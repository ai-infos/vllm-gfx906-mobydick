# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
import torch.nn as nn

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import (
        _topk_topp,
        _topp_split_mask,
        _topp_split_stats,
        _topp_split_step,
        apply_top_k_top_p_triton,
    )

logger = init_logger(__name__)


def register_top_k_top_p_warmups() -> None:
    """Register every native accelerator sampling kernel used at runtime."""
    if HAS_TRITON and not current_platform.is_cpu():
        _topk_topp.register_warmup()
        if current_platform.is_cuda_alike():
            _topp_split_stats.register_warmup()
            _topp_split_step.register_warmup()
            _topp_split_mask.register_warmup()


def _skip_aiter_sampler_on_gfx1250() -> bool:
    # Lazy ROCm-only import; keeps arch detection out of import time on CUDA/CPU.
    from vllm.platforms.rocm import on_gfx1250

    return on_gfx1250()


def flashinfer_sampler_supported() -> bool:
    """Decide whether FlashInfer's top-p/top-k sampler can be used.

    Returns False (with appropriate logging) when ``VLLM_USE_FLASHINFER_SAMPLER``
    is 0, when the platform isn't CUDA, when the GPU's compute capability is
    unsupported. Raises ``RuntimeError`` if the user explicitly opted in
    via the env var but FlashInfer is unavailable.

    Assumes flashinfer is installed, as guaranteed by ``requirements/cuda.txt``;
    otherwise importing the FlashInfer backend below raises ``ImportError``.

    Note: callers must additionally ensure ``logprobs_mode`` doesn't require
    post-top-k/top-p logits/logprobs for any request whose logprobs will be
    returned in this step, since FlashInfer doesn't expose those.
    """
    if not current_platform.is_cuda():
        return False
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        logger.info_once(
            "FlashInfer top-p/top-k sampling disabled via "
            "VLLM_USE_FLASHINFER_SAMPLER=0."
        )
        return False
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    capability = current_platform.get_device_capability()
    assert capability is not None
    unsupported_reason: str | None = None
    if not FlashInferBackend.supports_compute_capability(capability):
        unsupported_reason = (
            f"unsupported compute capability {capability.as_version_str()}"
        )

    if unsupported_reason is None:
        logger.info_once("Using FlashInfer for top-p & top-k sampling.", scope="global")
        return True
    if envs.is_set("VLLM_USE_FLASHINFER_SAMPLER"):
        raise RuntimeError(
            f"FlashInfer top-p/top-k sampling unavailable: {unsupported_reason}. "
            "Unset VLLM_USE_FLASHINFER_SAMPLER=1."
        )
    logger.warning_once(
        "FlashInfer top-p/top-k sampling unavailable: %s; falling back. "
        "Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.",
        unsupported_reason,
    )
    return False


class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel
        if current_platform.is_cuda():
            # FlashInfer doesn't expose post-top-k/top-p logits/logprobs,
            # so it can't be used when the configured mode requires them.
            can_use_flashinfer = (
                logprobs_mode not in PROCESSED_LOGPROBS_MODES
                and flashinfer_sampler_supported()
            )
            self.forward = (
                self.forward_cuda if can_use_flashinfer else self.forward_native
            )
        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            # Fall back to native implementation for POWERPC and RISCV.
            # On PowerPC argmax produces incorrect output with torch.compile.
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif current_platform.is_xpu():
            if envs.VLLM_XPU_USE_SAMPLER_KERNEL:
                self.forward = self.forward_xpu
            else:
                self.forward = self.forward_native
        elif (
            logprobs_mode not in PROCESSED_LOGPROBS_MODES
            and rocm_aiter_ops.is_enabled()
            and not _skip_aiter_sampler_on_gfx1250()  # TODO (JPVILLAM): Enable
        ):
            self.aiter_ops = None
            self._aiter_ops_import_failed = False
            logger.info_once(
                "Using aiter sampler on ROCm (lazy import, sampling-only)."
            )
            self.forward = self.forward_hip
        else:
            self.forward = self.forward_native

        # Every accelerator backend can fall back to native sampling at runtime.
        register_top_k_top_p_warmups()

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return (
            random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        # Fall back to the PyTorch-native path when FlashInfer has nothing
        # to do (no top-k / top-p filter) or when per-request generators
        # are present (unsupported by FlashInfer 0.2.3+).
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in PROCESSED_LOGPROBS_MODES, (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling for CPU.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        if not generators and not self.use_fp64_gumbel:
            return compiled_random_sample(logits), logits_to_return

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = empty_exponential_noise_like(probs, self.use_fp64_gumbel)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

        return sample_with_exponential_noise(probs, q), logits_to_return

    def _init_aiter_ops(self) -> bool:
        if self._aiter_ops_import_failed:
            return False
        try:
            import aiter.ops.sampling  # noqa: F401
        except ImportError:
            self._aiter_ops_import_failed = True
            self.forward = self.forward_native
            logger.warning_once(
                "aiter.ops.sampling is not available on ROCm. "
                "Falling back to PyTorch-native implementation."
            )
            return False
        self.aiter_ops = torch.ops.aiter
        return True

    def forward_hip(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Optimized ROCm/aiter path (same structure as forward_cuda)."""
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once(
                    "aiter sampler does not support per-request generators; "
                    "falling back to PyTorch-native."
                )
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in PROCESSED_LOGPROBS_MODES, (
            "aiter sampler does not support returning logits/logprobs."
        )
        if self.aiter_ops is None and not self._init_aiter_ops():
            return self.forward_native(logits, generators, k, p)
        return self.aiter_sample(logits, k, p, generators), None

    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """Sample from logits using aiter ops."""
        assert self.aiter_ops is not None
        use_top_k = k is not None
        use_top_p = p is not None
        # Joint k+p path
        if use_top_p and use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                *_to_tensor_scalar_tuple(k),
                *_to_tensor_scalar_tuple(p),
                deterministic=True,
            )
            return next_token_ids.view(-1)
        # Top-p only path
        elif use_top_p:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_p_sampling_from_probs(
                probs, None, *_to_tensor_scalar_tuple(p), deterministic=True
            )
            return next_token_ids.view(-1)
        # Top-k only path
        elif use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            renorm_probs = self.aiter_ops.top_k_renorm_probs(
                probs, *_to_tensor_scalar_tuple(k)
            )
            return torch.multinomial(renorm_probs, num_samples=1).view(-1)
        raise RuntimeError("aiter_sample was called with no active top-k or top-p.")

    def forward_xpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if generators:
            logger.warning_once(
                "xpu kernel topk_topp_sampler does not support "
                "per-request generators. Falling back to "
                "PyTorch-native implementation."
            )
            return self.forward_native(logits, generators, k, p)
        random_sampled = torch.empty(
            logits.shape[0], dtype=torch.int64, device=logits.device
        )
        logits_to_return = None
        if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            logits_to_return = torch.empty_like(logits)

        assert len(generators) != logits.shape[0], (
            "xpu kernel topk_topp_sampler does not support batch-wise generators."
        )
        generator = torch.xpu.default_generators[logits.device.index]

        state = generator.get_state()
        seed, offset = state.view(torch.int64)
        seeds = torch.tensor(
            [seed, offset], dtype=torch.int64, device=torch.device("cpu")
        )
        # The XPU kernel expects k as int64 (Long), but the input batch
        # stores top_k as int32. Cast here to avoid dtype mismatch.
        if k is not None:
            k = k.to(torch.int64)
        torch.ops.vllm.xpu_topk_topp_sampler(
            random_sampled, logits_to_return, logits, k, p, self.logprobs_mode, seeds
        )
        # The custom XPU sampler kernel consumes RNG values internally, so advance
        # the default generator's offset to keep future draws deterministic.
        # pytorch: offset must be multiple of 4
        offset = (offset + logits.numel() + 3) // 4 * 4
        state.view(torch.int64)[1] = offset
        generator.set_state(state)
        return random_sampled, logits_to_return


# Note: this is a workaround for
# https://github.com/pytorch/pytorch/pull/151218
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if HAS_TRITON:
        return apply_top_k_top_p_triton(logits, k, p)

    # SYV-4 (gfx906): small-batch top-k/top-p without the full-vocab sort.
    # Technique ported from syv-ai/qwen38-27b-rtx3090 (docs/optimizations.md,
    # fetched 2026-09-02; see docs/gfx906/RECON-syv-qwen38-27b-rtx3090.md
    # SYV-4); implementation is ours for the gfx906 ROCm path. The pytorch
    # path below sorts all ~248k logits per row (~0.35 ms at B=1); when every
    # row's k is small and host-known, torch.topk(k) + a threshold mask is
    # O(V log k) and bit-equivalent modulo fp rounding.
    if (
        _sort_free_small_k_enabled()
        and _can_use_sort_free_small_k(logits, k)
    ):
        return apply_top_k_top_p_sort_free(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    is_cpu = current_platform.is_cpu()
    return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=is_cpu)


def _sort_free_small_k_enabled() -> bool:
    """Opt-out for the SYV-4 sort-free small-k path (default ON)."""
    return os.environ.get("VLLM_GFX906_SORT_FREE_SMALL_K", "1") == "1"


def _can_use_sort_free_small_k(
    logits: torch.Tensor, k: torch.Tensor | None
) -> bool:
    """Preconditions for the sort-free path.

    ``k`` is the per-row top-k from the input batch (GPU int32); rows whose
    request does not use top-k are padded with ``vocab_size`` by
    ``gpu_input_batch``. Such full-vocab rows cannot be handled here (top-p
    over the whole vocabulary needs the sort), so any of them forces the
    fallback. The max-k check is one small device reduction + sync — same
    class of sync the reference path already performs.
    """
    if k is None or not k.is_cuda:
        return False
    if logits.shape[0] >= 8:
        return False
    # One device reduction + sync (the reference path performs the same class
    # of sync). gpu_input_batch only ever stores values in [1, vocab_size],
    # so max(k) >= vocab_size iff some row is a disabled/padded full-vocab row.
    kmax = int(k.max())
    return kmax < logits.shape[1] and kmax <= 64


def apply_top_k_top_p_sort_free(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """SYV-4: top-k/top-p without sorting the full vocabulary.

    Technique ported from syv-ai/qwen38-27b-rtx3090 (docs/optimizations.md,
    fetched 2026-09-02; see docs/gfx906/RECON-syv-qwen38-27b-rtx3090.md
    SYV-4); implementation is ours for the gfx906 ROCm path.

    Equivalent to ``apply_top_k_top_p_pytorch`` for batches where every row's
    k is small (<= 64). The reference path sorts all ~248k logits per row
    (~0.35 ms at B=1) and scatters back; here one ``torch.topk(k)`` gives the
    descending candidates, from which both thresholds are derived:

    - top-k: mask everything below the per-row k-th largest value (the
      reference's ascending-sort gather picks the same value up to tie order,
      and tied values are numerically equal);
    - top-p: cumulative softmax mass over the row's own top-k candidates in
      descending order; mask below the crossing value. This mirrors the
      reference exactly — it masks the sub-k-threshold entries first (they
      contribute 0 to the softmax), so its cumsum also runs over the top-k
      survivors only.

    Masked entries contribute exp(-inf) = 0 to the final softmax denominator,
    so renormalized probabilities match the reference up to fp rounding.
    At least one entry always survives (the crossing value itself is kept).

    The logits tensor is updated in-place.
    """
    assert k is not None
    kk = int(k.max())  # <= 64 by precondition; all rows partial here
    vals, _ = logits.topk(kk, dim=1)  # [B, kk], descending per row

    # Per-row top-k threshold: the k-th largest value.
    k_thresh = vals.gather(1, (k - 1).unsqueeze(1)).squeeze(1)  # [B]
    logits.masked_fill_(logits < k_thresh.unsqueeze(1), -float("inf"))

    if p is not None:
        # Per-row top-p over each row's own top-k candidates only. The reference
        # softmaxes the full vocab with non-candidates masked to -inf, so its
        # per-candidate probabilities equal this restricted softmax exactly.
        cand = vals.masked_fill_(
            torch.arange(kk, device=vals.device).unsqueeze(0) >= k.unsqueeze(1),
            -float("inf"),
        )
        probs_desc = cand.softmax(dim=-1)          # [B, kk], descending order
        cum = torch.cumsum(probs_desc, dim=-1)     # cum[j] = mass of top-(j+1)
        # keep_count = smallest n with cum[n-1] >= p  ==  (# of j with cum[j] < p) + 1.
        # `below` is monotone [1..1 0..0] (cum is non-decreasing), so its sum is
        # exactly that count — no argmax/bool op needed (ROCm has no bool kernels).
        below = (cum < p.unsqueeze(1)).to(torch.int32)
        # keep_count = smallest n with cum[n-1] >= p, clamped to this row's own
        # k: (a) at p=1.0 the final cum entry can round just under 1.0, pushing
        # the count to kk+1; (b) when a row's top-k mass cannot reach its target
        # p (e.g. top_k=3 with p=0.95), the reference keeps exactly that row's
        # k candidates — without the per-row clamp the gather would land on a
        # -inf slot and top-p would silently no-op for the row.
        keep_count = torch.clamp(torch.minimum(below.sum(dim=1) + 1, k), min=1)
        p_thresh = vals.gather(1, (keep_count - 1).unsqueeze(1)).squeeze(1)
        logits.masked_fill_(logits < p_thresh.unsqueeze(1), -float("inf"))

    return logits


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def empty_exponential_noise_like(
    probs: torch.Tensor, use_fp64_gumbel: bool
) -> torch.Tensor:
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    return torch.empty(probs.shape, dtype=dtype, device=probs.device)


def sample_with_exponential_noise(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator] = {},  # noqa
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.
    """
    import flashinfer

    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)


def _to_tensor_scalar_tuple(x):
    return (x, 0) if isinstance(x, torch.Tensor) else (None, x)

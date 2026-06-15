# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Torch SDPA against vLLM Triton unified attention.

This is a small ROCm/CUDA microbenchmark intended for local backend tuning.
It measures:

* dense Torch SDPA: best-case contiguous Q/K/V attention,
* paged Torch SDPA: gathers vLLM-style KV cache blocks then calls SDPA,
* vLLM Triton unified_attention: the actual paged kernel used by the backend.

Example:
    .venv/bin/python benchmarks/benchmark_sdpa_vs_unified_attention.py \
        --prefill-tokens 10000 --kv-tokens 10000 --iters 20 --warmup 5
"""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BenchResult:
    label: str
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    token_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=("both", "prefill", "decode"),
                        default="both")
    parser.add_argument("--prefill-tokens", type=int, default=10_000)
    parser.add_argument("--kv-tokens", type=int, default=10_000)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--local-q-heads", type=int, default=3)
    parser.add_argument("--local-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dtype",
                        choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--backends",
                        nargs="+",
                        choices=("auto", "flash", "math", "efficient"),
                        default=("auto", "flash"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seq-threshold-3d", type=int, default=128)
    parser.add_argument("--num-segments", type=int, default=16)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@contextlib.contextmanager
def sdpa_backend(name: str) -> Iterator[None]:
    if name == "auto":
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_map = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
        }
        with sdpa_kernel([backend_map[name]]):
            yield
        return
    except (ImportError, AttributeError):
        pass

    # Older PyTorch fallback.
    enable_flash = name == "flash"
    enable_math = name == "math"
    enable_mem_efficient = name == "efficient"
    with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient,
            enable_cudnn=False,
    ):
        yield


def make_block_table(kv_tokens: int, block_size: int,
                     device: torch.device) -> torch.Tensor:
    num_blocks = math.ceil(kv_tokens / block_size)
    return torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)


def make_inputs(
    q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(kv_tokens / block_size)
    padded_kv_tokens = num_blocks * block_size
    q = torch.randn(q_tokens, q_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(num_blocks,
                          block_size,
                          kv_heads,
                          head_dim,
                          dtype=dtype,
                          device=device)
    v_cache = torch.randn_like(k_cache)
    block_table = make_block_table(padded_kv_tokens, block_size, device)
    return q, k_cache, v_cache, block_table


def repeat_kv_heads(x: torch.Tensor, q_heads: int) -> torch.Tensor:
    kv_heads = x.shape[1]
    if kv_heads == q_heads:
        return x
    return x.repeat_interleave(q_heads // kv_heads, dim=1)


def cache_to_dense(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = k_cache.shape[1]
    num_blocks = math.ceil(kv_tokens / block_size)
    block_ids = block_table[0, :num_blocks].to(torch.long)
    k = k_cache.index_select(0, block_ids).reshape(-1, *k_cache.shape[2:])
    v = v_cache.index_select(0, block_ids).reshape(-1, *v_cache.shape[2:])
    return k[:kv_tokens], v[:kv_tokens]


def as_sdpa_q(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).unsqueeze(0)


def torch_dense_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    q4 = as_sdpa_q(q)
    k4 = as_sdpa_q(repeat_kv_heads(k, q.shape[1]))
    v4 = as_sdpa_q(repeat_kv_heads(v, q.shape[1]))
    with sdpa_backend(backend):
        out = F.scaled_dot_product_attention(q4,
                                             k4,
                                             v4,
                                             is_causal=causal,
                                             scale=scale)
    return out.squeeze(0).transpose(0, 1)


def torch_paged_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_tokens: int,
    *,
    backend: str,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    k, v = cache_to_dense(k_cache, v_cache, block_table, kv_tokens)
    return torch_dense_sdpa(q, k, v, backend=backend, causal=causal, scale=scale)


def make_unified_attention_runner(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_tokens: int,
    seq_threshold_3d: int,
    num_segments: int,
) -> Callable[[], torch.Tensor]:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    q_tokens, q_heads, head_dim = q.shape
    device = q.device
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, q_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([kv_tokens], dtype=torch.int32, device=device)
    head_dim_padded = next_power_of_2(head_dim)
    softmax_segm_output = torch.empty(
        (seq_threshold_3d, q_heads, num_segments, head_dim_padded),
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3d, q_heads, num_segments),
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_expsum = torch.empty_like(softmax_segm_max)
    scale = head_dim**-0.5

    def run() -> torch.Tensor:
        unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=q_tokens,
            seqused_k=seq_lens,
            max_seqlen_k=kv_tokens,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=seq_threshold_3d,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
        )
        return out

    return run


def bench(label: str, fn: Callable[[], torch.Tensor], tokens: int, warmup: int,
          iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = statistics.mean(times)
    return BenchResult(
        label=label,
        mean_ms=mean_ms,
        median_ms=statistics.median(times),
        min_ms=min(times),
        p90_ms=statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
        token_rate=tokens / (mean_ms / 1000.0),
    )


def maybe_check(label: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    max_abs = (got.float() - ref.float()).abs().max().item()
    mean_abs = (got.float() - ref.float()).abs().mean().item()
    print(f"  check {label}: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}")


def print_result(result: BenchResult) -> None:
    print(
        f"{result.label:34s} "
        f"mean={result.mean_ms:9.3f} ms  "
        f"median={result.median_ms:9.3f} ms  "
        f"min={result.min_ms:9.3f} ms  "
        f"p90={result.p90_ms:9.3f} ms  "
        f"tok/s={result.token_rate:10.2f}")


def run_group(
    name: str,
    q_tokens: int,
    kv_tokens: int,
    causal: bool,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    print(f"\n== {name}: q_tokens={q_tokens}, kv_tokens={kv_tokens} ==")
    q, k_cache, v_cache, block_table = make_inputs(
        q_tokens,
        kv_tokens,
        args.local_q_heads,
        args.local_kv_heads,
        args.head_dim,
        args.block_size,
        dtype,
        device,
    )
    scale = args.head_dim**-0.5
    k_dense, v_dense = cache_to_dense(k_cache, v_cache, block_table, kv_tokens)

    vllm_runner = None
    if not args.skip_vllm:
        try:
            vllm_runner = make_unified_attention_runner(
                q,
                k_cache,
                v_cache,
                block_table,
                kv_tokens,
                args.seq_threshold_3d,
                args.num_segments,
            )
        except Exception as err:
            print(f"  skipping vLLM unified_attention: {err}")

    reference: torch.Tensor | None = None
    for backend in args.backends:
        dense_fn = lambda backend=backend: torch_dense_sdpa(
            q, k_dense, v_dense, backend=backend, causal=causal, scale=scale)
        paged_fn = lambda backend=backend: torch_paged_sdpa(
            q,
            k_cache,
            v_cache,
            block_table,
            kv_tokens,
            backend=backend,
            causal=causal,
            scale=scale,
        )

        try:
            dense_result = bench(f"torch dense sdpa/{backend}", dense_fn, q_tokens,
                                 args.warmup, args.iters)
            print_result(dense_result)
            if args.check and reference is None:
                reference = dense_fn()
        except Exception as err:
            print(f"{'torch dense sdpa/' + backend:34s} FAILED: {err}")

        try:
            paged_result = bench(f"torch paged sdpa/{backend}", paged_fn, q_tokens,
                                 args.warmup, args.iters)
            print_result(paged_result)
            if args.check and reference is not None:
                maybe_check(f"paged/{backend}", paged_fn(), reference)
        except Exception as err:
            print(f"{'torch paged sdpa/' + backend:34s} FAILED: {err}")

    if vllm_runner is not None:
        result = bench("vLLM triton unified", vllm_runner, q_tokens, args.warmup,
                       args.iters)
        print_result(result)
        if args.check and reference is not None:
            maybe_check("vLLM unified", vllm_runner(), reference)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs a CUDA/ROCm GPU")
    if args.local_q_heads % args.local_kv_heads != 0:
        raise ValueError("--local-q-heads must be divisible by --local-kv-heads")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    print(f"torch={torch.__version__} hip={torch.version.hip} "
          f"cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device)} dtype={dtype}")
    print(f"q_heads={args.local_q_heads} kv_heads={args.local_kv_heads} "
          f"head_dim={args.head_dim} block_size={args.block_size}")
    print(f"backends={list(args.backends)} warmup={args.warmup} iters={args.iters}")

    if args.mode in ("both", "prefill"):
        run_group(
            "prefill",
            q_tokens=args.prefill_tokens,
            kv_tokens=args.prefill_tokens,
            causal=True,
            args=args,
            dtype=dtype,
            device=device,
        )
    if args.mode in ("both", "decode"):
        run_group(
            "decode",
            q_tokens=args.decode_tokens,
            kv_tokens=args.kv_tokens,
            causal=False,
            args=args,
            dtype=dtype,
            device=device,
        )


if __name__ == "__main__":
    main()

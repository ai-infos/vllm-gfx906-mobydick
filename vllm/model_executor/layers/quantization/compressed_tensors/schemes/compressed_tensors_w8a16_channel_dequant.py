# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""W8A16 channel-wise dense layers (compressed-tensors) on gfx906.

Nemotron-3.5-Lightning-30B-class checkpoints store the mamba/attention
dense projections and lm_head as symmetric bias-128 int8 with
per-output-channel scales (packed 4 int8 per int32 word along K,
little-endian ascending). The only stock ROCm mixed-precision kernel
for this layout (ConchLinearKernel) costs ~3.8 ms per M=1 GEMV on MI50
(75% of Nemotron decode GPU time across the 46 dense projections), so
this scheme has two paths:

int8 in-kernel (opt-in on gfx906 + fp16; NH-2 — measured NO-GO for the
    serving mode, see DEVLOG-nemotron-h.md 2026-08-30):
    the packed bytes are pre-shifted in-place to signed int8 at load
    and served as a no-copy view — no fp16 materialization, 1
    byte/weight. The Triton GEMV (M=1) and GEMM (M>1) kernels dequant
    in-register. Enable with VLLM_GFX906_W8A16_INT8=1 (default 0);
    it wins only at M=1 on the K=2688/large-N shapes (lm_head 1.60x,
    mamba in_proj 1.42x) and loses at M>1 (0.19-0.80x vs the
    hand-tuned CUDA/hipBLAS paths) — Nemotron's spec-decode serving
    mode is M=6/step.

load-time dequant (fallback, and all non-gfx906 ROCm):
    the packed int8 is dequantized to fp16 once at load and the
    optimized gfx906 unquantized GEMV family runs on it (2x the int8
    weight bytes in VRAM; prefill is a plain GEMM).

Both paths use the same convention: w = (byte - 128) * scale.

On the int8 path the packed storage is pre-shifted in-place at load
(`byte ^= 0x80`, i.e. (byte - 128) mod 256) so `weight_i8` is a signed
int8 view of it and the kernels run the P2-validated 3-op element chain
(w.to(f32) * x; no per-element f32 subtract — the M=1 GEMV is
f32-ALU-bound at Nemotron's K, the 4th op cost ~25 % there).
"""

import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)

__all__ = ["CompressedTensorsW8A16ChannelDequant"]

_DEQUANT_CHUNK_ROWS = 8192
_ENV_INT8 = "VLLM_GFX906_W8A16_INT8"
# NH-2': CUDA byte-load + in-register per-channel dequant GEMV family
# (dense_gemv_i8_gfx906 / dense_gemv_i8_m4_gfx906), opt-in on top of the
# int8 path. Supported: M <= 4, K % 16 == 0, N even; everything else
# falls back to the Triton kernels below (bit-exact same convention).
_ENV_INT8_CUDA = "VLLM_GFX906_W8A16_INT8_CUDA"


def _i8_cuda_kchunk(k: int) -> int | None:
    """KC (bytes of weight per thread-slice) for a given K, or None when the
    CUDA family cannot serve it. KC must be in {1024, 2048, 4096} (whole
    wavefronts); ksplit = ceil(K / KC)."""
    if k % 16 != 0:
        return None
    for kc in (2048, 4096):
        if kc >= k:
            return kc
    # K > 4096: split only when divisible by a supported KC.
    if k % 2048 == 0:
        return 2048
    if k % 1024 == 0:
        return 1024
    return None


@triton.jit
def k_w8a16_gemv(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    N,
    K,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """M=1 GEMV over signed int8 rows: out[n] = s[n] * sum_k w[n,k] * x[k].

    w is int8 [N, K] (the packed storage pre-shifted to two's complement
    at load, see module docstring); s is fp16 [N]; out is fp16 [N]. For
    SPLIT > 1 the caller zero-initializes out and partials are
    atomic-added. (P2-validated element chain: 3 f32 ops, no subtract.)
    """
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    per = K // SPLIT
    for k0 in range(pk * per, (pk + 1) * per, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < (pk + 1) * per
        x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0,
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    acc = acc * s
    if SPLIT == 1:
        tl.store(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=rmask)
    else:
        tl.atomic_add(o_ptr + rows, acc.to(o_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def k_w8a16_gemm(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """M>1 W8A16 GEMM: dequant each weight tile in-register to fp16
    (bit-identical to the load-time dequant: f32 w*s -> f16, w signed) and
    tl.dot with fp32 accumulation. w is loaded [BK, BN] so it feeds the
    dot directly (no transpose). Requires K % BK == 0 (checked by the
    launcher; K is a multiple of the pack factor 4).
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + ks[None, :], mask=mmask[:, None], other=0.0
        )
        w = tl.load(w_ptr + rn[None, :] * K + ks[:, None], mask=nmask[None, :], other=0)
        s = tl.load(s_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        w16 = (w.to(tl.float32) * s[None, :]).to(tl.float16)
        acc = tl.dot(x, w16, acc)
    tl.store(
        o_ptr + rm[:, None] * N + rn[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mmask[:, None] & nmask[None, :],
    )


# (k, n) -> (BN, BK, SPLIT): measured winners on MI50
# (bench_w8a16_gfx906.py, 2026-08-30, us): k=2688: n=131072 (32,128,1)
# 530.8 | n=10304 (64,128,1) 49.3 | n=4096 (32,128,1) 28.2 | n=256
# (16,128,4) 16.9; k=4096: n=2688 (16,512,1) 51.3. The K=4096 family
# loses to production at every config (P2's fill rule picks SPLIT=2,
# which measured worse: 51.3 vs 66.1 at SPLIT=1).
_MEASURED_GEMV = {
    (2688, 131072): (32, 128, 1),
    (2688, 10304): (64, 128, 1),
    (2688, 4096): (32, 128, 1),
    (2688, 256): (16, 128, 4),
    (4096, 2688): (16, 512, 1),
}


def _gemv_geometry(n: int, k: int) -> tuple[int, int, int]:
    """(BN, BK, SPLIT) for the M=1 GEMV. Measured winners for the known
    Nemotron shapes; P2's _pick rules otherwise (BN by N band; SPLIT
    doubled only while the (row, split) program count still under-fills
    ~5 waves of the 60 CUs; BK = largest power of two (<= 512, >= 64)
    dividing k — the kernel's kmask handles a partial tail tile)."""
    if (k, n) in _MEASURED_GEMV:
        return _MEASURED_GEMV[(k, n)]
    bn = 64 if n >= 12288 else (32 if n >= 4096 else 16)
    split = 1
    while split < 8 and k % (split * 2) == 0 and triton.cdiv(n, bn) * split < 300:
        split *= 2
    bk = 512
    while bk > 64 and k % bk:
        bk //= 2
    assert k % bk == 0 and k >= bk, f"N={n} K={k}: no usable BK"
    return bn, bk, split


def _gemm_geometry(m: int, k: int) -> tuple[int, int, int]:
    """(BM, BN, BK) for the M>1 GEMM: K is a multiple of 4, so a BK in
    {64, 32, 16} always divides it (16 only for exotic K)."""
    bm = 16 if m <= 16 else 64
    for bk in (64, 32, 16):
        if k % bk == 0:
            return bm, 64, bk
    raise ValueError(f"W8A16 GEMM: K={k} not a multiple of 16")


def w8a16_gemv(
    w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """M=1: x [1, K] fp16 -> out [N] fp16. w is int8 [N, K] (the packed
    storage pre-shifted to signed at load), scale fp16 [N] or [N, 1]."""
    n, k = w.shape
    bn, bk, split = _gemv_geometry(n, k)
    dev = w.device
    out = torch.zeros(n, dtype=scale.dtype, device=dev) if split > 1 else torch.empty(
        n, dtype=scale.dtype, device=dev
    )
    k_w8a16_gemv[(triton.cdiv(n, bn), split)](
        x, w, scale, out, n, k, BN=bn, BK=bk, SPLIT=split, num_warps=8
    )
    return out


def w8a16_gemm(
    w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """M>1: x [M, K] fp16 -> out [M, N] fp16."""
    n, k = w.shape
    m = x.shape[0]
    bm, bn, bk = _gemm_geometry(m, k)
    out = torch.empty(m, n, dtype=scale.dtype, device=w.device)
    k_w8a16_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, w, scale, out, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=4
    )
    return out


class CompressedTensorsW8A16ChannelDequant(CompressedTensorsScheme):
    """Symmetric pack-quantized int8 channel-wise weights.

    On gfx906 with fp16 params and VLLM_GFX906_W8A16_INT8=1 the packed
    bytes are served in-kernel (NH-2 — default off, measured NO-GO for
    the serving mode, see module docstring); otherwise they are
    dequantized to ``params_dtype`` in process_weights_after_loading
    and the unquantized GEMV family runs on the result.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # Selection is platform-gated (gfx906 only); mirror the WNA16 gate
        # for the _check_scheme_supported path.
        return 75

    def __init__(self, layer_name: str | None = None):
        self.layer_name = layer_name

    def _int8_path(self, params_dtype: torch.dtype) -> bool:
        from vllm.platforms.rocm import on_gfx906

        # default OFF: measured NO-GO for the serving mode
        # (DEVLOG-nemotron-h.md, 2026-08-30); =1 for the M=1-only wins
        return (
            on_gfx906()
            and params_dtype == torch.float16
            and os.environ.get(_ENV_INT8, "0") == "1"
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        layer.serve_int8 = self._int8_path(params_dtype)
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False

        pack_factor = 4  # 32 / num_bits, num_bits == 8
        if input_size_per_partition % pack_factor != 0:
            raise ValueError(
                "W8A16 channel dequant requires input_size_per_partition "
                f"divisible by 4 (got {input_size_per_partition})"
            )

        # [N, K/4] int32, 4 int8 values per word along K (little-endian,
        # ascending k) — the compressed-tensors pack-quantized layout.
        packed_input_dim = input_size_per_partition // pack_factor
        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                packed_input_dim,
                dtype=torch.int32,
            ),
        )

        # Channel scales are per output row [N, 1]: column-parallel layers
        # load their row slice, row-parallel layers keep all rows (matches
        # the WNA16 channel convention — scales follow the output dim only).
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0,
            weight_loader=weight_loader,
            data=torch.empty(output_size_per_partition, 1, dtype=params_dtype),
        )

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_packed = layer.weight_packed.data
        scale = layer.weight_scale.data
        n, k_words = w_packed.shape
        k = k_words * 4
        dtype = scale.dtype

        if scale.shape != (n, 1):
            raise ValueError(
                f"W8A16 channel dequant expects scale shape [{n}, 1], "
                f"got {tuple(scale.shape)}"
            )

        if getattr(layer, "serve_int8", False):
            # The packed bytes are the bias-128 codes; XOR 0x80 turns each
            # byte into the two's-complement code of (byte - 128) — an
            # in-place 1-op/weight pre-shift (no copy, no extra VRAM) so
            # the kernels consume plain signed int8 (3-op element chain).
            # After this, weight_packed's storage is no longer the
            # checkpoint's raw codes.
            w_bytes = w_packed.view(torch.uint8)
            w_bytes.bitwise_xor_(0x80)
            w_i8 = w_bytes.view(torch.int8).reshape(n, k)
            layer.weight_i8 = torch.nn.Parameter(w_i8, requires_grad=False)
            # weight_packed is the storage; keep it alive. weight_shape is
            # load metadata only.
            del layer.weight_shape
            return

        # uint8 view: [N, K] with ascending k within each int32 word
        # (little-endian bytes). Symmetric int8 uses the bias-128
        # convention: w = (q - 128) * scale.
        w_u8 = w_packed.view(torch.uint8)
        dequant = torch.empty(n, k, dtype=dtype, device=w_packed.device)
        for row0 in range(0, n, _DEQUANT_CHUNK_ROWS):
            row1 = min(row0 + _DEQUANT_CHUNK_ROWS, n)
            chunk = w_u8[row0:row1].to(torch.float32)
            chunk -= 128.0
            chunk *= scale[row0:row1].to(torch.float32)
            dequant[row0:row1] = chunk.to(dtype)

        layer.weight = torch.nn.Parameter(dequant, requires_grad=False)
        # Free the packed tensors: the dequantized weight replaces them.
        del layer.weight_packed
        del layer.weight_scale
        del layer.weight_shape

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if getattr(layer, "serve_int8", False):
            if bias is not None:
                raise ValueError(
                    f"{self.layer_name}: the gfx906 W8A16 int8 in-kernel "
                    "path does not support bias"
                )
            if x.dtype != torch.float16:
                raise ValueError(
                    f"{self.layer_name}: the gfx906 W8A16 int8 in-kernel "
                    f"path requires fp16 activations (got {x.dtype})"
                )
            x2d = x.reshape(-1, x.shape[-1])
            if not x2d.is_contiguous():
                x2d = x2d.contiguous()
            w = layer.weight_i8.data
            s = layer.weight_scale.data.squeeze(-1)
            m = x2d.shape[0]
            # NH-2': CUDA byte-load + in-register per-channel dequant
            # (opt-in on top of the int8 path; A/B-gated, default off).
            # M=1 routes through the M<=4 kernel too (M is a template
            # parameter there) so one dispatch covers the whole serving
            # range; kchunk selection and the fallback to Triton are
            # shape-driven.
            if (
                os.environ.get(_ENV_INT8_CUDA, "0") == "1"
                and m <= 4
                and w.shape[0] % 2 == 0
            ):
                kc = _i8_cuda_kchunk(w.shape[1])
                if kc is not None:
                    from vllm import _custom_ops as ops

                    out = ops.dense_gemv_i8_m4_gfx906(
                        w, s, x2d.contiguous(), kc
                    )
                    return out.view(*x.shape[:-1], out.shape[-1])
            if m == 1:
                out = w8a16_gemv(w, s, x2d[0].contiguous())
            else:
                out = w8a16_gemm(w, s, x2d)
            return out.view(*x.shape[:-1], out.shape[-1])

        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

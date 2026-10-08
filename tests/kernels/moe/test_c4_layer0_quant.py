# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Unit tests for C4 load-time int4 quantization (c4_layer0_moe).

Covers, CPU-only:
- the asymmetric AWQ dequant accuracy bound of the group quantizer;
- bit-exact packing (qweight k-pairs, qzeros n-pairs) vs an independent
  unpack;
- agreement with `_repack_w4a16_wna16_layout`: feeding the quantizer output
  through the gfx906 repack must equal the exllama-shuffled layout built
  independently from the same (q, scale, zp);
- the env gate.
"""

import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _repack_w4a16_gfx906_expert,
)
from vllm.model_executor.layers.quantization.c4_layer0_moe import (
    _quantize_fp16_to_moe_wna16,
    c4_quant_layer0_enabled,
)


def _reference_quant(w: torch.Tensor, group_size: int):
    """Independent reference: returns (q [N,K] uint8, scale [N,G], zp [N,G]).

    Mirrors the implementation's convention: codepoints are chosen against
    the *stored* fp16 scale (what the kernel applies), so bit-exact agreement
    is expected.
    """
    N, K = w.shape
    G = K // group_size
    wf = w.float()
    groups = wf.reshape(N, G, group_size)
    mx = groups.amax(dim=2, keepdim=True)
    mn = groups.amin(dim=2, keepdim=True)
    scale32 = ((mx - mn).clamp(min=1e-5) / 15.0).squeeze(2)  # [N, G] fp32
    scale_w = scale32.to(w.dtype)  # fp16, the kernel's convention
    sw = scale_w.float().unsqueeze(2)
    zp = torch.round(mn.abs() / sw).clamp(0, 15).squeeze(2).int()
    q = torch.round(groups / sw) + zp.unsqueeze(2)
    q = q.clamp(0, 15).to(torch.uint8)  # [N, G, gs]
    return q.reshape(N, K), scale_w, zp


@pytest.mark.parametrize("group_size", (32, 128))
@pytest.mark.parametrize("n,k", ((512, 2048), (2048, 512)))
def test_roundtrip_accuracy(group_size, n, k):
    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.float16) * 0.05
    qweight, scales, qzeros = _quantize_fp16_to_moe_wna16(w, group_size)

    # Unpack the packed tensors back to nibbles.
    G = k // group_size
    lo = (qweight & 0xF).to(torch.int32)
    hi = ((qweight >> 4) & 0xF).to(torch.int32)
    q_unpacked = torch.stack([lo, hi], dim=2).reshape(n, k)

    zlo = (qzeros & 0xF).to(torch.int32)
    zhi = ((qzeros >> 4) & 0xF).to(torch.int32)
    zp_unpacked = torch.stack([zlo, zhi], dim=1).reshape(n, G)

    dequant = (q_unpacked - zp_unpacked.repeat_interleave(group_size, dim=1)) \
        * scales.float().repeat_interleave(group_size, dim=1)
    err = (w.float() - dequant).abs()
    # Per-element bound: half a step of the element's own group, plus the
    # fp16-scale-rounding term. Codepoints use sw (the stored scale), but the
    # group range was computed with the fp32 scale c; when fp16 rounds c down
    # (c/sw - 1 <= 2^-11), the max weight can exceed 15 codepoints and clamp,
    # adding at most 15*(c - sw) ~= 0.7% of a step.
    step_elem = scales.float().repeat_interleave(group_size, dim=1)
    assert (err <= step_elem * 0.51 + 1e-6).all(), (
        f"max err {err.max().item():.3e} vs bound "
        f"{(step_elem * 0.51).max().item():.3e}"
    )
    # L2 relative error is the meaningful accuracy metric for int4 weight
    # quantization (~6-8% for unit-variance data); the final quality gate is
    # the PPL/coherence check on the real model.
    rel_l2 = err.norm() / w.float().norm()
    assert rel_l2.item() < 0.15, f"L2 relative error too large: {rel_l2.item()}"


@pytest.mark.parametrize("group_size", (32, 128))
def test_packing_bit_exact(group_size):
    torch.manual_seed(1)
    n, k = 256, 1024
    w = torch.randn(n, k, dtype=torch.float16)
    q_ref, scale_ref, zp_ref = _reference_quant(w, group_size)
    G = k // group_size

    qweight, scales, qzeros = _quantize_fp16_to_moe_wna16(w, group_size)

    # Scales must match the reference exactly (fp16 cast of the same value).
    assert torch.equal(scales.to(torch.float32), scale_ref.to(torch.float32))

    # Unpack and compare nibble-for-nibble.
    lo = (qweight & 0xF).to(torch.int32)
    hi = ((qweight >> 4) & 0xF).to(torch.int32)
    q_unpacked = torch.stack([lo, hi], dim=2).reshape(n, k)
    assert torch.equal(q_unpacked, q_ref.to(torch.int32))

    zlo = (qzeros & 0xF).to(torch.int32)
    zhi = ((qzeros >> 4) & 0xF).to(torch.int32)
    zp_unpacked = torch.stack([zlo, zhi], dim=1).reshape(n, G)
    assert torch.equal(zp_unpacked, zp_ref.to(torch.int32))


def test_repack_matches_independent_exllama_layout():
    """Feeding the quantizer output through the gfx906 repack must equal the
    exllama-shuffled layout built directly from (q, scale, zp)."""
    torch.manual_seed(2)
    E, N, K = 4, 512, 2048
    group_size = 128
    w = torch.randn(E, N, K, dtype=torch.float16) * 0.05

    q_ref_list, sc_ref_list, zp_ref_list = [], [], []
    qw_list, sc_list, zq_list = [], [], []
    for e in range(E):
        q, sc, zp = _reference_quant(w[e], group_size)
        q_ref_list.append(q)
        sc_ref_list.append(sc)
        zp_ref_list.append(zp)
        qw, s, z = _quantize_fp16_to_moe_wna16(w[e], group_size)
        qw_list.append(qw)
        sc_list.append(s)
        zq_list.append(z)

    q_stacked = torch.stack(q_ref_list)          # [E, N, K]
    sc_stacked = torch.stack(sc_ref_list)        # [E, N, G]
    zp_stacked = torch.stack(zp_ref_list)        # [E, N, G]
    w_in = torch.stack(qw_list).contiguous()     # [E, N, K/2] uint8
    sc_in = torch.stack(sc_list).contiguous()    # [E, N, G] fp16
    z_in = torch.stack(zq_list).contiguous()     # [E, N/2, G] uint8

    wq, sc_out, zp_out = _repack_w4a16_gfx906_expert(w_in, sc_in, z_in)

    # Independent exllama shuffle: for k = 8*qk + j, nibble j goes to bits
    # [4j] (even j) or [16 + 4*(j-1)] (odd j). q_stacked is [E, N, K].
    G = K // group_size
    shifts_out = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    wq_ref = (
        (q_stacked.to(torch.int32).view(E, N, K // 8, 8)
         << shifts_out.view(1, 1, 8)).sum(dim=3)
        .permute(0, 2, 1).contiguous()
    )
    assert torch.equal(wq, wq_ref), "exllama shuffle mismatch"

    # Scales pass through to [E, G, N].
    assert sc_out.shape == (E, G, N)
    assert torch.equal(
        sc_out.to(torch.float32),
        sc_stacked.permute(0, 2, 1).contiguous().to(torch.float32),
    )

    # Zero points: [E, G, N/8] int32, 8 nibbles per word, n ascending
    # (bit 4j holds n = 8m + j). zp_stacked is [E, N, G]. Shifts must be
    # int32 so the top nibble wraps exactly as in the kernel's int32 OR.
    zp_nk = zp_stacked.to(torch.int32)
    zp_ref = (
        zp_nk.view(E, N // 8, 8, G) << (4 * torch.arange(8, dtype=torch.int32)).view(1, 1, 8, 1)
    ).sum(dim=2).permute(0, 2, 1).contiguous()
    assert zp_out.shape == (E, G, N // 8)
    assert torch.equal(zp_out, zp_ref), "zero-point packing mismatch"


def test_env_gate(monkeypatch):
    # Default ON since the gate passed (DEVLOG-c4-layer0-quant.md); "0" is the
    # kill switch.
    monkeypatch.delenv("VLLM_GFX906_QUANT_LAYER0_MOE", raising=False)
    assert c4_quant_layer0_enabled()
    monkeypatch.setenv("VLLM_GFX906_QUANT_LAYER0_MOE", "1")
    assert c4_quant_layer0_enabled()
    monkeypatch.setenv("VLLM_GFX906_QUANT_LAYER0_MOE", "0")
    assert not c4_quant_layer0_enabled()

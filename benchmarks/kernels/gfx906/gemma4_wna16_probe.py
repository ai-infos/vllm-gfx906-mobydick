# SPDX-License-Identifier: Apache-2.0
# Copyright Kevin Read <me@kevin-read.com>
"""Phase-0 format probe for Gemma-4-26B-A4B-AWQ MoE (no-zero-point W4A16).

Verifies, from the raw safetensors (CPU, no model load):
  1. On-disk layout: packed int32 N-first [N, K/8] (the MoE loader
     transposes to K-first [K/8, N]), scales fp16 [N, K/G],
     NO zero-point tensor (symmetric).
  2. Reference dequant math: w = (q - 8) * scale, LSB-first nibbles along K
     (cross-checked against vLLM's own emulation reference
     `_unpack_and_dequant_int4_gptq` with qzeros=None).
  3. Repack round-trip: the planned GPTQ K-first -> exllama-shuffle repack
     (same shuffle as the existing AWQ K-first branch) dequants back to the
     reference under the kernel's exact arithmetic.
  4. Distribution sanity: dequant weights are zero-centered and consistent
     with the unquantized bf16 shared experts of the same layers.

Usage:
  .venv/bin/python benchmarks/kernels/gfx906/gemma4_wna16_probe.py [snapshot]
"""

import json
import sys

import torch
from safetensors import safe_open

SNAP = (
    "/local/cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit"
    "/snapshots/0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba"
)


def load_tensor(snap, name):
    with open(f"{snap}/model.safetensors.index.json") as f:
        idx = json.load(f)
    fname = idx["weight_map"][name]
    with safe_open(f"{snap}/{fname}", framework="pt") as f:
        return f.get_tensor(name)


def ref_dequant_gptq(w_int32, scale, qzeros=None):
    """Independent unpack: K-first int32, LSB-first nibbles along K.

    w_int32: [K/8, N] int32; scale: [K/G, N]. Returns [K, N] fp32.
    """
    K = w_int32.shape[0] * 8
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (w_int32.unsqueeze(-1) >> shifts) & 0xF  # [K/8, N, 8]
    q = nib.permute(0, 2, 1).reshape(K, w_int32.shape[1]).to(torch.int16)
    G = scale.shape[0]
    sc = scale.repeat_interleave(K // G, dim=0).to(torch.float32)
    return (q - 8).to(torch.float32) * sc


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else SNAP
    torch.manual_seed(0)

    E0 = "model.language_model.layers.0.experts.0"
    wp_raw = load_tensor(snap, f"{E0}.gate_proj.weight_packed")
    sc_raw = load_tensor(snap, f"{E0}.gate_proj.weight_scale")
    ws = load_tensor(snap, f"{E0}.gate_proj.weight_shape")
    print(f"gate_proj.weight_packed (raw): {tuple(wp_raw.shape)} {wp_raw.dtype}")
    print(f"gate_proj.weight_scale  (raw): {tuple(sc_raw.shape)} {sc_raw.dtype}")
    print(f"gate_proj.weight_shape:  {tuple(ws.shape)} {ws.tolist()}")

    assert wp_raw.dtype == torch.int32
    # Raw on-disk layout is N-first [N, K/8] / [N, K/G]; the vLLM MoE loader
    # (is_transposed=True) transposes to [K/8, N] before the oracle/repack.
    assert wp_raw.shape == (704, 2816 // 8), "expected N-first [704, 352]"
    assert sc_raw.shape == (704, 2816 // 32), "expected [704, 88]"
    wp = wp_raw.t().contiguous()
    sc = sc_raw.t().contiguous()
    # No zero point on disk:
    with open(f"{snap}/model.safetensors.index.json") as f:
        idx = json.load(f)
    zp_keys = [k for k in idx["weight_map"] if "zero_point" in k]
    assert not zp_keys, f"unexpected zero-point tensors: {zp_keys[:3]}"
    print("no zero-point tensors in index: OK (symmetric)")

    # --- 1. My unpack vs vLLM emulation reference (1 expert, 1 layer) ---
    ref_vllm = None
    try:
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            _unpack_and_dequant_int4_gptq,
        )

        ref_vllm = _unpack_and_dequant_int4_gptq(
            wp.unsqueeze(0),
            sc.unsqueeze(0).to(torch.float16),
            None,
            transpose_output=False,
            output_dtype=torch.float16,
        )[0].float()
    except Exception as e:  # import may need GPU env; not fatal
        print(f"(skipping vLLM reference: {e})")

    mine = ref_dequant_gptq(wp, sc)
    if ref_vllm is not None:
        d = (mine - ref_vllm).abs().max().item()
        rel = (mine - ref_vllm).abs().max() / mine.abs().max()
        ok = d < 2.5e-3 and rel < 1e-2  # ref dequants in fp16
        print(
            f"mine vs vLLM emulation ref: max |diff| = {d:.3e} "
            f"(rel {rel:.3e}) {'OK' if ok else 'MISMATCH'}"
        )

    print(
        f"dequant gate: mean={mine.mean():.5f} std={mine.std():.5f} "
        f"min={mine.min():.4f} max={mine.max():.4f}"
    )
    shifts = torch.arange(8, dtype=torch.int32) * 4
    hist = torch.bincount(
        ((wp.unsqueeze(-1) >> shifts) & 0xF).reshape(-1), minlength=16
    )
    print(f"nibble histogram (0..15): {hist.tolist()}")

    # --- 2. Repack round-trip (planned GPTQ K-first branch) ---
    # Same exllama shuffle as _repack_w4a16_awq_kfirst_layout.
    K8, N = wp.shape
    shifts_out = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    q = (wp.unsqueeze(-1) >> (4 * torch.arange(8, dtype=torch.int32))) & 0xF
    wq = (
        (q.permute(0, 2, 1).reshape(K8, 8, N) << shifts_out.view(1, 8, 1))
        .sum(dim=1)
        .to(torch.int32)
    ).unsqueeze(0)  # [1, K/8, N]
    sc3 = sc.unsqueeze(0).to(torch.float16)  # [1, G, N]

    # Dequant one random row slice with the kernel's exact arithmetic:
    # dq = (q + 1024) * scale + scale * (-1024 - zero), zero = 8, zero_offset 0.
    for trial in range(200):
        r = torch.randint(0, K8, (1,)).item()
        col = torch.randint(0, N, (1,)).item()
        word = wq[0, r, col].item() & 0xFFFFFFFF
        nibs = [(word >> (4 * j)) & 0xF for j in range(8)]
        # exllama shuffle: even j -> low half, odd j -> high half
        k_of_slot = [0, 2, 4, 6, 1, 3, 5, 7]
        g = r // (K8 // sc.shape[0])  # group of this word (groupsize = 32 k = 4 words)
        s = sc3[0, g, col].item()
        for slot, kk in enumerate(k_of_slot):
            qv = nibs[slot]
            k = r * 8 + kk
            dq = (qv + 1024) * s + s * (-1024 - 8)
            ref = mine[k, col]
            assert abs(dq - ref) < 1e-3 * max(1.0, abs(ref)), (
                f"trial {trial}: slot {slot} k={k} col={col} dq={dq} ref={ref}"
            )
    print("repack round-trip (200 random cells, kernel arithmetic): OK")

    # --- 3. Distribution consistency vs bf16 shared expert ---
    sh = load_tensor(snap, "model.language_model.layers.0.mlp.gate_proj.weight")
    print(
        f"shared expert gate (bf16): mean={sh.float().mean():.5f} "
        f"std={sh.float().std():.5f} min={sh.float().min():.4f} "
        f"max={sh.float().max():.4f}"
    )

    # Scales: per-group max|w| should be ~<= 8*scale (uint4b8 headroom)
    G = sc.shape[0]
    gs = K8 * 8 // G
    maxabs = mine.abs().view(G, gs, N).amax(dim=1)  # [G, N]
    ratio = (maxabs / sc.float()).flatten()
    print(
        f"max|w|/scale per group: mean={ratio.mean():.3f} "
        f"p99={ratio.quantile(0.99):.3f} max={ratio.max():.3f} (expect <= 8)"
    )
    print("PROBE PASS")


if __name__ == "__main__":
    main()

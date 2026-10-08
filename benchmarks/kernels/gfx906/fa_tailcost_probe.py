#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""S8 decode-tax kernel breakdown (plan_masked_fa.md step 1).

The S8 gap (39.9 -> 29.9 t/s at 131072 -> 262144, TP=2 dense-27B, ~8.4
ms/step at ~1.5k live context) is attributed to the capture-frozen
two-kernel fallback that runs at Sk_pad > 65535:
  1. gather_paged_kv_fp16 (V2 kernel, grid (B, Hkv, ceil(Sk/16)))
  2. quantize_q8_0 (dense pass over the full [B, Hkv, Sk, D] K buffer)
plus the FA-compute kernel, which is live-bounded (kv_max) and
therefore NOT part of the tax.

rocprofv3 full-model traces are unusable on this box for dense runs
(DEVLOG-dense-decode.md: exit-finalization race), so this probe measures
the kernels directly at the exact S8 shapes. Kernel GPU times transfer
from standalone to in-model (house rule: launch-regime evidence — the
serving A/B remains the gate).

Shapes (Qwen3.8-27B-AWQ-INT4, 16 FA layers, Hq=24, D=256, block 16):
  B=1, live seq_len=1536, Sk in {3328, 65536, 131072, 262144},
  Hkv in {2 (TP=2 serving), 4 (TP=1)}.

Run (local venv, GPU 0):
  cd /local/git/vllm-gfx906-mobydick
  source ~/env-rocm-7.14-gfx906.sh
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u \
      benchmarks/kernels/gfx906/fa_tailcost_probe.py
"""
import os

import torch

dev = "cuda"
torch.manual_seed(0)

Hq = 24
D = 256
BLOCK = 16
BPR = (D // 32) * 34  # 272 uint8 per Q8 row
LIVE = 1536           # ~S8 real context (1.5k)
N_LAYERS = 16         # FA layers in the 27B
HBM_BW = 798e9        # measured MI50 HBM read BW (P3-0 Q1)


def time_us(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us/call


def ref_gather(kc, vc, bt, sl, sk, hkv):
    """Torch reference: rows [0, sl) per seq, V tail zeroed."""
    nb = (sk + BLOCK - 1) // BLOCK
    bt_l = bt[:, :nb].long()
    k = kc[bt_l].view(1, -1, hkv, D)[:, :sk]
    v = vc[bt_l].view(1, -1, hkv, D)[:, :sk]
    pos = torch.arange(sk, device=dev)
    m = (pos[None, :] < sl[:, None]).view(1, sk, 1, 1).to(torch.float16)
    v = v * m
    return (k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous())


def main():
    from vllm import _gfx906_fa_C as fa

    sks = [3328, 65536, 131072, 262144]
    maxsk = max(sks)
    num_blocks = maxsk // BLOCK + 8

    rows = []
    for hkv in (2, 4):
        # fp16 paged caches for the LEGACY (two-kernel fallback) path:
        kcf = torch.randn(num_blocks, BLOCK, hkv, D,
                          dtype=torch.float16, device=dev)
        vcf = torch.randn(num_blocks, BLOCK, hkv, D,
                          dtype=torch.float16, device=dev)

        bt = torch.zeros(1, num_blocks, dtype=torch.int32, device=dev)
        bt[0, : num_blocks - 8] = torch.arange(num_blocks - 8,
                                               dtype=torch.int32)
        sl = torch.tensor([LIVE], dtype=torch.int32, device=dev)

        for sk in sks:
            kb = torch.empty(1, hkv, sk, D, dtype=torch.float16, device=dev)
            vb = torch.empty(1, hkv, sk, D, dtype=torch.float16, device=dev)
            gather = lambda: fa.gather_paged_kv_fp16(
                kcf, vcf, bt, sl, sk, k_out=kb, v_out=vb)
            gather()
            kq = torch.empty(1, hkv, sk, BPR, dtype=torch.uint8, device=dev)

            # correctness vs torch reference (in-range rows + V tail zero)
            k_ref, v_ref = ref_gather(kcf, vcf, bt, sl, sk, hkv)
            k_ok = torch.equal(kb[:, :, :LIVE], k_ref[:, :, :LIVE])
            v_ok = torch.equal(vb[:, :, :LIVE], v_ref[:, :, :LIVE])
            if not (k_ok and v_ok):
                raise SystemExit(f"correctness FAIL hkv={hkv} sk={sk} "
                                 f"k={k_ok} v={v_ok}")

            g_us = time_us(gather)
            quant = lambda: fa.quantize_q8_0(kb)
            quant()
            q_us = time_us(quant)
            del kq

            # fused-quant single kernel (the <= 65535 path): the small-Sk
            # reference the replacement kernel must not regress.
            fq_us = float("nan")
            if sk <= 65535:
                kqb = torch.empty(1, hkv, sk, BPR, dtype=torch.uint8,
                                  device=dev)
                fq = lambda: fa.gather_paged_kv_quantized(
                    kcf, vcf, bt, sl, sk, k_out=kqb, v_out=vb)
                fq_us = time_us(fq)
                del kqb

            # FA compute at LIVE width (not part of the tax): q fp32
            # [B, Hq, 1, D], kv_max = live.
            lsk = (LIVE + 31) // 32 * 32
            klf = torch.randn(1, hkv, lsk, D, dtype=torch.float16,
                              device=dev)
            vlf = torch.randn(1, hkv, lsk, D, dtype=torch.float16,
                              device=dev)
            lkq = fa.quantize_q8_0(klf)
            q = torch.randn(1, Hq, 1, D, dtype=torch.float32, device=dev)
            fwd = lambda: fa.forward(q, lkq, vlf, 0.04419417382415922,
                                     kv_max=sl)
            out = fwd()
            if not torch.isfinite(out.float()).all().item():
                raise SystemExit(f"FA forward non-finite hkv={hkv}")
            fa_us = time_us(fwd)

            # OOB work arithmetic (per layer): V-zero stores by the
            # gather, read-quantize-write by quantize_q8_0.
            oob_rows = hkv * (sk - LIVE)
            vzero_b = (sk // BLOCK - LIVE // BLOCK) * hkv * BLOCK * D * 2
            quant_b = oob_rows * (D * 2 + BPR)
            rows.append({
                "hkv": hkv, "sk": sk,
                "gather_us": g_us, "quant_us": q_us, "fq_us": fq_us,
                "fa_us": fa_us,
                "vzero_mbps": vzero_b / 1e6, "quant_mbps": quant_b / 1e6,
                "g_implicit_gbps": vzero_b / (g_us * 1e-6) / 1e9,
                "q_implicit_gbps": quant_b / (q_us * 1e-6) / 1e9,
            })
            del kb, vb, klf, vlf, lkq, out
            torch.cuda.empty_cache()

    print(f"{'Hkv':>3} {'Sk':>7} | {'gatherV2 us':>12} {'quant us':>9} "
          f"{'fused us':>9} | {'FA@live us':>11} | "
          f"{'gather+quant ms/step':>22} | {'vzero GB/s':>11} "
          f"{'quant GB/s':>11}")
    for r in rows:
        gq_ms = (r["gather_us"] + r["quant_us"]) / 1000 * N_LAYERS
        fq = f"{r['fq_us']:9.1f}" if r["fq_us"] == r["fq_us"] else "        -"
        print(f"{r['hkv']:3d} {r['sk']:7d} | {r['gather_us']:12.1f} "
              f"{r['quant_us']:9.1f} {fq} | {r['fa_us']:11.1f} | "
              f"{gq_ms:22.2f} | {r['g_implicit_gbps']:11.1f} "
              f"{r['q_implicit_gbps']:11.1f}")

    # S8 comparison: TP=2 (hkv=2) 262144 vs 131072 gather+quant delta.
    def gq(sk, hkv):
        return next(r for r in rows if r["sk"] == sk and r["hkv"] == hkv)
    d = (gq(262144, 2)["gather_us"] + gq(262144, 2)["quant_us"]
         - gq(131072, 2)["gather_us"] - gq(131072, 2)["quant_us"]) / 1000
    print(f"\nS8 check (hkv=2, x{N_LAYERS} layers): "
          f"262144-131072 gather+quant delta = {d:.2f} ms/step "
          f"(measured S8 gap: 8.4 ms/step)")
    a = gq(262144, 2)
    print(f"262144 hkv=2: gather {a['gather_us']:.0f} us + quant "
          f"{a['quant_us']:.0f} us = {(a['gather_us']+a['quant_us'])/1000*N_LAYERS:.1f} "
          f"ms/step of tail work; FA@live {a['fa_us']:.0f} us/layer "
          f"({a['fa_us']/1000*N_LAYERS:.1f} ms/step)")


if __name__ == "__main__":
    main()

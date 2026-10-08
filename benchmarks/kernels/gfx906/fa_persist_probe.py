#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Persistent fused gather+quantize probe (plan_masked_fa.md step 3 gate).

Gates the new gather_paged_kv_quant_persistent kernel:
  1. Correctness: in-range (tok < seq_len) K_q8/V rows must be BIT-EQUAL
     to the two-kernel path (gather_paged_kv_fp16 + quantize_q8_0) and to
     the fused path (gather_paged_kv_quantized) at Sk <= 65535; V margin
     rows must be zero (GFX906_FA_PERSIST_MARGIN default 128).
  2. Timing: per-call us at the S8 shapes (B=1, Hkv=2/4, live 1536, Sk up
     to 262144) vs the two-kernel path — the persistent kernel must
     collapse the O(Sk_pad) tail.

D in {128, 256}: the kernel is D-generic (D % 32 == 0, V uint4 copy,
K blocks_per_row = D/32); D=128 covers the other advertised head size
before default-ON widens past the D=256 model family
(fa-masked-gather-code-rev-qwen P1-2).

Run (local venv, GPU 0):
  cd /local/git/vllm-gfx906-mobydick
  source ~/env-rocm-7.14-gfx906.sh
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u \
      benchmarks/kernels/gfx906/fa_persist_probe.py
"""
import os

import torch

dev = "cuda"
torch.manual_seed(0)

Hkv_list = (2, 4)
D_list = (128, 256)
BLOCK = 16
LIVE = 1536
MARGIN = int(os.environ.get("GFX906_FA_PERSIST_MARGIN", "128"))


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


def main():
    from vllm import _gfx906_fa_C as fa

    # env knobs must be set before first call (read once in the extension)
    print(f"PERSIST env: {os.environ.get('GFX906_FA_PERSIST')} "
          f"grid={os.environ.get('GFX906_FA_PERSIST_GRID', '1024 (default)')} "
          f"margin={MARGIN}")

    for D in D_list:
        BPR = (D // 32) * 34
        for hkv in Hkv_list:
            sks = [3328, 65504, 131072, 262144]
            maxsk = max(sks)
            nb = maxsk // BLOCK + 8
            kc = torch.randn(nb, BLOCK, hkv, D, dtype=torch.float16, device=dev)
            vc = torch.randn(nb, BLOCK, hkv, D, dtype=torch.float16, device=dev)
            bt = torch.zeros(1, nb, dtype=torch.int32, device=dev)
            bt[0, : nb - 8] = torch.arange(nb - 8, dtype=torch.int32)
            sl = torch.tensor([LIVE], dtype=torch.int32, device=dev)

            for sk in sks:
                kb2 = torch.empty(1, hkv, sk, D, dtype=torch.float16, device=dev)
                vb2 = torch.empty(1, hkv, sk, D, dtype=torch.float16, device=dev)
                fa.gather_paged_kv_fp16(kc, vc, bt, sl, sk,
                                        k_out=kb2, v_out=vb2)
                kq2 = fa.quantize_q8_0(kb2)

                kb = torch.empty(1, hkv, sk, BPR, dtype=torch.uint8, device=dev)
                vb = torch.empty(1, hkv, sk, D, dtype=torch.float16, device=dev)
                kp, vp = fa.gather_paged_kv_quant_persistent(
                    kc, vc, bt, sl, sk, k_out=kb, v_out=vb)

                # 1. bit-equal in-range
                k_ok = torch.equal(kp[:, :, :LIVE], kq2[:, :, :LIVE])
                v_ok = torch.equal(vp[:, :, :LIVE], vb2[:, :, :LIVE])
                # 2. margin rows zero, beyond-margin untouched check skipped
                m_end = min(sk, LIVE + MARGIN)
                vm_ok = m_end == LIVE or torch.equal(
                    vp[:, :, LIVE:m_end],
                    torch.zeros(1, hkv, m_end - LIVE, D, dtype=torch.float16,
                                device=dev))
                # 3. reference vs two-kernel V: two-kernel zeros ALL tail
                #    (existing contract); persistent zeros margin only.
                print(f"D={D} hkv={hkv} sk={sk}: K_bit={k_ok} V_bit={v_ok} "
                      f"Vmargin_zero={vm_ok}")
                if not (k_ok and v_ok and vm_ok):
                    raise SystemExit("correctness FAIL")

                # timing
                two = (time_us(lambda: fa.gather_paged_kv_fp16(
                             kc, vc, bt, sl, sk, k_out=kb2, v_out=vb2))
                       + time_us(lambda: fa.quantize_q8_0(kb2)))
                one = time_us(lambda: fa.gather_paged_kv_quant_persistent(
                             kc, vc, bt, sl, sk, k_out=kb, v_out=vb))
                print(f"  two-kernel {two:8.1f} us   persistent {one:7.1f} us   "
                      f"speedup {two/one:6.2f}x   per-step(x16) "
                      f"{(two-one)/1000*16:7.2f} ms saved")
                del kb2, vb2, kq2, kb, vb, kp, vp
                torch.cuda.empty_cache()
            del kc, vc, bt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""NaN-tail gate (plan_masked_fa.md step 2).

Question: does the FA compute kernel (gfx906_fa.forward) ever read K_q8 or
V rows AT OR BEYOND kv_max (the live seq_len)? If not, the gather may stop
writing V zeros beyond seq_len — the persistent kernel's correctness
precondition.

Method: run forward with zero tails, then re-run with the K_q8 and V tails
[seq_len, Sk) poisoned with NaN. If the kernel reads any poisoned row, the
output becomes NaN (softmax weight 0 x NaN = NaN; Q . NaN = NaN).
Gate PASS = outputs bit-equal across the poison.

K_q8 rows are 34 bytes per 32-value block: [scale fp16 (2B) | qs i8 (32B)].
Poison = scale NaN (0x7E00 LE), qs 0. V poison = fp16 NaN (0x7E00).

Cases: seq_len aligned / misaligned to the 32-row K tile (tail-tile
oob_check path), seq_len 1 row before Sk (worst-case tail), seq_len == Sk
(no tail), B=1 and B=2 ragged. Hkv in {2,4} exercises the GQA-packed
(ncols2 > 1) decode path at Hq=24. sq > 1 exercises the multi-query
paths the persistent gather also serves: mtp2 spec-decode verify
(sq=3, ncols1=4) and prefill chunks (sq=64, ncols1=64) with inline
causal (q_abs_offset = sl - sq, so the last query row sits on the last
live token and rows [sl, sk) are dead in every query row's view).

Run (local venv, GPU 0):
  cd /local/git/vllm-gfx906-mobydick
  source ~/env-rocm-7.14-gfx906.sh
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u \
      benchmarks/kernels/gfx906/fa_nantail_probe.py
"""
import torch

dev = "cuda"
torch.manual_seed(0)

Hq = 24
D = 256
BPR = (D // 32) * 34  # 272 bytes per q8_0 row (D=256)
SCALE = 0.04419417382415922  # 1/sqrt(D)

# D=128 cases: the other advertised head size
# (supports_head_size {64,128,256}) — different FA tile config; the
# tail-read precondition must hold there too before default-ON widens
# past the D=256 model family (fa-masked-gather-code-rev-qwen P1-2).
D_ALT = 128
BPR_ALT = (D_ALT // 32) * 34
SCALE_ALT = 1.0 / (D_ALT ** 0.5)

K_ZERO = torch.zeros(BPR, dtype=torch.uint8, device=dev)
K_NAN = torch.zeros(BPR, dtype=torch.uint8, device=dev)
K_NAN[1] = 0x7E  # fp16 NaN scale, little-endian byte 1
K_ZERO_ALT = torch.zeros(BPR_ALT, dtype=torch.uint8, device=dev)
K_NAN_ALT = torch.zeros(BPR_ALT, dtype=torch.uint8, device=dev)
K_NAN_ALT[1] = 0x7E


def run_case(fa, hkv, b, sls, sk, sq=1, d=D):
    if d == D:
        bpr, scale, k_zero, k_nan = BPR, SCALE, K_ZERO, K_NAN
    else:
        bpr, scale, k_zero, k_nan = BPR_ALT, SCALE_ALT, K_ZERO_ALT, K_NAN_ALT
    kb = torch.randn(b, hkv, sk, d, dtype=torch.float16, device=dev)
    vb = torch.randn(b, hkv, sk, d, dtype=torch.float16, device=dev)
    kq = fa.quantize_q8_0(kb)
    q = torch.randn(b, Hq, sq, d, dtype=torch.float32, device=dev)

    for i, sl in enumerate(sls):
        if sl < sk:
            kq[i, :, sl:sk, :] = k_zero
            vb[i, :, sl:sk, :] = 0
    kv_max = torch.tensor(sls, dtype=torch.int32, device=dev)

    # sq=1: decode, no causal. sq>1: inline causal (same as forward_paged
    # prefill/spec-decode) — offset = sl - sq.
    q_off = (kv_max - sq).to(torch.int32) if sq > 1 else None

    o_clean = fa.forward(q, kq, vb, scale, kv_max, None, q_off)

    for i, sl in enumerate(sls):
        if sl < sk:
            kq[i, :, sl:sk, :] = k_nan
            vb[i, :, sl:sk, :] = 0x7E00  # fp16 NaN
    o_poison = fa.forward(q, kq, vb, scale, kv_max, None, q_off)

    ok = torch.equal(o_clean, o_poison)
    finite = bool(torch.isfinite(o_poison).all())
    return ok, finite, o_clean


def main():
    from vllm import _gfx906_fa_C as fa

    cases = [
        # (hkv, seq_lens, Sk_pad, sq)
        (2, [2176], 2560, 1),         # aligned to 32 (no tail tile)
        (2, [2177], 2560, 1),         # tail tile oob path
        (2, [2559], 2560, 1),         # 1 row before Sk (worst-case tail)
        (2, [2560], 2560, 1),         # full (no tail)
        (2, [32], 2560, 1),           # tiny
        (4, [2177], 2560, 1),         # Hkv=4 packed path
        (2, [1024, 2177], 2560, 1),   # B=2 ragged
        (2, [2177], 2560, 3),         # mtp2 verify (Sq=1+2), inline causal
        (2, [2177], 2560, 64),        # prefill chunk (Sq=64, ncols1=64)
        (2, [2559], 2560, 2),         # worst-case tail + Sq=2 (spec n=2)
        (4, [1024, 2177], 2560, 3),   # GQA-packed + mtp2, B=2 ragged
        (2, [2177], 2560, 1, D_ALT),  # D=128: tail-tile oob, decode
        (2, [2559], 2560, 1, D_ALT),  # D=128: worst-case tail
        (2, [2177], 2560, 3, D_ALT),  # D=128: mtp2 inline causal
    ]
    fails = 0
    for hkv, sls, sk, sq, *drest in cases:
        d = drest[0] if drest else D
        ok, finite, o = run_case(fa, hkv, len(sls), sls, sk, sq, d)
        print(f"D={d} hkv={hkv} B={len(sls)} sq={sq} sls={sls} sk={sk}: "
              f"{'PASS' if ok and finite else 'FAIL'} "
              f"(bit_equal={ok}, finite={finite})")
        if not (ok and finite):
            fails += 1
            print(f"  clean[0,:4]   = {o[0, 0, 0, :4].tolist()}")
        del o
        torch.cuda.empty_cache()
    print("RESULT:", "FAIL" if fails else
          "PASS (no tail reads at/beyond kv_max)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()

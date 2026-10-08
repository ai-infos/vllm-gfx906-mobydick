#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Capture/replay bit-exactness gate (plan_masked_fa.md step 4).

Capture the persistent gather + FA forward into a torch CUDA graph at the
frozen launch dim Sk_pad (like a FULL cudagraph capture: metadata built
once, seq_lens a persistent GPU tensor whose CONTENTS change per replay).
Replay at several live seq_lens; outputs must be bit-equal to eager
(no-graph) execution at the same seq_lens.

Gate PASS = all replays bit-equal and finite.

Run (local venv, GPU 0):
  cd /local/git/vllm-gfx906-mobydick
  source ~/env-rocm-7.14-gfx906.sh
  HIP_VISIBLE_DEVICES=0 .venv/bin/python -u \
      benchmarks/kernels/gfx906/fa_capture_replay_probe.py
"""
import torch

dev = "cuda"
torch.manual_seed(0)

Hq = 24
Hkv = 2
D = 256
BPR = (D // 32) * 34
BLOCK = 16
SK_PAD = 262144  # frozen graph launch dim (256k model)
SCALE = 0.04419417382415922


def eager_step(fa, q, kc, vc, bt, sl_t, sk, kbuf, vbuf):
    kp, vp = fa.gather_paged_kv_quant_persistent(
        kc, vc, bt, sl_t, sk, k_out=kbuf, v_out=vbuf)
    return fa.forward(q, kp, vp, SCALE, sl_t)


def main():
    from vllm import _gfx906_fa_C as fa

    # B=1..4: FULL capture sizes are [1,2,3,4] in the S5 serving config, so
    # the kernel's per-seq register prefix must hold for all of them.
    # nb = SK_PAD//BLOCK per seq (+ headroom) so that the sl=SK_PAD replays
    # materialize ALL rows: block_tab_idx must stay < max_blocks_per_seq,
    # else the kernel's OOB guard skips the row in every path (bit-equal
    # but unmaterialized) and the sweep is shallower than it looks.
    max_b = 4
    nb = SK_PAD // BLOCK * max_b + 16 * max_b
    kc = torch.randn(nb, BLOCK, Hkv, D, dtype=torch.float16, device=dev)
    vc = torch.randn(nb, BLOCK, Hkv, D, dtype=torch.float16, device=dev)
    fails = 0
    for b in [1, 2, 3, 4]:
        # Full-coverage block table: every token of [0, SK_PAD) maps to
        # its own physical block. (The pre-fix fill covered only off-16
        # entries/seq, so the last 256 tokens per seq aliased phys block
        # 0 through the zero-filled table tail — rows were written from
        # real (aliased) data, but not the seq's own blocks.)
        off = (nb - 16 * max_b) // max_b
        bt = torch.zeros(b, off, dtype=torch.int32, device=dev)
        for s_ in range(b):
            bt[s_, :] = torch.arange(
                s_ * off, (s_ + 1) * off, dtype=torch.int32)
        q = torch.randn(b, Hq, 1, D, dtype=torch.float32, device=dev)
        sl_t = torch.zeros(b, dtype=torch.int32, device=dev)

        kbuf = torch.empty(b, Hkv, SK_PAD, BPR, dtype=torch.uint8,
                           device=dev)
        vbuf = torch.empty(b, Hkv, SK_PAD, D, dtype=torch.float16, device=dev)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for v in [128, 1024]:
                sl_t.fill_(v)
                eager_step(fa, q, kc, vc, bt, sl_t, SK_PAD, kbuf, vbuf)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        sl_t.fill_(1024)  # capture-time live value (irrelevant to dims)
        with torch.cuda.graph(g):
            kp, vp = fa.gather_paged_kv_quant_persistent(
                kc, vc, bt, sl_t, SK_PAD, k_out=kbuf, v_out=vbuf)
            o_graph = fa.forward(q, kp, vp, SCALE, sl_t)
        print(f"B={b}: captured at Sk_pad={SK_PAD}, grid frozen")

        for sl in [128, 1536, SK_PAD - 32, SK_PAD]:
            sl_t.fill_(sl)
            g.replay()
            torch.cuda.synchronize()
            o_e = eager_step(fa, q, kc, vc, bt, sl_t, SK_PAD, kbuf, vbuf)
            ok = torch.equal(o_graph, o_e)
            finite = bool(torch.isfinite(o_graph).all())
            print(f"  B={b} sl={sl:>7}: "
                  f"{'PASS' if ok and finite else 'FAIL'} "
                  f"(bit_equal={ok}, finite={finite})")
            fails += 0 if (ok and finite) else 1
        del kbuf, vbuf, bt, q, sl_t
        torch.cuda.empty_cache()
    print("RESULT:", "FAIL" if fails else "PASS (capture/replay bit-exact)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()

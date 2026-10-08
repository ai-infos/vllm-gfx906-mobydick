#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""P3-3a Day-1 gate: fused KV gather + Q-fp32 micro-bench at serving shapes.

Measures gfx906_fa.gather_paged_kv_q8 in isolation at B=1 (single-request
decode, the _bench_gfx906.py serving config), using pre-allocated output
buffers — the steady-state serving path after warmup. Also measures the
per-FA-layer Q-fp32 side costs (q.float(), q-pad zero/copy, out unpack)
that ride along with every CUSTOM FA call.

Decision gate (plan-decode-phase3.md §4): if the gather kernel costs
> ~80 us/layer at Sk~2816, the 72 us FA kernel win is eaten by the
gather tax at B=1 and P3-3a stays suspended.

Model shape: Qwen3.5-35B-A3B — 10 FA layers, Hq=16, Hkv=2, D=256,
block_size=16, Q8 side-buffer bytes_per_row = (256/32)*34 = 272.

Run in the gfx906 vLLM image with the repo source-mounted:
  python3 -u /bench/bench_gfx906_fa_gather.py
"""
import torch

dev = "cuda"
torch.manual_seed(0)

Hq, Hkv, D, BLOCK = 16, 2, 256, 16
BPR = (D // 32) * 34  # 272 uint8 per Q8 row
SK_LIST = [2048, 2560, 2816, 3072, 3328]
HBM_BW = 798e9  # P3-0 Q1: measured MI50 HBM read BW
GATE_US = 80.0  # plan §4: gather > ~80 us/layer -> P3-3a stays suspended


def time_us(fn, warmup=20, iters=200):
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


def ref_gather(kc, vc, bt, sl, sk):
    """Torch reference mirroring _gather_kv_q8 (gfx906_fa_paged.py)."""
    nb = (sk + BLOCK - 1) // BLOCK
    bt_l = bt[:, :nb].long()
    k = kc[bt_l].view(1, -1, Hkv, BPR)[:, :sk]
    v = vc[bt_l].view(1, -1, Hkv, D)[:, :sk]
    pos = torch.arange(sk, device=dev)
    m = (pos[None, :] < sl[:, None]).view(1, sk, 1, 1).to(torch.float16)
    v = v * m
    return (k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous())


def main():
    from vllm import _gfx906_fa_C as fa

    num_blocks = (max(SK_LIST) + BLOCK - 1) // BLOCK + 8
    kc = torch.randint(0, 255, (num_blocks, BLOCK, Hkv, BPR),
                       dtype=torch.uint8, device=dev)
    vc = torch.randn(num_blocks, BLOCK, Hkv, D, dtype=torch.float16, device=dev)
    bt_full = torch.zeros(1, num_blocks, dtype=torch.int32, device=dev)
    bt_full[0] = torch.arange(num_blocks, dtype=torch.int32, device=dev)

    # -------- correctness (fused vs torch reference) --------
    # Tail test: seq_lens = Sk-15 leaves 15 masked rows inside the last block.
    sk_t = 2816
    sl_t = torch.tensor([sk_t - 15], dtype=torch.int32, device=dev)
    k_ref, v_ref = ref_gather(kc, vc, bt_full, sl_t, sk_t - 15)
    ko = torch.empty(1, Hkv, sk_t, BPR, dtype=torch.uint8, device=dev)
    vo = torch.empty(1, Hkv, sk_t, D, dtype=torch.float16, device=dev)
    k_f, v_f = fa.gather_paged_kv_q8(kc, vc, bt_full, sl_t, sk_t,
                                     k_out=ko, v_out=vo)
    k_ok = torch.equal(k_f[:, :, :sk_t - 15], k_ref)
    v_ok = torch.equal(v_f[:, :, :sk_t - 15], v_ref)
    tail_zero = bool((v_f[:, :, sk_t - 15:] == 0).all())
    print(f"correctness: K==ref {k_ok}  V==ref {v_ok}  V_tail_zero {tail_zero}")
    assert k_ok and v_ok and tail_zero, "fused gather disagrees with reference"

    # -------- gather kernel timing, steady state (buffer reuse) --------
    print(f"\ngather_paged_kv_q8  B=1 Hkv={Hkv} D={D} bs={BLOCK} "
          f"(pre-allocated out buffers)")
    print(f"{'Sk':>6} {'us/call':>9} {'MB moved':>10} {'GB/s':>7} "
          f"{'floor_us':>9} {'x_floor':>8}")
    gather = {}
    for sk in SK_LIST:
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        ko = torch.empty(1, Hkv, sk, BPR, dtype=torch.uint8, device=dev)
        vo = torch.empty(1, Hkv, sk, D, dtype=torch.float16, device=dev)
        bt = bt_full[:, : (sk + BLOCK - 1) // BLOCK]

        def call():
            fa.gather_paged_kv_q8(kc, vc, bt, sl, sk, k_out=ko, v_out=vo)

        us = time_us(call)
        moved = 2 * Hkv * sk * (BPR + D * 2)  # read+write, K_q8 + V_fp16
        floor = moved / HBM_BW * 1e6
        gather[sk] = us
        print(f"{sk:>6} {us:>9.1f} {moved/1e6:>10.2f} "
              f"{moved/us/1e3:>7.0f} {floor:>9.1f} {us/floor:>8.1f}")

    # -------- Q-fp32 side costs per FA layer (decode Sq=1) --------
    q16 = torch.randn(1, Hq, D, dtype=torch.float16, device=dev)
    q32 = q16.float()
    q_pad = torch.empty(1, Hq, 2, D, dtype=torch.float32, device=dev)
    out_pad = torch.randn(1, Hq, 2, D, dtype=torch.float32, device=dev)

    us_cast = time_us(lambda: q16.float())
    us_zero = time_us(q_pad.zero_)
    us_qcopy = time_us(lambda: q_pad[:, :, :1, :].copy_(q32.unsqueeze(2)))
    us_unpack = time_us(
        lambda: out_pad[:, :, 0, :].reshape(1, Hq * D).contiguous())
    side = us_cast + us_zero + us_qcopy + us_unpack
    print("\nQ-fp32 side costs per FA layer (Sq=1):")
    print(f"  q.float()       {us_cast:6.1f} us")
    print(f"  q_pad.zero_()   {us_zero:6.1f} us")
    print(f"  q copy into pad {us_qcopy:6.1f} us")
    print(f"  out unpack      {us_unpack:6.1f} us")
    print(f"  side total      {side:6.1f} us/layer")

    # -------- LEGACY=1 serving path: torch _gather_kv (fp16 K) --------
    # The default serving path (GFX906_FA_LEGACY=1) gathers fp16 K + fp16 V
    # via PyTorch fancy indexing with mask+permute copies; the P3-0/v10
    # serving profile suggests it costs far more than the fused Q8 gather.
    from vllm.gfx906_fa.gfx906_fa_paged import _gather_kv

    k16 = torch.randn(num_blocks, BLOCK, Hkv, D, dtype=torch.float16,
                      device=dev)
    print("\nLEGACY torch _gather_kv  B=1 Hkv=2 D=256 (fp16 K + fp16 V, "
          "mask+permute)")
    print(f"{'Sk':>6} {'us/call':>9} {'vs fused q8':>12}")
    for sk in SK_LIST:
        sl = torch.tensor([sk], dtype=torch.int32, device=dev)
        bt = bt_full[:, : (sk + BLOCK - 1) // BLOCK]

        def call_legacy():
            _gather_kv(k16, vc, bt, sl, sk)

        us = time_us(call_legacy)
        print(f"{sk:>6} {us:>9.1f} {us/gather[sk]:>11.1f}x")

    # -------- gate verdict --------
    sk_g = 2816
    g = gather[sk_g]
    print(f"\nGATE (plan §4, P3-3a): gather @ Sk={sk_g} = {g:.1f} us/layer "
          f"vs ~{GATE_US:.0f} us threshold")
    print(f"  + Q-side {side:.1f} us/layer -> combined {g + side:.1f} us/layer")
    if g <= GATE_US:
        print("  => gather <= threshold: P3-3a may RESUME (M1 work, per plan)")
    else:
        print("  => gather > threshold: P3-3a stays SUSPENDED")


if __name__ == "__main__":
    main()

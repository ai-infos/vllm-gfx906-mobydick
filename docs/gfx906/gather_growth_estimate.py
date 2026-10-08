#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Re-derive the pre-fix _gather_retired growth (plan §4 gate 1).

The fix plan must not cite a retire-dict figure that is not
reproducible from a committed script. This computes the cumulative
bytes kept alive by the PRE-FIX policy (exact-Sk match + sticky
_gather_captured latch: every replaced generation is retired) for the
256k Qwen3.8-27B TP=2 run-4 config (chunk 1024, max_num_seqs 2).

Per generation:
    K: B x Hkv x Sk_pad x (D/32)*34 bytes   (uint8 q8_0)
    V: B x Hkv x Sk_pad x D x 2 bytes       (fp16)

The chunk i prefill carries Sk = i*chunk tokens (last chunk short);
the capture-time generation is B = max_num_seqs at width
max_model_len.

Usage:
    python3 gather_growth_estimate.py            # run-4 defaults
    python3 gather_growth_estimate.py --chunk 8192 --tokens 250000
"""

import argparse


def gen_bytes(b: int, hkv: int, sk: int, d: int = 256) -> int:
    return b * hkv * sk * ((d // 32) * 34 + d * 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk", type=int, default=1024,
                   help="max_num_batched_tokens (run 4: 1024)")
    p.add_argument("--tokens", type=int, default=250_000,
                   help="prompt length of the needle request")
    p.add_argument("--max-model-len", type=int, default=262_144)
    p.add_argument("--hkv", type=int, default=2,
                   help="kv heads per rank (27B: 4 heads / TP2)")
    p.add_argument("--b-serving", type=int, default=1,
                   help="num_seqs of the prefill forward (single request)")
    p.add_argument("--b-capture", type=int, default=2,
                   help="B of the capture-time generation (max_num_seqs)")
    a = p.parse_args()

    pad = lambda x: ((x + 31) // 32) * 32
    capture_gen = gen_bytes(a.b_capture, a.hkv, pad(a.max_model_len))
    total = capture_gen
    n = 0
    i = 0
    sk = 0
    while sk < a.tokens:
        i += 1
        sk = min(i * a.chunk, a.tokens)
        g = gen_bytes(a.b_serving, a.hkv, pad(sk))
        total += g
        n += 1
        if i % 25 == 0 or sk == a.tokens:
            print(f"chunk {i:3d}  Sk={sk:7d}  gen={g / 2**20:9.2f} MiB  "
                  f"cum_retired={total / 2**30:8.2f} GiB")
    print(f"\n{a.tokens} tokens, chunk {a.chunk}: {n} serving generations "
          f"+ 1 capture generation ({capture_gen / 2**20:.1f} MiB)")
    print(f"pre-fix total retired (B={a.b_serving} serving, "
          f"B={a.b_capture} capture): {total / 2**30:.2f} GiB")
    print(f"post-fix total retired: {capture_gen / 2**30:.2f} GiB "
          f"(the capture-time generation only)")


if __name__ == "__main__":
    main()

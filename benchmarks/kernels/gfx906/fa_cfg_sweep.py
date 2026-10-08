#!/usr/bin/env python3
"""Config sweep for the FA decode kernel AT THE k=4 VERIFY SHAPE (Sq_pad=8).

All prior NC2/KVSPLIT sweeps ran at Sq=2 (greedy), where the kernel is
bandwidth-saturated and config barely matters. At Sq=8 (ncols1=8) it is
compute-bound (~46% of HBM floor @120k, see PROFILE-k4-step-ledger.md §5),
so the config space is unexplored. This driver runs the standalone bench
one-process-per-config (the extension reads GFX906_FA_* at import) and
collects the reported us/layer into a CSV.

Env: SWEEP_SKS (default "98304,122880"), SWEEP_NC2 ("1,2,4"),
     SWEEP_KVSPLIT ("1,2,4,8"), BENCH_FA_SQ (default 8).
"""
import csv
import os
import subprocess
import sys

REPO = "/local/git/vllm-gfx906-mobydick"
BENCH = f"{REPO}/benchmarks/kernels/gfx906/bench_gfx906_fa_decode_qwen27.py"
PY = f"{REPO}/.venv/bin/python"
OUT = "/local/tmp/mtp1/fa_cfg_sweep_sq8.csv"

SWS = [int(x) for x in os.environ.get("SWEEP_SKS", "98304,122880").split(",")]
NC2S = [int(x) for x in os.environ.get("SWEEP_NC2", "1,2").split(",")]  # Hq/Hkv=6: nc2 must divide 6 (launcher guard)
KSPLITS = [int(x) for x in os.environ.get("SWEEP_KVSPLIT", "1,2,4,8").split(",")]


def main():
    rows = []
    n = len(SWS) * len(NC2S) * len(KSPLITS)
    i = 0
    for sk in SWS:
        for nc2 in NC2S:
            for kv in KSPLITS:
                i += 1
                env = dict(os.environ,
                           BENCH_FA_SQ=os.environ.get("BENCH_FA_SQ", "8"),
                           BENCH_FA_SK=str(sk),
                           GFX906_FA_NC2=str(nc2),
                           GFX906_FA_KVSPLIT=str(kv))
                print(f"[{i}/{n}] Sk={sk} NC2={nc2} KVSPLIT={kv}", flush=True)
                try:
                    r = subprocess.run([PY, "-u", BENCH], env=env,
                                       cwd=REPO, capture_output=True,
                                       text=True, timeout=600)
                    out = r.stdout + r.stderr
                    # bench line: "Sk=122880:    976.7 us   eff  1183.6 GB/s (148.3% of HBM)   floor ..."
                    mline = next((l for l in out.splitlines()
                                  if l.startswith("Sk=")), "")
                    import re as _re
                    um = _re.search(r"Sk=\d+:\s*([\d.]+)\s*us", mline)
                    hm = _re.search(r"\(([\d.]+)% of HBM\)", mline)
                    us = um.group(1) if um else ""
                    floor = (hm.group(1) + "%") if hm else ""
                    print(f"   -> {mline.strip()}", flush=True)
                except subprocess.TimeoutExpired:
                    line, us, floor = "TIMEOUT", "", ""
                    print("   -> TIMEOUT", flush=True)
                rows.append({"Sk": sk, "NC2": nc2, "KVSPLIT": kv,
                             "us_per_layer": us, "hbm_floor_note": floor})
                with open(OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    print(f"\nSWEEP DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

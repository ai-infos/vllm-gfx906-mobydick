# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""G1 probe: per-node replay cost of a captured decode graph (gfx906).

Motivation (ROADMAP G1): the 2026-08-29 LEGACY adjudication left ~1.55 ms/step
unexplained, with extra captured-graph nodes (16-32/decode step) as the
leading unmeasured hypothesis. This probe measures the wall-clock replay cost
per graph node directly:

  * build a decode-shaped graph: LAYERS layers x KNODES kernels (~120 nodes,
    the order of magnitude of a real 40-layer MoE decode step's routing+FA
    chain), each kernel doing real small work (matmul / elementwise);
  * re-capture with N dummy no-op Triton launches appended per layer for
    N in {0, 16, 32, 64};
  * replay each graph WARM + ITERS times and measure host wall time per
    replay (with a device sync per iteration — the vLLM decode-step shape).

Output: one line per N with mean/median us/replay; the final line reports the
slope in us/node. Under torchrun (TP=2), an allreduce after each layer's work
mimics the TP=2 graph's inter-layer collective placement.

Run:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/gfx906/ \
      g1_node_replay_probe.py            # single GPU (TP=1)
  bash /local/tmp/g1_tp2_launch.sh       # TP=2 shape (allreduce per layer);
                                         # pins each rank to its own GPU
                                         # before torch import (ROCm 7.14's
                                         # device_count() misreports 0, so
                                         # torchrun-style indexing fails).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _noop_kernel(x_ptr):
    tl.store(x_ptr, 1.0)


LAYERS = 40          # Qwen3.5-35B MoE layer count (decode shape)
KNODES = 3           # work kernels per layer (matmul + add + scale)
SINK_KERNELS = 3     # w.float() cast + .sum() reduce + in-place add to sink
N_DUMMY_STEPS = (0, 16, 32, 64)
WARM = 50
ITERS = 300


def build_work(device: torch.device):
    """Real small work per kernel slot: alternating matmul / elementwise."""
    a = torch.randn(512, 512, device=device, dtype=torch.float16)
    b = torch.randn(512, 512, device=device, dtype=torch.float16)
    x = torch.randn(512, device=device, dtype=torch.float16)
    return a, b, x


def run_layer(a, b, x, out_sink):
    """KNODES kernels of real work; returns tensors feeding the next layer."""
    y = torch.matmul(a, b)            # kernel 1 (512x512 matmul)
    z = x + y[0]                      # kernel 2 (elementwise add, 512)
    w = z * 0.5                       # kernel 3 (scale, 512)
    out_sink += w.float().sum()       # keep the graph honest (no DCE)
    return a, b, x


def capture_graph(n_dummy: int, device: torch.device, dist=None):
    a, b, x = build_work(device)
    sink = torch.zeros((), device=device)
    dummy_buf = torch.zeros(1, device=device)

    def body():
        nonlocal a, b, x
        for _ in range(LAYERS):
            a, b, x = run_layer(a, b, x, sink)
            if dist is not None:
                t = sink.unsqueeze(0).float()
                dist.all_reduce(t)    # TP=2 shape: collective per layer
            if n_dummy:
                for _ in range(n_dummy):
                    _noop_kernel[(1,)](dummy_buf)

    # warm the kernels (Triton JIT etc.) outside capture
    body()
    torch.cuda.synchronize(device)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g


def measure(g: torch.cuda.CUDAGraph, device: torch.device) -> dict:
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize(device)
    times = []
    import time as _time

    for _ in range(ITERS):
        t0 = _time.perf_counter()
        g.replay()
        torch.cuda.synchronize(device)
        times.append((_time.perf_counter() - t0) * 1e6)  # us, host wall incl. sync
    return {
        "mean_us": statistics.fmean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
    }


def main():
    dist = None
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Launched by /local/tmp/g1_tp2_launch.sh: each rank is pinned to its
        # own physical GPU via HIP_VISIBLE_DEVICES *before* torch import, so
        # "cuda" (cuda:0) in this rank's view IS its own GPU. We init NCCL
        # WITHOUT device_id because this ROCm 7.14 stack misreports
        # torch.cuda.device_count()==0, which makes the device_id validation
        # in init_process_group reject even cuda:0. Tensors are created on the
        # rank-local default device; NCCL uses it per op.
        import torch.distributed as _dist

        device = torch.device("cuda")
        _dist.init_process_group(backend="nccl")
        dist = _dist
    else:
        # Non-distributed: the caller pins the GPU via HIP_VISIBLE_DEVICES
        # (a visibility mask, per repo convention), so this process sees
        # exactly one device — always "cuda" in its view.
        device = torch.device("cuda")

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"G1 node-replay probe: LAYERS={LAYERS} KNODES={KNODES} "
              f"base_nodes={LAYERS * KNODES} tp={'2' if dist else '1'}", flush=True)

    results = {}
    # per-layer real kernel count (work + sink bookkeeping); the dummy nodes
    # are what we vary, so only LAYERS*n enters the slope.
    base_nodes = LAYERS * (KNODES + SINK_KERNELS)
    for n in N_DUMMY_STEPS:
        g = capture_graph(n, device, dist)
        m = measure(g, device)
        nodes = base_nodes + (LAYERS if dist else 0) + LAYERS * n
        results[n] = m
        if rank == 0:
            print(f"N={n:3d}  nodes={nodes:4d}  mean={m['mean_us']:9.1f} us/replay  "
                  f"median={m['median_us']:9.1f}  min={m['min_us']:9.1f}", flush=True)
        del g
        torch.cuda.empty_cache()

    if rank == 0:
        base = results[0]["mean_us"]
        for n in N_DUMMY_STEPS[1:]:
            dn = LAYERS * n
            slope = (results[n]["mean_us"] - base) / dn
            print(f"slope N=0->{n}: {slope:.2f} us/node "
                  f"({LAYERS * n} nodes added, "
                  f"+{results[n]['mean_us'] - base:.1f} us/replay)", flush=True)
        if dist is not None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

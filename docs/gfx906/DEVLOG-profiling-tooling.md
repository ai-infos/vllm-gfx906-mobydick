# Profiling tooling — rocprofv3 / torch-profiler / py-spy on gfx906

**VERDICT:** OPEN (safe paths established; torch profiler BANNED until
the host RAM upgrade) · **GATE:** n/a (tooling capability record)
**Branch:** gfx906/fa-decode-fp16 (2026-09-11 → 2026-09-12)
**Operational rules live in:** `.agents/skills/gfx906-rocprofv3-kernel-trace/SKILL.md`
**Full detail:** stall-log §13.17/§13.20–§13.25 pre-restructure

## 2026-09-11 — rocprofv3 + rocpd WORKS (single-process)

`rocprofv3 -d <dir> --output-format rocpd --kernel-trace --stats -- <app>`
with a clean-exit flush window collects kernel dispatches on gfx906
(rocpd = sqlite; query recipe in the skill). **vLLM-process collection
is EMPTY** (3 documented attempts: TP=2 serve, TP=1 in-process, any flag
set) — do not assume the build is broken; the focused single-process
microbench is the workaround. The §13.1 "no libkineto" diagnosis was
RETRACTED (kineto statically linked; `-DUSE_KINETO -DHAS_ROCTRACER`
present) — the earlier AGENTS.md stale note stands for GPU-domain
traces: torch-profiler chrome traces on this build carry NO GPU-domain
kernel events.

## 2026-09-12 — CPU-side attribution method (cracked M3)

Torch-profiler chrome traces (offline in-process path) carry cpu_op +
python_function + user_annotation. **`aten::item` durations = D2H-sync
stalls**: bimodal p50 ~µs / p90 ~70 ms (instant vs full-backlog drain);
total item-time ≈ host stall. Containment analysis against
python_function frames (same-tid filter; frame names carry the def
line, not the executing line) located the M3 syncs in one pass. This
method + the export hazards are documented in the skill.

## 2026-09-12 — THE THREE FREEZES → torch profiler BANNED

Host-fatal on this torch/ROCm build with vLLM's TP=2 multiprocess
executor, at any window length, on any stack:
- Y12: 120-s offline window → export thrash (power-cycle).
- Y13: 20-s serving window → worker export at SIGTERM teardown
  (power-cycle); only the driver-side 460-KB trace materialized.
- Y14: 20-s offline window → `start_profile` hung the engine →
  shm_broadcast wedge → journald starvation (power-cycle).

Also: py-spy is YAMA-blocked for non-child attach (ptrace_scope=1);
`--nonblocking` false-positives "process exited" at the multiprocessing
spawn; blocking + `--idle` required for stall hunts (and even then the
Y13 stall stacks were missed).

## Safe paths until the RAM upgrade

- rocprofv3 single-process microbenches (never caused an incident).
- /metrics counter polling (RAM-safe; ~10-s publisher cadence — fine
  for phase decomposition, not per-step).
- Offline in-process runs WITHOUT the profiler.
- Per-TID CPU sampler + wchan (run_stall4 harness) — host-side only.

## RAM/swap answer (Kevin's question, 2026-09-12)

More RAM (→64 GB) = YES — re-enables the profiler (the export transient
is 10–20 GB), removes the 19.6-GB checkpoint-staging warning. More swap
= NO (4 GB is enough; more swap converts an OOM kill into a thrash
hang — exactly the observed failure; consider earlyoom/systemd-oomd
instead). Stopping background services = marginal (~3.4 GB baseline).
The Y12/Y13/Y14 freeze journal entries are in the respective
`journalctl -b -N` outputs; PSI-critical lines from hermes/kanban mark
onset in all three.

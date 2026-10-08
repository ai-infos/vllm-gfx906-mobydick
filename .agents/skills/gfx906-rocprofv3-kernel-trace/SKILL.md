---
name: gfx906-rocprofv3-kernel-trace
description: "Profile vLLM/torch GPU kernels on the gfx906 MI50 stack (ROCm 7.14, rocprofiler-sdk 1.3.2) with rocprofv3 + rocpd output. Use when per-kernel GPU attribution is needed (which kernel family eats time in a prefill/decode step), when torch-profiler/kineto GPU traces come up empty, when rocprofv3 'output generation' runs empty, or before attempting a torch rebuild for profiling reasons. Records the verified-working invocation, the vLLM multiprocess failure mode, and the focused-microbench workaround. Validated 2026-09-11; CPU-side attribution + export-RAM hazard added 2026-09-12."
---

# rocprofv3 GPU kernel tracing for vLLM on gfx906

## TL;DR — the verified-working invocation

```bash
# single-process torch app (verified: 16 kernel dispatch records, names + durations)
setsid nohup rocprofv3 -d <outdir> --output-format rocpd --kernel-trace --stats -- \
  env FLUSH_WAIT=20 <venv>/bin/python tiny_mm.py > run.log 2>&1 < /dev/null &
# tiny_mm.py must END with a flush window: time.sleep(float(os.environ["FLUSH_WAIT"]))
```

- Output: `<outdir>/mi50-01/*_results.db` — **rocpd IS a sqlite DB**
  (`rocpd_kernel_dispatch`, `rocpd_info_kernel_symbol`, `rocpd_string`, ...).
- `--kernel-trace` (and/or `--stats`) is REQUIRED: without it, rocprofv3
  collects nothing and "output generation" runs ~0.000 s (two documented
  empty runs). With it: DB materializes at teardown.
- The app must exit CLEANLY and hold the process tree alive ~15-20 s so the
  per-process buffers flush (see the teardown race below).
- gfx906 IS supported by rocprofiler-sdk 1.3.2 on this stack:
  `rocprofv3-avail list` recognizes it; verified with a hipBLASLt GEMM
  (`Cijk_...ISA906...` kernel, 13 launches captured).

## Per-kernel query (rocpd = sqlite)

```python
import sqlite3
c = sqlite3.connect("<outdir>/mi50-01/<pid>_results.db")
S = "<rocpd_uuid_suffix>"   # read from the table names
kd, ks = f"rocpd_kernel_dispatch_{S}", f"rocpd_info_kernel_symbol_{S}"
for r in c.execute(f"""
    select ks.kernel_name, count(*), round(avg(k.end-k.start)/1e3,1),
           round(sum(k.end-k.start)/1e6,2)
    from {kd} k join {ks} ks on k.kernel_id = ks.id
    group by ks.kernel_name order by sum(k.end-k.start) desc limit 20"""):
    print(r)
```

## The vLLM-process failure mode (OPEN — do not assume the build is broken)

Running the whole `vllm serve` / in-process vLLM under rocprofv3 (any flag
set, including the verified one, TP=2 multiprocess AND TP=1
VLLM_ENABLE_V1_MULTIPROCESSING=0 clean-exit) produced **zero records**
("output generation :: 0.0000xx s"). Three documented attempts
(2026-09-11 17:37/17:58/18:06). Candidates: per-process buffer loss on
vLLM's force-kill teardown; triton/HIP-module kernel registration;
record-volume limits. **Do NOT conclude "rocprof is broken on gfx906" from
this — the single-process case above proves the tool works.**

**Workaround that works today: the focused microbench.** Isolate the kernel
family in a single-process script and profile that. Example (the H2
follow-up): build the paged KV + Q tensors directly and call
`vllm/gfx906_fa/gfx906_fa_paged.py::forward_paged` in a loop — dozens of
launches, clean exit, rocpd flushes. Sweep the context length A at fixed
chunk to measure per-token×A kernel-time terms directly.

**Other GPU-visibility tools that DO work on this stack** (for attribution
without rocprof):
- vLLM torch-profiler CPU op table (`profiler_config={"profiler":"torch"}`):
  host-side op counts/times — confirms call structure (e.g. 48 GDN layers ×
  steps), but is host-LAUNCH-distorted under enforce_eager (159 s CPU ≈
  166 s wall) — NOT usable for GPU time.
- Per-TID CPU sampler + `wchan` (see run_stall4_session.sh): host-side.
- Backend A/B by difference: force `TRITON_ATTN`/`ROCM_ATTN` for the
  full-attn layers (see rocm.py potential-backends list) and compare step
  walls — attributes residual cost to the custom kernel vs elsewhere.
- The old kineto GPU-trace route is DEAD on this torch build for vLLM
  processes (§13.1) AND the "no libkineto in torch/lib" §13.1 diagnosis was
  WRONG: `torch.__config__.show()` shows `-DUSE_KINETO -DHAS_ROCTRACER`
  (kineto statically linked into libtorch_cpu — never a separate .so).
  Runtime collection still fails with kineto (cause open); the 1-min
  diagnostic is `KINETO_LOGLEVEL=5 KINETO_TRACE=1` on the standalone repro.

## Torch-profiler CPU-side attribution (works; RAM-capped) — 2026-09-12

The current torch build's chrome traces carry **NO GPU-domain kernel
events** (verified again on a 4×122880 in-process mtp3 run: only
cpu_op / python_function / user_annotation cats). GPU per-kernel
attribution stays parked. **But the CPU-side data is a real attribution
tool** — it located the M3 residual in one pass:

- **`aten::item` durations = D2H-sync stalls.** Distribution shape is
  the signature: p50 ~µs (queue empty, instant) with a heavy p90+ tail
  (~70–120 ms = blocking until the queued GPU work drains). Total
  item-time ≈ the host-side stall. Worked example: 10,959 aten::item =
  112.5 s of a 120-s window, 7,956 of them inside `forward_paged`
  spans → the per-seq `int(cu_seqlens_q[...])` syncs (M3, fixed).
- **Stack attribution**: python_function events (6M in a 120-s window)
  allow containment analysis — an op's enclosing frames are the
  python_function events with `ts <= op.ts <= ts+dur` on the **SAME
  tid** (filter by tid! process-wide containment cross-contaminates
  from monitor threads). Frame names carry the **def line, not the
  executing line** — they tell you WHICH function, never the statement.
- Traces land as `.pt.trace.json.gz` → `gzip.open(path, "rt")`.

**RAM HAZARD — trace export can freeze the whole host.** The profiler
buffers events in RAM and serializes at stop/exit: a 120-s window on
the campaign shape (4×122880, mtp3, TP=2) built multi-GB tables per
worker; the simultaneous export pushed the 31-GB host into PSI-critical
thrash (hermes logged it, systemd-logind timed out on session creation,
machine unresponsive → power-cycle; NO kernel OOM, journal otherwise
clean). Rules: **cap windows ≤30 s** for campaign-scale shapes on this
31-GB host; expect export RAM ≈ 20–40× the on-disk gz size; stagger
per-worker exports or profile one rank; save traces to persistent disk
(/local/tmp — /tmp is tmpfs and dies with the power cycle you may cause).
Incident: `docs/gfx906/ttft-prefill-stall.md` §13.21.

**2026-09-12 update (incident §13.24): it is worse than window length.**
On the vLLM SERVING stack, `stop_profile` returns 200 but does not
reliably export the worker-side tables — they flush at SIGTERM teardown
instead, and that export froze the host (PSI critical, power-cycle) even
for a 20-s window. The only trace materialized was the driver-side
`async_llm` component (python frames only, no kernels — no value).
**Rule: do not attach the torch profiler to the vLLM serving stack on
this box at all.** CPU-side `aten::item` attribution stays valid on the
OFFLINE in-process path (LLM() in one process — exports cleanly).
Also: put `ignore_eos: true` on D1c-style holder requests — a greedy
run hit EOS at 322/4096 tokens (kv-split near-tie nondeterminism) and
invalidated the probe timing.

**2026-09-12 final update: the profiler is BANNED on this box until the
host RAM is upgraded (31 GB → more).** Three freezes in one day, three
different failure points: 120-s window → export thrash (boot Y12);
20-s window → worker export at SIGTERM teardown (boot Y13); 20-s window
→ `start_profile` itself hung the engine → shm_broadcast wedge →
journald starvation (boot Y14). Window length was never the variable;
the profiler machinery with vLLM's TP=2 multiprocess executor is
host-fatal on this torch/ROCm build. GPU attribution = rocprofv3
single-process microbenches (never caused an incident) + /metrics
counter polling (RAM-safe) + offline in-process runs WITHOUT the
profiler. Counter-polling decomposition example: the D1c residual is a
batch-level collapse under mixed prefill+decode admission (probe ttft
7.1 s, holder 40 → 1.6 t/s), not a per-layer tax — found with zero RAM
exposure (§13.25).

## Flag semantics learned (rocprofv3 1.3.2)

| flag | behavior |
|---|---|
| `--output-format rocpd` | rocpd sqlite DB (sqlite3 format itself was DROPPED in v3; csv/json/pftrace/otf2/rocpd remain) |
| `--kernel-trace` | REQUIRED for kernel-dispatch collection; with rocpd also emits per-process raw `kernel_trace.dat` |
| `--stats` | per-kernel stats summary; include it (verified-working set) |
| `-d <dir>` | output dir for the DB |
| `--output-format csv` | works, but CSV historically broke on teardown crashes — prefer rocpd; csv+kernel-trace verified to materialize on a clean-exit run |
| `--pmc <counters>` | hardware counters (AMD-Skills wrapper uses GPU_UTIL/MfmaUtil/BANDWIDTH_EA on MI100+); gfx906 pmc support via rocprofiler-sdk UNVERIFIED |
| (none of kernel-trace/stats) | EMPTY collection — the default collects nothing |

`ROCPROFILER_LOG=1` did NOT produce /tmp/rocprofiler_log.txt on this stack
in the one test run (env may be rocprofiler-legacy-era; revisit if needed).

## Teardown race

vLLM's shutdown force-kills the EngineCore/worker children; rocprofv3's
signal handler then finalizes with their buffered records LOST (observed:
"force killing remaining process EngineCore" at T-9 s before the 0.246-s
rocpd flush of a LATER clean run). Mitigation: the app holds the process
tree alive at exit (flush window ≥15 s) so children exit cleanly and
buffers flush. For `vllm serve` (long-lived) send SIGTERM to the SERVER pid
only after the traced request completes, and prefer in-process/TP=1 shapes.

## Scripts (this dir)

- `tiny_mm.py` — minimal sanity app (matmul + flush window). Run FIRST to
  verify rocprofv3+rocpd collection on the current stack.
- `run_rp3_test.sh` — the vLLM in-process attempt wrapper (kb_probe.py under
  rocprofv3; currently hits the empty-collection failure — kept for whoever
  debugs it).
- `kb_probe.py` — in-process vLLM prefill probe with env knobs:
  `KB_TP` (default 2), `KB_EAGER=1` (skips inductor; TP=1 hits an inductor
  "User compiler error" without it), `KB_MAXLEN` (TP=1 needs 65536: 7.44 GiB
  KV available vs 9.29 required at 131072), `KB_PREFILL`, `PROF_PP`,
  `KB_FLUSH_WAIT`, and `VLLM_ENABLE_V1_MULTIPROCESSING=0` recommended.

## Process-group safety

Launch long sessions detached from the agent shell (`setsid nohup ... <
/dev/null &`) — tool aborts/timeouts kill the agent's process GROUP and a
plain `nohup &` child dies with it (cost a D1 session on 2026-09-11). See
`/local/git/AGENTS.md`.

## References

- `/local/git/AMD-Skills/` (cloned 2026-09-11): `rocprofv3-profiler` (PMC
  wrapper + bottleneck classifier, MI100/MI200/MI300-oriented — pmc on
  gfx906 unverified), `rocm-profiler-analysis` (vLLM/SGLang triage).
- ROCm/rocprofiler-sdk issue #137: AMD-recommended vLLM pattern
  (`rocprofv3 --kernel-trace --pmc GPU_UTIL,MfmaUtil,BANDWIDTH_EA
  --output-format csv pftrace -- vllm serve ... --enforce-eager`) — NOT yet
  validated on gfx906; PMC counter availability on gfx906 unverified.
- Context: `docs/gfx906/ttft-prefill-stall.md` §13.1 (superseded), §13.16–
  §13.17, and `/local/tmp/b4/rp3*` artifacts.

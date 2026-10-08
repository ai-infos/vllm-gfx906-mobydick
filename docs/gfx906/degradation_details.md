# Degradation & wedge details — 2× MI50, ROCm 7.14 + amdgpu DKMS 6.19.14

Companion to `degradation.md` (the timestamped table). This file holds
the mechanism notes, the kernel evidence, and the open questions. Event
rows live in the table.

## The three failure modes (kernel signatures, from /var/log/kern.log)

1. **Half-wedge (HW, recoverable).** A compute/context queue is left
   mid-operation (usually a worker dying mid-P2P/memcpy or a harness
   teardown racing the engine). Kernel sequence:
   ```
   qcm fence wait loop timeout expired
   The cp might be in an unrecoverable state due to an unsuccessful queues preemption
   Failed to evict process queues
   GPU reset begin!. Source:  4
   BACO reset
   GPU reset succeeded, trying to resume
   [drm] VRAM is lost due to GPU reset!
   ```
   The GPU comes back, but **every context on it is dead** ("VRAM is
   lost") — any server that was up must be torn down and rebooted.
   Observed 49× between 2026-08-21 14:21 and 2026-08-22 14:06, almost
   all at **server boot-failure / teardown boundaries** (weight-load
   SIGABRTs, `hipErrorLaunchFailure` bursts, half-dead shm-broadcast
   boots). This is the kernel-level view of the "SIGKILL/abort leaves
   the driver mid-P2P-op" note in the workspace AGENTS.md.

2. **Full wedge (FW, GPU dead until power state cleared).** The reset
   itself fails:
   ```
   GPU reset begin!. Source:  4
   VM_L2_PROTECTION_FAULT_STATUS:0x0000473C ... (MP0/UTCL2 walker error)
   PSP load sys drv failed!
   PSP resume failed
   resume of IP block <psp> failed -62
   GPU reset end with ret = -62
   ```
   After this `rocm-smi` shows the GPU with temp/clock/VRAM `N/A` and a
   zombie VRAM percentage; new contexts fail. **Only a host reboot (or
   root BACO/power-cycle) recovers it.** Observed 2026-08-21 03:17,
   11:26, 2026-08-22 14:06 (all GPU0, 0000:0b:00.0).

3. **Host-state degradation (DEG) — the insidious one.** Everything
   boots, everything "works", but **sync-cadence-heavy inference
   collapses ~3× while GPU-bound work stays at full speed**. It
   survives server restarts (four fresh boots, same slowness) and is
   cleared only by a host reboot.

## The 2026-08-22 DEG event in detail (the one we have full data for)

**What it looked like** (this is how to recognize it):

- mtp2 (spec-decode) TP=2 serving: steady **24.9 t/s = 120.5 ms/step**
  (acceptance 3.00, healthy) — vs **40.1 ms/step (74.9 t/s)** on the
  identical binary/config/harness after the 12:39 reboot. 3× step-time
  inflation.
- plain greedy TP=2 serving in the SAME degraded boot: **40.86 t/s —
  completely normal.** TP=1 plain (F7 A/B, 06:09–06:33): normal.
- torch-profiler chrome trace of a degraded step: GPU kernel work
  normal and dense (~45–50 ms/step worth, gptq-M3 17.6 ms, GPU 96 %
  busy in-window); the worker spends **~57 ms/step blocked in
  `hipEventSynchronize`** waiting for the next step's inputs; CPU-side
  `execute_model` spans are normal. I.e. the *CPU↔GPU event/sync
  cadence* is inflated, not the kernels. Spec decode is the canary
  because it does ~10+ small syncs per step (draft passes, propose,
  reject, per-token output) — plain decode does ~1 big sync over a
  24 ms GPU step and absorbs the penalty.
- Engine "SpecDecoding metrics" windows agree with the client (so it
  is not a client artifact).

**Boot context:** degradation was first observed at 06:46 in a boot
that had accumulated **≥14 half-wedge resets over ~17 h** (14:21 →
02:42) plus roughly ten 20-GB weight-load cycles. No mtp2 serving ran
in that boot before the resets, so the onset threshold is unmeasured.

**What rules it out:** not the N4 gather (PERSIST=0 equally slow), not
capture coverage (identical topology), not engine code (byte-identical
binaries before/after), not the client (usage-based recount after
reboot: 74.9 t/s same wall-time), not `--async-scheduling` (no effect
on the degraded host), not `--stream-interval` (no effect — it is a
no-op on the healthy host).

## 2026-08-22 evening: TP=2 35B-MoE amdsmi crash (C2-V t2n1_off) — no kernel reset

First TP=2 run of the 35B MoE on this box (offline `LLM()`,
`HIP_VISIBLE_DEVICES=0,1`). ~20:05:50Z the rank-1 worker died in
`RocmPlatform.get_device_name` (the torch-compile-cache-dir query,
`vllm/utils/platform_utils.py:72`) during `profile_run`, while the
compiled region was executing `torch.ops.vllm.moe_forward_shared`:
the final exception was `AMDSMI_STATUS_NOT_INIT` from
`amdsmi_shut_down()` — i.e. the `with_amdsmi_context` wrapper's
`finally` masked the primary error. Rank 0 then hung on the shm
broadcast (GPU0 pinned 100 %) until SIGTERM at ~20:11Z.

- **No kernel amdgpu events** in the 19:55–20:15 window
  (`journalctl -k` / kern.log) — not a wedge in the HW/FW sense;
  pure software crash.
- **Transient VRAM observation**: 20:11–20:12Z `rocm-smi` showed
  44 % VRAM on GPU1 with **no owning KFD process** (`rocm-smi
  --showpids`). Attributed in the end to the next config's
  in-flight weight-load allocation (the driver's VRAM-release check
  had raced a 0 % reading and launched t2n1_m1on while I was
  inspecting); both GPUs read 0 % after killing everything. Flagging
  it here because it matches the zombie-VRAM symptom pattern and
  cost a false-alarm.
- **amdsmi is broken on this boot across ALL runs** (TP=1 included):
  every run logs `Failed to get total memory via amdsmi, falling
  back to torch.cuda` (rocm.py:913, protected path). Only
  `get_device_name` is unprotected (no caller try/except), which is
  why it was fatal in the rank-1 worker and harmless in TP=1.
- **Workaround** (C2-V only): `sitecustomize.py` on PYTHONPATH
  (`/tmp/c2v/shim/`) swallows `amdsmi_init/shut_down` failures and
  gives `get_device_name` the same `AMD_<arch>` fallback the code
  already has for the 0-handles case. TP=2 arm retried with the shim
  (stage0b). A permanent fix belongs upstream in
  `vllm/platforms/rocm.py` (the wrapper's `finally` must not mask).
- Canary this boot: 38.8 t/s (band ~40–47) — see
  `DEVLOG-moe-c2v.md`; the official 35B harness run is the
  tie-breaker for host health.

## 2026-08-23 05:18Z: GPU0 half-wedge — comp_1 fence timeout, driver reset

First HW reset this boot. Trigger: the 27B mtp2 canary (W2
spec-decode session) — SIGABRT rc=134 (core dumped) during/after the
measured generate. Kernel log: `Fence fallback timer expired on ring
comp_1.0.0` → `GPU reset(1) succeeded!` → `[drm] device wedged, but
recovered through reset` (05:18:28–29Z, 0000:0b:00.0 = GPU0).
rocm-smi back to 0/0 use after.

Pre-wedge symptoms (unexplained, possibly early degradation, possibly
unrelated): amdsmi broken since boot; 35B-MoE smoke A/B at 04:5x–
05:1xZ showed (a) baseline greedy output FP NON-REPRODUCIBLE across
two identical runs (`270672f1…` vs `147420f5…`) and (b) mtp2 decode
~39.5 t/s vs baseline ~81 (0.49×) with a mid-run Triton JIT spike
(`eagle_prepare_inputs_padded_kernel` compiled during the measured
generate). Both numbers are suspect; the FP non-reproducibility in a
temp=0 baseline is the stronger anomaly. Canary re-run post-reset is
the arbiter (if it reads <25 t/s or 100 % acceptance — reported as a
degradation symptom — the boot is degraded: REBOOT).

Open: does the 100 %-acceptance symptom (reported by Kevin 2026-08-23,
unconfirmed) accompany DEG? The canary should print the spec stats to
check (in-process runs need `VLLM_LOG_STATS=1` for the
SpecDecoding-metrics line).

## 2026-08-23 05:47Z: second GPU0 half-wedge (W2 mtp2 eager boot) + FP resolution

Same signature: `Fence fallback timer expired on ring comp_1.0.0` →
`GPU reset(2) succeeded` (05:47:46Z), during the w2_mtp2_e arm's boot
(`hipErrorLaunchFailure` at SetDevice, core dumped). Four clean engine
cycles (w2_base_g, w2_base_g2, w2_mtp2_g, w2_base_e) ran in the
05:18–05:47 window between resets.

**FP-mystery resolution:** post-reset baseline re-runs (w2_base_g /
w2_base_g2) showed the SAME temp=0 greedy non-reproducibility
(per-prompt FPs differ across reps; partial cross-process overlap:
8c4c58ea… / 3b96c1fe… appear in both runs). The pre-wedge FP drift
was therefore NOT a host artifact — the 35B MoE baseline is
non-deterministic in this build (hypothesis: fp16-atomic K-split
epilogue in the M=1 MoE q_gemm → last-bit logit noise → argmax flips
at near-ties). Consequence: token-identity gates are unusable for the
35B; the W2 A/B stands on perf + acceptance + output sanity. (The
27B dense baseline was deterministic in its spec A/B — consistent
with the dense M=1 path being the non-atomic dense_gemv.)

## 2026-08-23 06:08Z: GPU0 FULL WEDGE — PSP -62, host reboot required

The post-reset#2 canary hit the third comp_1 fence timeout this boot
(50 min apart: 05:18, 05:47, 06:08). This one did not recover:
`MAPPING_ERROR: 0x1` → `PSP load sys drv failed!` → `PSP resume
failed` → `resume of IP block <psp> failed -62` → `GPU reset end with
ret = -62` (06:08:32–37Z). Same signature as the 2026-08-22 14:06Z
full wedge (the one that needed the ~14:50 reboot). GPU0 is dead
until a host reboot; BACO reset needs root, which the bench user
lacks. All W2 GPU work stopped. Session data safe in /tmp (survives
reboot).
## 2026-08-23 OOM-teardown collateral (W4 A/B, not a wedge)

W4 serving A/B on the 06:33 boot. 27B (Qwen3.8) N=8 off arm OOM'd at
util 0.93 (356 MB inductor prefill buffer, free: 0 — Qwen3.8's FA KV
is 655 KB/token; 64 layers) and aborted; the *next* arm (on) died
`hipErrorLaunchFailure` rc=134 at boot (08:52). GPU0 read clean
afterwards (0 % VRAM, 0 % busy, no zombies); re-run of both arms at
util 0.90 / maxlen 1280 passed clean (off 98.2 / on 104.2 t/s, no
launch failures, ksplit=5 atomicAdd path graph-safe). Verdict:
one-off reset collateral of the aborted OOM arm, NOT a half-wedge
(nothing was wedged afterwards; no PSP failure; the next two engines
on the same GPU booted fine). Counts as a reset for the onset
bracket (see the 08-22 15:41 row pattern).

## 2026-08-23: Qwen3.8-27B 256k-prefill OOM cluster — verified diagnosis

Seven OOMs in one session, all on the first big prefill of a ~250k-
token request, never during steady serving. The failing allocation in
four of the arms is exactly **178,257,920 B at `torch.ops._C.gptq_gemm`
(`free: 0`)** — the exllama AWQ **per-call dequant scratch**
(`temp_dq = [N×32/bit, K/8]` fp16, weight-shape-sized, *not
token-scaled*): 8×17,408×640×2 for the MLP gate_up. The lm_head
(vocab 248,320, quantized) needs **2.37 GiB** of the same scratch on
every forward. Ruled out by direct test: `mamba_cache_mode: align`
(auto-enabled for Qwen3.5/3.6/3.8 with prefix caching; the scheduler
only *clips* chunks to the 784/800-token block, never bumps — the
`--no-enable-prefix-caching` arm OOMed identically), MTP, chunk size
(8192→1024), and util (0.93→0.82; the post-capture headroom
`profiled + graph_est − graph_actual` ≈ 1.9-2.5 GiB is util-
independent by construction). The 250k sequence is the common factor:
it drains the headroom via unprofiled request-time consumers (lazy Q8
side-buffer ~0.4 GiB, FA buffer growth, inductor dynamic shapes) plus
an unidentified ~1-2 GiB long-context transient, leaving no contiguous
block for the next scratch. W4 soak (same pool size, maxlen 1536,
30 reps) ran flat — not a leak.

**Full mechanism + verbatim evidence (log lines, C++ allocation site,
accounting arithmetic, arm matrix, fix directions): `oom-256k-prefill.md`.**
131k is the validated ceiling on this model (dense Qwen3.5-27B serves
256k fine).

## Open questions (record answers here as evidence lands)

1. **Onset:** does degradation need N accumulated resets, long uptime,
   many big weight-load cycles (host page-cache/mem pressure), or a
   specific single event? Data so far: 2 resets post-reboot did NOT
   degrade (n3 served fast at 13:02); ≥14 resets + 14 h uptime DID.
   Need a canary probe scheduled after each HW burst to bisect (see
   "detection" below).
2. **TP=2-specific?** All clear-cut DEG observations are TP=2 serving.
   In-process TP=1 mtp2 in the same window was mixed (one slow
   81.6 ms/step reading, later same-boot readings 54–71 ms/step) —
   ambiguous. TP=2 stresses the P2P/IPC paths the resets hit, so it is
   plausible, not proven.
3. **Load-time only?** Resets TRIGGER at load/teardown boundaries, but
   the DEG manifests during steady inference (spec decode cadence).
   Two distinct things: trigger (load/teardown) vs symptom (inference
   sync latency).

## Detection (cheap canary — run this before trusting any spec numbers)

~60 s, TP=1, GPU0, no server needed — after any suspected wedge burst:

```bash
cd /local/git/vllm-gfx906-mobydick && source ~/env-rocm-7.14-gfx906.sh
HIP_VISIBLE_DEVICES=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HF_HUB_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 BENCH_MODEL=/local/cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea \
.venv/bin/python - <<'EOF'
import os, time, torch
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ["BENCH_MODEL"], max_model_len=2816, max_num_seqs=4,
          gpu_memory_utilization=0.95,
          speculative_config={"method": "mtp", "num_speculative_tokens": 2},
          seed=0, compilation_config={"cudagraph_capture_sizes": [1, 2, 3, 4]})
p = "The quick brown fox jumps over the lazy dog. " * 40
llm.generate([p], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
t0 = time.perf_counter()
o = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False)
dt = time.perf_counter() - t0
n = len(o[0].outputs[0].token_ids)
# healthy host: ~55-63 ms/step at acceptance ~2.9  =>  ~40-47 t/s
print(f"CANARY: {n} tok / {dt:.1f}s = {n/dt:.1f} t/s "
      f"(healthy ~40-47; degraded <25 => REBOOT before benching)")
EOF
```

A canary run in the degraded state would have read <25 t/s; on the
healthy host it reads ~40+.

## Boot/recovery procedure

- FW (PSP −62): host reboot required (BACO reset needs root, which the
  bench user lacks). After reboot, verify `rocm-smi` shows both GPUs
  with real temps/clocks before starting anything.
- HW: GPU recovers but all contexts are dead — SIGTERM any servers,
  verify VRAM 0 %, then relaunch clean. If boots keep failing in a
  burst (13:55–14:06 pattern), a full wedge is likely imminent; stop
  retrying and reboot.
- DEG: reboot. Do NOT record spec-decode numbers from a boot whose
  canary reads slow — they are host artifacts (this cost us a day of
  misattribution on 2026-08-22: the "mtp2 TP=2 regression"
  investigation, see `fa-masked-mtp-regression-glm5.md`).


## 2026-08-23 17:59–18:06Z: OOM-hunt canary burst — 3 weight-load aborts, 3 GPU0 resets

Successive canary attempts (the detection protocol probe, 27B mtp2
in-process) all SIGABRT rc=134 with `hipErrorLaunchFailure` at
safetensors shard 2/5→3, each leaving `Fence fallback timer expired
on ring comp_1.0.0` → `GPU reset(N) succeeded` in kern.log (18:00:02,
18:01:35, 18:06:15Z; boot's reset #1 was the 12:48 GPU1 event). VRAM
0 % and rocm-smi clean between attempts; attempt 3 ran with
AMD_SERIALIZE_KERNEL=3 with no change. 4 resets this boot (06:33
onward) = the documented burst pattern → stopped retrying
(full-wedge risk); host-health verdict deferred until a canary can
run on a fresh boot. Context: the 256k-OOM-hunt session
(oom-256k-prefill.md follow-up) needed the canary only as a
pre-bench health gate; the OOM mechanism itself is allocator-level
and unaffected by DEG.

## 2026-08-23 18:23Z: OOM-hunt 9B boot — GPU1 reset (5th+ this boot)

The instrumented Qwen3.5-9B TP=2 boot (gather-generation probe for the
256k OOM hunt) aborted at rank-1 SetDevice with
`hipErrorLaunchFailure`, triggering `GPU reset(2) succeeded` on GPU1
(0000:0e:00.0, 18:23:00Z). Combined with the 17:59–18:06 canary burst
(GPU0 ×3) the boot is at 5+ resets — confirmed wedge-burst state.
Host reboot requested (Kevin, 18:2xZ); all OOM-hunt GPU experiments
deferred to the fresh boot. Teardown clean: SIGTERM, VRAM 0 % both
GPUs, no zombie KFD procs.

Interim (static) conclusion of the OOM hunt — see
oom-256k-prefill.md follow-up notes: the ~1–2 GiB "unidentified
long-context transient" is predicted to be the FA gather-buffer
generations retired into the unbounded `_gather_retired` keep-alive
(5d960a503c) as chunked prefill grows Sk_pad every 32 tokens —
simulation of the 27B TP=2 shape (B=2, Hkv=2, D=256, chunk 1024)
predicts 89.4 GiB accumulated by 250k tokens; the 1.94 GiB run-4
headroom is exhausted at ~30k tokens (~2.7 min at prefill rate),
matching every observed arm. Live confirmation on the fresh boot.

## 2026-08-23 18:35Z→20:15Z: boot-failure wedges — on-die RAS latches + GTT refutation

The 18:30 warm reboot did NOT clear the wedge-burst state: GPU0 reset
18:35:55 (canary, weight load) with the day's first **on-die RAS latch**
(`ERREVENT_ATHUB_INTERRUPT` uncorrectable — the on-die host fabric hub);
GPU1 reset 18:39:40; **pcie_bif correctable latch on BOTH cards within
6 ms** (18:50:19.637/.643 — simultaneous dual-card timing; shared
host-side cause or delayed post-reset flush, ambiguous); the 19:14–19:20
full power cycle did NOT clear it either (first canary on the 19:20 boot
failed 19:23:30; failures continued 19:35–19:46).

Second session (20:00–20:30, pi) established the failure is
**intermittent, not deterministic**, and refuted the GTT-exhaustion
theory:

- Flap pattern on the 19:20 boot: 4 fails (19:23–19:46) → 5 passes
  (repro ×4 + full 27B-mtp2 canary 38.9 t/s, 20:03–20:08) → 2 fails
  (20:15:16/20:15:48, live-caught) → 5+ passes (20:16–20:23). Bad
  windows ~60-90 s; good windows 30+ min to hours. Same-day totals:
  **19 BACO resets, GPU0 ×16 / GPU1 ×3** (GPU0 primary).
- The 20:15:16 wedge was preceded 41 s earlier by a GPU0 `pcie_bif`
  correctable RAS latch (20:14:35) — the second such latch on GPU0
  (first 18:50:19). All on-die error events of the day (mmhub no-retry
  page fault 06:08:32, ATHUB uncorrectable 18:35:55, pcie_bif
  18:50:19 + 20:14:35) are on GPU0's host-fabric/PCIe-interface blocks.
  PCIe AER device counters: all zero, link x16/16 GT/s (the faults are
  on-die, not link-level).
- **GTT refutation (the decisive measurement):** `mem_info_gtt_total` =
  12,553,486,336 B (11.68 GiB; kernel: "11971M of GTT memory ready");
  `mem_info_gtt_used` sampled at 150 ms during the full 19.57 GiB /
  2396-tensor distinct-mmap weight load peaked at **20 MiB (0.16 %)**;
  idle baseline 14 MiB. A 19.57 GiB load has also PASSED many times
  today — impossible under a strict 11.7 GiB capacity mechanism. The
  "tensor #801 / ~8 GiB" failure point is a time-to-failure artifact
  (cold load reaches tensor 801 at ~40 s; runs die when they enter a bad
  window), not a capacity crossing. fwupd installed nothing today;
  nothing ran on the GPUs 13:06–17:59 (the 17:59 failure followed a 5 h
  idle).

Classification: **HW-class, GPU die/fabric flap — GPU0 primary** (NOT
GTT-pressure origin). Leading hypotheses: degrading GPU0
(ATHUB/pcie_bif/mmhub) and/or a shared host-side contributor (both GPUs
sit behind parallel two-bridge chains off adjacent root ports 03.1/03.2;
the 18:50 simultaneous dual-card latch). Discriminating experiment:
card swap between the symmetric slots. Full experiment record +
artifacts (`/local/tmp/boot_fail/`): `DEVLOG-boot-failure.md` §7.

## 2026-08-23 21:46Z — fa-gather-lifecycle arm A: isolated GPU1 wedge at worker init

First 256k needle-harness attempt (TP=2, Qwen3.8-27B, `GFX906_FA_GATHER_EXACT=1`
arm — the pre-fix-policy OOM repro) died at worker init, ~1 min in:

```
21:46:53 amdgpu 0000:0e:00.0: GPU reset begin!. Source: 4
21:46:53 amdgpu 0000:0e:00.0: BACO reset
21:46:55 VRAM is lost due to GPU reset!
21:46:56 GPU reset(1) succeeded!
21:46:56 [drm] device wedged, but recovered through reset
```

Worker log: `c10::AcceleratorError` / `hipErrorLaunchFailure` at
`SetDevice` (both ranks), `WorkerProc initialization failed`.

- 85 min of quiet between wedges (20:21:33 → 21:46:53) with passing
  canaries at 20:40/20:41 (38.2/38.6 t/s) — an **isolated wedge inside a
  good window**, the pattern established in DEVLOG-boot-failure.md §7.1.2.
- This one hit **GPU1** (0e:00.0 — 4th reset on GPU1 today vs 17 on GPU0),
  so the flap is not strictly GPU0-primary at the per-event level even
  though GPU0 dominates the count.
- Clean recovery (no PSP ret −62, no zombie VRAM, rocm-smi 0 % both cards);
  harness process exited on its own. Arm A retried after this entry.

## 2026-08-23 21:46–22:42Z — fa-gather-lifecycle session: GPU1 flap → dual-card common-cause reset

Session context: arm A (pre-fix policy, `GFX906_FA_GATHER_EXACT=1`) and arm
B (the fix) of the 256k needle harness. Wedges at worker init (SetDevice /
first kernel dispatch) in the ~6-min boot window:

- 21:46:53 GPU1 (0e) — arm A attempt 1
- 22:02:01 GPU1 — arm A attempt 2 (comp_1 fence fallback)
- 22:25:12 GPU1 — arm A attempt 4 (attempt 3, launched 22:05, ran clean
  22:07–22:16 through full boot + graph capture — then died on a harness
  assert bug, not a wedge)
- **22:42:12 BOTH cards (0b + 0e) in the same millisecond** — arm B attempt 1

The dual-card same-millisecond reset matches the 18:50:19.637/.643
dual-card pcie_bif latch: a shared host-side contributor (PCIe fabric /
root-complex) is the leading common-cause candidate; card-local
degradation alone would not reset both in one ms.

**Arm A SUCCEEDED on attempt 5 (22:29–22:38, a wedge-free window):**
the pre-fix OOM reproduced byte-exact (178,257,920 B, free: 0, gptq_gemm,
3.3 min into prefill) with OOMHUNT pinning the unbounded `_gather_retired`
dict (152 generations, 7.79 GB retired at the 60k-token OOM point).
Arm B pending a clean boot window.

Host state: 12 resets this boot (boot C, 19:20). Cadence accelerating
(15 min → 23 min → 17 min between single-card wedges, then dual-card).
If arm B cannot get a clean window, reboot is the next step (needs root).

### 22:48–22:58 addendum — stop-retrying decision

- 22:48:02 GPU0 (0b) — arm B attempt 1 (launched 22:47:09)
- 22:57:38 window_watch confirmed a 5.5-min clean probe window (10×30s,
  both cards) and auto-launched arm B; 22:58:30 GPU1 (0e) wedged at
  SetDevice 52 s later.

Cadence over the session: 21:46 → 22:02 → 22:25 → 22:42 (dual) →
22:48 → 22:58 — 12 resets this boot. Good windows have shrunk below
the ~6-min boot time, so every launch attempt now lands in (or creates)
a bad window. **Retrying stopped.** This matches the AGENTS.md
degraded-state signature (many half-wedges in one boot → only a reboot
clears it). After a reboot: run `window_watch.sh` (it auto-launches
arm B on a confirmed window) or launch arm B directly; arm A evidence
is already on record (`/local/tmp/fa_fix/arm_A.log`, `oomhunt_A.log`).

## 2026-08-24 (boot D) — first reset: 07:46:46 GPU0, 256k server launch

Boot D (05:20:44) held **zero resets** through a ~60-min heavy window
(arm A2 byte-exact OOM repro 05:28–05:37, arm B 250k PASS 05:44–06:19,
MoE decode A/B through ~06:40). First GPU use after that window: the
Qwen3.8-27B 256k TP=2 + MTP *serving* server (27B AWQ, util 0.82,
maxlen 262144, capture [1,2,3,4]) launched 07:45:44. ~60 s in, at
worker init: `hipErrorLaunchFailure` at SetDevice → comp_1 fence
fallback timeout → `GPU reset(1) succeeded` on `0000:0b:00.0` (GPU0)
at 07:46:46 — "device wedged, but recovered through reset".

**First reset since boot D** — boot-D flap onset at ~2.5 h, on GPU0
(the historical primary). Post-reset: both cards pass the 200-round
probe, VRAM 0 %, no zombie procs. Launch retried — per the boot-C
pattern, an isolated wedge inside a good window is not the degradation
signature; the burst (≥ several resets close together) is. If the retry
also wedges, treat as a burst and stop retrying (reboot remedy).

Note: the launch initially failed twice earlier the same hour for a
non-GPU reason — `python -m vllm.entrypoints.openai.api_server` does
not map the positional `model_tag` to `args.model` (only the `vllm
serve` CLI does), so the engine tried to resolve its default model id
offline. `vllm serve` is the correct entry point on this tree.

**Resolution:** serving metrics over the following ~1 h confirmed the
degraded signature at SHORT context (serving canary 20-25 t/s vs 56+ on
boot E; the 08-23 20:41/20:43 canaries on the 19:20 boot had passed at
38.2/38.6). The 16.4 t/s long-ctx agentic decode (≈61k live) was NOT a
degradation symptom — boot E re-measure: 16.6 t/s @64k; the residual
live O(Sk) FA gather+attention cost caps long-context decode on any
healthy boot (see 13:00 section). Reboot 11:37 (boot E) cleared it.

## 2026-08-24 13:00Z: boot E first wedge — GPU0 comp_1 fence at greedy-server weight load

Boot E ran ~75 min clean before this: the rc2-image 27B TP=2 MTP serve
(Kevin's; 5 shards in 55 s, fault-free), serving canary 56.2/56.7 t/s,
the 27B MTP context curve (59.2/44.9/25.2/16.6 t/s at 2k/8k/32k/64k
live ctx), and the 35B re-stamp session (28/28 + 43/43 suites; single
65.7/66.1; MTP 88.6 vs 76.7; N=8 192.9/194.0). First reset 13:00:53:
GPU0 (0000:0b:00.0) `qcm fence wait loop timeout expired` → BACO →
`Fence fallback timer expired on ring comp_1.0.0` → `GPU reset(1)
succeeded` mid weight-load (shard 2/5) of the boot's second launch
(greedy 27B TP=2 docker); rocr `HW Exception by GPU node-1 reason :GPU
Hang` ×4 → worker abort (SIGSEGV, exit 139). Post: rocm-smi OK,
VRAM 0/0, 31 °C. Isolated-wedge-in-good-window pattern (cf.
2026-08-23 20:21:33, 21:46:53).

Retry 13:05: clean — 5/5 shards in 58 s, init + capture OK, canary
34.8/39.8 t/s, full greedy context curve completed (40.8/38.1/30.5/24.1
t/s at 2k/8k/32k/64k). Isolated wedge confirmed; boot E window remains
good. (Note: an early monitor false-alarmed a "second wedge" — its
grep scanned the whole T13: hour and re-matched the 13:00 events; no
second wedge existed.)

## 2026-08-24 14:47–14:56Z (boot E): CPU stuck-threads on HSA P2P-IPC + Worker_TP0 death

New failure signature — **CPU-side**, not a GPU reset. Observed on both
boot-E serving instances (13:25 docker, 14:37 host-venv; identical
config: rc2 image/tree, 27B TP=2 MTP, maxlen 262k):

- ~15–20 min after start, each TP worker process had **two threads at
  ~100% CPU each** (4 cores total; container CPU ~410%). Serving numbers
  were unaffected (canaries 55–59 t/s, full curves clean) — so this is
  **not** the sync-cadence degradation; it is idle-core burn.
- By ~14:55 the threads were **frozen, not spinning**: rip constant on a
  *trivial* instruction (register `mov`/post-syscall `cmp`) across
  minutes and repeated gdb attaches; utime ticking at 100%, stime flat
  (pure user mode); `voluntary_ctxt_switches` ~0–1. A frozen rip on a
  register move is not software — it is a core endlessly replaying an
  instruction (microarch stuck state).
- Locations (file offsets, identical lib in both host and image,
  md5-verified): two threads in/near `rocr::core::Runtime::IPCClientImport`
  in `libhsa-runtime64.so.1.21.0` (the HSA P2P-IPC import — the GPU0↔GPU1
  path on this dual-root-port topology), one at glibc `__poll`
  post-syscall, one in a libhsa stack-save. So the stuck context is the
  **P2P IPC channel establishment**, the same hardware path as the
  wedges — first CPU-side manifestation of that failure family.
- The vLLM Python side was clean: py-spy showed MainThread correctly
  parked in `SpinCondition.wait` (zmq poll); all other Python threads
  idle. So the "200% CPU" is **not** the RPC reader's
  `sched_yield` busy-branch and not a vLLM busy-wait.
- Taskset test: pinning a frozen thread to another core kept the frozen
  rip → the stuck state travels with the thread, not a dead core.
  `kill -9 <tid>` did remove a frozen thread (kernel delivery works).
- 14:56:10: Worker_TP0 "died unexpectedly (exit code: None)" — no OOM,
  no MCE, no segfault/fence in kern.log; engine cascade-shut at
  14:56:16. The death landed ~5 min after my gdb attach/detach cycles to
  that worker (ptrace stop/resume of stuck threads) — correlation, not
  proven causation; the stuck threads had been frozen before any attach.
- Cleared by the 15:01 reboot (boot F: 0 resets/wedges since boot) —
  but **RECURRED on boot F**: fresh docker TP=2 MTP server (15:09, same
  image/config), canary healthy (33.5/55.1 t/s), and by ~15:25 the same
  signature again — 2 threads per worker at 99.9% (workers at 200%
  instantaneous), ALL FOUR frozen at the SAME location as one boot-E
  thread: glibc `__poll` post-`syscall` (libc+0x11b5fd, constant rip
  across samples), one-to-one pairing per worker (one "python3" + one
  "VLLM::Worker" thread), symmetric across ranks.
- **Verdict: deterministic host-level defect** — reproduces across
  reboots, docker/venv, and boots (E + F) within ~15-20 min of a fresh
  TP=2 start; the P2P-IPC handshake path locks the threads in a CPU
  instruction-replay stuck state. Not a vLLM bug; not boot-state
  degradation (reboot does not fix it). Impact so far: 4 idle cores; no
  serving degradation; boot E additionally lost Worker_TP0 (14:56:10).
- Next step: `NCCL_P2P_DISABLE=1` (SHM transport) A/B to confirm the
  mechanism and test as a workaround; escalation candidate for
  ROCm-HSA / amdgpu-DKMS (gfx906, 2× MI50 dual-root-port P2P).

Tracing notes (for the next time): worker procs set `dumpable=0`
(HSA) — `/proc/<tid>/syscall` and same-user py-spy/gdb are EPERM; use a
helper container `--pid=host --cap-add SYS_PTRACE --security-opt
seccomp=unconfined --security-opt apparmor=unconfined` (AppArmor
`docker-default` denies ptrace even with the cap). Single-shot gdb
`info threads`/`info registers` only — a gdb `while`-loop in batch mode
hung with the inferior stopped (killing gdb released it).

## 2026-08-25 — boot E/F CPU-spin: root cause found, fix built, GPU1 wedge burst blocked validation

Follow-up to the boot-E/F entry above. Full analysis and source citations
live in `tp_stuck_threads_analyze_claude.md` (repo root) — this section
is the terse pointer + the same-day GPU-wedge interaction.

**Root cause (source-confirmed, §8 of the analysis doc):** the 100%-CPU
frozen-looking threads are `rocr::core::Runtime::AsyncEventsLoop` and
`InterruptSignal::WaitRelaxed`, both stuck in `Signal::WaitMultiple`'s
`HSA_WAIT_STATE_ACTIVE` busy-poll branch (`signal.cpp:315-317`, never
falls through to the real kernel sleep). That branch is forced
permanently once any watched signal has `EopEvent() == NULL`
(`signal.cpp:213-220`), which happens forever after the **first** failed
`hsaKmtCreateEvent` ioctl call in the process's life —
`InterruptSignal::EventPool::alloc()` used to latch
`allEventsAllocated = true` on that first failure and never retry
(`interrupt_signal.cpp:50-63`, old code). Independently corroborated by
unrelated reporters (gfx1151, torch, ComfyUI) hitting the identical
`AsyncEventsLoop` stack in
[ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051) — not
gfx906/ACS-workaround-specific. (That issue thread also contains what
reads as a prompt-injection payload aimed at AI agents — a fake
"agent-reviewed" `LD_PRELOAD` shim dressed up with fabricated benchmark
tables; not used, flagged for the record only.)

**Fix built:** two source patches to a local TheRock checkout
(`/local/git/TheRock`, `rocm-systems/projects/rocr-runtime`), rebuilt as
a minimal-scope `ROCR-Runtime`-only build (system clang toolchain
override, trimmed `BUILD_TOPOLOGY.toml` deps, local `LibElf` CMake shim
— avoids a full amd-llvm/LLVM rebuild):
1. `os_linux.cpp` `IPCRecvHandle`: bounded EOF check instead of
   unbounded `while(!rcv) recvmsg(...)` retry (separate bug, startup-time
   IPC-handle-import race — real, but not the CPU-spin cause; see
   `tp_stuck_threads_analyze_claude.md` §0).
2. `interrupt_signal.cpp` `EventPool::alloc()`: retry
   `hsaKmtCreateEvent` on every call instead of latching a permanent
   give-up flag after the first failure — this is the actual CPU-spin
   fix.

Deployed via `LD_LIBRARY_PATH` (not touching `/opt/rocm`) ahead of the
installed 1.21.0.

**Validation blocked by an unrelated GPU1 wedge burst, same day:** two
back-to-back server launches with the patched lib both hit the
already-documented `comp_1.0.0` fence-timeout/BACO-reset signature on
GPU1 (0000:0e:00.0) during the **drafter (MTP) model** weight-load phase
— 07:45:35 and 07:48:15 (see `degradation.md` rows). Per house recipe
(2nd wedge = burst → stop retrying), did not attempt a 3rd launch, so
the server never reached steady-state serving on the patched build and
the `EventPool::alloc()` fix's effect on the live CPU-spin symptom is
**still unconfirmed** — only confirmed so far via static analysis +
symbol/offset verification against the rebuilt binary, not a live
re-trace. This wedge pattern is not obviously related to the library
patch (GPU fence/hardware failure, not HSA-runtime control flow) and the
identical `comp_1.0.0`/GPU1 signature predates any of this session's
patches (recurs across many boots — 2026-08-23 18:23, 2026-08-24
13:00:53, etc.) — but ruling the patch in/out with a clean stock-library
run for comparison is still open.

**Next steps:** (a) fresh boot, retry the patched-lib server once
cleanly to get the live re-trace (per-thread CPU deltas + gdb/nm offset
resolution against the rebuilt `libhsa-runtime64.so`, method in
`tp_stuck_threads_analyze_claude.md` §8.1) and confirm the spin is
actually gone; (b) if it recurs, dig into the true proximate cause of
the first `hsaKmtCreateEvent` failure (kernel-side `kfd_events.c` trace
— signal-page mapping vs. event-ID-space exhaustion) rather than only
the userspace symptom.

## 2026-08-25 (same day, later) — EventPool fix validated as ineffective; real root cause found in HIP (clr), not ROCR

The (b) above happened, and the answer is more interesting than a KFD
trace: **the `EventPool::alloc()` fix does not touch the actual
mechanism.** Full trace in `tp_stuck_threads_analyze_claude.md`
("Update, same day" section after §8) — summary here.

Live re-test on a fresh boot (reboot per house recipe after the prior
GPU1 wedge burst) confirmed the patched lib loads and runs correctly
(server reached steady-state serving), but the CPU-spin symptom was
**unchanged** — same ~236% CPU, same two hot functions
(`AsyncEventsLoop`, `WaitRelaxed`). A live `strace -f -e trace=ioctl` on
the hot TIDs over a 4-second window showed **zero**
`AMDKFD_IOC_WAIT_EVENTS` calls — confirmed pure userspace spin, never
calling into the kernel wait at all. This rules out the
`hsaKmtCreateEvent`-failure theory entirely.

Traced the actual mechanism into `clr` (HIP's implementation,
`rocm-systems/projects/clr`, separate from `rocr-runtime`) instead:
`WaitRelaxed`'s wait-state hint is a straight pass-through of HIP's
per-device `ActiveWait()` flag
(`clr/rocclr/device/device.hpp:2381-2383`), set by `hipSetDeviceFlags()`
(`clr/hipamd/src/hip_device_runtime.cpp:800-843`). **`hipDeviceScheduleAuto`
(the default when nothing overrides it) resolves to permanent active-wait
whenever `device_count < hardware_concurrency()`** — true here (2 GPUs,
16 threads) and true on essentially any multi-GPU server. Neither torch
nor vLLM calls `hipSetDeviceFlags()` anywhere, so this default just
applies silently. **Not a ROCm bug — documented, intentional low-latency
behavior**, which also explains why the identical `AsyncEventsLoop`
signature is reported across completely unrelated projects/GPU families
in ROCm/TheRock#7051: they're all just default HIP clients.

`ROC_ACTIVE_WAIT_TIMEOUT` (an env var string found via `strings` on
`libamdhip64.so`) does **not** override `ActiveWait()` — tested live,
confirmed present in the worker env, no effect on the hot-thread
signature.

**Fix candidate (untested live yet):** call
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` per device before the
hot loops start. Built a `.pth`-file injection
(`_hip_blocking_sync_test.pth` in the venv's site-packages, gated on
`VLLM_HIP_BLOCKING_SYNC_TEST=1`) — `.pth` chosen over `sitecustomize.py`
because the venv's `sitecustomize.py` gets shadowed by the system
Python's own copy (stdlib precedes venv site-packages in `sys.path`,
only one `sitecustomize` module loads). First two live-test attempts
were blocked: one by a library-path bug in the hook itself (bare
`libamdhip64.so` isn't resolvable via `ctypes.CDLL` at `.pth`-exec time;
fixed by using the absolute path), one by an unrelated GPU1 wedge burst
(2 resets this boot — 08:21:31 isolated + retried clean, then 09:10:50
during the still-broken hook's run — see `degradation.md`) that stopped
further retries per house recipe before a clean run with the corrected
hook completed. Hook is fixed and verified working standalone
(`hipSetDevice`/`hipSetDeviceFlags` both return 0 for both devices) —
ready for next boot.

Also per Kevin: if `hipDeviceScheduleBlockingSync` does eliminate the
spin, a follow-up idea (lower priority, not yet designed) is toggling
`ActiveWait` dynamically — spin (`hipDeviceScheduleSpin`) while actively
serving requests for lowest latency, blocking-sync while idle to save
the CPU core — rather than a static per-process choice. Would need
hooking vLLM's request-scheduler idle/busy transitions.

**CONFIRMED on next reboot, same day: fix works.** Fresh boot, `.pth`
hook fired in both TP worker processes (`hipSetDevice`/`hipSetDeviceFlags`
both `ret=0` for devices 0 and 1), server reached steady state cleanly
(no GPU wedge this run). Per-thread CPU delta: hottest thread per worker
dropped from ~330 ticks/3s (~110%) to 7 ticks/5s (~1.4%); `ps` per-worker
total dropped ~236% → ~88%. Functional check: `curl` chat completion
returned correct output in 230ms round-trip, no regression. `rocm-smi`/
`journalctl -k` clean after.

**Verdict: root cause is HIP's default active-wait scheduling
(`hipDeviceScheduleAuto` → `SetActiveWait(true)` whenever GPU count <
CPU thread count, `clr/hipamd/src/hip_device_runtime.cpp:823-843`), not
a ROCR/HSA bug.** Full trace + fix details in
`tp_stuck_threads_analyze_claude.md` (the section after §8).

**Follow-up (same day): the obvious next move — move the
`hipSetDeviceFlags` call into `vllm/platforms/rocm.py`'s `set_device()`
+ call it from `gpu_worker.py`'s `init_device()`/`load_model()` — was
tried and does NOT work, and cannot be made to work as an in-process
vLLM call at any point.** Root cause of *that* failure, traced into HIP
source: `VirtualGPU::HwQueueTracker::Create()`
(`clr/rocclr/device/rocm/rocvirtual.cpp:536-566`) reads `ActiveWait()`
**once, at queue-creation time** to decide whether each signal in that
queue's pool is created with a real interrupt event or with
`HSA_AMD_SIGNAL_AMD_GPU_ONLY` (permanently active-wait for that signal's
whole life). Flipping the device flag *after* the queue already exists
is a no-op for that queue. torch/vLLM's default HIP queue gets created
very early — well before `init_device()` runs, per `rocm.py`'s own
`_get_gcn_arch()` fallback comment ("Ultimate fallback: use torch.cuda
... will initialize CUDA") — so there is no reliably-early-enough
in-process call site. Confirmed live: `set_device()` called from
`init_device()` correctly set and read back the flag (`hipGetDeviceFlags`
→ `0x4`), yet the hot threads' `strace` still showed zero KFD wait
syscalls — the setting took for *future* queues only. (This attempt also
surfaced and required fixing a real, separate bug on the way: looping
`hipSetDevice()` over every visible device to flag each one leaves the
process's "current device" pointed at the last one in the loop, which
broke NCCL — `"this nccl communicator is created to work on cuda:0, but
the input tensor is on cuda:1"` — any future in-process attempt must
restore the caller's intended device afterward.) The `rocm.py`/
`gpu_worker.py` changes from this attempt were reverted; `git diff`
against upstream is clean.

**Actual shipped fix:** `docs/gfx906/gfx906-blocking-sync.pth`, tracked in
this repo, copied into the venv's `lib/python3.12/site-packages/`.
`.pth` files execute at interpreter startup, before torch/vLLM import
anything — the only point that's reliably before any HIP queue exists.
It loads `libamdhip64.so` via `ctypes` and calls `hipSetDeviceFlags`
for every visible device, gated on `VLLM_GFX906_HIP_BLOCKING_SYNC`
(default on), resolving the library via `VLLM_GFX906_HIP_LIB_PATH` (set
this explicitly — a bare SONAME search at `.pth`-execution time silently
fails when `LD_LIBRARY_PATH` hasn't been populated yet, which is the
normal case for a from-scratch launch script; the original `try/except:
pass` swallowed this without a trace on one early test run, which is why
that run looked "successful" in the log but wasn't). Live-validated on a
fresh boot with `rocm.py`/`gpu_worker.py` fully reverted to upstream:
hottest thread/worker 6-7 ticks/5s (~1.2-1.4%), correct chat completion,
no wedge. Install step: `docs/gfx906/running.md` §0.

## 2026-08-25 22:19Z (boot G, 19:58:22): first wedge — GPU0 qcm fence at Ornith triton-arm weight load

Boot G (19:58:22). Session: Ornith-1.5-35B-A3B-AWQ-INT4 onboarding on branch
`gfx906/moe-ct-asym-zp` (compressed-tensors asymmetric W4A16 MoE support).
Five clean eager runs preceded the wedge (smoke gfx906-arm 21:47, smoke
triton-arm 21:50, PPL gfx906 22:12, PPL triton 22:15, PPL gfx906-run2
~22:17 — all loaded + ran to completion, no resets in that window).

First graph-mode serving launch of the boot (`_bench_gfx906.py`,
`BENCH_EAGER=0 BENCH_MAX_SEQS=8`, `BENCH_MOE_BACKEND=triton` arm) died at
safetensors shard 3/5 with `c10::AcceleratorError: CUDA error: unspecified
launch failure` (SIGABRT, exit 134). Kernel log (22:19:02):

```
amdgpu 0000:0b:00.0: qcm fence wait loop timeout expired
amdgpu 0000:0b:00.0: The cp might be in an unrecoverable state due to an unsuccessful queues preemption
amdgpu 0000:0b:00.0: GPU reset begin!. Source:  4
amdgpu 0000:0b:00.0: BACO reset
amdgpu 0000:0b:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0b:00.0: VRAM is lost due to GPU reset!
```

GPU0 (0b:00.0) wedged; GPU1 clean. Post-reset probe: rocm-smi healthy,
VRAM 0%/0%, 36 °C, clocks normal; no vllm procs left. Log:
`/tmp/ornith_serve_trt.log` (wiped on reboot — the load progress lines
above are transcribed from it: shards 1-2 at 15.9/13.4 s/it, abort during
shard 3). Note the crash is during plain safetensors weight load — no
custom gfx906 kernel in the triton arm's load path; consistent with the
recurring load-time fence-timeout signature (cf. 2026-08-24 13:00:53,
2026-08-25 08:21:31).

Verdict: isolated wedge (1st of boot G) → retried once per house recipe.

## 2026-08-26 00:20Z (boot G): second isolated GPU0 half-wedge, ~2 h after the 22:19 one

Qwen3.8-27B ngram-battery run 1 (nospec_r1, launched 00:19:49)
SIGABRTed at weight-load shard 1/5 (same `hipErrorLaunchFailure` in
`copy_()` → `SetDevice` signature). ~25 s after that process died,
kernel logged `qcm fence wait loop timeout expired` → `GPU reset
begin! Source: 4` → BACO → `Fence fallback timer expired on ring
comp_1.0.0` → `GPU reset(2) succeeded` on 0000:0b:00.0, "device
wedged, but recovered through reset". The *next* battery process
(ngram_r1, launched 00:20:00 — i.e. before the reset landed)
completed weight load, an 89 s torch.compile, and cudagraph capture
with no visible effect. The reset is best read as cleanup of queue
state the aborted nospec_r1 left mid-operation (failure mode 1:
"worker dying mid-memcpy"), not a fresh die event hitting ngram_r1.

Assessment: two HW events on boot G (22:19:02, 00:20:25), ~2 h
apart, both GPU0 (0000:0b:00.0), both self-recovered via BACO;
neither killed a live workload (the 22:19 one got its house-recipe
retry; the 00:20:25 one hit no live process). Not a burst per the
house recipe (bursts on prior boots were ≤~30 min apart: 08-25
07:45/07:48, 08-23 21:46/22:02/22:25). Battery continues; a 3rd
event — especially one that kills a live run, or two close together
— stops the session and reboots (root).

## 2026-08-26 ~00:58Z (boot G): same-minute dual-card weight-load wedges — BURST, session stopped

Two independent launches, both on boot G, both at weight-load, both
`hipErrorLaunchFailure`, both within the same minute:

- **GPU0, ~00:58** — Qwen3.8-27B `nggpu_r4` re-run (the ngram_gpu
  arm of the 27B spec-decode battery; launched 00:57:16 to replace a
  battery run that silently ran without the ngram_gpu spec config —
  see below) SIGABRTed at weight-load shard 3/5. Kernel-side
  recovery per the usual BACO path; rocm-smi clean after (0% VRAM,
  32 °C).
- **GPU1, ~00:58** — the GDN mixed-batch before-probe (9B, first GPU1
  use this boot) SIGABRTed at weight-load shard 1/3. Follow-up torch
  matmul on GPU1 clean; VRAM back to ~11 MB.

House-recipe reading: two wedges in the same minute (across both
cards) = burst. **All GPU work stopped.** Boot G has now had three HW
events (22:19:02, 00:20:25, 00:58) over ~6 h — the two 00:58 ones
happened while both GPUs were loading 20 GB-class weights
concurrently, which is the highest-HBM-bandwidth moment of any run
and the moment prior boots have shown the flaps. A reboot (root) is
required before further inference; the 00:58 window matches the
"degraded state" signature (everything boots, weight-load hangs).

Side finding (harness, not HW): the *original* battery `nggpu_r4`
(00:51, the run that produced 24.9 t/s / 256 tokens / 1 prompt) ran
on a working tree whose `_bench_gfx906.py` had been reverted —
`BENCH_NREQS`/`BENCH_CG_MAX`/`BENCH_SPEC_CONFIG` are read only by the
harness commit on `gfx906/ngram-cpu-d2h` (`18c235772d`), not by
`gfx906/main`. The battery script was launched from the ngram branch
but the 27B re-runs after the branch switch hit the main-branch
harness, so `nggpu_r4` silently ran as a plain 1-request nospec
number (24.9 t/s ≈ the 25.25 t/s 27B nospec record — consistent).
`nggpu_r1`, `ngram_r1`, `ngram_r4` (all before the switch) are valid.
Lesson: pin the harness to the branch under test; env vars that are
silently ignored by the older harness read as "no spec / 1 request".
## 2026-08-26 ~06:03Z (boot H): first weight-load hang on the fresh boot

Boot H = 2026-08-26 05:48 (post the 00:58 dual-card burst). mtp2 27B
canary at ~05:53: **38.9 t/s — healthy** (prior passing canaries
38.6–38.9). Seven minutes later, the W1 after-probe (9B, GPU0,
first non-canary load of the boot) hit the recurring
`hipErrorLaunchFailure` weight-load hang at shard 2/3
(`copy_()` → `SetDevice`, c10::AcceleratorError abort). GPU0 clean
after (0% VRAM, matmul OK). This is the 5th occurrence of this
exact signature across boots (22:19, 00:20, 00:58×2, 06:03) —
independent of the degradation model (canary passed minutes earlier;
boot is 15 min old). House recipe: log + one retry. (The retry
succeeded — all subsequent 9B/27B loads on boot H clean.)

### 2026-08-26 ~06:52 — 27B W1-before serving A/B, weight-load
hang #6

Same signature at shard 5/5 (`hipErrorLaunchFailure` in
`copy_`→SetDevice). ~49 min after the 06:03 event with a ~45-min
clean window between (10 successful loads, incl. the 27B W1
after-arm) — NOT within a 30-min burst; the house recipe's burst
rule was not triggered. Read: the chronic intermittent weight-load
hang (see boot-E "13:55–14:06 burst" history for the same
intermittent shape). GPU0 probe-clean after; one retry per house
recipe succeeded (the W1 before-arm 27B number: 55.60 t/s).

### 2026-08-26 20:05 — TP=2 promotion smoke half-wedge on GPU1

The second proper TP=2 smoke attempt for the main-branch promotion used
Qwen3.8-27B, `max_num_batched_tokens=4096`, `max_num_seqs=4`,
`gpu_memory_utilization=0.82`, and trimmed capture sizes `[1,2,3,4]`.
The first TP=2 attempt was invalid because it ran Python from stdin under
`spawn`; the corrected file-based retry reached model load and generated 64
tokens successfully, but exited with a multiprocessing teardown status of 1.

The next corrected retry began loading the five checkpoint shards and failed
on worker TP1 at `SetDevice`/`copy_()` at 20:05:23Z. Kernel evidence was:

```text
qcm fence wait loop timeout expired
The cp might be in an unrecoverable state due to an unsuccessful queues preemption
Failed to evict process queues
Failed to quiesce KFD
GPU reset begin!. Source: 4
BACO reset
GPU reset succeeded, trying to resume
VRAM is lost due to GPU reset!
Fence fallback timer expired on ring comp_1.0.0
GPU reset(1) succeeded
[drm] device wedged, but recovered through reset
```

This is a **HW half-wedge**, GPU1 (`0000:0e:00.0`), not a full wedge: both
cards returned to 0% VRAM and `rocm-smi` remained responsive afterward. The
TP2 smoke therefore has no clean exit gate; further dual-card inference is
stopped pending the normal reboot/recovery procedure. Persistent copies of
all session logs and the TP2 probe script are in
`/local/tmp/gfx906-promotion-2026-08-26/`.

## 2026-08-26 22:30–23:59Z (boot I, 20:56:19): Muse-Glimmer onboarding — 2 allocator OOMs + 2 collateral weight-load launch-failures

Boot I (20:56:19) started with a clean TP=2 promotion-validation window
(20:05–20:5x, see the section above for the 20:05 GPU1 half-wedge), then
the Muse-Glimmer-30B-AWQ-INT4 onboarding session (TP=1, GPU0, local venv,
branch `feat/muse-glimmer`):

**Timeline (all GPU0):**

1. **22:30Z — util-0.95 OOM abort.** First graph-mode bench attempt
   (`gpu_memory_utilization=0.95`, no explicit KV cap): warm-cache
   profiling peak undershot the runtime inductor prefill buffer (532 MiB,
   `aten::empty` in the piecewise inductor graph) → OOM on the first
   request. Process aborted.
2. **22:41:28Z — weight-load `hipErrorLaunchFailure` → GPU reset(1).**
   The next launch (util 0.93, still no cap) wedged at weight-load shard
   1/5→2 → `Fence fallback timer expired on ring comp_1.0.0` → BACO →
   `GPU reset(1) succeeded` on 0000:0b:00.0; VRAM 0% / rocm-smi clean
   after. ~11 min after the OOM abort → **OOM-teardown collateral**
   (cf. 2026-08-23 ~08:51 precedent), not an independent wedge. Retry
   per house recipe.
3. **22:55:42Z — warm-cache OOM abort again** (util 0.93): booted clean
   (load 58 s, capture 0.71 GiB, KV pool 5.17 GiB from profiling) then
   OOM'd on the first prefill (same 532 MiB buffer, free: 0 ×2) → HSA
   `HSA_STATUS_ERROR_OUT_OF_RESOURCES` abort in `_fwd_kernel`
   (rocdevice.cpp:4207), no kernel reset in journal. Root cause:
   profiling-based KV sizing vs the 532 MiB runtime buffer (the 532 MiB
   buffer is larger than Qwen3.8-27B's 356 MiB, so 0.93 — which works for
   27B — is too tight here). Fix: explicit `kv_cache_memory_bytes` cap +
   `BENCH_KV_MEM` harness hook. SIGTERM teardown, VRAM 0%.
4. **~23:38Z — weight-load `hipErrorLaunchFailure` (eager probe, 2 GiB
   KV cap).** SIGABRT at shard 3/5; journal clean (no BACO/reset this
   time), rocm-smi 0% after. ~60 s after the previous run's OOM-teardown
   exit (23:37) → same collateral pattern.
5. **~23:59Z — weight-load `hipErrorLaunchFailure` (graph run, 0.75 GiB
   KV).** SIGABRT at shard 3/5; journal clean, rocm-smi 0% after. Again
   immediately after an OOM-teardown exit (23:57) → collateral pattern.
6. **~00:0xZ (08-27) — retry clean.** Weights loaded 100%, capture OK;
   at this KV cap the first prefill still OOMed (free 128–132 MiB —
   allocator-level; the all-CUSTOM arm's per-layer buffers are ~2.6 GiB
   larger than the hybrid's — see `DEVLOG-muse-glimmer.md`, memory
   forensics). A `BENCH_BATCHED_TOKENS=1024` re-run then loaded clean
   and completed the full gate bench.

**Assessment:** two allocator-level OOM aborts (expected — the model's
first-request memory profile exceeded the profiling-based KV sizing) and
two weight-load launch-failures, both within a minute or two of an
OOM-teardown exit. No independent wedge signature (no unexplained BACO,
no full wedge, journal otherwise clean). Both cards clean at session
end; no vllm procs left. If weight-load failures start appearing WITHOUT
a preceding OOM abort, treat as independent wedges and apply the burst
rule.

Logs: `/local/tmp/muse/` (`bench_hybrid_graph*.log`,
`bench_allcustom*.log`, `smoke_*.log`, `probe_allcustom*.log`); dev log:
`DEVLOG-muse-glimmer.md`.

## 2026-08-27 08:2x–08:3xZ (boot I): weight-load `hipErrorLaunchFailure` under a concurrent 16-way build — retry clean

**Context.** Boot I is ~11 h 45 min old; the 08:1x–08:2x window-FA
follow-up session ran two full B=4 bench launches that loaded clean.
The 3rd launch (B=4 `GFX906_FA_KVSPLIT=2` arm) SIGABRT'd at weight-load
shard 4/5 with `c10::AcceleratorError: CUDA error: unspecified launch
failure` (HIP `hipErrorLaunchFailure`).

**Unusual confounder.** A 16-way ccache/clang rebuild of the gfx906 FA
extension (`setup.py build_ext --inplace`, Phase C kernel change) was
running concurrently — the first observed overlap of a heavy CPU build
with a weight load. No OOM preceded it (unlike both boot-I events from
the onboarding session), so per the onboarding-session rule this counts
as an independent launch-failure, not OOM collateral.

**Unknowns.** `dmesg`/`journalctl -k` unreadable (no root) — cannot tell
whether a BACO/kernel reset ran; rocm-smi showed both cards clean
immediately after (VRAM 0%/0%, 38/31 °C, normal SCLK/MCLK), and a torch
matmul probe was not needed because the retry loaded and served clean.

**Outcome.** Isolated (1st since the ~00:0x session) → one retry per
house recipe at 08:41Z: weights loaded clean, full 4-sample bench
completed (20.55 t/s). No further launch-failures; the two subsequent
pp4096 long-context bench launches also loaded clean.

**Open question for the table:** does sustained high CPU load (16-way
clang) during a 20 GB weight load raise the launch-failure rate? One
data point; if weight-load failures recur under concurrent builds,
serialize builds and loads.

## 2026-08-27 14:19–14:32Z (boot I): OOM teardown → 2 consecutive weight-load/init `hipErrorLaunchFailure` — burst, session stopped

**Context.** Review-round-2 session: the LEGACY=0 Q8-side-buffer smoke
(first one with an *uncapped* KV pool on this model) OOM'd at
14:19:25Z — `aten::empty` in `gptq_gemm` inside the AOT/inductor
runtime (the 104k-token pool + ~1.5 GiB Q8 side buffer + inductor
headroom exceeds 32 GiB; allocator-level, not HW). The same smoke with
a 0.375 GiB KV cap then ran to completion (garbage output = the
expected side-buffer desync finding).

**The wedge.** ~11 min after the OOM teardown, the standard
LEGACY=1 smoke SIGABRT'd at weight-load shard 2/5
(`c10::AcceleratorError: CUDA error: unspecified launch failure`,
`hipErrorLaunchFailure`, raised in SetDevice) — `/local/tmp/muse/
smoke_final.log`. rocm-smi remained fully responsive throughout
(both cards VRAM 0%, 33/31 °C, SCLK 938 MHz) — the silent-wedge
variant: the device answers management queries but rejects launches.
The one permitted retry (14:32Z) also SIGABRT'd, at engine init before
any weight load — `/local/tmp/muse/smoke_final2.log`.

**Classification.** The OOM-teardown collateral pattern (precedents:
2026-08-26 22:41, ~23:38, ~23:59; 2026-08-23 08:51) — both failures
follow the 14:19:25 teardown with no other GPU work between. But two
consecutive failures = **burst per house recipe**: GPU work stopped;
reboot (root) required before further inference.

**What completed before the wedge.** All of review round 2's
verification: the pre-fix repro of the P1 clip bug (old .so), the
post-fix verification (new .so), 45/45 suite, the LEGACY=0 garbage
smoke (desync blocker), the window-check rejection probes, and the
pp8192 clip A/B (later identified as a null test — both arms ran the
gather path). Nothing GPU-dependent is pending.

## 2026-08-27 17:40–17:52Z (boot J): TP=2 serving OOM (pool sizing) → force-kill → 2 consecutive `hipErrorLaunchFailure` relaunches — stopped, reboot required

**Timeline.**

1. **17:30Z** — Muse-Glimmer-30B **TP=2** ngram serving launched clean
   (first TP=2 launch this boot; official-driver stack). Weights 12.68
   GiB/GPU, KV pool **9.02 GiB/GPU** (1,358,787 tokens; util 0.82, no cap),
   graphs 1.28 GiB. Single-request checks + a 4-parallel batch (short
   prompts, all <100 tokens) + tool/reasoning parser checks all clean.
2. **17:40:33Z** — first request with a real 4096-token prefill (bench
   sanity, pp4097): OOM `aten::empty` in `gptq_gemm` (AOT runtime).
   Budget math from the engine log: steady state 14.48 (weights +
   non-torch) + 1.28 (graphs) + 9.02 (KV) = 24.78 GiB of 31.98 physical
   → <7.2 GiB headroom; the runtime bt4096 inductor prefill buffer
   exceeded it (profiled peak activation was only 2.73 GiB — the
   warm/cold gap is much larger than the documented 0.16 GiB 27B case
   at this model size/batch). Allocator-level, the same class as the
   TP=1 532 MiB case (boot I 22:30) — the pool was simply oversized.
   **The engine force-killed the surviving worker** (`force killing
   remaining process EngineCore`) — a SIGKILL mid TP=2 P2P op.
3. **17:45:49Z** — relaunch (with `--kv-cache-memory-bytes 6 GiB` cap):
   `hipErrorLaunchFailure` at worker init/SetDevice, both ranks.
4. **~17:49Z** — TP=1 mtp2-27B canary: **38.4 t/s, healthy** (no P2P).
5. **17:52:06Z** — retry: `hipErrorLaunchFailure` again, both ranks,
   22 s into weight load. Journal unreadable (no root) — BACO/reset
   status unknown. rocm-smi clean after (both 0%/0%, 32–33 °C, 938 MHz),
   no stale processes.

**Classification.** Two components, both matching documented patterns:

- The OOM itself is pure pool-sizing (SW): fixed by the explicit KV cap
  (`--kv-cache-memory-bytes 6442450944` → ~900k-token pool, ~10 GiB
  headroom/GPU). The uncapped 0.82 config is NOT usable for this
  model at bt4096 — the README row must carry the cap.
- The two consecutive `hipErrorLaunchFailure`s after a mid-P2P SIGKILL
  match the AGENTS.md TP=2 teardown note exactly ("SIGKILL leaves the
  driver mid-P2P-op and the next init wedges GPU1 … BACO reset + retry
  needs root"). The healthy TP=1 canary between the two failures argues
  against host degradation (which would slow TP=1 spec decode too);
  this is a P2P-path stall. Per house recipe, 2 consecutive launch
  failures = stop; a BACO reset (root) or a reboot is required before
  further TP=2 work on this boot.

**Pending on next clean boot (reboot, then canary first).**

- Muse TP=2 ngram serving validation + benchmark grid
  (`/local/tmp/muse/bench_serve_grid.py`: pp2048/8192/16384 ×
  tg256/512, B=1 streaming TTFT/decode split + one B=4 point at
  pp2048/tg256) with the capped pool; server args preserved in
  `/local/tmp/muse/muse_tp2_ngram3.log` (header).
- README model-row numbers for the serving config (decode t/s per
  context + prefill t/s) — see the DEVLOG-muse-glimmer round-3 notes.
- Optional: greedy (no-spec) baseline B=1 for the ngram-lift number
  (needs a spec-off server).

## 2026-08-27 18:23–18:29Z (boot K): chronic weight-load hang (GPU1 reset) → clean retry → silent process-group kill (operator-aborted launcher), not HW

**Timeline.**

1. **~18:19Z** — boot K (after the 17:52 stop). rocm-smi both cards clean
   (0%/0%, 32–33 °C, 938 MHz). mtp2-27B canary **38.8 t/s** (healthy;
   prior passing canaries 38.4–38.9).
2. **18:22:31Z** — Muse TP=2 serve launched with the 6 GiB KV cap
   (`--kv-cache-memory-bytes 6442450944`; args unchanged otherwise).
3. **18:23:30Z** — weight load hung at shard 2/5→3 on both ranks
   (`hipErrorLaunchFailure` in `copy_`→SetDevice) — the chronic
   weight-load-hang signature (6th occurrence across boots).
4. **18:23:47Z** — kernel: `Fence fallback timer expired on ring
   comp_1.0.0` → `GPU reset(1) succeeded` on **0000:0e:00.0 (GPU1)** →
   "device wedged, but recovered through reset". First HW event of
   boot K. rocm-smi clean after.
5. **~18:25:30Z** — retry per house recipe: weights 5/5 in 42 s (clean),
   **KV pool 904,164 tokens — exactly the 6 GiB cap** (3.45× the 256k
   max, vs 1,358,787 uncapped), graph capture 9 s / 0.89 GiB (vs 1.28
   uncapped — smaller pool, smaller piecewise workspace), "Application
   startup complete" 18:28:07Z.
6. **~18:29Z** — the entire process group (API + EngineCore + both
   workers) died with **no log output, no kernel events, no OOM-kill**
   in kern.log, and full VRAM release (both cards 0%/0%, 34 °C).

**Classification.**

- Event 3/4 = the chronic intermittent weight-load hang, isolated on a
  fresh boot with a healthy canary and no OOM/SIGKILL precursor —
  same class as 08-26 06:03/06:52 (boot H) and 08-27 08:2x (boot I);
  all of those cleared on one retry, and so did this one.
- Event 6 is **not a GPU event**. The launcher was a single shell call
  (`nohup vllm serve … & sleep 240; grep …`); when that call was
  interrupted by the operator ("Loaded" interjection), the tool killed
  the call's process group — `nohup` only ignores SIGHUP, so the
  backgrounded server died with it. Distinguishing evidence: kern.log
  is empty for 18:28–18:31 (no fence timeout, no reset, no amdgpu
  error), no "Killed process" OOM lines, and the GPUs released all
  VRAM cleanly (a mid-op GPU kill would leave zombie VRAM / N/A
  rocm-smi). Boot J's launches used the identical pattern but *completed*
  their calls, which is why their servers survived.
- **Process-management lesson (recurring trap):** launch long-lived
  servers detached from the interactive call (`setsid nohup … &` from
  a stable terminal, or a plain `nohup … &` whose call is never
  interrupted) and check readiness in *separate* short calls.

**Validation state of the 6 GiB-cap config (as of this entry).**
Load, pool sizing (exactly the requested 904,164 tokens), and graph
capture all clean on the retry. The one remaining validation is the
**first real 4096-token prefill** — the exact site of the boot-J OOM.
Pending with the operator-relaunched server: sanity request (pp4096) →
the prefill grid (`docs/gfx906/_bench_serve_grid_gfx906.py`) → the
README serving row.

## 2026-08-27 20:40–20:42Z (boot K): in-process OOM-attribution probe wedges GPU0 mid weight-load; driver self-recovers

**What ran.** The OOM-attribution probe for the boot-J/K first-prefill
OOM (`/local/tmp/muse/probe_oom_attribution.py`, custom arm = our
GFX906_FA backend, TP=1, in-process, 0.5 GiB explicit KV cap, PP=4097
so the first prefill chunk is 4096 = the OOM site,
`VLLM_USE_AOT_COMPILE=0` after the AOT-worker crash, see
DEVLOG-muse-glimmer.md round 3 for the inductor-host-crash chase).

**Timeline.**
1. **~20:40:59Z** — model load starts (5th in-process probe launch on
   boot K: custom3/4/6 died earlier in host-level inductor crashes,
   custom5 in an import-time AttributeError — none of those touched a
   GPU kernel launch hard, but the boot has also carried the 17:40/17:52
   boot-J relaunch failures and the 18:23 GPU1 reset).
2. **~20:41:0xZ** — at 20% into weight load (shard 1/5→2 boundary):
   `terminate called after throwing an instance of 'c10::AcceleratorError'
   what(): CUDA error: unspecified launch failure` — the **7th
   occurrence** of the chronic weight-load hang across boots (same
   family: 08-26 06:03/06:52, 08-27 08:2x ×2, 17:40/17:52, 18:23).
   kern.log/journal unreadable (no root), so the reset type is
   inferred, not observed.
3. **~20:42Z** — rocm-smi: both cards back at the 10.8 MB VRAM
   baseline, 0% util. The driver completed its own reset and the GPU
   is usable again — an *isolated* wedge with self-recovery.

**Assessment.** Isolated per the house recipe → one retry of the probe.
If the retry wedges again that is a burst (2 consecutive launch
failures on this boot's tail) → stop GPU work, reboot (root). Note the
boot has now accumulated 3 GPU-side incidents (18:23 GPU1, 20:40 GPU0,
plus the boot-J carry-over); the degradation-onset question (when does
a boot enter the state where weight loads wedge?) stays open — see the
Open questions section.

## 2026-08-27 21:57–21:59Z (boot K): OOM-attribution probe (q_pad-fix verification run) wedges GPU0; driver self-recovers

**Context.** Boot K's 4th GPU-side incident (after 18:23 GPU1, 20:40
GPU0, plus the 21:0x/21:1x/21:2x in-process probe launches that ran
clean). The run was the post-fix verification of the q_pad ClassVar
change (DEVLOG-muse-glimmer round 4 root cause;
`attr_tp1_custom18_fixed.log`): in-process custom arm, TP=1, 0.5 GiB
KV cap, PP=4097, weight load started ~20 s in.

**Timeline.**
1. **21:57:28Z** — engine init: backend registered, weight load
   begins (GPU0).
2. **~21:58:3xZ** — `terminate called after throwing an instance of
   'c10::AcceleratorError' what(): CUDA error: unspecified launch
   failure` — the 8th occurrence of the chronic launch-failure
   family across boots. kern.log/journal unreadable (no root).
3. **~22:00Z** — rocm-smi: GPU0 back at the 11.2 MB VRAM baseline,
   0% util; the driver completed its own reset (self-recovery, same
   pattern as 20:40). GPU1 unaffected (a test suite ran on it
   throughout).

**4. ~22:03Z** — a 30 s torch canary on GPU0 (200 fp16 matmuls)
passed: 3.01 s, clean. GPU1 unaffected throughout (the 51/51
q_pad-fix test suite completed on it at ~22:06).
5. **~22:06Z** — the ONE allowed retry of the probe
(`attr_tp1_custom19_fixed_retry.log`) wedged GPU0 again — 2nd
consecutive launch failure, same `unspecified launch failure`
signature. And after this one the driver did NOT self-recover: GPU0
stuck at **24.9 GB zombie VRAM**, 0% util (the 20:40 and 21:58
crashes both released to the ~11 MB baseline; this one held the
reservation).

**Assessment.** **BURST per house recipe (2 consecutive launch
failures) → all GPU work stopped; the host needs a reboot (root)** —
BACO reset also needs root, and the 2nd-failure rule stops short of
even attempting it. Boot K's GPU incident count: 18:23 (GPU1), 20:40
(GPU0, self-recovered), 21:58 (GPU0, self-recovered), 22:06 (GPU0,
ZOMBIE VRAM). This boot is done for GPU work.

**Pending post-reboot** (in order): (1) canary (Qwen3.8-27B mtp2,
expect 38–47 t/s); (2) q_pad-ClassVar fix verification — re-run the
OOM-attribution custom arm (expect: one-time 256 MiB grow, survival
at the 0.5 GiB KV cap, transient ≈ model core + 0.26 + churn, i.e.
~1.5 GiB vs the 3.785 pre-fix); (3) M1 gather-clip e2e A/B
(pp8192/B=1 tg256, `GFX906_FA_GATHER_CLIP` 1 vs 0, record recipe);
(4) bt4096 TP=2 serving re-validation — the fix removes the
6.7 GiB/GPU q_pad growth that forced the bt2048 workaround, so the
boot K launch recipe can drop `--max-num-batched-tokens 2048` (and
the 6 GiB KV cap can shrink) if prefill clears; (5) write the
gfx906-mem-attribution skill (roadmap Housekeeping) with the
validated recipe (3-arm matrix + per-layer `memory_allocated()`
hooks + the env traps: `VLLM_USE_AOT_COMPILE=0`, thread compile
pool, `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`).

## 2026-08-28 ~06:14Z (boot L): post-reboot canary + q_pad-fix verification — clean

1. **~06:14Z** — fresh boot (uptime 1 min at check); both GPUs at the
   10.8 MB baseline, 0% util.
2. **Canary** (Qwen3.8-27B mtp2, GPU0, `canary_bootL.log`): **38.8
   t/s** — in the recent healthy band (38.4–38.9). Host clear.
3. **q_pad-ClassVar fix verification** (pending list item 1; custom
   attribution arm, TP=1, 0.5 GiB KV cap, PP=4097,
   `attr_tp1_custom20_bootL.log`): **SURVIVED**, peak transient
   **1.285 GiB** (vs 3.785 pre-fix on boot K), 4.89 GiB free after
   (vs 0.00). The boot J/K first-prefill OOM root cause is confirmed
   fixed (DEVLOG-muse-glimmer round 4). Remaining pending: M1
   gather-clip e2e A/B, bt4096 TP=2 serving re-validation, the
   gfx906-mem-attribution skill.
4. **~06:40Z** — **M1 gather-clip e2e A/B: PASS, +8.1%** (harness
   record recipe, pp8192/B=1/tg256, `GFX906_FA_GATHER_CLIP` 1 vs 0:
   6.042 vs 5.587 t/s; DEVLOG-muse-glimmer round 5).
5. **~06:55Z** — **bt4096 TP=2 serving re-validation: PASS** (boot K
   recipe + `--max-num-batched-tokens 4096`; first real 8192 request
   cleared — the exact boot J/K OOM site — cold prefill 452 t/s,
   warm ~99 t/s decode @8k/B=1, 8.7 GiB headroom/GPU; clean SIGTERM
   teardown to the 10.8 MB baseline; the bt2048 workaround is
   droppable — README updated).
6. **~07:05Z** — **gfx906-mem-attribution skill written**
   (`/home/kread/.agents/skills/gfx906-mem-attribution/SKILL.md`);
   the attribution probe persisted in-repo
   (`docs/gfx906/_probe_mem_attribution_gfx906.py`). All boot-K
   post-reboot pending items complete. Boot L clean throughout
   (canary 38.8 t/s; no wedges).
7. **~12:43Z** — **1st wedge of boot L: dual weight-load launch
   failures (GPU0 + GPU1 simultaneous)**. Two in-process harness runs
   (G1 B=2 / G2 B=4 `GFX906_FA_WINDOW_CLIP` A/B for the LEGACY=0
   default-flip gate, Muse-Glimmer, default LEGACY=1) both aborted at
   weight load — `terminate called after throwing an instance of
   'c10::AcceleratorError' ... CUDA error: unspecified launch
   failure` (`hipErrorLaunchFailure`), in the Exllama load path
   (checkpoint stats logged, then terminate — before any FA kernel
   ran, so the round-6 fused-clip rebuild is not implicated by the
   crash site). Both cards affected in the same ~30 s window ~6.5 h
   into boot L; rocm-smi self-recovered (0%/0% both cards, no zombie
   VRAM) within ~1 min of the crashes — matches the chronic
   weight-load-hang family (9th occurrence across boots) and the
   host-degradation signature (sync-cadence-heavy load work dies,
   driver recovers). Evidence: `/tmp/g1_clipon_b2.log`,
   `/tmp/g2_clipon_b4.log` (boot-volatile; signatures quoted here).
   **Canary skipped on operator instruction** ("card was working just
   now") — if the retry loads but t/s land low, the host is the
   suspect. Protocol state: isolated → 1 retry (G1 on GPU0); a 2nd
   launch failure = burst → stop all GPU work + reboot (root).
8. **~12:54Z** — **2nd boot-L wedge pair: concurrent G1(n=2)/G2(n=4)
   relaunch, both cards, same weight-load `unspecified launch
   failure`** (evidence: `/tmp/g1_clipon_b2_n2.log`,
   `/tmp/g2_clipon_b4_n4.log`; self-recovered to 10.8 MB both).
   Context: the 12:43 pair's single-card retry (G1, ~12:52) loaded
   clean and ran to completion at **6.06 t/s** (B=1/pp8192/tg256 vs
   the 6.042 pre-wedge record — host healthy mid-interval). Pattern
   so far on boot L: every double wedge followed a CONCURRENT
   two-card launch; every clean run was single-card. Hypothesis
   (unverified): two heavy HSA loads in the same ~30 s window race a
   host/driver resource on this dual-root-port topology. Mitigation
   from here: serialize all GPU work (one in-process or one TP=2 job
   at a time). Protocol state: 2nd observation, chain was broken by
   the 12:52 success → G1 retried once; a 3rd observation or a retry
   failure = burst → stop + reboot (root).
9. **~14:38Z** — **3rd boot-L wedge: G3 LEGACY=0 TP=2 bake retry,
   `hipErrorLaunchFailure` in weight `copy_`→SetDevice (Worker_TP1,
   both ranks; evidence: `/tmp/g3_legacy0_serve.log` 2nd write, boot-
   volatile; signatures quoted here).** First G3 attempt (~14:20)
   died with a CODE error instead — a capture-unsafe D2H sync
   (`int(cu[s+1]-cu[s])`) in the direct-paged branch's Sq>1 Q-pad
   loop, hit for the first time in production (LEGACY=0 + ngram spec
   + B≥2 decode = Sq=6 direct-paged under FULL decode capture);
   fixed in-tree (uniform-batch fast paths, mirroring the gather
   branch's capture-safe idiom; 57/57 suite incl. the nq=6/B=2
   bit-identity A/B now exercising them). The retry then wedged at
   weight load — the 14:20 code death had already torn down both
   workers, so the wedge may be collateral to that teardown (cf. the
   documented SIGKILL-mid-P2P → next-init wedge), or a 3rd
   accumulation. Either way: **3rd wedge observation of boot L =
   BURST per the protocol recorded at 12:54Z → ALL GPU WORK STOPPED;
   host needs a REBOOT (root).** Pending post-reboot: the G3 LEGACY=0
   TP=2 serving bake (recipe: `HIP_VISIBLE_DEVICES=0,1
   GFX906_FA_LEGACY=0` + README TP=2 flags; grid
   `_bench_serve_grid_gfx906.py` default ×3; control = the boot L
   LEGACY=1 records 111.5/99/46.7 @2k/8k/B=4). Pattern note: all 3
   boot-L wedges involved two-card launches; all 6 single-card
   in-process runs (incl. the post-wedge A/Bs, 13:0x–14:3x) were
   clean — the concurrent-two-card race hypothesis from 12:54Z is
   untested by design (TP=2 IS two cards), and boot L is ~8 h in
   with 3 wedges accumulated — consistent with the degradation-onset
   model (enough half-wedge resets in one boot degrade the host).
10. **~15:00Z (boot M)** — **1st boot-M wedge: G3 LEGACY=0 TP=2
    attempt 1, `hipErrorLaunchFailure` both ranks at weight load**
    (~5.5 min in; evidence: `/tmp/g3_legacy0_serve.log` boot-M
    write, boot-volatile; signatures quoted here). Context: fresh
    boot (uptime ~8 min), canary **38.9 t/s** at ~14:58Z (2 min
    prior) — host healthy at launch. Chronic weight-load-hang
    family, 10th occurrence across boots; the TP=2 two-rank weight
    load is the usual trigger (cf. boot K 18:23:30, 7th). Driver
    self-recovered to the 10.8 MB baseline within ~4 min; no zombie
    VRAM. Protocol state: isolated → 1 retry; a 2nd launch failure
    on boot M = burst → stop + reboot (root).
11. **~17:18Z (boot M)** — **2nd boot-M wedge: M6 Part B bake arm 1
    (LEGACY=0 + `GFX906_FA_DIRECT_PAGED_Q8=1` TP=2 serve),
    `hipErrorLaunchFailure` in `copy_`→SetDevice at weight-load
    shard 4/5→5, Worker_TP0 (both ranks; evidence:
    `/tmp/m6b_serve_l0_dpa1.log`, boot-volatile; signatures quoted
    here)**. Kernel: `GPU reset(2) succeeded` on 0000:0e:00.0
    (GPU1) + `GPU reset(1) succeeded` on 0000:0b:00.0 (GPU0) at
    17:18:27Z, both "device wedged, but recovered through reset".
    Chronic weight-load-hang family, 11th occurrence across boots.
    Context: ~2 h clean single-card window since 15:00Z (unit suite
    60/60, in-process B=4/pp8192 bench arm, canary **38.4 t/s** at
    ~16:59Z) — the 15:10Z clean G3-attempt-2 success already broke
    the 15:00 chain (boot L 12:52 precedent: a success resets the
    consecutive-failure count), so this is a NEW isolated
    observation, not the 15:00 chain's 2nd link. Pattern note: as
    on boot L, both boot-M wedges are two-card (TP=2) launches;
    all boot-M single-card work clean. Protocol state: isolated →
    1 retry per house recipe; **a retry failure = burst → stop +
    reboot (root)** (boot M would then be at 2 failures in the
    retry chain, 3 wedge observations total ~2.4 h in).
12. **2026-08-29 ~05:5x–06:00:02Z (boot M, ~15 h in)** — **3rd boot-M
    wedge: an in-process (TP=1, GPU0) Muse 2×130k-token prefill hung
    mid-stream** — observed ~05:57Z as GPU use 0 % + main thread
    200 % CPU spin (two cores), no log progress since the 05:50:51
    warmup line; SIGTERM 05:59:16Z; kernel then logged `qcm fence
    wait loop timeout expired` (05:59:59) → `GPU reset(2) succeeded`
    (06:00:02) on 0000:0b:00.0 (GPU0), "device wedged, but recovered
    through reset". First boot-M wedge on a SINGLE-CARD launch (the
    15:00/17:18 pair were TP=2 weight-load hangs) and on RUNNING
    INFERENCE, with the `qcm fence wait loop timeout` signature
    (GPU stops honoring a fence mid-stream; the QCM CPU wait-loop
    gives up) rather than the chronic `hipErrorLaunchFailure`
    load-family.

    **Bisection (all TILE_CLIP=0 / the pre-M2 path — the wedge is
    NOT M2-attributable), boot M:** pp16384 completes
    (128.4 s/sample, 2×16k); pp32768 pass-1 136 s, pp65536 pass-1
    306 s; the later "stalls" were slow passes meeting my timeouts,
    not hangs (a 130k pass at the boot's prefill rate is ~9 min).

    **RESOLVED 2026-08-29 ~07:58Z on the fresh boot (boot N):** the
    same 32k shape (in-process, TP=1, bt4096, main build) measured
    warmup 135.48 s + sample 136.51 s on a ~2-min-old boot with 0
    wedges and a 38.9 t/s canary — matching boot M's 136 s exactly,
    no pass-to-pass drift. **Verdict: the ~214–256 t/s in-process
    TP=1 prefill rate is the TRUE rate for this hybrid model on this
    box, not degradation.** Consistency check: the ~450–540 t/s
    records are TP=2 — 32.8k tokens / 500 t/s = 65.6 s ≈ 135/2.06,
    i.e. prefill scales ~2× with TP as expected for the
    GEMM/FA-bound path. Boot M's only genuine anomaly remains the
    05:59Z fence wedge (isolated; no recurrence on boot N through
    ~08:10Z). Baseline recorded in the README Muse row so the rate
    is not re-derived. Evidence: `/local/tmp/m2iso_32k_bootN.log`
    (persists), boot-M copies in `/tmp/m2iso_*.log` (wiped by the
    reboot).

## 2026-08-29 13:40:36Z — boot N, B=1 LEGACY A/B bake arm B (LEGACY=0 TP=2)

    B=1 decode-gap item, serving A/B bake (Qwen3.8-27B TP=2, maxlen
    32768, the `_serve_tp2_gfx906.sh` recipe with the new
    EXTRA_SERVE_ENV passthrough). Arm A (LEGACY=1) loaded clean and
    produced the control numbers (B=1 39.76/40.12 t/s @ pp2048/tg256,
    `/local/tmp/b1ab_armA.log`); arm B (EXTRA_SERVE_ENV=
    "GFX906_FA_LEGACY=0") wedged at weight load, 13:40:36: Worker_TP1
    `c10::AcceleratorError: CUDA error: unspecified launch failure`
    surfaced at the load-model memory probe
    (`rocm.get_current_memory_usage`, an async launch failure from an
    earlier kernel in the load; no HwException/PSP lines in the log,
    rocm-smi showed both cards back at the 10.8 MB baseline / 0 %
    within ~1 min — the driver self-recovered, no zombie VRAM). This
    is the chronic two-card weight-load-hang family (12th occurrence
    across boots; boots L/M each had 3 before reboot) and the first
    HW observation on boot N. It is NOT plausibly LEGACY=0-specific:
    the failure is in the generic load path (the Q8 side buffer is
    built at KV-cache append time, not load time), and arm A loaded
    the identical weights ~10 min earlier on the same cards. Protocol
    outcome: isolated → one retry of arm B; retry failure = burst →
    stop + reboot (root). Server log: `/local/tmp/lcbench_b1ab_b_server.log`.

    **RETRY FAILED — 2026-08-29 13:42:05–13:43:04Z.** Identical
    signature: both ranks `c10::AcceleratorError: CUDA error:
    unspecified launch failure` (the log shows the duplicated
    `terminate` from both workers; 2 occurrences of `unspecified
    launch failure`), at the same weight-load stage; both cards
    self-recovered to the 10.8 MB baseline / 0 % within ~1 min,
    no zombie VRAM, no PSP/HwException lines. **2 consecutive
    launch failures on boot N = BURST per the house recipe → all GPU
    work stopped; reboot (root) required.** The two-card-launch vs
    single-card-clean split holds on boot N exactly as it did on boot
    L (all recent wedges two-card, all recent clean runs single-card
    GPU0). Post-reboot pending (see the degradation.md row): canary,
    then bake arms B/C on `feat/fa-legacy0-b1-decode` (arm A control
    already recorded: 39.76/40.12 t/s @ pp2048/tg256).

## 2026-08-29 15:51 + 16:09Z — boot O, B=1 bake arms A and C (two wedges, non-consecutive)

    Post-reboot bake of `feat/fa-legacy0-b1-decode` (arm A control
    already recorded on boot N: 39.76/40.12 t/s). Boot O: canary
    39.3 t/s clean at 15:49.

    **Wedge 1 (15:51:02, arm A 1st attempt):** kernel log — GPU1
    (0000:0e:00.0) `qcm fence wait loop timeout expired`,
    `The cp might be in an unrecoverable state due to an unsuccessful
    queues preemption`, `Failed to evict process queues`,
    `Failed to quiesce KFD`, `GPU reset begin! Source: 4`, PSP
    `UNLOAD_TA(0x2) failed (0x117)`, **BACO reset**, `GPU reset
    succeeded, trying to resume` + coredump file created; userspace
    `unspecified launch failure` both ranks at the shard-5 weight-load
    stage. Just before: `amdkfd_restore_userptr_worker hogged CPU
    >10000us` (workqueue notice; the boot's kernel log is otherwise
    clean from the 15:37 driver init — no earlier reset this boot).

    **Between the wedges, four clean two-card launches:** bare
    torch/RCCL probe (50 allreduces, 35.2 ms/iter, ~15:55), arm A
    retry (loaded 345 s, served, benched 40.11/40.12 t/s, clean
    teardown), arm B (LEGACY=0) (loaded 325 s, served, benched
    37.61/37.56 t/s, clean teardown).

    **Wedge 2 (16:09:11, arm C 1st attempt):** identical GPU1
    signature (`qcm fence wait loop timeout` → evict/KFD-quiet fail
    → GPU reset), userspace `unspecified launch failure` both ranks at
    weight load.

    Assessment: boot O is taking intermittent GPU1 qcm-fence wedges
    under two-card vLLM weight loads, with clean two-card launches
    between them — NOT the consecutive-failure burst pattern (the
    clean runs break the chain per the boot-L 12:52 precedent), but
    this is the 4th consecutive boot (L, M, N, O) with the two-card
    load-wedge pattern, and every boot-O wedge is GPU1. Chronic
    hardware/driver suspicion is now the working hypothesis (a
    reboot alone has not cleared it across L→M→N→O). Protocol: arm C
    gets one retry; a 3rd boot-O wedge OR the retry wedging stops
    GPU work pending host investigation.

## 2026-08-31 (boot P) — C2 combined A/B session: chronic load wedge + worker-cgroup OOM finding

    Boot P started 2026-08-30 20:51:32. Early-boot wedges
    (interactive session, pre-cron): GPU0 02:55:13 and GPU1 04:24:55 —
    both `qcm fence wait loop timeout` → BACO → "recovered through
    reset", the chronic two-card weight-load-hang family, intermittent
    with clean windows between (not a burst).

    **Cron C2 combined A/B (TP=2 M=1 default-on decision), 08:36–09:1xZ:**
    - Pre-bench canary both GPUs: 16 TFLOP/s warm, 0 resets in the
      window. Host judged healthy for A/B gating.
    - **TP=1 off arm (in-process harness): 82.37 ± 0.07 t/s** Δ-metric
      (fp d2e5262183c6b92f) — clean, load 391 s.
    - **Wedge 08:49:34 (TP=1 `MOE_M1=1` arm):** `unspecified launch
      failure` at weight-load shard 9/9 → qcm fence timeout +
      unsuccessful queue preemption → BACO reset on GPU0, "device
      wedged, but recovered through reset". NOT a MOE_M1 kernel fault:
      the flag only changes decode-time gemm2 dispatch (the engine died
      during load, before any MoE GEMM ran); v2-tile source unchanged
      since C2-V measured it clean (08-22/23). 3rd reset this boot but
      intermittent (clean windows + the full off arm between events) →
      isolated per house recipe. Post-reset canary: both GPUs 16
      TFLOP/s, VRAM 0%/0%.
    - **TP=2 in-process OOM finding (new, software):** the first TP=2
      engine (off arm) was SIGKILL'd at NCCL init by the **hermes
      worker cgroup** (`memory.max` cap on
      `app.slice/hermes-worker-proc_*.scope`, ~4 GiB): two TP workers
      (~1.8 GB RSS each + shmem + inductor) exceed it. No GPU event.
      Fix per the standing rule: long vLLM work under **systemd user
      services with MemoryMax=infinity** (new unit `c2arm@.service`,
      one-shot per arm). Under systemd the TP=2 off arm completed
      clean: **81.58 ± 0.58 t/s**, fp d2e5262183c6b92f (identical to
      the TP=1 off arm — numerics gate passes at the control point),
      unit peak 13.9 GB, 0 resets in the window.
    - Remaining arms (m1 / npt2 / both) run under the same systemd
      path with a burst guard (abort on 2 consecutive failures or ≥3
      resets since run start; canary before/after each arm).

    **Completion (09:1x–12:0xZ, interactive + cron sessions):** all
    four TP=2 arms done under systemd (m1 83.80, npt2 83.88, both
    85.65 t/s; off 81.58 — full table in `DEVLOG-moe-c2v.md` final
    section). The two missing TP=1 arms (m1, npt2) were re-run
    in-process at 11:5x–12:0xZ (the cgroup OOM only affects TP=2's
    dual-worker footprint): m1 84.41, npt2 84.64 t/s; the final TP=1
    both arm ran in-process at 12:53–12:56Z: 84.74 t/s. All 8 arms
    share output fingerprint d2e5262183c6b92f (numerics gate PASS).
    0 new resets after the 08:49:34 event; VRAM drained to ~10 MB/card
    between every arm; post-run canary clean. GPUs idle at session end.

## 2026-09-01 (boot Q) + 2026-09-02 — MTP-1 session: three wedges, two from old-vLLM code path

Boot Q = the clean reboot requested before trusting MTP-1 crossover data
(~16:35 UTC 2026-09-01). Three HW events across the MTP-1 work:

1. **2026-09-01 14:00:45 (GPU1, pre-reboot tail):** first Qwen3.8-27B TP=2
   MTP serve launch this boot — worker-init `hipErrorLaunchFailure` → qcm
   fence timeout → BACO reset, recovered. Chronic weight-load-hang family.
2. **2026-09-01 20:29:04 (GPU0, boot Q):** in-process TP=2 *profiler* launch
   wedged at weight load; `VRAM is lost due to GPU reset!`, PSP resume OK.
   First HW reset on boot Q — the in-process weight-load trigger recurs even
   on a clean-booted host. Both GPUs passed matmul post-reset. Did NOT
   invalidate the MTP-1 sweep (both arms completed 20:09, pre-event).
3. **2026-09-02 03:12:41–53 (BOTH GPUs):** old-vLLM docker image
   `aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0` TP=2
   serve → `HW Exception ... GPU Hang` at weight-load shard ~2/5 on both
   cards simultaneously; both wedged, both recovered via reset + PSP resume.
   First dual-GPU simultaneous hang on this host. Userspace: segfault in
   `__clone`. Suspect: old ROCm 7.2 container userland vs modern kernel
   driver (6.8.12-acso).
4. **2026-09-02 06:10:17 (GPU0):** old-vLLM 0.23.1 **in-process** via
   PYTHONPATH on the MODERN ROCm 7.14 userland (platform detection fixed via
   sitecustomize RocmPlatform pin — the old fork predates our amdsmi→
   `torch.version.hip` fallback, and under this userland `import vllm`
   poisons amdsmi to 0 handles) → same `unspecified launch failure` at
   weight-load shard ~2/5 → BACO reset, recovered. **Same signature as event
   3 on a different userland** → root cause is the old vLLM code path itself
   (predates our gfx906 weight-load/FA fixes), not userland or harness env.
   Old-vLLM A/B abandoned as not-runnable on this host.

**NEW failure mode — zombie KFD handle (event 4):** dead worker PID 44222
(gone from /proc; `kill -9` is a no-op) still holds **10.99 GB on GPU0** per
`rocm-smi --showpids`. Kernel-level amdgpu leak post-reset that userspace
cannot clear — only a reboot releases it. Consequence: GPU0 has ~23 GB free,
below the ~29.7 GB a TP=2 27B AWQ @util 0.93 needs → all remaining MTP-1 work
(rocprofv3 decode kernel breakdown @120k) is blocked until reboot.

**Takeaways:** (a) old-vLLM branches must not be pointed at this model on this
host — they wedge the GPUs at load; (b) a GPU reset can leave an unreclaimable
KFD VRAM handle even when rocm-smi shows "recovered" — always re-check
`--showpids` + actual free VRAM before planning the next run, and treat
"recovered but <N GB free with no live process" as a reboot-required state;
(c) in-process weight-load wedges are chronic on boot Q too (event 2) — keep
using serve-based systemd launches for load-sensitive work.

## 2026-09-02 (boots R + S) — MTP-1b session: wedges #4–#7, all GPU0, all non-deterministic

Boot R (~08:5x) and boot S (~15:4x) reboots were both user-authorized
(`~/bin/hermes-reb.sh`) to clear host state before trusting MTP-1 data.
Four more GPU0 wedges landed across them; every one recovered via BACO
reset with a passing matmul canary immediately after, and VRAM returned to
the ~11 MB baseline (no zombie KFD handles this time — the 06:10 leak did
NOT recur).

4. **2026-09-02 15:4x (boot S):** in-process TP=2 `LLM()` + mode-NONE
   phase-profiler diagnostic run — `hipErrorLaunchFailure` at worker init →
   BACO reset on 0000:0e:00.0. Same family as the boot-R 08:54/10:56 events.

5. **2026-09-02 16:43:34 (boot S):** in-process MTP-arm profiler run —
   `hipErrorLaunchFailure` at init → kernel wording escalated to **"The cp
   might be in an unrecoverable state"** + failed queue evict/quiesce → BACO
   reset. The greedy arm of the identical config completed clean minutes
   earlier → non-deterministic race, not a deterministic code bug.

6. **2026-09-02 19:57:57 (boot S):** standard `vllm serve` startup for the
   MTP-1b k=1 arm — wedged at the CUDA-graph-capture phase, same "cp
   unrecoverable" wording. **First serve-based TP=2 wedge on this host** —
   all five prior were in-process LLM()+mode-NONE or old-vLLM code paths.
   Retry started clean and served the full k=1 sweep.

7. **2026-09-02 23:24:15 (boot S):** k=2 server on the kv-split-fixed build
   with a `GFX906_FA_KVSPLIT=1` drop-in (a causation-check run) — worker
   init failed at `SetDevice`, same "cp unrecoverable" + BACO reset. Hit
   before ANY FA kernel ran → unrelated to the kv_split change.

**Assessment:** seven non-deterministic GPU0 wedges across two boots,
canaries passing between each, no config-family commonality (in-process
mode-NONE, serve graph-capture, and bare SetDevice all hit). This is strong
evidence of genuine GPU0 HW degradation — the RMA/replace question should be
raised with Kevin rather than continuing to burn boots on retries. The
kv-split A/B result stands regardless: it was measured on a healthy window
(2.38–2.80× at 64k/96k/120k, n=3 each) and the causation question is
answered by unit-level correctness (kv_split ∈ {1,8,16} bit-identical for
Sq ≤ 1024 on both paths) plus that A/B.

## 2026-09-03 (boot T) — SYV-3/SYV-9 session: wedges #8–#13; tp=2 instability accepted as HW-related

Kevin's ruling for this boot (2026-09-03): tp=2 instability on this dev
system is **accepted as hardware-related** — no RMA escalation, no
wedge-chasing; keep canary checks + log entries. Wedges #8–#11 (earlier in
the day) were the usual mix: 3× in-process `LLM()` mode-NONE weight-load
hangs (06:47, 11:29 GPU0; 16:45 GPU1 — first GPU1 of the boot), all
recovering via BACO reset with clean canaries after.

12. **2026-09-03 22:34:52 (boot T):** serve-based TP=2 MTP arm0 weight load
    (`mtp1srv@mtp`, util 0.85, k=2, skinny-GEMV drop-in OFF) — ~60s into
    weight load, `unspecified launch failure` both ranks → fence-fallback +
    BACO reset on GPU0 (reset #5). **First serve-based TP=2 weight-load wedge
    this boot** (all prior serve launches today were clean). Confounder: a
    standalone Triton GEMV microbench ran concurrently during the window —
    first wedge with concurrent GPU work; unproven as cause. Process survived
    but never became healthy; driver killed it at the 675s health timeout.

13. **2026-09-03 22:45:56 (boot T):** arm1 weight load (same config, skinny
    GEMV default ON — final reviewed code) — same signature ~11 min later,
    **no concurrent GPU work** → the #12 confounder is not required; plain
    weight-load coin-flip continues. Driver killed at 675s; SYV-3 A/B run
    voided. Canary clean after (39–41 t/s).

14. **2026-09-04 00:52:04 (boot T):** arm1 re-run via `mtp1srv-syv3arm1`
    (run_server.sh wrapper, GFX906_SKINNY_GEMV=1) — worker init wedged at
    `c10::cuda::SetDevice` ~60s in → fence-fallback + BACO reset on GPU0
    (reset #7). Note: an earlier raw launch failed on a *different* error
    (missing FLASH_ATTENTION_TRITON_AMD_ENABLE → flash-attn ImportError),
    not a wedge. Canary immediately after: **38.2–38.3 t/s, down from 41.2
    at 22:3x** — degradation trend on this boot.

15. **2026-09-04 01:02:36 (boot T):** arm1 re-run #2 — identical SetDevice
    wedge ~70s in, after a 3-min cooldown + canary recheck (38.2 t/s) and a
    clean KFD check (no zombie handles). **4th GPU0 wedge since 22:34 and
    3rd consecutive launch failure**; every reset "recovered" but the canary
    trend is downward (41.2 → 39.2 → 38.2).

**Assessment:** fifteen wedges this boot, all GPU0 except #11 (GPU1), all
recovered via BACO reset with baseline VRAM afterward (no zombie KFD
handles). Per Kevin's ruling this is the accepted HW-related tp=2
instability of this dev system. Operational lessons carried forward:
(1) no concurrent GPU work during a TP=2 weight-load window; (2) expect an
occasional wedge on any serve launch — retry once, don't reboot; (3) health
timeout 675s is too tight when a reset happens mid-load (a wedged-then-
recovered process may still be compiling) — bump to ~1200s on retries.
**New as of #14–#15:** three consecutive launch failures with a downward
canary trend (41.2 → 38.2 t/s) is beyond the "retry once" band — decision
taken: **reboot via ~/bin/hermes-reb.sh** for a clean state before the
definitive SYV-3 arm1 run; if SetDevice wedges recur on a fresh boot, this
exceeds the accepted-HW pattern and should be escalated (RMA conversation).

## 2026-09-04 (boot U) — post-wedge-reboot SYV-3 A/B resume: wedge #16 at first launch

Boot U started 01:06:52. Arrival state clean (31/32 °C, 0%/0% VRAM,
938 MHz). mtp1canary@mtp **38.3 t/s ×2** (01:26, 01:29) — within the host's
known passing band (38.2–39.3 across boots J–T), below the ~40–47 nominal
but consistent with every recent boot; above the <35 hard stop.
`syv3_final_test.py` ALL PASS on GPU at 01:30 (K=1 err 2.0e-3, K=1 kernel
1.7 µs / 742 GB/s vs torch.mm 6.9 µs — 4.01×).

16. **2026-09-04 01:31:41 (boot U):** SYV-3 A/B arm0 first launch
    (`mtp1srv@mtp`, TP=2 MTP serve, util 0.85, k=2, skinny-GEMV drop-in OFF),
    launched 01:30:35 — worker init wedged at `c10::cuda::SetDevice` ~68 s in
    (userspace `unspecified launch failure` both ranks) → kernel `Fence
    fallback timer expired on ring comp_1.0.0` + BACO reset on 0000:0b:00.0,
    "device wedged, but recovered through reset", 01:31:41–01:31:44. VRAM back
    to baseline; **no zombie KFD handle** (rocm-smi --showpids: only the
    gpuagent daemon).

    **This is the first SetDevice/weight-load wedge on a fresh boot — the
    boot-T recurrence pattern has crossed the reboot boundary**, which is the
    single event that triggers the pre-committed escalation clause in #15.
    Context before judging: every fresh boot since Aug 24 (D, E, G, H, J, K,
    M, O) also produced an early isolated wedge that recovered cleanly; what
    exceeds the accepted-HW pattern is a BURST (consecutive failures) on the
    fresh boot. House recipe authorizes exactly one retry — if arm0's retry
    ALSO wedges at SetDevice/weight load, that is 2 consecutive launch
    failures on boot U = burst → STOP all GPU work + escalate per #15.

    **Mapping note (new pitfall):** the BDF↔GPU-index mapping FLIPPED vs
    boot T. This boot's kernel says `Initialized amdgpu ... for 0000:0b:00.0
    on minor 0` and rocm-smi reports GUID 10709 for GPU[0] — the same GUID
    this machine has reported for its "GPU0" in recent rocm-smi output, so
    0b:00.0 = GPU0 on boot U (boot T had 0e:00.0 = GPU0). The historical
    table's BDF labels are boot-specific; always re-derive the mapping per
    boot from the `Initialized ... on minor N` kernel line + rocm-smi GUIDs.

17. **2026-09-04 01:39:21 (boot U):** SYV-3 A/B arm0 RETRY (driver's built-in
    one-shot retry after #16; `mtp1srv@mtp`, same TP=2 MTP config) — worker
    init wedged at `c10::cuda::SetDevice` again → kernel `qcm fence wait loop
    timeout expired` + BACO reset on 0000:0e:00.0 (**GPU1** this time),
    "device wedged, but recovered through reset", 01:39:21–01:39:24. VRAM back
    to baseline; no zombie KFD handle.

    **2 consecutive launch failures on boot U = BURST per house recipe → ALL
    GPU WORK STOPPED.** The SetDevice/weight-load pattern crossed the reboot
    boundary (#16) and recurred within ~8 minutes on the OTHER card (#17).
    This is exactly the scenario #15 pre-committed as escalation-worthy: it
    exceeds the accepted-HW "retry once" band. Post-stop canary 01:4x:
    **38.3 t/s — no perf degradation**, so this is a launch-wedge burst, not a
    DEG state; both cards probe-clean afterward (VRAM baseline, no zombies).

    Non-wedge observation for the record: arm1's launch at 01:42 (after the
    driver skipped arm0) **loaded weights cleanly** (5/5 shards 32 s + drafter
    12.5 s) but then hung at shm-broadcast ("No available shared memory
    broadcast block found in 60 seconds") — the known one-off init-deadlock
    family (cf. 2026-08-23 ~11:36 entry). Stopped by operator per the burst
    decision before it could resolve; NOT counted as a wedge.

**Assessment (boot U):** two SetDevice wedges in the first ~9 minutes of GPU
use, on both cards, both recovered via BACO with clean post-state and a flat
canary (38.3 t/s ×3). Per Kevin's 2026-09-03 ruling tp=2 instability is
accepted as HW-related for this dev system — but the pre-committed clause in
#15 ("if SetDevice wedges recur on a fresh boot, escalate") has now been
triggered. **Escalation to Kevin: RMA/replace conversation warranted.** No
further GPU work until Kevin decides; SYV-3 A/B is blocked at arm0 (arm0
skipped after 2 failed launches; arm1 never benched). Code state unchanged:
the head_dtype gate fix + one-shot diagnostic remain in the working tree,
verified by syv3_final_test.py ALL PASS (01:30) and by both canary runs
showing "SYV-3 skinny GEMV activated" — so the A/B is runnable as soon as a
launch window opens.

## 2026-09-04 (boot U) — CAT-1 pilot A/B: wedge #18 mid-decode @120k (silent, no kernel event)

**Context:** CAT-1 draft-vocab pilot driver (`/local/tmp/mtp1/cat1_pilot_driver.sh`),
arm0 = stock MTP k=2 TP=2 `mtp1srv@mtp` :8123 (util 0.85, maxlen 131072, capture
[1,2,3,4], NCCL Tree+LL). Canary pre-flight 39.2 t/s (~11:56, healthy band).

**Timeline:**
- 12:02:35 driver start (SKIP_CANARY=1 — canary passed minutes earlier)
- 12:12:20 arm0 ready after ~585s (weights 34.0+14.8 s; torch.compile 87.7 s
  backbone + 12.6 s eagle_head on WARM 31 GB AOT cache; graph capture 9 s)
- bench pp=65536: reps 38.9/38.8/38.8 t/s (median 38.85, n=3) — matches the
  known k=2-fixed @64k class (37.95 on boot S); coherence sample clean
- ~12:24:35 last server log line (`Running: 1`, gen throughput 25.6 t/s logger)
  while rep at pp=122880 was in flight → **both GPUs pinned 100%**, bench client
  (pid 34530) stuck in `do_sys_poll` with only 8 s CPU time, zero new log lines
- ~12:27 teardown: systemd stop + SIGTERM; my first manual pkill self-matched its
  own shell command line (`pkill -f "vllm serve"` contains the pattern) and
  SIGTERM'd itself mid-loop — sloppy but harmless (systemd's stop completed,
  GPU quiesced within seconds)
- post-teardown: no zombie KFD handle (only arm1's new PIDs on /dev/kfd);
  kernel journal window has NO amdgpu events for this hang

**Signature:** first serve-based wedge NOT at weight-load/SetDevice/prefill —
this one hit during DECODE at 120k ctx, silently (no BACO, no kernel log).
Long-context zone per the degradation.md risk note; cf. boot-R 10:56
(prefill@122880 crash, GDN-Triton-decode path active). Non-deterministic HW
per Kevin's 2026-09-03 ruling (tp=2 accepted as HW-related on this dev system).

**Consequence:** arm0 pp=122880 point lost; re-run that single point after arm1
completes. Driver continued to arm1 (CAT-1 pilot work-dir, :8125) which loaded
weights cleanly (50.6 s + 18.1 s — the extra `model_extra_tensors.safetensors`
419 MB draft head included).

**Assessment:** single event, canary healthy minutes before and after; no burst.
Continue per house recipe (retry the lost point once arm1 is done). If decode-
zone wedges recur in this session, stop + report rather than rebooting mid-pilot.

## 2026-09-04 (boot U) — S1 startup-time session: wedge #19 at profile stage with --language-model-only

**Context:** S1 (startup compile/capture speed) instrumented-boot session. Baseline
stock boot (`s1base`, 15:13–15:20) completed cleanly: init engine 233.85 s,
compilation counter 0.95 s (warm AOT cache hit — vs 100.3 s on the 12:0x pilot
boot whose config-key cache was cold). Stack dumps (in-process SIGUSR1/dumper)
proved the ~213 s "pre-compile gap" is the **vision-encoder dummy forward**:
Worker_TP0 pinned in `gpu_model_runner.py:6611 profile_run → qwen3_vl.py:2882
embed_multimodal → vit_attn_wrappers.py vit_flash_attn_wrapper` for
15:15:26→15:18:51 (42 consecutive 5 s samples). The checkpoint is
`Qwen3_5ForConditionalGeneration` with `vision_config: true` + 333 `model.visual.*`
tensors, so every startup profiles a max-feature-size dummy image even for
text-only serving.

**The wedge:** L3 lever probe (`LANG_ONLY=1` → `--language-model-only`, stock
config otherwise) launched 15:55 under `systemd-run --user -p MemoryMax=infinity`.
Flag applied cleanly (both TP workers logged "All limits of multimodal modalities
supported by the model are set to 0, running in text-only mode" at 15:57:12).
At ~15:57:35 Worker_TP1 threw `c10::AcceleratorError: CUDA error: unspecified
launch failure` (`hipErrorLaunchFailure`) during init; kernel journal:
`Fence fallback timer expired on ring comp_1.0.0` → `GPU reset(2) succeeded!
[drm] device wedged, but recovered through reset` on 0000:0e:00.0 (GPU1).
API server then raised "Engine core initialization failed".

**Attribution:** NOT the flag's code path — `language_model_only` only zeroes
modality limits in config before any GPU work; the stock boot 10 minutes earlier
ran the identical init sequence cleanly, and this is the same chronic TP=2 HW
instability family (#16/#17 SetDevice, #18 mid-decode). Per Kevin's 2026-09-03
ruling: accepted HW-related on this dev system; no RMA escalation.

**State after:** GPU quiesced (both 0%), VRAM baseline, `fuser /dev/kfd` empty
(no zombie KFD handles — reboot not required). L3 startup win still unmeasured;
retry after a fresh canary, or measure the same lever on the TP=1 text-only path
(lower wedge risk per Kevin's "one GPU first" guidance).

**Instrumentation note (for future sessions):** v1 stack dumper (daemon thread +
`faulthandler.dump_traceback(all_threads)`) labels the *dumper* thread as
"Current thread", hiding the main thread — parse `Thread 0x…` blocks instead.
v2 (SIGUSR1 handler, no dumper thread) fixed this but needs an age-gate before
signaling: a bare SIGUSR1 to the bash launcher pre-`exec` kills it (default
action), and spawned workers only arm their handler at interpreter startup.

## 2026-09-04 (boot U) — wedge #20: L3 re-run dies at the same stage (post-reset degradation suspected)

**Context:** S1 startup-time session, ~83 min after wedge #19. L3 probe re-run
(`LANG_ONLY=1` = `--language-model-only`, stock config otherwise), launched under
`systemd-run --user -p MemoryMax=infinity` via the fixed sampler (age-gated
SIGUSR1 + shell exclusion). Unit `s1lang.rerun.service`.

**Timeline:**
- 17:19:31 APIServer up; both workers log "All limits of multimodal modalities ... set to 0, running in text-only mode" (flag active, as intended)
- NO "Encoder cache will be initialized" line → vision-encoder profiling correctly skipped (hypothesis confirmed again at the log level)
- ~17:20:30 both workers throw `c10::AcceleratorError: CUDA error: unspecified launch failure` from `SetDevice` (HIPFunctions.cpp:334); EngineCore reports "WorkerProc initialization failed due to an exception in a background process"
- 17:20:31 kernel: `Fence fallback timer expired on ring comp_1.0.0` → BACO `GPU reset(3) succeeded! device wedged, but recovered through reset` on 0000:0e:00.0 (**GPU0**)

**Stack-dump evidence (decisive for the flag's innocence):**
- 233 dumps written across 5 pids; **zero** contain vision frames (`vit_flash_attn|embed_multimodal|qwen3_vl|vision_encoder|image_processor`) — the ViT forward was genuinely not running, consistent with `--language-model-only` doing its job.
- Last EngineCore sample: idle in `wait_for_ready`; APIServer in `wait_for_engine_startup`. The dying work was in the C-level worker-init/profile path (no Python frames), same shape as #19.

**Why this is NOT attributed to `--language-model-only`:**
1. Its code path only zeroes modality limits at config time — before any GPU work. A software fault there would surface as an assertion/shape/value error, not a fence timeout + BACO reset.
2. The stock baseline boot 90 min earlier (no flag) ran the identical init to healthy (39.2 t/s canary later confirmed the system fine at 15:1x).
3. Kernel history for this boot: GPU0 already BACO-reset at 15:57 (#19); GPU1 reset twice earlier (01:31, 15:12). Pattern = chronic TP=2 HW instability family; **post-reset degradation of GPU0** is the leading hypothesis (a card that just went through a BACO reset may be flaky until reboot).

**Recovery:** automatic via BACO. No zombie vLLM KFD handles post-run (only `gpuagent` monitor at 0 VRAM), so no reboot *required* for handle hygiene — but the prudent next step before any further TP=2 GPU work is a **canary probe**, and if stock also wedges, stop + reboot via `~/bin/hermes-reb.sh`.

**S1 consequence:** L3 end-to-end startup number remains unmeasured. The 213 s vision-profiling gap it targets stays PROVEN by the clean baseline dumps (Worker_TP0 pinned in `profile_run → embed_multimodal → vit_flash_attn_wrapper`, 15:15:26→15:18:51). L3 is expected to remove ~213 s of a 234 s init; exact post-fix number pending a healthy GPU window.

**Do NOT:** retry L3 (or any TP=2 boot) immediately — canary first, reboot if the canary itself wedges.

## 2026-09-04 (boot U, evening) — T-1 session: wedge #21 on in-process probe launch

T-1 work resumed ~22:15 UTC on boot U (still the same boot as the morning
SYV-3 burst; no reboot in between). Sequence:

1. 22:15:54 `mtp1canary@mtp` started (fresh, post-morning-burst clean state).
   Model load 39.2 s; canary **39.2 t/s** — healthy band, PASS. Clean KFD after
   (only gpuagent), VRAM 0% both cards.
2. 22:19 `t1probe` service started — in-process TP=2 `LLM()` (util 0.85,
   maxlen 131072, compilation mode NONE, MTP k=2) with the t1_phase plugin armed
   (hooks target lm_head + drafter). Wedged at weight-load shard ~2/5:
   userspace `c10::AcceleratorError: unspecified launch failure` on both ranks →
   kernel `amdgpu 0000:0b:00.0: qcm fence wait loop timeout expired` +
   `The cp might be in an unrecoverable state due to an unsuccessful queues
   preemption` + `Failed to evict process queues` + `GPU reset begin!`. Source 4.
3. After teardown: VRAM back to 0% baseline both cards, KFD clean (only gpuagent),
   no zombie vllm PID.

**Assessment:** same chronic in-process-LLM() weight-load-hang family as wedges
#9/#10/#11 (all boot T) and the morning #16/#17 burst. Non-deterministic; a clean
canary passed ~4 min before, so this is not a code fault (the plugin only attaches
hooks AFTER load_model returns — it cannot affect weight loading). The in-process
weight-load path remains the trigger family on this host.

**Decision per house recipe:** one retry authorized. If the retry wedges → stop GPU
work + reboot via `~/bin/hermes-reb.sh`, then re-run the probe SERVE-BASED (the
serve path booted clean 6× today and is the cleaner family). The probe itself is a
one-shot measurement; it does not need to be in-process.

## 2026-09-04 (boot U, evening) — T-1 session: wedge #22 on serve-based probe launch

**Context.** Per #21's decision, the probe was moved off the in-process `LLM()`
path onto a **serve-based** launcher (`run_t1probe_serve.sh`, `vllm serve` TP=2
util 0.85, mode-NONE eager, MTP k=2) + the `t1_phase` plugin armed via
`/local/tmp/t1/t1_arm.cfg`. The plugin's hook-attach bug (shared-lm_head double
hook) was fixed and synced to site-packages before launch.

**Timeline.**
1. 22:46 canary `mtp1canary@mtp` **39.2 t/s PASS** (healthy band), clean SIGTERM
   shutdown, VRAM released.
2. 22:49 `t1probe-serve.service` started under systemd (MemoryMax=inf).
3. ~59s in, worker init wedged at `c10::cuda::SetDevice`: userspace
   `c10::AcceleratorError: CUDA error: unspecified launch failure` on BOTH ranks →
   kernel `amdgpu 0000:0b:00.0: qcm fence wait loop timeout expired` + `The cp
   might be in an unrecoverable state due to an unsuccessful queues preemption` +
   `Failed to evict process queues` + `GPU reset begin!` (Source 4) → **BACO reset**
   → `GPU reset succeeded, trying to resume`. EngineCore raised
   `WorkerProc initialization failed`; API server exited code 1.

**Post-wedge state.** VRAM back to 0% baseline both cards; KFD clean (only
`gpuagent`); no zombie vllm PID. **Kernel SELF-recovered** — unlike the
reboot-only zombie-KFD states, this one needs no reboot.

**Assessment.** This is the **serve-based family** (worker-init `SetDevice`),
distinct from #21's in-process-LLM() weight-load family. Non-deterministic; a clean
canary passed ~4 min before. The plugin cannot be the cause — it attaches hooks only
AFTER `load_model` returns, and this wedged during worker init (before model load).
Consistent with the chronic non-deterministic tp=2 weight-load/SetDevice wedge family
that Kevin has accepted as HW-related on this dev system.

**Decision per house recipe:** post-reset canary gate, then **ONE** serve-based probe
attempt (first of its family on boot U). If it wedges again → full stop + report to
Kevin (do not keep relaunching a repeatedly-wedging GPU without direction).

## 2026-09-05 (boot U) — T-1 A/B arm1 launches: wedges #23/#24, pre-transform weight-load failures

**Context.** T-1 Task 3 serving A/B on branch `gfx906/t1-int8-fp16-mass`. Design:
same branch for both arms; the single variable is the `T1_INT8_MASS` env flag
(arm0 = OFF via `mtp1srv@mtp`, arm1 = ON via the new `mtp1srv-t1on@` unit with an
`Environment=T1_INT8_MASS=1` line). Unit check re-run first: 31/31 ALL PASS.

**Timeline.**
1. ~07:44 arm0 (`mtp1srv@mtp`, flag OFF) booted clean; log verified to contain NO
   "int8 mass armed" line (transform correctly skipped).
2. 07:46–08:49 arm0 sweep complete: s9 @64k/96k/120k ×3 reps, medians
   **38.83 / 30.43 / 26.14 t/s**, acceptance 2.0 (per-pos 1.0/1.0) on all 9 reps —
   matches the k2fix reference band (37.95 / 29.88 / 25.70). Clean SIGTERM teardown,
   GPUs to 0%.
3. 08:51 arm1 (`mtp1srv-t1on@mtp`, flag ON) started; ~70s into weight load both ranks
   died with `c10::AcceleratorError: CUDA error: unspecified launch failure`
   (hipErrorLaunchFailure); EngineCore init failed, unit exited 1.

**Why this is NOT a T-1 fault.** The "T-1 int8 mass armed" log line never appeared —
the transform runs post-load in `load_model` and was never reached; the crash is in
weight loading itself. Grep-verified: `T1_INT8_MASS` is read only at load_model time
(`t1_int8_mass_enabled()`), the module has no import-time side effects, and arm0 ran
the identical code path with the flag false ~75 min earlier on this same boot. The
only code delta vs a main-branch boot is gated behind that false check.

**Post-wedge state.** KFD clean (only `gpuagent`), VRAM 10.8 MB baseline both cards,
no zombie PID — driver self-recovered. Boot U uptime ~13.4 h at the event; arm0's
full cycle (boot + 9 long-ctx reps) completed healthy in between.

**Assessment.** 23rd non-deterministic wedge; serve-based TP=2 weight-load family
(same as #22, and the chronic load-hang family generally). Per Kevin's standing
position: accepted HW-related on this dev system — log + canary + retry, no
wedge-chasing.

**Decision per house recipe:** post-wedge canary gate (`mtp1canary@mtp`, PASS bar
per boot-U band 38.2–39.3), then **ONE** arm1 retry; if it wedges again → stop +
report (no blind third launch).

### Outcome of the authorized retry (09:00)

Canary `mtp1canary@mtp` at 08:58 = **39.3 t/s PASS** (top of boot-U band, well above
the <25 degraded line). Arm1 retry launched ~08:59; it progressed FURTHER than attempt
1 — main-model weights loaded clean ("Loading weights took 31.69 seconds", both ranks
past the attempt-1 SetDevice crash point) — then `c10::AcceleratorError: CUDA error:
unspecified launch failure` on both ranks during post-load init (drafter load /
graph-capture window); EngineCore init failed, unit exited 1 at 09:00:22. The "T-1
int8 mass armed" line was **again absent** → the transform never ran; T-1 still not
implicated (crash is in weight-load/init, before `load_model` returns).

**2nd consecutive arm1 launch failure (wedges #23/#24) = BURST per house recipe → ALL GPU WORK STOPPED.**
GPUs clean after (0%/0%, KFD only `gpuagent`, no zombie PID) — non-deterministic HW,
not host degradation (canary passed between the two attempts) and not code.

**T-1 A/B state at stop:** arm0 COMPLETE (38.83 / 30.43 / 26.14 t/s @64k/96k/120k
medians, acc 2.0 all 9 reps). **arm1 BLOCKED — needs a fresh boot** to clear the
wedge burst before the `T1_INT8_MASS=1` server can come up. Reboot is Kevin's call
(`~/bin/hermes-reb.sh`); on the next clean boot, re-run arm1 (canary first), then the
comparator. The arm0 data + arm1 unit + comparator are all persisted and reusable as-is.

## 2026-09-05 ~15:4xZ (boot U) — T-1 x k=4 boot attempt: wedge #25 at worker init

**Context.** After the k=4 depth A/B landed as a decisive win (mtp4 arm, T-1 OFF:
+15.4/+17.6/+18.4% vs k=2 baseline, perfect 4/4 acceptance), the next experiment was
the T-1 x k=4 interaction: same config with `T1_INT8_MASS=1` armed (unit
`mtp1srv-t1on@mtp4`, port 8128).

**Attempt 1 — operator error, not HW.** mtp4 teardown race: my stop-wait loop matched
the wrong process pattern and declared "down" while the old server still held :8128;
the new instance died at bind. No GPU damage.

**Attempt 2 — wedge #25.** On verified-clean GPUs (0% VRAM, KFD only gpuagent):
`c10::AcceleratorError: CUDA error: unspecified launch failure` (hipErrorLaunchFailure)
at `c10::cuda::SetDevice`, both ranks, ~60 s into worker init. EngineCore init failed.
The T-1 transform (post-load hook) was never reached — "int8 mass armed" line absent —
so no T-1 code executed; only config deltas vs the clean mtp4 boot are spec depth +
capture sizes, both of which that arm already served 9/9 reps with ~35 min earlier.

**Classification.** Serve-based worker-init family (like #23/#24). Non-deterministic:
mtp4 served clean all morning on this boot; GPUs read healthy after (temps 32-40 C,
no zombie handles, driver self-recovered). Per house pattern this is the accepted
TP=2 HW instability — no RMA escalation. Reboot required to clear GPU0 state before
retrying.

**State at stop.** k=4 win data complete + committed (`fc548cba19`). T-1 x k=4 test
BLOCKED on reboot; everything needed for the retry is persisted (unit, launcher mtp4
config, sweep client). Post-reboot: canary gate, then ONE T-1@k=4 boot attempt, then
sweep vs the mtp4 baseline.

## Wedges #30/#31 (boot X, 2026-09-07 08:35–08:37 UTC) — mtp4ag burst

**Context.** Boot X (up since ~05:49, post-reboot from the boot-W burst).
Sequence: canary 38.8 t/s PASS (05:58) → mtp4@s9 TP=2 9/9 reps clean
(06:01–07:14) → mtp5@s9 TP=2 9/9 reps clean (07:14–08:32, sweep + text
probe) → background swap: old driver killed 08:32:58 on verified-clean
GPUs (VRAM 0/0) → lean queue launched (pid 38641) → canary 38.9 t/s PASS
(08:32:58–08:35:04) → mtp4ag launch.

**Attempt 1 — wedge #30 (08:35:53, GPU0 only).** mtp4ag = the mtp4 server
config with an agent-corpus *client* (the corpus name never reaches the
server; server config byte-identical to the arm that ran clean 1.7 h
earlier). Died ~49 s into worker init: userspace
`c10::AcceleratorError: CUDA error: unspecified launch failure`
(hipErrorLaunchFailure), EngineCore init failed. Kernel (0000:0b:00.0 =
GPU0, minor 0 this boot): `qcm fence wait loop timeout expired` +
"The cp might be in an unrecoverable state due to an unsuccessful queues
preemption" + `Failed to evict process queues` + `Failed to quiesce KFD` +
BACO reset, "GPU reset succeeded, trying to resume", "VRAM is lost due to
GPU reset!". Strongest kernel wording of the family so far (cf. boot-S #5),
but the driver self-recovered (clean VRAM/KFD at the 08:36:00 teardown
check).

**Attempt 2 — wedge #31 (08:37:23–08:37:32, BOTH GPUs).** Launched
08:36:41 on verified-clean GPUs (one retry per house recipe). Progressed
to weight load (shard 0/5 at 08:37:12), then hipErrorLaunchFailure both
ranks; EngineCore init failed at 08:37:37. Kernel: identical
fence/cp-unrecoverable/queue-evict sequence on GPU0 (0b:00.0) at
08:37:23 and GPU1 (0e:00.0) at 08:37:32; both BACO reset, both self-
recovered. GPUs clean at 09:52 (VRAM 0/0, KFD only gpuagent, 31–32 °C).

**Classification.** Same chronic serve-based TP=2 weight-load/SetDevice
family as #12–#17/#21–#29. This boot's signature: two full clean TP=2
arms + two passing canaries first, then two consecutive launch failures —
the 5th occurrence of the "post-long-serve, later launch" pattern on this
host (cf. #23, #25, #26, #28). The mtp4ag config is indistinguishable
from the clean mtp4@s9 boot (client-side corpus only), so there is no code
or config delta to suspect. Non-deterministic HW per the accepted
classification.

**State at stop.** mtp4@s9 and mtp5@s9 numbers complete and valid
(deterministic filler corpus; boot-independent): mtp5 medians
50.47/40.80/35.72 t/s @64/96/120k vs mtp4 45.43/36.63/31.77 =
+11.1/+11.4/+12.4%; acceptance 1.0 at all positions (0–4), all 9 reps
each; acc_mean 5.0; text probe token-identical to mtp4 (1568 chars,
trigram_rep 0.965). mtp7@s9 gate (pos-4 acc ≥ 0.8) passed with wide
margin — but the s9 mtp7 arm was deliberately skipped (filler ceiling
adds nothing; agent corpus is the decision data). **Agent-corpus queue
(mtp4ag → mtp5ag → cat1k4ag → mtp7ag-gated → final canary) BLOCKED —
fresh boot required** (reboot via ~/bin/hermes-reb.sh, root; kread has no
sudo). Post-reboot: `run_arms3.sh` as-is (canary-gated; one retry per
arm; abort-on-2-failures).

## Wedge #32 + driver double-launch bug (boot Y, 2026-09-07 11:33–11:41 UTC)

**Context.** Boot Y (up since ~11:31, Kevin's reboot after the boot-X abort).
Canary (mtp2 TP=1, GPU0) PASS 38.4 t/s 11:33–11:36. mtp4ag (mixed corpus —
client-side only; server config identical to boots V/W/X) launched 11:36:07.

**Attempt 1 — wedge #32 (11:37:13, GPU1 only).** Single-instance launch on
clean GPUs (VRAM 0/0, KFD gpuagent only). Died ~66 s into weight load:
fence timeout + cp-unrecoverable + queue-evict failure + KFD quiesce
failure + BACO reset on 0000:0e:00.0 (GPU1, minor 1 this boot), self-
recovered 11:37:15. Chronic serve-based weight-load family. Notable: this
was the FIRST TP=2 launch of the boot (no prior long serve), so the
"post-long-serve, later launch" pattern (cf. #23/#25/#26/#28) does not
hold for #32 — the family is broader than that pattern.

**"Attempt 2" — the double-launch bug (11:38:10–11:40:29, NOT a wedge).**
The run_arm retry loop in run_arms2.sh/run_arms3.sh launched the server at
BOTH the top and the bottom of the for-loop body. After attempt 1 failed:
teardown → sleep 30 → launch at loop bottom (instance A, APIServer 4084) →
loop iteration 2 → launch at loop top (instance B, APIServer 4085) — two
`vllm serve` processes ~simultaneously. Both loaded the full TP=2 model
concurrently on the same GPUs; both engines' KV-cache checks then failed
(4.74 GiB needed for one 131072 request vs 4.28 / 3.54 GiB available; a
clean single instance measures **15.29 GiB** available — the boot-X mtp4
log). Journal clean in the window (no fence/reset events) — pure memory
contention from the duplicate instance, no hardware event. Signature for
future logs: two "ROCm switched to: /opt/rocm" lines at the server-log
head, consecutive APIServer PIDs, two EngineCores, two "Initializing a V1
LLM engine" lines.

**Consequence + correction.** Boot X's "wedge #31" has the same signature
in its (since-overwritten) attempt-2 log — 2 APIServers + 2 EngineCores —
and is reclassified as a double-launch artifact; the fence events on both
GPUs there are consistent with two concurrent weight loads. Genuine wedges
stand at #30 (boot X, GPU0) and #32 (boot Y, GPU1).

**Fix (run_arms3.sh, 2026-09-07):** one launch per attempt (bottom-of-loop
launch removed), pre-launch alive guard (teardown if anything is already
running), and the "ROCm switched" line count is logged at READY (must be
1). **Procedural lesson: the burst rule counts genuine single-instance
launch failures; a diagnosed software failure is fixed and retried on the
same boot — a reboot is only for genuine wedge pairs.**

**State at write (12:2x UTC).** GPUs clean: VRAM 0/0, KFD gpuagent only;
both cards probe 9.9/9.8 TFLOPS fp16 (no residual damage from #32).
Reranker: NO llama-server process exists on this boot (Kevin authorized
stopping it; nothing to stop — it is not a factor in any of these
failures). Mixed-corpus queue re-launching on the fixed driver (canary
gate first). If the re-run produces two genuine single-instance wedge
launches on boot Y, stop + reboot per rule.

## Boot Y burst — wedges #33/#34, ABORT + reboot (2026-09-07 14:07–14:11 UTC)

**Context.** Boot Y re-run (12:50, fixed driver, patched client): canary
38.8 PASS → mtp4ag (mixed) completed 9/9 clean (29.37/23.89/21.87 t/s
@64/96/120k; acc_mean 2.0–2.98; see DEVLOG-fa-attention.md) → mtp5ag
launch fired from the still-running queue (Kevin had just decided to skip
it; the safe-kill watcher was armed but mtp4ag's teardown finished ~1 min
before the window check would have fired… in practice the driver reached
the mtp5ag launch first — the 30 s window was consumed by the text probe
+ teardown timing this time).

**#33 (14:08:45, GPU0 0b:00.0) + #34 (14:10:26, GPU1 0e:00.0).** Two
consecutive single-instance mtp5 launches, each dying ~70–90 s into weight
load with the full chronic signature (fence timeout → cp-unrecoverable →
queue-evict/KFD-quiesce failure → BACO → self-recover → fence-fallback
timer). Different GPUs, ~100 s apart. No software failure candidate this
time: config identical to boot X's clean mtp5@s9 run, single instance
verified, VRAM 0/0 pre-state on both attempts, journal clean between them.

**Burst per rule → ABORT 14:10:33.** The run_arms3 driver's mechanical
2-failure abort fired; the new watch_tail_and_plain watcher detected the
driver's self-exit and HELD (exit 2, no tail/plain launch) for a human
decision — the exact path it was designed for. Boot Y's genuine-wedge
total: #32 (11:37, GPU1, first TP=2 launch of the boot) + #33 + #34 =
THREE. Reboot per house rule (Kevin executes via ~/bin/hermes-reb.sh).

**Pattern note (open question list).** Three wedges in one boot, all in
serve-based TP=2 weight loads, spanning the boot's first and ~2.5 h later;
the ~2.5 h between #32 and #33 included a full 9/9 clean sweep on the same
arm family. Cumulative-reset → degradation model (see top of file) remains
the working hypothesis; per-boot wedge budget empirically ≈ 2–3.

**Post-mortem findings (all fixed/recorded, 14:3x UTC):**
1. **Abort-path log loss**: run_arm's `rm -f "$slog"` ran before the
   ABORT check, deleting the attempt-2 server log (only the journal
   survives). run_tail.sh/run_plain.sh patched: abort check before rm.
   (run_arms3 has served its life; boots Z+ use run_tail.sh.)
2. **Client flat/nested corpus fix** verified in production: mtp4ag swept
   9/9 (boot Y pass 1 swept 0/27 — see dev log).
3. **Watch-window timing**: the W1 kill window (post-teardown, pre-next-
   launch, 30 s) can be missed when the preceding arm's text probe +
   teardown run long; the watcher's driver-exit fallback (hold for human)
   covered it safely. For boot Z the skip is already out of the queue —
   no intercept needed.

**State at write.** GPUs VRAM 0/0, no GPU processes, no stragglers
(checked 14:17). All boot-Z queue artifacts persistent in /local/tmp/a3
(run_tail.sh, run_plain.sh, sweep logs, driver logs). Mixed-corpus v2
(chat replay) queued after the tail + plain baseline on boot Z.

## Boot Z — wedge #35 on the chat-replay launch (2026-09-07 17:20:59 UTC)

**Context.** Boot Z (up ~14:47) ran the reduced depth matrix cleanly:
canary 38.8 PASS → mtp2ag 6/6 (30.95/20.71 t/s @64k/120k) → mtp3ag 3/3
(24.80 t/s @120k — the depth winner) → final canary 38.8 PASS → plain
6/6 (19.76/13.11). Five clean TP=2-equivalent launches. The 6th launch —
the chat-replay server (plain TP=2, port 8132, maxlen 65536; launched
17:19:47 on a verified-clean VRAM 0/0) — wedged.

**#35 (17:20:59, GPU1 0e:00.0).** Single instance (one "ROCm switched"
line, no other GPU processes). Rank 0 completed its weight load clean
(29.45 s, 9.15 GiB); the failure surfaced in the other rank's
`SetDevice` copy as `c10::AcceleratorError: CUDA error: unspecified
launch failure` (`hipErrorLaunchFailure`). Kernel: `qcm fence wait loop
timeout expired` → "cp might be in an unrecoverable state due to an
unsuccessful queues preemption" → `Failed to evict process queues` →
`Failed to quiesce KFD` → PSP `UNLOAD_TA(0x2) failed (0x117)` → BACO
reset → "GPU reset succeeded, trying to resume" 17:21:01, devcoredump
written. Full chronic family signature, GPU1.

**Disposition.** Boot Z's first genuine launch failure → house rule
allows one retry for the arm. Pre-retry verification: no stragglers,
VRAM 0/0, GPU1 matmul probe 9.7 TFLOPS fp16 OK (the canary only covers
GPU0 — the post-GPU1-wedge probe was required and passed). A 2nd
consecutive genuine failure = BURST → reboot per rule (Kevin executes).
Note: 6 weight loads in this boot — the family's per-boot accumulation
pattern is tracking boots W–Y (2–3 wedges/5–6 loads).

## Boot Z burst — wedge #36 + the watcher exec/pgrep false positive (2026-09-07 17:37–17:39 UTC)

**#36 (17:38:41, GPU1 0e:00.0).** The replay retry (launched 17:37:37 on a
verified-clean post-#35 state) died in the same phase, same GPU, same
signature: rank-0 load clean (5/5 shards in 24 s), other rank's SetDevice
copy → `hipErrorLaunchFailure` 17:38:43; kernel fence timeout →
cp-unrecoverable → KFD-quiesce failure → BACO → "GPU reset(2) succeeded"
+ "device wedged, but recovered through reset" 17:38:44. Two consecutive
genuine single-instance failures (#35 17:20:59, #36 17:38:41) = **BURST**
→ all GPU work stopped, reboot per house rule. Boot Z's launch record:
5 clean TP=2-equivalent launches (canary, mtp2, mtp3, canary, plain) +
2 wedged replay launches — the family is now biting on the 6th–7th
weight load of the boot, consistent with the 2–3 wedges/5–6 loads pattern
of boots W–Y.

**Watcher bug (why #35's driver line was a false positive).**
`run_replay_server.sh` line 22 `exec .venv/bin/vllm serve …` replaces the
bash process image — after exec, NO process carries the
`run_replay_server.sh` cmdline, so the watcher's death check
(`pgrep -f "run_replay_server[.]sh"`) is guaranteed to false-positive on
its first poll. On attempt 1 the watcher logged "replay server died
during startup" at 17:19:47 — the SAME SECOND it launched the server —
and exited, orphaning the setsid'd server (which kept loading and wedged
genuinely at 17:20:59, #35). Consequences: (a) #35's driver log line
mislabels the timing (the wedge is 72 s later, in the journal); (b) the
retry script had the same bug and false-positived the same way at
17:37:37, while the real death came at 17:38:41. Fix: death checks must
match the exec'd process (`pgrep -f "[v]llm serve .*max-model-len
65536"`); `run_replay_retry.sh` patched, `run_replay_attach.sh`
(attach-to-already-running, launch-free) written as the safe pattern for
orphaned servers.

**State at stop.** No vllm/replay procs, VRAM 0/0, attach driver never
launched. Boot-Z results all safe on disk: tail_driver.log,
sweep_mtp2ag.client.log (6/6), sweep_mtp3ag.client.log (3/3),
sweep_plainag.client.log (6/6), canary logs. Replay output
(`chat_replay_qwen38.jsonl`) not started. Next boot (W'): the replay
sequence (fixed driver) + the v2-mixed-corpus build + the SYV-12
generation-time gate + the mtp3ag@64k cell + the chat-frac arms; the
SYV-10 GDN-bounds port (fb0fc766e4) compile-checks on the first MTP
launch there.

## Boot W' burst — wedges #37/#38 on the mtpv2 launch (2026-09-07 20:27–20:31 UTC)

**Context.** Boot W' (up since ~17:46, Kevin's reboot after the boot-Z
burst) ran the post-replay queue: chat replay 18:13 (62/62 turns, 205,173
completion tokens, zero failures), v2 mixed-corpus build (20 % chat),
SYV-12 generation-time gate, canary 38.7 t/s, mtp3 launch 19:35 (clean,
KV 15.43 GiB single instance) — 3 clean TP=2-equivalent weight loads
before the burst.

**Harness bug (mtp3v2 first sweep run, 19:37:49).** `run_w1.sh`'s arm
function set `MTP1_PTS` but forgot `MTP1_PORT`, so the sweep client used
its default port 8123 (nothing listening there) and died on
connection-refused with rc=1; the original driver held for human by
design. Software failure, NOT a wedge — the mtp3 server was (and stayed)
healthy on 8134. A continuation driver ATTACHED to the running server
(launch-free, the boot-Z lesson) and re-ran the sweep with the correct
port: 6/6 reps, medians **27.44 @64k / 24.76 @120k** (acc_median
1.38/2.15). The 120k rep0 (20.76) is a warm-up outlier that includes the
first-ever compile of the SYV-10 ported GDN spec kernels (commit
fb0fc766e4) — reps 1–2 (24.76/25.07) are steady-state. **The port ran
thousands of MTP draft steps across the sweep with zero server errors:
compile-check + runtime check PASS.** (The mtp3v2 text probe was lost to
a `Permission denied` on text_probe.py — a harness chmod bug; re-run
post-reboot if wanted.)

**#37 (20:28:55, GPU1 0e:00.0) / #38 (20:30:26, GPU0 0b:00.0).** The
mtpv2 arm (k=2, the remaining chat-frac arm) wedged twice in the
chronic weight-load family: attempt 1 (launched 20:27:56 on verified-clean
GPUs) and attempt 2 (launched 20:29:25 post-#37, matmul probes
10.0/9.8 TFLOPS OK) both died at safetensors shard 0/5 —
`hipErrorLaunchFailure` at the SetDevice copy, single instance each (1
"ROCm switched" line). Kernel signature per #30–#36: fence timeout →
cp-unrecoverable → queue-evict/KFD-quiesce failure → GPU reset → BACO,
self-recovered both times. Two consecutive GENUINE single-instance
launch failures = **BURST** → continuation driver stopped all GPU work at
20:30:50 by design; **reboot required (Kevin executes)**.

**Wear pattern (update).** Genuine wedges per boot: X=1, Y=3, Z=2,
**W'=2**; bursts always at the 2nd consecutive genuine failure. Load
position of the first wedge varies (boot Y: first TP=2 launch of the
boot; boot Z/W': after 5/3 clean loads) — no reliable load-count
predictor; the 2-failures-stop rule remains the right instrument.

**State at stop.** No vllm procs, VRAM 0/0, both cards BACO-recovered.
Boot-W' data safe on disk: `chat_replay_qwen38.jsonl` (205k tokens),
v2 `corpus.json`, `syv12gen_w1_fixed.log` (gate GO, 25.3 % hit_frac),
`sweep_mtp3v2_w1.client.log` (6/6), canary logs, driver logs with the
MTP1_PORT correction line. **Remaining queue (post-reboot, ~2 weight
loads — well inside the per-boot budget):** mtpv2 6 reps (k=2 @ v2,
the same-corpus comparison arm for the k=2-vs-k=3 call) + final canary.

---

**Boot W'' (post-#37/#38 reboot, 2026-09-07 ~20:47).** Clean boot:
preflight 9.9/9.8 TFLOPS, canary 38.8 t/s, mtpv2 launched clean
attempt 1 (KV 15.57 GiB), 6/6 sweep rc=0, clean SIGTERM teardown
21:35 (VRAM 0/0). **Event #39 (22:23, GPU1 0e:00.0): SYV-12 probe
arm-1 launch wedged with the chronic SetDevice signature —
harness-induced.** Sequence: arm 0 (SYV-12 OFF, in-process TP=2 `LLM()`)
finished its 512-token run (result line + token ids saved) but its
multiproc EXECUTOR SHUTDOWN HUNG ("[shutdown] Executor: workers still
running after grace period; sending SIGTERM count=2" at 22:22:51); the
probe driver's inter-arm wait was only `sleep 10` + a VRAM% check
(0/0 — the workers had already released VRAM) and launched arm 1 at
22:23:02, 11 s after the SIGTERM. Arm 1 died ~10 s in at SetDevice
(`c10::AcceleratorError: unspecified launch failure`, both ranks);
kernel 22:24:15 on 0e:00.0: fence timeout → cp-unrecoverable →
queue-evict fail → `Failed to quiesce KFD` → UNLOAD_TA(0x2) 0x117 →
BACO reset, self-recovered. This is the documented TP=2 teardown hazard
(AGENTS.md: "SIGKILL leaves the driver mid-P2P-op and the next init
wedges GPU1 (hipErrorLaunchFailure; needs BACO reset + retry)") fired
through a hung-shutdown SIGTERM instead of a manual SIGKILL — i.e. a
mid-op kill of arm 0's workers, self-inflicted by the driver's
insufficient inter-arm drain. **Classification (per the boot-Y refined
burst rule): diagnosed software trigger → NOT counted toward the
genuine-failure burst pair; fix + retry on the same boot.** Driver fix:
inter-arm wait now (a) polls until ALL prior-arm python/worker procs
are gone (not just VRAM) and (b) re-runs both-GPU matmul probes before
the next launch. Post-#39 state verified 22:32: VRAM 0/0, matmul
9.9/9.8 TFLOPS both cards. Arm-0 reference result safe:
`/local/tmp/syv12/tokens_arm0.json` (512 ids, 478 s wall incl. 120k
prefill — the probe's decode-only split needs the RequestMetrics
first/last-token times; see the SYV-12 devlog).

**Event #40 (22:36:28, GPU1 0e:00.0): the probe RE-RUN arm 0 (OFF)
wedged SPONTANEOUSLY 56 s after launch, on a verified-clean pre-state**
(the fixed driver's pre-arm check passed at 22:35:28–32: no stragglers,
VRAM 0/0, matmul 10.0/9.7 both cards). Identical kernel signature to
#39 (fence → cp-unrecoverable → KFD-quiesce fail → UNLOAD_TA 0x117 →
BACO, self-recovered). Single instance. This is the 1st GENUINE
spontaneous launch failure of boot W'' (#39 was harness-induced and
doesn't count) — one retry is authorized per the burst rule, but
**decision: STOP GPU work pre-emptively (wear-based, not a formal
burst).** Rationale: two chronic-family BACO resets on GPU1 15 min
apart after 5 TP=2-equivalent loads; the remaining SYV-12 verification
is 4 more loads (probe off+on, A/B off+on) — running them on a
demonstrably wearing GPU1 buys little (the probe is a sanity check, and
its OFF reference is already captured from the 22:05 run) at real
wedge risk. The formal gate (serving A/B) only matters on a healthy
boot anyway (the degradation canary rule). Post-stop state verified
22:46: VRAM 0/0, matmul 10.0/9.8 TFLOPS, no stragglers. **Next boot:
Kevin reboots → staged verification = `run_probe.sh` (2 loads) then
`run_syv12_ab.sh` (2 loads); both drivers are final (drain + pre-launch
probes + MTP1_PORT explicit). If #40 recurs as the 2nd consecutive
genuine failure on the fresh boot, treat it as burst-grade per the
standing rule (reboot again + escalate the RMA question).**

## Boot Y2 — wedge #41 on the SYV-12 probe ON arm (2026-09-08 07:49:51 UTC)

**Event #41 (07:49:51, GPU1 0e:00.0): the SYV-12 probe arm 1 (ON) attempt
1 wedged SPONTANEOUSLY during weight load, on a verified-clean pre-state.**
Boot Y2 (up since ~06:31, 2nd clean reboot of the SYV-12 verification
campaign). Sequence: probe driver launched 07:34:48 with the #39-hardened
drain; arm 0 (OFF) ran a FULL clean pass — 120k prefill + 512-token decode
(25.335 t/s decode-only, 20.21 s) + graceful multiproc shutdown 07:48:39;
post-drain verified 07:48:53–57 (all procs gone, VRAM 0/0, both-GPU
matmul probes 9.9/9.7 TFLOPS OK); arm 1 (ON, `GFX906_SYV12=1`) launched
07:48:57 and both TP ranks died at safetensors shard 2–4/5 with
`hipErrorLaunchFailure` at the SetDevice copy. Kernel: fence timeout →
cp-unrecoverable → `Failed to evict process queues` → `Failed to quiesce
KFD` → UNLOAD_TA(0x2) 0x117 → BACO, "GPU reset succeeded" 07:49:53
(self-recovered). Single instance (no other processes on the GPUs).
Identical signature to the chronic serve-based weight-load family
(#34–#40); GPU1 0e:00.0 is again the victim (the recurring card).
Nothing SYV-12-specific runs during weight load (the extension's buffers
are trivial allocations made in the runner `__init__` before load; the
fill/append kernels fire only in the spec-verify step) — the OFF arm ran
the identical shape 30 s earlier and was fully clean, so this is
classified **GENUINE/spontaneous HW-family**, 1st genuine failure of boot
Y2.

**Action: one retry authorized per house rule** (`run_arm1_retry.sh`,
launched 07:52:07, same hardened pre-checks: proc-drain, VRAM 0/0,
both-GPU matmul probes). Arm 0's result + identity reference are safe
(`result_arm0.line`, `tokens_arm0.json`). **If the retry wedges
spontaneously: 2nd consecutive GENUINE launch failure on boot Y2 (with
#41) = BURST → stop all GPU work + reboot (Kevin), per the standing rule.**

**Boot Y2 continuation — #41 retry history + #42 (08:57:24, GPU1
0e:00.0): pre-emptive wear-based stop.** The one retry authorized after
#41 did not wedge — it hit two consecutive SOFTWARE bugs in the SYV-12
V1 port, each diagnosed and fixed on the same boot (diagnosed software
failures don't count toward the burst per the refined rule):

1. **Scheduler spec-stats sizing** (`SpecDecodingStats` sized by base k;
   the extended scheduled drafts are k+ext → `IndexError` in
   `observe_draft`; masked on the boot-Y probe because LLM() disabled
   log_stats). Fixed: sized by k+ext; asserts made length-based.
2. **GDN state-slot under-provisioning (the zero-output bug).** The ON
   arm ran end-to-end (both SYV-12 kernels fired per jit_monitor) but
   emitted an all-zero output stream from decode token 2, with near-
   saturated acceptance (3.94 — a self-consistent zero fixed point).
   A 16k debug probe (env-gated runner dump of the spec-decode input
   assembly; removed after diagnosis) showed the input assembly was
   correct ([bonus, d1, d2, fill(-1)]; the -1 fill slot is rejected by
   the standard one-hot path and the bonus is taken from the right
   position) — but the MTP draft was off by one loop position from
   step 1, i.e. the GDN state was corrupt. Root cause: the V1 port
   widened `MambaSpec.num_speculative_blocks` to k+ext (state pool +
   block table = 4 columns) but two GDN consumers still sized per-
   draft state by the BASE k: (a) the GDN attention backend's
   `num_spec` (`vllm/v1/attention/backends/gdn_attn.py`) —
   `spec_state_indices_tensor` [bs, k+1] (one column short), the
   spec token-index capacity, and — decisively —
   `max_query_len=spec_state_indices_tensor.size(-1)` in
   `causal_conv1d_update`, which capped the GDN conv kernel at 3 of the
   4 verify tokens; (b) the GDN layer's conv-state shape
   (`mamba/gdn/base.py`) — `conv_kernel-1 + k` snapshot columns, one
   short. Fixed both by adding `syv12_ext` (lockstep with
   MambaSpec). Post-fix 16k debug probe: correct 8-token s9 loop from
   decode token 1, 128/128 tokens. The full 120k ON re-run then wedged
   at weight load = #42 (same chronic family, GPU1, self-recovered).

**Stop decision (08:59):** two spontaneous chronic-family BACO resets
on boot Y2 (07:49, 08:57 — 68 min apart, 4 fully clean loads between)
meets the #40 pre-emptive criterion even though the formal burst pair
(2 *consecutive* genuine launch failures) was never formed. Remaining
verification = 3 more loads; at this boot's ~25 % per-load wedge rate
that's a coin-flip. Fresh boot + 3 loads is the lower-risk path. State
for next boot: OFF reference + identity reference persist
(`/local/tmp/syv12/tokens_arm0.json`, `result_arm0.line`); only the 120k
ON arm (`run_arm1_retry.sh`) + the guarded A/B (`run_syv12_ab.sh`,
unblock by touching `/local/tmp/syv12/V1_PORT_DONE` after the probe
PASSes) are owed.

**Boot Y3 — #43 (13:14:37, GPU1 0e:00.0).** First genuine launch failure on
boot Y3 (after 4 clean loads, including the SYV-12 probe + full A/B sweep
pair). A6 profiling server, EAGER k=3 MTP TP=2. Same chronic weight-load
signature (fence timeout → cp-unrecoverable → BACO, self-recovered in
~2 s). Retry 13:21:57 after the hardened pre-checks. Not a burst pair
yet — this boot's tally is 1. (The 13:11 agdn crash before this was the
harness wrapper arg-shift bug, diagnosed + fixed on the same boot.)

**Boot Y3 — #44 (21:16:40, GPU1 0e:00.0).** Second spontaneous
chronic-family BACO reset on boot Y3 (first: #43 at 13:14:37, 8 h
earlier; clean loads in between, including the A6 retry run and a
standalone GPU unit test of the new SYV-12 fill kernel at ~21:12).
Victim: the instrumented SYV-12 s9 probe (resurrection gate), attempt
1 — same chronic weight-load signature (qcm fence timeout →
cp-unrecoverable → queue-evict failure → KFD quiesce failure →
UNLOAD_TA 0x117 → BACO, self-recovered in ~3 s), single instance,
verified-clean pre-state (both-GPU matmul probes passed at 21:15).
Per-boot wedge rate now ~2/8 weight loads (25 %). The #40 pre-emptive
criterion (two spontaneous chronic-family resets on one boot) is met in
form; the retry was executed because (a) the house rule authorizes one
retry per arm, (b) the user explicitly directed this probe on this
boot, and (c) the retry's failure mode (burst → reboot) is identical to
the alternative (pre-emptive stop → reboot) while its success mode
delivers the resurrection-gate data. If the retry wedges spontaneously
→ burst pair with #44 → all GPU work stops and the SYV-12 fix + probe
carry to a fresh boot (all state is under /local, nothing lost).

**Boot Y3 — #45 (21:22:12, GPU1 0e:00.0) + BURST + STOP.** The
authorized retry of the instrumented SYV-12 probe (launched 21:21:22 on
verified-clean GPUs: 10.0/9.8 TFLOPS both cards at 21:21:17–22, no
procs, VRAM 0/0) wedged 50 s in at the same phase as #44 (safetensors
shard ~1/5, SetDevice hipErrorLaunchFailure both ranks) with the same
chronic kernel signature (fence timeout → cp-unrecoverable → queue-evict
→ KFD-quiesce-failure → BACO, self-recovered ~3 s). Single instance,
spontaneous. **2nd consecutive genuine single-instance launch failure
(with #44) = BURST per the house rule.** All GPU work stopped 21:24;
reboot requested (Kevin executes).

Session state at stop (all committed / staged under /local, nothing
lost): (1) the SYV-12 fill-contract root cause was identified and fixed
in-tree before the first wedge — the v1 fill kernel computed the
lookup suffix from the trailing history only, which ends at the anchor
and therefore continues into the d0 slot, off by k=2 positions from the
FILL slot it is stored in (derivation from the s9 loop structure; fully
consistent with both observed regimes: 0.000 on the period-9 s9 loop,
~0.6 % incidental hits on code). Fix: suffix = last (MIN_MATCH−k)
history tokens + the k base drafts (which end at d_{k-1}, the token
before the FILL position). Unit-verified 16/16 under the corrected
contract; the offline analyzer + synthetic healthy-pipeline simulation
validate end-to-end (synthetic s9-like stream: pos3 0.975, mean 3.95).
(2) SYV-13 closed as N/A (verify-only, no code change). (3) The B=4
harness extension landed (`_bench_serve_grid_gfx906.py`: n-ary cells,
corpus mode, /metrics acceptance deltas, stop/repetition screens;
offline unit checks pass). The next boot's SYV-12 probe is therefore a
VALIDATION run (expect s9 pos3 ≈ 1.000, mean acceptance ≈ 4.0), not a
localization run — gate step 2 of the resurrection ladder is what it
tests. Boot Y3 final: 3 spontaneous chronic-family wedges
(#43/#44/#45) across ~9 weight loads.
## Boot Y4 — wedge #46 on the SYV-12 validation probe attempt 1 (2026-09-09 06:45:03 UTC)

Boot Y4 (2026-09-08 21:31, after the Y3 burst reboot). Preflight
clean: VRAM 0/0, no procs, both-GPU matmul 10.0/9.8 TFLOPS, kernel
log clean since boot; BDF map 0b:00.0=GPU0, 0e:00.0=GPU1 (same as
Y2/Y3). Canary mtp2 TP=1 @38.7 t/s (mid healthy band 38.4–38.9).

**~9 h gap** between canary (21:36) and the first probe attempt
(2026-09-09 06:43:34) — an ssh outage; the machine idled cleanly
through it.

Wedge #46: instrumented SYV-12 s9 validation probe attempt 1
(in-process TP=2 `LLM()`, k=2, util 0.93, EAGER + per-step debug; the
fill-contract fix from 7b2fba0754 in tree) launched 06:43:34 on
verified-clean GPUs (prelaunch matmul 10.0/9.8). Both ranks died 89 s
in at the safetensors shard ~1/5 phase (SetDevice copy) with
`hipErrorLaunchFailure`; kernel 06:45:03 on 0000:0e:00.0 (**GPU1**,
drm-card minor 1): "qcm fence wait loop timeout expired" + "cp might
be in an unrecoverable state due to an unsuccessful queues preemption"
+ "Failed to evict process queues" + "Failed to quiesce KFD" + "GPU
reset begin" + "BACO reset" → "GPU reset(1) succeeded" + "device
wedged, but recovered through reset" 06:45:06. Self-recovered in ~3
s. Single instance, spontaneous — the chronic TP=2 weight-load
family, GPU1 again (the recurring victim). This is the 3rd weight
load of the in-process-TP=2 probe config to wedge (#44, #45 boot Y3);
every chronic-family wedge on this host is a TP=2 launch (canaries
are TP=1).

Post-drain verified clean (06:47): no procs, VRAM 0/0, both-GPU
matmul 10.0/9.8. ONE retry authorized per house rule.

**The retry exposed a software failure, not a wedge** (clean weight
load; Triton compile assert on the first extended step — see the dev
log SYV-12 boot-Y4 entry). Fixed in-tree (fb971c54a0) and re-run:
the validation probe then PASSED (attempt 2, 06:59:15 launch, 07:12
finish; 0 wedges; post-run VRAM 0/0).

Boot Y4 tally at the probe's completion: 1 spontaneous wedge (#46) +
1 diagnosed software failure (not counted) across 3 weight loads;
canary clean. Next: the SYV-12 v2 production A/B (2 more loads).

**Wedge #47** (07:29:13, GPU1 0e:00.0): the A/B's OFF arm (mtp2o4,
`mtp` k=2 TP=2 server, util 0.85) attempt 1, launched 07:28:12 on
verified-clean GPUs — died 71 s in at weight load; kernel 07:29:13:
fence timeout + BACO, self-recovered 07:29:16. Single instance,
spontaneous (log kept:
/local/tmp/a3/server_mtp2o4_syv12.log.attempt1). The driver's
authorized retry loaded clean (07:31:17, KV 15.56 GiB) and completed
the full 6-rep sweep + text probe + clean teardown.

**Wedge #48** (08:24:06, GPU1 0e:00.0): the A/B's ON arm (mtp2n4,
same server + GFX906_SYV12=1) attempt 1, launched 08:23:00 after the
OFF arm's clean teardown — died 66 s in at weight load; kernel
08:24:06: fence timeout + BACO, self-recovered 08:24:08. Single
instance, spontaneous (log kept:
/local/tmp/a3/server_mtp2n4_syv12.log). The driver's authorized
retry loaded clean (08:26:14, KV 15.42 GiB) and completed its sweep.

**Boot Y4 FINAL (at 09:17): 3 spontaneous chronic-family wedges
(#46 06:45, #47 07:29, #48 08:24 — all GPU1, all self-recovered) + 1
diagnosed software failure (not counted), across 8 TP=2 weight loads
(canary, probe x3 attempts [1 wedge, 1 software, 1 pass], A/B x4
attempts [2 wedges, 2 passes]) ≈ 38 % per-load wedge rate — the
highest per-boot rate recorded (Y3: ~33 % over 9 loads).** The A/B
verdict itself landed clean (SYV-12 v2: +6.4 % @120k, −5.3 % @64k —
payload-conditional; dev log boot-Y4 A/B entry). Per the pre-decision
recorded with #47 (and the #40/#42 wear criterion): ALL GPU WORK
STOPPED after the A/B (09:17 teardown verified VRAM 0/0); the B=4
campaign defers to the next boot.

## Boot Y4 (continued) — Kevin-directed B=4 campaign on the same boot; wedge #49 (2026-09-09 10:09–10:58 UTC)

Kevin overrode the stop at 10:0x: run the B=4 campaign now (same boot,
clean GPUs — no resident llama-servers this boot; UTIL=0.93; 120k
B=4 cell enabled). Campaign state at #49:

- **mtp3 B=1 anchor arm: COMPLETE.** Loaded clean 10:09 (KV 507,446
tokens/rank — +6.6 % vs the 0.85 pool), bench 64k×3 + 120k×2:
decode 26.7/31.1/39.0 @64k, 23.6/24.7 @120k (anchors match boot W'
mtp3v2 within spread). Clean SIGTERM teardown 10:51 (VRAM 0/0).
  Data: /local/tmp/b4/bench_mtp3_0909_1009.log.
- During this arm the driver's bench client surfaced the **~240 s
  @64k / ~580 s @120k per-request prefill stall** that every past TP=2
  sweep's ttft column hid (longstanding — all archive ttfts show the
  same values). Investigation (worker CPU 114 % during the stall,
  idle 0 %, no JIT, frozen-then-burst prefill): dev log
  `docs/gfx906/ttft-prefill-stall.md` (committed bdf1d814c4) with the
  T1–T4 theory matrix. Kevin's clarification: his long-standing
  "two pythons at 100 % during inference" observation means
  "not during idle", phase unspecified.
- **Wedge #49** (10:56:27, GPU1 0e:00.0): the cProfile diagnostic
  probe (in-process TP=2 `LLM()`, the dev log's §6.2 next step)
  launched ~10:52:30 after the clean mtp3 teardown — both ranks died
  at weight load; kernel 10:56:27: fence timeout + cp-unrecoverable +
  queue-evict + `Failed to quiesce KFD` + UNLOAD_TA 0x117 + BACO,
  self-recovered 10:56:29. Single instance, verified-clean pre-state —
  **SPONTANEOUS** (log: /local/tmp/b4/prof_64k.log). 4th spontaneous
  chronic-family reset on boot Y4 (4/9 loads ≈ 44 %).
- **Decision:** no retry spent on the diagnostic (it is not campaign
critical; the cProfile step stays queued for the next agent/boot).
  Next load = the campaign's greedy4 arm, under the standing directive.
  Burst rule unchanged: greedy4 attempt 1 wedging spontaneously = 2nd
  consecutive GENUINE launch failure (with #49) = BURST → stop +
  reboot (Kevin).

## Boot Y4 (continued) — wedge #50 on the kernel-breakdown probe launch (2026-09-09 19:06 UTC)

- **Context:** since #49 (10:56): ONE clean in-process TP=2 load — the
  TTFT matrix load A (`prof_64k.py`, 18:14:00 launch, full run1+run2
  data, clean atexit ~18:40; dev log `ttft-prefill-stall.md` §12,
  commit e867cdb1c0). Its three earlier launch attempts (18:08,
  18:11, 18:18) were all SELF-INFlicted (two my-profiler bugs aborted
  mid-load, one clean VRAM rejection against my own orphaned workers
  — no GPU event, not counted).
- **Event:** the per-step GPU kernel-breakdown probe (`kb_probe.py`,
  in-process TP=2 `LLM()`, mtp3 k=3, util 0.93, pp=1024, 8k traced
  prefill — the §12.6 next step) launched 19:06:10 on verified-clean
  GPUs (19:05: no python procs, VRAM 10.9 MB/card, KFD only gpuagent):
  weights reached ~60 % (shard 3/5, 19:06:40); both ranks then died
  with `c10::AcceleratorError: CUDA error: unspecified launch failure`
  (hipErrorLaunchFailure) — terminate() in both workers; the probe
  exited with leaked-shared-memory warnings. Self-recovered: 19:07
  check VRAM 10.9 MB/card, KFD only gpuagent, no stragglers. Single
  instance, verified-clean pre-state — **SPONTANEOUS**, chronic
  weight-load family (boot Y4's 5th; per-load rate 5/11 TP=2 loads ≈
  45 %). Log: /local/tmp/b4/kb_run.log. dmesg unreadable to kread
  (no sudo) — kernel-event BDF not recorded this time.
- **Decision (per house rule):** ONE retry authorized — this is the
  1st consecutive GENUINE single-instance launch failure since #49
  (the clean matrix load between resets the pair). Post-retry state
  verified before relaunch: no procs, VRAM baseline, both-GPU matmul
  probes (15.9/15.6 TFLOPS fp16).
- **OUTCOME (19:17:15–19:18:08): the retry WEDGED — same family,
  same phase** (weights ~60–80 %, both ranks `unspecified launch
  failure`, workers exited gracefully 19:18:08, self-recovered). Log:
  /local/tmp/b4/kb_run2.log. **2nd consecutive GENUINE single-instance
  launch failure (with #50) = BURST per house rule → ALL GPU WORK
  STOPPED; REBOOT required (Kevin executes — kread has no sudo).**
  Post-stop verified 19:21: no vllm/probe procs, VRAM baseline both
  cards, KFD only gpuagent. Nothing lost: §12 committed
  (e867cdb1c0); the kernel-breakdown step stays queued for the next
  boot (script staged at /local/tmp/b4/kb_probe.py +
  run_kernel_breakdown.sh). Boot Y4 final tally: 5 spontaneous
  chronic-family wedges (#46 06:45, #47 07:29, #48 08:24, #49 10:56,
  #50 19:06) + this burst retry, across ~14 TP=2 weight-load attempts
  (incl. 3 self-inflicted matrix aborts and the clean matrix load)
  ≈ 36 % per-attempt spontaneous wedge rate (6/14 incl. the retry).

## 2026-09-09 20:55:25 UTC — wedge #51 (boot Y5)

**Context:** boot Y5 (fresh boot after the #50 burst), up since ~19:22;
canary 38.8 t/s PASS at 19:26; then four fully clean TP=2 loads
(kb_run3 in-process, kb_run4 in-process, kb-serve serving, pp256
serving — all clean SIGTERM teardowns, VRAM to baseline).

**Event:** the §13.5 D1+D2+D3 combined session
(`/local/tmp/b4/run_stall3_session.sh`; mtp3 TP=2 serve, util 0.93,
pp=1024, port 8152 — a diagnostic/discriminator run, no campaign
data) launched 20:54:25 on verified-clean GPUs. TP0 reached
safetensors shard 3/5 (60 %); both ranks then threw
`c10::AcceleratorError: CUDA error: unspecified launch failure`
(hipErrorLaunchFailure) at the SetDevice copy; EngineCore init
failed; the APIServer exited cleanly. Self-recovered: 21:01 check
shows VRAM 0 % on both cards, temps 31/33 °C, sclk 938 / mclk 350
idle, rocm-smi fully nominal, no KFD stragglers. Single instance
(verified: the only process on the GPUs; one server launch).

**GPU:** dmesg unreadable to kread (dmesg_restrict) — BDF not
recorded, as with #50. GPU1 per the family's recurring-victim
pattern, unconfirmed.

**Classification:** GENUINE spontaneous, chronic serve-based
weight-load family. Boot Y5's 1st genuine failure (1/5 loads ≈ 20 %
— within clean-boot range). House rule: ONE retry after post-state
verification (no procs, VRAM baseline, both-GPU matmul probes); a
spontaneous wedge on the retry = 2nd consecutive GENUINE failure =
BURST → stop + reboot (Kevin).

**Session state:** fully staged under /local/tmp/b4/
(run_stall3.sh, stall3_client.py — holder+probe corrected design,
per_tid_sampler.py, stall3_analyze.py, run_stall3_session.sh). A
failed retry costs only the load; the pre-wedge work (D1 design
correction, D3 preliminary: the shim is a yield-poll synonym and
covers event syncs, the "100 % hot thread" re-analysis) is recorded
in `ttft-prefill-stall.md` §13.5 and survives.

## 2026-09-09 21:04:36 UTC — wedge #52 (boot Y5) — BURST

**Context:** retry authorized by the #51 entry. Pre-launch verification
complete at ~21:02: 0 vllm procs, VRAM baseline (0 %), both-GPU
matmul probes 15.9/15.6 TFLOPS fp16 (freshly probed, both cards).
Launch 21:03:29 (single instance — one server process, verified).

**Event:** progressed further than #51 — weight load reached 100 %
(5/5 shards on both ranks, ~6 s each); the crash moved to the
post-load init window (drafter load / graph-capture boundary): both
ranks `c10::AcceleratorError: unspecified launch failure`
(hipErrorLaunchFailure); EngineCore init failed 21:04:36; the
APIServer exited. Self-recovered: 21:07 check — VRAM 0 % both cards,
31/33 °C, sclk 938/mclk 350 idle, rocm-smi fully nominal, 0
stragglers. dmesg unreadable to kread — BDF not recorded (GPU1 per
family pattern, unconfirmed).

**Classification:** GENUINE spontaneous, chronic weight-load /
post-load-init family. **2nd consecutive GENUINE single-instance
launch failure (with #51, 9 min apart, a verified-clean inter-check
between them) = BURST per house rule.** ALL GPU WORK STOPPED 21:07;
REBOOT required (Kevin — kread has no sudo).

**Boot Y5 final:** 2 spontaneous wedges (#51 20:55, #52 21:04) across
5 TP=2 loads ≈ 40 %/load — worse than Y4 (36 %). Notable: the 4
pre-#51 loads on this boot (kb_run3, kb_run4 in-process; kb-serve,
pp256 serving) were all clean, so the wear manifested as a
consecutive pair at the 4th/5th load — the same shape as Y4's
#49→#50 pair (clean loads, then a pair).

**Session state (staged for the next boot):** /local/tmp/b4/ —
run_stall3.sh (mtp3 serve, port 8152, no profiler),
stall3_client.py (corrected D1: hold→complete→D1a warm-pool probe→
reset_prefix_cache→D1b cold control), per_tid_sampler.py,
stall3_analyze.py, run_stall3_session.sh (orchestrator with the
1 Hz clock logger + per-TID sampler + 30 s strace at t+90 s).
One launch + ~5.5 min wall. Expected reads: D1a ttft ~5.4 s
(batch-local) vs ~9.7 s (standing pool tax); per-TID CPU + strace
classify the §12.3 burner (yield-poll vs spin vs blocking KFD);
clocks pin the 78 %-MFU reference's clock state (D2, also the s0
outlier's clock-correlation baseline).

## 2026-09-10 10:45:17 UTC — wedge #53 (boot Y6)

**Context:** boot Y6 (fresh boot after the #52 burst), up since
~09:54; canary 38.8 t/s PASS 10:07; then two fully clean TP=2
serving sessions (stall3 10:08-10:16: D1a/D2/D3; stall4 10:31-10:40:
D1c + idle census + burner onset; both clean SIGTERM teardowns, VRAM
to baseline).

**Event:** the short stall5 burner-identity session (same mtp3 TP=2
server as stall4 plus `VLLM_STALL_PROF=1` with the new 20 s periodic
stack-dump; client: one 2k-in/4096-out request) launched 10:43:38 on
verified-clean GPUs (0 procs, VRAM baseline). Worker_TP0 reached
safetensors shard 4/5 (~80 %); both ranks then threw
`c10::AcceleratorError` terminate; EngineCore init failed 10:45:17;
the EngineCore exit path additionally segfaulted in
`__hipUnregisterFatBinary` (secondary — during exit handling after
the failure, not the cause). The known-benign cpuinfo
`JSONDecodeError` in the usage-reporting thread also appears (noise,
seen on clean boots). Self-recovered: 10:50 check — VRAM 0 % both
cards, 31/33 °C, sclk 938/mclk 350 idle, 0 stragglers. dmesg
unreadable to kread — BDF not recorded (GPU1 per family pattern,
unconfirmed).

**Classification:** GENUINE spontaneous, chronic weight-load family
— 3rd observed phase variant in this family across boots: #51 at
weights 60 %, #52 after weights 100 % (post-load init), #53 at
weights ~80 %. Boot Y6's 1st genuine failure (3rd TP=2 load). House
rule: ONE retry after post-state verification; a spontaneous wedge on
the retry = 2nd consecutive GENUINE failure = BURST → stop + reboot
(Kevin).

**Session state:** periodic stack dumps confirmed working (worker
profsamp files survived the unclean exit — the 20 s snapshot design
is exactly what SIGTERM-kill atexit-skip needs). Retry is the same
`run_stall5_session.sh`; the load is the only cost.

## 2026-09-10 10:52 UTC — wedge #54 (boot Y6) — BURST

**Context:** retry authorized by the #53 entry. Pre-launch
verification ~10:50:50: 0 vllm procs, VRAM baseline (0 %), both-GPU
matmul probes 15.9/15.6 TFLOPS fp16. Launch 10:51:15 (single
instance).

**Event:** died earlier than #53 — Worker_TP0 reached safetensors
shard 1/5 (20 %) and the crash hit by the 20→40 % boundary; both
ranks `c10::AcceleratorError` terminate; EngineCore init failed.
Self-recovered: 10:54:20 check — VRAM 0 % both cards, 31/33 °C,
sclk 938/mclk 350 idle, rocm-smi nominal, 0 stragglers. dmesg
unreadable to kread — BDF not recorded (GPU1 per family pattern,
unconfirmed).

**Classification:** GENUINE spontaneous, chronic weight-load family
(earliest failure phase yet in the family's observed spread:
#51 60 %, #53 80 %, #52 100 %+post-load, #54 ~20-40 %). **2nd
consecutive GENUINE single-instance launch failure (with #53, 9 min
apart, verified-clean inter-check between) = BURST per house rule.**
ALL GPU WORK STOPPED 10:54; REBOOT required (Kevin — kread has no
sudo).

**Boot Y6 final:** 2 spontaneous wedges (#53 10:45, #54 10:52)
across 4 TP=2 loads (stall3 10:08, stall4 10:31 clean) ≈ 50 %/load
— worst of the recent boots (Y4 36 %, Y5 40 %, Y6 50 %). Same
within-boot shape as Y4/Y5: clean loads first, then a consecutive
pair at the tail.

**Session state (staged for the next boot):** /local/tmp/b4/
run_stall5_session.sh (short burner-identity session: same mtp3 TP=2
server + VLLM_STALL_PROF with 20 s periodic stack dumps, one
2k-in/4096-out request, ~5 min wall). The periodic-dump mechanism is
proven (worker dumps survived #53's unclean SIGTERM-less exit). The
only open item it closes: the burner thread's dominant Python frame
(~1 core/worker, request-driven, onset 1 s after first request).

## Wedge #55 (2026-09-10 15:03, boot Y7)

**Timeline.** Boot Y7 up since 14:09. Canary 38.7 t/s PASS ~14:11.
stall5 (burner-identity, mtp3 TP=2 + VLLM_STALL_PROF) 14:14–14:16
clean — 51.3 s request, 0 AcceleratorErrors; TP0/TP1 stack dumps
collected (TP0 `async_output_busy_loop` 69 % of its samples in
`Stream.synchronize`; TP1's burner invisible to the frame sampler).
stall6 (wchan-sniffer attempt) 14:24–14:27 clean — 51.4 s request;
its external sniffer CSVs came back empty (diagnosed next, non-GPU).
~35 min of non-GPU /proc-debugging, then stall7 attempt 1 launched
15:02:01; kernel family event 15:03:11 on **0000:0e:00.0 (GPU1)** —
fence timeout → cp-unrecoverable → queue-evict fail → quiesce-KFD
fail → UNLOAD_TA(0x2) 0x117 → BACO → "GPU reset succeeded" 15:03:13;
userspace `c10::AcceleratorError` (hipErrorLaunchFailure) both ranks;
EngineCore init failed. Worker was at safetensors shard 4/5 in flight
(~60 % weights; 3/5 completed 15:03:01). Self-recovered; verified
15:06: VRAM 10.8 MB both cards, 0 vllm procs, KFD only gpuagent.

**Classification.** GENUINE spontaneous, chronic serve-based
weight-load family. **BDF confirmed via `journalctl -k`** (readable
to kread this boot — unlike boots Y5/Y6 where dmesg/journal were
unavailable): 0e:00.0 = GPU1, matching the family's recurring-victim
pattern.

**Bonus data from the unclean exit (the 20 s periodic dumps worked
again):**
- Worker census (profsamp_5229/5230, atexit): the per-tid
  comm/wchan reads were unreadable (proc view churn during teardown),
  but the TID LISTS survived: main + a first 15-TID group created at
  interpreter start (5235–5249) + three later groups of exactly 7
  (5275–5281, 5309–5315, 5332–5338) + scattered late TIDs. The 7-TID
  groups = torch OMP pool batches; OMP probe (new in sitecustomize
  v4): **`torch.get_num_threads()=8`, `OMP_NUM_THREADS='8'`** in both
  workers — 7 pool threads per worker.
- **`thread.ident` is NOT the OS TID on this platform**: ident values
  are pthread_self() addresses (e.g. 137047297746624) while the
  census TIDs are 5229–5444 (verified standalone: main ident
  136770900471936 vs tid 5961; the venv python also lacks
  `os.gettid` — custom/stripped build). profsamp's per-thread keys
  therefore cannot be joined to /proc TIDs; use thread NAMES
  (threading._active) + birth-order instead.
- **/proc sandbox churn (non-GPU observation, boot Y7 14:1x–15:0x):**
  interactive one-liner processes intermittently saw foreign pid
  ranges in `/proc/<pid>/task`, transient ENOENT on
  `/proc/<pid>/<tid>/stat`, and `$!`/Popen pids that did not match
  the content-view pids (off-by-a-few). Long-lived session
  infrastructure (stall5/6 samplers, in-process dumps) was
  unaffected in both directions. Guarded the stall7 census + per-TID
  sampler with a `stat.num_threads` cross-check (fail-closed SUSPECT
  marking / scan skip). Attributed to the tool's command sandbox,
  not host degradation — no GPU symptom; flagging here for
  completeness since it initially masqueraded as wedge fore-shadowing.

**Decision.** 1st genuine failure on boot Y7 → house rule authorizes
ONE retry of stall7 after post-state verification. If the retry
wedges spontaneously: 2nd consecutive GENUINE single-instance
launch failure (with #55) = BURST → stop + reboot (Kevin). The
stall7 experiment (burner-thread identity: birth-order census +
1 Hz per-TID CPU + in-process names) is fully staged; its value is
the last open item of the D3 stall-investigation arc.

## Wedge #56 (2026-09-10 15:29, boot Y7)

**Timeline.** Post-#55: the stall7 retry (15:13:38) loaded clean and
served the full session (45 s idle + 51.3 s request + 30 s post,
clean SIGTERM 15:18:04) — the D3 burner-identity data landed (see
`ttft-prefill-stall.md` §13.9). ~10 min of non-GPU analysis + two
standalone (no-vLLM) hip-spin burner-repro probes (both negative —
neither sustained matmul bursts nor per-iteration event syncs
reproduce the burn), then the stall8 shim-OFF A/B launched 15:28:34
(same mtp3 TP=2 session + `VLLM_GFX906_HIP_BLOCKING_SYNC=0`).
Weights 5/5 completed; post-load init (mamba page padding) 15:29:35;
kernel family event 15:29:44 on **0000:0e:00.0 (GPU1)**; userspace
SetDevice `hipErrorLaunchFailure` Worker_TP0 (pid 7450); EngineCore
init failed. Self-recovered; verified 15:33: VRAM baseline, 0 procs.

**Classification.** GENUINE spontaneous, chronic serve-based
weight-load family, post-load-init phase (cf. #52). The shim-OFF env
var cannot plausibly be the trigger (it only defers .pth context
creation to torch init_device; the family has hit every phase and
config on this host, and the pre-state was verified clean).

**Decision (wear-based stop, #40/#42 criterion).** Boot Y7 now carries
TWO spontaneous chronic-family BACO resets — #55 15:03 (GPU1) and
this 15:29 (GPU1 again, 26 min apart). Per the #40/#42 precedent the
authorized retry is NOT spent: the remaining experiment (the shim-OFF
A/B, exactly 1 load) tests a D3 nicety (burner causality), not a core
result — the burner identity itself is closed on the shim-ON data.
GPU1 showing two BACO resets in 26 min on one boot is the same shape
that ended Y3 (#44/#45, 50 s apart) and Y4's runs. REBOOT
recommended; the shim-OFF A/B (run_stall8_session.sh, ~6 min wall,
1 load) is the first job of the next boot after the canary, alongside
the 120kxB4 campaign decision.

**Boot Y7 final (pending Kevin's reboot):** 2 spontaneous wedges
(#55 15:03, #56 15:29) across 6 loads (canary, stall5, stall6,
stall7a, stall7b, stall8a) ≈ 33 %/load; GPU1 was the victim of both
(journal-confirmed 0e:00.0 — first journal-readable recent boot).

## Boot Y8 burst — wedges #57/#58 (2026-09-11): MBT-1 bt4096 arm, chronic weight-load family claims its 4th consecutive boot

**Session context.** Boot Y8 (fresh post-#56, up 09-10 ~16:23) had been the
best boot in days: ~5 clean TP=2 loads over two days (S13.10 shim-OFF A/B
2/2, S13.13 120k×B4 campaign server, 2026-09-11 stall5 burner session 08:29,
MBT bt2048 server 08:38→10:04 with a full clean 81-min 4×120k prefill). The
MBT-1 matrix (prefill-multibatch-tax.md) was ⅔ done — bt=1024 baseline
(§13.13, 75.4 min) and bt=2048 (81.1 min, chunk-INVARIANT — the decisive E1
result) — when the bt=4096 arm hit the family.

**#57 (10:10:03, GPU1 0e:00.0).** bt4096 attempt 1, launched 10:08:5x on
verified-clean GPUs (driver pre-probes 10.0/9.7 TFLOPS, VRAM 0/0, kernel log
clean). Worker_TP0 died ~90 s into weight load; kernel: qcm fence wait loop
timeout expired → cp-unrecoverable → Failed to evict process queues → Failed
to quiesce KFD → UNLOAD_TA(0x2) 0x117 → BACO, GPU reset (Source: 4).
Userspace: `hipErrorLaunchFailure` (CUDA error: unspecified launch failure)
from c10::cuda::SetDevice during a copy_ — the same SetDevice→copy_ weight-
load signature as the 2026-08-25 family entry. journalctl -k readable
(0e:00.0 confirmed — Y7's journal readability persists on Y8). Self-
recovered: driver post-check probes 10.0/9.8 TFLOPS, VRAM 0/0.

**#58 (10:12:4x, GPU1 again).** The authorized retry, same config, verified-
clean pre-state (post-#57 probes clean): died the same way ~70 s into weight
load; same fence-timeout/BACO signature on 0e:00.0 (~2.5 min after #57).
Post-check clean again.

**Decision (house rule + wear criterion).** 2nd consecutive genuine
single-instance launch failure (with #57) = BURST → all GPU work stopped.
Independently, Y8 now carries TWO spontaneous chronic-family BACO resets on
GPU1 in one boot — the #40/#42/#56 wear criterion says stop regardless.
**REBOOT required (Kevin).** NOT an OOM and NOT bt=4096-specific: both deaths
are the standard weight-load fence timeout at the standard phase; bt=4096's
prefill behavior remains untested (both attempts died before the first
chunk — the inductor-buffer OOM risk never got a chance to matter).

**Boot Y8 final:** 2 spontaneous wedges (#57 10:10, #58 10:12) across ~7
TP=2 loads ≈ 29 %/load; GPU1 victim of both; first wedges appeared only
after ~5 clean loads (late-boot wear shape, consistent with Y7).

**What the boot delivered despite the burst** (see
prefill-multibatch-tax.md E1 RESULT + ttft-prefill-stall.md §13.14):
stall5 burner outcome (native KFD/HIP thread, python-invisible — D3 final);
**E1 MBT-1 bt2048 = 4868.7 s (81.1 min) vs bt1024 75.3 min — the O(live-
context) tax is CHUNK-INVARIANT** (per-step-repeated overhead model refuted;
per-prefill-token × live-context work confirmed; predicts ~10–20× attention-
FLOP cost → kernel-efficiency target in long-context prefill attention).
Queued post-reboot: MBT-2 (seqs2), MBT-1-complement bt4096 (optional — bt2048
already establishes invariance), campaign 120k×B4 go, TP-1/FD-1.

## 2026-09-12 boot Y10 — fast-degrading morning (hold-trace session)

Timeline (UTC):
- 05:15:42 attempt 1 (mtp3b4 serve, port 8143, py-spy record --subprocesses
  --nonblocking as launcher wrapper): py-spy bailed at ~52 s ("process
  exited" false positive right at the EngineCore/Worker spawn; 25 sampling
  errors; server unaffected by py-spy's exit). Server weight load then
  faulted ~05:16:4x: c10::AcceleratorError "unspecified launch failure"
  (hipErrorLaunchFailure) in both workers ~40 s after Worker init. VRAM
  self-drained to 0/0. [degradation.md #60]
- 05:28:02 attempt 2 (retry, blocking-mode py-spy): engine loaded CLEAN
  (KV 501,558 tokens, 91% VRAM both GPUs, inductor artifacts reused) but
  the APIServer never opened its socket: uvicorn printed the full startup
  ladder ("Started server process" / "Waiting for application startup" /
  "Application startup complete") yet "Uvicorn running on ..." never
  appeared; ss + /proc/22536/net/tcp show NO listener on 8143; the main
  thread sat in do_epoll_wait (idle, utime 12.3 s total) for 7+ min.
  Health polls all refused; driver aborted at the 10-min cap. py-spy's
  speedscope (661 KB) holds only early-load samples — py-spy excludes
  idle threads by default, so the stall window (an IDLE epoll wait) was
  invisible to it. [degradation.md #61]
- 05:42:37 attempt 3 (py-spy-FREE discrimination, port 8144): Worker_TP0
  died at 05:43:45 in SetDevice — hipErrorLaunchFailure at the FIRST
  device touch, before weights. EngineCore aborted. [degradation.md #62]

Assessment:
- #60 + #62 are GPU-wedge-class and bracket the boot: faults at load and
  then at bare init = the AGENTS.md degraded-host signature. Reboot
  required; no GPU work attempted after #62.
- #61 is qualitatively different (software stall, engine healthy). Two
  hypotheses: (a) py-spy ptrace interference with uvicorn's startup
  (attempt 2 was the only py-spy-wrapped run that reached the API stage);
  (b) intrinsic vLLM async race — same flavor as the hold anomaly
  (R2/FD-1/mtp3b4-s1: a request admitted but never scheduled for minutes;
  here: a startup coroutine never resumed). The discrimination run was
  swallowed by #62; redo on a clean boot: first a plain launch, then a
  py-spy-wrapped one if the plain launch is clean.
- Tooling findings for gfx906-rocprofv3-kernel-trace / py-spy usage:
  py-spy 0.4.2 --nonblocking as a vLLM-launcher wrapper false-positives
  "process exited" at the multiprocessing spawn (~52 s); blocking mode
  survives but records only BUSY samples of short-lived phases unless
  --idle is passed; stall hunts on serving stacks need --idle + rate >=2.

## 2026-09-12 boot Y13 — wedge #67 (CAT-1 K2 A/B session; chronic weight-load family)

Context: boot Y13 was fresh (uptime 6 h) with no prior GPU work. Session was
the CAT-1 K2 serving arm (MTP k=2 + a 32,768-row draft head built from the
own-model corpus). Sequence of CUDA inits:

1. 20:06 canary (in-process TP=2, MTP k=2): **PASSED 38.9 t/s** at 20:09:21
   (`/local/tmp/mtp1/canary_mtp.log`); unit exited clean, VRAM back to
   10,924,032 B on both GPUs (verified before the next launch).
2. 20:09:22 `systemctl --user start mtp1srv@cat132k` (init #2). Worker died at
   20:10:27, ~65 s into engine init:

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure
Search for `hipErrorLaunchFailure' in .../HIP/ ... for more information.
```
   → `EngineCore initialization failed ... WorkerProc initialization failed
   due to an exception in a background process` (full trace:
   `/local/tmp/mtp1/server_cat132k.log` lines 55-120, 168-225).
3. Kernel log in the same second (20:10:26), both cards:

```
amdgpu 0000:0b:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0b:00.0: GPU reset(1) succeeded!
amdgpu 0000:0b:00.0: [drm] device wedged, but recovered through reset
amdgpu 0000:0e:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0e:00.0: GPU reset(1) succeeded!
amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

Assessment:
- Same chronic weight-load family as #65/#66 (boot Y12) and #57/#58 (Y8):
  fence timeout on `comp_1.0.0` at a TP=2 init, self-recovered by reset,
  VRAM cleared with no zombie processes (no BACO needed).
- Init-fault probability again followed the boot's 2nd CUDA init (Y10 #62,
  Y11 #63/#64, Y13 #67): the canary passes, then the first *server* init of
  the boot wedges. Consistent with the standing host-wear hypothesis.
- Retry discipline held: post-reset matmul probes (GPU0 10.0, GPU1 9.8
  TFLOPS vs the 7.0 gate) then TWO subsequent inits (#3 spec-less load,
  #4 the real MTP arm) loaded clean — hence a single event, not a burst.
- Operational note for this session type: the CAT-1 K2 arm needs exactly one
  TP=2 weight load. The canary costs an extra init, and the wedge record now
  shows init #2 of a boot is a *high-risk* init — so a canary pass, while
  required for health, does not protect the arm's own load; budget for one
  authorized retry per session.

## 2026-09-13 boot Y13 — wedge #69 (BACO reset; the same arm as #68) and the Y13 stop

Sequence (all on boot Y13, uptime 16 h at the stop):

- 20:09 canary PASS 38.9 t/s (init #1); 20:10 **#67** at `mtp1srv@cat132k` init
  (#2), both-GPU reset, self-recovered; 20:14 (#3), 20:20 (#4) clean loads;
  20:26-20:52 the CAT-1 A/B arms served 3×64k each cleanly.
- 21:29 **#68** at `mtp1srv@cat1ctrl` init (#7): `hipErrorLaunchFailure`,
  self-recovered, VRAM cleared.
- 06:13 (next visit) the **authorized retry** of that same arm (init #8):

```
amdgpu 0000:0e:00.0: GPU reset begin!. Source: 4
amdgpu 0000:0e:00.0: BACO reset
amdgpu 0000:0e:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0e:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0e:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0e:00.0: GPU reset(3) succeeded!
amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

- Both failures were the *same work dir* (`/local/tmp/mtp1/cat1_32768ctrl`);
  the three clean loads either side used `cat1_32768`, `cat1_pilot` and the
  pristine snapshot. With 2 consecutive failures the house rule applies:
  **GPU work stopped, reboot before the next session.**
- Discriminator to run first on the fresh boot: load `cat1_32768ctrl` once
  (1 load). Wedge again ⇒ suspect that work dir (rebuild it with a fresh
  `slice` and re-verify the index/extra-tensors pair); clean ⇒ boot wear
  (inits #7/#8 of a 16 h boot), no code implication.
- Two useful lessons recorded here rather than in the dev log: (a) an idle
  boot does not reset the init-fault probability — #68 and #69 were 9 h
  apart on the same boot; (b) the `hipErrorLaunchFailure` init wedge can
  escalate to a BACO reset on the retry, so retries are not free.

## 2026-09-13 boot Y14 — wedge #70 (first load-wedge of the boot, BACO reset)

Boot Y14 (07:10) was clean: canary PASS 38.9 t/s, one work-dir arm loaded and
served 1×64k decode, teardown clean (VRAM 10.9 MB both). Then the **pristine
snapshot** arm (init #3 of the boot) wedged at engine init:

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure)
...
amdgpu 0000:0e:00.0: GPU reset begin!. Source:  4
amdgpu 0000:0e:00.0: BACO reset
amdgpu 0000:0e:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0e:00.0: GPU reset(1) succeeded!
amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

Observations: (a) the init-wedge does **not** discriminate by model dir — the
previous boot's #68/#69 were the *work-dir* arm and I had begun to suspect that
directory; **this one is the pristine snapshot**, which exonerates the work dir
and re-confirms the boot-wear/load-family reading; (b) the BACO path now appears
in 2 of the last 3 wedges (#69, #70) — the escalation is no longer unusual;
(c) still clustering on the 1st-3rd CUDA init of a boot, minutes after a wedge-
free canary and a clean arm. One authorized retry per house rules.

## 2026-09-13 boot Y16 — wedge #72 (README-perf session's first big load; GPU0, in-place reset)

Boot Y16 (up 07:10) had already carried the V2-runner init wedge at 10:33.
This session: canary PASS **38.8 t/s** at 12:05 (GPU0), then the
headline-restamp script. Its first arm (MoE 35B) aborted for a **software**
reason (stale model path: `/local/models/QuantTrio/...` no longer exists — the
model now lives on `/data`; vLLM treated the path as a repo id and raised
`HFValidationError`). The script fell through to the dense arm, which died ~40 s
later at shard 0/8 of the 20.35 GiB checkpoint:

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure)
Exception raised from SetDevice at c10/hip/HIPFunctions.cpp:334
amdgpu 0000:0b:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0b:00.0: GPU reset(2) succeeded!
amdgpu 0000:0b:00.0: [drm] device wedged, but recovered through reset
```

Notes: (a) GPU0 (`0b:00.0`), unlike the GPU1-heavy pattern of the earlier
load-family events; (b) **no `VRAM is lost due to GPU reset!` and no BACO
line** — the fence-timeout path reset and recovered in place; (c) the trigger
sequence is *failed software launch → next load wedge*, i.e. the engine-init
path was cold rather than preceded by a long serve; (d) VRAM returned to the
10.9 MB/GPU idle baseline, `rocm-smi --showpids` shows only `gpuagent` — no
zombie KFD handles, so the house recipe applies: probe both GPUs, canary, then
ONE retry (two consecutive genuine failures = burst → stop + reboot).

## 2026-09-13 boot Y16 — wedge #73 + BURST (the retry of #72; GPU0 BACO, session stopped)

Retry sequence after #72 was textbook-clean until the second load: both GPUs
probed at **10.0 / 9.8 TFLOPS fp16** (healthy 10.0/9.7 band) and the canary
passed at **38.9 t/s**. Then, 5 min later, the MoE arm
(`/data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`, 23.71 GiB over AUTOFS) died at
shard 0/9 with the identical `hipErrorLaunchFailure`, and this time the kernel
took the BACO path:

```
amdgpu 0000:0b:00.0: GPU reset begin!. Source:  4
amdgpu 0000:0b:00.0: BACO reset
amdgpu 0000:0b:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0b:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0b:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0b:00.0: GPU reset(3) succeeded!
amdgpu 0000:0b:00.0: [drm] device wedged, but recovered through reset
```

This is the **2nd consecutive genuine single-instance load failure on GPU0
within ~6 min** (#72 at 12:08:55, reset(2); #73 at 12:14:33, reset(3)) ⇒ BURST
per the house recipe: all GPU work stopped. The script's third arm (dense,
which had reached 62 % of its 8-shard load without error) was killed rather
than allowed to continue; VRAM returned to the 10.9 MB/GPU idle baseline and
`rocm-smi --showpids` shows only `gpuagent`.

Reading: the pattern is the **chronic weight-load family on this boot** (both
failures at shard 0, both on GPU0, the *second* right after a clean probe and
canary), i.e. boot-wear rather than a model/code/path cause — the model path
change (NFS `/data`) and the software abort of the first arm are incidental.
Two GPU resets in six minutes is the pre-full-wedge wear signal, so the boot
is spent: **reboot before further GPU work** (a fresh boot's first load is the
cleanest slot we have; see the load-lottery tallies in this file).

## 2026-09-13 boot f27e8058 — wedge #74 (TP=2 first launch of the CAT-1 headline session; GPU1, in-place reset)

Clean-boot context: canary **39.0 t/s**, probes **10.0/9.8 TFLOPS**, four
harness runs completed with mclk verified at 1000 MHz (MoE/dense x cold/warm),
~90 min of GPU work with no event. Then the CAT-1 agentic session's first arm
(TP=2, `run_server.sh greedy`) launched and GPU1 reset in place:

```
amdgpu 0000:0e:00.0: PSP is resuming...
amdgpu 0000:0e:00.0: reserve 0x400000 from 0x87fe000000 for PSP TMR
amdgpu 0000:0e:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0e:00.0: GPU reset(1) succeeded!
amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

The GPU1 worker process died, so the driver's `ready` check reported "failed to
start" (the launcher deleted that attempt's server log — the campaign scripts
keep `.attempt1`; worth porting back). The **authorized retry loaded clean in
1.7 min** and the session continued; no BACO, no `VRAM is lost`. Same
load-family signature as #72/#73 but on the *other* GPU and after a long clean
stretch, i.e. boot-wear rather than a specific arm or config.

## 2026-09-13 boot f27e8058 — wedge #75 (CAT-1 arm first launch; GPU1 BACO, retry passed)

Second wedge of the boot, same GPU as #74 (GPU1 `0e:00.0`) and the same
weight-load family, this time escalating through BACO:

```
amdgpu 0000:0e:00.0: BACO reset
amdgpu 0000:0e:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0e:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0e:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0e:00.0: GPU reset(2) succeeded!
amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

Trigger: the third and last arm of the CAT-1 headline session (TP=2,
`cat1v3k3` = Qwen3.8-27B + the CAT-1 v3 draft head + MTP k=3) launching after
the k=3 arm's clean teardown. The worker died, so the arm reported "failed to
start"; the **authorized retry loaded clean in 5.7 min** and the sweep ran.
Between #74 (15:51) and #75 (16:51) the session completed six full sweeps
(greedy + MTP k=3 on the agent corpus, 4 reps each) with no event, i.e. two
resets an hour apart on the same card with long clean stretches between —
boot-wear/load-lottery behaviour, not an arm, model-dir or config dependency.

## 2026-09-13 boot f27e8058 — wedge #76 (0.29.0 validation; GPU0, weight-load family, hung process)

Context: after the upstream v0.29.0 merge, the worktree build succeeded and the
FA suite passed (89/89, 18:16). The PPL probe (fresh merge, in-process Qwen3.8-27B)
then died during checkpoint load:

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure)
amdgpu 0000:0b:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0b:00.0: GPU reset(2) succeeded!
amdgpu 0000:0b:00.0: [drm] device wedged, but recovered through reset
```

New detail worth recording: **the Python process survived the reset in a zombie
state**, still holding 20.8 GB (KFD table showed it on the physical GPU that
rocm-smi labels GPU[0] — note the numbering mismatch with `HIP_VISIBLE_DEVICES`),
with the traceback already printed and the interpreter stuck in teardown.
`SIGTERM` was ignored; `SIGKILL` released the memory immediately and VRAM
returned to the 11 MB idle baseline. House recipe for the next session: after a
wedge, check `rocm-smi --showpids` for a *live* holder of the model-sized
allocation before relaunching — a recovered-but-hung client is otherwise
indistinguishable from a clean state in the log alone.

## 2026-09-13 boot f27e8058 — wedge #77 + BURST (0.29.0 validation; GPU0)

Second consecutive genuine load failure on this boot, ~2 min after #76 and with
no successful load in between (the FA suite's kernel-level tests do not load a
checkpoint, so they do not count as a clean load for the wear tally):

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure)
Exception raised from SetDevice at /build/pytorch/c10/hip/HIPFunctions.cpp:334
amdgpu 0000:0b:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:0b:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:0b:00.0: Fence fallback timer expired on ring comp_1.0.0
amdgpu 0000:0b:00.0: GPU reset(4) succeeded!
amdgpu 0000:0b:00.0: [drm] device wedged, but recovered through reset
```

Per the house recipe this is a BURST: all GPU work stopped, reboot before any
further GPU work. Boot f27e8058 (up 14:23, ~4 h) has now recorded six events
(#72–#77), five of them GPU0 weight-load failures — the same accelerating
load-family wear seen on earlier boots. No hung holders this time (the smoke's
engine-core process died with the accelerator error; VRAM returned to the
10.9 MB baseline by itself).

Validation status at the stop (for the resume): the 0.29.0 worktree **build
succeeded**, all three compiled extensions import, the **FA suite passed
89/89 twice**, and the static sweeps are clean (no conflict markers,
`compileall`, ruff F821/F811 apart from one pre-existing upstream F811). The
remaining gates — the PPL probe (blocked by #76) and the V1 serving smoke
(blocked by #77) — need a fresh boot.


## 2026-09-16 07:10:05 — INT8-PACKED-1 first load wedges at weight load (half-wedge, reset-recovered)

**Context:** the first attempt to serve `lued/Qwen3.8-27B-INT8-W8A16-DFlash2` on this fork, after wiring
`quant_config` + `prefix` into `Qwen3_5Model.embed_tokens` (the W8A16 pack-quantized checkpoint's
embedding needs the quant method; see `DEVLOG-int8-packed.md`). Boot 2026-09-15 13:20 (the degraded boot that already carried wedges #93-#95 today), GPUs idle before the run.

**Signature:** `vllm serve` (TP=2, `--dtype float16`, `--kv-cache-dtype float16`, 8k ctx, no spec decode)
logged `Resolved architecture: Qwen3_5ForConditionalGeneration`, then died during weight load with
`Error: CUDA error: unspecified launch failure` (the `hipErrorLaunchFailure` class). Kernel log:

```
07:10:05 amdgpu 0000:0e:00.0: Fence fallback timer expired on ring comp_1.0.0
07:10:05 amdgpu 0000:0e:00.0: GPU reset(4) succeeded!
07:10:05 amdgpu 0000:0e:00.0: [drm] device wedged, but recovered through reset
```

**Outcome:** half-wedge — the device recovered through `reset(4)` (no BACO, no `PSP resume failed`), VRAM
returned to 0.00 GB on both GPUs, and the process exited. No `weight_packed` / loader-key error appeared,
i.e. the failure happened at/after the weight-load boundary rather than at key matching.

**Interpretation:** the same boot's load lottery as #68-#95 (this is a 5-hour-old boot with several wedges today, exactly the state where the degradation rule predicts load failures) (this INT8 checkpoint adds a new load path:
401 packed tensors + a Triton dequant-gather embedding kernel are staged at load time). One authorized retry
followed; a second consecutive genuine load failure would be a BURST (stop GPU work, reboot).

## 2026-09-26 — 35B MoE graph-capture page fault on the 0.30.0 base (new signature)

Boot 2026-09-24 19:00, ~36 h up, no passwordless sudo. Context: the **0.30.0
release gate** `_bench_gfx906.py /data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`
(`BENCH_EAGER=0 BENCH_GPU_UTIL=0.95 BENCH_SAMPLES=4 BENCH_PP=2048 BENCH_TG=256
BENCH_MAX_SEQS=32`, V2 default, `cudagraph_mode=FULL_DECODE_ONLY`).

**Signature (reproducible, 2/2 V2 runs).** During the FULL-graph capture phase
(after `Capturing CUDA graphs (FULL) 3/4`), the ROCm runtime reports:

```
Memory Fault Error [host: mi50-01, GPU index: 0, faulting addr: 0x784758401000,
 kernel: void vllm::moe_gptq_gfx906::moe_gemm_q4_kernel_gfx906<1, 2>(
   __half const*, __half*, unsigned int const*, __half const*, unsigned int const*,
   float const*, int const*, int const*, int const*, int ×11, bool, int, int)]
```

then `hipErrorIllegalAddress` and SIGABRT (rc=134). The retry window
(07:45–08:02) logged **916** `gfxhub0 no-retry page fault` lines — a fault storm,
all in that kernel. `moe_gemm_q4_kernel_gfx906<BLOCK_SIZE_M=1, N_PER_THREAD=2>` is
the default M=1 decode MoE GEMM (`csrc/rocm/moe_q_gemm_gfx906.cu`).

**What is ruled out.** `moe_q_gemm_gfx906.cu` is **unchanged by the merge**
(`git diff 524ac6f2d6 8893a50e54 -- csrc/rocm/moe_q_gemm_gfx906.cu` is empty), and
the merge's `CMakeLists.txt` change does not touch gfx906 kernel flags. The same
model passed the **eager** PPL probe minutes earlier (PPL 15.9840, 0 misses), and
the **dense** 27B canary captures graphs cleanly (38.2 t/s, low-normal not DEG).

**Two hypotheses.**
(a) **Host state**: this boot accumulated many wedge events; a degraded host can
trip sync-cadence-heavy paths. Counter-evidence: the documented DEG symptom is
*slowness*, and this log has **no page-fault precedent** — every prior event is
`hipErrorLaunchFailure` during *weight load*.
(b) **A real 0.30.0 regression** in the **V2 + 35B-MoE + FULL-capture** path:
0.29-V2 ran the identical bench at **59.86 t/s** (`/local/tmp/bench_flip.log`,
2026-09-18), and the fork's serving recipes pin **V1** (`_serve_tp2_gfx906.sh`),
so the shipped serving path may be unaffected.

**Outcome.** During the V1 control attempt the GPU hit the familiar
`hipErrorLaunchFailure` **during weight load** and reset (`GPU reset(1) succeeded`
08:00:53, `VRAM is lost`) — the second failure of the session, so GPU work
**stopped per the house burst rule**. No passwordless sudo → the host reboot is
the operator's call.

**Next step (discriminating test).** Reboot, then re-run the V2 35B bench once on
a clean boot. Healthy fail → hypothesis (b): bisect the merge's capture-path
changes (upstream's MRV2 FULL-graph work, #51700 / #54646 / #56382 / #56312) for
the fork-MoE interaction, and ship 0.30.0 with the V1 pin (as the recipes already
do) plus a roadmap item. Clean pass → hypothesis (a), record the boot state and
proceed with the release gates.

**2026-09-26, fresh-boot discrimination (host reboot 08:08:39, 0 wedge events):**

- **V2 retry on the fresh boot: same fault, harder error.** The 35B bench
  aborted again during capture, this time reported as
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access
  memory beyond the largest legal address` in the *same* kernel
  (`moe_gemm_q4_kernel_gfx906<1, 2>`) — no journal page-fault lines, but the
  same capture-phase abort. **This rules out hypothesis (a), host state: it is a
  real 0.30.0 regression** in the MRV2 FULL-graph capture path for this model.
- **V1 control on the fresh boot: clean, rc=0.** `VLLM_USE_V2_MODEL_RUNNER=0`
  ran the identical bench to completion: **60.446 / 60.444 / 60.404 / 60.388 t/s**
  (mean **60.42**) with no faults. (The merged worker logs
  `Using V2 Model Runner` only for V2, so the V1 attribution is unambiguous.) So
  the fork's shipped serving configuration — the recipes pin V1 — is unaffected,
  and 0.30.0 is ~0.9 % faster than the 0.29 house number (59.77–59.86).

**Verdict: `REL30-1` is a V2-only regression.** Ship 0.30.0 on the V1 pin (the
shipped config); the V2 + 35B-MoE + FULL-capture fault stays a `DFL2-2` item.

## `REL30-1` root cause (2026-09-26): padded `topk_ids` rows make the gfx906 MoE walk off the expert tables

**VERDICT: a real 0.30.0 V2-only regression in the fork's `Gfx906WNA16Experts`, not
host state and not the kernel.** Reproducible on a fresh boot; the V1 path is clean
(60.42 t/s). Root-caused with an env-gated shape dump in
`vllm/model_executor/layers/fused_moe/experts/gfx906_w4a16_moe.py`.

**Evidence chain.**

1. The aborted launch is `moe_gemm_q4_kernel_gfx906<1, 2>` with `grid=[2048, 2, 8]`,
   `group_seg_size=528`. `528 = BLOCK_SIZE_M * (256 + 8) * 2`, so **BLOCK_SIZE_M = 1**;
   `grid.y = cdiv(size_n, 256*2) = 2` and `grid.z = cdiv(size_k, 256) = 8` pin it to
   **gemm1** (N=1024, K=2048). `grid.x = num_token_blocks = 2048`.
2. `num_token_blocks = sorted_token_ids.size(0) / block_size_m` — a host-side bound
   (`csrc/rocm/moe_q_gemm_gfx906.cu:894`). With `block_size_m = 1`, the align buffer
   held **2048** entries.
3. `moe_align_block_size` sizes that buffer from **`topk_ids.numel()`**, while the
   fork's `apply()` derives `em = M * topk` from **`hidden_states.size(0)`**. The two
   agree only when `topk_ids.size(0) == hidden_states.size(0)`.
4. `BLOCK_SIZE_M = 1` requires `em <= 32` (`_block_size_m_for`), i.e. `M <= 4`, so
   `topk_ids.numel() = 2048` **cannot** come from `M*topk` — it requires
   **`topk_ids.size(0) = 256`** while `hidden_states` had <= 4 rows. The align then
   builds blocks from the *unfilled* rows of the padded buffer, whose expert ids are
   garbage; the kernel computes `b_q_weight + expert_id * expert_weight_stride` and
   reads far outside the allocation → page fault / aperture violation.
5. The shape dump confirms every non-faulting call is consistent
   (`M=8→tid=(8,8)/ntb=64`, `M=4→ntb=32`, `M=2→ntb=16`, profiling `M=4096→ntb=4320`);
   only the faulting capture phase has the mis-sized align.

**Why 0.30 and not 0.29.** Upstream's base `FusedMoEExpertsModular.moe_problem_size`
**asserts** `topk_ids.size(0) == a1.size(0)`; the fork **overrides** that method
without the assert, so the contract violation passes silently. V2's new input-batch /
capture machinery (`vllm/v1/worker/gpu/input_batch.py`, FULL capture for microbatched
steps #51700, warm-before-capture #55341) is what now hands the MoE a topk_ids with
more rows than `hidden_states`.

**Fix (implemented, awaiting validation).** In `Gfx906WNA16Experts.apply()`, restore
the contract before aligning:

```python
if topk_ids.size(0) != M:
    topk_ids = topk_ids[:M]
```

so the align buffers are sized from the same token count the GEMM uses. **Validation
is blocked**: both attempts to re-run the 35B bench after the fault died ~43 s in with
the boot's init-lottery `hipErrorLaunchFailure` (broken-amdsmi); a host reboot is
needed.

**Resolution of `REL30-1` (2026-09-27).** Fixed by gating the fork's fused M=1
align to an explicit V1 selection. Measured on the house config, both `rc=0` with
0 faults: **V1 60.07 t/s** (60.094/60.056/60.028/60.083, fused align on) and
**V2 58.90 t/s** (58.904/58.976/58.800/58.880, fused align skipped,
`Using V2 Model Runner`). A `register_fake` for the align op was tried and does
**not** fix the uninitialized out-params — recorded so it is not retried. The
diagnostic instrumentation used to find this (C++ out-param/ids dump, Python
shape dump) has been removed and the extension rebuilt.

## 2026-09-27 08:29–09:40 — observation #110 (bench and PPL normal, sync-heavy mamba suite ~10-20x down and stalling; amdsmi broken)

Fresh boot at 08:29 (the reboot after #109). Within the first hour:

- **Steady-state gates are fine**: the 35B house bench reads **59.14 t/s** (59.077 / 59.13 / 59.142 / 59.203; the release record is 59.79 and
  the release notes' V2 number 58.90), and the *merged tree's* PPL probe reads **10.5472 with 0 top-20 misses** — the exact reference value.
- **The sync-heavy suite is not**: `tests/kernels/mamba` (195 tests, ~56 s on a healthy boot, and 195 passed on this code family earlier in
  the week) ran **8 minutes to 57 %** on the release line at 100 % CPU (progressing, just ~15x slow) and **stalled at 54 % with 0.0 % CPU** on
  the merged tree — twice, the second time as the *only* GPU job on the box. Same 54 % position as the pre-reboot run of #109.
- **amdsmi is broken again**: `_query_gcn_arch_from_amdsmi()` raises `RuntimeError: amdsmi did not return valid GCN arch` on both trees, and both
  fall back to `torch.cuda` with `on_gfx906()` True. This is the same pre-wedge symptom recorded for #106 and #108.
- The journal has no GPU reset, no `Failed to resume KFD`, no OOM; `rocm-smi` reports both GPUs and VRAM returns to the ~10 MB baseline after each
  run stopped.

Interpretation: this is #108's asymmetry (dense GPU work at speed, sync-cadence workloads collapse) with the added end state of a stall rather than a
uniform slowdown, and it is **not** a property of either tree — both show the slow suite, and the trees differ only in AMD-side/`qwen4_exp` code that no
mamba test touches (verified: arch detection identical, `on_gfx906()` True in both, the merged tree's added `platforms/rocm.py` plausibility check is
unreachable because the amdsmi call raises before it). Consequence: **do not gate the suite on this boot**; the merge's suite verdict rests on the
healthy-boot runs of the same code (195 passed on this code family) plus the completed gates here (PPL 10.5472, QSA 22 + 19, PR #2 harnesses 52,
tiny rig boot + 3/3 requests).

---

## Carried from the QSA-FN (0.29) line — `degradation_details.md`

The Qwen3.8-Flash-Next/QSA work landed on this line from `gfx906/qsa-fn` (merged 2026-09-27). This section is that branch's content for this file, kept verbatim so the records survive the union; the entries above are this line's own.


## 2026-09-22 20:23 — wedge #106 (kill-and-relaunch of the standing 27B MTP k=3 server; GPU1 left half-wedged)

**Context.** A drafter-graph A/B needed the standing server restarted (two arms differing
only in whether the drafter inherits the target's cudagraph mode, mirroring the tester's
`GFX906_DRAFTER_GRAPHS` gate in PR #2 — a local, uncommitted patch, Python-only, no
rebuild). The running server (pid 7181, Qwen3.8-27B-AWQ-INT4, TP=2, fp16, MTP k=3,
maxlen 262144, util 0.82, both decks at 29.4 GiB) was SIGTERM-killed and the same config
relaunched ~15 s later.

**Event.** APIServer up 20:21:52; the weight load then died at worker init (20:23:21):

```
CUDA error: unspecified launch failure  /  hipErrorLaunchFailure
  from c10::cuda::SetDevice  (at::native::copy_ -> HIPFunctions.cpp:334)
Exception: WorkerProc initialization failed ... Failed core proc(s): {}
RuntimeError: Engine core initialization failed.
```

The traceback is the chronic weight-load family (#93–#105), raised before any of the
patched drafter code could run. The boot's pre-symptom matches the documented degradation
signature: `[rocm.py:1063] Failed to get total memory via amdsmi, falling back to
torch.cuda` (amdsmi broken since boot, as in #66/#67).

**Post-state.** GPU0 drained back to 10.9 MB once the failed workers exited, but **GPU1
stayed at 10.44 GB with no process in the KFD list** (only `gpuagent`) and did not clear
across 2+ minutes of polling — zombie VRAM, i.e. a half-wedge that needs a BACO reset or a
reboot to reclaim. Sequence observed right after the SIGTERM: GPU0 briefly held 2.1 GB
(not yet released) while GPU1 was already at 19 MB, so the leak is specific to the crashed
worker on GPU1.

**Action.** One authorized retry, on **GPU0 only (TP=1)** — the A/B is a *relative*
comparison of drafter graph capture, so TP=1 keeps the measurement valid while avoiding
the wedged deck; the absolute t/s is therefore not comparable to the recorded TP=2
numbers. A second consecutive wedge stops GPU work for the session (the "2 in a row →
stop" rule).

**Lesson for the entry protocol:** killing a long-running TP=2 server and immediately
re-launching the same config is a load-lottery roll like any other first-load — the kill
itself does not appear to be the trigger, but the retry budget should be assumed before
starting an A/B that needs a restart.

**Retry (20:34–20:55) — wedged again, differently, GPU work stopped.** The GPU0-only retry
(`HIP_VISIBLE_DEVICES=0`, TP=1, util 0.90, same model/MTP k=3 config, `GFX906_DRAFTER_GRAPHS=1`)
launched 20:34:16 and then **never reached the API server's first log line**: the process sat
in kernel `D` state with `wchan = amddrm_sched_entity_flush` for 11+ minutes, 0.3 % CPU, no
child processes, zero bytes of log, GPU0 back at the 10.9 MB baseline and GPU1 still holding
the 10.44 GB zombie. `SIGKILL` cleared the process once the driver call returned; no VRAM
came back on GPU1.

That is a **second consecutive wedge** in the same session (first = #106's load-lottery
`hipErrorLaunchFailure`, second = the retry stuck in the DRM scheduler), so per the
"2 in a row → stop" rule **GPU work stopped for the session**. The driver state explains
both: amdsmi has been broken since this boot (the documented pre-wedge symptom), and
`amddrm_sched_entity_flush` cannot complete while a scheduler entity from the killed
long-running server is still registered — the deck needs a BACO reset or a **host reboot**
(root required for BACO), after which GPU1's zombie 10.44 GB should clear.

**Consequence for the deferred work:** the drafter-graph A/B (the reason for the restart)
did not run. Nothing about it has been measured; the PR-review claim that the tester's
default would cost our spec configs their drafter graphs remains an argument from their own
+6 % measurement, not from our box.

## 2026-09-23 06:20 — observation #107 (draft-model load stall, TP=1 arm abandoned)

Boot from 2026-09-22 21:21 had served five clean TP=2 boots (the drafter-graph and
all-reduce A/B) plus the FD arm. The sixth launch — the TP=1 arm, after two earlier
non-GPU failures of the same arm (a KV-capacity refusal at `max-model-len 131072`,
then a `Free memory 13.51/31.98 GiB < 0.82` race against the previous server's VRAM
release) — loaded the target model normally (`Loading weights took 71.15 seconds`)
and then **stalled loading the draft model**: the log held

```
Loading safetensors checkpoint shards:   0%|  | 0/5 [00:00<?, ?it/s]
```

for ~13 minutes, unchanged (0 bytes of log growth over 30 s), with the EngineCore at
3.9 % CPU, 26 GB of VRAM held on GPU0, no error line, and `rocm-smi` showing both
decks healthy (no reset, no zombie VRAM). This differs from the #93–#106 family,
which dies or spins; here the load simply stopped. `SIGTERM` took the process down
and released the VRAM.

Recorded as a load-stall observation rather than a wedge (no reset, no leaked VRAM,
and the driver stayed serviceable). Consequence: the TP=1 drafter-graph arm was
abandoned, so the sweep's TP=1 half is unmeasured — the TP=2 half (three arms, three
contexts, ±0.2 % ms/step) answers the question it was asked.

## 2026-09-23 ~15:00–16:20 — observation #108 (quiet degradation: −4.7 % on both trees, no wedge)

The PR #2 integration bench looked like a regression at first: the 35B house bench on
the PR tip read **57.21 t/s** (57.16–57.30, mclk 1000) against the **59.79** recorded on
the 2026-09-22 boot. Three samples per arm, tight, so it was worth chasing.

Control: the *pre-cherry-pick* tip (`aa6982deb4`) benched on the **same boot** read
**57.00 t/s** (56.96–57.06) — the same number, so the ~4.7 % is the host, not the
branch. The branch is +0.4 % (inside noise) against its own control.

Boot context: rebooted 2026-09-22 21:21, ~19 h up, after ~15 GPU loads (the drafter A-B-A
session, the FD arm, the real-payload sweep, the tiny-rig boots, suites and two benches),
with the earlier wedge #107 (draft-model load stall) in the window. `amdsmi` is broken on
this boot (`_query_gcn_arch_from_amdsmi` raises `RuntimeError` — the same pre-wedge
symptom documented for #106 and for the 2026-09-16 degradation), both trees fall back to
`torch.cuda` and `on_gfx906()` is True in both, so the amdsmi fallback we merged from
PR #2 is behaviour-identical here. No GPU reset appears in the reachable logs; the
spec-decode canary was not run (the failure mode is a uniform slowdown, not the
sync-cadence-only collapse the canary detects).

Consequence: **perf gates measured on this boot are not comparable to the 09-22
numbers** (the flips' 59.79 baseline, the drafter sweep, the AR arms). Functional gates
(suites, tiny rig) are unaffected. Reboot before any further perf work, and re-run the
35B bench to confirm the host is back at ~59.8 before quoting branch-vs-main perf.

### #108, addendum (2026-09-24 ~18:30, next boot — degradation is not just the bench delta)

On the following boot the same picture shows up in the suites, and much larger than
4.7 %: `tests/kernels/mamba` (195 tests, ~56 s on a healthy boot, and 195 passed
earlier the same day) ran at **~8 % progress per 9 minutes** — a ~50x slowdown, i.e.
effectively the stalled pattern — and an earlier combined FA+mamba run on the
previous boot hung at 56 % for ~10 min before it was stopped. The 35B bench on this
boot is only ~4.7 % down, so the two do not scale together; sync-heavy / many-small-
kernel workloads are the ones that collapse, which is the same asymmetry as the
spec-decode-only collapse recorded for #106 (dense GPU work normal, sync cadence
gone). `amdsmi` is broken here as well. The combined run was stopped by hand; the
individual mamba run that followed died on its own on the next boot — see #109, which
supersedes this addendum. The PLE change in question touches only
`qwen4_exp/amd/ple_layer.py` and its test, so it cannot affect mamba kernels, and the
same suite passed on this branch before the cherry-picks. Re-run them after a reboot.

## 2026-09-24 18:13–18:45 — observation #109 (KFD resume failure 22 min into a fresh boot; one GPU's suite collapses)

The host was rebooted at **18:13:44** (`who -b`, uptime 33 min at the time of writing) — so this
is a *fresh* boot, and it is not clean. There is **no** GPU reset, `HwException`, ring timeout or
OOM anywhere in `journalctl -k` for the day; the only kernel-side anomaly is:

```
Sep 24 18:36:22 mi50-01 kernel: amdgpu: amdgpu_amdkfd_restore_userptr_worker: Failed to resume KFD
```

Timeline: at ~18:33 two GPU jobs start — `tests/kernels/mamba` on GPU1 and the tiny QSA rig on
GPU0. The suite reaches 46 % by 18:36, then slows to ~8 % per 9 minutes; at 18:45 it stops writing
and the process has exited **silently** — no pytest summary, no traceback, 560 bytes of log, no OOM
in the journal. The rig on the other GPU boots and serves its 1344/2016/4031-token sequence normally
in the same window, and `rocm-smi` still reports both GPUs (VRAM back to the ~10 MB baseline).

So this is neither the quiet uniform slowdown of #108 (the 35B bench was only ~4.7 % down there) nor
a full wedge (no reset, rocm-smi alive). It is the *sync-heavy-suite-collapse* shape on one GPU,
three minutes after that GPU's first real load of the boot, with a KFD resume failure at exactly
that moment. Compare #106: dense GPU work at full speed while the sync cadence is gone; the same
asymmetry, a different boot.

Consequences for the record: the mamba/FA *kernel* suites cannot be run on this boot, so their
verdict for the cherry-picks rests on (a) the suites that did complete on the changed code (their
PLE + GC harnesses + our QSA file: 74 passed; tiny rig boot + requests with MTP k=1/k=3), and
(b) construction — the PLE commits touch `qwen4_exp/amd/ple_layer.py` and its test, `config/vllm.py`
behind a `model_type == "qwen4_exp"` gate, and the drafter knob restored to upstream's default, so
no mamba or FA kernel is in the diff. Re-run them after a reboot; do not read their absence as a
verdict on the branch.


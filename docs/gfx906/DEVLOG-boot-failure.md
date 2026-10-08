# Boot-failure hunt — hipErrorLaunchFailure at weight load (2026-08-23)

> Investigation session 2026-08-23 ~17:59–20:00 UTC · `gfx906/main` @
> 4a9e24b5ca (no code changes) · host mi50-01 · started as the live-confirmation
> step of the 256k-prefill OOM hunt (`oom-256k-prefill.md`, fix design in
> `/local/tmp/gfx906-fa-fix.md`), blocked by a new host fault.

**STATUS: OPEN — unresolved, but INTERMITTENT, not deterministic** (corrected
20:30Z, §7): boots flap between good windows (full canary PASS, 38.9 t/s)
and bad windows (~1-2 min, every attempt aborts mid-load). The §4
GTT-exhaustion working theory was **refuted by direct measurement** in §7
(20 MiB peak of 12,260 MiB GTT during a full 19.57 GiB load). Current
leading cause: intermittent GPU die/fabric wedges, GPU0 primary (16/19
resets today; ATHUB uncorrectable + 2× pcie_bif latches on GPU0). Takeover
agent resumes from §7.

Repro logs + pause doc in `/local/tmp/oom_hunt/` (persistent),
snapshot of the pause doc also at `/tmp/start-error.md`; second-session
artifacts in `/local/tmp/boot_fail/`.

## 1. Symptom

Deterministic abort mid weight-load, ~40 s in, at **shard 3/5, tensor #801**
(`model.language_model.layers.31.mlp.down_proj.weight_packed`, 5120×2176
int32, ~44 MB) — near-identical across all 10+ attempts:

```
Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:27<...]
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure, rc=134)
Exception raised from SetDevice at HIPFunctions.cpp:334
  → at::native::copy_ ← torch dispatcher ← Python weights loop
```

The `SetDevice` frame is the *first API call to notice a sticky async error*
from an earlier kernel — not the faulting op itself. Kernel side, same
second: `qcm fence wait loop timeout expired` on **ring comp_1.0.0**
(compute queue — not SDMA) → `Failed to evict process queues` → `psp gfx
UNLOAD_TA failed (0x117)` → BACO **GPU reset**. One isolated
`ERREVENT_ATHUB_INTERRUPT` uncorrectable error (18:35, GPU0 only).
RAS: clean except 1 correctable pcie_bif per card (link x16/16 GT/s).

Timeline of the failure rate: real-weight boots passed all morning (W4 A/B,
soak, OOM arms 12:13–13:06); one intermittent failure at **12:48** (mid OOM
cluster, recorded in `degradation_details.md` § OOM cluster); then
**10/10 failures 17:59–19:47** on both cards, on the 18:30 warm reboot, and
on the **19:14–19:20 full power cycle** (6 min off). ~20 GPU resets today.

## 2. What it is NOT (each ruled out with a run)

| Hypothesis | Ruled out by |
|---|---|
| Host wedge residue | Persists on fresh boot AND full power cycle |
| GPU0 hw (full-wedge history 06:08, PSP −62) | Fails identically on **GPU1**; bare matmul + H2D pass both cards |
| Card-specific fault | Both cards, same cumulative failure point |
| The uncommitted FA instrumentation patch | No-op unless `OOMHUNT_LOG` set; repro fails with **zero vllm code** |
| Kernel optimizations (W4 skinny 10:50, C2-V 21:51) | (a) `load_format='dummy'` 9B boot **passed** full init+generate — quant param alloc, custom-op registration all fine; (b) merges touch decode-time GEMM dispatch only, env-gated off; (c) minimal repro (below) imports no vllm |
| The 10:55 `_C_stable_libtorch` rebuild | Same venv+binary passed real-weight boots repeatedly post-rebuild (W4 A/B ~09:3x, soak 10:0x, OOM arms 12:13–13:06) |
| SDMA copy engine | `HSA_ENABLE_SDMA=0`: identical failure; fence timeout is on compute ring comp_1 |
| Prefetch thread | vLLM logs "Auto-prefetch is disabled" (EXT4 + checkpoint > RAM) |
| Host RAM pressure | 17 GiB free at failure; fill test holds 20 GiB verified |
| "Too many copies" | 2000× repeated 200 MB RAM→GPU copies pass |

## 3. Minimal reproducer (no vLLM)

`/local/tmp/oom_hunt/safeopen_repro.log`, `mater_repro.log`: **pure torch +
safetensors `safe_open`**, iterating all 2396 tensors, copying each into a
fixed 200 MB GPU buffer with a sync per tensor. **Dies after ~800 tensors**
("800 tensors ok" then identical `hipErrorLaunchFailure` + comp_1 fence
timeout + GPU reset) — both in the mmap-direct form and with `.contiguous()`
materialization first (so not an mmap/page-fault pathology). A `load_file`
+ 4×512 MB H2D test passed — the failure needs **many distinct host tensors
copied to the GPU**, not bulk bytes.

## 4. Open paradox + working theory

The same venv/binary/model **passed the same load all morning**. Nothing on
disk changed (no rebuild, no config change, no new packages). What changed is
invisible state. Working theory: **cumulative exhaustion of a GPU-visible
host-mapping resource** (GTT/GART pages or kernel-queue objects) that crosses
a limit mid-shard-3 — ~800 distinct tensors ≈ ~8 GiB of distinct host
mappings touched, near the 12.5 GiB GTT aperture. Explains: consistent
cumulative point, both cards, power-cycle persistence (driver-static limit,
not card state), and the morning-passes/afternoon-fails drift if the
effective limit moved at the 12:48 event.

## 5. Next steps when resumed

1. **First-attempt-after-boot repro** — does the very first per-tensor loop
   of a boot pass where the second fails? (Boot-fresh vs drained state.)
2. Count/size sweep + `rocm-smi --showmeminfo gtt` sampled during the repro.
3. `HSA_ENABLE_SDMA=0` + `AMD_SERIALIZE_KERNEL=3` combined run; pinned
   staging buffer copies.
4. If GTT exhaustion confirmed: `amdgpu.gttsize` kernel parameter, or loader
   staging through a pinned buffer (fixes vLLM generally on this host).
5. Docker userspace A/B: `mixa3607/vllm-gfx906:0.28.0rc2-19e23ffedd-...`
   (isolates userspace torch/HIP vs driver).

## 6. Degradation records

All resets today (05:18, 05:47, 06:08 full wedge, 08:52, 12:48, 17:59×3,
18:23, 18:35, 18:39, 19:23, 19:40, 19:46 + power-cycle window) are recorded
in `degradation_details.md` (§ sections 2026-08-23). The 256k-OOM hunt
remains paused at "fix designed, live validation blocked"
(`oom-256k-prefill.md` follow-up + `/local/tmp/gfx906-fa-fix.md`).

## 7. Second session 20:00–20:30 UTC (pi, 19:20 boot): GTT refuted — hardware flap

**VERDICT:** the §4 working theory (GTT/GART exhaustion) is **refuted by
direct measurement**. The boot failure is **intermittent GPU compute-queue
wedging** (die/fabric class), flapping in ~1-2 min bad windows and
30+ min good windows; **GPU0 primary** (16 of 19 resets today, all
ATHUB/mmhub/pcie_bif on-die RAS latches). The host *is* usable in good
windows — a full 27B-mtp2 canary passed at 38.9 t/s (≈ healthy; the 06:33
BOOT line precedent: 38.6 = healthy-for-this-model) — but any long run
(soaks, 256k needle) risks a mid-run wedge.

### 7.1 Experiments (all on the 19:20 post-power-cycle boot)

1. **§5.1 answered — first-attempt-after-boot PASSES.** §3 repro re-run:
   PASS (2396 tensors, rc=0, ~28 s warm) — and 3 more consecutive passes,
   followed by a full vLLM canary (`/tmp/oom_hunt/canary.sh`) PASS
   (5/5 shards, 38.9 t/s). Meanwhile this same boot's earlier four attempts
   (19:23 canary, 19:35, 19:40, 19:46) all failed. Not boot-fresh dependent.
2. **Intermittency quantified.** 10× back-to-back repro loop (28 s/run):
   FAIL 20:15:17 (rc=134), FAIL 20:15:50, then PASS ×5+ through 20:18:17.
   The bad window lasted ~60-90 s. (Another repro PASS at 20:23:50 while
   sampling GTT, §7.1.4.) Then an **isolated wedge at 20:21:33** (comp_1
   fence timeout, ~30 s after the loop ended, nothing adjacent failing) —
   so good windows contain isolated wedges too: it flaps, it is not strict
   good/bad phases. Two more full canaries on the same invocation passed at
   20:40 (38.2 t/s) and 20:41 (38.6 t/s, `canary_auth*.log`).
3. **Live-caught wedge + RAS precursor.** The 20:15:16 failure shows the
   usual comp_1 fence-timeout signature, and the kernel logged
   `1 correctable hardware errors detected in pcie_bif block` (GPU0)
   **41 s before** the queue death (20:14:35) — an on-die PCIe-interface
   RAS latch preceding the wedge.
4. **GTT measurement — the decisive experiment (§5.2), GTT REFUTED.**
   Sampled `/sys/class/drm/card{0,1}/device/mem_info_gtt_used` at 150 ms
   while running the §3 full load (19.57 GiB, 2396 tensors, distinct
   mmap'd host regions — the exact "distinct-page churn" of §4):
   **peak 20 MiB of 12,260 MiB** (`mem_info_gtt_total` =
   12,553,486,336 B = 11.68 GiB; kernel: "11971M of GTT memory ready";
   idle baseline 14 MiB per card). A load moving 19.57 GiB of distinct
   host pages uses **0.16 %** of the GTT. A 11.7 GiB capacity limit cannot
   trip at a 20 MiB peak, and the same load pattern has passed many times
   today (a strict-capacity mechanism would forbid any 19.57 > 11.7 GiB
   load — yet 6 boots passed 11:41–13:06 and all of W4 this morning).
5. **Whole-day RAS tally** (kern.log, boots 06:33/18:30/19:20 confirmed
   via `Linux version` lines): on-die hardware-error events — GPU0: mmhub
   no-retry page fault 06:08:32; ATHUB uncorrectable 18:35:55; pcie_bif
   correctable 18:50:19.637 and 20:14:35. GPU1: pcie_bif correctable
   18:50:19.643 — **6 ms after GPU0's** (dual-card simultaneous timing;
   shared host-side cause or delayed post-reset flush — ambiguous). BACO
   resets: **19 total, GPU0 ×16 / GPU1 ×3**. PCIe AER device counters: all
   zero on both cards (link x16/16 GT/s per §1).
6. **More ruled-outs.** fwupd installed nothing today (13:35 refresh was
   metadata-only; `get-updates` empty). No GPU work ran 13:06–17:59
   (syslog silent except sysstat/fwupd) — the 17:59 failure was boot B's
   first GPU0 attempt after a 5 h idle. PCIe topology: both GPUs sit behind
   parallel two-bridge chains off adjacent root ports (03.1/03.2 →
   0b:00.0 / 0e:00.0).

### 7.2 Re-reading of §1/§3 observations

- **"Deterministic at tensor #801 / ~8 GiB": a time-to-failure artifact.**
  A cold load reaches tensor 801 at ~40 s; the run dies when it enters a
  bad window, and at steady copy rate that lands near the same elapsed time
  every time. The 2000×200 MB pass (§2) ran inside a good window. No
  capacity crossing is involved (§7.1.4).
- **§2 "GPU0 hw ruled out":** valid for a *card-only* fault; the RAS
  evidence now points at GPU0's die/fabric as primary (16/19 resets, all
  its latches), with the 18:50 dual-card simultaneous pcie_bif latch
  suggesting a shared host-side contributor (root complex / riser / power
  rail) for that event.

### 7.3 For the takeover agent

1. **Card-swap test** (the discriminating experiment): swap GPU0↔GPU1
   between the symmetric slots, run §3 repro; fault follows the card →
   GPU0 RMA/spare, stays in the slot → slot/riser/root port. Or run §3
   repro on these cards in another machine.
2. **Pinned-staging loader** (the other agent's fix items 1-2): a
   legitimate loader *optimization* (and good for a 23 GB-RAM box), but
   NOT the fix for this failure — GTT is not the trigger. Note for the
   write-up: on ROCm `non_blocking=False` does not change the H2D mapping
   path (both go through the same user-memory DMA); only pinned buffers do.
3. Keep logging wedges in `degradation.md` per protocol — rows from 18:35
   on (incl. the 20:14:35 pcie_bif latch + 20:15:16/20:15:48 double wedge)
   were added from kern.log in this session.
4. The FA OOM fix is tracked separately in `/local/tmp/gfx906-fa-fix.md`
   (unbounded `_gather_retired` growth under chunked-prefill Sk growth —
   the OOM-cluster story; verified against the code, 89.4 GiB retired by
   250k tokens in their simulation). Unrelated to the boot failure.

**Artifacts:** `/local/tmp/boot_fail/` — `repro.py` (§3 repro),
`gtt_sampler.py` + `gtt_trace1.out` (the §7.1.4 data: per-150 ms GTT
trace, PEAK 20 MiB), `canary_now.log` (the passing canary), `loop10.out`
(the fail/fail/pass/… sequence), `kern_before_exp` (kern.log snapshot).

## 8. Third session 20:30–20:45 UTC (pi, 19:20 boot): fresh boot REPRODUCIBLY CLEAN + auth GTT/root sentinels

**VERDICT:** on the 19:20 post-power-cycle boot the host is now **reproducibly
clean**: every full 19.57 GiB load and every vLLM canary **PASS**, zero GPU
resets across all runs — continuing the §7 pattern (intermittent flaps, not
persistence). The failure is NOT reproducible on a settled fresh boot. A
discriminating experiment (distinct-alloc vs one-sliced-alloc) was run and
is also NEGATIVE for any capacity mechanism.

**GATE:** full §3 repro (2396 tensors, rc=0) + full vLLM canary (5/5 shards,
m/svc), both clean on a fresh boot with no prior failed attempts this boot.

### 8.1 Experiments (all on the 19:20 boot, HIP_VISIBLE_DEVICES=0)

1. **Distinct-alloc vs sliced-alloc discriminator (synthetic).** 1200 ×
   1 MiB **distinct** anonymous `torch.empty` buffers, each copied once into a
   1 MiB GPU buf with per-copy sync → **COMPLETED all 1200**, GTT+VRAM flat
   (0.02 / 0.19 GiB). Then 1200 **views of ONE 1 GiB** host alloc → aborted at
   view #0 — but that was a **script bug** (view step < dst, shape-mismatch
   RuntimeError mislabeled as a wedge), not a device fault; re-run in
   foreground showed a pure Python shape error, GTT clean, no kernel events.
   Net: **1.2 GB of distinct-alloc churn passes** — no per-alloc/per-mapping
   capacity ceiling reproduces on a clean boot. (Synthetic; a real load is
   19.57 GiB with 2×2.37 GiB `lm_head`/`embed_tokens` tensors, which §7
   already measured flowing through at a 20 MiB GTT peak.)
2. **Bare-torch full-load ×3 IN-PROCESS, clean.** 19.57 GiB / 2396 tensors
   repeated 3× back-to-back in one interpreter: 26.6/26.8/26.5 s → **3/3 PASS**, `mem_info_gtt_used` flat at 0.02 GiB. No in-process accumulation.
3. **Real vLLM canary, FULL PIPELINE PASS.** `canary_auth.log`: main model
   5/5 shards, 53.8 s; drafter loaded; torch.compile cache hit; graph
   capture clean; **`CANARY: 256 tok / 6.7s = 38.2 t/s`**; clean exit; GTT/VRAM
   returned to idle (14/10 MiB). Zero amdgpu reset/fence events in kern.log
   during the whole run.
4. **Second canary back-to-back — PASS (38.6 t/s), zero resets.** Answers the
   §7.1.1 corollary: not only the first attempt passes on this boot, a
   **subsequent full canary in the same boot passes too** — no
   cross-process accumulation either.
5. **Bare-torch full load AFTER the two canaries — PASS (33.9 s).** Even after
   two full vLLM sessions have run and torn down, the identical-with-morning
   load passes. No residual device state from prior loads trips it.

### 8.2 What this session did NOT see

- **No wedge, no reset, no `qcm fence timeout`, no `hipErrorLaunchFailure`**
  in any run (kern.log grep across 20:38–20:46 = 0 events).
- GTT never left ~0.02–0.19 GiB; VRAM only reached 19.88 GiB **during a
  passing** canary model load (weights resident), dropping to idle after.
- The §7 flap (fail in a ~1-2 min window, then pass for 30+ min) had no bad
  window in 20:38–20:46.

### 8.3 Re-read + status

- Reinforces 7.2: the §1 tensor-#801 / cumulative-point "determinism" is a
  **time-in-window artifact**, not a capacity crossing — the same load passes
  cleanly 10+ times on this boot.
- **Root-enabled observables provided** (for the next real wedge):
  `sudo bash /local/tmp/oom_hunt/root_probe.sh /local/tmp/oom_hunt/root`
  — copies `devcoredump/data` (decisive wedge-time VM/queue state, root-only),
  dumps `/sys/kernel/debug/amdgpu` (KFD/VM nodes) + GTT/VRAM per card + rocm
  --showpids + amdgpu params (`vm_block_size/num_kcq/vm_update_mode/noretry`,
  all currently -1/0 = defaults). Run BEFORE a repro and again AFTER a wedge;
  compare. Snapshot dir populated 20:38 (`kern_203832.log`, no devcoredump).
- Current standing: the boot-failure is an **intermittent hardware/driver
  flake (die/fabric class, GPU0 primary, §7 RAS evidence) that reproduces in
  bad windows and is documented-clean on a fresh settled boot** — NOT a
  code, GTT, loader, or FA fault. Card-swap test (§7.3.1) + capturing the
  root/devcoredump artifact on the next real wedge remain the open next steps.

**Artifacts:** `/local/tmp/oom_hunt/` — `gtt_dissect.py`(+log, §8.1.1),
`gtt_trace_probe.py`(+log+`gtt_trace.tsv`), `canary_auth.log` (§8.1.3),
`canary_auth2.log` (§8.1.4), `root_probe.sh` (§8.3 root sentinel).

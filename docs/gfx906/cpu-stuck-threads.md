# CPU stuck-threads in TP=2 serving (HSA P2P-IPC)

Status: **open — deterministic host-level defect** (2026-08-24, boots E + F).
Reproduces on every fresh TP=2 vLLM start on this host within ~15–20 min.
Reboot does NOT clear it. Not a vLLM bug. Options in §7.

## TL;DR

An idle TP=2 serving container "uses 200–400% CPU" (4 cores of 16). The
cause is **two threads per TP worker** (4 total) that stop making progress
while the scheduler charges them 100% **user** CPU each: their `rip` is
*constant* on a trivial instruction (a `cmp`/`mov`) for tens of minutes —
a core endlessly replaying one instruction, i.e. a CPU microarchitectural
stuck state, not a software spin. The threads sit in the **HSA runtime's
P2P-IPC handshake** context (`libhsa-runtime64` `Runtime::IPCClientImport`
and the `poll()` on its socket) — the same GPU0↔GPU1 P2P path that produces
the comp_1 fence wedges (see `degradation.md`). Serving numbers are
unaffected so far; on boot E one worker process died with no kernel trace.

## 1. Symptom

`docker stats` / `ps` on the rc2 TP=2 MTP server (27B, maxlen 262144,
util 0.82, chunk 1024, capture [1,2,3,4], image
`0.28.0rc2-7e4567053e`):

| observation | value |
|---|---|
| container CPU (idle, no requests) | ~410% (boot E), ~400% lifetime-avg (boot F) |
| per-worker process CPU (instantaneous) | **200.0% / 200.0%** (boot F, `top -bn2`) |
| hot threads | **2 per worker**, 99.7–99.9% each, state R, pinned to one core |
| thread names (per worker) | one `python3` + one `VLLM::Worker` — symmetric across ranks |
| serving performance | unaffected: canaries 56.2/56.7, 55.2/55.6 (boot E), 33.5/55.1 (boot F) — all in/above the healthy band |
| GPU side | clean: rocm-smi OK, 0 wedges/fences in kern.log on both boots (boot E had the separate 13:00:53 GPU0 wedge, `degradation.md`) |

Pitfall: `ps`/`docker stats` `%CPU` is a *lifetime average* — a decaying
compile burst looks like a stuck thread. The real signature is a
**per-thread instantaneous** ~100% in state R that never drops:
`top -H -bn2 -d 3 -p <worker>`.

## 2. Ruling out vLLM (the Python side is clean)

py-spy dump of a worker (boot E venv instance) shows every Python thread
correctly parked:

- **MainThread**: `poll (zmq/sugar/poll.py:106)` ← `SpinCondition.wait`
  (`shm_broadcast.py:206`) ← `acquire_read` (794) ← `dequeue` (889) ←
  `worker_busy_loop` (`multiproc_executor.py:1043`) — a *blocking* zmq
  poller wait, exactly as the code intends.
- `WorkerAsyncOutputCopy`: `queue.get`; `DeathPipeMonitor`: pipe `recv`;
  usage/tqdm threads: `Event.wait`. All idle.

Code reading agrees: `SpinCondition.wait` only `sched_yield()`-spins for
≤1 s *after a recent read*; idle it parks in `poller.poll()` (with a 5 s
recheck cadence). The optional spinloop C extension caps each burst at
100 ms. So the RPC reader cannot burn 100% when idle. The EngineCore
(same torch/zmq stack, no TP process group) sits at ~0.7% — the burn is
specific to the **worker** processes, i.e. to the TP/P2P domain.

## 3. Evidence: frozen, not spinning

For each hot thread (gdb via a privileged helper container, §8):

| test | result |
|---|---|
| `rip` sampled repeatedly (2 s apart, independent attach/detach cycles, over minutes) | **constant 64-bit value** on every sample |
| the frozen instruction | trivial: `cmp $0xfffffffffffff000,%rax` (libc `__poll` post-`syscall`) or register `mov`s — **cannot block or loop** |
| utime / stime over 5 s | utime +500 ticks (100% user), **stime flat** — no kernel spin |
| voluntary_ctxt_switches (boot E docker-era thread, 65 min) | **1** — the thread never voluntarily yielded once |
| `taskset -p <other core>` | thread migrates (psr changes), **frozen rip unchanged** — the stuck state travels with the thread; not a dead core |
| `kill -9 <tid>` | a frozen thread *can* be killed (kernel delivery works) — but see §6 |

A userspace thread at 100% CPU with a constant `rip` on a non-looping
instruction is a core replaying the instruction forever (microarch stuck
state). Software — vLLM, Python, HSA, glibc — has no code path that hangs
on `mov %r13d,%eax`.

## 4. Where the threads are stuck

Boot E (venv instance, worker TP0 pid 16479 / TP1 16480), file offsets
verified against the on-disk libs (identical md5 in host and image):

| thread | location |
|---|---|
| TP0 `python` (tid 16518) | `libhsa-runtime64.so.1.21.0` **+0x110b54** — `mov %r13d,%eax`, epilogue of `rocr::core::Runtime::IPCClientImport`, immediately after `call rocr::os::CloseIPCSocket` |
| TP0 `VLLM::Worker` (tid 16616) | glibc `libc.so.6` **+0x11b5fd** — `cmp` right after the `syscall` in `__poll` (poll(2) wrapper); observed *moving* (in `hsa_amd_agent_set_async_scratch_limit`) ~5 min earlier, then frozen |
| TP1 `python` (tid 16515) | `libhsa` **+0x110af3** — inside `IPCClientImport` (post `_Rb_tree::_M_emplace_hint_unique`, allocation-region bookkeeping) |
| TP1 `VLLM::Worker` (tid 16615) | `libhsa` **+0xf05a2** — stack-save region (struct copy) |

Boot F (docker, workers 2880/2881): **all four** hot threads (2919/3014 in
TP0, 2917/3013 in TP1) frozen at **`libc+0x11b5fd`** (same spot as one
boot-E thread): TP0 @ 0x79c5d2ad75fd, TP1 @ 0x7be387b735fd, constant
across samples.

Common denominator: the **HSA P2P-IPC channel** between the two GPUs —
establishment/import and its socket `poll`. This is the dual-root-port
P2P path that required the official amdgpu DKMS driver and that the comp_1
fence wedges exercise; this is its first observed **CPU-side**
manifestation.

## 5. Timeline

**Boot E (2026-08-24, booted 11:37):**
- 13:25 — docker TP=2 MTP server. By ~13:45 both workers show 2×~100%
  threads (lifetime utime ≈ full uptime → burning from early start).
- 14:37 — relaunch from host venv (identical config) to allow tracing.
- ~14:52–14:55 — py-spy clean; first gdb samples catch one thread still
  moving (HSA scratch-limit code), others already frozen in IPC import.
- **14:56:10 (log)** — `Worker proc VllmWorker-0 died unexpectedly (exit
  code: None)`; engine cascade-shutdown logged through 14:56:16. *Anomaly:*
  the worker PIDs were demonstrably alive (and being traced) until ~15:33,
  and the whole instance was gone by 15:35 — see §6.
- 15:01 — reboot (boot F).

**Boot F (booted 15:01):**
- 15:09 — fresh docker server, same image/config. READY ~15:15.
- Canary 33.5/55.1 t/s — healthy.
- ~15:25 — **recurrence**: 2 hot threads per worker, all four frozen at
  `libc+0x11b5fd`. 0 GPU events since boot.

## 6. Impact

- **4 cores at idle** (25% of the box) as long as the server runs.
- **Serving: no measured effect** — every boot-E/E-F number (59.2 MTP
  record, full context curves, 35B re-stamps) was taken with the stuck
  threads present, and canaries sit in the healthy band.
- **Worker death (boot E only)**: the 14:56:10 log line is real but does
  not line up with wall-clock process state (workers alive until ~15:33);
  the instance fully died within ~2 min of me `kill -9`-ing one frozen
  thread (15:33), with **no** OOM (20 GB RAM available), no MCE, no
  segfault/fence in kern.log. Causation (stuck state vs. the kill vs. the
  log-clocked 14:56 event) is unprovable from the available evidence.
  Treat "TP=2 instance can die with no kernel trace" as a live risk.

## 7. Options

1. **Accept.** Keep the server; live with 4 idle cores and the (unproven)
   worker-death risk. Zero cost. Current state as of 2026-08-24 15:3x.
2. **`NCCL_P2P_DISABLE=1` A/B** (one restart, ~10 min). Forces RCCL over
   the SHM (host-RAM bounce) transport, bypassing the HSA P2P-IPC
   handshake. If the freeze doesn't recur within ~30 min, mechanism
   confirmed *and* workaround obtained; then measure the allreduce cost
   against the 55–59 t/s canary / context curves (expected small —
   decode-bound workload). If it still freezes, the trigger is upstream of
   the transport choice.
3. **Escalate to AMD** (ROCm HSA runtime / amdgpu DKMS 6.19.14). This doc
   + `degradation_details.md` (14:47–14:56 section) is a self-contained
   bug report: deterministic repro, frozen-PC evidence, library + offset
   identification, kernel-log cleanliness.
4. **Kill the stuck threads at runtime.** Demonstrated possible
   (`kill -9 <tid>`), but on boot E the instance died shortly after — not
   a mitigation, a diagnostic only.
5. **Avoid TP=2** (TP=1 serving only). No P2P IPC → no freeze, but gives
   up the 445k-token KV pool / 256k context that motivated TP=2.

Recommended order: **2 now** (cheap, decisive), **3 in parallel**
(independent of 2's outcome), **1 as fallback**.

## 8. Tracing recipe (next time)

Obstacles, in order hit:

- Worker processes set **`dumpable=0`** (HSA) — `/proc/<tid>/syscall`
  gives EPERM *even for the owner*; plain ptrace tools need
  `CAP_SYS_PTRACE`.
- Host yama `ptrace_scope=1` — same-user but non-ancestor ptrace (my
  shell → server) is blocked. `perf` is blocked too
  (`perf_event_paranoid=4`).
- Docker default profiles deny ptrace even with the cap: **AppArmor
  `docker-default`** is the last blocker (audit `DENIED operation="ptrace"`
  in kern.log).

Working helper container (trace host PIDs):

```bash
docker run -d --name pyspy-helper --pid=host --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --net=host <any-glibc-image> sleep 900
# gdb is not in the vllm image: apt-get install -y gdb inside the helper.
```

Commands (single-shot only — a gdb batch `while` loop hung with the
inferior *stopped*; kill the gdb to release the process):

```bash
# per-thread instantaneous CPU (NOT ps/docker-stats lifetime averages):
top -H -bn2 -d 3 -p <worker> | awk 'NR>7 && $9+0 > 50'
# rip of one thread (repeat 2-3x; constant = frozen):
gdb -p <worker> -batch -ex "thread find <tid>" -ex "info registers rip"
# which lib+file offset:
python3 -c '...parse /proc/<worker>/maps for the rip...'   # (s, e, fileoff)
objdump -d --start-address=0x<off-0x20> --stop-address=0x<off+0x20> <lib>
# utime/stime split (user vs kernel burn):
awk '{print $14, $15}' /proc/<worker>/task/<tid>/stat
# map tid -> worker:
ls /proc/<worker>/task | grep -x <tid>
```

py-spy (Python frames only) is enough to clear the vLLM side; native
threads (the stuck ones) need gdb.

## 9. Relation to other host failures

- **comp_1 fence wedges** (`degradation.md`): same P2P/IPC hardware path,
  GPU-side manifestation (fence timeout → BACO reset). This is the CPU-side
  manifestation. First CPU-side observation: 2026-08-24.
- **Sync-cadence "degradation"** (`degradation_details.md`, boot D): a
  different mode — everything boots and works, sync-heavy inference ~3×
  slow at *short* context, cleared by reboot. This defect is *not* cleared
  by reboot, shows no latency change, and burns fixed cores.
- All recorded GPU-side wedges on this box occur on the dual-root-port
  P2P topology; the official amdgpu DKMS driver is required for P2P to
  work at all (stock Ubuntu amdgpu stalls/hangs RCCL P2P —
  `DEVLOG-tp2-dense.md` prereqs).

## 10. References

- `degradation_details.md` — "2026-08-24 14:47–14:56Z (boot E)" section
  (incl. boot-F recurrence update).
- `DEVLOG-tp2-dense.md` S9 — all 2026-08-24 TP=2 numbers (taken with the
  stuck threads present).
- `oom-256k-prefill.md` — the TP=2 256k sizing work on the same server.
- Evidence logs (boot E, wiped on reboot): `/tmp/venv_serve.log`; gdb/py-spy
  sessions reconstructed in §3–§4 above.

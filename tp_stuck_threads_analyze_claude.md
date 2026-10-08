# Deeper analysis: CPU stuck-threads in TP=2 serving

Reviewer: Claude (Sonnet 5), CPU-only analysis on the live host, 2026-08-24.
Source doc reviewed: `docs/gfx906/cpu-stuck-threads.md` (boots E+F).

**Update (same day, later pass):** folded in a second independent review
(`tp_stuck_threads_analyze_gpt.md`, by GPT) and cross-checked both against
the actual ROCm 7.14 source, now available locally at
`/local/git/TheRock` (the `rocm-systems` submodule pins commit
`2b22ab0195...`, which is the exact build hash embedded in the installed
`libhsa-runtime64.so.1.21.0` — "`1.21.7-local-build-2b22ab0195`" — so this
is source-exact, not "close enough"). §0 below is the new, most important
material: it turns one of the two theories into a **confirmed real bug in
`Runtime::IPCRecvHandle`**, found by reading `os_linux.cpp` directly. The
rest of the doc (§1–§7) is my original binary-only pass, left intact
because it's still accurate and GPT's review reaches compatible
conclusions by an independent route; §0 supersedes the speculative parts
of §3 and §4 below with source-confirmed answers.

## Verdict up front

The doc's headline claim — **"a core endlessly replaying one instruction,
i.e. a CPU microarchitectural stuck state"** — does not survive scrutiny of
the evidence presented, and is now directly contradicted by source. The
`python`-thread half of the signature (boot E, frozen inside
`IPCClientImport`) is a **confirmed real bug**: `rocr::os::IPCRecvHandle`
(ROCm 7.14 source, `os_linux.cpp:1080-1102`) contains an unconditional
`while (!rcv) rcv = recvmsg(...)` with no timeout, backoff, or error path,
which spins forever once the peer's socket hits EOF — see §0. The other
half of the signature (the `__poll`-frozen `VLLM::Worker` threads on both
boots) most likely maps to RCCL's proxy service intentionally polling with
`timeout=0` while async ops are outstanding (`proxy.cc:1783-1787`,
explicitly commented as deliberate) — plausible, not yet TID-confirmed;
see §0 and §5. Both are **mundane non-blocking busy-poll/spin loops**, not
a CPU fault, and the `ptrace` attach/detach sampling method the doc used
cannot actually distinguish "frozen forever at address X" from "spends the
overwhelming majority of every loop iteration at or near address X" in the
first place (see §1–§2, my original pass, still valid). I did not have GPU
access for this pass (CPU-only per your instructions) so I could not
attach live and get a native backtrace to finish the TID-level attribution
between the two mechanisms — that's the one remaining open step, and it's
cheap (§5.1).

I also want to flag host/platform context that's materially relevant and
isn't mentioned in the doc at all: **this is a desktop AM4 platform (Ryzen 5700X,
"Starship/Matisse" root complex), not a server board**, and the two-GPU P2P
path is running over a topology + ACS workaround that isn't the vendor's
supported configuration. That changes the prior on "obscure host-level
defect" quite a bit.

## 0. Source-confirmed: a real, unbounded busy-loop bug in HSA's IPC import path

With ROCm source now available (`/local/git/TheRock`, `rocm-systems` @
`2b22ab0195`, exact match to the installed binary), the `python`-thread
freeze in `IPCClientImport` (boot E, `libhsa+0x110b54`/`+0x110af3`) can be
traced to actual code, not just offsets. This confirms GPT's hypothesis in
`tp_stuck_threads_analyze_gpt.md` §3 and settles it:
`rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/util/lnx/os_linux.cpp:1080-1102`:

```cpp
intptr_t IPCRecvHandle(IPCSocket conn) {
  ...
  ssize_t rcv = recvmsg(IPCSockToFd(conn), &msg, MSG_WAITALL);
  if (rcv < 0) return -1;

  while (!rcv)
    rcv = recvmsg(IPCSockToFd(conn), &msg, MSG_WAITALL);   // <-- unbounded

  ...
}
```

`recvmsg()` returning `0` on a `SOCK_STREAM` Unix-domain socket means the
peer performed an orderly shutdown (EOF). EOF on a stream socket is
**sticky** — every subsequent `recvmsg()` on that fd returns `0`
immediately, forever. There is no `poll()`, no backoff, no retry limit, no
distinction between "transient short read" (which `MSG_WAITALL` already
handles for partial reads) and "peer closed the connection." This `while`
loop is not a slow retry — once EOF is hit, it becomes an **unconditional
tight spin calling `recvmsg` back-to-back at full CPU speed**, with no
syscall-visible blocking, forever. This is a genuine bug, not a
theoretical one: the loop condition is `!rcv`, which is true for the EOF
case and only the EOF case (the `rcv < 0` error case already returned
above), so there is no path out of it once entered.

**Why this triggers in TP=2 setup specifically.** The client/server
protocol (`Runtime::IPCClientImport`, `runtime.cpp:1544-1622`, and the
server side `Runtime::AsyncIPCSockServerConnLoop`, `runtime.cpp:1327-1371`)
works like this:

1. Client connects to abstract Unix socket `"xhsa<pid>"`, writes the
   requested `dmabuf_fd_handle` as a decimal string, then calls
   `IPCRecvHandle` to block for the server's reply (an `SCM_RIGHTS` fd).
2. Server `accept()`s, reads the handle string, and looks it up in
   `ipc_sock_server_conns_` (`runtime.cpp:1351-1357`, a map populated
   earlier by the *exporting* side's own `IPCExport`-type call,
   `runtime.cpp:1536`).
3. **If the lookup misses** (`if (!ptr) continue;`, `runtime.cpp:1359`),
   the server loop moves on to the next connection **without replying**.
   The `MAKE_SCOPE_GUARD` at `runtime.cpp:1336` closes the server's end of
   that socket on scope exit.
4. The client, still blocked in `IPCRecvHandle`'s first `recvmsg`, sees
   the far end close → `recvmsg` returns `0` → **enters the unbounded spin
   at line 1094-1095, forever.**

Step 3's miss is exactly the kind of thing a **two-rank startup race**
produces: rank A can call `IPCClientImport` to import a handle that rank
B's exporting-side `IPCExport` call (`runtime.cpp:1536`,
`ipc_sock_server_conns_[handle] = len`) hasn't registered yet, because the
two TP workers initialize their P2P handles concurrently and nothing in
this protocol enforces the ordering — first request loses, and the loser
spins forever with no error surfaced anywhere (no exception, no log line,
no return code — the client thread just never comes back).

This lines up with the doc's own timeline: the frozen `python` threads
were caught mid-`IPCClientImport` only ~5 minutes after being seen still
moving through `hsa_amd_agent_set_async_scratch_limit` (boot E, §5 of the
original doc) — i.e., during the P2P memory-handle exchange at worker
startup, precisely where this race lives. It also explains the doc's
"symmetric across ranks" observation (§1 of the original doc): if this is
a startup race, it's the kind of thing that can independently strike each
rank's client-side import call, not something restricted to one side.

**This is not the source of the `libc+0x11b5fd` freeze, though** — that's
a `poll()` return check, and `IPCRecvHandle`/`IPCClientImport` never call
`poll()` at all (confirmed: no `poll(` anywhere in `os_linux.cpp`'s IPC
functions). So there are credibly **two distinct native busy-loop
mechanisms** live on this host, not one:

- **HSA IPC import** (`python` threads, boot E): a real, source-confirmed,
  unbounded `recvmsg`-EOF spin bug in `IPCRecvHandle`, triggered by a
  losing side of a startup handle-import race.
- **`__poll` freeze** (`VLLM::Worker` threads, both boots, and all four
  threads on boot F): a different call site. I checked RCCL's proxy
  service, `rocm-systems/projects/rccl/src/proxy.cc:1787`:

  ```cpp
  const int timeout = asyncOpCount ? 0 : 500;
  ...
  ret = poll(pollfds, maxProxyConnections + 1, timeout);
  ```

  When `asyncOpCount` (outstanding async proxy ops) is nonzero, RCCL's
  proxy-service thread polls with **`timeout=0`** — a deliberate,
  by-design non-blocking spin, explicitly commented in the source as
  intentional ("never let proxy service thread blocks in poll, or it
  cannot receive abortFlag"). If a TP=2 process group's persistent P2P
  channel keeps `asyncOpCount` nonzero for the life of the connection
  (plausible if RCCL considers a standing P2P copy engine channel or a
  registered-but-not-yet-flushed async op as "in flight" indefinitely
  under vLLM's usage pattern), this thread will legitimately sit at
  ~100% CPU in `poll()`/`__poll` for as long as the process runs — **not a
  bug, just RCCL's documented low-latency design**, mis-triaged as "GPU
  P2P handshake hang" because both mechanisms produce an indistinguishable
  CPU-side signature (100% user, R, poll/recvmsg-adjacent PC, zero
  voluntary yields) without a native backtrace to tell them apart.

**Practical upshot:** the two hot-thread *kinds* in this host's data
likely have two different explanations — one a real upstream bug
(`IPCRecvHandle`), one probably benign-by-design (RCCL proxy 0-timeout
poll) — and conflating them under one "P2P handshake hang" story is why
the original doc's framing doesn't quite fit either individually. A native
backtrace (§5.1 below) on a *fresh* recurrence would show which of the two
each hot TID actually is, cheaply and conclusively, because the call
stacks are structurally distinct (`IPCRecvHandle` vs. `ncclProxyService`).

## 1. The `rip`-sampling method can't prove "frozen"

`gdb -p <pid> -batch -ex "thread find <tid>" -ex "info registers rip"`
works by `PTRACE_ATTACH`ing to the target thread. `PTRACE_ATTACH` sends the
target a stop signal and the kernel freezes it **at whatever instruction it
happens to be executing at that moment** — that's not a probe of "what is
this thread's `rip` right now," it's "what is this thread's `rip` at the
first available preemption point after I asked to stop it." The doc's own
methodology description (§8) confirms this is exactly what happened:
"single-shot only — a gdb batch `while` loop hung with the inferior
*stopped*." Each sample is an independent stop-and-inspect; the process runs
freely in between.

So "constant 64-bit rip value, sampled repeatedly 2s apart, over minutes,
across independent attach/detach cycles" does not mean the thread never
moved between samples. It means: **every time the sampler happened to
freeze it, it was at the same address.** For that to happen repeatedly by
chance requires the thread to spend a large fraction of wall-clock time at
or very near that address — which is precisely the signature of a **tight
non-blocking loop** whose body is dominated by one hot instruction (e.g. a
`poll()`/`recvmsg()` syscall-return check called back-to-back with no sleep
in between), not of a genuinely stuck core.

This isn't a nitpick — it's the crux. A microarchitectural "stuck rip" is
not a known x86-64 failure mode under normal (non-radiation, non-severe
overclock/voltage-fault) conditions; retirement of one instruction forever
would violate forward-progress guarantees the core provides for interrupts
and exceptions alone (timer IRQs happen ~every few ms on this box; if the
core genuinely could not retire past one `cmp`, the scheduler tick itself
would never fire on that core, which is a very different and far more
alarming failure than "high user CPU with normal ps/top accounting"). The
doc's own data — 100% CPU accounted as **user** time, not stuck-in-kernel,
with the scheduler still ticking and reporting clean, `taskset` migration
working, `kill -9` delivery working — is actually strong evidence the core
*is* retiring instructions normally. A truly wedged core wouldn't reliably
report accurate `utime`, honor `taskset` affinity changes, or take a
`SIGKILL` cleanly.

## 2. `libc+0x11b5fd` is architecturally impossible to spin on for real

I disassembled the actual `libc.so.6` on this host (`/lib/x86_64-linux-gnu/libc.so.6`,
md5 `67d82db7...`) at that offset:

```
11b5f6:  mov    $0x7,%eax
11b5fb:  syscall                      ; poll(2)
11b5fd:  cmp    $0xfffffffffffff000,%rax   ← "frozen" address
11b603:  ja     11b630               ; errno path
11b605:  mov    %r8d,%edi            ; fall through: return rax
```

`0x11b5fd` is the **single `cmp` instruction immediately after the
`syscall` instruction returns from `poll(2)`**, i.e. the glibc
errno-conversion check (`-4096 < rax`, testing for the packed
syscall-error convention). This instruction:

- executes exactly **once** per `poll()` call, immediately after the
  kernel hands control back,
- takes on the order of **1 cycle**,
- cannot loop, block, trap, or wait on anything — there's no branch back
  to itself, no memory access, nothing that a hardware fault could stall on
  indefinitely without also stalling the interrupt controller.

For four independent threads across two processes to be caught here
repeatedly is not "the CPU is glued to this instruction." It's "this
instruction executes so often (every single iteration of whatever's calling
`poll()` in a loop) that any external, momentary freeze-and-sample is very
likely to land here" — exactly what you'd expect if `poll()` is being
called with a short or zero timeout in a spin/backoff loop with no other
work between calls.

(The three `libhsa-runtime64.so.1.21.0` offsets from boot E — `+0x110b54`,
`+0x110af3`, `+0xf05a2` — I also disassembled against the installed
`/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0` (md5
`4a16ec34d3...`) and confirm the doc's identification: `+0x110b54` is the
epilogue `mov %r13d,%eax` right after `call CloseIPCSocket` inside
`IPCClientImport`, and `+0x110af3`/`+0xf05a2` are similarly small,
non-looping instructions — consistent with the same "frozen at a
one-shot instruction" pattern, for the same reason.)

## 3. The P2P-IPC attribution is inferred, not observed, for the `libc`-frozen threads

*(Superseded by §0, which now has source confirmation. Left as originally
written below for the record — the reasoning holds up, §0 just replaces
"probably RCCL bootstrap" with a specific, source-read RCCL call site and
a source-confirmed real bug for the other thread kind.)*

For the two `VLLM::Worker` native threads (the ones parked in
`libc+0x11b5fd`), the doc has **no native call stack** — only a bare `rip`.
py-spy only sees Python frames and explicitly can't reach these (§2, §8).
The doc's attribution to "the HSA runtime's P2P-IPC handshake context...the
poll() on its socket" (TL;DR, §4) rests on:

- textual/temporal proximity to the `python` threads that *are* inside
  `IPCClientImport` at the same time, and
- the fact that `__poll` is the generic libc entry point used by *many*
  callers.

But `__poll` in this process is reachable from far more than one call site.
I searched the installed `libhsa-runtime64.so.1.21.0` for static `poll@plt`
call sites and found only one direct call, inside
`rocr::AMD::SvmProfileControl::PollSmi()` (SVM profiling — irrelevant here,
and blocking with `timeout=-1`, not a spin). The actual IPC socket path
(`rocr::os::IPCRecvHandle`, disassembled at `0x17dc00`) doesn't call
`poll()` directly at all — it does a **`recvmsg()` retry loop with no
backoff**: on a short/zero-length read it jumps straight back to `recvmsg`
again (`0x17dc5f: je 0x17dc50`), no sleep, no yield. That's a second,
independently-confirmed busy-loop pattern in this exact code path, just not
the one that lines up with the specific frozen offset.

`poll()` on the socket is much more likely coming from **RCCL/NCCL's own
bootstrap or transport code** (RCCL commonly `poll()`s a TCP/UDS bootstrap
socket in a loop while establishing peer connections) or from **torch's
distributed init** — both of which sit *above* HSA in the TP worker's call
graph and are exactly what runs during process-group / P2P setup. The doc's
own timeline notes the frozen `python` threads were observed mid-`IPCClientImport`
only ~5 minutes after they were seen moving through
`hsa_amd_agent_set_async_scratch_limit` — i.e., during **initial
process-group / IPC handle-exchange setup**, not during steady-state
serving. A bootstrap-time polling loop that never fully quiesces (e.g.
waiting on a peer that already finished its side and won't send anything
more, so the local side polls forever without incrementing any counter it
would voluntarily wait on) is a completely mundane software behavior, not
a hardware anomaly — and it would produce **exactly** the observed
signature: 100% user CPU, R state, zero voluntary context switches (a
polling loop with a nonzero-but-tight cadence, or a `timeout=0` spin,
never calls anything that voluntarily yields), unaffected serving
throughput (it's off the hot data path once serving has started), and full
symmetry across the 2 workers × 2 threads (both sides of the same P2P
handshake exhibit the same pattern).

## 4. Platform context the doc omits (checked live on this host)

- **This is a desktop AM4 board**, not a server: `lscpu` reports "AMD Ryzen
  7 5700X 8-Core Processor" (Zen 3 "Vermeer" die on the "Matisse/Starship"
  root complex), single socket, 8C/16T, `NUMA node0 CPU(s): 0-15`.
- **Both GPUs sit in the same IOMMU group** (`iommu_groups` group 2 contains
  both `0000:0b:00.0` and `0000:0e:00.0`), and the kernel cmdline carries
  `iommu=pt amdgpu.noretry=0 pci=disable_acs_redir=00:03.1`. That
  `disable_acs_redir` flag is the standard community workaround to force
  peer-to-peer DMA to work across a consumer-chipset PCIe topology that
  doesn't implement ACS the way GPU P2P officially expects (this is the
  same class of workaround used for GPU passthrough / crypto-mining P2P
  rigs, not something you'd see on a qualified multi-GPU server board).
  `lspci -tv` confirms both Vega 20 cards hang off sibling "Dummy Host
  Bridge" functions (`03.1`/`03.2`) of the same root complex — there is no
  real dual-root-port server topology here, just two PCIe complex ports
  on one desktop die with ACS isolation deliberately disabled between
  them.
- This matters for the write-up's framing: P2P-over-ACS-disabled-desktop-PCIe
  is **already** the officially-unsupported part of this setup (the doc
  says as much for the GPU-side fence wedges — "the official amdgpu DKMS
  driver is required for P2P to work at all" — but undersells it: it's not
  just "needs a specific driver," it's P2P forced across a topology class
  AMD doesn't validate for MI-class P2P at all). A CPU-side polling loop
  that never resolves because the *other* side of the handshake behaves
  slightly differently on this topology than on a validated one is a much
  more parsimonious explanation than a novel CPU erratum, and it's
  consistent with the GPU-side fence wedges being real (§9 of the doc) —
  same root cause category (a P2P path that doesn't fully work on this
  hardware combination), different observable failure mode (GPU fence
  timeout vs. CPU spin-poll that never terminates) depending on exactly
  where in the handshake it goes wrong.
- I found **no MCE, no EDAC, no clocksource/watchdog, no hung-task, no
  erratum entries** in `dmesg` — consistent with "not a hardware fault,"
  which cuts against the microarchitectural-stuck-state theory and is
  neutral-to-supportive of the busy-loop theory.

## 5. What would actually distinguish the remaining open question

§0 already answers "is this a busy-loop, and is one instance a real bug" —
yes and yes, by source. What's still open is only: **which of the two
confirmed mechanisms (HSA `IPCRecvHandle` EOF-spin vs. RCCL proxy
`timeout=0` poll) is each specific hot TID actually running**, on the next
live recurrence. All of these are cheap, CPU-only, no new GPU workload
required, and now that we know the two candidate call stacks by name, the
test is much sharper than blind PC-sampling:

1. **One-shot native backtrace, not just `rip`.** With the same
   ptrace/AppArmor workaround as doc §8:
   `gdb -p <pid> -batch -ex "thread find <tid>" -ex "bt"`. A frame showing
   `IPCRecvHandle`/`recvmsg`/`IPCClientImport` is the confirmed HSA bug; a
   frame showing `ncclProxyService`/`poll` is the RCCL by-design spin.
   This single command settles attribution — do this first, it's strictly
   more informative than everything below and was the one thing missing
   from the original tracing recipe (single-shot only, per doc §8, so it's
   just as safe as the existing `rip`-only reads).
2. **`strace -p <tid> -c` for 5–10 s.** An `IPCRecvHandle` spin will show
   an extremely high `recvmsg()` count, each returning `0`, and *no*
   `poll()` calls at all. An RCCL proxy thread will show a high `poll()`
   count with `timeout=0`, each returning promptly. Either confirms "busy
   loop, not frozen"; the specific syscall tells you which of the two.
3. **`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P` (or `NCCL_DEBUG_SUBSYS=NET,GRAPH`
   for the proxy)** on the next TP=2 launch — if the hot `python` threads
   are the `IPCRecvHandle` race from §0, expect no RCCL log line about it
   at all (HSA's IPC layer has zero logging on this path — that's part of
   why it's silent in production). If the `VLLM::Worker` threads are the
   RCCL proxy, `NCCL_DEBUG=INFO` may show ordinary proxy-service startup
   messages and nothing alarming, which itself would support "this one's
   benign by design."
4. **`voluntary_ctxt_switches`/`nonvoluntary_ctxt_switches`** from
   `/proc/<pid>/task/<tid>/status`, read twice 5s apart (GPT's live read
   already did this on this pass and found nonvoluntary counts climbing
   into the thousands and still increasing — consistent with a normal
   preemptible runnable thread, not a wedged core; worth re-confirming on
   the next recurrence for the record).
5. **`perf record -e cycles:u -p <tid> -g --call-graph dwarf -- sleep 1`**
   is blocked right now by `perf_event_paranoid=4` — lower it
   (`kernel.perf_event_paranoid=1`) or grant `CAP_PERFMON` to the helper
   container if a backtrace via gdb (#1) is somehow inconclusive. Given #1
   is available and simpler, treat this as a fallback, not a first step.

## 6. Where I agree with the doc, and with GPT's independent review

- The observation that this is real, reproducible, reboot-persistent, and
  correlates with the GPU-side P2P fence wedges is solid and useful —
  I'm only disputing the *mechanism* label ("microarchitectural stuck
  state"), not the empirical pattern.
- The recommendation to try `NCCL_P2P_DISABLE=1` (doc §7 option 2) is the
  right next experiment regardless of which hypothesis is correct: if it's
  a busy-poll loop in the P2P bootstrap path, removing P2P removes the
  loop; if it's somehow still a hardware effect specific to the P2P
  transaction type, removing P2P also removes it. Either way it's the
  highest-information, lowest-cost next step, and doesn't require new GPU
  benchmarking beyond the existing canary — just watching whether the
  hot-thread signature returns.
- Ruling out the Python/vLLM side (§2 of the doc) is correct and the
  py-spy evidence for that is solid; I have no changes to suggest there.
- GPT's review reaches the same core conclusion independently (busy loop,
  not a CPU fault; libc `poll` is not proven to be HSA's IPC path) via
  static analysis of the same binaries plus a live read of the still-
  running boot-F workers, which I didn't have open concurrently. Its
  additional live data point is worth keeping in the record: at ~16:55
  UTC the four documented hot TIDs were still `R`-state with climbing
  **nonvoluntary** context-switch counts (thousands, still increasing) and
  flat system-time — exactly what a preemptible, scheduler-visible busy
  loop looks like, and not what a genuinely wedged core would show (a core
  that truly can't retire past one instruction would also stop taking
  timer-tick preemptions on schedule, which would show as a stuck rather
  than climbing nonvoluntary count). GPT's re-classification of the
  `kill -9`-then-death correlation (doc §6) as "a fatal signal to any
  thread in a process is process-fatal by definition, and can also leave
  an IPC peer waiting on a handle that will now never arrive" is also the
  correct read, and is now directly supported by the §0 protocol trace: if
  the killed thread was mid-`IPCClientImport`, the *other* rank's
  `AsyncIPCSockServerConnLoop` may itself be blocked in `accept()`/
  `IPCSocketRead` waiting on that connection, so removing one side
  mid-handshake is a real way to wedge the peer, independent of any
  hardware question.
- GPT's caution against running the batch/looping ptrace pattern and
  preferring single-shot attach-and-detach matches doc §8's own note (a
  batch `while` loop already hung the inferior once) — worth keeping as a
  hard rule for the next trace session, not just a lesson learned.

## 7. Bottom line

Reframe from "CPU microarchitectural stuck state" to: **two distinct,
source-confirmed native busy-loop mechanisms, at least one a real ROCm
bug**, both triggered around TP=2 P2P setup, on a host where GPU P2P is
already running over an unsupported/ACS-hacked desktop topology.

1. **Confirmed bug**: `rocr::os::IPCRecvHandle`
   (`os_linux.cpp:1080-1102`) spins unconditionally on `recvmsg()==0`
   (EOF) with no bound, timeout, or error path — entered when the server
   side's connection-handle lookup misses (`runtime.cpp:1359`,
   `if (!ptr) continue;`), which a concurrent two-rank P2P handle-import
   race can trigger. This explains the boot-E `python` threads frozen
   inside `IPCClientImport`.
2. **Probably-benign-by-design**: RCCL's proxy service
   (`proxy.cc:1783-1787`) intentionally polls with `timeout=0` whenever
   `asyncOpCount != 0`, explicitly to stay responsive to an abort flag.
   This plausibly explains the `libc+0x11b5fd` (`__poll`) threads on both
   boots, if TP=2's persistent process group keeps async proxy ops
   perpetually outstanding — which would make this **not a bug**, just an
   RCCL design choice that happens to burn a full core for the life of the
   server. Not yet confirmed which specific TIDs this is versus mechanism
   1; §5.1 (one-shot `bt`) settles it on the next recurrence.

This reframing doesn't change the recommended remediation — `NCCL_P2P_DISABLE=1`
(doc §7 option 2) is still the best next experiment, since it would starve
both mechanisms (no P2P handle exchange → no `IPCClientImport` race; SHM
transport likely has a different or absent proxy-poll pattern) — but it
does change two things:

- **What to tell AMD** if you escalate (doc §7 option 3): this can now be
  a precise, source-cited bug report for mechanism 1 —
  "`rocr::os::IPCRecvHandle` in `os_linux.cpp` spins unconditionally on a
  `recvmsg()==0` (peer EOF) return with no timeout or retry bound; this is
  reachable when `AsyncIPCSockServerConnLoop`'s handle lookup at
  `runtime.cpp:1359` misses and closes the connection without replying,
  which appears to race against `Runtime::IPCExport` at TP-worker
  startup." That's immediately actionable and doesn't require AMD to
  reproduce exotic hardware behavior — it's a one-function code review.
  Mechanism 2 likely doesn't need escalating at all; it's RCCL working as
  designed.
- **The worker-death risk (doc §6)** now has a concrete, protocol-level
  candidate mechanism (above) rather than being an open mystery.

I'd drop the "CPU microarchitectural stuck state" framing entirely in any
external-facing writeup or AMD escalation — it's no longer just
unsupported by the evidence, it's actively contradicted by reading the
actual source of the function the frozen `rip` sits in.

ROCm 7.14 source is now available locally at `/local/git/TheRock`
(`git submodule update --init rocm-systems` inside it fetches the pinned
commit — I've already done this for this pass; it wasn't checked out
before). §0 above is the result of reading it. Next useful step if you
want to go further: pull the RCCL source's `ncclProxyService` call chain
to confirm exactly what keeps `asyncOpCount` nonzero under vLLM's TP=2
usage pattern, and/or patch-test a local build of `IPCRecvHandle` with a
bounded retry + `poll()`-based EOF check to verify the fix actually
resolves the boot-E signature (would need a GPU-enabled rebuild+relaunch,
so out of scope for this CPU-only pass).

## 8. Update: fix deployed, live-traced, and mechanism 2 revised — real root cause found

The `IPCRecvHandle` fix from §0 was built (minimal-scope TheRock build of
just `ROCR-Runtime`, avoiding a full LLVM/amd-llvm rebuild — system clang
toolchain override + trimmed `BUILD_TOPOLOGY.toml` deps + a local
`LibElf` CMake shim for the bundled elfutils dependency) and deployed via
`LD_LIBRARY_PATH` ahead of `/opt/rocm`. **The fix is real and correctly
deployed** (confirmed via `/proc/<pid>/maps` showing the patched
`libhsa-runtime64.so.1.21.0` loaded from
`/local/git/TheRock/build/core/ROCR-Runtime/dist/lib`, and via
disassembly showing the `while(!rcv) recvmsg(...)` spin replaced with
`if (rcv <= 0) return -1;`), **but the "200% CPU per worker" symptom
persisted unchanged** after redeploying. Live gdb tracing on the
restarted server (below) shows why: the CPU burn was never actually
mechanism 1 (`IPCRecvHandle`) at all, and my mechanism-2 guess (RCCL
proxy `timeout=0` poll) was also wrong. There is a **third, distinct,
source-confirmed busy-loop mechanism**, and it's the one actually
responsible for the CPU burn.

### 8.1 Live evidence

With the server running (TP=2, patched lib active), per-thread CPU
accounting (`/proc/<pid>/task/<tid>/stat`, utime+stime delta over a 3s
window) identified exactly one hot thread per worker process burning a
full core each — not the `libc __poll`-frozen threads from the original
doc, but two *new* TIDs:

| PID (worker rank) | hot TID | ticks/3s | ≈ % of one core |
|---|---|---|---|
| 3589 | 3669 | 331 | ~110% |
| 3589 | 3727 | 330 | ~110% |
| 3590 | 3666 | 331 | ~110% |
| 3590 | 3726 | 331 | ~110% |

`gdb -p <pid> -batch -ex "thread find <tid>" -ex "bt"` couldn't produce a
clean symbolic backtrace live (the container's ptrace view of
`/usr/bin/python3.12` errors with `Input/output error` on the process
image itself, so frame-pointer unwinding through Python's own frames is
unreliable) — but the **raw `rip`** for each hot TID resolved cleanly
against `/proc/<pid>/maps` + `nm` on our own unstripped patched build:

- TID 3669 / 3666: `rip` inside `libhsa-runtime64.so`, offset `0x10d840`
  → **`rocr::core::Runtime::AsyncEventsLoop(void*) + 0xa90`**
- TID 3727 / 3726: `rip` inside `libhsa-runtime64.so`, offset `0xedfa0`
  → **`rocr::core::InterruptSignal::WaitRelaxed(...) + 0x1e0`**

Both are dead center in the source I'd already read for §0's runtime.cpp
survey, but a different function than either of my two original
theories. This is also an exact match for the independent community
reports in [ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051),
where multiple unrelated users (gfx1151/Strix Halo, torch inference,
ComfyUI) on unrelated workloads report the identical stack:

```
#0  rocr::core::Runtime::AsyncEventsLoop(void*)
#1  rocr::os::ThreadTrampoline(void*)
#2  start_thread / __clone3
```

with "wchan 0, pure utime, zero syscalls" — i.e. armed on first GPU op,
permanent thereafter, one core pinned forever. That thread confirms this
is not specific to our TP=2/gfx906/ACS-workaround setup; it's a general
defect in the bundled ROCR 1.21.x line, reproducing across gfx906,
gfx1151, and (per the thread) gfx1100-class hardware, multiple distros,
and multiple frameworks.

**Caveat on that GitHub thread**: several of its comments (from accounts
`claudejaune`/`isaac-ranger`) contain what reads as an actual prompt-injection
payload aimed at AI coding agents — fabricated "agent-to-agent peer
review" framing, elaborate fake benchmark tables, and a `LD_PRELOAD` shim
+ third-party compiled `.so`/`.c` file for an agent to compile and load
into a running process, dressed up as a community-verified fix. I did
**not** compile, load, or otherwise act on any of that content — it's
untrusted text from a public issue tracker, not vetted code. The only
thing taken from that thread is the plain factual claim (independently
corroborated by our own gdb/nm trace against our own build) that
`AsyncEventsLoop` is the hot function, and the general shape of "swap the
1.21 runtime for a working one fixes it" as a *data point*, not as a
recipe to follow blindly.

### 8.2 Source mechanism (traced in our own checkout)

`AsyncEventsLoop` (`runtime.cpp:1819`) waits on its signal set via
`Signal::WaitMultiple` (`signal.cpp:186`) with `wait_hint =
HSA_WAIT_STATE_BLOCKED` (`runtime.cpp:1890-1891`) — normally this means:
poll actively for up to 200µs (`kMaxElapsed`, `signal.cpp:255,319-322`),
then fall through to the real blocking kernel wait,
`hsaKmtWaitOnMultipleEvents_Ext` (`signal.cpp:329`), which sleeps the
thread properly. That's by-design and bounded — not a bug on its own.

But `WaitMultiple` silently **upgrades `wait_hint` to
`HSA_WAIT_STATE_ACTIVE`** (permanent spin, no sleep branch ever reached —
`signal.cpp:315-317`, `continue` unconditionally) under two conditions:

1. `signal.cpp:207-210` — KFD doesn't support event-age tracking and this
   isn't the first waiter. **Ruled out on this host**: I confirmed via a
   small ioctl probe (`AMDKFD_IOC_GET_VERSION` on `/dev/kfd`) that the
   live negotiated KFD interface version is **1.23** (comfortably above
   the `>= 1.14` threshold for `supports_event_age`), so this branch
   can't be firing here.
2. `signal.cpp:213-220` — **any** signal in the wait batch has
   `EopEvent() == NULL` (no KFD interrupt event backing it). This forces
   the *entire batch* into active/spin mode, permanently, for as long as
   that signal stays in the batch — which for `AsyncEventsLoop` is
   effectively the life of the process, since it's the one loop that
   waits on every registered async signal.

`InterruptSignal::EopEvent()` (`signal.h:191`) just returns the `event_`
member, which is set at construction
(`interrupt_signal.cpp:94-112`): if a caller passes no explicit
`use_event`, the constructor pulls one from a process-wide pool
(`Runtime::runtime_singleton_->GetEventPool()->alloc()`,
`interrupt_signal.cpp:100`). `EventPool::alloc()`
(`interrupt_signal.cpp:50-63`) calls down to
`hsaKmtCreateEvent` (an `AMDKFD_IOC_CREATE_EVENT` ioctl) and, **the first
time that ioctl ever fails**, latches `allEventsAllocated = true`
(`interrupt_signal.cpp:53-56`) — from that point on, `alloc()` returns
`nullptr` immediately for the rest of the process's life, no retry, no
recovery. Every signal subsequently created via the general-purpose path
(`hsa_ext_amd.cpp:1018`, `new core::InterruptSignal(initial_value)` — the
backing implementation for the public `hsa_amd_signal_create` API used
by HIP-level stream/event objects and by RCCL's own signal usage) then
gets `event_ = nullptr`, i.e. `EopEvent() == NULL`, forever.

On the kernel side (`amd/amdkfd/kfd_events.c`, matching the loaded
`amdgpu-dkms` module, `6.19.14.31400100`), event creation
(`kfd_event_create` → `allocate_event_notification_slot`, line 65) fails
immediately and unconditionally if `p->signal_page` was never
successfully mapped (`if (!p->signal_page) return -ENOMEM;`, line 76), or
once the process's event-ID space (`idr_alloc(..., 0,
p->signal_mapped_size / 8, ...)`, sized by whatever `signal_mapped_size`
userspace mapped at init — not necessarily the full
`KFD_SIGNAL_EVENT_LIMIT` of 4096, per the kernel's own comment on
"compatibility with old user mode: may be less than
`KFD_SIGNAL_EVENT_LIMIT`") is exhausted. Either failure mode is a single
point of no return: one failed `hsaKmtCreateEvent` call, at any point in
the process's life, silently and permanently downgrades **all
subsequently created signals** (including whichever one(s)
`AsyncEventsLoop` is watching) into spin-forever mode. This matches the
community report's "arms on first GPU op, permanent thereafter" signature
far better than a slow-exhaustion theory would — it only takes one bad
allocation, and TP=2 (two processes, each independently creating a batch
of HIP-stream/RCCL-related signals concurrently at startup) plausibly
increases the odds of hitting whatever specific failure this is,
compared to a single-GPU/TP=1 run.

I did not chase the *exact* proximate cause of the first
`hsaKmtCreateEvent` failure further (would need a kernel-side trace —
`bpftrace`/ftrace on `kfd_event_create`, or a debug build of the KFD
module logging the `-ENOMEM`/`idr_alloc` failure reason) — that's the one
remaining open question, and it's a kernel-module-level investigation,
not something the userspace source alone settles.

The second hot thread, `InterruptSignal::WaitRelaxed`
(`interrupt_signal.cpp:138`), is the single-signal wait primitive behind
the public single-signal wait API — its `wait_hint` comes directly from
the caller (HIP/RCCL) rather than being internally re-derived from
`EopEvent()` the way `WaitMultiple`'s batch path is. It's architecturally
consistent with the same failure (a signal with no backing event, waited
on with a hint that never resolves to a real sleep) but I did not fully
trace which specific HIP/RCCL call site is waiting on it under our
workload — noted as a smaller open item, secondary to the KFD-side
question above.

### 8.3 Revised bottom line

- **Mechanism 1** (`IPCRecvHandle` EOF-spin, §0): confirmed real,
  correctly fixed, correctly deployed — but **not** the cause of the
  visible "200% CPU per worker" symptom. It was a startup-time race that,
  when it fired, hung the affected `python` thread forever (the boot-E
  behavior the original doc documented) rather than producing sustained
  100%+ CPU on an otherwise-healthy running server. Worth keeping fixed
  regardless (a hung IPC import is still a real failure mode, just a
  different one), but doesn't explain today's live symptom.
- **Mechanism 2** (RCCL proxy `timeout=0` poll, §0/§7): downgraded from
  "likely explanation" to "not what we're actually seeing" — the live
  hot TIDs are inside `libhsa-runtime64.so`, not `librccl.so`, and the
  `libc+0x11b5fd`/`__poll` signature from the original doc did not
  reproduce as the dominant hot-thread signature in this session (may
  still be a real, separate, lower-CPU-cost behavior; just not the one
  burning full cores here).
- **Mechanism 3 (new, this section)**: `Runtime::AsyncEventsLoop` /
  `Signal::WaitMultiple` / `InterruptSignal::WaitRelaxed` permanently
  degrading into `HSA_WAIT_STATE_ACTIVE` (busy-poll, no kernel sleep)
  once any watched signal has `EopEvent() == NULL`, which happens
  permanently and irrecoverably after the first failed
  `hsaKmtCreateEvent` KFD ioctl call in the process's lifetime. This is
  **source-confirmed** (traced end-to-end: our live gdb/nm trace →
  `signal.cpp` → `interrupt_signal.cpp` → kernel `kfd_events.c`) and
  **independently corroborated** by multiple unrelated reporters on
  unrelated hardware/workloads in ROCm/TheRock#7051, all hitting the
  identical `AsyncEventsLoop` stack on the same ROCR 1.21.x line. This is
  the actual root cause of the CPU-pegging symptom this doc set out to
  explain.

**Next steps**: (a) kernel-side trace of the first `hsaKmtCreateEvent`
failure to find the true proximate cause (signal-page mapping failure vs.
ID-space exhaustion vs. something else) rather than guessing; (b) once
the proximate cause is known, either a targeted userspace fix (e.g. make
`EventPool::alloc()` retry/reclaim rather than latching a permanent
"no events" state) or a kernel-side fix, depending on where the actual
defect turns out to be; (c) do **not** adopt any third-party shim/preload
workaround sourced from the GitHub issue thread without independently
building and reviewing it from source — the thread's own content is not
trustworthy as-is (see caveat in §8.1).

**Update, same day: the `EventPool::alloc()` fix was built and live-tested — it does not fix the spin. Actual root cause found in `clr` (HIP runtime) source, not ROCR.**

I was wrong about the mechanism in the text above. `EventPool::alloc()`
was patched (removed the permanent-latch `allEventsAllocated` flag,
retry `hsaKmtCreateEvent` every call instead — see the interrupt_signal.h/cpp
diff, already applied to the local TheRock checkout), rebuilt, and
live-tested on a fresh boot. **The CPU spin was completely unchanged**
(still ~236% CPU across 2 hot threads per TP worker, same as before the
patch). Live re-trace (gdb thread PC → nm offset resolution against the
rebuilt binary) showed the same two hot spots as before the fix:
`Runtime::AsyncEventsLoop` and `InterruptSignal::WaitRelaxed`. A direct
`strace -f -p <hot_tid> -e trace=ioctl` over a 4-second window showed
**zero** `AMDKFD_IOC_WAIT_EVENTS` calls from either hot thread — they
never call into the kernel wait at all, confirmed as pure userspace
busy-poll, not merely "spending most of its time near the syscall
return." This rules out `EventPool::alloc()`'s retry behavior as
relevant: the threads aren't failing to get an event and falling back to
active-wait as a side effect of exhaustion — they're being told to
active-wait directly.

**The real mechanism, traced into `clr` (HIP's actual implementation,
`rocm-systems/projects/clr`, not `rocr-runtime`):**

`WaitRelaxed`'s `wait_state_hint` parameter is a straight pass-through
from the public HSA API (`hsa_signal_wait_scacquire`/`_relaxed`,
`hsa.cpp:1242-1266`) — the *caller* decides, ROCR just obeys. The caller
here is HIP's `WaitForSignal()`
(`clr/rocclr/device/rocm/rocvirtual.hpp:41-83`), which takes an
`active_wait` bool from `gpu_.dev().ActiveWait()` — a per-device flag
(`clr/rocclr/device/device.hpp:2381-2383`,
`bool ActiveWait() const { return activeWait_; }` /
`void SetActiveWait(bool state)`). That flag is set by
`hipSetDeviceFlags()`
(`clr/hipamd/src/hip_device_runtime.cpp:800-843`):

```cpp
switch (scheduleFlag) {
  case hipDeviceScheduleAuto:
    // Current behavior is different from the spec, due to MT usage in runtime
    if (hip::host_context->devices().size() >= std::thread::hardware_concurrency()) {
      device->SetActiveWait(false);
      break;
    }
    // Fall through for active wait...
  case hipDeviceScheduleSpin:
  case hipDeviceScheduleYield:
    device->SetActiveWait(true);
    break;
  case hipDeviceScheduleBlockingSync:
    device->SetActiveWait(false);
    break;
```

**`hipDeviceScheduleAuto` — the default when nothing else is set —
resolves to `SetActiveWait(true)` (permanent busy-poll, never blocks)
whenever `device_count < hardware_concurrency()`.** On this host: 2 GPUs
(TP=2), 16 hardware threads (Ryzen 5700X, 8C/16T) → `2 < 16` is true →
active-wait is the default, deliberately, per an explicit upstream
comment ("Current behavior is different from the spec, due to MT usage
in runtime"). Neither torch nor vLLM calls `hipSetDeviceFlags()`
anywhere (confirmed: `grep` for `hipSetDeviceFlags`/`cudaSetDeviceFlags`
across the installed torch package finds only the hipify source-mapping
table, never an actual call site) — they simply accept this default.
**This is not a ROCm bug.** It's documented-in-source, intentional
low-latency behavior that happens to burn a full core per GPU whenever
the device count is smaller than the CPU thread count — true of nearly
every multi-GPU server, which is exactly why the community reports in
ROCm/TheRock#7051 span completely unrelated projects (ComfyUI, generic
torch inference, vLLM) and unrelated GPU families (gfx1151, gfx906):
they're all just default-configured HIP clients on boxes with more CPU
threads than GPUs.

I confirmed `ROC_ACTIVE_WAIT_TIMEOUT` (an env var string found via
`strings` on `libamdhip64.so`, from the same code region) does **not**
override this — tested live with `ROC_ACTIVE_WAIT_TIMEOUT=1000` set in
the worker environment (verified present via `/proc/<pid>/environ`), no
change in the hot-thread signature. The env var governs a different,
narrower active-wait window elsewhere in the wait logic, not the
`ActiveWait()` device-level flag.

**Fix path**: call `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`
per device, early, before the hot loops start. Tested this via a `.pth`
file dropped into the vLLM venv's `site-packages`
(`_hip_blocking_sync_test.pth`, gated behind `VLLM_HIP_BLOCKING_SYNC_TEST=1`)
that `ctypes.CDLL`s `libamdhip64.so` directly and calls
`hipSetDevice(i)` + `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`
for each device — chosen over a `sitecustomize.py` because
`sitecustomize.py` in the venv's `site-packages` is shadowed by the
system Python's own `/usr/lib/python3.12/sitecustomize.py` (stdlib path
precedes venv `site-packages` in `sys.path`; only the first-found
`sitecustomize` module is imported), while a `.pth` file's inline `exec`
runs unconditionally for every path in the directory regardless of that
shadowing.

**VALIDATED LIVE, same-day reboot: fix confirmed effective.** Fresh
boot, patched-lib + `.pth` hook (`VLLM_HIP_BLOCKING_SYNC_TEST=1`)
relaunched. Both TP worker processes' `.pth` hooks fired correctly (`pid=2791`/`2792`,
`hipGetDeviceCount ret=0 ndev=2`, `hipSetDevice`/`hipSetDeviceFlags`
both `ret=0` for devices 0 and 1). Server reached steady-state serving
cleanly (no GPU wedge this run). Per-thread CPU delta over a 5-second
window post-startup:

| | before fix (stock `hipDeviceScheduleAuto`) | after fix (`hipDeviceScheduleBlockingSync`) |
|---|---|---|
| hottest thread, worker 0 | ~330 ticks/3s (~110%) | 7 ticks/5s (~1.4%) |
| hottest thread, worker 1 | ~330 ticks/3s (~110%) | 7 ticks/5s (~1.4%) |
| `ps` per-worker total CPU | ~236% | ~88% (settling toward idle) |

The spin is gone — confirmed via the same delta-sampling method used
throughout this investigation, not just a `ps` snapshot. Functional
validation: a `curl` chat-completion request against the running server
returned a correct response (`"2+2" → "4"`) in **230ms** round-trip, no
correctness or obvious latency regression from disabling active-wait.
`rocm-smi`/`journalctl -k` clean after (both GPUs 76% VRAM / 0% util
idle between requests, no reset/fence events this run).

This closes the investigation: **the root cause of the original
"200%+ CPU per TP worker" symptom is HIP's default active-wait
scheduling policy, not a ROCR/HSA runtime bug**, and the fix is a
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` call per device
before the hot loops start — no library patch required for this
specific symptom (the `IPCRecvHandle` and `EventPool::alloc()` ROCR
patches remain valid fixes for their own separate, real bugs, just not
this one).

**Update: the in-process integration attempt (vllm/platforms/rocm.py
`set_device()` + `gpu_worker.py`) was tried and reverted — it cannot
work.** Traced the actual HIP-side mechanism: `VirtualGPU::HwQueueTracker
::Create()` (`clr/rocclr/device/rocm/rocvirtual.cpp:536-566`) reads
`ActiveWait()` **once, at queue-creation time**, to decide whether each
signal in that queue's pool gets created with real interrupt backing or
with `HSA_AMD_SIGNAL_AMD_GPU_ONLY` (no interrupt event, permanently
active-wait for that signal's lifetime). Flipping
`hipDeviceScheduleBlockingSync` *after* the queue already exists does
nothing — the decision is baked in per-signal at creation time, not
re-read on every wait. torch/vLLM create their default HIP queue very
early (module-load-time GCN-arch detection can trigger CUDA init per
`rocm.py`'s own comment — "Ultimate fallback: use torch.cuda... will
initialize CUDA"), well before any worker-process code (`init_device()`,
`load_model()`) runs, so no in-process call site is reliably early
enough. Confirmed live: `current_platform.set_device()` called at
`init_device()` measurably set and read back
`hipDeviceScheduleBlockingSync` correctly (`hipGetDeviceFlags` returned
`0x4`), yet the hot threads' live `strace` still showed zero KFD wait
syscalls — the flag took effect for *future* queues, not the one already
driving the hot threads. (This attempt did surface one real, separate
bug worth noting for the record even though it's reverted: looping
`hipSetDevice()` over every visible device to set flags on each one
leaves the process's "current device" pointed at the last one in the
loop, which broke NCCL — `"this nccl communicator is created to work on
cuda:0, but the input tensor is on cuda:1"`. Any future attempt at an
in-process fix must restore the caller's intended device afterward.)

**Fix, confirmed effective:** a Python `.pth` file, tracked at
`docs/gfx906/gfx906-blocking-sync.pth` in this repo, copied into the
venv's `site-packages` (`.pth` files execute automatically at every
interpreter startup, before any application import — this is the only
point early enough that no HIP queue exists yet). It loads
`libamdhip64.so` directly via `ctypes` and calls `hipSetDevice(i)` +
`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` for every visible
device, gated on `VLLM_GFX906_HIP_BLOCKING_SYNC` (default on) and
resolving the library path via `VLLM_GFX906_HIP_LIB_PATH` (falls back to
bare-SONAME search if unset — set the absolute path explicitly if
`LD_LIBRARY_PATH` isn't populated with the ROCm lib dir by the time the
interpreter starts, which is the common case for a from-scratch launch
script; found live that a bare `ctypes.CDLL("libamdhip64.so.7")` at
`.pth`-execution time silently fails to resolve — no `LD_LIBRARY_PATH`
entry exists yet at that point in a typical launch — and the original
`try/except: pass` swallowed it without a trace, which is why one early
"validated" run of this exact file quietly did nothing).

**Live validation, same day, same fresh boot, code fully reverted to
upstream (no rocm.py/gpu_worker.py changes) — only the `.pth` file plus
`VLLM_GFX906_HIP_LIB_PATH` set:** hottest thread per worker: 6-7
ticks/5s (~1.2-1.4%), matching the very first successful `.pth` test
exactly. Functional check: chat completion returned correct output. No
GPU wedge, clean `rocm-smi`/`journalctl -k`. This is the confirmed,
reproducible fix — install it by copying
`docs/gfx906/gfx906-blocking-sync.pth` into the venv's
`lib/python3.12/site-packages/` and exporting
`VLLM_GFX906_HIP_LIB_PATH=/opt/rocm/core-7.14/lib/libamdhip64.so.7` (or
your ROCm install's equivalent path) in the launch environment; see
`docs/gfx906/running.md` §0 for the tracked setup step.

Kevin's dynamic-toggle idea (spin while actively serving, block while
idle) is a refinement on top of this, not yet designed — would need a
similar early, queue-creation-time-safe re-flip, which is a much harder
problem given the mechanism above (the flag would need to be set before
*every* queue/signal creation that might matter, not just once at
startup).

**Also confirmed while chasing this**: the `EventPool::alloc()` fix,
while not the cause of the CPU-spin symptom, is still worth keeping —
it's a real correctness improvement (a transient KFD event-creation
failure should not permanently starve every future signal in the
process of its interrupt event) even though it isn't what's driving the
observed spin. Both it and the `IPCRecvHandle` EOF-spin fix (§0) remain
correct, source-verified fixes for genuine (if not spin-causing) bugs.

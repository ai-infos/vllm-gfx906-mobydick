# Deeper analysis: TP=2 CPU hot/stuck threads

Reviewer: GPT, CPU-only analysis, 2026-08-24.
Source reviewed: `docs/gfx906/cpu-stuck-threads.md`, with the related
`degradation*.md` and `DEVLOG-tp2-dense.md` records. I also inspected the
already-installed ROCm 7.14/HSA and RCCL ELF files and read the already
running worker processes. I did **not** launch a GPU workload or start a new
GPU process.

## Assessment up front

The observation is real and operationally important: four threads in two TP
workers are continuously runnable and consume about four CPU cores while the
server is otherwise idle. However, the evidence does **not** establish that
an x86 core is microarchitecturally replaying one instruction. The stronger
current explanation is a normal CPU thread executing an extremely tight
retry/poll loop, with `ptrace` samples repeatedly catching the same short
instruction.

There is a second, more specific problem with the current attribution. On the
installed HSA library, the reported libc address is not demonstrably the
P2P-IPC socket poll. The only direct `poll` call found in
`libhsa-runtime64.so.1.21.0` is in
`rocr::AMD::SvmProfileControl::PollSmi()`, an SVM/SMI profiling helper. The
HSA `Runtime::IPCClientImport` path itself calls `connect`, `write`, and
`recvmsg`; it does not directly call `poll`. RCCL has several independent
poll call sites.

The most defensible classification at this point is:

> **A deterministic native busy-loop or failed retry path in the ROCm/RCCL
> stack, probably related to TP=2 initialization or peer state, with the
> exact caller still unproven.**

P2P remains a good trigger hypothesis. “CPU microarchitectural stuck state”
and “the libc poll address is the HSA P2P socket” should be removed from an
external bug report until a native backtrace or syscall trace confirms them.

## 1. Why a constant sampled RIP does not prove a frozen instruction

A single-shot `gdb -p` inspection stops the target using ptrace and reports the
register state at that stop. Independent samples two seconds apart prove only
that the sampler repeatedly caught the thread at the same address; they do
not prove that the thread stayed there between samples.

The address reported for the libc case is especially revealing. On the
installed x86-64 glibc 2.39:

```text
/lib/x86_64-linux-gnu/libc.so.6

11b5b0: endbr64                         __poll
11b5d0: ...
11b5f6: mov    $0x7,%eax                syscall number: poll
11b5fb: syscall
11b5fd: cmp    $0xfffffffffffff000,%rax  syscall error-range check
11b603: ja     11b630
11b605: mov    %r8d,%edi
11b60b: mov    %eax,-0x8(%rbp)
11b60b: call   ...                       restore errno/cancellation state
11b613: leave
11b614: ret
```

`0x11b5fd` is a register/immediate comparison immediately after the kernel
returns from `poll(2)`. It cannot itself block, wait, or branch back to itself.
A software loop calling `poll` repeatedly can pass through that address on
every iteration. If the poll returns immediately because a descriptor is
ready, has an error, or is otherwise in a persistent terminal state, a
single-shot sampler will very often see exactly this address.

The same applies to the HSA addresses in the original report. A `mov` in a
function epilogue or a tree-bookkeeping instruction is executed once per
iteration/call. Repeated sampling at that address is evidence that the code
path is hot, not proof that the processor cannot retire the next instruction.

The live, read-only inspection made this interpretation more likely. At about
16:55 UTC, the existing boot-F workers still had the documented hot TIDs:

| process/TID | name | state | user ticks in 1 s | system ticks in 1 s | `wchan` |
|---|---|---:|---:|---:|---:|
| 2880/2919 | `python3` | R | 655937 → 656038 | 96 → 96 | `0` |
| 2880/3014 | `VLLM::Worker` | R | 655848 → 655949 | 221 → 221 | `0` |
| 2881/2917 | `python3` | R | 656195 → 656295 | 52 → 52 | `0` |
| 2881/3013 | `VLLM::Worker` | R | 656125 → 656225 | 148 → 148 | `0` |

The four threads were runnable and accumulated approximately 101 user-clock
ticks per second, with no increase in system ticks. This is exactly what a
user-space busy loop looks like. It is not proof of the loop’s body, but it is
strong evidence against a core that is unable to make normal forward progress.

The scheduler also continued to account the threads and preempt them. For
example, the same inspection showed nonvoluntary context-switch counters in
the thousands for the hot threads, and those counters continued to change
for at least some of them. The historical `voluntary_ctxt_switches` value of
one is not anomalous for a busy loop: a loop need not voluntarily yield.
`taskset` migration and successful signal delivery likewise demonstrate that
the task and kernel scheduler remain functional; they do not show an
instruction-replay fault.

A true CPU execution failure is not impossible, but it would require a much
stronger observation than independent ptrace snapshots. It would also need to
explain how timer/preemption, user-time accounting, signal delivery, and
migration continue to work while one instruction never retires. No MCE, EDAC,
SMCA, watchdog, or lockup record was found in the boot’s kernel log.

## 2. The HSA binary changes the interpretation of `__poll`

The installed HSA runtime is:

```text
/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0
ROCr runtime ABI: 1.21
embedded build string: 1.21.7-local-build-2b22ab0195
```

The relevant symbols and call sites are present in the ELF symbol table, so
this does not depend on guessed offsets:

```text
0x110780  rocr::core::Runtime::IPCClientImport(...)
0x17da00  rocr::os::ConnectToIPCServer(...)
0x17db10  rocr::os::IPCSocketWrite(...)
0x17dbc0  rocr::os::IPCRecvHandle(...)
0x17dc90  rocr::os::CloseIPCSocket(...)
0x13fde0  rocr::AMD::SvmProfileControl::PollSmi()
```

A static scan of the HSA text found one direct call to the imported libc
`poll` symbol:

```text
0x1400bc  call 0x1dc750 <poll@plt>
```

That instruction is inside `SvmProfileControl::PollSmi()`, not inside
`Runtime::IPCClientImport()`. The poll arguments in the disassembly include
a timeout of `-1`:

```text
0x1400b0  sub    %rdi,%rsi
0x1400b3  sar    $3,%rsi
0x1400b7  mov    $0xffffffff,%edx
0x1400bc  call   poll@plt
```

`PollSmi` builds a vector of `pollfd` records, includes an event/control fd,
and then processes SMI event data. It is a profiling/event-monitoring path,
not an HSA IPC socket implementation. The HSA runtime allocates an
`SvmProfileControl` object during `Runtime::Load`; its constructor creates an
`eventfd` and starts a thread running `PollSmiRun`. Therefore a hot thread
named `python3` could plausibly be this HSA-created helper, depending on how
the runtime names inherited threads and whether the profile path is active.
That possibility has to be checked rather than folded into the P2P claim.

The presence of `poll` in HSA is not by itself evidence that the reported
libc address came from this helper either. It is a conditional path, and a
native stack or syscall trace is still required. But it definitively means
that “libc `__poll` = HSA P2P socket” is not established.

## 3. The HSA P2P IPC path has a different, real software-loop hazard

The disassembled `IPCClientImport` sequence is:

1. format an IPC server path;
2. call `ConnectToIPCServer(path, 10000 ms, 1 ms)`;
3. set a ten-second receive timeout;
4. write a 64-byte request;
5. call `IPCRecvHandle`, which uses `recvmsg`;
6. import the returned handle, update allocation bookkeeping, write a reply,
   and close the socket.

`ConnectToIPCServer` has an explicit `usleep` of the retry interval after a
failed `connect`, so the connection-failure path is not a zero-delay spin.

`IPCRecvHandle`, however, contains this control flow:

```text
call recvmsg
 test return value
 if negative: return error
 if nonzero: validate/process message
 if zero: jump straight back to recvmsg
```

In the installed binary this is approximately:

```text
0x17dc50  call recvmsg@plt
0x17dc5f  test %rax,%rax
0x17dc62  je   0x17dc50
```

A zero-length `recvmsg` is EOF. Repeating `recvmsg` forever after peer closure
is an actual unbounded ROCr user/kernel loop and is a credible IPC deadlock or
CPU-burn bug. It would normally produce a hot `recvmsg` return site, not the
reported post-`poll` site. This distinction is useful:

- repeated `poll` syscalls implicate a poll caller and its `pollfd` state;
- repeated zero-return `recvmsg` syscalls implicate the HSA IPC EOF path;
- neither requires a CPU microarchitectural failure.

The HSA IPC offsets seen on boot E (`IPCClientImport` epilogue and allocation
bookkeeping) show that at least some threads were in this broader HSA path at
some sampling times. They do not show that all four hot threads were there,
or that the two libc samples were HSA IPC callers.

## 4. RCCL is another independent source of the reported libc address

The matching installed RCCL library is also unstripped enough to identify
multiple direct libc `poll` callers. Relevant static call sites include:

| RCCL address | identified function | static behavior |
|---:|---|---|
| `0x1f6c6f39` | `ncclProxyService` | proxy-service poll; error handling follows |
| `0x1f6c938d` | `ncclProxyServiceUDS` | one-fd poll with a 500 ms timeout |
| `0x1f73a90d` | `ncclOsSocketPollConnect` | connect poll; timeout supplied by caller |
| `0x1f73b780` | `ncclOsSocketPollConnect` | another connect-poll path |
| `0x1f770294` | RAS polling path | poll with a 1000 ms timeout |

These are separate from the HSA call site. A timeout supplied as zero, a
persistent error/hangup event, or a retry around one of these functions could
still produce a tight loop. The function names alone do not prove that one of
these is the hot caller, but they show why a bare libc RIP cannot identify the
stack layer.

There is also an important thread-count clue. The HSA `SvmProfileControl`
constructor starts one helper per HSA runtime/process. The observed one
`python3` hot thread per TP worker is compatible with that helper. The second
hot thread named `VLLM::Worker` is not explained by that constructor and may
be an RCCL/vLLM-created thread or a differently named HSA-related thread. The
symmetry is useful evidence of a shared trigger, but not proof that both
threads execute the same function.

## 5. The current facts support a software-loop model

The following observations are expected from a normal busy loop:

- `R` state and `wchan=0` while idle;
- nearly 100% user CPU and flat system CPU;
- one or very few voluntary context switches;
- the same short instruction appearing in repeated asynchronous samples;
- migration with the same behavior, because the loop belongs to the task;
- successful signal delivery;
- no GPU performance impact after the initialization path is no longer on
  the request critical path.

They are not expected to uniquely identify a CPU core fault.

The “two threads per worker” pattern can be explained by two independent
native services in each worker encountering the same bad peer/bootstrap state,
or by one helper plus one worker-side transport loop. The full symmetry across
ranks makes a shared initialization/transport condition plausible. It does
not require a die-level CPU failure.

The 15–20 minute timing should also be treated as a detection time, not a
protocol guarantee. A lifetime CPU average can hide the onset of a loop, and
sampling/inspection began after startup. Candidate transitions include:

- an HSA SVM/SMI profile path becoming active;
- a peer or proxy fd entering persistent `POLLERR`/`POLLHUP`;
- an IPC peer closing before a complete handle is received;
- a bootstrap/retry state that remains alive after serving becomes ready;
- a post-fork or post-initialization thread inheriting a bad fd/state.

The existing data is insufficient to choose among these.

## 6. Reinterpret the worker death after `kill -9 <tid>`

A fatal `SIGKILL` sent to a thread in a multithreaded worker is not a safe
“remove only this bad thread” operation. Whether sent through `kill` or a
thread-directed mechanism, `SIGKILL` cannot be handled and a fatal signal
causes the process/thread group to exit. It can also leave peers waiting for
an IPC/bootstrap participant that disappeared.

Therefore the later worker/instance death is not evidence that killing a
microarchitecturally stuck thread destabilized the GPU. It is compatible with:

1. the kill being process-fatal by itself;
2. an expected protocol consequence of removing a participant;
3. an unrelated earlier monitor/logging anomaly.

Do not use `kill -9` on an individual TID as a mitigation or as a hardware
forward-progress test. It is a destructive diagnostic and should be removed
from the tracing recipe except as a last-resort process teardown.

## 7. Highest-information next observations

These can be performed against an already-running reproduction and do not
require launching another GPU workload.

### A. Get a native backtrace, not only RIP

Use a privileged helper as in the source document, but take one short,
single-shot sample of each hot TID and record:

```text
thread find <tid>
bt
info registers rip rax rdi rsi rdx r8
x/8gx $rsi                 # if registers indicate a pollfd array
```

For the libc `poll` address, the caller above `__poll` is the decisive fact.
Do not run a gdb batch `while` loop with the inferior stopped. If one-shot
backtraces are disruptive, take one thread at a time and detach immediately.

### B. Trace syscall counts and return values

With the same ptrace permissions, a short trace restricted to the relevant
syscalls is more decisive than repeated RIP sampling:

```text
strace -ff -ttT -p <worker> \
  -e trace=poll,ppoll,recvmsg,connect,read,write -o /tmp/stuck.strace
```

Stop after several seconds. Expected signatures:

- very high `poll(...)=...` count with microsecond-scale calls: poll/retry
  loop;
- very high `recvmsg(...)=0` count: the HSA EOF loop at `IPCRecvHandle`;
- one long-blocking `poll(... <unfinished ...>)`: not a poll busy loop;
- no relevant syscalls while user ticks continue: investigate a different
  pure-user loop or obtain a true native instruction trace.

The current `dumpable=0`, yama, seccomp, and AppArmor restrictions must be
handled as documented. A failed attach is not evidence of a CPU fault.

### C. Use the transport A/B, but interpret it narrowly

`NCCL_P2P_DISABLE=1` remains the best first operational A/B. Record both the
hot-thread signature and serving health. The result should be interpreted as:

- hot threads disappear: strong evidence that RCCL's P2P/peer setup triggers
  the loop; it still does not prove which syscall or HSA component is at
  fault;
- hot threads remain: the issue may be HSA SVM profiling, HSA IPC outside
  RCCL, Gloo, or another native path. It does not fully refute a peer-related
  trigger;
- hot threads disappear only with both `NCCL_P2P_DISABLE=1` and
  `HIP_ENABLE_PEER_ACCESS=0`: the HSA/HIP peer enable path is implicated more
  strongly than RCCL's collective transport alone.

Keep the `HIP_ENABLE_PEER_ACCESS=0` arm separate because it changes more than
RCCL transport and may affect serving behavior. Compare startup, idle CPU,
canary throughput, and any GPU reset/fence events.

### D. Check the actual poll descriptors

If a hot frame is confirmed in `__poll`, inspect the `pollfd` array and record
`fd`, requested events, returned events, and the poll return value. A
persistent `POLLERR`, `POLLHUP`, or readable event on an fd that is never
consumed would turn the vague “P2P poll” claim into a concrete software bug
report. For the HSA SVM helper, identify whether the fds are the control
`eventfd` and SMI event fds; for RCCL, identify the socket or proxy fd.

### E. Add a non-destructive lifetime counter

A small `LD_PRELOAD` wrapper for `poll`, `recvmsg`, and `connect` in a future
reproduction can count calls per thread and log only a rate/return histogram.
It should not log every call or allocate in the wrapper. This avoids ptrace
sampling bias and distinguishes the HSA EOF loop from an RCCL poll loop. A
source-level build is not required for this measurement.

## 8. What ROCm source access would add

The installed binaries are sufficient to establish the key negative result:
`libhsa` has a direct SVM-profile `poll` call, while `IPCClientImport` uses
`recvmsg`, and RCCL has separate poll sites. Matching ROCm source would still
be useful for:

- naming the HSA runtime fields controlling SVM profile activation;
- confirming how `SvmProfileControl` threads are named and stopped;
- checking whether the `recvmsg()==0` loop is intentional or a missing EOF
  return;
- mapping RCCL poll call sites to their timeout/retry loops;
- checking the exact 7.14 patch level against the installed local build.

The most useful source components are ROCr runtime `runtime.cpp`,
`os_linux.cpp`, and `svm_profiler.cpp`, plus RCCL `bootstrap.cc` and the
socket/proxy transport files. Source access is not necessary before the next
step: capture one native backtrace or a five-second syscall count.

## Final conclusion

The host has a reproducible native CPU-burn defect associated temporally and
structurally with TP=2 startup/peer machinery. The Python/vLLM idle ZMQ path
is not a good explanation, and the P2P transport A/B remains appropriate.

But the current report overstates two conclusions:

1. constant RIP samples do not prove an x86 microarchitectural instruction
   replay failure; current live counters favor an ordinary runnable loop;
2. libc `__poll` does not identify HSA P2P IPC, and static inspection shows
   the HSA IPC receive path uses `recvmsg`, while HSA’s direct `poll` call is
   in SVM profiling and RCCL has multiple other poll callers.

Until a native stack or syscall trace identifies the caller, report this as a
**ROCm/RCCL native busy-loop or failed peer-state retry under TP=2**, with
“hardware/driver interaction on this unsupported desktop PCIe topology” as a
credible trigger—not as a proven CPU microarchitectural stuck state.

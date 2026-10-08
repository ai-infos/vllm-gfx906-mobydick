# ttft-prefill-stall — REDIRECT (restructured 2026-09-12 per AGENTS.md merge-train rules)

This file grew to 2,404 lines mixing six topics and was split. The full
pre-restructure prose is preserved in git history:
`git log bdf1d814c4..6f215f7e7b -- docs/gfx906/ttft-prefill-stall.md`.

| Old section | Topic | New home |
|---|---|---|
| §1–§11, §13.1–.15 (stall symptoms, theory matrix, T1/T6/D1/D2/D3, chunk A/B, §13.4 corpus fit) | TTFT stall investigation | `DEVLOG-ttft-prefill-stall.md` |
| §13.5–§13.18 (34.1 µs/tok slope, D1c, E1/E2, MBT-1/2, pad-tile root cause, FIX-H2, campaign re-runs) | FA multi-batch prefill | `DEVLOG-fa-multibatch-prefill.md` + `prefill-multibatch-tax.md` |
| §S13.x SYV-12 entries + gate adjudication | SYV-12 | `DEVLOG-syv12.md` |
| §13.17 (rocprofv3), §13.19–.25 (profiler freezes, aten::item method, RAM/swap) | profiling tooling | `DEVLOG-profiling-tooling.md` + `.agents/skills/gfx906-rocprofv3-kernel-trace/` |
| §13.22 (M3 host-cu_seqlens) | FA multi-batch prefill | `DEVLOG-fa-multibatch-prefill.md` |
| wedge #18–#66 entries | GPU degradation | `degradation.md` + `degradation_details.md` |
| depth matrix, FD-1 | spec decode | `DEVLOG-spec-decode.md` |

Verdicts live at the top of each new file. Review/adjudication trails
that used to live in `fa-decode-fp16-hunt-*.md` (deleted in the same
commit) are folded into the topic logs above; raw copies remain in git
history at `1ee7bd4fb6^`.

# <TOPIC> — one-line claim

> Branch `<branch>` off `<base>` · model `<model>` · date `YYYY-MM-DD` ·
> roadmap item `<roadmap-id>`.

**VERDICT:** `DEAD-END` | `SHIPPED` | `NEUTRAL` | `SUPERSEDED` | `OPEN`

**GATE:** the one measurement that decides this verdict (e.g. `serving,
graph mode, pp=2048/tg=256, 4 samples`, `_bench_gfx906.py`). Per-kernel
census / standalone harness results are evidence, *not* the gate — they
have repeatedly failed to transfer (see [moe-gemm1-retiling](DEVLOG-moe-gemm1-retiling.md) §3).

**DROPPED / REVERTED at commit:** `<hash>` (if verdict is DEAD-END/
SUPERSEDED). Prepared for git-revert or already reverted. vLLM no-busywork
rule: no shipped dispatch change without a measured gate benefit.

---

## HYPOTHESIS

One or two sentences: the original idea, why it seemed worth trying, what
it would change. **Prefer a falsifiable form.** ("If X is faster in the
standalone harness, it must show in serving wall-clock" is *falsified* by
the transfer failures.)

## What was done

Terse bullet list: the actual change (files/kernels, launch tweaks), the
configs swept, codes/commits touched. No prose padding. Log paths under
`/tmp` that back the numbers.

## Evidence — FOR

Bulleted. Numbers + the gate they were measured at. Standalone / corpus /
census results go here but are explicitly labeled as **launch-regime**
(not the gate).

## Evidence — AGAINST

Bulleted. Numbers + the gate. If the verdict is DEAD-END, the decisive
"does not transfer" row goes here and is called out as the blocker.

## Why it failed (if applicable)

The mechanism (measured or inferred): e.g. "the 8-slot fp16 accumulation
dwarfs the tiling gain"; "per-kernel rows are pipeline-state dependent,
not graph-context-only".

## Interactions / superseded-by

- What this experiment invalidated, or was invalidated by.
- Recurring lesson it reinforces (e.g. "3rd consecutive decode-size gemm1
  retiling fails transfer").

## Refrigerated residue

Cheap or near-hit calls this exploration surfaced but did not ship — for a
future reader sniffing the same shelf. (e.g. C3 zeroing fold, ~234 µs/step.
Do **not** restate roadmap candidates verbatim; cross-link them.)

## Search keys

`HYPOTHESIS:` `VERDICT:` intended to be grep-able in one pass.

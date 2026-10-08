# MTP depth matrix on the real payload — k=3 is the production depth; the s9 ceiling does not transfer

**VERDICT:** k=3 SHIPPED as the serving default at production context
(120k) · k=4 DEAD-END on real payloads (the s9 perfect-acceptance win did not
transfer) · k=5 unmeasured on real payloads (s9 ceiling only) · k=2 kept for
short-context/chat-heavy paths · **GATE:** same-boot serving A/B on the
*production corpus* (agent/chat bodies), 3 reps × 3 points, tg=256, temp 0,
TP=2, Qwen3.8-27B-AWQ-INT4
**Branch:** `gfx906/fa-decode-fp16` · 2026-09-07/09 · boots X/Y/Z/W'/W''
**Full detail:** `git show e963fd8c62:docs/gfx906/DEVLOG-fa-attention.md`
(A1/mtp4ag/depth-matrix entries, pre-split) · artifacts `/local/tmp/a3/`

## HYPOTHESIS

MTP depth was chosen on the **s9 filler** (perfect acceptance, acc_mean = k+1):
k=4 looked like a win there. If the real payload's acceptance is much lower,
then deeper drafting adds verify cost for ~0 marginal tokens and the depth
ranking inverts. The production depth must be picked by a same-corpus A/B.

## The matrix (medians t/s, `plain` = no MTP)

| arm | corpus | 64k | 96k | 120k | acc_mean |
|---|---|---|---|---|---|
| plain | v1 | **19.76** | — | **13.11** | — (slope anchor, boot Z) |
| mtp2ag (k=2) | v1 | **30.95** | — | **20.71** | — |
| mtp3ag (k=3) | v1 | — | — | **24.80** | — (reduced: 120k only) |
| mtp4ag (k=4) | v1 | 29.37 | 23.89 | 21.87 | 2.0–2.98 |
| mtp3v2 (k=3) | v2 (20 % chat) | **27.44** | — | **24.76** | 1.38/2.15 (medians) |
| mtpv2 (k=2) | v2 | **27.30** | — | **22.65** | 1.04–1.17 @64k, 1.58–1.71 @120k |
| (ref) mtp4 @s9 | s9 filler | 45.43 | 36.63 | 31.77 | 4.0 (= k+1, perfect) |

All launch sets clean (boot Z 5/5; boot W'' mtpv2 6/6, KV 15.57 GiB); canaries
38.7–38.8 t/s (healthy band 38.4–38.9); mclk median 1000 MHz in every window.
The 120k reps include the first-ever compile of the ported GDN spec kernels
(`fb0fc766e4`); reps 1–2 are steady state and the port ran thousands of draft
steps with zero server errors.

## Verdicts

1. **Item-2 threshold CLEARED:** k=2 @64k = 30.95 ≫ ~20 — spec decode stays
   net-positive on the production payload; no revert to plain/ngram. (Plain at
   long context is 18.9 @64k / 12.7–13.1 @120k, so k=4@mixed is still
   **+55–70 % vs plain** — the "k=4 is a net loss" wording of the first entry
   was a cross-regime misread and is retracted here; the loss is *vs k=2/k=3*.)
2. **k=3 wins @120k (v1):** 24.80 = **+89 % vs plain**, +13.4 % over k=4,
   +19.8 % over k=2.
3. **k=2 wins @64k (v1):** 30.95 vs k=4's 29.37 (+5.4 %) — the shallow verify
   pays where acceptance is weakest.
4. **k=3 keeps the 120k win on v2** (24.76 vs k=4's 21.87 = +13.2 %), and the
   same-corpus cell closes the ranking: **120k k=3 24.76 > k=2 22.65 (+9.3 %);
   64k k=3 27.44 vs k=2 27.30 = +0.5 % (a tie inside rep noise)**.
   Acceptance at 120k is markedly higher than at 64k (acc_per_pos ≈0.85/0.78 vs
   ≈0.66/0.45) — the longer agent tail is more copy-heavy.
5. **k=4 is last at 120k** on every corpus.

## Why k=3 is the sweet spot (mechanism, measured in the kernel track)

The verify block is 1 anchor + k drafts → rows k+1. **k=3 gives 4 rows, which
pads to the same `Sq_pad=4` occ-2 FA tile as k=2 (3 rows) — one cheap extra
head at no verify-tile cost. k=4 gives 5 rows → pads to `Sq_pad=8`, crossing to
the slow occ-1 tile (~19 vs 8.64 ns/token).** The depth call is therefore a
**tile-config** call, not only an acceptance call (see
`DEVLOG-fa-verify-sq8.md` for the tile cliff, and `DEVLOG-fa-attention.md` for
the Sq=8 ledger that first exposed it).

## A1 — k=5: the s9 *ceiling* measured, never validated on the payload

Boot X (s9 filler, 3×3 reps): **mtp5 = +11.1/+11.4/+12.4 % vs k=4**
(medians 50.47/40.80/35.72 @64/96/120k vs 45.43/36.63/31.77; acceptance 1.0 at
every position, acc_mean 5.0; text probe token-identical to mtp4).
**VERDICT: this is a perfect-acceptance CEILING, not a traffic prediction** —
it upper-bounds what k=5 could buy; the agent-corpus A/B (mtp4ag vs mtp5ag)
that would make it a verdict was blocked by wedges #30/#31, and **k=5 was
subsequently skipped by decision** (k=4 already lost on real payloads; the
sweet spot is k=2–3). Do not quote the +11–12 % as a serving expectation.

## Per-body variance is real (the measurement trap this campaign hit)

Reps rotate through different chat+agent bodies (`rep%8`), so `acc_mean` spans
2.0–2.98 and t/s spans 22.4–28.6 within one arm at 96k: **medians carry the
spread, single reps do not.** Acceptance decays steeply per position
(pos-0 0.73–0.82 → pos-3 0.29–0.47), which is what makes the depth ranking
corpus-dependent.

## Related

- **FD-1** (the `VLLM_GFX906_FUSED_DRAFT=1` offline gate at the *mtp3b4* B=4/120k
  shape, NEUTRAL and stack-confounded — offline vs serving) →
  `DEVLOG-spec-decode.md`; keep-or-strip analysis: `/local/tmp/b4/fd1-keep-strip-decision.md`.
- **SYV-12's generation-time gate** (25.3 % hit_frac over 39 convos /
  205,091 generated positions, clearing the ≥15 % gate) was a free ride on this
  campaign's replay corpus → `DEVLOG-syv12.md`.
- The B=4/120k prefill-tax campaign that hosted the FD-1 arm →
  `DEVLOG-fa-multibatch-prefill.md`, `ttft-prefill-stall.md`.

## Harness / process findings (worth keeping — each cost a session)

1. The sweep client assumed flat per-point token lists; the nested 8-body
   format made every rep fail `len(prompt)==pp` **with rc=0** (a silent 0/27
   sweep). Fixed: flat/nested unwrap + per-rep body selection, identical
   across arms.
2. On ABORT the driver deleted the attempt-2 server log before the abort check
   (lost the traceback). Fixed: abort check before `rm`; `.attempt1` rename for
   both dead attempts.
3. `run_w1.sh` set `MTP1_PTS` but not `MTP1_PORT` → the sweep hit the client's
   default port and got connection-refused (a *software* failure, driver held
   for a human; the server was healthy and a continuation driver attached to
   it and re-ran 6/6).
4. Wedge tally this campaign: #30/#31, #33/#34, #37/#38 bursts (chronic
   weight-load family, single-instance, BACO self-recovered) — the wear rule
   (2nd consecutive genuine failure = BURST → stop + reboot) held every time.
   `degradation*.md` holds the table.

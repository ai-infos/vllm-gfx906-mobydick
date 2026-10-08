# Code review — `gfx906/nh2-int8-gemv` (NH-3, amdsmi fixes, NH-2 int8, NH-5)

Review of the 13 commits on `gfx906/nh2-int8-gemv` ahead of
`gfx906/main` (merge base `284ce5ff6a`), performed before the merge to
main. Scope: all code commits since `b0c89b9f7d` (NH-3) — the earlier
onboard/C1 work was reviewed at its own merges (`c645b66ebf`, C1 review
docs).

## Verdict

**MERGE-READY — no blocking findings.** One hardening note (F1,
non-blocking) and one blast-radius observation recorded for the future.
All affected unit suites green on GPU0: align/topk/w8a16/gemm/amdsmi
**111 passed, 2 skipped**; mamba SSD restructure **94/94**. Ruff clean
on every touched file (no new errors vs `gfx906/main` versions).

## Findings

### F1 — NH-5 fused-align gate: EP safety invariant holds (verified, no change)

The `(128, 6)` instantiation of `moe_align_m1_gfx906` sizes its LDS
counter array to the GLOBAL expert count (`E=128`) and indexes it with
raw `topk_ids`. Under TP=2 + `--enable-expert-parallel` each rank holds
only 64 local experts while `topk_ids` carry global ids up to 127 — if
the fused path ever fired there, the kernel would be correct (it only
needs the id space) but the *generic-chain equivalence* it claims
(`num_tokens_post_pad == topk`, buffer sizes) would no longer match what
downstream expects under EP.

Verified the gate cannot fire under EP: `_use_fused_align_m1` requires
`expert_map is None`, and `determine_expert_map`
(`expert_map_manager.py`) returns `None` **iff `ep_size == 1`** — any
EP deployment passes a real map (global→local, −1 for remote) and the
fused path falls back to `moe_align_block_size`, which is EP-aware by
construction. Same invariant as C1 stage 1's original gate; no code
change needed. Recorded here so a future (E, topk) addition keeps the
`expert_map is None` check.

### F2 — NH-5 fast path blast radius beyond Nemotron (observation)

`_grouped_topk_single_group` fires for ANY model with
`n_group == 1 and topk_group == 1` (default ON), not just Nemotron-H:
`openpangu.py`, `lfm2_moe.py`, and `hy_v3.py` in-tree also pass
`num_expert_group=1`. The fast path is provably degenerate-identical to
the generic chain for that config (group topk over 1 group = no-op,
mask all-ones), and the unit suite pins bit-equality incl. tie-heavy
inputs across sigmoid/softmax × bias × renorm × scale — so the behavior
change is a pure node removal with identical numerics. Acceptable;
noted so a future reviewer doesn't mistake it for a Nemotron-only gate.

### F3 — env read inside `@torch.compile` (accepted, documented)

`VLLM_GFX906_TOPK_SINGLE_GROUP` is read via `os.environ.get` inside the
compiled `grouped_topk`. Dynamo constant-folds it per captured graph;
toggling at runtime re-traces (and a mid-session flip could in principle
serve two cached variants). The compiled env-toggle test
(`test_single_group_compiled_env_toggle_bit_equal`) proves both arms are
bit-equal, so even a mixed capture cannot diverge numerically. Serving
practices set the flag before launch; accepted as-is.

### F4 — NH-3 `torch.mv` path scope (verified)

The fp32 branch sits inside the existing skinny-GEMV guard
(`n == 1`, `m % 4 == 0 or m < 4`, `bias is None`, K whitelist), so it
only takes single-token fp32 GEMVs — exactly the
`force_fp32_compute` router-gate case. `torch.mv(weight, x_view[0])`
computes `w @ x` = `F.linear(x, w)` element-wise (fp32 in/out, no dtype
promotion); the boundary test pins that `m % 4 != 0` stays on the triton
path. No change.

### F5 — amdsmi cleanup pair (verified)

`with_amdsmi_context` now branches on query outcome (warning at process
scope on success, debug on failure — the propagating exception dominates
the failure path), and `rocm_platform_plugin`'s detection finally is
fail-soft so a NOT_INIT shut_down can no longer misattribute the
detection log. Both match the 2026-08-22 C2-V degradation entry's call
for a permanent fix. Tests pin init/shutdown call counts and the warning
contract. No change.

### F6 — ssd_chunk_scan pointer-yield restructure (verified)

The restructured `scf.if` branches load their own tiles and yield values
only; semantics are preserved because `HAS_INITSTATES` is a constexpr
set from `initial_states is not None` at the call site — so whenever the
init-states branch is live, `initstates_ptr` is non-null. The
non-init-states fresh-seq zero path and the same-sequence continuation
load are statement-identical to the pre-restructure logic. 94/94
reference tests incl. initial-states variants. No change.

### F7 — NH-2 int8 in-kernel path (verified, default OFF)

Opt-in behind `VLLM_GFX906_W8A16_INT8=1` (default 0, measured NO-GO for
the serving mode). The in-place XOR pre-shift of the packed storage is
one-way (documented: after it, `weight_packed` no longer holds raw
checkpoint codes) — acceptable because the process never re-reads raw
codes post-load, and the dequant fallback path (default) is untouched.
GEMV/GEMM kernels match the P2-validated 3-op element chain; geometry
table has measured winners for all six Nemotron shape families with a
fail-closed BK assertion otherwise. No change.

## Test record (GPU0, this boot)

| suite | result |
|---|---|
| `test_moe_align_m1_gfx906.py` + `test_grouped_topk_single_group.py` + `test_compressed_tensors_w8a16_channel_dequant.py` + `test_rocm_unquantized_gemm.py` + `test_rocm_amdsmi_context.py` | 111 passed, 2 skipped (35.9 s) |
| `test_mamba_ssm_ssd.py` | 94 passed (26.5 s) |

Logs: `/local/tmp/nh_review_tests.log`, `/local/tmp/nh_review_ssd.log`.

## Search keys

`HYPOTHESIS:` n/a (review) · `VERDICT:` MERGE-READY, no blocking
findings, F1 EP invariant expert_map None iff ep_size 1, F2 blast radius
single-group models openpangu lfm2_moe hy_v3, F3 env constant-fold
torch.compile, F4 torch.mv fp32 skinny guard, F5 amdsmi outcome branch,
F6 pointer-yield restructure HAS_INITSTATES constexpr, F7 int8 opt-in
default off XOR pre-shift.

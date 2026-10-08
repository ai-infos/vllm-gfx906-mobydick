# TP=2 crash analysis — `gfx906_fa` NC2 head-packing guard

Log: `/tmp/message (1).txt`

## TL;DR

All 4 TP workers crash in the **gfx906 FA native attention kernel** with
`RuntimeError: gfx906_fa_launch failed: invalid argument` during what looks
like **warmup / first inference** on a Mamba-hybrid model (Qwen3.5/Qwen3-Next).
The native launcher rejects the launch before any GPU work because the packed
attention tile count `nc2` defaults to **8** but the per-TP-shard query head
count is **`heads_q = 6`**, which is not divisible by 8:

```
[gfx906_fa] nc2=8 must be a power of two dividing heads_q=6
```

The EngineCore's `KeyError: 'chatcmpl-...'` and the `EngineDeadError` and the
connection reset are all **downstream** symptoms of the workers dying.

## The fault chain (bottom-up)

1. `qwen3_next.py` / `qwen3_5.py` forward → unified attention (`attention.py:777`)
   → gfx906 FA backend `forward` (`gfx906_fa_backend.py:650`)
   → `forward_paged` (`gfx906_fa_paged.py:605`) → `gfx906_fa.forward(...)` native.
2. Native `gfx906_fa_launch_impl` (`csrc/gfx906_fa/gfx906_fa_launcher.cu`) is
   entered with `heads_q=6`, `nc2=8` (default from
   `get_fa_nc2()` → env `GFX906_FA_NC2`, else **8**).
3. Launcher guard (`gfx906_fa_launcher.cu:100-103`):

   ```cpp
   } else if (heads_q % nc2 != 0 || (nc2 & (nc2 - 1)) != 0) {
       fprintf(stderr, "[gfx906_fa] nc2=%d must be a power of two dividing heads_q=%d\n", nc2, heads_q);
       return hipErrorInvalidValue;
   }
   ```

   `6 % 8 != 0` → prints the message, returns `hipErrorInvalidValue`.
4. Python `at::cuda::OptionalCUDAGuard`/launch wrapper turns the non-OK hip
   result into `RuntimeError: gfx906_fa_launch failed: invalid argument`.
5. All four `Worker_TP*` processes raise; EngineCore dies; API requests get
   `EngineDeadError` / connection-reset.

The 8 `nc2=8...` lines are exactly **2 per TP worker** (4 workers), consistent
with each shard hitting the guard (e.g. warmup capture + a live run), and all
four fail identically because TP shards all get the same `heads_q=6`.

## Root cause — the auto-downgrade is unreachable for `heads_q < 8`

The code *intends* to auto-downgrade `nc2` for GQA ratios that 8 doesn't divide
(see `gfx906_fa_launcher.cu` around line 150 and the commit
`b4873459f8`/`1a895e8a01`): the default `nc2=8` should fall back to `2` or `1`.
But the guard **order is wrong**:

- The GQA-ratio downgrade runs on an `else if` branch **below**
  (`gqa_ratio = heads_q / heads_kv`; `gqa_ratio % nc2 != 0` → `down => 2 or 1`).
- That downgrade path is **only reachable when line 100 passes** — i.e. when
  `heads_q % nc2 == 0` **and** `nc2` is a power of two.
- For `heads_q=6, nc2=8`, `6 % 8 != 0`, so line 100 fires **first** and returns
  an error **before** the ratio downgrade can ever be evaluated.

So the "default nc2=8 auto-downgrades 8→2→1" intent only works for head counts
**≥ 8 that are divisible by 8** (e.g. the `Hq=24/Hkv=4` case the commit
validated). Any per-shard `heads_q` that is **not itself divisible by 8** —
notably **6** — fails hard even though a perfectly valid `nc2=1` (MHA-style
unpacked) path exists.

## Why `heads_q=6` under TP=2

The gfx906 backend sets `heads_q = query.shape[1]`, i.e. the **per-TP-shard**
query head count. With tensor-parallel=2 the model splits `total_num_heads`
across the 2 GPUs (`qwen3_next.py:243` `num_heads = total_num_heads // tp_size`).
A model with `total_num_heads` that TP=2 turns into 6 per shard (e.g.
`12/2`, or the Mamba/linear-attn head geometry defined via
`linear_num_key/value_heads`) lands squarely on this. The dense Qwen3.5-27B
path (Hq=24) doesn't trip it because 24 % 8 == 0; it's the small-head,
TP-split mixed (Mamba + dense) config that does.

## Why "without recent bug fixes" still crashes

The `: 8` default and the `heads_q % nc2` guard both came in
`e8b3293554` ("B=1 decode parallelism — GQA head-packing + KV split") and are
present on **`gfx906/main`** unmodified (verified identical in `gfx906/main` vs
`HEAD`). So this isn't a regression introduced by the MTP/spec-decode work —
it's a latent bug on `gfx906/main` that only triggers on this mixed model with
a per-shard `heads_q` < 8 and not divisible by 8.

## Concrete fix

Reorder the guard so the divisibility-of-`heads_q` failure **feeds the
auto-downgrade instead of aborting**, or equivalently check the ratio first:

```cpp
if (nc2 <= 1) {
    nc2 = 1;
} else if ((nc2 & (nc2 - 1)) != 0) {
    fprintf(stderr, "[gfx906_fa] nc2=%d unsupported (must be a power of two)\n", nc2);
    return hipErrorInvalidValue;
} else {
    const int gqa_ratio = heads_q / heads_kv;
    // must not straddle a GQA group: tile shares ONE kv head.
    // ratio % nc2 must == 0  ==>  heads_q % (nc2*heads_kv) == 0 implicitly.
    if (gqa_ratio % nc2 != 0 || heads_q % nc2 != 0) {
        if (nc2 == 8) {
            const int down = (heads_q % 2 == 0) ? 2 : 1;
            if (gqa_ratio % down != 0) down = 1;
            nc2 = down;
        } else {
            ... reject ...
        }
    }
}
```

The key invariant is **`gqa_ratio % nc2 == 0`** (the docstring/comments already
state this is the real correctness constraint: a packed tile must share one KV
head). The extra `heads_q % nc2 != 0` check is redundant when `heads_kv` divides
`heads_q`, but as written it fires before the intended fallback and turns a
soft/valid case hard. Either drop it (relying on `gqa_ratio % nc2`) or make it
route into the same downgrade path as the ratio check.

Optionally also widen the instantiated set or derive the default `nc2` from the
actual heads rather than hard-8, but the minimal, correct patch is the guard
reorder above.

## Immediate workarounds (no rebuild)

- `GFX906_FA_NC2=1` for this model (disables GQA head-packing entirely — the
  `KVsplit`/packed-tile speedup is lost, workload is correct). 
- `GFX906_FA_NC2=2` if `heads_q` (6) is even — 6 % 2 == 0, ratio 6 % 2 == 0.
- Run TP=1 (per-last-shard `heads_q` likely not divisible by 8 either, but a
  full-head layout may differ) — verify first.
- Any run under TP=2 on the Mamba-hybrid model should pin `GFX906_FA_NC2=1`.

## What to grep for on the next repro to confirm the config

- `heads_q=6` value + `GFX906_FA_NC2` unset (default 8).
- The 8× message then 4× `RuntimeError: gfx906_fa_launch failed: invalid argument`
  in each `Worker_TP*`.
- Confirm TP=2 in the launch args; confirm the model is the Mamba-mixed
  (Qwen3.5-Next) one so per-shard heads land on 6.

## Classification

- **Severity**: crashing / release-blocking for TP=2 serving of this model on
  `gfx906/main`.
- **Not** a spec-decode/MTP regression. Pre-existing in the gfx906 FA stack.
- **Fix**: guarded ordering bug, small patch, needs a real-gfx906 numeric + a
  unit test that exercises `heads_q` values not divisible by 8 (6, 10, 12...)
  and asserts the downgrade to a valid `nc2`.
# gfx906 Latency-Hiding Ops (Measured)

> Source: <https://skyne98.github.io/wiki-gfx906/> — copyright Oleksii Halahan.
> All numbers were measured on MI50 32 GB (gfx906).

Instruction patterns on gfx906 (MI50/MI60) most useful for hiding latency in
quant/dequant-style kernels.

## Scope

Focus is on:

- wave-lane exchange (DPP, DS permute)
- LDS width (b32/b64/b128)
- global load width (dword vs dwordx4)
- scheduling behavior (s_waitcnt placement)

All kernels were compiled for `--offload-arch=gfx906` and validated with
emitted ISA.

## Key measured findings

1. **Use DPP first for row-local shuffles.** `v_mov_b32_dpp` row shift
   (`row_shr:1`) vs LDS+barrier equivalent: dpp_row_shr ~1778–1784 Gxchg/s,
   lds_row_shr ~906 Gxchg/s. For row-local lane movement, DPP gives about 2x
   throughput and removes barrier overhead.
2. **Use `ds_bpermute_b32` for general in-wave exchange.** XOR-neighbor
   exchange benchmark: `ds_bpermute_b32` ~962–970 Gxchg/s, LDS
   store+load+barriers equivalent ~905–907 Gxchg/s. Consistently better than
   LDS exchange when the shuffle pattern is not DPP-friendly.
3. **Prefer wide LDS ops for staging.** Pure LDS streaming kernels
   (instruction forms confirmed in ISA): `ds_read/write_b32` (l1) ~1.9–3.9
   TB/s; `_b64` (l2) ~4.3–8.8 TB/s; `_b128` (l4) ~9.5–11.2 TB/s. b128 LDS
   accesses are the strongest baseline for LDS-heavy staging paths.
4. **Wide global loads help when the memory path is healthy.** Compiler emits
   `global_load_dword` (scalar path) vs `global_load_dwordx4` (vector path).
   In uncongested runs, dwordx4 outperformed scalar (~867–873 GB/s vs
   ~814 GB/s). On a shared machine, global-memory numbers varied run-to-run;
   treat this as directionally positive, not a fixed constant.

## Scheduling behavior that matters

In ILP kernels, the compiler issues multiple loads first and delays waits:

- VMEM: staged `s_waitcnt vmcnt(3..0)`
- LDS: staged `s_waitcnt lgkmcnt(...)`

That pattern is the core latency-hiding mechanism on gfx906: keep multiple
memory operations in flight before consuming results.

## Practical checklist

- Row-local shuffle: use `v_mov_b32_dpp`.
- Arbitrary in-wave shuffle: use `ds_bpermute_b32` / `ds_permute_b32`.
- LDS staging: default to `ds_read/write_b128` where alignment allows.
- Global staging: prefer `global_load_dwordx4` for contiguous packed data.
- Structure loops to issue multiple independent loads before first use.
- Avoid immediate waits after each load; let the compiler keep VMEM/LDS
  queues populated.

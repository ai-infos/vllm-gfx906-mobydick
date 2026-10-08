# gfx906 LDS Layout Standard for LLM Blocks

> Source: <https://skyne98.github.io/wiki-gfx906/> — copyright Oleksii Halahan.
> Numbers measured on MI50 32 GB (gfx906) with `ds_read_b128`/`ds_write_b128`.

A practical LDS layout standard for gfx906 kernels (GEMM/attention-style
tiles) based on measured bank-pressure behavior.

## Why this matters

Many LLM kernels read one operand row-wise and another operand effectively
column-wise from LDS. On gfx906, column-style accesses can collapse bandwidth
if row stride aliases LDS banks.

## Measured result (key experiment)

Microbenchmark on real gfx906 using `ds_read_b128`/`ds_write_b128`:

- contiguous vec4 access baseline: ~4257 GB/s
- column-style access with ld=32 vec4: ~1865 GB/s
- same column-style access with ld=33 vec4 padding: ~3974 GB/s

Interpretation:

- ld=32 (power-of-two stride) is a bad default for column-like LDS reads.
- adding one vec4 of padding per row (ld=33) recovers most bandwidth.

## Layout standard for gfx906

- Use 16-byte vectorized LDS payloads (uint4/float4/packed int blocks).
- Keep base LDS buffers 16-byte aligned.
- For tiles consumed row-wise only: use natural row stride.
- For tiles that will be consumed column-wise (or transposed access), use
  padded leading dimension in vec4 units: `ld_vec = logical_ld_vec + 1`.
- Prefer `ds_read/write_b128` staging paths over scalar LDS traffic.

## Recommended defaults

- A-like operand (row-consumed): no pad needed.
- B-like operand (column-consumed by waves): +1 vec4 pad per row.
- If LDS budget is tight, test +1 first before more complex swizzles.

## Practical formula

If a row has `K_vec` vec4 elements, allocate:

- `stride_vec = K_vec` for row-only reads
- `stride_vec = K_vec + 1` for column-like reuse

LDS footprint increase is modest (~1/K_vec fractional overhead) and often
worth it.

## Power-of-two strides matter in global memory too (2026-08-28)

The same phenomenon extends beyond LDS to **global-memory plane
layouts**: `mxxm-t/mx-llama.cpp` PR #4 ("q8 repack", merged, real
MI50 hardware) pads its Q8_0 two-plane row stride by one 32-value
sub-block **iff the sub-block count is a power of two** — explicitly
"to avoid shared/global-memory bank conflicts" — and measures
+21…+51 % on its LDS-staged GEMM path with the padding in place (vs
llama.cpp native MMQ).

Rule of thumb: any plane/tile row stride that comes out a power of
two (in the access granularity actually used — dword, uint4, or
sub-block units) should get +1 unit of padding; check this whenever
a quantized layout is designed, not just for LDS. Incidental
example: the gfx906-fa Q8 tile stride of 136 B = 4×34 is never a
power of two at any granularity — safe by construction, but the
check belongs in the layout decision (see `plan_fa_part_A.md`
A1/A2).

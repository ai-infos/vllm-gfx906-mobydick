---
name: gfx906-isa-disassembly
description: "Extract and disassemble the gfx906 (amdgcn) GPU code of vLLM custom kernels in /local/git/vllm-gfx906-mobydick (2x MI50, ROCm 7.14). Use when a perf claim, review finding, or conflict between committed records rests on what the compiler actually emits: instruction fusion (e.g. half2 a*b+c into packed FMA), dot-instruction usage, per-kernel instruction counts, or whether a gfx906 ISA instruction exists at all. Validated 2026-08-29."
---

# gfx906 ISA disassembly & analysis

Adjudicating what the compiler *actually emits* for the gfx906 custom
kernels (`csrc/gfx906_fa/`, `csrc/rocm/*`) — for perf claims, review
findings, and ISA-availability questions.

## Provenance rule

A dump is only as good as the build it came from. Record the commit +
`.so` mtime alongside every count; re-extract from the current build
before citing. A review rejected a refutation built from a *parked,
unmerged branch's* dump (DEVLOG-fa-attention.md 2026-08-29, F1).

**Stale-object hazard:** after branch switches touching `csrc/`, the
build-tree `.o`/`.hip.o` can be from the *other* branch's sources
(hipify skip-check false-negative once produced a Frankenstein .so of
all-NaN outputs). Wipe
`build/temp.*/CMakeFiles/_gfx906_fa_C.dir build/temp.*/csrc/gfx906_fa`,
rebuild, and verify the `.o` mtime before disassembling.

## Extraction (validated 2026-08-29)

The built `.so` contains only host x86 ELFs — the amdgcn object is
embedded in the build-tree **`.hip.o` files' `.hip_fatbin` section**
as a clang offload bundle. Tools live in `/opt/rocm/lib/llvm/bin/`.
There is no `llvm-offload-packter`; the bundle tool is
`clang-offload-bundler`.

```bash
cd /local/git/vllm-gfx906-mobydick
LLVM=/opt/rocm/lib/llvm/bin
O=build/temp.linux-x86_64-cpython-312/CMakeFiles/_gfx906_fa_C.dir/csrc/gfx906_fa/gfx906_fa_launcher.hip.o
$LLVM/llvm-objcopy -O binary --only-section=.hip_fatbin $O /tmp/fatbin.bin
$LLVM/clang-offload-bundler -list -type=o -input /tmp/fatbin.bin
# -> hipv4-amdgcn-amd-amdhsa--gfx906   (and the host triple)
$LLVM/clang-offload-bundler -unbundle -type=o \
  -targets=hipv4-amdgcn-amd-amdhsa--gfx906 \
  -input /tmp/fatbin.bin -output /tmp/gpu.o
$LLVM/llvm-objdump -d /tmp/gpu.o > /tmp/disasm.txt   # ~250k lines for the FA launcher
$LLVM/llvm-nm -C /tmp/gpu.o | grep ' T ' | grep flash_attn   # instantiation list
```

Each `.hip.o` carries its own TU's kernels: FA tile kernels in
`gfx906_fa_launcher.hip.o`, quantizer in `gfx906_fa_quant.hip.o`,
gather in `gfx906_fa_gather.hip.o`.

## Per-kernel instruction counts

Function headers look like
`000000000006df00 <_ZL18flash_attn_tile_q8ILi128ELi128ELi16ELi2ELb0EEvPKc...>:`.
Traps (all hit in practice):

- **Addresses are 14 hex digits, not 8** — regex `[0-9a-f]{8}` silently
  matches zero lines; use `[0-9a-f]+`.
- **Itanium template mangling E-counting**: each arg of
  `ILi128ELi128ELi16ELi2ELb0EE` contributes a trailing `E` *plus* the
  list's closing `E`; hand-written regexes drop/double one and match
  nothing. Copy the exact mangled string from `llvm-nm -C` (or the
  disasm) instead of reconstructing it.
- **The last function header swallows the rest of the file** — no next
  header bounds it, so per-function counts for the last symbol include
  unrelated code. Bound by next header AND check the tail, or scope
  with `llvm-objdump -d --section=<sec>` per instantiation.

Counting: split on headers, then `len(re.findall(rf"\b{mnem}\b",
body))` per mnemonic. Fusion discriminant: a fused half2 accumulate is
the **in-place packed FMA** `v_pk_fma_f16 vN, vA, vB, vN` (src3 == dst);
an un-fused one shows `v_pk_mul_f16` + `v_pk_add_f16` pairs with
distinct dsts. Worked example (2026-08-29): `flash_attn_tile_q8<128,
128,16,2>` (NC2 prefill) P·V = 1024× in-place `v_pk_fma_f16` vs 54×
`v_pk_mul_f16` (one-shot QK dequant scales) and 0× `v_pk_add_f16`;
counts scale with the unroll across instantiations.

## Compiler-contraction facts

- Source `acc += v * p` contracts to `v_pk_fma_f16` under the
  **default fp-contract**: ROCm 7.14's `amd_hip_fp16.h` defines plain
  `__hmul2` as contractible; only `__hmul2_rn` has
  `#pragma clang fp contract(off)`. "Two source ops" is NOT evidence
  of two instructions — disassemble.
- ISA rate/availability record: `docs/gfx906/dequant-instructions.md`
  (heed the SUPERSEDED markers — corrected 2026-08-29 over exactly
  this source-vs-ISA misread).
- **Availability gate = compile-to-object, not `-S`**:
  `clang++ --offload-arch=gfx906 -c probe.cu` (llvm-mc
  accepts/rejects); `-S` output alone is not the test.
- f16 test constants: derive at runtime with `__ushort_as_half`,
  never by hand (0x3C00 is 1.0, not 1.5 — the dot-probe false alarm
  was all hand-derived constants).

## Standalone probe (no full vLLM build)

For a single-kernel question, compile a minimal `.cu`/`.hip` with
`clang++ --offload-arch=gfx906 -c probe.cu -o probe.o` and run the same
objcopy/unbundle/objdump chain on `probe.o`. Precedent:
`benchmarks/kernels/gfx906/dot_isa_probe.py` (rates recorded in
dequant-instructions.md).

## Worked case

2026-08-29, M3 review F1: the roadmap's dot2 P·V refutation and
`dequant-instructions.md` disagreed on whether FA-Q8 P·V accumulates
via `v_pk_fma_f16` or a mul+add pair. One offload-bundle objdump of the
*current merged build* settled it (fused FMA, full packed rate); the
losing record was marked SUPERSEDED. Devlog:
`docs/gfx906/DEVLOG-fa-attention.md`, 2026-08-29 review-fixes section.

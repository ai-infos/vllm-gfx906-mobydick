# gfx906/v0.30.0.x integration audit

This branch integrates the v0.30.0 reference and retains its optimized defaults.
Host checks and compiler checks are separate from device validation. No GPU
correctness, model quality, graph replay, or new speedup is established here.
The reference benchmarks remain attributed historical evidence.
In particular, [FA-D96](DEVLOG-fa-d96.md),
[head-padding gates](DEVLOG-fa-coverage.md), and the reference release reviews
remain the evidence for those defaults, rather than new measurements here.

## Pinned integration inputs

| Input | Commit |
| --- | --- |
| Local `main` before integration | `ff063e44e2be3ed37f47790e81dd94fa5e1cadb9` |
| KIntegrated `gfx906/v0.30.0` | `57530039efaf273ac65eed05a304023d646b31aa` |
| Official `v0.30.0` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |

Both inputs were fetched and ancestry checked locally. Both baselines are
ancestors of the reference; `gfx906/v0.30.0.x` was created at local `main` and
advanced with `git merge --ff-only`. No local commits were replayed. `main`
remains at its original commit. Follow-up fixes are separate from imported
history and attribution. No step-2 branch, PR, or publication was created.

Complete rename-aware inventories, without GitHub's 300-file truncation:

- [Against local v0.23.1](upgrade-0.30.0-inventory/local-0.23.1.tsv): 4,977 entries,
  891,926 added lines / 157,662 removed lines.
- [Against official v0.30.0](upgrade-0.30.0-inventory/official-0.30.0.tsv): 392
  entries, 79,825 added lines / 1,589 removed lines.

These inventories describe the pinned imported snapshot. Follow-up changes are
available with `git diff 57530039efaf273ac65eed05a304023d646b31aa HEAD`.
Regenerate with `git diff --name-status --find-renames <baseline> <reference>`.
Each TSV retains Git's status and old/new names for renames.

## Capability mapping and merge review

“Preserved” means the implementation/dispatch remains in source, not that it
has passed device tests here. Review focused on merge-sensitive interfaces,
reachable gfx906 paths, cache and quantization contracts, and imported tests;
it is not an exhaustive proof of every changed kernel or supported model.

| Capability | Status | v0.30.0 implementation and review |
| --- | --- | --- |
| Dense Q8 attention, prefill/decode | Adapted | `vllm/gfx906_fa/`, `csrc/gfx906_fa/`; fused K/V cache, planar Q8 side cache, head padding and gather/direct dispatch retained; causality and gather sizing fixed below. |
| Sliding windows and bidirectional attention | Preserved / unresolved | Causal window clipping survives in both kernel copies. Bool noncausal works in source, but noncausal **windowed** attention still drops the window and attends a superset; see limitations below. |
| Sparse MLA | Preserved | `rocm_aiter_mla_sparse.py`, sparse indexer and software/reference fp16 paths retained. A backend name containing AITER does not itself prove that a matrix-core kernel is selected. Device dispatch remains unverified. |
| MiniMax-M3 including FP32 caches | Adapted | `vllm/models/minimax_m3/{common,amd}/`, cache/attention dtype schemas and aliases retained; fp32 GEMV reductions and fp16 narrowing before dot/native fused ops remain. FP32 cache storage does not imply full FP32 dot arithmetic. |
| AWQ and static GPTQ | Adapted | gfx906 repack/oracle, Exllama routing, zero-offset conventions, fp16 scales and max-ILP M=1 twin retained; split-K initialization repaired. |
| GPTQ group activation order | Intentionally superseded | Official v0.30.0 removes runtime group activation ordering. Existing validator rejects grouped `desc_act=True`, including dynamic overrides; channelwise `group_size=-1` normalizes the no-op to false. Do not reinstate `g_idx` or old operator signatures. |
| FP8 linear / MoE fallbacks | Adapted | Software E4M3 decode and fp16 dot paths retained; subnormal, NaN and FNUZ decoding fixed. Native FP8 instructions are not required by this software decoder. Other FP8 shapes/routes need device tests. |
| W4A16 MoE | Adapted | gfx906 expert GEMM, routing alignment, shape-gated tiles, zeroed workspaces and fused output reduction retained; malformed block/expert/group guards added. |
| Fused single-token MoE alignment | Preserved | Only explicit `VLLM_USE_V2_MODEL_RUNNER=0`, supported E/top-k pairs, int32 routes, BM=1 and no expert map qualify. V2 and automatic runner selection use generic alignment. |
| Speculative decoding | Preserved | MTP/suffix/draft optimizations, bounded accepted-token reads and optional rollback flags survive the imported runner changes; runner staging/capture is unverified here. |
| Communication workarounds | Preserved | RCCL communicator paths, blocking-sync `.pth` shim, topology notes and gfx906 device naming retained. TP/P2P depends on the actual host and driver. |

The original upstream merge is `8893a50e5468c605d989c0d71f452b24fdbbb21f`
(parents `524ac6f2d6668f12e844d1ae11a71bf33b20fce0` and official v0.30.0).
Its recorded conflict resolutions were inspected alongside final source and
automatically merged interfaces: seven-argument `gptq_gemm`, two-argument
`gptq_shuffle`, repack tuples, `seq_lens_cpu_upper_bound`, MiniMax model split,
cache schemas, routing/runner gates, and attention registration. Cache tests
distinguish the backend's logical five-axis allocation shape from fused physical
K/V storage; noncontiguous views are exercised directly.

Root contribution safeguards and domain guides are retained. No instructions
were weakened. The imported upstream-merge skill was used for the static sweep;
the explicitly approved absence of GPU validation is recorded here. No upstream
PR is proposed, so PR duplicate checks and human-submitter gates do not arise.

## Demonstrated defects fixed

1. Tensor/per-token causality previously warned and returned `True`, changing
   attention semantics. Metadata now raises `NotImplementedError` before
   execution, with `--attention-backend TRITON_ATTN` as the alternative.
2. Padded text heads reserved gather rows at the unpadded width. The wrapper's
   buffer-fit check then rejected those buffers and allocated replacements.
   Reservations now use `padded_head_size`; the exact-fit rollback remains.
3. Explicit source-tree registration could advertise CUSTOM on gfx906 without
   the native extension. Registration now checks extension availability.
4. E4M3 subnormal bytes were mapped to FP16 subnormal bits (e.g. byte 1 became
   `2^-17` instead of E4M3FN `2^-9`); NaNs also became finite values. Both linear
   and MoE sites use one software decoder covering FN/FNUZ. Byte reinterpretation
   is restricted to gfx906 in the linear launcher.
5. Shared module imports queried CUDA properties on non-HIP PyTorch. ROCm
   predicates now return an unknown/false architecture on non-HIP builds and
   warn if a HIP build has no queryable device; actual gfx906 detection is kept.
6. GPTQ split-K CTAs raced when z=0 cleared output inside the same launch as
   other CTAs' atomic adds. `empty` followed by `torch::stable::zero_` initializes
   on the stream
   first; the in-kernel clearing was removed from all four variants and the
   max-ILP twin. This adds an initialization launch; no performance result is
   claimed. MoE continues to require its caller's existing zeroed outputs.
7. Native MoE divided by unchecked `block_size_m` and accepted inconsistent
   expert/group extents. Host guards reject unsupported/zero block sizes,
   incomplete blocks and mismatched routing/scale/zero-point capacities.
8. The torch gather rollback used `view` on an advanced-indexing result that
   is not contiguous for fused K/V views. CPU regression reproduced the failure;
   `reshape` now preserves data and accepts those strides.
9. Out-of-tree HIP conversion processed original source paths instead of their
   copied build-tree counterparts. Quoted headers were therefore excluded from
   conversion, leaving `cuda_runtime.h` in `torch_utils.h` and failing native
   compilation. Conversion now consistently processes the copied tree, supports
   CMake's relative build-tree paths, and leaves original sources untouched.
   In-source conversion and unchanged-source `.hip` output are also tested.

No optimized default was flipped. Escape switches remain, including
`GFX906_FA_LEGACY=1`, `GFX906_FA_DIRECT_PAGED=0`, `GFX906_FA_FUSED=0`,
`GFX906_FA_PAD=0`, `GFX906_FA_PAD96=0`, `GFX906_FA_GATHER_EXACT=1`,
`GFX906_FA_NO_NONCAUSAL=1`, `GFX906_FA_VIT=0`,
`VLLM_GFX906_ALIGN_M1=0`, and `VLLM_GFX906_QGEMM_M1_MAXILP=0`.

## Hardware and numerical review

gfx906 is Vega20, with no MFMA/WMMA matrix cores. Packed arithmetic is useful
and retained. The compiler probe checks `__builtin_amdgcn_sdot4` and packed
half2 arithmetic, rather than rejecting every INT8 operation. Generated ISA
contains `v_dot4_i32_i8`, `v_pk_fma_f16` and wavefront size 64, without MFMA/WMMA.
The compiler's target evidence is separate from GPU execution. Primary ISA
reference: [AMD Vega 7nm ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-7nm-shader-instruction-set-architecture.pdf).

The custom FA shim sets wave64; kernels also use logical 32-lane subgroups with
width-limited shuffles. Those logical widths are not proof of a wave32 launch.
Instantiated text head widths are 64/96/128/256; NC2 packing is restricted for
prefill. Static shared-array sizing is below 64 KiB for these config-table rows
(including the 96-wide row), but compiled resource limits and runtime occupancy
must still be assessed per entry point. RDNA WMMA sources are separate from the
gfx906 expert and dense routes; AITER availability and manual backend overrides
must not be treated as universal gfx906 compatibility.

Q8 attention quantizes Q/K in groups of 32 with fp16 scales. P·V uses fp16
intermediates, while split metadata/normalization use wider values. Scale
rounding, padding inside the final Q8 group, and split order can change output.
The CPU FP8 decoder tests use exact comparison (`atol=rtol=0`, NaNs equivalent,
finite sign bits equal); this is **not** a GEMM or model-quality tolerance.
CPU padding checks compare actual padding with unquantized float64 attention,
using the original head-size scale, at `atol=rtol=1e-12`. They do not emulate
Q8 quantization, paged device dispatch, or decode position alignment.
Existing device attention/MoE suites retain their own reference tolerances;
they must be run before asserting GPU correctness.

## Build inputs and compatibility

The canonical recipe is `docker/Dockerfile.gfx906`, driven by
`build_and_push_docker.sh` and [immutable inputs](../../docker/gfx906-build.env):

| Component | Pinned input |
| --- | --- |
| Python | 3.12 from the immutable Linux x86_64 base |
| ROCm/PyTorch base | `mixa3607/pytorch-gfx906:v2.13.0-rocm-7.14`, digest `sha256:269e24c70e22b1f9cd948c4b18e6f78a2183370739ea56da6b9d2c45e6932e77` |
| Observed PyTorch / HIP | `2.13.0+gfx906.20260802001858` / `7.14.60850` |
| Triton source | official v3.8.0, `c01b6774b1865984607d89d89d3a10833de92037` |
| Flash-attention wrapper | ai-infos fork, `0ac8e77b2a6cf773ecf17bc486e1a11fe1e066e0` |
| uv | 0.12.23 Linux amd64 tarball, SHA256 in the input file |

Every uv install is constrained to that exact Torch build and Triton 3.8.0, preventing replacement with CUDA/CPU Torch or a PyPI Triton wheel. Flash-attention forces a source wheel
and skips CK under its Triton AMD mode. Rust uses the project venv interpreter
and the repository's `rust-toolchain.toml`.

The reference reports a PyPI Triton 3.8.0 import segfault on its gfx906 host
([Triton recon](RECON-triton-1.md)); the image therefore builds from source.
The CPU/offline PyPI checks here do not establish that wheel's suitability on
that host. The recipe sets `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON` to avoid the
Ninja/LLVM install-RPATH relink failure and leaves
`TRITON_BUILD_WITH_CLANG_LLD` unset rather than requiring bare clang tools.

The previous ROCm 6.3.x / PyTorch 2.11 / Triton 3.6 stack is **unverified** for
this branch. v0.30.0 build metadata requires Torch 2.13 and stable-libtorch APIs;
Triton 3.6 needs the old gfx906 fork patches. Old image existence is not ABI or
kernel evidence. No automatic legacy dependency downgrade is provided.

The main toolchain sources/artifacts are pinned; apt and transitive Python
dependencies still resolve from their indexes. A successful full build records
both build and runtime dependency freezes under `/usr/share/vllm-gfx906/`.
This is a reproducible source/toolchain recipe, not a bit-for-bit hermetic lock
of every transitive dependency. The full vLLM image has not been built here.

```bash
# Linux x86_64, Docker BuildKit; uses this checkout even from another cwd.
bash build_and_push_docker.sh
MAX_JOBS=4 bash build_and_push_docker.sh my-gfx906-test
# Publication is a separate, explicitly requested action; not run in this audit.
# bash build_and_push_docker.sh <versioned-tag> --push
```

The helper never clones `ai-infos/main`, never logs in, and rejects `latest`.
Dirty checkouts may build locally with a `.dirty` package suffix; publication
requires a clean checkout. Package versions are
`0.30.0+gfx906.<12-digit-commit>`; branch suffix `.x` is not a package version.
The Docker build asserts that the wheel contains `_gfx906_fa_C*.so` and the
backend, the Rust frontend and every registered Rust Python extension, and that
the sdist includes the vendored attention kernel source. Rust is required for
this image build, rather than silently skipped as an optional wheel component.

For an editable install inside the pinned build environment, after installing
the pinned Triton/flash-attention wheels and `requirements/rocm.txt`:

```bash
uv venv --python 3.12 --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements/lint.txt
.venv/bin/python -m pre_commit install
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_ROCM_ARCH=gfx906 VLLM_TARGET_DEVICE=rocm
export VLLM_VERSION_OVERRIDE="0.30.0+gfx906.$(git rev-parse --short=12 HEAD)"
export VLLM_RS_BUILD_VERSION="$VLLM_VERSION_OVERRIDE"
bash build_rust.sh
uv pip install --python .venv/bin/python --no-build-isolation -e .
.venv/bin/python setup.py build_ext --inplace
```

## Validation record (2026-10-08)

Validation ran under WSL Ubuntu 24.04 / Python 3.12.3 with a uv-managed `.venv`
and CPU Torch 2.13.0, plus the pinned Docker toolchain for HIP compilation.
Pre-commit and its commit hooks were installed. Source distribution and host
checks do not substitute for a linked HIP wheel or serving tests.

| Check | Result / scope |
| --- | --- |
| Host capabilities, cache strides, backend registration and metadata, padding/schema | 51 passed. |
| Additional CPU attention padding math | 4 passed, 29 deselected; float64 unquantized reference only. |
| MiniMax-M3 FP32/dtype aliases and GPTQ configuration | 13 passed, 2 skipped, 1 deselected. |
| FP8 FN/FNUZ decoder against CPU PyTorch | 2 passed; every byte, exact finite sign bits and NaNs. |
| Real PyTorch HIP conversion | 3 passed plus 1 CMake-path case passed separately; in-source, out-of-tree, unchanged source and relative build-tree paths. |
| Changed Python lint | No new Ruff diagnostics across 15 files against the pinned reference; 11 inherited diagnostics remain in edited reference files. Selected pre-commit Ruff check/format hooks passed for new helpers and compact regression files. The additional source-conversion test and helper pass standalone Ruff check/format. |
| Type check | `mypy --follow-imports=skip vllm/utils/gfx906.py`: success, one file. Full-project typing was not run. |
| Static merge sweep | Conflict-marker, F821/F811 and in-memory syntax checks passed. Duplicate-conflict-file sweep was inapplicable to this fast-forward. |
| Shell / whitespace | Bash syntax, ShellCheck 0.11.0 and `git diff --check` passed; changed native lines formatted with clang-format 21.1.2. |
| Build helper | Docker argument recorder passed local-context/versioned-default/no-implicit-push and invalid-input/dirty-push rejection checks. No registry login or real push. |
| Docker frontend / base | BuildKit `--check` passed without warnings; immutable `base` stage built and observed Torch/HIP versions matched. |
| HIP device compilation | FA launcher, Q8 quantizer, gather, GPTQ, max-ILP twin and native MoE translation units compiled for gfx906. No link or device execution. |
| ISA inspection | Packed dot4/half2 probe generated supported instructions and wave64; no MFMA/WMMA. Both FP8 decoders compiled offline for gfx906 with Triton 3.8.0, zero shared memory and no MFMA/WMMA. This used the PyPI compiler for the check, not the source-built image. |
| Packaging | Real sdist and empty-target Python wheel built as `0.30.0+gfx906.57530039efaf`; contents/version assertions passed. Sdist contains vendored FA, GPTQ/MoE, Rust build sources and new Python fallbacks. Wheel contains Python backends/fallbacks, **no native shared objects**. |

The two inherited MiniMax skips describe a stale pre-v0.30 fused-op fake and a
stale ROCm cache-layout fake; they were not changed into passing mocks. The
deselected GPTQ case requires a device kernel. Host runs use `--noconftest` to
avoid the global device-testing setup while exercising the production host
helpers, builder, gather and configuration code directly.
The full all-file pre-commit suite was not run; imported formatting/lint debt
was retained instead of bundled into this upgrade. Windows cannot execute the
installed Linux-venv hook shebang, so equivalent selected checks ran under WSL
and the Windows commits bypass those local hook entry points. Generated-by
trailers retain attribution; no human review or signature is fabricated.

Reproduce host checks in the project uv environment (CPU Torch 2.13.0+cpu,
torchvision 0.28.0+cpu, common/lint requirements, pytest and Triton 3.8.0):

```bash
VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest --noconftest \
  tests/utils/test_gfx906.py \
  tests/kernels/attention/test_gfx906_head_dim_pad.py -q -ra
# The combined command now includes the four padding-math cases (55 total).
.venv/bin/python -m pytest --noconftest \
  tests/utils/test_rocm_source_conversion.py -q -ra
VLLM_TARGET_DEVICE=cpu .venv/bin/python -m pytest --noconftest \
  tests/models/minimax_m3/test_fp32_kv_config.py \
  tests/quantization/test_auto_gptq.py -k 'not quantization_method' -q -ra
TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  --noconftest tests/kernels/quantization/test_gfx906_fp8_decode.py -q -ra
.venv/bin/python -m mypy --follow-imports=skip vllm/utils/gfx906.py
bash -n build_and_push_docker.sh build_rust.sh docker/gfx906-build.env
.venv/bin/shellcheck --norc -x -e SC1091 \
  build_and_push_docker.sh build_rust.sh
git diff --check
```

ShellCheck uses `--norc` because the imported Windows `.shellcheckrc` has CRLF
and otherwise fails parsing before checking the scripts. SC1091 is excluded
for the sourced pin file; that file is checked with Bash syntax.

The static sweep used `.agents/skills/upstream-merge/scripts/post-merge-sweep.sh`
(LF copy on Windows). Compilation used the pinned base, `hipcc -O3
--offload-arch=gfx906 -std=c++17`, and the following sources/options:

```bash
# Inside the pinned base with this checkout at /src, cwd /src:
hipcc -O3 --offload-arch=gfx906 -std=c++17 -Icsrc/gfx906_fa/kernel \
  -c -x hip csrc/gfx906_fa/gfx906_fa_launcher.cu -o /tmp/fa.o
# Same flags for gfx906_fa_quant.cu; gather adds -DGGML_HIP_GFX906.
/opt/vllm-venv/bin/python cmake/hipify.py -p /src/csrc -o /src/build/csrc \
  csrc/libtorch_stable/quantization/gptq/q_gemm.cu \
  csrc/libtorch_stable/quantization/gptq/q_gemm_m1_maxilp.cu \
  csrc/rocm/moe_q_gemm_gfx906.cu
# Compile those generated .hip files with the same gfx906 flags, -DUSE_ROCM,
# -Ibuild/csrc and Torch include, Torch csrc/api/include and Python include dirs.
```

Compiler warnings remain: FA occupancy targets were not always met (17 gather
warnings), GPTQ ignores two HIP status returns, and Torch headers warn about
C++20 bit-field initializers under C++17. They are not runtime correctness or
performance evidence. The max-ILP clone was updated alongside its parent.

Packaging used `VLLM_TARGET_DEVICE=empty`, the explicit version override and
`.venv/bin/python -m build --sdist/--wheel --no-isolation`. To avoid slow Windows
mount copies, the final packaging probe rebuilt an extracted source snapshot
on Linux storage after copying all follow-up files. Rust compilation was
optionally skipped there because no host Rust compiler was installed. These
artifacts are local, ignored, unpublished **payload probes**, not release/GPU
wheels. Full source-built Triton, Rust, linked native vLLM/FA wheel, final image
and container-serving smoke checks remain unverified. The Docker recipe makes
native FA and Rust payload presence mandatory when that full build is run.

## Runtime uncertainties

- No gfx906 device was available. All GPU tests, model evaluations, graph
  replay/capture, peak memory, serving A/B, and TP/P2P remain unverified.
- Noncausal sliding-window batches still attend a superset in the reference
  wrapper. For exact windowed bidirectional semantics select `TRITON_ATTN`;
  disable the custom noncausal claim with `GFX906_FA_NO_NONCAUSAL=1` for automatic
  fallback. Tensor causality is rejected regardless of this switch.
- The Q8 side cache aliases cache identity and persistent allocations; shared
  gather/Q buffers, their growth between capture and prefill, prefix reuse,
  changing sequence bounds, and split-K empty partitions need the existing
  device regression suite. CPU checks exercise only host contracts/math.
- Manual AITER/BF16/FP8/matrix-core backend choices and newly upstream-supported
  models/kernels are not automatically certified for gfx906. FP32 cache support
  can narrow Q/K/V for compute and is not an end-to-end FP32 guarantee.
- Native guards and GPTQ initialization changes require linked-device tests and
  serving performance measurements. Preserve the reference's
  [REL30-1 V1-only align gate](RELEASE-0.30.0-final.md).

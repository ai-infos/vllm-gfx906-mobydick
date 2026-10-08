# PR #22: fp32 GDN decode readout integration

The precision issue remains present on the v0.30.0 main baseline
`a2df265ba8b5ffb407aadf6d71ea83767bb38158`. The original
[PR #22](https://github.com/ai-infos/vllm-gfx906-mobydick/pull/22) and
[upstream #54146](https://github.com/vllm-project/vllm/pull/54146) by **Jack Danger**
identify the readout allocation defect. This follow-up adapts that idea to the
current packed decode, layer output buffer, mixed-batch merge and warmup.
AI assistance was used. This is local fork work; no duplicate upstream PR is
proposed, and no human review or model evaluation is claimed.

## Why the original patch is insufficient

The generic wrapper accumulates in fp32, then originally stores in `q.dtype`.
Widening only that allocation leaves the later activation-dtype `core_attn_out`
copy before normalization. Default non-spec decode instead uses the packed
kernel, which stores directly into that same layer buffer. Speculative output
can also become fp32 while prefill stays fp16; `index_copy_` needs matching
source and destination dtypes.

For a deterministic counterexample, let q=e1, k=e2, v=0, K=128, and every
state row have first component +/-2,000,000. Set A_log=a=b=dt_bias=0. Decay is
one half and the delta correction is zero. With the kernel's L2 epsilon, the
readout is approximately +/-88,388.3, exceeding fp16's finite range. Narrowing
before RMSNorm yields infinity and then NaN; keeping fp32 through normalization
allows a finite fp16 normalized result.

## Behavior and compatibility

- Widen only fp16 activation / fp32 recurrent-state readouts. Other dtype
  combinations retain their existing output dtype.
- Allocate the generic result and standard GPU layer output with the same
  policy. The packed kernel already writes using its destination pointer type.
- Merge mixed outputs using the destination dtype, converting sources before
  indexed copies.
- Keep z in activation precision; cast the normalized result back to that dtype
  before projection. Cache dtype/layout, CLI and kernel signatures are unchanged.
- Bypass external ROCm AITER for this affected combination, using the in-tree
  generic/packed implementation. Warmup includes the fp32-input/fp16-gate norm.
- Retain padding initialization and invalid-state guards. This does not repair
  prefill intermediate overflow or recycled-state corruption, nor prove that
  every repeated-exclamation failure is solved.

Fp32 output storage doubles readout bytes for the affected configuration and
adds a post-normalization conversion. No memory or speed measurement is claimed.
Source review found no added network access, subprocess execution,
deserialization or external dependency. Kernel recurrence and state-index guards
are unchanged; this is not a hardware memory-safety certification.

## Validation

The CPU regression runs the real GPU forward, kernel wrappers, output merge,
native normalization and torch projections. Only leaf device computations are
replaced with the analytical recurrence and bounded prefill/identity convolution.
The host fixture selects CPU tensors, native normalization and the non-gfx906
allocation branch independently of the machine running the tests.
It covers both signs, packed/generic decode, speculative/prefill and
decode/prefill batches, padding, and unchanged bf16/fp16-state behavior.
Additional device tests exercise both readout kernels and mixed-dtype norm.

Run host checks in a uv-managed environment with the common/lint requirements,
CPU Torch 2.13 and Triton 3.8 (Python-only editable install for host tests):

```bash
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
    --noconftest tests/kernels/mamba/test_gdn_readout_dtype.py \
    tests/model_executor/test_qwen_triton_warmup.py -q -ra
git diff --check
bash -n build_and_push_docker.sh build_rust.sh docker/gfx906-build.env
```

The empty visible-device setting permits real Triton imports for CPU wrapper
and compile-key tests; it does not create a GPU or execute device kernels.
On a GPU host also run the existing sigmoid, packed decode and layernorm suites,
then dense/MoE model gates and eager/graph serving including speculative decode.

Host/compiler checks on 2026-10-08 used Python 3.12.3, CPU Torch 2.13.0,
Triton 3.8.0 and an isolated Linux source snapshot of the changed files:

| Check | Result |
| --- | --- |
| CPU layer regression and warmup; existing packed-decode suite | 27 passed, 9 GPU-dependent cases skipped |
| New sigmoid/packed and normalization device regressions | 6 skipped without a GPU |
| Offline Triton compilation targeting HIP/gfx906, wave64 | Generic, speculative, packed fp32 readout and fp32-input/fp16-gate norm all compiled |
| CPU Triton interpreter on the new device regressions | All 6 positive/negative readout and normalization checks passed |
| Original main allocation replayed through the host regression | Rejected: readout became fp16 infinity before normalization |
| Ruff 0.14.0 check/format on changed non-vendored Python files | Passed |
| Markdownlint CLI2 0.21.0 on this review record | Passed |
| Local SPDX, lazy-import, forbidden-import, CUDA-API, boolean-context and config hooks | Passed |
| Bash syntax and ShellCheck on the build helpers | Passed |
| Source distribution and Python-only wheel (`VLLM_TARGET_DEVICE=empty`) | Built successfully; wheel contains the corrected readout code |
| `git diff --check` | Passed |

The interpreter check used CPU tensors and one row per normalization block.
Packed-kernel `exp`/`log` aliases were rebound to the same Triton language
operations so the interpreter could resolve them. Production code was unchanged.
Offline compilation used the PyPI Triton compiler, not the image's source-built
Triton. These checks establish numerical and compiler evidence, not hardware
execution or the full native build.

The repository's mypy 1.20.2 / Python 3.12 hook passed the changed tests.
Production checks reported 31 errors in the existing attention/warmup modules;
repeating the check with unchanged main sources produced the same findings,
with no additional production errors. Selected hook commands ran directly;
the complete pre-commit suite was not run.

GPU execution, model evaluations, graph replay, peak memory and serving
performance are **unverified**. No Docker image was built or published as part
of this integration.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""NH-4: grouped Mixer2RMSNormGated through the fused triton kernel.

`Mixer2RMSNormGated.forward_cuda` previously only used the fused
`rms_norm_gated` triton kernel for `n_groups == 1`; every grouped config
(Nemotron-H: 8 groups of 1024) fell back to `forward_native`, a ~7-launch
eager elementwise chain that is launch-tail dominated at decode sizes.
With `VLLM_GFX906_MAMBA_FUSED_GROUP_NORM=1` the grouped case routes through
the same fused kernel, which already supports per-group reduction via
`group_size`.

These tests pin:
  * fused-grouped output == forward_native reference (fp32/fp16/bf16),
    at Nemotron geometry and several other group layouts;
  * the gate falls back to native when the per-rank hidden dim is not an
    integer multiple of the group size (partial groups);
  * `use_rms_norm=False` behavior is unchanged.

Reference semantics (forward_native):
    x = x * silu(gate.to(fp32))                      # fp32, then cast
    var = mean over each group of 1024 (fp32)
    out = weight * (x * rsqrt(var + eps)).to(dtype)
"""

import os
import unittest.mock as mock

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 gated-norm Triton kernel requires a CUDA-alike device.",
)


def _make_norm(full_hidden_size: int, full_n_groups: int, tp_size: int = 1):
    with (
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        norm = Mixer2RMSNormGated(
            full_hidden_size=full_hidden_size,
            full_n_groups=full_n_groups,
            use_rms_norm=True,
            eps=1e-5,
        )
    return norm


# ---------------------------------------------------------------------------
# Direct kernel-level check: fused grouped rms_norm_gated vs eager reference.
# This is the numerical contract; the model-level tests below pin dispatch.
# ---------------------------------------------------------------------------


def test_fused_grouped_kernel_matches_eager_reference(default_vllm_config):
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
        rms_norm_gated,
    )

    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    # Nemotron-H per-rank geometry at TP=1: 8 groups of 1024.
    M, N = 4, 8192
    group_size = 1024
    eps = 1e-5
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(M, N, device=device, dtype=dtype)
        z = torch.randn(M, N, device=device, dtype=dtype)
        w = torch.rand(N, device=device, dtype=dtype) + 0.5

        # Eager reference (fp32 intermediates, like forward_native).
        xf = x * torch.nn.functional.silu(z.to(torch.float32))
        xg = xf.view(M, N // group_size, group_size)
        var = xg.pow(2).mean(-1, keepdim=True)
        wg = w.to(torch.float32).view(N // group_size, group_size)
        ref = (wg * (xg * torch.rsqrt(var + eps))).view(M, N)

        got = rms_norm_gated(
            x, w, bias=None, z=z, eps=eps, group_size=group_size,
            norm_before_gate=False,
        )
        if dtype == torch.float32:
            # fp32: not bit-equal — tl.sum's reduction order over 1024
            # elements differs from torch.mean's; ~2e-6 max is the
            # measured floor. Tolerances are loose enough to absorb
            # reordering, tight enough to catch a wrong group layout
            # (which would be O(1) errors).
            got_cmp, ref_cmp = got.to(torch.float32), ref
            atol, rtol = 1e-4, 5e-5
        else:
            # 16-bit: the kernel rounds its output to dtype once; put both
            # sides on the same grid so the residual is pure reduction-order
            # noise (a value flips at most one ulp when it lands near a
            # rounding boundary).
            got_cmp, ref_cmp = got, ref.to(dtype)
            atol, rtol = (5e-3, 1e-3) if dtype == torch.float16 else (1e-2, 2e-2)
        torch.testing.assert_close(got_cmp, ref_cmp, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "full_hidden_size,full_n_groups",
    [
        (8192, 8),   # Nemotron-H at TP=1
        (4096, 4),   # Nemotron-H per rank at TP=2
        (512, 4),    # small grouped layout
        (1024, 2),
        (8192, 16),  # many groups
    ],
)
def test_fused_grouped_matches_native_reference(
    default_vllm_config, full_hidden_size: int, full_n_groups: int
):
    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    M = 3
    # (norm eps is the _make_norm default of 1e-5)

    norm = _make_norm(full_hidden_size, full_n_groups, tp_size=1)
    w = torch.rand(full_hidden_size, device=device, dtype=torch.float16) + 0.5
    norm.weight.data = w
    x = torch.randn(M, full_hidden_size, device=device, dtype=torch.float16)
    gate = torch.randn(M, full_hidden_size, device=device, dtype=torch.float16)

    ref = norm.forward_native(x, gate).clone()

    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "1"
    try:
        got = norm.forward_cuda(x, gate)
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-3)


def test_gate_falls_back_to_native_on_partial_groups(default_vllm_config):
    """per-rank hidden not a multiple of group size -> fused path refused.

    Proven with a sentinel: if the gated branch (wrongly) fired, it would
    call rms_norm_gated and raise _FusedCalled; instead forward_cuda must
    proceed down the native path (which itself rejects this degenerate
    geometry, so we only assert that the sentinel was NOT hit).
    """

    class _FusedCalled(Exception):
        pass

    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    norm = _make_norm(64, 3, tp_size=1)
    norm.weight.data = (
        torch.rand(64, device=device, dtype=torch.float16) + 0.5
    )
    x = torch.randn(2, 64, device=device, dtype=torch.float16)
    gate = torch.randn(2, 64, device=device, dtype=torch.float16)

    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "1"
    try:
        with mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2.rms_norm_gated",
            side_effect=_FusedCalled,
        ):
            try:
                norm.forward_cuda(x, gate)
                native_raised = False
            except _FusedCalled:
                pytest.fail("gate fired the fused path on partial groups")
            except RuntimeError:
                # native's grouped view rejects 64 = 3 x 21 + 1 — expected;
                # what matters is it was NOT the sentinel.
                native_raised = True
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    assert native_raised, "neither path ran as expected"


def test_env_off_keeps_native_path(default_vllm_config):
    """Default (env unset/0): grouped case must be bit-equal to native."""
    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    norm = _make_norm(8192, 8, tp_size=1)
    w = torch.rand(8192, device=device, dtype=torch.float16) + 0.5
    norm.weight.data = w
    x = torch.randn(2, 8192, device=device, dtype=torch.float16)
    gate = torch.randn(2, 8192, device=device, dtype=torch.float16)

    ref = norm.forward_native(x, gate).clone()
    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "0"
    try:
        got = norm.forward_cuda(x, gate)
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_no_rms_norm_unchanged(default_vllm_config):
    """use_rms_norm=False: the gate must not alter that path at all.

    (Note: forward_cuda's no-rms-norm expression is NOT bit-identical to
    forward_native's — pre-existing, different fp32/fp16 rounding order —
    so the reference here is the env-OFF forward_cuda result, which proves
    the gate leaves the path untouched.)
    """
    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    with (
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        norm = Mixer2RMSNormGated(
            full_hidden_size=1024,
            full_n_groups=8,
            use_rms_norm=False,
        )
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    gate = torch.randn(2, 1024, device=device, dtype=torch.float16)

    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "0"
    try:
        ref = norm.forward_cuda(x, gate).clone()
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "1"
    try:
        got = norm.forward_cuda(x, gate)
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_tp2_nemotron_geometry_dispatch(default_vllm_config):
    """Actual TP=2 Nemotron-H dispatch (8 groups of 1024, per-rank 4096).

    Mirrors production serving geometry: full_hidden_size=8192 with
    tp_size=2 -> per_rank_hidden_size=4096, n_groups=8. The fused path
    must fire (env on) and match forward_native within fp16 tolerance.
    """
    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    with (
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=2,
        ),
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        norm = Mixer2RMSNormGated(
            full_hidden_size=8192,
            full_n_groups=8,
            use_rms_norm=True,
            eps=1e-5,
        )
    assert norm.per_rank_hidden_size == 4096
    w = torch.rand(4096, device=device, dtype=torch.float16) + 0.5
    norm.weight.data = w
    x = torch.randn(2, 4096, device=device, dtype=torch.float16)
    gate = torch.randn(2, 4096, device=device, dtype=torch.float16)

    ref = norm.forward_native(x, gate).clone()
    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "1"
    try:
        got = norm.forward_cuda(x, gate)
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]

    torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-3)


def test_tp_driven_partial_groups_refused(default_vllm_config):
    """TP-driven partial groups: 8 groups of 1024 at tp_size=16.

    per_rank_hidden_size = 512 < group_size = 1024, so the fused path
    must be refused even with env on (sentinel proves it never fires);
    native proceeds down its redundant all-gather branch.
    """

    class _FusedCalled(Exception):
        pass

    torch.manual_seed(0)
    device = f"{DEVICE}:0"
    with (
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=16,
        ),
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        norm = Mixer2RMSNormGated(
            full_hidden_size=8192,
            full_n_groups=8,
            use_rms_norm=True,
            eps=1e-5,
        )
    assert norm.per_rank_hidden_size == 512

    os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"] = "1"
    try:
        with mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2.rms_norm_gated",
            side_effect=_FusedCalled,
        ):
            try:
                norm.forward_cuda(
                    torch.randn(1, 512, device=device, dtype=torch.float16),
                    torch.randn(1, 512, device=device, dtype=torch.float16),
                )
            except _FusedCalled:
                pytest.fail("gate fired the fused path on TP-driven partial groups")
            except AssertionError as exc:
                # Expected: native's redundant all-gather branch needs a real
                # 16-rank process group, which this unit test does not have.
                # The contract is that we got HERE (native), not to the fused
                # path — so any non-sentinel failure from native passes.
                assert "tensor model parallel" in str(exc)
    finally:
        del os.environ["VLLM_GFX906_MAMBA_FUSED_GROUP_NORM"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

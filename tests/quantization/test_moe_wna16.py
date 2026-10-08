# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    convert_to_wna16_moe_kernel_format,
    map_wna16_backend,
)
from vllm.model_executor.layers.quantization import moe_wna16
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.platforms import current_platform


def test_map_wna16_backend_supports_triton():
    assert map_wna16_backend("triton") == WNA16MoEBackend.TRITON


@pytest.mark.parametrize(
    "config",
    [
        {"desc_act": True, "group_size": 128},
        {
            "desc_act": False,
            "group_size": 128,
            "dynamic": {r"+:model\.layers\.0\..*": {"desc_act": True}},
        },
    ],
)
def test_moe_wna16_rejects_gptq_group_activation_order(config):
    config.update({"quant_method": "gptq", "bits": 4, "sym": True})
    with pytest.raises(ValueError, match="group activation ordering"):
        MoeWNA16Config.from_config(config)


def test_moe_wna16_accepts_channelwise_gptq_activation_order():
    config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": -1,
        "desc_act": True,
        "sym": True,
    }
    assert MoeWNA16Config.is_moe_wna16_compatible(config)
    MoeWNA16Config.from_config(config)


@pytest.mark.parametrize(
    ("backend", "quant_config", "may_have_zp", "may_have_bias", "expected"),
    [
        (
            WNA16MoEBackend.TRITON,
            AutoAWQConfig(4, 128, True, False),
            True,
            False,
            "AutoAWQ weight layout",
        ),
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            False,
            False,
            "activation ordering",
        ),
        # DYNAMIC is a distinct actorder value (not an alias of GROUP):
        # the Triton gate must reject it too, or a g_idx-ordered
        # checkpoint reaches the repack and mis-dequants silently.
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.DYNAMIC,
            ),
            False,
            False,
            "activation ordering",
        ),
        # WEIGHT-ordered asymmetric CT is g_idx-free: the Triton gate
        # must accept it (reaches the repack, which is the supported
        # CT-asym fallback).
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.WEIGHT,
            ),
            False,
            False,
            None,
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            True,
            "bias",
        ),
        (
            WNA16MoEBackend.MARLIN,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
        (
            WNA16MoEBackend.GFX906_HIP,
            # Symmetric GPTQ has no stored zero points.
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            False,
            "zero points are required",
        ),
        (
            WNA16MoEBackend.RDNA3,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            False,
            "compressed-tensors",
        ),
        (
            WNA16MoEBackend.RDNA3,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
            ),
            True,
            False,
            "asymmetric",
        ),
        (
            WNA16MoEBackend.RDNA3,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
                dynamic=False,
            ),
            False,
            False,
            "group-wise scales",
        ),
    ],
)
def test_wna16_oracle_rejects_incompatible_quant_structures(
    backend, quant_config, may_have_zp, may_have_bias, expected
):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config()

    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
        allow_tile_padding=True,
    )

    if expected is None:
        assert reason is None
        return
    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize(
    "quant_config",
    [
        AutoAWQConfig(4, 128, True, False),
        MoeWNA16Config(
            linear_quant_method="awq",
            weight_bits=4,
            group_size=128,
            has_zp=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
    ],
)
def test_gfx906_hip_oracle_accepts_awq_style_zero_points(quant_config):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Realistic Qwen3.5-A3B MoE shapes: the gfx906 kernel's shape gate
    # (intermediate % 8, hidden % group_size) rejects the dummy 1x1 config.
    moe_config = make_dummy_moe_config(
        hidden_dim=2048, intermediate_size=1024)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


@pytest.mark.parametrize(
    "hidden_dim, intermediate_size, group_size, expected",
    [
        (2048, 10, 128, "intermediate size must be a multiple of 8"),
        (200, 1024, 128, "hidden size must be divisible by the group size"),
    ],
)
def test_gfx906_hip_oracle_shape_gate(hidden_dim, intermediate_size,
                                      group_size, expected):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config(
        hidden_dim=hidden_dim, intermediate_size=intermediate_size)
    quant_config = AutoAWQConfig(4, group_size, True, False)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize(
    ("quant_config", "expected"),
    [
        # W8A16: the kernel is W4A16 only.
        (
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
            ),
            "4-bit weights",
        ),
        # Dynamic scales: the kernel consumes static per-group scales.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=True,
                group_size=128,
            ),
            "static (non-dynamic) scales",
        ),
        # Group size not a multiple of 32: the kernel's per-32-K-slice
        # group tracking contract is violated (group 64 IS accepted since
        # the Nemotron-3.5-Lightning onboarding; see the accept test below).
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=48,
            ),
            "multiple of 32",
        ),
        # Channel strategy: no [E, G, N] group scales for the kernel.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
                dynamic=False,
            ),
            "group strategy",
        ),
        # g_idx activation ordering: weights are stored in original
        # column order and need a runtime reordering the kernel lacks.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            "g_idx activation ordering",
        ),
        # DYNAMIC is an alias of GROUP with the same runtime contract.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.DYNAMIC,
            ),
            "g_idx activation ordering",
        ),
        # WEIGHT is format-identical to no activation ordering: the
        # repack consumes the stored weights in natural order, so the
        # gate must not reject it.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.WEIGHT,
            ),
            None,
        ),
    ],
)
def test_gfx906_hip_oracle_symmetric_no_zp_contract_gate(quant_config, expected):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Qwen3.5-A3B-shaped config so the shape gate passes and the no-zp
    # gate is what fires.
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=1024)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    if expected is None:
        assert reason is None
        return
    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_gfx906_hip_oracle_accepts_symmetric_no_zp(group_size):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Gemma-4-26B-A4B-shaped config (group-32 symmetric no-zp): the
    # shipped no-zp path (180f030ee3) must keep passing the gate. Group
    # 64 is the Nemotron-3.5-Lightning-30B contract (K=2688/1856 both
    # divisible by 64; any positive multiple of 32 is accepted).
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=704)
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=group_size,
    )

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


@pytest.mark.parametrize(
    ("quant_config", "expected"),
    [
        # W8A16: the kernel is W4A16 only.
        (
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
            ),
            "4-bit weights",
        ),
        # Dynamic scales: the kernel consumes static per-group scales.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=True,
                group_size=128,
            ),
            "static (non-dynamic) scales",
        ),
        # Group size outside the validated 32/128 set.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=64,
            ),
            "group size 32 or 128",
        ),
        # Channel strategy: no [E, G, N] group scales for the kernel.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=False,
                dynamic=False,
            ),
            "group strategy",
        ),
        # g_idx activation ordering: weights are stored in original
        # column order and need a runtime reordering the kernel lacks.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            "g_idx activation ordering",
        ),
        # DYNAMIC is an alias of GROUP with the same runtime contract.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.DYNAMIC,
            ),
            "g_idx activation ordering",
        ),
        # WEIGHT is format-identical to no activation ordering: the
        # repack consumes the stored weights in natural order, so the
        # gate must not reject it.
        (
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=False,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.WEIGHT,
            ),
            None,
        ),
    ],
)
def test_gfx906_hip_oracle_asym_ct_contract_gate(quant_config, expected):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Qwen3.5-A3B-shaped config so the shape gate passes and the
    # asymmetric-CT gate is what fires.
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=1024)

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    if expected is None:
        assert reason is None
        return
    assert reason is not None
    assert expected in reason


@pytest.mark.parametrize("group_size", [32, 128])
def test_gfx906_hip_oracle_accepts_asym_ct(group_size):
    from tests.kernels.moe.utils import make_dummy_moe_config

    # Ornith-1.5-35B-A3B-shaped config (group-32 asymmetric
    # pack-quantized): the stored int32-packed zps arrive K-first
    # [E, G, N/8] and the gate must accept the config.
    moe_config = make_dummy_moe_config(hidden_dim=2048, intermediate_size=1024)
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=False,
        dynamic=False,
        group_size=group_size,
    )

    reason = _backend_incompatibility_reason(
        backend=WNA16MoEBackend.GFX906_HIP,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=True,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


def test_compressed_tensors_weights_are_transposed_for_triton():
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
    )
    w13 = torch.arange(16, dtype=torch.int32).reshape(1, 2, 8)
    w2 = torch.arange(12, dtype=torch.int32).reshape(1, 2, 6)
    w13_scale = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
    w2_scale = torch.arange(18, dtype=torch.float16).reshape(1, 3, 6)

    converted = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    assert converted is not None
    assert torch.equal(converted[0], w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[1], w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[2], w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(converted[3], w2_scale.transpose(1, 2).contiguous())


def test_moe_wna16_setup_forwards_selected_backend(monkeypatch):
    method = object.__new__(MoeWNA16Method)
    method.experts_cls = object
    method.wna16_backend = WNA16MoEBackend.HUMMING
    method.moe = object()
    quant_config = object()
    method.get_fused_moe_quant_config = lambda layer: quant_config
    layer = SimpleNamespace(_expert_routing_tables=lambda: (None, None, None))
    captured = {}
    kernel = object()

    def fake_make_wna16_moe_kernel(**kwargs):
        captured.update(kwargs)
        return kernel

    monkeypatch.setattr(moe_wna16, "make_wna16_moe_kernel", fake_make_wna16_moe_kernel)

    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert captured["backend"] == WNA16MoEBackend.HUMMING


def test_moe_wna16_humming_adapter_repacks_uint8_tensors():
    qweight = torch.arange(32, dtype=torch.uint8).reshape(1, 4, 8)
    scales = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    qzeros = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

    converted = _convert_moe_wna16_humming_tensors(
        {"qweight": qweight, "scales": scales, "qzeros": qzeros},
        has_zero_point=True,
    )

    assert torch.equal(converted["weight"], qweight.view(torch.int32))
    assert converted["weight"].shape == (1, 4, 2)
    assert torch.equal(converted["weight_scale"], scales)
    expected_qzeros = (
        qzeros.transpose(-1, -2)
        .contiguous()
        .view(torch.int32)
        .transpose(-1, -2)
        .contiguous()
    )
    assert torch.equal(converted["zero_point"], expected_qzeros)
    assert converted["zero_point"].shape == (1, 2, 2)


def test_moe_wna16_uses_humming_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import humming_utils

    method = object.__new__(MoeWNA16Method)
    method.wna16_backend = WNA16MoEBackend.HUMMING
    layer = object()
    quant_config = object()
    monkeypatch.setattr(
        humming_utils,
        "get_humming_moe_quant_config",
        lambda actual_layer, *args, **kwargs: (
            quant_config if actual_layer is layer else None
        ),
    )

    assert method.get_fused_moe_quant_config(layer) is quant_config


def test_repack_qzeros_kfirst_for_triton_matches_kernel_indexing():
    """The Triton int4 WNA16 MoE kernel reads column n's zero point from
    word ``n // 2`` (axis 1), nibble ``(n % 2) * 4``, group ``g`` (axis 2);
    the repack must make the checkpoint's K-first 8-zp-per-word packing
    satisfy exactly that indexing."""
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _repack_qzeros_kfirst_for_triton,
    )

    assert _repack_qzeros_kfirst_for_triton(None, 128) is None

    E, G, N = 3, 16, 128  # N/8 = 16 words
    zp = (
        torch.randint(0, 2**32, (E, G, N // 8), dtype=torch.int64)
        .to(torch.int32)
        .contiguous()
    )
    out = _repack_qzeros_kfirst_for_triton(zp, N)
    assert out.shape == (E, N // 2, G)
    assert out.dtype == torch.int32
    # Logical [E, N // 2, G] backed by physical [E, G, N // 2]: the kernel
    # walks axis 1 with axis 2 fixed, so axis 1 must be the contiguous one.
    assert out.stride() == (G * (N // 2), 1, N // 2)

    n = torch.arange(N)
    # Checkpoint convention: column n = word n // 8, nibble (n % 8) * 4.
    expected = (zp[:, :, n // 8] >> ((n % 8) * 4)) & 0xF  # [E, G, N]
    got = (out[:, n // 2, :] >> ((n % 2) * 4)[:, None]).transpose(1, 2) & 0xF
    assert torch.equal(got, expected)  # [E, G, N]


@pytest.mark.parametrize(
    ("zp_shape", "zp_dtype", "n_out"),
    [
        # Packed width not N_out // 8 (e.g. a quantization source whose
        # qzeros use a different packing convention): fail closed.
        ((3, 16, 8), torch.int32, 128),
        ((3, 16, 64), torch.int32, 128),
        # N_out not a multiple of 8: the 8-per-word packing is impossible.
        ((3, 16, 1), torch.int32, 10),
        # Non-int32 storage.
        ((3, 16, 16), torch.uint8, 128),
    ],
)
def test_repack_qzeros_kfirst_for_triton_rejects_unexpected_layouts(
    zp_shape, zp_dtype, n_out
):
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _repack_qzeros_kfirst_for_triton,
    )

    zp = torch.zeros(*zp_shape, dtype=zp_dtype)
    with pytest.raises(ValueError, match="qzeros must be K-first"):
        _repack_qzeros_kfirst_for_triton(zp, n_out)
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Compressed-tensors Humming WNA16 MoE requires CUDA",
)
@pytest.mark.parametrize("num_bits", [3, 5, 6, 7])
def test_compressed_tensors_wna16_moe_create_weights_uses_ceil_packed_shapes(
    num_bits,
):
    pytest.importorskip("humming")

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    quant_args = QuantizationArgs(
        num_bits=num_bits,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    moe_config = make_dummy_moe_config(
        num_experts=2,
        hidden_dim=256,
        intermediate_size=512,
    )
    moe_config.moe_backend = "humming"
    method = CompressedTensorsWNA16MoEMethod(quant_args, None, moe_config)
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=512,
        params_dtype=torch.float16,
    )

    packed_hidden = (256 * num_bits + 31) // 32
    packed_intermediate = (512 * num_bits + 31) // 32
    assert method.wna16_backend == WNA16MoEBackend.HUMMING
    assert layer.w13_weight_packed.shape == (2, 1024, packed_hidden)
    assert layer.w2_weight_packed.shape == (2, 256, packed_intermediate)
    assert layer.w13_weight_scale.shape == (2, 1024, 2)
    assert layer.w2_weight_scale.shape == (2, 256, 4)
    assert layer.w13_weight_packed.dtype is torch.int32
    assert layer.w2_weight_scale.dtype is torch.float16


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Compressed-tensors Humming WNA16 MoE requires CUDA",
)
def test_compressed_tensors_wna16_moe_converts_and_sets_up_humming_kernel():
    pytest.importorskip("humming")

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    quant_args = QuantizationArgs(
        num_bits=3,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    moe_config = make_dummy_moe_config(
        num_experts=2,
        hidden_dim=256,
        intermediate_size=512,
    )
    moe_config.moe_backend = "humming"
    method = CompressedTensorsWNA16MoEMethod(quant_args, None, moe_config)
    layer = torch.nn.Module()
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16
    layer.layer_name = "test.humming_moe"
    layer._expert_routing_tables = lambda: (None, None, None)

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=512,
        params_dtype=torch.bfloat16,
    )
    layer.cuda()
    for parameter in layer.parameters():
        parameter.data.zero_()

    method.process_weights_after_loading(layer)

    assert method.wna16_backend == WNA16MoEBackend.HUMMING
    assert method.moe_kernel is not None
    assert set(layer.weight_schemas) == {"w13", "w2"}
    assert set(layer.humming_configs) == {"w13", "w2"}
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
    assert layer.w13_weight.dtype is torch.int32
    assert layer.w2_weight.dtype is torch.int32


def test_moe_wna16_forwards_packed_modules_mapping_to_linear_delegate(monkeypatch):
    """The linear delegate must receive packed_modules_mapping.

    It is rebuilt from the raw HF quantization dict, which lists shard names and
    never fused ones, so without the mapping a fused layer resolves to
    `UnquantizedLinearMethod` and the checkpoint's qweight has nowhere to load.
    """
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    config = MoeWNA16Config(
        linear_quant_method="gptq",
        weight_bits=4,
        group_size=128,
        has_zp=False,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config={
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "quant_method": "gptq",
            # As emitted by AutoGPTQ: shard names, never the fused name.
            "modules_in_block_to_quantize": [["mlp.gate_proj", "mlp.up_proj"]],
        },
    )
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    seen: dict[str, dict[str, list[str]]] = {}
    monkeypatch.setattr(
        AutoGPTQConfig,
        "get_quant_method",
        lambda self, layer, prefix: seen.setdefault(
            "mapping", self.packed_modules_mapping
        ),
    )
    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    config.get_quant_method(layer, "model.layers.0.mlp.gate_up_proj")

    assert seen["mapping"] == {"gate_up_proj": ["gate_proj", "up_proj"]}


def test_xpu_platform_supports_moe_wna16():
    """Regression guard for the XPU quantization allowlist."""
    try:
        from vllm.platforms.xpu import XPUPlatform
    except ImportError:
        pytest.skip("vllm_xpu_kernels not importable outside an XPU stack")

    assert "moe_wna16" in XPUPlatform.supported_quantization


def _channelwise_int4_args() -> QuantizationArgs:
    """A per-channel int4 checkpoint, which leaves ``group_size`` unset."""
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    assert args.group_size is None, "premise: CHANNEL leaves group_size unset"
    return args


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="check_moe_marlin_supports_config rejects every config on ROCm",
)
@pytest.mark.parametrize("backend", [WNA16MoEBackend.MARLIN, WNA16MoEBackend.TRITON])
def test_wna16_oracle_accepts_unset_group_size(backend):
    """Both backends must *accept* a per-channel config, not just survive it.

    -1 is a supported Marlin group size and the shapes below pass the Marlin
    tiling checks, while Triton never reads group_size for QuantizationArgs.
    A reason string from either backend would mean the unset group_size cost
    the layer its preferred kernel instead of raising TypeError.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config

    # hidden_dim % 128 and intermediate % 64 must hold, or the Marlin shape
    # check rejects the config before group_size is ever read.
    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=make_dummy_moe_config(
            num_experts=2, hidden_dim=256, intermediate_size=512
        ),
        quant_config=_channelwise_int4_args(),
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )

    assert reason is None


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Marlin is only a candidate WNA16 MoE backend on CUDA; elsewhere "
    "__init__ takes the non-Marlin branch, which rejects channelwise",
)
def test_compressed_tensors_wna16_moe_marlin_prep_with_unset_group_size():
    """Load a per-channel checkpoint through the Marlin path end to end.

    ``__init__`` and Marlin weight prep read ``group_size`` independently, so
    both have to normalise the unset value. The post-repack shapes prove prep
    ran with the Marlin K/N rather than merely returning something.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    num_experts, hidden_size, intermediate_size = 2, 256, 512
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
    )
    moe_config.moe_backend = "marlin"

    method = CompressedTensorsWNA16MoEMethod(_channelwise_int4_args(), None, moe_config)
    assert method.wna16_backend == WNA16MoEBackend.MARLIN
    assert method.group_size == -1

    layer = torch.nn.Module()
    layer.intermediate_size_per_partition = intermediate_size
    layer._expert_routing_tables = lambda: (None, None, None)
    method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        intermediate_size_full=intermediate_size,
        params_dtype=torch.bfloat16,
    )
    layer.cuda()
    for parameter in layer.parameters():
        parameter.data.zero_()

    method.process_weights_after_loading(layer)

    # gptq_marlin_moe_repack packs to (size_k // 16, size_n * 2) for int4; w13
    # is repacked with size_k=hidden_size, size_n=2*intermediate_size, and w2
    # the other way round. Channelwise keeps one scale group per channel.
    assert layer.w13_weight_packed.shape == (
        num_experts,
        hidden_size // 16,
        4 * intermediate_size,
    )
    assert layer.w2_weight_packed.shape == (
        num_experts,
        intermediate_size // 16,
        2 * hidden_size,
    )
    assert layer.w13_weight_scale.shape == (num_experts, 1, 2 * intermediate_size)
    assert layer.w2_weight_scale.shape == (num_experts, 1, hidden_size)

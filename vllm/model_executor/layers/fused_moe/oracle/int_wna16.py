# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
import sys
from enum import Enum
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    BatchedMarlinExperts,
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxint4_moe import (
    TrtLlmMxint4ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
    marlin_act_int8_process_scales,
    marlin_moe_padded_intermediate,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    moe_awq_to_marlin_zero_points,
    moe_packed_to_marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    pack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    HUMMING = "HUMMING"
    CPU = "CPU"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    TRITON = "TRITON"
    XPU = "XPU"
    EMULATION = "EMULATION"
    GFX906_HIP = "GFX906_HIP"
    RDNA3 = "RDNA3"


# WNA16 MoE backends whose kernels consume stored zero points (w1_zp /
# w2_zp) — the only ones an asymmetric (stored-zp) checkpoint may use.
# Single source of truth: the compressed-tensors asymmetric gate in
# CompressedTensorsWNA16MoEMethod.__init__ must fail closed for any
# backend outside this set, so a backend that later gains zp support is
# declared here rather than discovered in an unrelated assert.
WNA16_BACKENDS_WITH_STORED_ZP = frozenset(
    {
        WNA16MoEBackend.TRITON,
        WNA16MoEBackend.GFX906_HIP,
        # 0.29.0's int4 emulation experts consume the stored zero points by
        # baking them into the dequantized bf16 weights, so asymmetric
        # checkpoints are valid there too (the resulting tensors carry no
        # separate zp — see CompressedTensorsWNA16MoEMethod).
        WNA16MoEBackend.EMULATION,
    }
)


def _is_symmetric_no_zp(
    quant_config: QuantizationConfig | QuantizationArgs,
) -> bool:
    """Symmetric W4A16 with no stored zero points (compressed-tensors
    pack-quantized): dequant is (q - 8) * scale; the gfx906 kernel inlines
    the constant zero point 8 when the zp tensor is empty."""
    return isinstance(quant_config, QuantizationArgs) and quant_config.symmetric


def _gidx_actorder_reason(
    quant_config: QuantizationArgs,
    family: str,
) -> str | None:
    """Why this config's activation ordering needs the runtime g_idx
    reordering that the W4A16 MoE kernels (gfx906 custom and Triton
    fallback) cannot apply (None if the ordering is safe).

    Single source of truth for which compressed-tensors actorder values
    carry a g_idx — shared by the gfx906 symmetric/no-zp gate, the
    asymmetric-CT gate, and the Triton gate. `group` (and its `dynamic`
    spelling) store weights in original column order with a runtime g_idx
    permutation; `weight`/`static` are format-identical to natural order.
    A kernel without g_idx support would mis-dequant silently on the
    g_idx families, so every gate rejects them here.
    """
    if quant_config.actorder in ("group", "dynamic"):
        return f"{family} does not support g_idx activation ordering"
    return None


def _gfx906_no_zp_reason(
    quant_config: QuantizationConfig | QuantizationArgs,
) -> str | None:
    """Why a symmetric no-zp checkpoint cannot use the gfx906 W4A16
    kernel (None if it can).

    The kernel dequants (q - 8) * scale with static per-group scales,
    tracks group boundaries on 32-K slice boundaries, and has no g_idx
    activation reordering. A symmetric checkpoint outside that contract
    would pass the zero-point gates and either crash late in the repack
    or mis-dequant silently, so reject it here; the oracle then falls
    through to the Triton backend.
    """
    if not _is_symmetric_no_zp(quant_config):
        return None
    if quant_config.num_bits != 4:
        return (
            f"symmetric no-zp MoE requires 4-bit weights "
            f"(got {quant_config.num_bits}-bit)"
        )
    if quant_config.dynamic:
        return "symmetric no-zp MoE requires static (non-dynamic) scales"
    # The kernel reads [E, G, N] group scales and derives group
    # boundaries generically as K / groups, so any group size that is a
    # multiple of 32 works (its per-32-K-slice group tracking constraint).
    # Validated for 32/128 (Qwen3.5/Gemma-4 shapes) and 64 (Nemotron-3.5-
    # Lightning: K=2688 w13 / K=1856 w2, both divisible by 64).
    if quant_config.strategy != QuantizationStrategy.GROUP:
        return "symmetric no-zp MoE requires the group strategy"
    if (
        quant_config.group_size is None
        or quant_config.group_size <= 0
        or quant_config.group_size % 32 != 0
    ):
        return (
            "symmetric no-zp MoE requires a positive group size that is a "
            f"multiple of 32 (got {quant_config.group_size})"
        )
    # g_idx activation ordering (group/dynamic) needs a runtime weight
    # reordering the kernel lacks; weight/static are natural-order safe.
    return _gidx_actorder_reason(quant_config, "symmetric no-zp MoE")


def _gfx906_asym_ct_reason(
    quant_config: QuantizationConfig | QuantizationArgs,
) -> str | None:
    """Why an asymmetric compressed-tensors checkpoint cannot use the
    gfx906 W4A16 kernel (None if it can).

    Only compressed-tensors configs are checked here; the stored zero
    points are int32-packed 8-per-word along N (ascending n, low nibble
    first), which the MoE loader presents K-first as [E, G, N/8] — the
    kernel's native AWQ qzeros layout — and the dequant (q - zp) * scale
    matches the kernel's zero_offset=0 convention. Everything else must
    satisfy the same contract as the symmetric no-zp path.
    """
    if not isinstance(quant_config, QuantizationArgs):
        return None
    if _is_symmetric_no_zp(quant_config):
        return None
    if quant_config.num_bits != 4:
        return (
            f"asymmetric compressed-tensors MoE requires 4-bit weights "
            f"(got {quant_config.num_bits}-bit)"
        )
    if quant_config.dynamic:
        return (
            "asymmetric compressed-tensors MoE requires static (non-dynamic) scales"
        )
    if quant_config.strategy != QuantizationStrategy.GROUP:
        return "asymmetric compressed-tensors MoE requires the group strategy"
    if quant_config.group_size not in (32, 128):
        return (
            "asymmetric compressed-tensors MoE requires group size 32 or 128 "
            f"(got {quant_config.group_size})"
        )
    return _gidx_actorder_reason(
        quant_config, "asymmetric compressed-tensors MoE"
    )


def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == WNA16MoEBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        )

        return [
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    elif backend == WNA16MoEBackend.MARLIN:
        return [MarlinExperts]
    elif backend == WNA16MoEBackend.BATCHED_MARLIN:
        return [BatchedMarlinExperts]
    elif backend == WNA16MoEBackend.FLASHINFER_TRTLLM:
        return [TrtLlmMxint4ExpertsMonolithic]
    elif backend == WNA16MoEBackend.TRITON:
        return [TritonWNA16Experts]
    elif backend == WNA16MoEBackend.GFX906_HIP:
        from vllm.model_executor.layers.fused_moe.experts.gfx906_w4a16_moe import (
            Gfx906WNA16Experts,
        )

        return [Gfx906WNA16Experts]
    elif backend == WNA16MoEBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsWNA16,
        )

        return [XPUExpertsWNA16]
    elif backend == WNA16MoEBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            CPUExpertsInt4,
        )

        return [CPUExpertsInt4]
    elif backend == WNA16MoEBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
            Int4EmulationTritonExperts,
        )

        return [Int4EmulationTritonExperts]
    elif backend == WNA16MoEBackend.RDNA3:
        from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
            Rdna3WNA16Experts,
        )

        return [Rdna3WNA16Experts]
    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    if current_platform.is_cpu():
        return [WNA16MoEBackend.CPU]
    if current_platform.is_xpu():
        return [WNA16MoEBackend.XPU]

    backends: list[WNA16MoEBackend] = []
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx906

        if (
            on_gfx906()
            and hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_gfx906")
        ):
            backends.append(WNA16MoEBackend.GFX906_HIP)

    return backends + [
        # Native HIP kernel, gated on gfx1100 by _supports_current_device().
        WNA16MoEBackend.RDNA3,
        WNA16MoEBackend.FLASHINFER_TRTLLM,
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
        WNA16MoEBackend.TRITON,
        WNA16MoEBackend.HUMMING,
        WNA16MoEBackend.EMULATION,
    ]


def _backend_incompatibility_reason(
    backend: WNA16MoEBackend,
    moe_config: FusedMoEConfig,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool,
) -> str | None:
    if backend == WNA16MoEBackend.FLASHINFER_TRTLLM and (may_have_zp or may_have_bias):
        return "zero points and bias are not supported"

    # AWQ-style stored zero points, or symmetric no-zp (compressed-tensors;
    # the repack passes an empty zp and the kernel inlines the constant 8).
    # Asymmetric checkpoints without a zero-point source fall back to Triton.
    if (
        backend == WNA16MoEBackend.GFX906_HIP
        and not may_have_zp
        and not _is_symmetric_no_zp(quant_config)
    ):
        return "zero points are required (AWQ-style checkpoints)"

    if backend == WNA16MoEBackend.RDNA3:
        if not isinstance(quant_config, QuantizationArgs):
            return "only compressed-tensors checkpoints are supported"
        if may_have_zp:
            return "asymmetric checkpoints are not supported"
        if may_have_bias:
            return "expert bias is not supported"
        if quant_config.actorder == "group":
            return "group activation ordering is not supported"
        if quant_config.strategy != QuantizationStrategy.GROUP:
            return "only group-wise scales are supported"

    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

    # GPTQ-style checkpoints (AutoGPTQ) use a stored zero-point convention
    # (and may use activation ordering) which the gfx906 kernel and repack
    # do not implement: only the AutoAWQ K-first, MoeWNA16 N-first, and
    # compressed-tensors K-first (symmetric no-zp or asymmetric
    # pack-quantized) layouts are supported.
    if (
        backend == WNA16MoEBackend.GFX906_HIP
        and isinstance(quant_config, AutoGPTQConfig)
        and not _is_symmetric_no_zp(quant_config)
    ):
        return "GPTQ-style zero-point encoding is not supported"
    # compressed-tensors asymmetric (stored int32-packed zps): accept only
    # the pack-quantized contract; anything outside it falls through to
    # Triton instead of reaching the repack.
    if backend == WNA16MoEBackend.GFX906_HIP:
        asym_ct_reason = _gfx906_asym_ct_reason(quant_config)
        if asym_ct_reason is not None:
            return asym_ct_reason

    # Symmetric no-zp is the only zero-point-less path the gfx906 kernel
    # supports; reject the symmetric variants it cannot dequant instead
    # of letting them reach the repack.
    if backend == WNA16MoEBackend.GFX906_HIP:
        no_zp_reason = _gfx906_no_zp_reason(quant_config)
        if no_zp_reason is not None:
            return no_zp_reason

    # Shape contract of the gfx906 kernel (it derives group boundaries as
    # K / groups and packs 8 nibbles per int32 — violations are silent
    # garbage, so gate them here instead of in the kernel).
    if backend == WNA16MoEBackend.GFX906_HIP:
        n = moe_config.intermediate_size_per_partition
        k = moe_config.hidden_dim
        group_size = getattr(quant_config, "group_size", None)
        if n % 8 != 0:
            return "intermediate size must be a multiple of 8"
        if group_size is None or (group_size > 0 and k % group_size != 0):
            return "hidden size must be divisible by the group size"

    if backend == WNA16MoEBackend.TRITON:
        if may_have_bias:
            return "expert bias is not supported"
        if isinstance(quant_config, AutoAWQConfig):
            return "the AutoAWQ weight layout is not supported"
        if isinstance(quant_config, AutoGPTQConfig) and quant_config.desc_act:
            return "GPTQ activation ordering is not supported"
        if isinstance(quant_config, QuantizationArgs):
            # Shared with the gfx906 gates: both `group` and `dynamic`
            # carry a runtime g_idx the Triton kernel cannot apply. (The
            # previous check caught only `group`, silently letting a
            # `dynamic`-ordered checkpoint reach the repack and
            # mis-dequant.)
            gidx_reason = _gidx_actorder_reason(
                quant_config, "the Triton WNA16 MoE backend"
            )
            if gidx_reason is not None:
                return gidx_reason

    # Marlin only supports certain problem/group sizes.
    allow_marlin = not isinstance(quant_config, MoeWNA16Config)

    if allow_marlin and backend in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ):
        if isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, QuantizationArgs)):
            # QuantizationArgs leaves group_size unset for per-channel
            # strategies; -1 is the in-tree encoding for that.
            group_size = (
                -1 if quant_config.group_size is None else quant_config.group_size
            )
        else:
            return "Marlin not supported for this layer"

        if not check_moe_marlin_supports_config(
            moe_config, group_size, allow_tile_padding
        ):
            return "Marlin not supported for this layer"

    if not allow_marlin and backend in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
        WNA16MoEBackend.EMULATION,
    ):
        return "the MoeWNA16 checkpoint layout is not supported"

    return None


def map_wna16_backend(runner_backend: MoEBackend) -> WNA16MoEBackend:
    """Map user's MoEBackend to WNA16MoEBackend."""
    mapping = {
        "triton": WNA16MoEBackend.TRITON,
        "marlin": WNA16MoEBackend.MARLIN,
        "humming": WNA16MoEBackend.HUMMING,
        "flashinfer_trtllm": WNA16MoEBackend.FLASHINFER_TRTLLM,
        "emulation": WNA16MoEBackend.EMULATION,
        "rdna3": WNA16MoEBackend.RDNA3,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for WNA16 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool = False,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_key: The QuantKey describing the weight quantization.
                    Must have int4 or int8 type.
        quant_config: Quantization structure and checkpoint format description.
        may_have_zp: Whether the integration can provide weight zero points.
        may_have_bias: Whether the integration can provide expert bias.

    Returns:
        A tuple of (``WNA16MoEBackend``, experts class or ``None``).
    """

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: WNA16MoEBackend):
        return f"Using '{backend.value}' WNA16 MoE backend."

    def _make_log_unsupported(backend: WNA16MoEBackend, reason: str | None) -> str:
        if reason:
            return (
                f"WNA16 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"WNA16 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: WNA16MoEBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_wna16_backend(runner_backend)
        reason = _backend_incompatibility_reason(
            requested_backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            raise ValueError(_make_log_unsupported(requested_backend, reason))
        return _return_or_raise(
            requested_backend, config, weight_key, None, activation_format
        )

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _get_priority_backends()

    for backend in AVAILABLE_BACKENDS:
        reason = _backend_incompatibility_reason(
            backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            logger.debug_once(_make_log_unsupported(backend, reason), scope="local")
            continue
        activation_key = None  # always BF16 activation for WNA16 MoE
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No WNA16 MoE backend supports the deployment configuration."
    )


def make_wna16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    num_bits: int,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    gemm1_clamp_limit: float | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
) -> FusedMoEQuantConfig:
    """Create the FusedMoEQuantConfig for 4 or 8-bit WNA16 MoE."""
    if num_bits == 4:
        return int4_w4a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            block_shape=[0, group_size],
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
        )
    else:
        assert num_bits == 8
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            block_shape=[0, group_size],
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
        )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: WNA16MoEBackend = WNA16MoEBackend.MARLIN,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsInt4,
    )
    from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
        Int4EmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
        Rdna3WNA16Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )

    # Currently, we only support TrtLlmMxint4ExpertsMonolithic, MarlinExperts,
    # BatchedMarlinExperts, XPUExpertsWNA16, CPUExpertsInt4, the Humming
    # grouped/indexed experts, and Int4EmulationTritonExperts
    allowed_experts: tuple[type[mk.FusedMoEExperts], ...] = (
        MarlinExperts,
        BatchedMarlinExperts,
        TritonWNA16Experts,
        TrtLlmMxint4ExpertsMonolithic,
        XPUExpertsWNA16,
        CPUExpertsInt4,
        Int4EmulationTritonExperts,
        Rdna3WNA16Experts,
    )
    if backend == WNA16MoEBackend.HUMMING:
        allowed_experts += tuple(backend_to_kernel_cls(WNA16MoEBackend.HUMMING))
    if backend == WNA16MoEBackend.GFX906_HIP:
        allowed_experts += tuple(backend_to_kernel_cls(WNA16MoEBackend.GFX906_HIP))
    assert experts_cls in allowed_experts

    is_monolithic = experts_cls.is_monolithic()

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")
    logger.info_once("Using %s", experts_cls.__name__, scope="local")

    extra_args: dict[str, Any] = {}
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_args["max_num_tokens"] = max_num_tokens
        extra_args["num_dispatchers"] = prepare_finalize.num_dispatchers()

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_args,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )


# ---------------------------------------------------------------------------
# Per-backend weight post-processing
# ---------------------------------------------------------------------------


def _process_weights_flashinfer(
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Flashinfer (TRT-LLM MXINT4) weight post-processing.

    Steps
    -----
    1. Transform weights/scales via ``prepare_static_weights_for_trtllm_mxint4_moe``.
    2. Return transformed tensors and biases.
    """
    from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
        prepare_static_weights_for_trtllm_mxint4_moe,
    )

    dict_weights_mxint4 = prepare_static_weights_for_trtllm_mxint4_moe(
        w13_qweight,
        w13_scales,
        w2_qweight,
        w2_scales,
    )

    return (
        dict_weights_mxint4["gemm1_weights"],
        dict_weights_mxint4["gemm2_weights"],
        dict_weights_mxint4["gemm1_scales"],
        dict_weights_mxint4["gemm2_scales"],
        None,
        None,
        None,
        None,
        w13_bias,
        w2_bias,
    )


def _pad_w13_shard_cols(x: torch.Tensor, unit: int, padded_unit: int) -> torch.Tensor:
    """Zero-pad each of the two gate/up shards of a ``(E, rows, 2 * unit)``
    tensor along its last dim, from ``unit`` to ``padded_unit`` columns."""
    if padded_unit == unit:
        return x
    e, rows, _ = x.shape
    x = x.view(e, rows, 2, unit)
    x = torch.nn.functional.pad(x, (0, padded_unit - unit))
    return x.reshape(e, rows, 2 * padded_unit).contiguous()


def _pad_rows(x: torch.Tensor, padded_rows: int) -> torch.Tensor:
    """Zero-pad a ``(E, rows, cols)`` tensor to ``padded_rows`` rows."""
    if padded_rows == x.size(1):
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, padded_rows - x.size(1)))


def _pad_w13_bias(bias: torch.Tensor, n: int, padded_n: int) -> torch.Tensor:
    """Zero-pad each gate/up shard of a ``(E, 2 * n)`` bias to ``padded_n``."""
    if padded_n == n:
        return bias
    e = bias.size(0)
    bias = bias.view(e, 2, n)
    bias = torch.nn.functional.pad(bias, (0, padded_n - n))
    return bias.reshape(e, 2 * padded_n).contiguous()


def _process_weights_marlin(
    layer: torch.nn.Module,
    input_dtype: torch.dtype | None,
    num_bits: int,
    pack_factor: int,
    group_size: int,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Standard Marlin weight post-processing shared by MARLIN and
    BATCHED_MARLIN backends.

    Steps
    -----
    1. Optional FP8 preprocessing of packed weights / scales.
    2. Repack weights via ``gptq_marlin_moe_repack``.
    3. Permute scales (and optionally extract INT8 global scales).
    4. Permute bias tensors.
    """
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    marlin_w13_qweight: torch.Tensor
    marlin_w2_qweight: torch.Tensor
    marlin_w13_scales: torch.Tensor
    marlin_w2_scales: torch.Tensor
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    # --- FP8 weight / scale adjustment ---
    if input_dtype == torch.float8_e4m3fn:
        # NOTE: for non-zp quantization format only
        marlin_w13_qweight = ops.marlin_int4_fp8_preprocess(w13_qweight, inplace=False)
        marlin_w2_qweight = ops.marlin_int4_fp8_preprocess(w2_qweight, inplace=False)
        marlin_w13_scales = w13_scales.data * 512
        marlin_w2_scales = w2_scales.data * 512
    else:
        marlin_w13_qweight = w13_qweight
        marlin_w2_qweight = w2_qweight
        marlin_w13_scales = w13_scales
        marlin_w2_scales = w2_scales

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # GPTQ packs along K: w13's N is in the (shard) columns, w2's N in the rows.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        marlin_w13_qweight = _pad_w13_shard_cols(marlin_w13_qweight, N, padded_N)
        marlin_w2_qweight = _pad_rows(marlin_w2_qweight, padded_N // pack_factor)
        marlin_w13_scales = _pad_w13_shard_cols(marlin_w13_scales, N, padded_N)
        if group_size > 0:
            marlin_w2_scales = _pad_rows(marlin_w2_scales, padded_N // group_size)
        if w13_qzeros is not None:
            w13_qzeros = _pad_w13_shard_cols(
                w13_qzeros, N // pack_factor, padded_N // pack_factor
            )
        if w2_qzeros is not None and group_size > 0:
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

    # --- Repack weights ---
    marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        marlin_w13_qweight,
        marlin_w13_qweight.shape[1] * pack_factor,
        marlin_w13_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        marlin_w2_qweight,
        marlin_w2_qweight.shape[1] * pack_factor,
        marlin_w2_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )

    # --- Permute scales ---
    marlin_w13_scales = marlin_moe_permute_scales(
        s=marlin_w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=marlin_w13_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    group_size_or_pack_factor = group_size if group_size != -1 else pack_factor
    marlin_w2_scales = marlin_moe_permute_scales(
        s=marlin_w2_scales,
        size_k=marlin_w2_scales.shape[1] * group_size_or_pack_factor,
        size_n=marlin_w2_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    if input_dtype == torch.int8:
        if layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
        if layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )

    # --- Permute zero points ---
    if w13_qzeros is not None and w2_qzeros is not None:
        w13_qzeros = moe_packed_to_marlin_zero_points(
            w13_qzeros,
            size_k=w13_qzeros.shape[1],
            size_n=w13_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        w2_qzeros = moe_packed_to_marlin_zero_points(
            w2_qzeros,
            size_k=w2_qzeros.shape[1],
            size_n=w2_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )

    # --- Permute bias ---
    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        w13_qzeros,
        w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )


def _process_awq_weights_marlin(
    layer: torch.nn.Module,
    weight_bits: int,
    pack_factor: int,
    group_size: int,
    input_dtype: torch.dtype | None,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """AWQ-specific Marlin weight post-processing.

    AWQ checkpoints use a different packing order than GPTQ, so they need
    AWQ-specific weight repacking and zero-point conversion before Marlin runs.
    """
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    if input_dtype == torch.float8_e4m3fn:
        ops.marlin_int4_fp8_preprocess(
            w13_qweight.view(-1, w13_qweight.size(2)),
            w13_qzeros.view(-1, w13_qzeros.size(2)),
            inplace=True,
        )
        ops.marlin_int4_fp8_preprocess(
            w2_qweight.view(-1, w2_qweight.size(2)),
            w2_qzeros.view(-1, w2_qzeros.size(2)),
            inplace=True,
        )
        w13_scales = w13_scales.data * 512
        w2_scales = w2_scales.data * 512

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # AWQ packs along N: w13's N is in the (shard) columns, w2's N in the rows.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        w13_qweight = _pad_w13_shard_cols(
            w13_qweight, N // pack_factor, padded_N // pack_factor
        )
        w2_qweight = _pad_rows(w2_qweight, padded_N)
        w13_scales = _pad_w13_shard_cols(w13_scales, N, padded_N)
        w13_qzeros = _pad_w13_shard_cols(
            w13_qzeros, N // pack_factor, padded_N // pack_factor
        )
        if group_size > 0:
            w2_scales = _pad_rows(w2_scales, padded_N // group_size)
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

    marlin_w13_qweight = ops.awq_marlin_moe_repack(
        w13_qweight,
        size_k=w13_qweight.shape[1],
        size_n=w13_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.awq_marlin_moe_repack(
        w2_qweight,
        size_k=w2_qweight.shape[1],
        size_n=w2_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )

    marlin_w13_scales = marlin_moe_permute_scales(
        s=w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w13_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w13 > 1:
        marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
            marlin_w13_scales
        )

    marlin_w2_scales = marlin_moe_permute_scales(
        s=w2_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w2_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w2 > 1:
        marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
            marlin_w2_scales
        )

    marlin_w13_qzeros = moe_awq_to_marlin_zero_points(
        w13_qzeros,
        size_k=w13_qzeros.shape[1],
        size_n=w13_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qzeros = moe_awq_to_marlin_zero_points(
        w2_qzeros,
        size_k=w2_qzeros.shape[1],
        size_n=w2_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )

    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        marlin_w13_qzeros,
        marlin_w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )


def _synthesize_rdna3_qzeros(
    groups: int, out_features: int, device: torch.device
) -> torch.Tensor:
    """Create the packed zero-point tensor for symmetric quantization.

    GPTQv1 +1 quirk: the kernel adds 1 to the stored zeros, so encode
    (bias - 1) = 7 for uint4b8 (bias=8).
    """
    zeros = torch.full(
        (groups, out_features),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    return pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)


def _process_weights_rdna3(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """RDNA3 (gfx1100) W4A16 weight post-processing.

    Interleaves the packed nibbles per expert (the exllama shuffle the dense
    RDNA3 kernel also uses) and synthesizes the symmetric zero points that
    ``moe_gptq_gemm_rdna3`` dequantizes with. The packed layout
    ``[E, K // 8, N]`` and the ``[E, groups, N]`` scales are already what the
    kernel wants, so neither is repacked.
    """
    device = w13.device
    num_experts = w13.size(0)

    for e in range(num_experts):
        w13_e = w13[e].contiguous()
        ops.gptq_shuffle(w13_e, 4)
        w13[e] = w13_e
        w2_e = w2[e].contiguous()
        ops.gptq_shuffle(w2_e, 4)
        w2[e] = w2_e

    def _qzeros(w: torch.Tensor) -> torch.Tensor:
        qz = _synthesize_rdna3_qzeros((w.size(1) * 8) // group_size, w.size(2), device)
        return qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous()

    return (
        w13,
        w2,
        w13_scale.contiguous(),
        w2_scale.contiguous(),
        _qzeros(w13),
        _qzeros(w2),
        None,  # w13_input_global_scale
        None,  # w2_input_global_scale
        None,  # w13_bias
        None,  # w2_bias
    )


def _process_weights_cpu(
    quant_config: QuantizationConfig | QuantizationArgs | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """CPU INT4 W4A16 weight post-processing."""
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        prepare_int4_moe_layer_for_cpu,
    )
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
    )
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQConfig,
    )

    # Detect packing format.
    # AWQ: qweight is [E, K, 2*N//8] (packed along output/N dim).
    # GPTQ: qweight is [E, K//8, 2*N] (packed along input/K dim).
    # compressed-tensors: qweight is [E, K//8, 2*N] (packed along input/K dim).
    if isinstance(quant_config, AutoAWQConfig):
        # AWQ: K is stored unpacked in dim 1.
        cpu_quant_algo = ops.CPUQuantAlgo.AWQ
    elif isinstance(quant_config, (AutoGPTQConfig, QuantizationArgs)):
        # GPTQ / compressed-tensors: K//8 is stored packed in dim 1.
        if isinstance(quant_config, AutoGPTQConfig) and quant_config.desc_act:
            raise NotImplementedError(
                "CPU WNA16 MoE backend does not support GPTQ with "
                "desc_act=True. The fused MoE kernel has no activation "
                "reordering support."
            )
        cpu_quant_algo = ops.CPUQuantAlgo.GPTQ
    else:
        raise TypeError(
            "CPU WNA16 MoE backend requires AutoAWQConfig, AutoGPTQConfig "
            f"or QuantizationArgs, got {type(quant_config).__name__}."
        )

    # Determine zero points for repacking.
    w13_zeros: torch.Tensor | None = None
    w2_zeros: torch.Tensor | None = None
    if w13_qzeros is not None:
        w13_zeros = (
            w13_qzeros.data.view(torch.int32)
            if w13_qzeros.dtype != torch.int32
            else w13_qzeros.data
        )
    if w2_qzeros is not None:
        w2_zeros = (
            w2_qzeros.data.view(torch.int32)
            if w2_qzeros.dtype != torch.int32
            else w2_qzeros.data
        )

    (
        blocked_w13,
        blocked_w2,
        blocked_s13,
        blocked_s2,
        blocked_z13,
        blocked_z2,
    ) = prepare_int4_moe_layer_for_cpu(
        w13,
        w2,
        w13_scale,
        w2_scale,
        quant_algo=cpu_quant_algo,
        w13_zeros=w13_zeros,
        w2_zeros=w2_zeros,
    )
    return (
        blocked_w13,
        blocked_w2,
        blocked_s13,
        blocked_s2,
        blocked_z13,
        blocked_z2,
        None,  # w13_input_global_scale
        None,  # w2_input_global_scale
        w13_bias.to(torch.float32) if w13_bias is not None else None,
        w2_bias.to(torch.float32) if w2_bias is not None else None,
    )


def _process_weights_xpu(
    layer: torch.nn.Module,
    quant_config: QuantizationConfig,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Repack GPTQ-format INT4 MoE weights into the layout
    `vllm_xpu_kernels.fused_moe_interface.xpu_fused_moe(is_int4=True)` expects:

        w13: [E, 2*N, K] int4 (uint8 storage [E, 2*N, K // 2])
        w13_scales: [E, 2*N, K // group_size] params_dtype
        w2:  [E, K, N]   int4 (uint8 storage [E, K, N // 2])
        w2_scales:  [E, K, N // group_size]   params_dtype

    Input GPTQ layout from MoERunner.weight_loader:
        w13: [E, K // 8, 2*N] int32 (8 nibbles per int32 along the input dim)
        w13_scales: [E, K // group_size, 2*N] params_dtype
        w2:  [E, N // 8, K] int32
        w2_scales:  [E, N // group_size, K] params_dtype

    Transpose dim 1 ↔ dim 2 then view int32 → uint8 to recover sequential
    int4-packed bytes along the input dim. Each packed int32 holds 8 nibbles
    `(n7<<28)|(n6<<24)|...|(n1<<4)|n0` in ascending K order; on a
    little-endian host the int32→uint8 view exposes them as bytes
    `[n1<<4|n0, n3<<4|n2, n5<<4|n4, n7<<4|n6]`, i.e. two nibbles per byte
    with the lower nibble = lower input-K index. xpu_fused_moe(is_int4=True)
    expects this convention; on a big-endian host the byte order reverses
    and the kernel would silently miscompute, so we hard-fail.
    """
    del layer, quant_config  # unused — kept for parity with the marlin helper

    if sys.byteorder != "little":
        raise NotImplementedError(
            "_process_weights_xpu requires a little-endian host: the GPTQ "
            "int32 → uint8 nibble repack relies on LE byte ordering."
        )

    w13_xpu = w13_qweight.transpose(1, 2).contiguous().view(torch.uint8)
    w2_xpu = w2_qweight.transpose(1, 2).contiguous().view(torch.uint8)
    w13_scales_xpu = w13_scales.transpose(1, 2).contiguous()
    w2_scales_xpu = w2_scales.transpose(1, 2).contiguous()

    return (
        w13_xpu,
        w2_xpu,
        w13_scales_xpu,
        w2_scales_xpu,
        w13_bias,
        w2_bias,
    )


def _humming_wna16_weight_schema(
    quant_config: QuantizationConfig | QuantizationArgs | None,
) -> dict[str, Any]:
    """Humming weight schema for a WNA16 checkpoint, derived from the quant
    config rather than the running kernel."""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    if isinstance(quant_config, AutoAWQConfig):
        return {
            "quant_method": "awq",
            "bits": quant_config.weight_bits,
            "group_size": quant_config.group_size,
            "zero_point": quant_config.zero_point,
        }
    if isinstance(quant_config, AutoGPTQConfig):
        return {
            "quant_method": "gptq",
            "bits": quant_config.weight_bits,
            "group_size": quant_config.group_size,
            "desc_act": quant_config.desc_act,
            "sym": quant_config.is_sym,
        }
    if isinstance(quant_config, QuantizationArgs):
        quant_type = getattr(quant_config.type, "value", quant_config.type)
        quant_strategy = getattr(quant_config.strategy, "value", quant_config.strategy)
        return {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "type": str(quant_type),
            "num_bits": quant_config.num_bits,
            "strategy": str(quant_strategy),
            "group_size": quant_config.group_size,
            "symmetric": quant_config.symmetric,
        }
    raise TypeError(
        "Humming WNA16 checkpoint schema requires AutoAWQConfig, "
        "AutoGPTQConfig or QuantizationArgs, "
        f"got {type(quant_config).__name__}."
    )


def _convert_moe_wna16_humming_tensors(
    tensors: dict[str, torch.Tensor], has_zero_point: bool
) -> dict[str, torch.Tensor]:
    """Convert MoeWNA16's N-first uint8 packing to Humming's int32 packing."""
    if sys.byteorder != "little":
        raise NotImplementedError(
            "MoeWNA16 to Humming conversion requires a little-endian host."
        )

    output = {
        "weight": tensors["qweight"].contiguous().view(torch.int32),
        "weight_scale": tensors["scales"],
    }
    if has_zero_point:
        qzeros = tensors["qzeros"]
        output["zero_point"] = (
            qzeros.transpose(-1, -2)
            .contiguous()
            .view(torch.int32)
            .transpose(-1, -2)
            .contiguous()
        )
    return output


class _MoeWNA16HummingWeightSchema:
    """Adapter from MoeWNA16's generic packed layout to Humming's layout."""

    def __init__(self, bits: int, group_size: int, has_zero_point: bool) -> None:
        self.bits = bits
        self.group_size = group_size
        self.has_zero_point = has_zero_point

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        del shape_n_stacks, shape_k_stacks, num_experts
        from vllm.utils.humming import HummingWeightSchema, dtypes

        output = _convert_moe_wna16_humming_tensors(
            tensors, has_zero_point=self.has_zero_point
        )
        output["weight_scale"] = output["weight_scale"].to(param_dtype)
        schema = HummingWeightSchema(
            b_dtype=dtypes.DataType.from_str(f"uint{self.bits}"),
            weight_scale_group_size=self.group_size,
            has_zero_point=self.has_zero_point,
        )
        return schema, output


def _unpack_and_dequant_int4_gptq(
    w_int32: torch.Tensor,
    scale: torch.Tensor,
    qzeros: torch.Tensor | None,
    transpose_output: bool,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack GPTQ-packed int4 weights and dequantize to output_dtype.

    Args:
        w_int32: packed weights, shape [E, K_packed, N] where K_packed = K//8
                 (8 nibbles per int32, LSB-first in the K dimension).
        scale:   per-group scales, shape [E, K//group_size, N], float16.
        qzeros:  optional asymmetric zero-points, shape [E, K//gs, N//8], int32.
                 None for symmetric (uint4b8 with implicit bias 8).
        transpose_output: if True return [E, N, K]; if False return [E, K, N].
        output_dtype: target floating-point dtype (bfloat16 or float16).

    Returns:
        Dequantized weight tensor in the requested layout.
    """
    E, K_packed, N = w_int32.shape
    K = K_packed * 8

    # Unpack: [E, K_packed, N] -> [E, K_packed, N, 8] via bit-shifts.
    # The nibble index (last dim) enumerates K rows within each packed column,
    # so we must fuse K_packed and the nibble dim, not N and the nibble dim.
    # Permute to [E, K_packed, 8, N] before reshaping to [E, K, N].
    shifts = torch.arange(8, device=w_int32.device, dtype=torch.int32) * 4
    nibbles = (w_int32.unsqueeze(-1) >> shifts) & 0xF  # [E, K_packed, N, 8]

    # Reshape to [E, K, N]: fuse K_packed and nibble index (dim 1 and 3)
    w = nibbles.permute(0, 1, 3, 2).reshape(E, K, N).to(torch.int16)

    if qzeros is None:
        # Symmetric uint4b8: subtract bias so the range is [-8, 7]
        w = w - 8
    else:
        # Asymmetric: unpack zero-points (same 8-nibble packing) and subtract
        # qzeros shape: [E, K//gs, N//8] int32
        gs = K // scale.shape[1]
        n_gs = scale.shape[1]
        zp_shifts = torch.arange(8, device=qzeros.device, dtype=torch.int32) * 4
        zp_nibbles = (qzeros.unsqueeze(-1) >> zp_shifts) & 0xF  # [E, n_gs, N//8, 8]
        zp = zp_nibbles.reshape(E, n_gs, N).to(torch.int16)  # [E, n_gs, N]
        zp = zp.repeat_interleave(gs, dim=1)  # [E, K, N]
        w = w - zp

    # Broadcast scale [E, K//gs, N] -> [E, K, N]
    gs = K // scale.shape[1]
    scale_broadcast = scale.repeat_interleave(gs, dim=1).to(output_dtype)

    w_dequant = w.to(output_dtype) * scale_broadcast  # [E, K, N]

    if transpose_output:
        return w_dequant.permute(0, 2, 1).contiguous()  # [E, N, K]
    return w_dequant.contiguous()  # [E, K, N]


def _unpack_and_dequant_int4_awq(
    w_int32: torch.Tensor,
    scale: torch.Tensor,
    qzeros: torch.Tensor | None,
    transpose_output: bool,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack AWQ-packed int4 weights and dequantize to output_dtype.

    AWQ packs along the N (column) dimension with an interleave permutation
    [0,2,4,6,1,3,5,7] applied before packing, so unpacking must undo that.

    Args:
        w_int32: packed weights, shape [E, K, N_packed] where N_packed = N//8
                 (8 nibbles per int32, packed along N with AWQ interleaving).
        scale:   per-group scales, shape [E, K//group_size, N], float16.
        qzeros:  asymmetric zero-points, shape [E, K//gs, N_packed], int32.
                 None for symmetric (uint4b8 with implicit bias 8).
        transpose_output: if True return [E, N, K]; if False return [E, K, N].
        output_dtype: target floating-point dtype (bfloat16 or float16).

    Returns:
        Dequantized weight tensor in the requested layout.
    """
    E, K, N_packed = w_int32.shape
    N = N_packed * 8

    # Unpack 8 nibbles per int32 along the N dimension (LSB-first)
    shifts = torch.arange(8, device=w_int32.device, dtype=torch.int32) * 4
    # [E, K, N_packed, 8] -> [E, K, N_packed*8] = [E, K, N_interleaved]
    nibbles = (w_int32.unsqueeze(-1) >> shifts) & 0xF
    w_interleaved = nibbles.reshape(E, K, N)  # [E, K, N] but column-interleaved

    # Undo AWQ interleave: packed order is [0,2,4,6,1,3,5,7] within each group
    # of 8. Inverse: position i in packed -> original column interleave[i].
    # To reverse: we need the inverse permutation so that
    # w[:, :, inv_interleave] = w_interleaved gives the natural column order.
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=w_int32.device)
    inv_interleave = torch.empty_like(interleave)
    inv_interleave[interleave] = torch.arange(8, device=w_int32.device)

    # Apply inverse interleave within each group of 8 columns
    w_reshaped = w_interleaved.reshape(E, K, N // 8, 8)  # [E, K, groups, 8]
    w_reordered = w_reshaped[:, :, :, inv_interleave]  # undo interleave
    w = w_reordered.reshape(E, K, N).to(torch.int16)  # [E, K, N]

    if qzeros is None:
        w = w - 8
    else:
        # qzeros: [E, K//gs, N_packed] int32, same AWQ column packing
        gs = K // scale.shape[1]
        n_gs = scale.shape[1]
        zp_nibbles = (qzeros.unsqueeze(-1) >> shifts) & 0xF  # [E, n_gs, N_packed, 8]
        zp_interleaved = zp_nibbles.reshape(E, n_gs, N)
        zp_reshaped = zp_interleaved.reshape(E, n_gs, N // 8, 8)
        zp_reordered = zp_reshaped[:, :, :, inv_interleave]
        zp = zp_reordered.reshape(E, n_gs, N).to(torch.int16)  # [E, n_gs, N]
        zp = zp.repeat_interleave(gs, dim=1)  # [E, K, N]
        w = w - zp

    gs = K // scale.shape[1]
    scale_broadcast = scale.repeat_interleave(gs, dim=1).to(output_dtype)  # [E, K, N]

    w_dequant = w.to(output_dtype) * scale_broadcast  # [E, K, N]

    if transpose_output:
        return w_dequant.permute(0, 2, 1).contiguous()  # [E, N, K]
    return w_dequant.contiguous()  # [E, K, N]


def _process_weights_emulation_gptq(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None,
    w2_qzeros: torch.Tensor | None,
) -> tuple:
    """Dequantize int4 weights to BF16 for the emulation backend.

    Inputs are in GPTQ packed format:
        w13: [E, K//8, 2*N]   int32  (gate+up proj stacked on dim 2)
        w2:  [E, N//8, K]     int32
        w13_scale: [E, K//gs, 2*N]  float16
        w2_scale:  [E, N//gs, K]    float16

    Outputs (what TritonExperts expects):
        w13_out: [E, 2*N, K]  bfloat16
        w2_out:  [E, K, N]    bfloat16
    """
    # w13: packed along K (dim 1), output cols are 2*N (dim 2)
    # transpose_output=True yields [E, 2*N, K]
    w13_bf16 = _unpack_and_dequant_int4_gptq(
        w13, w13_scale, w13_qzeros, transpose_output=True
    )

    # w2: packed along N (dim 1 is N//8), output cols are K (dim 2)
    # After unpacking we get [E, N, K]; we want [E, K, N] for TritonExperts
    # transpose_output=False gives [E, N, K], then we permute once more
    w2_unpacked = _unpack_and_dequant_int4_gptq(
        w2, w2_scale, w2_qzeros, transpose_output=False
    )  # [E, N, K]
    w2_bf16 = w2_unpacked.permute(0, 2, 1).contiguous()  # [E, K, N]

    dummy = torch.ones(1, dtype=torch.float16, device=w13.device)
    return (
        w13_bf16,  # w13_qweight  (now bf16, not int32)
        w2_bf16,  # w2_qweight   (now bf16, not int32)
        dummy,  # w13_scales   (unused; nulled out in Int4EmulationTritonExperts)
        dummy,  # w2_scales    (unused)
        None,  # w13_qzeros
        None,  # w2_qzeros
        None,  # w13_input_global_scale
        None,  # w2_input_global_scale
        None,  # w13_bias
        None,  # w2_bias
    )


def _process_weights_emulation_awq(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None,
    w2_qzeros: torch.Tensor | None,
) -> tuple:
    """Dequantize AWQ int4 weights to BF16 for the emulation backend.

    AWQ inputs:
        w13: [E, K, 2*N//8]       int32  (packed along N, gate+up on dim 2)
        w2:  [E, N, K//8]         int32  (packed along K)
        w13_scale: [E, K//gs, 2*N]  float16
        w2_scale:  [E, N//gs, K]    float16

    Outputs (what TritonExperts expects):
        w13_out: [E, 2*N, K]  bfloat16
        w2_out:  [E, K, N]    bfloat16
    """
    # w13: AWQ-packed along N (dim 2), K is unpacked in dim 1
    # _unpack_and_dequant_int4_awq with transpose_output=True yields [E, 2*N, K]
    w13_bf16 = _unpack_and_dequant_int4_awq(
        w13, w13_scale, w13_qzeros, transpose_output=True
    )

    # w2: AWQ packs along K (dim 2 is K//8), N is unpacked in dim 1.
    # AWQ w2 is [E, N, K//8] — same column-pack format applied to the K dim.
    # _unpack_and_dequant_int4_awq expects [E, rows, N_packed] where the
    # packed dim is columns. Treat dim 1 as rows and dim 2 as N_packed:
    # unpacking gives [E, N, K]. Then permute to [E, K, N].
    w2_unpacked = _unpack_and_dequant_int4_awq(
        w2, w2_scale, w2_qzeros, transpose_output=False
    )  # [E, N, K]
    w2_bf16 = w2_unpacked.permute(0, 2, 1).contiguous()  # [E, K, N]

    dummy = torch.ones(1, dtype=torch.float16, device=w13.device)
    return (
        w13_bf16,
        w2_bf16,
        dummy,
        dummy,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _repack_w4a16_gfx906_expert(
    w: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack one W4A16 MoE weight set into the gfx906 kernel layout.

    Three source layouts are supported (detected by shape/dtype):

    MoeWNA16 (N-first uint8; the AWQ-on-ROCm fallback path via
    MoeWNA16Method.create_weights):
      w:      [E, N, K/2] uint8  (byte j holds k=2j low, k=2j+1 high)
      scales: [E, N, G]   fp16
      qzeros: [E, N/2, G] uint8 (byte i holds n=2i low, n=2i+1 high)

    AutoAWQMoEMethod (K-first int32; Marlin-supported path):
      w:      [E, K, N/8] int32 (word m holds n=8m..8m+7, low nibble first)
      scales: [E, G, N]   fp16
      qzeros: [E, G, N/8] int32 (word m holds n=8m..8m+7, low nibble first)

    compressed-tensors (GPTQ-style K-first int32; symmetric no-qzeros —
    e.g. Gemma-4-26B-A4B-AWQ — or asymmetric pack-quantized with stored
    zps — e.g. Ornith-1.5-35B-A3B-AWQ-INT4). Raw on-disk tensors are
    N-first [N, K/8] (zps [N/8, G]); the MoE weight loader
    (is_transposed) presents them K-first:
      w:      [E, K/8, N] int32 (word holds k=8q..8q+7, low nibble first)
      scales: [E, G, N]   fp16
      qzeros: [E, G, N/8] int32 (asymmetric only; word holds
               n=8m..8m+7, low nibble first — the kernel's native layout)

    Detection is collision-free: the packed dim of the int32 layouts is
    dim 2 (AWQ: N/8) or dim 1 (GPTQ: K/8), so w.shape[2] is N/8 (AWQ),
    N (GPTQ), or matches scales only for uint8 MoeWNA16.

    Outputs (all):
      wq:  [E, K/8, N] int32 exllama shuffle
            (even/odd interleaved: bits[3:0]=k0 [7:4]=k2 [11:8]=k4
             [15:12]=k6 [19:16]=k1 [23:20]=k3 [27:24]=k5 [31:28]=k7
             for k = 8*qk .. 8*qk+7)
      sc:  [E, G, N] fp16
      zp:  [E, G, N/8] int32 (8 nibbles per word, ascending n order);
            None for symmetric (no-zp) inputs — the kernel inlines the
            constant zero point 8
    """
    if w.dtype == torch.uint8 and w.shape[1] == scales.shape[1]:
        return _repack_w4a16_wna16_layout(w, scales, qzeros)
    N = scales.shape[2]
    if w.shape[2] * 8 == N:
        return _repack_w4a16_awq_kfirst_layout(w, scales, qzeros, N)
    if w.shape[2] == N:
        return _repack_w4a16_gptq_kfirst_layout(w, scales, qzeros, N)
    raise ValueError(
        f"unrecognized W4A16 MoE weight layout: w={tuple(w.shape)} "
        f"dtype={w.dtype}, scales={tuple(scales.shape)}"
    )


def _repack_w4a16_wna16_layout(
    w: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """MoeWNA16 N-first uint8 layout (see _repack_w4a16_gfx906_expert)."""
    E, N, k_half = w.shape
    K = 2 * k_half
    assert N % 8 == 0 and K % 8 == 0

    b = w.to(torch.int32).view(E, N, K // 8, 4)
    lo = b & 0xF  # k = 2j
    hi = (b >> 4) & 0xF  # k = 2j + 1
    wq = (
        (
            lo[..., 0]
            | (lo[..., 1] << 4)
            | (lo[..., 2] << 8)
            | (lo[..., 3] << 12)
            | (hi[..., 0] << 16)
            | (hi[..., 1] << 20)
            | (hi[..., 2] << 24)
            | (hi[..., 3] << 28)
        )
        .permute(0, 2, 1)
        .contiguous()
    )

    sc = scales.to(torch.float16).permute(0, 2, 1).contiguous()

    if qzeros is None:
        # Symmetric quant: no zp tensor — the kernel inlines the constant
        # zero point 8 (uint4 midpoint) when passed an empty tensor.
        zp = None
    else:
        z = qzeros.to(torch.int32)
        zf = torch.stack([z & 0xF, (z >> 4) & 0xF], dim=2).reshape(E, N, -1)
        zr = zf.view(E, N // 8, 8, -1)
        zp = (
            (
                zr[..., 0, :]
                | (zr[..., 1, :] << 4)
                | (zr[..., 2, :] << 8)
                | (zr[..., 3, :] << 12)
                | (zr[..., 4, :] << 16)
                | (zr[..., 5, :] << 20)
                | (zr[..., 6, :] << 24)
                | (zr[..., 7, :] << 28)
            )
            .permute(0, 2, 1)
            .contiguous()
        )

    return wq, sc, zp


def _repack_w4a16_awq_kfirst_layout(
    w: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """AutoAWQMoEMethod K-first int32 layout (see
    _repack_w4a16_gfx906_expert). Scales and zero points are already in the
    kernel's [E, G, N] / [E, G, N/8] layout and pass through unchanged."""
    E, K, _ = w.shape
    assert K % 8 == 0

    # Shifts that place nibble j (k = 8*qk + j) into its exllama-shuffled
    # position: even j -> bits[4j], odd j -> bits[16 + 4*(j-1)].
    shifts_out = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], device=w.device, dtype=torch.int32
    )
    nib_shifts = 4 * torch.arange(8, device=w.device, dtype=torch.int32)

    wq = torch.empty(E, K // 8, N, dtype=torch.int32, device=w.device)
    # Process one expert at a time to keep the temporary [K, N] unpack small.
    for e in range(E):
        q = ((w[e].unsqueeze(-1) >> nib_shifts) & 0xF).reshape(K, N)
        wq[e] = (
            (q.view(K // 8, 8, N) << shifts_out.view(1, 8, 1))
            .sum(dim=1)
            .to(torch.int32)
        )

    sc = scales.to(torch.float16).contiguous()
    # Symmetric quant: no zp tensor — the kernel inlines the constant zero
    # point 8 when passed an empty tensor.
    zp = None if qzeros is None else qzeros.to(torch.int32).contiguous()

    return wq, sc, zp


def _repack_w4a16_gptq_kfirst_layout(
    w: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """compressed-tensors GPTQ-style K-first int32 layout (see
    _repack_w4a16_gfx906_expert). The nibbles are already packed
    8-per-word along K exactly like the AutoAWQ K-first layout, so the
    exllama shuffle is identical and scales pass through unchanged.
    Asymmetric (pack-quantized) inputs carry stored zps that the MoE
    loader presents K-first [E, G, N/8] int32 — the kernel's native
    layout — and pass through; symmetric inputs have no zp tensor and
    the kernel inlines the constant zero point 8."""
    E, K8, _ = w.shape

    shifts_out = torch.tensor(
        [0, 16, 4, 20, 8, 24, 12, 28], device=w.device, dtype=torch.int32
    )
    nib_shifts = 4 * torch.arange(8, device=w.device, dtype=torch.int32)

    wq = torch.empty(E, K8, N, dtype=torch.int32, device=w.device)
    # Process one expert at a time to keep the temporary [K, N] unpack small.
    for e in range(E):
        # [K8, N, 8] -> [K8, 8, N]: the nibble dim (8, along K) must merge
        # with K8, not with N (unlike the AWQ branch where dim 2 is N/8).
        q = (
            ((w[e].unsqueeze(-1) >> nib_shifts) & 0xF)
            .permute(0, 2, 1)
            .reshape(K8 * 8, N)
        )
        wq[e] = (
            (q.view(K8, 8, N) << shifts_out.view(1, 8, 1)).sum(dim=1).to(torch.int32)
        )

    sc = scales.to(torch.float16).contiguous()
    if qzeros is None:
        return wq, sc, None
    # Asymmetric pack-quantized: the loader already presents the zps in
    # the kernel's layout ([E, G, N/8] int32, 8 nibbles per word, n
    # ascending) — validate and pass through.
    if qzeros.shape != (E, scales.shape[1], N // 8):
        raise ValueError(
            "compressed-tensors asymmetric MoE zps must be [E, G, N/8] "
            f"int32-packed, got shape {tuple(qzeros.shape)} for "
            f"w={tuple(w.shape)}, scales={tuple(scales.shape)}"
        )
    return wq, sc, qzeros.to(torch.int32).contiguous()


def _process_weights_gfx906(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None,
    w2_qzeros: torch.Tensor | None,
) -> tuple:
    w13_qweight, w13_scales, w13_zp = _repack_w4a16_gfx906_expert(
        w13, w13_scale, w13_qzeros
    )
    w2_qweight, w2_scales, w2_zp = _repack_w4a16_gfx906_expert(w2, w2_scale, w2_qzeros)
    return (
        w13_qweight,
        w2_qweight,
        w13_scales,
        w2_scales,
        w13_zp,
        w2_zp,
        None,  # w13_input_global_scale
        None,  # w2_input_global_scale
        None,  # w13_bias
        None,  # w2_bias
    )


def _repack_qzeros_kfirst_for_triton(
    qzeros: torch.Tensor | None,
    n_out: int,
) -> torch.Tensor | None:
    """Convert int4 qzeros from the checkpoint K-first layout to the layout
    the Triton WNA16 MoE kernel indexes.

    Checkpoint (CT/GPTQ, is_transposed loader): ``[E, G, N // 8]`` int32,
    8 zps per word, output column ``n = 8 * w + j`` in nibble ``j``.

    Triton kernel (``fused_moe_kernel_gptq_awq`` int4 zp branch): column
    ``n`` reads word ``n // 2`` (axis 1), nibble ``(n % 2) * 4``, group
    ``g`` on axis 2 — i.e. ``[E, N // 2, G]`` with 2 zps per word.

    The result is stored physically as ``[E, G, N // 2]`` and returned as
    a transposed view: the kernel walks axis 1 (``n // 2``) with axis 2
    (``g``) fixed inside each k-block, so N-major storage makes those
    int32 loads contiguous (a plain ``[E, N // 2, G]`` contiguous tensor
    would stride them by G words).

    Note (2026-08-25, Ornith A/B): the upstream kernel's int4 zp branch
    is pathologically slow on gfx906 regardless of this layout choice —
    both layouts measured identical decode (267-270 ms/tok, ~30x below
    the no-zp class); the slowness is in the kernel's has_zp path, not
    the zp storage. See DEVLOG-ornith-wna16.md.

    ``n_out`` is the weight's output width (K-first axis 2); the packed
    width is validated against it so a qzeros tensor in an unexpected
    layout (e.g. a different quantization source's convention) fails
    closed at weight load instead of silently mis-dequantizing.
    """
    if qzeros is None:
        return None
    E, G, words = qzeros.shape
    if qzeros.dtype != torch.int32 or words != n_out // 8 or n_out % 8:
        raise ValueError(
            "WNA16 MoE qzeros must be K-first [E, G, N_out // 8] int32 "
            "(8 zps per word) for the weight output width, got shape "
            f"{tuple(qzeros.shape)} dtype {qzeros.dtype} for N_out={n_out}"
        )
    N = n_out
    n = torch.arange(N, device=qzeros.device, dtype=torch.int32)
    z = (qzeros[:, :, n // 8] >> ((n % 8) * 4)) & 0xF  # [E, G, N]
    packed = (z[:, :, 0::2] | (z[:, :, 1::2] << 4)).to(torch.int32)
    # physical [E, G, N // 2]; logical [E, N // 2, G]
    return packed.transpose(1, 2)


def convert_to_wna16_moe_kernel_format(
    backend: WNA16MoEBackend,
    layer: torch.nn.Module,
    quant_config: QuantizationConfig | QuantizationArgs | None,
    input_dtype: torch.dtype | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> (
    tuple[
        torch.Tensor,  # w13_qweight
        torch.Tensor,  # w2_qweight
        torch.Tensor,  # w13_scales
        torch.Tensor,  # w2_scales
        torch.Tensor | None,  # w13_qzeros
        torch.Tensor | None,  # w2_qzeros
        torch.Tensor | None,  # w13_input_global_scale
        torch.Tensor | None,  # w2_input_global_scale
        torch.Tensor | None,  # w13_bias
        torch.Tensor | None,  # w2_bias
    ]
    | None
):
    """Dispatch weight post-processing to the appropriate per-backend handler.

    To add a new backend, implement a ``_process_weights_<name>`` helper and
    add a branch here. Backends that rewrite the layer's parameters in place
    (e.g. Humming) return ``None``; the caller then skips the param scatter.

    Args:
        backend: the selected ``WNA16MoEBackend``.
        layer: the ``MoERunner`` layer whose parameters are being prepared.
        quant_config: the ``QuantizationConfig`` for this layer.
        input_dtype: optional activation dtype, usually should be 16 bit.
    """
    if backend == WNA16MoEBackend.HUMMING:
        from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            convert_to_humming_moe_kernel_format,
        )

        if isinstance(quant_config, MoeWNA16Config):
            from vllm.utils.humming import HummingInputSchema

            convert_to_humming_moe_kernel_format(
                layer,
                weight_schema=_MoeWNA16HummingWeightSchema(
                    bits=quant_config.weight_bits,
                    group_size=layer.group_size,
                    has_zero_point=quant_config.has_zp,
                ),
                input_schema=HummingInputSchema(),
            )
        else:
            convert_to_humming_moe_kernel_format(
                layer, quant_config=_humming_wna16_weight_schema(quant_config)
            )
        return None

    if backend in (
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
    ):
        from vllm.model_executor.layers.quantization.auto_awq import (
            AutoAWQConfig,
        )
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQConfig,
        )

        if isinstance(quant_config, AutoAWQConfig):
            num_bits = quant_config.weight_bits
            pack_factor = quant_config.pack_factor
            group_size = quant_config.group_size
        elif isinstance(quant_config, AutoGPTQConfig):
            num_bits = quant_config.quant_type.size_bits
            pack_factor = quant_config.pack_factor
            group_size = quant_config.group_size
        elif isinstance(quant_config, QuantizationArgs):
            num_bits = quant_config.num_bits
            pack_factor = 32 // quant_config.num_bits
            group_size = (
                -1 if quant_config.group_size is None else quant_config.group_size
            )
        else:
            raise TypeError(
                "Marlin WNA16 MoE backend requires AutoGPTQConfig, AutoAWQConfig or "
                f"QuantizationArgs, got {type(quant_config).__name__}."
            )

        if isinstance(quant_config, AutoAWQConfig):
            if w13_qzeros is None or w2_qzeros is None:
                raise ValueError("AWQ Marlin MoE requires zero-point tensors.")

            return _process_awq_weights_marlin(
                layer,
                num_bits,
                pack_factor,
                group_size,
                input_dtype,
                w13,
                w2,
                w13_scale,
                w2_scale,
                w13_qzeros,
                w2_qzeros,
                w13_bias,
                w2_bias,
            )
        else:
            return _process_weights_marlin(
                layer,
                input_dtype,
                num_bits,
                pack_factor,
                group_size,
                w13,
                w2,
                w13_scale,
                w2_scale,
                w13_qzeros,
                w2_qzeros,
                w13_bias,
                w2_bias,
            )
    elif backend == WNA16MoEBackend.RDNA3:
        assert isinstance(quant_config, QuantizationArgs)
        return _process_weights_rdna3(
            w13,
            w2,
            w13_scale,
            w2_scale,
            quant_config.group_size,
        )
    elif backend == WNA16MoEBackend.CPU:
        return _process_weights_cpu(
            quant_config,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_qzeros,
            w2_qzeros,
            w13_bias,
            w2_bias,
        )
    elif backend == WNA16MoEBackend.FLASHINFER_TRTLLM:
        return _process_weights_flashinfer(
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_bias,
            w2_bias,
        )
    elif backend == WNA16MoEBackend.XPU:
        assert quant_config is not None
        (
            w13_xpu,
            w2_xpu,
            w13_scale_xpu,
            w2_scale_xpu,
            w13_bias_out,
            w2_bias_out,
        ) = _process_weights_xpu(
            layer,
            quant_config,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_bias,
            w2_bias,
        )
        return (
            w13_xpu,
            w2_xpu,
            w13_scale_xpu,
            w2_scale_xpu,
            None,  # w13_qzeros — sym int4 on XPU has none; kernel does uint4b8→s4
            None,  # w2_qzeros
            None,  # w13_input_global_scale
            None,  # w2_input_global_scale
            w13_bias_out,
            w2_bias_out,
        )
    elif backend == WNA16MoEBackend.EMULATION:
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        if isinstance(quant_config, AutoAWQConfig):
            return _process_weights_emulation_awq(
                w13,
                w2,
                w13_scale,
                w2_scale,
                w13_qzeros,
                w2_qzeros,
            )
        return _process_weights_emulation_gptq(
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_qzeros,
            w2_qzeros,
        )
    elif backend == WNA16MoEBackend.GFX906_HIP:
        return _process_weights_gfx906(
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_qzeros,
            w2_qzeros,
        )
    elif backend == WNA16MoEBackend.TRITON:
        # Two possible input layouts depending on the quantization source:
        #
        # MoeWNA16 (uint8):              (E, N_out, K // bit8_pack)  — N-first
        #   → just view as uint8 (no-op)
        #
        # AutoGPTQ/compressed-tensors (int32, K-first):
        #   (E, K // pack32, N_out)
        #   → transpose to N-first, then view as uint8 to get
        #     (E, N_out, K // bit8_pack)  [int32 = 4 bytes → 4 uint8s]
        #   Scales: (E, K // gs, N_out) → transpose → (E, N_out, K // gs)
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQConfig,
        )

        if isinstance(quant_config, (AutoGPTQConfig, QuantizationArgs)):
            # These integrations build in K-first format even when the Triton
            # backend is selected. Transpose to N-first first.
            w13_uint8 = w13.transpose(1, 2).contiguous().view(torch.uint8)
            w2_uint8 = w2.transpose(1, 2).contiguous().view(torch.uint8)
            w13_scale = w13_scale.transpose(1, 2).contiguous()
            w2_scale = w2_scale.transpose(1, 2).contiguous()
            # The checkpoint qzeros are K-first [E, G, N_out // 8] (8 zps
            # per word) for both supported sources — compressed-tensors
            # pack-quantized and auto-gptq (pack_cols2ints, same
            # 8-per-word convention); the Triton kernel indexes
            # [E, N_out // 2, G] (2 zps per word). Repack; the passthrough
            # read a transposed layout and mis-dequantized asymmetric
            # checkpoints.
            w13_qzeros = _repack_qzeros_kfirst_for_triton(
                w13_qzeros, w13.shape[2]
            )
            w2_qzeros = _repack_qzeros_kfirst_for_triton(
                w2_qzeros, w2.shape[2]
            )
        else:
            # MoeWNA16 uses N-first uint8 weights and scales.
            w13_uint8 = w13.view(torch.uint8)
            w2_uint8 = w2.view(torch.uint8)
        return (
            w13_uint8,
            w2_uint8,
            w13_scale,
            w2_scale,
            w13_qzeros,
            w2_qzeros,
            None,
            None,
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(f"Unsupported wna16 MoE backend: {backend.value}")

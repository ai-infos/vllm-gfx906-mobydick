# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""C4 (Tier 2): load-time int4 quantization of unquantized MoE layers.

Some AWQ checkpoints (e.g. Qwen3.5-35B-A3B-AWQ) list `model.layers.0.` in
`modules_to_not_convert`, so the first decoder layer's routed experts load as
plain fp16 and run on the generic Triton unquantized MoE path. On gfx906 that
path costs ~4x per call vs the custom W4A16 kernel (measured 740 us/call vs
182 us/call at M=1, C4 scoping probe of 2026-09-01), because it reads 4x more
weight bytes. This module quantizes those fp16 experts to int4 right after
loading (group size from the checkpoint's AWQ config, asymmetric AWQ
zero-point convention) and routes them through the same gfx906 W4A16 kernel
as every other MoE layer.

Default on since 2026-09-18; `VLLM_GFX906_QUANT_LAYER0_MOE=0` is the kill
switch. Quantizing a layer the checkpoint author deliberately left unquantized is
a quality trade-off, so it was opt-in until its gates passed - they all did
(DEVLOG-c4-layer0-quant.md, 2026-09-01): unit 8/8; PPL 15.9531 -> 15.9929
(delta +0.04 against a 0.5 gate); greedy serving fingerprint bit-identical; serving
A/B 84.95 -> 87.51 t/s = +3.0 % against a ~1.8 % noise floor; ~1.5 GiB returned to
graph capture.

Design notes:
- Weights load exactly as the unquantized path does (inherited create_weights
  + weight loader), then `process_weights_after_loading` quantizes in place
  and delegates to `MoeWNA16Method.process_weights_after_loading`, which runs
  the shared gfx906 repack + kernel setup. No new kernel code.
- The quantizer emits the MoeWNA16 N-first uint8 layout that
  `_repack_w4a16_wna16_layout` auto-detects:
    w:      [E, N, K/2] uint8 (byte j holds k=2j low, k=2j+1 high)
    scales: [E, N, G]   fp16
    qzeros: [E, N/2, G] uint8 (byte i holds n=2i low, n=2i+1 high)
  Dequant convention (matches the gfx906 kernel `zero_offset=0` and vllm's
  `quantize_weights`): w ~= (q - zp) * scale.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)

logger = init_logger(__name__)


def c4_quant_layer0_enabled() -> bool:
    """Env gate for C4 (default on since the gate passed; see below)."""
    return os.environ.get("VLLM_GFX906_QUANT_LAYER0_MOE", "1") != "0"


def _on_gfx906() -> bool:
    from vllm.platforms.rocm import on_gfx906

    return on_gfx906()


def _quantize_fp16_to_moe_wna16(
    w: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one expert-set weight [N, K] fp16 to the MoeWNA16 layout.

    Uses the asymmetric AWQ convention (vllm `quantize_weights`):
        scale = (max - min).clamp(min=1e-5) / 15   per group of K/group_size
        zp    = round(|min| / scale), clamped to [0, 15]
        q     = round(w / scale) + zp, clamped to [0, 15]
        dequant ~= (q - zp) * scale

    Codepoints are chosen against the *stored* fp16 scale (the value the
    kernel applies), so the reconstruction error stays within half a step of
    that scale.

    Args:
        w: [N, K] floating-point weights, K divisible by group_size.
        group_size: AWQ group size from the checkpoint config.

    Returns:
        (qweight [N, K/2] uint8, scales [N, G] in w's dtype, qzeros
        [N/2, G] uint8) with the packing documented in the module docstring.
    """
    N, K = w.shape
    assert K % group_size == 0, (
        f"K={K} not divisible by group_size={group_size}"
    )
    G = K // group_size

    wf = w.float()
    groups = wf.reshape(N, G, group_size)  # [N, G, gs]
    max_val = groups.max(dim=2, keepdim=True).values  # [N, G, 1]
    min_val = groups.min(dim=2, keepdim=True).values

    scale = ((max_val - min_val).clamp(min=1e-5) / 15.0).squeeze(2)  # [N, G]
    # The kernel applies the *stored* (fp16) scale, so pick codepoints
    # against that exact value: the reconstruction error is then bounded by
    # half a step of the scale actually used, with no extra rounding term.
    scale_w = scale.to(w.dtype)  # [N, G]
    sw = scale_w.float().unsqueeze(2)  # [N, G, 1], exact fp32 upcast

    zp = torch.round(min_val.abs() / sw).clamp(0, 15).squeeze(2).int()  # [N, G]
    q = torch.round(groups / sw) + zp.unsqueeze(2)
    q = q.clamp(0, 15).to(torch.uint8)  # [N, G, gs]

    # Pack k-pairs: byte j holds k=2j (low nibble), k=2j+1 (high nibble).
    q_nk = q.reshape(N, K)
    qweight = (q_nk[:, 0::2] | (q_nk[:, 1::2] << 4)).contiguous()  # [N, K/2]

    # Pack n-pairs for the zero points: byte i holds n=2i (low), n=2i+1 (high).
    qzeros = (zp[0::2] | (zp[1::2] << 4)).contiguous()  # [N/2, G]

    return qweight, scale_w.contiguous(), qzeros


class C4QuantizedLayer0MoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized MoE load path + post-load int4 quantization for the
    deliberately-unquantized first MoE layer on gfx906.

    Falls back to the plain unquantized behaviour whenever the C4 gate is off
    or the platform/shape preconditions are not met, so it is safe to install
    for any skipped RoutedExperts layer (for Qwen3.5-35B-A3B-AWQ that is
    exactly `model.layers.0`).
    """

    def __init__(self, moe, awq_config):
        super().__init__(moe)
        self._awq_config = awq_config
        self._c4_active = False
        self._wna16_method: MoeWNA16Method | None = None

    def process_weights_after_loading(self, layer) -> None:
        from vllm.platforms import current_platform

        if not (
            c4_quant_layer0_enabled()
            and current_platform.is_rocm()
            and _on_gfx906()
            and getattr(self._awq_config, "weight_bits", 4) == 4
        ):
            super().process_weights_after_loading(layer)
            return

        group_size = int(getattr(self._awq_config, "group_size", 128))
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        E, N13, K13 = w13.shape
        _, N2, K2 = w2.shape
        if K13 % group_size or K2 % group_size:
            logger.warning_once(
                "C4: hidden/intermediate sizes (K13=%d, K2=%d) not divisible "
                "by AWQ group_size=%d; keeping the unquantized path.",
                K13, K2, group_size,
            )
            super().process_weights_after_loading(layer)
            return

        device = w13.device
        logger.info(
            "C4: quantizing %d routed experts to int4 (group_size=%d) on %s",
            E, group_size, device.type,
        )

        q13 = torch.empty(E, N13, K13 // 2, dtype=torch.uint8, device=device)
        s13 = torch.empty(
            E, N13, K13 // group_size, dtype=w13.dtype, device=device
        )
        z13 = torch.empty(
            E, N13 // 2, K13 // group_size, dtype=torch.uint8, device=device
        )
        q2 = torch.empty(E, N2, K2 // 2, dtype=torch.uint8, device=device)
        s2 = torch.empty(
            E, N2, K2 // group_size, dtype=w2.dtype, device=device
        )
        z2 = torch.empty(
            E, N2 // 2, K2 // group_size, dtype=torch.uint8, device=device
        )

        # Quantize per expert (keeps temporaries small: one [N, K] fp32 at a
        # time). The loaded weights are contiguous logical tensors here —
        # the unquantized path's ROCm 512-byte padding happens later, inside
        # its own process_weights_after_loading, which we do not run.
        for e in range(E):
            qw, sc, zq = _quantize_fp16_to_moe_wna16(w13[e], group_size)
            q13[e].copy_(qw)
            s13[e].copy_(sc)
            z13[e].copy_(zq)
            qw, sc, zq = _quantize_fp16_to_moe_wna16(w2[e], group_size)
            q2[e].copy_(qw)
            s2[e].copy_(sc)
            z2[e].copy_(zq)

        # Register the quantized parameters under the names
        # MoeWNA16Method.process_weights_after_loading reads.
        _register_param(layer, "w13_qweight", q13)
        _register_param(layer, "w2_qweight", q2)
        _register_param(layer, "w13_scales", s13)
        _register_param(layer, "w2_scales", s2)
        _register_param(layer, "w13_qzeros", z13)
        _register_param(layer, "w2_qzeros", z2)

        # Build the WNA16 config directly from the AutoAWQConfig's already-
        # parsed fields (the checkpoint dict uses AWQ key names like "w_bit"
        # which MoeWNA16Config.from_config would not find). This mirrors
        # exactly what layers 1+ get, so layer 0 joins the same kernel path.
        moe_wna16_config = MoeWNA16Config(
            linear_quant_method="awq",
            weight_bits=int(self._awq_config.weight_bits),
            group_size=group_size,
            has_zp=bool(self._awq_config.zero_point),
            lm_head_quantized=bool(self._awq_config.lm_head_quantized),
            modules_to_not_convert=self._awq_config.modules_to_not_convert,
            full_config=self._awq_config.full_config,
        )
        layer.quant_config = moe_wna16_config
        layer.group_size = group_size
        layer.group_size_div_factor = 1

        # Free the fp16 weights: unregistering the parameters (not just
        # dropping references) releases them from the module's parameter
        # registry too, so CUDA graph capture / weight reloads never see a
        # stale fp16 tensor. The gfx906 repack re-registers w13_weight /
        # w2_weight as the int4 storage afterwards.
        layer.register_parameter("w13_weight", None)
        layer.register_parameter("w2_weight", None)
        del w13
        del w2
        torch.accelerator.empty_cache()

        # Shared gfx906 repack + kernel setup.
        self._wna16_method = MoeWNA16Method(moe_wna16_config, layer.moe_config)
        self._wna16_method.process_weights_after_loading(layer)

        # Sync the runner-visible state from the delegate. The MoE runner
        # reads these off *this* method (self._quant_method), not the
        # delegate: moe_kernel drives _fused_output_is_reduced,
        # supports_internal_mk (property over moe_kernel),
        # mk_can_overlap_shared_experts, topk_indices_dtype, and
        # do_naive_dispatch_combine. Without this sync layer 0 would look
        # like a non-MK method to the runner even though its forward runs
        # through the WNA16 kernel.
        self.moe_kernel = self._wna16_method.moe_kernel
        self.moe_quant_config = self._wna16_method.moe_quant_config
        self.experts_cls = self._wna16_method.experts_cls

        self._c4_active = True

    @property
    def supports_eplb(self) -> bool:
        # UnquantizedFusedMoEMethod advertises True, but the active WNA16
        # path does not support EPLB (base default False). Follow the path
        # that actually runs.
        if self._c4_active and self._wna16_method is not None:
            return self._wna16_method.supports_eplb
        return super().supports_eplb

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input):
        if self._c4_active and self._wna16_method is not None:
            return self._wna16_method.apply(
                layer, x, topk_weights, topk_ids, shared_experts,
                shared_experts_input,
            )
        return super().apply(
            layer, x, topk_weights, topk_ids, shared_experts,
            shared_experts_input,
        )


def _register_param(layer, name: str, tensor: torch.Tensor) -> None:
    """Register a fresh (non-trainable) parameter on the experts module."""
    layer.register_parameter(
        name, torch.nn.Parameter(tensor, requires_grad=False)
    )

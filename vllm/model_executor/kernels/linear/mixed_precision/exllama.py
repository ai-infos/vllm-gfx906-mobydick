# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx906
else:

    def on_gfx906() -> bool:
        return False


class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,
        scalar_types.uint8b128,
        scalar_types.uint4,
    ]
    # In theory supports `scalar_types.uint2b2, scalar_types.uint3b4` too but
    # currently untested so not added to the list

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        if c.act_type != torch.float16 and not (
            on_gfx906() and c.act_type == torch.float32
        ):
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )
        if c.weight_type == scalar_types.uint4 and not on_gfx906():
            return False, "AWQ Exllama routing is only enabled on gfx906"

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        if c.zero_points:
            def transform_w_zp(x):
                permute_param_layout_(x, input_dim=0, output_dim=1)
                return x.data.contiguous()

            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        else:
            # For Exllama, we need to set a zero-point tensor if there is not one
            self.w_zp_name = "qzeros"
            assert c.weight_type.has_bias()
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]
            zero_bias = c.weight_type.bias if on_gfx906() else c.weight_type.bias - 1
            zeros = torch.full(
                (groups, out_features),
                zero_bias,
                dtype=torch.int32,
                device=device,
            )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            ops.gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            dtype = (
                torch.float16
                if on_gfx906() and c.act_type == torch.float32
                else c.act_type
            )
            return x.to(dtype=dtype)

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp = self._get_weight_params(layer)
        # gfx906: AWQ (uint4) routing keeps fp16 scales and uses the GPTQv2
        # zero-point format; every other path keeps upstream's GPTQv1.
        use_v2_format = on_gfx906()

        assert w_zp is not None, "Zero points are required by Exllama"

        x_2d_fp16 = (
            x_2d.to(torch.float16)
            if on_gfx906() and x_2d.dtype == torch.float32
            else x_2d
        )

        output = ops.gptq_gemm(
            x_2d_fp16,
            w_q,
            w_zp,
            w_s,
            True,
            use_v2_format,
            c.weight_type.size_bits,
        )

        if output.dtype != x.dtype:
            output = output.to(x.dtype)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

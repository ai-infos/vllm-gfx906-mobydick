# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe import RoutedExperts, fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


def _make_inc_config(extra_config: dict[str, dict]) -> INCConfig:
    return INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        extra_config=extra_config,
    )


def test_inc_uses_embedding_method_for_unquantized_vocab_embedding():
    config = _make_inc_config(
        {"language_model.model.embed_tokens": {"bits": 16, "data_type": "float"}}
    )

    with (
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        embedding = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=8,
            quant_config=config,
            prefix="language_model.model.embed_tokens",
        )

    assert isinstance(embedding.quant_method, UnquantizedEmbeddingMethod)


def test_inc_uses_embedding_method_for_unquantized_lm_head():
    config = _make_inc_config(
        {"language_model.lm_head": {"bits": 16, "data_type": "float"}}
    )

    with (
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        lm_head = ParallelLMHead(
            num_embeddings=16,
            embedding_dim=8,
            quant_config=config,
            prefix="language_model.lm_head",
        )

    assert isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod)


def _import_minimax_m3_classes(module_name: str):
    module = importlib.import_module(module_name)
    return (
        module.MiniMaxM3SparseForCausalLM,
        module.MiniMaxM3SparseForConditionalGeneration,
    )


def test_minimax_m3_mapper_handles_terminal_vision_mlp_names():
    _, vl_cls = _import_minimax_m3_classes("vllm.models.minimax_m3.amd.model")

    mapped = vl_cls.hf_to_vllm_mapper.apply_dict(
        {
            "vision_tower.vision_model.encoder.layers.24.mlp.fc2": {
                "bits": 16
            },
            "multi_modal_projector.linear_1": {"bits": 16},
        }
    )

    assert mapped == {
        "vision_tower.vision_model.encoder.layers.24.fc2": {"bits": 16},
        "vision_tower.multi_modal_projector.linear_1": {"bits": 16},
    }


def test_minimax_m3_platform_copies_publish_quant_mappings():
    for module_name in (
        "vllm.models.minimax_m3.amd.model",
        "vllm.models.minimax_m3.nvidia.model",
    ):
        causal_cls, vl_cls = _import_minimax_m3_classes(module_name)

        for model_cls in (causal_cls, vl_cls):
            assert model_cls.packed_modules_mapping["qkv_proj"] == [
                "q_proj",
                "k_proj",
                "v_proj",
            ]
            assert model_cls.packed_modules_mapping["gate_up_proj"] == [
                "gate_proj",
                "up_proj",
            ]


def test_inc_resolves_unquantized_fused_minimax_m3_qkv_from_shards():
    _, vl_cls = _import_minimax_m3_classes("vllm.models.minimax_m3.amd.model")
    layer_name = "vision_tower.vision_model.encoder.layers.24.self_attn.qkv_proj"
    config = _make_inc_config(
        {
            layer_name.replace("qkv_proj", shard): {
                "bits": 16,
                "data_type": "float",
            }
            for shard in ("q_proj", "k_proj", "v_proj")
        }
    )
    config.packed_modules_mapping = vl_cls.packed_modules_mapping

    assert config.get_layer_config(torch.nn.Module(), layer_name) == (16, 128, True)


def test_inc_gptq_moe_wna16_fallback_sets_desc_act_false():
    config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        backend="gptq",
    )
    layer = object.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace()

    quant_method = config.get_quant_method(
        layer,
        "language_model.model.layers.3.block_sparse_moe.experts",
    )

    assert isinstance(quant_method, MoeWNA16Method)
    assert quant_method.quant_config.full_config["desc_act"] is False


def test_wna16_moe_quant_configs_preserve_swiglu_oai_params():
    for builder in (int4_w4a16_moe_quant_config, int8_w8a16_moe_quant_config):
        quant_config = builder(
            w1_scale=torch.ones(1),
            w2_scale=torch.ones(1),
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )

        assert quant_config.gemm1_alpha == 1.702
        assert quant_config.gemm1_beta == 1.0
        assert quant_config.gemm1_clamp_limit == 7.0


def test_moe_wna16_method_preserves_minimax_swiglu_oai_params():
    quant_config = MoeWNA16Config(
        linear_quant_method="gptq",
        weight_bits=4,
        group_size=128,
        has_zp=False,
        lm_head_quantized=False,
        modules_to_not_convert=[],
        full_config={"quant_method": "gptq", "bits": 4},
    )
    method = MoeWNA16Method(quant_config, SimpleNamespace())
    layer = SimpleNamespace(
        w13_scales=torch.ones(1),
        w2_scales=torch.ones(1),
        group_size=128,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    )

    fused_quant_config = method.get_fused_moe_quant_config(layer)

    assert fused_quant_config.gemm1_alpha == 1.702
    assert fused_quant_config.gemm1_beta == 1.0
    assert fused_quant_config.gemm1_clamp_limit == 7.0


def test_fused_experts_forwards_swiglu_oai_params_to_custom_op(monkeypatch):
    captured_kwargs = {}

    def fake_fused_experts(**kwargs):
        captured_kwargs.update(kwargs)
        return torch.empty_like(kwargs["hidden_states"])

    monkeypatch.setattr(
        fused_moe_module.torch.ops.vllm,
        "fused_experts",
        fake_fused_experts,
        raising=False,
    )
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )

    fused_experts(
        hidden_states=torch.ones(1, 2),
        w1=torch.ones(1, 4, 2),
        w2=torch.ones(1, 2, 2),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        quant_config=quant_config,
    )

    assert captured_kwargs["gemm1_alpha"] == 1.702
    assert captured_kwargs["gemm1_beta"] == 1.0
    assert captured_kwargs["gemm1_clamp_limit"] == 7.0


def test_fused_experts_op_forwards_swiglu_oai_params_to_impl(monkeypatch):
    captured_args = ()

    def fake_fused_experts_impl(*args):
        nonlocal captured_args
        captured_args = args
        return torch.empty_like(args[0])

    monkeypatch.setattr(
        fused_moe_module,
        "fused_experts_impl",
        fake_fused_experts_impl,
    )

    fused_moe_module.fused_experts_op(
        hidden_states=torch.ones(1, 2),
        w1=torch.ones(1, 4, 2),
        w2=torch.ones(1, 2, 2),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE.value,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )

    assert captured_args[-3:] == (1.702, 1.0, 7.0)


def test_inc_gptq_skips_unsupported_layers_without_quant_debug(caplog):
    config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        backend="gptq",
    )

    with caplog.at_level(logging.DEBUG):
        quant_method = config.apply_gptq_quant_layer(
            torch.nn.Module(),
            "language_model.model.layers.0.self_attn.attn",
            backend="gptq",
        )

    assert quant_method is None
    assert "Type: Module" not in caplog.text

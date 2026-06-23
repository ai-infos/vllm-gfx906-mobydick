# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerTritonImpl,
    select_indexer_impl_cls,
)
from vllm.models.minimax_m3.common.ops.index_topk import (
    _use_fp16_dot_for_fp32_inputs as _use_index_fp16_dot_for_fp32_inputs,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _use_fp16_dot_for_fp32_inputs as _use_sparse_fp16_dot_for_fp32_inputs,
)
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseBackend
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype


def test_cache_config_accepts_float32_kv_cache_dtype():
    config = CacheConfig(cache_dtype="float32")

    assert config.cache_dtype == "float32"
    assert kv_cache_dtype_str_to_dtype(config.cache_dtype, None) is torch.float32


def test_attention_config_normalizes_indexer_fp32_alias():
    config = AttentionConfig(indexer_kv_dtype="fp32")

    assert config.indexer_kv_dtype == "float32"


def test_minimax_m3_backends_advertise_float32_support():
    assert torch.float32 in MiniMaxM3SparseBackend.supported_dtypes
    assert torch.float32 in MiniMaxM3IndexerBackend.supported_dtypes
    assert "float32" in MiniMaxM3SparseBackend.supported_kv_cache_dtypes
    assert "float32" in MiniMaxM3IndexerBackend.supported_kv_cache_dtypes


def test_minimax_m3_indexer_cache_uses_float32_spec_dtype():
    vllm_config = VllmConfig(cache_config=CacheConfig(block_size=128))

    with set_current_vllm_config(vllm_config):
        index_cache = MiniMaxM3IndexerCache(
            head_dim=128,
            prefix="test_minimax_m3_fp32.index_cache",
            indexer_kv_dtype="float32",
        )

    kv_cache_spec = index_cache.get_kv_cache_spec(vllm_config)

    assert index_cache.indexer_kv_dtype == "float32"
    assert index_cache.dtype is torch.float32
    assert kv_cache_spec.dtype is torch.float32


def test_minimax_m3_indexer_triton_impl_accepts_float32():
    assert (
        select_indexer_impl_cls(indexer_kv_dtype="float32")
        is MiniMaxM3IndexerTritonImpl
    )
    assert (
        select_indexer_impl_cls(indexer_kv_dtype="fp32")
        is MiniMaxM3IndexerTritonImpl
    )


@pytest.mark.parametrize(
    ("q_dtype", "cache_dtype", "expected"),
    [
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_minimax_m3_sparse_attention_uses_fp16_dot_for_native_mixed_fp32(
    q_dtype, cache_dtype, expected
):
    assert _use_sparse_fp16_dot_for_fp32_inputs(q_dtype, cache_dtype) is expected


@pytest.mark.parametrize(
    "fp8_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_minimax_m3_sparse_attention_keeps_fp8_cache_path_separate(fp8_dtype):
    assert not _use_sparse_fp16_dot_for_fp32_inputs(torch.float32, fp8_dtype)


@pytest.mark.parametrize(
    ("q_dtype", "cache_dtype", "expected"),
    [
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_minimax_m3_indexer_uses_fp16_dot_for_mixed_fp32(
    q_dtype, cache_dtype, expected
):
    assert _use_index_fp16_dot_for_fp32_inputs(q_dtype, cache_dtype) is expected


def test_amd_dense_attention_casts_fp32_qkv_to_fp16_and_restores_output(
    monkeypatch,
):
    amd_module = importlib.import_module("vllm.models.minimax_m3.amd.model")
    captured = {}

    def fake_fused_op(qkv, *args):
        captured["fused_qkv_dtype"] = qkv.dtype
        captured["cos_sin_cache_dtype"] = args[2].dtype

    def fake_qkv_proj(hidden_states):
        qkv = torch.zeros(hidden_states.shape[0], 8, dtype=torch.float32)
        return qkv, None

    def fake_attn(q, k, v):
        captured["attn_q_dtype"] = q.dtype
        captured["attn_k_dtype"] = k.dtype
        captured["attn_v_dtype"] = v.dtype
        return q

    def fake_o_proj(attn_output):
        captured["o_proj_input_dtype"] = attn_output.dtype
        return attn_output, None

    monkeypatch.setattr(
        amd_module.ops,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        fake_fused_op,
    )

    layer = SimpleNamespace(
        q_size=4,
        kv_size=2,
        num_heads=2,
        num_kv_heads=1,
        qkv_proj=fake_qkv_proj,
        q_norm=SimpleNamespace(
            weight=torch.zeros(2),
            variance_epsilon=1e-6,
        ),
        k_norm=SimpleNamespace(weight=torch.zeros(2)),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.empty(1, dtype=torch.float32),
            rotary_dim=2,
        ),
        attn=fake_attn,
        o_proj=fake_o_proj,
    )

    hidden_states = torch.zeros(3, 4, dtype=torch.float32)
    output = amd_module.MiniMaxM3Attention.forward(
        layer, torch.arange(3), hidden_states
    )

    assert captured == {
        "fused_qkv_dtype": torch.float16,
        "cos_sin_cache_dtype": torch.float16,
        "attn_q_dtype": torch.float16,
        "attn_k_dtype": torch.float16,
        "attn_v_dtype": torch.float16,
        "o_proj_input_dtype": torch.float32,
    }
    assert output.dtype is torch.float32


def test_amd_sparse_attention_casts_fp32_qkv_to_fp16_and_restores_output(
    monkeypatch,
):
    amd_module = importlib.import_module("vllm.models.minimax_m3.amd.model")
    captured = {}

    def fake_qkv_proj(hidden_states):
        qkv = torch.zeros(hidden_states.shape[0], 10, dtype=torch.float32)
        return qkv, None

    def fake_fused_op(qkv, *args):
        captured["fused_qkv_dtype"] = qkv.dtype
        captured["cos_sin_cache_dtype"] = args[2].dtype
        captured["kv_cache_dtype"] = args[13].dtype
        captured["index_cache_dtype"] = args[14].dtype
        q_out = args[16]
        index_q_out = args[17]
        q_out.copy_(qkv[:, : q_out.shape[1]])
        index_start = q_out.shape[1] + 2 * 2
        index_q_out.copy_(qkv[:, index_start : index_start + index_q_out.shape[1]])

    def fake_run_attention(query, index_query, output):
        captured["query_dtype"] = query.dtype
        captured["index_query_dtype"] = index_query.dtype
        return query

    def fake_o_proj(attn_output):
        captured["o_proj_input_dtype"] = attn_output.dtype
        return attn_output, None

    monkeypatch.setattr(
        amd_module.ops,
        "fused_minimax_m3_qknorm_rope_kv_insert",
        fake_fused_op,
    )
    monkeypatch.setattr(
        amd_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            slot_mapping={
                "layers.3.self_attn.attn": torch.arange(3),
                "layers.3.self_attn.attn.index_cache": torch.arange(3),
            }
        ),
    )

    index_cache = SimpleNamespace(
        prefix="layers.3.self_attn.attn.index_cache",
        kv_cache=torch.empty(1, 128, 2, dtype=torch.float16),
    )
    layer = SimpleNamespace(
        hidden_size=4,
        q_size=4,
        kv_size=2,
        num_heads=2,
        num_kv_heads=1,
        idx_head_dim=2,
        index_q_size=2,
        num_idx_heads=1,
        qkv_proj=fake_qkv_proj,
        q_norm=SimpleNamespace(
            weight=torch.zeros(2),
            variance_epsilon=1e-6,
        ),
        k_norm=SimpleNamespace(weight=torch.zeros(2)),
        index_q_norm=SimpleNamespace(weight=torch.zeros(2)),
        index_k_norm=SimpleNamespace(weight=torch.zeros(2)),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.empty(1, dtype=torch.float32),
            rotary_dim=2,
        ),
        layer_name="layers.3.self_attn.attn",
        kv_cache=torch.empty(1, 2, 128, 1, 2, dtype=torch.float16),
        indexer=SimpleNamespace(index_cache=index_cache),
        _fp8_kv=False,
        _run_attention=fake_run_attention,
        o_proj=fake_o_proj,
    )

    hidden_states = torch.zeros(3, 4, dtype=torch.float32)
    output = amd_module.MiniMaxM3SparseAttention.forward(
        layer, torch.arange(3), hidden_states
    )

    assert captured == {
        "fused_qkv_dtype": torch.float16,
        "cos_sin_cache_dtype": torch.float16,
        "kv_cache_dtype": torch.float16,
        "index_cache_dtype": torch.float16,
        "query_dtype": torch.float16,
        "index_query_dtype": torch.float16,
        "o_proj_input_dtype": torch.float32,
    }
    assert output.dtype is torch.float32


def test_rocm_fp32_kv_cache_update_uses_triton_writer(monkeypatch):
    try:
        from vllm.v1.attention.backends import rocm_attn as rocm_attn_module
    except Exception as exc:
        pytest.skip(f"ROCm attention backend unavailable: {exc}")

    captured = {}

    def fail_native_writer(*args, **kwargs):
        raise AssertionError("native ROCm writer should not handle float32 KV")

    def fake_triton_writer(*args, **kwargs):
        captured["called"] = True
        captured["kv_cache_dtype"] = args[5]

    monkeypatch.setattr(
        rocm_attn_module.PagedAttention,
        "write_to_paged_cache",
        fail_native_writer,
    )
    monkeypatch.setattr(
        rocm_attn_module,
        "triton_reshape_and_cache_flash",
        fake_triton_writer,
    )

    impl = object.__new__(rocm_attn_module.RocmAttentionImpl)
    impl.attn_type = rocm_attn_module.AttentionType.DECODER
    impl.num_kv_heads = 1
    impl.head_size = 128
    impl.kv_cache_dtype = "float32"

    layer = SimpleNamespace(_k_scale=torch.ones(()), _v_scale=torch.ones(()))
    key = torch.zeros(1, 1, 128, dtype=torch.float32)
    value = torch.zeros(1, 1, 128, dtype=torch.float32)
    kv_cache = torch.zeros(2, 1, 16, 1, 128, dtype=torch.float32)
    slot_mapping = torch.zeros(1, dtype=torch.long)

    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    assert captured == {"called": True, "kv_cache_dtype": "float32"}

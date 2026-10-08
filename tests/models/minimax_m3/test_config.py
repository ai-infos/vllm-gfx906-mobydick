# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.multimodal import MultiModalConfig
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerCache,
    MiniMaxM3IndexerTritonImpl,
    select_indexer_impl_cls,
)
from vllm.models.minimax_m3.common.mm_preprocess import MiniMaxM3VLProcessingInfo
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseBackend
from vllm.transformers_utils.processors.minimax_m3 import (
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLVideoProcessor,
)


def test_attention_config_normalizes_indexer_fp16_alias():
    config = AttentionConfig(indexer_kv_dtype="fp16")

    assert config.indexer_kv_dtype == "float16"


def test_minimax_m3_indexer_cache_uses_float16_spec_dtype():
    vllm_config = VllmConfig(cache_config=CacheConfig(block_size=128))

    with set_current_vllm_config(vllm_config):
        index_cache = MiniMaxM3IndexerCache(
            head_dim=128,
            prefix="test_minimax_m3.index_cache",
            indexer_kv_dtype="float16",
        )

    kv_cache_spec = index_cache.get_kv_cache_spec(vllm_config)

    assert index_cache.indexer_kv_dtype == "float16"
    assert index_cache.dtype is torch.float16
    assert kv_cache_spec.dtype is torch.float16


def test_minimax_m3_indexer_triton_impl_accepts_float16():
    assert (
        select_indexer_impl_cls(topk_blocks=16, indexer_kv_dtype="float16")
        is MiniMaxM3IndexerTritonImpl
    )
    assert (
        select_indexer_impl_cls(topk_blocks=16, indexer_kv_dtype="fp16")
        is MiniMaxM3IndexerTritonImpl
    )


def test_minimax_m3_backends_advertise_float16_kv_cache():
    assert "float16" in MiniMaxM3SparseBackend.supported_kv_cache_dtypes
    assert "float16" in MiniMaxM3IndexerBackend.supported_kv_cache_dtypes


class _FakeModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeImpl:
    def __init__(self, *args, **kwargs):
        pass


def _make_sparse_text_config():
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        partial_rotary_factor=0.5,
        max_position_embeddings=1024,
        sparse_attention_config={
            "sparse_num_index_heads": 1,
            "sparse_index_dim": 8,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )


def test_minimax_m3_platform_sparse_attention_passes_indexer_dtype():
    for module_name in (
        "vllm.models.minimax_m3.amd.model",
        "vllm.models.minimax_m3.nvidia.model",
    ):
        module = importlib.import_module(module_name)
        indexer_kwargs = []

        def make_indexer(*args, **kwargs):
            indexer_kwargs.append(kwargs)
            return _FakeModule()

        vllm_config = VllmConfig(
            attention_config=AttentionConfig(indexer_kv_dtype="fp16"),
            cache_config=CacheConfig(cache_dtype="float16", block_size=128),
        )

        # The amd and nvidia sparse-attention classes diverged (split indexer
        # projections + the AITER selector vs the fused QKV projection), so
        # patch only the names each module actually exposes.
        patches = [
            patch.object(
                module, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(module, "RowParallelLinear", _FakeModule),
            patch.object(module, "MiniMAXGemmaRMSNorm", _FakeModule),
            patch.object(
                module,
                "select_main_backend_and_impl_cls",
                return_value=(object, _FakeImpl),
            ),
            patch.object(module, "MiniMaxM3Indexer", side_effect=make_indexer),
        ]
        # amd builds the rotary with `_build_rotary_emb`, nvidia with `get_rope`.
        for fn_name in ("_build_rotary_emb", "get_rope"):
            if hasattr(module, fn_name):
                patches.append(
                    patch.object(module, fn_name, return_value=SimpleNamespace())
                )
        for cls_name in (
            "QKVParallelLinear",
            "ReplicatedLinear",
            "MinimaxM3QKVParallelLinearWithIndexer",
        ):
            if hasattr(module, cls_name):
                patches.append(patch.object(module, cls_name, _FakeModule))
        if hasattr(module, "select_aiter_indexer_impl_cls"):
            patches.append(
                patch.object(module, "select_aiter_indexer_impl_cls", return_value=None)
            )
        if hasattr(module, "get_tensor_model_parallel_rank"):
            patches.append(
                patch.object(module, "get_tensor_model_parallel_rank", return_value=0)
            )

        with set_current_vllm_config(vllm_config):
            for p in patches:
                p.start()
            try:
                module.MiniMaxM3SparseAttention(
                    config=_make_sparse_text_config(),
                    layer_id=3,
                    prefix=f"{module_name}.layers.3.self_attn",
                    cache_config=vllm_config.cache_config,
                )
            finally:
                for p in patches:
                    p.stop()

        assert indexer_kwargs[0]["indexer_kv_dtype"] == "float16"


class _MiniMaxM3ProcessingInfoForTest(MiniMaxM3VLProcessingInfo):
    def __init__(self):
        pass

    def get_image_processor(self, **kwargs):
        return MiniMaxM3VLImageProcessor()

    def get_video_processor(self, **kwargs):
        return MiniMaxM3VLVideoProcessor()


def test_minimax_m3_default_video_profile_budget_is_worst_case():
    info = _MiniMaxM3ProcessingInfoForTest()

    assert info.get_num_frames_with_most_features(1_048_576, {"video": 1}) == 500
    assert info.get_max_video_tokens(1_048_576, {"video": 1}) == 192000


def test_minimax_m3_video_zero_limit_disables_video_budget():
    ctx = SimpleNamespace(
        get_mm_config=lambda: MultiModalConfig(
            limit_per_prompt={"image": 1, "video": 0}
        )
    )
    info = MiniMaxM3VLProcessingInfo(ctx)  # type: ignore[arg-type]

    assert info.allowed_mm_limits["image"] == 1
    assert info.allowed_mm_limits["video"] == 0


def test_minimax_m3_gfx906_routes_to_triton_not_cdna_aiter():
    """gfx906 must take the generic gfx906-safe paths, never the CDNA/AITER ones.

    The 0.30.0 merge took upstream's selector (`select_main_backend_and_impl_cls`
    / `select_aiter_indexer_impl_cls`), which are gfx942/gfx950-gated. This pins
    that the MI50 still lands on `MiniMaxM3SparseBackend` + the Triton impl and
    on the `common.ops` sparse-attn kernels (which carry the gfx906 LDS-safe
    launch kwargs) rather than `amd.ops`.
    """
    from vllm.models.minimax_m3.common import sparse_attention as sa_module
    from vllm.models.minimax_m3.common.ops import sparse_attn as common_attn
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseTritonImpl,
        select_main_backend_and_impl_cls,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx906

    if not (current_platform.is_rocm() and on_gfx906()):
        import pytest

        pytest.skip("gfx906-only routing gate")

    backend, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=16, kv_cache_dtype="float16", num_kv_heads=1
    )
    assert backend is MiniMaxM3SparseBackend
    assert impl_cls is MiniMaxM3SparseTritonImpl

    # The attend symbols must come from common.ops (LDS-safe on MI50), not amd.ops.
    assert sa_module.minimax_m3_sparse_attn is common_attn.minimax_m3_sparse_attn
    assert (
        sa_module.minimax_m3_sparse_attn.__module__
        == "vllm.models.minimax_m3.common.ops.sparse_attn"
    )

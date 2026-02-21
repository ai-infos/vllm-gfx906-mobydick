# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


class ROCMAiterMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]:
        return ROCMAiterMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]:
        return ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]:
        return ROCMAiterMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class ROCMAiterMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    qo_indptr: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indptr_rest: torch.Tensor | None = None

    block_size: int = 1
    topk_tokens: int = 2048


@dataclass
class ROCMAiterMLASparseMetadataBuilder(
    AttentionMetadataBuilder[ROCMAiterMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.qo_indptr = torch.arange(
                0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
            )
            self.paged_kv_last_page_len = torch.ones(
                max_num_batched_tokens, dtype=torch.int32, device=device
            )

            # These two needs to be calculated in runtime,
            # but we still needs to prepare the buffer
            self.paged_kv_indices = torch.zeros(
                [max_num_batched_tokens * self.topk_tokens],
                dtype=torch.int32,
                device=device,
            )
            self.paged_kv_indptr = torch.zeros(
                [max_num_batched_tokens + 1], dtype=torch.int32, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ROCMAiterMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
            self.paged_kv_indices.fill_(0)
            self.paged_kv_indptr.fill_(0)
            qo_indptr = self.qo_indptr[: num_tokens + 1]
            paged_kv_last_page_len = self.paged_kv_last_page_len[:num_tokens]
            paged_kv_indices = self.paged_kv_indices[: num_tokens * self.topk_tokens]
            paged_kv_indptr = self.paged_kv_indptr[: num_tokens + 1]
            paged_kv_indptr_rest = self.paged_kv_indptr[num_tokens + 1 :]
        else:
            qo_indptr = None
            paged_kv_last_page_len = None
            paged_kv_indices = None
            paged_kv_indptr = None
            paged_kv_indptr_rest = None


        metadata = ROCMAiterMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            qo_indptr=qo_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indptr_rest=paged_kv_indptr_rest,
        )
        return metadata


# Take from
# https://github.com/deepseek-ai/FlashMLA/blob/082094b793fcc7452977d0a71a00e266a2e3061e/tests/ref.py
def reference_mla_sparse_prefill(
    q: torch.Tensor, # in kv dtype
    kv: torch.Tensor, 
    indices: torch.Tensor, 
    sm_scale: float, 
    d_v: int,
) -> torch.Tensor:
    """
    Returns:
    - o: [s_q, h_q, dv]
    """
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    s_q, h_q, d_qk = q.shape  # [s_q, h_q, d_qk]
    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_mask = (indices < 0) | (indices >= s_kv)    # [s_q, topk]
    indices[invalid_mask] = 0

    gathered_kv = kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk)   # [s_q, topk, d_qk]
    if kv.dtype == torch.float32:
        P = q @ gathered_kv.transpose(1, 2) # [s_q, h_q, topk]
    else: # q and kv are both fp16 or bf16
        P = (q @ gathered_kv.transpose(1, 2)).float() # 16 bits matmul for performance
    P.masked_fill_(invalid_mask.unsqueeze(1), float("-inf"))
    P *= sm_scale

    orig_lse = torch.logsumexp(P, dim=-1)   # [s_q, h_q]
    s_for_o = torch.exp(P - orig_lse.unsqueeze(-1))
    if kv.dtype == torch.float32:
        out = s_for_o @ gathered_kv[..., :d_v]
    else:
        out = s_for_o.to(kv.dtype) @ gathered_kv[..., :d_v]

    return out # in kv dtype


class ROCMAiterMLASparseImpl(SparseMLAAttentionImpl[ROCMAiterMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        assert indexer is not None
        self.topk_indices_buffer: torch.Tensor | None = indexer.topk_indices_buffer

    def _forward_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: ROCMAiterMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)

        if envs.VLLM_ROCM_MLA_SPARSE_FP16:
            output = reference_mla_sparse_prefill(
                q, kv_c_and_k_pe_cache, topk_indices,
                self.softmax_scale, self.kv_lora_rank,
            )
        else:
            seq_len = (topk_indices != -1).sum(dim=-1)
            torch.cumsum(seq_len, dim=0, out=attn_metadata.paged_kv_indptr[1:])
            attn_metadata.paged_kv_indptr_rest.fill_(attn_metadata.paged_kv_indptr[-1])
            fetch_id_to_ragged_triton(
                topk_indices,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.topk_tokens,
            )
            output = torch.empty(
                [num_tokens, self.num_heads, self.kv_lora_rank],
                dtype=q.dtype,
                device=q.device,
            )
            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_c_and_k_pe_cache,
                output,
                self.scale,
                attn_metadata.qo_indptr,
                1,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
            )

        return output[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        attn_out = self._forward_kv(
            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
        )

        return attn_out, None

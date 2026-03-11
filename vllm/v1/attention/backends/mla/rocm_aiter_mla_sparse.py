# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
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
    MultipleOf,
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
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "half",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 32, 64]

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
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


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
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

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


@triton.jit
def _mla_sparse_vec_kernel(
    output_ptr, query_ptr, kv_ptr, topk_indices_ptr,
    stride_out_sq: tl.int64, stride_out_hq: tl.int64,
    stride_q_sq: tl.int64, stride_q_hq: tl.int64,
    stride_kv_skv: tl.int64,
    stride_idx_sq: tl.int64,
    scale,
    s_kv: tl.int32,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    TOPK: tl.constexpr,
    D_SCORE_CHUNK: tl.constexpr,
    D_V_CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_hb = tl.program_id(1)
    pid_dv = tl.program_id(2)

    head_start = pid_hb * BLOCK_M
    dv_start = pid_dv * D_V_CHUNK

    offs_m = tl.arange(0, BLOCK_M)
    offs_dk = tl.arange(0, D_SCORE_CHUNK)
    offs_dv = tl.arange(0, D_V_CHUNK)
    offs_tk = tl.arange(0, TILE_K)

    q_base = query_ptr + pid_token * stride_q_sq
    idx_base = topk_indices_ptr + pid_token * stride_idx_sq

    M_val = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L_val = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_V_CHUNK], dtype=tl.float32)

    for t_start in range(0, TOPK, TILE_K):
        t_mask = (t_start + offs_tk) < TOPK
        kv_pos = tl.load(idx_base + t_start + offs_tk, mask=t_mask, other=0)
        valid = t_mask & (kv_pos >= 0) & (kv_pos < s_kv)
        kv_pos = tl.where(valid, kv_pos, 0)

        S = tl.zeros([BLOCK_M, TILE_K], dtype=tl.float32)

        for d_start in range(0, D_QK, D_SCORE_CHUNK):
            d_offs = d_start + offs_dk
            q_ptrs = (q_base
                      + (head_start + offs_m[:, None]) * stride_q_hq
                      + d_offs[None, :])
            Q_d = tl.load(q_ptrs)

            k_ptrs = (kv_ptr
                      + kv_pos[None, :] * stride_kv_skv
                      + d_offs[:, None])
            K_d = tl.load(k_ptrs, mask=valid[None, :], other=0.0)

            S = tl.dot(Q_d, K_d, acc=S)

        S *= scale
        S = tl.where(valid[None, :], S, float("-inf"))

        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(M_val, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)

        alpha = tl.exp(M_val - m_new)
        P = tl.exp(S - m_new[:, None])
        l_j = tl.sum(P, axis=1)

        acc = acc * alpha[:, None]
        L_val = L_val * alpha + l_j
        M_val = m_new

        v_ptrs = (kv_ptr
                  + kv_pos[:, None] * stride_kv_skv
                  + (dv_start + offs_dv)[None, :])
        V = tl.load(v_ptrs, mask=valid[:, None], other=0.0)

        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    acc = acc / L_val[:, None]

    out_ptrs = (output_ptr
                + pid_token * stride_out_sq
                + (head_start + offs_m[:, None]) * stride_out_hq
                + (dv_start + offs_dv)[None, :])
    tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))


def triton_mla_sparse_vec(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    BLOCK_M: int = 16,
    TILE_K: int = 32,
    D_V_CHUNK: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    kv_flat = kv.view(s_kv, d_qk).contiguous()

    if indices.ndim == 3:
        indices = indices[:, 0, :]
    indices = indices.to(torch.int32).contiguous()
    topk = indices.shape[1]

    out = torch.empty((s_q, h_q, d_v), device=q.device, dtype=q.dtype)

    D_SCORE_CHUNK = 64
    num_hb = h_q // BLOCK_M
    num_dv = d_v // D_V_CHUNK

    grid = (s_q, num_hb, num_dv)

    _mla_sparse_vec_kernel[grid](
        output_ptr=out, query_ptr=q, kv_ptr=kv_flat,
        topk_indices_ptr=indices,
            stride_out_sq=out.stride(0), stride_out_hq=out.stride(1),
            stride_q_sq=q.stride(0), stride_q_hq=q.stride(1),
            stride_kv_skv=kv_flat.stride(0),
            stride_idx_sq=indices.stride(0),
            scale=sm_scale, s_kv=s_kv,
            D_QK=d_qk, D_V=d_v, TOPK=topk,
            D_SCORE_CHUNK=D_SCORE_CHUNK,
            D_V_CHUNK=D_V_CHUNK, BLOCK_M=BLOCK_M, TILE_K=TILE_K,
            num_warps=num_warps, num_stages=1,
        )
    return out


# Inspired from
# https://github.com/deepseek-ai/FlashMLA/blob/082094b793fcc7452977d0a71a00e266a2e3061e/tests/ref.py
def reference_mla_sparse_prefill(
    q: torch.Tensor,       # [s_q, h_q, d_qk] in kv dtype
    kv: torch.Tensor,      # [total_kv, 1, d_qk]
    indices: torch.Tensor, # [s_q, 1, topk]
    sm_scale: float,
    d_v: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    #GPU reference that chunks over QUERY tokens to avoid OOM.
    #Memory per chunk: chunk_size * topk * d_qk * 4 bytes (FP32)
    #With chunk_size=512, topk=2048, d_qk=576:
    # 512 * 2048 * 576 * 4 = 2416 MB (safe on 32GB GPU)

    indices = indices[:, 0, :]  # [s_q, topk]
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    s_q, h_q, d_qk = q.shape  # [s_q, h_q, d_qk]

    out = torch.empty(s_q, h_q, d_v, device=q.device, dtype=kv.dtype)
    
    for start in range(0, s_q, chunk_size):
        end = min(start + chunk_size, s_q)

        idx_chunk = indices[start:end]  # [cs, topk]

        # Mark invalid indices (-1 or out-of-bounds)
        invalid_mask = (idx_chunk < 0) | (idx_chunk >= s_kv) # [s_q, topk]
        idx_chunk[invalid_mask] = 0

        # Gather: [cs, topk, d_qk] this is the memory-critical step
        gathered_kv = kv.index_select(dim=0, index=idx_chunk.flatten()).reshape(end-start, topk, d_qk)  # [cs, topk, d_qk]

        if kv.dtype == torch.float32:
            P = q[start:end] @ gathered_kv.transpose(1, 2) # [s_q, h_q, topk]
        else: # q and kv are both fp16 or bf16
            P = (q[start:end] @ gathered_kv.transpose(1, 2)).float() # 16 bits matmul for performance
    
        P.masked_fill_(invalid_mask.unsqueeze(1), float("-inf"))
        P *= sm_scale
        
        orig_lse = torch.logsumexp(P, dim=-1)   # [s_q, h_q]
        s_for_o = torch.exp(P - orig_lse.unsqueeze(-1)) # [cs, h_q, topk]
        
        if kv.dtype == torch.float32:
            out[start:end] = s_for_o @ gathered_kv[..., :d_v]
        else:
            out[start:end] = s_for_o.to(kv.dtype) @ gathered_kv[..., :d_v]

        # Free gather tensor immediately to reduce memory pressure
        del gathered_kv, P

    return out # in kv dtype


def mla_sparse(
    q: torch.Tensor, # in kv dtype
    kv: torch.Tensor, 
    indices: torch.Tensor, 
    sm_scale: float, 
    d_v: int,
) -> torch.Tensor:
    """
    Returns:
    - o: [s_q, h_q, dv]
    
    Smart dispatch implementation to avoid OOM on MI50 (gfx906) while maximizing speed.
    """
    s_q = q.shape[0]
    
    if s_q == 1:
        # Decode: no VRAM risk, maximize rocBLAS speed (effectively unchunked)
        return reference_mla_sparse_prefill(q, kv, indices, sm_scale, d_v)
    else:
        # Prefill: VRAM is the primary concern for large batches.
        # We can route to Triton if specifically instructed.
        if envs.VLLM_ROCM_MLA_SPARSE_FP16_TRITON:
            return triton_mla_sparse_vec(q, kv, indices, sm_scale, d_v)
        else:
            # By default, use chunked PyTorch. 
            # Note: 512 is the sweet-spot for MI50 rocBLAS performance!
            return reference_mla_sparse_prefill(q, kv, indices, sm_scale, d_v, chunk_size=512)


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
            # Force ref Torch (instead of using mla_sparse) as triton is still slower than chunked torch (1.5 vs 8 TFLOPS) and not steady enough (HSA_STATUS_ERROR_OUT_OF_RESOURCES when running with max-num-batched-tokens 8192)
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

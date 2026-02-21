# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, : block_size * head_dim]
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "NHD",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            block_id * kv_cache_stride
            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "NHD",
        head_dim,
        block_tile_size,
        head_tile_size,
    )


@triton.jit
def _deepgemm_fp16_paged_mqa_logits_stage1(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,           # FP16
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,          # FP16 [NumBlocks, BlockSize, Heads, Dim]
    stride_k_blk,       # Stride(0): Jump to next physical block (numel per block)
    stride_k_tok,       # Stride(1): Jump to next token inside block (usually Heads*Dim)
    context_len_ptr,
    block_table,        # [Batch, MaxNumBlocks] - Note: This is now a BLOCK table, not token indices
    weights,            # [Batch*NextN, Heads] FP32
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_num_blocks,     # Changed from max_blk_len (tokens) to max_num_blocks
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr, # Must match BlockSize (e.g., 64)
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    # Grid logic
    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)
    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
        
    # Cap the length to the assigned split
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    # 1. Load Q (dtype)
    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )

    # 2. Load Weights (FP32)
    # Based on reference: weight_slice = weights[i * next_n:(i + 1) * next_n, ...]
    # These act as gating/scaling factors after ReLU.
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    # Loop over KV blocks
    # CRITICAL CHANGE: We iterate aligned to ChunkK (64)
    # We assume ChunkK == BlockSize for this implementation
    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        # 3. Paged Attention Indexing
        # a. Calculate which logical block we are in
        current_logical_block = context_idx // ChunkK
        
        # b. Load the Physical Block ID from the Block Table
        #    block_table shape: [Batch, MaxNumBlocks]
        physical_block_id = tl.load(
            block_table + pid_batch * max_num_blocks + current_logical_block
        )

        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length

        # 4. Load K (FP16)
        #    Pointer = Base of Physical Block + Offset for Token inside Block
        #    KV_buffer shape is [NumPhysicalBlocks, BlockSize, 1, Dim]
        k = tl.load(
            KV_buffer
            + physical_block_id * stride_k_blk      # Jump to the correct physical block
            + tl.arange(0, ChunkK)[:, None] * stride_k_tok # Iterate tokens 0..63 inside that block
            + tl.arange(0, HiddenDim)[None, :],     # Iterate dim
            mask=mask_kv[:, None],
            other=0.0,
        )

        # 5. Dot Product (FORCE FP32 ACCUMULATION)
        o = tl.dot(q.to(tl.float16) if q.dtype == tl.float32 else q, k.T, out_dtype=tl.float32)
        
        # 6. Activation (ReLU)
        # Matches reference: s = torch.relu(s)
        o = tl.maximum(o, 0.0)

        # 7. Apply Weights (Gating)
        # Matches reference: s = s * weight_slice
        # scale_weight shape is [ChunkQ], o is [ChunkQ, ChunkK]
        # We broadcast scale_weight to apply per-head scaling
        o = o * scale_weight[None, :].T

        # 8. Masking (Causal/Padding)
        mask = (
            context_idx + tl.arange(0, ChunkK) <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        # Store Output
        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
            + (context_idx + tl.arange(0, ChunkK)[None, None, :]),
            o[:, None, :],
        )
        

# First implementation attempt, not fully optimized but faster than torch reference
def deepgemm_fp16_paged_mqa_logits_stage1(
    q: torch.Tensor,           # [Batch, NextN, Heads, Dim] (--dtype)
    kv_cache: torch.Tensor,    # [NumBlocks, BlockSize, 1, Dim] (FP16)
    weights: torch.Tensor,     # [Batch * NextN, Heads] (FP32)
    out_qk: torch.Tensor,      # Output Logits (FP32)
    context_lens: torch.Tensor,
    block_tables: torch.Tensor, # [Batch, MaxNumBlocks] (Not per-token as blockSize !=1)
    max_model_len: int,
    # MI50 TUNED DEFAULTS:
    ChunkQ: int = 16,           # Smaller tiles = More thread blocks = Higher Occupancy on MI50's 60 CUs.
    ChunkK: int = 64,           # Must match BlockSize, 64 has slightly better performance than 32 (with ChunkQ=32).
    TotalCuCount: int = 60,     # MI50 has 60 CU
    WavePerEU: int = 1,         # 1 for gfx906
    num_warps: int = 4,   # Sweet spot for register usage on gfx906.
    num_stages: int = 1,  # Essential for stability (avoids LDS crashes)
):
    
    # Check Block Size matches ChunkK
    block_size = kv_cache.size(1) 
    assert block_size == ChunkK, f"Kernel requires BlockSize ({block_size}) == ChunkK ({ChunkK})"
    
    batch_size, next_n, heads, hidden_dim = q.size()
    _, max_num_blocks = block_tables.size()

    # Calculate strides for the KV Cache
    # KV Cache: [NumBlocks, BlockSize, 1, Dim]
    stride_k_blk = kv_cache.stride(0)  # Steps to jump 1 physical block
    stride_k_tok = kv_cache.stride(1)  # Steps to jump 1 token inside a block

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    config = {
        "ChunkQ": ChunkQ,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }
    
    grid = (batch_size * next_n * (heads // config["ChunkQ"] * SplitKV),)
    
    _deepgemm_fp16_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache,
        stride_k_blk,  # Stride per block
        stride_k_tok,  # Stride per token
        context_lens,
        block_tables,  # Block Table
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_num_blocks, # Max Blocks, not tokens
        num_warps=num_warps,
        num_stages=num_stages,
        **config,
    )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp16_paged_mqa_logits_torch(
    q: torch.Tensor, # --dtype
    kv_cache: torch.Tensor, # fp16
    weights: torch.Tensor, # fp32
    context_lens: torch.Tensor, # int32
    block_tables: torch.Tensor, # int32
    max_model_len: int,
):

    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    
    is_context_lens_2d = False # assumed to never be 2d
   
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.full((next_n, ), context_len, device='cuda', dtype=torch.int32) if is_context_lens_2d \
                    else torch.arange(context_len - next_n, context_len, device='cuda')
        
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        
        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]                 # [num_blocks, block_size, kv_heads, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)    # [kv_heads, dim, total_tokens]
        qx = q[i].transpose(0, 1)                       # q[i]: [next_n, heads, dim] -> [heads, next_n, dim]
        s = torch.matmul(qx.to(torch.float16) if qx.dtype == torch.float32 else qx, kx).float()       # [heads, next_n, dim] @ [1, dim, total_tokens] -> [heads, next_n, total_tokens] in fp32 here

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float('-inf'))     # mask shape: [1, next_n, total_tokens]
        s = torch.relu(s) * weight_slice[..., None]             # weight_slice: [heads, next_n] -> [heads, next_n, 1]
        s = s.sum(dim=0)                                        # [next_n, total_tokens]
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))

    return logits


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def rocm_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller, or --dtype fp16/fp32.
        kv_cache: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
            Or FP16 : [num_blocks, block_size, 1, D].
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    batch_size, next_n, heads, _ = q.shape

    if envs.VLLM_ROCM_MLA_SPARSE_FP16_TRITON:
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp16_paged_mqa_logits_stage1(
            q,
            kv_cache,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
        )
        return out_qk.sum(dim=0)
    elif envs.VLLM_ROCM_MLA_SPARSE_FP16:
        return fp16_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables, max_model_len)

    from vllm._aiter_ops import rocm_aiter_ops

    @functools.lru_cache
    def paged_mqa_logits_module():
        paged_mqa_logits_module_path = None
        if importlib.util.find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
            paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
        elif (
            importlib.util.find_spec("aiter.ops.triton.attention.pa_mqa_logits")
            is not None
        ):
            paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

        if paged_mqa_logits_module_path is not None:
            try:
                module = importlib.import_module(paged_mqa_logits_module_path)
                return module
            except ImportError:
                return None
        return None

    aiter_paged_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()
    # FIXME(ganyi): Temporarily disable the aiter path until nightly docker
    # update aiter to the fix PR.
    aiter_paged_mqa_logits_module = None

    if aiter_paged_mqa_logits_module is not None:
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits_stage1(
            q,
            kv_cache,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_torch(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )

# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp16_mqa_logits_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. --dtype
        kv: `kv` has shape [N, D] with dtype `torch.float16` 
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    if q.dtype != torch.float16:
        q = q.half()
    
    seq_len_kv = kv.shape[0]
    num_q, num_heads, _ = q.shape


    # 1. Pre-allocate output
    final_logits = torch.full(
        (num_q, seq_len_kv), 
        float("-inf"), 
        device=q.device, 
        dtype=torch.float32
    )

    # 2. Prepare Mask (Broadcasting later)
    mask_lo = (torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None])
    mask_hi = (torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None])
    mask = mask_lo & mask_hi

    # Accumulator
    weighted_sum = torch.zeros((num_q, seq_len_kv), device=q.device, dtype=torch.float32)

    # Permute for easier slicing: [H, M, D]
    q_per_head = q.permute(1, 0, 2) 
    weights_per_head = weights.t() # [H, M]
    k_t = kv.t() # [D, N]

    # 3. Chunked Loop
    # 4 (VLLM_FP16_MQA_TORCH_HEAD_CHUNK_SIZE) * 512 tokens (batch size) * 2 (peak factor) * 32k (max context) * 4 bytes (fp32) ~= 0.5 GB memory peak / GPU.
    # num_heads = 128 for Deepseek V3.2 , so 32 (=128/4) loop iterations if VLLM_FP16_MQA_TORCH_HEAD_CHUNK_SIZE = 4

    for i in range(0, num_heads, envs.VLLM_FP16_MQA_TORCH_HEAD_CHUNK_SIZE):
        end = min(i + envs.VLLM_FP16_MQA_TORCH_HEAD_CHUNK_SIZE, num_heads)
        
        # Slice the chunk: Shape [Chunk_Size, M, D]
        q_chunk = q_per_head[i:end] 
        w_chunk = weights_per_head[i:end].unsqueeze(-1) # [Chunk, M, 1]

        # Matmul: [Chunk, M, D] @ [D, N] -> [Chunk, M, N]
        # This is the heavy lifting.
        score_chunk = torch.matmul(q_chunk, k_t).float()
        score_chunk = torch.relu(score_chunk)
        
        # Weighted sum: Sum over the chunk dimension (dim 0)
        # [Chunk, M, N] * [Chunk, M, 1] -> [Chunk, M, N] -> Sum -> [M, N]
        chunk_sum = (score_chunk * w_chunk).sum(dim=0)
        weighted_sum.add_(chunk_sum)
        
        # Free tensor immediately to reduce memory pressure before next chunk
        del score_chunk
        del chunk_sum

    # 4. Final Masking
    final_logits = torch.where(mask, weighted_sum, final_logits)
    return final_logits


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def rocm_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller, or --dtype fp16/fp32.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`. Or FP16 ([N, D])
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    if envs.VLLM_ROCM_MLA_SPARSE_FP16:
        # only torch kernel available here for kv_fp16 (with q_fp16 or q_fp32) as it is much faster than triton equivalent for gfx906
        return fp16_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    @functools.lru_cache
    def mqa_logits_module():
        mqa_logits_module_path = None
        if importlib.util.find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
            mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
        elif (
            importlib.util.find_spec("aiter.ops.triton.attention.fp8_mqa_logits")
            is not None
        ):
            mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

        if mqa_logits_module_path is not None:
            try:
                module = importlib.import_module(mqa_logits_module_path)
                return module
            except ImportError:
                return None
        return None

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        kv, scale = kv
        return fp8_mqa_logits(q, kv, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run - workspace reservation is done by the caller
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace during profiling run so lock doesn't fail later
        if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
            current_workspace_manager().get_simultaneous(
                ((total_seq_lens, head_dim), fp8_dtype),
                ((total_seq_lens, 4), torch.uint8),
            )
        else:
            current_workspace_manager().get_simultaneous(
                ((total_seq_lens, head_dim), torch.float16),
            )
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )
    else:
        ops.indexer_k_cache_fp16(
            k,
            kv_cache,
            slot_mapping,
        )

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        # Pre-allocate workspace buffers once, reuse across all chunks
        workspace_manager = current_workspace_manager()
        if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
            k_fp_full, k_scale_full = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), fp8_dtype),
                ((total_seq_lens, 4), torch.uint8),
            )
        else:
            (k_fp_full,) = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), torch.float16),
            )

        for chunk in prefill_metadata.chunks:
            if not envs.VLLM_ROCM_MLA_SPARSE_FP16:
                k_fp = k_fp_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                k_fp = k_fp_full[: chunk.total_seq_lens]
                ops.cp_gather_indexer_k_cache_fp16(
                    kv_cache,
                    k_fp,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            logits = rocm_mqa_logits(
                q[chunk.token_start : chunk.token_end],
                k_fp if envs.VLLM_ROCM_MLA_SPARSE_FP16 else (k_fp, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_decode_tokens = pack_seq_triton(
                q[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_decode_tokens = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_decode_tokens.shape[0]
        next_n = padded_q_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_paged_mqa_logits(
            padded_q_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        
        if decode_metadata.use_large_context_topk:
            if next_n == 1:
                lengths = decode_metadata.seq_lens
            else:
                # (bs,) -> (bs, 1) + (next_n,) -> (bs, next_n) -> (bs * next_n,)
                lengths = (
                    decode_metadata.seq_lens.unsqueeze(1)
                    - next_n
                    + 1
                    + decode_metadata.offsets
                ).flatten() 

            torch.ops._C.large_context_topk(
                logits,
                topk_indices,
                lengths,
                None,
            )
        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer

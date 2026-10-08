# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch SDPA attention kernels."""

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F

import vllm.envs as envs

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except (ImportError, AttributeError):
    SDPBackend = None
    sdpa_kernel = None

try:
    from torch.nn.attention.bias import causal_lower_right
except (ImportError, AttributeError):
    causal_lower_right = None


# constants
_LOGGED_TORCH_SDPA_PREFILL = False
_LOGGED_TORCH_SDPA_DECODE = False
_LOGGED_TORCH_SDPA_MTP_DECODE = False


@contextlib.contextmanager
def _torch_sdpa_backend() -> Iterator[None]:
    if envs.VLLM_TORCH_SDPA_BACKEND == "auto" or sdpa_kernel is None:
        yield
        return
    if envs.VLLM_TORCH_SDPA_BACKEND == "math" and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
        return
    yield


def _causal_lower_right_mask(
    query_len: int,
    kv_len: int,
    device: torch.device,
    query_start_pos: int = 0,
    full_query_len: int | None = None,
) -> torch.Tensor:
    if full_query_len is None:
        full_query_len = query_len
    context_len = kv_len - full_query_len
    query_pos = query_start_pos + torch.arange(query_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return kv_pos <= context_len + query_pos


def _log_torch_sdpa_prefill_once() -> None:
    global _LOGGED_TORCH_SDPA_PREFILL
    if _LOGGED_TORCH_SDPA_PREFILL:
        return
    _LOGGED_TORCH_SDPA_PREFILL = True
    logger.info(
        "Using experimental Torch SDPA prefill path "
        "(backend=%s, min_tokens=%d, max_tokens=%d, q_chunk_size=%d)",
        envs.VLLM_TORCH_SDPA_BACKEND,
        envs.VLLM_TORCH_SDPA_PREFILL_MIN_TOKENS,
        envs.VLLM_TORCH_SDPA_PREFILL_MAX_TOKENS,
        envs.VLLM_TORCH_SDPA_PREFILL_Q_CHUNK_SIZE,
    )


def _log_torch_sdpa_decode_once() -> None:
    global _LOGGED_TORCH_SDPA_DECODE
    if _LOGGED_TORCH_SDPA_DECODE:
        return
    _LOGGED_TORCH_SDPA_DECODE = True
    logger.info(
        "Using experimental Torch SDPA decode path "
        "(backend=%s, max_seqs=%d)",
        envs.VLLM_TORCH_SDPA_BACKEND,
        envs.VLLM_TORCH_SDPA_DECODE_MAX_SEQS,
    )


def _log_torch_sdpa_mtp_decode_once() -> None:
    global _LOGGED_TORCH_SDPA_MTP_DECODE
    if _LOGGED_TORCH_SDPA_MTP_DECODE:
        return
    _LOGGED_TORCH_SDPA_MTP_DECODE = True
    logger.info(
        "Using experimental Torch SDPA MTP decode verifier path "
        "(backend=%s)",
        envs.VLLM_TORCH_SDPA_BACKEND,
    )


def can_use_torch_sdpa_prefill(
    num_actual_tokens: int,
    max_query_len: int,
    query_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor | None,
    attn_type: AttentionType,
    kv_quant_mode: KVQuantMode,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    alibi_slopes: torch.Tensor | None,
    use_alibi_sqrt: bool,
    sinks: torch.Tensor | None,
    logits_soft_cap: float,
    sliding_window: tuple[int, int],
    chunk_lookback: int,
    output_scale: torch.Tensor | None,
    mm_prefix_range_tensor: torch.Tensor | None,
) -> bool:
    if not envs.VLLM_TORCH_SDPA_PREFILL:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        logger.debug("Torch SDPA prefill skip: CUDA graph capture is active")
        return False
    if attn_type != AttentionType.DECODER:
        logger.debug("Torch SDPA prefill skip: attn_type %s != DECODER", attn_type)
        return False
    if max_query_len <= 1:
        # Decode step, normal behavior to skip
        return False
    if output_scale is not None:
        logger.debug("Torch SDPA prefill skip: output_scale is not None")
        return False
    if kv_quant_mode != KVQuantMode.NONE:
        logger.debug("Torch SDPA prefill skip: kv_quant_mode %s != NONE", kv_quant_mode)
        return False
    if q_dtype not in (torch.float16, torch.bfloat16):
        logger.debug("Torch SDPA prefill skip: q_dtype %s not in (FP16, BF16)", q_dtype)
        return False
    if k_dtype != q_dtype or v_dtype != q_dtype:
        logger.debug(
            "Torch SDPA prefill skip: dtype mismatch Q:%s K:%s V:%s",
            q_dtype,
            k_dtype,
            v_dtype,
        )
        return False
    if alibi_slopes is not None or use_alibi_sqrt:
        logger.debug("Torch SDPA prefill skip: ALiBi not supported")
        return False
    if sinks is not None:
        logger.debug("Torch SDPA prefill skip: Sinks not supported")
        return False
    if logits_soft_cap != 0:
        logger.debug(
            "Torch SDPA prefill skip: logits_soft_cap %s != 0", logits_soft_cap
        )
        return False
    if sliding_window != (-1, -1):
        logger.debug(
            "Torch SDPA prefill skip: sliding_window %s != (-1, -1)",
            sliding_window,
        )
        return False
    if chunk_lookback != -1:
        logger.debug("Torch SDPA prefill skip: chunk_lookback %s != -1", chunk_lookback)
        return False
    if mm_prefix_range_tensor is not None:
        logger.debug("Torch SDPA prefill skip: mm_prefix_range_tensor is not None")
        return False

    if (query_lens_cpu <= 1).any().item():
        logger.debug(
            "Torch SDPA prefill skip: mixed batch with decode tokens in "
            "query_lens_cpu"
        )
        return False

    total_prefill_tokens = int(query_lens_cpu.sum().item())
    if total_prefill_tokens < envs.VLLM_TORCH_SDPA_PREFILL_MIN_TOKENS:
        logger.debug(
            "Torch SDPA prefill skip: total_tokens %d < MIN_TOKENS %d",
            total_prefill_tokens,
            envs.VLLM_TORCH_SDPA_PREFILL_MIN_TOKENS,
        )
        return False
    if (
        envs.VLLM_TORCH_SDPA_PREFILL_MAX_TOKENS > 0
        and total_prefill_tokens > envs.VLLM_TORCH_SDPA_PREFILL_MAX_TOKENS
    ):
        logger.debug(
            "Torch SDPA prefill skip: total_tokens %d > MAX_TOKENS %d",
            total_prefill_tokens,
            envs.VLLM_TORCH_SDPA_PREFILL_MAX_TOKENS,
        )
        return False

    if seq_lens_cpu is None:
        logger.debug("Torch SDPA prefill skip: seq_lens_cpu is None")
        return False
    if (seq_lens_cpu < query_lens_cpu).any().item():
        logger.debug("Torch SDPA prefill skip: seq_lens_cpu < query_lens_cpu")
        return False
    return True


def can_use_torch_sdpa_mtp_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_type: AttentionType,
    max_query_len: int,
    max_decode_query_len: int,
    kv_quant_mode: KVQuantMode,
    alibi_slopes: torch.Tensor | None,
    use_alibi_sqrt: bool,
    sinks: torch.Tensor | None,
    logits_soft_cap: float,
    sliding_window: tuple[int, int],
    chunk_lookback: int,
    seq_lens_cpu: torch.Tensor | None,
    query_start_loc_cpu: torch.Tensor,
    is_prefilling: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    mm_prefix_range_tensor: torch.Tensor | None,
) -> bool:
    if not envs.VLLM_TORCH_SDPA_MTP_DECODE:
        return False

    # This path uses CPU lengths to choose gather sizes and SDPA mask shapes.
    # Capturing it would bake dummy capture-time lengths into the graph.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False

    if attn_type != AttentionType.DECODER:
        return False

    if max_query_len <= 1:
        return False

    if max_query_len > max_decode_query_len:
        logger.debug(
            "Torch SDPA MTP decode skip: max_query_len %d > "
            "max_decode_query_len %d",
            max_query_len,
            max_decode_query_len,
        )
        return False

    if is_prefilling is not None:
        num_reqs = query_start_loc_cpu.shape[0] - 1
        if is_prefilling[:num_reqs].any().item():
            logger.debug("Torch SDPA MTP decode skip: prefill batch")
            return False

    if output_scale is not None:
        return False

    if kv_quant_mode != KVQuantMode.NONE:
        return False

    if q.dtype not in (torch.float16, torch.bfloat16):
        return False

    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        return False

    if alibi_slopes is not None or use_alibi_sqrt:
        return False

    if sinks is not None:
        return False

    if logits_soft_cap != 0:
        return False

    if sliding_window != (-1, -1):
        return False

    if chunk_lookback != -1:
        return False

    if mm_prefix_range_tensor is not None:
        return False

    if seq_lens_cpu is None:
        return False

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    if query_lens_cpu.shape[0] > seq_lens_cpu.shape[0]:
        return False

    active = query_lens_cpu > 0
    if not active.any().item():
        return False

    active_query_lens = query_lens_cpu[active]
    if (active_query_lens <= 1).all().item():
        return False

    if (active_query_lens > max_query_len).any().item():
        return False

    active_seq_lens = seq_lens_cpu[:query_lens_cpu.shape[0]][active]
    if (active_seq_lens < active_query_lens).any().item():
        return False

    return True


def torch_sdpa_prefill_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_kv_heads: int,
    num_queries_per_kv: int,
    scale: float,
    output: torch.Tensor,
    log_prefill: bool = True,
) -> torch.Tensor:
    if log_prefill:
        _log_torch_sdpa_prefill_once()

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

    if key_cache.ndim == 5:
        block_size = key_cache.shape[3]
    else:
        block_size = key_cache.shape[1]
    head_size = q.shape[-1]

    with _torch_sdpa_backend():
        for seq_idx in range(query_lens_cpu.shape[0]):
            query_start = int(query_start_loc_cpu[seq_idx].item())
            query_end = int(query_start_loc_cpu[seq_idx + 1].item())
            query_len = int(query_lens_cpu[seq_idx].item())
            kv_len = int(seq_lens_cpu[seq_idx].item())

            if query_len == 0:
                continue

            num_kv_blocks = (kv_len + block_size - 1) // block_size

            block_ids = block_table[seq_idx, :num_kv_blocks].to(torch.long)

            if key_cache.ndim == 5:
                # ROCM_ATTN layout:
                # key_cache: (num_blocks, num_kv_heads, head_size // x, block_size, x)
                # value_cache: (num_blocks, num_kv_heads, head_size, block_size)
                k = key_cache.index_select(0, block_ids).permute(0, 3, 1, 2, 4).reshape(
                    -1, num_kv_heads, head_size)[:kv_len]
                v = value_cache.index_select(0, block_ids).permute(0, 3, 1, 2).reshape(
                    -1, num_kv_heads, head_size)[:kv_len]
            else:
                # Triton/CUDA layout:
                # key_cache/value_cache:
                # (num_blocks, block_size, num_kv_heads, head_size)
                k = key_cache.index_select(0, block_ids).reshape(
                    -1, num_kv_heads, head_size)[:kv_len]
                v = value_cache.index_select(0, block_ids).reshape(
                    -1, num_kv_heads, head_size)[:kv_len]

            if num_queries_per_kv != 1:
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)

            k_seq = k.transpose(0, 1).unsqueeze(0)
            v_seq = v.transpose(0, 1).unsqueeze(0)

            chunk_size = envs.VLLM_TORCH_SDPA_PREFILL_Q_CHUNK_SIZE
            if chunk_size <= 0 or chunk_size >= query_len:
                q_seq = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
                attn_mask = None
                is_causal = True
                if kv_len != query_len:
                    is_causal = False
                    if causal_lower_right is not None:
                        attn_mask = causal_lower_right(query_len, kv_len)
                    else:
                        attn_mask = _causal_lower_right_mask(
                            query_len, kv_len, q.device)

                out = F.scaled_dot_product_attention(
                    q_seq,
                    k_seq,
                    v_seq,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=scale,
                )
                output[query_start:query_end].copy_(
                    out.squeeze(0).transpose(0, 1))
                continue

            for local_start in range(0, query_len, chunk_size):
                local_end = min(local_start + chunk_size, query_len)
                q_start = query_start + local_start
                q_end = query_start + local_end
                q_seq = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
                attn_mask = _causal_lower_right_mask(
                    local_end - local_start,
                    kv_len,
                    q.device,
                    query_start_pos=local_start,
                    full_query_len=query_len,
                )
                out = F.scaled_dot_product_attention(
                    q_seq,
                    k_seq,
                    v_seq,
                    attn_mask=attn_mask,
                    is_causal=False,
                    scale=scale,
                )
                output[q_start:q_end].copy_(out.squeeze(0).transpose(0, 1))
    return output


def torch_sdpa_mtp_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_kv_heads: int,
    num_queries_per_kv: int,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    _log_torch_sdpa_mtp_decode_once()
    return torch_sdpa_prefill_attention(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        scale=scale,
        output=output,
        log_prefill=False,
    )


def can_use_torch_sdpa_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_type: AttentionType,
    max_query_len: int,
    _kv_quant_mode: KVQuantMode,
    alibi_slopes: torch.Tensor | None,
    use_alibi_sqrt: bool,
    sinks: torch.Tensor | None,
    logits_soft_cap: float,
    sliding_window: tuple[int, int],
    chunk_lookback: int,
    seq_lens_cpu: torch.Tensor | None,
    query_start_loc_cpu: torch.Tensor,
    output_scale: torch.Tensor | None,
    mm_prefix_range_tensor: torch.Tensor | None,
) -> bool:
    if not envs.VLLM_TORCH_SDPA_DECODE:
        return False

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False

    if attn_type != AttentionType.DECODER:
        return False

    if max_query_len != 1:
        return False

    if output_scale is not None:
        return False

    if _kv_quant_mode != KVQuantMode.NONE:
        return False

    if q.dtype not in (torch.float16, torch.bfloat16):
        return False

    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        return False

    if alibi_slopes is not None or use_alibi_sqrt:
        return False

    if sinks is not None:
        return False

    if logits_soft_cap != 0:
        return False

    if sliding_window != (-1, -1):
        return False

    if chunk_lookback != -1:
        return False

    if mm_prefix_range_tensor is not None:
        return False

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

    if (query_lens_cpu != 1).any().item():
        return False

    if seq_lens_cpu is None:
        return False

    max_seqs = envs.VLLM_TORCH_SDPA_DECODE_MAX_SEQS
    if query_lens_cpu.shape[0] > max_seqs:
        return False

    return True


def torch_sdpa_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_kv_heads: int,
    num_queries_per_kv: int,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    """
    Experimental Torch SDPA decode path.

    Intended only for max_query_len == 1 decode batches.
    This gathers paged KV into temporary contiguous K/V tensors, so it may be
    slower than the Triton paged-attention kernel for normal decode.
    """
    _log_torch_sdpa_decode_once()

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

    if key_cache.ndim == 5:
        block_size = key_cache.shape[3]
    else:
        block_size = key_cache.shape[1]

    head_size = q.shape[-1]

    with _torch_sdpa_backend():
        for seq_idx in range(query_lens_cpu.shape[0]):
            query_start = int(query_start_loc_cpu[seq_idx].item())
            query_end = int(query_start_loc_cpu[seq_idx + 1].item())
            query_len = int(query_lens_cpu[seq_idx].item())
            kv_len = int(seq_lens_cpu[seq_idx].item())

            # This function is decode-only.
            if query_len != 1:
                raise RuntimeError(
                    f"torch_sdpa_decode_attention expected query_len == 1, "
                    f"got query_len={query_len}"
                )

            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_ids = block_table[seq_idx, :num_kv_blocks].to(torch.long)

            if key_cache.ndim == 5:
                # ROCM_ATTN layout:
                # key_cache:   (num_blocks, num_kv_heads, head_size // x, block_size, x)
                # value_cache: (num_blocks, num_kv_heads, head_size, block_size)
                k = (
                    key_cache.index_select(0, block_ids)
                    .permute(0, 3, 1, 2, 4)
                    .reshape(-1, num_kv_heads, head_size)[:kv_len]
                )
                v = (
                    value_cache.index_select(0, block_ids)
                    .permute(0, 3, 1, 2)
                    .reshape(-1, num_kv_heads, head_size)[:kv_len]
                )
            else:
                # Triton/CUDA layout:
                # key_cache/value_cache:
                # (num_blocks, block_size, num_kv_heads, head_size)
                k = (
                    key_cache.index_select(0, block_ids)
                    .reshape(-1, num_kv_heads, head_size)[:kv_len]
                )
                v = (
                    value_cache.index_select(0, block_ids)
                    .reshape(-1, num_kv_heads, head_size)[:kv_len]
                )

            if num_queries_per_kv != 1:
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)

            q_seq = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
            k_seq = k.transpose(0, 1).unsqueeze(0)
            v_seq = v.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                attn_mask=None,
                is_causal=False,
                scale=scale,
            )

            output[query_start:query_end].copy_(out.squeeze(0).transpose(0, 1))

    return output

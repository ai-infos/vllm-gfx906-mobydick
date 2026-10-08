# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Custom-FA path for the Qwen3.5-family ViT (VIT-1).

The Qwen3.5/3.8 vision tower runs **bidirectional, cache-free, ragged**
(`cu_seqlens`) fp16 attention with `head_dim = 72` (16 heads, hidden 1152),
prefill-only, on the critical path of every image-bearing prompt. Upstream
serves it through flash-attn; on gfx906 that is the Triton-AMD path, which costs
a per-boot JIT compile + graph-capture stall and is not MI50-tuned.

Why the dense entry fits: `gfx906_fa.forward` is the non-paged launcher (plain
contiguous `K`/`V`, no block table) and the kernel masks *either* via a
materialised mask *or* the inline-causal `q_abs_offset` — with **neither** it
computes **full bidirectional attention**, which is exactly the ViT case.

**Input layout (production).** The Qwen VL towers keep the whole batch of images
in one packed token stream: `Qwen3VLModel.forward` does
`hidden_states.unsqueeze(1)` before the blocks, so the attention sees
`[seq_len, 1, hidden]` — i.e. `B = 1`, `S = sum(real lengths)`, with
`cu_seqlens = [0, l_0, l_0+l_1, …]` and `cu_seqlens[-1] == S` (this is also why
flash-attn's varlen wrapper asserts `cu_seqlens_q[-1] == total_seqlen_q`: the
tensor is packed, never padded). The kernel wants one batch row per *sequence*
(`kv_max` is per row), so this adapter groups the packed stream into runs of
equal sequence length, hands each run to the kernel as a separate batch row via
zero-copy views, and scatters the results back into the packed output.

A `cu_seqlens` of `B+1` entries over a `[B, S, …]` tensor (one sequence per batch
item, optionally padded) is also handled — per-item calls when lengths differ,
one batched call when they do not.

Why head_dim is padded: the launcher serves `head_dim in {64, 96, 128, 256}` and
requires `head_size % 32 == 0`. The pad map is `{64, 96, 128, 256}` — so 72 is
zero-padded to **96** (`GFX906_FA_PAD96=0` restores 128). The padding is exact:

* padded `Q` dims contribute 0 to the QK dot,
* padded `K` dims quantise to zero q8_0 blocks (0 contribution),
* padded `V` dims contribute 0 to P·V,
* the padding is in the **head** dim, so the softmax denominator is unchanged.

Padding cost is real (the QK/PV work grows with the padded head dim); the 96-wide
instantiation makes the 96..127 all-zero q8_0 block disappear (2026-09-16).
"""

from __future__ import annotations

import os

import torch

from vllm import _gfx906_fa_C as gfx906_fa


_INSTANTIATED_HEAD_DIMS = (64, 96, 128, 256)
_FALLBACK_HEAD_DIMS = (64, 128, 256)


def _pad_head_dim(head_size: int) -> int | None:
    """Smallest servable kernel head dim that fits (None if none does).

    The pad map is (64, 96, 128, 256) since the FA-D96 gate (2026-09-16, default on): the
    ViT's 72 pads onto 96 instead of 128. ``GFX906_FA_PAD96=0`` restores the
    (64, 128, 256) map. Mirrors `gfx906_fa_backend._pad_head_dim` (the text path).
    """
    dims = _INSTANTIATED_HEAD_DIMS if _pad96_enabled() else _FALLBACK_HEAD_DIMS
    for hd in dims:
        if head_size <= hd:
            return hd
    return None


def _pad96_enabled() -> bool:
    """96-wide pad map (see the backend's twin for the gate evidence). Default on."""
    return os.environ.get("GFX906_FA_PAD96", "1") == "1"


def vit_enabled() -> bool:
    """Kill switch: `GFX906_FA_VIT=0` restores the upstream flash-attn ViT path."""
    return os.environ.get("GFX906_FA_VIT", "1") == "1"


def vit_auto_enabled() -> bool:
    """Whether the ViT path is selected *automatically* on gfx906.

    Default ON since the VIT-1 serving gate (2026-09-15): image-prompt TTFT
    -11.5 % at 1024x1024 (5.81 -> 5.14 s, 3 fresh images/arm) and -55 s of
    fresh-boot Triton JIT, with the image-conditioned distribution perturbed only
    in the tail (top-1 preserved, max |dlogprob| 0.66 at rank 4+; mean logprob
    +0.0126/token). `GFX906_FA_VIT=0` restores the upstream flash-attn ViT path
    outright; `GFX906_FA_VIT_AUTO=0` only opts out of the automatic selection
    (an explicit `--mm-encoder-attn-backend custom` still works). Unsupported
    shapes/dtypes fall through to the upstream backends in
    `ROCmPlatform.get_vit_attn_backend`.
    """
    return vit_enabled() and os.environ.get("GFX906_FA_VIT_AUTO", "1") == "1"


def vit_unsupported_reason(head_size: int, dtype: torch.dtype) -> str | None:
    """Why the custom ViT path cannot serve this shape/dtype (None = it can).

    Separate from `vit_supported` so the caller can *say* why it fell back: a
    silent fall-through loses the MI50-tuned kernel, the -11.5 % image-prompt
    TTFT win and the fresh-boot saving, and depending on what is installed can
    land on unfused SDPA instead of flash-attn (VIT-1, DEVLOG-vit1.md).
    """
    if not vit_enabled():
        return "custom ViT path disabled by GFX906_FA_VIT=0"
    if dtype not in (torch.float16, torch.float32):
        return (
            f"dtype {dtype} is neither fp16 nor fp32 — the Q8 FA kernel takes fp16 "
            "Q/K/V, and casting to reach it would be lossy for bf16"
        )
    if _pad_head_dim(head_size) is None:
        return (
            f"head_size {head_size} exceeds every instantiated kernel head dim "
            f"{_INSTANTIATED_HEAD_DIMS} (the launcher dispatches those only)"
        )
    return None


def vit_supported(head_size: int, dtype: torch.dtype) -> bool:
    return vit_unsupported_reason(head_size, dtype) is None


def _seq_plan(
    b: int, s: int, cu: torch.Tensor | None
) -> list[tuple[int, int, int]]:
    """Map the input to `(batch_row, start, length)` per attention sequence.

    Packed input (`b == 1`, `cu[-1] == s`): sequences are contiguous slices of
    row 0. Per-item input (`cu.numel() == b + 1`): each sequence starts at the
    beginning of its own batch row, and `length <= s` (padded rows are simply not
    read, which is what `kv_max` is for).
    """
    if cu is None:
        return [(i, 0, s) for i in range(b)]

    cu = cu.to(dtype=torch.int32)
    lens = (cu[1:] - cu[:-1]).tolist()
    total = b * s

    if b == 1 and sum(lens) == total:
        plan, off = [], 0
        for ln in lens:
            plan.append((0, off, ln))
            off += ln
        return plan

    if len(lens) == b:
        return [(i, 0, ln) for i, ln in enumerate(lens)]

    raise ValueError(
        f"ViT FA cannot map cu_seqlens ({cu.numel()} entries, "
        f"lens={lens[:8]}{'…' if len(lens) > 8 else ''}) onto a [{b}, {s}] "
        f"tensor: packed input needs a single batch row with cu[-1] == {total}, "
        f"per-item input needs {b + 1} entries."
    )


def forward_vit(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: torch.Tensor | None,
    scale: float,
    head_size: int,
) -> torch.Tensor:
    """Bidirectional ragged attention through the gfx906 dense FA entry.

    Args:
        query/key/value: `[B, S, H|Hkv, D]` fp16 (upstream ViT layout). For the
            Qwen VL towers `B == 1` and the token stream is packed.
        cu_seqlens: int32 `[num_seqs+1]` real lengths, or None for one sequence
            per batch item covering the whole `S`.
        max_seqlen: int32 `[1]` upstream FA convention (informational here).
        scale: the model's own softmax scale, computed for the real head size.
        head_size: real (unpadded) head dim.

    Returns:
        `[B, S, H, D]` in `query`'s dtype.
    """
    assert query.dtype == torch.float16, "ViT FA expects fp16 Q/K/V"
    b, sq, heads, d_in = query.shape
    sk = key.shape[1]
    hkv = key.shape[2]
    assert d_in == head_size, (d_in, head_size)
    assert value.shape == (b, sk, hkv, head_size)
    assert sq == sk, f"ViT FA is bidirectional; Sq {sq} != Skv {sk}"

    pad = _pad_head_dim(head_size)
    assert pad is not None, f"head_size {head_size} is not pad-able"

    plan = _seq_plan(b, sq, cu_seqlens)

    def to_kernel(x: torch.Tensor) -> torch.Tensor:
        # [B, S, H, D] -> [B, H, S, D], zero-padded to the kernel head dim.
        x = x.transpose(1, 2)
        if pad == head_size:
            return x.contiguous()
        out = x.new_zeros((x.shape[0], x.shape[1], x.shape[2], pad))
        out[..., :head_size] = x
        return out

    out = torch.empty(
        (b, sq, heads, head_size), dtype=query.dtype, device=query.device
    )

    # Group consecutive sequences of equal length so an equal-length batch (the
    # common case: one image, or N images of the same size) is a single kernel
    # call, and the view stays contiguous.
    groups: list[list[tuple[int, int, int]]] = []
    for row, start, ln in plan:
        prev = groups[-1] if groups else None
        if (
            prev
            and prev[-1][2] == ln
            and prev[-1][0] == row
            and prev[-1][1] + ln == start
        ):
            prev.append((row, start, ln))
        else:
            groups.append([(row, start, ln)])

    for grp in groups:
        ln = grp[0][2]
        if not ln:
            continue
        # Whole-row fast path only when the sequence really is the whole row
        # (a single image / one sequence per batch item covering all of S). A
        # packed sequence that merely *starts* at 0 is shorter than the stream,
        # and passing the whole row would compute attention for every other
        # image's queries too (right answer for the rows we keep, since later
        # groups overwrite them, but ~2x the work).
        whole_row = len(grp) == 1 and grp[0][1] == 0 and ln == sq
        if whole_row:
            qs = query[grp[0][0] : grp[0][0] + 1]
            ks = key[grp[0][0] : grp[0][0] + 1]
            vs = value[grp[0][0] : grp[0][0] + 1]
            kv_max = torch.full(
                (1,), ln, dtype=torch.int32, device=query.device
            )
        else:
            # Contiguous run of equally long sequences: a zero-copy view.
            qs = query[grp[0][0], grp[0][1] : grp[0][1] + ln * len(grp)].view(
                len(grp), ln, heads, head_size
            )
            ks = key[grp[0][0], grp[0][1] : grp[0][1] + ln * len(grp)].view(
                len(grp), ln, hkv, head_size
            )
            vs = value[grp[0][0], grp[0][1] : grp[0][1] + ln * len(grp)].view(
                len(grp), ln, hkv, head_size
            )
            kv_max = torch.full(
                (len(grp),), ln, dtype=torch.int32, device=query.device
            )

        # mask=None + q_abs_offset=None => full bidirectional attention; kv_max
        # bounds the scan to the sequence. NOTE: `forward` takes Q as
        # [B, H, S, D] but returns the BSHD-native output [B, S, H, D].
        # kv_split=1: the shape-aware default (32 for any Sq >= 4) is a DECODE
        # rule. With Sq in the hundreds-to-thousands the per-split partial
        # buffer [B, Sq, Hq, y, D] fp32 grows with Sq and the split-combine
        # dominates: measured 34-42 ms at Sq=1536/1728 with y=32, against 7 ms
        # at Sq=2304 where the 512 MiB budget happens to force y=1. Bidirectional
        # prefill-shaped attention has no KV-split parallelism to win: the query
        # axis already supplies the grid.
        got = gfx906_fa.forward(
            to_kernel(qs).float().contiguous(),
            gfx906_fa.quantize_q8_0(to_kernel(ks).contiguous()),
            to_kernel(vs).contiguous(),
            float(scale),
            kv_max,
            kv_split=1,
        )[..., :head_size].to(query.dtype)

        if whole_row:
            out[grp[0][0] : grp[0][0] + 1] = got
        else:
            out[grp[0][0], grp[0][1] : grp[0][1] + ln * len(grp)] = got.reshape(
                -1, heads, head_size
            )

    return out

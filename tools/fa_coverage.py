# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Enumerate which attention backend gfx906 selects, and why not CUSTOM when it is not.

Drives the selectors the engine uses, so silent fallbacks off the custom gfx906 FA are
predicted without loading a model:

* text attention via ``RocmPlatform.get_valid_backends`` (the matrix covers head
  size, block size, sliding window, non-causality, sinks, attention type; every local
  checkpoint's real configs are reported per tensor-parallel width);
* the vision tower via ``vit_unsupported_reason`` (that path is separate, and only
  the ViT one pads head dims, so the supported sets differ).

A fallback is a performance cliff: the non-CUSTOM backend may be unable to use CUDA
graphs and is often several times slower per step. See DEVLOG-fa-coverage.md.

Usage: ``.venv/bin/python tools/fa_coverage.py [--models DIR ...] [--tp 1,2,4,8]``
"""

import argparse
import glob
import json
import os

import torch

from vllm.gfx906_fa.gfx906_fa_mm_encoder import vit_unsupported_reason
from vllm.platforms.interface import DeviceCapability
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import AttentionSelectorConfig

GFX906 = DeviceCapability(major=9, minor=0)
CUSTOM = "CUSTOM"
HEAD_SIZES = (64, 96, 128, 256, 288, 512)
BLOCK_SIZES = (16, 32, 64)
VISION_HEAD_SIZES = (32, 64, 72, 80, 96, 112, 128, 160, 256, 288)
VISION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_MODEL_ROOTS = (
    "/local/models",
    "/data/models",
    "/local/cache/huggingface/hub",
    "/data/cache/huggingface/hub",
)


def select(cfg: AttentionSelectorConfig, num_heads: int | None = None):
    """Return (chosen backend name or None, reasons recorded against CUSTOM)."""
    valid, invalid = RocmPlatform.get_valid_backends(
        device_capability=GFX906,
        attn_selector_config=cfg,
        num_heads=num_heads,
    )
    chosen = min(valid, key=lambda probe: probe[1])[0].name if valid else None
    reasons = {
        backend.name: reason
        for backend, reason in invalid.items()
        if backend.name == CUSTOM
    }
    return chosen, reasons


def config(**kwargs) -> AttentionSelectorConfig:
    base: dict = {
        "head_size": 128,
        "dtype": torch.float16,
        "kv_cache_dtype": "auto",
        "block_size": 16,
        "attn_type": AttentionType.DECODER,
    }
    base.update(kwargs)
    return AttentionSelectorConfig(**base)


def text_matrix() -> None:
    print("## synthetic text matrix (only configs where CUSTOM is not chosen)")
    rows: list[str] = []
    total = 0
    for head_size in HEAD_SIZES:
        for block_size in BLOCK_SIZES:
            for has_sliding in (False, True):
                for use_non_causal in (False, True):
                    for has_sink in (False, True):
                        for attn_type in (AttentionType.DECODER, "encoder"):
                            total += 1
                            cfg = config(
                                head_size=head_size,
                                block_size=block_size,
                                has_sliding_window=has_sliding,
                                use_non_causal=use_non_causal,
                                has_sink=has_sink,
                                attn_type=attn_type,
                            )
                            chosen, reasons = select(cfg, num_heads=32)
                            if chosen == CUSTOM:
                                continue
                            reason = reasons.get(CUSTOM, ["?"])[0]
                            rows.append(
                                f"  head={head_size:3d} block={block_size:3d} "
                                f"sliding={int(has_sliding)} "
                                f"non_causal={int(use_non_causal)} "
                                f"sink={int(has_sink)} attn={attn_type:10s} "
                                f"-> {chosen} ({reason})"
                            )
    print("\n".join(rows))
    print(f"  {len(rows)}/{total} synthetic text configs do not get CUSTOM")
    print(
        "  NOTE: above 256 (288/512 here) cannot be padded to an instantiated kernel\n"
        "        dim, so those need a new kernel instance rather than a gate change;\n"
        "        Gemma-4's heterogeneous 256/512 layers are in this class."
    )


def vision_matrix() -> None:
    print("\n## vision tower (mm-encoder) matrix — why the custom ViT path is not used")
    misses = 0
    total = 0
    for head_size in VISION_HEAD_SIZES:
        for dtype in VISION_DTYPES:
            total += 1
            reason = vit_unsupported_reason(head_size, dtype)
            if reason is None:
                continue
            misses += 1
            name = str(dtype).replace("torch.", "")
            print(f"  head_dim={head_size:3d} dtype={name:8s} -> no CUSTOM: {reason}")
    print(f"  {misses}/{total} vision (head_dim, dtype) combinations fall back")
    print(
        "  NOTE: get_vit_attn_backend(head_size, dtype) takes no sliding/causality\n"
        "        argument, so a windowed ViT (e.g. Qwen2.5-VL) selects CUSTOM with no\n"
        "        way to express the window; verify the ViT op honours it, or that is\n"
        "        is a correctness gap rather than a performance one."
    )


def _dig(node, *keys, default=None):
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def text_configs(path: str) -> list[tuple[int, bool, int, int]]:
    """``(head_dim, has_sliding, num_heads, num_kv_heads)`` for a text checkpoint."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return []
    cfg = raw.get("text_config") or raw.get("llm_config") or raw
    heads = _dig(cfg, "num_attention_heads")
    kv_heads = _dig(cfg, "num_key_value_heads", default=heads)
    head_dim = _dig(cfg, "head_dim")
    if head_dim is None:
        hidden = _dig(cfg, "hidden_size")
        head_dim = hidden // heads if (hidden and heads) else None
    if not head_dim or not heads:
        return []
    layer_types = [str(t) for t in (_dig(cfg, "layer_types") or [])]
    kinds = {("sliding" in t) for t in layer_types} or set()
    if not kinds:
        kinds = {bool(_dig(cfg, "sliding_window"))}
    # Hybrid models (e.g. iRoPE) select per layer group: evaluate each kind.
    return [
        (int(head_dim), sliding, int(heads), int(kv_heads or heads))
        for sliding in kinds
    ]


def vision_configs(path: str) -> list[tuple[int, int, bool]]:
    """``(head_dim, num_heads, windowed)`` for a vision tower, if present."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return []
    found = []
    for key in ("vision_config", "vision_tower_config", "visual"):
        vit = raw.get(key)
        if not isinstance(vit, dict):
            continue
        heads = _dig(vit, "num_attention_heads", "num_heads")
        head_dim = _dig(vit, "head_dim")
        if head_dim is None:
            hidden = _dig(vit, "hidden_size", "embed_dim")
            head_dim = hidden // heads if (hidden and heads) else None
        windowed = bool(
            _dig(vit, "window_size")
            or _dig(vit, "sliding_window")
            or any("window" in str(t) for t in (_dig(vit, "layer_types") or []))
        )
        if head_dim and heads:
            found.append((int(head_dim), int(heads), windowed))
    return found


def scan_models(roots: list[str], tps: list[int]) -> None:
    print("\n## local checkpoints (only configs where CUSTOM is not chosen)")
    seen: set = set()
    reads = 0
    misses = 0
    for root in roots:
        if os.path.basename(root).startswith("hub"):
            repo_dirs = glob.glob(os.path.join(root, "models--*"))
            stubs = [
                d for d in repo_dirs if not glob.glob(os.path.join(d, "snapshots", "*"))
            ]
            if stubs:
                print(
                    f"  NOTE: {len(stubs)} repo(s) under {root} are refs-only stubs "
                    "(no snapshots/, so nothing to enumerate) - e.g. "
                    f"{os.path.basename(stubs[0])}"
                )
            patterns = [
                os.path.join(root, "models--*", "snapshots", "*", "config.json")
            ]
        else:
            patterns = [
                os.path.join(root, pattern, "config.json")
                for pattern in ("*", "*/*", "*/*/*")
            ]
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                name = os.path.basename(os.path.dirname(path))[:44]
                for head_size, sliding, heads, kv_heads in text_configs(path):
                    reads += 1
                    for tp in tps:
                        if heads % tp:
                            continue
                        cfg = config(head_size=head_size, has_sliding_window=sliding)
                        chosen, reasons = select(cfg, num_heads=heads // tp)
                        if chosen == CUSTOM:
                            continue
                        key = (path, head_size, tp, chosen)
                        if key in seen:
                            continue
                        seen.add(key)
                        misses += 1
                        reason = reasons.get(CUSTOM, ["?"])[0]
                        print(
                            f"  {name:44s} TEXT   tp={tp} heads={heads} "
                            f"kv={kv_heads} head_dim={head_size:3d} "
                            f"sliding={int(sliding)} -> {chosen} ({reason})"
                        )
                for head_size, heads, windowed in vision_configs(path):
                    reason = vit_unsupported_reason(head_size, torch.float16)
                    if reason is None:
                        continue
                    key = (path, head_size, "vision")
                    if key in seen:
                        continue
                    seen.add(key)
                    misses += 1
                    print(
                        f"  {name:44s} VISION head_dim={head_size:3d} heads={heads} "
                        f"windowed={int(windowed)} -> no CUSTOM: {reason}"
                    )
    print(f"  {misses} non-CUSTOM configs among {reads} checkpoint config reads")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_ROOTS))
    parser.add_argument("--tp", default="1,2,4,8")
    parser.add_argument("--skip-matrix", action="store_true")
    args = parser.parse_args()
    if not args.skip_matrix:
        text_matrix()
        vision_matrix()
    scan_models(args.models, [int(t) for t in args.tp.split(",")])


if __name__ == "__main__":
    main()

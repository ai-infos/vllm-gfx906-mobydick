#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""FN-3 — build a tiny Qwen3.8-Flash-Next (`qwen4_exp`) config for gfx906 dev.

The real checkpoint is ~120 B params of MoE (W4A16 ≈ 60 GB, plus a PLE ngram
table the MI210 recipe offloads 60 GB of), so it cannot be loaded on 2× MI50 —
which leaves the whole QSA + PLE + hyperconnection + MoE path unexercised on
this box. This writes a `config.json` that keeps the **architecture identical**
(all four layer types, PLE with a real ngram table, hyperconnection, the QSA
indexer and sparse attention, MTP) and shrinks only the dimensions, so the model
builds and runs on one MI50.

Weights come from vLLM's `--load-format dummy` (random init), so this is an
**execution/A-B harness, not a quality gate** — see
`DEVLOG-qwen38-flash-qsa.md` for what it can and cannot decide.

Only the tokenizer comes from the real checkpoint:

    hf download Qwen/Qwen3.8-Flash-Next \\
        tokenizer_config.json vocab.json merges.txt chat_template.jinja \\
        generation_config.json --local-dir <dir>
    python docs/gfx906/_qsa_tiny_model.py <dir>
    .venv/bin/vllm serve <dir> --load-format dummy ...

Usage: `_qsa_tiny_model.py <dir>` writes `<dir>/config.json` and prints a
per-field diff against the real model (kept in REAL for that purpose).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The real Qwen/Qwen3.8-Flash-Next text_config, for the printed comparison and
# for every field this harness deliberately does *not* shrink.
REAL = {
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "ple_embed_dim": 2560,
    "ngram_vocab_size_base": 20_000_000,
    "indexer_n_heads": 4,
    "indexer_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262_144,
}

# Shrunk for one MI50. Architecture-bearing fields are kept: hc_count, the
# PLE/ngram layout, the QSA indexer (budget/compress_ratio are constrained by
# Qwen4ExpTextConfig._validate_qsa_config: budget/ratio must be 512 or 2048),
# the 4-layer linear/full attention mix, and the MTP layer count.
TEXT_CONFIG = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 256,
    "head_dim": 64,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    # vocab_size must stay at the tokenizer's size (248 320): the tokenizer
    # emits real ids and the embedding would index out of range otherwise.
    "vocab_size": 248_320,
    "max_position_embeddings": 4096,
    "dtype": "bfloat16",  # faithful: gfx906 then auto-selects fp16
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": True,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    },
    "bos_token_id": 248_044,
    "eos_token_id": 248_044,
    # GDN / linear attention. value heads must be a multiple of key heads.
    "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    # MoE (every layer; decoder_sparse_step=1 and num_experts > 0).
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "decoder_sparse_step": 1,
    "norm_topk_prob": True,
    "output_router_logits": False,
    "router_aux_loss_coef": 0.001,
    "mlp_only_layers": [],
    # Hyperconnection: hc_count > 1 is required by the config; 4 is the real
    # model's value and costs almost nothing at hidden_size 256.
    "hc_count": 4,
    "hc_lowrank": 32,
    # PLE + ngram table. ple_embed_dim must divide by (ngram_size-1)*heads_per_ngram
    # = 16, so head_dim is 256/16 = 16; ngram_vocab_size_base sets the table
    # (≈ ngram_heads × base rows) — 4096 keeps it at a few MB.
    "ple_layer_ids": [2],
    "ple_embed_dim": 256,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 4096,
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": 128,
    "output_gate_type": "sigmoid",
    # QSA: indexer_kv_heads must be 1; budget/compress_ratio ∈ {512, 2048};
    # indexer_head_dim must cover head_dim * partial_rotary_factor = 16.
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 64,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    # MTP: one layer, as in the real model. The nested `mtp` dict in the real
    # config is not read by this vLLM revision (only mtp_num_hidden_layers is).
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
}

CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp",
    "text_config": TEXT_CONFIG,
    "tie_word_embeddings": False,
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    print(f"wrote {target / 'config.json'}")
    print(f"{'field':<34} {'real':>12} {'tiny':>12}")
    for field, tiny in TEXT_CONFIG.items():
        if field in REAL:
            print(f"{field:<34} {REAL[field]!s:>12} {tiny!s:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

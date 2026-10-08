# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Qwen4Exp PLE n-gram lookup split.

The PLE lookup gathers over a *host-resident* n-gram table: it copies the ids
device->host with a blocking copy so the host can read the mmap'd shards, then
copies the result back through pinned buffers. HIP refuses that inside a stream
capture, so the op has to stay outside the captured graph pieces. vLLM does that
by appending the op to ``splitting_ops`` for Qwen4Exp models -- and the guard
compares the family, not one spelling of it, because a checkpoint's ``model_type``
varies (``qwen4_exp``, ``qwen4_exp_text``, ...). The guard used to compare
``model_type == "qwen4_exp"`` exactly, silently missed
``logic65/Whittle-Qwen-3.8-35B-A3B`` (``qwen4_exp_text``), and engine init died
with ``hipErrorStreamCaptureUnsupported``.

The same guard must not *over*-fire: the op only exists when the text config
declares ``ple_layer_ids`` (the gate `amd/model_state.py` / its nvidia twin
uses), so a PLE-less member of the family -- including the MTP drafter, whose
config is rewritten to ``qwen4_exp_mtp`` -- must keep the capture mode it asked
for rather than being downgraded to PIECEWISE for an op that is never emitted.

These tests build synthetic config dirs, so they run anywhere (no checkpoint
download); the last one additionally checks the real Whittle checkpoint when its
config is cached on this machine.
"""

import json
from pathlib import Path

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.platforms import current_platform

PLE_NGRAM_OP = "vllm::qwen4_exp_amd_ple_ngram_embedding"

# The split exists because the *AMD* path gathers the host table inside one op
# with a blocking device->host copy, which HIP refuses inside a capture. The CUDA
# path prefetches through streams over a device-side table, so the guard must not
# fire there -- those assertions only mean something on ROCm.
_AMD_PATH_REQUIRED = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the host-resident PLE lookup only needs the eager region on ROCm",
)

_WHITTLE = Path(
    "/biglocal/cache/hf/hub/models--logic65--Whittle-Qwen-3.8-35B-A3B"
    "/snapshots/926d1dc370db65c9b905b5578989600954bbc2b7"
)

# The two spellings a checkpoint of this family can present as. `qwen4_exp_mtp`
# is a SpeculativeConfig rewrite of the same checkpoint (draft model), and it
# never carries PLE layers.
_FAMILY_SPELLINGS = ["qwen4_exp", "qwen4_exp_text"]

# Everything the family's config classes need to instantiate; the values only
# have to be self-consistent (nothing here builds a model).
_BASE_CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}

_LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "vocab_size": 1024,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
}


def _write_config(tmp_path: Path, name: str, config: dict) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def _vllm_config_for(model_dir: Path) -> VllmConfig:
    model_config = ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tokenizer_mode="auto",
        trust_remote_code=False,
        skip_tokenizer_init=True,
    )
    return VllmConfig(model_config=model_config)


@_AMD_PATH_REQUIRED
@pytest.mark.parametrize("model_type", _FAMILY_SPELLINGS)
@pytest.mark.parametrize("ple_layer_ids", [[2], [1, 2]], ids=["one", "two"])
def test_ple_lookup_is_split_out_of_the_captured_graph(
    tmp_path, model_type, ple_layer_ids
):
    """Every family spelling that uses the n-gram memory must split the op out."""
    model_dir = _write_config(
        tmp_path,
        f"{model_type}-{len(ple_layer_ids)}",
        {
            **_BASE_CONFIG,
            "model_type": model_type,
            "ple_layer_ids": ple_layer_ids,
        },
    )
    vllm_config = _vllm_config_for(model_dir)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert splitting_ops is not None
    assert PLE_NGRAM_OP in splitting_ops, (
        f"{PLE_NGRAM_OP} must be split out for model_type={model_type!r}; "
        "otherwise the blocking device->host copy inside the PLE lookup runs "
        "inside a cudagraph capture and engine init fails"
    )
    # FULL capture would swallow the split op again, so the guard downgrades.
    assert not vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()


@pytest.mark.parametrize("model_type", _FAMILY_SPELLINGS)
def test_allow_full_cudagraph_hatch_keeps_the_requested_mode(
    tmp_path, model_type, monkeypatch
):
    """The escape hatch must keep FULL, not merely stay quiet about it.

    QSA-FN-12 / tracker #7: the downgrade is a judgement about this HIP, so the
    judgement has to be *measurable*. With the hatch set the split stays and the
    full-graph mode survives -- the arm then fails at capture with
    ``hipErrorStreamCaptureUnsupported``, which is the point: a loud failure with
    the real cause beats silently running a mode nobody asked for.
    """
    model_dir = _write_config(
        tmp_path,
        f"{model_type}-hatch",
        {**_BASE_CONFIG, "model_type": model_type, "ple_layer_ids": [2]},
    )
    monkeypatch.setenv("VLLM_GFX906_QWEN4_EXP_ALLOW_FULL_CUDAGRAPH", "1")
    vllm_config = _vllm_config_for(model_dir)

    assert PLE_NGRAM_OP in vllm_config.compilation_config.splitting_ops
    assert vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(), (
        "with the hatch set the guard must stand aside so full-graph capture can "
        "be attempted (and fail visibly) on this HIP"
    )


@pytest.mark.parametrize("value", ["0", ""], ids=["zero", "empty"])
def test_allow_full_cudagraph_hatch_is_exactly_one(tmp_path, monkeypatch, value):
    """Only the exact string ``1`` opens the hatch -- like every other GFX906 switch."""
    model_dir = _write_config(
        tmp_path,
        f"hatch-{value or 'empty'}",
        {**_BASE_CONFIG, "model_type": "qwen4_exp_text", "ple_layer_ids": [2]},
    )
    monkeypatch.setenv("VLLM_GFX906_QWEN4_EXP_ALLOW_FULL_CUDAGRAPH", value)
    vllm_config = _vllm_config_for(model_dir)

    assert not vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()


@pytest.mark.parametrize("model_type", _FAMILY_SPELLINGS)
@pytest.mark.parametrize(
    "ple", [[], None], ids=["empty", "absent"]
)  # `None` = key missing from config.json entirely
def test_family_without_ple_layers_keeps_its_capture_mode(tmp_path, model_type, ple):
    """A PLE-less family checkpoint must not be downgraded for a no-op split."""
    config = {**_BASE_CONFIG, "model_type": model_type}
    if ple is not None:
        config["ple_layer_ids"] = ple
    model_dir = _write_config(tmp_path, f"{model_type}-nople", config)
    vllm_config = _vllm_config_for(model_dir)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert splitting_ops is not None
    assert PLE_NGRAM_OP not in splitting_ops
    assert vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(), (
        "a Qwen4Exp checkpoint without ple_layer_ids emits no PLE lookup, so "
        "downgrading its full-graph capture would be pure cost"
    )


def test_non_family_model_is_untouched(tmp_path):
    """The widened family match must not reach other architectures."""
    model_dir = _write_config(tmp_path, "llama", _LLAMA_CONFIG)
    vllm_config = _vllm_config_for(model_dir)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert splitting_ops is not None
    assert PLE_NGRAM_OP not in splitting_ops
    assert vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()


@_AMD_PATH_REQUIRED
def test_repeated_configs_do_not_duplicate_the_split(tmp_path):
    """The append must not accumulate across VllmConfig builds (shared list)."""
    model_dir = _write_config(
        tmp_path,
        "dupes",
        {**_BASE_CONFIG, "model_type": "qwen4_exp_text", "ple_layer_ids": [2]},
    )
    seen = []
    for _ in range(3):
        vllm_config = _vllm_config_for(model_dir)
        seen.append(list(vllm_config.compilation_config.splitting_ops))

    assert all(ops.count(PLE_NGRAM_OP) == 1 for ops in seen), seen


@_AMD_PATH_REQUIRED
@pytest.mark.skipif(
    not (_WHITTLE / "config.json").is_file(),
    reason="the Whittle checkpoint's config.json is not cached on this machine",
)
def test_real_checkpoint_with_host_resident_ple_is_split(tmp_path):
    """The checkpoint that motivated this: `qwen4_exp_text` + `ple_layer_ids`."""
    real = json.loads((_WHITTLE / "config.json").read_text())
    assert real["model_type"] == "qwen4_exp_text"
    assert real["ple_layer_ids"], "this checkpoint is the host-resident-PLE case"

    model_dir = _write_config(tmp_path, "whittle", real)
    vllm_config = _vllm_config_for(model_dir)

    assert PLE_NGRAM_OP in (vllm_config.compilation_config.splitting_ops or [])
    assert not vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()

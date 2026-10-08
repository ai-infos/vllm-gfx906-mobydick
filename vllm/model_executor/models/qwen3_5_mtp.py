# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3_5 MTP model."""

import hashlib
import json
import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)

# --- draft-vocab manifest (written by tools/build_draft_vocab.py) ------------
# The work dir may carry cat1_manifest.json: which corpus the list came from,
# and hashes binding the ids file to the sliced head rows (the row order *is*
# the id mapping, so an equal-count swap of the two files would otherwise be
# silent). Absent manifest = older work dir, no checks, unchanged behaviour.
_MANIFEST_NAME = "cat1_manifest.json"


def _ids_sha1(ids) -> str:
    return hashlib.sha1(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


def _file_sha1(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_draft_vocab_manifest(model_dir: str, ids: torch.Tensor) -> dict | None:
    """Verify the manifest against the loaded ids (and the head file, unless
    MTP_DRAFT_VOCAB_STRICT=0). Returns the manifest so the caller can log its
    provenance, or None when there is none."""
    path = os.path.join(model_dir, _MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        manifest = json.load(fh)
    want = manifest.get("ids_sha1")
    got = _ids_sha1(ids.tolist())
    if want and got != want:
        raise ValueError(
            f"{path}: mtp_draft_vocab_ids.pt does not match the manifest "
            f"(ids sha1 {got} != {want}) - the ids file and the sliced head are "
            f"not a matched pair. Re-run build_draft_vocab.py slice."
        )
    head_file = manifest.get("head_file")
    if (
        head_file
        and manifest.get("head_sha1")
        and os.environ.get("MTP_DRAFT_VOCAB_STRICT", "1") != "0"
    ):
        got_h = _file_sha1(os.path.join(model_dir, head_file))
        if got_h != manifest["head_sha1"]:
            raise ValueError(
                f"{path}: {head_file} does not match the manifest "
                f"(sha1 {got_h} != {manifest['head_sha1']}) - the sliced "
                f"draft head is not the one this ids list was built with."
            )
    return manifest


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # CAT-1 (ported from syv-ai/qwen38-27b-rtx3090, patches/
        # qwen3_5-mtp-draft-vocab.patch, adapted for our bf16 unquantized
        # lm_head): vocab-truncated draft head. If the model directory ships
        # mtp_draft_vocab_ids.pt (built by tools/build_draft_vocab.py) the
        # drafter scores only those rows (mtp.draft_lm_head.*) instead of the
        # full 248k-row lm_head; logits for all other ids are -inf.
        # Speculative decoding stays exact, only the acceptance rate can
        # change. File absence = baseline; MTP_DRAFT_VOCAB=0 forces baseline.
        self.draft_lm_head = None
        self.draft_vocab_ids = None
        _ids_path = os.path.join(model_config.model, "mtp_draft_vocab_ids.pt")
        if (
            os.path.exists(_ids_path)
            and os.environ.get("MTP_DRAFT_VOCAB", "1") != "0"
        ):
            _ids = torch.load(_ids_path, map_location="cpu")
            self.draft_vocab_ids = _ids
            # Manifest (optional): provenance + ids<->head pairing check.
            _manifest = _check_draft_vocab_manifest(model_config.model, _ids)
            if _manifest is not None:
                _prov = _manifest.get("provenance") or {}
                _srcs = (
                    ", ".join(
                        os.path.basename(f.get("path", "?"))
                        for f in _prov.get("corpus_files", [])[:3]
                    )
                    or "?"
                )
                logger.info(
                    "MTP draft-vocab manifest: %d ids, sha1 %s, built %s, "
                    "corpus %s (%.1fM tokens, holdout %s, control tokens %s)",
                    _manifest.get("n_ids", int(_ids.numel())),
                    str(_manifest.get("ids_sha1"))[:12],
                    _prov.get("created", _manifest.get("created", "?")),
                    _srcs,
                    _prov.get("corpus_tokens", 0) / 1e6,
                    _prov.get("holdout", "?"),
                    _prov.get("control_tokens", "?"),
                )
            # Our lm_head is unquantized bf16 (AWQ ignore list), so the sliced
            # rows are plain bf16 too — quant_config=None (upstream passed
            # vllm_config.quant_config because their rows were int8-packed).
            self.draft_lm_head = ParallelLMHead(
                int(_ids.numel()),
                config.hidden_size,
                quant_config=None,
                prefix=maybe_prefix(prefix, "draft_lm_head"),
            )
            logger.info("MTP drafter uses a %d-token draft head", int(_ids.numel()))

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        fc_quant = (
            None
            if (quant_config and quant_config.get_name() == "modelopt_fp4")
            else quant_config
        )
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )

        # GPTQ: quantized checkpoints may exclude MTP from quantization via
        # quantization_config.dynamic with "-:pattern" entries. When detected,
        # disable quantization for MTP layers so they use unquantized params.
        original_quant = vllm_config.quant_config
        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):
            hf_qc = getattr(model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    vllm_config.quant_config = None
        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        vllm_config.quant_config = original_quant
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen3NextSparseMoeBlock,
            "mlp",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Branch on the inputs, not the rank: the drafter is built entirely on
        # the last PP stage, where `is_first_rank` is False.
        if intermediate_tensors is None:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        # CAT-1 (syv port): processor for the truncated draft head.
        _draft_ids = getattr(self.model, "draft_vocab_ids", None)
        self.draft_logits_processor = (
            LogitsProcessor(int(_draft_ids.numel()))
            if _draft_ids is not None
            and getattr(self.model, "draft_lm_head", None) is not None
            else None
        )
        # CAT-1 guard: use_local_argmax_reduction scores drafts via
        # get_top_tokens() -> self.lm_head and would silently bypass the
        # shortlist (degraded acceptance, no error). Fail loudly instead.
        if self.draft_logits_processor is not None:
            _spec_cfg = getattr(vllm_config, "speculative_config", None)
            if getattr(_spec_cfg, "use_local_argmax_reduction", False):
                raise ValueError(
                    "MTP draft-vocab shortlist is incompatible with "
                    "use_local_argmax_reduction: the local-argmax fast path "
                    "reads the full lm_head and would ignore the shortlist. "
                    "Disable use_local_argmax_reduction."
                )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # CAT-1 (syv port): score with the truncated draft head, then scatter
        # into a full-vocab buffer (-inf everywhere else). Rejection sampling
        # uses the target model, so this stays exact.
        if self.draft_logits_processor is not None:
            sub = self.draft_logits_processor(self.model.draft_lm_head, hidden_states)
            if sub is None:
                return None
            ids = self.model.draft_vocab_ids
            assert ids is not None  # set together with draft_lm_head in __init__
            if ids.device != sub.device:
                # One-shot lazy migration. Safe under cudagraphs: this branch
                # can only fire on the first eager call (ids start on CPU),
                # which happens during warmup before any graph capture; after
                # it, the device matches and the branch is dead code in every
                # captured replay.
                ids = ids.to(sub.device)
                self.model.draft_vocab_ids = ids
            full = sub.new_full((sub.shape[0], self.config.vocab_size), float("-inf"))
            full.index_copy_(1, ids, sub)
            # Marker for the V2 bring-up check: V2's MTP path has its own top-k
            # sharing, so a run that never logs this line is NOT using the
            # shortlist (silent loss of CAT-1's read saving). Grep the server log
            # for it when validating V2.
            logger.info_once(
                "MTP draft-vocab shortlist ACTIVE (%d ids; logits outside the "
                "list are -inf)",
                int(ids.numel()),
            )
            return full
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits."""
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            for name, weight in weights:
                # CAT-1 (syv port): skip the truncated draft head when it is
                # disabled (no mtp_draft_vocab_ids.pt / MTP_DRAFT_VOCAB=0).
                if "draft_lm_head" in name and self.model.draft_lm_head is None:
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()

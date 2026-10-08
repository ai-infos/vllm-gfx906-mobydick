# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Print the compressed-tensors scheme vLLM resolves for a checkpoint's config groups.

CPU-only (no GPU, no weights): builds the checkpoint's ``CompressedTensorsConfig`` and
asks it for the scheme of representative layer names. Useful when a quantised checkpoint
fails to load and the question is whether the routing (no scheme, or the wrong one) or
the loader is at fault.

Usage: ``.venv/bin/python tools/ct_scheme_probe.py <model_dir>``
"""

import json
import sys

from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)

LAYER_NAMES = (
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.embed_tokens",
    "lm_head",
)


def main(model_dir: str) -> None:
    with open(f"{model_dir}/config.json", encoding="utf-8") as f:
        qc = json.load(f)["quantization_config"]
    cfg = ct.CompressedTensorsConfig.from_config(
        {
            "quantization_config": qc,
            "config_groups": qc["config_groups"],
            "format": qc["format"],
            "ignore": qc.get("ignore", []),
            "quant_method": qc["quant_method"],
            "version": qc.get("version"),
        }
    )
    for group, spec in qc["config_groups"].items():
        weight_quant = QuantizationArgs(**spec["weights"])
        print(
            f"{group}: targets={spec['targets']} strategy={weight_quant.strategy} "
            f"bits={weight_quant.num_bits} group_size={weight_quant.group_size}"
        )
        for name in LAYER_NAMES:
            try:
                scheme = cfg._get_scheme_from_parts(  # noqa: SLF001 - diagnostic tool
                    weight_quant=weight_quant,
                    input_quant=None,
                    format=qc["format"],
                    layer_name=name,
                )
                extra = " ".join(
                    f"{a}={getattr(scheme, a)}"
                    for a in ("strategy", "num_bits", "group_size", "symmetric")
                    if hasattr(scheme, a)
                )
                print(f"  {name} -> {type(scheme).__name__} {extra}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {name} -> {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) Nick — nick413@gmail.com
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Vendored from https://github.com/cassettesgoboom/gfx906-fa-vllm
# (FlashAttention-style custom attention backend for vLLM on AMD gfx906).
#
"""Custom FlashAttention Q8 attention backend for AMD gfx906 (MI50/MI60).

This package wraps the shipped ``vllm._gfx906_fa_C`` CUDA/HIP extension and
registers :class:`Gfx906FABackend` as vLLM's ``CUSTOM`` attention backend.

Importing this package (or loading the ``vllm.general_plugins`` entry point
``vllm.gfx906_fa.gfx906_fa_backend:register``) registers the backend so it can
be selected with ``VLLM_ATTENTION_BACKEND=CUSTOM``.
"""

try:
    from vllm import _gfx906_fa_C as ext  # noqa: E402  (compiled by the vLLM build)
except ImportError:
    # The extension is only built for gfx906 ROCm. This package is loaded
    # on every vLLM startup by the vllm.general_plugins entry point, so it
    # must import cleanly (with a no-op register) on other platforms.
    ext = None

if ext is not None:
    from vllm.gfx906_fa.gfx906_fa_backend import Gfx906FABackend, register

    __all__ = ["ext", "Gfx906FABackend", "register"]
else:
    def register() -> None:
        pass
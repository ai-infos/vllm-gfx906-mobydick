# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
"""Guards for the with_amdsmi_context cleanup contract.

On gfx906-native ROCm 7.14 builds, amdsmi_init() can return success with
0 processor handles after torch import (the state ROCmPlatform's
get_device_name GCN-arch fallback is designed for), in which case
amdsmi_shut_down() raises AMDSMI_STATUS_NOT_INIT. The wrapper must not
let the cleanup failure mask the wrapped call's outcome: a query that
returned normally must keep its result, and a query that failed must
keep its exception. This crashed a vLLM worker (Nemotron-H, TP=2 + EP,
first mamba SSD decode config lookup) on 2026-08-30.
"""

from unittest.mock import MagicMock, patch

import pytest

import vllm.platforms.rocm as rocm


def _not_init():
    raise RuntimeError("32 | AMDSMI_STATUS_NOT_INIT")


def _broken_query():
    raise ValueError("underlying query failed")


def test_wrapped_result_survives_shut_down_failure():
    init, shut_down = MagicMock(), MagicMock(side_effect=_not_init)
    with (
        patch.object(rocm, "amdsmi_init", init, create=True),
        patch.object(rocm, "amdsmi_shut_down", shut_down, create=True),
        patch.object(rocm.logger, "warning_once") as warn,
    ):
        assert rocm.with_amdsmi_context(lambda: "AMD_GFX906")() == "AMD_GFX906"
    init.assert_called_once()
    shut_down.assert_called_once()
    # The diagnostic must be visible from every rank of a TP run.
    warn.assert_called_once()
    assert warn.call_args.args[0].startswith("amdsmi_shut_down failed")
    assert warn.call_args.kwargs.get("scope") == "process"


def test_wrapped_failure_still_propagates_when_shut_down_also_fails():
    shut_down = MagicMock(side_effect=_not_init)
    with (
        patch.object(rocm, "amdsmi_init", create=True),
        patch.object(rocm, "amdsmi_shut_down", shut_down, create=True),
        patch.object(rocm.logger, "warning_once") as warn,
        pytest.raises(ValueError, match="underlying query failed"),
    ):
        rocm.with_amdsmi_context(_broken_query)()
    shut_down.assert_called_once()
    # A failed query logs at debug, not warning — the warning text claims
    # success, so it must not fire here.
    warn.assert_not_called()

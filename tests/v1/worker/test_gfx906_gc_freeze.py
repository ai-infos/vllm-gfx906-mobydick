# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the gfx906 GC freeze guard.

Covers `GFX906_GC_FREEZE` / `GFX906_GC_THAW` / `GFX906_GC_DUMP` as they are
implemented in `vllm/v1/worker/gpu/model_runner.py` -- the guard itself, not a
copy of it. The claims under test, in the order the reviewer asked about them:

* **OFF is untouched** -- with the env var unset the guard is a pure
  passthrough: `gc.collect` is still the very object the caller had put there,
  `gc.freeze` is never called, the enabled-state is not touched, the depth
  counter stays 0, and vLLM's own `_gc_maybe_collect()` still collects.
* **ON is shadowed inside** -- inside the region `gc.collect` *is*
  `_gc_collect_noop`, GC is disabled, and `_gc_maybe_collect()` collects
  nothing.
* **One real collect on outermost exit** -- with `GFX906_GC_THAW=1` exactly one
  real collection runs, and only at the outermost exit; with the default (park)
  it is `gc.freeze()` again and nothing is ever traversed.
* **Nesting** -- an inner region leaves the shadow in place and restores
  nothing; the outermost exit restores once.
* **Exception path** -- a raise inside propagates and everything is still
  restored.
* **Enabled-state restore** -- whatever `gc.isenabled()` was on entry is what
  it is on exit, including "already disabled".

The guard mutates the *process-global* `gc` module and two module globals, by
design (that is the whole point of it). So these tests instrument `gc` with
counting wrappers and let pytest undo the patching, and a fixture asserts the
depth counter is back to 0 after every case and un-parks the heap.
"""

import gc

import pytest

mr = pytest.importorskip(
    "vllm.v1.worker.gpu.model_runner",
    reason="V2 model runner is not importable in this build",
)

FREEZE = mr._GC_FREEZE_ENV
THAW = mr._GC_THAW_ENV


class _Spies:
    """Counting wrappers over the real gc entry points.

    `collect` is the interesting one: the guard captures whatever is in
    `gc.collect` when the outermost region opens and puts it back on the
    outermost exit, so a wrapper there counts *real* collections -- including
    the one the guard performs itself.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("freeze", "unfreeze", "disable", "enable", "collect"):
            real = getattr(gc, name)

            def make(name=name, real=real):
                def wrapper(*a, **k):
                    self.calls.append(name)
                    return real(*a, **k)

                return wrapper

            monkeypatch.setattr(gc, name, make())

    def count(self, name: str) -> int:
        return self.calls.count(name)

    def reset(self) -> None:
        self.calls.clear()


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch):
    """Instrument gc, and guarantee the process is left as we found it."""
    for var in (FREEZE, THAW, mr._GC_DUMP_ENV):
        monkeypatch.delenv(var, raising=False)
    gc.enable()  # before the wrappers go on, so this call is not counted
    s = _Spies()
    s.install(monkeypatch)
    yield s
    # A park-mode region leaves everything the process was holding in the
    # permanent generation; un-park so later tests in this process are not
    # measured against a heap with no young generation.
    if mr._GC_FROZEN_DEPTH != 0:  # a failed assertion must not corrupt the run
        mr._GC_FROZEN_DEPTH = 0
        mr._GC_REAL_COLLECT = None
    gc.unfreeze()
    gc.enable()
    s.reset()


# --------------------------------------------------------------------------
# 1. OFF: the guard is not in the program at all
# --------------------------------------------------------------------------


def test_off_env_unset_means_disabled():
    assert mr._gc_freeze_enabled() is False


def test_off_leaves_gc_collect_object_untouched(spies, monkeypatch):
    sentinel = gc.collect
    with mr._gc_freeze_guard("profile_run"):
        assert gc.collect is sentinel, "OFF must not shadow gc.collect"
    assert gc.collect is sentinel


def test_off_never_touches_the_collector(spies):
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert spies.calls == [], f"OFF region touched gc: {spies.calls}"


def test_off_preserves_enabled_state(spies):
    before = gc.isenabled()
    with mr._gc_freeze_guard("capture_model"):
        assert gc.isenabled() is before
    assert gc.isenabled() is before


def test_off_keeps_upstreams_own_collect(spies):
    """`_gc_maybe_collect()` is vLLM's pre-capture collection: with the guard
    OFF it must still collect exactly once."""
    spies.reset()
    mr._gc_maybe_collect()
    assert spies.count("collect") == 1


def test_off_depth_counter_stays_zero(spies):
    with mr._gc_freeze_guard("profile_run"):
        assert mr._GC_FROZEN_DEPTH == 0
    assert mr._GC_FROZEN_DEPTH == 0


def test_off_decorator_is_transparent(spies):
    @mr._gc_freeze_around
    def fn(x):
        return x * 2

    assert fn(21) == 42
    assert fn.__name__ == "fn", "@functools.wraps must survive"
    assert spies.calls == []


# --------------------------------------------------------------------------
# 2. ON: shadowed for the duration of the region
# --------------------------------------------------------------------------


def test_on_env_gate_reads_the_environment(monkeypatch):
    assert mr._gc_freeze_enabled() is False
    monkeypatch.setenv(FREEZE, "1")
    assert mr._gc_freeze_enabled() is True
    monkeypatch.setenv(FREEZE, "0")
    assert mr._gc_freeze_enabled() is False
    monkeypatch.setenv(FREEZE, "")
    assert mr._gc_freeze_enabled() is False, "empty string is not an enable"


def test_on_shadows_gc_collect_inside(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    outside = gc.collect
    with mr._gc_freeze_guard("profile_run"):
        assert gc.collect is mr._gc_collect_noop
        assert gc.collect is not outside
    assert gc.collect is outside, "restored on exit"


def test_on_noop_returns_zero_and_takes_any_args(spies):
    assert mr._gc_collect_noop() == 0
    assert mr._gc_collect_noop(2, gen=1, foo="bar") == 0


def test_on_disables_gc_inside(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    assert gc.isenabled() is True
    with mr._gc_freeze_guard("profile_run"):
        assert gc.isenabled() is False
        assert spies.count("disable") == 1


def test_on_freezes_once_on_entry(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    spies.reset()
    with mr._gc_freeze_guard("profile_run"):
        assert spies.count("freeze") == 1
        assert mr._GC_FROZEN_DEPTH == 1


def test_on_suppresses_vllms_own_collect(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    with mr._gc_freeze_guard("profile_run"):
        spies.reset()
        mr._gc_maybe_collect()
        assert spies.count("collect") == 0, "pre-capture collect must not run"


def test_on_third_party_caller_is_shadowed_too(spies, monkeypatch):
    """The reason shadowing exists: `gc.disable()` alone does not stop an
    explicit `gc.collect()` from a third-party frame."""
    monkeypatch.setenv(FREEZE, "1")

    def third_party():
        return gc.collect()  # attribute lookup at call time -> the noop

    with mr._gc_freeze_guard("capture_model"):
        spies.reset()
        assert third_party() == 0
        assert spies.count("collect") == 0


# --------------------------------------------------------------------------
# 3. The exit path: park (default) vs thaw
# --------------------------------------------------------------------------


def test_park_mode_default_collects_nothing(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    spies.reset()
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert mr._gc_thaw_enabled() is False, "park is the default"
    assert spies.count("collect") == 0, "park must never traverse"
    assert spies.count("unfreeze") == 0
    assert spies.count("freeze") == 2, "entry freeze + exit park, no walk"


def test_thaw_mode_runs_exactly_one_real_collect(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    monkeypatch.setenv(THAW, "1")
    spies.reset()
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert spies.count("unfreeze") == 1
    assert spies.count("collect") == 1, "exactly one real collection on exit"
    assert spies.count("freeze") == 1, "thaw does not re-park"


def test_thaw_collect_runs_after_the_shadow_is_restored(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    monkeypatch.setenv(THAW, "1")
    seen = {}

    def probe(*a, **k):
        seen["gc.collect is noop"] = gc.collect is mr._gc_collect_noop
        return 0

    # The guard captures this wrapper as "the real collect"; when it calls it on
    # exit, the shadow must already be gone (otherwise the exit collect is a
    # no-op and the thaw is a lie).
    monkeypatch.setattr(gc, "collect", probe)
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert seen.get("gc.collect is noop") is False


def test_real_collect_reference_is_cleared(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    with mr._gc_freeze_guard("profile_run"):
        assert mr._GC_REAL_COLLECT is not None
    assert mr._GC_REAL_COLLECT is None
    assert mr._GC_FROZEN_DEPTH == 0


# --------------------------------------------------------------------------
# 4. Nesting
# --------------------------------------------------------------------------


def test_nested_region_leaves_shadow_in_place(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    with mr._gc_freeze_guard("profile_run"):
        outer_noop = gc.collect
        with mr._gc_freeze_guard("capture_model"):
            assert gc.collect is outer_noop
            assert mr._GC_FROZEN_DEPTH == 2
        assert gc.collect is outer_noop, "inner exit must not restore early"
        assert mr._GC_FROZEN_DEPTH == 1
    assert mr._GC_FROZEN_DEPTH == 0


def test_nested_thaw_collects_once_at_the_outermost_exit(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    monkeypatch.setenv(THAW, "1")
    spies.reset()
    with (
        mr._gc_freeze_guard("profile_run"),
        mr._gc_freeze_guard("capture_model"),
        mr._gc_freeze_guard("innermost"),
    ):
        pass
    assert spies.count("collect") == 1
    assert spies.count("unfreeze") == 1


def test_captured_reference_is_only_taken_at_depth_zero(spies, monkeypatch):
    """If the inner region re-captured, the shadow would restore to itself and
    `gc.collect` would be permanently the noop."""
    monkeypatch.setenv(FREEZE, "1")
    first = gc.collect
    with mr._gc_freeze_guard("outer"):
        captured = mr._GC_REAL_COLLECT
        with mr._gc_freeze_guard("inner"):
            assert mr._GC_REAL_COLLECT is captured
    assert gc.collect is first


# --------------------------------------------------------------------------
# 5. Exception path
# --------------------------------------------------------------------------


def test_exception_propagates_and_restores_everything(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    sentinel = gc.collect
    with (
        pytest.raises(RuntimeError, match="boom"),
        mr._gc_freeze_guard("profile_run"),
    ):
        raise RuntimeError("boom")
    assert gc.collect is sentinel
    assert mr._GC_FROZEN_DEPTH == 0
    assert mr._GC_REAL_COLLECT is None
    assert gc.isenabled() is True, "must be re-enabled even after a raise"


def test_exception_from_inner_region_still_lets_outer_restore(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    sentinel = gc.collect
    with (
        pytest.raises(RuntimeError),
        mr._gc_freeze_guard("outer"),
        mr._gc_freeze_guard("inner"),
    ):
        raise RuntimeError("boom")
    assert gc.collect is sentinel
    assert mr._GC_FROZEN_DEPTH == 0


def test_decorator_restores_on_exception(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    sentinel = gc.collect

    @mr._gc_freeze_around
    def fn():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        fn()
    assert gc.collect is sentinel


# --------------------------------------------------------------------------
# 6. Enabled-state restore
# --------------------------------------------------------------------------


def test_previously_disabled_gc_stays_disabled(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    gc.disable()
    try:
        with mr._gc_freeze_guard("profile_run"):
            assert gc.isenabled() is False
        assert gc.isenabled() is False, "the guard must not enable on our behalf"
    finally:
        gc.enable()


def test_previously_enabled_gc_is_re_enabled(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    assert gc.isenabled() is True
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert gc.isenabled() is True


def test_thaw_with_previously_disabled_gc(spies, monkeypatch):
    monkeypatch.setenv(FREEZE, "1")
    monkeypatch.setenv(THAW, "1")
    gc.disable()
    try:
        with mr._gc_freeze_guard("profile_run"):
            pass
        assert gc.isenabled() is False
        assert spies.count("collect") == 1, "thaw collects regardless"
    finally:
        gc.enable()


# --------------------------------------------------------------------------
# 7. GFX906_GC_DUMP
# --------------------------------------------------------------------------


def test_dump_env_only_arms_faulthandler(spies, monkeypatch):
    """The dump flag must not change collection behaviour at all: tvm_ffi
    replaces the SIGSEGV handler, so re-arming faulthandler is the only effect."""
    monkeypatch.setenv(FREEZE, "1")
    monkeypatch.setenv(mr._GC_DUMP_ENV, "1")
    armed = []
    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: armed.append((a, k)))
    spies.reset()
    with mr._gc_freeze_guard("profile_run"):
        pass
    assert armed, "faulthandler.enable() must have been called"
    assert spies.count("collect") == 0

"""Tests for the Tinker deadline utilities: expiry, overrides, classification."""

from __future__ import annotations

import threading
import time
from typing import NoReturn

import pytest
from llm_waterfall import is_capacity_error

from wmh.distill.deadlines import (
    DEFAULT_DEADLINES_S,
    DeadlineFuture,
    TinkerCallKind,
    TinkerDeadlineError,
    call_with_deadline,
    deadline_for,
    env_var_for,
    wait_with_deadline,
)

_SHORT_DEADLINE = "0.05"
_GENEROUS_WAIT_S = 5.0
"""An expiry must land well before this (the deadline is 0.05s)."""


class _BlockingFuture:
    """A fake SDK future that never completes; result(timeout) honors the timeout.

    Mirrors the real `APIFuture.result(timeout)` contract: block on an event
    that is never set, then raise the builtin `TimeoutError`.
    """

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


class _ReadyFuture:
    """A fake SDK future whose result is immediately available."""

    def result(self, timeout: float | None = None) -> str:
        del timeout
        return "ready"


class _BrokenFuture:
    """A fake SDK future that fails with a non-timeout error."""

    def result(self, timeout: float | None = None) -> NoReturn:
        del timeout
        raise RuntimeError("terminal SDK failure")


def test_defaults_match_the_documented_values() -> None:
    assert DEFAULT_DEADLINES_S == {
        "sample": 300.0,
        "compute_logprobs": 300.0,
        "forward_backward": 900.0,
        "optim_step": 600.0,
        "save_state": 600.0,
        "load_state": 600.0,
        "save_weights_for_sampler": 600.0,
        "connect": 60.0,
    }


def test_wait_with_deadline_returns_the_result() -> None:
    future: DeadlineFuture[str] = _ReadyFuture()
    assert wait_with_deadline("sample", future) == "ready"


@pytest.mark.parametrize("kind", sorted(DEFAULT_DEADLINES_S))
def test_blocking_future_expires_within_the_shortened_deadline(
    kind: TinkerCallKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_var_for(kind), _SHORT_DEADLINE)
    started = time.monotonic()
    with pytest.raises(TinkerDeadlineError) as info:
        wait_with_deadline(kind, _BlockingFuture())
    assert time.monotonic() - started < _GENEROUS_WAIT_S
    assert info.value.kind == kind
    assert info.value.deadline_s == pytest.approx(0.05)
    message = str(info.value)
    assert "timed out" in message
    assert "wedged" in message
    assert env_var_for(kind) in message
    assert f"{info.value.elapsed_s:.1f}s" in message


def test_blocking_call_expires_within_the_shortened_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fully blocking call (no timeout parameter anywhere): the bounded wait
    # abandons the daemon thread, which stays parked on the never-set event.
    monkeypatch.setenv(env_var_for("connect"), _SHORT_DEADLINE)
    never = threading.Event()
    started = time.monotonic()
    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        call_with_deadline("connect", never.wait)
    assert time.monotonic() - started < _GENEROUS_WAIT_S


def test_call_with_deadline_returns_the_result() -> None:
    assert call_with_deadline("connect", lambda: 41 + 1) == 42


def test_call_with_deadline_reraises_the_calls_own_error() -> None:
    def explode() -> NoReturn:
        raise ValueError("bad model name")

    with pytest.raises(ValueError, match="bad model name"):
        call_with_deadline("connect", explode)


def test_wait_with_deadline_propagates_non_timeout_errors_unchanged() -> None:
    with pytest.raises(RuntimeError, match="terminal SDK failure"):
        wait_with_deadline("sample", _BrokenFuture())


def test_nested_deadline_error_is_not_rewrapped() -> None:
    inner = TinkerDeadlineError("compute_logprobs", elapsed_s=0.1, deadline_s=0.1)

    class _NestedFuture:
        def result(self, timeout: float | None = None) -> NoReturn:
            del timeout
            raise inner

    with pytest.raises(TinkerDeadlineError) as info:
        wait_with_deadline("sample", _NestedFuture())
    assert info.value is inner


def test_env_override_parses_as_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env_var_for("sample"), "0.25")
    assert deadline_for("sample") == 0.25


def test_unset_env_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env_var_for("sample"), raising=False)
    assert deadline_for("sample") == 300.0


def test_sub_millisecond_override_clamps_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env_var_for("sample"), "0.0000001")
    assert deadline_for("sample") == 0.001


@pytest.mark.parametrize("raw", ["fast", "", "0", "-5", "inf", "nan"])
def test_invalid_override_raises_naming_the_env_var(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_var_for("optim_step"), raw)
    with pytest.raises(ValueError, match="WMH_TINKER_DEADLINE_OPTIM_STEP"):
        deadline_for("optim_step")


def test_deadline_error_is_a_capacity_error_for_the_retry_stack() -> None:
    # Load-bearing pin: the provider retry wrapper (and the waterfall) only
    # retries what llm_waterfall classifies as capacity. The error subclasses
    # TimeoutError and says "timed out" precisely so this holds.
    error = TinkerDeadlineError("sample", elapsed_s=120.4, deadline_s=120.0)
    assert isinstance(error, TimeoutError)
    assert is_capacity_error(error) is True

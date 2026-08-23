"""Bounded classifier execution tests."""

from __future__ import annotations

import time

import pytest

from exp.runtime.gateway.guardrails.bounded import ClassifierTimeoutError, run_bounded


def test_run_bounded_returns_the_inspect_result() -> None:
    """A timely inspect result is returned to the caller."""
    assert run_bounded(lambda: 7, 0.5) == 7


def test_run_bounded_raises_without_waiting_for_a_blocking_inspect() -> None:
    """A hung inspect times out on the caller before the worker returns."""

    def hang() -> int:
        """Sleep longer than the allowed timeout."""
        time.sleep(0.25)
        return 1

    started = time.monotonic()
    with pytest.raises(ClassifierTimeoutError):
        run_bounded(hang, 0.05)
    assert time.monotonic() - started < 1.0


def test_run_bounded_propagates_inspect_errors() -> None:
    """Adapter exceptions surface after the worker finishes inside the budget."""

    def boom() -> int:
        """Fail immediately."""
        raise RuntimeError("classifier unavailable")

    with pytest.raises(RuntimeError, match="classifier unavailable"):
        run_bounded(boom, 0.5)

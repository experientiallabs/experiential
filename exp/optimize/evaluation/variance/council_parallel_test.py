"""Offline concurrency, native spend context and cancellation coverage for council draws."""

from contextvars import ContextVar
from functools import partial
from pathlib import Path
from threading import Barrier

import pytest

from exp.optimize.evaluation.variance.council_parallel import council_draws
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    current_spend_ceiling,
    install_spend_ceiling,
    operation_spend_scope,
)


def test_parallel_draws_preserve_order_and_real_spend_context(tmp_path: Path) -> None:
    """Independent callbacks overlap, inherit accounting, and leave caller context unchanged."""
    barrier = Barrier(2)
    marker: ContextVar[str] = ContextVar("council-marker", default="outside")
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})

    def draw(index: int) -> tuple[int, str]:
        """Require both workers and the caller's actual native operation scope."""
        assert current_spend_ceiling() is ceiling
        barrier.wait(timeout=5)
        value = marker.get()
        marker.set(str(index))
        return index, value

    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        token = marker.set("inherited")
        result = council_draws(tuple(partial(draw, i) for i in range(2)), 2)
        assert marker.get() == "inherited"
        marker.reset(token)
    assert result == ((0, "inherited"), (1, "inherited"))
    assert ceiling.summary().unpriced_attempts == 0


def test_terminal_failure_leaves_queued_draws_undispatched() -> None:
    """The actual failure is retained and later callbacks do not turn into retries."""
    invoked: list[int] = []

    def draw(index: int) -> int:
        """Record a callback and fail the first native owner."""
        invoked.append(index)
        if index == 0:
            raise ValueError("native owner failed")
        return index

    with pytest.raises(ValueError, match="native owner failed"):
        council_draws(tuple(partial(draw, i) for i in range(8)), 1)
    assert invoked == [0]


def test_invalid_bound_fails_without_invoking_draws() -> None:
    """A nonpositive bound is rejected even with no callbacks."""
    with pytest.raises(ValueError, match="positive"):
        council_draws((), 0)

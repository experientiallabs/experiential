"""The held-out aggregate line states every denominator, and a failed panel stops buying."""

import io
import threading
from typing import cast

import pytest
import typer
from rich.console import Console

from exp.cli.evaluation import held_out_fidelity
from exp.cli.evaluation.held_out_fidelity import PANEL_CONCURRENCY, aggregate_line, execute_runs
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import PreparedFidelity
from exp.optimize.evaluation.trace_replay import ReplayAggregate
from exp.runtime.models.providers.transport import ProviderTransportError


def test_aggregate_line_reports_replies_fields_judge_and_coverage() -> None:
    """One line, with explicit counts and a dash when nothing was judged."""
    total = ReplayAggregate(
        traces=2, exact=6, steps=9, fields_matched=40, fields_total=50, judge_mean=3.75
    )
    assert aggregate_line(total, 3) == (
        "Held-out fidelity  6/9 replies exact (67%) · 80% reply fields match (40/50) · "
        "judge 3.8/5 · 2/3 traces"
    )
    empty = total.model_copy(update={"judge_mean": None, "fields_total": 0, "fields_matched": 0})
    assert "- reply fields match (0/0) · judge -" in aggregate_line(empty, 3)


def test_a_failed_trace_starts_none_of_the_queued_panel_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every call fails at once, no worker starts a trace after the first failure."""
    started: list[object] = []
    lock = threading.Lock()

    def execute(_store: object, prepared: object, **_kwargs: object) -> None:
        """Fail immediately, as a revoked key would."""
        with lock:
            started.append(prepared)
        raise ProviderTransportError("unauthorized", status_code=401)

    monkeypatch.setattr(held_out_fidelity, "execute_fidelity_evaluation", execute)
    traces = tuple(range(PANEL_CONCURRENCY * 25))
    with pytest.raises(typer.Exit):
        execute_runs(
            Console(file=io.StringIO()),
            cast("ProjectStore", None),
            cast("tuple[PreparedFidelity, ...]", traces),
            lambda _event: None,
        )
    assert len(started) <= PANEL_CONCURRENCY

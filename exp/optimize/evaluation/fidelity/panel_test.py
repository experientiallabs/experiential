"""A failed panel trace stops the panel from starting any queued trace."""

import threading
from contextvars import ContextVar
from pathlib import Path
from typing import cast

import pytest

from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import panel as replay_panel
from exp.optimize.evaluation.fidelity.panel import PANEL_CONCURRENCY, execute_panel
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    current_spend_ceiling,
    install_spend_ceiling,
    operation_spend_scope,
)


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

    monkeypatch.setattr(replay_panel, "execute_fidelity_evaluation", execute)
    # No trace was interrupted before, so nothing needs resuming in place.
    monkeypatch.setattr(replay_panel, "interrupted", lambda _store, _prepared: False)
    traces = tuple(range(PANEL_CONCURRENCY * 25))
    with pytest.raises(ProviderTransportError):
        execute_panel(
            cast("ProjectStore", None),
            cast("tuple[PreparedFidelity, ...]", traces),
        )
    assert len(started) <= PANEL_CONCURRENCY


def test_panel_workers_inherit_the_operation_ceiling_in_independent_contexts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every concurrent trace retains the shared spend ceiling without leaking local changes."""
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={})
    marker: ContextVar[str] = ContextVar("panel-operation-marker", default="outside")
    caller = threading.get_ident()
    started: list[object] = []
    lock = threading.Lock()
    concurrent = threading.Barrier(PANEL_CONCURRENCY)

    def execute(_store: object, prepared: object, **_kwargs: object) -> None:
        """Observe the actual operation scope inside each pool worker before local mutation."""
        assert threading.get_ident() != caller
        assert current_spend_ceiling() is ceiling
        assert marker.get() == "operation"
        marker.set("trace-local")
        # Force the first group to overlap: one shared Context cannot service them all.
        if cast("int", prepared) < PANEL_CONCURRENCY:
            concurrent.wait(timeout=10)
        with lock:
            started.append(prepared)

    monkeypatch.setattr(replay_panel, "execute_fidelity_evaluation", execute)
    monkeypatch.setattr(replay_panel, "interrupted", lambda _store, _prepared: False)
    traces = tuple(range(PANEL_CONCURRENCY * 3))
    assert current_spend_ceiling() is None
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        token = marker.set("operation")
        try:
            execute_panel(
                cast("ProjectStore", None),
                cast("tuple[PreparedFidelity, ...]", traces),
            )
            assert marker.get() == "operation"
            assert current_spend_ceiling() is ceiling
        finally:
            marker.reset(token)
    assert current_spend_ceiling() is None
    assert marker.get() == "outside"
    assert sorted(cast("list[int]", started)) == list(traces)

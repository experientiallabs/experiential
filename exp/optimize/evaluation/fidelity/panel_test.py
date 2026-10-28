"""A failed panel trace stops the panel from starting any queued trace."""

import threading
from typing import cast

import pytest

from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import panel as replay_panel
from exp.optimize.evaluation.fidelity.panel import PANEL_CONCURRENCY, execute_panel
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity
from exp.runtime.models.providers.transport import ProviderTransportError


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

"""Explicit original deadlines reach the actual native decision physical transport."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.build_records import VarianceBuildSpec
from exp.optimize.evaluation.variance.build_test import TASK_BANK, DecisionWire, variance_project
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope


def test_native_decision_clients_keep_physical_receipts_and_cached_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native decision HTTP calls settle in the original ledger and replay without redispatch."""
    project = variance_project(tmp_path, monkeypatch)
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.build_clients.HttpxJsonTransport", lambda: wire
    )
    spec = VarianceBuildSpec(
        decision_alias="jev", task_bank=TASK_BANK, clustering=None, concurrency=1
    )
    ceiling = operation_spend_ceiling(project, "original-operation")
    ceiling.authorize("preparation", ceiling_usd=0, unpriced_attempts=1000)
    checked: list[bool] = []

    def check() -> None:
        """Record actual physical dispatch boundary checks."""
        checked.append(True)

    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        report = run_variance_build(
            project,
            spec,
            code_revision="fixture",
            execution_deadline=datetime.now(UTC) + timedelta(minutes=5),
            before_claim=check,
        )
        assert run_variance_build(project, spec, code_revision="fixture") == report
    assert wire.states and len(checked) == len(wire.states)
    assert ceiling.summary().unpriced_attempts == len(wire.states)
    receipts = [
        identity
        for identity in project.artifacts.list_ids()
        if project.artifacts.read(identity).manifest.artifact_type == "provider-physical-attempt"
    ]
    assert receipts


def test_original_absolute_deadline_refuses_before_decision_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh tagging-hours default cannot renew an explicitly supplied expired original stop."""
    project = variance_project(tmp_path, monkeypatch)
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.build_clients.HttpxJsonTransport", lambda: wire
    )
    spec = VarianceBuildSpec(
        decision_alias="jev", task_bank=TASK_BANK, clustering=None, concurrency=1
    )
    with pytest.raises(TimeoutError, match="deadline reached before physical dispatch"):
        run_variance_build(
            project,
            spec,
            code_revision="fixture",
            execution_deadline=datetime(2000, 1, 1, tzinfo=UTC),
        )
    assert not wire.states
    assert not any(
        identity.endswith("-result")
        for identity in project.artifacts.list_ids()
        if identity.startswith("variance-build-")
    )

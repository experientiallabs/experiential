"""Cascade deciders are checked before credentials and leave a durable receipt per decision."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.models import ModelRequest, load_model_catalog
from exp.common.project import ArtifactStore, ProjectPaths
from exp.optimize.evaluation.analysis.decision_cascade_test import drafts, systemone
from exp.optimize.evaluation.analysis.decision_points_test import catalog_model, project_fixture
from exp.optimize.evaluation.continuations.deciders import (
    DecisionReceipt,
    RecordedDecider,
    preflight_decider,
    saved_decision,
)
from exp.runtime.agents.cascade import DeciderDefinition, NextActionDecision
from exp.runtime.agents.cascade_test import CANDIDATES, Decider, ranked, request


def test_preflight_checks_kind_capability_and_frozen_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat alias cannot rank, a decision alias cannot draft, and a drifted snapshot refuses."""
    project = project_fixture(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    preflight_decider(catalog, systemone(project))
    preflight_decider(catalog, drafts(project))
    with pytest.raises(ValueError, match="TypeSafe decision alias"):
        preflight_decider(
            catalog,
            DeciderDefinition(
                kind="systemone", alias="judge", model=catalog_model(project, "judge")
            ),
        )
    with pytest.raises(ValueError, match="outside the admitted HTTP transport"):
        preflight_decider(
            catalog,
            drafts(project).model_copy(update={"alias": "jev", "model": systemone(project).model}),
        )
    with pytest.raises(ValueError, match="frozen snapshot"):
        preflight_decider(
            catalog,
            systemone(project).model_copy(update={"model": catalog_model(project, "world")}),
        )


class Interrupting(Decider):
    """A decider whose dispatch is interrupted like a killed process."""

    def decide(self, request: ModelRequest, candidates: tuple[str, ...]) -> NextActionDecision:
        """Raise after the durable intent was written."""
        raise KeyboardInterrupt


def test_each_decision_is_receipted_once_and_never_redrawn(tmp_path: Path) -> None:
    """Answers and failures are saved; an orphaned intent reads as an unknown outcome."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    inner = Decider(
        "systemone", [ranked("close_ticket", 0.9, CANDIDATES), RuntimeError("provider down")]
    )
    recorded = RecordedDecider(inner, store=store, prefix="run", inputs=(), code_revision="test")
    actor = request("close_ticket", "open_ticket")
    decision = recorded.decide(actor, CANDIDATES)
    assert saved_decision(store, "run-decision-0") == DecisionReceipt(decision=decision)
    with pytest.raises(RuntimeError):
        recorded.decide(actor, CANDIDATES)
    assert saved_decision(store, "run-decision-1") == DecisionReceipt(failure="RuntimeError")
    with pytest.raises(ValueError, match="already dispatched"):
        recorded.decide_at("run-decision-0", actor, CANDIDATES)
    interrupted = RecordedDecider(
        Interrupting("systemone", []), store=store, prefix="crash", inputs=(), code_revision="t"
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.decide(actor, CANDIDATES)
    assert saved_decision(store, "crash-decision-0") == DecisionReceipt(
        failure="UnfinishedDispatchIntent"
    )
    assert saved_decision(store, "never-started") is None
    assert recorded.definition == inner.definition

"""The held-out exposure decision needs the build's grouping in the ledger it decides with."""

import hashlib
from functools import partial
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.tasks.store import load_task_set
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.audit.exposure import held_out_exposure, held_out_members
from exp.optimize.evaluation.audit.lifecycle_test import audit
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    held_out_lineages,
)
from exp.optimize.evaluation.audit.receipts_test import _dataset
from exp.simulation.build import build_project as build_artifacts
from exp.simulation.mining.exposure_ledger import ExposureLedger
from exp.simulation.mining.exposure_ledger_test import DATASET, EDITED, trace
from exp.simulation.mining.fingerprint import TRACE_CONTENT
from exp.simulation.mining.lineage import assign_source_lineages
from exp.simulation.mining.service import MiningSpec


def test_exposure_is_decided_only_with_a_ledger_that_recorded_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every build records itself; a ledger without that build cannot decide independence."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    members = held_out_members(store)
    assert set(members) == held_out_lineages(store)
    assert all(item.vertices and not item.unresolved for item in members.values())
    decision = held_out_exposure(store)
    assert decision.exposed == ()
    assert decision.ledger == ExposureLedger(store.paths.root).snapshot().identity
    with pytest.raises(ValueError, match="never recorded its traces and grouping"):
        held_out_exposure(store, ExposureLedger(tmp_path / "elsewhere").snapshot())


@pytest.mark.parametrize("source_project", ["retail", "alternate-source"])
def test_a_group_with_an_unresolved_member_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_project: str
) -> None:
    """A5: one unfingerprintable member makes its whole lineage ineligible for independence."""
    originals = [episode(i) for i in range(USERS)]
    # More than half the lineages are held out, so every source-path-dependent split must
    # include both classes: only six customers have an unresolved sibling, and six do not.
    held_out_target = USERS // 2 + 2
    monkeypatch.setattr(
        "exp.cli.build.app.build_project",
        partial(
            build_artifacts,
            mining_spec=MiningSpec(
                fit_task_budget=USERS - held_out_target, held_out_task_budget=held_out_target
            ),
        ),
    )
    # Even customers also have a repeat in the same conversation recorded without call IDs,
    # whose trace ID sorts after the original's, so the selected trace is the fingerprintable one.
    follow_ups: list[JsonObject] = [
        {**episode(i), "repeat": _later_tag(f"user_{i:02d}"), "no_call_ids": True}
        for i in range(0, USERS, 2)
    ]
    provider = SimulatedProvider(originals)
    _, store, _ = build_project(
        tmp_path, monkeypatch, provider, episodes=originals + follow_ups, project=source_project
    )
    members = held_out_members(store)
    assert len(members) == held_out_target
    unresolved = {lineage for lineage, item in members.items() if item.unresolved}
    assert unresolved and unresolved != set(members)
    decision = held_out_exposure(store)
    assert {item.lineage for item in decision.exposed if item.kind == "unresolved"} == unresolved
    with audit(store, USERS) as opened:
        assert set(opened.receipt.lineages) == set(members) - unresolved
        assert opened.receipt.basis == "independent" and opened.excluded == len(unresolved)
    # The excluded groups' selected traces are themselves fingerprintable siblings.
    traces = {t.trace_id: t for t in load_trace_dataset(store.artifacts, _dataset(store)).traces}
    build = store.load_project().build
    assert build is not None
    selected = [
        traces[min(task.source_trace_ids)]
        for task in load_task_set(store.artifacts, build.task_set.artifact_id).tasks
        if task.lineage_group_id in unresolved
    ]
    assert selected and all(TRACE_CONTENT.compute(trace) is not None for trace in selected)


def _later_tag(user: str) -> str:
    """A repeat tag whose fixture trace ID sorts after the original trace's."""
    original = hashlib.sha256(user.encode()).hexdigest()
    return next(
        tag
        for tag in (f"follow-up-{n}" for n in range(100))
        if hashlib.sha256((user + tag).encode()).hexdigest() > original
    )


def test_unknown_scheme_nodes_without_equivalence_refuse_independent_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A4: unresolved nodes under an unknown scheme refuse and are counted; re-audits proceed."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    elsewhere = ExposureLedger(tmp_path / "elsewhere", schemes=(EDITED,))
    elsewhere.record_build(
        project="far",
        build="task-set-far",
        dataset=DATASET,
        traces=(trace(1), trace(2)),
        lineages=assign_source_lineages((trace(1), trace(2))),
        near_duplicates=(),
    )
    assert ExposureLedger(store.paths.root).merge(elsewhere.export()) == (EDITED.identity,)
    with pytest.raises(ValueError, match="2 under trace-content v1") as refusal, audit(store):
        pass
    assert EDITED.identity.definition_sha256[:12] in str(refusal.value)
    with audit(store, include_inspected=True) as reaudit:
        assert reaudit.receipt.basis == "development-re-audit"

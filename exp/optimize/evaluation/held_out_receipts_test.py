"""Held-out audit receipts, their status, and exposure across projects, imports and grouping.

Builds run offline through the routed fake of :mod:`held_out_fidelity_test`; every project
shares one ``.exp`` root, so its exposure ledger is shared too.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.held_out_exposure import held_out_exposure
from exp.optimize.evaluation.held_out_fidelity import HeldOutPanel, register_held_out_fidelity
from exp.optimize.evaluation.held_out_fidelity_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    held_out_command,
    held_out_lineages,
)
from exp.optimize.evaluation.held_out_receipts import (
    RECEIPT_TYPE,
    AuditReceipt,
    audit_receipts,
    record_audit_receipt,
)
from exp.optimize.evaluation.trace_learning import learning_plan
from exp.simulation.mining.exposure_ledger import (
    FINGERPRINT_SCHEME,
    ExposureLedger,
    trace_fingerprint,
)
from exp.simulation.world_model.experience_memory import memory_head

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def register(
    store: ProjectStore, count: int = 1, *, include_inspected: bool = False
) -> HeldOutPanel:
    """Register a source-RAG held-out panel without provider calls."""
    return register_held_out_fidelity(
        store,
        count,
        profile="source-rag",
        code_revision=REVISION,
        clock=lambda: NOW,
        include_inspected=include_inspected,
    )


def record(
    store: ProjectStore,
    panel: HeldOutPanel,
    *,
    fresh: bool,
    detail: bool = False,
    revision: str = REVISION,
    at: datetime = NOW,
) -> AuditReceipt:
    """Record a reported audit of ``panel``."""
    return record_audit_receipt(
        store,
        panel,
        fresh=fresh,
        results_shown=True,
        detail_shown=detail,
        code_revision=revision,
        clock=lambda: at,
    )


def test_receipt_status_is_independent_replay_or_development_reaudit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first fresh report is independent; repeats replay it or re-audit for development."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    panel = register(store)
    lineage = panel.cases[0].task.lineage_group_id
    assert panel.identity == register(store).identity, "identity follows cases and runs only"
    assert panel.identity != register(store, 2).identity

    first = record(store, panel, fresh=True)
    assert (first.status, first.replays, first.sequence) == ("independent", None, 1)
    assert first.panel == panel.identity and first.code_revision == REVISION
    assert first.lineages == (lineage,) and first.runs == (panel.prepared[0].spec.run_id,)
    assert first.scheme == FINGERPRINT_SCHEME
    assert first.ledger == ExposureLedger(store.paths.root).snapshot().identity
    assert first.exposed == () and not first.detail_shown

    replay = record(store, panel, fresh=False, at=NOW + timedelta(seconds=1))
    assert (replay.status, replay.replays) == ("replay", first.receipt_id)
    other_code = record(store, panel, fresh=False, revision="d" * 40, at=NOW + timedelta(2))
    assert other_code.status == "development-re-audit", "a code-revision change is no replay"
    assert held_out_exposure(store).exposed == (), "aggregates expose nothing"

    # Saved results that no receipt ever reported are not a fresh evaluation.
    unreported = register(store, 2)
    assert record(store, unreported, fresh=False).status == "development-re-audit"

    # Per-trace detail exposes the lineage at the root; the default panel moves past it.
    detail = record(store, panel, fresh=False, detail=True, at=NOW + timedelta(3))
    assert detail.status == "replay" and detail.detail_shown
    exposure = held_out_exposure(store)
    assert exposure.lineages == {lineage} and exposure.ledger != first.ledger
    [exposed] = exposure.exposed
    assert exposed.path.path == () and exposed.path.component_size >= 1
    moved = register(store)
    assert moved.cases[0].task.lineage_group_id != lineage
    assert moved.excluded == 1 and moved.independent
    assert record(store, moved, fresh=True).status == "independent"

    # A fresh evaluation that touches exposed cases is a development re-audit.
    touched = register(store, 3, include_inspected=True)
    assert touched.inspected == (lineage,) and not touched.independent
    reaudit = record(store, touched, fresh=True)
    assert reaudit.status == "development-re-audit" and reaudit.reaudited == (lineage,)
    assert [item.lineage for item in reaudit.exposed] == [lineage]

    # Receipts are immutable artifacts: later audits add receipts and change none.
    receipts = audit_receipts(store)
    assert receipts[0] == first and [r.sequence for r in receipts] == list(range(1, 8))
    assert len(store.artifacts.list_ids(artifact_type=RECEIPT_TYPE)) == 7


def test_exhausted_held_out_panel_refuses_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once every held-out lineage is exposed, an independent audit needs fresh traces."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    panel = register(store, USERS)
    assert {case.task.lineage_group_id for case in panel.cases} == held_out_lineages(store)
    record(store, panel, fresh=True, detail=True)
    purchased = len(provider.requests)
    with pytest.raises(ValueError, match="fresh traces") as refusal:
        register(store)
    assert "new lineages" in str(refusal.value) and "--include-inspected" in str(refusal.value)
    assert len(provider.requests) == purchased
    assert not register(store, include_inspected=True).independent


def test_exposure_follows_content_across_projects_imports_and_regrouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exposed A grouped with B in one build; C later groups only with B: C is exposed."""
    # Every customer also has a follow-up trace (B) in its own conversation that declares the
    # same task identity as the original (A), so whichever A is exposed has a grouped B.
    follow_ups: list[JsonObject] = [
        {
            **episode(i),
            "instruction": f"{episode(i)['instruction']} Also, what is my order total?",
            "conversation": f"follow-up-{i}",
            "repeat": "follow-up",
        }
        for i in range(USERS)
    ]
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, one, _ = build_project(
        tmp_path, monkeypatch, provider, episodes=[episode(i) for i in range(USERS)] + follow_ups
    )
    panel = register(one)
    receipt = record(one, panel, fresh=True, detail=True)
    traces = {t.trace_id: t for t in load_trace_dataset(one.artifacts, _dataset(one)).traces}
    exposed_trace = traces[receipt.traces[0]]
    customer = str(exposed_trace.initial_context["task_lineage"])
    index = int(customer.removeprefix("user_"))
    a = trace_fingerprint(exposed_trace)

    # A later import drops A. B arrives with the same content but a new trace ID and a new
    # declared identity, which only C shares; A' is A's content under a new trace ID.
    b_again: JsonObject = {**follow_ups[index], "lineage": "regrouped", "repeat": "again"}
    c: JsonObject = {**episode(USERS + 1), "lineage": "regrouped", "conversation": "fresh"}
    a_again: JsonObject = {**episode(index), "repeat": "reimport"}
    others = [episode(i) for i in range(USERS) if i != index]
    _, two, _ = build_project(
        tmp_path,
        monkeypatch,
        provider,
        episodes=[*others, b_again, c, a_again],
        project="second",
    )
    snapshot = ExposureLedger(two.paths.root).snapshot()
    by_content = {trace_fingerprint(t): t for t in _traces(two)}
    c_trace = next(
        t
        for t in by_content.values()
        if t.initial_context["task_lineage"] == "regrouped" and "Also" not in t.task
    )
    found = snapshot.exposure((trace_fingerprint(c_trace),))
    assert found is not None and found.exposed_node == a
    assert found.path[0].reason == "declared-task-identity"
    assert (found.path[0].task_lineage, found.path[0].project) == ("regrouped", "second")
    assert found.path[-1].project == "retail" and len(found.path) == 2
    assert found.component_size >= 3
    # Reimported content is exposed itself, with no path; another customer's trace is not.
    direct = snapshot.exposure((a,))
    assert a in by_content and direct is not None and direct.path == ()
    neighbour = f"user_{(index + 1) % USERS:02d}"
    unrelated = next(t for t in _traces(two) if t.initial_context["task_lineage"] == neighbour)
    assert snapshot.exposure((trace_fingerprint(unrelated),)) is None


def test_held_out_command_reports_receipts_checkpoint_and_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI names its checkpoint and ledger, classifies receipts, and refuses when spent."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(
        tmp_path, monkeypatch, provider, "--learn", "--hold-out", "--memory-recall", "jev"
    )
    plan = learning_plan(store, scope="fit")
    head = memory_head(store.artifacts, plan.lineage)
    assert head is not None
    held = held_out_lineages(store)
    assert len(held) >= 3

    dry = held_out_command(root, 1, "--dry-run", "--verbose")
    [prepared] = audit_receipts(store)
    assert not prepared.results_shown and not prepared.detail_shown
    assert "Prepared without provider calls" in dry and "Audit receipt #1" in dry

    output = held_out_command(root, 1, "--verbose")
    assert (
        f"Audited checkpoint: memory {plan.lineage} v{head.version}, scope fit "
        "(held-out lineages reserved before learning)" in output
    )
    assert (
        "An all-traces build (exp build --learn without --hold-out) is a different checkpoint "
        "and was not independently evaluated" in output
    )
    receipt = audit_receipts(store)[-1]
    assert f"Audit receipt #2 {receipt.receipt_id}: independent" in output
    assert f"Exposure ledger {receipt.ledger} ({FINGERPRINT_SCHEME})" in output
    assert "Limit: Knowledge a person carries over by hand is not tracked." in output
    assert "Held-out fidelity 3/3" in output and "now development data" in output
    assert receipt.detail_shown and receipt.checkpoint is not None
    assert (receipt.checkpoint.lineage, receipt.checkpoint.version) == (plan.lineage, head.version)
    [inspected] = receipt.lineages
    assert held_out_exposure(store).lineages == {inspected}

    following = held_out_command(root, 1)
    assert "Skipped 1 held-out lineage exposed in the exposure ledger" in following
    assert f"{inspected} (skipped): component of" in following
    fresh = audit_receipts(store)[-1]
    assert fresh.status == "independent" and fresh.lineages != (inspected,)
    replay = held_out_command(root, 1)
    assert "Held-out fidelity (replay)" in replay
    assert f"replay of receipt #{fresh.sequence} (independent)" in replay

    reaudit = held_out_command(root, len(held), "--include-inspected")
    assert "Re-audit on inspected cases (development)" in reaudit
    assert "not an independent measurement" in reaudit
    assert f"{inspected} (in this panel)" in reaudit
    assert audit_receipts(store)[-1].status == "development-re-audit"

    # Every remaining result was computed by the re-audit, so this is no fresh evaluation.
    rest = held_out_command(root, len(held), "--verbose")
    assert "Held-out re-audit (development)" in rest
    assert held_out_exposure(store).lineages == held
    purchased = len(provider.requests)
    receipts = len(audit_receipts(store))
    refusal = held_out_command(root, 1, exit_code=2)
    assert "independent audit needs fresh traces" in refusal
    assert len(provider.requests) == purchased and len(audit_receipts(store)) == receipts


def _dataset(store: ProjectStore) -> str:
    """The build's trace dataset artifact."""
    build = store.load_project().build
    assert build is not None
    return build.trace_dataset.artifact_id


def _traces(store: ProjectStore) -> tuple[Trace, ...]:
    """Every trace of the project's build."""
    return load_trace_dataset(store.artifacts, _dataset(store)).traces

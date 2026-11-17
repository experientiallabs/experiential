"""Held-out audit receipts and reports: headline, exposure, exhaustion, and the CLI report.

Builds run offline through the routed fake of :mod:`held_out_fidelity_test`; every project
shares one ``.exp`` root, so its exposure ledger is shared too.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.audit.exposure import LineageExposure, held_out_exposure
from exp.optimize.evaluation.audit.lifecycle_test import FailingOnce, audit, clock, run
from exp.optimize.evaluation.audit.panel import saved_held_out_results
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    USERS,
    SimulatedProvider,
    audit_report,
    build_project,
    episode,
    held_out_command,
    held_out_lineages,
    learn_fit,
)
from exp.optimize.evaluation.audit.receipts import (
    QUALIFIED,
    AuditNotice,
    audit_receipts,
    find_receipt,
    receipt_panel,
    receipt_view,
    record_notice,
    record_report,
)
from exp.optimize.evaluation.learning.traces import learning_plan, recorded_plan
from exp.simulation.mining.exposure_ledger import (
    ExposureLedger,
)
from exp.simulation.mining.fingerprint import TRACE_CONTENT
from exp.simulation.world_model.experience_memory import memory_head


def test_reports_append_to_the_receipt_and_detail_exposes_its_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reports never rewrite the receipt; per-trace detail exposes, and selection moves on."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    with audit(store) as opened:
        receipt = opened.receipt
        record_report(
            store, receipt, opened.panel, kind="prepared", detail_shown=False, clock=clock
        )
        run(store, opened)
        record_report(
            store, receipt, opened.panel, kind="evaluation", detail_shown=False, clock=clock
        )
        reserved = held_out_exposure(store).exposed
        assert [item.kind for item in reserved] == ["reserved"], "aggregates expose nothing"
        record_report(store, receipt, opened.panel, kind="replay", detail_shown=True, clock=clock)
    view = receipt_view(store, receipt)
    assert [report.kind for report in view.reports] == ["prepared", "evaluation", "replay"]
    assert view.headline == "independent" and view.reported
    assert view.reports[-1].runs == (receipt.cases[0].base_run,)
    exposure = held_out_exposure(store, owner=receipt.receipt_id)
    [exposed] = exposure.exposed
    assert exposed.lineage == receipt.lineages[0] and exposed.kind == "exposed"
    assert exposed.path is not None and exposed.path.by and exposed.path.path == ()

    with audit(store) as moved:
        assert moved.receipt != receipt and moved.receipt.basis == "independent"
        assert moved.excluded == 1 and moved.receipt.lineages != receipt.lineages
    with audit(store, include_inspected=True) as reaudit:
        assert reaudit.receipt.basis == "development-re-audit"
        assert receipt.lineages[0] in reaudit.receipt.reaudited
        assert receipt_view(store, reaudit.receipt).headline == "development re-audit"
    assert [r.sequence for r in audit_receipts(store)] == [1, 2, 3]


def test_exhausted_held_out_panel_refuses_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once every held-out lineage is exposed or reserved, an audit needs fresh traces."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    with audit(store, USERS) as opened:
        assert set(opened.receipt.lineages) == held_out_lineages(store)
    purchased = len(provider.requests)
    with pytest.raises(ValueError, match="fresh traces") as refusal, audit(store):
        pass
    assert "new lineages" in str(refusal.value) and "--include-inspected" in str(refusal.value)
    assert len(provider.requests) == purchased
    with audit(store, include_inspected=True) as reaudit:
        assert reaudit.receipt.basis == "development-re-audit"


def test_the_audit_command_refuses_an_exhausted_panel_before_fit_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exp eval --audit`` checks the panel before its paid fit-learning step."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    with audit(store, USERS) as opened:
        assert set(opened.receipt.lineages) == held_out_lineages(store)
    purchased = len(provider.requests)
    refusal = held_out_command(root, 1, exit_code=2)
    assert "independent audit needs fresh traces" in refusal
    assert len(provider.requests) == purchased and not recorded_plan(store, "fit")


def test_exposure_follows_content_across_projects_imports_and_regrouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exposed A's lineage holds B; C later groups only with B's content: C is exposed."""
    # Every customer also has a follow-up trace (B) in its own conversation that declares the
    # same task identity as the original (A). The build joins declared identities within one
    # source and customer, so whichever A is exposed shares its held-out lineage with B.
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
    with audit(one) as opened:
        record_report(
            one, opened.receipt, opened.panel, kind="replay", detail_shown=True, clock=clock
        )
        receipt = opened.receipt
    traces = {t.trace_id: t for t in load_trace_dataset(one.artifacts, _dataset(one)).traces}
    exposed_trace = traces[receipt.cases[0].trace_id]
    customer = str(exposed_trace.initial_context["task_lineage"])
    index = int(customer.removeprefix("user_"))
    a = vertex(exposed_trace)

    # The exposed trace A is either the original or the follow-up; B is the other one.
    originals = {"original": episode(index), "follow-up": follow_ups[index]}
    exposed_kind = "follow-up" if "Also" in exposed_trace.task else "original"
    other = originals["original" if exposed_kind == "follow-up" else "follow-up"]
    # A later import drops A. B arrives with the same content but a new trace ID and a new
    # declared identity, which only C shares.
    b_again: JsonObject = {**other, "lineage": "regrouped", "repeat": "again"}
    c: JsonObject = {**episode(USERS + 1), "lineage": "regrouped", "conversation": "fresh"}
    others = [episode(i) for i in range(USERS) if i != index]
    _, two, _ = build_project(
        tmp_path, monkeypatch, provider, episodes=[*others, b_again, c], project="second"
    )
    snapshot = ExposureLedger(two.paths.root).snapshot()
    by_content = {vertex(t): t for t in _traces(two)}
    c_trace = next(
        t
        for t in by_content.values()
        if t.initial_context["task_lineage"] == "regrouped" and "#W1013" in t.task
    )
    # Per-trace detail exposed every member of A's lineage, so B's content is exposed itself:
    # C reaches it in one step through its new declared identity in the second project, the
    # retail build's declared edge still joins B to A, and the same report marked both.
    b = next(
        vertex(t)
        for t in traces.values()
        if t.initial_context["task_lineage"] == customer and t.trace_id != exposed_trace.trace_id
    )
    found = snapshot.exposure((vertex(c_trace),))
    assert found is not None and found.exposed_node == b
    [step] = found.path
    assert (step.reason, step.task_lineage, step.project) == (
        "declared-task-identity",
        "regrouped",
        "second",
    )
    [shown] = [item for item in snapshot.records.exposures if item.record_id in found.by]
    assert {a, b} <= set(shown.vertices) and shown.project == "retail"
    assert any(
        edge.reason == "declared-task-identity" and {edge.left, edge.right} == {a, b}
        for edge in snapshot.records.edges
        if edge.project == "retail"
    )
    assert found.component_size >= 3
    assert a not in by_content, "A left the later import"
    neighbour = f"user_{(index + 1) % USERS:02d}"
    unrelated = next(t for t in _traces(two) if t.initial_context["task_lineage"] == neighbour)
    assert snapshot.exposure((vertex(unrelated),)) is None
    # A third import carries A's content again under a new trace ID: exposed itself, no path.
    a_again: JsonObject = {**originals[exposed_kind], "repeat": "reimport"}
    _, three, _ = build_project(
        tmp_path, monkeypatch, provider, episodes=[*others, a_again], project="third"
    )
    assert a in {vertex(t) for t in _traces(three)}
    direct = ExposureLedger(three.paths.root).snapshot().exposure((a,))
    assert direct is not None and direct.path == ()


def test_held_out_command_reports_receipts_checkpoint_and_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI names its checkpoint, ledger and receipt, labels re-audits, refuses when spent."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    plan = learning_plan(store, scope="fit")
    head = memory_head(store.artifacts, plan.lineage)
    assert head is not None
    held = held_out_lineages(store)
    assert len(held) >= 3

    dry = held_out_command(root, 1, "--dry-run", "--verbose")
    [receipt] = audit_receipts(store)
    assert "Prepared without provider calls" in dry and receipt.receipt_id in dry

    output = held_out_command(root, 1, "--verbose")
    assert "Report:" in output
    assert "Held-out fidelity 3/3" in output and "now development data" in output
    output = audit_report(store)  # the receipt, checkpoint and exposure detail
    assert (
        f"Audited checkpoint: memory {plan.lineage} v{head.version}, scope fit, "
        f"{head.version} fit traces learned, simulator {receipt.world_model.artifact_id} "
        "(held-out lineages reserved before learning)" in output
    )
    assert (
        "The build's all-traces memory is a different checkpoint "
        "and was not independently evaluated" in output
    )
    assert audit_receipts(store) == (receipt,), "the dry run's receipt is resumed"
    assert f"Audit receipt #1 {receipt.receipt_id}: independent · this run: evaluation" in output
    scheme = TRACE_CONTENT.identity
    assert f"Exposure ledger {receipt.ledger} ({scheme.name} v{scheme.version} definition" in output
    assert receipt.schemes == (scheme,)
    assert "Limit: Knowledge a person carries over by hand is not tracked." in output
    assert "Limit: Reservations and panel leases are atomic only for runs sharing" in output
    assert "Held-out fidelity 3/3" in output
    assert receipt.checkpoint is not None
    assert (receipt.checkpoint.lineage, receipt.checkpoint.version) == (plan.lineage, head.version)
    assert held_out_exposure(store).lineages == set(receipt.lineages)

    following = held_out_command(root, 1)
    assert "Report:" in following
    following += " " + audit_report(store)
    assert "Skipped 1 held-out lineage exposed, reserved or unresolved" in following
    assert f"{receipt.lineages[0]} (exposed, skipped): component of" in following
    second = audit_receipts(store)[-1]
    assert second.sequence == 2 and second.lineages != receipt.lineages
    purchased_before_replay = len(provider.requests)
    replay = held_out_command(root, 1)
    assert "Held-out fidelity 3/3" in replay and "Report:" in replay
    assert len(provider.requests) == purchased_before_replay
    replay += " " + audit_report(store)
    assert "Held-out fidelity 3/3" in replay and "this run: replay" in replay
    assert f"Audit receipt #2 {second.receipt_id}: replay (independent)" in replay

    reaudit = held_out_command(root, len(held), "--include-inspected") + " " + audit_report(store)
    assert "Re-audit on inspected cases (development)" in reaudit
    assert "not an independent measurement" in reaudit
    third = audit_receipts(store)[-1]
    assert f"Audit receipt #3 {third.receipt_id}: development re-audit · this run" in reaudit

    purchased = len(provider.requests)
    receipts = len(audit_receipts(store))
    refusal = held_out_command(root, 2, exit_code=2)
    assert "independent audit needs fresh traces" in refusal
    assert len(provider.requests) == purchased and len(audit_receipts(store)) == receipts


def test_a_trace_reattempt_qualifies_the_headline_and_reports_both_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crash, --retry-incomplete resumes the trace in place under the same receipt."""
    provider = FailingOnce([episode(i) for i in range(USERS)])
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    provider.armed = True
    crashed = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--audit", "2"] + ["--yes", "--non-interactive"],
    )
    assert crashed.exit_code == 130, crashed.output  # interrupted after a claimed slot
    stopped = held_out_command(root, 2, "--dry-run")
    assert "dispatched, outcome unknown" in stopped
    terminal = held_out_command(root, 2, "--retry-incomplete")
    output = terminal + " " + audit_report(store)
    [receipt] = audit_receipts(store)
    view = receipt_view(store, receipt)
    [child] = view.reattempts
    assert view.headline == QUALIFIED
    assert "Report:" in output
    details = audit_report(store)
    assert f"Audit receipt #1 {receipt.receipt_id}: {QUALIFIED}" in details
    assert "Held-out fidelity (independent, qualified: trace re-attempts) 6/6 replies" in output
    assert "Unaffected traces only 3/3 replies exact (100%)" in output
    assert "Re-attempted traces only 3/3 replies exact (100%)" in output
    assert terminal.count("· 1/1 traces") == 2, "each subset states its own denominator"
    assert (
        "re-dispatched after an unknown outcome prediction 1 · judge 0 · investigation 0 · "
        "1 of 2 traces affected"
    ) in details
    assert child.in_place and child.retired_run == child.replacement_run
    held_out_command(root, 2)
    assert (
        f"re-attempt of {child.trace_id}: {child.retired_run} resumed in place "
        f"({child.claimed_slots} claimed slots); step 1 re-dispatched after an unknown outcome "
        "(prediction, 2 attempts)"
    ) in held_out_command(root, 2) + " " + audit_report(store)
    # A later invalidation keeps the qualification beside it, never replacing either.
    notice = AuditNotice(
        receipt=receipt.receipt_id,
        at=clock(),
        ledger=receipt.ledger,
        exposed=(LineageExposure(lineage=receipt.lineages[0]),),
    )
    assert record_notice(store, receipt, notice)
    assert receipt_view(store, receipt).headline == (
        "independent, invalidated: predating exposure, qualified: trace re-attempts"
    )


def vertex(trace: Trace) -> str:
    """The trace's vertex under the current scheme."""
    return f"{TRACE_CONTENT.identity.key}:{TRACE_CONTENT.compute(trace)}"


def _dataset(store: ProjectStore) -> str:
    """The build's trace dataset artifact."""
    build = store.load_project().build
    assert build is not None
    return build.trace_dataset.artifact_id


def _traces(store: ProjectStore) -> tuple[Trace, ...]:
    """Every trace of the project's build."""
    return load_trace_dataset(store.artifacts, _dataset(store)).traces


def test_a_saved_receipt_rebuilds_its_panel_and_unknown_receipts_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuilt panel has the receipt's identity, runs and saved results; no calls are made."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    with audit(store, 2) as opened:
        run(store, opened)
        receipt, panel = opened.receipt, opened.panel
    calls = len(provider.requests)
    rebuilt = receipt_panel(store, find_receipt(store, receipt.receipt_id))
    assert rebuilt.identity == receipt.panel == panel.identity
    assert rebuilt.prepared == panel.prepared and rebuilt.cases == panel.cases
    assert saved_held_out_results(store, rebuilt) == saved_held_out_results(store, panel)
    assert len(provider.requests) == calls
    with pytest.raises(ValueError, match="no held-out audit receipt 'absent'"):
        find_receipt(store, "absent")
    foreign = receipt.model_copy(update={"panel": "another-panel"})
    with pytest.raises(ValueError, match="no longer matches its saved runs"):
        receipt_panel(store, foreign)

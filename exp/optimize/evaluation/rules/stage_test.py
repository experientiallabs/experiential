"""Offline tests for the learned-rules stage over scripted provider replies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.project import ProjectStore
from exp.common.project.manifests import artifact_input
from exp.common.traces import Trace
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.challenges import ChallengeCase
from exp.optimize.evaluation.rules.extraction_test import (
    RoutedTransport,
    _answer,
    _setup,
    _tool_reply,
    _wire,
)
from exp.optimize.evaluation.rules.stage import (
    RULE_BOOK_TYPE,
    RulesStageReport,
    RulesStageSpec,
    export_rulebook,
    load_rulebook,
    run_rules_stage,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan, Role
from exp.simulation.rules.contracts_test import refund_trace
from exp.simulation.rules.model import Transition, Witness

_BASE_STATE: JsonObject = {"orders": {"7": {"status": "paid", "amount": 40}}}


def _plan(world: tuple[str, ...], probes: tuple[str, ...] = ("probe",)) -> ExposurePlan:
    """An exposure plan giving each trace its own lineage.

    Args:
        world: Traces reserved for world learning.
        probes: Traces reserved as mining probes.

    Returns:
        The plan.
    """
    roles: dict[str, Role] = {f"l-{name}": "world_learning" for name in world}
    for name in probes:
        roles[f"l-{name}"] = "mining_probes"
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.3,
        roles=roles,
        traces={name: f"l-{name}" for name in (*world, *probes)},
    )


def _spec(settings: AuthorSettings) -> RulesStageSpec:
    """A stage registration that generates refund boundary workloads.

    Args:
        settings: Frozen extraction settings.

    Returns:
        The spec.
    """
    return RulesStageSpec(
        author=settings,
        concurrency=3,
        base_state=_BASE_STATE,
        examples={"refund": {"order_id": "7", "amount": 10}},
    )


def _session(workflow_id: str, amount: int, reply: JsonObject) -> WorkflowEvidence:
    """One simulated session with a single refund step.

    Args:
        workflow_id: Workflow identity.
        amount: Refund amount requested.
        reply: What the simulated environment answered.

    Returns:
        The evidence.
    """
    return WorkflowEvidence(
        workflow_id=workflow_id,
        task="Refund order 7.",
        initial_state=_BASE_STATE,
        steps=(
            WorkflowStep(
                step=0,
                action={"name": "refund", "arguments": {"order_id": "7", "amount": amount}},
                status="valid",
                reply=reply,
                before=_BASE_STATE,
                after=_BASE_STATE,
            ),
        ),
    )


def _save_session(project: ProjectStore, evidence: WorkflowEvidence) -> ArtifactInput:
    """Save a session as a workflow evaluation intent holding ``evidence.json``.

    Args:
        project: Project receiving the artifact.
        evidence: The session.

    Returns:
        Its reference.
    """
    manifest = project.artifacts.write_json(
        artifact_id="session-" + evidence.workflow_id,
        artifact_type="workflow-evaluation-intent",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"evidence.json": evidence},
    )
    return artifact_input(manifest)


def _traces(*names: str) -> tuple[Trace, ...]:
    """Refund trajectories with the given identities."""
    return tuple(refund_trace(name) for name in names)


def _observed_permit(amount: int) -> Transition:
    """A refund the simulated environment permitted, as observed for a workload.

    Args:
        amount: Refund amount the workload requested.

    Returns:
        The observed transition.
    """
    return Transition(
        before=_BASE_STATE,
        after=_BASE_STATE,
        action={"name": "refund", "arguments": {"order_id": "7", "amount": amount}},
        observation={"ok": True},
    )


def test_full_run_saves_a_rule_book_challenges_it_and_replays_without_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contracts, draws, merge, rule book, properties and workloads land in one saved report."""
    project, settings = _setup(tmp_path, monkeypatch)
    traces = _traces("alpha", "beta", "gamma")
    transport = RoutedTransport(
        {
            "alpha": _tool_reply(_answer("alpha")),
            "beta": _tool_reply(_answer("beta")),
            "gamma": _tool_reply(_answer("gamma", evidence="trace:gamma:span:invented")),
        }
    )
    _wire(monkeypatch, transport)
    refused = _session("refused", 400, {"error": "refund exceeds payment"})
    permitted = _save_session(project, _session("permitted", 500, {"ok": True}))
    plan = _plan(("alpha", "beta", "gamma"))

    report = run_rules_stage(
        project,
        _spec(settings),
        exposure=plan,
        traces=traces,
        code_revision="test",
        sessions=(refused, permitted),
    )

    assert len(transport.requests) == 3
    assert list(report.contracts) == ["lookup", "refund"]
    assert [draw.trace_id for draw in report.failures] == ["gamma"]
    assert report.failures[0].response is not None
    assert report.rules == 1
    assert report.rule_counts == {"supported": 1, "contradicted": 0, "unknown": 0}
    assert report.open_questions == 1
    (rule_id,) = report.diagnostic
    assert report.requires_witness == (rule_id,)
    assert [workload.workload_id for workload in report.workloads] == [
        f"{rule_id}/{n}" for n in range(3)
    ]
    assert [workload.expected for workload in report.workloads] == [
        "permitted",
        "permitted",
        "refused",
    ]
    assert report.triage == ()
    assert report.session_ids == ("refused", "permitted")
    assert report.sessions == (permitted,)
    assert report.properties is not None
    (result,) = report.properties.results
    assert (result.holds, result.violated) == (1, 1)
    assert result.violations == ("workflow:permitted:step:0",)
    assert report.families_must_stay_diagnostic == (rule_id,)

    stored = project.artifacts.read(report.rulebook.artifact_id)
    assert stored.manifest.artifact_type == RULE_BOOK_TYPE
    assert sorted(stored.payloads) == [
        "README.md",
        "environment.md",
        "rules.json",
        "tools/lookup.md",
        "tools/refund.md",
    ]
    learned_from = sorted(
        (draw.response for draw in report.draws if draw.rules is not None and draw.response),
        key=lambda ref: ref.artifact_id,
    )
    assert stored.manifest.inputs == tuple(learned_from)
    book = load_rulebook(project, report.rulebook)
    assert book.sources == ("alpha", "beta")
    exported = export_rulebook(project, report.rulebook, tmp_path / "book")
    assert {
        path.relative_to(tmp_path / "book").as_posix(): path.read_bytes() for path in exported
    } == dict(stored.payloads)

    replay = run_rules_stage(
        project,
        _spec(settings),
        exposure=plan,
        traces=traces,
        code_revision="test",
        sessions=(refused, permitted),
    )

    assert replay == report
    assert len(transport.requests) == 3
    saved = project.artifacts.read_bytes(report.identity + "-result", "report.json")
    assert RulesStageReport.model_validate_json(saved) == report


def test_a_trace_outside_world_learning_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mining-probe trace stops the stage before contracts, draws or saved artifacts."""
    project, settings = _setup(tmp_path, monkeypatch)
    transport = RoutedTransport({})
    _wire(monkeypatch, transport)

    with pytest.raises(ValueError, match="not reserved for world_learning"):
        run_rules_stage(
            project,
            _spec(settings),
            exposure=_plan(("alpha",)),
            traces=_traces("alpha", "probe"),
            code_revision="test",
        )

    assert transport.requests == []
    assert project.artifacts.list_ids() == ()


def test_a_failed_draw_is_kept_and_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused answer stays a failure with its response, even when a later run re-keys."""
    project, settings = _setup(tmp_path, monkeypatch)
    transport = RoutedTransport({"alpha": _tool_reply(_answer("beta"))})
    _wire(monkeypatch, transport)
    plan = _plan(("alpha",))

    report = run_rules_stage(
        project, _spec(settings), exposure=plan, traces=_traces("alpha"), code_revision="test"
    )

    (failure,) = report.failures
    assert failure.response is not None and failure.rules is None
    assert failure.failure is not None and "not 'alpha'" in failure.failure
    assert report.rules == 0 and report.workloads == ()
    assert project.artifacts.read(report.rulebook.artifact_id).manifest.inputs == ()

    later = run_rules_stage(
        project,
        _spec(settings),
        exposure=plan,
        traces=_traces("alpha"),
        code_revision="test",
        sessions=(_session("later", 5, {"ok": True}),),
    )

    assert later.identity != report.identity
    assert later.failures == report.failures
    assert len(transport.requests) == 1


def test_witnesses_lift_diagnostic_status_without_redrawing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supporting witness clears a rule from every diagnostic list; draws are replayed."""
    project, settings = _setup(tmp_path, monkeypatch)
    traces = _traces("alpha", "beta")
    transport = RoutedTransport({name: _tool_reply(_answer(name)) for name in ("alpha", "beta")})
    _wire(monkeypatch, transport)
    plan = _plan(("alpha", "beta"))
    spec = _spec(settings)

    first = run_rules_stage(project, spec, exposure=plan, traces=traces, code_revision="test")
    (rule_id,) = first.diagnostic
    observed = {f"{rule_id}/2": _observed_permit(41)}

    unwitnessed = run_rules_stage(
        project, spec, exposure=plan, traces=traces, code_revision="test", observed=observed
    )

    (item,) = unwitnessed.triage
    assert item.verdict == "contradicts_rule" and item.needs_witness
    assert unwitnessed.families_must_stay_diagnostic == (rule_id,)
    assert unwitnessed.diagnostic == (rule_id,)

    witness = Witness(kind="specification", reference="spec:refunds#limit", supports=True)
    witnessed = run_rules_stage(
        project,
        spec,
        exposure=plan,
        traces=traces,
        code_revision="test",
        witnesses={rule_id: (witness,)},
        observed=observed,
    )

    assert len(transport.requests) == 2
    assert witnessed.extraction_identity == first.extraction_identity
    assert witnessed.rulebook != first.rulebook
    assert witnessed.diagnostic == ()
    assert witnessed.requires_witness == ()
    (item,) = witnessed.triage
    assert item.verdict == "contradicts_rule" and item.witnessed and not item.needs_witness
    assert witnessed.families_must_stay_diagnostic == ()
    assert load_rulebook(project, witnessed.rulebook).rules[0].witnesses == (witness,)

    with pytest.raises(ValueError, match="not in the rule book"):
        run_rules_stage(
            project,
            spec,
            exposure=plan,
            traces=traces,
            code_revision="test",
            witnesses={"precondition.refund.0000000000000000": (witness,)},
        )


class _PermitAll:
    """A bound world that permits every refund, counting what it executes.

    Attributes:
        executed: Case keys executed so far.
    """

    identity = "1" * 64

    def __init__(self) -> None:
        """Start with nothing executed."""
        self.executed: list[str] = []

    def execute(self, case: ChallengeCase) -> Transition | str | None:
        """Permit the case's refund without any state change."""
        self.executed.append(case.key)
        return Transition(
            before=case.workload.state,
            after=case.workload.state,
            action=case.workload.action,
            observation={"ok": True},
        )


def test_an_explicit_challenge_executor_runs_every_workload_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executed workloads are triaged, contradict the limit and replay without executing."""
    project, settings = _setup(tmp_path, monkeypatch)
    traces = _traces("alpha", "beta")
    transport = RoutedTransport({name: _tool_reply(_answer(name)) for name in ("alpha", "beta")})
    _wire(monkeypatch, transport)
    plan = _plan(("alpha", "beta"))
    world = _PermitAll()

    report = run_rules_stage(
        project,
        _spec(settings),
        exposure=plan,
        traces=traces,
        code_revision="test",
        challenge=world,
    )
    replay = run_rules_stage(
        project,
        _spec(settings),
        exposure=plan,
        traces=traces,
        code_revision="test",
        challenge=world,
    )

    assert replay == report and len(world.executed) == 3
    (rule_id,) = report.diagnostic
    assert report.challenges is not None
    (row,) = report.challenges.rules
    assert (row.prior, row.status) == ("supported", "contradicted")
    assert row.contradicting == (f"{rule_id}/2",)
    assert [item.verdict for item in report.triage] == [
        "consistent_with_rule",
        "consistent_with_rule",
        "contradicts_rule",
    ]
    assert report.families_must_stay_diagnostic == (rule_id,)
    plain = run_rules_stage(
        project, _spec(settings), exposure=plan, traces=traces, code_revision="test"
    )
    assert plain.challenges is None and "challenges" not in plain.model_dump(mode="json")
    with pytest.raises(ValueError, match="not both"):
        run_rules_stage(
            project,
            _spec(settings),
            exposure=plan,
            traces=traces,
            code_revision="test",
            observed={f"{rule_id}/2": _observed_permit(41)},
            challenge=world,
        )

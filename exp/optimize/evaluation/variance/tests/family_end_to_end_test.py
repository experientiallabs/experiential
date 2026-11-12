"""Decision families end to end over a real grounded world, with no stage replaced.

``run_actor_critic`` and ``run_decision_family`` run for real on the grounded ticket world of
:mod:`~exp.optimize.evaluation.variance.tests.grounded_world_test`: variant checkpoints, replays,
witnesses, option contrasts, fresh confirmations and rubric grading all execute through their
owners. Only the wire is scripted: the actor, the grounded predictor and the role draws (judge,
critic, decision-state and scenario authors) answer through scripted transports, and the rubric
grader answers through a scripted decision client.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.action_contrast import ActionContrast
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.continuations.study import required_verifier
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.qualification import (
    ExecutableQualificationSpec,
    load_executable_qualification,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable.task import persist_executable_task
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, run_actor_critic
from exp.optimize.evaluation.variance.actor_critic_test import (
    _exposure,
    _point,
    _spec,
    _tool_reply,
)
from exp.optimize.evaluation.variance.admission_test import bound_judging, unfrozen_judging
from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL, SCENARIO_TOOL
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import (
    DecisionFamilySpec,
    FamilyReport,
    export_family_harbor,
    family_executable_task,
    run_decision_family,
)
from exp.optimize.evaluation.variance.grader_test import _SNAPSHOT, Answer, _Decider, _evidence
from exp.optimize.evaluation.variance.judged_test import JudgeRouter, state_judge
from exp.optimize.evaluation.variance.records import AuditDisposition, TrainingRecords
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    ACTIONS,
    RULE,
    GroundedFixture,
    grounded_fixture,
    grounded_rules_stage,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase
from exp.simulation.evaluation.tasks import TaskVerifier, VerifierChallenge
from exp.simulation.mining.decisions import DecisionsTraceTagger

_INTENT = "Escalate or close a support ticket as its service tier requires."


def _decision_answer(*, lock_axis: bool) -> JsonObject:
    """A decision-state answer at point 1, grounded in the before-state and the tier rule."""
    axes: list[JsonObject] = [
        {"path": "/tickets/severity", "values": ["high", "low"], "effect": "The tier decides."}
    ]
    grounding: list[JsonObject] = [
        {
            "path": "/tickets/severity",
            "revealed_by": ["check_sla"],
            "refs": ["state:/tickets/severity", f"rule:{RULE.rule_id}"],
        }
    ]
    if lock_axis:
        axes.append({"path": "/tickets/lock", "values": ["none", "held"], "effect": "A hold."})
        grounding.append(
            {"path": "/tickets/lock", "revealed_by": ["check_sla"], "refs": ["state:/tickets/lock"]}
        )
    return {
        "state": {
            "step": 1,
            "decision": "Escalate or close the ticket.",
            "options": [
                {"option_id": "escalate", "action": "Hand the ticket to the second line"},
                {"option_id": "close", "action": "Close the ticket as handled"},
            ],
            "relevant_paths": ["/tickets"],
            "axes": axes,
            "criteria": [{"criterion_id": "handled", "text": "The ticket was handled correctly."}],
            "evidence_refs": ["point", f"rule:{RULE.rule_id}"],
        },
        "grounding": grounding,
        "limitations": ["Simulated tickets only."],
    }


def _scenario(scenario_id: str, severity: str, *, lock: str | None = None) -> JsonObject:
    """One scenario answer setting the tier, and optionally the lock."""
    edits: list[JsonObject] = [{"path": "/tickets/severity", "value": severity}]
    if lock is not None:
        edits.append({"path": "/tickets/lock", "value": lock})
    return {
        "scenario_id": scenario_id,
        "edits": edits,
        "correct_option": "escalate" if severity == "high" else "close",
        "deciding_paths": ["/tickets/severity"],
        "witness": [{"tool": "check_sla", "arguments": {}, "reveals": ["/tickets/severity"]}],
        "rationale": f"A {severity} tier ticket needs the matching handling.",
        "criteria": ["handled"],
    }


def _roles(*, held: bool) -> ScriptedJsonTransport:
    """Judge, critic, decision-state author and scenario author replies, in call order.

    The authors answer twice: once for the unfrozen registration that builds each variant's start
    and once for the registration bound to the world freezes of those starts.
    """
    estimate = {"probability": 0.6, "rationale": "Base rate.", "evidence_refs": ["group"]}
    scenarios = [_scenario("urgent", "high"), _scenario("routine", "low")]
    if held:
        scenarios.append(_scenario("held", "low", lock="held"))
    authoring = (
        _tool_reply(DECISION_STATE_TOOL, _decision_answer(lock_axis=held)),
        _tool_reply(SCENARIO_TOOL, {"scenarios": scenarios, "limitations": ["Tier variants."]}),
    )
    return ScriptedJsonTransport(
        (
            _tool_reply(
                "choose_points",
                {
                    "points": [_point(0), _point(1)],
                    "audit": [
                        {
                            "category": "guess_required",
                            "explanation": "The tier is never read in some attempts.",
                            "evidence_refs": ["group"],
                        }
                    ],
                    "limitations": ["Simulated grades only."],
                },
            ),
            _tool_reply(
                "estimate_points",
                {
                    "estimates": [{"step": 0, **estimate}, {"step": 1, **estimate}],
                    "limitations": ["Simulated grades only."],
                },
            ),
            *authoring,
            *authoring,
        )
    )


@dataclass
class _Run:
    """One grounded project with a saved actor-critic run and a registered family.

    Attributes:
        fixture: The grounded world, its job and transports.
        roles: The role transport.
        report: The actor-critic run.
        exposure: The frozen data roles.
        unfrozen: The family run before any variant was frozen, which built each start.
        spec: The family registration bound to the world freezes of those starts.
    """

    fixture: GroundedFixture
    roles: JudgeRouter
    report: ActorCriticReport
    exposure: ExposurePlan
    unfrozen: FamilyReport
    spec: DecisionFamilySpec

    @property
    def project(self) -> ProjectStore:
        """The fixture project."""
        return self.fixture.project

    def family(
        self, spec: DecisionFamilySpec | None = None, answer: Answer | None = None
    ) -> FamilyReport:
        """Run or replay the family at point 1, grading with a scripted decision client."""
        decider = _Decider(answer or (lambda question, text: 0.9))
        return run_decision_family(
            self.project,
            spec or self.spec,
            exposure=self.exposure,
            report=self.report,
            point=1,
            code_revision="test",
            grader=DecisionsTraceTagger(decider, _SNAPSHOT),
        )

    def disposed(self, report: FamilyReport) -> DecisionFamilySpec:
        """Return the spec with a reviewer disposition for every finding of a report."""
        return self.spec.model_copy(
            update={
                "dispositions": tuple(
                    AuditDisposition(
                        finding_id=finding.finding_id,
                        resolution="accepted_with_reason",
                        rationale="Reviewed against the saved grounded trajectories.",
                        resolver="fixture-reviewer",
                    )
                    for finding in report.findings
                )
            }
        )


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, first: str, held: bool = False
) -> _Run:
    """Run the actor-critic stage for real over the grounded world and register a family."""
    fixture = grounded_fixture(tmp_path, monkeypatch, first=first)
    roles = JudgeRouter(_roles(held=held), state_judge(("tickets", "outcome"), "right"))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: roles
    )
    project, job = fixture.project, fixture.job
    exposure = _exposure(project, job, "mining_probes")
    report = run_actor_critic(project, _spec(project, job), exposure=exposure, code_revision="test")
    assert all(row.failure is None for row in report.points)
    scenarios = ("urgent", "routine", "held")
    unbound = unfrozen_judging(project, job.study.checkpoint, job.configuration)
    spec = DecisionFamilySpec(
        decision_author=fixture.settings,
        scenario_author=fixture.settings,
        rules_stage=grounded_rules_stage(project, fixture.settings),
        actions=ACTIONS,
        judging=dict.fromkeys(scenarios, unbound),
        repeats=4,
        confirmation_repeats=4,
        tiers={job.study.actor_model.model_id: "strong"},
        measure_decisions=False,
    )
    unfrozen = run_decision_family(
        project,
        spec,
        exposure=exposure,
        report=report,
        point=1,
        code_revision="test",
        grader=DecisionsTraceTagger(_Decider(lambda question, text: 0.9), _SNAPSHOT),
    )
    starts = {row.scenario_id: row.decision_checkpoint for row in unfrozen.variants}
    judging = {sid: unbound for sid in scenarios}
    for sid, start in starts.items():
        if start is not None:
            judging[sid] = bound_judging(project, start, job.configuration, variant=sid)
    frozen = spec.model_copy(update={"judging": judging})
    return _Run(fixture, roles, report, exposure, unfrozen, frozen)


def _qualification(run: _Run, report: FamilyReport, scenario_id: str) -> ArtifactInput:
    """Qualify the exact executable task of one variant with a feasible run and controls."""
    project = run.project
    task = family_executable_task(report, scenario_id, intent=_INTENT)
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision="test"
    )
    study = execute_continuation_job(project, task.job, code_revision="test")
    feasibility = continuation_result_reference(project, task.job, code_revision="test")
    verifier = required_verifier(task.job.study)
    independent = TaskVerifier(
        description="Authored ticket acceptance with an independent outcome schema.",
        outcome_schema={
            "properties": {
                "final_state": {
                    "type": "object",
                    "properties": {
                        "tickets": {
                            "type": "object",
                            "properties": {"outcome": {"const": "right"}},
                            "required": ["outcome"],
                        }
                    },
                    "required": ["tickets"],
                }
            },
            "required": ["final_state"],
        },
        allowed_tools=verifier.allowed_tools,
        preserved_paths=verifier.preserved_paths,
    )
    specification = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="A handled ticket's outcome is right and its tier and lock are kept.",
            predicate=independent,
            subject="outcome",
        ),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    cases = []
    for partition in ("development", "confirmation"):
        for index, kind in enumerate(
            ("valid-one", "valid-two", "goal-miss", "side-effect", "unknown")
        ):
            outcome = deepcopy(study.branches[0].outcome)
            final = outcome["final_state"]
            assert isinstance(final, dict) and isinstance(final["tickets"], dict)
            outcome["final_answer"] = f"{partition} control {index}"
            if kind == "goal-miss":
                final["tickets"]["outcome"] = "wrong"
            elif kind == "side-effect":
                final["tickets"]["severity"] = "low"
            elif kind == "unknown":
                outcome["steps"] = None
            expected = (
                "pass"
                if kind.startswith("valid")
                else "unassessable"
                if kind == "unknown"
                else "fail"
            )
            witness = record_label_witness(
                project.artifacts,
                specification=specification,
                outcome=outcome,
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            cases.append(
                CalibrationCase(
                    challenge=VerifierChallenge(
                        id=f"{partition}-{kind}",
                        outcome=outcome,
                        expected=expected,
                        authority="synthetic_specification",
                        witness=witness,
                        rationale="Authored observation-level verifier challenge.",
                    ),
                    lineage=f"ticket-control-{partition}-{index}",
                    partition=partition,
                )
            )
    qualification = qualify_executable_task(
        project,
        ExecutableQualificationSpec(task=reference, feasibility=feasibility, cases=tuple(cases)),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert load_executable_qualification(project, qualification).qualified
    return qualification


def _cited(project: ProjectStore, references: tuple[ArtifactInput, ...]) -> set[ArtifactInput]:
    """Return the evidence every saved action contrast of an option contrast cites."""
    return {
        evidence
        for reference in references
        for evidence in ActionContrast.model_validate_json(
            project.artifacts.read_bytes(reference.artifact_id, "report.json")
        ).spec.evidence
    }


def test_a_grounded_family_is_qualified_confirmed_eligible_and_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both tier variants flip and reproduce; dispositions make the family eligible to export."""
    run = _run(tmp_path, monkeypatch, first="read_ticket")
    report = run.family()

    assert report.failure is None, report.failure
    assert report.selection is not None and report.selection.qualifies
    assert report.confirmation is not None and report.confirmation.winners == ("close", "escalate")
    unfrozen = {row.scenario_id: row.decision_checkpoint for row in run.unfrozen.variants}
    for row in report.variants:
        assert row.variant.replay_turns == 0 and row.replay is None and row.stopped is None
        assert row.frozen is not None and row.frozen == unfrozen[row.scenario_id]
        assert row.witness is not None and row.witness.holds
        assert row.selection is not None and row.confirmation is not None
        assert row.witness_artifact in _cited(run.project, row.selection.contrasts)
        assert set(row.selection.contrasts) <= _cited(run.project, row.confirmation.contrasts)
    assert set(report.grades) == {"urgent", "routine"} and report.ambiguous_criteria == ()
    assert [c.name for c in report.eligibility.conditions if not c.holds] == ["audit_dispositions"]
    assert report.records is None and len(run.roles.requests) == 6

    qualification = _qualification(run, report, "urgent")
    diagnostic = export_family_harbor(
        run.project,
        report,
        "urgent",
        tmp_path / "diagnostic",
        name="fixture/ticket-diagnostic",
        intent=_INTENT,
        code_revision="test",
        qualification=qualification,
    )
    assert verify_harbor_export(diagnostic)["qualification"] == "diagnostic_only"
    exported = verify_executable_package(run.project, diagnostic)

    eligible = run.family(run.disposed(report))

    assert eligible.eligibility.eligible, eligible.eligibility.reasons
    assert isinstance(eligible.records, TrainingRecords)
    groups = eligible.records.groups["strong"]
    assert sorted((group.scenario_id, group.option_id) for group in groups) == [
        ("routine", "close"),
        ("routine", "escalate"),
        ("urgent", "close"),
        ("urgent", "escalate"),
    ]
    assert all(group.episodes and None not in group.advantages for group in groups)
    assert {record.point for record in eligible.records.critics} == {0, 1}
    assert len(run.roles.requests) == 6
    qualified = export_family_harbor(
        run.project,
        eligible,
        "urgent",
        tmp_path / "qualified",
        name="fixture/ticket",
        intent=_INTENT,
        code_revision="test",
        qualification=qualification,
    )
    binding = verify_harbor_export(qualified)
    assert binding["qualification"] == "qualified_observable_contract"
    assert run.spec.judging is not None
    assert binding["episode_judge"] == run.spec.judging["urgent"].judge.identity
    assert "rubric" not in json.dumps(binding)
    assert binding["qualification_evidence"] == qualification.model_dump(mode="json")
    assert verify_executable_package(run.project, qualified) == exported
    unqualified = export_family_harbor(
        run.project,
        eligible,
        "routine",
        tmp_path / "routine",
        name="fixture/ticket-routine",
        intent=_INTENT,
        code_revision="test",
    )
    assert verify_harbor_export(unqualified)["qualification"] == "diagnostic_only"


def test_variants_forked_before_the_decision_replay_to_it_or_diverge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``check_sla`` revealed the tier, so variants replay it; a held ticket's replay diverges.

    The replayed decisions are then frozen, and the frozen family runs from them without
    replaying again; the held ticket has no frozen start and diverges again.
    """
    run = _run(tmp_path, monkeypatch, first="check_sla", held=True)
    ambiguous = run.family(answer=_evidence(0.9, 0.9, draw=0.1))

    rows = {row.scenario_id: row for row in run.unfrozen.variants}
    frozen = {row.scenario_id: row for row in ambiguous.variants}
    for scenario_id in ("urgent", "routine"):
        row = rows[scenario_id]
        assert (row.variant.fork_turn, row.variant.replay_turns) == (0, 1)
        assert row.replay is not None and row.replay.status == "replayed"
        assert row.decision_checkpoint == row.replay.checkpoint != row.variant.reference
        assert row.selection is not None and row.selection.checkpoint == row.decision_checkpoint
        assert row.confirmation is not None
        start = frozen[scenario_id]
        assert start.frozen == row.replay.checkpoint and start.replay is None
        assert start.selection is not None and start.selection.checkpoint == start.frozen
    held = frozen["held"]
    assert held.replay is not None and held.replay.status == "replay_diverged"
    assert held.stopped is not None and held.stopped.startswith("replay_diverged: ")
    assert held.selection is None and held.decision_checkpoint is None
    assert ambiguous.selection is not None and ambiguous.selection.qualifies
    assert ambiguous.confirmation is not None and ambiguous.confirmation.qualifies
    assert ambiguous.ambiguous_criteria == ("handled",)
    assert [c.name for c in ambiguous.eligibility.conditions if not c.holds] == [
        "audit_dispositions",
    ]
    assert ambiguous.records is None

    eligible = run.family(run.disposed(ambiguous))

    assert eligible.eligibility.eligible, eligible.eligibility.reasons
    assert isinstance(eligible.records, TrainingRecords)
    assert [row.frozen for row in eligible.variants] == [row.frozen for row in ambiguous.variants]
    assert len(run.roles.requests) == 6
    replayed = rows["routine"].replay
    assert replayed is not None and replayed.calls[0].observation is not None
    assert json.loads(replayed.calls[0].observation.content) == {"lock": "none", "severity": "low"}

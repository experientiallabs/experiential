"""Decision-family orchestration over a real actor-critic run on the native workbook, offline.

The actor-critic run, the before-state probe and both authors run for real over the workbook
fixture with scripted transports. The workbook world is native and cannot carry pointer edits, so
these tests replace the per-variant stage owners with deterministic stand-ins that record every
dispatch, to exercise orchestration, replay of saved reports and dispositions cheaply. The same
pipeline runs with no stand-in over a grounded world in
:mod:`~exp.optimize.evaluation.variance.tests.family_end_to_end_test`.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, Sha256
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RULE_BOOK_TYPE,
    RulesStageReport,
    RulesStageSpec,
    rulebook_files,
)
from exp.optimize.evaluation.variance import family as family_module
from exp.optimize.evaluation.variance import family_search as family_search_module
from exp.optimize.evaluation.variance import family_stages as stages_module
from exp.optimize.evaluation.variance import variant_search as search_module
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, run_actor_critic
from exp.optimize.evaluation.variance.actor_critic_test import (
    _done,
    _edit,
    _exposure,
    _point,
    _spec,
    _tool_reply,
    _wire,
)
from exp.optimize.evaluation.variance.admission import VariantJudging
from exp.optimize.evaluation.variance.admission_test import bound_judging, unfrozen_judging
from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL, SCENARIO_TOOL
from exp.optimize.evaluation.variance.bounded import run_keyed
from exp.optimize.evaluation.variance.briefing_review import DEFAULT_REVIEW_VARIANTS, REVIEW_TOOL
from exp.optimize.evaluation.variance.contracts import (
    DecisionState,
    Scenario,
    ScenarioSet,
    StatePart,
)
from exp.optimize.evaluation.variance.contrast import OptionContrast, WitnessResult
from exp.optimize.evaluation.variance.contrast_test import _outcome
from exp.optimize.evaluation.variance.decision_measure import (
    MEASURE_TYPE,
    DecisionMeasure,
    DecisionMeasureSpec,
    JointVerdict,
    StateEvidence,
    StateMeasure,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import (
    DecisionFamilySpec,
    FamilyReport,
    export_family_harbor,
    family_decisions,
    run_decision_family,
)
from exp.optimize.evaluation.variance.grader import EpisodeView
from exp.optimize.evaluation.variance.grader_test import _SNAPSHOT, _Decider, _evidence
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgeCoverage
from exp.optimize.evaluation.variance.judged_test import JudgeRouter
from exp.optimize.evaluation.variance.nudge_ladder import (
    LadderRound,
    NudgeLadder,
    NudgeLadderSpec,
    NudgeRung,
)
from exp.optimize.evaluation.variance.nudges import (
    NudgeReview,
    ProposedNudge,
    ladder_decision,
    rung_scenario,
)
from exp.optimize.evaluation.variance.records import (
    AuditDisposition,
    TrainingRecords,
    audit_finding,
    confirmation_job,
)
from exp.optimize.evaluation.variance.rewards import PointDecision
from exp.optimize.evaluation.variance.variant_search import (
    LadderMeasurer,
    Measurer,
    SearchRound,
    VariantSearch,
    VariantSearchSpec,
)
from exp.optimize.evaluation.variance.variants import CoherenceCheck, VariantCheckpoint
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.rules.model import Rule, RuleBook, ToolContract

_TOOL = workbook_case().first.name
_RULE = Rule(
    rule_id="orders.rows",
    tool=_TOOL,
    kind="effect",
    statement="Deleting a row removes it from the Orders sheet.",
    status="supported",
    support=2,
    contradiction=0,
)


def _decision_answer() -> JsonObject:
    """A decision-state answer for point 0, grounded in the probed rows and one rule."""
    return {
        "state": {
            "step": 0,
            "decision": "How many order rows to delete.",
            "options": [
                {"option_id": "delete-1", "action": "Delete one order row"},
                {"option_id": "delete-2", "action": "Delete two order rows"},
            ],
            "relevant_paths": ["/rows"],
            "axes": [{"path": "/rows", "values": [None, 6, 7], "effect": "row count decides"}],
            "criteria": [{"criterion_id": "rows", "text": "The right rows were deleted."}],
            "audit": [
                {
                    "category": "toolset_or_harness_gap",
                    "explanation": "Only one tool can read the sheet.",
                    "evidence_refs": ["point"],
                }
            ],
            "evidence_refs": ["point", f"rule:{_RULE.rule_id}"],
        },
        "grounding": [
            {"path": "/rows", "revealed_by": [_TOOL], "refs": ["state:/rows", "prefix:0"]}
        ],
        "limitations": ["Simulated rows only."],
    }


def _scenario(scenario_id: str, rows: int, option: str) -> JsonObject:
    """One scenario editing the row count, revealed by one workbook call."""
    return {
        "scenario_id": scenario_id,
        "edits": [{"path": "/rows", "value": rows}],
        "correct_option": option,
        "deciding_paths": ["/rows"],
        "witness": [
            {"tool": _TOOL, "arguments": workbook_case().first.arguments, "reveals": ["/rows"]}
        ],
        "rationale": f"With {rows} rows, {option} is right.",
        "criteria": ["rows"],
    }


def _roles(scenarios: list[JsonObject] | None = None) -> list[JsonHttpResponse]:
    """Judge, critic, decision-state author and scenario author replies, in call order."""
    estimate = {"probability": 0.5, "rationale": "Base rate.", "evidence_refs": ["group"]}
    return [
        _tool_reply(
            "choose_points",
            {
                "points": [_point(0), _point(1)],
                "audit": [
                    {
                        "category": "guess_required",
                        "explanation": "The target row is never shown.",
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
        _tool_reply(DECISION_STATE_TOOL, _decision_answer()),
        _tool_reply(
            SCENARIO_TOOL,
            {
                "scenarios": scenarios
                or [_scenario("six", 6, "delete-1"), _scenario("seven", 7, "delete-2")],
                "limitations": ["Two variants."],
            },
        ),
        # Every fresh briefing is leak-reviewed under each reading stance; these reviews are clean.
        *(
            _tool_reply(REVIEW_TOOL, {"verdict": "clean", "limitations": ["Fixture review."]})
            for scenario in scenarios or ()
            if scenario.get("initialization") == "fresh_briefing"
            for _ in DEFAULT_REVIEW_VARIANTS
        ),
    ]


def _rules_stage(project: ProjectStore, settings: AuthorSettings) -> ArtifactInput:
    """Save a rule book with a contract for the workbook tool and a stage report naming it."""
    book = RuleBook(contracts={_TOOL: ToolContract(name=_TOOL)}, rules=(_RULE,))
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision="test"
    )
    rulebook = artifact_input(
        project.artifacts.write(
            artifact_id="rule-book-fixture",
            artifact_type=RULE_BOOK_TYPE,
            envelope=envelope,
            files=rulebook_files(book),
        )
    )
    report = RulesStageReport(
        spec=RulesStageSpec(author=settings),
        identity="rules-stage-fixture",
        extraction_identity="rules-extraction-fixture",
        contracts=dict(book.contracts),
        draws=(),
        failures=(),
        rulebook=rulebook,
        rules=1,
        rule_counts={"supported": 1, "contradicted": 0, "unknown": 0},
        open_questions=0,
        diagnostic=(),
        requires_witness=(),
        sessions=(),
        session_ids=(),
        properties=None,
        workloads=(),
        triage=(),
        families_must_stay_diagnostic=(),
    )
    return artifact_input(
        project.artifacts.write_json(
            artifact_id="rules-stage-fixture-result",
            artifact_type=REPORT_TYPE,
            envelope=envelope.model_copy(update={"inputs": (rulebook,)}),
            files={"report.json": report},
        )
    )


@dataclass
class _Stages:
    """Stand-ins for the executed owners, keyed like them so a repeated call replays.

    Attributes:
        report: The actor-critic run whose point checkpoints serve as variant checkpoints.
        confirm: Successes per option for each scenario's fresh confirmation.
        dispatches: Every first-time stage execution, by stage name.
        witnesses: Witnesses already produced, by scenario.
        contrasts: Contrasts already produced, by stage key.
        judges: Judge identity each contrast was labelled by, by stage key.
        required: State parts each variant build was asked to hold, by scenario.
    """

    report: ActorCriticReport
    confirm: dict[str, tuple[int, int]]
    dispatches: list[str] = field(default_factory=list)
    witnesses: dict[str, WitnessResult] = field(default_factory=dict)
    contrasts: dict[str, OptionContrast] = field(default_factory=dict)
    judges: dict[str, str] = field(default_factory=dict)
    required: dict[str, frozenset[StatePart]] = field(default_factory=dict)

    def checkpoint(self, scenario_id: str) -> ArtifactInput:
        """Return the distinct, loadable checkpoint standing in for one variant."""
        return self.report.points[0 if scenario_id == "six" else 1].checkpoint

    def build(
        self,
        project: ProjectStore,
        *,
        base: ArtifactInput,
        branch_checkpoints: tuple[ArtifactInput, ...],
        decision: DecisionState,
        scenario: Scenario,
        facts: InstanceFacts | None,
        code_revision: str,
        maximum_model_calls: int | None = None,
        required_state: frozenset[StatePart] = frozenset(),
    ) -> VariantCheckpoint:
        """Return a persisted variant with no fork before the decision; seven needs review.

        Args:
            project: Unused project.
            base: The decision's checkpoint, which must be the trajectory's boundary.
            branch_checkpoints: The picked attempt's boundaries.
            decision: Unused decision state.
            scenario: The scenario realized.
            facts: Unused instance facts.
            code_revision: Unused revision.
            maximum_model_calls: Unused allowance.
            required_state: Recorded state parts the variant must hold.

        Returns:
            The stand-in variant.
        """
        del project, decision, facts, code_revision, maximum_model_calls
        self.required[scenario.scenario_id] = required_state
        assert branch_checkpoints[0] == base
        reference = self.checkpoint(scenario.scenario_id)
        checks = (CoherenceCheck(name="trajectory", status="passed", detail="one boundary"),)
        if scenario.scenario_id == "seven":
            checks += (
                CoherenceCheck(name="clock_and_events", status="review", detail="pending event"),
            )
        return VariantCheckpoint(
            scenario_id=scenario.scenario_id,
            base=base,
            reference=reference,
            fork=base,
            fork_turn=0,
            revealing_turn=None,
            replay_turns=0,
            edited_state_sha256="0" * 64,
            checks=checks,
            problems=(),
        )

    def witness(
        self,
        project: ProjectStore,
        *,
        checkpoint: ArtifactInput,
        scenario: Scenario,
        decision: DecisionState,
        job: ContinuationJob,
        code_revision: str,
    ) -> WitnessResult:
        """Return a holding witness, dispatched once per scenario.

        Args:
            project: Unused project.
            checkpoint: The variant checkpoint.
            scenario: The scenario.
            decision: Unused decision state.
            job: Unused job.
            code_revision: Unused revision.

        Returns:
            The stand-in witness.
        """
        del project, decision, job, code_revision
        if scenario.scenario_id not in self.witnesses:
            self.dispatches.append(f"witness:{scenario.scenario_id}")
            self.witnesses[scenario.scenario_id] = WitnessResult(
                scenario=scenario,
                checkpoint=checkpoint,
                correct_action="Delete rows",
                state={"rows": 6},
                calls=(),
                remaining_calls=4,
                paths=(),
            )
        return self.witnesses[scenario.scenario_id]

    def cited(
        self,
        project: ProjectStore,
        *,
        checkpoint: ArtifactInput,
        scenario: Scenario,
        job: ContinuationJob,
        code_revision: str,
    ) -> ArtifactInput:
        """Cite the stand-in variant's own saved checkpoint as its witness artifact.

        Args:
            project: Unused project.
            checkpoint: The variant checkpoint.
            scenario: The scenario, which must have a stand-in witness.
            job: Unused job.
            code_revision: Unused revision.

        Returns:
            A saved artifact reference.
        """
        del project, job, code_revision
        assert scenario.scenario_id in self.witnesses
        return checkpoint

    def contrast(
        self,
        project: ProjectStore,
        *,
        checkpoint: ArtifactInput,
        decision: DecisionState,
        actions: Mapping[str, ToolCall],
        job: ContinuationJob,
        repeats: int,
        judge: EpisodeJudge,
        code_revision: str,
        evidence: tuple[ArtifactInput, ...] = (),
    ) -> OptionContrast:
        """Return a contrast whose winner follows the scenario, or ``confirm`` when fresh.

        Args:
            project: Unused project.
            checkpoint: The variant checkpoint, which names the scenario.
            decision: The decision state.
            actions: The option mapping, which must cover every option.
            job: The frozen job; its selector marks a fresh confirmation.
            repeats: Pairs per option.
            judge: The variant's frozen judge, recorded per stage key.
            code_revision: Unused revision.
            evidence: Unused evidence.

        Returns:
            The stand-in contrast, its judge coverage counting every planned cell.
        """
        del project, code_revision, evidence
        assert set(actions) == {option.option_id for option in decision.options}
        scenario = "six" if checkpoint == self.checkpoint("six") else "seven"
        fresh = "fresh confirmation" in job.study.selector
        key = f"{'confirmation' if fresh else 'selection'}:{scenario}"
        if key not in self.contrasts:
            self.dispatches.append(key)
            one, two = (repeats, 0) if scenario == "six" else (0, repeats)
            if fresh:
                one, two = self.confirm.get(scenario, (one, two))
            self.judges[key] = judge.identity
            self.contrasts[key] = OptionContrast(
                step=decision.step,
                decision=decision.decision,
                checkpoint=checkpoint,
                reference_option="delete-1",
                repeats=repeats,
                options=tuple(
                    _outcome(option, wins, repeats).model_copy(
                        update={
                            "coverage": JudgeCoverage(
                                judge=judge.identity,
                                planned=repeats,
                                passed=wins,
                                failed=repeats - wins,
                            )
                        }
                    )
                    for option, wins in (("delete-1", one), ("delete-2", two))
                ),
                judge=judge.identity,
            )
        return self.contrasts[key]


def _restores(project: ProjectStore, reference: ArtifactInput, **_: Sha256 | None) -> bool:
    """Accept every stand-in variant as restoring to its edited state."""
    del project, reference
    return True


def _episodes(
    project: ProjectStore, scenario_id: str, contrast: OptionContrast
) -> tuple[EpisodeView, ...]:
    """Render two stand-in confirmation episodes per variant."""
    del project, contrast
    return tuple(
        EpisodeView(episode_id=f"{scenario_id}-{n}", text=f"{scenario_id} episode {n}")
        for n in range(2)
    )


@dataclass
class _Setup:
    """One project with a saved actor-critic run and everything a family needs.

    Attributes:
        project: The project.
        spec: The family registration.
        report: The actor-critic run.
        exposure: The frozen data roles.
        roles: The role transport, shared by judges, critics and authors.
        stages: Stand-in executed owners.
        decider: Decision boundary answering the rubric grader.
        job: The fixture job.
    """

    project: ProjectStore
    spec: DecisionFamilySpec
    report: ActorCriticReport
    exposure: ExposurePlan
    roles: JudgeRouter
    stages: _Stages
    decider: _Decider
    job: ContinuationJob

    def run(self, spec: DecisionFamilySpec | None = None) -> FamilyReport:
        """Run or replay the family at point 0 with the option mapping and grader."""
        return run_decision_family(
            self.project,
            spec or self.spec,
            exposure=self.exposure,
            report=self.report,
            point=0,
            code_revision="test",
            actions=_actions,
            grader=DecisionsTraceTagger(self.decider, _SNAPSHOT),
        )


def _actions(decision: DecisionState) -> dict[str, ToolCall]:
    """Map each authored option to deleting that many rows."""
    call = workbook_case().first
    return {
        option.option_id: call.model_copy(
            update={
                "arguments": {
                    **call.arguments,
                    "deletions": [
                        {"sheet": "Orders", "start": 2, "count": int(option.option_id[-1])}
                    ],
                }
            }
        )
        for option in decision.options
    }


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    confirm: dict[str, tuple[int, int]] | None = None,
    scenarios: list[JsonObject] | None = None,
) -> _Setup:
    """Run the actor-critic stage for real and wire stand-ins for the executed owners."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    actor_spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    _, roles = _wire(monkeypatch, group + continuations, _roles(scenarios))
    exposure = _exposure(project, job, "mining_probes")
    report = run_actor_critic(project, actor_spec, exposure=exposure, code_revision="test")
    stages = _Stages(report=report, confirm=confirm or {})
    monkeypatch.setattr(stages_module, "build_variant_checkpoint", stages.build)
    monkeypatch.setattr(stages_module, "verify_restoration", _restores)
    monkeypatch.setattr(stages_module, "run_observation_witness", stages.witness)
    monkeypatch.setattr(stages_module, "witness_reference", stages.cited)
    monkeypatch.setattr(stages_module, "run_option_contrast", stages.contrast)
    monkeypatch.setattr(stages_module, "confirmation_episodes", _episodes)
    settings = actor_spec.judges[0]
    identities = [str(row["scenario_id"]) for row in scenarios or ()] or ["six", "seven"]
    spec = DecisionFamilySpec(
        decision_author=settings,
        scenario_author=settings,
        rules_stage=_rules_stage(project, settings),
        repeats=8,
        confirmation_repeats=8,
        judging=_frozen(project, stages, job, identities),
        tiers={report.spec.initial.study.actor_model.model_id: "strong"},
        measure_decisions=False,
    )
    decider = _Decider(lambda question, text: 0.9)
    return _Setup(project, spec, report, exposure, roles, stages, decider, job)


def _frozen(
    project: ProjectStore,
    stages: _Stages,
    job: ContinuationJob,
    identities: list[str],
    *,
    trainable: bool = True,
) -> dict[str, VariantJudging]:
    """Bind each scenario to a world freeze whose saved start is its stand-in checkpoint."""
    return {
        sid: bound_judging(
            project, stages.checkpoint(sid), job.configuration, variant=sid, trainable=trainable
        )
        for sid in identities
    }


def _disposed(setup: _Setup, report: FamilyReport) -> DecisionFamilySpec:
    """Return the spec with a reviewer disposition for every finding of a report."""
    return setup.spec.model_copy(
        update={
            "dispositions": tuple(
                AuditDisposition(
                    finding_id=finding.finding_id,
                    resolution="accepted_with_reason",
                    rationale="Reviewed against the saved trajectories.",
                    resolver="fixture-reviewer",
                )
                for finding in report.findings
            )
        }
    )


def test_family_qualifies_is_confirmed_and_yields_records_once_disposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both variants flip and reproduce on fresh samples; dispositions make it eligible."""
    setup = _setup(tmp_path, monkeypatch)
    report = setup.run()

    assert report.failure is None and report.decision.answer is not None
    assert report.scenarios is not None and report.scenarios.accepted
    assert report.selection is not None and report.selection.qualifies
    assert report.confirmation is not None and report.confirmation.winners == (
        "delete-1",
        "delete-2",
    )
    # Scenarios run at once, so only each scenario's own order and selection before
    # confirmation are fixed.
    dispatched = setup.stages.dispatches
    assert sorted(dispatched) == sorted(
        [
            "witness:six",
            "selection:six",
            "witness:seven",
            "selection:seven",
            "confirmation:six",
            "confirmation:seven",
        ]
    )
    assert max(dispatched.index(f"selection:{s}") for s in ("six", "seven")) < min(
        dispatched.index(f"confirmation:{s}") for s in ("six", "seven")
    )
    assert all(
        dispatched.index(f"witness:{s}") < dispatched.index(f"selection:{s}")
        for s in ("six", "seven")
    )
    assert set(report.grades) == {"six", "seven"} and report.ambiguous_criteria == ()
    sources = {finding.source for finding in report.findings}
    assert sources == {"actor_critic", "decision_state", "variant_review"}
    assert [c.name for c in report.eligibility.conditions if not c.holds] == ["audit_dispositions"]
    assert report.records is None
    assert len(setup.roles.requests) == 4

    disposed = setup.run(_disposed(setup, report))

    assert disposed.identity != report.identity
    assert disposed.eligibility.eligible, disposed.eligibility.reasons
    assert isinstance(disposed.records, TrainingRecords)
    assert set(disposed.records.groups) == {"strong"}
    assert [(g.scenario_id, g.option_id) for g in disposed.records.groups["strong"]] == [
        ("seven", "delete-1"),
        ("seven", "delete-2"),
        ("six", "delete-1"),
        ("six", "delete-2"),
    ]
    assert setup.spec.judging is not None
    judge = setup.spec.judging["six"].judge.identity
    assert all(g.judge == judge for g in disposed.records.groups["strong"])
    assert set(setup.stages.judges.values()) == {judge}
    assert set(disposed.coverage) == {"six", "seven"}
    assert disposed.judging == setup.spec.judging
    assert {record.point for record in disposed.records.critics} == {0, 1}
    assert not any(record.trainable for record in disposed.records.critics)
    assert all(not judge.trainable for judge in disposed.records.judges)
    assert len(setup.roles.requests) == 4
    assert len(setup.stages.dispatches) == 6


def test_scenarios_and_confirmations_run_at_once_with_the_serial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both scenarios are in their witness at once and both confirm at once; same report."""
    serial_setup = _setup(tmp_path / "serial", monkeypatch)
    monkeypatch.setattr(
        family_module, "run_keyed", lambda calls, keys, _: run_keyed(calls, keys, 1)
    )
    serial = serial_setup.run()
    monkeypatch.undo()
    setup = _setup(tmp_path / "parallel", monkeypatch)
    witnessed, contrasted = setup.stages.witness, setup.stages.contrast
    barriers = {
        "witness": threading.Barrier(2, timeout=30),
        "confirm": threading.Barrier(2, timeout=30),
    }

    def witness(*args: object, **kwargs: object) -> object:
        barriers["witness"].wait()  # both scenarios must be in flight together
        return witnessed(*args, **kwargs)

    def contrast(*args: object, **kwargs: object) -> object:
        job = cast(ContinuationJob, kwargs["job"])
        if "fresh confirmation" in job.study.selector:
            barriers["confirm"].wait()  # both winners must confirm together
        return contrasted(*args, **kwargs)

    monkeypatch.setattr(stages_module, "run_observation_witness", witness)
    monkeypatch.setattr(stages_module, "run_option_contrast", contrast)
    report = setup.run()
    assert [row.scenario_id for row in report.variants] == [
        row.scenario_id for row in serial.variants
    ]
    assert report.selection is not None and serial.selection is not None
    assert report.selection.qualifies and serial.selection.qualifies
    assert [(v.scenario_id, v.status) for v in report.selection.variants] == [
        (v.scenario_id, v.status) for v in serial.selection.variants
    ]
    assert report.confirmation is not None and serial.confirmation is not None
    assert report.confirmation.winners == serial.confirmation.winners
    assert sorted(setup.stages.dispatches) == sorted(serial_setup.stages.dispatches)


def test_family_that_qualifies_on_selection_but_not_on_fresh_samples_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A winner that does not separate again on fresh samples withdraws qualification."""
    setup = _setup(tmp_path, monkeypatch, confirm={"seven": (4, 4)})
    first = setup.run()
    report = setup.run(_disposed(setup, first))

    assert report.selection is not None and report.selection.qualifies
    assert report.confirmation is not None and not report.confirmation.qualifies
    seven = next(v for v in report.confirmation.variants if v.scenario_id == "seven")
    assert seven.status == "unseparated"
    assert [c.name for c in report.eligibility.conditions if not c.holds] == ["fresh_confirmation"]
    assert any("seven" in reason for reason in report.eligibility.reasons)
    assert report.records is None


def test_pending_audit_findings_block_eligibility_until_each_is_disposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disposing every finding but one still leaves the family ineligible, naming the count."""
    setup = _setup(tmp_path, monkeypatch)
    report = setup.run()
    flag = audit_finding("actor_critic", setup.report.audit[0])
    assert flag in report.findings
    partial = _disposed(setup, report)
    partial = partial.model_copy(
        update={
            "dispositions": tuple(
                d for d in partial.dispositions if d.finding_id != flag.finding_id
            )
        }
    )

    blocked = setup.run(partial)

    assert not blocked.eligibility.eligible and blocked.records is None
    assert blocked.eligibility.reasons == (
        f"audit_dispositions: 1 of {len(report.findings)} audit findings have no disposition",
    )


def test_a_saved_family_replays_without_any_call_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second run returns the saved report before any draw, probe or stage dispatch."""
    setup = _setup(tmp_path, monkeypatch)
    report = setup.run()
    requests, dispatches, asked = (
        len(setup.roles.requests),
        list(setup.stages.dispatches),
        len(setup.decider.requests),
    )
    setup.job.runtime_file.unlink()

    assert setup.run() == report
    assert len(setup.roles.requests) == requests
    assert setup.stages.dispatches == dispatches
    assert len(setup.decider.requests) == asked


def test_option_actions_must_be_given_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither or both mapping strategies refuse before any draw."""
    setup = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly once"):
        run_decision_family(
            setup.project,
            setup.spec,
            exposure=setup.exposure,
            report=setup.report,
            point=0,
            code_revision="test",
        )
    assert len(setup.roles.requests) == 2


def test_confirmation_job_redraws_only_by_its_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh confirmation keeps the frozen policy and differs only in the selector."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    del project
    fresh = confirmation_job(job, 3)
    assert fresh.study.selector.endswith("fresh confirmation seed 3")
    assert fresh.model_copy(update={"study": job.study}) == job
    assert confirmation_job(job, 3) == fresh and confirmation_job(job, 4) != fresh


def test_harbor_export_refuses_a_variant_the_family_did_not_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only persisted variants can be packaged."""
    setup = _setup(tmp_path, monkeypatch)
    report = setup.run()
    with pytest.raises(ValueError, match="no persisted decision checkpoint"):
        export_family_harbor(
            setup.project,
            report,
            "missing",
            tmp_path / "package",
            name="fixture/family",
            intent="Choose the deletion the row count calls for.",
            code_revision="test",
        )


def test_uncertain_calibration_criteria_are_diagnostic_and_never_block_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grading draws that straddle the band are recorded but do not decide eligibility."""
    setup = _setup(tmp_path, monkeypatch)
    setup.decider = _Decider(_evidence(0.9, 0.9, draw=0.1))
    first = setup.run()
    report = setup.run(_disposed(setup, first))

    assert report.confirmation is not None and report.confirmation.qualifies
    assert report.ambiguous_criteria == ("rows",)
    assert report.eligibility.eligible, report.eligibility.reasons


def test_an_untrainable_world_freeze_and_a_missing_review_are_each_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admission policy names every failed check for each confirmed variant."""
    setup = _setup(tmp_path, monkeypatch)
    judging = _frozen(setup.project, setup.stages, setup.job, ["six", "seven"], trainable=False)
    judging["seven"] = judging["seven"].model_copy(update={"compatibility": None})
    setup.spec = setup.spec.model_copy(update={"judging": judging})
    first = setup.run()
    report = setup.run(_disposed(setup, first))

    failed = [c.name for c in report.eligibility.conditions if not c.holds]
    assert failed == ["rubric_compatibility", "world_freeze"]
    assert any("seven: no compatibility review" in reason for reason in report.eligibility.reasons)
    assert any("six: the world freeze is not trainable" in r for r in report.eligibility.reasons)
    assert report.records is None


def test_frozen_variants_are_what_contrasts_run_from_and_unfrozen_ones_fail_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound freeze's saved start runs every contrast; a variant without one is recorded."""
    setup = _setup(tmp_path, monkeypatch)
    assert setup.spec.judging is not None
    judging = dict(setup.spec.judging)
    judging["seven"] = unfrozen_judging(
        setup.project, setup.stages.checkpoint("seven"), setup.job.configuration
    )
    setup.spec = setup.spec.model_copy(update={"judging": judging})
    first = setup.run()
    report = setup.run(_disposed(setup, first))

    rows = {row.scenario_id: row for row in report.variants}
    frozen = setup.stages.checkpoint("six")
    assert rows["six"].frozen == frozen == rows["six"].decision_checkpoint
    for contrast in (rows["six"].selection, rows["six"].confirmation):
        assert contrast is not None and contrast.checkpoint == frozen
    assert rows["seven"].frozen is None and rows["seven"].confirmation is not None
    failed = [c.name for c in report.eligibility.conditions if not c.holds]
    assert failed == ["world_freeze", "world_provenance"]
    assert any(
        "seven: no world-freeze report binds this variant" in reason
        for reason in report.eligibility.reasons
    )
    assert report.records is None


def test_variant_judges_must_be_given_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family without a frozen judge binding never falls back to mechanical labels."""
    setup = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="variant judges"):
        run_decision_family(
            setup.project,
            setup.spec.model_copy(update={"judging": None}),
            exposure=setup.exposure,
            report=setup.report,
            point=0,
            code_revision="test",
            actions=_actions,
        )


def test_fresh_briefing_scenarios_flow_through_the_family_with_required_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With fresh briefings enabled the author's view admits a briefed scenario; else refused."""
    briefed = {
        **_scenario("seven", 7, "delete-2"),
        "initialization": "fresh_briefing",
        "briefing": {"text": "Pick up the workbook task from where it stands."},
    }
    scenarios = [_scenario("six", 6, "delete-1"), briefed]
    setup = _setup(tmp_path, monkeypatch, scenarios=scenarios)
    spec = setup.spec.model_copy(update={"fresh_briefing": True, "required_state": ("files",)})

    report = setup.run(spec)

    assert report.scenarios is not None and report.scenarios.accepted, report.scenarios
    assert report.scenarios.scenarios is not None
    starts = [s.initialization for s in report.scenarios.scenarios.scenarios]
    assert starts == ["resume", "fresh_briefing"]
    assert setup.stages.required == {"six": frozenset({"files"}), "seven": frozenset({"files"})}
    assert spec.draws_identity(setup.report, 0, "test") != setup.spec.draws_identity(
        setup.report, 0, "test"
    )


def test_a_fresh_briefing_without_the_actor_view_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spec that does not enable fresh briefings gives the checks no view, so they refuse."""
    briefed = {
        **_scenario("seven", 7, "delete-2"),
        "initialization": "fresh_briefing",
        "briefing": {"text": "Pick up the workbook task from where it stands."},
    }
    setup = _setup(tmp_path, monkeypatch, scenarios=[_scenario("six", 6, "delete-1"), briefed])

    report = setup.run()

    assert report.scenarios is not None and not report.scenarios.accepted
    assert any("actor's view" in p.reason for p in report.scenarios.problems)
    assert report.failure is not None and report.failure.startswith("scenarios:")
    assert setup.stages.required == {}


def _verdict(admitted: bool) -> JointVerdict:
    """A joint verdict that admits with score 0.5 or refuses for lack of a choice shift."""
    return JointVerdict(
        states=(),
        choice_shift=admitted,
        consequence=admitted,
        clarity=admitted,
        flip=admitted,
        admitted=admitted,
        score=0.5 if admitted else 0.0,
        deficits=() if admitted else ("choices do not change across states",),
    )


def _fake_measure(admitted: bool, seen: list) -> object:
    """Return a stand-in for the joint measure that records its settings and answers a verdict."""
    reference = ArtifactInput(artifact_id="fixture-measure", sha256="5" * 64)

    def measure(project: ProjectStore, spec: DecisionMeasureSpec, **kwargs: object) -> tuple:
        """Record the settings and scenarios, then return the scripted verdict."""
        scenarios = kwargs["scenarios"]
        assert isinstance(scenarios, tuple)
        seen.append((spec, [s.scenario_id for s in scenarios if isinstance(s, Scenario)]))
        decision = kwargs["decision"]
        assert isinstance(decision, DecisionState)
        result = DecisionMeasure(
            identity="m", decision=decision, states=(), verdict=_verdict(admitted)
        )
        return reference, result, None

    return measure


def test_the_joint_measure_runs_by_default_under_the_family_authors_before_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no settings the measure uses the authors; a refusal blocks confirmation by name."""
    setup = _setup(tmp_path, monkeypatch)
    seen: list = []
    monkeypatch.setattr(family_module, "measure_family", _fake_measure(False, seen))
    default = setup.spec.model_copy(update={"measure_decisions": True})

    blocked = setup.run(default)

    assert default.measure_spec == DecisionMeasureSpec(
        classifier=default.scenario_author, reviewer=default.decision_author
    )
    assert seen == [(default.measure_spec, ["six", "seven"])]
    assert blocked.selection is not None and blocked.selection.qualifies
    assert blocked.confirmation is None and blocked.records is None
    assert isinstance(blocked.decision_variance, ArtifactInput)
    reasons = blocked.eligibility.reasons
    assert "decision_variance: choices do not change across states" in reasons
    assert any(r.startswith("fresh_confirmation: not run: the decision_variance") for r in reasons)
    assert "confirmation:six" not in setup.stages.dispatches
    assert default.draws_identity(setup.report, 0, "test") == setup.spec.draws_identity(
        setup.report, 0, "test"
    )
    assert default.identity(setup.report, 0, "test") != setup.spec.identity(setup.report, 0, "test")

    monkeypatch.setattr(family_module, "measure_family", _fake_measure(True, seen))
    admitted = setup.run(
        default.model_copy(update={"dispositions": _disposed(setup, blocked).dispositions})
    )
    condition = next(c for c in admitted.eligibility.conditions if c.name == "decision_variance")
    assert condition.holds and condition.detail == "admitted, joint score 0.500"
    assert admitted.confirmation is not None and "confirmation:six" in setup.stages.dispatches
    assert admitted.eligibility.eligible, admitted.eligibility.reasons
    assert isinstance(admitted.records, TrainingRecords) and admitted.records.selectors
    assert admitted.records.selectors.objective.version == "selector-decision-variance-v2"
    assert any(
        draw.score and draw.score.variances == {"point-0": 0.5}
        for draw in admitted.records.selectors.draws
    )
    assert blocked.decision.answer is not None
    measure = DecisionMeasure(
        identity="m", decision=blocked.decision.answer.state, states=(), verdict=_verdict(True)
    )
    saved = setup.project.artifacts.write_json(
        artifact_id="family-measure",
        artifact_type=MEASURE_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"measure.json": measure},
    )
    measured = admitted.model_copy(update={"decision_variance": artifact_input(saved)})
    assert family_decisions(
        setup.project, [measured, blocked.model_copy(update={"decision_variance": "no states"})]
    ) == {"point-0": PointDecision(point_id="point-0", joint_score=0.5)}


class _Measured:
    """A stand-in joint measure that admits every set and makes ``seven`` the hardest state."""

    def __init__(self) -> None:
        """Start with no measured set."""
        self.sets: list[list[str]] = []

    def __call__(self, project: ProjectStore, spec: DecisionMeasureSpec, **kwargs: object) -> tuple:
        """Save a measure whose states' free choices pick their correct option less for seven."""
        del spec
        scenarios = cast(tuple[Scenario, ...], kwargs["scenarios"])
        decision = kwargs["decision"]
        assert isinstance(decision, DecisionState)
        self.sets.append([s.scenario_id for s in scenarios])
        free = ArtifactInput(artifact_id="free-study", sha256="4" * 64)
        states = tuple(
            StateMeasure(
                state_id=s.scenario_id,
                free=free,
                choices=(),
                reviews=(),
                evidence=StateEvidence(
                    state_id=s.scenario_id,
                    correct_option=s.correct_option,
                    choices=(s.correct_option,) * (1 if s.scenario_id == "seven" else 3)
                    + ("delete-1" if s.correct_option == "delete-2" else "delete-2",) * 2,
                ),
            )
            for s in scenarios
        )
        identity = f"fixture-measure-{len(self.sets)}"
        measure = DecisionMeasure(
            identity=identity, decision=decision, states=states, verdict=_verdict(True)
        )
        saved = project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=MEASURE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"measure.json": measure},
        )
        return artifact_input(saved), measure, None


def test_a_searched_set_is_contrasted_and_an_admitted_family_ladders_its_hardest_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search's admitted set replaces the draw; the ladder's rungs become their own tasks."""
    setup = _setup(tmp_path, monkeypatch)
    project, stages, job = setup.project, setup.stages, setup.job
    measured = _Measured()
    monkeypatch.setattr(family_module, "measure_family", measured)
    monkeypatch.setattr(search_module, "measure_family", measured)
    asked: list[list[str]] = []
    frozen: dict[str, VariantJudging] = {}

    def judges(state: DecisionState, scenarios: Sequence[Scenario]) -> dict[str, VariantJudging]:
        """Bind each set once, the way the pipeline authors one chain per set."""
        del state
        asked.append([s.scenario_id for s in scenarios])
        for s in scenarios:
            if s.scenario_id not in frozen:
                frozen[s.scenario_id] = bound_judging(
                    project,
                    stages.checkpoint(s.scenario_id),
                    job.configuration,
                    variant=s.scenario_id,
                )
        return {s.scenario_id: frozen[s.scenario_id] for s in scenarios}

    found = [Scenario.model_validate(_scenario(n, r, o)) for n, r, o in _SEARCHED]

    def search(
        project: ProjectStore, spec: VariantSearchSpec, *args: object, **kwargs: object
    ) -> tuple[ArtifactInput, VariantSearch]:
        """Measure one admitted round through the family's measurer and save the search."""
        identity = kwargs["identity"]
        measure = cast(Measurer, kwargs["measure"])
        assert isinstance(identity, str)
        chosen = ScenarioSet(scenarios=tuple(found), limitations=("Searched.",))
        result = measure(chosen, 0)
        assert not isinstance(result, str), result
        row = SearchRound(
            round=0,
            stance=spec.stances[0],
            transcript=result[0],
            scenarios=chosen,
            measure=result[0],
            verdict=result[1].verdict,
            accepted=True,
        )
        record = VariantSearch(
            identity=identity, rounds=(row,), accepted_round=0, scenarios=chosen, measure=result[0]
        )
        saved = project.artifacts.write_json(
            artifact_id=identity,
            artifact_type="variant-search",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"search.json": record},
        )
        return artifact_input(saved), record

    def ladder(
        project: ProjectStore,
        spec: NudgeLadderSpec,
        decision: DecisionState,
        base: Scenario,
        **kwargs: object,
    ) -> tuple[ArtifactInput, NudgeLadder]:
        """Measure the base with one passing nudge and admit it as the strong rung."""
        laddered = ladder_decision(decision, base, str(kwargs["instruction"]), ["Check the rows."])
        axis = laddered.instruction_axis
        assert axis is not None
        value = next(v for v in axis.values if v.value_id == "nudge-0")
        rung = rung_scenario(base, value, f"{base.scenario_id}-nudge-0")
        measure = cast(LadderMeasurer, kwargs["measure"])
        result = measure(laddered, ScenarioSet(scenarios=(base, rung), limitations=("Nudged.",)))
        assert not isinstance(result, str), result
        nudge = ProposedNudge(
            strength="strong", text="Check the rows.", source="user", rationale="Points at rows."
        )
        rungs = (
            NudgeRung(
                position=0,
                strength="strong",
                scenario=rung,
                nudge=nudge,
                review=NudgeReview(leak=(), realism=()),
                admitted=True,
            ),
            NudgeRung(position=1, strength="none", scenario=base, admitted=True),
        )
        identity = str(kwargs["identity"])
        record = NudgeLadder(
            identity=identity,
            base=base.scenario_id,
            rounds=(
                LadderRound(
                    round=0,
                    stance=spec.author_stances[0],
                    transcript=result[0],
                    decision=laddered,
                    measure=result[0],
                    rungs=rungs,
                ),
            ),
            chosen_round=0,
            rungs=rungs,
        )
        saved = project.artifacts.write_json(
            artifact_id=identity,
            artifact_type="nudge-ladder",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"ladder.json": record},
        )
        return artifact_input(saved), record

    monkeypatch.setattr(family_search_module, "search_variants", search)
    monkeypatch.setattr(family_search_module, "build_nudge_ladder", ladder)
    settings = setup.spec.decision_author
    spec = setup.spec.model_copy(
        update={
            "judging": None,
            "measure_decisions": True,
            "search": VariantSearchSpec(author=settings),
            "ladder": NudgeLadderSpec(
                author=settings, leak_reviewer=settings, realism_reviewer=settings
            ),
        }
    )

    report = run_decision_family(
        project,
        spec,
        exposure=setup.exposure,
        report=setup.report,
        point=0,
        code_revision="test",
        actions=_actions,
        judging=judges,
    )

    assert report.search is not None and report.scenarios is not None
    assert report.scenarios.accepted and report.scenarios.scenarios is not None
    assert [s.scenario_id for s in report.scenarios.scenarios.scenarios] == ["six", "seven"]
    assert asked[0] == ["six", "seven"] and ["seven", "seven-nudge-0"] in asked
    assert measured.sets == [["six", "seven"], ["six", "seven"], ["seven", "seven-nudge-0"]]
    assert stages.dispatches.count("selection:six") == 1
    assert isinstance(report.ladder, ArtifactInput)
    assert [(r.scenario_id, r.strength, r.position) for r in report.rungs] == [
        ("seven-nudge-0", "strong", 0),
        ("seven", "none", 1),
    ]
    assert report.rungs[0].nudge == "Check the rows." and report.rungs[-1].nudge is None
    assert all(r.variant.selection is not None for r in report.rungs)
    assert report.rungs[0].judging == frozen["seven-nudge-0"]
    assert report.confirmation is not None
    with pytest.raises(ValueError, match="measure_decisions"):
        spec.model_copy(update={"measure_decisions": False}).model_validate(
            {**spec.model_dump(mode="json"), "measure_decisions": False}
        )
    assert setup.spec.draws_identity(setup.report, 0, "test") != spec.draws_identity(
        setup.report, 0, "test"
    )
    unset = spec.model_copy(update={"search": None, "ladder": None, "judging": setup.spec.judging})
    assert unset.draws_identity(setup.report, 0, "test") == setup.spec.model_copy(
        update={"measure_decisions": True}
    ).draws_identity(setup.report, 0, "test")

    again = run_decision_family(
        project,
        spec,
        exposure=setup.exposure,
        report=setup.report,
        point=0,
        code_revision="test",
        actions=_actions,
        judging=judges,
    )
    assert again == report and len(measured.sets) == 3


_SEARCHED = (("six", 6, "delete-1"), ("seven", 7, "delete-2"))
"""The admitted searched set: the same two row counts the scenario author would draw."""

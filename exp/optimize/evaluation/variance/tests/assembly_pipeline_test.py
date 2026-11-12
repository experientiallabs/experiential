"""The assembled pipeline from five actor tiers to qualified Harbor packages, offline.

The measured end-to-end fixture runs every stage through its real owner with scripted
transports, now with the point qualification bar (no fixed point count, the whole trajectory),
blast radius judged across tiers, hybrid dense reward on the training records and the
difficulty curriculum ordering every tier's schedule. The decision model's answers to the bar,
the selectors' justified picks and the severity reviews are explicit fixtures at their seams.
Every qualified export (variants and nudge-ladder rungs) is verified as an executable package
binding its frozen judge and dense reward, and the replay makes no call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.optimize.evaluation.executable.judged_acceptance import load_judged_acceptance
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.blast_radius import SEVERITY_TOOL, BlastSpec
from exp.optimize.evaluation.variance.decision_measure import CHOICE_TOOL
from exp.optimize.evaluation.variance.decision_measure_test import FORCED
from exp.optimize.evaluation.variance.difficulty_curriculum import load_difficulty_curriculum
from exp.optimize.evaluation.variance.pipeline import (
    load_stage_record,
    plan_decision_variance,
    stage_outputs,
)
from exp.optimize.evaluation.variance.pipeline_behaviors_test import MixedTagger
from exp.optimize.evaluation.variance.pipeline_dense import HybridDenseSettings
from exp.optimize.evaluation.variance.pipeline_points import load_blast_ranking
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    FamiliesOutputs,
    MiningOutputs,
    PipelineReport,
    PointQualification,
    RecordsOutputs,
    StageDependencies,
    TraceMLOutputs,
    VariantQualification,
)
from exp.optimize.evaluation.variance.pipeline_test import (
    ANSWERS,
    REVISION,
    TIER_B,
    FixtureBuild,
    Run,
    add_tier,
    dispositions_for,
    fixture_rules_stage,
    mining_rubric,
    rebound,
    selector_answer,
    step_tagger,
    write_bank,
)
from exp.optimize.evaluation.variance.point_bar import DEFAULT_BAR
from exp.optimize.evaluation.variance.point_selection import BarSelection, load_point_bar
from exp.optimize.evaluation.variance.reward_states import PointState
from exp.optimize.evaluation.variance.tests.decision_variance_end_to_end_test import (
    BANK,
    ChoosingActor,
    MeasuredWire,
    choice_answer,
    measured_spec,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTagger
from exp.simulation.mining.views import Unit

TIERS = ("actor", TIER_B, "actor-c", "actor-d", "actor-e")
"""Five actor tiers, weakest first."""


class BarTagger:
    """The decision model at the tagging seam: every step clears the bar; other banks delegate.

    Attributes:
        delegate: The fixture's mixed tagger for every other bank.
        bars: Steps answered against the bar.
    """

    def __init__(self, delegate: TraceTagger) -> None:
        """Wrap the fixture tagger."""
        self.delegate = delegate
        self.bars = 0

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer the bar's meta-rules as holding on every step; delegate other banks."""
        if bank.bank_id != DEFAULT_BAR.bar_id:
            return self.delegate.tag(units, bank)
        self.bars += len(units)
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            view="step",
            question_ids=bank.question_ids(),
            rows=tuple(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(0.95 for _ in bank.question_ids()),
                )
                for unit in units
            ),
        )


def _opening(payload: JsonObject) -> str:
    """Return a request's opening user content."""
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    return str(messages[1]["content"])


def bar_selector_answer(payload: JsonObject) -> JsonObject:
    """The selector's pick, justified against every required rule when the bar is shown."""
    answer = selector_answer()
    if "qualification_bar" not in _opening(payload):
        return answer
    points = answer["points"]
    assert isinstance(points, list)
    required = [rule.rule_id for rule in DEFAULT_BAR.rules if rule.required]
    justifications: list[JsonValue] = []
    for point in points:
        assert isinstance(point, dict)
        refs = point["evidence_refs"]
        justifications.append(
            {
                "point": point["step"],
                "rules": [
                    {"rule_id": rule, "holds": True, "event_refs": refs} for rule in required
                ],
                "rationale": "The handling choice is open, observable and decides the outcome.",
            }
        )
    return {**answer, "justifications": justifications}


def _actor_events(value: JsonValue, found: list[str]) -> list[str]:
    """Collect every scored actor event id in a request's opening, in order."""
    if isinstance(value, dict):
        if value.get("owner") == "actor" and value.get("scope") == "scored":
            event = value.get("event_id")
            if isinstance(event, str) and event not in found:
                found.append(event)
        for item in value.values():
            _actor_events(item, found)
    elif isinstance(value, list):
        for item in value:
            _actor_events(item, found)
    return found


def severity_answer(payload: JsonObject) -> JsonObject:
    """A severity review citing the first failing actor event it was shown."""
    events = _actor_events(json.loads(_opening(payload)), [])
    assert events, "the severity review received no failing actor event"
    return {
        "severity": "severe",
        "failure_events": events[:1],
        "rationale": "Closing an urgent ticket without escalation breaches its service tier.",
    }


def assembled_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Run, ChoosingActor]:
    """The measured wiring with five actor tiers and the decision model answering the bar."""
    base = grounded_fixture(root, monkeypatch, priced=True)
    for alias in TIERS[2:]:
        add_tier(base.project, alias)
    fixture = rebound(base, TIER_B)
    wire = MeasuredWire(usage=True)
    actor = ChoosingActor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_stages.run_rules_stage", fixture_rules_stage
    )
    monkeypatch.setitem(ANSWERS, CHOICE_TOOL, choice_answer)
    monkeypatch.setitem(ANSWERS, "choose_points", bar_selector_answer)
    monkeypatch.setitem(ANSWERS, SEVERITY_TOOL, severity_answer)
    bank = write_bank(root)
    (bank / "behavior.toml").write_text(BANK)
    run = Run(
        fixture=fixture,
        wire=wire,
        rubric=mining_rubric(fixture),
        bank=bank,
        executors={"variance-build": FixtureBuild(fixture)},
        dependencies=StageDependencies(
            traceml_tagger=BarTagger(MixedTagger(step_tagger(fixture.project))),
            rules_decisions=step_tagger(fixture.project),
            harbor_root=root / "harbor",
        ),
    )
    return run, actor


def assembled_spec(run: Run) -> DecisionVarianceSpec:
    """Five tiers, the qualification bar with blast reviewers, and hybrid dense records."""
    spec = measured_spec(run)
    settings = run.fixture.settings
    assert spec.records.dense is not None
    dense = spec.records.dense.model_copy(
        update={
            "hybrid": HybridDenseSettings(
                bank_id="behavior", decision_alias=spec.traceml.decision_alias, maximum_steps=16
            )
        }
    )
    mining = spec.mining.model_copy(
        update={
            "actors": TIERS,
            "qualification": PointQualification(blast=BlastSpec(reviewer=settings, profile=FORCED)),
        }
    )
    return spec.model_copy(
        update={"mining": mining, "records": spec.records.model_copy(update={"dense": dense})}
    )


def _outputs[Model: ContractModel](
    run: Run, report: PipelineReport, index: int, model: type[Model]
) -> Model:
    """One stage's typed outputs."""
    record = load_stage_record(run.project, report.stages[index].identity)
    assert record is not None
    return stage_outputs(record, model)


def test_five_tiers_reach_qualified_packages_through_the_bar_blast_radius_and_hybrid_dense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling, bar-qualified mining, blast radius, families, records, schedule and replay."""
    run, actor = assembled_run(tmp_path, monkeypatch)
    spec = assembled_spec(run)
    plan = plan_decision_variance(
        run.project, spec, code_revision=REVISION, executors=run.executors
    )
    mining_plan = next(row for row in plan.stages if row.stage == "error-mining")
    assert any("qualification bar" in line for line in mining_plan.lines)
    first = run.run(spec)
    assert first.complete, first.stopped
    qualification = VariantQualification()
    reviewed = spec.model_copy(
        update={
            "authoring": spec.authoring.model_copy(
                update={"dispositions": dispositions_for(run.project, first)}
            ),
            "records": spec.records.model_copy(update={"qualification": qualification}),
        }
    )
    actor.careful = True
    report = run.run(reviewed)
    assert report.complete, report.stopped
    project = run.project

    # The decision model answered the bar over every step of every tier's attempts.
    traceml = _outputs(run, report, 2, TraceMLOutputs)
    assert set(traceml.point_bars) == set(TIERS)
    tagger = run.dependencies.traceml_tagger
    assert isinstance(tagger, BarTagger) and tagger.bars > 0

    # Mining: five tiers, selectors justified picks against the bar, chosen points continued.
    mining = _outputs(run, report, 3, MiningOutputs)
    assert [tier.tier for tier in mining.tiers] == list(TIERS)
    for tier in mining.tiers:
        mined = ActorCriticReport.model_validate_json(
            project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
        )
        assert mined.spec.gate is not None and mined.point_schedule is not None
        assert mined.bar_record is not None
        assert all(isinstance(d.selection, BarSelection) for d in mined.judges if d.selection)
        continued = set(mined.point_schedule.continued)
        assert continued and continued == {row.point for row in mined.points}
        records = load_point_bar(project.artifacts, traceml.point_bars[tier.tier])
        qualified = set(records[mined.picked].qualified)
        assert continued <= qualified
        states = {t.subject: t.target for t in mined.bar_record.transitions}
        assert {states[f"point-{p}"] for p in continued} == {PointState.CONTINUED.value}

    # Blast radius: every continued checkpoint measured across tiers, judged and ranked.
    assert mining.blast is not None
    ranked = load_blast_ranking(project.artifacts, mining.blast)
    assert ranked and all(row.severity == "severe" for row in ranked if row.score is not None)
    assert any(row.score is not None and row.tiers_graded >= 2 for row in ranked)

    # Families: searched variants, the nudge ladder and revised rubrics, admitted.
    families = _outputs(run, report, 4, FamiliesOutputs).families
    admitted = [row for row in families if row.admitted]
    assert admitted and all(row.search is not None and row.ladder is not None for row in admitted)

    # Records: hybrid dense training records, qualified variant and rung packages, and every
    # tier's schedule ordered by the difficulty curriculum.
    records = _outputs(run, report, 5, RecordsOutputs)
    assert records.training
    hybrid = 0
    for reference in records.training.values():
        for dense in project.artifacts.read(reference.artifact_id).manifest.inputs:
            kind = project.artifacts.read(dense.artifact_id).manifest.artifact_type
            hybrid += kind == "dense-hybrid-assessment"
    assert hybrid > 0
    qualified = [row for row in records.harbor if not row.diagnostic]
    assert any(row.strength is not None for row in qualified)
    versions: dict[str, str] = {}
    for row in qualified:
        assert row.task is not None and row.qualification is not None
        assert load_judged_acceptance(project, row.qualification).qualified
        task = load_executable_task(project, row.task)
        assert task.dense is not None and task.judge is not None
        package = Path(row.path)
        assert verify_executable_package(project, package, require_provenance=True) == row.task
        provenance = task.provenance
        assert provenance is not None and provenance.step == row.point
        rubric = provenance.rubric
        assert rubric is not None and (rubric.version, rubric.epoch, rubric.parent_sha256) == (
            1,
            1,
            None,
        )
        assert rubric.items == () and rubric.version_sha256 == versions.setdefault(
            "v1", rubric.version_sha256
        )
        assert provenance.tier == row.tier and set(provenance.tier_pass_rates) == set(TIERS)
        # The fixture's start is authored, so its task names no source trace; the record keeps
        # exactly what the start's task holds.
        start = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
        assert provenance.source_trace_ids == start.task.source_trace_ids
        assert provenance.blast_radius is not None and provenance.blast_rank is not None
        binding = json.loads((package / "frozen-binding.json").read_text())
        assert binding["provenance"] == provenance.model_dump(mode="json")
    for alias in TIERS:
        schedule = records.schedules[alias]
        assert schedule.failure is None and schedule.difficulty is not None
        curriculum = load_difficulty_curriculum(project.artifacts, schedule.difficulty)
        assert list(curriculum.ladder.actors) == list(TIERS)

    # Replay: every stage is saved and nothing is called again.
    calls, decisions = (*run.calls(), len(actor.requests)), dict(actor.counts)
    replay = run.run(reviewed, approve=False)
    assert [row.replayed for row in replay.stages] == [True] * 6
    assert (*run.calls(), len(actor.requests)) == calls and actor.counts == decisions

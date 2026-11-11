"""Deferred-phase accounting follows retained point results without repeating execution."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance import actor_critic, actor_critic_phases, pipeline_test
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.actor_critic_phases import carry_deferred, latest_phase
from exp.optimize.evaluation.variance.evidence_tools import total_usage
from exp.optimize.evaluation.variance.pipeline import (
    load_stage_record,
    plan_decision_variance,
    stage_outputs,
)
from exp.optimize.evaluation.variance.pipeline_points import cost_per_point
from exp.optimize.evaluation.variance.pipeline_spec import (
    CostLine,
    DecisionVarianceSpec,
    MiningOutputs,
    PointQualification,
    StageContext,
    StageRun,
)
from exp.optimize.evaluation.variance.pipeline_test import (
    ANSWERS,
    REVISION,
    Run,
    make_spec,
    step_tagger,
    wired,
)
from exp.optimize.evaluation.variance.tests.assembly_pipeline_test import BarTagger
from exp.optimize.evaluation.variance.tests.point_carry_test import every_point_answer


class _MiningSaved(RuntimeError):
    """Stop the fixture before any family authoring or export work."""


class _StopAfterMining:
    """End fixture setup once the real mining stage has saved its report."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """No external content is needed for the fixture stop."""
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """No calls or costs follow the saved mining report in this fixture."""
        return (), ()

    def run(self, context: StageContext) -> StageRun:
        """Stop before entering the family stage."""
        raise _MiningSaved


def _first_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Run, ActorCriticReport, ArtifactInput, int]:
    """Save a real scripted first phase whose allowance leaves qualified points deferred."""
    run = wired(tmp_path, monkeypatch)
    monkeypatch.setitem(ANSWERS, "choose_points", every_point_answer)
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    spec = spec.model_copy(
        update={"mining": spec.mining.model_copy(update={"qualification": PointQualification()})}
    )
    allowance = cost_per_point(spec)
    run.dependencies = replace(
        run.dependencies,
        traceml_tagger=BarTagger(step_tagger(run.project)),
        remaining_attempts=lambda: allowance,
    )
    run.executors["families"] = _StopAfterMining()
    plan = plan_decision_variance(
        run.project, spec, code_revision=REVISION, executors=run.executors
    )
    with pytest.raises(_MiningSaved):
        run.run(spec)
    saved = load_stage_record(run.project, plan.stages[3].identity)
    assert saved is not None
    (tier,) = stage_outputs(saved, MiningOutputs).tiers
    report = ActorCriticReport.model_validate_json(
        run.project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
    )
    assert report.counts is not None
    assert report.point_schedule is not None and report.point_schedule.deferred
    return run, report, tier.report, allowance


@pytest.mark.parametrize("mode", ["graded", "unknown", "refused"])
def test_carrying_points_recounts_actual_outcomes_and_continuation_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Added, unscored and refused points update only dependent accounting and role scores."""
    run, first, _, allowance = _first_phase(tmp_path, monkeypatch)
    schedule = first.point_schedule
    assert schedule is not None
    if mode == "unknown":
        monkeypatch.setattr(pipeline_test, "semantic", lambda request: "unassessable")
    elif mode == "refused":

        def refuse(project: ProjectStore, job: ContinuationJob, *, code_revision: str) -> NoReturn:
            """Refuse a new continuation before actor dispatch; completed studies still replay."""
            del project, job, code_revision
            raise RuntimeError("fixture continuation refused")

        monkeypatch.setattr(actor_critic, "execute_continuation_job", refuse)

    reference, carried = carry_deferred(
        run.project, first.spec, allowance * (len(schedule.deferred) + 1), REVISION
    )

    previous = {row.point: row for row in first.points}
    for row in carried.points:
        if row.point in previous:
            assert row == previous[row.point]
    assert len(carried.points) > len(first.points)
    assert carried.spec == first.spec and carried.pick == first.pick
    assert carried.group == first.group and carried.group_cost == first.group_cost
    assert carried.group_tier == first.group_tier and carried.picked == first.picked
    assert carried.counts is not None and first.counts is not None
    actual = {
        "continuations_recorded": sum(
            row.outcome.graded + row.outcome.unknown for row in carried.points
        ),
        "continuations_graded": sum(row.outcome.graded for row in carried.points),
        "continuations_unknown": sum(row.outcome.unknown for row in carried.points),
        "point_failures": sum(row.failure is not None for row in carried.points),
    }
    measured = total_usage([row.cost.actor for row in carried.points if row.cost is not None])
    observed = {
        "counts": carried.counts.model_dump(include=set(actual)),
        "usage": carried.usage["continuations"].model_dump(),
    }
    expected = {"counts": actual, "usage": measured.model_dump()}
    assert observed == expected, {"observed": observed, "expected": expected}
    assert carried.counts.model_dump(exclude=set(actual)) == first.counts.model_dump(
        exclude=set(actual)
    )
    assert carried.usage["continuations"] == total_usage(
        [row.cost.actor for row in carried.points if row.cost is not None]
    )
    assert {key: value for key, value in carried.usage.items() if key != "continuations"} == {
        key: value for key, value in first.usage.items() if key != "continuations"
    }
    for before, after in zip(first.critics, carried.critics, strict=True):
        assert after.assessment == before.assessment and after.score == before.score
    stored = ActorCriticReport.model_validate_json(
        run.project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )
    assert stored == carried
    if mode == "unknown":
        assert actual["continuations_unknown"] > first.counts.continuations_unknown
    if mode == "refused":
        assert actual["point_failures"] > first.counts.point_failures
        assert all(
            row.study is None and row.cost is None
            for row in carried.points
            if row.point not in previous
        )


def _saved_phase(
    project: ProjectStore, report: ActorCriticReport, phase: int, source: ArtifactInput
) -> ArtifactInput:
    """Commit a representative cached phase with its original point evidence retained."""
    manifest = project.artifacts.write_json(
        artifact_id=f"{report.spec.identity(REVISION)}-result-phase{phase}",
        artifact_type="actor-critic-report",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=REVISION,
            inputs=(source,),
        ),
        files={"report.json": report},
    )
    return artifact_input(manifest)


def test_cached_accounting_repairs_are_immutable_readonly_to_plan_and_safe_to_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair stale phases without execution, preserving progress and absent legacy evidence."""
    run, first, _, allowance = _first_phase(tmp_path, monkeypatch)
    schedule = first.point_schedule
    assert schedule is not None
    completed_ref, completed = carry_deferred(
        run.project, first.spec, allowance * (len(schedule.deferred) + 1), REVISION
    )
    stale = completed.model_copy(update={"counts": first.counts, "usage": first.usage})
    stale_ref = _saved_phase(run.project, stale, 3, completed_ref)
    raw = run.project.artifacts.read_bytes(stale_ref.artifact_id, "report.json")
    identities = run.project.artifacts.list_ids()
    assert latest_phase(run.project, first.spec, REVISION) == (stale_ref, stale)
    assert run.project.artifacts.list_ids() == identities
    calls = run.calls()

    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        """A cached accounting correction has no authority to call actors or judges."""
        raise AssertionError("cached repair attempted execution")

    monkeypatch.setattr(actor_critic, "execute_continuation_job", unexpected)
    monkeypatch.setattr(actor_critic, "judge_study", unexpected)
    barrier = threading.Barrier(2)

    def repair() -> tuple[ArtifactInput, ActorCriticReport]:
        """Begin both public repair invocations together."""
        barrier.wait(timeout=5)
        return carry_deferred(run.project, first.spec, None, REVISION)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(repair) for _ in range(2)]
        repaired = [future.result(timeout=15) for future in futures]
    assert repaired[0] == repaired[1]
    repaired_ref, accounted = repaired[0]
    assert repaired_ref != stale_ref and accounted == completed
    assert set(run.project.artifacts.list_ids()) - set(identities) == {repaired_ref.artifact_id}
    assert run.project.artifacts.read_bytes(stale_ref.artifact_id, "report.json") == raw
    manifest = run.project.artifacts.read(repaired_ref.artifact_id).manifest
    assert manifest.inputs == (stale_ref,)
    assert (
        ArtifactInput.model_validate_json(
            run.project.artifacts.read_bytes(repaired_ref.artifact_id, "source.json")
        )
        == stale_ref
    )
    identities = run.project.artifacts.list_ids()
    assert latest_phase(run.project, first.spec, REVISION) == repaired[0]
    assert run.project.artifacts.list_ids() == identities

    missing_counts = completed.model_copy(update={"counts": None, "usage": first.usage})
    missing_counts_ref = _saved_phase(run.project, missing_counts, 4, completed_ref)
    _, no_counts = carry_deferred(run.project, first.spec, None, REVISION)
    assert no_counts.counts is None and no_counts.points == completed.points
    assert no_counts.usage == completed.usage
    retained = ActorCriticReport.model_validate_json(
        run.project.artifacts.read_bytes(missing_counts_ref.artifact_id, "report.json")
    )
    assert retained.counts is None and retained.usage == first.usage

    recorded = completed.points[0]
    assert recorded.study is not None and recorded.cost is not None
    missing_cost = completed.model_copy(
        update={
            "points": (
                recorded.model_copy(update={"cost": None}),
                *completed.points[1:],
            )
        }
    )
    missing_cost_ref = _saved_phase(run.project, missing_cost, 5, completed_ref)
    old_ref, unavailable = carry_deferred(run.project, first.spec, None, REVISION)
    assert unavailable.points == missing_cost.points and unavailable.counts == completed.counts
    assert "continuations" not in unavailable.usage
    assert unavailable.usage == {
        key: value for key, value in completed.usage.items() if key != "continuations"
    }
    assert unavailable.group_cost == completed.group_cost
    assert unavailable.points[0].cost is None
    assert (
        "continuations"
        in ActorCriticReport.model_validate_json(
            run.project.artifacts.read_bytes(missing_cost_ref.artifact_id, "report.json")
        ).usage
    )

    monkeypatch.setattr(actor_critic_phases, "PHASE_ACCOUNTING_PROTOCOL", "test-next-accounting")
    new_ref, new_accounting = carry_deferred(run.project, first.spec, None, REVISION)
    assert new_ref != old_ref and new_accounting == unavailable
    later = _saved_phase(run.project, completed, 6, completed_ref)
    identities = run.project.artifacts.list_ids()
    assert latest_phase(run.project, first.spec, REVISION) == (later, completed)
    assert run.project.artifacts.list_ids() == identities
    assert run.calls() == calls

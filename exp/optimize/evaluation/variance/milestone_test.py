"""Offline tests for the preregistered judge-versus-heuristic point selection comparison."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.project.manifests import ArtifactManifest
from exp.common.project.store import ArtifactStore
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
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
from exp.optimize.evaluation.variance.contracts import DecisionState
from exp.optimize.evaluation.variance.contrast_test import _deletion, _finish_transport
from exp.optimize.evaluation.variance.judged_test import JudgeRouter, route_judge, state_judge
from exp.optimize.evaluation.variance.milestone import (
    SelectionComparison,
    SelectionComparisonSpec,
    SelectionStrategy,
    choose_points,
    compare,
    register_option_mapping,
    run_selection_comparison,
)
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.continuations import ContinuationCheckpoint


def _measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ContinuationJob, ActorCriticReport, AuthorSettings]:
    """Run the actor-critic fixture: three single-edit attempts, a judge choosing points 0 and 1."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    estimate = {"probability": 0.5, "rationale": "Base rate.", "evidence_refs": ["group"]}
    roles = [
        _tool_reply("choose_points", {"points": [_point(0), _point(1)], "limitations": ["x"]}),
        _tool_reply(
            "estimate_points",
            {"estimates": [{"step": s, **estimate} for s in (0, 1)], "limitations": ["x"]},
        ),
    ]
    _wire(monkeypatch, group + continuations, roles)
    exposure = _exposure(project, job, "mining_probes")
    report = run_actor_critic(project, spec, exposure=exposure, code_revision="test")
    return project, job, report, spec.judges[0]


_ACTIONS = {"delete-1": _deletion(1), "delete-2": _deletion(2)}


def _late_rows(decision: DecisionState, checkpoint: ContinuationCheckpoint) -> dict[str, ToolCall]:
    """Map options to the separating deletions at the start and to two failing ones later.

    Args:
        decision: The authored decision.
        checkpoint: Checkpoint the options are forced from.

    Returns:
        The concrete forced action per option.
    """
    if checkpoint.actor.calls == 0:
        return {option.option_id: _ACTIONS[option.option_id] for option in decision.options}
    return {"delete-1": _deletion(2), "delete-2": _deletion(3)}


register_option_mapping("late-rows", _late_rows)


def _decision_answer(step: int, observed: JsonValue) -> JsonHttpResponse:
    """One scripted decision-state answer for a point whose probe shows ``observed`` rows."""
    state: JsonObject = {
        "step": step,
        "decision": "How many order rows to delete.",
        "options": [
            {"option_id": "delete-1", "action": "Delete one order row"},
            {"option_id": "delete-2", "action": "Delete two order rows"},
        ],
        "relevant_paths": ["/rows"],
        "axes": [{"path": "/rows", "values": [observed, 7], "effect": "Row count decides it."}],
        "criteria": [{"criterion_id": "rows", "text": "The right rows were deleted."}],
        "evidence_refs": ["point"],
    }
    return _tool_reply(
        "author_decision_state",
        {
            "state": state,
            "grounding": [
                {"path": "/rows", "revealed_by": ["edit_workbook"], "refs": ["state:/rows"]}
            ],
            "limitations": ["Simulated grades only."],
        },
    )


def _comparison_spec(
    settings: AuthorSettings,
    strategies: tuple[SelectionStrategy, ...],
    *,
    points: int,
    mapping: str | None = None,
) -> SelectionComparisonSpec:
    """Register a comparison with four pairs per option contrast."""
    return SelectionComparisonSpec(
        strategies=strategies,
        points=points,
        repeats=4,
        seed=3,
        author=settings,
        option_actions=None if mapping else _ACTIONS,
        option_mapping=mapping,
    )


def _author_transport(monkeypatch: pytest.MonkeyPatch, *replies: JsonHttpResponse) -> JudgeRouter:
    """Route decision-state author calls to a script, and judge one-row deletions as passes."""
    return route_judge(monkeypatch, replies, state_judge(("rows",), 6))


def _units(result: SelectionComparison, count: int) -> list[SelectionComparison]:
    """Replicate one result over ``count`` distinct lineages, as independent tasks would be."""
    return [
        result.model_copy(
            update={"plan": result.plan.model_copy(update={"lineage": f"lineage-{index}"})}
        )
        for index in range(count)
    ]


def test_equal_budgets_replay_and_a_null_comparison_is_not_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every strategy chooses the same K points; equal yields never show a judge advantage."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    strategies: tuple[SelectionStrategy, ...] = (
        "judge",
        "drop",
        "uniform",
        "late_failure",
        "midpoint",
    )
    spec = _comparison_spec(settings, strategies, points=2)
    authors = _author_transport(monkeypatch, _decision_answer(0, None), _decision_answer(1, 6))
    actor = _finish_transport(monkeypatch, 16)
    exposure = _exposure(project, job, "mining_probes")

    def run() -> SelectionComparison:
        """Run or replay the comparison."""
        return run_selection_comparison(
            project, spec, exposure=exposure, report=report, code_revision="test"
        )

    result = run()

    assert result.plan.pool == (0, 1)
    assert all(points == (0, 1) for points in result.plan.points.values())
    assert [row.strategy for row in result.strategies] == list(strategies)
    assert {len(row.points) for row in result.strategies} == {2}
    assert all(trial.separated and trial.leader == "delete-1" for trial in result.trials)
    assert len(authors.requests) == 2 and len(actor.requests) == 16
    judge, drop, uniform = result.strategies[:3]
    assert (judge.author_calls, judge.contrast_draws) == (2, 16)
    assert judge.selection_role_calls == 2 and judge.selection_continuations == 4
    assert drop.total_cost == judge.total_cost == 2 + 4 + 2 + 16
    assert uniform.selection_role_calls == uniform.selection_continuations == 0
    assert uniform.total_cost == 18 and uniform.yield_per_cost == pytest.approx(2 / 18)
    registration = project.artifacts.read(result.registration.artifact_id).manifest
    assert registration.artifact_type == "selection-comparison-plan"

    assert run() == result
    assert len(authors.requests) == 2 and len(actor.requests) == 16
    pooled = compare(_units(result, 12))
    assert pooled.verdict == "not_shown"
    assert all(d.estimate == 0.0 and not d.exceeds_noise for d in pooled.differences)
    cheap, judged = pooled.totals[2].yield_per_cost, pooled.totals[0].yield_per_cost
    assert cheap is not None and judged is not None and cheap > judged


def test_judge_wins_only_beyond_sampling_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge's point separates and the late heuristic's does not; one task is noise."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(settings, ("judge", "late_failure"), points=1, mapping="late-rows")
    _author_transport(monkeypatch, _decision_answer(0, None), _decision_answer(1, 6))
    _finish_transport(monkeypatch, 16)

    result = run_selection_comparison(
        project,
        spec,
        exposure=_exposure(project, job, "mining_probes"),
        report=report,
        code_revision="test",
    )

    judge, late = result.strategies
    assert (judge.points, judge.separated) == ((0,), (0,))
    assert (late.points, late.separated, late.unseparated) == ((1,), (), (1,))
    assert compare([result]).verdict == "not_shown"
    assert compare(_units(result, 3)).verdict == "not_shown"
    won = compare(_units(result, 8))
    assert won.verdict == "judge_better"
    assert won.differences[0].wilson_95 is not None and won.differences[0].wilson_95[0] > 0
    assert (won.totals[0].separated, won.totals[1].separated) == (8, 0)


def test_judge_must_beat_every_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A baseline that matches the judge keeps the verdict at not_shown."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(
        settings, ("judge", "late_failure", "midpoint"), points=1, mapping="late-rows"
    )
    _author_transport(monkeypatch, _decision_answer(0, None), _decision_answer(1, 6))
    _finish_transport(monkeypatch, 16)

    result = run_selection_comparison(
        project,
        spec,
        exposure=_exposure(project, job, "mining_probes"),
        report=report,
        code_revision="test",
    )

    assert result.plan.points["midpoint"] == (0,)
    verdict = compare(_units(result, 12))
    assert [d.exceeds_noise for d in verdict.differences] == [True, False]
    assert verdict.verdict == "not_shown"


def test_an_unmeetable_budget_is_refused_before_any_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three points cannot come from a two-point trajectory, so nothing is registered or drawn."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(settings, ("judge", "uniform"), points=3)
    authors = _author_transport(monkeypatch)
    actor = _finish_transport(monkeypatch, 0)

    with pytest.raises(ValueError, match="fewer than the budget of 3"):
        run_selection_comparison(
            project,
            spec,
            exposure=_exposure(project, job, "mining_probes"),
            report=report,
            code_revision="test",
        )

    assert authors.requests == [] and actor.requests == []
    assert not project.artifacts.exists(spec.identity(report, "test"))


def test_a_lost_author_draw_is_saved_as_a_failure_and_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draw with no response fails its point; a crash before the result never redraws it."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(settings, ("judge", "midpoint"), points=1)
    rejected = JsonHttpResponse(status_code=400, body={"error": {"message": "bad request"}})
    authors = _author_transport(monkeypatch, rejected)
    exposure = _exposure(project, job, "mining_probes")
    original = ArtifactStore.write_json

    def crash(
        self: ArtifactStore,
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Fail only the final comparison write, as an interrupted process would."""
        if artifact_type == "selection-comparison":
            raise OSError("interrupted")
        return original(
            self,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )

    monkeypatch.setattr(ArtifactStore, "write_json", crash)
    with pytest.raises(OSError, match="interrupted"):
        run_selection_comparison(
            project, spec, exposure=exposure, report=report, code_revision="test"
        )
    monkeypatch.setattr(ArtifactStore, "write_json", original)
    calls = len(authors.requests)
    _author_transport(monkeypatch, _decision_answer(0, None))

    result = run_selection_comparison(
        project, spec, exposure=exposure, report=report, code_revision="test"
    )

    trial = result.trials[0]
    assert trial.draft is not None and trial.draft.response is None
    assert trial.failure is not None and trial.failure.startswith("author")
    assert all(row.failed == (0,) for row in result.strategies)
    assert len(authors.requests) == calls


def test_a_refused_decision_state_is_an_explicit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer for the wrong step fails its point and costs one author call, no contrast."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(settings, ("judge", "midpoint"), points=1)
    authors = _author_transport(monkeypatch, _decision_answer(5, None))
    actor = _finish_transport(monkeypatch, 0)
    exposure = _exposure(project, job, "mining_probes")

    result = run_selection_comparison(
        project, spec, exposure=exposure, report=report, code_revision="test"
    )

    assert [trial.point for trial in result.trials] == [0]
    trial = result.trials[0]
    assert trial.contrast is None and trial.failure is not None and "author" in trial.failure
    assert all(row.failed == (0,) and row.separated == () for row in result.strategies)
    assert result.strategies[1].total_cost == 1
    assert len(authors.requests) == 1 and actor.requests == []


def test_late_failure_anchors_on_the_first_failed_tool_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a failed call the rule takes the last point; an error moves it before that turn."""
    project, job, report, settings = _measured(tmp_path, monkeypatch)
    spec = _comparison_spec(settings, ("judge", "late_failure"), points=1)
    branch = load_study(project.artifacts, report.group).branches[report.picked]
    assert choose_points("late_failure", spec, report, branch, (0, 1)) == (1,)
    steps = branch.outcome["steps"]
    assert isinstance(steps, list) and isinstance(steps[0], dict)
    reply = steps[0]["reply"]
    assert isinstance(reply, dict)
    failed = branch.model_copy(
        update={
            "outcome": {
                **branch.outcome,
                "steps": [{**steps[0], "reply": {**reply, "is_error": True}}],
            }
        }
    )
    assert choose_points("late_failure", spec, report, failed, (0, 1)) == (0,)
    assert choose_points("midpoint", spec, report, branch, (0, 1, 2, 3, 4)) == (2,)
    uniform = choose_points("uniform", spec, report, branch, (0, 1, 2, 3, 4))
    assert uniform == choose_points("uniform", spec, report, branch, (0, 1, 2, 3, 4))
    assert choose_points("judge", spec, report, branch, (0, 1)) == (report.ranking[0],)
    with pytest.raises(ValueError, match="judge once"):
        _comparison_spec(settings, ("uniform", "midpoint"), points=1)

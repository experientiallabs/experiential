"""Deferred points carry into the next phase under the live ledger, with no new registration.

Phase one's live allowance fits one point's continuations, so the bar-qualified picks beyond it
are deferred on the run's persisted record. Phase two reads a larger live balance: the mining
stage re-enters because deferred points are recorded, continues exactly the deferred points as
the run's next phase report, and replays every point already continued without buying it again.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.pipeline import load_stage_record, stage_outputs
from exp.optimize.evaluation.variance.pipeline_points import cost_per_point
from exp.optimize.evaluation.variance.pipeline_spec import (
    MiningOutputs,
    PipelineReport,
    PointQualification,
)
from exp.optimize.evaluation.variance.pipeline_test import (
    ANSWERS,
    Run,
    make_spec,
    selector_answer,
    step_tagger,
    wired,
)
from exp.optimize.evaluation.variance.point_bar import DEFAULT_BAR
from exp.optimize.evaluation.variance.tests.assembly_pipeline_test import BarTagger


def every_point_answer(payload: JsonObject) -> JsonObject:
    """Pick every offered point, each justified against the bar with its own locator."""
    answer = selector_answer()
    if "qualification_bar" not in json.dumps(payload):
        return answer
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    opening = json.loads(str(messages[1]["content"]))
    offered = sorted(int(row["point"]) for row in opening["points"])
    template = answer["points"]
    assert isinstance(template, list) and isinstance(template[0], dict)
    required = [rule.rule_id for rule in DEFAULT_BAR.rules if rule.required]
    points: list[JsonValue] = []
    justifications: list[JsonValue] = []
    for step in offered:
        refs = [f"point:{step}", "group"]
        points.append({**template[0], "step": step, "evidence_refs": refs})
        justifications.append(
            {
                "point": step,
                "rules": [
                    {"rule_id": rule, "holds": True, "event_refs": refs} for rule in required
                ],
                "rationale": "An open, observable choice the outcome depends on.",
            }
        )
    return {**answer, "points": points, "justifications": justifications}


def _mining(run: Run, report: PipelineReport) -> MiningOutputs:
    """The error-mining stage's outputs."""
    record = load_stage_record(run.project, report.stages[3].identity)
    assert record is not None
    return stage_outputs(record, MiningOutputs)


def test_deferred_points_continue_in_the_next_phase_without_rebuying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small first allowance defers; the next pass continues only the deferred points."""
    run = wired(tmp_path, monkeypatch)
    monkeypatch.setitem(ANSWERS, "choose_points", every_point_answer)
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    spec = spec.model_copy(
        update={"mining": spec.mining.model_copy(update={"qualification": PointQualification()})}
    )
    balance = {"left": cost_per_point(spec)}
    run.dependencies = replace(
        run.dependencies,
        traceml_tagger=BarTagger(step_tagger(run.project)),
        remaining_attempts=lambda: balance["left"],
    )
    first = run.run(spec)
    assert first.complete, first.stopped
    (tier,) = _mining(run, first).tiers
    phase_one = ActorCriticReport.model_validate_json(
        run.project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
    )
    schedule = phase_one.point_schedule
    assert schedule is not None and len(schedule.continued) == 1 and schedule.deferred
    continued = {row.point: row for row in phase_one.points}
    assert set(continued) == set(schedule.continued)

    # Phase two: more of the cumulative ledger is left; nothing is re-registered.
    balance["left"] = cost_per_point(spec) * (len(schedule.deferred) + 1)
    actor_before = len(run.fixture.actor.requests)
    second = run.run(spec)
    assert second.complete, second.stopped
    assert not second.stages[3].replayed
    assert second.stages[3].identity != first.stages[3].identity
    (carried,) = _mining(run, second).tiers
    assert carried.report.artifact_id.endswith("-result-phase2")
    phase_two = ActorCriticReport.model_validate_json(
        run.project.artifacts.read_bytes(carried.report.artifact_id, "report.json")
    )
    assert phase_two.spec == phase_one.spec
    merged = phase_two.point_schedule
    assert merged is not None and not merged.deferred
    assert set(merged.continued) == set(schedule.continued) | set(schedule.deferred)
    points = {row.point: row for row in phase_two.points}
    for point, row in continued.items():
        # Already-continued points keep their saved study: they were not bought again.
        assert points[point].study == row.study and points[point].judged == row.judged
    assert set(points) == set(merged.continued)
    bought = len(run.fixture.actor.requests) - actor_before
    assert bought > 0

    # A third pass with nothing deferred replays every saved phase with no call.
    calls = (*run.calls(), len(run.fixture.actor.requests))
    third = run.run(spec)
    assert third.complete
    fourth = run.run(spec, approve=False)
    assert [row.replayed for row in fourth.stages] == [True] * 6
    assert (*run.calls(), len(run.fixture.actor.requests)) == calls
    stable = (*run.calls(), len(run.fixture.actor.requests))
    run.run(spec, approve=False)
    assert (*run.calls(), len(run.fixture.actor.requests)) == stable

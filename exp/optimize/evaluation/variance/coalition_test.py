"""Offline tests for gathering the behavioral coalition from a real error-mining run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject
from exp.common.project import artifact_input
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance.actor_critic import run_actor_critic
from exp.optimize.evaluation.variance.actor_critic_test import (
    _done,
    _edit,
    _exposure,
    _point,
    _spec,
    _tool_reply,
    _wire,
    deletes_row,
)
from exp.optimize.evaluation.variance.coalition import (
    STEP_LABELS_TYPE,
    BehaviorPattern,
    load_step_labels,
    mining_coalition,
)
from exp.optimize.evaluation.variance.judged import judge_study

LABELS: JsonObject = {
    "traces": [
        {
            "trace_id": "attempt-trace-0",
            "tier": "council-a",
            "steps": [{"index": 0, "dominant": "explore"}, {"index": 1, "dominant": "recover"}],
            "pivots": [{"kind": "pivot", "step": 1, "triggers": ["changes_approach"]}],
            "returns": [],
        }
    ]
}


def test_the_coalition_holds_every_attempt_label_selector_critic_and_step_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mining run's whole group, judged by the current judge, becomes the coalition."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job).model_copy(update={"actor_tier": "council-a"})
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    estimate = {"probability": 0.5, "rationale": "Base rate.", "evidence_refs": ["group"]}
    roles = [
        _tool_reply(
            "choose_points",
            {"points": [_point(0), _point(1)], "limitations": ["Simulated grades only."]},
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [{"step": 0, **estimate}, {"step": 1, **estimate}],
                "limitations": ["Simulated grades only."],
            },
        ),
    ]
    _wire(monkeypatch, group + continuations, roles, deletes_row(2))
    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )
    _, judged = judge_study(
        project,
        report.group,
        spec.episode_judge,
        configuration=job.configuration,
        code_revision="test",
    )
    labels = project.artifacts.write_json(
        artifact_id="traceml-fixture",
        artifact_type=STEP_LABELS_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"labels.json": LABELS},
    )
    pattern = BehaviorPattern(
        pattern_id="wrong-row",
        kind="mistake",
        description="Deleted a row other than the cancelled order.",
        attempts=("council-a.g1", "council-a.g2"),
    )

    coalition = mining_coalition(
        project,
        [report],
        judged_groups={"council-a": judged},
        judge=spec.episode_judge.identity,
        patterns=(pattern,),
        step_labels=artifact_input(labels),
    )

    assert [(row.attempt_id, row.label) for row in coalition.attempts] == [
        ("council-a.g0", "pass"),
        ("council-a.g1", "fail"),
        ("council-a.g2", "fail"),
    ]
    assert {row.label_status for row in coalition.attempts} == {"judged"}
    (tier,) = coalition.tiers
    assert (tier.tier, tier.attempts, tier.passed, tier.failed) == ("council-a", 3, 1, 2)
    assert tier.patterns == ("wrong-row",)
    assert [(row.draw, row.step) for row in coalition.selections] == [(0, 0), (0, 1)]
    assert [row.estimate for row in coalition.critics] == [0.5, 0.5]
    assert all(row.judge == spec.episode_judge.identity for row in coalition.critics)
    assert {row.label for row in coalition.step_labels} == {"explore", "recover", "pivot"}
    assert report.group in coalition.references and artifact_input(labels) in coalition.references
    assert coalition.unavailable == ("no judged action-contrast outcomes were supplied",)
    assert coalition.mentions(["council-a did it"]) == ("council-a",)
    coalition.check_exposure(_exposure(project, job, "mining_probes"))
    with pytest.raises(ValueError, match="final evaluation"):
        coalition.check_exposure(_exposure(project, job, "final_evaluation"))

    historical = mining_coalition(
        project, [report], judged_groups={"council-a": judged}, judge="5" * 64
    )
    assert {row.label_status for row in historical.attempts} == {"historical"}
    unjudged = mining_coalition(project, [report])
    assert {row.label for row in unjudged.attempts} == {"unjudged"}
    assert "tier council-a: its group of attempts was not judged" in unjudged.unavailable
    with pytest.raises(ValueError, match="share tier"):
        mining_coalition(project, [report, report])
    assert len(load_step_labels(project, artifact_input(labels))) == 3


def test_a_long_trigger_text_is_cut_to_the_step_label_limit() -> None:
    """Run-31 pivots carried trigger texts past 1500 characters; they are cut, not refused."""
    from exp.optimize.evaluation.variance.coalition import DETAIL_LIMIT, StepLabel, _bounded_detail

    detail = _bounded_detail("triggers " + "x" * 4000, " (uncertain)")
    assert len(detail) == DETAIL_LIMIT and detail.endswith("[truncated] (uncertain)")
    StepLabel(trace_id="t", step=1, label="pivot", detail=detail)
    assert _bounded_detail("triggers short", "") == "triggers short"

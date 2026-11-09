"""Cross-tier blast radius: the other tiers continue a tier's top error points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance import pipeline_points
from exp.optimize.evaluation.variance.blast_radius import (
    BlastPoint,
    SeverityJudgment,
    TierOutcome,
    load_cross_tier,
)
from exp.optimize.evaluation.variance.blast_radius_test import spec_for
from exp.optimize.evaluation.variance.difficulty_curriculum import (
    ActorLadder,
    TaskAttempts,
    difficulty_tiers,
    merge_curriculum,
)
from exp.optimize.evaluation.variance.difficulty_curriculum_test import stage

LADDER = ("gemma", "glm-flash", "glm", "kimi", "mimo")
PASSES = {"glm-flash": 1, "glm": 2, "kimi": 7, "mimo": 8}
"""Planted: at gemma's point every tier through glm fails; kimi (tier 4) and mimo recover."""


def _ref(name: str) -> ArtifactInput:
    return ArtifactInput(artifact_id=name, sha256="0" * 64)


@dataclass(frozen=True)
class _Study:
    checkpoint: ArtifactInput | None = None
    repeats: int = 8

    def model_copy(self, update: dict[str, object]) -> _Study:
        known = {k: v for k, v in update.items() if k in ("checkpoint", "repeats")}
        return replace(self, **known)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Job:
    actor_alias: str
    study: _Study = field(default_factory=_Study)
    configuration: dict[str, object] = field(default_factory=dict)
    maximum_physical_attempts: int = 1

    def model_copy(self, update: dict[str, object]) -> _Job:
        return replace(self, **update)  # type: ignore[arg-type]


def test_a_point_a_weak_tier_fails_and_tier_four_recovers_is_read_across_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other tiers continue gemma's top point; shares, recoveries and order follow."""
    project = ProjectStore(tmp_path, "cross")
    tiers = [
        SimpleNamespace(
            tier=alias,
            report=_ref(f"report-{alias}"),
            judge=SimpleNamespace(settings=SimpleNamespace(model_alias="judge")),
            job=_Job(actor_alias=alias),
        )
        for alias in LADDER
    ]
    planted, flat = _ref("checkpoint-planted"), _ref("checkpoint-flat")
    reports = {
        "report-gemma": SimpleNamespace(
            points=(SimpleNamespace(point=3, checkpoint=planted, judged=None),)
        ),
        "report-glm": SimpleNamespace(
            points=(SimpleNamespace(point=5, checkpoint=flat, judged=None),)
        ),
    }
    points = {
        planted.artifact_id: BlastPoint(
            point_id=planted.artifact_id,
            point=3,
            checkpoint=planted,
            tiers=(TierOutcome(tier="gemma", successes=0, failures=8),),
        ),
        flat.artifact_id: BlastPoint(
            point_id=flat.artifact_id,
            point=5,
            checkpoint=flat,
            tiers=(TierOutcome(tier="glm", successes=0, failures=8),),
        ),
    }
    calls: list[tuple[str, str, int, int]] = []

    def saved(project: object, job: _Job, *, code_revision: str) -> SimpleNamespace:
        assert job.study.checkpoint is not None
        calls.append(
            (
                job.actor_alias,
                job.study.checkpoint.artifact_id,
                job.study.repeats,
                job.maximum_physical_attempts,
            )
        )
        return SimpleNamespace(branches=(None,) * job.study.repeats)

    def reference(project: object, job: _Job, *, code_revision: str) -> ArtifactInput:
        assert job.study.checkpoint is not None
        return _ref(f"study-{job.actor_alias}-{job.study.checkpoint.artifact_id}")

    def judge(project: object, ref: ArtifactInput, judge: object, **kwargs: object) -> tuple:
        if ref.artifact_id == "study-glm-checkpoint-planted":
            raise RuntimeError("world restore failed")  # one continuation fails, is skipped
        return _ref("judged-" + ref.artifact_id), SimpleNamespace(
            branches=ref.artifact_id, judge="j"
        )

    def rewards(project: object, branches: str, judge: str) -> tuple[float, ...]:
        tier = branches.split("-")[1] if not branches.startswith("study-glm-flash") else "glm-flash"
        passed = PASSES.get(tier, 0) if "planted" in branches else 0
        return (1.0,) * passed + (0.0,) * (8 - passed)

    monkeypatch.setattr(
        pipeline_points,
        "load_report",
        lambda project, ref: reports.get(ref.artifact_id, SimpleNamespace(points=())),
    )
    monkeypatch.setattr(pipeline_points, "tier_outcomes", lambda reports: dict(points))
    monkeypatch.setattr(
        pipeline_points,
        "judge_severity",
        lambda project, review, key, *a, **k: SeverityJudgment(point_id=key, severity="severe"),
    )
    monkeypatch.setattr(pipeline_points, "admitted_concurrency", lambda project, *aliases: 4)
    monkeypatch.setattr(pipeline_points, "saved_continuation_job", saved)
    monkeypatch.setattr(pipeline_points, "continuation_result_reference", reference)
    monkeypatch.setattr(pipeline_points, "judge_study", judge)
    monkeypatch.setattr(pipeline_points, "slot_rewards", rewards)
    blast = spec_for()
    spec = SimpleNamespace(
        mining=SimpleNamespace(
            qualification=SimpleNamespace(blast=blast),
            actors=LADDER,
            continuations=8,
            continuation_ceiling=lambda repeats=None: 160,
        )
    )

    result = pipeline_points.mine_blast_radius(
        project,
        spec,
        tiers,
        code_revision="test",
        inputs=(),  # type: ignore[arg-type]
    )

    assert result is not None
    _, ranked, cross_ref = result
    assert cross_ref is not None
    assert len(calls) == 8  # two origin points, four other tiers each
    assert all(repeats == 8 and ceiling == 160 for *_, repeats, ceiling in calls)
    record = load_cross_tier(project.artifacts, cross_ref)
    row = next(p for p in record.points if p.point_id == planted.artifact_id)
    assert row.origin == "gemma"
    assert row.failing_tiers == ("gemma", "glm-flash")
    assert row.pass_rates["kimi"] == pytest.approx(7 / 8)
    assert ("gemma", "kimi") in row.recoveries and ("glm-flash", "kimi") in row.recoveries
    assert "glm" in row.failures and "glm" not in row.pass_rates  # failed continuation skipped
    assert row.failing_share == pytest.approx(2 / 4)
    assert row.teaching == pytest.approx(4 / 5)  # 4 of 5 weaker-failing pairs recover
    flat_row = next(p for p in record.points if p.point_id == flat.artifact_id)
    assert flat_row.recoveries == () and flat_row.teaching == 0.0
    by_point = {r.point_id: r for r in ranked}
    assert by_point[planted.artifact_id].failing_share == pytest.approx(0.5)
    assert by_point[planted.artifact_id].ever_passed
    assert by_point[flat.artifact_id].failing_share == pytest.approx(1.0)

    # The curriculum puts the planted point's ladder first inside a strength, even though
    # its identity sorts after the flat point's.
    tasks = difficulty_tiers(
        [TaskAttempts(task_id="t", actor=a, outcomes=(0.0,) * 4) for a in LADDER],
        ActorLadder(actors=LADDER),
    )
    stages = [stage("ladder-a-flat", 0, "strong"), stage("ladder-z-planted", 0, "strong")]
    teaching = {
        "ladder-a-flat": record.teaching()[flat.artifact_id],
        "ladder-z-planted": record.teaching()[planted.artifact_id],
    }
    plain = merge_curriculum(tasks, {"t": stages}, ladder=ActorLadder(actors=LADDER))
    taught = merge_curriculum(
        tasks, {"t": stages}, ladder=ActorLadder(actors=LADDER), teaching=teaching
    )
    assert [e.ladder for e in plain.entries] == ["ladder-a-flat", "ladder-z-planted"]
    assert [e.ladder for e in taught.entries] == ["ladder-z-planted", "ladder-a-flat"]

    replay = pipeline_points.mine_blast_radius(
        project,
        spec,
        tiers,
        code_revision="other",
        inputs=(),  # type: ignore[arg-type]
    )
    assert replay is not None and replay[2] == cross_ref  # content-addressed, no code revision

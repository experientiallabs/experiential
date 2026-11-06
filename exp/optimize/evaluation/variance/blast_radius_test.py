"""Blast radius: failing share across tiers, ever passed, judged severity and decision variance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorTier,
    PointResult,
)
from exp.optimize.evaluation.variance.behavior_differences_test import (
    settings as fixture_settings,
)
from exp.optimize.evaluation.variance.blast_radius import (
    SEVERITY_TOOL,
    BlastPoint,
    BlastSpec,
    Severity,
    SeverityJudgment,
    TierOutcome,
    blast_radius,
    blast_reference,
    judge_severity,
    order_by_blast,
    rank_blast,
    save_blast_radius,
    tier_outcomes,
)
from exp.optimize.evaluation.variance.decision_measure_test import FORCED, reply, role_fixture
from exp.optimize.evaluation.variance.dense_hybrid_test import weakened
from exp.optimize.evaluation.variance.dense_judge_test import MODEL
from exp.optimize.evaluation.variance.rewards import PointDecision, PointOutcome
from exp.runtime.models.providers.transport import JsonHttpResponse

TIERS = ("gemma", "glm-flash", "glm", "kimi", "mimo")


def tiers(*rates: tuple[int, int]) -> tuple[TierOutcome, ...]:
    """One outcome per tier from (successes, failures)."""
    return tuple(
        TierOutcome(tier=tier, successes=s, failures=f)
        for tier, (s, f) in zip(TIERS, rates, strict=False)
    )


def severity(point: str, level: Severity | None) -> SeverityJudgment:
    """A reconciled severity."""
    return SeverityJudgment(point_id=point, severity=level)


def spec_for(settings: AuthorSettings | None = None) -> BlastSpec:
    """Blast settings with a forced reviewer profile."""
    return BlastSpec(reviewer=settings or fixture_settings(), profile=FORCED)


def test_points_rank_by_failing_share_severity_and_decision_variance() -> None:
    """Widest severe failure first; a passable point beats a never-passed one at equal score."""
    spec = spec_for()
    split = PointDecision(
        point_id="p", choices=("a",) * 4 + ("b",) * 4, rewards=(1.0,) * 4 + (0.0,) * 4
    )
    points = {
        "wide-critical": BlastPoint(
            point_id="wide-critical",
            point=3,
            attempt=0,
            tiers=tiers((0, 8), (0, 8), (1, 7), (2, 6), (6, 2)),
            decision=split,
        ),
        "narrow-critical": BlastPoint(
            point_id="narrow-critical",
            point=5,
            attempt=0,
            tiers=tiers((0, 8), (8, 0), (8, 0), (8, 0), (8, 0)),
            decision=split,
        ),
        "wide-minor": BlastPoint(
            point_id="wide-minor",
            point=1,
            attempt=0,
            tiers=tiers((0, 8), (0, 8), (0, 8), (0, 8), (0, 8)),
        ),
        "unpassed": BlastPoint(
            point_id="unpassed",
            point=7,
            attempt=0,
            tiers=tiers((0, 8), (0, 8), (0, 8), (0, 8), (0, 8)),
            decision=PointDecision(point_id="unpassed", joint_score=1.0),
        ),
        "unjudged": BlastPoint(point_id="unjudged", point=9, tiers=tiers((0, 8))),
    }
    levels: dict[str, Severity | None] = {
        "wide-critical": "critical",
        "narrow-critical": "critical",
        "wide-minor": "minor",
        "unpassed": "critical",
        "unjudged": None,
    }
    rows = [blast_radius(points[k], severity(k, levels[k]), spec) for k in points]
    ranked = rank_blast(rows)

    assert [row.point_id for row in ranked] == [
        "unpassed",
        "wide-critical",
        "narrow-critical",
        "wide-minor",
        "unjudged",
    ]
    by_id = {row.point_id: row for row in ranked}
    wide = by_id["wide-critical"]
    assert wide.tiers_failing == 4 and wide.failing_share == pytest.approx(0.8)
    assert wide.ever_passed and wide.decision_variance == pytest.approx(0.875)
    assert wide.score == pytest.approx(0.75)
    assert wide.failing_tiers == ("gemma", "glm-flash", "glm", "kimi")
    assert not by_id["unpassed"].ever_passed and by_id["unpassed"].score == 1.0
    assert by_id["unjudged"].score is None and by_id["unjudged"].rank == 5
    assert by_id["wide-minor"].score == pytest.approx(0.125)
    assert "decision variance unmeasured, counted as 0" in by_id["wide-minor"].reasons

    tie = [
        blast_radius(points["wide-minor"], severity("wide-minor", "minor"), spec),
        blast_radius(
            points["wide-minor"].model_copy(
                update={"point_id": "passable", "tiers": tiers((1, 8), (0, 8), (0, 8))}
            ),
            severity("passable", "minor"),
            spec,
        ),
    ]
    assert [row.point_id for row in rank_blast(tie)] == ["passable", "wide-minor"]


def test_the_ranking_is_a_selector_reference_and_orders_family_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved as an error-points artifact keyed by attempt; family order follows blast rank."""
    project, _, _ = role_fixture(tmp_path, monkeypatch, [])
    spec = spec_for()
    ranked = rank_blast(
        [
            blast_radius(
                BlastPoint(point_id="point-2", point=2, attempt=4, tiers=tiers((0, 4), (0, 4))),
                severity("point-2", "severe"),
                spec,
            ),
            blast_radius(
                BlastPoint(point_id="point-6", point=6, attempt=4, tiers=tiers((0, 4), (4, 0))),
                severity("point-6", "severe"),
                spec,
            ),
        ]
    )

    artifact = save_blast_radius(project.artifacts, ranked, code_revision="test")
    source = blast_reference(artifact)

    assert source.kind == "error_points" and source.file == "blast.json"
    record = json.loads(project.artifacts.read_bytes(artifact.artifact_id, "blast.json"))
    assert record["attempts"][0]["attempt"] == 4
    assert [row["point"] for row in record["attempts"][0]["points"]] == [2, 6]
    assert save_blast_radius(project.artifacts, ranked, code_revision="test") == artifact
    assert order_by_blast((6, 9, 2, 1), ranked) == (2, 6, 9, 1)
    assert order_by_blast((6, 2), ranked, attempt=3) == (6, 2)


def severity_reply(level: str, events: list[str]) -> JsonHttpResponse:
    """A forced severity answer."""
    return reply(
        [
            (
                SEVERITY_TOOL,
                {
                    "severity": level,
                    "failure_events": events,
                    "rationale": "Deleting the assertion hides the unfixed bug.",
                },
            )
        ]
    )


def test_severity_is_the_strict_majority_of_agentic_stances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three stances; an answer citing a world event votes unknown; majority still holds."""
    replies = [
        severity_reply("critical", ["e1"]),
        severity_reply("critical", ["e1", "e5"]),
        severity_reply("minor", ["e2"]),
    ]
    project, settings, transport = role_fixture(tmp_path, monkeypatch, replies)
    spec = spec_for(settings)
    evidence, _ = weakened()

    judgment = judge_severity(
        project, spec, "point-1", (evidence,), sources=(), code_revision="test"
    )

    assert judgment.severity == "critical"
    assert judgment.votes == {"critical": 2, "unknown": 1}
    assert judgment.reviews[2].failure and "scored actor events" in judgment.reviews[2].failure
    assert "STANCE (consequence_first)" in json.dumps(transport.requests[0].payload)
    assert len(transport.requests) == 3
    again = judge_severity(project, spec, "point-1", (evidence,), sources=(), code_revision="test")
    assert again.severity == "critical" and len(transport.requests) == 3


def test_split_stances_leave_severity_unknown_and_the_point_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No strict majority: severity None, so the point is never guessed into the ranking."""
    replies = [
        severity_reply("critical", ["e1"]),
        severity_reply("moderate", ["e1"]),
        severity_reply("minor", ["e1"]),
    ]
    project, settings, _ = role_fixture(tmp_path, monkeypatch, replies)
    spec = spec_for(settings)
    evidence, _ = weakened()

    judgment = judge_severity(project, spec, "p", (evidence,), sources=(), code_revision="test")
    row = blast_radius(BlastPoint(point_id="p", point=1, tiers=tiers((0, 4))), judgment, spec)

    assert judgment.severity is None and row.score is None
    assert judge_severity(project, spec, "q", (), sources=(), code_revision="t").severity is None


def test_tier_outcomes_combine_tiers_by_shared_checkpoint() -> None:
    """Two tiers continuing from one checkpoint form one point; tiers are never pooled."""
    checkpoint = ArtifactInput(artifact_id="checkpoint-a", sha256="1" * 64)
    other = ArtifactInput(artifact_id="checkpoint-b", sha256="2" * 64)

    def result(tier: str, where: ArtifactInput, s: int, f: int) -> PointResult:
        return PointResult.model_construct(
            point=3,
            checkpoint=where,
            study=ArtifactInput(artifact_id=f"study-{tier}", sha256="3" * 64),
            outcome=PointOutcome(point_id="point-3", step=3, successes=s, failures=f),
            tier=ActorTier(label=tier, actor_model=MODEL, profile_sha256="4" * 64),
        )

    reports = [
        ActorCriticReport.model_construct(
            picked=2, points=(result("gemma", checkpoint, 0, 8), result("gemma", other, 1, 1))
        ),
        ActorCriticReport.model_construct(picked=2, points=(result("kimi", checkpoint, 6, 2),)),
    ]

    points = tier_outcomes(reports)

    assert sorted(points) == ["checkpoint-a", "checkpoint-b"]
    shared = points["checkpoint-a"]
    assert [(t.tier, t.successes, t.failures) for t in shared.tiers] == [
        ("gemma", 0, 8),
        ("kimi", 6, 2),
    ]
    assert shared.attempt == 2 and shared.point == 3

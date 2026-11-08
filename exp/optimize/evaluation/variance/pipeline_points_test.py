"""Blast-radius mining across tiers: one unreviewable point never sinks the ranking."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance import pipeline_points
from exp.optimize.evaluation.variance.blast_radius import BlastPoint, SeverityJudgment
from exp.optimize.evaluation.variance.blast_radius_test import spec_for, tiers


def test_a_point_whose_review_cannot_be_prepared_is_unscored_and_the_rest_rank_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Severity reviews run at once within the admission bound; a refusal leaves one unscored.

    Before, the reviews ran one point at a time and one ValueError (a review that could not be
    prepared) aborted the error-mining stage after every tier's mining was paid for.
    """
    points = {
        f"p{index}": BlastPoint(point_id=f"p{index}", point=index, tiers=tiers((0, 8), (0, 8)))
        for index in range(4)
    }
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def judge(
        project: object, review: object, key: str, *args: object, **kwargs: object
    ) -> SeverityJudgment:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        if key == "p2":
            raise ValueError("request too large for the reviewer")
        return SeverityJudgment(point_id=key, severity="severe")

    saved: dict[str, Any] = {}

    def save(store: object, ranked: object, **kwargs: object) -> ArtifactInput:
        saved["ranked"] = ranked
        return ArtifactInput(artifact_id="blast", sha256="0" * 64)

    monkeypatch.setattr(pipeline_points, "tier_outcomes", lambda reports: points)
    monkeypatch.setattr(pipeline_points, "judge_severity", judge)
    monkeypatch.setattr(pipeline_points, "save_blast_radius", save)
    monkeypatch.setattr(pipeline_points, "admitted_concurrency", lambda project, *aliases: 2)
    spec = SimpleNamespace(mining=SimpleNamespace(qualification=SimpleNamespace(blast=spec_for())))
    project = SimpleNamespace(artifacts=None)

    result = pipeline_points.mine_blast_radius(
        project,  # type: ignore[arg-type]
        spec,  # type: ignore[arg-type]
        [],
        code_revision="test",
        inputs=(),
    )

    assert result is not None
    ranked = {row.point_id: row for row in result[1]}
    assert set(ranked) == set(points)
    assert ranked["p2"].score is None
    assert all(ranked[key].score is not None for key in ("p0", "p1", "p3"))
    assert state["peak"] == 2

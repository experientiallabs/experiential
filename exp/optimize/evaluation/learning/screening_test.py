"""Screening decisions and the fitted threshold, without a project or provider.

The end-to-end path (learning, durable dispatch, publication) is covered by ``lessons_test``.
"""

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
from exp.optimize.evaluation.learning.screening import (
    Deliveries,
    LearningMiss,
    LessonScreeningSpec,
    MissScreening,
    _probabilities,
    fit_threshold,
    screening_request,
    screening_score,
)


def miss(name: str, tool: str, reply: JsonValue = None) -> LearningMiss:
    """A miss of ``tool`` with a hypothesis, identified by ``name``."""
    ref = ArtifactInput(artifact_id=name, sha256="0" * 64)
    return LearningMiss(
        record=ref,
        run_id="run",
        trace_id="trace",
        task_lineage="lineage",
        source=ref,
        action=[{"name": tool, "arguments": {}}],
        predicted_reply={"status": "pending"},
        observed_reply=reply,
        investigation={"hypothesis": f"{tool} replies with its status"},
    )


def screened(name: str, tool: str, behavioural: float) -> MissScreening:
    """An answered screening with one behavioural probability."""
    return MissScreening(
        miss=miss(name, tool),
        status="answered",
        behavioural=behavioural,
        covered=0.0,
        supported=1.0,
    )


def test_fit_selects_the_threshold_with_the_best_same_tool_delivered_outcome() -> None:
    """Deliveries to another tool's prediction are not evidence; ties keep the higher value."""
    rows = [screened("rule", "read", 0.8), screened("data", "list", 0.3)]
    deliveries: Deliveries = {
        "rule": [("read", True), ("read", True), ("list", False)],
        "data": [("list", False), ("list", False), ("read", True)],
    }
    fit = fit_threshold(rows, deliveries, thresholds=(0.1, 0.5, 0.9), lineage="m", version=3)
    assert fit.objective == (0, 2, 0) and fit.retained == (2, 1, 0)
    assert fit.selected == 0.5 and fit.status == "fitted" and fit.deliveries == 4


def test_without_delivery_evidence_the_most_conservative_threshold_is_kept() -> None:
    """Unknown usefulness screens nothing in."""
    fit = fit_threshold(
        [screened("rule", "read", 0.8)], {}, thresholds=(0.1, 0.5), lineage="m", version=1
    )
    assert fit.selected == 0.5 and fit.status == "no-delivery-evidence"


def test_unanswered_screenings_have_no_score_and_are_not_fitted() -> None:
    """An unknown outcome stays unknown: it is neither scored nor credited."""
    unknown = MissScreening(miss=miss("lost", "read"), status="unknown", reason="lost")
    fit = fit_threshold(
        [unknown], {"lost": [("read", True)]}, thresholds=(0.1,), lineage="m", version=1
    )
    assert unknown.score is None and fit.deliveries == 0


def test_requests_cut_long_replies_and_answers_must_be_three_probabilities() -> None:
    """A long reply is a labelled prefix; any other answer shape is refused."""
    request = screening_request(miss("m", "read", "x" * 5_000), [], maximum_reply_bytes=300)
    state = request.state
    assert isinstance(state, dict)
    shown = state["miss"]["recorded_reply"]  # type: ignore[index]
    assert isinstance(shown, dict) and shown["total_bytes"] == 5_002
    assert set(request.questions) == {"behavioural", "covered", "supported"}
    good: JsonObject = {
        "answers": {key: {"type": "noul", "noul": 0.5} for key in request.questions}
    }
    assert _probabilities(good) == {"behavioural": 0.5, "covered": 0.5, "supported": 0.5}
    broken_answers: tuple[JsonObject, ...] = (
        {"answers": {"behavioural": {"type": "noul", "noul": 0.5}}},
        {"answers": {key: {"type": "score", "score": 1} for key in request.questions}},
        {"answers": {key: {"type": "noul", "noul": 1.5} for key in request.questions}},
    )
    for broken in broken_answers:
        assert _probabilities(broken) is None


def test_the_threshold_grid_must_increase_inside_the_unit_interval() -> None:
    """A registered grid is ordered and excludes 1, which would screen everything out."""
    with pytest.raises(ValueError, match="increase strictly"):
        LessonScreeningSpec(resolution="screen-lessons", thresholds=(0.5, 0.2))
    with pytest.raises(ValueError, match="increase strictly"):
        LessonScreeningSpec(resolution="screen-lessons", thresholds=(1.0,))


def test_the_score_is_reusable_times_supported_times_uncovered() -> None:
    """One formula scores both learning misses and agent lessons."""
    assert screening_score(0.5, 0.0, 0.8) == pytest.approx(0.4)
    assert screening_score(1.0, 1.0, 1.0) == 0.0

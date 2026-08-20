"""Held-out disagreement selection and labeling tests."""

from __future__ import annotations

from exp.common.judging.calibration_metrics import OutOfFoldPrediction, worst_disagreements


def _prediction(label_id: str, human_score: int, calibrated_score: float) -> OutOfFoldPrediction:
    """Return one held-out prediction with the given human and calibrated scores.

    Args:
        label_id: Unique label identity used for deterministic tie ordering.
        human_score: Explicit human score for the rollout.
        calibrated_score: Calibrated judge score for the same rollout.

    Returns:
        One valid held-out prediction fixture.
    """
    return OutOfFoldPrediction(
        label_id=label_id,
        rollout_id=f"rollout-{label_id}",
        lineage_id=f"lineage-{label_id}",
        dimension_id="task-success",
        fold_index=0,
        raw_score=human_score,
        human_score=human_score,
        calibrated_score=calibrated_score,
        absolute_error=abs(calibrated_score - human_score),
        optimistic_error=max(calibrated_score - human_score, 0.0),
    )


def test_exact_agreements_are_never_reported_as_disagreements() -> None:
    """Predictions whose calibrated score equals the human score are excluded."""
    predictions = (
        _prediction("label-a", 1, 1.0),
        _prediction("label-b", 0, 0.0),
        _prediction("label-c", 1, 0.25),
        _prediction("label-d", 0, 0.75),
    )

    result = worst_disagreements(predictions)

    assert tuple(item.prediction.label_id for item in result) == ("label-c", "label-d")
    assert tuple(item.direction for item in result) == ("pessimistic", "optimistic")


def test_all_exact_agreements_produce_no_disagreements() -> None:
    """A fully agreeing sample reports zero disagreements."""
    predictions = tuple(_prediction(f"label-{index}", 1, 1.0) for index in range(5))

    assert worst_disagreements(predictions) == ()


def test_disagreements_keep_the_ten_largest_nonzero_errors() -> None:
    """Only the ten largest nonzero errors survive, in deterministic order."""
    predictions = tuple(
        _prediction(f"label-{index:02d}", 0, (index + 1) / 100.0) for index in range(12)
    )

    result = worst_disagreements(predictions)

    assert len(result) == 10
    assert result[0].prediction.label_id == "label-11"
    assert all(item.direction == "optimistic" for item in result)
    assert all(item.prediction.absolute_error > 0 for item in result)

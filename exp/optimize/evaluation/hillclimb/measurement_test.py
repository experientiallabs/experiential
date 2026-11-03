"""Paired gains, rep-half floors and the stage rules on hand-written outcomes."""

import pytest

from exp.optimize.evaluation.hillclimb.measurement import (
    ArmOutcomes,
    PairedEstimate,
    _decide,
    halves,
    pass_rate,
    require_headroom,
    unknown_count,
)


def _estimate(value: float | None, interval: tuple[float, float] | None) -> PairedEstimate:
    """A paired estimate with only the fields the rules read."""
    return PairedEstimate(
        method="case_bootstrap",
        units=4,
        estimate=value,
        interval=interval,
        unit_differences={},
    )


def test_halves_split_each_lineage_and_unknowns_stay_counted() -> None:
    """A/A halves pair a lineage's first and second repeats; unknown slots are not dropped."""
    outcomes: ArmOutcomes = {"a": (1, 0, 1, None), "b": (0, 0, 1, 1)}
    first, second = halves(outcomes)
    assert first == {"a": (1, 0), "b": (0, 0)} and second == {"a": (1, None), "b": (1, 1)}
    assert unknown_count(outcomes) == 1
    assert pass_rate(outcomes) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_screen_needs_a_gain_above_the_floor_and_confirm_also_excludes_zero() -> None:
    """Screening compares the estimate; confirmation also needs the interval and every outcome."""
    assert _decide("screen", _estimate(0.2, (-0.1, 0.5)), 0.1, 3)[0]
    assert not _decide("screen", _estimate(0.1, (0.0, 0.2)), 0.1, 0)[0]
    assert not _decide("confirm", _estimate(0.2, (-0.1, 0.5)), 0.1, 0)[0]
    assert not _decide("confirm", _estimate(0.2, (0.1, 0.3)), 0.25, 0)[0]
    assert not _decide("confirm", _estimate(0.4, (0.1, 0.6)), 0.1, 1)[0]
    passed, reason = _decide("final", _estimate(0.4, (0.1, 0.6)), 0.1, 0)
    assert passed and "excludes zero" in reason
    assert not _decide("confirm", _estimate(None, None), None, 0)[0]


def test_a_saturated_or_unknown_train_set_is_refused() -> None:
    """All-pass, all-fail and no-known outcomes leave no measurable headroom."""
    headroom: ArmOutcomes = {"a": (1, 0), "b": (1, 1)}
    require_headroom(headroom)
    cases: tuple[tuple[ArmOutcomes, str], ...] = (
        ({"a": (1, 1), "b": (1, None)}, "passes every"),
        ({"a": (0, 0), "b": (0, 0)}, "fails every"),
        ({"a": (None, None)}, "no train outcome"),
    )
    for outcomes, message in cases:
        with pytest.raises(ValueError, match=message):
            require_headroom(outcomes)

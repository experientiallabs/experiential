"""Saturation follows the registered rule, and bands use the actor as the independent unit."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.uncertainty import wilson_interval
from exp.optimize.evaluation.factory.difficulty import (
    PassBand,
    SaturationRule,
    panel_difficulty,
)


def _metrics(*rows: tuple[int, int, int]) -> JsonObject:
    """Saved panel metrics with (planned, successes, unknown) per actor."""
    return {
        "cases": 1,
        "actors": [
            {"alias": f"actor-{i}", "planned": planned, "successes": passes, "unknown": unknown}
            for i, (planned, passes, unknown) in enumerate(rows)
        ],
    }


@pytest.mark.parametrize(
    "rows,rule,outcome",
    [
        (((4, 4, 0), (4, 4, 0)), SaturationRule(), "saturated"),
        (((4, 4, 0), (4, 3, 0)), SaturationRule(), "separating"),
        (((4, 0, 0), (4, 0, 0)), SaturationRule(), "all_fail"),
        (((4, 3, 1), (4, 4, 0)), SaturationRule(), "undetermined"),
        (((4, 3, 1), (4, 4, 0)), SaturationRule(maximum_unknown_fraction=0.25), "saturated"),
        (((4, 0, 4), (4, 4, 0)), SaturationRule(maximum_unknown_fraction=0.5), "undetermined"),
        (((4, 3, 0), (4, 4, 0)), SaturationRule(minimum_pass_fraction=0.75), "saturated"),
    ],
)
def test_registered_rule_decides_saturation(
    rows: tuple[tuple[int, int, int], ...], rule: SaturationRule, outcome: str
) -> None:
    """Unknown attempts never count as passes; their share is a registered limit."""
    assert panel_difficulty(_metrics(*rows), rule, PassBand()).outcome == outcome


def test_band_reports_actor_unit_interval_and_missing_bounds() -> None:
    """Repeats are averaged per actor first; unknowns stay bounds, not interval width."""
    difficulty = panel_difficulty(
        _metrics((4, 1, 0), (4, 2, 2), (4, 4, 0)),
        SaturationRule(maximum_unknown_fraction=0.5),
        PassBand(lower=0.3, upper=0.7),
    )
    estimate = (0.25 + 1.0 + 1.0) / 3
    assert difficulty.pass_rate == pytest.approx(estimate)
    assert difficulty.interval == pytest.approx(wilson_interval(estimate, 3))
    assert difficulty.missing_bounds == pytest.approx((7 / 12, 9 / 12))
    assert difficulty.band == "above" and difficulty.interval_within_band is False
    assert difficulty.outcome == "separating" and difficulty.unknown == 2


def test_multi_case_panels_and_empty_bands_are_refused() -> None:
    """A variant band is a one-case reading; a band must have width."""
    with pytest.raises(ValueError, match="one-case"):
        panel_difficulty({**_metrics((1, 1, 0)), "cases": 2}, SaturationRule(), PassBand())
    with pytest.raises(ValueError, match="below its upper"):
        PassBand(lower=0.5, upper=0.5)

"""Tests for Wilson intervals at the independent unit count and missing-outcome bounds."""

import pytest

from exp.common.evaluations.uncertainty import (
    Z95_ONE_SIDED,
    missing_outcome_bounds,
    paired_bootstrap_summary,
    proportion_summary,
    unit_difference_summary,
    unit_mean_summary,
    wilson_interval,
)


def test_wilson_interval_matches_the_reference_calculation() -> None:
    """Twelve of sixteen independent trials give roughly 50.5 to 89.8 percent."""
    low, high = wilson_interval(12 / 16, 16)
    assert low == pytest.approx(0.505, abs=5e-4)
    assert high == pytest.approx(0.898, abs=5e-4)
    assert wilson_interval(0.0, 4)[0] == 0.0 and wilson_interval(1.0, 4)[1] == 1.0
    one_sided = wilson_interval(0.0, 10, z=Z95_ONE_SIDED)[1]
    assert one_sided == pytest.approx(0.2129, abs=5e-4)
    with pytest.raises(ValueError, match="at least one"):
        wilson_interval(0.5, 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        wilson_interval(1.5, 3)


def test_fewer_independent_units_than_trials_widens_the_interval() -> None:
    """Related trials are counted once per unit, not once per trial."""
    trials = proportion_summary(12, 16)
    clustered = proportion_summary(12, 16, units=4)
    assert trials["estimate"] == clustered["estimate"] == 0.75
    assert clustered["independent_units"] == 4
    trial_interval, unit_interval = trials["wilson_95"], clustered["wilson_95"]
    assert isinstance(trial_interval, list) and isinstance(unit_interval, list)
    assert unit_interval[0] < trial_interval[0] and unit_interval[1] > trial_interval[1]
    assert proportion_summary(0, 0)["wilson_95"] is None
    with pytest.raises(ValueError, match="between one and the trial count"):
        proportion_summary(1, 2, units=3)


def test_unit_means_average_repeats_before_estimating_across_units() -> None:
    """Eight repeats of one checkpoint are one lineage, not eight independent tasks."""
    summary = unit_mean_summary(
        {"lineage-a": [1, 1, 1, 1, 1, 1, 1, 0], "lineage-b": [0, 0], "c": []}
    )
    assert summary["independent_units"] == 2 and summary["units_without_outcomes"] == 1
    assert summary["estimate"] == pytest.approx((7 / 8 + 0) / 2)
    interval = summary["wilson_95"]
    assert isinstance(interval, list)
    assert interval == list(wilson_interval((7 / 8) / 2, 2))
    assert unit_mean_summary({})["estimate"] is None


def test_unit_differences_pair_units_before_averaging_and_rescale_the_interval() -> None:
    """Differences are taken within units, weigh units equally and keep the [-1, 1] scale."""
    summary = unit_difference_summary(
        {"a": [1, 1, 1, 0], "b": [0, 0], "c": [1]},
        {"a": [1, 1], "b": [1], "d": [0]},
    )
    assert summary["unit_differences"] == {"a": pytest.approx(-0.25), "b": -1.0}
    assert summary["independent_units"] == 2 and summary["units_without_paired_outcomes"] == 2
    assert summary["estimate"] == pytest.approx(-0.625)
    low, high = wilson_interval((1 - 0.625) / 2, 2)
    assert summary["wilson_95"] == [pytest.approx(2 * low - 1), pytest.approx(2 * high - 1)]
    identical = unit_difference_summary({"a": [1]}, {"a": [1]})
    interval = identical["wilson_95"]
    assert isinstance(interval, list) and interval[0] < 0
    assert interval[0] == pytest.approx(-interval[1])
    assert unit_difference_summary({"a": []}, {"a": [1]})["estimate"] is None


def test_missing_outcomes_are_bounds_not_intervals() -> None:
    """S/N through (S+U)/N keeps every planned slot in the denominator."""
    assert missing_outcome_bounds(5, 3, 10) == [0.5, 0.8]
    assert missing_outcome_bounds(0, 0, 0) is None
    with pytest.raises(ValueError, match="planned denominator"):
        missing_outcome_bounds(5, 6, 10)


def test_paired_bootstrap_resamples_whole_units_reproducibly() -> None:
    """Repeats average within a unit; a seeded resample gives the same percentile interval."""
    first = {"a": [1, 1], "b": [1, 0], "c": [1, 1], "d": [0, 0], "e": []}
    second = {"a": [0, 0], "b": [0, 0], "c": [1, 1], "d": [0, 0]}
    summary = paired_bootstrap_summary(first, second, resamples=500, seed=3)
    assert summary["independent_units"] == 4
    assert summary["units_without_paired_outcomes"] == 1
    assert summary["unit_differences"] == {"a": 1.0, "b": 0.5, "c": 0.0, "d": 0.0}
    assert summary["estimate"] == 0.375
    interval = summary["bootstrap_95"]
    assert isinstance(interval, list) and 0 <= interval[0] < 0.375 < interval[1] <= 1
    assert paired_bootstrap_summary(first, second, resamples=500, seed=3) == summary
    constant = paired_bootstrap_summary({"a": [1], "b": [1]}, {"a": [0], "b": [0]})
    assert constant["bootstrap_95"] == [1.0, 1.0]
    empty = paired_bootstrap_summary({"a": [1]}, {"b": [0]})
    assert empty["estimate"] is None and empty["bootstrap_95"] is None


def test_paired_bootstrap_refuses_too_few_resamples() -> None:
    """A handful of resamples cannot estimate a 95 percent interval."""
    with pytest.raises(ValueError, match="at least 100 resamples"):
        paired_bootstrap_summary({"a": [1]}, {"a": [0]}, resamples=10)

"""Binomial uncertainty over independent units, kept apart from repeated draws of one unit.

Evaluation reports state a proportion, the number of independent units behind it and a Wilson
score interval at that unit count. A unit is whatever the report declares independent: a task
lineage, a lineage component or a case. Repeated attempts on one checkpoint, related variants of
one source and several critics forecasting the same task are not separate units.

When a unit contributes a fraction rather than one binary outcome (several repeats, several
related cases), the estimate is the equal-weight mean over units and the interval is the Wilson
interval at the unit count. A [0, 1] unit mean with mean p has variance at most p(1-p), so this
bounds the between-unit spread by the Bernoulli case instead of treating every repeat as new
evidence. Planned slots with unknown outcomes are reported as bounds, never as intervals.

A paired difference can instead be summarized by a case bootstrap: whole units are resampled with
replacement and the percentile interval of the resampled mean difference is reported. It assumes
nothing about the outcome scale, but with few units it is narrower than the Wilson bound and
should be read alongside the unit count.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from exp.common.core.artifacts import JsonObject

Z95 = 1.959963984540054  # two-sided 95 percent normal quantile
Z95_ONE_SIDED = 1.6448536269514722  # one-sided 95 percent normal quantile


def wilson_interval(estimate: float, units: int, *, z: float = Z95) -> tuple[float, float]:
    """Return the Wilson score interval for a proportion measured over independent units.

    Args:
        estimate: Observed proportion in [0, 1].
        units: Independent units behind the proportion; at least one.
        z: Normal quantile; ``Z95`` gives a two-sided 95 percent interval, ``Z95_ONE_SIDED``
            gives the one-sided 95 percent bound as either endpoint.

    Returns:
        Lower and upper interval endpoints in [0, 1].

    Raises:
        ValueError: The estimate is outside [0, 1] or there are no units.
    """
    if units < 1:
        raise ValueError("a Wilson interval needs at least one independent unit")
    if not 0 <= estimate <= 1 or not math.isfinite(estimate):
        raise ValueError("a Wilson interval needs a proportion in [0, 1]")
    scale = z * z / units
    center = (estimate + scale / 2) / (1 + scale)
    margin = (
        z * math.sqrt(estimate * (1 - estimate) / units + z * z / (4 * units * units)) / (1 + scale)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def proportion_summary(successes: int, trials: int, *, units: int | None = None) -> JsonObject:
    """Summarize a counted proportion with its Wilson interval at the independent unit count.

    Args:
        successes: Counted successes.
        trials: Counted trials, the denominator of the estimate.
        units: Independent units behind the trials; None treats every trial as independent.
            Fewer units than trials widens the interval to the unit count.

    Returns:
        The counts, the estimate, the unit count and a two-sided 95 percent Wilson interval;
        the estimate and interval are None without trials.
    """
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and the trial count")
    independent = trials if units is None else units
    if trials and not 1 <= independent <= trials:
        raise ValueError("independent units must lie between one and the trial count")
    estimate = successes / trials if trials else None
    return {
        "successes": successes,
        "trials": trials,
        "estimate": estimate,
        "independent_units": independent if trials else 0,
        "wilson_95": None if estimate is None else list(wilson_interval(estimate, independent)),
    }


def _unit_means(outcomes: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Average each unit's known outcomes, skipping units without any, in unit order."""
    return {unit: sum(values) / len(values) for unit, values in sorted(outcomes.items()) if values}


def unit_mean_summary(outcomes: Mapping[str, Sequence[float]]) -> JsonObject:
    """Average repeats within each independent unit, then estimate across units.

    Args:
        outcomes: Known outcomes in [0, 1] keyed by independent unit; units without known
            outcomes are skipped and counted.

    Returns:
        The equal-weight unit mean, the per-unit means, the unit count and a two-sided 95
        percent Wilson interval at that count, or None estimates without a measured unit.
    """
    means = _unit_means(outcomes)
    estimate = sum(means.values()) / len(means) if means else None
    return {
        "independent_units": len(means),
        "units_without_outcomes": len(outcomes) - len(means),
        "unit_means": means,
        "estimate": estimate,
        "wilson_95": None if estimate is None else list(wilson_interval(estimate, len(means))),
    }


def unit_difference_summary(
    first: Mapping[str, Sequence[float]], second: Mapping[str, Sequence[float]]
) -> JsonObject:
    """Average repeats within each unit under two paired conditions, then estimate the difference.

    A unit enters when both conditions have known outcomes for it. Its difference of unit means
    lies in [-1, 1], so ``(d + 1) / 2`` is a [0, 1] unit mean and the Wilson interval at the unit
    count bounds it exactly as in ``unit_mean_summary``; the endpoints are mapped back to the
    difference scale. Pairing by unit keeps the comparison within units instead of treating the
    two conditions as independent samples.

    Args:
        first: Known outcomes in [0, 1] keyed by independent unit, the minuend condition.
        second: Known outcomes in [0, 1] keyed by the same units, the subtrahend condition.

    Returns:
        The equal-weight mean of per-unit differences, the per-unit differences, the paired unit
        count, the units missing an outcome under either condition and a two-sided 95 percent
        interval on the difference scale, or None estimates without a paired unit.
    """
    left, right = _unit_means(first), _unit_means(second)
    differences = {unit: left[unit] - right[unit] for unit in sorted(left.keys() & right.keys())}
    units = len(differences)
    estimate = sum(differences.values()) / units if units else None
    interval = (
        None
        if estimate is None
        else [2 * bound - 1 for bound in wilson_interval((estimate + 1) / 2, units)]
    )
    return {
        "independent_units": units,
        "units_without_paired_outcomes": len(first.keys() | second.keys()) - units,
        "unit_differences": differences,
        "estimate": estimate,
        "wilson_95": interval,
    }


def paired_bootstrap_summary(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> JsonObject:
    """Bootstrap the mean paired difference by resampling whole independent units.

    Repeats are averaged within each unit first, exactly as in ``unit_difference_summary``, so a
    unit is resampled with all of its repeats. The interval is the 2.5 and 97.5 percentiles of the
    resampled mean difference, drawn from a seeded generator so the summary reproduces.

    Args:
        first: Known outcomes in [0, 1] keyed by independent unit, the minuend condition.
        second: Known outcomes in [0, 1] keyed by the same units, the subtrahend condition.
        resamples: Bootstrap resamples; at least 100.
        seed: Generator seed recorded with the summary.

    Returns:
        The equal-weight mean of per-unit differences, the per-unit differences, the paired unit
        count, the units missing an outcome under either condition, the resample count and seed,
        and a two-sided 95 percent percentile interval, or None estimates without a paired unit.

    Raises:
        ValueError: Fewer than 100 resamples were requested.
    """
    if resamples < 100:
        raise ValueError("a paired bootstrap needs at least 100 resamples")
    left, right = _unit_means(first), _unit_means(second)
    differences = {unit: left[unit] - right[unit] for unit in sorted(left.keys() & right.keys())}
    units = len(differences)
    estimate = sum(differences.values()) / units if units else None
    interval: list[float] | None = None
    if units:
        values = list(differences.values())
        generator = random.Random(seed)
        means = sorted(sum(generator.choices(values, k=units)) / units for _ in range(resamples))
        interval = [
            means[math.floor(0.025 * (resamples - 1))],
            means[math.ceil(0.975 * (resamples - 1))],
        ]
    return {
        "independent_units": units,
        "units_without_paired_outcomes": len(first.keys() | second.keys()) - units,
        "unit_differences": differences,
        "estimate": estimate,
        "resamples": resamples,
        "seed": seed,
        "bootstrap_95": interval,
    }


def missing_outcome_bounds(successes: int, unknown: int, planned: int) -> list[float] | None:
    """Bound a success fraction over every planned slot under each possible missing outcome.

    Args:
        successes: Known successes.
        unknown: Planned slots whose outcome is unknown.
        planned: Every planned slot, the fixed denominator.

    Returns:
        ``[S/N, (S+U)/N]``, or None without planned slots. These bound missingness; they are
        not sampling uncertainty and never replace an interval at the independent unit count.
    """
    if planned < 0 or not 0 <= successes <= successes + unknown <= planned:
        raise ValueError("known successes and unknown slots must fit the planned denominator")
    if not planned:
        return None
    return [successes / planned, (successes + unknown) / planned]

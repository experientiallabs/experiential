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
"""

from __future__ import annotations

import math
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


def unit_mean_summary(outcomes: Mapping[str, Sequence[float]]) -> JsonObject:
    """Average repeats within each independent unit, then estimate across units.

    Args:
        outcomes: Known outcomes in [0, 1] keyed by independent unit; units without known
            outcomes are skipped and counted.

    Returns:
        The equal-weight unit mean, the per-unit means, the unit count and a two-sided 95
        percent Wilson interval at that count, or None estimates without a measured unit.
    """
    means = {unit: sum(values) / len(values) for unit, values in sorted(outcomes.items()) if values}
    estimate = sum(means.values()) / len(means) if means else None
    return {
        "independent_units": len(means),
        "units_without_outcomes": len(outcomes) - len(means),
        "unit_means": means,
        "estimate": estimate,
        "wilson_95": None if estimate is None else list(wilson_interval(estimate, len(means))),
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

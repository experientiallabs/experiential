"""Registered saturation and pass-rate band readings of one variant's saved actor panel.

A variant is one case, so repeated attempts are not independent units. The pass rate is the
equal-weight mean over panel actors after averaging each actor's known attempts, with a Wilson
interval at the actor count (``exp.common.evaluations.uncertainty``). Unknown attempts stay
explicit bounds. The band is reported for every variant; it never selects or tunes a verifier.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.evaluations.uncertainty import missing_outcome_bounds, unit_mean_summary

PanelOutcome = Literal["saturated", "separating", "all_fail", "undetermined"]


class SaturationRule(ContractModel):
    """When a strong panel has solved a variant, registered before any panel runs.

    Attributes:
        minimum_pass_fraction: Every actor's known pass fraction must reach this for saturation.
        maximum_unknown_fraction: More unknown attempts than this leave the panel undetermined.
        minimum_known_per_actor: Fewer known attempts for any actor leave it undetermined.
    """

    minimum_pass_fraction: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    maximum_unknown_fraction: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    minimum_known_per_actor: int = Field(default=1, ge=1, le=100, strict=True)


class PassBand(ContractModel):
    """Target pass-rate band, reported per variant and never used to change a verifier.

    Attributes:
        lower: Lowest pass rate inside the band.
        upper: Highest pass rate inside the band.
    """

    lower: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    upper: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Reject an empty band."""
        if self.lower >= self.upper:
            raise ValueError("pass band lower bound must be below its upper bound")
        return self


class ActorCount(ContractModel):
    """One actor's counted attempts on the variant.

    Attributes:
        alias: Panel actor alias.
        planned: Planned attempts.
        passes: Known passing attempts.
        unknown: Attempts without a known outcome.
    """

    alias: str
    planned: int = Field(ge=0)
    passes: int = Field(ge=0)
    unknown: int = Field(ge=0)


class PanelDifficulty(ContractModel):
    """Saturation outcome and pass-rate band of one variant under its registered rules.

    Attributes:
        outcome: Saturated, separating, all known attempts failing, or undetermined.
        actors: Per-actor counts in panel order.
        planned: Planned attempts over all actors.
        unknown: Attempts without a known outcome.
        pass_rate: Equal-weight mean of per-actor known pass fractions.
        interval: Two-sided 95 percent Wilson interval at the measured actor count.
        missing_bounds: Pooled pass fraction with every unknown failing, then passing.
        band: Position of ``pass_rate`` against the registered band.
        interval_within_band: Whether the whole interval lies inside the band.
        unit: The declared independent unit behind the interval.
    """

    outcome: PanelOutcome
    actors: tuple[ActorCount, ...]
    planned: int
    unknown: int
    pass_rate: float | None
    interval: tuple[float, float] | None
    missing_bounds: tuple[float, float] | None
    band: Literal["below", "within", "above"] | None
    interval_within_band: bool | None
    unit: str = "panel actor; each actor's repeated attempts on this one case are averaged first"


def _int(row: JsonObject, key: str) -> int:
    """Read one counted field from saved panel metrics."""
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("actor panel metrics lack a counted " + key)
    return value


def actor_counts(metrics: JsonObject) -> tuple[ActorCount, ...]:
    """Read per-actor counts from saved ``panel_metrics`` output."""
    rows = metrics.get("actors")
    if not isinstance(rows, list) or not rows:
        raise ValueError("actor panel metrics lack actor rows")
    counts = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("alias"), str):
            raise ValueError("actor panel metrics contain a malformed actor row")
        counts.append(
            ActorCount(
                alias=str(row["alias"]),
                planned=_int(row, "planned"),
                passes=_int(row, "successes"),
                unknown=_int(row, "unknown"),
            )
        )
    return tuple(counts)


def _outcome(actors: tuple[ActorCount, ...], rule: SaturationRule) -> PanelOutcome:
    """Apply the registered saturation rule to counted attempts."""
    planned = sum(row.planned for row in actors)
    unknown = sum(row.unknown for row in actors)
    if (
        not planned
        or unknown / planned > rule.maximum_unknown_fraction
        or any(row.planned - row.unknown < rule.minimum_known_per_actor for row in actors)
    ):
        return "undetermined"
    if all(
        row.passes / (row.planned - row.unknown) >= rule.minimum_pass_fraction for row in actors
    ):
        return "saturated"
    if not any(row.passes for row in actors):
        return "all_fail"
    return "separating"


def panel_difficulty(metrics: JsonObject, rule: SaturationRule, band: PassBand) -> PanelDifficulty:
    """Classify a saved one-case panel and report its pass-rate band with intervals.

    Args:
        metrics: Saved ``ActorPanelReport.metrics`` for a panel with exactly one case.
        rule: Registered saturation rule.
        band: Registered reporting band.

    Returns:
        Outcome, per-actor counts, the actor-unit pass rate and interval, and missing bounds.
    """
    if metrics.get("cases") != 1:
        raise ValueError("variant difficulty requires a one-case actor panel")
    actors = actor_counts(metrics)
    planned = sum(row.planned for row in actors)
    unknown = sum(row.unknown for row in actors)
    summary = unit_mean_summary(
        {
            row.alias: [1.0] * row.passes + [0.0] * (row.planned - row.unknown - row.passes)
            for row in actors
        }
    )
    estimate = summary["estimate"]
    wilson = summary["wilson_95"]
    rate = float(estimate) if isinstance(estimate, int | float) else None
    interval = (
        (float(wilson[0]), float(wilson[1]))
        if isinstance(wilson, list) and all(isinstance(v, int | float) for v in wilson)
        else None
    )
    bounds = missing_outcome_bounds(sum(row.passes for row in actors), unknown, planned)
    return PanelDifficulty(
        outcome=_outcome(actors, rule),
        actors=actors,
        planned=planned,
        unknown=unknown,
        pass_rate=rate,
        interval=interval,
        missing_bounds=None if bounds is None else (bounds[0], bounds[1]),
        band=None
        if rate is None
        else "below"
        if rate < band.lower
        else "above"
        if rate > band.upper
        else "within",
        interval_within_band=None
        if interval is None
        else band.lower <= interval[0] and interval[1] <= band.upper,
    )

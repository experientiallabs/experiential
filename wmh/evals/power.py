"""Finite Monte Carlo gate for a preregistered paired-confirmation simulation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from scipy.stats import beta

PAIRED_POWER_GATE_VERSION: Literal["2"] = "2"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_CP_MIN_DECIMAL_PRECISION = 120
_CP_MAX_OUTWARD_STEPS = 65_536


class PairedPowerGateDesign(BaseModel):
    """Frozen operating-characteristic thresholds for one locked simulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["2"] = PAIRED_POWER_GATE_VERSION
    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    target_effect: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    maximum_type_i_error: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_power: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    monte_carlo_alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    replications_per_scenario: StrictInt = Field(ge=1)

    @field_validator(
        "target_effect",
        "maximum_type_i_error",
        "minimum_power",
        "monte_carlo_alpha",
        mode="before",
    )
    @classmethod
    def _reject_boolean_thresholds(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("paired power thresholds cannot be boolean")
        return value

    @field_validator("replications_per_scenario", mode="before")
    @classmethod
    def _reject_boolean_replications(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired power replications cannot be boolean")
        return value

    @property
    def digest(self) -> str:
        """Return the canonical identity of this complete gate."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedPowerTrial(BaseModel):
    """One simulator replicate projected to the frozen primary decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_evaluation_design_digest: str = Field(pattern=_DIGEST_PATTERN)
    scenario: Literal["weak-null", "target-alternative"]
    replicate: StrictInt = Field(ge=1)
    primary_passed: StrictBool

    @field_validator("replicate", mode="before")
    @classmethod
    def _reject_boolean_replicates(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired power replicate identities cannot be boolean")
        return value


class PairedPowerGateReport(BaseModel):
    """Reload-safe Monte Carlo evidence, bounds, and decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_version: Literal["2"]
    design: PairedPowerGateDesign
    trial_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    null_rejections: StrictInt = Field(ge=0)
    target_rejections: StrictInt = Field(ge=0)
    empirical_type_i_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    type_i_error_upper_bound: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    empirical_power: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    power_lower_bound: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    type_i_error_passed: bool
    power_passed: bool
    report_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_derived_evidence(self) -> Self:
        count = self.design.replications_per_scenario
        if self.null_rejections > count or self.target_rejections > count:
            raise ValueError("paired power rejection counts cannot exceed frozen replications")
        expected_type_i_error = self.null_rejections / count
        expected_power = self.target_rejections / count
        expected_type_i_upper = _binomial_upper_bound(
            self.null_rejections,
            count,
            alpha=self.design.monte_carlo_alpha,
        )
        expected_power_lower = _binomial_lower_bound(
            self.target_rejections,
            count,
            alpha=self.design.monte_carlo_alpha,
        )
        derived = {
            "empirical_type_i_error": expected_type_i_error,
            "type_i_error_upper_bound": expected_type_i_upper,
            "empirical_power": expected_power,
            "power_lower_bound": expected_power_lower,
            "type_i_error_passed": (expected_type_i_upper <= self.design.maximum_type_i_error),
            "power_passed": expected_power_lower >= self.design.minimum_power,
        }
        for field, expected in derived.items():
            if getattr(self, field) != expected:
                raise ValueError(f"paired power report {field} differs from its frozen evidence")
        expected_digest = _canonical_digest(self.model_dump(mode="json", exclude={"report_digest"}))
        if self.report_digest != expected_digest:
            raise ValueError("paired power report digest differs from its frozen evidence")
        return self

    @property
    def passed(self) -> bool:
        """Return whether both preregistered operating-characteristic gates pass."""
        return self.type_i_error_passed and self.power_passed

    @property
    def digest(self) -> str:
        """Return the canonical identity binding the design, evidence, and decisions."""
        return self.report_digest


def evaluate_paired_power_gate(
    design: PairedPowerGateDesign,
    trials: tuple[PairedPowerTrial, ...],
) -> PairedPowerGateReport:
    """Evaluate complete locked null and target simulations without optional stopping.

    The simulator, its data-generating assumptions, seeds, exact paired analysis,
    and mapping from a replicate to ``primary_passed`` live behind the frozen
    ``simulation_design_digest``. The separate paired-evaluation digest binds the
    exact roster, lane attempts, e-value bets, and observed floor exercised by the
    simulator. This gate rejects digest drift, duplicate or missing replicate
    identities, and extra replicates. It uses one-sided exact Clopper-Pearson bounds
    at the preregistered Monte Carlo alpha. Passing supports only the design's frozen
    target effect and assumptions; it does not establish power for an untested
    effect size.
    """
    expected_replicates = tuple(range(1, design.replications_per_scenario + 1))
    expected_scenarios = ("weak-null", "target-alternative")
    if any(trial.simulation_design_digest != design.simulation_design_digest for trial in trials):
        raise ValueError("paired power trial simulation design digest differs from the gate")
    if any(
        trial.paired_evaluation_design_digest != design.paired_evaluation_design_digest
        for trial in trials
    ):
        raise ValueError("paired power trial evaluation design digest differs from the gate")
    keys = [(trial.scenario, trial.replicate) for trial in trials]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"paired power trials contain duplicate replicate identities: {duplicates}"
        )
    observed = {
        scenario: tuple(sorted(trial.replicate for trial in trials if trial.scenario == scenario))
        for scenario in expected_scenarios
    }
    expected = {scenario: expected_replicates for scenario in expected_scenarios}
    if observed != expected:
        raise ValueError("paired power trials must exactly fill both frozen scenarios")

    null_rejections = sum(trial.primary_passed for trial in trials if trial.scenario == "weak-null")
    target_rejections = sum(
        trial.primary_passed for trial in trials if trial.scenario == "target-alternative"
    )
    canonical_trials = tuple(sorted(trials, key=lambda trial: (trial.scenario, trial.replicate)))
    trial_evidence_digest = _canonical_digest(
        [trial.model_dump(mode="json") for trial in canonical_trials]
    )
    count = design.replications_per_scenario
    empirical_type_i_error = null_rejections / count
    empirical_power = target_rejections / count
    type_i_error_upper_bound = _binomial_upper_bound(
        null_rejections,
        count,
        alpha=design.monte_carlo_alpha,
    )
    power_lower_bound = _binomial_lower_bound(
        target_rejections,
        count,
        alpha=design.monte_carlo_alpha,
    )
    report_payload: dict[str, JsonValue] = {
        "gate_version": PAIRED_POWER_GATE_VERSION,
        "design": design.model_dump(mode="json"),
        "trial_evidence_digest": trial_evidence_digest,
        "null_rejections": null_rejections,
        "target_rejections": target_rejections,
        "empirical_type_i_error": empirical_type_i_error,
        "type_i_error_upper_bound": type_i_error_upper_bound,
        "empirical_power": empirical_power,
        "power_lower_bound": power_lower_bound,
        "type_i_error_passed": type_i_error_upper_bound <= design.maximum_type_i_error,
        "power_passed": power_lower_bound >= design.minimum_power,
    }
    return PairedPowerGateReport.model_validate(
        {**report_payload, "report_digest": _canonical_digest(report_payload)}
    )


def _binomial_upper_bound(successes: int, trials: int, *, alpha: float) -> float:
    if successes == trials:
        return 1.0
    candidate = float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))
    threshold = Decimal.from_float(alpha)
    for _ in range(_CP_MAX_OUTWARD_STEPS):
        if _binomial_lower_tail_upper(successes, trials, probability=candidate) <= threshold:
            return candidate
        next_candidate = math.nextafter(candidate, math.inf)
        if next_candidate == candidate or next_candidate > 1.0:
            break
        candidate = next_candidate
    raise RuntimeError("could not certify an outward Clopper-Pearson upper bound")


def _binomial_lower_bound(successes: int, trials: int, *, alpha: float) -> float:
    if successes == 0:
        return 0.0
    candidate = float(beta.ppf(alpha, successes, trials - successes + 1))
    threshold = Decimal.from_float(alpha)
    for _ in range(_CP_MAX_OUTWARD_STEPS):
        if _binomial_upper_tail_upper(successes, trials, probability=candidate) <= threshold:
            return candidate
        next_candidate = math.nextafter(candidate, -math.inf)
        if next_candidate == candidate or next_candidate < 0.0:
            break
        candidate = next_candidate
    raise RuntimeError("could not certify an outward Clopper-Pearson lower bound")


def _binomial_lower_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
) -> Decimal:
    """Return a directed-rounding upper bound on ``P[X <= successes]``.

    The recurrence starts from ``q**n`` and advances through positive terms.
    Every division, multiplication, and addition rounds toward positive infinity,
    so the result is an upper bound. The local precision is large enough to make
    ``q = 1-p`` exact for the binary-float candidate.
    """
    if probability <= 0.0:
        return Decimal(1)
    if probability >= 1.0:
        return Decimal(0) if successes < trials else Decimal(1)
    probability_decimal = Decimal.from_float(probability)
    with localcontext() as context:
        context.prec = _cp_decimal_precision(probability_decimal)
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        if context.add(probability_decimal, complement) != one:
            raise RuntimeError("Clopper-Pearson decimal precision did not preserve q = 1-p")
        term = context.power(complement, trials)
        total = term
        odds_upper = context.divide(probability_decimal, complement)
        for observed in range(successes):
            count_ratio_upper = context.divide(
                Decimal(trials - observed),
                Decimal(observed + 1),
            )
            term = context.multiply(
                context.multiply(term, count_ratio_upper),
                odds_upper,
            )
            total = context.add(total, term)
        return +total


def _binomial_upper_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
) -> Decimal:
    """Return a directed-rounding upper bound on ``P[X >= successes]``.

    The recurrence starts from ``p**n`` and moves backward through positive
    terms, with every operation rounded toward positive infinity.
    """
    if probability <= 0.0:
        return Decimal(0) if successes > 0 else Decimal(1)
    if probability >= 1.0:
        return Decimal(1)
    probability_decimal = Decimal.from_float(probability)
    with localcontext() as context:
        context.prec = _cp_decimal_precision(probability_decimal)
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        if context.add(probability_decimal, complement) != one:
            raise RuntimeError("Clopper-Pearson decimal precision did not preserve q = 1-p")
        term = context.power(probability_decimal, trials)
        total = term
        reverse_odds_upper = context.divide(complement, probability_decimal)
        for observed in range(trials, successes, -1):
            count_ratio_upper = context.divide(
                Decimal(observed),
                Decimal(trials - observed + 1),
            )
            term = context.multiply(
                context.multiply(term, count_ratio_upper),
                reverse_odds_upper,
            )
            total = context.add(total, term)
        return +total


def _cp_decimal_precision(value: Decimal) -> int:
    """Return enough precision to form the exact unit complement of ``value``."""
    value_tuple = value.as_tuple()
    if not isinstance(value_tuple.exponent, int):
        raise RuntimeError("Clopper-Pearson certification requires a finite probability")
    exact_complement_digits = max(len(value_tuple.digits), 1 - value_tuple.exponent)
    return max(_CP_MIN_DECIMAL_PRECISION, exact_complement_digits + 8)


def _canonical_digest(value: JsonValue) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()

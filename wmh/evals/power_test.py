"""Tests for the preregistered paired-simulation power gate."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Decimal, localcontext

import pytest
from pydantic import ValidationError
from scipy.stats import beta

from wmh.evals.power import (
    PairedPowerGateDesign,
    PairedPowerGateReport,
    PairedPowerTrial,
    _binomial_lower_bound,
    _binomial_upper_bound,
    evaluate_paired_power_gate,
)

_SIMULATION_DIGEST = "sha256:" + "a" * 64
_PAIRED_DESIGN_DIGEST = "sha256:" + "c" * 64


def _design() -> PairedPowerGateDesign:
    # These fixture thresholds exercise the gate only. They are not the study's
    # target MDE or power claim; those must arrive in the locked simulation design.
    return PairedPowerGateDesign(
        simulation_design_digest=_SIMULATION_DIGEST,
        paired_evaluation_design_digest=_PAIRED_DESIGN_DIGEST,
        target_effect=0.2,
        maximum_type_i_error=0.05,
        minimum_power=0.9,
        monte_carlo_alpha=0.01,
        replications_per_scenario=100,
    )


def _trials(*, null_rejections: int, target_rejections: int) -> tuple[PairedPowerTrial, ...]:
    return tuple(
        PairedPowerTrial(
            simulation_design_digest=_SIMULATION_DIGEST,
            paired_evaluation_design_digest=_PAIRED_DESIGN_DIGEST,
            scenario=scenario,
            replicate=replicate,
            primary_passed=(
                replicate <= null_rejections
                if scenario == "weak-null"
                else replicate <= target_rejections
            ),
        )
        for scenario in ("weak-null", "target-alternative")
        for replicate in range(1, 101)
    )


def test_strong_locked_simulation_fixture_passes_both_power_gates() -> None:
    design = _design()
    report = evaluate_paired_power_gate(
        design,
        _trials(null_rejections=0, target_rejections=100),
    )

    assert report.design == design
    assert report.design.paired_evaluation_design_digest == _PAIRED_DESIGN_DIGEST
    assert report.trial_evidence_digest.startswith("sha256:")
    assert report.digest == report.report_digest
    assert report.empirical_type_i_error == 0.0
    assert report.type_i_error_upper_bound < design.maximum_type_i_error
    assert report.empirical_power == 1.0
    assert report.power_lower_bound > design.minimum_power
    assert report.type_i_error_passed is True
    assert report.power_passed is True
    assert report.passed is True
    assert PairedPowerGateReport.model_validate_json(report.model_dump_json()) == report
    assert (
        evaluate_paired_power_gate(
            design,
            tuple(reversed(_trials(null_rejections=0, target_rejections=100))),
        ).trial_evidence_digest
        == report.trial_evidence_digest
    )


def test_weak_locked_simulation_fixture_fails_without_a_power_claim() -> None:
    design = _design()
    report = evaluate_paired_power_gate(
        design,
        _trials(null_rejections=10, target_rejections=70),
    )

    assert report.type_i_error_upper_bound > design.maximum_type_i_error
    assert report.power_lower_bound < design.minimum_power
    assert report.type_i_error_passed is False
    assert report.power_passed is False
    assert report.passed is False


def test_power_gate_requires_the_exact_locked_replication_matrix() -> None:
    design = _design()
    trials = _trials(null_rejections=0, target_rejections=100)

    with pytest.raises(ValueError, match="exactly fill"):
        evaluate_paired_power_gate(design, trials[:-1])
    with pytest.raises(ValueError, match="duplicate replicate"):
        evaluate_paired_power_gate(design, (*trials, trials[0]))
    with pytest.raises(ValueError, match="digest differs"):
        evaluate_paired_power_gate(
            design,
            (
                trials[0].model_copy(update={"simulation_design_digest": "sha256:" + "b" * 64}),
                *trials[1:],
            ),
        )
    with pytest.raises(ValueError, match="evaluation design digest differs"):
        evaluate_paired_power_gate(
            design,
            (
                trials[0].model_copy(
                    update={"paired_evaluation_design_digest": "sha256:" + "d" * 64}
                ),
                *trials[1:],
            ),
        )


def test_clopper_pearson_bounds_round_outward_at_exact_boundaries() -> None:
    trials = 100
    alpha = 0.01
    raw_upper = float(beta.ppf(1.0 - alpha, 1, trials))
    upper = _binomial_upper_bound(0, trials, alpha=alpha)
    raw_lower = float(beta.ppf(alpha, trials, 1))
    lower = _binomial_lower_bound(trials, trials, alpha=alpha)

    assert upper == math.nextafter(raw_upper, math.inf)
    assert lower == math.nextafter(raw_lower, -math.inf)
    with localcontext() as context:
        context.prec = 80
        exact_alpha = Decimal.from_float(alpha)
        exact_lower = exact_alpha ** (Decimal(1) / Decimal(trials))
        exact_upper = Decimal(1) - exact_lower
        assert Decimal.from_float(upper) >= exact_upper
        assert Decimal.from_float(lower) <= exact_lower
    assert upper < 0.05
    assert _binomial_lower_bound(0, trials, alpha=alpha) == 0.0
    assert _binomial_upper_bound(trials, trials, alpha=alpha) == 1.0


def test_clopper_pearson_upper_moves_past_the_one_ulp_regression() -> None:
    alpha = 0.05
    raw = float(beta.ppf(1.0 - alpha, 2, 2))
    one_ulp = math.nextafter(raw, math.inf)
    certified = _binomial_upper_bound(1, 3, alpha=alpha)

    assert certified > one_ulp
    assert _independent_binomial_tail_upper(
        1,
        3,
        probability=certified,
        lower_tail=True,
    ) <= Decimal.from_float(alpha)


def test_clopper_pearson_exhaustive_independent_root_grid_through_200() -> None:
    alpha = 0.05
    threshold = Decimal.from_float(alpha)
    for trials in range(1, 201):
        for successes in range(trials):
            upper = _binomial_upper_bound(successes, trials, alpha=alpha)
            assert (
                _independent_binomial_tail_upper(
                    successes,
                    trials,
                    probability=upper,
                    lower_tail=True,
                )
                <= threshold
            ), ("upper", successes, trials, upper)
        for successes in range(1, trials + 1):
            lower = _binomial_lower_bound(successes, trials, alpha=alpha)
            assert (
                _independent_binomial_tail_upper(
                    successes,
                    trials,
                    probability=lower,
                    lower_tail=False,
                )
                <= threshold
            ), ("lower", successes, trials, lower)


def test_power_report_rejects_mutated_derived_or_bound_evidence() -> None:
    report = evaluate_paired_power_gate(
        _design(),
        _trials(null_rejections=0, target_rejections=100),
    )

    def reject(update: dict[str, object], match: str) -> None:
        payload = report.model_dump(mode="json")
        payload.update(update)
        with pytest.raises(ValidationError, match=match):
            PairedPowerGateReport.model_validate(payload)

    reject({"null_rejections": 101}, "cannot exceed frozen replications")
    reject({"target_rejections": 99}, "empirical_power differs")
    reject({"empirical_type_i_error": 0.01}, "empirical_type_i_error differs")
    reject({"empirical_power": 0.99}, "empirical_power differs")
    reject({"type_i_error_upper_bound": 0.0}, "type_i_error_upper_bound differs")
    reject({"power_lower_bound": 0.0}, "power_lower_bound differs")
    reject({"type_i_error_passed": False}, "type_i_error_passed differs")
    reject({"power_passed": False}, "power_passed differs")

    changed_design = report.design.model_copy(update={"target_effect": 0.3})
    reject({"design": changed_design.model_dump(mode="json")}, "report digest differs")
    reject({"trial_evidence_digest": "sha256:" + "e" * 64}, "report digest differs")
    reject({"report_digest": "sha256:" + "f" * 64}, "report digest differs")


def _independent_binomial_tail_upper(
    successes: int,
    trials: int,
    *,
    probability: float,
    lower_tail: bool,
) -> Decimal:
    """Direct positive-term Decimal sum independent of the production recurrence."""
    if probability <= 0.0:
        return Decimal(1 if lower_tail else 0)
    if probability >= 1.0:
        return Decimal(0 if lower_tail and successes < trials else 1)
    probability_decimal = Decimal.from_float(probability)
    value_tuple = probability_decimal.as_tuple()
    assert isinstance(value_tuple.exponent, int)
    precision = max(180, 1 - value_tuple.exponent + 12)
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_CEILING
        one = Decimal(1)
        complement = context.subtract(one, probability_decimal)
        assert context.add(probability_decimal, complement) == one
        observed_values = range(successes + 1) if lower_tail else range(successes, trials + 1)
        total = Decimal(0)
        for observed in observed_values:
            term = context.multiply(
                Decimal(math.comb(trials, observed)),
                context.multiply(
                    context.power(probability_decimal, observed),
                    context.power(complement, trials - observed),
                ),
            )
            total = context.add(total, term)
        return +total

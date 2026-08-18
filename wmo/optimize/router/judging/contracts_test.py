"""Tests for router judgment contracts."""

import pytest
from pydantic import ValidationError

from wmo.common.judging.lm import PORTABLE_RATIONALE_JSON_SCHEMA
from wmo.common.judging.lm_test import _axis_schema
from wmo.common.models import OperationEconomics, PricingSource
from wmo.optimize.router.judging.contracts import (
    JudgeCalibrationBudget,
    ManualJudgeReviewPricing,
    judge_feedback_schema,
)


def test_stored_budget_without_pricing_source_stays_unknown() -> None:
    """Older audit payloads remain readable without inventing a pricing source."""
    budget = JudgeCalibrationBudget.model_validate(
        {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
            "maximum_input_tokens_per_call": 4096,
            "maximum_attempts_per_call": 3,
            "call_count": 1,
            "estimated_cost_usd": 0.01,
            "maximum_cost_usd": 1.0,
        }
    )

    assert budget.pricing_source is PricingSource.UNKNOWN


def test_new_budget_defaults_to_sixteen_k_output_tokens() -> None:
    """Budgets built without an explicit output reservation get the current default."""
    budget = JudgeCalibrationBudget(
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=32_768,
        maximum_attempts_per_call=3,
        call_count=1,
        estimated_cost_usd=0.01,
        maximum_cost_usd=1.0,
    )

    assert budget.maximum_output_tokens_per_call == 16_384


def test_persisted_legacy_budget_with_4096_output_tokens_still_loads() -> None:
    """Previously approved audits priced at 4096 output tokens keep validating."""
    budget = JudgeCalibrationBudget.model_validate(
        {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
            "maximum_input_tokens_per_call": 4096,
            "maximum_output_tokens_per_call": 4096,
            "maximum_attempts_per_call": 3,
            "call_count": 1,
            "estimated_cost_usd": 0.01,
            "maximum_cost_usd": 1.0,
        }
    )

    assert budget.maximum_output_tokens_per_call == 4_096
    assert budget.model_dump(mode="json")["maximum_output_tokens_per_call"] == 4_096


def test_unsupported_output_budget_is_rejected() -> None:
    """Output reservations outside the supported set fail closed."""
    with pytest.raises(ValidationError, match="judge output tokens per call"):
        JudgeCalibrationBudget.model_validate(
            {
                "input_usd_per_million_tokens": 1.0,
                "output_usd_per_million_tokens": 2.0,
                "maximum_input_tokens_per_call": 4096,
                "maximum_output_tokens_per_call": 8192,
                "maximum_attempts_per_call": 3,
                "call_count": 1,
                "estimated_cost_usd": 0.01,
                "maximum_cost_usd": 1.0,
            }
        )


def test_review_pricing_defaults_new_and_loads_legacy_output_budget() -> None:
    """New review pricing reserves 16384 while persisted 4096 pricing keeps loading."""
    fresh = ManualJudgeReviewPricing(
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=32_768,
        maximum_attempts_per_call=3,
        authorized_call_count=1,
        maximum_reserved_cost_usd=1.0,
        observed_economics=OperationEconomics(),
    )
    legacy = ManualJudgeReviewPricing.model_validate(
        {
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 2.0,
            "maximum_input_tokens_per_call": 4096,
            "maximum_output_tokens_per_call": 4096,
            "maximum_attempts_per_call": 3,
            "authorized_call_count": 1,
            "maximum_reserved_cost_usd": 1.0,
            "observed_economics": OperationEconomics().model_dump(mode="json"),
        }
    )

    assert fresh.maximum_output_tokens_per_call == 16_384
    assert legacy.maximum_output_tokens_per_call == 4_096


def test_scalar_feedback_schema_matches_the_portable_rationale_contract() -> None:
    """Scalar setup schemas keep a required score and an optional nullable rationale."""
    schema = judge_feedback_schema("scalar")
    properties, required = _axis_schema(schema)

    assert properties["rationale"] == PORTABLE_RATIONALE_JSON_SCHEMA
    assert required == ["dimension_id", "raw_score"]
    assert "evidence_span_ids" not in properties

"""Tests for router judgment contracts."""

from wmo.common.models import PricingSource
from wmo.optimize.router.judging.contracts import JudgeCalibrationBudget


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

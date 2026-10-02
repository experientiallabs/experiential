"""Reservation pricing retains its public owner and integer per-surface ceilings."""

from __future__ import annotations

from exp.common.models.catalog import GatewayTokenPrices
from exp.runtime.gateway import attempt_costs, budgets
from exp.runtime.gateway.attempt_tokens import worst_case_output_tokens
from exp.runtime.gateway.budgets_test import _deployment, _request
from exp.runtime.gateway.decisions_contracts import DecisionRequest, NoulQuestion


def test_public_budget_pricing_exports_retain_owned_functions() -> None:
    """Extracted pricing stays the identical implementation used by existing callers."""
    assert budgets.maximum_attempt_cost_nano_usd is attempt_costs.maximum_attempt_cost_nano_usd
    assert (
        budgets.LONG_CONTEXT_TIER_MARGIN_PERCENT == attempt_costs.LONG_CONTEXT_TIER_MARGIN_PERCENT
    )


def test_integer_rounding_and_decision_allowance_are_preserved() -> None:
    """Completion pricing rounds up, while decisions use their own output allowance."""
    deployment = _deployment()
    completion = _request("hi").model_copy(update={"maximum_output_tokens": 1})
    tiny = deployment.model_copy(
        update={
            "gateway": deployment.gateway.model_copy(
                update={
                    "prices": GatewayTokenPrices(
                        input_nano_usd_per_million_tokens=1, output_nano_usd_per_million_tokens=1
                    )
                }
            )
        }
    )
    assert attempt_costs.maximum_attempt_cost_nano_usd(completion, tiny, input_tokens=1) == 1
    decision = DecisionRequest(
        state="state", questions={"ready": NoulQuestion(instructions="Ready?")}
    )
    output = worst_case_output_tokens(decision, deployment)
    assert deployment.capabilities is not None
    assert deployment.capabilities.maximum_output_tokens is not None
    assert output > deployment.capabilities.maximum_output_tokens
    assert (
        attempt_costs.maximum_attempt_cost_nano_usd(decision, deployment, input_tokens=7)
        == 7 + 2 * output
    )

"""Reservation pricing retains its public owner and integer per-surface ceilings."""

from __future__ import annotations

from exp.common.models.catalog import GatewayTokenPrices
from exp.common.models.catalog_prices import GatewayLongContextTier
from exp.runtime.gateway import attempt_costs, budgets
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.audio_contracts import SpeechRequest
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


def test_token_priced_audio_reserves_at_a_reachable_long_context_schedule() -> None:
    """Audio input bounds are byte ceilings, so a premium tier always prices the hold."""
    request = SpeechRequest(input="Hello there", voice="alloy")
    base = _deployment()
    tier = GatewayLongContextTier(
        input_threshold_tokens=1_000_000,
        input_nano_usd_per_million_tokens=3_000_000,
        output_nano_usd_per_million_tokens=5_000_000,
    )
    tiered = base.model_copy(
        update={
            "gateway": base.gateway.model_copy(
                update={"prices": base.gateway.prices.model_copy(update={"long_context": tier})}
            )
        }
    )
    input_tokens = worst_case_input_tokens(request)
    output_tokens = worst_case_output_tokens(request, tiered)
    expected = (input_tokens * 3_000_000 + output_tokens * 5_000_000 + 999_999) // 1_000_000
    assert attempt_costs.maximum_attempt_cost_nano_usd(request, tiered) == expected
    flat = attempt_costs.maximum_attempt_cost_nano_usd(request, base)
    assert flat is not None
    assert flat < expected
    unpriced_tier = tier.model_copy(update={"output_nano_usd_per_million_tokens": None})
    unpriced = tiered.model_copy(
        update={
            "gateway": tiered.gateway.model_copy(
                update={
                    "prices": tiered.gateway.prices.model_copy(
                        update={"long_context": unpriced_tier}
                    )
                }
            )
        }
    )
    assert attempt_costs.maximum_attempt_cost_nano_usd(request, unpriced) is None

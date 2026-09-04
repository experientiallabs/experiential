"""Tests for the pure ledger cost-attribution helpers."""

from exp.runtime.gateway.contracts import GatewayUsage
from exp.runtime.gateway.ledger_valuation import estimated_cost_micro_usd, optional_int


def test_subset_tokens_price_at_their_own_rates() -> None:
    """Cached-input and reasoning subsets bill at their rates, remainders at base."""
    usage = GatewayUsage(
        input_tokens=1_000,
        cached_input_tokens=400,
        output_tokens=200,
        reasoning_tokens=50,
    )
    cost = estimated_cost_micro_usd(
        usage,
        input_rate=10_000_000,
        cached_input_rate=1_000_000,
        output_rate=20_000_000,
        reasoning_rate=40_000_000,
    )
    # 600*10 + 400*1 + 150*20 + 50*40 = 11_400 micro-USD.
    assert cost == 11_400


def test_missing_rate_for_a_reported_subset_preserves_unknown_pricing() -> None:
    """A priced base rate never silently substitutes for a missing subset rate."""
    usage = GatewayUsage(input_tokens=100, cached_input_tokens=10, output_tokens=5)
    assert (
        estimated_cost_micro_usd(
            usage,
            input_rate=1_000_000,
            cached_input_rate=None,
            output_rate=1_000_000,
            reasoning_rate=None,
        )
        is None
    )


def test_malformed_subset_counts_clamp_to_their_totals() -> None:
    """Detail counts exceeding their totals clamp instead of going negative."""
    usage = GatewayUsage(
        input_tokens=10,
        cached_input_tokens=50,
        output_tokens=4,
        reasoning_tokens=9,
    )
    cost = estimated_cost_micro_usd(
        usage,
        input_rate=1_000_000,
        cached_input_rate=2_000_000,
        output_rate=3_000_000,
        reasoning_rate=5_000_000,
    )
    # 0*1 + 10*2 + 0*3 + 4*5 = 40 micro-USD.
    assert cost == 40


def test_absent_usage_or_counts_preserve_unknown_cost() -> None:
    """No usage, or usage without token counts, yields no estimate."""
    assert (
        estimated_cost_micro_usd(
            None, input_rate=1, cached_input_rate=1, output_rate=1, reasoning_rate=1
        )
        is None
    )
    tool_only = GatewayUsage(tool_names=("web_search",))
    assert (
        estimated_cost_micro_usd(
            tool_only,
            input_rate=1,
            cached_input_rate=1,
            output_rate=1,
            reasoning_rate=1,
        )
        is None
    )


def test_optional_int_preserves_null_and_narrows_values() -> None:
    """SQLite nullable integers convert precisely and keep None."""
    assert optional_int(None) is None
    assert optional_int(7) == 7

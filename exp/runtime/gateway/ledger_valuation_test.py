"""Tests for the pure ledger cost-attribution helpers."""

import sqlite3

import pytest

from exp.common.models.catalog import MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.ledger_errors import GatewayLedgerError
from exp.runtime.gateway.ledger_valuation import (
    MAXIMUM_NANO_USD,
    NanoUsdOverflowError,
    budget_settlement_nano_usd,
    estimated_cost_nano_usd,
    optional_int,
    terminal_values,
)


@pytest.mark.parametrize(
    ("surface", "cost", "reserved", "rejected", "expected"),
    [
        ("decisions", None, 100, False, None),
        ("decisions", None, 100, True, 0),
        ("chat_completions", None, 100, False, 100),
        ("chat_completions", None, None, False, None),
        ("decisions", 12, 100, False, 12),
    ],
)
def test_budget_settlement_keeps_unknown_and_unmetered_liability(
    surface: str,
    cost: int | None,
    reserved: int | None,
    rejected: bool,
    expected: int | None,
) -> None:
    """Pure settlement preserves conservative bounds and releases only proven decision rejection."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT ? AS api_surface, ? AS budget_reserved_nano_usd", (surface, reserved)
        ).fetchone()
        event = GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            decision_provider_rejected=rejected,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.UNAVAILABLE, safe_message="unavailable"
            ),
        )
        assert budget_settlement_nano_usd(row, cost, None, event) == expected
        with pytest.raises(GatewayLedgerError, match="integer capacity"):
            budget_settlement_nano_usd(row, MAXIMUM_NANO_USD + 1, None, None)
    finally:
        connection.close()


def test_terminal_value_normalization_keeps_unknown_usage_and_sanitized_failure() -> None:
    """The pure owner preserves terminal accounting without inventing observed usage."""
    event = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0)
    assert terminal_values(event, None) == ("completed", None, None, None)
    cancelled = GatewayFailure(
        failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"
    )
    assert terminal_values(None, cancelled) == ("cancelled", "cancelled", None, None)
    with pytest.raises(GatewayLedgerError, match="needs a terminal"):
        terminal_values(None, None)


def test_subset_tokens_price_at_their_own_rates() -> None:
    """Cached-input and reasoning subsets bill at their rates, remainders at base."""
    usage = GatewayUsage(
        input_tokens=1_000,
        cached_input_tokens=400,
        output_tokens=200,
        reasoning_tokens=50,
    )
    cost = estimated_cost_nano_usd(
        usage,
        input_rate=10_000_000,
        cached_input_rate=1_000_000,
        output_rate=20_000_000,
        reasoning_rate=40_000_000,
    )
    # 600*10 + 400*1 + 150*20 + 50*40 = 11_400 nano-USD.
    assert cost == 11_400


def test_missing_rate_for_a_reported_subset_preserves_unknown_pricing() -> None:
    """A priced base rate never silently substitutes for a missing subset rate."""
    usage = GatewayUsage(input_tokens=100, cached_input_tokens=10, output_tokens=5)
    assert (
        estimated_cost_nano_usd(
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
    cost = estimated_cost_nano_usd(
        usage,
        input_rate=1_000_000,
        cached_input_rate=2_000_000,
        output_rate=3_000_000,
        reasoning_rate=5_000_000,
    )
    # 0*1 + 10*2 + 0*3 + 4*5 = 40 nano-USD.
    assert cost == 40


def test_cache_write_prices_at_its_surcharge_rate() -> None:
    """Cache-write tokens bill at their own rate, disjoint from cache-read."""
    fresh = GatewayUsage(input_tokens=1_000, output_tokens=10)
    written = GatewayUsage(
        input_tokens=1_000,
        cache_creation_input_tokens=1_000,
        cache_creation_1h_input_tokens=0,
        output_tokens=10,
    )
    # Synthetic nano-USD rates: the fresh request costs 3_150, cache writes cost 3_900.
    assert (
        estimated_cost_nano_usd(
            fresh,
            input_rate=3_000_000,
            cached_input_rate=300_000,
            cache_creation_input_rate=3_750_000,
            output_rate=15_000_000,
            reasoning_rate=15_000_000,
        )
        == 3_150
    )
    assert (
        estimated_cost_nano_usd(
            written,
            input_rate=3_000_000,
            cached_input_rate=300_000,
            cache_creation_input_rate=3_750_000,
            output_rate=15_000_000,
            reasoning_rate=15_000_000,
        )
        == 3_900
    )
    # Mixed: 400 cached read + 300 cache write + 300 fresh.
    mixed = GatewayUsage(
        input_tokens=1_000,
        cached_input_tokens=400,
        cache_creation_input_tokens=300,
        cache_creation_1h_input_tokens=0,
        output_tokens=10,
    )
    assert (
        estimated_cost_nano_usd(
            mixed,
            input_rate=3_000_000,
            cached_input_rate=300_000,
            cache_creation_input_rate=3_750_000,
            output_rate=15_000_000,
            reasoning_rate=15_000_000,
        )
        == 300 * 3_000_000 // 1_000_000
        + 400 * 300_000 // 1_000_000
        + 300 * 3_750_000 // 1_000_000
        + 10 * 15_000_000 // 1_000_000
    )


def test_missing_cache_write_rate_preserves_unknown_pricing() -> None:
    """A cache-write without its rate stays unknown even when base rate is known."""
    usage = GatewayUsage(
        input_tokens=100,
        cache_creation_input_tokens=10,
        cache_creation_1h_input_tokens=0,
        output_tokens=5,
    )
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=1_000_000,
            cached_input_rate=1_000_000,
            cache_creation_input_rate=None,
            output_rate=1_000_000,
            reasoning_rate=None,
        )
        is None
    )


def test_malformed_cache_write_clamps_to_remaining_input() -> None:
    """Cache-write exceeding the remaining input clamps to input - cached."""
    usage = GatewayUsage(
        input_tokens=100,
        cached_input_tokens=60,
        cache_creation_input_tokens=90,
        cache_creation_1h_input_tokens=0,
        output_tokens=5,
    )
    # cached=60, creation clamps to 40, fresh=0.
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=1_000_000,
            cached_input_rate=2_000_000,
            cache_creation_input_rate=3_000_000,
            output_rate=1_000_000,
            reasoning_rate=None,
        )
        == 60 * 2_000_000 // 1_000_000 + 40 * 3_000_000 // 1_000_000 + 5
    )


def test_absent_usage_or_counts_preserve_unknown_cost() -> None:
    """No usage, or usage without token counts, yields no estimate."""
    assert (
        estimated_cost_nano_usd(
            None, input_rate=1, cached_input_rate=1, output_rate=1, reasoning_rate=1
        )
        is None
    )
    tool_only = GatewayUsage(tool_names=("web_search",))
    assert (
        estimated_cost_nano_usd(
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


def test_nano_usd_cost_is_the_micro_usd_cost_at_three_more_digits() -> None:
    """The nano-USD ledger prices the same tokens at rates a thousand times
    finer, so where a micro-USD cost was exact (the numerator divisible by one
    million) the nano cost is exactly a thousand times it; at every other
    numerator the two differ by at most half a micro-USD, because each rounds
    half-up at its OWN unit. The rule, pinned by these examples, is that the
    nano figure is the finer truth and the micro figure was its rounding, never
    the other way around.
    """
    cases = (
        # (tokens x micro-rate numerator, old micro-USD cost, new nano-USD cost)
        (1_000_000, 1, 1_000),  # exact: nano == micro x 1000
        (2_000_000, 2, 2_000),
        (1_400_000, 1, 1_400),  # micro rounded 1.4 down; nano keeps it
        (1_499_500, 1, 1_500),  # nano rounds its own half up; micro did not
        (1_500_000, 2, 1_500),  # micro rounded 1.5 up; nano keeps 1.5 exactly
        (999_499, 1, 999),  # micro rounded up to 1; nano says 0.999
        (499_999, 0, 500),  # sub-micro-dollar work is no longer billed as zero
        (0, 0, 0),
    )
    for numerator, micro_expected, nano_expected in cases:
        usage = GatewayUsage(input_tokens=1, output_tokens=0)
        micro = (numerator + 500_000) // 1_000_000
        assert micro == micro_expected, numerator
        nano = estimated_cost_nano_usd(
            usage,
            input_rate=numerator * 1_000,
            cached_input_rate=None,
            output_rate=0,
            reasoning_rate=None,
        )
        assert nano == nano_expected, numerator
        assert abs(nano - micro * 1_000) <= 500, numerator
        if numerator % 1_000_000 == 0:
            assert nano == micro * 1_000


def test_nano_usd_cost_never_exceeds_the_int8_ledger_column() -> None:
    """A cost past the signed 64-bit ledger column raises a named error instead
    of being stored as a wrapped or coerced value."""
    usage = GatewayUsage(input_tokens=10**13, output_tokens=0)
    with pytest.raises(NanoUsdOverflowError):
        estimated_cost_nano_usd(
            usage,
            input_rate=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
            cached_input_rate=None,
            output_rate=0,
            reasoning_rate=None,
        )
    assert MAXIMUM_NANO_USD == 2**63 - 1


def test_max_authored_rate_and_million_token_context_fit_the_int8_column() -> None:
    """Headroom pin: the highest authored rate today is $600 per million tokens
    (6e11 nano-USD per million); at the rate CEILING on every dimension of a
    1M-context request with the full output ceiling, both the pre-division
    numerator and the settled cost stay far inside the int8 ledger column."""
    tokens_per_dimension = 1_050_000
    dimensions = 4
    numerator = tokens_per_dimension * MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS * dimensions
    assert 600_000_000 * 1_000 < MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS
    assert numerator < MAXIMUM_NANO_USD
    usage = GatewayUsage(
        input_tokens=2 * tokens_per_dimension,
        cached_input_tokens=tokens_per_dimension,
        output_tokens=2 * tokens_per_dimension,
        reasoning_tokens=tokens_per_dimension,
    )
    cost = estimated_cost_nano_usd(
        usage,
        input_rate=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
        cached_input_rate=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
        output_rate=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
        reasoning_rate=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
    )
    assert cost == numerator // 1_000_000
    assert cost < MAXIMUM_NANO_USD // 1_000


@pytest.mark.parametrize("hour", [0, 200, 600])
def test_mixed_ttl_writes_are_priced_once(hour: int) -> None:
    """Provider TTL evidence partitions writes without changing the input total."""
    usage = GatewayUsage(
        input_tokens=1000,
        output_tokens=10,
        cached_input_tokens=100,
        cache_creation_input_tokens=600,
        cache_creation_1h_input_tokens=hour,
    )
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=3_000_000_000,
            cached_input_rate=300_000_000,
            cache_creation_input_rate=3_750_000_000,
            cache_creation_1h_input_rate=6_000_000_000,
            output_rate=15_000_000_000,
            reasoning_rate=None,
        )
        == 300 * 3000 + 100 * 300 + (600 - hour) * 3750 + hour * 6000 + 10 * 15000
    )


@pytest.mark.parametrize("hour, hour_rate", [(None, 6_000_000_000), (200, None)])
def test_unobserved_ttl_or_missing_hour_rate_preserves_unknown_cost(
    hour: int | None, hour_rate: int | None
) -> None:
    """A request or default price must never invent a missing observed TTL split."""
    usage = GatewayUsage(
        input_tokens=1000,
        output_tokens=10,
        cache_creation_input_tokens=600,
        cache_creation_1h_input_tokens=hour,
    )
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=3_000_000_000,
            cached_input_rate=300_000_000,
            cache_creation_input_rate=3_750_000_000,
            cache_creation_1h_input_rate=hour_rate,
            output_rate=15_000_000_000,
            reasoning_rate=None,
        )
        is None
    )


def test_cache_write_nano_usd_rounding_and_overflow_are_bounded() -> None:
    """Write dimensions use the shared single rounding and signed-ledger limit."""
    usage = GatewayUsage(
        input_tokens=2,
        output_tokens=0,
        cache_creation_input_tokens=2,
        cache_creation_1h_input_tokens=1,
    )
    assert (
        estimated_cost_nano_usd(
            usage,
            input_rate=None,
            cached_input_rate=None,
            cache_creation_input_rate=250_000,
            cache_creation_1h_input_rate=250_000,
            output_rate=None,
            reasoning_rate=None,
        )
        == 1
    )
    with pytest.raises(NanoUsdOverflowError):
        estimated_cost_nano_usd(
            usage,
            input_rate=None,
            cached_input_rate=None,
            cache_creation_input_rate=10**30,
            cache_creation_1h_input_rate=10**30,
            output_rate=None,
            reasoning_rate=None,
        )

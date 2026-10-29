"""Tests for cache-write pricing, tier and off-peak selection, and per-unit media cards."""

from datetime import UTC, date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from exp.common.models.catalog import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.catalog_prices import (
    MAXIMUM_RATE_NANO_USD_PER_UNIT,
    MAXIMUM_UNIT_VARIANT_CHARACTERS,
    BilledUnitKind,
    GatewayOffPeakPrices,
    GatewayPriceWindow,
    GatewayUnitPrices,
)


def test_requested_service_tier_keeps_its_cache_write_schedule() -> None:
    """Tier selection uses the requested rates without falling back to base writes."""
    prices = GatewayTokenPrices(
        cache_creation_input_nano_usd_per_million_tokens=99,
        priority=GatewayServiceTierPrices(
            cache_creation_input_nano_usd_per_million_tokens=10,
            cache_creation_1h_input_nano_usd_per_million_tokens=20,
        ),
    )
    selected = prices.for_service_tier("priority")
    assert selected.cache_creation_input_nano_usd_per_million_tokens == 10
    assert selected.cache_creation_1h_input_nano_usd_per_million_tokens == 20
    assert selected.long_context is None


def test_service_tier_keeps_its_own_complete_long_context_schedule() -> None:
    """Tier selection never erases or inherits a different processing schedule."""
    long = GatewayLongContextTier(
        input_threshold_tokens=272_001,
        input_nano_usd_per_million_tokens=8_000_000_000,
        cache_creation_input_nano_usd_per_million_tokens=10_000_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=16_000_000_000,
    )
    prices = GatewayTokenPrices(
        long_context=long.model_copy(update={"input_threshold_tokens": 100}),
        priority=GatewayServiceTierPrices(long_context=long),
    )
    assert prices.for_service_tier("priority").long_context == long
    assert prices.for_service_tier("default") is prices


def test_unset_long_context_does_not_change_snapshot_identity_projection() -> None:
    """Schema-five snapshots exclude declared defaults recursively, including this new None."""
    card = GatewayServiceTierPrices(input_nano_usd_per_million_tokens=7)
    assert card.model_dump(mode="json", exclude_defaults=True) == {
        "input_nano_usd_per_million_tokens": 7
    }
    prices = GatewayTokenPrices(priority=card)
    assert prices.model_dump(mode="json", exclude_defaults=True) == {
        "priority": {"input_nano_usd_per_million_tokens": 7}
    }


def test_unit_card_prices_a_variant_or_falls_back_to_the_flat_rate() -> None:
    """A named variant prices itself; an unauthored one uses the flat rate, else nothing."""
    card = GatewayUnitPrices(
        kind=BilledUnitKind.VIDEO_SECOND, rates={"": 80_000_000, "1080p": 120_000_000}
    )
    assert card.rate_for("1080p") == 120_000_000
    assert card.rate_for("4k") == 80_000_000
    assert card.rate_for("") == 80_000_000
    flatless = GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"720p": 1})
    assert flatless.rate_for("1080p") is None


@pytest.mark.parametrize(
    "rates",
    [
        {},
        {"": -1},
        {"": MAXIMUM_RATE_NANO_USD_PER_UNIT + 1},
        {"v" * (MAXIMUM_UNIT_VARIANT_CHARACTERS + 1): 1},
    ],
)
def test_unit_card_rejects_empty_negative_oversized_or_unbounded_rates(
    rates: dict[str, int],
) -> None:
    """A unit card fails closed at authoring instead of pricing at a corrupt rate."""
    with pytest.raises(ValidationError):
        GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates=rates)


def test_unit_card_is_tier_independent_and_absent_by_default() -> None:
    """Processing tiers reprice tokens only, and a token card serializes unchanged."""
    units = GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": 15_000})
    card = GatewayTokenPrices(
        flex=GatewayServiceTierPrices(input_nano_usd_per_million_tokens=1), units=units
    )
    assert card.for_service_tier("flex").units == units
    assert "units" not in GatewayTokenPrices().model_dump(exclude_defaults=True)


# DeepSeek's published schedule (api-docs.deepseek.com/quick_start/pricing, 2026-10-19):
# peak 01:00-04:00 and 06:00-10:00 UTC Monday to Friday, every other hour half price.
_DEEPSEEK_PEAK = (
    GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=60, end_minute=240),
    GatewayPriceWindow(days=(1, 2, 3, 4, 5), start_minute=360, end_minute=600),
)


def _deepseek_flash(*, off_peak_dates: tuple[date, ...] = ()) -> GatewayTokenPrices:
    """DeepSeek V4.1 Flash: peak $0.30/$0.006/$1.20, off-peak exactly half."""
    return GatewayTokenPrices(
        input_nano_usd_per_million_tokens=300_000_000,
        cached_input_nano_usd_per_million_tokens=6_000_000,
        output_nano_usd_per_million_tokens=1_200_000_000,
        off_peak=GatewayOffPeakPrices(
            prices=GatewayServiceTierPrices(
                input_nano_usd_per_million_tokens=150_000_000,
                cached_input_nano_usd_per_million_tokens=3_000_000,
                output_nano_usd_per_million_tokens=600_000_000,
            ),
            peak_windows=_DEEPSEEK_PEAK,
            off_peak_dates=off_peak_dates,
        ),
    )


def _utc(day: int, hour: int, minute: int = 0, second: int = 0, micro: int = 0) -> datetime:
    """An instant in the week of Monday 2026-10-19 (day 19) to Sunday 2026-10-25."""
    return datetime(2026, 10, day, hour, minute, second, micro, tzinfo=UTC)


@pytest.mark.parametrize(
    ("instant", "peak"),
    [
        (_utc(19, 0, 44, 59, 999_999), False),
        (_utc(19, 0, 45), True),  # lead band: fifteen minutes before 01:00 is peak
        (_utc(19, 0, 59), True),
        (_utc(19, 1), True),
        (_utc(19, 3, 59, 59, 999_999), True),
        (_utc(19, 4), True),  # guard band: the first minute after 04:00 is peak
        (_utc(19, 4, 0, 59, 999_999), True),
        (_utc(19, 4, 1), False),
        (_utc(19, 5, 44, 59, 999_999), False),
        (_utc(19, 5, 45), True),
        (_utc(19, 6), True),
        (_utc(19, 9, 59, 59, 999_999), True),
        (_utc(19, 10, 0, 59, 999_999), True),
        (_utc(19, 10, 1), False),
        (_utc(19, 23, 59, 59), False),
        (_utc(23, 9, 59), True),  # Friday
        (_utc(24, 1, 30), False),  # Saturday
        (_utc(25, 7), False),  # Sunday
        (_utc(25, 23, 59, 59, 999_999), False),  # Sunday into Monday (00:00 is off-peak)
        (datetime(2026, 10, 26, 0, 44, 59, tzinfo=UTC), False),
        (datetime(2026, 10, 26, 0, 45, tzinfo=UTC), True),  # Monday lead band
        (datetime(2026, 10, 26, 1, tzinfo=UTC), True),  # Monday again
    ],
)
def test_off_peak_window_boundaries_are_half_open_utc(instant: datetime, peak: bool) -> None:
    """Weekday peak [01:00, 04:00) and [06:00, 10:00) UTC, from 15 min before to 1 min after."""
    prices = _deepseek_flash()
    effective = prices.effective_at(instant)
    assert effective.off_peak is None
    if peak:
        assert effective == prices.model_copy(update={"off_peak": None})
    else:
        assert effective.input_nano_usd_per_million_tokens == 150_000_000
        assert effective.cached_input_nano_usd_per_million_tokens == 3_000_000
        assert effective.output_nano_usd_per_million_tokens == 600_000_000
        assert effective.off_peak is None


def test_off_peak_date_discounts_a_weekday_peak_hour_all_day() -> None:
    """A provider holiday (UTC date) is off-peak even inside a weekly peak window."""
    prices = _deepseek_flash(off_peak_dates=(date(2026, 10, 20), date(2027, 1, 1)))
    assert prices.effective_at(_utc(20, 7)).input_nano_usd_per_million_tokens == 150_000_000
    assert prices.effective_at(_utc(21, 7)).input_nano_usd_per_million_tokens == 300_000_000


def test_off_peak_selection_uses_the_utc_instant_in_any_zone_and_across_dst() -> None:
    """Zones normalize to UTC; a DST transition never moves a UTC window."""
    prices = _deepseek_flash()

    def is_peak(instant: datetime) -> bool:
        return prices.effective_at(instant).input_nano_usd_per_million_tokens == 300_000_000

    # 2026-11-01 is the US DST end; 05:59 UTC there is 01:59 EDT / 00:59 EST.
    new_york = ZoneInfo("America/New_York")
    assert is_peak(datetime(2026, 11, 2, 1, 30, tzinfo=new_york))
    assert not is_peak(datetime(2026, 11, 2, 0, 30, tzinfo=new_york))
    # 2027-03-28 is the EU DST start; Monday 2027-03-29 09:30 CEST is 07:30 UTC, peak.
    berlin = ZoneInfo("Europe/Berlin")
    assert is_peak(datetime(2027, 3, 29, 9, 30, tzinfo=berlin))
    assert not is_peak(datetime(2027, 3, 29, 12, 30, tzinfo=berlin))
    beijing = timezone(timedelta(hours=8))
    assert is_peak(datetime(2026, 10, 19, 9, 0, tzinfo=beijing))
    with pytest.raises(ValueError, match="timezone-aware"):
        prices.effective_at(datetime(2026, 10, 19, 7))  # noqa: DTZ001


def test_off_peak_card_flattens_with_its_own_long_context_tier() -> None:
    """The off-peak card replaces the whole schedule, including the long-context tier."""
    long = GatewayLongContextTier(
        input_threshold_tokens=200_000, input_nano_usd_per_million_tokens=600_000_000
    )
    prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=300_000_000,
        long_context=long,
        off_peak=GatewayOffPeakPrices(
            prices=GatewayServiceTierPrices(
                input_nano_usd_per_million_tokens=150_000_000,
                long_context=long.model_copy(update={"input_nano_usd_per_million_tokens": 1}),
            ),
            peak_windows=_DEEPSEEK_PEAK,
        ),
    )
    flat = prices.effective_at(_utc(24, 12))
    assert flat.long_context is not None
    assert flat.long_context.input_nano_usd_per_million_tokens == 1
    assert flat == GatewayTokenPrices(
        input_nano_usd_per_million_tokens=150_000_000, long_context=flat.long_context
    )


def test_cards_without_a_schedule_select_themselves_and_keep_their_identity() -> None:
    """No schedule is the same object at every instant and adds nothing to identity."""
    prices = GatewayTokenPrices(input_nano_usd_per_million_tokens=7)
    assert prices.effective_at(_utc(24, 12)) is prices
    assert prices.model_dump(mode="json", exclude_defaults=True) == {
        "input_nano_usd_per_million_tokens": 7
    }
    with_schedule = _deepseek_flash().model_dump(mode="json", exclude_defaults=True)
    assert with_schedule["off_peak"]["peak_windows"][0] == {
        "days": [1, 2, 3, 4, 5],
        "start_minute": 60,
        "end_minute": 240,
    }
    assert GatewayTokenPrices.model_validate_json(_deepseek_flash().model_dump_json()) == (
        _deepseek_flash()
    )


def _with_off_peak(**overrides: object) -> dict[str, object]:
    """A valid DeepSeek card as a plain payload, with chosen keys replaced."""
    payload = _deepseek_flash(off_peak_dates=(date(2027, 1, 1),)).model_dump(mode="json")
    off_peak = payload["off_peak"]
    assert isinstance(off_peak, dict)
    for key, value in overrides.items():
        if key in off_peak:
            off_peak[key] = value
        else:
            payload[key] = value
    return payload


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "prices": {
                    "input_nano_usd_per_million_tokens": 300_000_001,
                    "cached_input_nano_usd_per_million_tokens": 3_000_000,
                    "output_nano_usd_per_million_tokens": 600_000_000,
                }
            },
            "exceeds the peak rate",
        ),
        (
            {
                "prices": {
                    "input_nano_usd_per_million_tokens": 1,
                    "output_nano_usd_per_million_tokens": 1,
                }
            },
            "set exactly when the peak rate is",
        ),
        ({"peak_windows": []}, "at least 1"),
        (
            {"peak_windows": [{"days": [1], "start_minute": 60, "end_minute": 240}] * 2},
            "must not overlap",
        ),
        (
            {"peak_windows": [{"days": [2, 1], "start_minute": 60, "end_minute": 240}]},
            "strictly increasing ISO weekdays",
        ),
        (
            {"peak_windows": [{"days": [1], "start_minute": 240, "end_minute": 240}]},
            "must end after it starts",
        ),
        (
            {"peak_windows": [{"days": [8], "start_minute": 0, "end_minute": 60}]},
            "less than or equal to 7",
        ),
        ({"off_peak_dates": ["2027-01-02", "2027-01-01"]}, "strictly increasing"),
        (
            {"flex": {"input_nano_usd_per_million_tokens": 1}},
            "cannot be combined with tier or unit cards",
        ),
        (
            {"units": {"kind": "image", "rates": {"": 1}}},
            "cannot be combined with tier or unit cards",
        ),
        (
            {
                "long_context": {
                    "input_threshold_tokens": 1,
                    "input_nano_usd_per_million_tokens": 1,
                }
            },
            "long-context tier exactly when the base does",
        ),
    ],
)
def test_off_peak_schedule_validation_refuses_ambiguous_or_overpriced_cards(
    overrides: dict[str, object], message: str
) -> None:
    """Every off-peak leg sits at or below peak, and schedules have one canonical form."""
    payload = _with_off_peak(**overrides)
    with pytest.raises(ValidationError, match=message):
        GatewayTokenPrices.model_validate(payload)


def test_off_peak_long_context_tier_keeps_the_base_threshold() -> None:
    """A different off-peak threshold would price one request under two tier rules."""
    long = GatewayLongContextTier(input_threshold_tokens=200_000)
    with pytest.raises(ValidationError, match="keep the base threshold"):
        GatewayTokenPrices(
            long_context=long,
            off_peak=GatewayOffPeakPrices(
                prices=GatewayServiceTierPrices(
                    long_context=long.model_copy(update={"input_threshold_tokens": 100_000})
                ),
                peak_windows=_DEEPSEEK_PEAK,
            ),
        )

"""Bounded integer nano-USD gateway pricing cards and schedule selection."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ContractModel

MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS = 1_000_000_000_000
"""Upper bound on any authored rate: $1,000 per million tokens in nano-USD.

Every published price today is far below it (the highest authored rate is
$600 per million, 6e11 nano-USD), and at this ceiling on every dimension a
1M-context request with the full output ceiling still sums to well under the
signed 64-bit ledger column before the per-million division, so no authored
catalog can produce an attempt cost the ledger cannot hold.
"""

NanoUsdRatePerMillionTokens = Annotated[
    int | None, Field(ge=0, le=MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS)
]
"""One optional integer nano-USD-per-million-tokens rate; ``None`` is unknown, never zero."""


class GatewayLongContextTier(ContractModel):
    """Premium rates a provider applies to whole long-context requests.

    When ``usage.input_tokens >= input_threshold_tokens``, tier rates replace
    base rates for every dimension of the whole request, not just excess tokens.
    This models Gemini and Anthropic's long-context premium schedules. A ``None``
    tier rate stays unknown; it never inherits the base rate.
    """

    input_threshold_tokens: int = Field(gt=0)
    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_12h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    image_output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None


class GatewayServiceTierPrices(ContractModel):
    """PASS-THROUGH rates for one provider processing tier (flex / priority).

    OpenAI's ``service_tier`` reprices the WHOLE request (``flex`` discounted,
    ``priority`` premium): these rates replace the base schedule for every
    dimension at cost, no markup. ``None`` on a dimension is unknown exactly as
    on the base schedule (never the base rate). Its own long-context schedule
    applies independently of the standard processing schedule.
    """

    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_12h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    image_output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    long_context: GatewayLongContextTier | None = None


MAXIMUM_RATE_NANO_USD_PER_UNIT = 1_000_000_000_000
"""Upper bound on one authored per-unit rate: $1,000 per billed unit in nano-USD.

The priciest published media unit today is a second of 4K video at well under
$1 (1e9 nano-USD), so this bound only rejects authoring mistakes. A request's
unit ceiling multiplies it by a bounded quantity and is checked against the
signed 64-bit ledger column like every other reservation.
"""

MAXIMUM_UNIT_VARIANT_CHARACTERS = 64


class BilledUnitKind(StrEnum):
    """What one non-token billed unit measures.

    Media models are priced by what they produce or consume rather than by
    tokens: speech synthesis by input character, transcription by second of
    input audio, video generation by second of output video, and per-image
    priced image models by image.
    """

    CHARACTER = "character"
    AUDIO_SECOND = "audio_second"
    VIDEO_SECOND = "video_second"
    IMAGE = "image"


class GatewayUnitPrices(ContractModel):
    """Per-unit rates for a deployment priced in billed units instead of (or beside) tokens.

    ``rates`` maps a variant to integer nano-USD per unit. The empty variant
    ``""`` is the flat rate; named variants price provider SKUs that differ by
    output shape (``"720p"``, ``"1080p/audio"``, ``"hd/1024x1024"``). A request
    whose variant has no rate falls back to the flat rate when one exists and
    is otherwise unpriced (fail-closed: never billed at an invented rate).

    Attributes:
        kind: What one billed unit measures.
        rates: Variant to integer nano-USD per unit; at least one entry.
    """

    kind: BilledUnitKind
    rates: dict[str, int] = Field(min_length=1)

    @field_validator("rates")
    @classmethod
    def _bounded_rates(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject an out-of-range rate or an unbounded variant name."""
        for variant, rate in value.items():
            if len(variant) > MAXIMUM_UNIT_VARIANT_CHARACTERS:
                msg = (
                    f"unit variant {variant[:16]!r}... exceeds "
                    f"{MAXIMUM_UNIT_VARIANT_CHARACTERS} characters"
                )
                raise ValueError(msg)
            if rate < 0 or rate > MAXIMUM_RATE_NANO_USD_PER_UNIT:
                msg = (
                    f"unit rate for variant {variant!r} must be within "
                    f"0..{MAXIMUM_RATE_NANO_USD_PER_UNIT}"
                )
                raise ValueError(msg)
        return value

    def rate_for(self, variant: str) -> int | None:
        """Return the rate for one variant, falling back to the flat rate.

        Args:
            variant: The request's priced variant; ``""`` asks for the flat rate.

        Returns:
            Integer nano-USD per unit, or None when neither the variant nor a
            flat rate is authored.
        """
        if variant in self.rates:
            return self.rates[variant]
        return self.rates.get("")


MINUTES_PER_DAY = 24 * 60
MAXIMUM_PEAK_WINDOWS = 32
MAXIMUM_OFF_PEAK_DATES = 128
PEAK_GUARD_BAND = timedelta(minutes=1)
"""How far each peak window is widened after its end, against billing-clock skew."""
PEAK_LEAD_BAND = timedelta(minutes=15)
"""How far each peak window is widened before its start.

Selection happens at the attempt's start, but a provider may bill by completion
time (DeepSeek's pricing page does not say which). A long request that starts
just before a peak window could then be billed at peak while charged off-peak;
treating the preceding fifteen minutes as peak bounds that exposure. Owner
decision (Silen, 2026-10-19).
"""

_TOKEN_RATE_FIELDS = (
    "input_nano_usd_per_million_tokens",
    "cached_input_nano_usd_per_million_tokens",
    "cache_creation_input_nano_usd_per_million_tokens",
    "cache_creation_1h_input_nano_usd_per_million_tokens",
    "cache_creation_12h_input_nano_usd_per_million_tokens",
    "output_nano_usd_per_million_tokens",
    "reasoning_nano_usd_per_million_tokens",
    "image_output_nano_usd_per_million_tokens",
)
"""Every per-million token rate shared by base, tier, long-context, and off-peak cards."""


class GatewayPriceWindow(ContractModel):
    """One weekly UTC window, half-open ``[start_minute, end_minute)`` on each listed day.

    A window never wraps midnight; an overnight span is authored as two windows.
    ``days`` are ISO weekdays (Monday 1 to Sunday 7), strictly increasing so one
    schedule has exactly one serialization.

    Attributes:
        days: ISO weekdays the window recurs on.
        start_minute: First covered minute after 00:00 UTC.
        end_minute: First minute after the window; 1440 ends it at midnight.
    """

    days: tuple[Annotated[int, Field(ge=1, le=7)], ...] = Field(min_length=1, max_length=7)
    start_minute: int = Field(ge=0, lt=MINUTES_PER_DAY)
    end_minute: int = Field(gt=0, le=MINUTES_PER_DAY)

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Reject an empty span or unordered, repeated weekdays."""
        if self.start_minute >= self.end_minute:
            raise ValueError("a price window must end after it starts on the same UTC day")
        if any(left >= right for left, right in zip(self.days, self.days[1:], strict=False)):
            raise ValueError("price window days must be strictly increasing ISO weekdays")
        return self

    def covers(self, weekday: int, minute: int) -> bool:
        """Report whether one UTC weekday and minute of day fall inside this window."""
        return weekday in self.days and self.start_minute <= minute < self.end_minute


class GatewayOffPeakPrices(ContractModel):
    """A provider's discounted schedule outside its published peak hours (UTC only).

    DeepSeek, for example, halves every token rate outside 01:00 to 04:00 and
    06:00 to 10:00 UTC on weekdays, and all day on Chinese public holidays. The
    base card stays the peak (list) price; this card replaces it whole, with its
    own long-context tier, whenever an attempt STARTS off-peak. Selection happens
    once per attempt, so reservation and settlement always use the same card.
    Each peak window counts as peak from fifteen minutes before its start
    (:data:`PEAK_LEAD_BAND`) to one minute after its end (:data:`PEAK_GUARD_BAND`).

    Attributes:
        prices: Whole replacement card charged off-peak.
        peak_windows: Weekly UTC windows that charge the base card.
        off_peak_dates: UTC dates charged off-peak all day (provider holidays),
            strictly increasing.
    """

    prices: GatewayServiceTierPrices
    peak_windows: tuple[GatewayPriceWindow, ...] = Field(
        min_length=1, max_length=MAXIMUM_PEAK_WINDOWS
    )
    off_peak_dates: tuple[date, ...] = Field(default=(), max_length=MAXIMUM_OFF_PEAK_DATES)

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Reject overlapping peak windows and unordered or repeated holidays."""
        for index, window in enumerate(self.peak_windows):
            for other in self.peak_windows[index + 1 :]:
                if (
                    set(window.days) & set(other.days)
                    and window.start_minute < other.end_minute
                    and other.start_minute < window.end_minute
                ):
                    raise ValueError("off-peak schedule peak windows must not overlap")
        dates = self.off_peak_dates
        if any(left >= right for left, right in zip(dates, dates[1:], strict=False)):
            raise ValueError("off-peak dates must be strictly increasing")
        return self

    def applies_at(self, instant: datetime) -> bool:
        """Report whether the off-peak card prices an attempt started at ``instant``.

        Each peak window is widened by :data:`PEAK_LEAD_BAND` before its start
        and :data:`PEAK_GUARD_BAND` after its end, so an attempt that starts in
        the fifteen minutes before a peak window, or within a minute after one,
        pays peak. A long request cannot then be charged off-peak while the
        provider bills its completion at peak, and clock skew can only
        overcharge a little near a boundary.

        Args:
            instant: Timezone-aware attempt start; any zone is normalized to UTC.

        Returns:
            True when no instant within the guard band of ``instant`` is peak.

        Raises:
            ValueError: ``instant`` is naive, so its UTC position is unknown.
        """
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("off-peak selection requires a timezone-aware instant")
        moment = instant.astimezone(UTC)
        # Windows are minute-granular and at least a minute long, so probing at
        # every whole minute from -PEAK_GUARD_BAND to +PEAK_LEAD_BAND covers
        # exactly [start - lead, end + guard): peak is any window touching it.
        lead = int(PEAK_LEAD_BAND / timedelta(minutes=1))
        guard = int(PEAK_GUARD_BAND / timedelta(minutes=1))
        return not any(
            self._peak(moment + timedelta(minutes=offset)) for offset in range(-guard, lead + 1)
        )

    def _peak(self, moment: datetime) -> bool:
        """Whether one UTC instant falls in a peak window on a non-holiday date."""
        if moment.date() in self.off_peak_dates:
            return False
        weekday = moment.isoweekday()
        minute = moment.hour * 60 + moment.minute
        return any(window.covers(weekday, minute) for window in self.peak_windows)


class GatewayTokenPrices(ContractModel):
    """Integer gateway attribution rates for one provider deployment.

    Values are integer nano-USD per million provider-reported tokens (one nano-USD is a
    billionth of a dollar: $1.25 per million is ``1_250_000_000``), bounded above by
    ``MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS``. ``None`` means the rate is unknown; it must
    never be interpreted as zero. Evaluation snapshots may freeze this complete card separately
    from provider invoice charges. Explicit four-rate snapshots remain a distinct price contract.
    Cache-creation rates price disjoint non-one-hour and one-hour write tokens. The
    unqualified rate prices the remainder after observed 1-hour writes; missing
    TTL evidence leaves write cost unknown unless both authored write rates are equal, never
    inferred from requested TTL or provider name.

    Attributes:
        input_nano_usd_per_million_tokens: Fresh input tokens.
        cached_input_nano_usd_per_million_tokens: Cache-read tokens inside the input.
        cache_creation_input_nano_usd_per_million_tokens: Five-minute cache writes.
        cache_creation_1h_input_nano_usd_per_million_tokens: One-hour cache writes.
        cache_creation_12h_input_nano_usd_per_million_tokens: OpenAI 12-hour cache writes.
            None (the default, excluded from identity while None) prices them at 2.5
            times the input rate, OpenAI's published total write rate
            (``token_cost.twelve_hour_write_rate``), never at a shorter lifetime's rate.
        output_nano_usd_per_million_tokens: Output tokens.
        reasoning_nano_usd_per_million_tokens: Reasoning tokens inside the output.
        image_output_nano_usd_per_million_tokens: Generated-image tokens inside the
            output. None (the default) prices them at the output rate, the behavior of
            every card without it; it is excluded from identity while None.
        long_context: Whole-request premium schedule for long-context input, or None.
        flex: Pass-through rates for ``service_tier='flex'``, or None.
        priority: Pass-through rates for ``service_tier='priority'``, or None.
        units: Per-unit rates for media billed by character, second, or image;
            None (the default) on every token-priced deployment, so a token
            card's snapshot identity is unchanged by the field's existence.
        off_peak: Time-of-day discount schedule, or None (the default, excluded
            from identity). The base rates are then the peak rates.
    """

    input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cached_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_1h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    cache_creation_12h_input_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    reasoning_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    image_output_nano_usd_per_million_tokens: NanoUsdRatePerMillionTokens = None
    long_context: GatewayLongContextTier | None = None
    """Whole-request premium schedule for long-context input, when one exists.

    Verified against the providers' published schedules (2026-08-30):
    Gemini prices ``prompts > 200k tokens`` at a higher whole-request rate
    for input, output, and cache reads; Anthropic's Claude 4.6+ models serve
    the full 1M window at standard pricing (no tier), so current Anthropic
    deployments leave this ``None``.
    """
    flex: GatewayServiceTierPrices | None = None
    """Pass-through rates when the caller requests ``service_tier='flex'``."""
    priority: GatewayServiceTierPrices | None = None
    """Pass-through rates when the caller requests ``service_tier='priority'``."""
    units: GatewayUnitPrices | None = None
    off_peak: GatewayOffPeakPrices | None = None
    """Off-peak replacement card and its UTC schedule; resolve it with :meth:`effective_at`."""

    @model_validator(mode="after")
    def _off_peak_bounded_by_base(self) -> Self:
        """Keep every off-peak rate at or below its peak rate, leg for leg.

        Paths that never resolve the schedule (cost ordering, price filters,
        batch) then price at the base card, which can only overstate an
        off-peak cost, never understate it. Tier and unit cards are refused
        beside a schedule: no provider combines them and the composition order
        would otherwise be ambiguous.
        """
        if self.off_peak is None:
            return self
        if self.flex is not None or self.priority is not None or self.units is not None:
            raise ValueError("an off-peak schedule cannot be combined with tier or unit cards")
        _require_bounded_card(self, self.off_peak.prices, "off-peak")
        base_long, off_long = self.long_context, self.off_peak.prices.long_context
        if (base_long is None) != (off_long is None):
            raise ValueError(
                "an off-peak card must carry a long-context tier exactly when the base does"
            )
        if base_long is not None and off_long is not None:
            if base_long.input_threshold_tokens != off_long.input_threshold_tokens:
                raise ValueError("an off-peak long-context tier must keep the base threshold")
            _require_bounded_card(base_long, off_long, "off-peak long-context")
        return self

    def effective_at(self, instant: datetime) -> GatewayTokenPrices:
        """Select the complete schedule for an attempt that starts at ``instant``.

        Args:
            instant: Timezone-aware attempt start.

        Returns:
            This card itself when it has no schedule (callers may test
            identity). Otherwise a schedule-free card: the base rates when
            ``instant`` is peak, else the off-peak card as a complete schedule,
            so a frozen attempt card never carries a window table.

        Raises:
            ValueError: ``instant`` is naive.
        """
        off_peak = self.off_peak
        if off_peak is None:
            return self
        if not off_peak.applies_at(instant):
            return self.model_copy(update={"off_peak": None})
        card = off_peak.prices
        return GatewayTokenPrices(
            input_nano_usd_per_million_tokens=card.input_nano_usd_per_million_tokens,
            cached_input_nano_usd_per_million_tokens=card.cached_input_nano_usd_per_million_tokens,
            cache_creation_input_nano_usd_per_million_tokens=card.cache_creation_input_nano_usd_per_million_tokens,
            cache_creation_1h_input_nano_usd_per_million_tokens=card.cache_creation_1h_input_nano_usd_per_million_tokens,
            cache_creation_12h_input_nano_usd_per_million_tokens=card.cache_creation_12h_input_nano_usd_per_million_tokens,
            output_nano_usd_per_million_tokens=card.output_nano_usd_per_million_tokens,
            reasoning_nano_usd_per_million_tokens=card.reasoning_nano_usd_per_million_tokens,
            image_output_nano_usd_per_million_tokens=card.image_output_nano_usd_per_million_tokens,
            long_context=card.long_context,
        )

    def service_tier(self, tier: str | None) -> GatewayServiceTierPrices | None:
        """Find the pass-through card for a requested processing tier.

        Args:
            tier: Requested tier; only flex and priority have separate cards.

        Returns:
            The configured card, or None for base pricing or an unconfigured tier.
        """
        if tier == "flex":
            return self.flex
        if tier == "priority":
            return self.priority
        return None

    def for_service_tier(self, tier: str | None) -> GatewayTokenPrices:
        """Select the complete effective schedule without inheriting missing tier prices.

        Args:
            tier: Requested provider processing tier.

        Returns:
            The tier card as a complete schedule with its own long-context pricing,
            or this schedule when no configured tier card applies.
        """
        card = self.service_tier(tier)
        if card is None:
            return self
        return GatewayTokenPrices(
            input_nano_usd_per_million_tokens=card.input_nano_usd_per_million_tokens,
            cached_input_nano_usd_per_million_tokens=card.cached_input_nano_usd_per_million_tokens,
            cache_creation_input_nano_usd_per_million_tokens=card.cache_creation_input_nano_usd_per_million_tokens,
            cache_creation_1h_input_nano_usd_per_million_tokens=card.cache_creation_1h_input_nano_usd_per_million_tokens,
            cache_creation_12h_input_nano_usd_per_million_tokens=card.cache_creation_12h_input_nano_usd_per_million_tokens,
            output_nano_usd_per_million_tokens=card.output_nano_usd_per_million_tokens,
            reasoning_nano_usd_per_million_tokens=card.reasoning_nano_usd_per_million_tokens,
            image_output_nano_usd_per_million_tokens=card.image_output_nano_usd_per_million_tokens,
            long_context=card.long_context,
            # Processing tiers reprice tokens only; a unit card is tier-independent.
            units=self.units,
        )


def _require_bounded_card(
    base: GatewayTokenPrices | GatewayLongContextTier,
    discounted: GatewayServiceTierPrices | GatewayLongContextTier,
    label: str,
) -> None:
    """Require each discounted rate to exist exactly where the base does, at or below it."""
    for name in _TOKEN_RATE_FIELDS:
        peak = getattr(base, name)
        rate = getattr(discounted, name)
        if (peak is None) != (rate is None):
            msg = f"the {label} {name} must be set exactly when the peak rate is"
            raise ValueError(msg)
        if peak is not None and rate is not None and rate > peak:
            msg = f"the {label} {name} exceeds the peak rate"
            raise ValueError(msg)

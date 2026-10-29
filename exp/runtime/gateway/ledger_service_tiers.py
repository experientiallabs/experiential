# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Persist frozen service-tier authority and keep unconfirmed billing reserved."""

from __future__ import annotations

import sqlite3

from exp.common.models.catalog_prices import GatewayTokenPrices
from exp.runtime.gateway.budgets import settle_attempt_budgets
from exp.runtime.gateway.contracts import GatewayEvent, GatewayUsage
from exp.runtime.gateway.ledger_errors import GatewayLedgerError
from exp.runtime.gateway.ledger_valuation import estimated_cost_nano_usd, usage_source_label
from exp.runtime.gateway.service_tiers import (
    GatewayServiceTierAdmission,
    GatewayServiceTierSettlement,
)


def record_tier_admission(
    connection: sqlite3.Connection, attempt_id: str, tier: GatewayServiceTierAdmission | None
) -> None:
    """Write complete immutable schedules in the same transaction as the reservation."""
    if tier is not None:
        connection.execute(
            "UPDATE gateway_attempts SET service_tier_admission = ? WHERE attempt_id = ?",
            (tier.model_dump_json(), attempt_id),
        )


def settle_tier(
    connection: sqlite3.Connection, attempt_id: str, receipt: GatewayServiceTierSettlement | None
) -> GatewayServiceTierSettlement | None:
    """Validate the receipt against persisted authority; missing evidence becomes a hold."""
    row = connection.execute(
        "SELECT service_tier_admission FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
    ).fetchone()
    if row is None or row[0] is None:
        if receipt is not None:
            raise GatewayLedgerError("service-tier receipt has no admitted schedules")
        return None
    admission = GatewayServiceTierAdmission.model_validate_json(row[0])
    if receipt is None:
        receipt = admission.settlement(served=None, resolution="missing")
    expected = admission.settlement(served=receipt.served, resolution=receipt.resolution)
    if receipt != expected:
        raise GatewayLedgerError("service-tier receipt differs from the admitted schedules")
    if receipt.prices is not None:
        prices = receipt.prices
        connection.execute(
            "UPDATE gateway_attempts SET input_rate = ?, cached_input_rate = ?, "
            "cache_creation_input_rate = ?, cache_creation_1h_input_rate = ?, "
            "output_rate = ?, reasoning_rate = ?, long_context_threshold_tokens = ?, "
            "long_context_input_rate = ?, long_context_cached_input_rate = ?, "
            "long_context_cache_creation_input_rate = ?, "
            "long_context_cache_creation_1h_input_rate = ?, "
            "long_context_output_rate = ?, long_context_reasoning_rate = ? WHERE attempt_id = ?",
            (
                prices.input_nano_usd_per_million_tokens,
                prices.cached_input_nano_usd_per_million_tokens,
                prices.cache_creation_input_nano_usd_per_million_tokens,
                prices.cache_creation_1h_input_nano_usd_per_million_tokens,
                prices.output_nano_usd_per_million_tokens,
                prices.reasoning_nano_usd_per_million_tokens,
                *long_context_values(prices),
                attempt_id,
            ),
        )
    connection.execute(
        "UPDATE gateway_attempts SET service_tier_settlement = ? WHERE attempt_id = ?",
        (receipt.model_dump_json(), attempt_id),
    )
    return receipt


def reconcile_tier_receipt(
    connection: sqlite3.Connection,
    attempt_id: str,
    receipt: GatewayServiceTierSettlement | None,
    terminal: GatewayEvent | None,
    usage: GatewayUsage | None,
) -> None:
    """Resolve held tier liability through an exact terminal replay, without another dispatch.

    The caller's transaction fences the state. Confirmed receipts can arrive after
    restart, but neither the original observed meter nor an already settled tier
    may change. A crash row without a meter can accept complete observed usage;
    its delivery outcome remains unknown. Rates must match the durable admission.
    """
    if receipt is None:
        return
    row = connection.execute(
        "SELECT * FROM gateway_attempts WHERE attempt_id = ?", (attempt_id,)
    ).fetchone()
    assert row is not None
    if row["service_tier_admission"] is None:
        raise GatewayLedgerError("service-tier receipt has no admitted schedules")
    admission = GatewayServiceTierAdmission.model_validate_json(row["service_tier_admission"])
    if receipt != admission.settlement(served=receipt.served, resolution=receipt.resolution):
        raise GatewayLedgerError("service-tier receipt differs from the admitted schedules")
    previous = row["service_tier_settlement"]
    prior = None if previous is None else GatewayServiceTierSettlement.model_validate_json(previous)
    if prior is not None and prior.resolution == "confirmed" and prior != receipt:
        raise GatewayLedgerError("service-tier receipt differs from the settled tier")
    if receipt.resolution != "confirmed":
        return
    fields = (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_creation_1h_input_tokens",
        "reasoning_tokens",
    )
    # A committed safe hold is already acknowledged, even without a priceable meter.
    # Only identical durable evidence is a no-op; a new receipt still owes validation.
    if (
        prior == receipt
        and row["budget_settled_nano_usd"] is None
        and all(
            row[field] == (None if usage is None else getattr(usage, field)) for field in fields
        )
        and row["usage_source"]
        == usage_source_label(usage, estimated=terminal is not None and terminal.usage_estimated)
        and tier_usage_cost(receipt, usage, terminal) is None
    ):
        return
    crash_meter = (
        row["state"] == "unknown_after_crash"
        and row["usage_source"] == "unknown"
        and row["budget_settled_nano_usd"] is None
        and all(row[field] is None for field in fields)
    )
    if crash_meter:
        if (
            usage is None
            or not usage.has_token_counts
            or terminal is None
            or terminal.usage_estimated
            or terminal.usage_incomplete_due_to_disconnect
        ):
            raise GatewayLedgerError("crash tier recovery requires complete observed usage")
    else:
        if usage is None or any(row[field] != getattr(usage, field) for field in fields):
            raise GatewayLedgerError("service-tier receipt replay differs from the durable usage")
        if row["usage_source"] not in ("observed", "estimated") or (
            row["usage_source"] == "estimated"
        ) != (terminal is not None and terminal.usage_estimated):
            raise GatewayLedgerError("service-tier receipt replay differs from the usage evidence")
    cost = tier_usage_cost(receipt, usage, terminal)
    if row["budget_settled_nano_usd"] is not None:
        if cost != row["budget_settled_nano_usd"]:
            raise GatewayLedgerError("service-tier replay differs from the settled amount")
        return
    if cost is None:
        raise GatewayLedgerError(
            "confirmed service-tier receipt still lacks a complete price or meter"
        )
    settle_tier(connection, attempt_id, receipt)
    if crash_meter:
        assert usage is not None
        # Only billing evidence is recovered. Delivery remains unknown after the crash.
        connection.execute(
            "UPDATE gateway_attempts SET input_tokens = ?, output_tokens = ?, "
            "cached_input_tokens = ?, cache_creation_input_tokens = ?, "
            "cache_creation_1h_input_tokens = ?, reasoning_tokens = ?, usage_source = 'observed' "
            "WHERE attempt_id = ? AND state = 'unknown_after_crash'",
            (*(getattr(usage, field) for field in fields), attempt_id),
        )
    connection.execute(
        "UPDATE gateway_attempts SET estimated_cost_nano_usd = ?, budget_settled_nano_usd = ? "
        "WHERE attempt_id = ? AND budget_settled_nano_usd IS NULL",
        (cost, cost, attempt_id),
    )
    settle_attempt_budgets(connection, attempt_id=attempt_id, settled_nano_usd=cost)


def tier_usage_cost(
    receipt: GatewayServiceTierSettlement, usage: GatewayUsage | None, terminal: GatewayEvent | None
) -> int | None:
    """Price only confirmed evidence, preserving unknown rates and incomplete meters."""
    prices = receipt.prices
    if prices is None or (
        terminal is not None
        and terminal.usage_incomplete_due_to_disconnect
        and not terminal.usage_estimated
    ):
        return None
    return card_usage_cost(prices, usage)


def long_context_values(prices: GatewayTokenPrices) -> tuple[int | None, ...]:
    """Freeze every long-context dimension beside the effective base schedule."""
    tier = prices.long_context
    if tier is None:
        return (None,) * 7
    return (
        tier.input_threshold_tokens,
        tier.input_nano_usd_per_million_tokens,
        tier.cached_input_nano_usd_per_million_tokens,
        tier.cache_creation_input_nano_usd_per_million_tokens,
        tier.cache_creation_1h_input_nano_usd_per_million_tokens,
        tier.output_nano_usd_per_million_tokens,
        tier.reasoning_nano_usd_per_million_tokens,
    )


def card_usage_cost(prices: GatewayTokenPrices, usage: GatewayUsage | None) -> int | None:
    """Apply the whole-request long-context schedule and every disjoint write dimension."""
    schedule = prices
    if (
        prices.long_context is not None
        and usage is not None
        and usage.input_tokens is not None
        and usage.input_tokens >= prices.long_context.input_threshold_tokens
    ):
        schedule = prices.long_context
    return estimated_cost_nano_usd(
        usage,
        input_rate=schedule.input_nano_usd_per_million_tokens,
        cached_input_rate=schedule.cached_input_nano_usd_per_million_tokens,
        cache_creation_input_rate=schedule.cache_creation_input_nano_usd_per_million_tokens,
        cache_creation_1h_input_rate=schedule.cache_creation_1h_input_nano_usd_per_million_tokens,
        output_rate=schedule.output_nano_usd_per_million_tokens,
        reasoning_rate=schedule.reasoning_nano_usd_per_million_tokens,
    )

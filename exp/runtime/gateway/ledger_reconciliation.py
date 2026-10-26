"""Reconcile late observed meters without replacing an attempt's delivery outcome."""

from __future__ import annotations

import sqlite3

from exp.runtime.gateway.budgets import MAXIMUM_NANO_USD, settle_attempt_budgets
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayEvent,
    GatewayFailure,
    GatewayServiceTierSettlement,
    GatewayUsage,
)
from exp.runtime.gateway.ledger_errors import GatewayLedgerError
from exp.runtime.gateway.ledger_service_tiers import (
    reconcile_tier_receipt,
    settle_tier,
    tier_usage_cost,
)
from exp.runtime.gateway.ledger_valuation import (
    billed_unit_columns,
    budget_settlement_nano_usd,
    frozen_usage_cost,
    observed_usage_cost,
    optional_int,
)

_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)
_UNIT_FIELDS = (
    "billed_unit_kind",
    "billed_unit_variant",
    "billed_quantity_milli",
    "billed_unit_rate",
)
_REPLAY_CONFLICT = (
    "attempt settlement replay differs from durable accounting evidence (durable usage or cost)"
)


def record_recovery(
    connection: sqlite3.Connection, attempt_id: str, terminal: GatewayEvent | None
) -> None:
    """Persist trusted provisional provenance alongside the initial terminal transaction."""
    if terminal is not None and terminal.accounting_recovery:
        if terminal.usage is not None or terminal.failure is None:
            raise GatewayLedgerError("accounting recovery requires an unmetered failure")
        connection.execute(
            "UPDATE gateway_attempts SET accounting_recovery = 1 WHERE attempt_id = ?",
            (attempt_id,),
        )


def reconcile_terminal_attempt(
    connection: sqlite3.Connection,
    attempt_id: str,
    state: str,
    terminal: GatewayEvent | None,
    failure: GatewayFailure | None,
    usage: GatewayUsage | None,
    receipt: GatewayServiceTierSettlement | None,
) -> None:
    """Upgrade provisional billing atomically while preserving the first delivery outcome.

    The caller owns the SQLite write transaction. Complete observed usage can
    replace an explicitly estimated meter or unknown usage with persisted
    recovery provenance, including cancellation/crash states. A provider's
    final terminal without usage remains immutable; another estimate or partial
    observation cannot replace any terminal meter. The original delivery state,
    failure and timestamps remain
    intact. An identical observed meter is replayable even when its outcome
    differs from the earlier cancellation or crash outcome.
    """
    row = connection.execute(
        "SELECT a.*, r.api_surface FROM gateway_attempts AS a "
        "JOIN gateway_requests AS r ON r.request_id = a.request_id WHERE a.attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    assert row is not None
    if (
        terminal is not None
        and terminal.accounting_release_reservation
        and (row["accounting_recovery"] or row["state"] in {"cancelled", "unknown_after_crash"})
        and (row["usage_source"] == "unknown" or row["budget_settled_nano_usd"] == 0)
    ):
        # A late customer waiver can follow an earlier provisional cleanup.
        # Keep provider evidence and the original delivery outcome. An already
        # waived row also acknowledges retry after a late exact meter commits.
        _replace_budget_charges(
            connection, attempt_id, optional_int(row["budget_settled_nano_usd"]), 0
        )
        connection.execute(
            "UPDATE gateway_attempts SET budget_settled_nano_usd = 0 WHERE attempt_id = ?",
            (attempt_id,),
        )
        return
    complete = (
        terminal is not None
        and not terminal.usage_estimated
        and not terminal.usage_incomplete_due_to_disconnect
        and usage is not None
        and (
            usage.has_token_counts
            or (
                usage.billed_units is not None
                and all(getattr(usage, field) is None for field in _TOKEN_FIELDS)
            )
        )
    )
    if complete and row["usage_source"] in {"unknown", "estimated"}:
        assert terminal is not None and usage is not None
        if (
            row["usage_source"] == "unknown"
            and not row["accounting_recovery"]
            and row["state"] not in {"cancelled", "unknown_after_crash"}
        ):
            raise GatewayLedgerError(_REPLAY_CONFLICT)
        _upgrade_observed(connection, row, attempt_id, terminal, failure, usage, receipt)
        return
    if complete and row["usage_source"] == "observed":
        assert terminal is not None and usage is not None
        same = all(row[field] == getattr(usage, field) for field in _TOKEN_FIELDS) and tuple(
            row[field] for field in _UNIT_FIELDS
        ) == billed_unit_columns(row, usage)
        prior = _prior_tier(row)
        if prior is not None and prior.resolution == "confirmed":
            if receipt not in (None, prior):
                raise GatewayLedgerError("service-tier receipt differs from the settled tier")
            same = same and _replay_cost_matches(row, tier_usage_cost(prior, usage, terminal))
        elif row["service_tier_admission"] is None:
            if receipt is not None:
                raise GatewayLedgerError("service-tier receipt has no admitted schedules")
            same = same and _replay_cost_matches(row, observed_usage_cost(row, usage, terminal))
        if not same:
            raise GatewayLedgerError(_REPLAY_CONFLICT)
        if prior is not None and prior.resolution != "confirmed" and receipt is not None:
            # The exact meter may predate its tier receipt. Validate the schedule
            # and resolve its hold without overwriting an explicit refund.
            _upgrade_observed(connection, row, attempt_id, terminal, failure, usage, receipt)
        return
    if row["state"] == state or (row["state"] == "unknown_after_crash" and receipt is not None):
        reconcile_tier_receipt(connection, attempt_id, receipt, terminal, usage)
        return
    raise GatewayLedgerError("attempt is already settled with another terminal state")


def _replay_cost_matches(row: sqlite3.Row, cost: int | None) -> bool:
    """Keep durable unknown pricing unchanged when exact public replay omits private markers."""
    # Public event JSON omits the trusted disconnect marker. Its exact observed
    # meter can therefore look priceable on replay, but the durable unknown cost
    # and conservative charge must remain untouched. Never promote that replay
    # into a billing correction; only known costs owe exact recomputation here.
    return row["estimated_cost_nano_usd"] is None or row["estimated_cost_nano_usd"] == cost


def _prior_tier(row: sqlite3.Row) -> GatewayServiceTierSettlement | None:
    """Decode the immutable tier evidence retained by the original terminal write."""
    previous = row["service_tier_settlement"]
    return None if previous is None else GatewayServiceTierSettlement.model_validate_json(previous)


def _upgrade_observed(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    attempt_id: str,
    terminal: GatewayEvent,
    failure: GatewayFailure | None,
    usage: GatewayUsage,
    receipt: GatewayServiceTierSettlement | None,
) -> None:
    """Replace only provisional usage and its automatic budget amounts."""
    prior = _prior_tier(row)
    if prior is not None and prior.resolution == "confirmed" and receipt not in (None, prior):
        raise GatewayLedgerError("service-tier receipt differs from the settled tier")
    tier = settle_tier(connection, attempt_id, prior if receipt is None else receipt)
    cost = (
        observed_usage_cost(row, usage, terminal)
        if tier is None
        else tier_usage_cost(tier, usage, terminal)
    )
    settled = budget_settlement_nano_usd(row, cost, usage, terminal) if tier is None else cost
    old_settled = optional_int(row["budget_settled_nano_usd"])
    provisional = optional_int(row["estimated_cost_nano_usd"])
    if provisional is None:
        provisional = optional_int(row["budget_reserved_nano_usd"])
    explicit_charge = old_settled is not None and (
        row["api_surface"] == GatewayApiSurface.DECISIONS.value
        or old_settled != provisional
        or old_settled == 0
    )
    # Preserve explicit liability assignments and previous guardrail refunds.
    # Their accounting decision is independent of the newly observed usage.
    # Legacy rows do not persist the refund flag, so a zero estimate cannot be
    # distinguished from a zero-charge disposition. Keep that charge unchanged
    # while retaining the complete observed meter and provider cost for audit.
    normalized_failure = failure or terminal.failure
    if explicit_charge:
        settled = old_settled
    elif (
        normalized_failure is not None
        and normalized_failure.safe_details.get("input_guardrail_denied") is True
    ):
        settled = 0
    _replace_budget_charges(connection, attempt_id, old_settled, settled)
    counterfactual = (
        None
        if row["preferred_deployment_id"] is None
        else frozen_usage_cost(row, usage, prefix="preferred_")
    )
    connection.execute(
        "UPDATE gateway_attempts SET input_tokens = ?, cached_input_tokens = ?, "
        "cache_creation_input_tokens = ?, cache_creation_1h_input_tokens = ?, output_tokens = ?, "
        "reasoning_tokens = ?, usage_source = 'observed', estimated_cost_nano_usd = ?, "
        "counterfactual_cost_nano_usd = ?, budget_settled_nano_usd = ?, "
        "billed_unit_kind = ?, billed_unit_variant = ?, "
        "billed_quantity_milli = ?, billed_unit_rate = ? "
        "WHERE attempt_id = ?",
        (
            *(getattr(usage, field) for field in _TOKEN_FIELDS),
            cost,
            counterfactual,
            settled,
            *billed_unit_columns(row, usage),
            attempt_id,
        ),
    )


def _replace_budget_charges(
    connection: sqlite3.Connection,
    attempt_id: str,
    previous: int | None,
    settled: int | None,
) -> None:
    """Apply a replacement delta to each original allocation, retaining manual overrides."""
    rows = connection.execute(
        "SELECT c.budget_id, c.reserved_nano_usd, c.settled_nano_usd, "
        "b.settled_nano_usd AS budget_settled, b.reserved_nano_usd AS budget_reserved, "
        "b.unknown_cost_attempts FROM gateway_attempt_budget_charges AS c "
        "JOIN gateway_monthly_budgets AS b ON b.budget_id = c.budget_id WHERE c.attempt_id = ?",
        (attempt_id,),
    ).fetchall()
    for row in rows:
        old = optional_int(row["settled_nano_usd"])
        # A separately reconciled unknown-cost allocation belongs to its operator.
        if old is None or previous is None or old != previous or old == settled:
            continue
        total = int(row["budget_settled"]) - old + (settled or 0)
        reserved = int(row["budget_reserved"])
        unknown = int(row["unknown_cost_attempts"])
        if settled is None:
            if row["reserved_nano_usd"] is None:
                unknown += 1
            else:
                reserved += int(row["reserved_nano_usd"])
        if not 0 <= total <= MAXIMUM_NANO_USD or not 0 <= reserved <= MAXIMUM_NANO_USD:
            raise GatewayLedgerError(
                "reconciled monthly gateway cost exceeds SQLite integer capacity"
            )
        updated = connection.execute(
            "UPDATE gateway_monthly_budgets SET settled_nano_usd = ?, reserved_nano_usd = ?, "
            "unknown_cost_attempts = ? WHERE budget_id = ? AND settled_nano_usd >= ?",
            (total, reserved, unknown, row["budget_id"], old),
        )
        if updated.rowcount != 1:
            raise GatewayLedgerError("monthly budget counters are inconsistent")
        connection.execute(
            "UPDATE gateway_attempt_budget_charges SET settled_nano_usd = ? "
            "WHERE budget_id = ? AND attempt_id = ?",
            (settled, row["budget_id"], attempt_id),
        )
    # Still-held allocations use the existing reserve/unknown-to-settled path.
    settle_attempt_budgets(connection, attempt_id=attempt_id, settled_nano_usd=settled)

"""Serialized terminal request certificates for safe capacity reentry."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayFailure
from exp.runtime.gateway.ledger_errors import GatewayLedgerError
from exp.runtime.gateway.ledger_valuation import terminal_values


def finish_request(
    connection: sqlite3.Connection,
    *,
    authorization: AuthorizationSnapshot,
    failure: GatewayFailure,
    terminal_at: datetime,
    certify_no_effects: bool = False,
) -> bool:
    """Persist one immutable no-effects certificate under the dispatch write fence.

    Args:
        connection: Open serialized write transaction; callers expose results after commit.
        authorization: Frozen authority identifying the accepted request.
        failure: Sanitized terminal failure.
        terminal_at: Wall-clock terminal time, never used to identify the newest owner.
        certify_no_effects: Trusted admission attestation that no paid prework was possible.
            Defaults false; historical or repeated finishes cannot upgrade a saved failure.

    Returns:
        Whether this failed direct request has a persisted certificate and no attempts.
    """
    state, _, _, _ = terminal_values(None, failure)
    row = connection.execute(
        """
        SELECT organization_id, terminal_state, failed_without_effects,
          NOT EXISTS (SELECT 1 FROM gateway_attempts AS a
            WHERE a.request_id = gateway_requests.request_id) AS no_dispatch
        FROM gateway_requests WHERE request_id = ?
        """,
        (authorization.request_id,),
    ).fetchone()
    if row is None:
        raise GatewayLedgerError("request was not durably accepted")
    if str(row["organization_id"]) != authorization.organization_id:
        raise GatewayLedgerError("request authority differs from accepted request")
    if row["terminal_state"] is not None:
        if str(row["terminal_state"]) == state:
            return bool(row["failed_without_effects"] and row["no_dispatch"])
        raise GatewayLedgerError("request is already settled with another terminal state")
    certified = (
        certify_no_effects
        and isinstance(authorization.target, DirectTarget)
        and state == "failed"
        and bool(row["no_dispatch"])
    )
    connection.execute(
        """
        UPDATE gateway_requests
        SET terminal_state = ?, terminal_at = ?, failed_without_effects = ?
        WHERE request_id = ? AND terminal_state IS NULL
        """,
        (state, utc_text(terminal_at), int(certified), authorization.request_id),
    )
    return certified

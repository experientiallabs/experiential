"""Host pricing of a decided native settlement before its terminal bytes flush."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.class_throttles import tier_settlement
from exp.runtime.gateway.disconnect_estimate import settled_terminal
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
    authority_error,
)
from exp.runtime.gateway.native_components import SettlementPricer, SyncWriteLedger
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_settlement import (
    settlement_metadata,
    terminal_from_settlement,
    tool_search_requests_from_terminal,
    tool_search_requests_kwarg,
    web_search_requests_from_terminal,
    web_search_requests_kwarg,
)
from exp.runtime.gateway.rung_admission import RungLoadRegistry


def quote_settlement(
    data: JsonObject,
    entry: InflightRequest,
    *,
    loads: RungLoadRegistry,
    ledger: SyncWriteLedger,
) -> str:
    """Hand the host the exact terminal its later settlement will write.

    The terminal is normalized exactly as ``_commit_settlement`` normalizes
    it (the disconnect estimate freezes its cache fraction on ``entry``, so
    the durable write reproduces the priced meter), and the host's
    ``price_attempt`` receives the same keywords ``finish_attempt`` will.
    Nothing is retained: a failed quote leaves no pending settlement, and the
    data plane then settles before any terminal byte.

    Args:
        data: The decided settlement payload the data plane will deliver.
        entry: The live in-flight request that owns the attempt.
        loads: Rung load registry supplying the organization's cache fraction.
        ledger: The write ledger; only a ``SettlementPricer`` can price.

    Returns:
        An empty JSON object once the host priced the settlement.

    Raises:
        NativeBridgeError: The ledger cannot price settlements, or this one failed.
    """
    attempt_id = str(data["attempt_id"])
    parsed = terminal_from_settlement(data, surface=entry.authorization.surface)
    terminal, failure = settled_terminal(data, entry, parsed=parsed, loads=loads)
    if isinstance(data.get("opened"), bool):
        terminal = terminal.with_attempt_observation(
            stream_opened=data["opened"] is True,
            paid_prework=not entry.no_paid_prework,
        )
    if not isinstance(ledger, SettlementPricer):
        # This ledger keeps every terminal frame behind a durable settlement.
        raise NativeBridgeError(accounting_recovering_error())
    price = ledger.price_attempt
    try:
        price(
            attempt_id=attempt_id,
            terminal_event=terminal,
            failure=failure,
            finalize_request=bool(data.get("finalize", True)),
            **settlement_metadata(data, price, **tier_settlement(entry, attempt_id)),
            **web_search_requests_kwarg(price, web_search_requests_from_terminal(terminal)),
            **tool_search_requests_kwarg(price, tool_search_requests_from_terminal(terminal)),
        )
    except Exception as exc:  # noqa: BLE001 - the data plane settles first instead.
        raise authority_error(exc) from exc
    return "{}"

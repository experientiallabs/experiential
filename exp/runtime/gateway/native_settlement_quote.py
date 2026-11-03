"""Host pricing of a decided native settlement before its terminal bytes flush."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from exp.runtime.gateway.class_throttles import tier_settlement
from exp.runtime.gateway.disconnect_estimate import settled_terminal
from exp.runtime.gateway.native_accounting_errors import (
    NativeBridgeError,
    accounting_recovering_error,
    authority_error,
)
from exp.runtime.gateway.native_components import SettlementPricer
from exp.runtime.gateway.native_recovery import retain_recovery_observation_time
from exp.runtime.gateway.native_settlement import (
    settlement_metadata,
    terminal_from_settlement,
    tool_search_requests_from_terminal,
    tool_search_requests_kwarg,
    web_search_requests_from_terminal,
    web_search_requests_kwarg,
)

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


def quote_settlement(accounting: NativeAttemptAccounting, argument: str) -> str:
    """Hand the host the exact terminal its later settlement will write.

    The terminal is normalized exactly as ``_commit_settlement`` normalizes
    it (the disconnect estimate freezes its cache fraction on the entry and
    the first terminal arrival is stamped, so the durable write reproduces
    the priced meter), and the host's ``price_attempt`` receives the keywords
    ``finish_attempt`` will. A priced settlement is retained on the entry as
    the pending settlement, so from that moment the recovery sweep replays
    it rather than abandoning the request, and the data plane's own later
    delivery commits that same payload exactly once.

    Args:
        accounting: The control plane that owns the in-flight request.
        argument: The decided settlement payload the data plane will deliver.

    Returns:
        An empty JSON object once the host priced the settlement.

    Raises:
        NativeBridgeError: The ledger cannot price settlements, the request is
            no longer live or already recovering, or the quote failed. The
            data plane then settles before any terminal byte.
    """
    ledger = accounting._write_ledger
    if not isinstance(ledger, SettlementPricer):
        # This ledger keeps every terminal frame behind a durable settlement.
        raise NativeBridgeError(accounting_recovering_error())
    data = json.loads(argument)
    with accounting._lock:
        entry = accounting._inflight.get(str(data["request_id"]))
    if entry is None or entry.pending_settlement is not None:
        raise NativeBridgeError(accounting_recovering_error())
    attempt_id = str(data["attempt_id"])
    parsed = terminal_from_settlement(data, surface=entry.authorization.surface)
    retain_recovery_observation_time(accounting.recovery, entry, attempt_id)
    terminal, failure = settled_terminal(data, entry, parsed=parsed, loads=accounting._loads)
    if isinstance(data.get("opened"), bool):
        terminal = terminal.with_attempt_observation(
            stream_opened=data["opened"] is True,
            paid_prework=not entry.no_paid_prework,
        )
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
    with accounting._lock:
        if accounting._inflight.get(str(data["request_id"])) is entry:
            if entry.pending_settlement is not None:
                raise NativeBridgeError(accounting_recovering_error())
            entry.pending_settlement = data
    return "{}"

"""Per-organization admission fence over failed durable accounting writes.

A terminal write that fails (an attempt settlement, an abandonment, or a
request-only finish) belongs to the organization whose authorization admitted
the request. While that write stays retained, only that organization's new
work is refused with ``accounting_unavailable``; every other organization on
the same worker keeps serving. The fence lifts per organization as soon as the
recovery sweep lands that organization's retained writes.

A deliberate ledger refusal (:class:`AccountingRefusedError`) is a clean
per-request error. Its write is still retained and replayed, but it never
fences any organization.

A process-wide accounting outage is not decided here. When the durable writer
cannot commit at all, every organization's admission already fails per request
at ``accept_request``, and the composition's own health answers readiness.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from exp.runtime.gateway.ledger_errors import AccountingRefusedError

if TYPE_CHECKING:
    from exp.runtime.gateway.native_execution import InflightRequest


def fences_after(exc: BaseException) -> bool:
    """Return whether a failed terminal write fences its organization.

    Args:
        exc: The exception the durable write raised.

    Returns:
        False for a deliberate ledger refusal, True for any other write failure.
    """
    return not isinstance(exc, AccountingRefusedError)


def entry_fences(entry: InflightRequest) -> bool:
    """Return whether one in-flight request holds a fencing failed write.

    Args:
        entry: The admitted request.

    Returns:
        Whether its retained settlement or abandonment failed to land for a
        reason other than a deliberate ledger refusal.
    """
    failed = entry.settlement_write_failed or entry.abandon_write_failed
    return failed and not entry.refused_write


def fenced_entry_organizations(entries: Iterable[InflightRequest]) -> set[str]:
    """Return the organizations owning a fencing failed write among ``entries``.

    Args:
        entries: Admitted in-flight requests, read under the registry lock.

    Returns:
        The organization ids whose new work must be refused.
    """
    return {entry.authorization.organization_id for entry in entries if entry_fences(entry)}

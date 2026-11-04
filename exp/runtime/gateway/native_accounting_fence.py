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

Admission asks :class:`OrganizationFence` in constant time: each registry
records the request ids holding a fencing write per organization as the
write fails, and releases them as the write lands or the request leaves, so
the check never scans in-flight work and an organization's entry is gone
once its last fencing write is.

A process-wide accounting outage is not decided here. When the durable writer
cannot commit at all, every organization's admission already fails per request
at ``accept_request``, and the composition's own health answers readiness.
"""

from __future__ import annotations

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


class OrganizationFence:
    """Request ids holding a fencing failed write, indexed by organization.

    Not thread-safe: the owning registry calls every method under its own
    lock. Lookups and updates are constant time, and an organization's entry
    is removed with its last request id, so memory follows the fenced writes
    outstanding rather than every organization ever seen.
    """

    def __init__(self) -> None:
        """Start with no fenced organization."""
        self._requests: dict[str, set[str]] = {}

    def mark(self, organization_id: str, request_id: str, *, fencing: bool) -> None:
        """Record whether one request currently holds a fencing failed write.

        Args:
            organization_id: The request's admitting organization.
            request_id: The request whose write state changed.
            fencing: Whether its retained write fences the organization now.
        """
        if fencing:
            self._requests.setdefault(organization_id, set()).add(request_id)
            return
        requests = self._requests.get(organization_id)
        if requests is None:
            return
        requests.discard(request_id)
        if not requests:
            del self._requests[organization_id]

    def sync(self, inflight: dict[str, InflightRequest], entry: InflightRequest) -> None:
        """Re-derive one request's fence state from its write flags.

        Only the live registry entry can fence: a request already removed or
        replaced releases its id instead, so a write that fails after its
        request left the registry never strands a fence.

        Args:
            inflight: The registry's in-flight requests, under its lock.
            entry: The admitted request whose write flags changed.
        """
        authorization = entry.authorization
        live = inflight.get(authorization.request_id) is entry
        self.mark(
            authorization.organization_id,
            authorization.request_id,
            fencing=live and entry_fences(entry),
        )

    def drop(self, inflight: dict[str, InflightRequest], request_id: str) -> InflightRequest | None:
        """Remove one request from ``inflight`` and release any fence it held.

        Args:
            inflight: The registry's in-flight requests, under its lock.
            request_id: The request leaving the registry.

        Returns:
            The removed request, or ``None`` when it was already gone.
        """
        entry = inflight.pop(request_id, None)
        if entry is not None:
            self.mark(entry.authorization.organization_id, request_id, fencing=False)
        return entry

    def fenced(self, organization_id: str) -> bool:
        """Return whether ``organization_id`` holds any fencing failed write."""
        return organization_id in self._requests

    def organizations(self) -> frozenset[str]:
        """Return every organization holding a fencing failed write."""
        return frozenset(self._requests)

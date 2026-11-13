"""Explicit admission handoff to the gateway's sole native accounting owner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from exp_gateway_native import NativeCallbacks

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.native_execution import InflightRequest

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


@dataclass(frozen=True, slots=True)
class NativeRequestHandoff:
    """Release initial host admission memos after native registration succeeds."""

    callbacks: NativeCallbacks
    registered: Callable[[], None]


class NativeRequestFactory(Protocol):
    """Bind both serving and failed admission to a retained native ledger.

    No request can select a Python accounting fallback. All callbacks must be
    interpreter independent, including terminal retries after serving stops.
    """

    def __call__(self, entry: InflightRequest, /) -> NativeCallbacks | NativeRequestHandoff:
        """Retain native financial authority for a fully resolved request."""
        ...

    def admission(self, authorization: AuthorizationSnapshot, /) -> NativeCallbacks:
        """Retain native terminal authority if admission cannot resolve a route."""
        ...


def register_admitted_request(
    accounting: NativeAttemptAccounting, entry: InflightRequest, prepare: Callable[[], None]
) -> None:
    """Start native input work before transferring the immutable admission context."""
    accounting.register(entry, prepare=prepare)


def uses_native(accounting: NativeAttemptAccounting, request_id: str) -> bool:
    """Check that the only serving owner registered the admitted request."""
    return accounting.native.owns_request(request_id)


def accounting_callbacks(accounting: NativeAttemptAccounting) -> NativeCallbacks:
    """Return the native owner registry bound to this control plane."""
    return accounting.native.callbacks()


def accounting_drained(accounting: NativeAttemptAccounting) -> bool:
    """Keep host resources alive until every native write is acknowledged."""
    return accounting.native.drained()

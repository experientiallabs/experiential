"""Initial handoff of local SQLite authority to the native accounting owner."""

from __future__ import annotations

import json
import time

from exp_gateway_native import NativeCallbacks, NativeSQLiteLedger

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.native_continuation_store import NativeContinuationStore
from exp.runtime.gateway.native_execution import InflightRequest


def sqlite_request(
    ledger: NativeSQLiteLedger,
    entry: InflightRequest,
    continuations: NativeContinuationStore | None = None,
) -> NativeCallbacks:
    """Freeze one admitted route; later reservations and settlements stay in Rust.

    Args:
        ledger: Shared native writer for the initialized local database.
        entry: Initial authority and certified deployment order.

    Returns:
        Retained interpreter-independent ledger target for this request.
    """
    return ledger.request(
        json.dumps(
            {
                "authorization": entry.authorization.model_dump(mode="json"),
                "snapshot": entry.route.snapshot.model_copy(
                    update={"authorization": entry.authorization}
                ).model_dump(mode="json"),
                "deployments": [
                    deployment.model_dump(mode="json") for deployment in entry.route.deployments
                ],
                "remaining_seconds": max(0.0, entry.deadline_monotonic - time.monotonic()),
            },
            separators=(",", ":"),
        ),
        continuations=None if continuations is None else continuations.native,
    )


class SQLiteRequestFactory:
    """Bind initial local admission to native accounting and continuation storage."""

    def __init__(self, ledger: NativeSQLiteLedger, continuations: NativeContinuationStore) -> None:
        """Retain the local native resources shared by all admitted requests."""
        self.ledger = ledger
        self.continuations = continuations

    def __call__(self, entry: InflightRequest, /) -> NativeCallbacks:
        """Hand one resolved request to its native ledger owner."""
        return sqlite_request(self.ledger, entry, self.continuations)

    def admission(self, authorization: AuthorizationSnapshot, /) -> NativeCallbacks:
        """Retain a writer even when admission ends before a route is resolved."""
        return self.ledger.admission(authorization.model_dump_json())

"""Content-free SQLite request, attempt, recovery, and usage accounting."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from wmo.common.core.artifacts import ContractModel
from wmo.common.models.gateway_catalog import ExactModelDeployment
from wmo.runtime.gateway.auth import utc_text
from wmo.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from wmo.runtime.gateway.interfaces import GatewayClock
from wmo.runtime.gateway.sqlite.migrations import connect_database, initialize_database
from wmo.runtime.gateway.sqlite.store import SystemGatewayClock


class GatewayLedgerError(ValueError):
    """A request or attempt transition violates the content-free ledger contract."""


class IdempotencyConflictError(GatewayLedgerError):
    """A caller operation key was reused with different canonical request content."""


class IdempotencyReplayUnavailableError(GatewayLedgerError):
    """A completed or accepted keyed request exists but its content cannot be replayed."""


class UsageTerminalCount(ContractModel):
    """Count of attempts ending in one normalized terminal state."""

    state: str
    attempts: int


class IdentityUsage(ContractModel):
    """Content-free usage totals for one identity."""

    organization_id: str
    identity_id: str
    requests: int
    attempts: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    known_estimated_cost_micro_usd: int
    unknown_cost_attempts: int
    total_latency_ms: int
    average_latency_ms: float | None
    terminal_counts: tuple[UsageTerminalCount, ...]


class SQLiteAttemptLedger:
    """Durable content-free attempt ledger sharing the gateway control database."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: GatewayClock | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Initialize a ledger on an existing or new gateway database.

        Args:
            database_path: Shared gateway SQLite path.
            clock: Injectable wall and monotonic clock.
            busy_timeout_ms: Maximum SQLite lock wait.
        """
        self.database_path = database_path
        self._clock = SystemGatewayClock() if clock is None else clock
        self._busy_timeout_ms = busy_timeout_ms
        initialize_database(database_path, busy_timeout_ms=busy_timeout_ms)

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.

        Raises:
            IdempotencyConflictError: The caller operation exists for another request.
            IdempotencyReplayUnavailableError: The matching operation already exists.
        """
        now = self._clock.now()
        remaining = max(0.0, authorization.deadline_monotonic - self._clock.monotonic())
        deadline_at = now + timedelta(seconds=remaining)
        with self._transaction() as connection:
            if authorization.caller_operation_sha256 is not None:
                prior = connection.execute(
                    """
                    SELECT canonical_request_sha256, terminal_state
                    FROM gateway_requests
                    WHERE organization_id = ? AND identity_id = ?
                      AND alias_revision_id = ? AND api_surface = ?
                      AND caller_operation_sha256 = ?
                    ORDER BY accepted_at DESC LIMIT 1
                    """,
                    (
                        authorization.organization_id,
                        authorization.identity_id,
                        authorization.alias_revision_id,
                        authorization.surface.value,
                        authorization.caller_operation_sha256,
                    ),
                ).fetchone()
                if prior is not None:
                    if str(prior["canonical_request_sha256"]) != (
                        authorization.canonical_request_sha256
                    ):
                        raise IdempotencyConflictError(
                            "caller operation key was reused with different request content"
                        )
                    if str(prior["terminal_state"]) not in {
                        "expired_before_dispatch",
                        "unknown_after_crash",
                    }:
                        raise IdempotencyReplayUnavailableError(
                            "matching keyed request exists but durable content replay "
                            "is unavailable"
                        )
            alias_row = connection.execute(
                """
                SELECT alias_id FROM alias_revisions
                WHERE organization_id = ? AND revision_id = ?
                """,
                (authorization.organization_id, authorization.alias_revision_id),
            ).fetchone()
            if alias_row is None:
                raise GatewayLedgerError("authorized alias revision is not present in the ledger")
            connection.execute(
                """
                INSERT INTO gateway_requests (
                    request_id, organization_id, identity_id, key_id, alias_id,
                    alias_revision_id, api_surface, canonical_request_sha256,
                    caller_operation_sha256, accepted_at, deadline_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    authorization.request_id,
                    authorization.organization_id,
                    authorization.identity_id,
                    authorization.virtual_key_id,
                    str(alias_row["alias_id"]),
                    authorization.alias_revision_id,
                    authorization.surface.value,
                    authorization.canonical_request_sha256,
                    authorization.caller_operation_sha256,
                    utc_text(now),
                    utc_text(deadline_at),
                ),
            )

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
    ) -> AttemptId:
        """Durably mark a provider dispatch before starting network work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.

        Returns:
            Stable new attempt ID.
        """
        if deployment.deployment_id not in snapshot.deployment_ids:
            raise GatewayLedgerError("attempt deployment is absent from the execution snapshot")
        if deployment.exact_model_id != snapshot.exact_model_id:
            raise GatewayLedgerError("attempt deployment changes the selected exact model")
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        prices = deployment.gateway.prices
        with self._transaction() as connection:
            request = connection.execute(
                """
                SELECT organization_id, terminal_state FROM gateway_requests
                WHERE request_id = ?
                """,
                (snapshot.authorization.request_id,),
            ).fetchone()
            if request is None:
                raise GatewayLedgerError("attempt request was not durably accepted")
            if str(request["organization_id"]) != snapshot.authorization.organization_id:
                raise GatewayLedgerError("attempt authority differs from accepted request")
            if request["terminal_state"] is not None:
                raise GatewayLedgerError("attempt request is already terminal")
            connection.execute(
                """
                INSERT INTO gateway_attempts (
                    attempt_id, request_id, organization_id, attempt_ordinal, route_depth,
                    deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
                    billing_source,
                    pricing_source, pricing_effective_at,
                    input_rate, cached_input_rate, output_rate, reasoning_rate,
                    state, started_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'dispatched', ?
                )
                """,
                (
                    attempt_id,
                    snapshot.authorization.request_id,
                    snapshot.authorization.organization_id,
                    attempt_ordinal,
                    route_depth,
                    deployment.deployment_id,
                    deployment.provider,
                    snapshot.exact_model_id,
                    snapshot.pool_id,
                    snapshot.authorization.catalog_sha256,
                    deployment.billing_source.value,
                    deployment.gateway.pricing_source,
                    (
                        None
                        if deployment.gateway.pricing_effective_at is None
                        else utc_text(deployment.gateway.pricing_effective_at)
                    ),
                    prices.input_micro_usd_per_million_tokens,
                    prices.cached_input_micro_usd_per_million_tokens,
                    prices.output_micro_usd_per_million_tokens,
                    prices.reasoning_micro_usd_per_million_tokens,
                    utc_text(self._clock.now()),
                ),
            )
        return attempt_id

    def record_route_context(
        self,
        *,
        attempt_id: AttemptId,
        route_reason: str | None,
        fallback_reason: str | None,
    ) -> None:
        """Attach display-safe learned-route context without request content.

        Args:
            attempt_id: Stable dispatched attempt ID.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
        """
        for value in (route_reason, fallback_reason):
            if value is not None and (len(value) > 512 or any(ord(char) < 32 for char in value)):
                raise GatewayLedgerError("route context must be a short display-safe code")
        with self._transaction() as connection:
            result = connection.execute(
                """
                UPDATE gateway_attempts SET route_reason = ?, fallback_reason = ?
                WHERE attempt_id = ? AND state = 'dispatched'
                """,
                (route_reason, fallback_reason, attempt_id),
            )
            if result.rowcount != 1:
                raise GatewayLedgerError("route context requires a dispatched attempt")

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Idempotently settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
        """
        state, normalized_failure, usage = _terminal_values(terminal_event, failure)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT request_id, state, input_rate, cached_input_rate,
                       output_rate, reasoning_rate
                FROM gateway_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise GatewayLedgerError("attempt does not exist")
            current_state = str(row["state"])
            if current_state != "dispatched":
                if current_state == state:
                    return
                raise GatewayLedgerError("attempt is already settled with another terminal state")
            cost = _estimated_cost(
                usage,
                input_rate=_optional_int(row["input_rate"]),
                cached_input_rate=_optional_int(row["cached_input_rate"]),
                output_rate=_optional_int(row["output_rate"]),
                reasoning_rate=_optional_int(row["reasoning_rate"]),
            )
            connection.execute(
                """
                UPDATE gateway_attempts
                SET state = ?, terminal_at = ?, failure_class = ?,
                    input_tokens = ?, cached_input_tokens = ?, output_tokens = ?,
                    reasoning_tokens = ?, usage_source = ?, estimated_cost_micro_usd = ?
                WHERE attempt_id = ? AND state = 'dispatched'
                """,
                (
                    state,
                    utc_text(self._clock.now()),
                    normalized_failure,
                    None if usage is None else usage.input_tokens,
                    None if usage is None else usage.cached_input_tokens,
                    None if usage is None else usage.output_tokens,
                    None if usage is None else usage.reasoning_tokens,
                    "unknown" if usage is None else "observed",
                    cost,
                    attempt_id,
                ),
            )
            if finalize_request and state in {"completed", "failed", "cancelled", "incomplete"}:
                connection.execute(
                    """
                    UPDATE gateway_requests SET terminal_state = ?, terminal_at = ?
                    WHERE request_id = ? AND terminal_state IS NULL
                    """,
                    (state, utc_text(self._clock.now()), str(row["request_id"])),
                )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Idempotently terminalize accepted work that never reached dispatch.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
        """
        state, normalized_failure, _ = _terminal_values(None, failure)
        del normalized_failure
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT organization_id, terminal_state FROM gateway_requests
                WHERE request_id = ?
                """,
                (authorization.request_id,),
            ).fetchone()
            if row is None:
                raise GatewayLedgerError("request was not durably accepted")
            if str(row["organization_id"]) != authorization.organization_id:
                raise GatewayLedgerError("request authority differs from accepted request")
            current = row["terminal_state"]
            if current is not None:
                if str(current) == state:
                    return
                raise GatewayLedgerError("request is already settled with another terminal state")
            connection.execute(
                """
                UPDATE gateway_requests SET terminal_state = ?, terminal_at = ?
                WHERE request_id = ? AND terminal_state IS NULL
                """,
                (state, utc_text(self._clock.now()), authorization.request_id),
            )

    def reconcile_crashed_requests(self, *, cleanup_grace: timedelta) -> tuple[int, int]:
        """Settle expired pre-dispatch and dispatched work after a crash.

        Args:
            cleanup_grace: Additional bound allowed for upstream cleanup after deadline.

        Returns:
            Counts of expired pre-dispatch requests and unknown dispatched attempts.
        """
        if cleanup_grace < timedelta(0):
            raise ValueError("cleanup grace cannot be negative")
        now = self._clock.now()
        expired_requests = 0
        unknown_attempts = 0
        with self._transaction() as connection:
            request_rows = connection.execute(
                """
                SELECT r.request_id, r.deadline_at,
                       EXISTS(
                           SELECT 1 FROM gateway_attempts AS a WHERE a.request_id = r.request_id
                       ) AS has_attempt
                FROM gateway_requests AS r WHERE r.terminal_state IS NULL
                """
            ).fetchall()
            for request in request_rows:
                deadline = datetime.fromisoformat(str(request["deadline_at"]))
                if int(request["has_attempt"]) == 0 and deadline <= now:
                    connection.execute(
                        """
                        UPDATE gateway_requests
                        SET terminal_state = 'expired_before_dispatch', terminal_at = ?
                        WHERE request_id = ? AND terminal_state IS NULL
                        """,
                        (utc_text(now), str(request["request_id"])),
                    )
                    expired_requests += 1
            attempt_rows = connection.execute(
                """
                SELECT a.attempt_id, a.request_id, r.deadline_at
                FROM gateway_attempts AS a
                JOIN gateway_requests AS r ON r.request_id = a.request_id
                WHERE a.state = 'dispatched'
                """
            ).fetchall()
            for attempt in attempt_rows:
                deadline = datetime.fromisoformat(str(attempt["deadline_at"]))
                if deadline + cleanup_grace > now:
                    continue
                connection.execute(
                    """
                    UPDATE gateway_attempts
                    SET state = 'unknown_after_crash', terminal_at = ?,
                        usage_source = 'unknown'
                    WHERE attempt_id = ? AND state = 'dispatched'
                    """,
                    (utc_text(now), str(attempt["attempt_id"])),
                )
                connection.execute(
                    """
                    UPDATE gateway_requests
                    SET terminal_state = 'unknown_after_crash', terminal_at = ?
                    WHERE request_id = ? AND terminal_state IS NULL
                    """,
                    (utc_text(now), str(attempt["request_id"])),
                )
                unknown_attempts += 1
        return expired_requests, unknown_attempts

    def usage(
        self, *, organization_id: str, identity_id: str | None = None
    ) -> tuple[IdentityUsage, ...]:
        """Aggregate request, usage, cost, and terminal states by identity.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional identity filter.

        Returns:
            Stable identity usage rows without prompts or outputs.
        """
        parameters: tuple[str, ...]
        predicate = "i.organization_id = ?"
        if identity_id is None:
            parameters = (organization_id,)
        else:
            predicate += " AND i.identity_id = ?"
            parameters = (organization_id, identity_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT i.identity_id,
                       COUNT(DISTINCT r.request_id) AS requests,
                       COUNT(a.attempt_id) AS attempts,
                       COALESCE(SUM(a.input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(a.cached_input_tokens), 0) AS cached_input_tokens,
                       COALESCE(SUM(a.output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(a.reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(a.estimated_cost_micro_usd), 0) AS known_cost,
                       COALESCE(SUM(CASE
                           WHEN a.attempt_id IS NOT NULL
                            AND a.estimated_cost_micro_usd IS NULL THEN 1 ELSE 0 END), 0
                       ) AS unknown_cost_attempts,
                       COALESCE(SUM(CASE WHEN a.terminal_at IS NOT NULL THEN
                           ROUND((julianday(a.terminal_at) - julianday(a.started_at)) * 86400000)
                           ELSE 0 END), 0) AS total_latency_ms,
                       AVG(CASE WHEN a.terminal_at IS NOT NULL THEN
                           (julianday(a.terminal_at) - julianday(a.started_at)) * 86400000
                           ELSE NULL END) AS average_latency_ms
                FROM identities AS i
                LEFT JOIN gateway_requests AS r
                  ON r.organization_id = i.organization_id AND r.identity_id = i.identity_id
                LEFT JOIN gateway_attempts AS a ON a.request_id = r.request_id
                WHERE {predicate}
                GROUP BY i.identity_id ORDER BY i.identity_id
                """,
                parameters,
            ).fetchall()
            terminal_rows = connection.execute(
                f"""
                SELECT i.identity_id, a.state, COUNT(*) AS attempts
                FROM identities AS i
                JOIN gateway_requests AS r
                  ON r.organization_id = i.organization_id AND r.identity_id = i.identity_id
                JOIN gateway_attempts AS a ON a.request_id = r.request_id
                WHERE {predicate} AND a.state != 'dispatched'
                GROUP BY i.identity_id, a.state ORDER BY i.identity_id, a.state
                """,
                parameters,
            ).fetchall()
        terminal_by_identity: dict[str, list[UsageTerminalCount]] = {}
        for row in terminal_rows:
            terminal_by_identity.setdefault(str(row["identity_id"]), []).append(
                UsageTerminalCount(state=str(row["state"]), attempts=int(row["attempts"]))
            )
        return tuple(
            IdentityUsage(
                organization_id=organization_id,
                identity_id=str(row["identity_id"]),
                requests=int(row["requests"]),
                attempts=int(row["attempts"]),
                input_tokens=int(row["input_tokens"]),
                cached_input_tokens=int(row["cached_input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                reasoning_tokens=int(row["reasoning_tokens"]),
                known_estimated_cost_micro_usd=int(row["known_cost"]),
                unknown_cost_attempts=int(row["unknown_cost_attempts"]),
                total_latency_ms=int(row["total_latency_ms"]),
                average_latency_ms=(
                    None if row["average_latency_ms"] is None else float(row["average_latency_ms"])
                ),
                terminal_counts=tuple(terminal_by_identity.get(str(row["identity_id"]), [])),
            )
            for row in rows
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open and close one configured read connection."""
        connection = connect_database(self.database_path, busy_timeout_ms=self._busy_timeout_ms)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one explicit immediate transaction with rollback on failure."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")


def _terminal_values(
    terminal_event: GatewayEvent | None,
    failure: GatewayFailure | None,
) -> tuple[str, str | None, GatewayUsage | None]:
    """Normalize one finish call to state, safe failure class, and usage."""
    event_failure = None if terminal_event is None else terminal_event.failure
    normalized = failure or event_failure
    if terminal_event is None and normalized is None:
        raise GatewayLedgerError("attempt finish needs a terminal event or failure")
    if terminal_event is not None and terminal_event.kind not in {
        GatewayEventKind.COMPLETED,
        GatewayEventKind.INCOMPLETE,
        GatewayEventKind.FAILED,
    }:
        raise GatewayLedgerError("attempt finish event must be terminal")
    if normalized is not None:
        state = (
            "cancelled" if normalized.failure_class == GatewayFailureClass.CANCELLED else "failed"
        )
        return (
            state,
            normalized.failure_class.value,
            (None if terminal_event is None else terminal_event.usage),
        )
    assert terminal_event is not None
    return terminal_event.kind.value, None, terminal_event.usage


def _estimated_cost(
    usage: GatewayUsage | None,
    *,
    input_rate: int | None,
    cached_input_rate: int | None,
    output_rate: int | None,
    reasoning_rate: int | None,
) -> int | None:
    """Compute attributed integer micro-USD or preserve unknown pricing."""
    if usage is None:
        return None
    dimensions = (
        (usage.input_tokens, input_rate),
        (usage.cached_input_tokens or 0, cached_input_rate),
        (usage.output_tokens, output_rate),
        (usage.reasoning_tokens or 0, reasoning_rate),
    )
    if any(tokens > 0 and rate is None for tokens, rate in dimensions):
        return None
    numerator = sum(tokens * (rate or 0) for tokens, rate in dimensions)
    return (numerator + 500_000) // 1_000_000


def _optional_int(value: int | None) -> int | None:
    """Convert one nullable SQLite integer value to its precise type."""
    return None if value is None else int(value)

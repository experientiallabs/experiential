"""Content-free SQLite request, attempt, recovery, and usage accounting."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path

from exp_gateway_native import NativeSQLiteLedger
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.sqlite.connection import persistent_connection
from exp.common.sqlite.native import native_connection
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.budgets import (
    BudgetRefusalBinding,
    BudgetReservationRejected,
    BudgetScope,
)
from exp.runtime.gateway.contracts import (
    AttemptId,
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayFailure,
    GatewayServiceTierAdmission,
    GatewayServiceTierSettlement,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.ledger_errors import (
    AccountingRecoverySupersededError,
    AccountingRecoveryUnresolvedTerminalError,
)
from exp.runtime.gateway.ledger_errors import (
    AttemptRejectedError as AttemptRejectedError,
)
from exp.runtime.gateway.ledger_errors import (
    GatewayLedgerError as GatewayLedgerError,
)
from exp.runtime.gateway.ledger_errors import (
    IdempotencyConflictError as IdempotencyConflictError,
)
from exp.runtime.gateway.ledger_errors import (
    IdempotencyReplayUnavailableError as IdempotencyReplayUnavailableError,
)
from exp.runtime.gateway.ledger_usage import (
    BillingSourceUsage,
    IdentityUsage,
    LedgerUsageSnapshot,
    billing_source_usage_rows,
    client_app_usage_rows,
    identity_usage_rows,
)
from exp.runtime.gateway.ledger_valuation import terminal_values as _terminal_values
from exp.runtime.gateway.model_chain_authority import (
    ChainOperation,
    LocalSnapshotMemoOwner,
    SnapshotClassificationMemo,
    SQLiteChainPreflight,
    prepare_sqlite_chain_authority,
    serving_snapshot_limit,
)
from exp.runtime.gateway.native_continuation_store import NativeContinuationStore
from exp.runtime.gateway.native_sqlite import SQLiteRequestFactory
from exp.runtime.gateway.sqlite.migrations import initialize_database
from exp.runtime.gateway.sqlite.store import SystemGatewayClock
from exp.runtime.openai_protocol.state import BoundedContinuationStore


class SQLiteAttemptLedger(LocalSnapshotMemoOwner):
    """Durable content-free attempt ledger sharing the gateway control database."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: GatewayClock | None = None,
        busy_timeout_ms: int = 5_000,
        serving_snapshot_max_bytes: int | None = None,
        classification_memo: SnapshotClassificationMemo | None = None,
    ) -> None:
        """Initialize a ledger on an existing or new gateway database.

        Args:
            database_path: Shared gateway SQLite path.
            clock: Injectable wall and monotonic clock.
            busy_timeout_ms: Maximum SQLite lock wait.
            serving_snapshot_max_bytes: Per-file serving bound, default 64 MiB; no settings lookup.
            classification_memo: Borrowed composition memo; omitted creates a private close owner.
        """
        self._bind_classification_memo(classification_memo)
        self.serving_snapshot_max_bytes = serving_snapshot_limit(serving_snapshot_max_bytes)
        self.database_path = database_path
        self._clock = SystemGatewayClock() if clock is None else clock
        self._busy_timeout_ms = busy_timeout_ms
        initialize_database(database_path, busy_timeout_ms=busy_timeout_ms)
        self._native = NativeSQLiteLedger(
            str(database_path),
            busy_timeout_ms=busy_timeout_ms,
            maximum_snapshot_bytes=self.serving_snapshot_max_bytes,
        )

    def _native_apply(
        self, connection: sqlite3.Connection, operation: str, payload: JsonObject
    ) -> JsonValue:
        """Invoke the native writer inside a cold caller's existing transaction."""
        payload["now"] = utc_text(self._clock.now())
        try:
            return json.loads(
                self._native.apply(operation, json.dumps(payload), native_connection(connection))
            )
        except ValueError as error:
            detail = json.loads(str(error))
            if detail.get("code") == "idempotency_conflict":
                raise IdempotencyConflictError(detail["message"]) from error
            if detail.get("code") == "idempotency_replay_unavailable":
                raise IdempotencyReplayUnavailableError(detail["message"]) from error
            if detail.get("code") == "accounting_recovery_superseded":
                raise AccountingRecoverySupersededError(detail["message"]) from error
            if detail.get("code") == "accounting_recovery_unresolved":
                raise AccountingRecoveryUnresolvedTerminalError(detail["message"]) from error
            if detail.get("code") == "accounting_integrity":
                raise sqlite3.IntegrityError(detail["message"]) from error
            raise GatewayLedgerError(
                detail.get("message", "Native accounting write failed")
            ) from error

    def native_request_factory(
        self, continuations: BoundedContinuationStore
    ) -> SQLiteRequestFactory:
        """Bind local serving to native writes and the same native continuation store."""
        if not isinstance(continuations, NativeContinuationStore):
            raise TypeError("Local native accounting requires a NativeContinuationStore")
        return SQLiteRequestFactory(
            self._native,
            continuations,
        )

    @contextmanager
    def prepare_chain_authority(
        self,
        authorization: AuthorizationSnapshot,
        operation: ChainOperation,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[SQLiteChainPreflight | None]:
        """Classify before BEGIN, borrowing the writer's connection when supplied."""
        with (
            self._connect() if connection is None else nullcontext(connection) as connection,
            prepare_sqlite_chain_authority(
                connection,
                authorization.organization_id,
                authorization.alias_revision_id,
                request_id=authorization.request_id,
                operation=operation,
                maximum_bytes=self.serving_snapshot_max_bytes,
                remaining_seconds=authorization.deadline_monotonic - self._clock.monotonic(),
                classification_memo=self.classification_memo,
            ) as proof,
        ):
            yield proof

    def _require_chain_authority(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
        chain_preflight: SQLiteChainPreflight | None,
        operation: ChainOperation,
        staged: bool = False,
    ) -> None:
        """Require a live preflight and fence policy at the atomic write boundary."""
        SQLiteChainPreflight.require_ledger(
            connection, authorization, chain_preflight, operation, staged=staged
        )

    @property
    def busy_timeout_ms(self) -> int:
        """Return the configured SQLite lock-wait bound."""
        return self._busy_timeout_ms

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Persist accepted authority before route selection or dispatch.

        Args:
            authorization: Frozen authority and request identity.

        Raises:
            IdempotencyConflictError: The caller operation exists for another request.
            IdempotencyReplayUnavailableError: The matching operation already exists.
        """
        with (
            self._connect() as connection,
            self.prepare_chain_authority(authorization, "accept", connection=connection) as proof,
            self._transaction(connection=connection),
        ):
            self.apply_accept_request(
                connection, authorization=authorization, chain_preflight=proof
            )

    def apply_accept_request(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
        chain_preflight: SQLiteChainPreflight | None = None,
    ) -> None:
        """Run the acceptance write inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            authorization: Frozen authority and request identity.
            chain_preflight: Exact live pre-BEGIN proof; missing or closed proof is refused.

        Raises:
            IdempotencyConflictError: The caller operation exists for another request.
            IdempotencyReplayUnavailableError: The matching operation already exists.
        """
        self._require_chain_authority(
            connection,
            authorization=authorization,
            chain_preflight=chain_preflight,
            operation="accept",
        )
        remaining = max(0.0, authorization.deadline_monotonic - self._clock.monotonic())
        self._native_apply(
            connection,
            "accept",
            {
                "authorization": authorization.model_dump(mode="json"),
                "remaining_seconds": remaining,
                "deadline_at": utc_text(self._clock.now() + timedelta(seconds=remaining)),
            },
        )

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_nano_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
        service_tier: GatewayServiceTierAdmission | None = None,
    ) -> AttemptId:
        """Durably mark a provider dispatch before starting network work.

        Args:
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_nano_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, given
                only when it differs from ``deployment``; its base rates are
                frozen for the settle-time counterfactual cost.

        Returns:
            Stable new attempt ID.
        """
        with (
            self._connect() as connection,
            self.prepare_chain_authority(
                snapshot.authorization, "reserve", connection=connection
            ) as proof,
            self._transaction(connection=connection),
        ):
            return self.apply_start_attempt(
                connection,
                chain_preflight=proof,
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_nano_usd=maximum_cost_nano_usd,
                reserved_input_tokens=reserved_input_tokens,
                reserved_output_tokens=reserved_output_tokens,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
                service_tier=service_tier,
            )

    def apply_start_attempt(
        self,
        connection: sqlite3.Connection,
        *,
        chain_preflight: SQLiteChainPreflight | None = None,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_nano_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
        service_tier: GatewayServiceTierAdmission | None = None,
    ) -> AttemptId:
        """Run the dispatch reservation inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            chain_preflight: Exact live pre-BEGIN proof; missing or closed proof is refused.
            snapshot: Route-bound immutable request plan.
            deployment: Exact deployment about to receive the request.
            attempt_ordinal: Zero-based physical dispatch position for this request.
            route_depth: Zero-based operational route position.
            maximum_cost_nano_usd: Conservative charge reserved before dispatch.
            route_reason: Optional learned-selection reason code.
            fallback_reason: Optional embedding or router fallback reason code.
            dispatch_reason: Optional policy-dispatch disclosure code.
            preferred_deployment: The route's bypassed preferred rung, given
                only when it differs from ``deployment``; its base rates are
                frozen for the settle-time counterfactual cost.

        Returns:
            Stable new attempt ID.
        """
        self._require_chain_authority(
            connection,
            authorization=snapshot.authorization,
            chain_preflight=chain_preflight,
            operation="reserve",
            staged=bool(snapshot.model_stages),
        )
        lanes: list[JsonObject | None] = [None] * len(snapshot.deployment_ids)
        lanes[route_depth] = deployment.model_dump(mode="json")
        if preferred_deployment is not None:
            lanes[snapshot.deployment_ids.index(preferred_deployment.deployment_id)] = (
                preferred_deployment.model_dump(mode="json")
            )
        result = self._native_apply(
            connection,
            "start",
            {
                "authorization": snapshot.authorization.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
                "deployments": lanes,
                "remaining_seconds": max(
                    0.0, snapshot.authorization.deadline_monotonic - self._clock.monotonic()
                ),
                "dispatch": {
                    "ordinal": attempt_ordinal,
                    "depth": route_depth,
                    "prices": (
                        deployment.gateway.prices
                        if service_tier is None
                        else service_tier.requested_prices
                    ).model_dump(mode="json"),
                    "maximum_cost_nano": maximum_cost_nano_usd,
                    "reserved_input_tokens": reserved_input_tokens,
                    "reserved_output_tokens": reserved_output_tokens,
                    "route_reason": route_reason,
                    "fallback_reason": fallback_reason,
                    "dispatch_reason": dispatch_reason,
                    "preferred_deployment_id": None
                    if preferred_deployment is None
                    else preferred_deployment.deployment_id,
                    "service_tier": None
                    if service_tier is None
                    else service_tier.model_dump(mode="json"),
                },
            },
        )
        if not isinstance(result, dict):
            raise GatewayLedgerError("Native reservation returned an invalid result")
        if result.get("refused") is not None:
            refusal = result["refused"]
            if not isinstance(refusal, dict) or not isinstance(refusal.get("error"), dict):
                raise GatewayLedgerError("Native reservation returned an invalid refusal")
            error = refusal["error"]
            assert isinstance(error, dict)
            reason = error.get("message")
            if not isinstance(reason, str):
                raise GatewayLedgerError("Native reservation returned an invalid reason")
            scope = BudgetScope.model_validate(refusal["budget_scope"])
            raise BudgetReservationRejected(
                scope_kind=scope.kind,
                reason=reason,
                binding=BudgetRefusalBinding.model_validate(
                    {
                        "request_id": snapshot.authorization.request_id,
                        "organization_id": snapshot.authorization.organization_id,
                        "alias_revision_id": snapshot.authorization.alias_revision_id,
                        "scope": scope,
                        "application": refusal["application"],
                    }
                ),
            )
        return str(result["attempt_id"])

    def finish_attempt(
        self,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int = 0,
        tool_search_requests: int = 0,
        service_tier: GatewayServiceTierSettlement | None = None,
    ) -> None:
        """Idempotently settle one attempt with normalized content-free fields.

        Args:
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
            retry_after_seconds: Provider-stated ``Retry-After`` wait, when one was harvested.
            ratelimit_limit_requests: Provider-stated request-rate ceiling.
            ratelimit_remaining_requests: Provider-stated requests remaining.
            ratelimit_limit_tokens: Provider-stated token-rate ceiling.
            ratelimit_remaining_tokens: Provider-stated tokens remaining.
            upstream_provider: The upstream an aggregator rung (OpenRouter) named as serving.
            web_search_requests: Gateway-executed web searches billed to the attempt.
            tool_search_requests: Gateway-executed tool-search rounds billed to the attempt.
                Both meters are priced by the hosted ledger; not yet persisted or priced locally.
        """
        with self._transaction() as connection:
            self.apply_finish_attempt(
                connection,
                attempt_id=attempt_id,
                terminal_event=terminal_event,
                failure=failure,
                finalize_request=finalize_request,
                first_token_at=first_token_at,
                retry_after_seconds=retry_after_seconds,
                ratelimit_limit_requests=ratelimit_limit_requests,
                ratelimit_remaining_requests=ratelimit_remaining_requests,
                ratelimit_limit_tokens=ratelimit_limit_tokens,
                ratelimit_remaining_tokens=ratelimit_remaining_tokens,
                upstream_provider=upstream_provider,
                web_search_requests=web_search_requests,
                tool_search_requests=tool_search_requests,
                service_tier=service_tier,
            )

    def apply_finish_attempt(
        self,
        connection: sqlite3.Connection,
        *,
        attempt_id: AttemptId,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int = 0,
        tool_search_requests: int = 0,
        service_tier: GatewayServiceTierSettlement | None = None,
    ) -> None:
        """Run the attempt settlement inside the caller's open write transaction.

        Args:
            connection: Open write transaction owned by the caller.
            attempt_id: Stable attempt ID.
            terminal_event: Provider terminal event, possibly carrying usage.
            failure: Sanitized failure when no successful terminal event exists.
            finalize_request: Whether this attempt is the final route for its parent request.
            first_token_at: Wall-clock time the attempt streamed its first token, or ``None``.
            retry_after_seconds: Provider-stated ``Retry-After`` wait, when one was harvested.
            ratelimit_limit_requests: Provider-stated request-rate ceiling.
            ratelimit_remaining_requests: Provider-stated requests remaining.
            ratelimit_limit_tokens: Provider-stated token-rate ceiling.
            ratelimit_remaining_tokens: Provider-stated tokens remaining.
            upstream_provider: The upstream an aggregator rung (OpenRouter) named as serving.
            web_search_requests: Accepted for the shared settle signature (see ``finish_attempt``).
            tool_search_requests: Accepted alongside ``web_search_requests``; same local status.
        """
        state, _, _, usage = _terminal_values(terminal_event, failure)
        normalized = failure or (None if terminal_event is None else terminal_event.failure)
        terminal: JsonObject = (
            {}
            if terminal_event is None
            else {
                "accounting_recovery": terminal_event.accounting_recovery,
                "accounting_release_reservation": terminal_event.accounting_release_reservation,
                "usage_estimated": terminal_event.usage_estimated,
                "usage_incomplete_due_to_disconnect": (
                    terminal_event.usage_incomplete_due_to_disconnect
                ),
                "decision_provider_rejected": terminal_event.decision_provider_rejected,
            }
        )
        terminal.update(
            {
                "state": state,
                "usage": None if usage is None else usage.model_dump(mode="json"),
                "failure": None if normalized is None else normalized.model_dump(mode="json"),
                "input_guardrail_denied": normalized is not None
                and normalized.safe_details.get("input_guardrail_denied") is True,
                "finalize": finalize_request,
                "first_token_at": None if first_token_at is None else utc_text(first_token_at),
                "service_tier": None
                if service_tier is None
                else service_tier.model_dump(mode="json"),
                "observations": {
                    "retry_after_seconds": retry_after_seconds,
                    "ratelimit_limit_requests": ratelimit_limit_requests,
                    "ratelimit_remaining_requests": ratelimit_remaining_requests,
                    "ratelimit_limit_tokens": ratelimit_limit_tokens,
                    "ratelimit_remaining_tokens": ratelimit_remaining_tokens,
                    "upstream_provider": upstream_provider,
                    "web_search_requests": web_search_requests,
                    "tool_search_requests": tool_search_requests,
                },
            }
        )
        self._native_apply(connection, "settle", {"attempt_id": attempt_id, "terminal": terminal})

    def reconcile_decision_liability(
        self,
        *,
        attempt_id: AttemptId,
        assigned_cost_nano_usd: int,
    ) -> None:
        """Resolve one terminal decision's held liability at an operator-assigned cost.

        Assignment changes budget accounting only, never provider usage or its
        unknown cost estimate. Repeating the same assignment is a no-op; a
        conflicting assignment or an attempt without a held bound is refused.
        """
        if (
            isinstance(assigned_cost_nano_usd, bool)
            or not isinstance(assigned_cost_nano_usd, int)
            or not 0 <= assigned_cost_nano_usd <= 9_223_372_036_854_775_807
        ):
            raise ValueError("assigned decision cost must fit a nonnegative SQLite integer")
        with self._transaction() as connection:
            self._native_apply(
                connection,
                "reconcile_liability",
                {"attempt_id": attempt_id, "assigned_cost_nano_usd": assigned_cost_nano_usd},
            )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Terminalize accepted work and return its committed no-effects certificate.

        Args:
            authorization: Frozen authority identifying the accepted request.
            failure: Sanitized pre-dispatch terminal failure.
            certify_no_effects: Trusted proof that admission could not perform paid prework.
            web_search_requests: Completed searches with no model attempt to own the meter.
        """
        with self._transaction() as connection:
            return self.apply_finish_request(
                connection,
                authorization=authorization,
                failure=failure,
                certify_no_effects=certify_no_effects,
                web_search_requests=web_search_requests,
            )

    def apply_finish_request(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Apply :meth:`finish_request` inside the caller's serialized write transaction."""
        result = self._native_apply(
            connection,
            "finish",
            {
                "authorization": authorization.model_dump(mode="json"),
                "terminal": {
                    "state": "cancelled"
                    if failure.failure_class.value == "cancelled"
                    else "failed",
                    "failure": failure.model_dump(mode="json"),
                },
                "certify_no_effects": certify_no_effects,
                "web_search_requests": web_search_requests,
            },
        )
        if not isinstance(result, bool):
            raise GatewayLedgerError("Native accounting returned an invalid certificate")
        return result

    def reconcile_crashed_requests(self, *, cleanup_grace: timedelta) -> tuple[int, int]:
        """Settle expired pre-dispatch and dispatched work after a crash.

        Args:
            cleanup_grace: Additional bound allowed for upstream cleanup after deadline.

        Returns:
            Counts of expired pre-dispatch requests and unknown dispatched attempts.
        """
        if cleanup_grace < timedelta(0):
            raise ValueError("cleanup grace cannot be negative")
        with self._transaction() as connection:
            counts = self._native_apply(
                connection, "recover", {"cleanup_grace_seconds": int(cleanup_grace.total_seconds())}
            )
            if not isinstance(counts, list) or len(counts) != 2:
                raise GatewayLedgerError("Native recovery returned invalid counts")
            expired, unknown = counts
            if not isinstance(expired, int) or not isinstance(unknown, int):
                raise GatewayLedgerError("Native recovery returned invalid counts")
            return expired, unknown

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
        return self.usage_snapshot(
            organization_id=organization_id,
            identity_id=identity_id,
        ).identities

    def usage_by_billing_source(
        self,
        *,
        organization_id: str,
        identity_id: str | None = None,
    ) -> tuple[BillingSourceUsage, ...]:
        """Aggregate physical attempts by their frozen credential ownership source.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional identity filter applied through the parent request.

        Returns:
            Deterministic source buckets without partitioning logical request counts.
        """
        return self.usage_snapshot(
            organization_id=organization_id,
            identity_id=identity_id,
        ).by_billing_source

    def usage_snapshot(
        self,
        *,
        organization_id: str,
        identity_id: str | None = None,
    ) -> LedgerUsageSnapshot:
        """Read identity and source aggregates from one explicit SQLite snapshot.

        Args:
            organization_id: Tenant boundary.
            identity_id: Optional exact identity filter.

        Returns:
            Internally conserving usage aggregates from one WAL read transaction.
        """
        parameters: tuple[str, ...]
        predicate = "i.organization_id = ?"
        source_predicate = "r.organization_id = ?"
        if identity_id is None:
            parameters = (organization_id,)
        else:
            predicate += " AND i.identity_id = ?"
            source_predicate += " AND r.identity_id = ?"
            parameters = (organization_id, identity_id)
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                identities = identity_usage_rows(
                    connection,
                    organization_id=organization_id,
                    predicate=predicate,
                    parameters=parameters,
                )
                by_billing_source = billing_source_usage_rows(
                    connection,
                    predicate=source_predicate,
                    parameters=parameters,
                )
                by_client_app = client_app_usage_rows(
                    connection, predicate=source_predicate, parameters=parameters
                )
            finally:
                connection.rollback()
        return LedgerUsageSnapshot(
            identities=identities, by_billing_source=by_billing_source, by_client_app=by_client_app
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Check out one reusable configured connection."""
        with persistent_connection(
            self.database_path, busy_timeout_ms=self._busy_timeout_ms
        ) as connection:
            yield connection

    @contextmanager
    def _transaction(
        self, *, connection: sqlite3.Connection | None = None
    ) -> Iterator[sqlite3.Connection]:
        """Run an immediate transaction, optionally borrowing the preflight connection."""
        with self._connect() if connection is None else nullcontext(connection) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")

"""SQLite composition adapter for the storage-neutral gateway platform."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from exp.common.core.artifacts import stable_id
from exp.common.models.gateway_catalog import read_pinned_normalized_snapshot
from exp.runtime.gateway.budgets import (
    BudgetScope,
    BudgetScopeKind,
    SQLiteBudgetStore,
)
from exp.runtime.gateway.contracts import GatewayFailureClass, GatewayUsage, ProjectTarget
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.management import require_gateway_servable_provider
from exp.runtime.gateway.platform import (
    ActivateAliasRevisionCommand,
    AliasMutationCommand,
    AliasRevisionRecord,
    AttemptReservationRecord,
    AttemptReservationRequest,
    AttemptSettlementRecord,
    AttemptSettlementRequest,
    AttemptTerminalState,
    AttemptUsageSource,
    BillingSourceUsageAttribution,
    CreateIdentityCommand,
    DisableAliasCommand,
    ExactPoolRevision,
    ExactPoolRevisionAuthority,
    GrantAliasCommand,
    GrantMutationCommand,
    GrantRecord,
    IdentityRecord,
    IdentityUsageAttribution,
    IssueVirtualKeyCommand,
    ManagementAction,
    ManagementReceipt,
    MonthlyBudgetRecord,
    MonthlyBudgetScope,
    MonthlyBudgetScopeKind,
    NaturalMutationAction,
    NaturalMutationOutcome,
    OneTimeVirtualKeyResult,
    OpaqueSecretReference,
    OpaqueSecretScheme,
    OrganizationRecord,
    ProviderConnectionMutationCommand,
    ProviderConnectionRevision,
    SetMonthlyBudgetCommand,
    UpsertProviderConnectionCommand,
    UsageAttribution,
    UsageTerminalCount,
    VirtualKeyRecord,
)
from exp.runtime.gateway.snapshot_integrity import refuse_self_inconsistent_snapshot
from exp.runtime.gateway.sqlite.migrations import connect_database
from exp.runtime.gateway.sqlite.platform_records import (
    alias_record as _alias_record,
)
from exp.runtime.gateway.sqlite.platform_records import (
    key_record as _key_record,
)
from exp.runtime.gateway.sqlite.platform_records import (
    optional_datetime as _optional_datetime,
)
from exp.runtime.gateway.sqlite.platform_records import (
    optional_int as _optional_int,
)
from exp.runtime.gateway.sqlite.platform_records import (
    provider_binding as _provider_binding,
)
from exp.runtime.gateway.sqlite.platform_records import (
    require_reservation_replay as _require_reservation_replay,
)
from exp.runtime.gateway.sqlite.platform_records import (
    require_settlement_replay as _require_settlement_replay,
)
from exp.runtime.gateway.sqlite.platform_records import (
    required_datetime as _datetime,
)
from exp.runtime.gateway.sqlite.platform_records import (
    reservation_record as _reservation_record,
)
from exp.runtime.gateway.sqlite.provider_commands import sqlite_connection_config
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore


class SQLiteGatewayPlatform:
    """Compose existing SQLite authorities behind the neutral platform contract."""

    def __init__(
        self,
        database_path: Path,
        *,
        budgets: SQLiteBudgetStore | None = None,
        attempts: SQLiteAttemptLedger | None = None,
        pool_revisions: ExactPoolRevisionAuthority | None = None,
        pepper_path: Path | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Compose existing authorities, constructing defaults when omitted."""
        if budgets is None:
            budgets = SQLiteBudgetStore(database_path, busy_timeout_ms=busy_timeout_ms)
        if attempts is None:
            attempts = SQLiteAttemptLedger(database_path, busy_timeout_ms=busy_timeout_ms)
        self.database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self.control = SQLiteGatewayStore(
            database_path,
            pepper_path=pepper_path,
            busy_timeout_ms=busy_timeout_ms,
        )
        if budgets.database_path != database_path or attempts.database_path != database_path:
            raise ValueError("SQLite platform components must share one database path")
        self.budgets = budgets
        self.attempts = attempts
        self._pool_revisions = pool_revisions

    def execute(self, command: CreateIdentityCommand) -> ManagementReceipt:
        """Execute or atomically replay one identity creation command."""
        self.control.create_identity(
            organization_id=command.organization_id,
            identity_id=command.identity_id,
            display_name=command.display_name,
            description=command.description,
            operation_id=command.operation_id,
        )
        return self._receipt(
            organization_id=command.organization_id,
            operation_id=command.operation_id,
            expected_action=ManagementAction.CREATE_IDENTITY,
        )

    def issue_key(self, command: IssueVirtualKeyCommand) -> OneTimeVirtualKeyResult:
        """Issue a key while keeping its raw material outside durable records."""
        issued = self.control.issue_virtual_key(
            organization_id=command.organization_id,
            identity_id=command.identity_id,
            key_id=command.key_id,
            expires_at=command.expires_at,
            operation_id=command.operation_id,
        )
        receipt = self._receipt(
            organization_id=command.organization_id,
            operation_id=command.operation_id,
            expected_action=ManagementAction.ISSUE_VIRTUAL_KEY,
        )
        key = self._key(organization_id=command.organization_id, key_id=command.key_id)
        return OneTimeVirtualKeyResult(receipt=receipt, key=key, raw_key=issued.raw_key)

    def mutate_grant(self, command: GrantMutationCommand) -> NaturalMutationOutcome:
        """Add or remove a grant through existing naturally idempotent methods."""
        if isinstance(command, GrantAliasCommand):
            changed = self.control.grant_alias(
                organization_id=command.organization_id,
                identity_id=command.identity_id,
                alias_id=command.alias_id,
            )
            action = NaturalMutationAction.GRANT_ALIAS
        else:
            changed = self.control.revoke_alias_grant(
                organization_id=command.organization_id,
                identity_id=command.identity_id,
                alias_id=command.alias_id,
            )
            action = NaturalMutationAction.REVOKE_ALIAS_GRANT
        return NaturalMutationOutcome(
            organization_id=command.organization_id,
            action=action,
            resource_id=stable_id(
                "gateway-identity-alias-grant",
                {
                    "organization_id": command.organization_id,
                    "identity_id": command.identity_id,
                    "alias_id": command.alias_id,
                },
            ),
            changed=changed,
        )

    def mutate_provider_connection(
        self,
        command: ProviderConnectionMutationCommand,
    ) -> NaturalMutationOutcome:
        """Upsert or disable a provider connection, failing closed on non-servable providers."""
        if isinstance(command, UpsertProviderConnectionCommand):
            require_gateway_servable_provider(
                connection_id=command.connection_id, provider=command.provider
            )
            changed, authority = self.control.upsert_provider_connection(
                organization_id=command.organization_id,
                connection_id=command.connection_id,
                revision_id=command.revision_id,
                config=sqlite_connection_config(command),
                replace=command.replace,
            )
            if not changed and authority.revision_id != command.revision_id:
                raise ValueError("provider connection replay names a different immutable revision")
            action = NaturalMutationAction.UPSERT_PROVIDER_CONNECTION
        else:
            changed = self.control.disable_provider_connection(
                organization_id=command.organization_id,
                connection_id=command.connection_id,
            )
            action = NaturalMutationAction.DISABLE_PROVIDER_CONNECTION
        return NaturalMutationOutcome(
            organization_id=command.organization_id,
            action=action,
            resource_id=command.connection_id,
            changed=changed,
        )

    def mutate_alias(self, command: AliasMutationCommand) -> NaturalMutationOutcome:
        """Activate or disable an alias through existing idempotent transitions."""
        if isinstance(command, DisableAliasCommand):
            return NaturalMutationOutcome(
                organization_id=command.organization_id,
                action=NaturalMutationAction.DISABLE_ALIAS,
                resource_id=command.alias_id,
                changed=self.control.disable_alias(
                    organization_id=command.organization_id,
                    alias_id=command.alias_id,
                ),
            )
        existing = self._alias_revision(
            organization_id=command.organization_id,
            revision_id=command.revision_id,
        )
        if existing is not None:
            self._require_alias_replay(existing, command=command)
            alias_enabled = bool(existing["active"])
            if alias_enabled and str(existing["active_revision_id"]) != command.revision_id:
                raise ValueError("alias activation replay names a historical inactive revision")
            changed = not alias_enabled
            if changed:
                if isinstance(command.target, ProjectTarget):
                    raise ValueError(
                        "disabled project alias revisions require a new activation revision"
                    )
                changed = self._reactivate_alias_revision(
                    organization_id=command.organization_id,
                    alias_id=command.alias_id,
                    revision_id=command.revision_id,
                )
        else:
            self._ensure_catalog_snapshot(command=command)
            try:
                self.control.activate_alias_revision(
                    organization_id=command.organization_id,
                    alias_id=command.alias_id,
                    alias_name=command.alias_name,
                    revision_id=command.revision_id,
                    target=command.target,
                    snapshot_ref=command.snapshot_ref,
                    catalog_sha256=command.catalog_sha256,
                    provider_connections=tuple(
                        _provider_binding(item) for item in command.provider_connections
                    ),
                    refusal_failover=command.refusal_failover,
                )
                changed = True
            except sqlite3.IntegrityError:
                concurrent = self._alias_revision(
                    organization_id=command.organization_id,
                    revision_id=command.revision_id,
                )
                if concurrent is None:
                    raise
                self._require_alias_replay(concurrent, command=command)
                changed = False
        return NaturalMutationOutcome(
            organization_id=command.organization_id,
            action=NaturalMutationAction.ACTIVATE_ALIAS_REVISION,
            resource_id=command.alias_id,
            changed=changed,
        )

    def set_monthly_budget(
        self,
        command: SetMonthlyBudgetCommand,
    ) -> NaturalMutationOutcome:
        """Set a monthly budget through the existing naturally idempotent store."""
        changed, budget = self.budgets.set_limit(
            organization_id=command.organization_id,
            period=command.period,
            scope=BudgetScope(
                kind=BudgetScopeKind(command.scope.kind.value),
                identity_id=command.scope.identity_id,
                alias_id=command.scope.alias_id,
                pool_id=command.scope.pool_id,
                deployment_id=command.scope.deployment_id,
            ),
            limit_micro_usd=command.limit_micro_usd,
            replace=command.replace,
        )
        return NaturalMutationOutcome(
            organization_id=command.organization_id,
            action=NaturalMutationAction.SET_MONTHLY_BUDGET,
            resource_id=budget.budget_id,
            changed=changed,
        )

    def organization(self, *, organization_id: str) -> OrganizationRecord | None:
        """Read one organization through an explicit tenant predicate."""
        rows = self._rows(
            """
            SELECT organization_id, slug, display_name, active, created_at, updated_at
            FROM organizations WHERE organization_id = ?
            """,
            (organization_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return OrganizationRecord(
            organization_id=str(row["organization_id"]),
            slug=str(row["slug"]),
            display_name=str(row["display_name"]),
            active=bool(row["active"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
        )

    def identities(self, *, organization_id: str) -> tuple[IdentityRecord, ...]:
        """List identities selected through an explicit tenant predicate."""
        rows = self._rows(
            """
            SELECT identity_id, display_name, description, active, created_at, updated_at
            FROM identities WHERE organization_id = ? ORDER BY identity_id
            """,
            (organization_id,),
        )
        return tuple(
            IdentityRecord(
                organization_id=organization_id,
                identity_id=str(row["identity_id"]),
                display_name=str(row["display_name"]),
                description=None if row["description"] is None else str(row["description"]),
                active=bool(row["active"]),
                created_at=_datetime(row["created_at"]),
                updated_at=_datetime(row["updated_at"]),
            )
            for row in rows
        )

    def keys(self, *, organization_id: str) -> tuple[VirtualKeyRecord, ...]:
        """List key metadata for one tenant without fingerprints or raw values."""
        rows = self._rows(
            """
            SELECT key_id, identity_id, prefix, expires_at, revoked_at,
                   created_at, last_used_at
            FROM virtual_keys WHERE organization_id = ? ORDER BY key_id
            """,
            (organization_id,),
        )
        now = datetime.now().astimezone()
        return tuple(_key_record(row, organization_id=organization_id, now=now) for row in rows)

    def grants(self, *, organization_id: str) -> tuple[GrantRecord, ...]:
        """List grants joined only within one explicit tenant."""
        rows = self._rows(
            """
            SELECT g.identity_id, g.alias_id, a.alias_name, g.created_at
            FROM identity_alias_grants AS g
            JOIN gateway_aliases AS a
              ON a.organization_id = g.organization_id AND a.alias_id = g.alias_id
            WHERE g.organization_id = ?
            ORDER BY g.identity_id, g.alias_id
            """,
            (organization_id,),
        )
        return tuple(
            GrantRecord(
                organization_id=organization_id,
                identity_id=str(row["identity_id"]),
                alias_id=str(row["alias_id"]),
                alias_name=str(row["alias_name"]),
                created_at=_datetime(row["created_at"]),
            )
            for row in rows
        )

    def provider_connection_revisions(
        self,
        *,
        organization_id: str,
    ) -> tuple[ProviderConnectionRevision, ...]:
        """List every immutable provider revision for one tenant."""
        rows = self._rows(
            """
            SELECT r.*, c.active, c.active_revision_id
            FROM provider_connection_revisions AS r
            JOIN provider_connections AS c
              ON c.organization_id = r.organization_id
             AND c.connection_id = r.connection_id
            WHERE r.organization_id = ?
            ORDER BY r.connection_id, r.revision_number
            """,
            (organization_id,),
        )
        return tuple(
            ProviderConnectionRevision(
                organization_id=organization_id,
                connection_id=str(row["connection_id"]),
                revision_id=str(row["revision_id"]),
                revision_number=int(row["revision_number"]),
                provider=str(row["provider"]),
                base_url=None if row["base_url"] is None else str(row["base_url"]),
                api_version=None if row["api_version"] is None else str(row["api_version"]),
                azure_api_surface=row["azure_api_surface"],
                region=None if row["region"] is None else str(row["region"]),
                secret_reference=(
                    None
                    if row["api_key_env"] is None
                    else OpaqueSecretReference(
                        scheme=OpaqueSecretScheme.ENVIRONMENT,
                        reference=str(row["api_key_env"]),
                    )
                ),
                access_key_id_reference=(
                    None
                    if row["aws_access_key_id_env"] is None
                    else OpaqueSecretReference(
                        scheme=OpaqueSecretScheme.ENVIRONMENT,
                        reference=str(row["aws_access_key_id_env"]),
                    )
                ),
                bedrock_auth_mode=(
                    None
                    if row["bedrock_auth_mode"] is None
                    else cast(
                        'Literal["access_key_pair", "api_key"]', str(row["bedrock_auth_mode"])
                    )
                ),
                connection_sha256=str(row["connection_sha256"]),
                active=bool(row["active"]) and row["active_revision_id"] == row["revision_id"],
                created_at=_datetime(row["created_at"]),
            )
            for row in rows
        )

    def alias_revisions(self, *, organization_id: str) -> tuple[AliasRevisionRecord, ...]:
        """List every immutable alias revision for one tenant."""
        rows = self._alias_rows(organization_id=organization_id)
        return tuple(_alias_record(row, organization_id=organization_id) for row in rows)

    def exact_pool_revisions(
        self,
        *,
        organization_id: str,
    ) -> tuple[ExactPoolRevision, ...]:
        """Read complete pool revisions from injected or local immutable catalogs."""
        if self._pool_revisions is not None:
            records = self._pool_revisions.exact_pool_revisions(
                organization_id=organization_id,
            )
            if any(item.organization_id != organization_id for item in records):
                raise RuntimeError("exact pool revision authority crossed the tenant boundary")
            return records
        return self._local_pool_revisions(organization_id=organization_id)

    def _local_pool_revisions(
        self,
        *,
        organization_id: str,
    ) -> tuple[ExactPoolRevision, ...]:
        """Reconstruct complete pool records from catalog snapshots pinned by SQLite."""
        snapshots: dict[tuple[str, str], datetime] = {}
        for row in self._alias_rows(organization_id=organization_id):
            key = (str(row["snapshot_ref"]), str(row["catalog_sha256"]))
            created_at = _datetime(row["created_at"])
            snapshots[key] = min(created_at, snapshots.get(key, created_at))
        records: dict[str, ExactPoolRevision] = {}
        state_dir = self.database_path.parent.resolve()
        for (snapshot_ref, catalog_sha256), created_at in sorted(snapshots.items()):
            snapshot_path = (state_dir / snapshot_ref).resolve()
            if not snapshot_path.is_relative_to(state_dir):
                raise RuntimeError("catalog snapshot reference escapes gateway state")
            try:
                catalog = read_pinned_normalized_snapshot(
                    snapshot_path.read_bytes(), catalog_sha256
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError("catalog snapshot is unreadable or invalid") from exc
            for pool in catalog.pools:
                revision_id = stable_id(
                    "gateway-exact-pool-revision",
                    {
                        "organization_id": organization_id,
                        "snapshot_ref": snapshot_ref,
                        "catalog_sha256": catalog_sha256,
                        "pool": pool.model_dump(mode="json", exclude_none=False),
                    },
                )
                records[revision_id] = ExactPoolRevision(
                    organization_id=organization_id,
                    revision_id=revision_id,
                    pool_id=pool.pool_id,
                    exact_model_id=pool.exact_model_id,
                    deployment_ids=pool.deployment_ids,
                    equivalence=pool.equivalence,
                    snapshot_ref=snapshot_ref,
                    catalog_sha256=catalog_sha256,
                    created_at=created_at,
                )
        return tuple(records[key] for key in sorted(records))

    def monthly_budgets(
        self,
        *,
        organization_id: str,
        period: str,
    ) -> tuple[MonthlyBudgetRecord, ...]:
        """List typed budget balances for one tenant and UTC month."""
        return tuple(
            MonthlyBudgetRecord(
                budget_id=item.budget.budget_id,
                organization_id=item.budget.organization_id,
                period=item.budget.period,
                scope=MonthlyBudgetScope(
                    kind=MonthlyBudgetScopeKind(item.budget.scope.kind.value),
                    identity_id=item.budget.scope.identity_id,
                    alias_id=item.budget.scope.alias_id,
                    pool_id=item.budget.scope.pool_id,
                    deployment_id=item.budget.scope.deployment_id,
                ),
                limit_micro_usd=item.budget.limit_micro_usd,
                reserved_micro_usd=item.reserved_micro_usd,
                settled_micro_usd=item.settled_micro_usd,
                remaining_micro_usd=item.remaining_micro_usd,
                unknown_cost_attempts=item.unknown_cost_attempts,
                exhausted=item.exhausted,
                created_at=item.budget.created_at,
                updated_at=item.budget.updated_at,
            )
            for item in self.budgets.remaining(
                organization_id=organization_id,
                period=period,
            )
        )

    def reserve_attempt(
        self,
        request: AttemptReservationRequest,
    ) -> AttemptReservationRecord:
        """Reserve once or replay the exact natural request-ordinal identity."""
        existing = self._attempt_for_reservation(request)
        if existing is not None:
            return _require_reservation_replay(existing, request=request)
        try:
            attempt_id = self.attempts.start_attempt(
                snapshot=request.snapshot,
                deployment=request.deployment,
                attempt_ordinal=request.attempt_ordinal,
                route_depth=request.route_depth,
                maximum_cost_micro_usd=request.maximum_cost_micro_usd,
            )
        except sqlite3.IntegrityError:
            concurrent = self._attempt_for_reservation(request)
            if concurrent is None:
                raise
            return _require_reservation_replay(concurrent, request=request)
        return self._reservation(
            organization_id=request.organization_id,
            attempt_id=attempt_id,
        )

    def settle_attempt(
        self,
        request: AttemptSettlementRequest,
    ) -> AttemptSettlementRecord:
        """Settle only an attempt proven to belong to the requested tenant."""
        self._reservation(
            organization_id=request.organization_id,
            attempt_id=request.attempt_id,
        )
        self.attempts.finish_attempt(
            attempt_id=request.attempt_id,
            terminal_event=request.terminal_event,
            failure=request.failure,
            finalize_request=request.finalize_request,
            first_token_at=request.first_token_at,
        )
        row = self._attempt_row(
            organization_id=request.organization_id,
            attempt_id=request.attempt_id,
        )
        if request.finalize_request and str(row["request_terminal_state"]) != str(row["state"]):
            raise ValueError(
                "attempt settlement replay cannot finalize its non-terminal parent request"
            )
        usage = (
            None
            if row["input_tokens"] is None or row["output_tokens"] is None
            else GatewayUsage(
                input_tokens=int(row["input_tokens"]),
                cached_input_tokens=(
                    None if row["cached_input_tokens"] is None else int(row["cached_input_tokens"])
                ),
                cache_creation_input_tokens=(
                    None
                    if row["cache_creation_input_tokens"] is None
                    else int(row["cache_creation_input_tokens"])
                ),
                output_tokens=int(row["output_tokens"]),
                reasoning_tokens=(
                    None if row["reasoning_tokens"] is None else int(row["reasoning_tokens"])
                ),
            )
        )
        settlement = AttemptSettlementRecord(
            reservation=_reservation_record(row, organization_id=request.organization_id),
            state=AttemptTerminalState(str(row["state"])),
            terminal_at=_datetime(row["terminal_at"]),
            failure_class=(
                None
                if row["failure_class"] is None
                else GatewayFailureClass(str(row["failure_class"]))
            ),
            usage=usage,
            usage_source=AttemptUsageSource(str(row["usage_source"] or "unknown")),
            estimated_cost_micro_usd=_optional_int(row["estimated_cost_micro_usd"]),
            settled_micro_usd=_optional_int(row["budget_settled_micro_usd"]),
            first_token_at=_optional_datetime(row["first_token_at"]),
        )
        _require_settlement_replay(settlement, request=request)
        return settlement

    def usage_attribution(
        self,
        *,
        organization_id: str,
        identity_id: str | None = None,
    ) -> UsageAttribution:
        """Forward one tenant-scoped consistent usage snapshot."""
        snapshot = self.attempts.usage_snapshot(
            organization_id=organization_id,
            identity_id=identity_id,
        )
        return UsageAttribution(
            organization_id=organization_id,
            identities=tuple(
                IdentityUsageAttribution(
                    organization_id=item.organization_id,
                    identity_id=item.identity_id,
                    requests=item.requests,
                    attempts=item.attempts,
                    input_tokens=item.input_tokens,
                    cached_input_tokens=item.cached_input_tokens,
                    output_tokens=item.output_tokens,
                    reasoning_tokens=item.reasoning_tokens,
                    known_estimated_cost_micro_usd=item.known_estimated_cost_micro_usd,
                    unknown_cost_attempts=item.unknown_cost_attempts,
                    total_latency_ms=item.total_latency_ms,
                    average_latency_ms=item.average_latency_ms,
                    terminal_counts=tuple(
                        UsageTerminalCount(
                            state=AttemptTerminalState(count.state),
                            attempts=count.attempts,
                        )
                        for count in item.terminal_counts
                    ),
                )
                for item in snapshot.identities
            ),
            by_billing_source=tuple(
                BillingSourceUsageAttribution(
                    billing_source=item.billing_source,
                    attempts=item.attempts,
                    input_tokens=item.input_tokens,
                    cached_input_tokens=item.cached_input_tokens,
                    output_tokens=item.output_tokens,
                    reasoning_tokens=item.reasoning_tokens,
                    known_estimated_cost_micro_usd=item.known_estimated_cost_micro_usd,
                    unknown_cost_attempts=item.unknown_cost_attempts,
                    terminal_counts=tuple(
                        UsageTerminalCount(
                            state=AttemptTerminalState(count.state),
                            attempts=count.attempts,
                        )
                        for count in item.terminal_counts
                    ),
                )
                for item in snapshot.by_billing_source
            ),
        )

    def _alias_revision(
        self,
        *,
        organization_id: str,
        revision_id: str,
    ) -> sqlite3.Row | None:
        """Read one alias revision under an explicit tenant predicate."""
        rows = self._rows(
            """
            SELECT r.*, a.alias_name, a.active, a.active_revision_id
            FROM alias_revisions AS r
            JOIN gateway_aliases AS a
              ON a.organization_id = r.organization_id AND a.alias_id = r.alias_id
            WHERE r.organization_id = ? AND r.revision_id = ?
            """,
            (organization_id, revision_id),
        )
        return None if not rows else rows[0]

    def _require_alias_replay(
        self,
        row: sqlite3.Row,
        *,
        command: ActivateAliasRevisionCommand,
    ) -> AliasRevisionRecord:
        """Reject reuse of an alias revision ID with different immutable input."""
        record = _alias_record(row, organization_id=command.organization_id)
        expected_bindings = tuple(
            sorted(
                (
                    item.connection_id,
                    item.connection_revision_id,
                    item.connection_sha256,
                )
                for item in command.provider_connections
            )
        )
        actual_bindings = tuple(
            (
                item.connection_id,
                item.revision_id,
                item.connection_sha256,
            )
            for item in self.control.alias_provider_connections(
                organization_id=command.organization_id,
                alias_id=command.alias_id,
                alias_revision_id=command.revision_id,
            )
        )
        if (
            record.alias_id != command.alias_id
            or record.alias_name != command.alias_name
            or record.target != command.target
            or record.snapshot_ref != command.snapshot_ref
            or record.catalog_sha256 != command.catalog_sha256
            or record.refusal_failover != command.refusal_failover
            or actual_bindings != expected_bindings
        ):
            raise ValueError("alias revision ID was reused with different input")
        return record

    def _reactivate_alias_revision(
        self,
        *,
        organization_id: str,
        alias_id: str,
        revision_id: str,
    ) -> bool:
        """Conditionally reactivate a disabled revision with current provider bindings."""
        connection = connect_database(
            self.database_path,
            busy_timeout_ms=self._busy_timeout_ms,
        )
        try:
            connection.execute("BEGIN IMMEDIATE")
            alias = connection.execute(
                """
                SELECT active, active_revision_id FROM gateway_aliases
                WHERE organization_id = ? AND alias_id = ?
                """,
                (organization_id, alias_id),
            ).fetchone()
            if alias is None:
                raise ValueError("alias revision cannot be reactivated")
            current_revision = str(alias["active_revision_id"])
            if bool(alias["active"]):
                if current_revision == revision_id:
                    connection.rollback()
                    return False
                raise ValueError("alias activation advanced to another revision")
            if current_revision != revision_id:
                raise ValueError("disabled alias no longer points at the requested revision")
            bindings = connection.execute(
                """
                SELECT b.connection_id, b.connection_revision_id,
                       b.connection_sha256, c.active, c.active_revision_id,
                       r.connection_sha256 AS current_sha256
                FROM alias_revision_provider_connections AS b
                LEFT JOIN provider_connections AS c
                  ON c.organization_id = b.organization_id
                 AND c.connection_id = b.connection_id
                LEFT JOIN provider_connection_revisions AS r
                  ON r.organization_id = c.organization_id
                 AND r.connection_id = c.connection_id
                 AND r.revision_id = c.active_revision_id
                WHERE b.organization_id = ? AND b.alias_id = ?
                  AND b.alias_revision_id = ?
                """,
                (organization_id, alias_id, revision_id),
            ).fetchall()
            for binding in bindings:
                if (
                    not bool(binding["active"])
                    or str(binding["active_revision_id"]) != str(binding["connection_revision_id"])
                    or str(binding["current_sha256"]) != str(binding["connection_sha256"])
                ):
                    raise ValueError(
                        "alias revision provider bindings are no longer active and current"
                    )
            result = connection.execute(
                """
                UPDATE gateway_aliases
                SET active = 1, active_revision_id = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
                WHERE organization_id = ? AND alias_id = ? AND active = 0
                  AND active_revision_id = ?
                """,
                (revision_id, organization_id, alias_id, revision_id),
            )
            if result.rowcount != 1:
                raise ValueError("alias revision cannot be reactivated")
            connection.commit()
            return True
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_catalog_snapshot(self, *, command: ActivateAliasRevisionCommand) -> None:
        """Register a missing exact snapshot while preserving replay safety."""
        rows = self._rows(
            """
            SELECT snapshot_ref, catalog_sha256 FROM catalog_snapshot_refs
            WHERE organization_id = ? AND (
                snapshot_ref = ? OR catalog_sha256 = ?
            )
            """,
            (command.organization_id, command.snapshot_ref, command.catalog_sha256),
        )
        if rows:
            if len(rows) != 1 or (
                str(rows[0]["snapshot_ref"]),
                str(rows[0]["catalog_sha256"]),
            ) != (command.snapshot_ref, command.catalog_sha256):
                raise ValueError("catalog snapshot reference conflicts with existing authority")
            return
        # Never pin a self-inconsistent snapshot (the persistent hydration bug).
        refuse_self_inconsistent_snapshot(
            self.database_path.parent, command.snapshot_ref, command.catalog_sha256
        )
        try:
            self.control.register_catalog_snapshot(
                organization_id=command.organization_id,
                snapshot_ref=command.snapshot_ref,
                catalog_sha256=command.catalog_sha256,
            )
        except sqlite3.IntegrityError:
            concurrent = self._rows(
                """
                SELECT snapshot_ref, catalog_sha256 FROM catalog_snapshot_refs
                WHERE organization_id = ? AND snapshot_ref = ? AND catalog_sha256 = ?
                """,
                (command.organization_id, command.snapshot_ref, command.catalog_sha256),
            )
            if len(concurrent) != 1:
                raise

    def _receipt(
        self,
        *,
        organization_id: str,
        operation_id: str,
        expected_action: ManagementAction,
    ) -> ManagementReceipt:
        """Read and validate one tenant-owned durable operation receipt."""
        rows = self._rows(
            """
            SELECT operation_kind, request_sha256, resource_kind, resource_id, created_at
            FROM operation_receipts
            WHERE organization_id = ? AND operation_id = ?
            """,
            (organization_id, operation_id),
        )
        if len(rows) != 1 or str(rows[0]["operation_kind"]) != expected_action.value:
            raise RuntimeError("management mutation did not produce its expected receipt")
        row = rows[0]
        return ManagementReceipt(
            organization_id=organization_id,
            operation_id=operation_id,
            action=expected_action,
            command_sha256=str(row["request_sha256"]),
            resource_kind=str(row["resource_kind"]),
            resource_id=str(row["resource_id"]),
            created_at=_datetime(row["created_at"]),
        )

    def _key(self, *, organization_id: str, key_id: str) -> VirtualKeyRecord:
        """Read one non-secret tenant-owned key after issuance."""
        rows = self._rows(
            """
            SELECT key_id, identity_id, prefix, expires_at, revoked_at,
                   created_at, last_used_at
            FROM virtual_keys WHERE organization_id = ? AND key_id = ?
            """,
            (organization_id, key_id),
        )
        if len(rows) != 1:
            raise RuntimeError("issued virtual key is absent from durable authority")
        return _key_record(
            rows[0], organization_id=organization_id, now=datetime.now().astimezone()
        )

    def _alias_rows(self, *, organization_id: str) -> tuple[sqlite3.Row, ...]:
        """Read immutable alias rows under one tenant predicate."""
        return self._rows(
            """
            SELECT r.*, a.alias_name, a.active, a.active_revision_id
            FROM alias_revisions AS r
            JOIN gateway_aliases AS a
              ON a.organization_id = r.organization_id AND a.alias_id = r.alias_id
            WHERE r.organization_id = ?
            ORDER BY r.alias_id, r.revision_number
            """,
            (organization_id,),
        )

    def _attempt_row(self, *, organization_id: str, attempt_id: str) -> sqlite3.Row:
        """Read one attempt joined to tenant-owned request attribution."""
        rows = self._rows(
            """
            SELECT a.*, r.identity_id, r.alias_id, r.alias_revision_id,
                   r.terminal_state AS request_terminal_state
            FROM gateway_attempts AS a
            JOIN gateway_requests AS r
              ON r.organization_id = a.organization_id AND r.request_id = a.request_id
            WHERE a.organization_id = ? AND a.attempt_id = ?
            """,
            (organization_id, attempt_id),
        )
        if len(rows) != 1:
            raise ValueError("attempt does not belong to the requested organization")
        return rows[0]

    def _attempt_for_reservation(
        self,
        request: AttemptReservationRequest,
    ) -> sqlite3.Row | None:
        """Read one attempt by its tenant, request, and physical ordinal."""
        rows = self._rows(
            """
            SELECT a.*, r.identity_id, r.key_id, r.alias_id, r.alias_revision_id,
                   r.api_surface, r.canonical_request_sha256,
                   r.caller_operation_sha256,
                   r.terminal_state AS request_terminal_state
            FROM gateway_attempts AS a
            JOIN gateway_requests AS r
              ON r.organization_id = a.organization_id AND r.request_id = a.request_id
            WHERE a.organization_id = ? AND a.request_id = ? AND a.attempt_ordinal = ?
            """,
            (
                request.organization_id,
                request.snapshot.authorization.request_id,
                str(request.attempt_ordinal),
            ),
        )
        if len(rows) > 1:
            raise RuntimeError("attempt reservation natural key is not unique")
        return None if not rows else rows[0]

    def _reservation(
        self,
        *,
        organization_id: str,
        attempt_id: str,
    ) -> AttemptReservationRecord:
        """Read one precise tenant-owned reservation record."""
        return _reservation_record(
            self._attempt_row(organization_id=organization_id, attempt_id=attempt_id),
            organization_id=organization_id,
        )

    def _rows(
        self,
        query: str,
        parameters: tuple[str, ...] = (),
    ) -> tuple[sqlite3.Row, ...]:
        """Execute one bounded read against the shared database."""
        connection = connect_database(
            self.database_path,
            busy_timeout_ms=self._busy_timeout_ms,
        )
        try:
            return tuple(connection.execute(query, parameters).fetchall())
        finally:
            connection.close()


__all__ = ["SQLiteGatewayPlatform"]

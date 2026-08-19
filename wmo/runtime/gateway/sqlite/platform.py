"""SQLite composition adapter for the storage-neutral gateway platform."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from wmo.common.core.artifacts import stable_id
from wmo.common.models import BillingSource, ConnectionConfig
from wmo.common.models.gateway_catalog import NormalizedGatewayCatalog
from wmo.runtime.gateway.budgets import (
    BudgetScope,
    BudgetScopeKind,
    SQLiteBudgetStore,
)
from wmo.runtime.gateway.contracts import (
    DirectTarget,
    GatewayFailureClass,
    GatewayUsage,
    ProjectTarget,
)
from wmo.runtime.gateway.ledger import SQLiteAttemptLedger
from wmo.runtime.gateway.platform import (
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
    ProviderRevisionBinding,
    SetMonthlyBudgetCommand,
    UpsertProviderConnectionCommand,
    UsageAttribution,
    UsageTerminalCount,
    VirtualKeyRecord,
)
from wmo.runtime.gateway.sqlite.migrations import connect_database
from wmo.runtime.gateway.sqlite.provider_authority import ProviderConnectionBinding
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore


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
        """Upsert or disable a provider connection without claiming a receipt."""
        if isinstance(command, UpsertProviderConnectionCommand):
            secret_reference = command.secret_reference
            if (
                secret_reference is not None
                and secret_reference.scheme is not OpaqueSecretScheme.ENVIRONMENT
            ):
                raise ValueError(
                    "SQLite provider connections currently require an environment secret reference"
                )
            changed, _authority = self.control.upsert_provider_connection(
                organization_id=command.organization_id,
                connection_id=command.connection_id,
                revision_id=command.revision_id,
                config=ConnectionConfig(
                    provider=command.provider,
                    base_url=command.base_url,
                    api_key_env=(None if secret_reference is None else secret_reference.reference),
                    api_version=command.api_version,
                    region=command.region,
                ),
                replace=command.replace,
            )
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
            changed = False
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
                region=None if row["region"] is None else str(row["region"]),
                secret_reference=(
                    None
                    if row["api_key_env"] is None
                    else OpaqueSecretReference(
                        scheme=OpaqueSecretScheme.ENVIRONMENT,
                        reference=str(row["api_key_env"]),
                    )
                ),
                connection_sha256=str(row["connection_sha256"]),
                active=bool(row["active"])
                and str(row["active_revision_id"]) == str(row["revision_id"]),
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
                catalog = NormalizedGatewayCatalog.model_validate_json(snapshot_path.read_bytes())
            except (OSError, ValueError) as exc:
                raise RuntimeError("catalog snapshot is unreadable or invalid") from exc
            if catalog.identity_sha256() != catalog_sha256:
                raise RuntimeError("catalog snapshot digest differs from SQLite authority")
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
        """Forward to the existing atomic budget and attempt transaction."""
        attempt_id = self.attempts.start_attempt(
            snapshot=request.snapshot,
            deployment=request.deployment,
            attempt_ordinal=request.attempt_ordinal,
            route_depth=request.route_depth,
            maximum_cost_micro_usd=request.maximum_cost_micro_usd,
        )
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
    ) -> None:
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

    def _ensure_catalog_snapshot(self, *, command: ActivateAliasRevisionCommand) -> None:
        """Register a missing exact snapshot while preserving replay safety."""
        rows = self._rows(
            """
            SELECT snapshot_ref, catalog_sha256 FROM catalog_snapshot_refs
            WHERE organization_id = ? AND (
                snapshot_ref = ? OR catalog_sha256 = ?
            )
            """,
            (
                command.organization_id,
                command.snapshot_ref,
                command.catalog_sha256,
            ),
        )
        if rows:
            if len(rows) != 1 or (
                str(rows[0]["snapshot_ref"]),
                str(rows[0]["catalog_sha256"]),
            ) != (command.snapshot_ref, command.catalog_sha256):
                raise ValueError("catalog snapshot reference conflicts with existing authority")
            return
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
                (
                    command.organization_id,
                    command.snapshot_ref,
                    command.catalog_sha256,
                ),
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


def _key_record(
    row: sqlite3.Row,
    *,
    organization_id: str,
    now: datetime,
) -> VirtualKeyRecord:
    """Decode one durable key row without touching fingerprint material."""
    expires_at = _optional_datetime(row["expires_at"])
    revoked_at = _optional_datetime(row["revoked_at"])
    return VirtualKeyRecord(
        organization_id=organization_id,
        identity_id=str(row["identity_id"]),
        key_id=str(row["key_id"]),
        prefix=str(row["prefix"]),
        active=revoked_at is None and (expires_at is None or expires_at > now),
        expires_at=expires_at,
        revoked_at=revoked_at,
        created_at=_datetime(row["created_at"]),
        last_used_at=_optional_datetime(row["last_used_at"]),
    )


def _provider_binding(binding: ProviderRevisionBinding) -> ProviderConnectionBinding:
    """Convert one neutral provider binding to the existing SQLite contract."""
    return ProviderConnectionBinding(
        connection_id=binding.connection_id,
        connection_revision_id=binding.connection_revision_id,
        connection_sha256=binding.connection_sha256,
    )


def _require_settlement_replay(
    settlement: AttemptSettlementRecord,
    *,
    request: AttemptSettlementRequest,
) -> None:
    """Reject a replay whose accounting evidence differs from durable settlement."""
    event_failure = None if request.terminal_event is None else request.terminal_event.failure
    failure = request.failure or event_failure
    usage = None if request.terminal_event is None else request.terminal_event.usage
    if failure is not None:
        state = (
            AttemptTerminalState.CANCELLED
            if failure.failure_class is GatewayFailureClass.CANCELLED
            else AttemptTerminalState.FAILED
        )
        failure_class = failure.failure_class
    else:
        if request.terminal_event is None:
            raise ValueError("attempt settlement needs a terminal event or failure")
        state = AttemptTerminalState(request.terminal_event.kind.value)
        failure_class = None
    usage_source = AttemptUsageSource.OBSERVED if usage is not None else AttemptUsageSource.UNKNOWN
    if (
        settlement.state is not state
        or settlement.failure_class is not failure_class
        or settlement.usage != usage
        or settlement.usage_source is not usage_source
    ):
        raise ValueError("attempt settlement replay differs from durable accounting evidence")


def _alias_record(row: sqlite3.Row, *, organization_id: str) -> AliasRevisionRecord:
    """Decode one immutable alias revision and discriminated target."""
    target = (
        DirectTarget(pool_id=str(row["pool_id"]))
        if str(row["target_kind"]) == "direct"
        else ProjectTarget(
            project_ref=str(row["project_ref"]),
            activation_ref=str(row["activation_ref"]),
            catalog_sha256=str(row["catalog_sha256"]),
        )
    )
    return AliasRevisionRecord(
        organization_id=organization_id,
        alias_id=str(row["alias_id"]),
        alias_name=str(row["alias_name"]),
        revision_id=str(row["revision_id"]),
        revision_number=int(row["revision_number"]),
        target=target,
        snapshot_ref=str(row["snapshot_ref"]),
        catalog_sha256=str(row["catalog_sha256"]),
        refusal_failover=bool(row["refusal_failover"]),
        active=bool(row["active"]) and str(row["active_revision_id"]) == str(row["revision_id"]),
        created_at=_datetime(row["created_at"]),
    )


def _reservation_record(
    row: sqlite3.Row,
    *,
    organization_id: str,
) -> AttemptReservationRecord:
    """Decode one reservation from the existing atomic attempt row."""
    return AttemptReservationRecord(
        organization_id=organization_id,
        attempt_id=str(row["attempt_id"]),
        request_id=str(row["request_id"]),
        identity_id=str(row["identity_id"]),
        alias_id=str(row["alias_id"]),
        alias_revision_id=str(row["alias_revision_id"]),
        catalog_sha256=str(row["catalog_sha256"]),
        pool_id=str(row["pool_id"]),
        exact_model_id=str(row["exact_model_id"]),
        deployment_id=str(row["deployment_id"]),
        provider=str(row["provider"]),
        billing_source=BillingSource(str(row["billing_source"])),
        input_rate=_optional_int(row["input_rate"]),
        cached_input_rate=_optional_int(row["cached_input_rate"]),
        output_rate=_optional_int(row["output_rate"]),
        reasoning_rate=_optional_int(row["reasoning_rate"]),
        attempt_ordinal=int(row["attempt_ordinal"]),
        route_depth=int(row["route_depth"]),
        period=str(row["budget_period_start"])[:7],
        reserved_micro_usd=_optional_int(row["budget_reserved_micro_usd"]),
        started_at=_datetime(row["started_at"]),
    )


def _datetime(value: object) -> datetime:
    """Parse one required SQLite timestamp."""
    if value is None:
        raise RuntimeError("required platform timestamp is missing")
    return datetime.fromisoformat(str(value))


def _optional_datetime(value: object) -> datetime | None:
    """Parse one optional SQLite timestamp."""
    return None if value is None else datetime.fromisoformat(str(value))


def _optional_int(value: object) -> int | None:
    """Decode one optional SQLite integer."""
    return None if value is None else int(str(value))


__all__ = ["SQLiteGatewayPlatform"]

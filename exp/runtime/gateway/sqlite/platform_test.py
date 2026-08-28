"""Conformance tests for the SQLite gateway platform adapter."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ConnectionConfig, ModelCapabilities
from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway import (
    ActivateAliasRevisionCommand,
    AttemptAccountingAuthority,
    AttemptReservationRequest,
    AttemptSettlementRequest,
    CreateIdentityCommand,
    DisableAliasCommand,
    DisableProviderConnectionCommand,
    ExactPoolRevision,
    ExactPoolRevisionAuthority,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayPlatform,
    GatewayRequest,
    GatewayUsage,
    GrantAliasCommand,
    GrantAuthority,
    GrantMutationAuthority,
    IssueVirtualKeyCommand,
    ManagementCommandAuthority,
    MonthlyBudgetAuthority,
    MonthlyBudgetMutationAuthority,
    MonthlyBudgetScope,
    MonthlyBudgetScopeKind,
    OpaqueSecretReference,
    OpaqueSecretScheme,
    OrganizationIdentityKeyAuthority,
    ProviderConnectionMutationAuthority,
    ProviderConnectionRevisionAuthority,
    ProviderRevisionBinding,
    RevokeAliasGrantCommand,
    RoutingMutationAuthority,
    RoutingRevisionAuthority,
    SetMonthlyBudgetCommand,
    UpsertProviderConnectionCommand,
    UsageAttributionAuthority,
)
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.contracts import DirectTarget, ExecutionSnapshot
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform
from exp.runtime.gateway.sqlite.store import (
    GatewayStoreError,
    OperationConflictError,
    OperationReplayUnavailableError,
)

_DIGEST = "a" * 64


class _PoolRevisions:
    """Injected complete exact-pool revision reader."""

    def exact_pool_revisions(self, *, organization_id: str) -> tuple[ExactPoolRevision, ...]:
        """Return one complete singleton pool for the requested tenant."""
        if organization_id != "org-one":
            return ()
        return (
            ExactPoolRevision(
                organization_id=organization_id,
                revision_id="pool-revision-one",
                pool_id="coding-pool",
                exact_model_id="exact-coding",
                deployment_ids=("deployment-one",),
                snapshot_ref="catalog-one",
                catalog_sha256=_DIGEST,
                created_at=datetime(2026, 8, 19, tzinfo=UTC),
            ),
        )


def _platform(tmp_path: Path) -> SQLiteGatewayPlatform:
    """Create a SQLite platform with two isolated organizations."""
    path = tmp_path / "gateway.db"
    platform = SQLiteGatewayPlatform(
        path,
        budgets=SQLiteBudgetStore(path),
        attempts=SQLiteAttemptLedger(path),
        pool_revisions=_PoolRevisions(),
    )
    platform.control.create_organization(
        organization_id="org-one",
        slug="one",
        display_name="One",
    )
    platform.control.create_organization(
        organization_id="org-two",
        slug="two",
        display_name="Two",
    )
    return platform


def _require_static_protocol_conformance(platform: SQLiteGatewayPlatform) -> None:
    """Make static checking prove every narrow SQLite adapter protocol."""
    management: ManagementCommandAuthority = platform
    grants: GrantMutationAuthority = platform
    providers: ProviderConnectionMutationAuthority = platform
    routing: RoutingMutationAuthority = platform
    budgets: MonthlyBudgetMutationAuthority = platform
    attempts: AttemptAccountingAuthority = platform
    usage: UsageAttributionAuthority = platform
    complete: GatewayPlatform = platform
    del management, grants, providers, routing, budgets, attempts, usage, complete


def test_management_replay_is_atomic_tenant_scoped_and_secret_free(tmp_path: Path) -> None:
    """Commands replay one receipt while key material remains one-time only."""
    platform = _platform(tmp_path)
    _require_static_protocol_conformance(platform)
    command = CreateIdentityCommand(
        operation_id="create-builders",
        organization_id="org-one",
        identity_id="builders",
        display_name="Builders",
    )

    first = platform.execute(command)
    replay = platform.execute(command)
    issued = platform.issue_key(
        IssueVirtualKeyCommand(
            operation_id="issue-builders",
            organization_id="org-one",
            identity_id="builders",
            key_id="builders-key",
            expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        )
    )

    assert isinstance(platform, GatewayPlatform)
    assert isinstance(platform, ManagementCommandAuthority)
    assert isinstance(platform, OrganizationIdentityKeyAuthority)
    assert isinstance(platform, GrantAuthority)
    assert isinstance(platform, GrantMutationAuthority)
    assert isinstance(platform, ProviderConnectionRevisionAuthority)
    assert isinstance(platform, ProviderConnectionMutationAuthority)
    assert isinstance(platform, RoutingRevisionAuthority)
    assert isinstance(platform, RoutingMutationAuthority)
    assert isinstance(platform, ExactPoolRevisionAuthority)
    assert isinstance(platform, MonthlyBudgetAuthority)
    assert isinstance(platform, MonthlyBudgetMutationAuthority)
    assert isinstance(platform, AttemptAccountingAuthority)
    assert isinstance(platform, UsageAttributionAuthority)
    assert replay == first
    assert first.resource_id == "builders"
    assert platform.identities(organization_id="org-one")[0].identity_id == "builders"
    assert platform.identities(organization_id="org-two") == ()
    assert platform.organization(organization_id="org-one") is not None
    assert platform.organization(organization_id="missing") is None
    assert platform.keys(organization_id="org-one")[0].key_id == "builders-key"
    assert issued.raw_key.startswith("exp_vk_")
    assert issued.receipt.resource_id == "builders-key"
    assert issued.raw_key not in issued.receipt.model_dump_json()
    assert issued.raw_key.encode() not in (tmp_path / "gateway.db").read_bytes()

    with pytest.raises(OperationConflictError, match="different input"):
        platform.execute(command.model_copy(update={"display_name": "Changed"}))
    with pytest.raises(OperationReplayUnavailableError, match="cannot be revealed again"):
        platform.issue_key(
            IssueVirtualKeyCommand(
                operation_id="issue-builders",
                organization_id="org-one",
                identity_id="builders",
                key_id="builders-key",
                expires_at=datetime(2027, 1, 1, tzinfo=UTC),
            )
        )


def test_default_adapter_constructs_existing_atomic_components(tmp_path: Path) -> None:
    """The default adapter is ergonomic while retaining optional injection."""
    platform = SQLiteGatewayPlatform(tmp_path / "gateway.db")

    assert platform.budgets.database_path == platform.database_path
    assert platform.attempts.database_path == platform.database_path
    assert platform.exact_pool_revisions(organization_id="missing") == ()


def test_default_adapter_reads_complete_local_pool_revisions(tmp_path: Path) -> None:
    """SQLite resolves complete pools from its pinned immutable local snapshots."""
    platform = SQLiteGatewayPlatform(tmp_path / "gateway.db")
    platform.control.create_organization(
        organization_id="org-one",
        slug="one",
        display_name="One",
    )
    normalized = NormalizedGatewayCatalog(
        deployments=(
            ExactModelDeployment(
                deployment_id="deployment-one",
                source_alias="deployment-one",
                exact_model_id="exact-coding",
                connection="openai",
                provider="openai",
                provider_model="gpt-test",
                connection_sha256="b" * 64,
                capabilities_sha256="c" * 64,
            ),
        ),
        pools=(
            ExactModelPool(
                pool_id="coding-pool",
                exact_model_id="exact-coding",
                deployment_ids=("deployment-one",),
            ),
        ),
    )
    digest = normalized.identity_sha256()
    snapshot_ref = f"catalog-snapshots/{digest}.json"
    snapshot_path = tmp_path / snapshot_ref
    snapshot_path.parent.mkdir()
    snapshot_path.write_bytes(canonical_json_bytes(normalized))
    platform.control.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref=snapshot_ref,
        catalog_sha256=digest,
    )
    platform.control.activate_alias_revision(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref=snapshot_ref,
        catalog_sha256=digest,
    )

    revisions = platform.exact_pool_revisions(organization_id="org-one")

    assert len(revisions) == 1
    assert revisions[0].exact_model_id == "exact-coding"
    assert revisions[0].deployment_ids == ("deployment-one",)
    assert revisions[0].snapshot_ref == snapshot_ref


def test_naturally_idempotent_mutations_are_tenant_scoped_and_unreceipted(
    tmp_path: Path,
) -> None:
    """Current non-receipted mutations replay naturally within one tenant."""
    platform = _platform(tmp_path)
    for organization_id, identity_id in (
        ("org-one", "builders"),
        ("org-two", "reviewers"),
    ):
        platform.control.create_identity(
            organization_id=organization_id,
            identity_id=identity_id,
            display_name=identity_id,
        )
    provider = UpsertProviderConnectionCommand(
        organization_id="org-one",
        connection_id="openai",
        revision_id="provider-revision-one",
        provider="openai",
        secret_reference=OpaqueSecretReference(
            scheme=OpaqueSecretScheme.ENVIRONMENT,
            reference="OPENAI_API_KEY",
        ),
    )
    with pytest.raises(GatewayStoreError, match="unsupported provider 'tinker'"):
        platform.mutate_provider_connection(
            provider.model_copy(update={"connection_id": "training", "provider": "tinker"})
        )
    assert platform.provider_connection_revisions(organization_id="org-one") == ()
    assert platform.mutate_provider_connection(provider).changed
    assert not platform.mutate_provider_connection(provider).changed
    with pytest.raises(ValueError, match="different immutable revision"):
        platform.mutate_provider_connection(
            provider.model_copy(update={"revision_id": "provider-revision-two"})
        )
    assert platform.provider_connection_revisions(organization_id="org-two") == ()
    assert not platform.mutate_provider_connection(
        DisableProviderConnectionCommand(
            organization_id="org-two",
            connection_id="openai",
        )
    ).changed
    assert platform.mutate_provider_connection(
        DisableProviderConnectionCommand(
            organization_id="org-one",
            connection_id="openai",
        )
    ).changed
    assert not platform.mutate_provider_connection(
        DisableProviderConnectionCommand(
            organization_id="org-one",
            connection_id="openai",
        )
    ).changed

    alias = ActivateAliasRevisionCommand(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    assert platform.mutate_alias(alias).changed
    assert not platform.mutate_alias(alias).changed
    assert platform.alias_revisions(organization_id="org-two") == ()
    assert not platform.mutate_alias(
        DisableAliasCommand(organization_id="org-two", alias_id="coding")
    ).changed

    grant = GrantAliasCommand(
        organization_id="org-one",
        identity_id="builders",
        alias_id="coding",
    )
    assert platform.mutate_grant(grant).changed
    assert not platform.mutate_grant(grant).changed
    assert platform.grants(organization_id="org-two") == ()
    revoke = RevokeAliasGrantCommand(
        organization_id="org-one",
        identity_id="builders",
        alias_id="coding",
    )
    assert platform.mutate_grant(revoke).changed
    assert not platform.mutate_grant(revoke).changed

    budget = SetMonthlyBudgetCommand(
        organization_id="org-one",
        period="2026-08",
        scope=MonthlyBudgetScope(kind=MonthlyBudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )
    assert platform.set_monthly_budget(budget).changed
    assert not platform.set_monthly_budget(budget).changed
    assert platform.monthly_budgets(organization_id="org-two", period="2026-08") == ()
    disable = DisableAliasCommand(organization_id="org-one", alias_id="coding")
    assert platform.mutate_alias(disable).changed
    assert not platform.mutate_alias(disable).changed
    assert platform.mutate_alias(alias).changed
    assert platform.alias_revisions(organization_id="org-one")[0].active
    replacement = alias.model_copy(update={"revision_id": "alias-revision-two"})
    assert platform.mutate_alias(replacement).changed
    with pytest.raises(ValueError, match="historical inactive revision"):
        platform.mutate_alias(alias)

    with sqlite3.connect(platform.database_path) as database:
        assert database.execute("SELECT COUNT(*) FROM operation_receipts").fetchone()[0] == 0


def test_concurrent_alias_replay_converges_on_one_revision(tmp_path: Path) -> None:
    """Concurrent identical activations report one change and one stable replay."""
    platform = _platform(tmp_path)
    command = ActivateAliasRevisionCommand(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    barrier = threading.Barrier(2)

    def activate() -> bool:
        """Start one identical activation at the shared concurrency boundary."""
        barrier.wait(timeout=5)
        return platform.mutate_alias(command).changed

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(activate)
        second = executor.submit(activate)
        outcomes = (first.result(timeout=10), second.result(timeout=10))

    assert sorted(outcomes) == [False, True]
    assert len(platform.alias_revisions(organization_id="org-one")) == 1


def test_alias_reactivation_requires_current_provider_bindings(tmp_path: Path) -> None:
    """A disabled direct alias cannot revive a disabled or revised connection."""
    platform = _platform(tmp_path)
    provider = UpsertProviderConnectionCommand(
        organization_id="org-one",
        connection_id="openai",
        revision_id="provider-revision-one",
        provider="openai",
        secret_reference=OpaqueSecretReference(
            scheme=OpaqueSecretScheme.ENVIRONMENT,
            reference="OPENAI_API_KEY",
        ),
    )
    assert platform.mutate_provider_connection(provider).changed
    authority = platform.provider_connection_revisions(organization_id="org-one")[0]
    alias = ActivateAliasRevisionCommand(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
        provider_connections=(
            ProviderRevisionBinding(
                connection_id=authority.connection_id,
                connection_revision_id=authority.revision_id,
                connection_sha256=authority.connection_sha256,
            ),
        ),
    )
    assert platform.mutate_alias(alias).changed
    assert platform.mutate_alias(
        DisableAliasCommand(organization_id="org-one", alias_id="coding")
    ).changed
    assert platform.mutate_provider_connection(
        DisableProviderConnectionCommand(
            organization_id="org-one",
            connection_id="openai",
        )
    ).changed

    with pytest.raises(ValueError, match="no longer active and current"):
        platform.mutate_alias(alias)


def test_bedrock_auth_modes_survive_management_restart_and_alias_binding(
    tmp_path: Path,
) -> None:
    """Ambient, explicit-pair, and bearer authorities round-trip without ambiguity."""
    platform = _platform(tmp_path)
    secret = OpaqueSecretReference(
        scheme=OpaqueSecretScheme.ENVIRONMENT,
        reference="AWS_SECRET_ACCESS_KEY",
    )
    access_key_id = OpaqueSecretReference(
        scheme=OpaqueSecretScheme.ENVIRONMENT,
        reference="AWS_ACCESS_KEY_ID",
    )
    commands = (
        UpsertProviderConnectionCommand(
            organization_id="org-one",
            connection_id="bedrock-ambient",
            revision_id="bedrock-ambient-revision",
            provider="bedrock",
            region="us-west-2",
        ),
        UpsertProviderConnectionCommand(
            organization_id="org-one",
            connection_id="bedrock-pair",
            revision_id="bedrock-pair-revision",
            provider="bedrock",
            region="us-west-2",
            secret_reference=secret,
            access_key_id_reference=access_key_id,
            bedrock_auth_mode="access_key_pair",
        ),
        UpsertProviderConnectionCommand(
            organization_id="org-one",
            connection_id="bedrock-bearer",
            revision_id="bedrock-bearer-revision",
            provider="bedrock",
            region="us-west-2",
            secret_reference=secret.model_copy(update={"reference": "BEDROCK_API_KEY"}),
            bedrock_auth_mode="api_key",
        ),
    )
    for command in commands:
        assert platform.mutate_provider_connection(command).changed
    pair = next(
        revision
        for revision in platform.provider_connection_revisions(organization_id="org-one")
        if revision.connection_id == "bedrock-pair"
    )
    platform.control.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    assert platform.mutate_alias(
        ActivateAliasRevisionCommand(
            organization_id="org-one",
            alias_id="coding",
            alias_name="coding",
            revision_id="alias-revision-one",
            target=DirectTarget(pool_id="coding-pool"),
            snapshot_ref="catalog-one",
            catalog_sha256=_DIGEST,
            provider_connections=(
                ProviderRevisionBinding(
                    connection_id=pair.connection_id,
                    connection_revision_id=pair.revision_id,
                    connection_sha256=pair.connection_sha256,
                ),
            ),
        )
    ).changed

    restarted = SQLiteGatewayPlatform(
        platform.database_path,
        budgets=SQLiteBudgetStore(platform.database_path),
        attempts=SQLiteAttemptLedger(platform.database_path),
        pool_revisions=_PoolRevisions(),
    )
    revisions = {
        revision.connection_id: revision
        for revision in restarted.provider_connection_revisions(organization_id="org-one")
    }

    assert revisions["bedrock-ambient"].bedrock_auth_mode is None
    assert revisions["bedrock-pair"].bedrock_auth_mode == "access_key_pair"
    assert revisions["bedrock-pair"].access_key_id_reference == access_key_id
    assert revisions["bedrock-bearer"].bedrock_auth_mode == "api_key"
    assert revisions["bedrock-bearer"].access_key_id_reference is None
    assert restarted.alias_revisions(organization_id="org-one")[0].active


def test_revision_reads_forward_existing_sqlite_authority(tmp_path: Path) -> None:
    """The adapter exposes exact provider, alias, and catalog-owned pool revisions."""
    platform = _platform(tmp_path)
    platform.control.upsert_provider_connection(
        organization_id="org-one",
        connection_id="openai",
        revision_id="provider-revision-one",
        config=ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY"),
        replace=False,
    )
    platform.control.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.control.activate_alias_revision(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.budgets.set_limit(
        organization_id="org-one",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_micro_usd=1_000,
    )

    provider = platform.provider_connection_revisions(organization_id="org-one")[0]
    alias = platform.alias_revisions(organization_id="org-one")[0]
    pool = platform.exact_pool_revisions(organization_id="org-one")[0]

    assert provider.secret_reference is not None
    assert provider.secret_reference.reference == "OPENAI_API_KEY"
    assert alias.revision_id == "alias-revision-one"
    assert pool.pool_id == "coding-pool"
    assert pool.exact_model_id == "exact-coding"
    assert pool.deployment_ids == ("deployment-one",)
    assert pool.snapshot_ref == "catalog-one"
    assert (
        platform.monthly_budgets(
            organization_id="org-one",
            period="2026-08",
        )[0].remaining_micro_usd
        == 1_000
    )


def test_natural_provider_replay_preserves_azure_api_surface(tmp_path: Path) -> None:
    """The storage-neutral command and revision view round-trip the Foundry selector."""
    platform = _platform(tmp_path)
    command = UpsertProviderConnectionCommand(
        organization_id="org-one",
        connection_id="foundry",
        revision_id="foundry-revision-one",
        provider="azure",
        base_url="https://resource.services.ai.azure.com",
        api_version="2024-05-01-preview",
        azure_api_surface="model_inference",
        secret_reference=OpaqueSecretReference(
            scheme=OpaqueSecretScheme.ENVIRONMENT,
            reference="AZURE_FOUNDRY_API_KEY",
        ),
    )

    assert platform.mutate_provider_connection(command).changed
    assert not platform.mutate_provider_connection(command).changed
    (revision,) = platform.provider_connection_revisions(organization_id="org-one")
    assert revision.azure_api_surface == "model_inference"
    assert platform.alias_revisions(organization_id="org-two") == ()


def test_attempt_wrapper_returns_precise_reservation_and_settlement(
    tmp_path: Path,
) -> None:
    """Reservation and settlement forward existing atomic ledger transitions."""
    platform = _platform(tmp_path)
    platform.control.create_identity(
        organization_id="org-one",
        identity_id="builders",
        display_name="Builders",
    )
    platform.control.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.control.activate_alias_revision(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.control.grant_alias(
        organization_id="org-one",
        identity_id="builders",
        alias_id="coding",
    )
    raw_key = platform.control.issue_virtual_key(
        organization_id="org-one",
        identity_id="builders",
        key_id="builders-key",
    ).raw_key
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="bounded request"),),
        maximum_output_tokens=16,
    )
    authorization = platform.control.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=10**9,
    )
    platform.attempts.accept_request(authorization=authorization)
    snapshot = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-coding",
        pool_id="coding-pool",
        deployment_ids=("deployment-one",),
    )
    deployment = ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="deployment-one",
        exact_model_id="exact-coding",
        connection="openai",
        provider="openai",
        provider_model="gpt-test",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=16),
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_000_000,
                output_micro_usd_per_million_tokens=1_000_000,
            )
        ),
    )

    reservation = platform.reserve_attempt(
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=snapshot,
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )
    )
    assert (
        platform.reserve_attempt(
            AttemptReservationRequest(
                organization_id="org-one",
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=100,
            )
        )
        == reservation
    )
    with pytest.raises(ValueError, match="differs from durable accounting input"):
        platform.reserve_attempt(
            AttemptReservationRequest(
                organization_id="org-one",
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=101,
            )
        )
    with pytest.raises(ValueError, match="differs from durable accounting input"):
        platform.reserve_attempt(
            AttemptReservationRequest(
                organization_id="org-one",
                snapshot=snapshot,
                deployment=deployment.model_copy(
                    update={
                        "gateway": deployment.gateway.model_copy(
                            update={"pricing_source": "changed"}
                        )
                    }
                ),
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_micro_usd=100,
            )
        )
    settlement = platform.settle_attempt(
        AttemptSettlementRequest(
            organization_id="org-one",
            attempt_id=reservation.attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=0,
                usage=GatewayUsage(input_tokens=10, output_tokens=5),
            ),
        )
    )

    assert reservation.reserved_micro_usd == 100
    assert settlement.reservation == reservation
    assert settlement.state == "completed"
    assert settlement.usage == GatewayUsage(input_tokens=10, output_tokens=5)
    assert settlement.settled_micro_usd == 15
    with pytest.raises(ValueError, match="differs from durable"):
        platform.settle_attempt(
            AttemptSettlementRequest(
                organization_id="org-one",
                attempt_id=reservation.attempt_id,
                terminal_event=GatewayEvent(
                    kind=GatewayEventKind.COMPLETED,
                    sequence_number=0,
                    usage=GatewayUsage(input_tokens=11, output_tokens=5),
                ),
            )
        )
    second_authorization = platform.control.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request.model_copy(
            update={"messages": (GatewayMessage(role="user", content="second request"),)}
        ),
        deadline_monotonic=10**9,
    )
    platform.attempts.accept_request(authorization=second_authorization)
    second_reservation = platform.reserve_attempt(
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=snapshot.model_copy(update={"authorization": second_authorization}),
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )
    )
    terminal = GatewayEvent(
        kind=GatewayEventKind.COMPLETED,
        sequence_number=0,
        usage=GatewayUsage(input_tokens=1, output_tokens=1),
    )
    platform.settle_attempt(
        AttemptSettlementRequest(
            organization_id="org-one",
            attempt_id=second_reservation.attempt_id,
            terminal_event=terminal,
            finalize_request=False,
        )
    )
    with pytest.raises(ValueError, match="cannot finalize"):
        platform.settle_attempt(
            AttemptSettlementRequest(
                organization_id="org-one",
                attempt_id=second_reservation.attempt_id,
                terminal_event=terminal,
                finalize_request=True,
            )
        )
    usage = platform.usage_attribution(organization_id="org-one")
    assert usage.identities[0].known_estimated_cost_micro_usd == 17
    assert usage.identities[0].terminal_counts[0].state == "completed"
    with pytest.raises(ValueError, match="does not belong"):
        platform.settle_attempt(
            AttemptSettlementRequest(
                organization_id="org-two",
                attempt_id=reservation.attempt_id,
                terminal_event=GatewayEvent(
                    kind=GatewayEventKind.COMPLETED,
                    sequence_number=0,
                ),
            )
        )


def test_settlement_surfaces_first_token_time_for_ttft(tmp_path: Path) -> None:
    """A settlement request carrying first_token_at durably surfaces it on the record."""
    platform = _platform(tmp_path)
    platform.control.create_identity(
        organization_id="org-one",
        identity_id="builders",
        display_name="Builders",
    )
    platform.control.register_catalog_snapshot(
        organization_id="org-one",
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.control.activate_alias_revision(
        organization_id="org-one",
        alias_id="coding",
        alias_name="coding",
        revision_id="alias-revision-one",
        target=DirectTarget(pool_id="coding-pool"),
        snapshot_ref="catalog-one",
        catalog_sha256=_DIGEST,
    )
    platform.control.grant_alias(
        organization_id="org-one",
        identity_id="builders",
        alias_id="coding",
    )
    raw_key = platform.control.issue_virtual_key(
        organization_id="org-one",
        identity_id="builders",
        key_id="builders-key",
    ).raw_key
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="bounded request"),),
    )
    authorization = platform.control.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=10**9,
    )
    platform.attempts.accept_request(authorization=authorization)
    snapshot = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-coding",
        pool_id="coding-pool",
        deployment_ids=("deployment-one",),
    )
    deployment = ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="deployment-one",
        exact_model_id="exact-coding",
        connection="openai",
        provider="openai",
        provider_model="gpt-test",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=1_000_000,
                output_micro_usd_per_million_tokens=1_000_000,
            )
        ),
    )
    reservation = platform.reserve_attempt(
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=snapshot,
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=100,
        )
    )
    first_token_at = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
    settlement = platform.settle_attempt(
        AttemptSettlementRequest(
            organization_id="org-one",
            attempt_id=reservation.attempt_id,
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=0,
                usage=GatewayUsage(input_tokens=10, output_tokens=5),
            ),
            first_token_at=first_token_at,
        )
    )
    assert settlement.first_token_at == first_token_at

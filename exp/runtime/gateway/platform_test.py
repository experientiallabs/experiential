"""External call-site tests for storage-neutral gateway platform contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import exp.runtime.gateway as gateway_api
import exp.runtime.gateway.platform as platform_contracts
import exp.runtime.gateway.sqlite as sqlite_api
import exp.runtime.gateway.sqlite.platform as sqlite_platform
from exp.common.core.artifacts import ContractModel
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import BillingSource, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway import (
    ActivateAliasRevisionCommand,
    AliasRevisionRecord,
    AttemptReservationRecord,
    AttemptReservationRequest,
    AttemptSettlementRecord,
    AttemptSettlementRequest,
    AttemptTerminalState,
    AttemptUsageSource,
    BillingSourceUsageAttribution,
    CreateIdentityCommand,
    DirectTarget,
    DisableAliasCommand,
    DisableProviderConnectionCommand,
    ExactPoolRevision,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailureClass,
    GatewayPlatform,
    GatewayUsage,
    GrantAliasCommand,
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
    ProjectTarget,
    ProviderConnectionRevision,
    ProviderRevisionBinding,
    RevokeAliasGrantCommand,
    SetMonthlyBudgetCommand,
    UpsertProviderConnectionCommand,
    UsageAttribution,
    UsageTerminalCount,
    VirtualKeyRecord,
)
from exp.runtime.gateway.contracts import AuthorizationSnapshot, ExecutionSnapshot


def test_external_management_call_site_separates_receipts_from_one_time_secrets() -> None:
    """A caller receives raw key material only from the dedicated issuance result."""

    def provision(platform: GatewayPlatform) -> tuple[ManagementReceipt, OneTimeVirtualKeyResult]:
        """Exercise the intended storage-neutral management call site."""
        receipt = platform.execute(
            CreateIdentityCommand(
                operation_id="op-create-identity",
                organization_id="org-one",
                identity_id="identity-one",
                display_name="Build agents",
            )
        )
        issued = platform.issue_key(
            IssueVirtualKeyCommand(
                operation_id="op-issue-key",
                organization_id="org-one",
                identity_id="identity-one",
                key_id="key-one",
            )
        )
        return receipt, issued

    assert callable(provision)
    assert "raw_key" not in ManagementReceipt.model_fields
    assert "raw_key" not in VirtualKeyRecord.model_fields
    assert "raw_key" in OneTimeVirtualKeyResult.model_fields


def test_public_export_lists_are_explicit_and_complete() -> None:
    """Root packages export every intended contract and only SQLite implementations."""
    assert set(platform_contracts.__all__).issubset(gateway_api.__all__)
    assert set(sqlite_api.__all__) == {"SQLiteGatewayStore"}
    assert set(sqlite_platform.__all__) == {"SQLiteGatewayPlatform"}


def test_domain_models_reject_ambiguous_secret_and_tenant_state() -> None:
    """Platform records require opaque references and tenant-owned identifiers."""
    reference = OpaqueSecretReference(
        scheme=OpaqueSecretScheme.ENVIRONMENT,
        reference="OPENAI_API_KEY",
    )
    revision = ProviderConnectionRevision(
        organization_id="org-one",
        connection_id="openai",
        revision_id="connection-revision-one",
        revision_number=1,
        provider="openai",
        secret_reference=reference,
        connection_sha256="a" * 64,
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
    )
    organization = OrganizationRecord(
        organization_id="org-one",
        slug="one",
        display_name="One",
        active=True,
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        updated_at=datetime(2026, 8, 19, tzinfo=UTC),
    )

    assert revision.secret_reference is not None
    assert revision.secret_reference.reference == "OPENAI_API_KEY"
    assert organization.organization_id == revision.organization_id
    assert "OPENAI_API_KEY" in revision.model_dump_json()
    with pytest.raises(ValidationError):
        OpaqueSecretReference(scheme=OpaqueSecretScheme.ENVIRONMENT, reference="")


@pytest.mark.parametrize(
    "raw_value",
    [
        "exp_vk_0123456789ab_raw-secret",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "SK-ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "vault/path\ninjected",
        "vault/path\x7f",
    ],
)
def test_opaque_secret_reference_rejects_raw_key_material(raw_value: str) -> None:
    """Opaque locators reject recognizable keys and control characters."""
    with pytest.raises(ValidationError):
        OpaqueSecretReference(
            scheme=OpaqueSecretScheme.EXTERNAL_STORE,
            reference=raw_value,
        )


def test_opaque_secret_reference_requires_scheme_specific_locator_syntax() -> None:
    """Each secret source accepts identifiers, not arbitrary credential strings."""
    assert (
        OpaqueSecretReference(
            scheme=OpaqueSecretScheme.EXTERNAL_STORE,
            reference="vault://team/openai",
        ).reference
        == "vault://team/openai"
    )
    assert (
        OpaqueSecretReference(
            scheme=OpaqueSecretScheme.PROVIDER_MANAGED,
            reference="aws-default-chain",
        ).reference
        == "aws-default-chain"
    )
    with pytest.raises(ValidationError, match="environment"):
        OpaqueSecretReference(
            scheme=OpaqueSecretScheme.ENVIRONMENT,
            reference="actual-secret-value",
        )
    with pytest.raises(ValidationError, match="locator URI"):
        OpaqueSecretReference(
            scheme=OpaqueSecretScheme.EXTERNAL_STORE,
            reference="actual-secret-value",
        )


def _authorization() -> AuthorizationSnapshot:
    """Create one valid frozen authority for platform contract tests."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="org-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="coding",
        alias_revision_id="alias-revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        deadline_monotonic=100.0,
    )


def _execution() -> ExecutionSnapshot:
    """Create one valid route-bound execution snapshot."""
    return ExecutionSnapshot(
        authorization=_authorization(),
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("deployment-one",),
    )


def _deployment() -> ExactModelDeployment:
    """Create one exact deployment for attempt contract tests."""
    return ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="deployment-one",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="gpt-test",
        connection_sha256="c" * 64,
        capabilities_sha256="d" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=16),
        gateway=GatewayDeploymentMetadata(),
    )


def _round_trip(model: ContractModel) -> None:
    """Require deterministic JSON serialization and exact typed reconstruction."""
    payload = model.model_dump_json()
    rebuilt = type(model).model_validate_json(payload)
    assert rebuilt == model
    assert rebuilt.model_dump_json() == payload


def test_public_contract_categories_have_stable_json_round_trips() -> None:
    """Every public contract category round-trips through stable JSON."""
    now = datetime(2026, 8, 19, tzinfo=UTC)
    reservation = AttemptReservationRecord(
        organization_id="org-one",
        attempt_id="attempt-one",
        request_id="request-one",
        identity_id="identity-one",
        alias_id="alias-one",
        alias_revision_id="alias-revision-one",
        catalog_sha256="a" * 64,
        pool_id="pool-one",
        exact_model_id="exact-one",
        deployment_id="deployment-one",
        provider="openai",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        attempt_ordinal=0,
        route_depth=0,
        period="2026-08",
        reserved_micro_usd=10,
        started_at=now,
    )
    terminal = UsageTerminalCount(state=AttemptTerminalState.COMPLETED, attempts=1)
    secret = OpaqueSecretReference(
        scheme=OpaqueSecretScheme.ENVIRONMENT,
        reference="OPENAI_API_KEY",
    )
    models: tuple[ContractModel, ...] = (
        secret,
        OrganizationRecord(
            organization_id="org-one",
            slug="one",
            display_name="One",
            active=True,
            created_at=now,
            updated_at=now,
        ),
        IdentityRecord(
            organization_id="org-one",
            identity_id="identity-one",
            display_name="Identity",
            active=True,
            created_at=now,
            updated_at=now,
        ),
        VirtualKeyRecord(
            organization_id="org-one",
            identity_id="identity-one",
            key_id="key-one",
            prefix="0123456789ab",
            active=True,
            created_at=now,
        ),
        GrantRecord(
            organization_id="org-one",
            identity_id="identity-one",
            alias_id="alias-one",
            alias_name="coding",
            created_at=now,
        ),
        ProviderConnectionRevision(
            organization_id="org-one",
            connection_id="connection-one",
            revision_id="connection-revision-one",
            revision_number=1,
            provider="openai",
            secret_reference=secret,
            connection_sha256="a" * 64,
            created_at=now,
        ),
        AliasRevisionRecord(
            organization_id="org-one",
            alias_id="alias-one",
            alias_name="coding",
            revision_id="alias-revision-one",
            revision_number=1,
            target=DirectTarget(pool_id="pool-one"),
            snapshot_ref="catalog-snapshots/aaaaaaaa.json",
            catalog_sha256="a" * 64,
            active=True,
            created_at=now,
        ),
        ExactPoolRevision(
            organization_id="org-one",
            revision_id="pool-revision-one",
            pool_id="pool-one",
            exact_model_id="exact-one",
            deployment_ids=("deployment-one",),
            snapshot_ref="catalog-snapshots/aaaaaaaa.json",
            catalog_sha256="a" * 64,
            created_at=now,
        ),
        MonthlyBudgetRecord(
            budget_id="budget-one",
            organization_id="org-one",
            period="2026-08",
            scope=MonthlyBudgetScope(kind=MonthlyBudgetScopeKind.TEAM),
            limit_micro_usd=100,
            reserved_micro_usd=10,
            settled_micro_usd=20,
            remaining_micro_usd=70,
            unknown_cost_attempts=0,
            exhausted=False,
            created_at=now,
            updated_at=now,
        ),
        MonthlyBudgetScope(kind=MonthlyBudgetScopeKind.TEAM),
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=_execution(),
            deployment=_deployment(),
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_micro_usd=10,
        ),
        reservation,
        AttemptSettlementRequest(
            organization_id="org-one",
            attempt_id="attempt-one",
            terminal_event=GatewayEvent(
                kind=GatewayEventKind.COMPLETED,
                sequence_number=0,
            ),
        ),
        AttemptSettlementRecord(
            reservation=reservation,
            state=AttemptTerminalState.COMPLETED,
            terminal_at=now,
            usage=GatewayUsage(input_tokens=1, output_tokens=2),
            usage_source=AttemptUsageSource.OBSERVED,
            estimated_cost_micro_usd=3,
            settled_micro_usd=3,
        ),
        UsageAttribution(
            organization_id="org-one",
            identities=(
                IdentityUsageAttribution(
                    organization_id="org-one",
                    identity_id="identity-one",
                    requests=1,
                    attempts=1,
                    input_tokens=1,
                    cached_input_tokens=0,
                    output_tokens=2,
                    reasoning_tokens=0,
                    known_estimated_cost_micro_usd=3,
                    unknown_cost_attempts=0,
                    total_latency_ms=5,
                    average_latency_ms=5,
                    terminal_counts=(terminal,),
                ),
            ),
            by_billing_source=(
                BillingSourceUsageAttribution(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    attempts=1,
                    input_tokens=1,
                    cached_input_tokens=0,
                    output_tokens=2,
                    reasoning_tokens=0,
                    known_estimated_cost_micro_usd=3,
                    unknown_cost_attempts=0,
                    terminal_counts=(terminal,),
                ),
            ),
        ),
        terminal,
        CreateIdentityCommand(
            operation_id="operation-one",
            organization_id="org-one",
            identity_id="identity-one",
            display_name="Identity",
        ),
        IssueVirtualKeyCommand(
            operation_id="operation-two",
            organization_id="org-one",
            identity_id="identity-one",
            key_id="key-one",
        ),
        ManagementReceipt(
            organization_id="org-one",
            operation_id="operation-one",
            action=ManagementAction.CREATE_IDENTITY,
            command_sha256="a" * 64,
            resource_kind="identity",
            resource_id="identity-one",
            created_at=now,
        ),
        OneTimeVirtualKeyResult(
            receipt=ManagementReceipt(
                organization_id="org-one",
                operation_id="operation-two",
                action=ManagementAction.ISSUE_VIRTUAL_KEY,
                command_sha256="b" * 64,
                resource_kind="virtual_key",
                resource_id="key-one",
                created_at=now,
            ),
            key=VirtualKeyRecord(
                organization_id="org-one",
                identity_id="identity-one",
                key_id="key-one",
                prefix="0123456789ab",
                active=True,
                created_at=now,
            ),
            raw_key="exp_vk_0123456789ab_one-time-material",
        ),
        GrantAliasCommand(
            organization_id="org-one",
            identity_id="identity-one",
            alias_id="alias-one",
        ),
        RevokeAliasGrantCommand(
            organization_id="org-one",
            identity_id="identity-one",
            alias_id="alias-one",
        ),
        UpsertProviderConnectionCommand(
            organization_id="org-one",
            connection_id="connection-one",
            revision_id="connection-revision-one",
            provider="openai",
            secret_reference=secret,
        ),
        DisableProviderConnectionCommand(
            organization_id="org-one",
            connection_id="connection-one",
        ),
        ProviderRevisionBinding(
            connection_id="connection-one",
            connection_revision_id="connection-revision-one",
            connection_sha256="a" * 64,
        ),
        ActivateAliasRevisionCommand(
            organization_id="org-one",
            alias_id="alias-one",
            alias_name="coding",
            revision_id="alias-revision-one",
            target=DirectTarget(pool_id="pool-one"),
            snapshot_ref="snapshot-one",
            catalog_sha256="a" * 64,
        ),
        DisableAliasCommand(organization_id="org-one", alias_id="alias-one"),
        SetMonthlyBudgetCommand(
            organization_id="org-one",
            period="2026-08",
            scope=MonthlyBudgetScope(kind=MonthlyBudgetScopeKind.TEAM),
            limit_micro_usd=100,
        ),
        NaturalMutationOutcome(
            organization_id="org-one",
            action=NaturalMutationAction.GRANT_ALIAS,
            resource_id="grant-one",
            changed=True,
        ),
    )

    for model in models:
        _round_trip(model)


def test_target_scope_attempt_and_settlement_validators_fail_closed() -> None:
    """Cross-boundary drift and incoherent settlement records are rejected."""
    with pytest.raises(ValidationError, match="target catalog"):
        AliasRevisionRecord(
            organization_id="org-one",
            alias_id="alias-one",
            alias_name="coding",
            revision_id="alias-revision-one",
            revision_number=1,
            target=ProjectTarget(
                project_ref="project-one",
                activation_ref="activation-one",
                catalog_sha256="b" * 64,
            ),
            snapshot_ref="snapshot-one",
            catalog_sha256="a" * 64,
            active=True,
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="invalid identifiers"):
        MonthlyBudgetScope(
            kind=MonthlyBudgetScopeKind.IDENTITY,
            identity_id="identity-one",
            alias_id="alias-one",
        )
    with pytest.raises(ValidationError, match="absent"):
        AttemptReservationRequest(
            organization_id="org-one",
            snapshot=_execution(),
            deployment=_deployment().model_copy(update={"deployment_id": "deployment-two"}),
            attempt_ordinal=0,
            route_depth=0,
        )
    with pytest.raises(ValidationError, match="successful"):
        AttemptSettlementRecord(
            reservation=AttemptReservationRecord(
                organization_id="org-one",
                attempt_id="attempt-one",
                request_id="request-one",
                identity_id="identity-one",
                alias_id="alias-one",
                alias_revision_id="alias-revision-one",
                catalog_sha256="a" * 64,
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_id="deployment-one",
                provider="openai",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                attempt_ordinal=0,
                route_depth=0,
                period="2026-08",
                started_at=datetime(2026, 8, 19, tzinfo=UTC),
            ),
            state=AttemptTerminalState.COMPLETED,
            terminal_at=datetime(2026, 8, 19, tzinfo=UTC),
            failure_class=GatewayFailureClass.INTERNAL,
            usage_source=AttemptUsageSource.UNKNOWN,
        )

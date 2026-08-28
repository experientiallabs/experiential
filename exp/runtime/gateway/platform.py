"""Storage-neutral authority, management, accounting, and usage contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256
from exp.common.models.catalog import BillingSource, GatewayEquivalenceCertification
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelDeployment,
    ExactModelId,
    ExactModelPoolId,
)
from exp.runtime.gateway.contracts import (
    AttemptId,
    ExecutionSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayTarget,
    GatewayUsage,
    IdentityId,
    OrganizationId,
    VirtualKeyId,
)

ConnectionId = ArtifactId
ProviderConnectionRevisionId = ArtifactId
ExactPoolRevisionId = ArtifactId
ManagementOperationId = ArtifactId


class OpaqueSecretScheme(StrEnum):
    """Supported non-secret locator schemes."""

    ENVIRONMENT = "environment"
    EXTERNAL_STORE = "external_store"
    PROVIDER_MANAGED = "provider_managed"


class OpaqueSecretReference(ContractModel):
    """A locator that identifies secret material without containing its value."""

    scheme: OpaqueSecretScheme
    reference: str = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def _require_locator_syntax(self) -> OpaqueSecretReference:
        """Require scheme-specific locator syntax that cannot be a bare credential."""
        if any(ord(character) < 32 or ord(character) == 127 for character in self.reference):
            raise ValueError("secret references must not contain control characters")
        lowered = self.reference.lower()
        if lowered.startswith(("exp_vk_", "sk-")):
            raise ValueError("secret references must identify a locator, not raw key material")
        if self.scheme is OpaqueSecretScheme.ENVIRONMENT:
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", self.reference) is None:
                raise ValueError("environment secret references must name one variable")
            return self
        if self.scheme is OpaqueSecretScheme.EXTERNAL_STORE:
            parsed = urlsplit(self.reference)
            if (
                not parsed.scheme
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("external secret references must be credential-free locator URIs")
            return self
        if (
            re.fullmatch(
                r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*",
                self.reference,
            )
            is None
        ):
            raise ValueError("provider-managed secret references must use an opaque identifier")
        return self


class OrganizationRecord(ContractModel):
    """One tenant organization and its lifecycle state."""

    organization_id: OrganizationId
    slug: ArtifactId
    display_name: str = Field(min_length=1, max_length=256)
    active: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime


class IdentityRecord(ContractModel):
    """One non-secret principal owned by an organization."""

    organization_id: OrganizationId
    identity_id: IdentityId
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2_048)
    active: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime


class VirtualKeyRecord(ContractModel):
    """Durable virtual-key metadata that never contains raw key material."""

    organization_id: OrganizationId
    identity_id: IdentityId
    key_id: VirtualKeyId
    prefix: str = Field(min_length=1, max_length=64)
    active: bool
    expires_at: AwareDatetime | None = None
    revoked_at: AwareDatetime | None = None
    created_at: AwareDatetime
    last_used_at: AwareDatetime | None = None


class GrantRecord(ContractModel):
    """One explicit identity-to-alias authorization."""

    organization_id: OrganizationId
    identity_id: IdentityId
    alias_id: ArtifactId
    alias_name: ArtifactId
    created_at: AwareDatetime


class ProviderConnectionRevision(ContractModel):
    """One immutable provider endpoint revision with only an opaque secret locator."""

    organization_id: OrganizationId
    connection_id: ConnectionId
    revision_id: ProviderConnectionRevisionId
    revision_number: int = Field(ge=1)
    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_version: str | None = Field(default=None, max_length=256)
    region: str | None = Field(default=None, max_length=256)
    secret_reference: OpaqueSecretReference | None = None
    connection_sha256: Sha256
    active: bool = True
    created_at: AwareDatetime


class AliasRevisionRecord(ContractModel):
    """One immutable alias target revision."""

    organization_id: OrganizationId
    alias_id: ArtifactId
    alias_name: ArtifactId
    revision_id: ArtifactId
    revision_number: int = Field(ge=1)
    target: GatewayTarget
    snapshot_ref: str = Field(min_length=1, max_length=2_048)
    catalog_sha256: Sha256
    refusal_failover: bool = False
    active: bool
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _require_target_catalog(self) -> AliasRevisionRecord:
        """Require project targets to carry the alias revision catalog digest."""
        target_digest = getattr(self.target, "catalog_sha256", self.catalog_sha256)
        if target_digest != self.catalog_sha256:
            raise ValueError("alias target catalog differs from its revision")
        return self


class ExactPoolRevision(ContractModel):
    """One complete ordered exact-model pool from an immutable catalog snapshot."""

    organization_id: OrganizationId
    revision_id: ExactPoolRevisionId
    pool_id: ExactModelPoolId
    exact_model_id: ExactModelId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    equivalence: GatewayEquivalenceCertification | None = None
    snapshot_ref: str = Field(min_length=1, max_length=2_048)
    catalog_sha256: Sha256
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _require_pool_coherence(self) -> ExactPoolRevision:
        """Require unique routes and explicit equivalence for multi-route pools."""
        if len(set(self.deployment_ids)) != len(self.deployment_ids):
            raise ValueError("exact pool revision deployments must not repeat")
        if len(self.deployment_ids) > 1 and self.equivalence is None:
            raise ValueError("multi-deployment pool revisions require equivalence evidence")
        if len(self.deployment_ids) == 1 and self.equivalence is not None:
            raise ValueError("singleton pool revisions must not assert equivalence evidence")
        return self


class MonthlyBudgetScopeKind(StrEnum):
    """Supported hard-limit scope categories."""

    TEAM = "team"
    IDENTITY = "identity"
    POOL = "pool"
    DEPLOYMENT = "deployment"


class MonthlyBudgetScope(ContractModel):
    """One precise organization-relative monthly budget scope."""

    kind: MonthlyBudgetScopeKind
    identity_id: IdentityId | None = None
    alias_id: ArtifactId | None = None
    pool_id: ExactModelPoolId | None = None
    deployment_id: DeploymentId | None = None

    @model_validator(mode="after")
    def _require_scope_shape(self) -> MonthlyBudgetScope:
        """Require exactly the identifiers owned by the selected scope."""
        present = (
            self.identity_id is not None,
            self.alias_id is not None,
            self.pool_id is not None,
            self.deployment_id is not None,
        )
        expected = {
            MonthlyBudgetScopeKind.TEAM: (False, False, False, False),
            MonthlyBudgetScopeKind.IDENTITY: (True, False, False, False),
            MonthlyBudgetScopeKind.POOL: (False, True, True, False),
            MonthlyBudgetScopeKind.DEPLOYMENT: (False, True, True, True),
        }[self.kind]
        if present != expected:
            raise ValueError(f"{self.kind.value} budget scope has invalid identifiers")
        return self


class MonthlyBudgetRecord(ContractModel):
    """One organization-owned hard monthly allocation and materialized balance."""

    budget_id: ArtifactId
    organization_id: OrganizationId
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    scope: MonthlyBudgetScope
    limit_micro_usd: int = Field(ge=0)
    reserved_micro_usd: int = Field(ge=0)
    settled_micro_usd: int = Field(ge=0)
    remaining_micro_usd: int = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    exhausted: bool
    created_at: AwareDatetime
    updated_at: AwareDatetime


class AttemptReservationRequest(ContractModel):
    """A trusted route-bound request to atomically reserve and persist one dispatch.

    The gateway constructs this callback input only after authorization and catalog
    selection. An accounting adapter records that frozen decision, it does not
    independently authorize provider or pricing metadata.
    """

    organization_id: OrganizationId
    snapshot: ExecutionSnapshot
    deployment: ExactModelDeployment
    attempt_ordinal: int = Field(ge=0)
    route_depth: int = Field(ge=0)
    maximum_cost_micro_usd: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_matching_organization(self) -> AttemptReservationRequest:
        """Reject tenant, deployment-list, or exact-model drift."""
        if self.organization_id != self.snapshot.authorization.organization_id:
            raise ValueError("attempt reservation organization differs from its snapshot")
        if self.deployment.deployment_id not in self.snapshot.deployment_ids:
            raise ValueError("attempt deployment is absent from its execution snapshot")
        if self.deployment.exact_model_id != self.snapshot.exact_model_id:
            raise ValueError("attempt deployment changes the selected exact model")
        return self


class AttemptReservationRecord(ContractModel):
    """One durable physical-attempt reservation created before provider dispatch."""

    organization_id: OrganizationId
    attempt_id: AttemptId
    request_id: ArtifactId
    identity_id: IdentityId
    alias_id: ArtifactId
    alias_revision_id: ArtifactId
    catalog_sha256: Sha256
    pool_id: ExactModelPoolId
    exact_model_id: ExactModelId
    deployment_id: DeploymentId
    provider: str = Field(min_length=1, max_length=128)
    billing_source: BillingSource
    input_rate: int | None = Field(default=None, ge=0)
    cached_input_rate: int | None = Field(default=None, ge=0)
    output_rate: int | None = Field(default=None, ge=0)
    reasoning_rate: int | None = Field(default=None, ge=0)
    attempt_ordinal: int = Field(ge=0)
    route_depth: int = Field(ge=0)
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    reserved_micro_usd: int | None = Field(default=None, ge=0)
    started_at: AwareDatetime


class AttemptSettlementRequest(ContractModel):
    """A tenant-scoped request to settle one reserved attempt exactly once."""

    organization_id: OrganizationId
    attempt_id: AttemptId
    terminal_event: GatewayEvent | None = None
    failure: GatewayFailure | None = None
    finalize_request: bool = True
    first_token_at: AwareDatetime | None = None
    """Wall-clock time the winning attempt streamed its first token, or ``None``."""

    @model_validator(mode="after")
    def _require_terminal_input(self) -> AttemptSettlementRequest:
        """Require a terminal event or sanitized failure with coherent failure fields."""
        if self.terminal_event is None and self.failure is None:
            raise ValueError("attempt settlement needs a terminal event or failure")
        if self.terminal_event is not None and self.terminal_event.kind not in {
            GatewayEventKind.COMPLETED,
            GatewayEventKind.INCOMPLETE,
            GatewayEventKind.FAILED,
        }:
            raise ValueError("attempt settlement event must be terminal")
        if (
            self.terminal_event is not None
            and self.terminal_event.failure is not None
            and self.failure is not None
            and self.terminal_event.failure != self.failure
        ):
            raise ValueError("attempt settlement failures disagree")
        return self


class AttemptTerminalState(StrEnum):
    """Closed durable terminal states for physical attempts."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    UNKNOWN_AFTER_CRASH = "unknown_after_crash"


class AttemptUsageSource(StrEnum):
    """Closed provenance values for attempt usage."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class AttemptSettlementRecord(ContractModel):
    """One precise durable attempt settlement and attributed usage."""

    reservation: AttemptReservationRecord
    state: AttemptTerminalState
    terminal_at: AwareDatetime
    failure_class: GatewayFailureClass | None = None
    usage: GatewayUsage | None = None
    usage_source: AttemptUsageSource
    estimated_cost_micro_usd: int | None = Field(default=None, ge=0)
    settled_micro_usd: int | None = Field(default=None, ge=0)
    first_token_at: AwareDatetime | None = None
    """Wall-clock time this attempt streamed its first token, or ``None`` when it never did.

    The platform stores this alongside ``terminal_at`` to derive time-to-first-token relative
    to the parent request's ``accepted_at`` and to narrow generation duration to the streaming
    window. It is absent for attempts that failed before any token or crashed unobserved.
    """

    @model_validator(mode="after")
    def _require_settlement_coherence(self) -> AttemptSettlementRecord:
        """Require terminal state, failure, and usage provenance to agree."""
        if self.state in {AttemptTerminalState.COMPLETED, AttemptTerminalState.INCOMPLETE}:
            if self.failure_class is not None:
                raise ValueError("successful attempt states cannot carry a failure class")
        elif (
            self.failure_class is None
            and self.state is not AttemptTerminalState.UNKNOWN_AFTER_CRASH
        ):
            raise ValueError("failed and cancelled attempts require a failure class")
        if self.state is AttemptTerminalState.CANCELLED:
            if self.failure_class is not GatewayFailureClass.CANCELLED:
                raise ValueError("cancelled attempts require the cancelled failure class")
        elif self.failure_class is GatewayFailureClass.CANCELLED:
            raise ValueError("cancelled failure class requires cancelled attempt state")
        if self.state is AttemptTerminalState.UNKNOWN_AFTER_CRASH and (
            self.failure_class is not None
            or self.usage is not None
            or self.usage_source is not AttemptUsageSource.UNKNOWN
            or self.first_token_at is not None
        ):
            raise ValueError("unknown crash settlement cannot claim failure or usage evidence")
        if self.usage is None and self.usage_source is AttemptUsageSource.OBSERVED:
            raise ValueError("observed usage source requires usage")
        if self.usage is not None and self.usage_source is AttemptUsageSource.UNKNOWN:
            raise ValueError("unknown usage source cannot carry usage")
        return self


class UsageTerminalCount(ContractModel):
    """Count of physical attempts in one terminal state."""

    state: AttemptTerminalState
    attempts: int = Field(ge=0)


class IdentityUsageAttribution(ContractModel):
    """Content-free request, attempt, token, cost, and latency attribution."""

    organization_id: OrganizationId
    identity_id: IdentityId
    requests: int = Field(ge=0)
    attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    known_estimated_cost_micro_usd: int = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    total_latency_ms: int = Field(ge=0)
    average_latency_ms: float | None = Field(default=None, ge=0)
    terminal_counts: tuple[UsageTerminalCount, ...]


class BillingSourceUsageAttribution(ContractModel):
    """Physical-attempt attribution by frozen credential ownership."""

    billing_source: BillingSource
    attempts: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    known_estimated_cost_micro_usd: int = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    terminal_counts: tuple[UsageTerminalCount, ...]


class UsageAttribution(ContractModel):
    """One tenant-scoped usage snapshot across logical and billing ownership."""

    organization_id: OrganizationId
    identities: tuple[IdentityUsageAttribution, ...]
    by_billing_source: tuple[BillingSourceUsageAttribution, ...]


class ManagementAction(StrEnum):
    """Idempotent management mutations supported by the initial contract."""

    CREATE_IDENTITY = "create_identity"
    ISSUE_VIRTUAL_KEY = "issue_virtual_key"


class CreateIdentityCommand(ContractModel):
    """Idempotently create one organization-owned identity."""

    kind: Literal["create_identity"] = "create_identity"
    operation_id: ManagementOperationId
    organization_id: OrganizationId
    identity_id: IdentityId
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2_048)


class IssueVirtualKeyCommand(ContractModel):
    """Idempotently issue one key whose raw value may be returned only once."""

    kind: Literal["issue_virtual_key"] = "issue_virtual_key"
    operation_id: ManagementOperationId
    organization_id: OrganizationId
    identity_id: IdentityId
    key_id: VirtualKeyId
    expires_at: AwareDatetime | None = None


class ManagementReceipt(ContractModel):
    """Durable secret-free proof of one atomic management mutation."""

    schema_version: Literal[1] = 1
    organization_id: OrganizationId
    operation_id: ManagementOperationId
    action: ManagementAction
    command_sha256: Sha256
    resource_kind: str = Field(min_length=1, max_length=128)
    resource_id: ArtifactId
    created_at: AwareDatetime


class OneTimeVirtualKeyResult(ContractModel):
    """Dedicated one-time result carrying newly generated raw key material."""

    receipt: ManagementReceipt
    key: VirtualKeyRecord
    raw_key: str = Field(min_length=1)


class NaturalMutationAction(StrEnum):
    """Naturally idempotent mutations without durable operation receipts."""

    GRANT_ALIAS = "grant_alias"
    REVOKE_ALIAS_GRANT = "revoke_alias_grant"
    UPSERT_PROVIDER_CONNECTION = "upsert_provider_connection"
    DISABLE_PROVIDER_CONNECTION = "disable_provider_connection"
    ACTIVATE_ALIAS_REVISION = "activate_alias_revision"
    DISABLE_ALIAS = "disable_alias"
    SET_MONTHLY_BUDGET = "set_monthly_budget"


class NaturalMutationOutcome(ContractModel):
    """Typed result for a naturally idempotent non-receipted mutation."""

    organization_id: OrganizationId
    action: NaturalMutationAction
    resource_id: ArtifactId
    changed: bool


class GrantAliasCommand(ContractModel):
    """Naturally idempotently add one identity-to-alias grant."""

    kind: Literal["grant_alias"] = "grant_alias"
    organization_id: OrganizationId
    identity_id: IdentityId
    alias_id: ArtifactId


class RevokeAliasGrantCommand(ContractModel):
    """Naturally idempotently remove one identity-to-alias grant."""

    kind: Literal["revoke_alias_grant"] = "revoke_alias_grant"
    organization_id: OrganizationId
    identity_id: IdentityId
    alias_id: ArtifactId


GrantMutationCommand = Annotated[
    GrantAliasCommand | RevokeAliasGrantCommand,
    Field(discriminator="kind"),
]


class UpsertProviderConnectionCommand(ContractModel):
    """Create or explicitly revise one provider connection."""

    kind: Literal["upsert_provider_connection"] = "upsert_provider_connection"
    organization_id: OrganizationId
    connection_id: ConnectionId
    revision_id: ProviderConnectionRevisionId
    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_version: str | None = Field(default=None, max_length=256)
    region: str | None = Field(default=None, max_length=256)
    secret_reference: OpaqueSecretReference | None = None
    replace: bool = False


class DisableProviderConnectionCommand(ContractModel):
    """Naturally idempotently disable one unreferenced provider connection."""

    kind: Literal["disable_provider_connection"] = "disable_provider_connection"
    organization_id: OrganizationId
    connection_id: ConnectionId


ProviderConnectionMutationCommand = Annotated[
    UpsertProviderConnectionCommand | DisableProviderConnectionCommand,
    Field(discriminator="kind"),
]


class ProviderRevisionBinding(ContractModel):
    """One alias binding to an exact provider connection revision."""

    connection_id: ConnectionId
    connection_revision_id: ProviderConnectionRevisionId
    connection_sha256: Sha256


class ActivateAliasRevisionCommand(ContractModel):
    """Naturally idempotently activate one immutable alias revision."""

    kind: Literal["activate_alias_revision"] = "activate_alias_revision"
    organization_id: OrganizationId
    alias_id: ArtifactId
    alias_name: ArtifactId
    revision_id: ArtifactId
    target: GatewayTarget
    snapshot_ref: str = Field(min_length=1, max_length=2_048)
    catalog_sha256: Sha256
    provider_connections: tuple[ProviderRevisionBinding, ...] = ()
    refusal_failover: bool = False

    @model_validator(mode="after")
    def _require_target_catalog(self) -> ActivateAliasRevisionCommand:
        """Require project target and activation catalog identity to match."""
        target_digest = getattr(self.target, "catalog_sha256", self.catalog_sha256)
        if target_digest != self.catalog_sha256:
            raise ValueError("alias activation target catalog differs from its command")
        names = tuple(item.connection_id for item in self.provider_connections)
        if len(set(names)) != len(names):
            raise ValueError("alias provider connection bindings must not repeat")
        return self


class DisableAliasCommand(ContractModel):
    """Naturally idempotently disable one alias."""

    kind: Literal["disable_alias"] = "disable_alias"
    organization_id: OrganizationId
    alias_id: ArtifactId


AliasMutationCommand = Annotated[
    ActivateAliasRevisionCommand | DisableAliasCommand,
    Field(discriminator="kind"),
]


class SetMonthlyBudgetCommand(ContractModel):
    """Create or explicitly replace one monthly budget."""

    kind: Literal["set_monthly_budget"] = "set_monthly_budget"
    organization_id: OrganizationId
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    scope: MonthlyBudgetScope
    limit_micro_usd: int = Field(ge=0)
    replace: bool = False


@runtime_checkable
class ManagementCommandAuthority(Protocol):
    """Idempotent management mutation seam."""

    def execute(self, command: CreateIdentityCommand) -> ManagementReceipt:
        """Execute or replay one secret-free management command."""
        ...

    def issue_key(self, command: IssueVirtualKeyCommand) -> OneTimeVirtualKeyResult:
        """Issue one virtual key and return its raw value exactly once."""
        ...


@runtime_checkable
class OrganizationIdentityKeyAuthority(Protocol):
    """Tenant, principal, and non-secret key metadata read seam."""

    def organization(self, *, organization_id: str) -> OrganizationRecord | None:
        """Read one explicitly selected organization."""
        ...

    def identities(self, *, organization_id: str) -> tuple[IdentityRecord, ...]:
        """List identities owned by one tenant."""
        ...

    def keys(self, *, organization_id: str) -> tuple[VirtualKeyRecord, ...]:
        """List non-secret key metadata owned by one tenant."""
        ...


@runtime_checkable
class GrantAuthority(Protocol):
    """Explicit tenant-scoped grant read seam."""

    def grants(self, *, organization_id: str) -> tuple[GrantRecord, ...]:
        """List grants owned by one tenant."""
        ...


@runtime_checkable
class GrantMutationAuthority(Protocol):
    """Naturally idempotent grant mutation seam."""

    def mutate_grant(self, command: GrantMutationCommand) -> NaturalMutationOutcome:
        """Add or remove one tenant-owned grant without claiming a receipt."""
        ...


@runtime_checkable
class ProviderConnectionRevisionAuthority(Protocol):
    """Immutable provider connection revision read seam."""

    def provider_connection_revisions(
        self, *, organization_id: str
    ) -> tuple[ProviderConnectionRevision, ...]:
        """List all exact provider revisions with current activity marked."""
        ...


@runtime_checkable
class ProviderConnectionMutationAuthority(Protocol):
    """Naturally idempotent provider connection mutation seam."""

    def mutate_provider_connection(
        self,
        command: ProviderConnectionMutationCommand,
    ) -> NaturalMutationOutcome:
        """Upsert or disable one tenant-owned provider connection."""
        ...


@runtime_checkable
class RoutingRevisionAuthority(Protocol):
    """Immutable alias revision read seam."""

    def alias_revisions(self, *, organization_id: str) -> tuple[AliasRevisionRecord, ...]:
        """List immutable alias revisions for one tenant."""
        ...


@runtime_checkable
class ExactPoolRevisionAuthority(Protocol):
    """Complete catalog-backed exact-pool revision read seam."""

    def exact_pool_revisions(self, *, organization_id: str) -> tuple[ExactPoolRevision, ...]:
        """List complete exact-pool revisions for one tenant."""
        ...


@runtime_checkable
class RoutingMutationAuthority(Protocol):
    """Naturally idempotent alias revision mutation seam."""

    def mutate_alias(self, command: AliasMutationCommand) -> NaturalMutationOutcome:
        """Activate or disable one tenant-owned alias."""
        ...


@runtime_checkable
class MonthlyBudgetAuthority(Protocol):
    """Monthly hard-limit balance read seam."""

    def monthly_budgets(
        self, *, organization_id: str, period: str
    ) -> tuple[MonthlyBudgetRecord, ...]:
        """List monthly limits and materialized balances for one tenant."""
        ...


@runtime_checkable
class MonthlyBudgetMutationAuthority(Protocol):
    """Naturally idempotent monthly budget mutation seam."""

    def set_monthly_budget(self, command: SetMonthlyBudgetCommand) -> NaturalMutationOutcome:
        """Create, replay, or explicitly replace one monthly hard limit."""
        ...


@runtime_checkable
class AttemptAccountingAuthority(Protocol):
    """Atomic physical-attempt reservation and settlement seam."""

    def reserve_attempt(self, request: AttemptReservationRequest) -> AttemptReservationRecord:
        """Atomically reserve budget and persist one attempt before dispatch."""
        ...

    def settle_attempt(self, request: AttemptSettlementRequest) -> AttemptSettlementRecord:
        """Atomically settle usage and budget for one tenant-owned attempt."""
        ...


@runtime_checkable
class UsageAttributionAuthority(Protocol):
    """Tenant-scoped content-free usage attribution seam."""

    def usage_attribution(
        self, *, organization_id: str, identity_id: str | None = None
    ) -> UsageAttribution:
        """Read one tenant-scoped content-free usage snapshot."""
        ...


@runtime_checkable
class GatewayPlatform(
    ManagementCommandAuthority,
    OrganizationIdentityKeyAuthority,
    GrantAuthority,
    GrantMutationAuthority,
    ProviderConnectionRevisionAuthority,
    ProviderConnectionMutationAuthority,
    RoutingRevisionAuthority,
    ExactPoolRevisionAuthority,
    MonthlyBudgetAuthority,
    MonthlyBudgetMutationAuthority,
    RoutingMutationAuthority,
    AttemptAccountingAuthority,
    UsageAttributionAuthority,
    Protocol,
):
    """Cohesive composition of the narrow storage-neutral platform seams."""


__all__ = [
    "ActivateAliasRevisionCommand",
    "AliasMutationCommand",
    "AliasRevisionRecord",
    "AttemptAccountingAuthority",
    "AttemptReservationRecord",
    "AttemptReservationRequest",
    "AttemptSettlementRecord",
    "AttemptSettlementRequest",
    "AttemptTerminalState",
    "AttemptUsageSource",
    "BillingSourceUsageAttribution",
    "CreateIdentityCommand",
    "DisableAliasCommand",
    "DisableProviderConnectionCommand",
    "ExactPoolRevision",
    "ExactPoolRevisionAuthority",
    "GatewayPlatform",
    "GrantAuthority",
    "GrantAliasCommand",
    "GrantMutationAuthority",
    "GrantMutationCommand",
    "GrantRecord",
    "IdentityRecord",
    "IdentityUsageAttribution",
    "IssueVirtualKeyCommand",
    "ManagementAction",
    "ManagementCommandAuthority",
    "ManagementReceipt",
    "MonthlyBudgetAuthority",
    "MonthlyBudgetMutationAuthority",
    "MonthlyBudgetRecord",
    "MonthlyBudgetScope",
    "MonthlyBudgetScopeKind",
    "NaturalMutationAction",
    "NaturalMutationOutcome",
    "OpaqueSecretReference",
    "OpaqueSecretScheme",
    "OneTimeVirtualKeyResult",
    "OrganizationIdentityKeyAuthority",
    "OrganizationRecord",
    "ProviderConnectionMutationAuthority",
    "ProviderConnectionMutationCommand",
    "ProviderConnectionRevision",
    "ProviderConnectionRevisionAuthority",
    "ProviderRevisionBinding",
    "RevokeAliasGrantCommand",
    "RoutingMutationAuthority",
    "RoutingRevisionAuthority",
    "SetMonthlyBudgetCommand",
    "UpsertProviderConnectionCommand",
    "UsageAttribution",
    "UsageAttributionAuthority",
    "UsageTerminalCount",
    "VirtualKeyRecord",
]

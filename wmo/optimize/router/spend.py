"""Versioned provider-spend evidence for hosted Project build and optimization stages."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    sorted_unique_inputs,
    stable_id,
)
from wmo.common.core.money import USD_ZERO, exact_usd, reserve_usd
from wmo.common.models import BillingSource, OperationEconomics, Usage
from wmo.common.project import ArtifactStore, ProjectStage, artifact_input

_LEDGER_ARTIFACT_TYPE = "provider-spend-ledger"
_LEDGER_FILE = "spend-ledger.json"


class ProviderSpendComponent(StrEnum):
    """Customer-safe provider-backed components in one hosted attempt."""

    WORLD_MODEL = "world_model"
    CANDIDATE = "candidate"
    JUDGE = "judge"
    RETRIEVAL_EMBEDDING = "retrieval_embedding"
    ROUTER_EMBEDDING = "router_embedding"
    OTHER_PROVIDER = "other_provider"


class ProviderSpendStatus(StrEnum):
    """Evidence strength associated with one charged or unused operation."""

    OBSERVED = "observed"
    LOCALLY_PRICED = "locally_priced"
    RESERVED = "reserved"
    NOT_INCURRED = "not_incurred"


class ProviderSpendEntry(ContractModel):
    """One alias-free component charge, reservation, or explicit zero-call record."""

    operation_id: ArtifactId
    component: ProviderSpendComponent
    billing_source: BillingSource
    status: ProviderSpendStatus
    operation_count: int = Field(ge=0)
    amount_usd: Decimal = Field(ge=0)
    usage: Usage | None = None
    evidence: ArtifactInput | None = None

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _require_finite_amount(cls, value: object) -> Decimal:
        """Return one conservative exact numeric(20,6) amount."""
        return reserve_usd(value)

    @model_validator(mode="after")
    def _require_status_evidence(self) -> ProviderSpendEntry:
        """Keep zero-call, measured, locally priced, and ambiguity records unambiguous."""
        if self.status == ProviderSpendStatus.NOT_INCURRED:
            if self.operation_count != 0 or self.amount_usd != 0 or self.usage is not None:
                raise ValueError("not-incurred spend must have zero calls, zero cost, and no usage")
        elif self.operation_count == 0:
            raise ValueError("incurred or reserved spend requires at least one operation")
        if self.status == ProviderSpendStatus.LOCALLY_PRICED and self.usage is None:
            raise ValueError("locally priced spend requires observed provider usage")
        return self


class ProviderSpendLedger(ArtifactEnvelope):
    """Complete conservative component accounting under one finite attempt ceiling."""

    ledger_id: ArtifactId
    project_id: ArtifactId
    stage: ProjectStage
    attempt_id: ArtifactId
    attempt_authority_sha256: Sha256
    ceiling_usd: Decimal = Field(gt=0)
    stage_outputs: tuple[ArtifactInput, ...] = ()
    entries: tuple[ProviderSpendEntry, ...]
    total_usd: Decimal = Field(ge=0)
    outcome: Literal["completed", "failed_closed"]
    restart: Literal["completed_stage_bundle", "blocked_ambiguous_operation"]

    @field_validator("ceiling_usd", mode="before")
    @classmethod
    def _require_exact_ceiling(cls, value: object) -> Decimal:
        """Return one exact positive numeric(20,6) ceiling."""
        return exact_usd(value)

    @field_validator("total_usd", mode="before")
    @classmethod
    def _require_exact_total(cls, value: object) -> Decimal:
        """Return one exact nonnegative numeric(20,6) ledger total."""
        return exact_usd(value, allow_zero=True)

    @field_validator("entries")
    @classmethod
    def _require_canonical_entries(
        cls,
        value: tuple[ProviderSpendEntry, ...],
    ) -> tuple[ProviderSpendEntry, ...]:
        """Require one sorted operation identity per ledger entry."""
        operation_ids = tuple(item.operation_id for item in value)
        if not value or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("provider spend entries need unique operation identities")
        if operation_ids != tuple(sorted(operation_ids)):
            raise ValueError("provider spend entries must be sorted by operation identity")
        return value

    @field_validator("stage_outputs")
    @classmethod
    def _require_canonical_stage_outputs(
        cls,
        value: tuple[ArtifactInput, ...],
    ) -> tuple[ArtifactInput, ...]:
        """Require sorted unique durable outputs completed by this ledger stage."""
        artifact_ids = tuple(item.artifact_id for item in value)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("provider spend stage outputs must not repeat")
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise ValueError("provider spend stage outputs must be sorted")
        return value

    @model_validator(mode="after")
    def _require_complete_reconciliation(self) -> ProviderSpendLedger:
        """Bind totals, ceiling, inputs, and recovery posture to the complete entry set."""
        if {item.component for item in self.entries} != set(ProviderSpendComponent):
            raise ValueError("provider spend ledger must represent every provider component")
        pairs_with_zero = {
            (item.component, item.billing_source)
            for item in self.entries
            if item.status == ProviderSpendStatus.NOT_INCURRED
        }
        pairs_with_spend = {
            (item.component, item.billing_source)
            for item in self.entries
            if item.status != ProviderSpendStatus.NOT_INCURRED
        }
        if pairs_with_zero & pairs_with_spend:
            raise ValueError(
                "not-incurred spend cannot coexist with charged spend for one billing source"
            )
        expected = sum((item.amount_usd for item in self.entries), start=USD_ZERO)
        if self.total_usd != expected:
            raise ValueError("provider spend total differs from its component entries")
        if self.total_usd > self.ceiling_usd:
            raise ValueError("provider spend total exceeds the accepted attempt ceiling")
        expected_inputs = sorted_unique_inputs(
            *(item.evidence for item in self.entries if item.evidence is not None),
            *self.stage_outputs,
        )
        if self.inputs != expected_inputs:
            raise ValueError("provider spend ledger inputs differ from entry evidence")
        if self.outcome == "completed" and self.restart != "completed_stage_bundle":
            raise ValueError("completed spend evidence must restart from a completed-stage bundle")
        if self.outcome == "failed_closed" and self.restart != "blocked_ambiguous_operation":
            raise ValueError("ambiguous provider spend must remain fail closed")
        return self


def spend_entry_from_economics(
    *,
    operation_id: str,
    component: ProviderSpendComponent,
    billing_source: BillingSource,
    economics: OperationEconomics,
    operation_count: int = 1,
    evidence: ArtifactInput | None = None,
) -> ProviderSpendEntry:
    """Convert one response economics record to explicit observed or local pricing evidence.

    Args:
        operation_id: Alias-free stable operation identity.
        component: Provider-backed component that produced the response.
        billing_source: Credential owner responsible for the provider operation.
        economics: Reconciled provider operation economics.
        operation_count: Provider dispatches aggregated into the economics record.
        evidence: Optional immutable artifact carrying the operation result.

    Returns:
        One component entry retaining usage and cost provenance.

    Raises:
        ValueError: Cost or usage is absent after a completed provider operation.
    """
    cost = economics.cost_usd
    if cost is None:
        raise ValueError("completed provider economics must expose a reconciled cost")
    status = ProviderSpendStatus.OBSERVED
    if cost.provenance == "estimated":
        status = (
            ProviderSpendStatus.LOCALLY_PRICED
            if economics.usage is not None
            else ProviderSpendStatus.RESERVED
        )
    return ProviderSpendEntry(
        operation_id=operation_id,
        component=component,
        billing_source=billing_source,
        status=status,
        operation_count=operation_count,
        amount_usd=reserve_usd(cost.value),
        usage=economics.usage,
        evidence=evidence,
    )


def persist_provider_spend_ledger(
    store: ArtifactStore,
    *,
    project_id: str,
    stage: ProjectStage,
    attempt_id: str,
    attempt_authority_sha256: str,
    ceiling_usd: Decimal,
    entries: tuple[ProviderSpendEntry, ...],
    stage_outputs: tuple[ArtifactInput, ...] = (),
    outcome: Literal["completed", "failed_closed"],
    created_at: datetime,
    code_revision: str,
) -> tuple[ProviderSpendLedger, ArtifactInput]:
    """Persist or verify one complete versioned ledger and return its exact pointer.

    Args:
        store: Project-local immutable artifact store.
        project_id: Project that owns this attempt.
        stage: Latest durable stage represented by the entries.
        attempt_id: Caller-owned stable attempt identity.
        attempt_authority_sha256: Durable external authority digest bound to this attempt.
        ceiling_usd: Accepted finite ceiling shared by every component.
        entries: Complete canonical component entries.
        stage_outputs: Exact durable outputs whose completed selection this ledger accompanies.
        outcome: Whether the stage completed or stopped at ambiguous paid work.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Stored ledger and exact manifest pointer.
    """
    ordered = tuple(sorted(entries, key=lambda item: item.operation_id))
    outputs = sorted_unique_inputs(*stage_outputs)
    inputs = sorted_unique_inputs(
        *(item.evidence for item in ordered if item.evidence is not None),
        *outputs,
    )
    material = {
        "version": "provider-spend-ledger-v1",
        "project_id": project_id,
        "stage": stage.value,
        "attempt_id": attempt_id,
        "attempt_authority_sha256": attempt_authority_sha256,
        "ceiling_usd": str(ceiling_usd),
        "stage_outputs": [item.model_dump(mode="json") for item in outputs],
        "entries": [item.model_dump(mode="json") for item in ordered],
        "outcome": outcome,
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "code_revision": code_revision,
    }
    ledger = ProviderSpendLedger(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        ledger_id=stable_id("provider-spend-ledger", material),
        project_id=project_id,
        stage=stage,
        attempt_id=attempt_id,
        attempt_authority_sha256=attempt_authority_sha256,
        ceiling_usd=ceiling_usd,
        stage_outputs=outputs,
        entries=ordered,
        total_usd=sum((item.amount_usd for item in ordered), start=USD_ZERO),
        outcome=outcome,
        restart=(
            "completed_stage_bundle" if outcome == "completed" else "blocked_ambiguous_operation"
        ),
    )
    stored, manifest = store.write_or_replay(
        artifact_id=ledger.ledger_id,
        artifact_type=_LEDGER_ARTIFACT_TYPE,
        envelope=ledger,
        envelope_path=_LEDGER_FILE,
        envelope_type=ProviderSpendLedger,
        files={_LEDGER_FILE: canonical_json_bytes(ledger)},
    )
    return stored, artifact_input(manifest)


def load_provider_spend_ledger(
    store: ArtifactStore,
    pointer: ArtifactInput,
) -> ProviderSpendLedger:
    """Load one exact immutable provider-spend ledger.

    Args:
        store: Project-local artifact store.
        pointer: Exact selected ledger manifest identity.

    Returns:
        Verified typed ledger.

    Raises:
        ValueError: The pointer, type, identity, or payload differs.
    """
    stored = store.read(pointer.artifact_id)
    if stored.manifest.artifact_type != _LEDGER_ARTIFACT_TYPE:
        raise ValueError("provider spend pointer names the wrong artifact type")
    if artifact_input(stored.manifest) != pointer:
        raise ValueError("provider spend manifest digest changed")
    value = ProviderSpendLedger.model_validate_json(
        store.read_bytes(pointer.artifact_id, _LEDGER_FILE)
    )
    if value.ledger_id != pointer.artifact_id:
        raise ValueError("provider spend ledger identity differs from its artifact")
    return value


def not_incurred_entry(
    component: ProviderSpendComponent,
    billing_source: BillingSource,
) -> ProviderSpendEntry:
    """Return the canonical explicit zero-call entry for one component and payer.

    Args:
        component: Provider component with no operation in this ledger.
        billing_source: Credential owner whose component operation was not incurred.

    Returns:
        Stable not-incurred entry.
    """
    return ProviderSpendEntry(
        operation_id=stable_id(
            "provider-spend-operation",
            {"component": component.value, "billing_source": billing_source.value},
        ),
        component=component,
        billing_source=billing_source,
        status=ProviderSpendStatus.NOT_INCURRED,
        operation_count=0,
        amount_usd=USD_ZERO,
    )

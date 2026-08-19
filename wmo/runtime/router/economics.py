"""Alias-free provider spend contracts for online routed completions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from wmo.common.core.artifacts import ContractModel
from wmo.common.models import (
    BillingSource,
    NumericMeasurement,
    OperationEconomics,
    Usage,
    combine_economics,
)


class RoutedProviderComponent(StrEnum):
    """Provider-backed component of one online routed interaction."""

    ROUTER_EMBEDDING = "router_embedding"
    SELECTED_CANDIDATE = "selected_candidate"


class RoutedSpendDisposition(StrEnum):
    """Evidence disposition for one possible provider operation."""

    RESERVED = "reserved"
    OBSERVED = "observed"
    LOCALLY_PRICED = "locally_priced"
    RESERVED_AMBIGUOUS = "reserved_ambiguous"
    DEFINITELY_NOT_INCURRED = "definitely_not_incurred"


class BillingSourceEconomics(ContractModel):
    """Alias-free operation economics attributed to one credential owner."""

    billing_source: BillingSource
    economics: OperationEconomics


class RoutedProviderOperation(ContractModel):
    """One alias-free provider operation or proven non-operation."""

    operation_id: str = Field(pattern=r"^routed-operation-[0-9a-f]{20}$")
    operation_ordinal: int = Field(gt=0)
    component: RoutedProviderComponent
    billing_source: BillingSource
    disposition: RoutedSpendDisposition
    operation_count: int = Field(ge=0, le=1)
    economics: OperationEconomics

    @model_validator(mode="after")
    def _require_disposition_accounting(self) -> RoutedProviderOperation:
        """Require call counts and economics to agree with the disposition."""
        if self.disposition == RoutedSpendDisposition.DEFINITELY_NOT_INCURRED:
            if self.operation_count != 0 or self.economics != zero_operation_economics():
                raise ValueError(
                    "not-incurred provider operations must carry exact zero accounting"
                )
        elif self.operation_count != 1:
            raise ValueError("provider reservations and incurred operations must count once")
        return self


class RoutedSpendLedger(ContractModel):
    """Cumulative alias-free provider accounting for one routed interaction."""

    operations: tuple[RoutedProviderOperation, ...]
    operation_count: int = Field(ge=0)
    by_billing_source: tuple[BillingSourceEconomics, ...]
    total: OperationEconomics

    @model_validator(mode="after")
    def _require_reconciliation(self) -> RoutedSpendLedger:
        """Require exact operation order, payer totals, and overall total."""
        ordinals = tuple(item.operation_ordinal for item in self.operations)
        if ordinals != tuple(range(1, len(self.operations) + 1)):
            raise ValueError("routed provider operation ordinals must be contiguous")
        if len({item.operation_id for item in self.operations}) != len(self.operations):
            raise ValueError("routed provider operation IDs must be unique")
        if any(item.disposition == RoutedSpendDisposition.RESERVED for item in self.operations):
            raise ValueError("settled routed spend ledgers cannot retain live reservations")
        if self.operation_count != sum(item.operation_count for item in self.operations):
            raise ValueError("routed provider operation count differs from its entries")
        expected_by_source = economics_by_billing_source(self.operations)
        if self.by_billing_source != expected_by_source:
            raise ValueError("routed provider payer totals differ from operation economics")
        expected_total = combine_economics(tuple(item.economics for item in self.operations))
        if self.total != expected_total:
            raise ValueError("routed provider total differs from operation economics")
        return self


class RoutedCompletionEconomics(ContractModel):
    """Cumulative alias-free economics for a completed routed interaction."""

    operations: tuple[RoutedProviderOperation, ...]
    operation_count: int = Field(ge=0)
    router_embedding: BillingSourceEconomics
    selected_candidate: BillingSourceEconomics
    by_billing_source: tuple[BillingSourceEconomics, ...]
    total: OperationEconomics

    @model_validator(mode="after")
    def _require_complete_total(self) -> RoutedCompletionEconomics:
        """Require exact component, payer, call-count, and overall reconciliation."""
        ledger = routed_spend_ledger(self.operations)
        if (
            self.operation_count != ledger.operation_count
            or self.by_billing_source != ledger.by_billing_source
            or self.total != ledger.total
        ):
            raise ValueError("routed completion totals differ from provider operations")
        expected_embedding = _component_economics(
            self.operations, RoutedProviderComponent.ROUTER_EMBEDDING
        )
        expected_candidate = _component_economics(
            self.operations, RoutedProviderComponent.SELECTED_CANDIDATE
        )
        if self.router_embedding != expected_embedding:
            raise ValueError("router embedding total differs from provider operations")
        if self.selected_candidate != expected_candidate:
            raise ValueError("selected candidate total differs from provider operations")
        return self


def zero_operation_economics() -> OperationEconomics:
    """Return exact evidence that no provider operation was incurred."""
    return OperationEconomics(
        usage=Usage(input_tokens=0, output_tokens=0),
        cost_usd=NumericMeasurement(value=0.0, provenance="estimated"),
    )


def routed_spend_ledger(
    operations: tuple[RoutedProviderOperation, ...],
) -> RoutedSpendLedger:
    """Build one reconciled cumulative spend ledger.

    Args:
        operations: Settled provider operations in interaction order.

    Returns:
        Exact aggregate accounting by source and overall.
    """
    return RoutedSpendLedger(
        operations=operations,
        operation_count=sum(item.operation_count for item in operations),
        by_billing_source=economics_by_billing_source(operations),
        total=combine_economics(tuple(item.economics for item in operations)),
    )


def routed_completion_economics(
    operations: tuple[RoutedProviderOperation, ...],
) -> RoutedCompletionEconomics:
    """Build completed accounting with component convenience totals.

    Args:
        operations: Settled embedding and candidate operations in interaction order.

    Returns:
        Reconciled completion economics without model aliases or identities.
    """
    ledger = routed_spend_ledger(operations)
    return RoutedCompletionEconomics(
        operations=operations,
        operation_count=ledger.operation_count,
        router_embedding=_component_economics(operations, RoutedProviderComponent.ROUTER_EMBEDDING),
        selected_candidate=_component_economics(
            operations, RoutedProviderComponent.SELECTED_CANDIDATE
        ),
        by_billing_source=ledger.by_billing_source,
        total=ledger.total,
    )


def economics_by_billing_source(
    operations: tuple[RoutedProviderOperation, ...],
) -> tuple[BillingSourceEconomics, ...]:
    """Aggregate provider operations by immutable credential owner.

    Args:
        operations: Alias-free provider operations.

    Returns:
        Sorted exact totals for every represented billing source.
    """
    sources = sorted({item.billing_source for item in operations}, key=lambda item: item.value)
    return tuple(
        BillingSourceEconomics(
            billing_source=source,
            economics=combine_economics(
                tuple(item.economics for item in operations if item.billing_source == source)
            ),
        )
        for source in sources
    )


def _component_economics(
    operations: tuple[RoutedProviderOperation, ...],
    component: RoutedProviderComponent,
) -> BillingSourceEconomics:
    """Aggregate one required component while proving its billing source is stable."""
    matches = tuple(item for item in operations if item.component == component)
    if not matches:
        raise ValueError(f"routed completion omits {component.value} accounting")
    sources = {item.billing_source for item in matches}
    if len(sources) != 1:
        raise ValueError(f"routed completion changes {component.value} billing source")
    return BillingSourceEconomics(
        billing_source=next(iter(sources)),
        economics=combine_economics(tuple(item.economics for item in matches)),
    )

"""Immutable cost provenance for budgeted harness search components."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.providers.base import ProviderConfig
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetAccountBinding,
    BudgetLedgerAuthority,
    BudgetPolicy,
    ProviderCostMeter,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    bind_budget_account,
    bind_timed_resource_account,
    open_shared_spend_ledger,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class SearchComponentRole(StrEnum):
    """Stable roles that can independently spend during one harness search."""

    PROPOSER = "proposer"
    SCORER = "scorer"
    HOLDOUT_SCORER = "holdout_scorer"


class ProviderCostBinding(BaseModel):
    """Path-free identity of one component's exact paid provider account."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_configuration_id: str = Field(min_length=1, max_length=1_024)
    provider_config: ProviderConfig
    account: BudgetAccountBinding


class TimedResourceCostBinding(BaseModel):
    """Path-free identity of one component's exact timed resource account."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_type: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )
    resource_class_digest: str = Field(pattern=_DIGEST_PATTERN)
    account: BudgetAccountBinding


class SearchComponentCostBinding(BaseModel):
    """Complete provider and resource attribution for one search component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: SearchComponentRole
    configuration_id: str = Field(min_length=1, max_length=1_024)
    scope_category: str = Field(min_length=1)
    providers: tuple[ProviderCostBinding, ...] = Field(min_length=1)
    timed_resources: tuple[TimedResourceCostBinding, ...] = ()

    @model_validator(mode="after")
    def _bind_component_accounts(self) -> Self:
        if self.scope_category != self.scope_category.strip():
            raise ValueError("search component scope category must be canonical")
        accounts: list[BudgetAccountBinding] = []
        for provider in self.providers:
            if provider.component_configuration_id != self.configuration_id:
                raise ValueError("provider binding component configuration differs")
            accounts.append(provider.account)
        accounts.extend(resource.account for resource in self.timed_resources)
        if any(account.scope.category != self.scope_category for account in accounts):
            raise ValueError("search component account scope category differs from its binding")
        identities = [_account_identity(account) for account in accounts]
        if len(identities) != len(set(identities)):
            raise ValueError("search component cost binding contains a duplicate account")
        return self


class SearchCostBinding(BaseModel):
    """Frozen public cost contract for one budgeted harness search.

    The binding deliberately contains no ledger path. It freezes the complete hard policy,
    durable ledger identity, attribution phase and run, component identities, provider routes,
    and any timed resource classes before a paid component is called.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wmh.search-cost-binding.v1"] = "wmh.search-cost-binding.v1"
    declared_hard_limit_nano_usd: int = Field(gt=0)
    policy: BudgetPolicy
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    phase: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    proposer: SearchComponentCostBinding
    scorer: SearchComponentCostBinding
    holdout_scorer: SearchComponentCostBinding | None = None

    @field_validator("policy", mode="before")
    @classmethod
    def _detach_policy(cls, value: object) -> object:
        if isinstance(value, BudgetPolicy):
            return value.model_dump()
        return value

    @field_validator("proposer", "scorer", "holdout_scorer", mode="before")
    @classmethod
    def _detach_component(cls, value: object) -> object:
        if isinstance(value, SearchComponentCostBinding):
            return value.model_dump()
        return value

    @model_validator(mode="after")
    def _validate_cost_contract(self) -> Self:
        if self.declared_hard_limit_nano_usd != self.policy.hard_limit_nano_usd:
            raise ValueError("declared hard limit differs from the frozen budget policy")
        if self.phase != self.phase.strip() or self.run_id != self.run_id.strip():
            raise ValueError("search cost phase and run_id must be canonical")
        if self.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError("search cost phase is absent from the frozen budget policy")
        expected_roles = (
            ("proposer", self.proposer, SearchComponentRole.PROPOSER),
            ("scorer", self.scorer, SearchComponentRole.SCORER),
            ("holdout scorer", self.holdout_scorer, SearchComponentRole.HOLDOUT_SCORER),
        )
        for label, component, expected_role in expected_roles:
            if component is not None and component.role is not expected_role:
                raise ValueError(f"{label} binding has the wrong role")
        components = [self.proposer, self.scorer]
        if self.holdout_scorer is not None:
            components.append(self.holdout_scorer)
        categories = [component.scope_category for component in components]
        if len(categories) != len(set(categories)):
            raise ValueError("search component scope categories must be distinct")

        accounts: list[BudgetAccountBinding] = []
        for component in components:
            for provider in component.providers:
                self._validate_account(provider.account)
                meter = self.policy.meters.get(provider.account.meter_id)
                if not isinstance(meter, ProviderCostMeter):
                    raise ValueError("search provider account must name a provider token meter")
                if meter.provider_config != provider.provider_config:
                    raise ValueError("search provider config differs from its frozen meter")
                accounts.append(provider.account)
            for resource in component.timed_resources:
                self._validate_account(resource.account)
                meter = self.policy.meters.get(resource.account.meter_id)
                if not isinstance(meter, TimedResourceCostMeter):
                    raise ValueError("search resource account must name a timed resource meter")
                if meter.resource_type != resource.resource_type:
                    raise ValueError("search resource type differs from its frozen meter")
                if meter.resource_class_digest != resource.resource_class_digest:
                    raise ValueError("search resource class differs from its frozen meter")
                accounts.append(resource.account)
        identities = [_account_identity(account) for account in accounts]
        if len(identities) != len(set(identities)):
            raise ValueError("search cost binding assigns one account more than once")
        return self

    def _validate_account(self, account: BudgetAccountBinding) -> None:
        if account.policy_digest != self.policy.policy_digest:
            raise ValueError("search account policy digest differs from the frozen policy")
        if account.ledger_identity != self.ledger_identity:
            raise ValueError("search account ledger identity differs from the frozen authority")
        if account.scope.phase != self.phase:
            raise ValueError("search account phase differs from the frozen phase")
        if account.scope.run_id != self.run_id:
            raise ValueError("search account run_id differs from the frozen run")

    @property
    def digest(self) -> str:
        """Return the canonical path-free identity retained by search checkpoints."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class SearchCostRuntime(BaseModel):
    """Host-private resolver for one binding and its bootstrapped ledger authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    authority: BudgetLedgerAuthority = Field(exclude=True, repr=False)
    binding: SearchCostBinding

    @field_validator("authority", mode="before")
    @classmethod
    def _detach_authority(cls, value: object) -> object:
        if isinstance(value, BudgetLedgerAuthority):
            return value.model_dump()
        return value

    @field_validator("binding", mode="before")
    @classmethod
    def _detach_binding(cls, value: object) -> object:
        if isinstance(value, SearchCostBinding):
            return value.model_dump()
        return value

    @model_validator(mode="after")
    def _audit_authority(self) -> Self:
        _audit_runtime_state(self.authority, self.binding)
        return self

    def provider_account(self, binding: ProviderCostBinding) -> BudgetAccount:
        """Mint one exact provider account already present in the frozen binding."""
        authority, frozen_binding = self._validated_state()
        validated = ProviderCostBinding.model_validate(binding.model_dump())
        if validated not in _provider_bindings(frozen_binding):
            raise ValueError("provider account is not present in the frozen search cost binding")
        return authority.provider_account(
            scope=validated.account.scope,
            meter_id=validated.account.meter_id,
        )

    def timed_resource_account(
        self,
        binding: TimedResourceCostBinding,
    ) -> TimedResourceBudgetAccount:
        """Mint one exact resource account already present in the frozen binding."""
        authority, frozen_binding = self._validated_state()
        validated = TimedResourceCostBinding.model_validate(binding.model_dump())
        if validated not in _resource_bindings(frozen_binding):
            raise ValueError("resource account is not present in the frozen search cost binding")
        return authority.timed_resource_account(
            scope=validated.account.scope,
            meter_id=validated.account.meter_id,
        )

    def _validated_state(self) -> tuple[BudgetLedgerAuthority, SearchCostBinding]:
        """Revalidate nested mutable policy containers before minting an account."""
        authority = BudgetLedgerAuthority.model_validate(self.authority.model_dump())
        binding = SearchCostBinding.model_validate(self.binding.model_dump())
        _audit_runtime_state(authority, binding)
        return authority, binding


def validate_search_cost_components(
    binding: SearchCostBinding,
    *,
    proposer: object,
    scorer: object,
    holdout_scorer: object | None = None,
) -> None:
    """Require live components to expose the exact binding before any paid call."""
    frozen = SearchCostBinding.model_validate(binding.model_dump())
    _validate_component_binding(proposer, frozen.proposer, label="proposer")
    _validate_component_binding(scorer, frozen.scorer, label="scorer")
    if (holdout_scorer is None) != (frozen.holdout_scorer is None):
        raise ValueError("search holdout scorer and cost binding must be present together")
    if holdout_scorer is not None:
        assert frozen.holdout_scorer is not None
        _validate_component_binding(
            holdout_scorer,
            frozen.holdout_scorer,
            label="holdout_scorer",
        )


def _validate_component_binding(
    component: object,
    expected: SearchComponentCostBinding,
    *,
    label: str,
) -> None:
    configuration_id = getattr(component, "configuration_id", None)
    if configuration_id != expected.configuration_id:
        raise ValueError(f"{label} component configuration_id differs from its cost binding")
    actual = getattr(component, "search_cost_binding", None)
    if not isinstance(actual, SearchComponentCostBinding):
        raise ValueError(
            f"budgeted search requires {label}.search_cost_binding to expose its exact accounts"
        )
    validated = SearchComponentCostBinding.model_validate(actual.model_dump())
    if validated != expected:
        raise ValueError(f"{label} cost binding differs from the frozen search cost binding")


def _audit_runtime_state(
    authority: BudgetLedgerAuthority,
    binding: SearchCostBinding,
) -> None:
    if authority.policy != binding.policy:
        raise ValueError("search budget authority policy differs from its cost binding")
    if authority.ledger_identity != binding.ledger_identity:
        raise ValueError("search budget authority ledger identity differs from its binding")
    open_shared_spend_ledger(
        authority.ledger_path,
        authority.policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    for provider in _provider_bindings(binding):
        account = authority.provider_account(
            scope=provider.account.scope,
            meter_id=provider.account.meter_id,
        )
        if bind_budget_account(account) != provider.account:
            raise ValueError("search provider account differs from its budget authority")
    for resource in _resource_bindings(binding):
        account = authority.timed_resource_account(
            scope=resource.account.scope,
            meter_id=resource.account.meter_id,
        )
        if bind_timed_resource_account(account) != resource.account:
            raise ValueError("search resource account differs from its budget authority")


def _provider_bindings(binding: SearchCostBinding) -> tuple[ProviderCostBinding, ...]:
    components = [binding.proposer, binding.scorer]
    if binding.holdout_scorer is not None:
        components.append(binding.holdout_scorer)
    return tuple(provider for component in components for provider in component.providers)


def _resource_bindings(binding: SearchCostBinding) -> tuple[TimedResourceCostBinding, ...]:
    components = [binding.proposer, binding.scorer]
    if binding.holdout_scorer is not None:
        components.append(binding.holdout_scorer)
    return tuple(resource for component in components for resource in component.timed_resources)


def _account_identity(account: BudgetAccountBinding) -> tuple[str, ...]:
    scope = account.scope
    return (
        account.policy_digest,
        account.ledger_identity,
        scope.phase,
        scope.category,
        scope.run_id,
        scope.lane or "",
        scope.arm or "",
        account.meter_id,
    )

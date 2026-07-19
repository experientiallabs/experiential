"""Tests for immutable search cost bindings and trusted ledger resolution."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentRole,
    SearchCostBinding,
    SearchCostRuntime,
    TimedResourceCostBinding,
    validate_search_cost_components,
)
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetIntegrityError,
    BudgetLedgerAuthority,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    TimedResourceCostMeter,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    ExternalDispatchRateBinding,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _provider_config(model: str) -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region="test-region")


def _policy(label: str) -> BudgetPolicy:
    return BudgetPolicy(
        study_id=f"study-{label}",
        manifest_digest=_digest(f"manifest-{label}"),
        hard_limit_nano_usd=15_000_000_000,
        phase_limits_nano_usd={"search": 15_000_000_000},
        meters={
            "proposer-provider": synthetic_provider_cost_meter(
                provider_config=(proposer_config := _provider_config("proposer-model")),
                provenance=synthetic_tariff_provenance(proposer_config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=2,
            ),
            "scorer-provider": synthetic_provider_cost_meter(
                provider_config=(scorer_config := _provider_config("scorer-model")),
                provenance=synthetic_tariff_provenance(scorer_config),
                input_nano_usd_per_token=2,
                output_nano_usd_per_token=4,
            ),
            "proposer-project": TimedResourceCostMeter(
                resource_type="proposer_project",
                resource_class_digest=_digest("proposer-project-class"),
                nano_usd_per_second=3,
                max_billing_seconds=600,
            ),
            "task-environment": TimedResourceCostMeter(
                resource_type="task_environment",
                resource_class_digest=_digest("task-environment-class"),
                nano_usd_per_second=5,
                max_billing_seconds=600,
            ),
        },
    )


def _provider_binding(
    authority: BudgetLedgerAuthority,
    *,
    category: str,
    meter_id: str,
    configuration_id: str,
) -> ProviderCostBinding:
    account = authority.provider_account(
        scope=BudgetScope(
            phase="search",
            category=category,
            run_id="optimizer-run",
        ),
        meter_id=meter_id,
    )
    meter = authority.policy.meters[meter_id]
    assert isinstance(meter, ProviderCostMeter)
    return ProviderCostBinding(
        component_configuration_id=configuration_id,
        provider_config=meter.provider_config,
        account=bind_budget_account(account),
    )


def _resource_binding(
    authority: BudgetLedgerAuthority,
    *,
    category: str,
    meter_id: str,
    configuration_id: str,
) -> TimedResourceCostBinding:
    account = authority.timed_resource_account(
        scope=BudgetScope(
            phase="search",
            category=category,
            run_id="optimizer-run",
        ),
        meter_id=meter_id,
    )
    meter = authority.policy.meters[meter_id]
    assert isinstance(meter, TimedResourceCostMeter)
    return TimedResourceCostBinding(
        component_configuration_id=configuration_id,
        resource_type=meter.resource_type,
        resource_class_digest=meter.resource_class_digest,
        account=bind_timed_resource_account(account),
    )


def _binding(
    authority: BudgetLedgerAuthority,
    *,
    proposer_configuration_id: str = "proposer-v1",
    scorer_configuration_id: str = "scorer-v1",
) -> SearchCostBinding:
    return SearchCostBinding(
        declared_hard_limit_nano_usd=authority.policy.hard_limit_nano_usd,
        policy=authority.policy,
        ledger_identity=authority.ledger_identity,
        phase="search",
        run_id="optimizer-run",
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id=proposer_configuration_id,
            scope_category="proposer",
            providers=(
                _provider_binding(
                    authority,
                    category="proposer",
                    meter_id="proposer-provider",
                    configuration_id=proposer_configuration_id,
                ),
            ),
            timed_resources=(
                _resource_binding(
                    authority,
                    category="proposer",
                    meter_id="proposer-project",
                    configuration_id=proposer_configuration_id,
                ),
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id=scorer_configuration_id,
            scope_category="scorer",
            providers=(
                _provider_binding(
                    authority,
                    category="scorer",
                    meter_id="scorer-provider",
                    configuration_id=scorer_configuration_id,
                ),
            ),
            timed_resources=(
                _resource_binding(
                    authority,
                    category="scorer",
                    meter_id="task-environment",
                    configuration_id=scorer_configuration_id,
                ),
            ),
        ),
    )


def _authority(tmp_path: Path, label: str) -> BudgetLedgerAuthority:
    return bootstrap_budget_ledger(tmp_path / f"{label}.sqlite3", _policy(label))


def _revalidate(binding: SearchCostBinding, **updates: object) -> SearchCostBinding:
    payload = binding.model_dump(mode="json")
    payload.update(updates)
    return SearchCostBinding.model_validate(payload)


def test_search_cost_binding_is_complete_stable_and_path_free(tmp_path: Path) -> None:
    authority = _authority(tmp_path, "path-free")
    binding = _binding(authority)

    serialized = binding.model_dump_json()

    assert binding.digest == SearchCostBinding.model_validate_json(serialized).digest
    assert binding.declared_hard_limit_nano_usd == authority.policy.hard_limit_nano_usd
    assert str(authority.ledger_path) not in serialized
    assert "ledger_path" not in serialized


def test_search_cost_binding_rejects_declared_hard_limit_drift(tmp_path: Path) -> None:
    binding = _binding(_authority(tmp_path, "limit-drift"))

    with pytest.raises(ValidationError, match="declared hard limit differs"):
        _revalidate(binding, declared_hard_limit_nano_usd=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_digest", _digest("other-policy"), "policy digest"),
        ("ledger_identity", _digest("other-ledger"), "ledger identity"),
    ],
)
def test_search_cost_binding_rejects_account_authority_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    binding = _binding(_authority(tmp_path, f"authority-{field}"))
    payload = binding.model_dump(mode="json")
    payload["scorer"]["providers"][0]["account"][field] = value

    with pytest.raises(ValidationError, match=message):
        SearchCostBinding.model_validate(payload)


@pytest.mark.parametrize(
    ("scope_field", "value", "message"),
    [
        ("phase", "confirmation", "phase"),
        ("run_id", "other-run", "run_id"),
        ("category", "proposer", "scope category"),
    ],
)
def test_search_cost_binding_rejects_account_scope_drift(
    tmp_path: Path,
    scope_field: str,
    value: str,
    message: str,
) -> None:
    binding = _binding(_authority(tmp_path, f"scope-{scope_field}"))
    payload = binding.model_dump(mode="json")
    payload["scorer"]["providers"][0]["account"]["scope"][scope_field] = value

    with pytest.raises(ValidationError, match=message):
        SearchCostBinding.model_validate(payload)


def test_search_cost_binding_rejects_provider_route_or_meter_drift(tmp_path: Path) -> None:
    binding = _binding(_authority(tmp_path, "provider-drift"))
    payload = binding.model_dump(mode="json")
    payload["scorer"]["providers"][0]["provider_config"]["model"] = "other-model"
    with pytest.raises(ValidationError, match="provider config differs"):
        SearchCostBinding.model_validate(payload)

    payload = binding.model_dump(mode="json")
    payload["scorer"]["providers"][0]["account"]["meter_id"] = "proposer-project"
    with pytest.raises(ValidationError, match="provider token meter"):
        SearchCostBinding.model_validate(payload)


def test_search_cost_binding_rejects_resource_role_or_class_drift(tmp_path: Path) -> None:
    binding = _binding(_authority(tmp_path, "resource-drift"))
    payload = binding.model_dump(mode="json")
    payload["scorer"]["timed_resources"][0]["resource_type"] = "agent_runner"
    with pytest.raises(ValidationError, match="resource type differs"):
        SearchCostBinding.model_validate(payload)

    payload = binding.model_dump(mode="json")
    payload["scorer"]["timed_resources"][0]["resource_class_digest"] = _digest("different-class")
    with pytest.raises(ValidationError, match="resource class differs"):
        SearchCostBinding.model_validate(payload)

    payload = binding.model_dump(mode="json")
    payload["scorer"]["timed_resources"][0]["component_configuration_id"] = "other-scorer"
    with pytest.raises(ValidationError, match="resource binding component configuration differs"):
        SearchCostBinding.model_validate(payload)


def test_search_cost_binding_rejects_duplicate_accounts_or_component_roles(
    tmp_path: Path,
) -> None:
    binding = _binding(_authority(tmp_path, "duplicates"))
    payload = binding.model_dump(mode="json")
    payload["scorer"]["providers"][0] = payload["proposer"]["providers"][0]
    payload["scorer"]["providers"][0]["component_configuration_id"] = "scorer-v1"
    with pytest.raises(ValidationError, match="scope category"):
        SearchCostBinding.model_validate(payload)

    payload = binding.model_dump(mode="json")
    payload["scorer"]["role"] = SearchComponentRole.PROPOSER.value
    with pytest.raises(ValidationError, match="scorer binding has the wrong role"):
        SearchCostBinding.model_validate(payload)


def test_search_cost_runtime_audits_authority_and_mints_only_bound_accounts(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, "runtime")
    binding = _binding(authority)
    runtime = SearchCostRuntime(authority=authority, binding=binding)

    proposer_runtime = runtime.for_component(SearchComponentRole.PROPOSER)
    scorer_runtime = runtime.for_component(SearchComponentRole.SCORER)

    proposer_account = proposer_runtime.provider_account(binding.proposer.providers[0])
    task_account = scorer_runtime.timed_resource_account(binding.scorer.timed_resources[0])
    serialized_runtime = runtime.model_dump_json()
    serialized_component_runtime = proposer_runtime.model_dump_json()

    assert proposer_account.ledger_path == authority.ledger_path
    assert proposer_account.ledger_identity == binding.ledger_identity
    assert proposer_account.scope == binding.proposer.providers[0].account.scope
    assert task_account.ledger_path == authority.ledger_path
    assert task_account.meter_id == "task-environment"
    assert str(authority.ledger_path) not in serialized_runtime
    assert "ledger_path" not in serialized_runtime
    assert str(authority.ledger_path) not in serialized_component_runtime
    assert "ledger_path" not in serialized_component_runtime

    provider_binding = binding.scorer.providers[0]
    unbound = provider_binding.model_copy(
        update={
            "account": provider_binding.account.model_copy(
                update={"scope": provider_binding.account.scope.model_copy(update={"lane": "x"})}
            )
        }
    )
    with pytest.raises(ValueError, match="not present in the scorer cost binding"):
        scorer_runtime.provider_account(unbound)


def test_component_runtime_rejects_cross_role_provider_and_resource_swizzles(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, "runtime-role-scope")
    binding = _binding(authority)
    proposer_runtime = SearchCostRuntime(
        authority=authority,
        binding=binding,
    ).for_component(SearchComponentRole.PROPOSER)

    with pytest.raises(ValueError, match="provider account is not present in the proposer"):
        proposer_runtime.provider_account(binding.scorer.providers[0])
    with pytest.raises(ValueError, match="resource account is not present in the proposer"):
        proposer_runtime.timed_resource_account(binding.scorer.timed_resources[0])


def test_component_runtime_rejects_an_absent_holdout_role(tmp_path: Path) -> None:
    authority = _authority(tmp_path, "runtime-no-holdout")

    with pytest.raises(ValueError, match="holdout_scorer has no frozen search cost binding"):
        SearchCostRuntime(
            authority=authority,
            binding=_binding(authority),
        ).for_component(SearchComponentRole.HOLDOUT_SCORER)


def test_search_cost_runtime_rejects_same_policy_on_a_different_ledger(tmp_path: Path) -> None:
    authority = _authority(tmp_path, "runtime-first")
    binding = _binding(authority)
    second = bootstrap_budget_ledger(tmp_path / "runtime-second.sqlite3", authority.policy)

    with pytest.raises(ValidationError, match="ledger identity differs"):
        SearchCostRuntime(authority=second, binding=binding)


def test_component_runtime_reaudits_the_ledger_before_each_account_mint(tmp_path: Path) -> None:
    authority = _authority(tmp_path, "runtime-reaudit")
    binding = _binding(authority)
    component_runtime = SearchCostRuntime(
        authority=authority,
        binding=binding,
    ).for_component(SearchComponentRole.PROPOSER)

    with sqlite3.connect(authority.ledger_path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version"):
        component_runtime.provider_account(binding.proposer.providers[0])


def test_binding_and_runtime_detach_mutable_source_models(tmp_path: Path) -> None:
    authority = _authority(tmp_path, "detached")
    binding = _binding(authority)
    runtime = SearchCostRuntime(authority=authority, binding=binding)

    authority.policy.meters.clear()
    binding.policy.meters.clear()

    assert runtime.authority.policy.meters
    assert runtime.binding.policy.meters
    assert runtime.for_component(SearchComponentRole.PROPOSER).provider_account(
        runtime.binding.proposer.providers[0]
    ).meter_id == ("proposer-provider")

    provider_binding = runtime.binding.proposer.providers[0]
    runtime.binding.policy.meters.clear()
    with pytest.raises((ValidationError, ValueError), match="meter"):
        runtime.for_component(SearchComponentRole.PROPOSER).provider_account(provider_binding)


class _BoundComponent:
    def __init__(
        self,
        configuration_id: str,
        binding: SearchComponentCostBinding | None,
        create_rate_binding: ExternalDispatchRateBinding | None = None,
    ) -> None:
        self.configuration_id = configuration_id
        self.create_rate_binding = create_rate_binding
        if binding is not None:
            self.search_cost_binding = binding


def test_component_validation_requires_exact_exposed_bindings(tmp_path: Path) -> None:
    binding = _binding(_authority(tmp_path, "components"))
    proposer = _BoundComponent("proposer-v1", binding.proposer)
    scorer = _BoundComponent("scorer-v1", binding.scorer)

    validate_search_cost_components(binding, proposer=proposer, scorer=scorer)

    with pytest.raises(ValueError, match="proposer.search_cost_binding"):
        validate_search_cost_components(
            binding,
            proposer=_BoundComponent("proposer-v1", None),
            scorer=scorer,
        )
    with pytest.raises(ValueError, match="scorer cost binding differs"):
        validate_search_cost_components(
            binding,
            proposer=proposer,
            scorer=_BoundComponent("scorer-v1", binding.proposer),
        )
    with pytest.raises(ValueError, match="component configuration_id differs"):
        validate_search_cost_components(
            binding,
            proposer=_BoundComponent("spoofed-id", binding.proposer),
            scorer=scorer,
        )


def test_component_validation_requires_one_shared_external_dispatch_authority(
    tmp_path: Path,
) -> None:
    first = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "first-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    second = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "second-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    binding = _revalidate(
        _binding(_authority(tmp_path, "shared-rate")),
        external_dispatch_rate_binding=first.binding.model_dump(mode="json"),
    )
    proposer = _BoundComponent(
        "proposer-v1",
        binding.proposer,
        create_rate_binding=first.binding,
    )
    scorer = _BoundComponent(
        "scorer-v1",
        binding.scorer,
        create_rate_binding=first.binding,
    )

    validate_search_cost_components(binding, proposer=proposer, scorer=scorer)

    with pytest.raises(ValueError, match="shared external dispatch authority"):
        validate_search_cost_components(
            binding,
            proposer=proposer,
            scorer=_BoundComponent(
                "scorer-v1",
                binding.scorer,
                create_rate_binding=second.binding,
            ),
        )

    without_rate = _binding(_authority(tmp_path, "missing-shared-rate"))
    with pytest.raises(ValueError, match="omits the shared external dispatch authority"):
        validate_search_cost_components(
            without_rate,
            proposer=_BoundComponent(
                "proposer-v1",
                without_rate.proposer,
                create_rate_binding=first.binding,
            ),
            scorer=_BoundComponent("scorer-v1", without_rate.scorer),
        )

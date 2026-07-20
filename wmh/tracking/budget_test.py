"""Hard-budget admission, audit, and provider-boundary metering tests."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse
from llm_waterfall.bedrock_chat import bedrock_converse_response
from llm_waterfall.types import (
    ChatChoice,
    ChatMessage,
    ChatUsage,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import ResponseUsage
from pydantic import JsonValue, ValidationError

from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    SingleDispatchProvider,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.bedrock import BedrockProvider
from wmh.providers.receipt import (
    ProviderResponseIdentity,
    build_chat_provider_receipt,
)
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_evidence_receipt,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetAccountBinding,
    BudgetBreachError,
    BudgetBreachKind,
    BudgetedProvider,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetReservation,
    BudgetScope,
    BudgetTerminalProvenance,
    ExternalSpendAuthority,
    ProviderCostMeter,
    ReservationStatus,
    SpendLedger,
    TimedResourceBudget,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceRole,
    TokenPriceCeiling,
    UnpricedProviderUsageError,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
    nano_usd_from_usd,
    open_shared_spend_ledger,
    orphaned_timed_resource_requires_reap,
    reconcile_orphaned_timed_resource,
    resolve_budget_account,
    resolve_timed_resource_account,
    validate_timed_resource_class,
)


def _policy(*, hard: int = 100, search: int = 80, final: int = 20) -> BudgetPolicy:
    provider_config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="model-1",
        region="test-region",
    )
    price = TokenPriceCeiling(
        input_nano_usd_per_token=2,
        output_nano_usd_per_token=5,
    )
    provenance = synthetic_tariff_provenance(provider_config)
    return BudgetPolicy(
        study_id="study-1",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=hard,
        phase_limits_nano_usd={"search": search, "final": final},
        meters={
            "worker": ProviderCostMeter(
                provider_config=provider_config,
                price=price,
                tariff_provenance=provenance,
                tariff_evidence_receipt=synthetic_tariff_evidence_receipt(
                    provider_config=provider_config,
                    price=price,
                    provenance=provenance,
                ),
                input_overhead_tokens=8,
            )
        },
    )


def _scope(phase: str = "search", category: str = "worker") -> BudgetScope:
    return BudgetScope(
        phase=phase,
        category=category,
        run_id="run-1",
        lane="haiku",
        arm="candidate",
    )


def _ledger_identity(path: Path, policy: BudgetPolicy) -> str:
    absolute = path.resolve()
    if absolute.exists():
        return SpendLedger(absolute, policy, allow_create=False).ledger_identity
    return bootstrap_budget_ledger(absolute, policy).ledger_identity


class _ProcessBarrier(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...


class _StringQueue(Protocol):
    def put(self, value: str) -> None: ...


def _reserve_in_process(
    path: str,
    policy_json: str,
    reservation_id: str,
    barrier: object,
    outcomes: object,
    meter_id: str = "worker",
    max_nano_usd: int = 60,
) -> None:
    process_barrier = cast("_ProcessBarrier", barrier)
    process_queue = cast("_StringQueue", outcomes)
    process_barrier.wait()
    ledger = SpendLedger(path, BudgetPolicy.model_validate_json(policy_json))
    try:
        ledger.reserve(
            _scope(),
            meter_id=meter_id,
            max_nano_usd=max_nano_usd,
            reservation_id=reservation_id,
        )
    except BudgetExceededError:
        process_queue.put("rejected")
    else:
        process_queue.put("admitted")


def _reserve_then_exit(path: str, policy_json: str) -> None:
    ledger = SpendLedger(path, BudgetPolicy.model_validate_json(policy_json))
    ledger.reserve(
        _scope(),
        meter_id="worker",
        max_nano_usd=80,
        reservation_id="orphaned-process-call",
    )
    os._exit(0)


def _bind_account_in_fresh_process(account_json: str, outcomes: object) -> None:
    process_queue = cast("_StringQueue", outcomes)
    try:
        bind_budget_account(BudgetAccount.model_validate_json(account_json))
    except BudgetIntegrityError as exc:
        process_queue.put(str(exc))
    else:
        process_queue.put("bound")


def test_currency_helpers_round_up_to_exact_nano_usd_ceiling() -> None:
    assert nano_usd_from_usd("15000") == 15_000_000_000_000
    assert TokenPriceCeiling.from_usd_per_million(
        input_usd="1.00",
        output_usd="5",
    ) == TokenPriceCeiling(
        input_nano_usd_per_token=1_000,
        output_nano_usd_per_token=5_000,
    )
    assert (
        TokenPriceCeiling.from_usd_per_million(
            input_usd="0.0001",
            output_usd="0.0001",
        ).input_nano_usd_per_token
        == 1
    )

    with pytest.raises(TypeError, match="floating-point"):
        nano_usd_from_usd(1.5)  # ty: ignore[invalid-argument-type]
    for value in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="finite"):
            nano_usd_from_usd(value)
    with pytest.raises(OverflowError, match="SQLite integer"):
        nano_usd_from_usd("9223372037")


def test_budget_inputs_reject_unknown_fields_instead_of_using_defaults() -> None:
    with pytest.raises(ValidationError, match="tariff_provenance"):
        ProviderCostMeter.model_validate(
            {
                "provider_config": ProviderConfig(
                    kind=ProviderKind.BEDROCK,
                    model="model-1",
                ).model_dump(mode="json"),
                "price": {
                    "input_nano_usd_per_token": 1,
                    "output_nano_usd_per_token": 1,
                },
            }
        )

    with pytest.raises(ValidationError, match="billing_quantum_second"):
        TimedResourceCostMeter.model_validate(
            {
                "resource_type": "sandbox",
                "resource_class_digest": "sha256:" + "b" * 64,
                "nano_usd_per_second": 1,
                "billing_quantum_second": 60,
                "max_billing_seconds": 120,
            }
        )

    with pytest.raises(ValidationError, match="unexpected_cap"):
        BudgetPolicy.model_validate({**_policy().model_dump(mode="json"), "unexpected_cap": 1})

    with pytest.raises(ValidationError, match="ledger_identity"):
        BudgetAccount.model_validate(
            {
                "ledger_path": "/tmp/unbound-budget.sqlite3",
                "policy": _policy().model_dump(mode="json"),
                "scope": _scope().model_dump(mode="json"),
                "meter_id": "worker",
            }
        )


def test_independently_initialized_ledgers_have_distinct_durable_identities(
    tmp_path: Path,
) -> None:
    policy = _policy()
    first = SpendLedger(tmp_path / "first.sqlite3", policy)
    second = SpendLedger(tmp_path / "second.sqlite3", policy)

    assert first.ledger_identity != second.ledger_identity
    assert SpendLedger(first.path, policy).ledger_identity == first.ledger_identity


def test_bootstrap_mints_bound_accounts_and_refuses_existing_paths(tmp_path: Path) -> None:
    policy = _policy().model_copy(update={"study_id": "bootstrap-study"})
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    account = authority.provider_account(scope=_scope(), meter_id="worker")

    assert account.ledger_identity == authority.ledger_identity
    assert bind_budget_account(account).ledger_identity == authority.ledger_identity
    with pytest.raises(BudgetIntegrityError, match="already exists"):
        bootstrap_budget_ledger(authority.ledger_path, policy)


def test_identity_bound_account_rejects_replaced_ledger_after_restart(tmp_path: Path) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "replacement-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    account = authority.provider_account(scope=_scope(), meter_id="worker")
    ledger = open_shared_spend_ledger(
        authority.ledger_path,
        policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=100, reservation_id="spent")
    authority.ledger_path.rename(tmp_path / "spent.sqlite3")
    with pytest.raises(BudgetIntegrityError, match="authority differs"):
        SpendLedger(authority.ledger_path, policy)

    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    process = context.Process(
        target=_bind_account_in_fresh_process,
        args=(account.model_dump_json(), outcomes),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert "authority differs" in outcomes.get(timeout=1)


def test_identity_bound_account_rejects_stale_ledger_snapshot_after_restart(
    tmp_path: Path,
) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "rollback-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    account = authority.provider_account(scope=_scope(), meter_id="worker")
    snapshot_path = tmp_path / "budget-before-spend.sqlite3"
    shutil.copyfile(authority.ledger_path, snapshot_path)
    ledger = open_shared_spend_ledger(
        authority.ledger_path,
        policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=100, reservation_id="spent")
    shutil.copyfile(snapshot_path, authority.ledger_path)

    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    process = context.Process(
        target=_bind_account_in_fresh_process,
        args=(account.model_dump_json(), outcomes),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert "rolled back" in outcomes.get(timeout=1)


def test_cached_paid_open_rejects_stale_ledger_snapshot(tmp_path: Path) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "cached-rollback-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    snapshot_path = tmp_path / "budget-before-spend.sqlite3"
    shutil.copyfile(authority.ledger_path, snapshot_path)
    ledger = open_shared_spend_ledger(
        authority.ledger_path,
        policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=100, reservation_id="spent")
    shutil.copyfile(snapshot_path, authority.ledger_path)

    with pytest.raises(BudgetIntegrityError, match="rolled back"):
        open_shared_spend_ledger(
            authority.ledger_path,
            policy,
            expected_ledger_identity=authority.ledger_identity,
        )


def test_rejected_operation_after_authority_backfill_remains_usable(tmp_path: Path) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "backfill-rejection-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    first = SpendLedger(authority.ledger_path, policy, allow_create=False)
    second = SpendLedger(authority.ledger_path, policy, allow_create=False)
    authority_snapshot = tmp_path / "authority-before-spend.sqlite3"
    shutil.copyfile(first.authority_path, authority_snapshot)
    first.reserve(_scope(), meter_id="worker", max_nano_usd=100, reservation_id="spent")
    shutil.copyfile(authority_snapshot, first.authority_path)

    with pytest.raises(BudgetExceededError):
        second.reserve(_scope(), meter_id="worker", max_nano_usd=1, reservation_id="rejected")

    second.audit()
    assert second.snapshot().reserved_nano_usd == 100


def test_ledger_commit_authority_rollback_recovers_without_releasing_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "authority-commit-gap-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    ledger = SpendLedger(authority.ledger_path, policy, allow_create=False)
    original_authority_transaction = ledger._authority_transaction
    original_authority_connection = ledger._authority_connection

    @contextmanager
    def fail_authority_commit() -> Iterator[sqlite3.Connection]:
        with original_authority_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.rollback()
                raise sqlite3.OperationalError("injected authority commit failure")

    monkeypatch.setattr(ledger, "_authority_transaction", fail_authority_commit)
    with pytest.raises(sqlite3.OperationalError, match="injected authority commit failure"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="committed")
    monkeypatch.setattr(ledger, "_authority_transaction", original_authority_transaction)

    ledger.audit()
    assert ledger.snapshot().reserved_nano_usd == 60


def test_ledger_commit_failure_rolls_back_both_files_and_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(hard=100, search=100, final=0).model_copy(
        update={"study_id": "ledger-commit-failure-study"}
    )
    authority = bootstrap_budget_ledger(tmp_path / "budget.sqlite3", policy)
    ledger = SpendLedger(authority.ledger_path, policy, allow_create=False)
    original_transaction = ledger._transaction
    original_connection = ledger._connection

    @contextmanager
    def fail_ledger_commit() -> Iterator[sqlite3.Connection]:
        with original_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.rollback()
                raise sqlite3.OperationalError("injected ledger commit failure")

    monkeypatch.setattr(ledger, "_transaction", fail_ledger_commit)
    with pytest.raises(sqlite3.OperationalError, match="injected ledger commit failure"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="rolled-back")
    monkeypatch.setattr(ledger, "_transaction", original_transaction)

    assert ledger.snapshot().reserved_nano_usd == 0
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="usable")
    assert ledger.snapshot().reserved_nano_usd == 60


def test_one_policy_cannot_bind_two_bootstrapped_ledgers(tmp_path: Path) -> None:
    policy = _policy().model_copy(update={"study_id": "two-ledger-study"})
    first = bootstrap_budget_ledger(tmp_path / "first.sqlite3", policy)
    second = bootstrap_budget_ledger(tmp_path / "second.sqlite3", policy)
    bind_budget_account(first.provider_account(scope=_scope(), meter_id="worker"))

    with pytest.raises(BudgetIntegrityError, match="different ledger path"):
        bind_budget_account(second.provider_account(scope=_scope(), meter_id="worker"))


def test_shared_ledger_cache_uses_canonical_parent_path(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    policy = _policy().model_copy(update={"study_id": "canonical-path-study"})
    authority = bootstrap_budget_ledger(real_parent / "budget.sqlite3", policy)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    first = open_shared_spend_ledger(
        authority.ledger_path,
        authority.policy,
        expected_ledger_identity=authority.ledger_identity,
    )
    second = open_shared_spend_ledger(
        alias_parent / "budget.sqlite3",
        authority.policy,
        expected_ledger_identity=authority.ledger_identity,
    )

    assert first is second


def test_budget_binding_is_path_independent_and_resolves_only_registered_policy(
    tmp_path: Path,
) -> None:
    policy = _policy()
    account = BudgetAccount(
        ledger_path=(tmp_path / "budget.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "budget.sqlite3", policy),
        policy=policy,
        scope=_scope(),
        meter_id="worker",
    )

    binding = bind_budget_account(account)

    assert binding == BudgetAccountBinding(
        policy_digest=policy.policy_digest,
        ledger_identity=SpendLedger(account.ledger_path, policy).ledger_identity,
        scope=_scope(),
        meter_id="worker",
    )
    assert "ledger_path" not in binding.model_dump_json()
    assert str(account.ledger_path) not in binding.model_dump_json()
    assert resolve_budget_account(binding) == account

    tampered = binding.model_copy(update={"ledger_identity": "sha256:" + "f" * 64})
    with pytest.raises(BudgetIntegrityError, match="registered ledger identity"):
        resolve_budget_account(tampered)

    fork = account.model_copy(
        update={"ledger_path": (tmp_path / "fork.sqlite3").resolve()},
        deep=True,
    )
    with pytest.raises(BudgetIntegrityError, match="different ledger path"):
        bind_budget_account(fork)


def test_timed_resource_budget_reserves_ceiling_and_settles_billing_quantum(
    tmp_path: Path,
) -> None:
    clock_values = iter([10.0, 70.001])
    provider = _policy().meters["worker"]
    assert isinstance(provider, ProviderCostMeter)
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "c" * 64,
        fixed_nano_usd=10,
        nano_usd_per_second=3,
        billing_quantum_seconds=60,
        max_billing_seconds=120,
    )
    policy = BudgetPolicy(
        study_id="timed-resource",
        manifest_digest="sha256:" + "d" * 64,
        hard_limit_nano_usd=1_000,
        phase_limits_nano_usd={"confirmation": 1_000},
        meters={"worker": provider, "sandbox": resource},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "resource.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "resource.sqlite3", policy),
        policy=policy,
        scope=_scope("confirmation", "sandbox"),
        meter_id="sandbox",
    )

    reservation = TimedResourceBudget(
        account,
        id_factory=lambda: "sandbox-1",
        monotonic=lambda: next(clock_values),
    ).reserve()
    assert SpendLedger(account.ledger_path, policy).snapshot().reserved_nano_usd == 370

    settled = reservation.settle()

    assert settled.status is ReservationStatus.SETTLED
    assert settled.charged_nano_usd == 370
    assert settled.usage_quantity == 120
    assert settled.usage_unit == "billing_second"
    assert settled.input_tokens is None
    assert settled.output_tokens is None
    assert resource.billed_seconds(0) == 60


def test_timed_resource_class_binds_role_resources_ttl_and_host_horizon(
    tmp_path: Path,
) -> None:
    resource_class = TimedResourceClass(
        role=TimedResourceRole.AGENT_RUNNER,
        cpu_count=2,
        memory_mb=2048,
        provider_ttl_seconds=420,
        create_request_timeout_seconds=30,
        cleanup_horizon_seconds=600,
    )
    meter = TimedResourceCostMeter(
        resource_type=resource_class.role.value,
        resource_class_digest=resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=resource_class.max_host_observation_seconds,
    )
    policy = BudgetPolicy(
        study_id="resource-class",
        manifest_digest="sha256:" + "7" * 64,
        hard_limit_nano_usd=2_000,
        phase_limits_nano_usd={"search": 2_000},
        meters={"runner": meter},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "class.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "class.sqlite3", policy),
        policy=policy,
        scope=_scope(),
        meter_id="runner",
    )

    assert resource_class.max_host_observation_seconds == 1_050
    assert validate_timed_resource_class(account, resource_class) == meter
    for drift in (
        resource_class.model_copy(update={"role": TimedResourceRole.TASK_ENVIRONMENT}),
        resource_class.model_copy(update={"memory_mb": 4096}),
        resource_class.model_copy(update={"provider_ttl_seconds": 421}),
    ):
        with pytest.raises(BudgetIntegrityError, match="role differs|class differs"):
            validate_timed_resource_class(account, drift)

    short_meter = meter.model_copy(update={"max_billing_seconds": 1_049})
    short_policy = policy.model_copy(update={"meters": {"runner": short_meter}}, deep=True)
    short_account = account.model_copy(update={"policy": short_policy}, deep=True)
    with pytest.raises(BudgetIntegrityError, match="horizon"):
        validate_timed_resource_class(short_account, resource_class)


def test_external_spend_authority_is_immutable_policy_authority() -> None:
    authority = ExternalSpendAuthority(
        provider="e2b",
        account_identity="team/account",
        verifier_digest="sha256:" + "8" * 64,
    )
    meter = TimedResourceCostMeter(
        resource_type="task_environment_build",
        resource_class_digest="sha256:" + "9" * 64,
        nano_usd_per_second=1,
        max_billing_seconds=30,
        external_spend_authority=authority,
    )
    policy = BudgetPolicy(
        study_id="external-spend-authority",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"preparation": 100},
        meters={"build": meter},
    )

    for mutation in (
        authority.model_copy(update={"provider": "other"}),
        authority.model_copy(update={"account_identity": "other/account"}),
        authority.model_copy(update={"verifier_digest": "sha256:" + "b" * 64}),
    ):
        mutated_meter = meter.model_copy(update={"external_spend_authority": mutation})
        mutated_policy = policy.model_copy(update={"meters": {"build": mutated_meter}})
        assert mutated_policy.policy_digest != policy.policy_digest


def test_unbound_external_authority_preserves_legacy_digest_and_reopens(
    tmp_path: Path,
) -> None:
    policy = BudgetPolicy(
        study_id="legacy-timed-resource",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={
            "runner": TimedResourceCostMeter(
                resource_type="agent_runner",
                resource_class_digest="sha256:" + "b" * 64,
                nano_usd_per_second=2,
                billing_quantum_seconds=60,
                max_billing_seconds=120,
            )
        },
    )
    expected_digest = "sha256:27406f282ffd8a014471b6d13d33160e79c0075bb3bd728c17d4c484c4438bcb"

    assert policy.policy_digest == expected_digest
    ledger_path = tmp_path / "legacy.sqlite3"
    authority = bootstrap_budget_ledger(ledger_path, policy)
    with sqlite3.connect(ledger_path) as connection:
        [policy_json] = connection.execute(
            "SELECT policy_json FROM budget_metadata WHERE id = 1"
        ).fetchone()
        [entry_json] = connection.execute(
            "SELECT entry_json FROM budget_events WHERE sequence = 1"
        ).fetchone()
    assert "external_spend_authority" not in policy_json
    assert "external_spend_authority" not in entry_json

    reopened = SpendLedger(ledger_path, policy, allow_create=False)

    assert reopened.policy == policy
    assert reopened.ledger_identity == authority.ledger_identity


def test_settlement_terminal_provenance_is_audited_and_legacy_compatible(
    tmp_path: Path,
) -> None:
    policy = _policy()
    provenance = BudgetTerminalProvenance(
        namespace="wmh.test.resource.v1",
        digest="sha256:" + "7" * 64,
    )
    bound_path = tmp_path / "bound.sqlite3"
    bound = SpendLedger(bound_path, policy)
    bound.reserve(
        _scope(),
        meter_id="worker",
        max_nano_usd=60,
        reservation_id="bound",
    )
    bound.settle(
        "bound",
        charged_nano_usd=20,
        input_tokens=0,
        output_tokens=4,
        terminal_provenance=provenance,
    )

    assert bound.settlement_provenance("bound") == provenance
    with sqlite3.connect(bound_path) as connection:
        [bound_event_json] = connection.execute(
            "SELECT entry_json FROM budget_events WHERE sequence = 3"
        ).fetchone()
    assert provenance.namespace in bound_event_json
    assert provenance.digest in bound_event_json
    reopened_bound = SpendLedger(bound_path, policy, allow_create=False)
    assert reopened_bound.settlement_provenance("bound") == provenance

    legacy_path = tmp_path / "legacy-settlement.sqlite3"
    legacy = SpendLedger(legacy_path, policy)
    legacy.reserve(
        _scope(),
        meter_id="worker",
        max_nano_usd=60,
        reservation_id="legacy",
    )
    legacy.settle(
        "legacy",
        charged_nano_usd=20,
        input_tokens=0,
        output_tokens=4,
    )
    with sqlite3.connect(legacy_path) as connection:
        [legacy_event_json] = connection.execute(
            "SELECT entry_json FROM budget_events WHERE sequence = 3"
        ).fetchone()
    assert "terminal_provenance" not in legacy_event_json
    reopened_legacy = SpendLedger(legacy_path, policy, allow_create=False)
    assert reopened_legacy.settlement_provenance("legacy") is None


def test_orphaned_timed_resource_join_is_exact_and_conservative(tmp_path: Path) -> None:
    resource = TimedResourceCostMeter(
        resource_type="agent_runner",
        resource_class_digest="sha256:" + "8" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="orphan-join",
        manifest_digest="sha256:" + "9" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"runner": resource},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "orphan.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "orphan.sqlite3", policy),
        policy=policy,
        scope=_scope(),
        meter_id="runner",
    )

    assert reconcile_orphaned_timed_resource(account, reservation_id="pre-admission") is None
    assert not orphaned_timed_resource_requires_reap(
        account,
        reservation_id="pre-admission",
    )

    TimedResourceBudget(account, id_factory=lambda: "orphaned").reserve()
    joined = reconcile_orphaned_timed_resource(account, reservation_id="orphaned")
    assert joined is not None
    assert joined.status is ReservationStatus.FORFEITED
    assert joined.failure_type == "OrphanedLease"
    assert joined.charged_nano_usd == resource.maximum_charge_nano_usd()
    assert orphaned_timed_resource_requires_reap(account, reservation_id="orphaned")


def test_provider_and_timed_resources_share_one_hard_cap(tmp_path: Path) -> None:
    provider = _policy().meters["worker"]
    assert isinstance(provider, ProviderCostMeter)
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "b" * 64,
        nano_usd_per_second=3,
        max_billing_seconds=100,
    )
    policy = BudgetPolicy(
        study_id="combined-budget",
        manifest_digest="sha256:" + "c" * 64,
        hard_limit_nano_usd=400,
        phase_limits_nano_usd={"confirmation": 400},
        meters={"worker": provider, "sandbox": resource},
    )
    ledger_path = (tmp_path / "combined.sqlite3").resolve()
    ledger = SpendLedger(ledger_path, policy)
    ledger.reserve(
        _scope("confirmation"),
        meter_id="worker",
        max_nano_usd=101,
        reservation_id="provider-1",
    )
    account = TimedResourceBudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=ledger.ledger_identity,
        policy=policy,
        scope=_scope("confirmation", "sandbox"),
        meter_id="sandbox",
    )

    with pytest.raises(BudgetExceededError, match="hard budget"):
        TimedResourceBudget(account, id_factory=lambda: "sandbox-1").reserve()

    assert ledger.snapshot().reserved_nano_usd == 101


def test_timed_resource_binding_is_path_free_and_type_checked(tmp_path: Path) -> None:
    provider = _policy().meters["worker"]
    assert isinstance(provider, ProviderCostMeter)
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "b" * 64,
        nano_usd_per_second=1,
        max_billing_seconds=10,
    )
    policy = BudgetPolicy(
        study_id="resource-binding",
        manifest_digest="sha256:" + "c" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"confirmation": 100},
        meters={"worker": provider, "sandbox": resource},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "binding.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "binding.sqlite3", policy),
        policy=policy,
        scope=_scope("confirmation", "sandbox"),
        meter_id="sandbox",
    )

    binding = bind_timed_resource_account(account)

    assert "ledger_path" not in binding.model_dump_json()
    assert str(account.ledger_path) not in binding.model_dump_json()
    assert resolve_timed_resource_account(binding) == account
    with pytest.raises(BudgetIntegrityError, match="provider token meter"):
        resolve_budget_account(binding)


@pytest.mark.parametrize("resource_type", [" ", "Sandbox", "sandbox/path", "s" * 65])
def test_timed_resource_type_must_be_bounded_and_canonical(resource_type: str) -> None:
    with pytest.raises(ValueError):
        TimedResourceCostMeter(
            resource_type=resource_type,
            resource_class_digest="sha256:" + "b" * 64,
            nano_usd_per_second=1,
            max_billing_seconds=1,
        )


def test_ledger_rejects_settlement_that_differs_from_frozen_tariff(
    tmp_path: Path,
) -> None:
    provider = _policy().meters["worker"]
    assert isinstance(provider, ProviderCostMeter)
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "b" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=60,
    )
    policy = BudgetPolicy(
        study_id="tariff-validation",
        manifest_digest="sha256:" + "c" * 64,
        hard_limit_nano_usd=1_000,
        phase_limits_nano_usd={"search": 1_000},
        meters={"worker": provider, "sandbox": resource},
    )
    provider_ledger = SpendLedger(tmp_path / "provider.sqlite3", policy)
    provider_ledger.reserve(
        _scope(),
        meter_id="worker",
        max_nano_usd=500,
        reservation_id="provider-1",
    )
    resource_ledger = SpendLedger(tmp_path / "resource.sqlite3", policy)
    resource_ledger.reserve(
        _scope(),
        meter_id="sandbox",
        max_nano_usd=120,
        reservation_id="resource-1",
    )

    with pytest.raises(BudgetIntegrityError, match="frozen tariff"):
        provider_ledger.settle(
            "provider-1",
            charged_nano_usd=0,
            input_tokens=2,
            output_tokens=3,
        )
    with pytest.raises(BudgetIntegrityError, match="frozen tariff"):
        resource_ledger.settle(
            "resource-1",
            charged_nano_usd=0,
            usage_quantity=60,
            usage_unit="billing_second",
        )
    with pytest.raises(BudgetIntegrityError, match="at least one billing quantum"):
        resource_ledger.settle(
            "resource-1",
            charged_nano_usd=0,
            usage_quantity=0,
            usage_unit="billing_second",
        )

    assert provider_ledger.reservations()[0].status is ReservationStatus.RESERVED
    assert resource_ledger.reservations()[0].status is ReservationStatus.RESERVED


def test_timed_resource_reservation_retries_after_precommit_ledger_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter([0.0, 0.1])
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "e" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="timed-resource-retry",
        manifest_digest="sha256:" + "f" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"sandbox": resource},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "retry.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "retry.sqlite3", policy),
        policy=policy,
        scope=_scope(),
        meter_id="sandbox",
    )
    reservation = TimedResourceBudget(
        account,
        id_factory=lambda: "sandbox-1",
        monotonic=lambda: next(clock_values),
    ).reserve()
    original_settle = SpendLedger.settle
    fail_once = True

    def flaky_settle(
        ledger: SpendLedger,
        reservation_id: str,
        *,
        charged_nano_usd: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        usage_quantity: int | None = None,
        usage_unit: str | None = None,
        breach_kind: BudgetBreachKind | None = None,
    ) -> BudgetReservation:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise sqlite3.OperationalError("transient")
        return original_settle(
            ledger,
            reservation_id,
            charged_nano_usd=charged_nano_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_quantity=usage_quantity,
            usage_unit=usage_unit,
            breach_kind=breach_kind,
        )

    monkeypatch.setattr(SpendLedger, "settle", flaky_settle)

    with pytest.raises(sqlite3.OperationalError, match="transient"):
        reservation.settle()
    settled = reservation.settle()

    assert settled.status is ReservationStatus.SETTLED
    assert settled.usage_quantity == 1


def test_timed_resource_reservation_requires_its_exact_maximum(tmp_path: Path) -> None:
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "e" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="timed-resource-ceiling",
        manifest_digest="sha256:" + "f" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"sandbox": resource},
    )
    ledger = SpendLedger(tmp_path / "ceiling.sqlite3", policy)

    with pytest.raises(ValueError, match="exact maximum"):
        ledger.reserve(
            _scope(),
            meter_id="sandbox",
            max_nano_usd=59,
            reservation_id="sandbox-1",
        )

    assert ledger.reservations() == []


def test_timed_resource_under_reservation_is_rejected_during_replay(tmp_path: Path) -> None:
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "e" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="timed-resource-replay",
        manifest_digest="sha256:" + "f" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"sandbox": resource},
    )
    path = tmp_path / "replay.sqlite3"
    SpendLedger(path, policy)
    TimedResourceBudget(
        TimedResourceBudgetAccount(
            ledger_path=path.resolve(),
            ledger_identity=_ledger_identity(path, policy),
            policy=policy,
            scope=_scope(),
            meter_id="sandbox",
        ),
        id_factory=lambda: "sandbox-1",
    ).reserve()

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("DROP TRIGGER budget_events_no_update")
        row = connection.execute(
            "SELECT entry_json FROM budget_events WHERE sequence = 2"
        ).fetchone()
        assert row is not None
        event = json.loads(row["entry_json"])
        event["action"]["max_nano_usd"] = 59
        unsigned = {key: value for key, value in event.items() if key != "digest"}
        event["digest"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        )
        connection.execute(
            "UPDATE budget_events SET digest = ?, entry_json = ? WHERE sequence = 2",
            (
                event["digest"],
                json.dumps(
                    event,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            ),
        )
        connection.execute(
            "UPDATE budget_reservations SET max_nano_usd = 59 WHERE reservation_id = 'sandbox-1'"
        )

    with pytest.raises(BudgetIntegrityError, match="exact maximum"):
        SpendLedger(path, policy)


def test_timed_resource_rejects_token_only_breach_kinds(tmp_path: Path) -> None:
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "e" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="timed-resource-breach-kind",
        manifest_digest="sha256:" + "f" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"sandbox": resource},
    )
    ledger = SpendLedger(tmp_path / "breach.sqlite3", policy)
    ledger.reserve(
        _scope(),
        meter_id="sandbox",
        max_nano_usd=60,
        reservation_id="sandbox-1",
    )

    with pytest.raises(BudgetIntegrityError, match="token breach"):
        ledger.settle(
            "sandbox-1",
            charged_nano_usd=2,
            usage_quantity=1,
            usage_unit="billing_second",
            breach_kind=BudgetBreachKind.INPUT_TOKEN_CEILING,
        )

    assert ledger.reservations()[0].status is ReservationStatus.RESERVED


def test_timed_resource_failure_forfeits_full_ceiling_and_is_terminal(tmp_path: Path) -> None:
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "e" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="timed-resource-failure",
        manifest_digest="sha256:" + "f" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"sandbox": resource},
    )
    account = TimedResourceBudgetAccount(
        ledger_path=(tmp_path / "resource.sqlite3").resolve(),
        ledger_identity=_ledger_identity(tmp_path / "resource.sqlite3", policy),
        policy=policy,
        scope=_scope(),
        meter_id="sandbox",
    )
    reservation = TimedResourceBudget(account, id_factory=lambda: "sandbox-1").reserve()

    forfeited = reservation.forfeit(failure_type="CreateUnknown")

    assert forfeited.status is ReservationStatus.FORFEITED
    assert forfeited.charged_nano_usd == 60
    with pytest.raises(BudgetIntegrityError, match="already terminal"):
        reservation.settle()


def test_forfeit_failure_type_is_a_bounded_nonsecret_code(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=10, reservation_id="r1")

    with pytest.raises(ValueError, match="failure code"):
        ledger.forfeit("r1", failure_type="/private/tmp/provider-error.txt")

    assert ledger.reservations()[0].status is ReservationStatus.RESERVED


def test_spend_ledger_reserves_settles_and_forfeits_without_releasing_history(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())

    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r1")
    reserved = ledger.snapshot()
    assert reserved.charged_nano_usd == 0
    assert reserved.reserved_nano_usd == 60
    assert reserved.remaining_nano_usd == 40

    ledger.settle("r1", charged_nano_usd=20, input_tokens=0, output_tokens=4)
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r2")
    ledger.forfeit("r2", failure_type="ConnectionError")

    snapshot = ledger.snapshot()
    assert snapshot.charged_nano_usd == 80
    assert snapshot.reserved_nano_usd == 0
    assert snapshot.remaining_nano_usd == 20
    assert snapshot.by_phase_nano_usd == {"final": 0, "search": 80}
    reservations = {item.reservation_id: item for item in ledger.reservations()}
    assert reservations["r1"].status is ReservationStatus.SETTLED
    assert reservations["r1"].input_tokens == 0
    assert reservations["r2"].status is ReservationStatus.FORFEITED
    assert len(ledger.events()) == 5  # genesis, two reserves, settle, forfeit
    ledger.audit()


def test_spend_ledger_enforces_phase_and_hard_caps_before_reservation(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=80, reservation_id="search-full")

    with pytest.raises(BudgetExceededError, match="phase 'search'.*remaining=0"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=1, reservation_id="phase-over")
    with pytest.raises(BudgetExceededError, match="hard budget.*remaining=20"):
        ledger.reserve(
            _scope("final", "confirmation"),
            meter_id="worker",
            max_nano_usd=21,
            reservation_id="hard-over",
        )

    ledger.reserve(
        _scope("final", "confirmation"),
        meter_id="worker",
        max_nano_usd=20,
        reservation_id="final-full",
    )
    assert ledger.snapshot().remaining_nano_usd == 0


def test_open_reservation_is_charged_as_exposure_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    SpendLedger(path, _policy()).reserve(
        _scope(), meter_id="worker", max_nano_usd=80, reservation_id="orphaned-call"
    )

    reopened = SpendLedger(path, _policy())
    with pytest.raises(BudgetExceededError):
        reopened.reserve(_scope(), meter_id="worker", max_nano_usd=1, reservation_id="must-not-run")
    assert reopened.snapshot().reserved_nano_usd == 80


def test_concurrent_reservations_are_admitted_atomically(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    policy = _policy(hard=100, search=100, final=0)
    SpendLedger(path, policy)
    barrier = threading.Barrier(3)
    admitted: list[str] = []
    rejected: list[str] = []

    def reserve(reservation_id: str) -> None:
        ledger = SpendLedger(path, policy)
        barrier.wait()
        try:
            ledger.reserve(
                _scope(), meter_id="worker", max_nano_usd=60, reservation_id=reservation_id
            )
        except BudgetExceededError:
            rejected.append(reservation_id)
        else:
            admitted.append(reservation_id)

    threads = [
        threading.Thread(target=reserve, args=(reservation_id,)) for reservation_id in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert len(admitted) == len(rejected) == 1
    assert SpendLedger(path, policy).snapshot().reserved_nano_usd == 60


def test_processes_initialize_and_reserve_atomically(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "process-budget.sqlite3"
    policy = _policy(hard=100, search=100, final=0)
    barrier = context.Barrier(3)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(str(path), policy.model_dump_json(), reservation_id, barrier, outcomes),
        )
        for reservation_id in ("process-a", "process-b")
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    observed = sorted(outcomes.get(timeout=20) for _ in processes)
    for process in processes:
        process.join(timeout=20)

    assert observed == ["admitted", "rejected"]
    assert all(process.exitcode == 0 for process in processes)
    assert SpendLedger(path, policy).snapshot().reserved_nano_usd == 60


def test_provider_and_timed_processes_compete_for_one_atomic_cap(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "mixed-process-budget.sqlite3"
    provider = _policy().meters["worker"]
    assert isinstance(provider, ProviderCostMeter)
    resource = TimedResourceCostMeter(
        resource_type="sandbox",
        resource_class_digest="sha256:" + "b" * 64,
        nano_usd_per_second=2,
        max_billing_seconds=30,
    )
    policy = BudgetPolicy(
        study_id="mixed-process-budget",
        manifest_digest="sha256:" + "c" * 64,
        hard_limit_nano_usd=100,
        phase_limits_nano_usd={"search": 100},
        meters={"worker": provider, "sandbox": resource},
    )
    barrier = context.Barrier(3)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(
                str(path),
                policy.model_dump_json(),
                f"process-{meter_id}",
                barrier,
                outcomes,
                meter_id,
                60,
            ),
        )
        for meter_id in ("worker", "sandbox")
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    observed = sorted(outcomes.get(timeout=20) for _ in processes)
    for process in processes:
        process.join(timeout=20)

    assert observed == ["admitted", "rejected"]
    assert all(process.exitcode == 0 for process in processes)
    assert SpendLedger(path, policy).snapshot().reserved_nano_usd == 60


def test_process_death_after_reservation_keeps_full_exposure(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "orphan-budget.sqlite3"
    policy = _policy(hard=100, search=100, final=0)
    process = context.Process(
        target=_reserve_then_exit,
        args=(str(path), policy.model_dump_json()),
    )
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    ledger = SpendLedger(path, policy)
    assert ledger.snapshot().reserved_nano_usd == 80
    with pytest.raises(BudgetExceededError, match="remaining=20"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=21, reservation_id="must-not-run")


def test_settlement_over_reservation_records_breach_and_blocks_future_spend(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r1")

    with pytest.raises(BudgetBreachError, match="exceeded its 60 nano-USD reservation"):
        ledger.settle("r1", charged_nano_usd=63, input_tokens=9, output_tokens=9)

    snapshot = ledger.snapshot()
    assert snapshot.breached
    assert snapshot.charged_nano_usd == 63
    with pytest.raises(BudgetBreachError, match="already breached"):
        ledger.reserve(_scope("final"), meter_id="worker", max_nano_usd=1, reservation_id="r2")
    ledger.audit()


def test_ledger_schema_and_hash_chain_fail_closed_on_external_mutation(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    ledger = SpendLedger(path, _policy())
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=10, reservation_id="r1")

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE budget_events SET entry_json = '{}' WHERE sequence = 2")
        connection.execute("DROP TRIGGER budget_events_no_update")
        connection.execute("UPDATE budget_events SET entry_json = '{}' WHERE sequence = 2")

    with pytest.raises(BudgetIntegrityError, match="event 2"):
        ledger.events()
    with pytest.raises(BudgetIntegrityError, match="event 2"):
        SpendLedger(path, _policy())


def test_full_audit_rejects_a_validly_rehashed_but_false_settlement(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    ledger = SpendLedger(path, _policy())
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r1")
    ledger.settle("r1", charged_nano_usd=14, input_tokens=2, output_tokens=2)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("DROP TRIGGER budget_events_no_update")
        row = connection.execute(
            "SELECT entry_json FROM budget_events WHERE sequence = 3"
        ).fetchone()
        assert row is not None
        event = json.loads(row["entry_json"])
        event["action"]["charged_nano_usd"] = 70
        unsigned = {key: value for key, value in event.items() if key != "digest"}
        event["digest"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
        )
        connection.execute(
            "UPDATE budget_events SET digest = ?, entry_json = ? WHERE sequence = 3",
            (
                event["digest"],
                json.dumps(
                    event,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            ),
        )
        connection.execute(
            "UPDATE budget_reservations SET charged_nano_usd = 70 WHERE reservation_id = 'r1'"
        )

    with pytest.raises(BudgetIntegrityError, match="frozen tariff"):
        SpendLedger(path, _policy())


def test_verified_state_does_not_release_spend_when_the_mutable_index_is_corrupted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.sqlite3"
    policy = _policy(hard=100, search=100, final=0)
    ledger = SpendLedger(path, policy)
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=80, reservation_id="r1")
    ledger.settle("r1", charged_nano_usd=56, input_tokens=8, output_tokens=8)
    assert ledger.snapshot().charged_nano_usd == 56

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM budget_reservations WHERE reservation_id = 'r1'")

    with pytest.raises(BudgetExceededError, match="remaining=44"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=90, reservation_id="r2")
    assert ledger.snapshot().charged_nano_usd == 56
    with pytest.raises(BudgetIntegrityError, match="index differs"):
        SpendLedger(path, policy)


def test_independent_ledgers_incrementally_verify_each_others_committed_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.sqlite3"
    first = SpendLedger(path, _policy())
    second = SpendLedger(path, _policy())

    first.reserve(_scope(), meter_id="worker", max_nano_usd=10, reservation_id="r1")
    second.reserve(_scope(), meter_id="worker", max_nano_usd=10, reservation_id="r2")

    assert first.snapshot().reserved_nano_usd == 20
    assert second.snapshot().reserved_nano_usd == 20


def test_open_ledger_rejects_path_replacement(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    ledger = SpendLedger(path, _policy())
    path.rename(tmp_path / "moved.sqlite3")
    path.touch()

    with pytest.raises(BudgetIntegrityError, match="file changed"):
        ledger.snapshot()


def test_ledger_rejects_policy_drift_and_symlink_paths(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    SpendLedger(path, _policy())
    with pytest.raises(BudgetIntegrityError, match="policy"):
        SpendLedger(path, _policy(hard=101))

    target = tmp_path / "target.sqlite3"
    target.touch()
    symlink = tmp_path / "link.sqlite3"
    symlink.symlink_to(target)
    with pytest.raises(BudgetIntegrityError, match="symlink") as error:
        SpendLedger(symlink, _policy())
    assert str(symlink) not in str(error.value)


def test_ledger_schema_v1_requires_a_new_v5_ledger(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    SpendLedger(path, _policy())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version 1"):
        SpendLedger(path, _policy())


def test_v3_tariff_policy_ledger_requires_a_new_v5_ledger(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    policy = _policy()
    SpendLedger(path, policy)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT policy_json FROM budget_metadata WHERE id = 1").fetchone()
        assert row is not None
        old_policy = json.loads(row["policy_json"])
        provenance = old_policy["meters"]["worker"]["tariff_provenance"]
        del old_policy["meters"]["worker"]["tariff_evidence_receipt"]
        source = provenance["source_snapshots"][0]
        provenance["schema_version"] = 3
        provenance["source_locator"] = source["source_locator"]
        provenance["source_snapshot_digest"] = source["source_snapshot_digest"]
        del provenance["source_snapshots"]
        del provenance["source_bindings"]
        route = provenance["route"]
        route["schema_version"] = 2
        route["billing_sku"] = route.pop("billing_mode")
        for meter in route["billing_meters"]:
            meter["schema_version"] = 1
            del meter["source_id"]
            del meter["source_record_path"]
            del meter["sku_id"]
            del meter["billing_region"]
            del meter["billing_mode"]
            del meter["effective_on"]
            del meter["source_price_usd"]
            del meter["source_price_unit"]
        old_policy_json = json.dumps(old_policy, sort_keys=True, separators=(",", ":"))
        old_policy_digest = "sha256:" + hashlib.sha256(old_policy_json.encode()).hexdigest()
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute(
            "UPDATE budget_metadata SET schema_version = 3, policy_digest = ?, policy_json = ? "
            "WHERE id = 1",
            (old_policy_digest, old_policy_json),
        )

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version 3"):
        SpendLedger(path, policy)


def test_v4_policy_without_tariff_receipt_requires_a_new_v5_ledger(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    policy = _policy()
    SpendLedger(path, policy)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT policy_json FROM budget_metadata WHERE id = 1").fetchone()
        assert row is not None
        old_policy = json.loads(row["policy_json"])
        del old_policy["meters"]["worker"]["tariff_evidence_receipt"]
        old_policy_json = json.dumps(old_policy, sort_keys=True, separators=(",", ":"))
        old_policy_digest = "sha256:" + hashlib.sha256(old_policy_json.encode()).hexdigest()
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute(
            "UPDATE budget_metadata SET schema_version = 4, policy_digest = ?, policy_json = ? "
            "WHERE id = 1",
            (old_policy_digest, old_policy_json),
        )

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version 4"):
        SpendLedger(path, policy)


def test_cached_ledger_full_audit_revalidates_metadata(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    ledger = SpendLedger(path, _policy())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version 1"):
        ledger.audit()


def test_exposed_policy_is_a_defensive_deep_copy(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())
    exposed = ledger.policy
    exposed.phase_limits_nano_usd["search"] = 1
    exposed_meter = exposed.meters["worker"]
    assert isinstance(exposed_meter, ProviderCostMeter)
    exposed.meters["worker"] = exposed_meter.model_copy(
        update={
            "provider_config": exposed_meter.provider_config.model_copy(update={"model": "mutated"})
        }
    )

    assert ledger.policy.phase_limits_nano_usd["search"] == 80
    persisted_meter = ledger.policy.meters["worker"]
    assert isinstance(persisted_meter, ProviderCostMeter)
    assert persisted_meter.provider_config.model == "model-1"


class _FakeToolProvider:
    paid_request_attempts: Literal[1] = 1

    def __init__(
        self,
        *,
        text_usage: TokenUsage | None = None,
        chat_usage: ChatUsage | None = None,
        chat_response: ChatResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.config = ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="model-1",
            region="test-region",
        )
        self._text_usage = text_usage
        self._chat_usage = chat_usage
        self._chat_response = chat_response
        self._error = error

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        del system, messages, temperature, max_tokens
        if self._error is not None:
            raise self._error
        if self._text_usage is None:
            return Completion(text="ok")
        return Completion(text="ok", usage=self._text_usage)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        if self._error is not None:
            raise self._error
        if self._chat_response is not None:
            return self._chat_response
        return ChatResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="ok"))],
            usage=self._chat_usage,
            model="model-1",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


def _budgeted_provider(
    tmp_path: Path,
    provider: SingleDispatchProvider,
    *,
    ids: Iterator[str],
    response_identity: ProviderResponseIdentity | None = None,
) -> tuple[BudgetedProvider, SpendLedger]:
    policy = _policy(hard=10_000_000, search=10_000_000, final=0)
    path = tmp_path / "provider-budget.sqlite3"
    ledger = SpendLedger(path, policy)
    account = BudgetAccount(
        ledger_path=path,
        ledger_identity=ledger.ledger_identity,
        policy=policy,
        scope=_scope(category="provider-call"),
        meter_id="worker",
    )
    return (
        BudgetedProvider(
            provider,
            account,
            response_identity=response_identity,
            id_factory=lambda: next(ids),
        ),
        ledger,
    )


def test_budgeted_provider_exposes_exact_binding_and_rejects_nested_wrapper(
    tmp_path: Path,
) -> None:
    provider, _ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1)),
        ids=iter(["inner"]),
    )

    assert provider.paid_request_attempts == 1
    assert provider.budget_binding.policy_digest == provider.budget_policy_digest
    assert provider.budget_binding.ledger_identity == provider.budget_ledger_identity
    assert provider.wrapped_provider_implementation.endswith("._FakeToolProvider")
    with pytest.raises(TypeError, match="cannot wrap another BudgetedProvider"):
        BudgetedProvider(
            provider,
            provider._account,  # noqa: SLF001 - exact nested-wrapper rejection contract
            id_factory=lambda: "outer",
        )


@pytest.mark.parametrize("attempt_contract", ["missing", "multiple"])
def test_budgeted_provider_rejects_non_single_dispatch_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_contract: str,
) -> None:
    """Unsafe provider retry contracts never enter the paid budget boundary."""
    policy = _policy(hard=10_000_000, search=10_000_000, final=0)
    path = tmp_path / "provider-contract-budget.sqlite3"
    ledger = SpendLedger(path, policy)
    account = BudgetAccount(
        ledger_path=path,
        ledger_identity=ledger.ledger_identity,
        policy=policy,
        scope=_scope(category="provider-call"),
        meter_id="worker",
    )
    provider = _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1))
    if attempt_contract == "missing":
        monkeypatch.delattr(_FakeToolProvider, "paid_request_attempts")
    else:
        monkeypatch.setattr(provider, "paid_request_attempts", 2)

    with pytest.raises(TypeError, match="single-dispatch"):
        BudgetedProvider(provider, account, id_factory=lambda: "must-not-reserve")

    assert ledger.reservations() == []


def test_budgeted_provider_rejects_wrapped_config_drift_before_reservation(
    tmp_path: Path,
) -> None:
    raw_provider = _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1))
    provider, ledger = _budgeted_provider(
        tmp_path,
        raw_provider,
        ids=iter(["must-not-reserve"]),
    )
    raw_provider.config = raw_provider.config.model_copy(update={"model": "model-2"})

    with pytest.raises(BudgetIntegrityError, match="provider config changed"):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_completion_tokens=20,
            )
        )

    assert provider.config.model == "model-1"
    assert ledger.reservations() == []


def test_budgeted_provider_reserves_before_call_and_settles_exact_usage(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=11, completion_tokens=7)),
        ids=iter(["chat-1"]),
    )
    request = ChatRequest(
        model="ignored",
        messages=[ChatMessage(role="user", content="hello")],
        max_completion_tokens=20,
    )

    response = provider.complete_chat(request)

    assert response.choices[0].message.content == "ok"
    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 11 * 2 + 7 * 5
    assert reservation.max_nano_usd > reservation.charged_nano_usd


def test_live_shaped_bedrock_submit_with_zero_cache_usage_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiveShapedBedrockClient:
        def converse(self, **_kwargs: object) -> dict[str, object]:
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-1",
                                    "name": "submit",
                                    "input": {"answer": "ok"},
                                    "type": "tool_use",
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {
                    "inputTokens": 2,
                    "outputTokens": 1,
                    "totalTokens": 3,
                    "cacheReadInputTokens": 0,
                    "cacheWriteInputTokens": 0,
                },
                "ResponseMetadata": {"RequestId": "bedrock-request-1"},
            }

    raw_provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="model-1",
            region="test-region",
        )
    )
    client = LiveShapedBedrockClient()
    monkeypatch.setattr(raw_provider, "_get_client", lambda: client)
    provider, ledger = _budgeted_provider(
        tmp_path,
        raw_provider,
        ids=iter(["bedrock-zero-cache"]),
    )

    returned = provider.complete_chat(
        ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "solve"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "description": "return the answer",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "max_completion_tokens": 20,
            }
        )
    )

    assert returned.choices[0].message.tool_calls is not None
    assert returned.choices[0].message.tool_calls[0].function.name == "submit"
    assert returned.usage is not None
    assert returned.usage.model_extra == {"total_tokens": 3}
    assert returned.provider_receipt is not None
    assert returned.provider_receipt.provider_request_id == "bedrock-request-1"
    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 2 * 2 + 1 * 5


@pytest.mark.parametrize(
    ("usage_dimension", "dimension_name"),
    [
        ({"cacheReadInputTokens": 1}, "cache_read_input_tokens"),
        ({"cacheWriteInputTokens": 1}, "cache_write_input_tokens"),
        ({"cacheReadInputTokens": False}, "cache_read_input_tokens"),
        ({"cacheWriteInputTokens": 0.0}, "cache_write_input_tokens"),
        ({"cacheDetails": [{"inputTokens": 1, "ttl": "5m"}]}, "cacheDetails"),
        ({"cacheDetails": {}}, "cacheDetails"),
        ({"cacheDetails": None}, "cacheDetails"),
    ],
)
def test_bedrock_submit_rejects_nonzero_nonempty_or_malformed_cache_usage(
    tmp_path: Path,
    usage_dimension: dict[str, JsonValue],
    dimension_name: str,
) -> None:
    response = bedrock_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-1",
                                "name": "submit",
                                "input": {"answer": "ok"},
                                "type": "tool_use",
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 2,
                "outputTokens": 1,
                "totalTokens": 3,
                **usage_dimension,
            },
        },
        "model-1",
        advertised_client_tools=frozenset({"submit"}),
    )
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_response=response),
        ids=iter(["bedrock-unsupported-cache"]),
    )

    with pytest.raises(UnpricedProviderUsageError, match=dimension_name):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="solve")],
                max_completion_tokens=20,
            )
        )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UnpricedUsage"


def test_budgeted_chat_forfeits_before_settlement_on_unpriced_usage_dimensions(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 3},
                }
            )
        ),
        ids=iter(["chat-unpriced"]),
    )
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="hello")],
        max_completion_tokens=20,
    )

    with pytest.raises(UnpricedProviderUsageError, match="prompt_tokens_details"):
        provider.complete_chat(request)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UnpricedUsage"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_chat_accepts_a_consistent_derived_total_token_field(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            )
        ),
        ids=iter(["chat-derived-total"]),
    )

    provider.complete_chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_completion_tokens=20,
        )
    )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 11 * 2 + 7 * 5


def test_budgeted_chat_accepts_openai_sdk_null_usage_detail_placeholders(
    tmp_path: Path,
) -> None:
    sdk_usage = CompletionUsage(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
    ).model_dump(mode="json")
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage.model_validate(sdk_usage)),
        ids=iter(["chat-sdk-null-details"]),
    )

    provider.complete_chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_completion_tokens=20,
        )
    )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 11 * 2 + 7 * 5


def test_budgeted_chat_accepts_responses_sdk_null_usage_detail_placeholders(
    tmp_path: Path,
) -> None:
    sdk_usage = ResponseUsage.model_construct(
        input_tokens=11,
        input_tokens_details=None,
        output_tokens=7,
        output_tokens_details=None,
        total_tokens=18,
    ).model_dump(mode="json")
    chat_usage = ChatUsage.model_validate(
        {
            "prompt_tokens": sdk_usage.pop("input_tokens"),
            "completion_tokens": sdk_usage.pop("output_tokens"),
            **sdk_usage,
        }
    )
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=chat_usage),
        ids=iter(["responses-sdk-null-details"]),
    )

    provider.complete_chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_completion_tokens=20,
        )
    )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 11 * 2 + 7 * 5


@pytest.mark.parametrize(
    "input_tokens_details",
    [
        {"cached_tokens": 3},
        {"cached_tokens": 3, "cache_write_tokens": 4},
    ],
)
def test_budgeted_chat_accepts_exact_responses_inclusive_usage_breakdowns(
    tmp_path: Path,
    input_tokens_details: dict[str, int],
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "input_tokens_details": input_tokens_details,
                    "output_tokens_details": {"reasoning_tokens": 5},
                }
            )
        ),
        ids=iter(["responses-inclusive-breakdowns"]),
    )

    provider.complete_chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_completion_tokens=20,
        )
    )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 11 * 2 + 7 * 5


@pytest.mark.parametrize(
    "usage_details",
    [
        {"input_tokens_details": {"cached_tokens": 3, "future_tokens": 1}},
        {"input_tokens_details": {"cached_tokens": True}},
        {"input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": True}},
        {"input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": -1}},
        {"input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 12}},
        {"input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 8}},
        {"input_tokens_details": {"cache_write_tokens": 3}},
        {"input_tokens_details": ["cached_tokens"]},
        {"output_tokens_details": {"reasoning_tokens": 8}},
    ],
)
def test_budgeted_chat_rejects_noncanonical_responses_usage_breakdowns(
    tmp_path: Path,
    usage_details: dict[str, object],
) -> None:
    [dimension] = usage_details
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    **usage_details,
                }
            )
        ),
        ids=iter(["responses-invalid-breakdown"]),
    )

    with pytest.raises(UnpricedProviderUsageError, match=dimension):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_completion_tokens=20,
            )
        )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UnpricedUsage"


def test_budgeted_chat_rejects_a_non_numeric_derived_total_token_field(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": None}
            )
        ),
        ids=iter(["chat-invalid-total"]),
    )

    with pytest.raises(UnpricedProviderUsageError, match="total_tokens"):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_completion_tokens=20,
            )
        )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED


def test_budgeted_chat_rejects_an_unknown_null_usage_dimension(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            chat_usage=ChatUsage.model_validate(
                {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "future_billable_tokens": None,
                }
            )
        ),
        ids=iter(["chat-unknown-null-dimension"]),
    )

    with pytest.raises(UnpricedProviderUsageError, match="future_billable_tokens"):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_completion_tokens=20,
            )
        )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED


def test_budgeted_text_forfeits_before_settlement_on_unpriced_usage_dimensions(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            text_usage=TokenUsage.model_validate(
                {"input_tokens": 4, "output_tokens": 2, "cache_read_input_tokens": 1}
            )
        ),
        ids=iter(["text-unpriced"]),
    )

    with pytest.raises(UnpricedProviderUsageError, match="cache_read_input_tokens"):
        provider.complete("", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UnpricedUsage"


def test_budgeted_text_accepts_exact_responses_inclusive_usage_breakdowns(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            text_usage=TokenUsage.model_validate(
                {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens_details": {"reasoning_tokens": 2},
                }
            )
        ),
        ids=iter(["text-responses-inclusive-breakdowns"]),
    )

    provider.complete("", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 4 * 2 + 2 * 5


def test_budgeted_azure_reasoning_completion_identity_and_usage_details_settle_exactly(
    tmp_path: Path,
) -> None:
    provider_config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        endpoint="https://example.openai.azure.com",
        deployment="gpt55-deploy",
        api_version="2024-10-21",
        reasoning_effort="high",
        responses_api_version="v1",
    )
    completion = Completion(
        text="ok",
        model="gpt-5.5-2026-06-01",
        system_fingerprint="fp-azure-responses-1",
        usage=TokenUsage.model_validate(
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 5},
            }
        ),
    )

    class AzureReasoningProvider(_FakeToolProvider):
        def __init__(self) -> None:
            super().__init__()
            self.config = provider_config

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            del system, messages, temperature, max_tokens
            return completion

    provenance = synthetic_tariff_provenance(provider_config)
    policy = BudgetPolicy(
        study_id="azure-reasoning-cost-boundary",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=100_000,
        phase_limits_nano_usd={"search": 100_000},
        meters={
            "worker": synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=provenance,
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            )
        },
    )
    ledger_path = tmp_path / "azure-reasoning-cost-boundary.sqlite3"
    ledger = SpendLedger(ledger_path, policy)
    provider = BudgetedProvider(
        AzureReasoningProvider(),
        BudgetAccount(
            ledger_path=ledger_path,
            ledger_identity=ledger.ledger_identity,
            policy=policy,
            scope=BudgetScope(
                phase="search",
                category="worker",
                run_id="azure-reasoning-cost-boundary",
            ),
            meter_id="worker",
        ),
        response_identity=ProviderResponseIdentity(
            provider=ProviderKind.AZURE_OPENAI,
            response_model="gpt-5.5-2026-06-01",
            system_fingerprint="fp-azure-responses-1",
        ),
        id_factory=lambda: "azure-reasoning-cost-boundary-1",
    )

    returned = provider.complete(
        "system",
        [Message(role="user", content="hello")],
        max_tokens=20,
    )

    assert returned is completion
    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 18


def test_budgeted_text_accepts_openai_sdk_null_usage_detail_placeholders(
    tmp_path: Path,
) -> None:
    sdk_usage = CompletionUsage(
        prompt_tokens=4,
        completion_tokens=2,
        total_tokens=6,
    ).model_dump(mode="json")
    text_usage = TokenUsage.model_validate(
        {
            "input_tokens": sdk_usage.pop("prompt_tokens"),
            "output_tokens": sdk_usage.pop("completion_tokens"),
            **sdk_usage,
        }
    )
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(text_usage=text_usage),
        ids=iter(["text-sdk-null-details"]),
    )

    provider.complete("", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.charged_nano_usd == 4 * 2 + 2 * 5


def test_budgeted_bedrock_chat_accepts_requested_model_with_null_served_receipt(
    tmp_path: Path,
) -> None:
    receipt = build_chat_provider_receipt(
        provider="bedrock",
        provider_request_id="bedrock-request-1",
        response_id=None,
        requested_model="model-1",
        response_model=None,
        system_fingerprint=None,
        request_payload={"messages": [], "inferenceConfig": {"maxTokens": 20}},
        temperature=0.7,
        max_tokens=20,
        max_tokens_field="inferenceConfig.maxTokens",
        started_at_unix_s=1.0,
        finished_at_unix_s=2.0,
    )
    response = ChatResponse(
        choices=[ChatChoice(message=ChatMessage(role="assistant", content="ok"))],
        usage=ChatUsage(prompt_tokens=1, completion_tokens=1),
        model="model-1",
        provider_receipt=receipt,
    )
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_response=response),
        response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
        ids=iter(["bedrock-chat-identity"]),
    )

    returned = provider.complete_chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.7,
            max_completion_tokens=20,
        )
    )

    assert returned is response
    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED


def test_budgeted_chat_ceiling_dominates_both_compatibility_token_fields(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=90)),
        ids=iter(["dual-limit-chat"]),
    )
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 90,
            "max_completion_tokens": 1,
        }
    )

    provider.complete_chat(request)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.output_tokens == 90


def test_budgeted_chat_rejects_unquoted_extra_dispatch_fields(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1)),
        ids=iter(["must-not-reserve"]),
    )
    request = ChatRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 10,
            "n": 2,
        }
    )

    with pytest.raises(ValueError, match="unquoted provider fields: n"):
        provider.complete_chat(request)

    assert ledger.reservations() == []


def test_budgeted_chat_rejects_unpriced_multimodal_content(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1)),
        ids=iter(["must-not-reserve"]),
    )
    request = ChatRequest.model_validate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/x"}},
                    ],
                }
            ],
            "max_completion_tokens": 10,
        }
    )

    with pytest.raises(ValueError, match="unpriced non-text content"):
        provider.complete_chat(request)

    assert ledger.reservations() == []


def test_budgeted_chat_forfeits_partial_usage_instead_of_defaulting_missing_tokens(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1)),
        ids=iter(["partial-usage"]),
    )
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="hello")],
        max_completion_tokens=10,
    )

    provider.complete_chat(request)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageUnavailable"


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), None])
def test_budgeted_provider_forfeits_full_reservation_when_usage_is_unproved(
    tmp_path: Path,
    failure: Exception | None,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(error=failure),
        ids=iter(["call-1"]),
    )

    if failure is None:
        provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)
    else:
        with pytest.raises(RuntimeError, match="provider failed"):
            provider.complete(
                "system",
                [Message(role="user", content="hello")],
                max_tokens=10,
            )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_provider_forfeits_partial_text_usage(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(text_usage=TokenUsage(input_tokens=1)),
        ids=iter(["partial-text-usage"]),
    )

    provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageUnavailable"


def test_budgeted_provider_forfeits_negative_text_usage_as_terminal_integrity_error(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(text_usage=TokenUsage(input_tokens=-1, output_tokens=1)),
        ids=iter(["negative-text-usage"]),
    )

    with pytest.raises(BudgetIntegrityError, match="nonnegative integer counts"):
        provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_provider_validates_primary_counts_before_unpriced_dimensions(
    tmp_path: Path,
) -> None:
    usage = TokenUsage.model_construct(
        input_tokens=cast("int", None),
        output_tokens=1,
        total_tokens=1,
    )
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(text_usage=usage),
        ids=iter(["invalid-count-with-extra-usage"]),
    )

    with pytest.raises(BudgetIntegrityError, match="nonnegative integer counts"):
        provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_provider_forfeits_negative_chat_usage_as_terminal_integrity_error(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=-1)),
        ids=iter(["negative-chat-usage"]),
    )

    with pytest.raises(BudgetIntegrityError, match="nonnegative integer counts"):
        provider.complete_chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hello")],
                max_completion_tokens=10,
            )
        )

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_provider_forfeits_unpriceable_usage_as_terminal_integrity_error(
    tmp_path: Path,
) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(
            text_usage=TokenUsage(input_tokens=2**63, output_tokens=1),
        ),
        ids=iter(["overflow-text-usage"]),
    )

    with pytest.raises(BudgetIntegrityError, match="cannot be priced safely"):
        provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_budgeted_provider_canonicalizes_invalid_exception_class_and_reraises_original(
    tmp_path: Path,
) -> None:
    invalid_error_type = type("provider/error", (RuntimeError,), {})
    original = invalid_error_type("provider failed")
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(error=original),
        ids=iter(["invalid-error-name"]),
    )

    with pytest.raises(invalid_error_type) as captured:
        provider.complete("system", [Message(role="user", content="hello")], max_tokens=10)

    assert captured.value is original
    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UnclassifiedError"


def test_budgeted_provider_rejects_unbounded_chat_and_unpriced_embeddings(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(chat_usage=ChatUsage(prompt_tokens=1, completion_tokens=1)),
        ids=iter(["unused"]),
    )

    with pytest.raises(ValueError, match="output-token limit"):
        provider.complete_chat(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))
    with pytest.raises(RuntimeError, match="embeddings are disabled"):
        provider.embed(["hi"])
    assert ledger.reservations() == []


def test_budgeted_provider_records_usage_ceiling_violation_as_breach(tmp_path: Path) -> None:
    provider, ledger = _budgeted_provider(
        tmp_path,
        _FakeToolProvider(text_usage=TokenUsage(input_tokens=1, output_tokens=11)),
        ids=iter(["call-1"]),
    )

    with pytest.raises(BudgetBreachError, match="output-token ceiling"):
        provider.complete("", [Message(role="user", content="x")], max_tokens=10)

    [reservation] = ledger.reservations()
    assert reservation.status is ReservationStatus.BREACHED
    assert ledger.snapshot().breached


def test_budget_account_requires_an_absolute_cross_process_ledger_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        BudgetAccount(
            ledger_path=Path("relative/budget.sqlite3"),
            ledger_identity="sha256:" + "f" * 64,
            policy=_policy(),
            scope=_scope(),
            meter_id="worker",
        )

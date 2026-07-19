"""Hard-budget admission, audit, and provider-boundary metering tests."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse
from llm_waterfall.types import (
    ChatChoice,
    ChatMessage,
    ChatUsage,
)

from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetBreachError,
    BudgetedProvider,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    ReservationStatus,
    SpendLedger,
    TokenPriceCeiling,
    nano_usd_from_usd,
)


def _policy(*, hard: int = 100, search: int = 80, final: int = 20) -> BudgetPolicy:
    provider_config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model-1")
    return BudgetPolicy(
        study_id="study-1",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=hard,
        phase_limits_nano_usd={"search": search, "final": final},
        meters={
            "worker": ProviderCostMeter(
                provider_config=provider_config,
                price=TokenPriceCeiling(
                    input_nano_usd_per_token=2,
                    output_nano_usd_per_token=5,
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
) -> None:
    process_barrier = cast("_ProcessBarrier", barrier)
    process_queue = cast("_StringQueue", outcomes)
    process_barrier.wait()
    ledger = SpendLedger(path, BudgetPolicy.model_validate_json(policy_json))
    try:
        ledger.reserve(
            _scope(),
            meter_id="worker",
            max_nano_usd=60,
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


def test_spend_ledger_reserves_settles_and_forfeits_without_releasing_history(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())

    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r1")
    reserved = ledger.snapshot()
    assert reserved.charged_nano_usd == 0
    assert reserved.reserved_nano_usd == 60
    assert reserved.remaining_nano_usd == 40

    ledger.settle("r1", charged_nano_usd=20, input_tokens=3, output_tokens=4)
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=60, reservation_id="r2")
    ledger.forfeit("r2", failure_type="ConnectionError")

    snapshot = ledger.snapshot()
    assert snapshot.charged_nano_usd == 80
    assert snapshot.reserved_nano_usd == 0
    assert snapshot.remaining_nano_usd == 20
    assert snapshot.by_phase_nano_usd == {"final": 0, "search": 80}
    reservations = {item.reservation_id: item for item in ledger.reservations()}
    assert reservations["r1"].status is ReservationStatus.SETTLED
    assert reservations["r1"].input_tokens == 3
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
        ledger.settle("r1", charged_nano_usd=90, input_tokens=9, output_tokens=9)

    snapshot = ledger.snapshot()
    assert snapshot.breached
    assert snapshot.charged_nano_usd == 90
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
    ledger.settle("r1", charged_nano_usd=20, input_tokens=2, output_tokens=2)

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

    with pytest.raises(BudgetIntegrityError, match="exceeded its reservation"):
        SpendLedger(path, _policy())


def test_verified_state_does_not_release_spend_when_the_mutable_index_is_corrupted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.sqlite3"
    policy = _policy(hard=100, search=100, final=0)
    ledger = SpendLedger(path, policy)
    ledger.reserve(_scope(), meter_id="worker", max_nano_usd=80, reservation_id="r1")
    ledger.settle("r1", charged_nano_usd=80, input_tokens=8, output_tokens=8)
    assert ledger.snapshot().charged_nano_usd == 80

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM budget_reservations WHERE reservation_id = 'r1'")

    with pytest.raises(BudgetExceededError, match="remaining=20"):
        ledger.reserve(_scope(), meter_id="worker", max_nano_usd=90, reservation_id="r2")
    assert ledger.snapshot().charged_nano_usd == 80
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
    with pytest.raises(BudgetIntegrityError, match="symlink"):
        SpendLedger(symlink, _policy())


def test_exposed_policy_is_a_defensive_deep_copy(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "budget.sqlite3", _policy())
    exposed = ledger.policy
    exposed.phase_limits_nano_usd["search"] = 1
    exposed.meters["worker"].provider_config.model = "mutated"

    assert ledger.policy.phase_limits_nano_usd["search"] == 80
    assert ledger.policy.meters["worker"].provider_config.model == "model-1"


class _FakeToolProvider:
    paid_request_attempts = 1

    def __init__(
        self,
        *,
        text_usage: TokenUsage | None = None,
        chat_usage: ChatUsage | None = None,
        error: Exception | None = None,
    ) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model-1")
        self._text_usage = text_usage
        self._chat_usage = chat_usage
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
    provider: _FakeToolProvider,
    *,
    ids: Iterator[str],
) -> tuple[BudgetedProvider, SpendLedger]:
    policy = _policy(hard=10_000_000, search=10_000_000, final=0)
    path = tmp_path / "provider-budget.sqlite3"
    ledger = SpendLedger(path, policy)
    account = BudgetAccount(
        ledger_path=path,
        policy=policy,
        scope=_scope(category="provider-call"),
        meter_id="worker",
    )
    return BudgetedProvider(provider, account, id_factory=lambda: next(ids)), ledger


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
            policy=_policy(),
            scope=_scope(),
            meter_id="worker",
        )

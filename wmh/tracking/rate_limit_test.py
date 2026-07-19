"""Tests for durable external-dispatch pacing."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
from pathlib import Path
from typing import Protocol, cast

import pytest

import wmh.tracking.rate_limit as rate_limit
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    ExternalDispatchRateIntegrityError,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
    resolve_external_dispatch_rate_authority,
    validate_e2b_sandbox_create_rate_policy,
)


class _ProcessEvent(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class _IntQueue(Protocol):
    def put(self, value: int) -> None: ...


def _hold_rate_lease(path: str, ready: object, release: object) -> None:
    ready_event = cast("_ProcessEvent", ready)
    release_event = cast("_ProcessEvent", release)
    with rate_limit._exclusive_state_lease(Path(path)):  # noqa: SLF001
        ready_event.set()
        if not release_event.wait(timeout=10):
            raise RuntimeError("rate lease test timed out")


def _acquire_rate_permit(path: str, outcomes: object) -> None:
    outcome_queue = cast("_IntQueue", outcomes)
    authority = ExternalDispatchRateAuthority.bootstrap(Path(path), _policy())
    outcome_queue.put(authority.acquire().sequence)


class _Clock:
    def __init__(self, now_ns: int = 10_000_000_000) -> None:
        self.now_ns = now_ns
        self.sleeps: list[float] = []

    def time_ns(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_ns += round(seconds * 1_000_000_000)


def _policy() -> ExternalDispatchRatePolicy:
    return E2B_SANDBOX_CREATE_RATE_POLICY


def test_rate_authority_spaces_dispatches_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "rate.json"
    clock = _Clock()
    first = ExternalDispatchRateAuthority.bootstrap(
        path,
        _policy(),
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )

    receipts = [first.acquire() for _ in range(3)]
    restarted = ExternalDispatchRateAuthority.bootstrap(
        path,
        _policy(),
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )
    receipts.append(restarted.acquire())

    assert [receipt.sequence for receipt in receipts] == [1, 2, 3, 4]
    assert [receipt.admitted_at_unix_ns for receipt in receipts] == [
        10_000_000_000,
        10_250_000_000,
        10_500_000_000,
        10_750_000_000,
    ]
    assert clock.sleeps == [0.25, 0.25, 0.25]
    assert all(receipt.policy_digest == _policy().digest for receipt in receipts)


def test_rate_binding_is_path_free_and_resolves_the_same_authority(tmp_path: Path) -> None:
    authority = ExternalDispatchRateAuthority.bootstrap(tmp_path / "rate.json", _policy())

    binding = bind_external_dispatch_rate_authority(authority)
    payload = binding.model_dump_json()

    assert str(tmp_path) not in payload
    assert resolve_external_dispatch_rate_authority(binding) is authority


def test_task_and_runner_consumers_share_one_no_burst_sequence(tmp_path: Path) -> None:
    clock = _Clock()
    authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        _policy(),
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )
    binding = bind_external_dispatch_rate_authority(authority)
    task_gate = resolve_external_dispatch_rate_authority(binding)
    runner_gate = resolve_external_dispatch_rate_authority(binding)

    receipts = [
        task_gate.acquire(),
        runner_gate.acquire(),
        task_gate.acquire(),
        runner_gate.acquire(),
        task_gate.acquire(),
    ]

    assert [item.sequence for item in receipts] == [1, 2, 3, 4, 5]
    assert [item.admitted_at_unix_ns for item in receipts] == [
        10_000_000_000,
        10_250_000_000,
        10_500_000_000,
        10_750_000_000,
        11_000_000_000,
    ]
    assert clock.sleeps == [0.25, 0.25, 0.25, 0.25]


@pytest.mark.skipif(os.name != "posix", reason="rate leases require POSIX file locking")
def test_rate_authority_waits_for_a_cross_process_ledger_lease(tmp_path: Path) -> None:
    path = (tmp_path / "rate.json").resolve()
    ExternalDispatchRateAuthority.bootstrap(path, _policy())
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    holder = context.Process(target=_hold_rate_lease, args=(str(path), ready, release))
    contender = context.Process(target=_acquire_rate_permit, args=(str(path), outcomes))
    try:
        holder.start()
        assert ready.wait(timeout=5)
        contender.start()
        with pytest.raises(queue.Empty):
            outcomes.get(timeout=0.2)

        release.set()
        assert outcomes.get(timeout=5) == 1
        holder.join(timeout=5)
        contender.join(timeout=5)
        assert holder.exitcode == 0
        assert contender.exitcode == 0
    finally:
        release.set()
        for process in (holder, contender):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_rate_authority_rejects_tampered_state_before_admission(tmp_path: Path) -> None:
    path = tmp_path / "rate.json"
    clock = _Clock()
    authority = ExternalDispatchRateAuthority.bootstrap(
        path,
        _policy(),
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )
    authority.acquire()
    payload = json.loads(path.read_text())
    payload["policy_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ExternalDispatchRateIntegrityError, match="ledger|policy"):
        authority.acquire()

    assert clock.sleeps == []


def test_rate_authority_rejects_nonprivate_state_before_admission(tmp_path: Path) -> None:
    path = (tmp_path / "rate.json").resolve()
    authority = ExternalDispatchRateAuthority.bootstrap(path, _policy())
    path.chmod(0o644)

    with pytest.raises(ExternalDispatchRateIntegrityError, match="private"):
        authority.acquire()


def test_rate_authority_rejects_relative_host_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ExternalDispatchRateAuthority.bootstrap(Path("rate.json"), _policy())


def test_state_persist_closes_temporary_descriptor_when_chmod_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = rate_limit._ExternalDispatchRateState.freeze(  # noqa: SLF001
        policy_digest=_policy().digest,
        ledger_identity="sha256:" + "1" * 64,
        sequence=0,
        last_admitted_at_unix_ns=None,
    )
    observed: list[int] = []
    closed: list[int] = []
    real_close = os.close

    def fail_fchmod(descriptor: int, _mode: int) -> None:
        observed.append(descriptor)
        raise OSError("synthetic chmod failure")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(rate_limit.os, "fchmod", fail_fchmod)
    monkeypatch.setattr(rate_limit.os, "close", record_close)

    with pytest.raises(OSError, match="synthetic chmod failure"):
        rate_limit._persist_state(tmp_path / "rate.json", state)  # noqa: SLF001

    assert observed and closed == observed
    with pytest.raises(OSError):
        os.fstat(observed[0])


def test_e2b_create_policy_is_exact_and_rejects_provider_limit_drift() -> None:
    assert validate_e2b_sandbox_create_rate_policy(_policy()) == _policy()

    with pytest.raises(ValueError, match="four-per-second"):
        validate_e2b_sandbox_create_rate_policy(
            _policy().model_copy(update={"maximum_dispatches": 5})
        )


def test_rate_authority_fails_closed_on_large_clock_regression(tmp_path: Path) -> None:
    path = tmp_path / "rate.json"
    clock = _Clock()
    authority = ExternalDispatchRateAuthority.bootstrap(
        path,
        _policy(),
        clock_ns=clock.time_ns,
        sleeper=clock.sleep,
    )
    authority.acquire()
    clock.now_ns -= 2_000_000_000

    with pytest.raises(ExternalDispatchRateIntegrityError, match="clock"):
        authority.acquire()

    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", " E2B"),
        ("operation", "sandbox create"),
        ("maximum_dispatches", True),
        ("maximum_dispatches", 0),
        ("period_milliseconds", True),
        ("period_milliseconds", 0),
    ],
)
def test_rate_policy_rejects_noncanonical_or_nonpositive_inputs(
    field: str,
    value: str | int | bool,
) -> None:
    payload: dict[str, str | int | bool] = {
        "provider": "e2b",
        "operation": "sandbox_create",
        "maximum_dispatches": 4,
        "period_milliseconds": 1000,
    }
    payload[field] = value

    with pytest.raises(ValueError):
        ExternalDispatchRatePolicy.model_validate(payload)

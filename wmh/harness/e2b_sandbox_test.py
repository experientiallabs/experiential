"""Unit tests for the sandbox plumbing: creation retries, error classification, protocol slice.

No SDK import anywhere: retryable errors are classified by exception NAME and message, so a
locally-defined `RateLimitException` stands in for e2b's, and a plain fake satisfies the
`SandboxHandle` protocol structurally.
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import ModuleType

import pytest

import wmh.harness.e2b_sandbox as e2b_sandbox_module
from wmh.harness.e2b_sandbox import (
    E2BCreateRateGate,
    E2BCreateRateLimitError,
    SandboxCleanupError,
    SandboxHandle,
    create_sandbox,
    default_sandbox_factory,
    kill_sandbox,
    resolve_e2b_template,
)

_ACQUIRE_E2B_CREATE_SLOT = e2b_sandbox_module.acquire_e2b_create_slot


class RateLimitException(Exception):
    """Name-matches e2b's 429 exception; classification never imports the SDK."""


class _FakeClock:
    def __init__(self) -> None:
        self._lock = Lock()
        self.now_ns = 0
        self.sleeps: list[float] = []

    def monotonic_ns(self) -> int:
        with self._lock:
            return self.now_ns

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now_ns += round(seconds * 1_000_000_000)


@pytest.fixture(autouse=True)
def _disable_process_rate_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2b_sandbox_module, "acquire_e2b_create_slot", lambda: None)


def test_e2b_create_gate_paces_ninety_shared_concurrent_admissions() -> None:
    clock = _FakeClock()
    gate = E2BCreateRateGate(monotonic_ns=clock.monotonic_ns, sleep=clock.sleep)

    with ThreadPoolExecutor(max_workers=45) as executor:
        list(executor.map(lambda _index: gate.acquire(), range(90)))

    assert clock.now_ns == 89 * 250_000_000  # 4/sec admission schedule stays monotonic
    assert len(clock.sleeps) == 89
    assert all(delay == pytest.approx(0.25) for delay in clock.sleeps)


def test_sync_and_async_e2b_create_paths_contend_on_one_process_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    gate = E2BCreateRateGate(monotonic_ns=clock.monotonic_ns, sleep=clock.sleep)
    monkeypatch.setattr(e2b_sandbox_module, "_E2B_CREATE_RATE_GATE", gate)
    monkeypatch.setattr(
        e2b_sandbox_module,
        "acquire_e2b_create_slot",
        _ACQUIRE_E2B_CREATE_SLOT,
    )

    async def contend() -> None:
        sync_calls = [
            asyncio.to_thread(e2b_sandbox_module.acquire_e2b_create_slot) for _index in range(8)
        ]
        async_calls = [e2b_sandbox_module.acquire_e2b_create_slot_async() for _index in range(8)]
        await asyncio.gather(*sync_calls, *async_calls)

    asyncio.run(contend())

    assert clock.now_ns == 15 * 250_000_000
    assert len(clock.sleeps) == 15


def test_e2b_create_gate_rejects_an_admission_beyond_its_wait_bound() -> None:
    clock = _FakeClock()
    gate = E2BCreateRateGate(max_wait_s=0.1, monotonic_ns=clock.monotonic_ns, sleep=clock.sleep)
    gate.acquire()

    with pytest.raises(E2BCreateRateLimitError, match="0.100 seconds"):
        gate.acquire()

    assert clock.sleeps == []


class _Result:
    """A minimal CommandOutput for the protocol slice."""

    def __init__(self) -> None:
        self.stdout = ""
        self.stderr = ""
        self.exit_code = 0


class _FakeCommands:
    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> _Result:
        return _Result()

    def send_stdin(self, pid: int, data: str, request_timeout: float | None = None) -> None:
        del pid, data, request_timeout
        return None


class _FakeFiles:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def write(self, path: str, data: str) -> None:
        self.store[path] = data

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del request_timeout, gzip
        return self.store[path]


class FakeSandbox:
    """The exact `SandboxHandle` slice, satisfied structurally."""

    def __init__(self) -> None:
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.kills = 0
        self.timeouts: list[int] = []
        self.dead = False

    def set_timeout(self, timeout: int) -> None:
        if self.dead:
            raise RuntimeError("sandbox not found")
        self.timeouts.append(timeout)

    def kill(self, request_timeout: float | None = None) -> bool:
        del request_timeout
        self.kills += 1
        return True


def test_create_sandbox_retries_rate_limit_by_name_with_fixed_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    attempts: list[int] = []

    def factory() -> SandboxHandle:
        attempts.append(1)
        if len(attempts) <= 2:
            raise RateLimitException("slow down")  # classified by the exception's NAME
        return fake

    assert create_sandbox(factory) is fake
    assert len(attempts) == 3
    assert sleeps == [1.0, 3.0]


def test_create_sandbox_spends_all_three_delays_then_final_attempt_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    attempts: list[int] = []

    def factory() -> SandboxHandle:
        attempts.append(1)
        if len(attempts) <= 3:
            raise RateLimitException("still throttled")
        return fake

    assert create_sandbox(factory) is fake
    assert len(attempts) == 4  # three retried failures + the final attempt
    assert sleeps == [1.0, 3.0, 9.0]


def test_create_sandbox_exhausted_retries_propagate_the_final_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def factory() -> SandboxHandle:
        raise RateLimitException("throttled forever")

    with pytest.raises(RateLimitException, match="throttled forever"):
        create_sandbox(factory)
    assert sleeps == [1.0, 3.0, 9.0]  # every delay spent before giving up


def test_create_sandbox_retries_capacity_shaped_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    errors: list[Exception] = [
        RuntimeError("HTTP 429 from the API"),
        RuntimeError("no sandbox capacity available"),
    ]

    def factory() -> SandboxHandle:
        if errors:
            raise errors.pop(0)
        return fake

    assert create_sandbox(factory) is fake
    assert sleeps == [1.0, 3.0]


def test_create_sandbox_does_not_retry_non_capacity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    attempts: list[int] = []

    def factory() -> SandboxHandle:
        attempts.append(1)
        raise ValueError("template does not exist")

    with pytest.raises(ValueError, match="template does not exist"):
        create_sandbox(factory)
    assert len(attempts) == 1  # auth/template/config errors fail immediately
    assert sleeps == []


def test_kill_sandbox_retries_transient_errors_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    attempts = 0

    request_timeouts: list[float | None] = []

    def flaky_kill(request_timeout: float | None = None) -> bool:
        nonlocal attempts
        attempts += 1
        request_timeouts.append(request_timeout)
        if attempts < 3:
            raise RuntimeError("connection closed")
        return True

    monkeypatch.setattr(fake, "kill", flaky_kill)

    kill_sandbox(fake)

    assert attempts == 3
    assert sleeps == [0.1, 0.5]
    assert request_timeouts == [5.0, 5.0, 5.0]


def test_kill_sandbox_fails_closed_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()
    attempts = 0

    def broken_kill(request_timeout: float | None = None) -> bool:
        nonlocal attempts
        del request_timeout
        attempts += 1
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(fake, "kill", broken_kill)

    with pytest.raises(SandboxCleanupError, match="cleanup failed after 3 attempts"):
        kill_sandbox(fake)

    assert attempts == 3
    assert sleeps == [0.1, 0.5]


def test_kill_sandbox_accepts_explicit_already_gone_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = FakeSandbox()

    def gone(request_timeout: float | None = None) -> bool:
        del request_timeout
        raise RuntimeError("sandbox not found")

    monkeypatch.setattr(fake, "kill", gone)

    kill_sandbox(fake)

    assert sleeps == []


def test_default_factory_passes_metadata_to_the_lazy_e2b_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSandbox()
    calls: list[dict[str, object]] = []

    class _SandboxSdk:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            calls.append(kwargs)
            return fake

    e2b = ModuleType("e2b")
    e2b.__dict__["Sandbox"] = _SandboxSdk
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    metadata = {"kind": "optimizer-evaluator", "run_id": "run-1"}

    factory = default_sandbox_factory(
        api_key="key",
        template="tmpl",
        timeout=61.9,
        metadata=metadata,
    )

    assert factory() is fake
    assert calls == [
        {
            "template": "tmpl",
            "timeout": 61,
            "api_key": "key",
            "metadata": metadata,
        }
    ]


def test_default_factory_reacquires_rate_gate_before_every_create_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity retries re-enter the shared gate; a retry storm cannot burst the account rate."""
    fake = FakeSandbox()
    creates: list[int] = []
    admissions: list[int] = []

    class _SandboxSdk:
        @staticmethod
        def create(**_kwargs: object) -> FakeSandbox:
            creates.append(1)
            if len(creates) < 4:
                raise RateLimitException("slow down")
            return fake

    e2b = ModuleType("e2b")
    e2b.__dict__["Sandbox"] = _SandboxSdk
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    monkeypatch.setattr(
        e2b_sandbox_module,
        "acquire_e2b_create_slot",
        lambda: admissions.append(1),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    assert create_sandbox(default_sandbox_factory(api_key="key")) is fake
    assert len(creates) == 4
    assert len(admissions) == 4


def test_default_factory_snapshots_template_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """$WMH_E2B_TEMPLATE is read once per factory; "" disables the environment fallback."""
    fake = FakeSandbox()
    calls: list[dict[str, object]] = []

    class _SandboxSdk:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            calls.append(kwargs)
            return fake

    e2b = ModuleType("e2b")
    e2b.__dict__["Sandbox"] = _SandboxSdk
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "template-at-construction")
    factory = default_sandbox_factory(api_key="key")
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "template-after-construction")

    assert factory() is fake
    assert calls[0]["template"] == "template-at-construction"
    assert resolve_e2b_template("") is None


def test_fake_sandbox_satisfies_the_sandbox_handle_protocol() -> None:
    assert isinstance(FakeSandbox(), SandboxHandle)

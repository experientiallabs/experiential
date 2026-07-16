"""Unit tests for the sandbox plumbing: creation retries, error classification, protocol slice.

No SDK import anywhere: retryable errors are classified by exception NAME and message, so a
locally-defined `RateLimitException` stands in for e2b's, and a plain fake satisfies the
`SandboxHandle` protocol structurally.
"""

from __future__ import annotations

import sys
import time
from types import ModuleType

import pytest

from wmh.harness.e2b_sandbox import (
    SandboxCleanupError,
    SandboxHandle,
    create_sandbox,
    default_sandbox_factory,
    kill_sandbox,
)


class RateLimitException(Exception):
    """Name-matches e2b's 429 exception; classification never imports the SDK."""


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


def test_fake_sandbox_satisfies_the_sandbox_handle_protocol() -> None:
    assert isinstance(FakeSandbox(), SandboxHandle)

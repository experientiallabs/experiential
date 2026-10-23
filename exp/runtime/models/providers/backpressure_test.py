"""Host-wide slots hold across processes, reclaim dead holders, and breakers open and recover."""

from __future__ import annotations

import multiprocessing
import os
import signal
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from exp.common.models.client import RequestAdmissionRefused
from exp.runtime.models.providers.backpressure import (
    DEFAULT_ADMISSION_LIMITS,
    AdmissionLimits,
    AliasLimit,
    HostProviderAdmission,
    ProviderAdmissionRefused,
    ProviderRoute,
    default_admission_path,
    holder_alive,
    load_admission_limits,
)

_ONE = AdmissionLimits(provider_max_in_flight=1, alias_max_in_flight=1)
_ROUTE = ProviderRoute(provider="openrouter", alias="writer")


def _hold_slot(path: str, ready: Event, release: Event | None) -> None:
    """Child process: hold one slot until released, or forever when ``release`` is None."""
    admission = HostProviderAdmission(Path(path), _ONE)
    with admission.admit(_ROUTE, wait_seconds=5):
        ready.set()
        if release is None:
            signal.pause()
        else:
            release.wait(30)


def _spawn(path: Path, *, hold_forever: bool) -> tuple[SpawnProcess, Event, Event]:
    """Start a separate interpreter that takes the only slot and report when it holds it."""
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    child = context.Process(
        target=_hold_slot, args=(str(path), ready, None if hold_forever else release)
    )
    child.start()
    assert ready.wait(30), "child never acquired its slot"
    return child, ready, release


def test_a_slot_held_by_another_process_blocks_this_one_until_released(tmp_path: Path) -> None:
    """Two processes on one host share the provider budget through the same file."""
    path = tmp_path / "admission.sqlite"
    child, _, release = _spawn(path, hold_forever=False)
    try:
        admission = HostProviderAdmission(path, _ONE)
        assert admission.in_flight(_ROUTE) == (1, 1)
        with pytest.raises(ProviderAdmissionRefused) as refused:
            with admission.admit(_ROUTE, wait_seconds=0.2):
                pytest.fail("admitted past the other process's slot")
        assert refused.value.reason == "capacity_wait"
        assert isinstance(refused.value, RequestAdmissionRefused)
        release.set()
        child.join(30)
        with admission.admit(_ROUTE, wait_seconds=5) as slot:
            assert slot.waited_seconds == 0.0
            assert admission.in_flight(_ROUTE) == (1, 1)
        assert admission.in_flight(_ROUTE) == (0, 0)
    finally:
        release.set()
        child.join(30)


def test_a_killed_holder_slot_is_reclaimed(tmp_path: Path) -> None:
    """A SIGKILLed holder never releases, so the next claimant reclaims its slot by liveness."""
    path = tmp_path / "admission.sqlite"
    child, _, _ = _spawn(path, hold_forever=True)
    admission = HostProviderAdmission(path, _ONE)
    assert child.pid is not None
    assert admission.in_flight(_ROUTE) == (1, 1)
    os.kill(child.pid, signal.SIGKILL)
    child.join(30)
    assert not holder_alive(child.pid, None)
    with admission.admit(_ROUTE, wait_seconds=1) as slot:
        assert slot.waited_seconds == 0.0
    assert admission.in_flight(_ROUTE) == (0, 0)


def test_a_waiting_claim_proceeds_once_a_slot_frees(tmp_path: Path) -> None:
    """A contended claim polls and enters as soon as the holder releases."""
    path = tmp_path / "admission.sqlite"
    admission = HostProviderAdmission(path, _ONE)
    held = admission.admit(_ROUTE, wait_seconds=1)
    held.__enter__()
    sleeps: list[float] = []

    def release_on_first_poll(seconds: float) -> None:
        """Release the held slot during the waiting claimant's first poll."""
        sleeps.append(seconds)
        if len(sleeps) == 1:
            held.__exit__(None, None, None)

    waiting = HostProviderAdmission(path, _ONE, sleep=release_on_first_poll)
    with waiting.admit(_ROUTE, wait_seconds=5) as slot:
        assert slot.waited_seconds > 0
    assert len(sleeps) == 1


def test_alias_limit_applies_inside_the_provider_limit(tmp_path: Path) -> None:
    """One alias cannot take the whole provider budget; another alias still enters."""
    limits = AdmissionLimits(
        provider_max_in_flight=2, aliases={"writer": AliasLimit(max_in_flight=1)}
    )
    admission = HostProviderAdmission(tmp_path / "admission.sqlite", limits)
    reader = ProviderRoute(provider="openrouter", alias="reader")
    with admission.admit(_ROUTE, wait_seconds=1):
        with pytest.raises(ProviderAdmissionRefused):
            with admission.admit(_ROUTE, wait_seconds=0.05):
                pytest.fail("alias limit ignored")
        with admission.admit(reader, wait_seconds=1):
            assert admission.in_flight(reader) == (2, 1)
            with pytest.raises(ProviderAdmissionRefused):
                with admission.admit(ProviderRoute("openrouter", "third"), wait_seconds=0.05):
                    pytest.fail("provider limit ignored")
            with admission.admit(ProviderRoute("anthropic", "third"), wait_seconds=1):
                pass


def test_breaker_opens_after_consecutive_overloads_and_recovers_after_a_probe(
    tmp_path: Path,
) -> None:
    """N overloads in a row open it; after the cool-down one probe closes it on success."""
    now = [1_000.0]
    limits = AdmissionLimits(breaker_threshold=3, breaker_cooldown_seconds=30)
    admission = HostProviderAdmission(tmp_path / "admission.sqlite", limits, clock=lambda: now[0])
    for status in (429, 503):
        admission.observe("openrouter", status_code=status)
    admission.observe("openrouter", status_code=200)
    assert admission.breaker_state("openrouter") == "closed"
    for status in (429, 503, 529):
        admission.observe("openrouter", status_code=status)
    assert admission.breaker_state("openrouter") == "open"
    with pytest.raises(ProviderAdmissionRefused) as refused:
        with admission.admit(_ROUTE, wait_seconds=1):
            pytest.fail("open breaker admitted a dispatch")
    assert refused.value.reason == "circuit_open"
    assert refused.value.retry_after_seconds == 30
    with admission.admit(ProviderRoute("anthropic"), wait_seconds=1):
        pass
    now[0] += 30
    with admission.admit(_ROUTE, wait_seconds=1):
        assert admission.breaker_state("openrouter") == "probing"
        with pytest.raises(ProviderAdmissionRefused, match="circuit breaker"):
            with admission.admit(ProviderRoute("openrouter", "reader"), wait_seconds=1):
                pytest.fail("a second dispatch joined the single probe")
        admission.observe("openrouter", status_code=200)
    assert admission.breaker_state("openrouter") == "closed"
    with admission.admit(_ROUTE, wait_seconds=1):
        pass


def test_a_failed_or_unobserved_probe_reopens_and_retry_after_extends_the_cooldown(
    tmp_path: Path,
) -> None:
    """An overloaded probe reopens for max(cool-down, Retry-After); an ambiguous one reopens."""
    now = [0.0]
    limits = AdmissionLimits(breaker_threshold=1, breaker_cooldown_seconds=10)
    admission = HostProviderAdmission(tmp_path / "admission.sqlite", limits, clock=lambda: now[0])
    admission.observe("openrouter", status_code=429, retry_after_seconds=25)
    now[0] = 24
    with pytest.raises(ProviderAdmissionRefused):
        with admission.admit(_ROUTE, wait_seconds=1):
            pytest.fail("Retry-After was cut short by the shorter cool-down")
    now[0] = 25
    with admission.admit(_ROUTE, wait_seconds=1):
        admission.observe("openrouter", status_code=503)
    assert admission.breaker_state("openrouter") == "open"
    now[0] = 35
    with admission.admit(_ROUTE, wait_seconds=1):
        admission.observe("openrouter", status_code=None)
    assert admission.breaker_state("openrouter") == "open"
    now[0] = 45
    with pytest.raises(RuntimeError, match="crashed"):
        with admission.admit(_ROUTE, wait_seconds=1):
            raise RuntimeError("probe crashed before any observation")
    assert admission.breaker_state("openrouter") == "open"


def test_ambiguous_failures_neither_open_nor_reset_a_closed_breaker(tmp_path: Path) -> None:
    """Unknown outcomes carry no overload evidence, so the consecutive count is untouched."""
    limits = AdmissionLimits(breaker_threshold=2)
    admission = HostProviderAdmission(tmp_path / "admission.sqlite", limits)
    admission.observe("openrouter", status_code=429)
    admission.observe("openrouter", status_code=None)
    assert admission.breaker_state("openrouter") == "closed"
    admission.observe("openrouter", status_code=429)
    assert admission.breaker_state("openrouter") == "open"


def test_registered_limits_load_from_toml_with_conservative_defaults(tmp_path: Path) -> None:
    """An absent file means defaults; a present one overrides by provider and alias."""
    path = tmp_path / "provider-admission.toml"
    assert load_admission_limits(path) == DEFAULT_ADMISSION_LIMITS
    assert DEFAULT_ADMISSION_LIMITS.provider_limit("openai") == 8
    assert DEFAULT_ADMISSION_LIMITS.alias_limit("any") == 4
    path.write_text(
        "breaker_threshold = 2\n"
        "[providers.openrouter]\nmax_in_flight = 16\n"
        '[aliases."writer"]\nmax_in_flight = 1\n'
    )
    limits = load_admission_limits(path)
    assert limits.breaker_threshold == 2
    assert limits.provider_limit("openrouter") == 16 and limits.provider_limit("openai") == 8
    assert limits.alias_limit("writer") == 1
    path.write_text("unknown = 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_admission_limits(path)
    path.write_text("not = [valid\n")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_admission_limits(path)


def test_default_database_lives_in_the_user_data_directory(tmp_path: Path) -> None:
    """Every process on the host resolves the same file beside the credential store."""
    del tmp_path
    path = default_admission_path()
    assert path.name == "provider-admission.sqlite"
    assert path.parent.name == "exp"


def test_a_foreign_schema_is_refused_rather_than_migrated(tmp_path: Path) -> None:
    """A database of another schema version fails closed with the remedy."""
    import sqlite3

    path = tmp_path / "admission.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    with pytest.raises(ValueError, match="delete it"):
        HostProviderAdmission(path)


def test_a_per_request_503_refusal_is_definite_not_overload(tmp_path: Path) -> None:
    """A gateway 503 that refuses one exact request resets the count instead of adding to it."""
    admission = HostProviderAdmission(
        tmp_path / "admission.sqlite", AdmissionLimits(breaker_threshold=2)
    )
    for _ in range(3):
        admission.observe("openrouter", status_code=503, error_code="invalid_provider_stream")
    assert admission.breaker_state("openrouter") == "closed"
    admission.observe("openrouter", status_code=503, error_code="gateway_overloaded")
    admission.observe("openrouter", status_code=503)
    assert admission.breaker_state("openrouter") == "open"

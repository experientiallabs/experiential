"""Offline tests for orphan selection, killing, and capacity checks: no SDK, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.harness.e2b_ledger import SandboxLedger, read_ledger_files
from wmo.harness.e2b_reap import (
    DEFAULT_E2B_SANDBOX_CAP,
    E2B_SANDBOX_CAP_ENV,
    AliveSandbox,
    check_capacity,
    execute_reap,
    is_credential_error,
    plan_reap,
    sandbox_cap,
)

_NOW = datetime(2026, 7, 24, 18, 0, 0, tzinfo=UTC)


def _alive(
    sandbox_id: str,
    *,
    age_minutes: float = 10.0,
    template_id: str = "wmo-hb-v1-abc",
    trial: str | None = "task__a1b2",
    environment: str | None = "task/environment",
) -> AliveSandbox:
    metadata: dict[str, str] = {}
    if trial is not None:
        metadata["session_id"] = f"{trial}__env"
    if environment is not None:
        metadata["environment_name"] = environment
    return AliveSandbox(
        sandbox_id=sandbox_id,
        template_id=template_id,
        started_at=_NOW - timedelta(minutes=age_minutes),
        metadata=metadata,
    )


def _ledger_with(directory: Path, *, pid: int, sandbox_ids: tuple[str, ...]) -> SandboxLedger:
    ledger = SandboxLedger(directory, pid=pid, now=lambda: _NOW - timedelta(minutes=30))
    for sandbox_id in sandbox_ids:
        ledger.record_created(
            sandbox_id=sandbox_id, template_id="wmo-hb-v1-abc", trial_name=f"trial-{sandbox_id}"
        )
    return ledger


# -- cap resolution --------------------------------------------------------------------------


def test_sandbox_cap_defaults_to_the_published_account_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(E2B_SANDBOX_CAP_ENV, raising=False)
    assert sandbox_cap() == DEFAULT_E2B_SANDBOX_CAP
    monkeypatch.setenv(E2B_SANDBOX_CAP_ENV, "250")
    assert sandbox_cap() == 250


@pytest.mark.parametrize("value", ["nope", "0", "-3"])
def test_a_bad_cap_override_is_rejected_with_the_variable_name(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(E2B_SANDBOX_CAP_ENV, value)
    with pytest.raises(ValueError, match=E2B_SANDBOX_CAP_ENV):
        sandbox_cap()


# -- candidate selection ---------------------------------------------------------------------


def test_dead_owner_ledger_entries_are_candidates_and_live_owners_are_not(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1", "dead-2"))
    _ledger_with(tmp_path, pid=222, sandbox_ids=("live-1",))
    alive = [_alive("dead-1", age_minutes=200), _alive("dead-2", age_minutes=50), _alive("live-1")]

    plan = plan_reap(
        alive=alive,
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        pid_alive=lambda pid: pid == 222,
    )

    # Oldest first, and every candidate carries its evidence.
    assert [candidate.sandbox_id for candidate in plan.candidates] == ["dead-1", "dead-2"]
    first = plan.candidates[0]
    assert first.source == "ledger"
    assert first.owner_pid == 111
    assert first.owner_alive is False
    assert first.trial_name == "trial-dead-1"
    assert first.template_id == "wmo-hb-v1-abc"
    assert first.age_seconds == pytest.approx(200 * 60)
    assert first.ledger_path is not None
    assert plan.vanished == ()


def test_a_dead_owner_record_whose_sandbox_is_gone_is_released_not_killed(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("gone-1",))

    plan = plan_reap(
        alive=[],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        pid_alive=lambda _pid: False,
    )

    assert plan.candidates == ()
    [(path, record)] = plan.vanished
    assert record.sandbox_id == "gone-1"
    assert path.exists()


def test_dead_owners_can_be_turned_off(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1",))

    plan = plan_reap(
        alive=[_alive("dead-1")],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        dead_owners=False,
        pid_alive=lambda _pid: False,
    )

    assert plan.candidates == ()
    assert plan.vanished == ()


def test_stale_minutes_matches_harbor_metadata_on_the_account(tmp_path: Path) -> None:
    alive = [
        _alive("old-trial", age_minutes=90),
        _alive("young-trial", age_minutes=15),
        _alive("not-a-trial", age_minutes=600, trial=None),
        _alive("no-environment", age_minutes=600, environment=None),
    ]

    plan = plan_reap(
        alive=alive,
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        stale_minutes=60,
        pid_alive=lambda _pid: False,
    )

    [candidate] = plan.candidates
    assert candidate.sandbox_id == "old-trial"
    assert candidate.source == "metadata"
    assert candidate.owner_pid is None
    assert candidate.owner_alive is None  # no local record: liveness is unknowable
    assert candidate.trial_name == "task__a1b2"


def test_stale_minutes_never_selects_a_sandbox_whose_local_owner_is_alive(tmp_path: Path) -> None:
    """A long-running trial of the CURRENT run must survive an account-wide sweep."""
    _ledger_with(tmp_path, pid=222, sandbox_ids=("mine",))

    plan = plan_reap(
        alive=[_alive("mine", age_minutes=600)],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        stale_minutes=60,
        pid_alive=lambda pid: pid == 222,
    )

    assert plan.candidates == ()


def test_a_ledger_candidate_is_not_duplicated_by_the_metadata_sweep(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1",))

    plan = plan_reap(
        alive=[_alive("dead-1", age_minutes=600)],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        stale_minutes=60,
        pid_alive=lambda _pid: False,
    )

    assert [candidate.sandbox_id for candidate in plan.candidates] == ["dead-1"]
    assert plan.candidates[0].source == "ledger"


# -- execution -------------------------------------------------------------------------------


def test_execute_reap_kills_every_id_releases_the_ledger_and_prunes(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1", "dead-2"))
    plan = plan_reap(
        alive=[_alive("dead-1"), _alive("dead-2")],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        pid_alive=lambda _pid: False,
    )
    killed: list[str] = []

    def killer(sandbox_id: str) -> bool:
        killed.append(sandbox_id)
        return True

    outcome = execute_reap(
        plan,
        killer=killer,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid=999,
        pid_alive=lambda _pid: False,
    )

    assert sorted(killed) == ["dead-1", "dead-2"]
    assert sorted(outcome.killed) == ["dead-1", "dead-2"]
    assert outcome.freed == 2
    assert outcome.failed == ()
    # Every recorded id is released, so the owning file is pruned.
    assert len(outcome.pruned_ledgers) == 1
    assert read_ledger_files(tmp_path) == ()


def test_one_failing_kill_does_not_abort_the_sweep(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("bad", "good"))
    plan = plan_reap(
        alive=[_alive("bad", age_minutes=100), _alive("good", age_minutes=50)],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        pid_alive=lambda _pid: False,
    )

    def killer(sandbox_id: str) -> bool:
        if sandbox_id == "bad":
            raise RuntimeError("500: gateway blew up")
        return True

    outcome = execute_reap(
        plan,
        killer=killer,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid=999,
        pid_alive=lambda _pid: False,
    )

    assert outcome.killed == ("good",)
    assert outcome.failed == (("bad", "RuntimeError: 500: gateway blew up"),)
    assert outcome.freed == 1
    # The failed id stays held so a later reap retries it; the file is therefore not pruned.
    [ledger_file] = read_ledger_files(tmp_path)
    assert [record.sandbox_id for record in ledger_file.held] == ["bad"]
    assert outcome.pruned_ledgers == ()


def test_an_already_gone_candidate_frees_no_slot_but_clears_the_ledger(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("expired",))
    plan = plan_reap(
        alive=[_alive("expired")],
        ledger_files=read_ledger_files(tmp_path),
        now=_NOW,
        pid_alive=lambda _pid: False,
    )

    outcome = execute_reap(
        plan,
        killer=lambda _sandbox_id: False,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid=999,
        pid_alive=lambda _pid: False,
    )

    assert outcome.killed == ()
    assert outcome.already_gone == ("expired",)
    assert outcome.freed == 0
    assert read_ledger_files(tmp_path) == ()


def test_vanished_records_are_released_without_any_kill(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("gone",))
    plan = plan_reap(
        alive=[], ledger_files=read_ledger_files(tmp_path), now=_NOW, pid_alive=lambda _pid: False
    )

    def killer(sandbox_id: str) -> bool:
        raise AssertionError(f"must not kill {sandbox_id}")

    outcome = execute_reap(
        plan,
        killer=killer,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid=999,
        pid_alive=lambda _pid: False,
    )

    assert outcome.killed == ()
    assert len(outcome.pruned_ledgers) == 1


# -- capacity check --------------------------------------------------------------------------


def test_sufficient_free_slots_reap_nothing(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1",))

    def killer(sandbox_id: str) -> bool:
        raise AssertionError(f"must not kill {sandbox_id} when slots are free")

    check = check_capacity(
        required=4,
        cap=10,
        lister=lambda: [_alive(f"s{index}") for index in range(6)],
        killer=killer,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid_alive=lambda _pid: False,
    )

    assert check.ok is True
    assert (check.alive, check.free, check.reaped) == (6, 4, 0)
    assert check.outcome is None


def test_insufficient_slots_reap_dead_owners_and_then_pass(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1", "dead-2", "dead-3"))
    alive = [_alive(f"other-{index}") for index in range(7)] + [
        _alive("dead-1"),
        _alive("dead-2"),
        _alive("dead-3"),
    ]

    check = check_capacity(
        required=3,
        cap=10,
        lister=lambda: alive,
        killer=lambda _sandbox_id: True,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid_alive=lambda _pid: False,
    )

    assert check.ok is True
    assert (check.alive_before, check.alive, check.reaped, check.free) == (10, 7, 3, 3)
    assert read_ledger_files(tmp_path) == ()


def test_still_insufficient_after_reaping_reports_the_numbers(tmp_path: Path) -> None:
    _ledger_with(tmp_path, pid=111, sandbox_ids=("dead-1",))
    alive = [_alive(f"other-{index}") for index in range(9)] + [_alive("dead-1")]

    check = check_capacity(
        required=8,
        cap=10,
        lister=lambda: alive,
        killer=lambda _sandbox_id: True,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid_alive=lambda _pid: False,
    )

    assert check.ok is False
    assert (check.alive_before, check.alive, check.reaped, check.free) == (10, 9, 1, 1)
    assert check.required == 8


def test_capacity_check_reads_the_cap_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(E2B_SANDBOX_CAP_ENV, "12")

    check = check_capacity(
        required=2,
        lister=lambda: [_alive("s1")],
        killer=lambda _sandbox_id: True,
        ledger_directory=tmp_path,
        now=lambda: _NOW,
        pid_alive=lambda _pid: False,
    )

    assert check.cap == 12
    assert check.ok is True


def test_credential_errors_are_classified_by_name() -> None:
    """No SDK import needed: e2b's AuthenticationException is matched by class name."""

    class AuthenticationException(Exception):
        pass

    assert is_credential_error(AuthenticationException("API key is required")) is True
    assert is_credential_error(RuntimeError("connection reset")) is False

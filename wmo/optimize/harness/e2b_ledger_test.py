"""Offline tests for the E2B sandbox ledger: plain files, no SDK, no network."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.optimize.harness.e2b_ledger import (
    SandboxLedger,
    harbor_trial_name,
    ledger_dir,
    prune_released_files,
    read_ledger_files,
)


def _clock(start: datetime | None = None) -> Callable[[], datetime]:
    """A monotonic fake clock advancing one second per call."""
    base = start or datetime(2026, 7, 24, 16, 55, 12, tzinfo=UTC)
    state = {"calls": 0}

    def now() -> datetime:
        moment = base + timedelta(seconds=state["calls"])
        state["calls"] += 1
        return moment

    return now


def _ledger(directory: Path, *, pid: int = 4242) -> SandboxLedger:
    return SandboxLedger(directory, pid=pid, now=_clock())


def _dead(_pid: int) -> bool:
    """Liveness stub: the owning process is gone, so its released file may be pruned."""
    return False


def test_ledger_dir_lives_under_the_wmo_user_state_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMO_HOME", "/tmp/wmo-home-fixture")
    assert ledger_dir() == Path("/tmp/wmo-home-fixture/e2b-sandboxes")


def test_create_appends_a_record_named_by_owning_pid(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "e2b-sandboxes")

    ledger.record_created(sandbox_id="ix1", template_id="wmo-hb-v1-abc", trial_name="task__a1b2")

    assert ledger.path.name.startswith("4242-")
    assert ledger.path.name.endswith(".jsonl")
    [line] = ledger.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record == {
        "event": "created",
        "sandbox_id": "ix1",
        "template_id": "wmo-hb-v1-abc",
        "created_at": record["created_at"],
        "trial_name": "task__a1b2",
        "pid": 4242,
    }
    # The directory is owner-only: it names live billable cloud resources.
    assert ledger.path.parent.stat().st_mode & 0o777 == 0o700


def test_release_marks_the_record_and_a_fully_released_file_is_pruned(tmp_path: Path) -> None:
    directory = tmp_path / "e2b-sandboxes"
    ledger = _ledger(directory)
    ledger.record_created(sandbox_id="ix1", template_id="tpl", trial_name="t1")
    ledger.record_created(sandbox_id="ix2", template_id="tpl", trial_name="t2")

    ledger.record_released("ix1")
    [held_file] = read_ledger_files(directory)
    assert [record.sandbox_id for record in held_file.held] == ["ix2"]
    assert held_file.released_ids == ("ix1",)
    assert held_file.fully_released is False
    assert held_file.owner_pid == 4242
    assert prune_released_files(directory, owner_alive=_dead) == ()

    ledger.record_released("ix2")
    [empty_file] = read_ledger_files(directory)
    assert empty_file.held == ()
    assert empty_file.fully_released is True

    assert prune_released_files(directory, owner_alive=_dead) == (ledger.path,)
    assert not ledger.path.exists()
    assert read_ledger_files(directory) == ()


def test_a_live_owners_released_file_is_kept(tmp_path: Path) -> None:
    """The owner can append a new create at any moment; deleting under it would lose that id."""
    directory = tmp_path / "e2b-sandboxes"
    ledger = _ledger(directory)
    ledger.record_created(sandbox_id="ix1", template_id="tpl")
    ledger.record_released("ix1")

    assert prune_released_files(directory, owner_alive=lambda _pid: True) == ()
    assert ledger.path.exists()


def test_a_hard_kill_leaves_an_unreleased_record_and_a_torn_line_is_skipped(
    tmp_path: Path,
) -> None:
    """SIGKILL cannot run cleanup, so the create record must survive on its own."""
    directory = tmp_path / "e2b-sandboxes"
    ledger = _ledger(directory)
    ledger.record_created(sandbox_id="ix1", template_id="tpl", trial_name="t1")
    ledger.record_created(sandbox_id="ix2", template_id="tpl", trial_name="t2")
    # Simulate the process dying mid-append: the last line never finished.
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "created", "sandbox_id": "ix3"')

    [held_file] = read_ledger_files(directory)

    assert [record.sandbox_id for record in held_file.held] == ["ix1", "ix2"]
    assert held_file.fully_released is False


def test_a_ledger_write_failure_is_logged_and_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    ledger = _ledger(blocker / "e2b-sandboxes")

    with caplog.at_level(logging.WARNING, logger="wmo.optimize.harness.e2b_ledger"):
        ledger.record_created(sandbox_id="ix1", template_id="tpl")
        ledger.record_released("ix1")

    assert not ledger.path.exists()
    assert len(caplog.records) == 2
    assert "could not record E2B sandbox ix1" in caplog.text


def test_read_ledger_files_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert read_ledger_files(tmp_path / "absent") == ()
    assert prune_released_files(tmp_path / "absent") == ()


@pytest.mark.parametrize(
    ("session_id", "expected"),
    [
        ("hello-world__bZZeEkw__env", "hello-world__bZZeEkw"),
        ("trial__env", "trial"),
        ("__env", None),
        ("hello-world__agent", None),
        ("", None),
    ],
)
def test_harbor_trial_name_reads_the_env_session_convention(
    session_id: str, expected: str | None
) -> None:
    assert harbor_trial_name(session_id) == expected

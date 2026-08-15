"""Offline tests for the E2B sandbox ledger: plain files, no SDK, no network."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.runtime.environments import sandbox_ledger as ledger_module
from wmo.runtime.environments.sandbox_ledger import SandboxLedger, read_ledger_files


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


def test_create_appends_a_record_named_by_owning_pid(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

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


def test_created_and_released_records_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(ledger_module.os, "fsync", calls.append)
    ledger = _ledger(tmp_path)

    ledger.record_created(sandbox_id="ix1", template_id="tpl")
    ledger.record_released("ix1")

    assert len(calls) == 2


def test_release_marks_the_record_and_reports_a_fully_released_file(tmp_path: Path) -> None:
    state_directory = tmp_path / "state"
    ledger = _ledger(state_directory)
    ledger.record_created(sandbox_id="ix1", template_id="tpl", trial_name="t1")
    ledger.record_created(sandbox_id="ix2", template_id="tpl", trial_name="t2")

    ledger.record_released("ix1")
    [held_file] = read_ledger_files(state_directory)
    assert [record.sandbox_id for record in held_file.held] == ["ix2"]
    assert held_file.released_ids == ("ix1",)
    assert held_file.fully_released is False
    assert held_file.owner_pid == 4242

    ledger.record_released("ix2")
    [empty_file] = read_ledger_files(state_directory)
    assert empty_file.held == ()
    assert empty_file.fully_released is True


def test_a_hard_kill_leaves_an_unreleased_record_and_a_torn_line_is_skipped(
    tmp_path: Path,
) -> None:
    """SIGKILL cannot run cleanup, so the create record must survive on its own."""
    state_directory = tmp_path / "state"
    ledger = _ledger(state_directory)
    ledger.record_created(sandbox_id="ix1", template_id="tpl", trial_name="t1")
    ledger.record_created(sandbox_id="ix2", template_id="tpl", trial_name="t2")
    # Simulate the process dying mid-append: the last line never finished.
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "created", "sandbox_id": "ix3"')

    [held_file] = read_ledger_files(state_directory)

    assert [record.sandbox_id for record in held_file.held] == ["ix1", "ix2"]
    assert held_file.fully_released is False


def test_a_ledger_write_failure_is_logged_and_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    ledger = _ledger(blocker)

    with caplog.at_level(logging.WARNING, logger="wmo.runtime.environments.sandbox_ledger"):
        ledger.record_created(sandbox_id="ix1", template_id="tpl")
        ledger.record_released("ix1")

    assert not ledger.path.exists()
    assert len(caplog.records) == 2
    assert "could not record E2B sandbox ix1" in caplog.text


def test_read_ledger_files_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert read_ledger_files(tmp_path / "absent") == ()

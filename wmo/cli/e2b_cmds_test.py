"""CLI tests for `wmo e2b reap`, driven via CliRunner against fake account state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

import wmo.cli.e2b_cmds as e2b_cmds_module
from wmo.cli import app
from wmo.harness.e2b_ledger import SandboxLedger, read_ledger_files
from wmo.harness.e2b_reap import AliveSandbox

runner = CliRunner()

_NOW = datetime(2026, 7, 24, 18, 0, 0, tzinfo=UTC)


def _alive(
    sandbox_id: str, *, age_minutes: float, trial: str | None = "task__a1b2"
) -> AliveSandbox:
    metadata = (
        {"session_id": f"{trial}__env", "environment_name": "task/environment"}
        if trial is not None
        else {}
    )
    return AliveSandbox(
        sandbox_id=sandbox_id,
        template_id="wmo-hb-v1-abc",
        started_at=_NOW - timedelta(minutes=age_minutes),
        metadata=metadata,
    )


def _record(directory: Path, *, pid: int, sandbox_ids: tuple[str, ...]) -> Path:
    ledger = SandboxLedger(directory, pid=pid, now=lambda: _NOW - timedelta(hours=3))
    for sandbox_id in sandbox_ids:
        ledger.record_created(
            sandbox_id=sandbox_id, template_id="wmo-hb-v1-abc", trial_name=f"trial-{sandbox_id}"
        )
    return ledger.path


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger_dir: Path,
    alive: list[AliveSandbox],
    killed: list[str] | None = None,
    kill_errors: dict[str, Exception] | None = None,
) -> None:
    """Point the command at a temporary ledger and a fake account (no SDK, no network)."""
    monkeypatch.setenv("WMO_HOME", str(ledger_dir.parent))
    # Wide enough that rich renders every candidate column in full (no ellipsis).
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(e2b_cmds_module, "_now", lambda: _NOW)
    monkeypatch.setattr(e2b_cmds_module, "list_alive_sandboxes", lambda: list(alive))

    def killer(sandbox_id: str) -> bool:
        error = (kill_errors or {}).get(sandbox_id)
        if error is not None:
            raise error
        if killed is not None:
            killed.append(sandbox_id)
        return True

    monkeypatch.setattr(e2b_cmds_module, "kill_sandbox", killer)


def _flat(result: Result) -> str:
    """Collapse rich wrapping (and typer's error-box borders) for substring asserts."""
    return " ".join(result.output.replace("│", " ").split())


def _reap(*extra: str) -> Result:
    return runner.invoke(app, ["e2b", "reap", *extra])


def test_dry_run_lists_candidates_with_their_evidence_and_kills_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    _record(ledger_dir, pid=999_001, sandbox_ids=("orphan-1",))
    killed: list[str] = []
    _patch(
        monkeypatch,
        ledger_dir=ledger_dir,
        alive=[_alive("orphan-1", age_minutes=192), _alive("busy", age_minutes=5)],
        killed=killed,
    )

    result = _reap()

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "E2B usage: 2/100 concurrent sandbox(es) running" in flat
    assert "98 slot(s) free" in flat
    assert "orphan-1" in flat and "3h12m" in flat and "ledger" in flat
    assert "999001" in flat  # the recorded owner pid
    assert "dry run" in flat and "1 sandbox(es) would be killed" in flat
    assert killed == []
    # Nothing was released, so the ledger still holds the orphan.
    [ledger_file] = read_ledger_files(ledger_dir)
    assert [record.sandbox_id for record in ledger_file.held] == ["orphan-1"]


def test_yes_kills_the_candidates_and_reports_freed_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    _record(ledger_dir, pid=999_001, sandbox_ids=("orphan-1", "orphan-2"))
    killed: list[str] = []
    _patch(
        monkeypatch,
        ledger_dir=ledger_dir,
        alive=[
            _alive("orphan-1", age_minutes=200),
            _alive("orphan-2", age_minutes=100),
            _alive("busy", age_minutes=5),
        ],
        killed=killed,
    )

    result = _reap("--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert sorted(killed) == ["orphan-1", "orphan-2"]
    assert "reaped 2 sandbox(es); usage now 1/100 (99 free)" in flat
    assert "pruned 1 fully released ledger file(s)" in flat
    assert read_ledger_files(ledger_dir) == ()


def test_a_failing_kill_is_reported_and_the_rest_still_die(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    _record(ledger_dir, pid=999_001, sandbox_ids=("bad", "good"))
    killed: list[str] = []
    _patch(
        monkeypatch,
        ledger_dir=ledger_dir,
        alive=[_alive("bad", age_minutes=200), _alive("good", age_minutes=100)],
        killed=killed,
        kill_errors={"bad": RuntimeError("503: try later")},
    )

    result = _reap("--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert killed == ["good"]
    assert "reaped 1 sandbox(es)" in flat
    assert "kill failed bad: RuntimeError: 503: try later" in flat
    [ledger_file] = read_ledger_files(ledger_dir)
    assert [record.sandbox_id for record in ledger_file.held] == ["bad"]


def test_stale_minutes_adds_account_wide_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    killed: list[str] = []
    _patch(
        monkeypatch,
        ledger_dir=ledger_dir,
        alive=[
            _alive("foreign-old", age_minutes=180),
            _alive("foreign-young", age_minutes=30),
            _alive("not-a-trial", age_minutes=600, trial=None),
        ],
        killed=killed,
    )

    without = _reap()
    assert "nothing to reap" in _flat(without)

    result = _reap("--stale-minutes", "60", "--yes")

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert killed == ["foreign-old"]
    assert "metadata" in flat and "unknown" in flat  # no local owner is recorded


def test_no_dead_owners_without_stale_minutes_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    _patch(monkeypatch, ledger_dir=ledger_dir, alive=[])

    result = _reap("--no-dead-owners")

    assert result.exit_code != 0
    assert "selects nothing" in _flat(result)


def test_a_listing_failure_becomes_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WMO_HOME", str(tmp_path / ".wmo"))

    def lister() -> list[AliveSandbox]:
        raise RuntimeError("401: unauthorized")

    monkeypatch.setattr(e2b_cmds_module, "list_alive_sandboxes", lister)

    result = _reap()

    assert result.exit_code != 0
    flat = _flat(result)
    assert "could not list E2B sandboxes" in flat
    assert "$E2B_API_KEY" in flat


def test_a_missing_e2b_extra_names_the_sync_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WMO_HOME", str(tmp_path / ".wmo"))

    def lister() -> list[AliveSandbox]:
        raise ImportError("the e2b SDK is not installed; run `uv sync --extra e2b` to manage it")

    monkeypatch.setattr(e2b_cmds_module, "list_alive_sandboxes", lister)

    result = _reap()

    assert result.exit_code != 0
    assert "uv sync --extra e2b" in _flat(result)


def test_a_live_owner_is_never_a_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reaper probes real pids, so this process's own record must survive."""
    import os

    ledger_dir = tmp_path / ".wmo" / "e2b-sandboxes"
    _record(ledger_dir, pid=os.getpid(), sandbox_ids=("mine",))
    _patch(monkeypatch, ledger_dir=ledger_dir, alive=[_alive("mine", age_minutes=600)])

    result = _reap("--stale-minutes", "1")

    assert result.exit_code == 0, result.output
    assert "nothing to reap" in _flat(result)

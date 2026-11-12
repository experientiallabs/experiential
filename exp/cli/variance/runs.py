"""Run records: what ``exp variance watch`` needs to find a consented front-door run again.

A consented run writes ``runtime/front-door-cli/<digest16>.json`` in the project: the plan
digest, the registration's journal directory, the spend operation, the process, its log and
its state. A run the CLI did not launch (an operator script's) has no record; ``watch`` then
finds it from the spend ledger (``runtime/spend/front-door-cluster-<digest16>``) and the
journal whose run was planned right after that ledger's first authorization.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from exp.cli.variance.sources import OPERATION_PREFIX, journal_segments, parse_time
from exp.common.project import ProjectStore


@dataclass(frozen=True)
class RunRecord:
    """One consented front-door run.

    Attributes:
        digest: The consented plan digest.
        journal: The registration's run directory name (``front-door-<spec16>``).
        operation: The spend operation.
        spec: The spec file.
        revision: The code revision.
        pid: The process running it.
        log: Its output file, when detached.
        started: When it was consented (ISO, UTC).
        state: ``running``, ``finished``, ``stopped`` or ``failed``.
        stopped: Why it stopped, when it did.
    """

    digest: str
    journal: str
    operation: str
    spec: str
    revision: str
    pid: int
    log: str | None = None
    started: str = ""
    state: str = "running"
    stopped: str | None = None

    def alive(self) -> bool:
        """Whether its process still runs (a reused PID is not distinguished)."""
        return process_alive(self.pid)


def process_alive(pid: int | None) -> bool:
    """Whether a local process exists."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def records_directory(project: ProjectStore) -> Path:
    """Where run records live."""
    return project.paths.runtime_directory / "front-door-cli"


def save_record(project: ProjectStore, record: RunRecord) -> Path:
    """Write (or replace) one run record atomically."""
    directory = records_directory(project)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.digest[:16]}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def finish_record(
    project: ProjectStore, record: RunRecord, state: str, stopped: str | None
) -> None:
    """Record how a run ended."""
    save_record(project, replace(record, state=state, stopped=stopped))


def load_records(project: ProjectStore) -> list[RunRecord]:
    """Every run record, newest first."""
    directory = records_directory(project)
    if not directory.is_dir():
        return []
    records = []
    for path in directory.glob("*.json"):
        try:
            records.append(RunRecord(**json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda row: row.started, reverse=True)


def now_iso() -> str:
    """The current UTC time as ISO text."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Located:
    """Where one run's sources are.

    Attributes:
        name: What the view calls it (the digest's first 12, or the journal name).
        journal: The transition journal file, when found.
        since: The run's start inside the journal (its first run planned at or after this).
        ledger: The spend ledger file, when found.
        record: The CLI run record, when the CLI launched it.
    """

    name: str
    journal: Path | None
    since: datetime | None
    ledger: Path | None
    record: RunRecord | None = None


def locate(project: ProjectStore, run: str | None) -> Located:
    """Find a run by plan digest (or prefix), journal name, or the newest when ``run`` is None.

    Raises:
        ValueError: Nothing matches.
    """
    runtime = project.paths.runtime_directory
    journals = runtime / "front-door"
    spend = runtime / "spend"
    records = load_records(project)
    if run is None:
        if records:
            return _from_record(project, records[0])
        candidates = sorted(journals.glob("front-door-*/transitions.jsonl"), key=_mtime)
        if not candidates:
            raise ValueError("no front-door run in this project yet")
        journal = candidates[-1]
        return Located(journal.parent.name, journal, None, _ledger_for(spend, journal, None))
    key = run.strip().lower().removeprefix(OPERATION_PREFIX)
    for record in records:
        if record.digest.startswith(key) or record.journal == run:
            return _from_record(project, record)
    if run.startswith("front-door-") and (journals / run / "transitions.jsonl").exists():
        journal = journals / run / "transitions.jsonl"
        return Located(run, journal, None, _ledger_for(spend, journal, None))
    ledgers = [path for path in spend.glob(f"{OPERATION_PREFIX}{key[:16]}*/ledger.sqlite")]
    if len(ledgers) == 1:
        ledger = ledgers[0]
        authorized = _first_authorization(ledger)
        journal = _journal_for(journals, authorized)
        return Located(key[:12], journal, authorized, ledger)
    if len(ledgers) > 1:
        raise ValueError(f"{run} names {len(ledgers)} runs; give more of the digest")
    raise ValueError(f"no front-door run matches {run}")


def _from_record(project: ProjectStore, record: RunRecord) -> Located:
    """Sources of a CLI-launched run."""
    runtime = project.paths.runtime_directory
    journal = runtime / "front-door" / record.journal / "transitions.jsonl"
    return Located(
        record.digest[:12],
        journal,
        parse_time(record.started),
        runtime / "spend" / record.operation / "ledger.sqlite",
        record,
    )


def _mtime(path: Path) -> float:
    """A file's modification time."""
    return path.stat().st_mtime


def _first_authorization(ledger: Path) -> datetime | None:
    """When the ledger's first consent was recorded."""
    from exp.cli.variance.sources import read_only

    try:
        with read_only(ledger) as db:
            row = db.execute("SELECT min(authorized_at) FROM authorizations").fetchone()
    except Exception:  # noqa: BLE001 - an unreadable ledger just matches nothing
        return None
    return parse_time(row[0]) if row else None


def _journal_for(journals: Path, authorized: datetime | None) -> Path | None:
    """The journal holding the first run planned at or after a consent."""
    if authorized is None:
        return None
    best: tuple[datetime, Path] | None = None
    for path in journals.glob("front-door-*/transitions.jsonl"):
        if _mtime(path) < authorized.timestamp():
            continue
        for segment in journal_segments(path):
            if segment.planned is not None and segment.planned >= authorized:
                if best is None or segment.planned < best[0]:
                    best = (segment.planned, path)
                break
    return None if best is None else best[1]


def _ledger_for(spend: Path, journal: Path, since: datetime | None) -> Path | None:
    """The ledger whose first consent came last before the journal's run was planned."""
    segments = journal_segments(journal)
    if not segments:
        return None
    target = (
        segments[-1]
        if since is None
        else next((row for row in segments if row.planned and row.planned >= since), segments[-1])
    )
    if target.planned is None:
        return None
    best: tuple[datetime, Path] | None = None
    for ledger in spend.glob(f"{OPERATION_PREFIX}*/ledger.sqlite"):
        authorized = _first_authorization(ledger)
        if authorized is None or authorized > target.planned:
            continue
        if best is None or authorized > best[0]:
            best = (authorized, ledger)
    return None if best is None else best[1]


def spawn_detached(arguments: list[str], log: Path) -> int:
    """Start ``python -m exp ARGUMENTS`` in its own session, output to ``log``; return its PID.

    The child has no terminal and its own process group, so closing this terminal (SIGHUP) or
    pressing Ctrl-C in a later watch does not stop it.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        child = subprocess.Popen(  # noqa: S603 - fixed interpreter and argument vector
            [sys.executable, "-m", "exp", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return child.pid

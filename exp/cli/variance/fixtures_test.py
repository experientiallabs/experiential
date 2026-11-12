"""Recorded-run fixtures shared by the ``exp variance`` tests: a journal and a spend ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

START = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)


def at(minutes: float) -> str:
    """An ISO time ``minutes`` after the fixture start."""
    return (START + timedelta(minutes=minutes)).isoformat()


def write_journal(path: Path, rows: list[tuple[float, str, str, str | None, str, str]]) -> Path:
    """Write a transition journal from ``(minute, machine, subject, from, event, to)`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    machines = {"run": "front-door-run", "phase": "front-door-phase"}
    with path.open("w", encoding="utf-8") as stream:
        for minute, machine, subject, before, event, after in rows:
            row = {
                "at": at(minute),
                "event": event,
                "from": before,
                "machine": machines[machine],
                "refused": False,
                "subject": subject,
                "to": after,
            }
            stream.write(json.dumps(row) + "\n")
    return path


RUN = "front-door-00aa11bb22cc33dd"
TASK = "task-307a285239d7acd18b5d"

RECORDED = [
    (0, "run", RUN, None, "planned", "planned"),
    (0, "phase", "project-build", None, "done", "done"),
    (0, "phase", "variance-build", None, "done", "done"),
    (0, "phase", "slot tag-banks", None, "planned", "planned"),
    (0, "phase", "bootstrap-preparation", None, "blocked", "blocked"),
    (0, "phase", "pipelines", None, "blocked", "blocked"),
    (0, "phase", "slot dense-hybrid", None, "planned", "planned"),
    (0, "run", RUN, "planned", "consented", "consented"),
    (0, "run", RUN, "consented", "started", "running"),
    (1, "phase", "slot tag-banks", "planned", "approved", "approved"),
    (1, "phase", "slot tag-banks", "approved", "started", "running"),
    (31, "phase", "slot tag-banks", "running", "finished", "done"),
    (31, "phase", "bootstrap-preparation", "blocked", "priced", "planned"),
    (31, "phase", "bootstrap-preparation", "planned", "approved", "approved"),
    (31, "phase", "bootstrap-preparation", "approved", "started", "running"),
    (60, "phase", "bootstrap-preparation", "running", "finished", "done"),
    (60, "phase", "pipelines", "blocked", "replayed", "done"),
    (61, "phase", f"pipeline {TASK}", None, "planned", "planned"),
    (61, "phase", f"pipeline {TASK}", "planned", "approved", "approved"),
    (61, "phase", f"pipeline {TASK}", "approved", "started", "running"),
]
"""A run that finished tag-banks and the preparation and is inside its first start's pipeline."""


def write_ledger(path: Path, calls: list[tuple[str, str, float]], allowance: int = 1000) -> Path:
    """Write a spend ledger with ``(role, state, minute)`` reservations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE authorizations (seq INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT, "
            "ceiling_usd REAL, unpriced_attempts INTEGER, authorized_at TEXT)"
        )
        db.execute(
            "CREATE TABLE reservations (id TEXT PRIMARY KEY, seq INTEGER UNIQUE, role TEXT, "
            "priced INTEGER, reserved_usd REAL, charge_usd REAL, state TEXT, overrun INTEGER, "
            "attempt_id TEXT, machine TEXT, pid INTEGER, start TEXT, reserved_at TEXT)"
        )
        db.execute(
            "INSERT INTO authorizations (stage, ceiling_usd, unpriced_attempts, authorized_at) "
            "VALUES ('a' || hex(randomblob(8)), 0, 0, ?)",
            (at(-1),),
        )
        db.execute(
            "INSERT INTO authorizations (stage, ceiling_usd, unpriced_attempts, authorized_at) "
            "VALUES ('operation-unpriced-allowance', 0, ?, ?)",
            (allowance, at(-1)),
        )
        for seq, (role, state, minute) in enumerate(calls, start=1):
            db.execute(
                "INSERT INTO reservations VALUES (?, ?, ?, 0, 0, 0, ?, 0, NULL, 'm', 999999999, "
                "NULL, ?)",
                (f"r{seq}", seq, role, state, at(minute)),
            )
    return path

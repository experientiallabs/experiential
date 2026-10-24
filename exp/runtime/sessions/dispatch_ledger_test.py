"""Offline tests for the invocation ledger: cap, per-invocation slots, durability, and schema."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from exp.runtime.sessions.dispatch_ledger import (
    DispatchCapExhaustedError,
    DispatchLedger,
    LedgerSchemaError,
)

NOW = "2026-10-10T12:00:00+00:00"
PHASE = "a" * 64


def test_claims_up_to_cap_then_refuses_without_writing(tmp_path: Path) -> None:
    """Exactly ``cap`` slots are granted; the next claim is refused and writes nothing."""
    ledger = DispatchLedger(tmp_path / "ledger.sqlite", cap=2, phase_config_sha256=PHASE)
    assert ledger.claim(NOW)[0] == 1
    assert ledger.claim(NOW)[0] == 2

    with pytest.raises(DispatchCapExhaustedError):
        ledger.claim(NOW)

    assert ledger.used() == 2


def test_each_invocation_gets_a_fresh_attempt_id_and_its_own_sequence(tmp_path: Path) -> None:
    """Two claims never share an attempt ID or a sequence number, even at identical timestamps."""
    ledger = DispatchLedger(tmp_path / "ledger.sqlite", cap=6, phase_config_sha256=PHASE)
    first_seq, first_attempt = ledger.claim(NOW)
    second_seq, second_attempt = ledger.claim(NOW)

    assert second_seq == first_seq + 1
    assert first_attempt != second_attempt
    assert len(first_attempt) == 32


def test_refused_claim_writes_nothing_so_repeated_refusals_stay_refusals(tmp_path: Path) -> None:
    """A refused claim does not increment the count, so further claims are still refused."""
    ledger = DispatchLedger(tmp_path / "ledger.sqlite", cap=1, phase_config_sha256=PHASE)
    ledger.claim(NOW)
    for _ in range(3):
        with pytest.raises(DispatchCapExhaustedError):
            ledger.claim(NOW)
    assert ledger.used() == 1


def test_slots_survive_reopen_and_count_against_the_cap(tmp_path: Path) -> None:
    """A new process opening the same file sees every earlier claim, failed ones included."""
    path = tmp_path / "ledger.sqlite"
    DispatchLedger(path, cap=2, phase_config_sha256=PHASE).claim(NOW)
    DispatchLedger(path, cap=2, phase_config_sha256=PHASE).claim(NOW)

    reopened = DispatchLedger(path, cap=2, phase_config_sha256=PHASE)
    assert reopened.used() == 2
    with pytest.raises(DispatchCapExhaustedError):
        reopened.claim(NOW)


def test_claimed_slot_is_durable_before_any_call_could_start(tmp_path: Path) -> None:
    """The slot row is committed as soon as claim returns, so a later crash still counts it."""
    path = tmp_path / "ledger.sqlite"
    seq, attempt = DispatchLedger(path, cap=3, phase_config_sha256=PHASE).claim(NOW)

    raw = sqlite3.connect(path)
    try:
        rows = raw.execute(
            "SELECT seq, attempt_id, phase_config_sha256, receipt_state FROM invocation_slots"
        ).fetchall()
    finally:
        raw.close()
    assert rows == [(seq, attempt, PHASE, "claimed")]


def test_nonpositive_cap_and_malformed_phase_digest_are_refused(tmp_path: Path) -> None:
    """A zero cap or a non-hex phase digest is a configuration error."""
    with pytest.raises(ValueError, match="cap"):
        DispatchLedger(tmp_path / "a.sqlite", cap=0, phase_config_sha256=PHASE)
    with pytest.raises(ValueError, match="SHA-256"):
        DispatchLedger(tmp_path / "b.sqlite", cap=1, phase_config_sha256="not-a-digest")


def test_foreign_schema_is_refused_rather_than_migrated(tmp_path: Path) -> None:
    """A database with another schema version is refused, never silently upgraded."""
    path = tmp_path / "foreign.sqlite"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()

    with pytest.raises(LedgerSchemaError):
        DispatchLedger(path, cap=1, phase_config_sha256=PHASE)


def test_concurrent_claims_never_exceed_the_cap(tmp_path: Path) -> None:
    """Threads racing for slots grant exactly ``cap`` of them, never more."""
    path = tmp_path / "ledger.sqlite"
    DispatchLedger(path, cap=4, phase_config_sha256=PHASE)
    granted: list[int] = []
    refused: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def attempt() -> None:
        """Claim one slot and record whether it was granted."""
        ledger = DispatchLedger(path, cap=4, phase_config_sha256=PHASE)
        barrier.wait()
        try:
            seq, _ = ledger.claim(NOW)
            with lock:
                granted.append(seq)
        except DispatchCapExhaustedError:
            with lock:
                refused.append("cap")

    threads = [threading.Thread(target=attempt) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(granted) == 4
    assert len(refused) == 8
    assert DispatchLedger(path, cap=4, phase_config_sha256=PHASE).used() == 4


def test_version_zero_foreign_database_is_refused_and_left_untouched(tmp_path: Path) -> None:
    """A version-0 SQLite file that holds someone else's tables is refused, not adopted."""
    path = tmp_path / "foreign.sqlite"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
    raw.execute("INSERT INTO customers (name) VALUES ('keep-me')")
    raw.commit()
    raw.close()

    with pytest.raises(LedgerSchemaError):
        DispatchLedger(path, cap=3, phase_config_sha256=PHASE)

    check = sqlite3.connect(path)
    try:
        tables = sorted(
            r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
        assert tables == ["customers"]
        assert check.execute("PRAGMA user_version").fetchone()[0] == 0
        assert check.execute("SELECT name FROM customers").fetchone()[0] == "keep-me"
    finally:
        check.close()


def test_fresh_empty_database_is_still_accepted(tmp_path: Path) -> None:
    """A new file with no tables gets the ledger schema, as before."""
    ledger = DispatchLedger(tmp_path / "fresh.sqlite", cap=2, phase_config_sha256=PHASE)
    assert ledger.claim(NOW)[0] == 1


@pytest.mark.parametrize("changed_cap,changed_phase", [(2, PHASE), (1, "b" * 64)])
def test_reopen_cannot_change_registered_authority(
    tmp_path: Path, changed_cap: int, changed_phase: str
) -> None:
    """A persisted phase cannot acquire new call authority by changing its constructor."""
    path = tmp_path / "ledger.sqlite"
    original = DispatchLedger(path, cap=1, phase_config_sha256=PHASE)
    original.claim(NOW)
    with pytest.raises(LedgerSchemaError, match="registration"):
        DispatchLedger(path, cap=changed_cap, phase_config_sha256=changed_phase)
    assert original.used() == 1


def test_registration_is_frozen_before_the_first_invocation(tmp_path: Path) -> None:
    """An empty ledger still binds its phase and cap before any provider dispatch."""
    path = tmp_path / "ledger.sqlite"
    original = DispatchLedger(path, cap=1, phase_config_sha256=PHASE)
    with pytest.raises(LedgerSchemaError, match="registration"):
        DispatchLedger(path, cap=2, phase_config_sha256=PHASE)
    assert original.used() == 0


@pytest.mark.parametrize("old_version", [1, 2])
def test_old_ledger_is_refused_without_changing_its_receipts(
    tmp_path: Path, old_version: int
) -> None:
    """A stale persisted schema is an explicit fresh-phase requirement, never migrated."""
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE invocation_slots (attempt_id TEXT)")
        connection.execute("INSERT INTO invocation_slots VALUES ('preserve')")
        connection.execute(f"PRAGMA user_version = {old_version}")
    with pytest.raises(LedgerSchemaError, match="fresh ledger"):
        DispatchLedger(path, cap=1, phase_config_sha256=PHASE)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == old_version
        assert connection.execute("SELECT attempt_id FROM invocation_slots").fetchall() == [
            ("preserve",)
        ]


def test_boolean_cap_cannot_create_a_registration(tmp_path: Path) -> None:
    """JSON true is not permission to dispatch one provider request."""
    path = tmp_path / "ledger.sqlite"
    with pytest.raises(ValueError, match="positive integer"):
        DispatchLedger(path, cap=True, phase_config_sha256=PHASE)
    assert not path.exists()


@pytest.mark.parametrize(
    "statement",
    ["DELETE FROM invocation_slots", "UPDATE invocation_slots SET claimed_at = 'changed'"],
)
def test_invocation_slots_cannot_be_edited_to_restore_authority(
    tmp_path: Path, statement: str
) -> None:
    """SQL edits cannot erase a spent slot or rewrite the original invocation receipt."""
    path = tmp_path / "ledger.sqlite"
    ledger = DispatchLedger(path, cap=1, phase_config_sha256=PHASE)
    ledger.claim(NOW)
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(statement)
        assert connection.execute("SELECT claimed_at FROM invocation_slots").fetchone()[0] == NOW
    assert ledger.used() == 1
    with pytest.raises(DispatchCapExhaustedError):
        ledger.claim(NOW)

"""Scoped durable gateway capture reader regressions."""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.runtime.gateway.local_capture_contracts import (
    CapturedExchange,
    CaptureProvenance,
    LocalCaptureScope,
)
from exp.runtime.gateway.local_capture_store import LocalCaptureStore


def test_reader_filters_scope_and_retention_and_resumes_cursor(tmp_path: Path) -> None:
    """A resumed consumer cannot read another application or expired content."""
    path = tmp_path / "capture.db"
    scope = LocalCaptureScope(user_id="user", application_id="app")
    experience = CapturedExchange(
        experience_id="exp-one",
        response_id="response-one",
        scope=scope,
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=CaptureProvenance(source_id="one", model_id="model"),
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY, "
        "user_id TEXT, application_id TEXT, expires_at INTEGER, payload TEXT)"
    )
    for sequence, application, expires in [
        (1, "app", 9999999999),
        (2, "other", 9999999999),
        (3, "app", 1),
    ]:
        connection.execute(
            "INSERT INTO gateway_captures VALUES (?, ?, ?, ?, ?)",
            (sequence, "user", application, expires, experience.model_dump_json()),
        )
    connection.commit()
    connection.close()
    store = LocalCaptureStore(path, scope)
    assert [row.sequence for row in store.read_after()] == [1]
    assert store.read_after(1) == ()


def _seed_capture(path: Path, scope: LocalCaptureScope, experience_id: str) -> None:
    """Write one live scoped row into a fresh capture database at `path`."""
    experience = CapturedExchange(
        experience_id=experience_id,
        response_id=f"response-{experience_id}",
        scope=scope,
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=CaptureProvenance(source_id="one", model_id="model"),
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY, "
        "user_id TEXT, application_id TEXT, expires_at INTEGER, payload TEXT)"
    )
    connection.execute(
        "INSERT INTO gateway_captures VALUES (?, ?, ?, ?, ?)",
        (1, scope.user_id, scope.application_id, 9999999999, experience.model_dump_json()),
    )
    connection.commit()
    connection.close()


def test_reader_refuses_a_database_path_that_is_a_symlink(tmp_path: Path) -> None:
    """A final-entry symlink is refused rather than followed to another database."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    elsewhere = tmp_path / "elsewhere.db"
    _seed_capture(elsewhere, scope, "not-ours")
    link = tmp_path / "capture.db"
    link.symlink_to(elsewhere)

    store = LocalCaptureStore(link, scope)

    for read in (store.read_after, store.read_snapshot, lambda: tuple(store.iter_snapshot())):
        with pytest.raises(ValueError, match="regular file"):
            read()


def test_reader_refuses_a_database_replaced_by_a_symlink_after_binding(
    tmp_path: Path,
) -> None:
    """The window the constructor cannot close: swapped between binding and reading."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    path = tmp_path / "capture.db"
    _seed_capture(path, scope, "ours")
    elsewhere = tmp_path / "elsewhere.db"
    _seed_capture(elsewhere, scope, "not-ours")
    store = LocalCaptureStore(path, scope)
    assert [row.experience.experience_id for row in store.read_after()] == ["ours"]

    path.unlink()
    path.symlink_to(elsewhere)

    with pytest.raises(ValueError, match="regular file"):
        store.read_after()


def test_reader_follows_a_symlinked_directory_to_the_named_file(tmp_path: Path) -> None:
    """Only the final entry is refused: an aliased parent directory still resolves."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    _seed_capture(real_directory / "capture.db", scope, "ours")
    aliased = tmp_path / "aliased"
    aliased.symlink_to(real_directory)

    store = LocalCaptureStore(aliased / "capture.db", scope)

    assert [row.experience.experience_id for row in store.read_after()] == ["ours"]


def test_reader_refuses_a_dangling_symlink_rather_than_reading_it_as_absent(
    tmp_path: Path,
) -> None:
    """A link to nothing is still a link, and must not be mistaken for no file."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    link = tmp_path / "capture.db"
    link.symlink_to(tmp_path / "never-created.db")

    store = LocalCaptureStore(link, scope)

    with pytest.raises(ValueError, match="regular file"):
        store.read_after()


def test_reader_fails_closed_when_the_entry_is_swapped_around_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-to-use race: validated, then replaced before SQLite resolves the name.

    The guard cannot hand SQLite its descriptor, so the two opens resolve the
    pathname separately. This drives the window directly by swapping the entry
    for a link to another database at the moment of connect, and asserts the
    reader refuses rather than serving the substitute's rows.
    """
    scope = LocalCaptureScope(user_id="user", application_id="app")
    path = tmp_path / "capture.db"
    _seed_capture(path, scope, "ours")
    elsewhere = tmp_path / "elsewhere.db"
    _seed_capture(elsewhere, scope, "not-ours")
    store = LocalCaptureStore(path, scope)
    real_connect = sqlite3.connect

    def swap_then_connect(database: str, **kwargs: object) -> sqlite3.Connection:
        path.unlink()
        path.symlink_to(elsewhere)
        return real_connect(database, uri=bool(kwargs.get("uri")), timeout=1.0)

    monkeypatch.setattr(sqlite3, "connect", swap_then_connect)

    with pytest.raises(ValueError, match="was replaced"):
        store.read_after()


def test_reader_fails_closed_when_the_swap_is_reverted_around_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The swap that undoes itself, which the entry's own identity cannot catch.

    An attacker who puts a substitute in place for SQLite's open and restores
    the original before the check leaves the pinned identity intact, so the
    entry comparison passes. The directory is the only remaining witness: its
    change time moved and cannot be set back.
    """
    scope = LocalCaptureScope(user_id="user", application_id="app")
    path = tmp_path / "capture.db"
    _seed_capture(path, scope, "ours")
    elsewhere = tmp_path / "elsewhere.db"
    _seed_capture(elsewhere, scope, "not-ours")
    store = LocalCaptureStore(path, scope)
    real_connect = sqlite3.connect
    held = tmp_path / "held.db"

    def swap_revert_then_connect(database: str, **kwargs: object) -> sqlite3.Connection:
        pinned_before = path.stat()
        os.rename(path, held)
        os.rename(elsewhere, path)
        connection = real_connect(database, uri=bool(kwargs.get("uri")), timeout=1.0)
        os.rename(path, elsewhere)
        os.rename(held, path)
        # The entry check alone cannot see this: the name is back on the inode
        # that was pinned, so only the directory still carries the evidence.
        assert (path.stat().st_dev, path.stat().st_ino) == (
            pinned_before.st_dev,
            pinned_before.st_ino,
        )
        return connection

    monkeypatch.setattr(sqlite3, "connect", swap_revert_then_connect)

    with pytest.raises(ValueError, match="was replaced"):
        store.read_after()


def test_reader_refuses_an_absent_database_without_opening_the_name(
    tmp_path: Path,
) -> None:
    """An entry that appears after the check must not be opened unvalidated."""
    scope = LocalCaptureScope(user_id="user", application_id="app")

    store = LocalCaptureStore(tmp_path / "absent.db", scope)

    with pytest.raises(ValueError, match="not present"):
        store.read_after()
    assert not (tmp_path / "absent.db").exists()


def test_reader_without_no_follow_opens_still_refuses_a_link_and_reads_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no no-follow open or directory descriptor, so it takes the portable path.

    That path cannot pin anything, so it cannot witness a replacement race. It
    must still refuse the substitution the reader exists to catch, and must
    still serve an ordinary database.
    """
    monkeypatch.setattr(os, "name", "nt")
    scope = LocalCaptureScope(user_id="user", application_id="app")
    elsewhere = tmp_path / "elsewhere.db"
    _seed_capture(elsewhere, scope, "not-ours")
    link = tmp_path / "linked.db"
    link.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="regular file"):
        LocalCaptureStore(link, scope).read_after()

    with pytest.raises(ValueError, match="not present"):
        LocalCaptureStore(tmp_path / "absent.db", scope).read_after()

    ordinary = tmp_path / "capture.db"
    _seed_capture(ordinary, scope, "ours")
    rows = LocalCaptureStore(ordinary, scope).read_after()
    assert [row.experience.experience_id for row in rows] == ["ours"]


def test_reader_serves_an_unswapped_database_normally(tmp_path: Path) -> None:
    """The identity check must not refuse the ordinary case it guards."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    path = tmp_path / "capture.db"
    _seed_capture(path, scope, "ours")

    store = LocalCaptureStore(path, scope)

    assert [row.experience.experience_id for row in store.read_after()] == ["ours"]
    assert [row.experience.experience_id for row in store.read_snapshot()] == ["ours"]
    assert [row.experience.experience_id for row in store.iter_snapshot()] == ["ours"]

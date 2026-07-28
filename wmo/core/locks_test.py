"""Tests for the cross-process write lock.

The lost-update guarantee is what this module exists for, and atomic replace does not provide it,
so every test here is about two writers rather than about one failing.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from wmo.core.files import write_text_atomic
from wmo.core.locks import FileLockTimeout, file_write_lock


def test_file_write_lock_reports_a_stuck_holder_instead_of_hanging(tmp_path: Path) -> None:
    path = tmp_path / "roster.toml"
    lock_path = path.with_name("roster.toml.lock")
    holder = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(FileLockTimeout, match=r"writing the roster at .*roster\.toml"):
            with file_write_lock(path, what="the roster", timeout_s=0.05):
                pass  # pragma: no cover - the lock is held, so this is never reached
    finally:
        os.close(holder)

    # Once the holder is gone the lock is free again: the kernel owns that release, so a leftover
    # lock FILE is never a held lock and cannot wedge the next run.
    with file_write_lock(path, what="the roster", timeout_s=0.05):
        pass


def test_file_write_lock_releases_on_an_exception_inside_the_block(tmp_path: Path) -> None:
    """A rejected write must not leave the file locked, or one bad input wedges every later run."""
    path = tmp_path / "roster.toml"

    with pytest.raises(ValueError, match="rejected"):
        with file_write_lock(path, what="the roster", timeout_s=0.05):
            raise ValueError("rejected")

    with file_write_lock(path, what="the roster", timeout_s=0.05):
        pass


def test_file_write_lock_serializes_a_read_modify_write(tmp_path: Path) -> None:
    """The lost-update guarantee: atomic replace alone does NOT give you this.

    Both threads read, edit, and write back the same file. Unlocked, each reads the same starting
    contents and the later write erases the earlier thread's line, with neither raising. The sleep
    inside the critical section makes that interleaving certain rather than timing-dependent.
    """
    path = tmp_path / "roster.toml"
    write_text_atomic(path, "")

    def _append(line: str) -> None:
        with file_write_lock(path, what="the roster"):
            current = path.read_text(encoding="utf-8")
            time.sleep(0.05)  # hold the section open so an unlocked version would interleave
            write_text_atomic(path, f"{current}{line}\n")

    writers = [threading.Thread(target=_append, args=(name,)) for name in ("a", "b")]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)
        assert not writer.is_alive(), "the lock wait is not bounded"

    assert sorted(path.read_text(encoding="utf-8").split()) == ["a", "b"]


def test_file_write_lock_does_not_lock_the_file_being_written(tmp_path: Path) -> None:
    """The lock has to live in a sibling, because an atomic write swaps the file's inode.

    A lock taken on the roster itself would stop protecting anything the moment the first writer
    renamed a new inode into place, and the failure would be silent.
    """
    path = tmp_path / "roster.toml"

    with file_write_lock(path, what="the roster"):
        write_text_atomic(path, "payload")

    assert path.with_name("roster.toml.lock").is_file()
    assert path.read_text(encoding="utf-8") == "payload"

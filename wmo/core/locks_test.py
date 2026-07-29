"""Tests for the cross-process write lock.

The lost-update guarantee is what this module exists for, and atomic replace does not provide it,
so every test here is about two writers rather than about one failing. Contending locks are taken
through `filelock` rather than a raw `fcntl.flock`, so these tests exercise the same backend the
module picks on whatever platform they run on.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock

from wmo.core.files import write_text_atomic
from wmo.core.locks import FileLockTimeout, file_write_lock


def test_file_write_lock_reports_a_stuck_holder_instead_of_hanging(tmp_path: Path) -> None:
    path = tmp_path / "roster.toml"
    holder = FileLock(path.with_name("roster.toml.lock"))
    holder.acquire()
    try:
        with pytest.raises(FileLockTimeout, match=r"writing the roster at .*roster\.toml"):
            with file_write_lock(path, what="the roster", timeout_s=0.05):
                pass  # pragma: no cover - the lock is held, so this is never reached
    finally:
        holder.release()

    # Once the holder is gone the lock is free again: the OS owns that release, so a leftover lock
    # FILE is never a held lock and cannot wedge the next run.
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


def test_file_write_lock_is_not_reentrant_and_says_so_within_the_bound(tmp_path: Path) -> None:
    """Nesting on one path deadlocks against itself, so it must TIME OUT rather than hang.

    The docstring warns against it; this pins that the warning is the whole story, and that the
    obvious next feature (promoting two aliases together by wrapping a locked helper twice) fails
    fast instead of wedging a terminal.
    """
    path = tmp_path / "roster.toml"

    with file_write_lock(path, what="the roster", timeout_s=0.05):
        with pytest.raises(FileLockTimeout):
            with file_write_lock(path, what="the roster", timeout_s=0.05):
                pass  # pragma: no cover - the outer lock is held


def test_two_symlinks_to_one_target_contend_on_a_single_lock(tmp_path: Path) -> None:
    """A shared roster symlinked from two project directories is ONE file to be serialized.

    The lock file is derived from the resolved write target, not from the path as given. Keyed off
    the link instead, each caller takes its own lock, both proceed, and each loses the other's
    update: precisely the failure the lock exists to prevent, and reachable only once writes follow
    symlinks through to their target.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "pool.toml"
    target.write_text("roster", encoding="utf-8")
    first = tmp_path / "project-a.toml"
    first.symlink_to(target)
    second = tmp_path / "project-b.toml"
    second.symlink_to(target)

    with file_write_lock(first, what="the model pool", timeout_s=0.05):
        with pytest.raises(FileLockTimeout):
            with file_write_lock(second, what="the model pool", timeout_s=0.05):
                pass  # pragma: no cover - the first lock is held

    assert [item.name for item in shared.glob("*.lock")] == ["pool.toml.lock"]
    assert list(tmp_path.glob("*.lock")) == [], "a lock was keyed off the symlink, not the target"

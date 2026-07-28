"""Tests for the durable-write helpers: atomic replace and the cross-process write lock.

Every assertion is about what survives a FAILURE, because that is the whole point of the module:
the happy path of any of these writers was already correct. Torn writes are modelled by failing
`os.fsync`, which is reached with the payload already on the temp file and the rename not yet
done, so it is the moment where an in-place writer would have destroyed the previous contents.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from wmo.core.files import (
    FileLockTimeout,
    file_write_lock,
    write_bytes_atomic,
    write_text_atomic,
)


def _fail_fsync(monkeypatch: pytest.MonkeyPatch, message: str = "disk full") -> None:
    def _boom(fd: int) -> None:
        raise OSError(message)

    monkeypatch.setattr(os, "fsync", _boom)


def test_write_text_atomic_writes_the_file_and_creates_its_parent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"

    write_text_atomic(path, '{"run": 1}')

    assert path.read_text(encoding="utf-8") == '{"run": 1}'


def test_write_text_atomic_replaces_existing_contents_whole(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_text_atomic(path, "a much longer previous payload")

    write_text_atomic(path, "short")

    assert path.read_text(encoding="utf-8") == "short"  # not overwritten in place, so no tail


def test_write_text_atomic_uses_a_staging_name_unique_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared staging name lets two writers rename each other's half-written file into place.

    That turns a lost update into a CORRUPT file, which is strictly worse: last-writer-wins at
    least leaves something loadable. The name is a uuid rather than the pid because two THREADS
    of one process share a pid, and the alias tables are written from threads in this suite.
    """
    renamed: list[str] = []
    real_replace = Path.replace

    def _record(self: Path, target: object) -> Path:
        renamed.append(self.name)
        return real_replace(self, target)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(Path, "replace", _record)
    write_text_atomic(tmp_path / "state.json", "one")
    write_text_atomic(tmp_path / "state.json", "two")

    assert len(set(renamed)) == 2, f"the staging name repeated across calls: {renamed}"
    assert all(name.startswith(".state.json.") for name in renamed)


def test_write_text_atomic_staging_name_differs_across_threads(tmp_path: Path) -> None:
    """The pid is not enough: same-process threads would collide on it and corrupt the file.

    Each thread writes its own long distinct payload many times over; if two shared a staging
    path they would interleave inside one file and the winner would publish a mixture. Every
    read must see exactly one thread's whole payload.
    """
    path = tmp_path / "shared.txt"
    payloads = {name: name * 5000 for name in ("a", "b", "c")}
    torn: list[str] = []

    def _hammer(name: str) -> None:
        for _ in range(20):
            write_text_atomic(path, payloads[name])
            seen = path.read_text(encoding="utf-8")
            if seen not in payloads.values():
                torn.append(seen[:40])

    writers = [threading.Thread(target=_hammer, args=(name,)) for name in payloads]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)

    assert torn == []
    assert list(tmp_path.glob("*.partial")) == []


def test_write_bytes_atomic_round_trips_bytes(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"

    write_bytes_atomic(path, b'{"weights": [1, 2]}')

    assert path.read_bytes() == b'{"weights": [1, 2]}'


def test_write_text_atomic_rejects_unencodable_text_without_leaving_a_staging_file(
    tmp_path: Path,
) -> None:
    """A lone surrogate reaches `json.dumps(..., ensure_ascii=False)` and cannot be encoded.

    That raises `UnicodeEncodeError`, which is a ValueError, not an OSError. Encoding before
    touching the filesystem means no staging file is created at all, rather than one that an
    `except OSError` cleanup would have missed and stranded in the run directory.
    """
    path = tmp_path / "state.json"

    with pytest.raises(UnicodeEncodeError):
        write_text_atomic(path, '{"text": "\ud83d"}')

    assert list(tmp_path.iterdir()) == []


def test_write_text_atomic_cleans_up_after_a_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C between the write and the rename must not strand a staging file.

    `KeyboardInterrupt` is a BaseException, so an `except OSError` cleanup would let it through
    and leave litter in an artifact directory that serving and the fitter both scan.
    """
    path = tmp_path / "policy.json"

    def _interrupt(fd: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "fsync", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_text_atomic(path, "payload")

    assert list(tmp_path.iterdir()) == []


def test_write_text_atomic_keeps_the_destinations_mode(tmp_path: Path) -> None:
    """`replace` installs a NEW inode, so a restricted file would silently widen to the umask.

    An operator (or an installer) who chmods a credential-adjacent config to 0600 must not have
    it reset to 0644 by the next write.
    """
    path = tmp_path / "settings.toml"
    write_text_atomic(path, "first")
    path.chmod(0o600)

    write_text_atomic(path, "second")

    assert path.stat().st_mode & 0o777 == 0o600


def test_write_text_atomic_leaves_the_previous_file_intact_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core guarantee: a reader sees the previous file or the whole new one, never a stub."""
    path = tmp_path / "aliases.toml"
    write_text_atomic(path, "[aliases]\nchampion = 3\n")
    _fail_fsync(monkeypatch)

    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(path, "[aliases]\nchampion = 4\n")

    assert path.read_text(encoding="utf-8") == "[aliases]\nchampion = 3\n"


def test_a_failure_after_the_rename_reports_the_new_contents_not_the_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent-directory fsync runs AFTER the rename, so its failure is not a no-op write.

    Some FUSE, SMB and NFS mounts reject `fsync` on a directory fd. The write has landed by then,
    and the docstring says so rather than claiming the file is untouched: a caller that reported
    "the promotion failed" and rolled back on that basis would be acting on the opposite of the
    truth. Only the durability of the rename against a power loss is unproven.
    """
    path = tmp_path / "aliases.toml"
    write_text_atomic(path, "old")
    real_fsync = os.fsync
    calls: list[int] = []

    def _fail_the_directory(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:  # 1 is the payload, 2 is the parent directory
            raise OSError("directory fsync unsupported")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fail_the_directory)
    with pytest.raises(OSError, match="directory fsync unsupported"):
        write_text_atomic(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.partial")) == []


def test_write_text_atomic_leaves_no_temp_behind_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "aliases.toml"
    write_text_atomic(path, "[aliases]\nchampion = 3\n")
    _fail_fsync(monkeypatch)

    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(path, "[aliases]\nchampion = 4\n")

    assert list(tmp_path.glob("*.tmp")) == []


def test_write_text_atomic_fsyncs_the_payload_and_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both, or a power loss can persist the rename while the data blocks are lost.

    The result of that is an EMPTY file behind a write that reported success, which is the
    failure mode the whole helper exists to prevent, so it is worth pinning rather than trusting.
    """
    synced: list[int] = []
    real_fsync = os.fsync

    def _record(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _record)
    write_text_atomic(tmp_path / "state.json", "payload")

    assert len(synced) == 2  # the temp file, then the directory that now holds the rename


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

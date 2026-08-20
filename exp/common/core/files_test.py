"""Tests for the atomic writers.

Every assertion is about what survives a FAILURE, because that is the whole point of the module:
the happy path of any of these writers was already correct. Torn writes are modelled by failing
`os.fsync`, which is reached with the payload already on the staging file and the rename not yet
done, so it is the moment where an in-place writer would have destroyed the previous contents.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import pytest

from exp.common.core.files import write_bytes_atomic, write_text_atomic


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


def test_a_directory_fsync_failure_does_not_fail_a_write_that_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent-directory fsync runs AFTER the rename, so raising there would be a lie.

    Some FUSE, SMB and NFS mounts reject `fsync` on a directory fd unconditionally. Propagating
    that would report every write on such a mount as failed for a file written correctly every
    time, and a caller acting on it (reporting a failed promotion, retrying, rolling back) would
    act on the opposite of the truth. It is logged instead, because what is genuinely lost is the
    proof of durability against a power loss, not the write.
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
    write_text_atomic(path, "new")  # must not raise

    assert path.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.partial")) == []


def test_a_directory_fsync_failure_is_logged_rather_than_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Not raising is not the same as saying nothing: the durability gap has to be recorded."""
    path = tmp_path / "state.json"
    real_fsync = os.fsync
    calls: list[int] = []

    def _fail_the_directory(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("directory fsync unsupported")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fail_the_directory)
    with caplog.at_level(logging.WARNING, logger="exp.common.core.files"):
        write_text_atomic(path, "payload")

    assert "not proven durable" in caplog.text
    assert str(tmp_path) in caplog.text


def test_a_directory_close_failure_does_not_fail_a_write_that_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`close` is the last post-rename step, and it can fail for real.

    NFS reports deferred write errors at `close`, so it can return EIO after the fsync succeeded.
    Letting that out would report a landed write as failed, which is the same defect as an
    unguarded directory fsync one line above, just harder to notice.

    The stub closes the descriptor for real before raising, so the test cannot leak a handle.
    """
    path = tmp_path / "state.json"
    real_open, real_close = os.open, os.close
    directory_fds: set[int] = set()

    def _record_open(target: object, flags: int, *args: int) -> int:
        fd = real_open(target, flags, *args)  # ty: ignore[invalid-argument-type]
        if target == path.parent:
            directory_fds.add(fd)
        return fd

    def _fail_close(fd: int) -> None:
        real_close(fd)
        if fd in directory_fds:
            raise OSError("close reported a deferred write error")

    monkeypatch.setattr(os, "open", _record_open)
    monkeypatch.setattr(os, "close", _fail_close)
    with caplog.at_level(logging.WARNING, logger="exp.common.core.files"):
        write_text_atomic(path, "payload")  # must not raise

    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == "payload"
    assert "could not close the directory handle" in caplog.text
    assert list(tmp_path.glob("*.partial")) == []


def test_write_text_atomic_writes_through_a_symlinked_destination(tmp_path: Path) -> None:
    """An operator who points a config at a shared file must get their writes there.

    `replace` swaps the LINK for a regular file and leaves the target stale, with nothing raised.
    An in-place `write_text` followed the link, so the three writers converted to atomic writes
    (`HarnessStore.set_alias`, `exp/common/config/card.py`, `OutcomeMatrix.save`) silently changed
    meaning; the ones that were always atomic had been clobbering links all along.
    """
    target = tmp_path / "shared.toml"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "config.toml"
    link.symlink_to(target)

    write_text_atomic(link, "new")

    assert link.is_symlink(), "the symlink was replaced by a regular file"
    assert target.read_text(encoding="utf-8") == "new"


def test_write_bytes_atomic_no_follow_replaces_a_swapped_destination_symlink(
    tmp_path: Path,
) -> None:
    """A security-checked pointer file must never write through a link swapped in after the check.

    With `follow_symlinks=False` the rename replaces the LINK itself with a regular file, so the
    link's target stays untouched and the pointer path holds the payload.
    """
    target = tmp_path / "victim.json"
    target.write_bytes(b"original")
    pointer = tmp_path / "pointer.json"
    pointer.symlink_to(target)

    write_bytes_atomic(pointer, b"selection", follow_symlinks=False)

    assert not pointer.is_symlink(), "the swapped symlink survived the pointer write"
    assert pointer.read_bytes() == b"selection"
    assert target.read_bytes() == b"original"


def test_write_bytes_atomic_no_follow_ignores_the_swapped_link_targets_mode(
    tmp_path: Path,
) -> None:
    """The mode carryover must not stat THROUGH a swapped link whose target mode is hostile.

    Whoever plants the link picks the target, so copying the target's mode lets them pick the new
    pointer file's permissions — a 0o000 target leaves the pointer written but unreadable, wedging
    the immediate read-back every pointer writer performs. The link is not a regular file, so no
    mode is carried over and the pointer keeps the staging file's default.
    """
    target = tmp_path / "victim.json"
    target.write_bytes(b"original")
    target.chmod(0o000)
    pointer = tmp_path / "pointer.json"
    pointer.symlink_to(target)

    write_bytes_atomic(pointer, b"selection", follow_symlinks=False)

    assert not pointer.is_symlink()
    assert pointer.stat().st_mode & 0o777 != 0o000, "the link target's mode leaked onto the pointer"
    assert pointer.read_bytes() == b"selection"


def test_write_text_atomic_stages_beside_the_symlink_target_not_the_link(tmp_path: Path) -> None:
    """Atomicity depends on this: `replace` across filesystems fails with EXDEV.

    A symlink pointing into another mount is exactly the case where staging next to the LINK would
    make the final rename cross a filesystem boundary and fail. Staging beside the resolved target
    keeps the rename local. A separate directory stands in for a separate mount here, which pins
    the staging location even though it cannot pin EXDEV itself.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "shared.toml"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "config.toml"
    link.symlink_to(target)
    staged: list[Path] = []
    real_replace = Path.replace

    def _record(self: Path, destination: object) -> Path:
        staged.append(self)
        return real_replace(self, destination)  # ty: ignore[invalid-argument-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "replace", _record)
        write_text_atomic(link, "new")

    assert [item.parent for item in staged] == [elsewhere], "staged beside the link, not the target"
    assert target.read_text(encoding="utf-8") == "new"


def test_write_text_atomic_keeps_the_symlink_targets_mode(tmp_path: Path) -> None:
    """The mode that matters is the target's, since that is the inode being replaced."""
    target = tmp_path / "shared.toml"
    target.write_text("original", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "config.toml"
    link.symlink_to(target)

    write_text_atomic(link, "new")

    # Both halves matter: without the content assertion this passes on the broken behaviour,
    # because a target that was never written trivially keeps its mode.
    assert target.read_text(encoding="utf-8") == "new"
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_text_atomic_creates_the_target_of_a_dangling_symlink(tmp_path: Path) -> None:
    """Writing through a broken link creates its target, as an in-place write would have."""
    missing = tmp_path / "not-there-yet.toml"
    link = tmp_path / "config.toml"
    link.symlink_to(missing)

    write_text_atomic(link, "created")

    assert missing.read_text(encoding="utf-8") == "created"
    assert link.is_symlink()

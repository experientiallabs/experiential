"""Actual Windows handle behavior, not mocked platform labels."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exp.common.core import files as atomic_files
from exp.common.core.files import write_bytes_atomic
from exp.runtime.gateway import snapshot_file
from exp.runtime.gateway.model_chain_authority import SnapshotClassificationMemo
from exp.runtime.gateway.snapshot_file_windows import _WindowsFiles

if sys.platform == "win32":
    import msvcrt

from exp.runtime.gateway.snapshot_file import (
    prepare_snapshot_file,
    read_snapshot_bytes,
    snapshot_stream,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Requires actual Windows kernel handles"
)


@pytest.mark.parametrize("mutation", ["atomic_replace", "rewrite_restore_mtime"])
def test_windows_memo_anchor_allows_publication_and_detects_change_time(
    tmp_path: Path, mutation: str
) -> None:
    """Permissive metadata anchors allow the real publisher and invalidate same-size rewrites."""
    path = tmp_path / "snapshot.json"
    path.write_bytes(b'{"model_chains":[]}')
    sidecar = path.with_suffix(".models.json")
    sidecar.write_bytes(b"{}")
    memo = SnapshotClassificationMemo()
    key = ("db", path.name, 1024)
    try:
        with (
            prepare_snapshot_file(tmp_path, path.name, 1024) as first,
            prepare_snapshot_file(tmp_path, sidecar.name, 1024) as second,
        ):
            memo.remember(key, (first, second))
        assert len(memo._entries) == 1
        before = path.stat()
        if mutation == "atomic_replace":
            write_bytes_atomic(path, b'{"model_chains":{}}')
        else:
            path.write_bytes(b'{"model_chains":{}}')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with (
            prepare_snapshot_file(tmp_path, path.name, 1024, read_content=False) as first,
            prepare_snapshot_file(tmp_path, sidecar.name, 1024, read_content=False) as second,
        ):
            assert not memo.matches(key, (first, second))
    finally:
        memo.close()


def test_windows_atomic_publisher_preserves_strict_reader_exclusion(tmp_path: Path) -> None:
    """Both normal and POSIX replacement refuse while a strict operation denies delete sharing."""
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"old")
    staging = tmp_path / "manual.partial"
    staging.write_bytes(b"new")
    with snapshot_stream(tmp_path, path.name) as stream:
        with pytest.raises(OSError):
            write_bytes_atomic(path, b"new")
        assert stream.read() == b"old"
        assert not list(tmp_path.glob(".snapshot.json.*.partial"))
        with pytest.raises(OSError):
            atomic_files._windows_replace_open_target(staging, path)
        assert path.read_bytes() == b"old" and staging.read_bytes() == b"new"
    staging.unlink()
    write_bytes_atomic(path, b"new")
    assert path.read_bytes() == b"new"


@pytest.mark.parametrize("payload", [b"", b"new bytes"])
@pytest.mark.parametrize("long_path", [False, True])
def test_windows_posix_atomic_replace_preserves_open_old_inode_and_unicode_path(
    tmp_path: Path, payload: bytes, long_path: bool
) -> None:
    """A permissive read handle retains old bytes while later opens see the complete new payload."""
    assert os.name == "nt"
    directory = tmp_path / "nested-日\U0001f30d"
    if long_path:
        directory = directory / ("nested" * 20) / ("deeper" * 20)
    directory.mkdir(parents=True)
    path = directory / "state-é\U0001f680.json"
    path.write_bytes(b"old")
    api = _WindowsFiles()
    # Metadata-only anchor is the production case; a delete-shared data reader
    # separately proves old bytes stay intact rather than being overwritten in place.
    handle = api.api.CreateFileW(str(path), 0x80000000, 7, None, 3, 0, None)
    assert handle is not None and handle != ctypes.c_void_p(-1).value
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        api.close(handle)
        raise
    with os.fdopen(descriptor, "rb") as old:
        write_bytes_atomic(path, payload)
        assert old.read() == b"old"
        assert path.read_bytes() == payload
        assert not list(directory.glob("*.partial"))


def test_windows_atomic_replace_unsupported_kernel_preserves_old_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure of the atomic kernel operation never falls through to delete or in-place writes."""
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"old")
    memo = SnapshotClassificationMemo()
    with (
        prepare_snapshot_file(tmp_path, path.name, 1024) as first,
        prepare_snapshot_file(tmp_path, "absent.models.json", 1024) as second,
    ):
        memo.remember(("db", path.name, 1024), (first, second))

    def unsupported(staging: Path, destination: Path) -> None:
        """Represent an unsupported FileRenameInfoEx operation after staging completed."""
        assert staging.read_bytes() == b"new" and destination == path
        raise OSError("FileRenameInfoEx unsupported")

    try:
        monkeypatch.setattr(atomic_files, "_windows_replace_open_target", unsupported)
        with pytest.raises(OSError, match="unsupported"):
            write_bytes_atomic(path, b"new")
        assert path.read_bytes() == b"old"
        assert not list(tmp_path.glob("*.partial"))
    finally:
        memo.close()


def test_windows_atomic_replace_keeps_readonly_and_nofollow_policy(tmp_path: Path) -> None:
    """POSIX replacement never ignores read-only attributes or follows a protected pointer link."""
    path = tmp_path / "readonly.json"
    path.write_bytes(b"old")
    path.chmod(0o444)
    try:
        with pytest.raises(OSError):
            write_bytes_atomic(path, b"new")
        assert path.read_bytes() == b"old"
        assert not list(tmp_path.glob("*.partial"))
    finally:
        path.chmod(0o666)
    victim = tmp_path / "victim.json"
    victim.write_bytes(b"victim")
    pointer = tmp_path / "pointer.json"
    try:
        pointer.symlink_to(victim)
    except OSError:
        pytest.skip("symlink creation is unavailable for this Windows account")
    write_bytes_atomic(pointer, b"selection", follow_symlinks=False)
    assert not pointer.is_symlink() and pointer.read_bytes() == b"selection"
    assert victim.read_bytes() == b"victim"
    pointer.unlink()
    pointer.symlink_to(victim)
    staging = tmp_path / "pointer.partial"
    staging.write_bytes(b"direct")
    atomic_files._windows_replace_open_target(staging, pointer)
    assert not pointer.is_symlink() and pointer.read_bytes() == b"direct"
    assert victim.read_bytes() == b"victim"
    staging.symlink_to(victim)
    victim_mode = victim.stat().st_mode
    with pytest.raises(OSError, match="non-reparse"):
        atomic_files._windows_replace_open_target(staging, pointer)
    assert staging.is_symlink() and victim.read_bytes() == b"victim"
    assert victim.stat().st_mode == victim_mode
    staging.unlink()


def test_windows_missing_change_time_disables_memo_not_secure_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported metadata identity never substitutes creation time for a change counter."""
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(snapshot_file, "windows_change_time", lambda _descriptor: None)
    memo = SnapshotClassificationMemo()
    try:
        with (
            prepare_snapshot_file(tmp_path, path.name, 1024) as first,
            prepare_snapshot_file(tmp_path, "missing.models.json", 1024) as second,
        ):
            assert first.take_bytes() == b"{}"
            memo.remember(("db", path.name, 1024), (first, second))
            assert not memo._entries
            first.validate_current()
    finally:
        memo.close()


def test_windows_directory_and_leaf_handles_block_replacement_and_close(tmp_path: Path) -> None:
    """Checked components cannot be renamed or overwritten until the read releases handles."""
    directory = tmp_path / "nested"
    directory.mkdir()
    leaf = directory / "snapshot"
    leaf.write_bytes(b"data")
    with snapshot_stream(tmp_path, "nested/snapshot") as stream:
        assert stream.read() == b"data"
        with pytest.raises(OSError):
            directory.rename(tmp_path / "moved")
        with pytest.raises(OSError):
            leaf.write_bytes(b"changed")
        with pytest.raises(OSError):
            leaf.unlink()
    leaf.write_bytes(b"changed")
    directory.rename(tmp_path / "moved")
    assert read_snapshot_bytes(tmp_path, "moved/snapshot", 7) == b"changed"


def test_windows_preflight_retains_handles_and_detects_absent_sidecar_appearance(
    tmp_path: Path,
) -> None:
    """Prepared classification owns write-denying handles until its operation exits."""
    directory = tmp_path / "nested"
    directory.mkdir()
    leaf = directory / "snapshot.json"
    leaf.write_bytes(b"{}")
    with prepare_snapshot_file(tmp_path, "nested/snapshot.json", 1024) as present:
        assert present.take_bytes() == b"{}"
        present.validate_current()
        with pytest.raises(OSError):
            leaf.write_bytes(b"[]")
        with prepare_snapshot_file(tmp_path, "nested/snapshot.models.json", 1024) as absent:
            assert absent.take_bytes() is None
            directory.joinpath("snapshot.models.json").write_bytes(b"{}")
            with pytest.raises(ValueError, match="changed"):
                absent.validate_current()
    leaf.write_bytes(b"[]")
    directory.rename(tmp_path / "moved")
    with pytest.raises(ValueError, match="closed"):
        present.validate_current()


def test_windows_intermediate_junction_is_rejected_and_handles_close(tmp_path: Path) -> None:
    """A real junction cannot redirect a snapshot read outside its trusted directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "snapshot").write_bytes(b"data")
    root = tmp_path / "root"
    root.mkdir()
    junction = root / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    try:
        with pytest.raises(ValueError, match="reparse"):
            read_snapshot_bytes(root, "junction/snapshot", 10)
    finally:
        junction.rmdir()
    root.rename(tmp_path / "renamed-root")


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "C:relative",
        "C:/absolute",
        "\\\\server\\share\\file",
        "snapshot:stream",
        "NUL",
        "CON",
        "nested./snapshot",
    ],
)
def test_windows_rejects_nonordinary_paths(tmp_path: Path, path: str) -> None:
    """Drive-relative, UNC overrides, streams and devices never become snapshot files."""
    with pytest.raises((ValueError, OSError)):
        read_snapshot_bytes(tmp_path, path, 100)


def test_windows_case_insensitive_component_identity_and_missing_cleanup(tmp_path: Path) -> None:
    """Canonical path comparisons accept normal Windows case and reject absent leaves cleanly."""
    nested = tmp_path / "MixedCase"
    nested.mkdir()
    (nested / "Snapshot.JSON").write_bytes(b"{}")
    assert read_snapshot_bytes(tmp_path, "mixedcase/snapshot.json", 2) == b"{}"
    with pytest.raises(OSError):
        read_snapshot_bytes(tmp_path, "MixedCase/missing", 100)
    nested.rename(tmp_path / "after-error")


def test_windows_stream_failure_releases_file_and_directory_handles(tmp_path: Path) -> None:
    """Exceptions raised by a reader release the CRT file and checked directory handles."""
    nested = tmp_path / "nested"
    nested.mkdir()
    leaf = nested / "snapshot"
    leaf.write_bytes(b"data")
    with pytest.raises(RuntimeError, match="reader failed"):
        with snapshot_stream(tmp_path, "nested/snapshot"):
            raise RuntimeError("reader failed")
    leaf.write_bytes(b"after")
    nested.rename(tmp_path / "after-reader-error")


def test_windows_existing_writer_is_not_shared_with_snapshot(tmp_path: Path) -> None:
    """An already-open writer prevents obtaining a supposedly immutable read handle."""
    leaf = tmp_path / "snapshot"
    leaf.write_bytes(b"data")
    with leaf.open("r+b"):
        with pytest.raises(OSError):
            read_snapshot_bytes(tmp_path, "snapshot", 4)
    assert read_snapshot_bytes(tmp_path, "snapshot", 4) == b"data"


def test_windows_fdopen_failure_closes_transferred_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CRT descriptor owns the leaf even when stream creation fails."""
    leaf = tmp_path / "snapshot"
    leaf.write_bytes(b"data")

    def fail_stream(descriptor: int, mode: str) -> None:
        """Fail after handle transfer; the production owner must close the descriptor."""
        raise OSError("stream allocation failed")

    monkeypatch.setattr(os, "fdopen", fail_stream)
    with pytest.raises(OSError, match="stream allocation failed"):
        read_snapshot_bytes(tmp_path, "snapshot", 4)
    leaf.write_bytes(b"after")
    leaf.unlink()

"""Actual Windows handle behavior, not mocked platform labels."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from exp.runtime.gateway.snapshot_file import read_snapshot_bytes, snapshot_stream

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Requires actual Windows kernel handles"
)


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

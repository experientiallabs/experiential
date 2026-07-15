# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for E2B workspace snapshotting and conflict-safe local reconciliation."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from wmh.cli.workspace_sync import (
    WorkspaceSyncError,
    apply_workspace_patch,
    snapshot_workspace,
    sync_workspace,
    write_conflict_archive,
)
from wmh.harness.workspace_patch import build_workspace_patch


def _archive(files: dict[str, tuple[bytes, int]]) -> bytes:
    """Build a regular-file-only gzip tar for a simulated final sandbox."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, (content, mode) in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_snapshot_skips_links_vcs_and_dependency_trees(tmp_path: Path) -> None:
    """Only source-like regular files enter the upload archive."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("large", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "src" / "app.py")

    snapshot = snapshot_workspace(tmp_path)

    assert set(snapshot.files) == {"src/app.py"}
    with tarfile.open(fileobj=io.BytesIO(snapshot.archive), mode="r:gz") as archive:
        assert archive.getnames() == ["src/app.py"]


def test_sync_applies_remote_add_modify_delete_and_mode(tmp_path: Path) -> None:
    """Uncontested remote filesystem changes automatically land in the local directory."""
    (tmp_path / "changed.txt").write_text("before", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("remove", encoding="utf-8")
    initial = snapshot_workspace(tmp_path)
    final = _archive(
        {
            "changed.txt": (b"after", 0o755),
            "added.txt": (b"new", 0o644),
        }
    )

    result = sync_workspace(tmp_path, initial, final)

    assert result.conflicts == ()
    assert set(result.applied) == {"added.txt", "changed.txt", "deleted.txt"}
    assert (tmp_path / "changed.txt").read_text(encoding="utf-8") == "after"
    assert (tmp_path / "changed.txt").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "added.txt").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "deleted.txt").exists()


def test_sync_preserves_concurrent_local_edit_and_applies_other_paths(tmp_path: Path) -> None:
    """A local edit wins its path while unrelated remote changes still sync."""
    (tmp_path / "same.txt").write_text("base", encoding="utf-8")
    (tmp_path / "other.txt").write_text("base", encoding="utf-8")
    initial = snapshot_workspace(tmp_path)
    (tmp_path / "same.txt").write_text("local", encoding="utf-8")
    final = _archive(
        {
            "same.txt": (b"remote", 0o644),
            "other.txt": (b"remote", 0o644),
        }
    )

    result = sync_workspace(tmp_path, initial, final)

    assert result.conflicts == ("same.txt",)
    assert result.applied == ("other.txt",)
    assert (tmp_path / "same.txt").read_text(encoding="utf-8") == "local"
    assert (tmp_path / "other.txt").read_text(encoding="utf-8") == "remote"
    recovery = write_conflict_archive(tmp_path, "session-1", final)
    assert recovery.read_bytes() == final


def test_incremental_patch_applies_uncontested_changes(tmp_path: Path) -> None:
    """A live remote patch lands before the hosted session finishes."""
    (tmp_path / "changed.txt").write_text("before", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("remove", encoding="utf-8")
    before = snapshot_workspace(tmp_path)
    after = _archive(
        {
            "changed.txt": (b"after", 0o755),
            "added.txt": (b"new", 0o644),
        }
    )
    patch = build_workspace_patch(before.archive, after)
    assert patch is not None

    result = apply_workspace_patch(tmp_path, patch)

    assert result.conflicts == ()
    assert set(result.applied) == {"added.txt", "changed.txt", "deleted.txt"}
    assert (tmp_path / "changed.txt").read_text(encoding="utf-8") == "after"
    assert (tmp_path / "changed.txt").stat().st_mode & 0o777 == 0o755
    assert not (tmp_path / "deleted.txt").exists()


def test_incremental_patch_preserves_local_conflict_and_applies_other_path(
    tmp_path: Path,
) -> None:
    """Live sync isolates a same-path conflict instead of stopping the stream."""
    (tmp_path / "same.txt").write_text("base", encoding="utf-8")
    (tmp_path / "other.txt").write_text("base", encoding="utf-8")
    before = snapshot_workspace(tmp_path)
    after = _archive(
        {
            "same.txt": (b"remote", 0o644),
            "other.txt": (b"remote", 0o644),
        }
    )
    patch = build_workspace_patch(before.archive, after)
    assert patch is not None
    (tmp_path / "same.txt").write_text("local", encoding="utf-8")

    result = apply_workspace_patch(tmp_path, patch)

    assert result.conflicts == ("same.txt",)
    assert result.applied == ("other.txt",)
    assert (tmp_path / "same.txt").read_text(encoding="utf-8") == "local"
    assert (tmp_path / "other.txt").read_text(encoding="utf-8") == "remote"


def test_sync_rejects_traversal_and_links(tmp_path: Path) -> None:
    """A malicious sandbox archive cannot escape staging or materialize links locally."""
    initial = snapshot_workspace(tmp_path)
    traversal = _archive({"../escape": (b"bad", 0o644)})
    with pytest.raises(WorkspaceSyncError, match="unsafe"):
        sync_workspace(tmp_path, initial, traversal)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(WorkspaceSyncError, match="regular file or directory"):
        sync_workspace(tmp_path, initial, buffer.getvalue())

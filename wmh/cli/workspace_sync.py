# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Safe local snapshot and three-way sync for hosted E2B agent workspaces."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from wmh.harness.workspace_patch import PatchFileState, parse_workspace_patch

MAX_WORKSPACE_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_WORKSPACE_UNPACKED_BYTES = 512 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 100_000

# Dependency trees, VCS internals, caches, and WMH recovery artifacts are not
# useful source inputs and can turn a small repository into a multi-GB upload.
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".wmh-conflicts",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


class WorkspaceSyncError(RuntimeError):
    """A workspace cannot be safely archived or synchronized."""


@dataclass(frozen=True)
class FileState:
    """Content and executable-mode identity used by the three-way merge."""

    sha256: str
    mode: int


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Initial upload archive plus its regular-file manifest."""

    archive: bytes
    files: dict[str, FileState]


@dataclass(frozen=True)
class SyncResult:
    """Paths applied automatically and paths preserved as local conflicts."""

    applied: tuple[str, ...]
    conflicts: tuple[str, ...]


def snapshot_workspace(root: Path) -> WorkspaceSnapshot:
    """Archive regular files under ``root`` and capture their initial identities."""
    resolved = root.resolve()
    files = _manifest(resolved)
    try:
        total = sum(path.stat().st_size for path in _paths_for_manifest(resolved, files))
    except OSError as error:
        msg = "workspace changed while it was being inspected; retry the run"
        raise WorkspaceSyncError(msg) from error
    if total > MAX_WORKSPACE_UNPACKED_BYTES:
        msg = f"workspace files exceed {MAX_WORKSPACE_UNPACKED_BYTES} uncompressed bytes"
        raise WorkspaceSyncError(msg)
    buffer = io.BytesIO()
    try:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for relative in sorted(files):
                source = resolved / relative
                info = tarfile.TarInfo(relative)
                file_stat = source.stat()
                info.size = file_stat.st_size
                info.mode = stat.S_IMODE(file_stat.st_mode)
                info.mtime = int(file_stat.st_mtime)
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
    except OSError as error:
        msg = "workspace changed while it was being archived; retry the run"
        raise WorkspaceSyncError(msg) from error
    if _manifest(resolved) != files:
        msg = "workspace changed while it was being archived; retry the run"
        raise WorkspaceSyncError(msg)
    content = buffer.getvalue()
    if len(content) > MAX_WORKSPACE_ARCHIVE_BYTES:
        msg = f"workspace archive exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} compressed bytes"
        raise WorkspaceSyncError(msg)
    return WorkspaceSnapshot(archive=content, files=files)


def sync_workspace(
    root: Path,
    initial: WorkspaceSnapshot,
    final_archive: bytes,
    *,
    protected_paths: frozenset[str] = frozenset(),
) -> SyncResult:
    """Apply remote changes unless the same path changed locally since upload."""
    resolved = root.resolve()
    if len(final_archive) > MAX_WORKSPACE_ARCHIVE_BYTES:
        msg = f"final workspace archive exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} bytes"
        raise WorkspaceSyncError(msg)
    with tempfile.TemporaryDirectory(prefix="wmh-workspace-") as staging_name:
        staging = Path(staging_name)
        _extract_archive(final_archive, staging)
        remote = _manifest(staging)
        current = _manifest(resolved)
        applied: list[str] = []
        conflicts: list[str] = []
        for relative in sorted(set(initial.files) | set(remote)):
            before = initial.files.get(relative)
            after = remote.get(relative)
            if before == after:
                continue
            target = resolved / relative
            now = current.get(relative)
            if (
                relative in protected_paths
                or _has_non_file_collision(target, now)
                or (now != before and now != after)
            ):
                conflicts.append(relative)
                continue
            if now == after:
                continue
            try:
                if after is None:
                    target.unlink(missing_ok=True)
                    _remove_empty_parents(target.parent, resolved)
                else:
                    _atomic_copy(staging / relative, target, root=resolved, mode=after.mode)
            except OSError:
                conflicts.append(relative)
                continue
            applied.append(relative)
    return SyncResult(applied=tuple(applied), conflicts=tuple(conflicts))


def apply_workspace_patch(root: Path, content: bytes) -> SyncResult:
    """Apply an incremental remote patch when each local path still matches its base."""
    resolved = root.resolve()
    patch = parse_workspace_patch(content)
    current = _manifest(resolved)
    applied: list[str] = []
    conflicts: list[str] = []
    for operation in patch.operations:
        target = resolved / operation.path
        now = current.get(operation.path)
        before = _file_state(operation.before)
        after = _file_state(operation.after)
        if _has_non_file_collision(target, now) or (now != before and now != after):
            conflicts.append(operation.path)
            continue
        if now == after:
            continue
        try:
            if after is None:
                target.unlink(missing_ok=True)
                _remove_empty_parents(target.parent, resolved)
            else:
                _atomic_write(patch.files[operation.path], target, root=resolved, mode=after.mode)
        except OSError:
            conflicts.append(operation.path)
            continue
        applied.append(operation.path)
    return SyncResult(applied=tuple(applied), conflicts=tuple(conflicts))


def write_conflict_archive(root: Path, session_id: str, content: bytes) -> Path:
    """Preserve a downloaded result archive when automatic reconciliation conflicts."""
    directory = root.resolve() / ".wmh-conflicts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.tar.gz"
    path.write_bytes(content)
    return path


def _manifest(root: Path) -> dict[str, FileState]:
    """Hash regular files without following symlinks or entering excluded trees."""
    manifest: dict[str, FileState] = {}
    entries = 0
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDED_DIRECTORY_NAMES and not (base / name).is_symlink()
        )
        for name in sorted(filenames):
            path = base / name
            try:
                file_stat = path.lstat()
            except OSError as error:
                raise WorkspaceSyncError(f"could not inspect workspace file: {path}") from error
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            relative = path.relative_to(root).as_posix()
            manifest[relative] = FileState(
                sha256=_sha256(path), mode=stat.S_IMODE(file_stat.st_mode)
            )
            entries += 1
            if entries > MAX_WORKSPACE_ENTRIES:
                msg = f"workspace has more than {MAX_WORKSPACE_ENTRIES} files"
                raise WorkspaceSyncError(msg)
    return manifest


def _paths_for_manifest(root: Path, manifest: dict[str, FileState]) -> list[Path]:
    """Resolve manifest paths for aggregate-size accounting."""
    return [root / relative for relative in manifest]


def _sha256(path: Path) -> str:
    """Hash one regular file without loading it all into memory."""
    digest = hashlib.sha256(usedforsecurity=False)
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise WorkspaceSyncError(f"could not read workspace file: {path}") from error
    return digest.hexdigest()


def _extract_archive(content: bytes, destination: Path) -> None:
    """Validate and extract regular files only into an isolated staging directory."""
    total_size = 0
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_WORKSPACE_ENTRIES:
                msg = f"workspace archive has more than {MAX_WORKSPACE_ENTRIES} entries"
                raise WorkspaceSyncError(msg)
            for member in members:
                relative = _normalized_name(member.name)
                if relative in seen:
                    raise WorkspaceSyncError(f"duplicate workspace path: {relative}")
                seen.add(relative)
                if not (member.isfile() or member.isdir()):
                    raise WorkspaceSyncError(
                        f"workspace entry must be a regular file or directory: {member.name}"
                    )
                total_size += member.size
                if total_size > MAX_WORKSPACE_UNPACKED_BYTES:
                    msg = f"workspace expands beyond {MAX_WORKSPACE_UNPACKED_BYTES} bytes"
                    raise WorkspaceSyncError(msg)
                target = destination if relative == "." else destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise WorkspaceSyncError(f"workspace file has no content: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
    except WorkspaceSyncError:
        raise
    except (tarfile.TarError, OSError, EOFError) as error:
        msg = "workspace must be a valid gzip tar archive"
        raise WorkspaceSyncError(msg) from error


def _normalized_name(name: str) -> str:
    """Normalize one tar name and reject absolute or traversing entries."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise WorkspaceSyncError(f"unsafe workspace path: {name}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    return PurePosixPath(*parts).as_posix() if parts else "."


def _has_non_file_collision(target: Path, state: FileState | None) -> bool:
    """Return true when a manifest-absent target still exists as a dir/link/special file."""
    return state is None and (target.exists() or target.is_symlink())


def _atomic_copy(source: Path, target: Path, *, root: Path, mode: int) -> None:
    """Copy through a sibling temporary file after proving the parent stays in ``root``."""
    relative_parent = target.parent.relative_to(root)
    current = root
    for part in relative_parent.parts:
        current /= part
        if current.is_symlink():
            raise OSError(f"workspace path crosses a symlink: {target}")
        if current.exists() and not current.is_dir():
            raise OSError(f"workspace parent is not a directory: {current}")
        current.mkdir(exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise OSError(f"workspace target is not a regular file: {target}")
    temporary = target.parent / f".{target.name}.wmh-{uuid.uuid4().hex}"
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write(content: bytes, target: Path, *, root: Path, mode: int) -> None:
    """Write patch bytes through the same collision-safe sibling replacement path."""
    with tempfile.TemporaryDirectory(prefix="wmh-patch-") as staging_name:
        source = Path(staging_name) / "content"
        source.write_bytes(content)
        _atomic_copy(source, target, root=root, mode=mode)


def _file_state(state: PatchFileState | None) -> FileState | None:
    """Translate the shared transport state into the CLI merge state."""
    if state is None:
        return None
    return FileState(sha256=state.sha256, mode=state.mode)


def _remove_empty_parents(directory: Path, root: Path) -> None:
    """Remove newly empty parents after a remote deletion, stopping at the sync root."""
    current = directory
    while current != root:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent

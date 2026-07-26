# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Hash-guarded incremental patch protocol for synchronized workspaces."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_WORKSPACE_PATCH_BYTES = 50 * 1024 * 1024
MAX_WORKSPACE_PATCH_UNPACKED_BYTES = 512 * 1024 * 1024
MAX_WORKSPACE_PATCH_ENTRIES = 100_000

_MANIFEST_PATH = "manifest.json"
_DATA_PREFIX = "data/"
_VERSION = 1


class WorkspacePatchError(ValueError):
    """An incremental workspace patch is malformed or unsafe."""


@dataclass(frozen=True)
class PatchFileState:
    """The content and executable mode expected before or after one operation."""

    sha256: str
    mode: int


@dataclass(frozen=True)
class WorkspacePatchOperation:
    """One conditional path transition in an incremental patch."""

    path: str
    before: PatchFileState | None
    after: PatchFileState | None


@dataclass(frozen=True)
class WorkspacePatch:
    """Validated operations and replacement bytes keyed by workspace path."""

    operations: tuple[WorkspacePatchOperation, ...]
    files: dict[str, bytes]


def build_workspace_patch(before_archive: bytes, after_archive: bytes) -> bytes | None:
    """Build a deterministic patch between two full workspace archives."""
    before = _read_workspace_archive(before_archive)
    after = _read_workspace_archive(after_archive)
    operations: list[WorkspacePatchOperation] = []
    for path in sorted(set(before) | set(after)):
        before_state = _state(before.get(path))
        after_state = _state(after.get(path))
        if before_state != after_state:
            operations.append(
                WorkspacePatchOperation(path=path, before=before_state, after=after_state)
            )
    if not operations:
        return None

    manifest = {
        "version": _VERSION,
        "operations": [
            {
                "path": operation.path,
                "before": _state_json(operation.before),
                "after": _state_json(operation.after),
            }
            for operation in operations
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        _add_bytes(archive, _MANIFEST_PATH, manifest_bytes, 0o600)
        for operation in operations:
            if operation.after is None:
                continue
            body, _mode = after[operation.path]
            _add_bytes(archive, f"{_DATA_PREFIX}{operation.path}", body, operation.after.mode)
    content = buffer.getvalue()
    if len(content) > MAX_WORKSPACE_PATCH_BYTES:
        raise WorkspacePatchError(f"workspace patch exceeds {MAX_WORKSPACE_PATCH_BYTES} bytes")
    return content


def parse_workspace_patch(content: bytes) -> WorkspacePatch:
    """Validate and decode one patch without writing any filesystem paths."""
    if len(content) > MAX_WORKSPACE_PATCH_BYTES:
        raise WorkspacePatchError(f"workspace patch exceeds {MAX_WORKSPACE_PATCH_BYTES} bytes")
    members: dict[str, tuple[bytes, int]] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            entries = archive.getmembers()
            if len(entries) > MAX_WORKSPACE_PATCH_ENTRIES + 1:
                raise WorkspacePatchError("workspace patch has too many entries")
            for member in entries:
                name = _normalized_name(member.name)
                if name in members:
                    raise WorkspacePatchError(f"duplicate workspace patch path: {name}")
                if not member.isfile():
                    raise WorkspacePatchError(
                        f"workspace patch entry must be a regular file: {member.name}"
                    )
                total += member.size
                if total > MAX_WORKSPACE_PATCH_UNPACKED_BYTES:
                    raise WorkspacePatchError("workspace patch expands beyond its limit")
                source = archive.extractfile(member)
                if source is None:
                    raise WorkspacePatchError(
                        f"workspace patch entry has no content: {member.name}"
                    )
                members[name] = (source.read(), member.mode & 0o777)
    except WorkspacePatchError:
        raise
    except (tarfile.TarError, OSError, EOFError) as error:
        raise WorkspacePatchError("workspace patch must be a gzip tar archive") from error

    manifest_entry = members.pop(_MANIFEST_PATH, None)
    if manifest_entry is None:
        raise WorkspacePatchError("workspace patch has no manifest")
    try:
        raw = json.loads(manifest_entry[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise WorkspacePatchError("workspace patch manifest is not valid JSON") from error
    if not isinstance(raw, dict) or raw.get("version") != _VERSION:
        raise WorkspacePatchError("unsupported workspace patch version")
    raw_operations = raw.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise WorkspacePatchError("workspace patch has no operations")

    operations: list[WorkspacePatchOperation] = []
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    for raw_operation in raw_operations:
        operation = _parse_operation(raw_operation)
        if operation.path in seen:
            raise WorkspacePatchError(f"duplicate workspace patch operation: {operation.path}")
        seen.add(operation.path)
        data_name = f"{_DATA_PREFIX}{operation.path}"
        entry = members.pop(data_name, None)
        if operation.after is None:
            if entry is not None:
                raise WorkspacePatchError(
                    f"deleted workspace patch path has replacement data: {operation.path}"
                )
        else:
            if entry is None:
                raise WorkspacePatchError(
                    f"workspace patch has no replacement data: {operation.path}"
                )
            body, mode = entry
            if _digest(body) != operation.after.sha256:
                raise WorkspacePatchError(f"workspace patch digest mismatch: {operation.path}")
            if mode != operation.after.mode:
                raise WorkspacePatchError(f"workspace patch mode mismatch: {operation.path}")
            files[operation.path] = body
        operations.append(operation)
    if members:
        extra = next(iter(sorted(members)))
        raise WorkspacePatchError(f"unexpected workspace patch entry: {extra}")
    return WorkspacePatch(operations=tuple(operations), files=files)


def _read_workspace_archive(content: bytes) -> dict[str, tuple[bytes, int]]:
    """Read regular files from a trusted full snapshot used to calculate a patch."""
    files: dict[str, tuple[bytes, int]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            entries = archive.getmembers()
            if len(entries) > MAX_WORKSPACE_PATCH_ENTRIES:
                raise WorkspacePatchError("workspace archive has too many entries")
            total = 0
            for member in entries:
                name = _normalized_name(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise WorkspacePatchError(
                        f"workspace entry must be a regular file or directory: {member.name}"
                    )
                if name in files:
                    raise WorkspacePatchError(f"duplicate workspace path: {name}")
                total += member.size
                if total > MAX_WORKSPACE_PATCH_UNPACKED_BYTES:
                    raise WorkspacePatchError("workspace archive expands beyond its limit")
                source = archive.extractfile(member)
                if source is None:
                    raise WorkspacePatchError(f"workspace file has no content: {name}")
                files[name] = (source.read(), member.mode & 0o777)
    except WorkspacePatchError:
        raise
    except (tarfile.TarError, OSError, EOFError) as error:
        raise WorkspacePatchError("workspace must be a gzip tar archive") from error
    return files


def _parse_operation(value: object) -> WorkspacePatchOperation:
    if not isinstance(value, dict):
        raise WorkspacePatchError("workspace patch operation must be an object")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise WorkspacePatchError("workspace patch operation has no path")
    path = _normalized_name(path_value)
    if path == ".":
        raise WorkspacePatchError("workspace patch operation cannot target the root")
    before = _parse_state(value.get("before"))
    after = _parse_state(value.get("after"))
    if before == after:
        raise WorkspacePatchError(f"workspace patch operation does not change: {path}")
    return WorkspacePatchOperation(path=path, before=before, after=after)


def _parse_state(value: object) -> PatchFileState | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise WorkspacePatchError("workspace patch file state must be an object")
    sha256 = value.get("sha256")
    mode = value.get("mode")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or not 0 <= mode <= 0o777
    ):
        raise WorkspacePatchError("workspace patch file state is invalid")
    return PatchFileState(sha256=sha256, mode=mode)


def _state(entry: tuple[bytes, int] | None) -> PatchFileState | None:
    if entry is None:
        return None
    body, mode = entry
    return PatchFileState(sha256=_digest(body), mode=mode)


def _state_json(state: PatchFileState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {"sha256": state.sha256, "mode": state.mode}


def _add_bytes(archive: tarfile.TarFile, name: str, body: bytes, mode: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(body)
    info.mode = mode
    info.mtime = 0
    archive.addfile(info, io.BytesIO(body))


def _normalized_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise WorkspacePatchError(f"unsafe workspace patch path: {name}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    return PurePosixPath(*parts).as_posix() if parts else "."


def _digest(body: bytes) -> str:
    return hashlib.sha256(body, usedforsecurity=False).hexdigest()

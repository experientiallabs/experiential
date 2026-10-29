"""Durable publication and secure reads of digest-addressed external artifact files."""

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from uuid import uuid4


def _read_artifact_file_snapshot(directory: Path, relative_path: str) -> bytes:
    """Read a regular descendant without following a replaced path component."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure artifact reads require O_NOFOLLOW")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_NOFOLLOW)
    current_fd = directory_fd
    try:
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise OSError("artifact directory is not a directory")
        parts = PurePosixPath(relative_path).parts
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise OSError("artifact data path has a non-directory component")
            except OSError:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise OSError("artifact data path is not a regular file")
            return _read_file_descriptor(file_descriptor)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(current_fd)


def _read_file_descriptor(file_descriptor: int) -> bytes:
    """Read all available bytes from one already-open regular file descriptor."""
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _fsync_directory_strict(directory: Path) -> None:
    """Persist a staged directory before atomically exposing it to readers."""
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def publish_blob(root: Path, relative_directory: PurePosixPath, payload: bytes) -> str:
    """Publish and fsync immutable bytes before their database reference can commit.

    Args:
        root: Explicit content root, the boundary for all descendant directory opens.
        relative_directory: Validated project blob directory beneath the root.
        payload: Exact artifact bytes whose digest is the final blob name.

    Returns:
        SHA-256 digest naming the durably published blob.
    """
    if relative_directory.is_absolute() or ".." in relative_directory.parts:
        raise ValueError("Blob directory must be a relative descendant.")
    digest = hashlib.sha256(payload).hexdigest()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _fsync_directory_strict(root.parent)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    staging = f".{uuid4().hex}.partial"
    try:
        for component in relative_directory.parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            else:
                os.fsync(descriptor)
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            staging,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            mode=0o600,
            dir_fd=descriptor,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    staging,
                    digest,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if not stat.S_ISREG(os.fstat(existing).st_mode) or (
                        _read_file_descriptor(existing) != payload
                    ):
                        raise ValueError(
                            "Existing artifact blob differs from its digest."
                        ) from None
                finally:
                    os.close(existing)
            os.fsync(descriptor)
        finally:
            os.unlink(staging, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    return digest

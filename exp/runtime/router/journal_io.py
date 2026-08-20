"""Filesystem durability helpers for routed interaction journals."""

from __future__ import annotations

import os
from pathlib import Path


class RuntimeJournalError(ValueError):
    """The runtime journal is corrupt or an attempted transition is invalid."""


def prepare_runtime_directory(path: Path) -> None:
    """Create a private runtime directory and reject symlinked journal targets.

    Args:
        path: Exact journal file path that will be opened without following symlinks.

    Raises:
        RuntimeJournalError: The directory or journal path is a symlink or invalid target.
    """
    directory = path.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeJournalError("runtime journal directory must be a real directory")
    if path.is_symlink():
        raise RuntimeJournalError("runtime journal path cannot be a symbolic link")


def fsync_directories(directories: tuple[Path, ...]) -> None:
    """Persist the journal and every project directory entry that names it.

    Args:
        directories: Ordered project directories whose entries must reach durable storage.

    Raises:
        RuntimeJournalError: Any supported directory cannot be opened or synchronized.
    """
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in directories:
        try:
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise RuntimeJournalError(
                f"cannot persist runtime journal directory {directory}"
            ) from exc


def truncate_torn_tail(path: Path) -> None:
    """Remove only a non-newline-terminated final record before the next append.

    Args:
        path: Journal whose possible crash-torn final record should be repaired.

    Raises:
        RuntimeJournalError: An existing journal cannot be read, truncated, or synchronized.
    """
    try:
        with path.open("r+b") as handle:
            payload = handle.read()
            if not payload or payload.endswith(b"\n"):
                return
            last_newline = payload.rfind(b"\n")
            handle.seek(last_newline + 1)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeJournalError(f"cannot repair torn runtime journal {path}") from exc

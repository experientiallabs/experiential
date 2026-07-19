"""Private nonblocking POSIX file leases shared by durable local stores."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if os.name == "posix":
    import fcntl
else:
    fcntl = None


@contextmanager
def exclusive_posix_file_lease(
    path: Path,
    *,
    unsupported_error: RuntimeError,
    irregular_file_error: OSError,
    contention_error: RuntimeError,
) -> Iterator[None]:
    """Hold one crash-safe POSIX lease without following the final path component."""
    if fcntl is None:
        raise unsupported_error

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with _exclusive_posix_descriptor_lease(
            descriptor,
            irregular_file_error=irregular_file_error,
            contention_error=contention_error,
        ):
            yield
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_posix_file_lease_at(
    directory_descriptor: int,
    name: str,
    *,
    unsupported_error: RuntimeError,
    irregular_file_error: OSError,
    contention_error: RuntimeError,
) -> Iterator[None]:
    """Hold one POSIX lease relative to an already pinned directory descriptor."""
    if fcntl is None:
        raise unsupported_error
    _validate_leaf_name(name)
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        with _exclusive_posix_descriptor_lease(
            descriptor,
            irregular_file_error=irregular_file_error,
            contention_error=contention_error,
        ):
            yield
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_posix_descriptor_lease(
    descriptor: int,
    *,
    irregular_file_error: OSError,
    contention_error: RuntimeError,
) -> Iterator[None]:
    if fcntl is None:  # pragma: no cover - guarded by each public entry point
        raise RuntimeError("POSIX file locking is unavailable")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise irregular_file_error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise contention_error from exc
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _validate_leaf_name(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError("file lease name must be one path component")

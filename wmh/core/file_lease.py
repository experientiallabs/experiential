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
    with _close_descriptor(descriptor):
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise irregular_file_error
        with _exclusive_posix_descriptor_lease(descriptor, contention_error=contention_error):
            yield


@contextmanager
def exclusive_posix_directory_lease(
    directory_descriptor: int,
    *,
    unsupported_error: RuntimeError,
    irregular_directory_error: OSError,
    contention_error: RuntimeError,
) -> Iterator[None]:
    """Lease one already pinned directory inode without owning its descriptor."""
    if fcntl is None:
        raise unsupported_error
    if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
        raise irregular_directory_error
    with _exclusive_posix_descriptor_lease(
        directory_descriptor,
        contention_error=contention_error,
    ):
        yield


@contextmanager
def _exclusive_posix_descriptor_lease(
    descriptor: int,
    *,
    contention_error: RuntimeError,
) -> Iterator[None]:
    if fcntl is None:  # pragma: no cover - guarded by each public entry point
        raise RuntimeError("POSIX file locking is unavailable")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise contention_error from exc
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"file lease unlock also failed: {cleanup_error}")


@contextmanager
def _close_descriptor(descriptor: int) -> Iterator[None]:
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"file lease descriptor close also failed: {cleanup_error}")

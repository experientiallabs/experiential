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
    finally:
        os.close(descriptor)

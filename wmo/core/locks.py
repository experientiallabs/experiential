"""The cross-process write lock for a file that is read, edited, and written back.

Separate from `wmo.core.files` because this is the half that needs `fcntl`, which is Unix only.
Atomic writing needs no locking, and `.wmo/config.toml`, `.wmo/settings.toml`, a model card, a
fitted policy and an outcome matrix are all written whole from their source of truth, so they take
`write_text_atomic` alone. Keeping the two in one module put `fcntl` on the import path of
`wmo.config`, and therefore of almost every command.

Rename atomicity is what stops a reader seeing half a file. It does NOT stop two writers from each
reading the same file, each applying an edit, and the later write erasing the earlier one with both
commands reporting success. A roster or an alias table needs this as well.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# How long a writer waits for another process to finish. These writes are a few file operations,
# so real contention never comes close; the bound exists so a hung holder is REPORTED instead of
# hanging the terminal forever.
DEFAULT_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.01


class FileLockTimeout(RuntimeError):
    """Another process held a file's write lock for longer than the bounded wait."""


@contextmanager
def file_write_lock(
    path: Path, *, what: str, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S
) -> Iterator[None]:
    """Hold the exclusive cross-process write lock for `path`, for a read-modify-write cycle.

    Take this around the WHOLE cycle, not just the write. Atomic replace alone does not stop two
    writers from each reading the same file, each applying their own edit, and the later write
    erasing the earlier one, with both commands reporting success: a registration an operator made
    is simply not in the file, and nothing says so.

    The lock sits in a sibling `<name>.lock` file, not in the file itself: writing goes through an
    atomic rename, which swaps the inode, so a lock taken on the file would stop protecting
    anything the moment the first writer landed.

    `flock` belongs to the open file description, so the kernel drops it when the holder exits,
    crashes, or is killed. A leftover `.lock` FILE is therefore never a held lock and can never
    wedge a later run (which is also why it is left in place: unlinking it would let a waiter
    block on an inode nobody else can reach). A live holder that hangs still could, so the wait is
    bounded and reports what to do instead of blocking forever.

    NOT REENTRANT. Each `with` opens its own file description, and flock conflicts between two
    descriptions of one file even inside a single process, so nesting this on the same path blocks
    for the full timeout and then reports a stuck OTHER process, which would be a lie. A caller
    that wants several edits to land together (promoting `champion` and `staging` in one step)
    must take the lock ONCE around all of them rather than calling a locked helper per edit.

    Args:
        path: The file being written. The lock file is created beside it.
        what: The noun for the error message ("the model pool"), naming what is being written from
            the perspective of someone who ran a command, not the file's role in the code.
        timeout_s: Seconds to keep retrying the lock before raising.

    Yields:
        None, with the lock held; it is released when the block exits, on any path.

    Raises:
        FileLockTimeout: The lock was still held after `timeout_s`.
        OSError: The lock could not be taken at all, which on a filesystem with no working
            advisory locks (NFS without lockd, some container volume drivers) is ENOLCK.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        f"another process has been writing {what} at {path} for more than "
                        f"{timeout_s:g}s (lock file {lock_path}); this write takes milliseconds, "
                        "so retry, and if it keeps failing look for a stuck process holding that "
                        "lock (the lock is released automatically when that process exits, so "
                        "the lock file itself is never the problem)"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

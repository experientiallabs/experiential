"""The cross-process write lock for a file that is read, edited, and written back.

Rename atomicity (`wmo.common.core.files`) stops a reader seeing half a file. It does not stop two
writers reading the same file, each applying an edit, and the later write erasing the earlier one
with both commands reporting success. Mutable project state and resumable run manifests need this.

Kept in its own module because most files WMO writes are rendered whole from their source of truth
and only need `write_text_atomic`; project configuration, review state, and run manifests need the
stronger read-modify-write boundary. Keeping locking separate leaves its dependency off read-only
paths.

`filelock` rather than a direct `fcntl.flock`: it is the same advisory-lock semantics, but it
selects `fcntl` on POSIX and `msvcrt` on Windows, so nothing here prevents WMO from importing on a
non-POSIX platform. It is declared directly in `pyproject.toml` because this module imports it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from wmo.common.core.files import resolve_write_target

# How long a writer waits for another process to finish. These writes are a few file operations,
# so real contention never comes close; the bound exists so a hung holder is REPORTED instead of
# hanging the terminal forever.
DEFAULT_LOCK_TIMEOUT_S = 10.0


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

    The sibling is chosen beside the file the write will actually LAND on, via
    `resolve_write_target`, not beside `path` as given. Two callers reaching one target through
    different links would otherwise take one lock each, both succeed, and lose an update apiece:
    exactly what this exists to prevent.

    The lock is released by the OS when the holder exits, crashes, or is killed, so a leftover
    `.lock` FILE is never a held lock and can never wedge a later run. That is also why it is left
    in place: unlinking it would let a waiter block on an inode nobody else can reach. A live
    holder that hangs still could, so the wait is bounded and reports what to do about it.

    NOT REENTRANT. Each `with` takes its own lock on the file, and two holders of one file conflict
    even inside a single process, so nesting this on the same path blocks for the full timeout and
    then reports a stuck OTHER process, which would be a lie. A caller that wants several edits to
    land together (promoting `champion` and `staging` in one step) must take the lock ONCE around
    all of them rather than calling a locked helper per edit.

    Args:
        path: The file being written. The lock file is created beside it.
        what: The noun for the error message ("project configuration"), naming what is written from
            the perspective of someone who ran a command, not the file's role in the code.
        timeout_s: Seconds to keep retrying the lock before raising.

    Yields:
        None, with the lock held; it is released when the block exits, on any path.

    Raises:
        FileLockTimeout: The lock was still held after `timeout_s`.
    """
    target = resolve_write_target(path)
    lock_path = target.with_name(f"{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path, timeout=timeout_s, mode=0o600)
    try:
        lock.acquire()
    except Timeout:
        raise FileLockTimeout(
            f"another process has been writing {what} at {path} for more than "
            f"{timeout_s:g}s (lock file {lock_path}); this write takes milliseconds, so retry, "
            "and if it keeps failing look for a stuck process holding that lock (the lock is "
            "released automatically when that process exits, so the lock file itself is never "
            "the problem)"
        ) from None
    try:
        yield
    finally:
        lock.release()

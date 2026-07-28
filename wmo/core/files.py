"""Durable writes for the small files wmo owns: atomic replace, plus a cross-process write lock.

Every file this covers is a registry or a config that a later command reads back: the candidate
pool roster, a store's `aliases.toml`, `.wmo/config.toml`, `.wmo/settings.toml`, a distill run's
resume state, a fitted policy, a paid sweep's outcome matrix. They share one failure mode, and it
is not losing the file. It is leaving a TRUNCATED or EMPTY file behind an apparently successful
write, so the next command fails to parse something nobody edited, and the recovery is hand-repair.

The two helpers answer different halves of that, which is why they are separate. Rename atomicity
means a reader never sees half a file, but two writers doing read-modify-write still overwrite
each other with both reporting success. A file written whole from its source of truth (a config
snapshot, a rendered document) needs `write_text_atomic` alone; a file that is read, edited, and
written back (a roster, an alias table) needs `file_write_lock` around the whole cycle too.

Not covered here: the credential and session writers (`wmo.config.dotenv`,
`wmo.platform.credentials`, `wmo.connect.credentials`, `wmo.cli.session_state`,
`wmo.cli.workspace_sync`). Those need 0600 from the moment of creation and refuse to follow a
symlink, which is a different contract, not a stricter setting of this one.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

# How long a writer waits for another process to finish. These writes are a few file operations,
# so real contention never comes close; the bound exists so a hung holder is REPORTED instead of
# hanging the terminal forever.
DEFAULT_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.01


class FileLockTimeout(RuntimeError):
    """Another process held a file's write lock for longer than the bounded wait."""


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` so a reader sees the previous file or the whole new one.

    Four properties, each answering a failure this codebase has actually shipped:

    - **Same-directory staging file plus `replace`.** A rename within a filesystem is atomic, so a
      crash or a full disk leaves the previous file intact instead of a truncated one. Writing in
      place does not: a half-written `aliases.toml` loses the champion pointer that
      `resolve_version` depends on, and a half-written `checkpoints.json` breaks resume exactly
      when the crash that caused it made resume necessary.
    - **A staging name unique PER CALL.** A shared fixed `.tmp` lets two writers rename each
      other's half-written file into place, which turns a lost update into a corrupt file. The
      name is a uuid rather than the pid because two THREADS of one process collide on a pid.
    - **fsync on the payload and then on its parent directory.** Without both, a power loss can
      persist the rename while the data blocks are lost, leaving an empty file behind a write that
      reported success. These files are small and written rarely: measured at 0.08 ms per write,
      about 0.1 s across a 200-step distill run whose steps are minutes of paid rollouts.
    - **The destination's mode is carried over.** `replace` installs a NEW inode, so without this
      a file an operator or an installer had restricted comes back as 0644 on the first write.

    Cleanup is `BaseException`, not `OSError`: a Ctrl-C between the write and the rename would
    otherwise strand a staging file in an artifact directory that serving and the fitter scan.

    Args:
        path: The destination file. Its parent directory is created when missing.
        payload: The complete file contents.

    Raises:
        OSError: The write, the rename, or either fsync failed. Whether `path` changed depends on
            where it failed, and the two cases are distinguishable by the caller only if it looks:
            BEFORE the rename (the common case: no space, no permission) `path` is untouched and
            the staging file is cleaned up; AFTER it (the parent-directory fsync, which some FUSE
            and NFS mounts reject) `path` already holds `payload` and only its durability against
            a power loss is unproven. Neither case can leave a partly written `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with staging.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            # `replace` installs the staging inode, so the destination's mode has to be carried
            # over explicitly or a restricted file silently widens to the umask default.
            staging.chmod(path.stat().st_mode & 0o7777)
        staging.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        staging.unlink(missing_ok=True)  # never leave a stray staging file beside the real one
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """`write_bytes_atomic` for UTF-8 text; see it for the guarantees.

    Encoding happens BEFORE anything touches the filesystem, so text that cannot be encoded (a
    lone surrogate reaching `json.dumps(..., ensure_ascii=False)`) fails without creating a
    staging file at all.
    """
    write_bytes_atomic(path, text.encode("utf-8"))


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

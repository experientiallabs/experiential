"""Atomic writes for the small files wmo owns: a reader sees the old file or the whole new one.

Every file this covers is durable state that a later command reads back: project configuration,
the model catalog, telemetry settings, review state, SFT run manifests, fitted policies, and paid
simulation outcomes. They share one failure mode: a TRUNCATED or EMPTY file behind an apparently
successful write, leaving the next command unable to parse state nobody edited.

This is only half the problem. Rename atomicity means a reader never sees half a file, but two
writers doing read-modify-write still overwrite each other with both reporting success. A file
written whole from its source of truth (an artifact or rendered policy) needs
`write_text_atomic` alone; a file that is read, edited, and written back (project configuration,
review state, or a run manifest) needs `wmo.common.core.locks.file_write_lock` around the whole
cycle too. The lock remains separate so read-only import paths do not load its dependency.

Not covered here: credential writers such as `wmo.common.config.dotenv`. Those need 0600 from the
moment of creation and refuse to follow a symlink, which is a different contract, not a stricter
setting of this one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` so a reader sees the previous file or the whole new one.

    Four properties, each answering a distinct durable-state failure:

    - **Same-directory staging file plus `replace`.** A rename within a filesystem is atomic, so a
      crash or a full disk leaves the previous file intact instead of a truncated one. Writing in
      place does not: a half-written `project.toml` loses durable configuration, and a
      half-written run manifest breaks resume exactly when the crash made resume necessary.
    - **A staging name unique PER CALL.** A shared fixed `.tmp` lets two writers rename each
      other's half-written file into place, which turns a lost update into a corrupt file. The
      name is a uuid rather than the pid because two THREADS of one process collide on a pid.
    - **fsync on the payload and then on its parent directory.** Without both, a power loss can
      persist the rename while the data blocks are lost, leaving an empty file behind a write that
      reported success. These files are small and written rarely: measured at 0.08 ms per write,
      about 0.1 s across a 200-step SFT run whose steps are minutes of paid work.
    - **The destination's mode is carried over.** `replace` installs a NEW inode, so without this
      a file an operator or an installer had restricted comes back as 0644 on the first write.
    - **A symlinked destination is written THROUGH, not replaced.** See `resolve_write_target`.

    Cleanup is `BaseException`, not `OSError`: a Ctrl-C between the write and the rename would
    otherwise strand a staging file in an artifact directory that serving and the fitter scan.

    Args:
        path: The destination file, or a symlink to it. Its parent directory is created when
            missing.
        payload: The complete file contents.

    Raises:
        OSError: The write or the rename failed, which means the destination is untouched and the
            staging file has been cleaned up. Everything that can fail here fails BEFORE the
            rename, so a raised error always means the write did not land: see
            `fsync_directory_best_effort` for the one step that is deliberately best effort.
    """
    path = resolve_write_target(path)
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
    except BaseException:
        staging.unlink(missing_ok=True)  # never leave a stray staging file beside the real one
        raise
    fsync_directory_best_effort(path.parent)


def resolve_write_target(path: Path) -> Path:
    """The real file to write, following `path` if it is a symlink.

    An in-place `write_text` follows a symlink: an operator who points `project.toml` at a
    shared or version-controlled file gets their writes where they asked for them. `replace` does
    the opposite, and silently: it swaps the LINK for a regular file and leaves the target holding
    stale contents, with nothing raised and nothing to notice until something reads the target and
    gets an old answer.

    Resolving keeps every writer, including `project.toml`, `models.toml`, and `settings.toml`,
    writing where the operator pointed it. Following the link is what they asked for in every case;
    nothing wants a symlink quietly turned into a file.

    Resolving is also what keeps the write atomic, and what keeps the LOCK correct.
    `wmo.common.core.locks.file_write_lock` derives its lock file from this answer, so two callers
    reaching one target through different symlinks contend on one lock instead of taking a lock
    each and both proceeding. And the staging file has to sit beside the FINAL target, because
    `replace` across filesystems fails with EXDEV, and a symlink into another mount is exactly the
    case where staging beside the link would break the rename.

    A broken symlink resolves to its missing target, which is then created: writing through a
    dangling link is what an in-place write would have done too.

    Raises:
        OSError: `path` is a symlink loop, or resolution otherwise fails. Better surfaced than
            written around.
    """
    return path.resolve() if path.is_symlink() else path


def fsync_directory_best_effort(directory: Path) -> None:
    """Persist the rename itself. Best effort, and deliberately so: never raises `OSError`.

    Every step is individually guarded, including the close. `close` is not a formality on every
    filesystem: NFS reports deferred write errors there, so it can fail with EIO after everything
    else succeeded, and letting that out would recreate the exact bug this function was written to
    remove one line further down.

    The rename has already landed by the time this runs, so raising here would report a write that
    DID happen as a failure, and a caller acting on that (reporting a failed promotion, retrying,
    rolling back) would be acting on the opposite of the truth. Worse, some FUSE, SMB and NFS
    mounts reject `fsync` on a directory fd unconditionally, which would turn every write on such
    a mount into an error for a file that was written correctly every time.

    What is lost by not raising is narrow and worth naming: without this fsync a power loss can
    persist the rename while the data blocks are lost, leaving an empty file behind a write that
    reported success. That is a real risk, so the failure is logged rather than swallowed, but it
    is a durability risk on a crash, not a correctness problem now, and the file is intact for
    every reader that looks before then.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        logger.warning(
            "could not open %s to persist a rename (the write itself landed): %s", directory, exc
        )
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        logger.warning(
            "could not fsync %s after a rename, so the write is not proven durable against a "
            "power loss (the file itself is written): %s",
            directory,
            exc,
        )
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            logger.warning(
                "could not close the directory handle for %s after a rename (the file itself is "
                "written): %s",
                directory,
                exc,
            )


def write_text_atomic(path: Path, text: str) -> None:
    """`write_bytes_atomic` for UTF-8 text; see it for the guarantees.

    Encoding happens BEFORE anything touches the filesystem, so text that cannot be encoded (a
    lone surrogate reaching `json.dumps(..., ensure_ascii=False)`) fails without creating a
    staging file at all.
    """
    write_bytes_atomic(path, text.encode("utf-8"))

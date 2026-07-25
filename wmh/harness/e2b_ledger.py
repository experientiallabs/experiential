"""A durable record of the E2B sandboxes this machine created, so they stay reapable by id.

E2B caps concurrent sandboxes per account. Every wmh harbor trial creates one sandbox with a
long timeout (hours), so a run that dies without graceful shutdown (crash, SIGKILL, budget
abort, machine sleep) leaves its sandboxes running until they expire on their own. The next
sweep then starves at the cap and every trial fails at sandbox creation with
`RateLimitException: 429 ... maximum number of concurrent E2B sandboxes`, which in a distill
run looks exactly like a model producing zero token spans.

This ledger makes those orphans reapable BY EXACT ID: each create appends one line to a
per-owning-process JSONL file under the user-global WMH state dir (`$WMH_HOME` or `~/.wmh`),
and each proven cleanup appends a matching release line. Writes are append + fsync, so a hard
kill still leaves the record on disk; a file whose every recorded id is released is deleted.
Killing exactly the ids a dead process recorded is defensible, whereas pattern-matching the
account is not, which is why `wmh e2b reap` treats this file as its primary evidence.

Ledger bookkeeping is best-effort by design: a write failure logs a warning and never fails
the trial that owns the sandbox. Nothing here imports the e2b SDK, so reading the ledger needs
neither the optional extra nor credentials.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from wmh.platform.credentials import wmh_home

logger = logging.getLogger(__name__)

LEDGER_DIRNAME = "e2b-sandboxes"
"""Subdirectory of the WMH user state dir holding one JSONL file per owning process."""

HARBOR_ENV_SESSION_SUFFIX = "__env"
"""Harbor tags a trial's environment sandbox with `session_id = "<trial name>__env"`."""


def harbor_trial_name(session_id: str) -> str | None:
    """The harbor trial name behind an environment session id, or None when it is not one."""
    if not session_id.endswith(HARBOR_ENV_SESSION_SUFFIX):
        return None
    trial = session_id[: -len(HARBOR_ENV_SESSION_SUFFIX)]
    return trial or None


def ledger_dir() -> Path:
    """The ledger directory (`$WMH_HOME/e2b-sandboxes`, else `~/.wmh/e2b-sandboxes`)."""
    return wmh_home() / LEDGER_DIRNAME


def pid_is_alive(pid: int) -> bool:
    """Whether the process that owns a ledger file still exists on this machine."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, it just belongs to another user
    return True


class SandboxCreated(BaseModel):
    """One sandbox a process created, recorded the moment E2B returned it."""

    model_config = ConfigDict(frozen=True)

    event: Literal["created"] = "created"
    sandbox_id: str
    template_id: str
    """The template identity the sandbox was created from: for harbor trials that is the
    resource-qualified alias, which E2B accepts wherever it accepts a template id."""

    created_at: datetime
    trial_name: str | None = None
    """The harbor trial that owns the sandbox, when the caller knows it."""

    pid: int
    """The process that created the sandbox, probed for liveness when reaping."""


class SandboxReleased(BaseModel):
    """Proof that a recorded sandbox was killed, so no reaper needs to consider it."""

    model_config = ConfigDict(frozen=True)

    event: Literal["released"] = "released"
    sandbox_id: str
    released_at: datetime
    pid: int
    """The process that observed the release (the owner, or a reaper)."""


LedgerRecord = Annotated[SandboxCreated | SandboxReleased, Field(discriminator="event")]
_RECORD_ADAPTER: TypeAdapter[SandboxCreated | SandboxReleased] = TypeAdapter(LedgerRecord)


class LedgerFile(BaseModel):
    """One owning process's ledger: where it lives, whose it is, and what it still holds."""

    model_config = ConfigDict(frozen=True)

    path: Path
    owner_pid: int
    held: tuple[SandboxCreated, ...]
    """Recorded sandboxes with no matching release record."""

    released_ids: tuple[str, ...]

    @property
    def fully_released(self) -> bool:
        """Whether every sandbox this file recorded has a release record."""
        return not self.held and bool(self.released_ids)


def read_ledger_files(directory: Path | None = None) -> tuple[LedgerFile, ...]:
    """Read every ledger file in `directory`, sorted by path.

    Unparseable lines are skipped: a hard kill can tear the last line of an append, and one
    torn record must not hide the intact records above it.

    Args:
        directory: The ledger directory; defaults to the user-global one.

    Returns:
        One `LedgerFile` per readable `*.jsonl` file (an unreadable file is skipped with a
        warning, since a reap must still consider the rest).
    """
    root = directory if directory is not None else ledger_dir()
    if not root.is_dir():
        return ()
    files: list[LedgerFile] = []
    for path in sorted(root.glob("*.jsonl")):
        parsed = _read_one(path)
        if parsed is not None:
            files.append(parsed)
    return tuple(files)


def _read_one(path: Path) -> LedgerFile | None:
    """Parse one ledger file into held and released ids, or None when it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("cannot read the E2B sandbox ledger %s: %s", path, error)
        return None
    created: dict[str, SandboxCreated] = {}
    released: dict[str, None] = {}
    record_pid: int | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = _RECORD_ADAPTER.validate_json(line)
        except ValidationError:
            logger.debug("skipping an unparseable E2B ledger line in %s", path)
            continue
        record_pid = record_pid if record_pid is not None else record.pid
        if isinstance(record, SandboxCreated):
            created[record.sandbox_id] = record
        else:
            released[record.sandbox_id] = None
    return LedgerFile(
        path=path,
        owner_pid=_owner_pid(path, record_pid),
        held=tuple(record for key, record in created.items() if key not in released),
        released_ids=tuple(released),
    )


def _owner_pid(path: Path, record_pid: int | None) -> int:
    """The owning pid, read from the `<pid>-<stamp>.jsonl` name and falling back to a record."""
    head = path.name.split("-", 1)[0]
    if head.isdigit():
        return int(head)
    return record_pid if record_pid is not None else 0


def append_release(path: Path, *, sandbox_id: str, pid: int, released_at: datetime) -> None:
    """Append one release record to an existing ledger file (used by a reaper).

    Records are single short lines appended to a file opened `O_APPEND`, so a reaper writing
    into another process's ledger cannot interleave with that process's own appends.

    Raises:
        OSError: If the record cannot be written; the caller decides whether that is fatal
            (`SandboxLedger` treats its own write failures as warnings).
    """
    _append_line(path, SandboxReleased(sandbox_id=sandbox_id, released_at=released_at, pid=pid))


def prune_released_files(
    directory: Path | None = None,
    *,
    owner_alive: Callable[[int], bool] = pid_is_alive,
) -> tuple[Path, ...]:
    """Delete every ledger file whose recorded sandboxes are all released.

    A fully released file whose owning process is STILL RUNNING is kept: that process can append
    a new create record at any moment, and deleting the file underneath it would lose the only
    record of a live sandbox.

    Args:
        directory: The ledger directory; defaults to the user-global one.
        owner_alive: Liveness probe for a file's owning pid (injected in tests).

    Returns:
        The paths that were deleted.
    """
    removed: list[Path] = []
    for ledger in read_ledger_files(directory):
        if not ledger.fully_released or owner_alive(ledger.owner_pid):
            continue
        try:
            ledger.path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("cannot delete the released E2B ledger %s: %s", ledger.path, error)
            continue
        removed.append(ledger.path)
    return tuple(removed)


class SandboxLedger:
    """Append-only record of the E2B sandboxes one process created.

    The file is named `<pid>-<utc stamp>.jsonl` and created lazily on the first record, so a
    process that never opens a sandbox leaves nothing behind. Recording is best-effort: any
    write failure is logged and swallowed, because losing a ledger line must never fail the
    trial whose sandbox is already running.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        pid: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Record into `directory` (default: the user-global ledger dir) as `pid`."""
        self._directory = directory if directory is not None else ledger_dir()
        self._pid = pid if pid is not None else os.getpid()
        self._now = now
        self._lock = Lock()
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        """This process's ledger file (the name is fixed on first access)."""
        with self._lock:
            if self._path is None:
                stamp = self._now().strftime("%Y%m%dT%H%M%SZ")
                self._path = self._directory / f"{self._pid}-{stamp}.jsonl"
            return self._path

    def record_created(
        self,
        *,
        sandbox_id: str,
        template_id: str,
        trial_name: str | None = None,
    ) -> None:
        """Record a sandbox that E2B just created, flushing it to disk before returning."""
        self._append(
            SandboxCreated(
                sandbox_id=sandbox_id,
                template_id=template_id,
                created_at=self._now(),
                trial_name=trial_name,
                pid=self._pid,
            )
        )

    def record_released(self, sandbox_id: str) -> None:
        """Record a sandbox whose kill this process proved, so no reaper considers it."""
        self._append(SandboxReleased(sandbox_id=sandbox_id, released_at=self._now(), pid=self._pid))

    def _append(self, record: SandboxCreated | SandboxReleased) -> None:
        """Append one record; a failure warns instead of propagating into the caller."""
        path = self.path
        try:
            _ensure_directory(path.parent)
            with self._lock:
                _append_line(path, record)
        except OSError as error:
            logger.warning(
                "could not record E2B sandbox %s (%s) in the ledger %s: %s",
                record.sandbox_id,
                record.event,
                path,
                error,
            )


def _append_line(path: Path, record: SandboxCreated | SandboxReleased) -> None:
    """Append one JSONL record and push it to disk before returning."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{record.model_dump_json()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ensure_directory(directory: Path) -> None:
    """Create every missing level owner-only, matching the rest of the WMH state dir.

    `mkdir(parents=True)` would create intermediate levels with the umask default, which for
    `~/.wmh` is a permanent 0755 around state that names live cloud resources. A umask can only
    narrow 0o700, so creating each level with that mode closes the window; existing directories
    are left alone.
    """
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for level in reversed(missing):
        with contextlib.suppress(FileExistsError):
            level.mkdir(mode=0o700)


_DEFAULT_LEDGER: SandboxLedger | None = None
_DEFAULT_GUARD = Lock()


def default_ledger() -> SandboxLedger:
    """The process-wide ledger every wmh-created E2B sandbox is recorded in."""
    global _DEFAULT_LEDGER
    with _DEFAULT_GUARD:
        if _DEFAULT_LEDGER is None:
            _DEFAULT_LEDGER = SandboxLedger()
        return _DEFAULT_LEDGER

"""Read-only identity-scoped consumption of the native gateway capture database."""

import errno
import os
import sqlite3
import stat
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope

_NOT_A_REGULAR_FILE = (
    "capture database must be a regular file; use a database path without a file symlink"
)


_ABSENT_DATABASE = "capture database is not present at that path"

_ENTRY_REPLACED = (
    "capture database was replaced while it was being opened; "
    "retry the read once the path is stable"
)


@dataclass(frozen=True)
class _PinnedEntry:
    """One capture entry and its directory, held open across the database open.

    Attributes:
        descriptor: Read-only descriptor on the validated final entry, or
            ``None`` where no-follow opens do not exist and the entry was
            validated by its metadata instead.
        directory: Read-only descriptor on the directory holding that entry,
            or ``None`` on the same platforms.
        directory_change_ns: That directory's change time when the entry was
            pinned, or ``None`` when the directory could not be pinned.
    """

    descriptor: int | None
    directory: int | None
    directory_change_ns: int | None


def _pin_capture_entry(path: Path) -> _PinnedEntry:
    """Open the final path entry without following it, and keep it pinned.

    ``O_NOFOLLOW`` refuses a final-entry symlink in the open itself, so there is
    no window between deciding the entry is a regular file and holding it. The
    descriptor stays open for the whole read: it pins the inode, so the identity
    checked against it cannot be recycled underneath it.

    A symlink at the database filename would redirect the reader to content the
    operator never named, and every scope check downstream then inspects the
    wrong file and finds nothing wrong. File mode is deliberately not checked:
    native capture owns this database and its permissions, so a read-only
    consumer is not the component that gets to refuse them.

    The directory is pinned as well, and its change time recorded, because the
    entry can be swapped and swapped back around the database open. That leaves
    the entry's own identity equal to the pinned one, so only the directory
    still carries evidence of it.

    Args:
        path: Capture database whose final entry has not been resolved.

    Windows has neither no-follow opens nor directory descriptors, so there the
    entry is validated by its own metadata without following it and nothing is
    pinned. That is the same split the trace store already makes: the portable
    check refuses a link, and the race hardening is POSIX-only.

    Returns:
        The pinned entry, its directory, and that directory's change time, or a
        metadata-validated entry holding no descriptors on Windows.

    Raises:
        OSError: The entry cannot be opened for a reason other than absence.
        ValueError: The path is absent, a symlink, or another non-regular file.
    """
    if os.name == "nt":
        _require_regular_entry(path)
        return _PinnedEntry(descriptor=None, directory=None, directory_change_ns=None)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Read before the entry is opened, so any swap that races the database
        # open falls inside the window this timestamp covers.
        directory_change_ns = os.fstat(directory).st_ctime_ns
        descriptor = _pin_regular_file(path)
    except BaseException:
        os.close(directory)
        raise
    return _PinnedEntry(
        descriptor=descriptor,
        directory=directory,
        directory_change_ns=directory_change_ns,
    )


def _require_regular_entry(path: Path) -> None:
    """Refuse a link or another non-regular entry without following it.

    The portable floor, used where a no-follow open is unavailable. It closes
    the substitution the issue reported; it cannot close a replacement that
    races the database open, which needs descriptors this platform lacks.

    Args:
        path: Capture database whose final entry has not been resolved.

    Raises:
        OSError: The entry's metadata cannot be read.
        ValueError: The path is absent, a symlink, or another non-regular file.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(_ABSENT_DATABASE) from error
    if not stat.S_ISREG(mode):
        raise ValueError(_NOT_A_REGULAR_FILE)


def _pin_regular_file(path: Path) -> int:
    """Open one final path entry with no-follow semantics and validate its type.

    Args:
        path: Capture database whose final entry has not been resolved.

    Returns:
        A descriptor on the validated entry.

    Raises:
        OSError: The entry cannot be opened for a reason other than absence.
        ValueError: The path is absent, a symlink, or another non-regular file.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as error:
        # Refused here rather than left to the connection: opening the pathname
        # anyway would resolve it a second time, and an entry that appears in
        # between is exactly the substitution this guards against. The caller
        # reports absence and substitution with the same actionable message.
        raise ValueError(_ABSENT_DATABASE) from error
    except OSError as error:
        # ELOOP is how O_NOFOLLOW reports "the final entry is a symlink",
        # including a dangling one, which must not read as an absent file.
        if error.errno not in {errno.ELOOP, errno.EMLINK}:
            raise
        raise ValueError(_NOT_A_REGULAR_FILE) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(_NOT_A_REGULAR_FILE)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _close_pinned_entry(pinned: _PinnedEntry) -> None:
    """Release whichever descriptors were held for one read."""
    try:
        if pinned.descriptor is not None:
            os.close(pinned.descriptor)
    finally:
        if pinned.directory is not None:
            os.close(pinned.directory)


def _require_pinned_identity(path: Path, pinned: _PinnedEntry) -> None:
    """Fail closed unless the entry SQLite opened is still the pinned one.

    SQLite is handed a pathname, not the pinned descriptor, so it resolves the
    name a second time and could open a different file than the one validated.
    Two things are checked, because either alone can be beaten.

    Comparing the name's CURRENT target against the pinned inode catches a
    straight replacement. It does not catch a replacement that is undone: an
    entry swapped out, opened by SQLite, and swapped back has the pinned
    identity again by the time it is compared.

    So the directory's change time is compared as well. Adding, removing or
    renaming an entry restages its directory, and a change time cannot be set
    back by the process that caused it, which makes the directory the one
    witness to a swap that was reverted. This runs before any query, and SQLite
    opens the database file when the connection is made, so the whole of its
    own path resolution falls inside the window compared here. A read-only
    connection does create WAL sidecars, which restage the directory too, but
    only once a statement runs.

    ``stat`` here, not ``lstat``, precisely because it must answer the question
    SQLite's own open asked: what does this name resolve to.

    Args:
        path: The database pathname handed to SQLite.
        pinned: The entry and directory held open across the database open.

    Raises:
        OSError: The path's metadata cannot be read.
        ValueError: The entry was replaced, or its directory was restaged,
            while the database was being opened.
    """
    if pinned.descriptor is None or pinned.directory is None:
        # Nothing was pinned, so there is nothing to compare against. The entry
        # was already refused unless it was a regular file, which is the whole
        # guard this platform can offer.
        return
    entry = os.fstat(pinned.descriptor)
    current = path.stat()
    if (current.st_dev, current.st_ino) != (entry.st_dev, entry.st_ino):
        raise ValueError(_ENTRY_REPLACED)
    if os.fstat(pinned.directory).st_ctime_ns != pinned.directory_change_ns:
        raise ValueError(_ENTRY_REPLACED)


@dataclass(frozen=True)
class CaptureRow:
    """One durable gateway exchange and its monotonically increasing cursor.

    Attributes:
        sequence: Database-local cursor; retention can leave gaps.
        experience: Validated exchange in the reader's identity/application scope.
    """

    sequence: int
    experience: CapturedExchange


class LocalCaptureStore:
    """Read bounded pages without acquiring write authority over captured traffic.

    Cursors are local to one database. Retention may create gaps; consumers must
    checkpoint the last returned sequence rather than assume contiguous IDs.
    """

    def __init__(self, database_path: Path, scope: LocalCaptureScope) -> None:
        """Bind an existing local content database and an explicit application scope."""
        # Resolve directory aliases without hiding a symlink at the final filename:
        # a whole-path resolve() answers with the link TARGET, which is exactly the
        # substitution the read-time check below exists to catch.
        self._path = database_path.parent.resolve() / database_path.name
        self._scope = scope

    def read_after(self, sequence: int = 0, *, limit: int = 100) -> tuple[CaptureRow, ...]:
        """Return at most one bounded page for the bound application.

        Args:
            sequence: Last consumed row sequence, or zero for retained history.
            limit: Maximum returned rows, between one and one thousand.

        Returns:
            Validated durable records ordered by increasing sequence.
        """
        if sequence < 0 or not 1 <= limit <= 1000:
            raise ValueError("sequence must be nonnegative and limit must be between 1 and 1000")
        return self._read(sequence=sequence, limit=limit)

    def read_snapshot(self) -> tuple[CaptureRow, ...]:
        """Read all retained scoped captures from one consistent SQLite read snapshot.

        This explicit convenience method materializes the snapshot. Importers use
        iter_snapshot to copy it to private disk without retaining the corpus in memory.
        """
        return self._read(sequence=0, limit=None)

    def iter_snapshot(self) -> Generator[CaptureRow]:
        """Yield retained scoped rows from one snapshot; close the iterator on early exit."""
        return self._iter_read(sequence=0, limit=None)

    def _read(self, *, sequence: int, limit: int | None) -> tuple[CaptureRow, ...]:
        """Materialize explicitly requested pages for callers needing random access."""
        return tuple(self._iter_read(sequence=sequence, limit=limit))

    def _iter_read(self, *, sequence: int, limit: int | None) -> Generator[CaptureRow]:
        """Own one snapshot and validate every payload against its durable partition."""
        # Validated per read rather than once at construction: a store outlives
        # the call that built it, so the entry can be replaced in between. Both
        # read paths land here, so neither can be left behind.
        pinned = _pin_capture_entry(self._path)
        try:
            connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        except BaseException:
            _close_pinned_entry(pinned)
            raise
        try:
            # Before any query: SQLite resolved the pathname itself, so this is
            # what proves it opened the entry that was validated.
            _require_pinned_identity(self._path, pinned)
            connection.execute("BEGIN")
            cursor = connection.execute(
                "SELECT sequence, payload FROM gateway_captures "
                "WHERE user_id = ? AND application_id = ? AND sequence > ? "
                "AND expires_at > unixepoch() ORDER BY sequence LIMIT ?",
                (
                    self._scope.user_id,
                    self._scope.application_id,
                    sequence,
                    limit if limit is not None else -1,
                ),
            )
            for row in cursor:
                experience = CapturedExchange.model_validate_json(row[1])
                if experience.scope != self._scope:
                    raise ValueError("experience payload scope differs from its durable partition")
                yield CaptureRow(sequence=row[0], experience=experience)
        finally:
            connection.close()
            # Held until the read is done: while the entry is open its inode
            # cannot be recycled, so the identity checked above stays meaningful.
            _close_pinned_entry(pinned)

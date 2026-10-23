"""Durable video job persistence: the host seam and its SQLite implementation.

The job row is what makes a video job survive the request that created it: any
process that loads it can poll the provider, settle the reservation, and
answer the owner. Hosts install their own store (the hosted platform keeps
jobs beside its ledger); the local gateway keeps them in its SQLite database
(``gateway_video_jobs``, schema 30).

Every owner-facing read is scoped to the creating organization AND identity,
so another identity's lookup of an existing id answers exactly like a missing
id. Writes to an open job are compare-and-set on ``settled = 0``: once one
process records the settled terminal state, a slower process's write cannot
regress it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol

from exp.common.sqlite.connection import persistent_connection
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.video.contracts import VideoJob

ListOrder = Literal["asc", "desc"]


class VideoJobStore(Protocol):
    """Host-installed durable persistence for video jobs."""

    def create(self, *, job: VideoJob) -> None:
        """Persist one new job before any provider call is made for it."""
        ...

    def load(self, *, video_id: str, organization_id: str, identity_id: str) -> VideoJob | None:
        """Return one undeleted job owned by exactly this identity, or None."""
        ...

    def load_any(self, *, video_id: str) -> VideoJob | None:
        """Return one job regardless of owner, for the server-side poller."""
        ...

    def save_open(self, *, job: VideoJob) -> bool:
        """Replace one unsettled job's record; False when it was already settled."""
        ...

    def mark_deleted(self, *, video_id: str, organization_id: str, identity_id: str) -> bool:
        """Hide one settled job owned by this identity; False when absent or open."""
        ...

    def list_jobs(
        self,
        *,
        organization_id: str,
        identity_id: str,
        limit: int,
        after: str | None,
        order: ListOrder,
    ) -> list[VideoJob]:
        """Return up to ``limit`` undeleted owned jobs strictly after the cursor id."""
        ...

    def open_jobs(self, *, limit: int) -> list[VideoJob]:
        """Return up to ``limit`` unsettled jobs, oldest first."""
        ...


class SQLiteVideoJobStore:
    """Video jobs in the local gateway database's ``gateway_video_jobs`` table."""

    def __init__(self, database_path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        """Bind an initialized gateway database (schema 30 or later).

        Args:
            database_path: Shared gateway SQLite path.
            busy_timeout_ms: Maximum SQLite lock wait.
        """
        self._path = database_path
        self._busy_timeout_ms = busy_timeout_ms

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Check out one reusable configured connection."""
        with persistent_connection(self._path, busy_timeout_ms=self._busy_timeout_ms) as connection:
            yield connection

    def create(self, *, job: VideoJob) -> None:
        """Insert one new job.

        Args:
            job: The job; its id must be new.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO gateway_video_jobs (
                    video_id, organization_id, identity_id, created_at, settled, deleted, job
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.video_id,
                    job.organization_id,
                    job.identity_id,
                    utc_text(job.created_at),
                    int(job.settled),
                    int(job.deleted),
                    job.model_dump_json(),
                ),
            )

    def load(self, *, video_id: str, organization_id: str, identity_id: str) -> VideoJob | None:
        """Return one undeleted job owned by this identity.

        Args:
            video_id: Public job id.
            organization_id: Caller's tenant.
            identity_id: Caller's identity.

        Returns:
            The job, or None when absent, deleted, or owned by anyone else.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job FROM gateway_video_jobs
                WHERE video_id = ? AND organization_id = ? AND identity_id = ? AND deleted = 0
                """,
                (video_id, organization_id, identity_id),
            ).fetchone()
        return None if row is None else VideoJob.model_validate_json(str(row["job"]))

    def load_any(self, *, video_id: str) -> VideoJob | None:
        """Return one job by id regardless of owner or deletion.

        Args:
            video_id: Public job id.

        Returns:
            The job, or None when absent.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT job FROM gateway_video_jobs WHERE video_id = ?", (video_id,)
            ).fetchone()
        return None if row is None else VideoJob.model_validate_json(str(row["job"]))

    def save_open(self, *, job: VideoJob) -> bool:
        """Replace one unsettled job's record (compare-and-set on ``settled = 0``).

        Args:
            job: The job's new state; ``settled`` may flip to True here.

        Returns:
            Whether the write landed; False when another writer settled it first.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE gateway_video_jobs SET job = ?, settled = ?, deleted = ?
                WHERE video_id = ? AND settled = 0
                """,
                (job.model_dump_json(), int(job.settled), int(job.deleted), job.video_id),
            )
        return cursor.rowcount == 1

    def mark_deleted(self, *, video_id: str, organization_id: str, identity_id: str) -> bool:
        """Hide one settled owned job from reads and listings.

        Args:
            video_id: Public job id.
            organization_id: Caller's tenant.
            identity_id: Caller's identity.

        Returns:
            Whether a settled, owned, undeleted job was hidden.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job FROM gateway_video_jobs
                WHERE video_id = ? AND organization_id = ? AND identity_id = ?
                  AND deleted = 0 AND settled = 1
                """,
                (video_id, organization_id, identity_id),
            ).fetchone()
            if row is None:
                return False
            job = VideoJob.model_validate_json(str(row["job"])).model_copy(update={"deleted": True})
            cursor = connection.execute(
                """
                UPDATE gateway_video_jobs SET job = ?, deleted = 1
                WHERE video_id = ? AND deleted = 0 AND settled = 1
                """,
                (job.model_dump_json(), video_id),
            )
        return cursor.rowcount == 1

    def list_jobs(
        self,
        *,
        organization_id: str,
        identity_id: str,
        limit: int,
        after: str | None,
        order: ListOrder,
    ) -> list[VideoJob]:
        """Return one page of owned, undeleted jobs ordered by creation.

        Args:
            organization_id: Caller's tenant.
            identity_id: Caller's identity.
            limit: Maximum jobs returned.
            after: Cursor id; the page starts strictly after it. An id the
                caller does not own is treated as no cursor.
            order: ``desc`` (newest first) or ``asc``.

        Returns:
            The page.
        """
        direction = "DESC" if order == "desc" else "ASC"
        comparison = "<" if order == "desc" else ">"
        with self._connect() as connection:
            cursor_row = (
                None
                if after is None
                else connection.execute(
                    """
                    SELECT created_at, rowid FROM gateway_video_jobs
                    WHERE video_id = ? AND organization_id = ? AND identity_id = ?
                    """,
                    (after, organization_id, identity_id),
                ).fetchone()
            )
            if cursor_row is None:
                rows = connection.execute(
                    f"""
                    SELECT job FROM gateway_video_jobs
                    WHERE organization_id = ? AND identity_id = ? AND deleted = 0
                    ORDER BY created_at {direction}, rowid {direction} LIMIT ?
                    """,  # noqa: S608 - direction is one of two literals.
                    (organization_id, identity_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    SELECT job FROM gateway_video_jobs
                    WHERE organization_id = ? AND identity_id = ? AND deleted = 0
                      AND (created_at, rowid) {comparison} (?, ?)
                    ORDER BY created_at {direction}, rowid {direction} LIMIT ?
                    """,  # noqa: S608 - direction and comparison are fixed literals.
                    (
                        organization_id,
                        identity_id,
                        cursor_row["created_at"],
                        cursor_row["rowid"],
                        limit,
                    ),
                ).fetchall()
        return [VideoJob.model_validate_json(str(row["job"])) for row in rows]

    def open_jobs(self, *, limit: int) -> list[VideoJob]:
        """Return the oldest unsettled jobs.

        Args:
            limit: Maximum jobs returned.

        Returns:
            Up to ``limit`` unsettled jobs, oldest first.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job FROM gateway_video_jobs WHERE settled = 0
                ORDER BY created_at ASC, rowid ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [VideoJob.model_validate_json(str(row["job"])) for row in rows]

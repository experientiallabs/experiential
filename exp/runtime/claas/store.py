"""Read-only scoped consumption of the native durable experience database."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from exp.common.claas import ClaasScope, Experience


@dataclass(frozen=True)
class ExperienceRow:
    """One durable experience and its monotonically increasing consumer cursor."""

    sequence: int
    experience: Experience


class ExperienceStore:
    """Read bounded pages without acquiring write authority over captured traffic.

    Cursors are local to one database. Retention may create gaps; consumers must
    checkpoint the last returned sequence rather than assume contiguous IDs.
    """

    def __init__(self, database_path: Path, scope: ClaasScope) -> None:
        """Bind an existing local content database and an explicit application scope."""
        self._path = database_path.resolve()
        self._scope = scope

    def read_after(self, sequence: int = 0, *, limit: int = 100) -> tuple[ExperienceRow, ...]:
        """Return at most one bounded page for the bound application.

        Args:
            sequence: Last consumed row sequence, or zero for retained history.
            limit: Maximum returned rows, between one and one thousand.

        Returns:
            Validated durable records ordered by increasing sequence.
        """
        if sequence < 0 or not 1 <= limit <= 1000:
            raise ValueError("sequence must be nonnegative and limit must be between 1 and 1000")
        connection = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            rows = connection.execute(
                "SELECT sequence, payload FROM claas_experiences "
                "WHERE user_id = ? AND application_id = ? AND sequence > ? "
                "AND expires_at > unixepoch() ORDER BY sequence LIMIT ?",
                (self._scope.user_id, self._scope.application_id, sequence, limit),
            ).fetchall()
        finally:
            connection.close()
        result = tuple(
            ExperienceRow(sequence=row[0], experience=Experience.model_validate_json(row[1]))
            for row in rows
        )
        if any(row.experience.scope != self._scope for row in result):
            raise ValueError("experience payload scope differs from its durable partition")
        return result

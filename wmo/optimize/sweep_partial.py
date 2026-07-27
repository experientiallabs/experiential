"""The sweep's crash-safe sidecar: rows on disk the moment they are measured, and the resume.

A sweep used to hold every measured cell in memory and write `matrix.json` once, at the end. One
transport fault, one Ctrl-C, one laptop lid at hour five, and a run that had already bought
hundreds of episodes had nothing to show for them. This module is the other half of that
contract: `<matrix>.partial.jsonl` gains one line per cell as the cell completes, and the next
invocation of the same plan measures only the cells the file does not already have.

The format is a JSON Lines log, not a document: line 1 is the plan identity these rows were
measured under, every later line is one `ScenarioOutcome`. Append-only, flushed and fsynced per
line, so what survives a kill is a prefix of what was measured rather than a corrupt file. The
last line for a cell wins, which is what makes a re-measured cell (`remeasured=True`) replace its
earlier row without rewriting anything.

Identity is checked, never assumed. Resuming a sidecar whose rows were measured under a different
pool, a different scenario cut, a different episode count, step budget, history window, or
compressor would silently mix two arms into one matrix, so a mismatch is refused with the field
that differs named. That check is also the answer to a gap the tau-grid runner filed against the
library: before this, nothing on disk recorded WHICH scenario cut a matrix came from.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict

from wmo.optimize.outcomes import ScenarioOutcome

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial.jsonl"
"""Appended to the matrix path: `matrix.json` -> `matrix.json.partial.jsonl`."""

PARTIAL_FORMAT_VERSION = 1
"""Bumped only by a change that makes an older sidecar unreadable, so the refusal can say so."""

IDENTITY_DIGEST_CHARS = 16
"""64 bits, the same width `OutcomeMatrix`'s provenance digest uses."""


class PartialSweepError(ValueError):
    """A sidecar cannot be used, for a reason the operator can fix.

    Raised rather than worked around on purpose: every alternative (ignore the file, delete it,
    mix its rows in) either throws away paid cells or fabricates a matrix whose rows were measured
    under two different plans. `wmo.optimize.sweep` re-raises these as `SweepError`.
    """


class PlanIdentity(BaseModel):
    """What a set of measured rows was measured UNDER: the cohort pins, as data.

    Two matrices are comparable when these agree and are different evidence when they do not, so
    this is exactly the resume check, and exactly what a sidecar records about itself. Fields are
    the properties that change what a cell MEASURES; how fast the sweep ran (concurrency) is not
    one of them, so raising concurrency mid-run resumes cleanly instead of re-buying the grid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: str  # digest of the candidate roster, prices included
    scenarios: tuple[str, ...]  # the scenario cut, by id, in sweep order
    episodes: int
    max_steps: int
    history_chars: int
    compression: str  # `wmo.optimize.compression.compression_signature`

    @property
    def digest(self) -> str:
        """A short stable hash of the whole identity, for stamping into artifacts and logs."""
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:IDENTITY_DIGEST_CHARS]

    def mismatch(self, other: PlanIdentity) -> str | None:
        """How `other` differs from this identity in operator terms, or None when it does not.

        Names ONE difference, the first in declaration order: an operator who changed the pool and
        the episode count fixes them one at a time anyway, and a wall of diffs buries the field
        that actually explains the refusal.
        """
        if self == other:
            return None
        if self.pool != other.pool:
            return "the candidate pool changed (different models, or different prices)"
        if self.scenarios != other.scenarios:
            return (
                f"the scenario cut changed ({len(other.scenarios)} scenario(s) then, "
                f"{len(self.scenarios)} now)"
            )
        if self.episodes != other.episodes:
            return f"episodes per cell changed ({other.episodes} then, {self.episodes} now)"
        if self.max_steps != other.max_steps:
            return f"the step budget changed ({other.max_steps} then, {self.max_steps} now)"
        if self.history_chars != other.history_chars:
            return (
                f"the observation window changed ({other.history_chars} chars then, "
                f"{self.history_chars} now)"
            )
        return f"the compression arm changed ({other.compression} then, {self.compression} now)"


class PartialHeader(BaseModel):
    """Line 1 of a sidecar: which plan the rows below it belong to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = PARTIAL_FORMAT_VERSION
    identity: PlanIdentity


def partial_path(out_path: Path) -> Path:
    """Where a matrix destination's sidecar lives: beside it, so one dir check covers both."""
    return out_path.with_name(out_path.name + PARTIAL_SUFFIX)


def read_partial(path: Path, identity: PlanIdentity) -> list[ScenarioOutcome]:
    """Rows a previous attempt at THIS plan measured, oldest first; empty when there are none.

    A torn final line is tolerated and dropped: the process was killed between the write and the
    flush, so that cell is simply unmeasured and gets bought again. A malformed line anywhere
    ELSE is refused, because a hole in the middle of the log means the file is not what this
    module wrote and guessing which rows are real is not a thing evidence may do.

    Raises:
        PartialSweepError: The sidecar is unreadable, was written by a newer format, or its rows
            were measured under a different plan.
    """
    if not path.is_file():
        return []
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise PartialSweepError(
            f"cannot read the partial sweep at {path}: {exc}. Delete it to start the sweep over "
            "(the cells it holds are then bought again), or fix the permissions to resume it"
        ) from exc
    if not lines:
        return []
    header = _parse_header(path, lines[0])
    difference = identity.mismatch(header.identity)
    if difference is not None:
        raise PartialSweepError(
            f"{path} holds {len(lines) - 1} cell(s) measured under a DIFFERENT plan: "
            f"{difference}. Those rows and this sweep's rows are not the same evidence, so they "
            "cannot be merged into one matrix. Re-run with the previous settings to finish that "
            f"sweep, or delete {path} to measure this plan from scratch"
        )
    return _parse_rows(path, lines[1:])


def _parse_header(path: Path, line: str) -> PartialHeader:
    """Line 1, or a refusal naming what the file is instead."""
    try:
        header = PartialHeader.model_validate_json(line)
    except ValueError as exc:
        raise PartialSweepError(
            f"{path} does not start with a partial-sweep header, so it was not written by this "
            f"sweep and its contents are unknown: {exc}. Move or delete it, then re-run"
        ) from exc
    if header.version != PARTIAL_FORMAT_VERSION:
        raise PartialSweepError(
            f"{path} was written in partial-sweep format v{header.version}; this build reads "
            f"v{PARTIAL_FORMAT_VERSION}. Finish that sweep with the build that started it, or "
            "delete the file to measure this plan from scratch"
        )
    return header


def _parse_rows(path: Path, lines: list[str]) -> list[ScenarioOutcome]:
    """Every row after the header, dropping only a torn LAST line."""
    rows: list[ScenarioOutcome] = []
    for number, line in enumerate(lines, start=2):
        try:
            rows.append(ScenarioOutcome.model_validate_json(line))
        except ValueError as exc:
            if number == len(lines) + 1:
                logger.warning(
                    "%s ends in a partly written row (line %d); that cell is measured again",
                    path,
                    number,
                )
                break
            raise PartialSweepError(
                f"{path} line {number} is not a measured cell: {exc}. The log is damaged in the "
                "middle, so which of its rows are real cannot be established; move or delete the "
                "file and re-run"
            ) from exc
    return rows


class PartialWriter:
    """Append-only writer for the sidecar: one flushed, fsynced line per completed cell.

    Not thread-safe by itself, deliberately. The sweep already serializes its per-cell callback
    under one lock (so progress printing cannot interleave), and writing under that same lock
    keeps the log's order the completion order instead of adding a second lock to reason about.
    """

    def __init__(self, path: Path, identity: PlanIdentity) -> None:
        self._path = path
        self._identity = identity
        # The matrix's directory is created by `OutcomeMatrix.save`, which does not run until the
        # end; the sidecar needs it at the start. Creating it here (rather than at plan time)
        # keeps the pre-spend path pure: an operator who declines the cost question leaves no
        # trace on disk.
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        if path.stat().st_size == 0:
            self._write(PartialHeader(identity=identity).model_dump_json())

    @property
    def path(self) -> Path:
        return self._path

    def append(self, outcome: ScenarioOutcome) -> None:
        """Persist one measured cell, durably, before the next one starts."""
        self._write(outcome.model_dump_json())

    def _write(self, payload: str) -> None:
        self._handle.write(payload + "\n")
        # Flushed AND fsynced per cell: a cell is minutes of wall clock and real money, and the
        # whole point of the sidecar is to survive a kill that a buffered write would lose. The
        # cost is one fsync per multi-minute episode.
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        """Close the handle. Idempotent; the file stays on disk."""
        if not self._handle.closed:
            self._handle.close()

    def discard(self) -> None:
        """Close and remove the sidecar: the matrix it was protecting is written."""
        self.close()
        self._path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

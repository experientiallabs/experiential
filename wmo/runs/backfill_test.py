"""Tests for the artifact-to-event backfill mapping.

The headline test is byte-exact reproduction of the shared-truth fixtures. That
directory is the seam between this emitter and the platform's ingest: both sides
test against it, so drift fails here rather than in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from wmo.core.types import JsonObject
from wmo.runs.backfill import grid_arm_events, optimize_events
from wmo.runs.schema import RUN_SEQ_BAND, RunEventType, cell_band

FIXTURES = Path.home() / "Desktop/Projects/wmh-plan/d-runs-fixtures"
ARTIFACTS = FIXTURES / "artifacts"
ARM = "llmlingua2-endpoint"
GRID = "jt/grid-c2"
MODEL = "tau-jt-toy"


def _obj(value: JsonValue) -> JsonObject:
    """Narrow a payload member to an object; a wrong shape is a test-worthy failure."""
    assert isinstance(value, dict)
    return value


def _rows(value: JsonValue) -> list[JsonValue]:
    """Narrow a payload member to a list, for the same reason."""
    assert isinstance(value, list)
    return value


def _expected() -> list[JsonObject]:
    """The canonical stream both sides of the seam are pinned to."""
    events = FIXTURES / "expected-events.jsonl"
    if not events.exists():  # pragma: no cover - fixtures live outside the repo
        pytest.skip(f"shared-truth fixtures not present at {events}")
    return [json.loads(line) for line in events.read_text().splitlines() if line.strip()]


def _derived() -> list[JsonObject]:
    """Both runs' events, in the order the reference walks them."""
    if not ARTIFACTS.exists():  # pragma: no cover - fixtures live outside the repo
        pytest.skip(f"shared-truth artifacts not present at {ARTIFACTS}")
    grid = grid_arm_events(ARTIFACTS, arm=ARM, grid_relpath=GRID)
    pipeline = optimize_events(ARTIFACTS / "optimize-run.json", model=MODEL)
    return [event.jsonl_row() for event in (*grid, *pipeline)]


def test_backfill_reproduces_the_shared_fixtures_exactly() -> None:
    """Every event matches the canonical stream, in order, field for field.

    Compared as parsed JSON rather than as text so a failure names the differing
    event instead of printing two walls of bytes; key order is asserted
    separately in `schema_test`, where the envelope shape lives.
    """
    expected = _expected()
    derived = _derived()

    assert len(derived) == len(expected)
    for index, (actual, wanted) in enumerate(zip(derived, expected, strict=True)):
        assert actual == wanted, f"event {index} (seq {wanted['seq']}) diverged"


def test_spend_totals_do_not_depend_on_the_interpreter() -> None:
    """The spend rule is explicit accumulation, not `sum()`.

    CPython 3.12 gave `sum()` a compensated fast path for floats, so the same
    ledger yields different low bits on 3.11 and 3.12. That difference is enough
    to break byte-exactness against the shared fixtures, which is how this was
    found: the canonical stream was generated on 3.11 and this package runs 3.12.
    Pinning the rule here means neither side drifts when an interpreter moves.
    """
    from wmo.runs.backfill import _total

    # A case where compensated and naive summation visibly disagree.
    pathological = [0.1] * 10 + [1e16, -1e16]

    assert _total(pathological) == 0.0
    assert sum(pathological) != _total(pathological)


def test_timestamps_come_only_from_the_artifacts() -> None:
    """No derived clock: every ts appears verbatim somewhere in the inputs.

    A ts this code invented would drift on every run, and the platform's guards
    treat a changed clock as a different event — so a backfill replayed later
    would either re-apply as new or be suppressed as stale.
    """
    raw = (
        (ARTIFACTS / "cohort.json").read_text()
        + (ARTIFACTS / "ledger.jsonl").read_text()
        + (ARTIFACTS / "optimize-run.json").read_text()
        + (ARTIFACTS / ARM / "matrix.meta.json").read_text()
        if (ARTIFACTS / ARM / "matrix.meta.json").exists()
        else ""
    )
    if not raw:
        raw = (
            (ARTIFACTS / "cohort.json").read_text()
            + (ARTIFACTS / "ledger.jsonl").read_text()
            + (ARTIFACTS / "optimize-run.json").read_text()
        )

    for event in _derived():
        assert str(event["ts"]) in raw, f"seq {event['seq']} invented a clock"


def test_cells_precede_their_chunks_ledger_line() -> None:
    """A chunk's cells happened before the ledger line that closed it."""
    grid = grid_arm_events(ARTIFACTS, arm=ARM, grid_relpath=GRID)
    order = [event.type for event in grid]

    first_cells = order.index(RunEventType.CELL_BATCH)
    first_ledger = order.index("ledger.line")
    assert first_cells < first_ledger


def test_completed_chunk_cells_ride_their_own_band() -> None:
    """Cells are banded by chunk; in-flight partials stay on the run-level walk."""
    grid = grid_arm_events(ARTIFACTS, arm=ARM, grid_relpath=GRID)

    for event in grid:
        band = (event.seq - 1) // RUN_SEQ_BAND
        if event.type != RunEventType.CELL_BATCH:
            assert band == 0, f"{event.type} left the run-level band"
            continue
        chunks = {_obj(cell).get("chunk") for cell in _rows(event.payload["cells"])}
        assert len(chunks) == 1, "one cell.batch must not span chunks"
        chunk = chunks.pop()
        assert isinstance(chunk, int)
        # Band 0 means the chunk was still in flight (no ledger line yet).
        assert band in (0, cell_band(chunk))


def test_a_live_arm_has_no_terminal_status(tmp_path: Path) -> None:
    """Absent `matrix.meta.json`, the stream ends at the heartbeat.

    Merge meta is the only evidence an arm finished. Inventing `completed` from a
    quiet ledger would make the platform's status lie about a running arm.
    """
    live = tmp_path / GRID
    (live / ARM).mkdir(parents=True)
    (live / "cohort.json").write_text(
        json.dumps({"created": "2026-07-27T09:00:00+00:00", "model_dir": "/x/tau-bench"})
    )
    (live / "ledger.jsonl").write_text(
        json.dumps(
            {
                "arm": ARM,
                "event": "chunk",
                "chunk": 0,
                "ts": "2026-07-27T09:05:00+00:00",
                "cells": 5,
                "scored": 4,
                "candidate_usd": 0.5,
                "compressor_usd": 0.1,
                "wm_usd": 0.25,
            }
        )
        + "\n"
    )

    events = grid_arm_events(live, arm=ARM, grid_relpath=GRID)

    assert [event.type for event in events] == [
        RunEventType.RUN_META,
        "ledger.line",
        RunEventType.HEARTBEAT,
    ]
    heartbeat = events[-1]
    assert heartbeat.payload["progress"] == {"done": 5, "total": None, "scored": 4}
    # compressor_usd rides along as reported; it is a subset of candidate_usd, so a
    # reader totals candidate + wm and never adds it in.
    assert heartbeat.payload["spend"] == {
        "candidate_usd": 0.5,
        "compressor_usd": 0.1,
        "wm_usd": 0.25,
    }


def test_a_repaired_chunk_counts_once_but_costs_twice(tmp_path: Path) -> None:
    """Progress dedupes per chunk index; spend does not. Both are deliberate.

    A chunk that failed and was re-run appears twice in a real ledger. Summing
    both attempts' cells double-counts the arm's progress — a 440-cell arm read
    880 on the live stack, contradicting the panel's per-cell rollup. The later
    attempt supersedes the earlier one. Its dollars do not: every attempt really
    spent money, which is the ledger's own spend-to-date rule.
    """
    live = tmp_path / GRID
    (live / ARM).mkdir(parents=True)
    (live / "cohort.json").write_text(
        json.dumps({"created": "2026-07-27T09:00:00+00:00", "model_dir": "/x/tau-bench"})
    )

    def line(chunk: int, ts: str, *, cells: int, scored: int, usd: float) -> str:
        return json.dumps(
            {
                "arm": ARM,
                "event": "chunk",
                "chunk": chunk,
                "ts": ts,
                "cells": cells,
                "scored": scored,
                "candidate_usd": usd,
                "compressor_usd": 0.0,
                "wm_usd": 0.0,
            }
        )

    (live / "ledger.jsonl").write_text(
        "\n".join(
            [
                # Chunk 0 twice: a failed attempt, then the repair that supersedes it.
                line(0, "2026-07-27T09:01:00+00:00", cells=10, scored=2, usd=1.0),
                line(0, "2026-07-27T09:05:00+00:00", cells=10, scored=9, usd=1.5),
                line(1, "2026-07-27T09:06:00+00:00", cells=10, scored=10, usd=2.0),
            ]
        )
        + "\n"
    )

    (heartbeat,) = [
        event
        for event in grid_arm_events(live, arm=ARM, grid_relpath=GRID)
        if event.type == RunEventType.HEARTBEAT
    ]

    # Two chunks of ten, counted once each — not three lines of ten.
    assert heartbeat.payload["progress"] == {"done": 20, "total": None, "scored": 19}
    # And every attempt's dollars are still in the total.
    assert heartbeat.payload["spend"] == {
        "candidate_usd": 4.5,
        "compressor_usd": 0.0,
        "wm_usd": 0.0,
    }


def test_an_empty_manifest_emits_nothing(tmp_path: Path) -> None:
    """A manifest with no completed stages has no facts to report yet."""
    manifest = tmp_path / "optimize-run.json"
    manifest.write_text(json.dumps({"version": 1, "world_model": MODEL, "stages": []}))

    assert optimize_events(manifest, model=MODEL) == []

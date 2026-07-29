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
from wmo.runs.backfill import (
    ArtifactError,
    _total,
    conforms_to_ledger_schema,
    grid_arm_events,
    optimize_events,
)
from wmo.runs.schema import LEDGER_LINE, RUN_SEQ_BAND, RunEventType, cell_band, ledger_walk_seq

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


def _expected_lines(fixtures: Path) -> list[str]:
    """The canonical stream's raw lines, which are what the seam is pinned to."""
    events = fixtures / "expected-events.jsonl"
    return [line for line in events.read_text().splitlines() if line.strip()]


def _expected(fixtures: Path) -> list[JsonObject]:
    """The canonical stream, parsed."""
    return [json.loads(line) for line in _expected_lines(fixtures)]


def _derived(artifacts: Path) -> list[JsonObject]:
    """Both runs' events, in the order the reference walks them."""
    grid = grid_arm_events(artifacts, arm=ARM, grid_relpath=GRID)
    pipeline = optimize_events(artifacts / "optimize-run.json", model=MODEL)
    return [event.jsonl_row() for event in (*grid, *pipeline)]


def test_backfill_reproduces_the_shared_fixtures_exactly(
    d_runs_fixtures: Path, d_runs_artifacts: Path
) -> None:
    """Every event matches the canonical stream byte for byte, in order.

    Compared as RAW LINES, not as parsed dicts: `cell_payload` promises `chunk` last
    "so the key order matches the fixtures", and a dict comparison cannot see key
    order at all, so that promise was unguarded. The platform tolerates any order,
    but the fixture file is the shared artifact two repos diff against, and a
    reordering would make every line differ for no reason.
    """
    expected_lines = _expected_lines(d_runs_fixtures)
    derived_lines = [json.dumps(row) for row in _derived(d_runs_artifacts)]

    assert len(derived_lines) == len(expected_lines)
    for index, (actual, wanted) in enumerate(zip(derived_lines, expected_lines, strict=True)):
        # Parsed first, so an ordinary value change names the field instead of
        # printing two walls of bytes; the raw check below then catches key order.
        assert json.loads(actual) == json.loads(wanted), f"event {index} diverged"
        assert actual == wanted, f"event {index} matches by value but not key order"


def test_spend_totals_do_not_depend_on_the_interpreter() -> None:
    """The spend rule is explicit accumulation, not `sum()`.

    CPython 3.12 gave `sum()` a compensated fast path for floats, so the same
    ledger yields different low bits on 3.11 and 3.12. That difference is enough
    to break byte-exactness against the shared fixtures, which is how this was
    found: the canonical stream was generated on 3.11 and this package runs 3.12.
    Pinning the rule here means neither side drifts when an interpreter moves.
    """
    # A case where compensated and naive summation visibly disagree.
    pathological = [0.1] * 10 + [1e16, -1e16]

    assert _total(pathological) == 0.0
    assert sum(pathological) != _total(pathological)


def _ledger(arm: str, chunk: int, ts: str, **overrides: JsonValue) -> JsonObject:
    """One conforming ledger line, as the runner writes them."""
    line: JsonObject = {
        "event": "chunk",
        "arm": arm,
        "chunk": chunk,
        "cells": 5,
        "scored": 4,
        "candidate_usd": 0.5,
        "compressor_usd": 0.0,
        "wm_usd": 0.25,
        "wall_s": 1.0,
        "ts": ts,
        "cumulative_usd": 0.75,
        "tip_sha": "abc123",
        "max_steps": 20,
        "episodes": 2,
    }
    line.update(overrides)
    return line


def test_a_non_conforming_line_does_not_consume_a_walk_position(tmp_path: Path) -> None:
    """Both paths must count positions over the same lines, or every seq after
    the odd line differs and the platform holds two copies of the rest of the run.

    The live path counts over what the runner accepted, and the runner validates with
    `extra="forbid"` and skips what fails. A backfill counting every json-parseable
    line instead renumbers everything downstream, and the duplicates read as derived
    positions, so the collision tripwire stays quiet while the spend curve doubles.
    """
    live = tmp_path / GRID
    (live / ARM).mkdir(parents=True)
    (live / "cohort.json").write_text(
        json.dumps({"created": "2026-07-27T09:00:00+00:00", "model_dir": "/x/tau-bench"})
    )
    rows = [
        _ledger(ARM, 0, "2026-07-27T09:01:00+00:00"),
        # Valid JSON, but a field the runner's model forbids: it is skipped there,
        # so it must not take a position here either.
        _ledger(ARM, 1, "2026-07-27T09:02:00+00:00", surprise="from a newer writer"),
        _ledger(ARM, 2, "2026-07-27T09:03:00+00:00"),
    ]
    (live / "ledger.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    events = grid_arm_events(live, arm=ARM, grid_relpath=GRID)
    ledger_events = [event for event in events if event.type == LEDGER_LINE]

    # Two counted lines, at positions 1 and 2 — not 1 and 3.
    assert [event.seq for event in ledger_events] == [ledger_walk_seq(1), ledger_walk_seq(2)]
    assert [event.payload["chunk"] for event in ledger_events] == [0, 2]
    # And the skipped line is not silently counted into progress either.
    (heartbeat,) = [event for event in events if event.type == RunEventType.HEARTBEAT]
    assert _obj(heartbeat.payload["progress"])["done"] == 10


def test_a_torn_final_ledger_line_is_recovered_not_refused(tmp_path: Path) -> None:
    """A half-written last line is what a crash leaves, and backfill exists for that.

    Refusing it would make the one artifact the tool is meant to recover
    unrecoverable. A malformed line EARLIER is the opposite case: the file is
    corrupted rather than truncated, so every later line is in doubt and a partial
    recovery would under-report the run.
    """
    live = tmp_path / GRID
    (live / ARM).mkdir(parents=True)
    (live / "cohort.json").write_text(
        json.dumps({"created": "2026-07-27T09:00:00+00:00", "model_dir": "/x/tau-bench"})
    )
    good = json.dumps(_ledger(ARM, 0, "2026-07-27T09:05:00+00:00"))
    # Interrupted mid-append: the writer died partway through the JSON object.
    (live / "ledger.jsonl").write_text(f"{good}\n" + good[: len(good) // 2])

    events = grid_arm_events(live, arm=ARM, grid_relpath=GRID)

    assert [event.type for event in events] == ["run.meta", "ledger.line", "heartbeat"]
    assert _obj(events[-1].payload["progress"]) == {"done": 5, "total": None, "scored": 4}

    # The same damage anywhere but the end is a refusal.
    (live / "ledger.jsonl").write_text(f"{good[: len(good) // 2]}\n{good}\n")
    with pytest.raises(ArtifactError, match="corrupted rather than truncated"):
        grid_arm_events(live, arm=ARM, grid_relpath=GRID)


def test_pipeline_spend_uses_the_same_deterministic_accumulation(tmp_path: Path) -> None:
    """The manifest legs follow the explicit rule too, not `sum()`.

    Same hazard as the grid legs: CPython 3.12's compensated `sum()` disagrees with
    3.11's in the low bits, and a byte-exact seam cannot rest on which interpreter
    ran. The pipeline path had been left on the builtin.
    """
    manifest = tmp_path / "optimize-run.json"
    legs = [0.1] * 10 + [1e16, -1e16]
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "world_model": MODEL,
                "stages": [
                    {
                        "stage": f"s{index}",
                        "completed_at": f"2026-07-27T09:0{index}:00+00:00",
                        "spend_usd": leg,
                    }
                    for index, leg in enumerate(legs)
                ],
            }
        )
    )

    (heartbeat,) = [
        event
        for event in optimize_events(manifest, model=MODEL)
        if event.type == RunEventType.HEARTBEAT
    ]

    # Left-to-right accumulation, which is what the reference and the grid legs use.
    spend = _obj(heartbeat.payload["spend"])
    assert spend["candidate_usd"] == _total(legs)
    assert spend["candidate_usd"] != sum(legs)


def test_timestamps_come_only_from_the_artifacts(d_runs_artifacts: Path) -> None:
    """No derived clock: every ts appears verbatim somewhere in the inputs.

    A ts this code invented would drift on every run, and the platform's guards
    treat a changed clock as a different event, so a backfill replayed later
    would either re-apply as new or be suppressed as stale.
    """
    sources = ["cohort.json", "ledger.jsonl", "optimize-run.json", f"{ARM}/matrix.meta.json"]
    raw = "".join(
        (d_runs_artifacts / name).read_text()
        for name in sources
        if (d_runs_artifacts / name).exists()
    )

    for event in _derived(d_runs_artifacts):
        assert str(event["ts"]) in raw, f"seq {event['seq']} invented a clock"


def test_cells_precede_their_chunks_ledger_line(d_runs_artifacts: Path) -> None:
    """A chunk's cells happened before the ledger line that closed it."""
    grid = grid_arm_events(d_runs_artifacts, arm=ARM, grid_relpath=GRID)
    order = [event.type for event in grid]

    first_cells = order.index(RunEventType.CELL_BATCH)
    first_ledger = order.index("ledger.line")
    assert first_cells < first_ledger


def test_completed_chunk_cells_ride_their_own_band(d_runs_artifacts: Path) -> None:
    """Cells are banded by chunk; in-flight partials stay on the run-level walk."""
    grid = grid_arm_events(d_runs_artifacts, arm=ARM, grid_relpath=GRID)

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
            _ledger(ARM, 0, "2026-07-27T09:05:00+00:00", compressor_usd=0.1),
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
    both attempts' cells double-counts the arm's progress: a 440-cell arm read
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
            _ledger(ARM, chunk, ts, cells=cells, scored=scored, candidate_usd=usd, wm_usd=0.0)
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

    # Two chunks of ten, counted once each, not three lines of ten.
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

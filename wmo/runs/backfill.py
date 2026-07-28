"""Derive a run's canonical event stream from artifacts already on disk.

`wmo runs backfill` replays work that has already happened — a grid arm's ledger and
chunk files, an optimize manifest — into the same events the live hooks emit. The two
paths MUST converge on identical `(run, seq)` identities, because the platform's
idempotency key is exactly that pair: converging makes a backfill over a live run
free, and diverging double-counts every fact.

Convergence is what forces the determinism rules below. They are binding, and the
shared fixtures (`d-runs-fixtures/expected-events.jsonl`) are the executable
statement of them — `backfill_test.py` asserts byte-exact reproduction, so a change
here that drifts from the seam fails locally rather than in production.

- Seqs are BANDED (see `schema.SeqBands`). Band 0 carries the run-level walk
  (`run.meta`, `ledger.line`, the final `heartbeat`, `run.status`) plus in-flight
  partial-file cells; completed chunk k's cells ride band k + 1. Pipeline runs use
  band 0 alone.
- Never invent a clock. Every `ts` is an artifact's own string, passed through
  untouched: the ledger line's `ts`, the cohort's `created`, a stage's
  `completed_at`, the matrix meta's `merged_at`. The platform's projection guards
  compare these, so a re-derived timestamp is a different event.
- A chunk's cells are emitted immediately BEFORE its `ledger.line`, because they
  happened first. An in-flight partial's cells come after every ledger line.
- Spend follows the ledger-sum rule: sum `candidate_usd` and `wm_usd` per arm line.
  `cumulative_usd` is advisory under concurrency and never read. `compressor_usd` is
  a SUBSET of `candidate_usd` and is passed through, never added into a total.
- A grid arm is `completed` only when its `matrix.meta.json` exists (ts =
  `merged_at`); a live arm's stream ends at the heartbeat. A manifest whose `report`
  stage completed is a finished optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import JsonValue

from wmo.core.types import JsonObject
from wmo.runs.schema import (
    CELL_BATCH_CAP,
    LEDGER_LINE,
    RUN_LEVEL_BAND,
    RunEvent,
    RunEventType,
    RunKind,
    SeqBands,
    cell_band,
    grid_arm_external_id,
    pipeline_external_id,
)

logger = logging.getLogger(__name__)

# The emitter truncates long free text before it leaves the machine; the platform's
# column checks are only a backstop. Keep this in step with the product rule.
PREVIEW_CHARS = 4096

# Outcome fields that can carry unbounded model text.
TEXT_PREVIEW_FIELDS = ("critique", "replies", "task")

# Ledger events that count toward a run's completed-cell progress.
_COUNTED_LEDGER_EVENTS = ("chunk", "chunk-skipped")


class ArtifactError(ValueError):
    """Raised when an artifact field is missing or not the type the mapping needs.

    Loud on purpose. These fields drive seq bands, spend totals, and event clocks,
    so a silently coerced or skipped value would produce a plausible-looking event
    stream that disagrees with the run it claims to describe.
    """


def _as_int(value: JsonValue, *, field: str) -> int:
    """Narrow a parsed-JSON value to an int."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer, got {value!r}"
        raise ArtifactError(msg)
    return value


def _as_float(value: JsonValue, *, field: str) -> float:
    """Narrow a parsed-JSON value to a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{field} must be a number, got {value!r}"
        raise ArtifactError(msg)
    return float(value)


def _as_str(value: JsonValue, *, field: str) -> str:
    """Narrow a parsed-JSON value to a string."""
    if not isinstance(value, str):
        msg = f"{field} must be a string, got {value!r}"
        raise ArtifactError(msg)
    return value


def _as_rows(value: JsonValue, *, field: str) -> list[JsonObject]:
    """Narrow a parsed-JSON value to a list of objects."""
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        msg = f"{field} must be a list of objects"
        raise ArtifactError(msg)
    return [dict(row) for row in value if isinstance(row, dict)]


def _preview(value: JsonValue) -> JsonValue:
    """Truncate long free text, recursing into lists of it."""
    if isinstance(value, str) and len(value) > PREVIEW_CHARS:
        return value[:PREVIEW_CHARS]
    if isinstance(value, list):
        return [_preview(item) for item in value]
    return value


def cell_payload(outcome: JsonObject, *, chunk: int | None = None) -> JsonObject:
    """Project one `ScenarioOutcome` row into a `cell.batch` entry.

    Public because the live hooks call it too: a cell emitted while a chunk runs
    and the same cell replayed from that chunk's file have to be the SAME payload
    by construction. Two call sites building it independently would drift into two
    dialects, and the platform would read the difference as a rewrite.

    An unscored cell keeps a null reward and carries its error text: a dropped
    request must never read as incapability. Compression and timing land in
    `detail`, with empty keys dropped so a row is not padded with nulls.

    Args:
        outcome: One `ScenarioOutcome` row as JSON.
        chunk: The chunk the cell belongs to, appended last when known (an
            in-flight partial's rows carry it too; only a chunkless caller omits it).

    Returns:
        The cell entry, `chunk` last so the key order matches the fixtures.
    """
    detail: JsonObject = {
        "call_seconds": outcome.get("call_seconds"),
        "remeasured": outcome.get("remeasured", False),
        "tokens_in_raw": outcome.get("tokens_in_raw"),
        "tokens_in_compressed": outcome.get("tokens_in_compressed"),
        "compressor_id": outcome.get("compressor_id"),
        "compressor_version": outcome.get("compressor_version"),
        "aggressiveness": outcome.get("aggressiveness"),
        "compressor_latency_s": outcome.get("compressor_latency_s"),
        "compressor_cost_usd": outcome.get("compressor_cost_usd"),
    }
    for field in TEXT_PREVIEW_FIELDS:
        raw = outcome.get(field)
        if raw is not None:
            detail[f"{field}_preview"] = _preview(raw)
    cell: JsonObject = {
        "cell_key": f"{outcome['scenario_id']}|{outcome['model']}|{outcome['episode']}",
        "scenario_id": outcome["scenario_id"],
        "model": outcome["model"],
        "episode": outcome["episode"],
        "reward": outcome.get("reward"),
        "success": outcome.get("success"),
        "steps": outcome.get("steps"),
        "stop_reason": outcome.get("stop_reason"),
        "error": outcome.get("error"),
        "usage": outcome.get("usage"),
        "cost_usd": outcome.get("cost_usd"),
        "detail": {key: value for key, value in detail.items() if value is not None},
    }
    if chunk is not None:
        cell["chunk"] = chunk
    return cell


def _total(values: list[float]) -> float:
    """Add floats left to right, explicitly.

    NOT `sum()`, and not for style: CPython 3.12 gave `sum()` a compensated
    (Neumaier) fast path for floats, so the same source over the same ledger
    produces different low bits on 3.11 and 3.12. A byte-exact seam cannot rest
    on that — the canonical fixtures would stop matching the moment either side
    changed interpreter. An explicit accumulation is IEEE-deterministic on every
    version, which is worth more here than the extra accuracy: run spend is
    display-only and the ledger's own precision is far coarser than the
    difference.
    """
    total = 0.0
    for value in values:
        total += value
    return total


def _ledger_total(lines: list[JsonObject], field: str) -> float:
    """Sum one spend leg across ledger lines, deterministically."""
    return _total([_as_float(line[field], field=f"ledger.{field}") for line in lines])


def _stage_total(stages: list[JsonObject], field: str) -> float:
    """Sum one spend leg across manifest stages, deterministically."""
    return _total([_as_float(record.get(field, 0.0), field=f"stage.{field}") for record in stages])


def _latest_chunk_lines(arm_lines: list[JsonObject]) -> list[JsonObject]:
    """The last ledger line per chunk index, which is what progress counts.

    Real ledgers carry repair attempts: a chunk that failed and was re-run appears
    two or three times, and summing all of them double-counts its cells — a
    440-cell arm reported 880 done on the live stack, disagreeing with the panel's
    own cell rollup, which is keyed per cell and therefore truthful. The later
    attempt supersedes the earlier one, so only the last line per index counts.

    Spend deliberately does NOT dedupe (see the caller): every attempt's dollars
    were really spent, which is the ledger's own spend-to-date rule.

    Lines with no chunk index are skipped: they describe the arm, not a chunk's
    cells. `retry` lines are already excluded by event type.
    """
    latest: dict[int, JsonObject] = {}
    for line in arm_lines:
        if line.get("event") not in _COUNTED_LEDGER_EVENTS:
            continue
        raw_chunk = line.get("chunk")
        if raw_chunk is None:
            continue
        latest[_as_int(raw_chunk, field="ledger.chunk")] = line
    return list(latest.values())


def _jsonl(path: Path) -> list[JsonObject]:
    """Read a JSONL file, skipping blank lines."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BackfillRefused(RuntimeError):
    """Raised when a backfill would double-count a run that was live-emitted."""


def ensure_backfillable(existing_events: int, *, force: bool = False) -> None:
    """Refuse to backfill a run that already holds events, unless forced.

    Live emission and backfill place RUN-LEVEL events in different bands — a live
    process uses the band of the first chunk it owns, because two or three of them
    drive one arm and all wanting band 0 is the collision the partition exists to
    stop, while a backfill writes the canonical band-0 walk. So replaying a run that
    was already live-emitted does not converge: the ledger lines land a SECOND time
    under band-0 seqs nobody used, and the panel's spend curve counts every dollar
    twice. Cell events are safe either way, since a chunk's band is a function of
    the chunk alone.

    Args:
        existing_events: Events the platform already holds for this run.
        force: Proceed anyway, accepting the double-count.

    Raises:
        BackfillRefused: When the run already holds events and `force` is False.
    """
    if existing_events == 0 or force:
        if existing_events and force:
            logger.warning(
                "backfilling a run that already holds %d events: its run-level events will "
                "land a second time under different seqs and the spend curve will "
                "double-count every dollar",
                existing_events,
            )
        return
    msg = (
        f"run already holds {existing_events} events, so it was emitted live; backfilling it "
        "would re-add its run-level events under band-0 seqs and double-count the spend "
        "curve. Pass --force to accept that."
    )
    raise BackfillRefused(msg)


class _Walker:
    """Accumulates one run's events, taking seqs from the shared band allocator."""

    def __init__(self, external_id: str) -> None:
        """Initialize the walker for one run."""
        self._external_id = external_id
        self._bands = SeqBands()
        self._events: list[RunEvent] = []

    def emit(self, band: int, event_type: str, ts: str, payload: JsonObject) -> None:
        """Append one event, consuming a seq from its writer's band."""
        self._events.append(
            RunEvent(
                external_id=self._external_id,
                seq=self._bands.take(band),
                ts=ts,
                type=event_type,
                payload=payload,
            )
        )

    def emit_cells(self, band: int, rows: list[JsonObject], *, chunk: int | None, ts: str) -> None:
        """Emit a chunk's rows as `cell.batch` events, in file order."""
        for start in range(0, len(rows), CELL_BATCH_CAP):
            cells = [cell_payload(row, chunk=chunk) for row in rows[start : start + CELL_BATCH_CAP]]
            self.emit(band, RunEventType.CELL_BATCH, ts, {"cells": cells})

    @property
    def events(self) -> list[RunEvent]:
        """The events walked so far, in emission order."""
        return self._events


def grid_arm_events(artifacts: Path, *, arm: str, grid_relpath: str) -> list[RunEvent]:
    """Derive one grid arm's event stream from its cohort, ledger, and chunk files.

    Args:
        artifacts: Grid directory holding `cohort.json`, `ledger.jsonl`, and the
            per-arm subdirectory.
        arm: Arm name; the ledger is filtered to it.
        grid_relpath: Grid directory path relative to `.wmo`, for the run's name.

    Returns:
        The arm's events in emission order.
    """
    cohort: JsonObject = json.loads((artifacts / "cohort.json").read_text())
    arm_lines = [line for line in _jsonl(artifacts / "ledger.jsonl") if line.get("arm") == arm]
    walker = _Walker(grid_arm_external_id(grid_relpath, arm))
    created = _as_str(cohort["created"], field="cohort.created")
    model_name = Path(_as_str(cohort["model_dir"], field="cohort.model_dir")).name

    walker.emit(
        RUN_LEVEL_BAND,
        RunEventType.RUN_META,
        created,
        {
            "kind": RunKind.GRID_ARM.value,
            "benchmark": model_name,
            "arm": arm,
            "world_model": model_name,
            "config": cohort,
            "started_at": created,
        },
    )

    arm_dir = artifacts / arm
    for line in arm_lines:
        line_ts = _as_str(line["ts"], field="ledger.ts")
        raw_chunk = line.get("chunk")
        if line.get("event") == "chunk" and raw_chunk is not None:
            chunk = _as_int(raw_chunk, field="ledger.chunk")
            chunk_file = arm_dir / f"chunk-{chunk}.json"
            if chunk_file.exists():
                parsed: JsonObject = json.loads(chunk_file.read_text())
                rows = _as_rows(parsed["outcomes"], field=f"chunk-{chunk}.outcomes")
                walker.emit_cells(cell_band(chunk), rows, chunk=chunk, ts=line_ts)
        walker.emit(RUN_LEVEL_BAND, LEDGER_LINE, line_ts, line)

    last_ts = _as_str(arm_lines[-1]["ts"], field="ledger.ts") if arm_lines else created
    # In-flight partials ride band 0: their chunk has no ledger line yet, so the
    # run-level walker owns them until the chunk completes and takes its own band.
    partials = sorted(arm_dir.glob("chunk-*.json.partial.jsonl")) if arm_dir.exists() else []
    for partial in partials:
        in_flight = int(partial.name.split("-")[1].split(".")[0])
        rows = _jsonl(partial)[1:]  # line 0 is the PartialHeader
        if rows:
            walker.emit_cells(RUN_LEVEL_BAND, rows, chunk=in_flight, ts=last_ts)

    counted = _latest_chunk_lines(arm_lines)
    walker.emit(
        RUN_LEVEL_BAND,
        RunEventType.HEARTBEAT,
        last_ts,
        {
            "progress": {
                # Integer counts, so ordinary `sum` is exact here; only the
                # float legs below need the explicit rule. `counted` is already
                # deduped per chunk index, while the spend legs below sum EVERY
                # line — the two are asymmetric on purpose.
                "done": sum(_as_int(line["cells"], field="ledger.cells") for line in counted),
                # A live arm does not know its denominator; the platform's progress
                # column is nullable for exactly this.
                "total": None,
                "scored": sum(_as_int(line["scored"], field="ledger.scored") for line in counted),
            },
            "spend": {
                "candidate_usd": _ledger_total(arm_lines, "candidate_usd"),
                "compressor_usd": _ledger_total(arm_lines, "compressor_usd"),
                "wm_usd": _ledger_total(arm_lines, "wm_usd"),
            },
        },
    )

    # The ARM's own merge meta, with no fallback to the grid root: a sibling arm's
    # merge says nothing about this one, and treating the root's meta as this arm's
    # would report a still-running arm as completed.
    meta_file = arm_dir / "matrix.meta.json"
    if meta_file.exists():
        meta: JsonObject = json.loads(meta_file.read_text())
        merged_at = _as_str(meta["merged_at"], field="matrix.meta.merged_at")
        walker.emit(
            RUN_LEVEL_BAND,
            RunEventType.RUN_STATUS,
            merged_at,
            {"status": "completed", "finished_at": merged_at},
        )
    return walker.events


def optimize_events(manifest_path: Path, *, model: str) -> list[RunEvent]:
    """Derive one optimize pipeline's event stream from its manifest.

    The manifest persists COMPLETED stages only, so every stage event is
    `completed`: there is no started_at, no running, and no failed stage to
    recover from it. Live emission is what reports a stage in flight.

    Args:
        manifest_path: Path to `optimize-run.json`.
        model: World-model name, for the run's name.

    Returns:
        The pipeline's events in emission order.
    """
    manifest: JsonObject = json.loads(manifest_path.read_text())
    stages = _as_rows(manifest["stages"], field="manifest.stages")
    walker = _Walker(pipeline_external_id(model))
    if not stages:
        return walker.events
    first_ts = _as_str(stages[0]["completed_at"], field="stage.completed_at")

    walker.emit(
        RUN_LEVEL_BAND,
        RunEventType.RUN_META,
        first_ts,
        {
            "kind": RunKind.PIPELINE.value,
            "benchmark": manifest["world_model"],
            "world_model": manifest["world_model"],
            "config": {
                "version": manifest["version"],
                "lifetime_spend_usd": manifest.get("lifetime_spend_usd", 0.0),
            },
            "started_at": first_ts,
        },
    )
    for record in stages:
        completed_at = _as_str(record["completed_at"], field="stage.completed_at")
        walker.emit(
            RUN_LEVEL_BAND,
            RunEventType.STAGE_UPSERT,
            completed_at,
            {
                "stage": record["stage"],
                "status": "completed",
                "fingerprint": record.get("fingerprint"),
                "spend": {
                    "candidate_usd": record.get("spend_usd", 0.0),
                    "compressor_usd": record.get("compressor_spend_usd", 0.0),
                    "wm_usd": record.get("world_model_spend_usd", 0.0),
                },
                "completed_at": completed_at,
                "artifact": {
                    "artifact_path": record.get("artifact_path"),
                    "artifact_identity": record.get("artifact_identity"),
                },
            },
        )

    last_ts = _as_str(stages[-1]["completed_at"], field="stage.completed_at")
    walker.emit(
        RUN_LEVEL_BAND,
        RunEventType.HEARTBEAT,
        last_ts,
        {
            "progress": {
                "done": len(stages),
                "total": None,
                "stage": _as_str(stages[-1]["stage"], field="stage.stage"),
            },
            "spend": {
                "candidate_usd": sum(record.get("spend_usd", 0.0) for record in stages),
                "compressor_usd": sum(record.get("compressor_spend_usd", 0.0) for record in stages),
                "wm_usd": sum(record.get("world_model_spend_usd", 0.0) for record in stages),
            },
        },
    )
    report_ts = next(
        (
            _as_str(record["completed_at"], field="stage.completed_at")
            for record in stages
            if record.get("stage") == "report"
        ),
        None,
    )
    if report_ts is not None:
        walker.emit(
            RUN_LEVEL_BAND,
            RunEventType.RUN_STATUS,
            report_ts,
            {"status": "completed", "finished_at": report_ts},
        )
    return walker.events

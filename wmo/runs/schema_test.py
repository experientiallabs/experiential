"""Tests for the D-RUNS wire contract and the seq-band allocator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmo.runs.schema import (
    CELL_BATCH_CAP,
    MAX_CELLS_PER_EVENT,
    RUN_LEVEL_BAND,
    RUN_SEQ_BAND,
    TERMINAL_STATUSES,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    SeqBandOverrun,
    SeqBands,
    cell_band,
    grid_arm_external_id,
    is_terminal_status,
    pipeline_external_id,
)


def test_bands_are_disjoint_and_independent() -> None:
    """Each writer's band is its own range, allocated without touching the others."""
    bands = SeqBands()

    assert bands.take(RUN_LEVEL_BAND) == 1
    assert bands.take(RUN_LEVEL_BAND) == 2
    # Chunk 0's band starts a full width up, and asking for it consumed nothing
    # from band 0, which is what lets a backfill walk chunks in any order and
    # still derive the seqs the live hooks used.
    assert bands.take(cell_band(0)) == RUN_SEQ_BAND + 1
    assert bands.take(cell_band(1)) == 2 * RUN_SEQ_BAND + 1
    assert bands.take(RUN_LEVEL_BAND) == 3


def test_band_overrun_fails_loudly_instead_of_walking_into_the_next_band() -> None:
    """Exhausting a band raises: silently continuing would collide with a neighbour.

    The platform's idempotency key is `(run_id, seq)`, so a seq that strays into
    another writer's band is accepted as that writer's and makes ITS real event
    look like a duplicate. Nothing downstream can detect that.
    """
    bands = SeqBands(band_width=3)

    taken = [bands.take(RUN_LEVEL_BAND) for _ in range(3)]

    assert taken == [1, 2, 3]
    with pytest.raises(SeqBandOverrun, match="band 0 is exhausted"):
        bands.take(RUN_LEVEL_BAND)
    # The neighbour is untouched and still starts where it should.
    assert bands.take(1) == 4


def test_resume_from_top_only_moves_down_and_refuses_the_wrong_direction() -> None:
    """The descending twin of `resume_at`, with the misuse it has to refuse.

    A resumed descending walk that restarts at the ceiling re-issues the seqs its
    previous invocation used, and the platform discards each as a replay. That is
    quiet and, for a terminal `run.status`, permanent: the run stays `running` on the
    panel forever.
    """
    bands = SeqBands()
    bands.resume_from_top(1, 2 * RUN_SEQ_BAND - 5)

    assert bands.take_from_top(1) == 2 * RUN_SEQ_BAND - 5
    # Only downward: a stale higher mark cannot rewind a walk already further down.
    bands.resume_from_top(1, 2 * RUN_SEQ_BAND)
    assert bands.take_from_top(1) == 2 * RUN_SEQ_BAND - 6

    # And a seq belonging to the ascending walk is refused AT THE RESUME, where the
    # caller's mistake is visible, rather than at some later heartbeat.
    ascending = SeqBands()
    for _ in range(4):
        ascending.take(1)
    with pytest.raises(SeqBandOverrun, match="belongs to the other direction"):
        ascending.resume_from_top(1, RUN_SEQ_BAND + 2)


def test_a_maximum_cannot_locate_a_descending_frontier() -> None:
    """Why `last_seq` is the wrong input to `resume_from_top`, as an executable note.

    A descending walk's frontier is the LOWEST seq it issued; the platform reports the
    HIGHEST seq in the run. After three live-only events the highest is still the
    ceiling, so `last_seq - 1` names a seq the previous invocation already used, and
    the resumed walk re-issues it.
    """
    bands = SeqBands()
    issued = [bands.take_from_top(1) for _ in range(3)]
    platform_last_seq = max(issued)

    assert platform_last_seq == bands.band_end(1)
    # The frontier is the minimum, three below the maximum.
    assert min(issued) == platform_last_seq - 2
    # So resuming at last_seq - 1 lands on an already-issued seq.
    assert platform_last_seq - 1 in issued


def test_terminal_status_is_a_membership_test_not_an_inversion() -> None:
    """An unrecognized status must not read as terminal.

    A reader uses this to tell a self-closed tail (the run finished) from a dropped
    connection. `status != "running"` agrees with this today and diverges the moment
    the platform adds a still-writing status like paused or resuming: that would read
    as terminal, and the tail would exit early and silently drop the rest of the run.
    Erring the other way leaves a stream open too long, which is recoverable.
    """
    assert is_terminal_status(RunStatus.COMPLETED) is True
    assert is_terminal_status(RunStatus.FAILED) is True
    assert is_terminal_status(RunStatus.STOPPED) is True
    assert is_terminal_status(RunStatus.RUNNING) is False
    # The whole point: a status this build has never heard of is treated as live.
    assert is_terminal_status("paused") is False
    assert is_terminal_status("resuming") is False
    # Every known status is accounted for, so adding one to the enum without
    # classifying it fails here rather than in a tail that ends early.
    assert {status.value for status in RunStatus} == TERMINAL_STATUSES | {"running"}


def test_cell_batch_cap_stays_clear_of_the_platform_refusal() -> None:
    """The batch shape is a choice; the platform's cap is a hard 422."""
    assert CELL_BATCH_CAP < MAX_CELLS_PER_EVENT


def test_external_ids_follow_the_run_naming_convention() -> None:
    """Run names are paths, which is why the platform routes them as `:path`."""
    assert grid_arm_external_id("jt/grid-c2", "identity") == "jt/grid-c2/identity"
    assert grid_arm_external_id("/jt/grid-c2/", "identity") == "jt/grid-c2/identity"
    assert pipeline_external_id("tau-jt-toy") == "tau-jt-toy/optimize"


def test_event_wire_shape_matches_the_shared_fixtures(d_runs_fixtures: Path) -> None:
    """The envelope is exactly the five keys the platform ingest reads.

    Pinned against the shared-truth fixtures rather than a hand-written literal:
    this file and the platform's ingest tests are the two halves of one seam.
    """
    events = d_runs_fixtures / "expected-events.jsonl"
    first = json.loads(events.read_text().splitlines()[0])

    parsed = RunEvent.model_validate(first)

    # The fixture line names its run; a PUSHED event does not, because the ingest
    # route takes the run in its URL path.
    assert parsed.jsonl_row() == first
    assert list(parsed.wire()) == ["seq", "ts", "type", "payload"]
    assert "external_id" not in parsed.wire()
    assert sorted(first) == ["external_id", "payload", "seq", "ts", "type"]
    assert parsed.type == RunEventType.RUN_META
    assert parsed.payload["kind"] == RunKind.GRID_ARM
    # The clock is the artifact's own string, not a re-serialized datetime.
    assert isinstance(parsed.ts, str)
    assert parsed.ts == first["ts"]


def test_fixture_seqs_respect_the_band_scheme(d_runs_fixtures: Path) -> None:
    """Every fixture event sits inside the band its content dictates."""
    events = d_runs_fixtures / "expected-events.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]

    for row in rows:
        band = (row["seq"] - 1) // RUN_SEQ_BAND
        if row["type"] == RunEventType.CELL_BATCH:
            # Completed-chunk cells ride their chunk's band; in-flight partial
            # cells have no chunk of their own and ride the run-level walk.
            chunks = {cell.get("chunk") for cell in row["payload"]["cells"]}
            assert len(chunks) == 1
            chunk = chunks.pop()
            assert band in (RUN_LEVEL_BAND, cell_band(chunk))
        else:
            assert band == RUN_LEVEL_BAND, row["type"]

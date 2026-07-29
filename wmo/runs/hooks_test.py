"""Tests for the live emission hooks: bands, buffering, control, and never breaking a run."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ConfigDict

from wmo.core.types import JsonObject, JsonValue
from wmo.optimize.outcomes import ScenarioOutcome
from wmo.optimize.pipeline import Stage, StageRecord
from wmo.platform.client import PlatformUnreachable
from wmo.runs.backfill import cell_payload
from wmo.runs.client import PUSH_ATTEMPTS, PushRejected, RunsSink
from wmo.runs.hooks import (
    ACK_ACKED,
    ACK_DONE,
    ACK_REJECTED,
    CONTROL_FORCE_FROM_STAGE,
    CONTROL_RETRY_UNSCORED,
    CONTROL_STOP,
    GridEmitter,
    GridSnapshot,
    PipelineEmitter,
)
from wmo.runs.schema import (
    CELL_BATCH_CAP,
    RUN_LEVEL_BAND,
    RUN_META_SEQ,
    RUN_SEQ_BAND,
    RunEventType,
    RunStatus,
    cell_band,
    ledger_walk_seq,
)

ORG = "org-1"
GRID = "jt/grid-c2"
ARM = "identity"
RUN = f"{GRID}/{ARM}"
CREATED = "2026-07-27T09:00:00+00:00"
COHORT: JsonObject = {"tip_sha": "abc123", "episodes": 2, "created": CREATED}


class Pushed(BaseModel):
    """One event as the transport received it, typed for assertions."""

    model_config = ConfigDict(frozen=True)

    seq: int
    ts: str
    type: str
    payload: JsonObject


def obj(value: JsonValue) -> JsonObject:
    """Read a nested JSON object out of a payload."""
    assert isinstance(value, dict)
    return value


def rows(value: JsonValue) -> list[JsonValue]:
    """Read a nested JSON list out of a payload."""
    assert isinstance(value, list)
    return value


class FakeTransport:
    """Records what would have been pushed, and can be told to fail or short-accept."""

    def __init__(
        self,
        *,
        failures: list[Exception] | None = None,
        accepted: int | None = None,
        control: list[JsonObject] | None = None,
        held_last_seq: int = 0,
    ) -> None:
        """Initialize the double.

        `held_last_seq` seeds a run the platform already holds events for (seqs 1..N), which is
        what the resume probe reports and what a re-numbered event then collides with.
        """
        self.pushes: list[list[Pushed]] = []
        self.acks: list[tuple[str, str, str | None]] = []
        self.probes = 0
        self.failures = list(failures or [])
        self._accepted = accepted
        self._control = control or []
        # The platform keys events on (run, seq) and DISCARDS a seq it already holds. Modelling that
        # here is what lets a test fail on a collision at all: a double that accepts everything
        # cannot distinguish the band design working from the band design being broken.
        self.held: set[int] = set(range(1, held_last_seq + 1)) if held_last_seq else set()

    def push_run_events(
        self,
        org_id: str,
        external_id: str,
        *,
        emitter_id: str,
        events: Sequence[JsonObject],
    ) -> JsonObject:
        """Record a push, raising a queued failure first, and hand back any control commands."""
        if not events:
            # Counted before any injected failure, so a probe that FAILS is still an attempt: the
            # question a test asks is how many times the emitter asked, not how many times it was
            # answered.
            self.probes += 1
        if self.failures:
            raise self.failures.pop(0)
        control, self._control = self._control, []
        if not events:
            # The resume probe: a zero-event push, answered with the run's mark and its pending
            # commands (the real ingest 422s an unknown run, which the sink turns into 0).
            mark = max(self.held) if self.held else 0
            return {"accepted": 0, "last_seq": mark, "control": control}
        types = [str(event["type"]) for event in events]
        if not self.held and RunEventType.RUN_META not in types:
            # The real ingest refuses a new run's first batch unless it declares the run, and that
            # refusal is PERMANENT: every later batch fails the same way until run.meta arrives.
            msg = "a new run's first batch must carry run.meta"
            raise PushRejected(msg, status_code=422)
        self.pushes.append([Pushed.model_validate(event) for event in events])
        seqs = [int(str(event["seq"])) for event in events]
        fresh = [seq for seq in seqs if seq not in self.held]
        self.held.update(seqs)
        return {
            "accepted": len(fresh) if self._accepted is None else self._accepted,
            "last_seq": max(self.held),
            "control": control,
        }

    def ack_run_control(
        self,
        org_id: str,
        external_id: str,
        control_id: str,
        *,
        status: str,
        note: str | None = None,
    ) -> JsonObject:
        """Record an ack."""
        self.acks.append((control_id, status, note))
        return {"ok": True}

    # -- helpers for assertions

    @property
    def events(self) -> list[Pushed]:
        """Every pushed event, in push order."""
        return [event for push in self.pushes for event in push]

    def of_type(self, event_type: str) -> list[Pushed]:
        """Every pushed event of one type."""
        return [event for event in self.events if event.type == event_type]

    def close(self) -> None:
        """No-op: this double owns no connection pool.

        Required by `RunsTransport`, which declares `close()` so a sink's release is
        a checked contract rather than a runtime `getattr` probe.
        """


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the sleep out of the transport's retry.

    These tests exercise what happens when a push fails, and the client's real backoff would spend
    seconds of suite time proving nothing about the hooks.
    """
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)


def _sink(transport: FakeTransport) -> RunsSink:
    return RunsSink(transport, org_id=ORG, emitter_id="test")


def _grid(
    transport: FakeTransport, *, band: int = 1, flush_cells: int = CELL_BATCH_CAP
) -> GridEmitter:
    return GridEmitter.create(
        grid_relpath=GRID,
        arm=ARM,
        band=band,
        factory=lambda: _sink(transport),
        snapshot=lambda: GridSnapshot(done=4, scored=3, total=10, candidate_usd=1.5, wm_usd=3.0),
        flush_cells=flush_cells,
    )


def _declared(transport: FakeTransport, **kwargs: int) -> GridEmitter:
    """A grid emitter whose run is already declared, which is every real run's first act.

    The platform refuses a new run's first batch unless it carries `run.meta`, so a test about
    ledger lines or cells has to start where a real arm starts.
    """
    emitter = _grid(transport, **kwargs)
    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    return emitter


def _ts(minute: int) -> str:
    """A distinct, ordered artifact timestamp."""
    return f"2026-07-27T10:{minute:02d}:00+00:00"


def _outcome(scenario: str = "s1", model: str = "haiku", episode: int = 0) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario,
        task="book a flight",
        model=model,
        episode=episode,
        reward=0.75,
        success=True,
        steps=3,
        cost_usd=0.01,
    )


def test_arm_start_declares_the_run_at_its_derived_position() -> None:
    """`run.meta` is first, flushed immediately, and numbered where the artifacts put it.

    Band 0 seq 1 no matter which chunks this process owns, because a backfill of the same arm puts
    it there too: the two converge on one declaration instead of each writing its own.
    """
    transport = FakeTransport()
    emitter = _grid(transport, band=cell_band(3))

    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)

    meta = transport.of_type("run.meta")
    assert len(meta) == 1
    assert meta[0].payload["kind"] == "grid_arm"
    assert meta[0].payload["arm"] == ARM
    assert meta[0].payload["config"] == COHORT
    assert meta[0].seq == RUN_META_SEQ


def test_ledger_lines_are_numbered_from_their_position_in_the_file() -> None:
    """The Nth line of the arm's ledger holds `1 + N`, the same seq a backfill derives for it."""
    transport = FakeTransport()
    emitter = _declared(transport)

    for position in (1, 2, 3):
        emitter.on_ledger_line(
            {"event": "chunk", "arm": ARM, "chunk": position - 1},
            ts=_ts(position),
            position=position,
        )

    assert [event.seq for event in transport.of_type("ledger.line")] == [
        ledger_walk_seq(1),
        ledger_walk_seq(2),
        ledger_walk_seq(3),
    ]
    assert [event.seq for event in transport.of_type("ledger.line")] == [2, 3, 4]


def test_heartbeats_and_status_descend_from_the_bands_ceiling() -> None:
    """Facts with no artifact position cannot be re-derived, so they get their own end of the band.

    Descending from the ceiling keeps them clear of the ascending derived walk, which is what lets
    one band carry both without either knowing the other's count.
    """
    transport = FakeTransport()
    emitter = _declared(transport, band=cell_band(0))

    emitter.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)
    emitter.on_status(RunStatus.COMPLETED)

    ceiling = 2 * RUN_SEQ_BAND
    beats = transport.of_type("heartbeat")
    status = transport.of_type("run.status")
    assert beats[0].seq == ceiling
    assert status[0].seq == ceiling - 1
    # The derived walk in band 0 is untouched by them.
    assert transport.of_type("ledger.line")[0].seq == 2


def test_cells_go_to_their_chunks_band_with_the_backfill_payload() -> None:
    """A live cell is numbered in its chunk's band and carries the same payload a replay would."""
    transport = FakeTransport()
    emitter = _declared(transport)
    outcome = _outcome()

    emitter.on_outcome(outcome, chunk=7)
    emitter.send_cells()

    batches = transport.of_type("cell.batch")
    assert len(batches) == 1
    seq = batches[0].seq
    assert 8 * RUN_SEQ_BAND < seq <= 9 * RUN_SEQ_BAND
    cells = rows(batches[0].payload["cells"])
    # Identical by construction, not by a parallel implementation: the hook calls the backfill's
    # own mapper, so a cell emitted live and the same cell replayed cannot drift.
    assert cells[0] == cell_payload(outcome.model_dump(mode="json"), chunk=7)


def test_cells_are_buffered_until_the_batch_is_full() -> None:
    """One request per cell would be one request per episode, so cells wait for a batch."""
    transport = FakeTransport()
    emitter = _declared(transport, flush_cells=3)

    for index in range(2):
        emitter.on_outcome(_outcome(episode=index), chunk=0)
    assert transport.of_type("cell.batch") == []

    emitter.on_outcome(_outcome(episode=2), chunk=0)
    batches = transport.of_type("cell.batch")
    assert len(batches) == 1
    assert len(rows(batches[0].payload["cells"])) == 3


def test_ledger_line_travels_verbatim_with_a_whole_run_heartbeat() -> None:
    """The ledger line is log-only and unedited; the heartbeat beside it is whole-run."""
    transport = FakeTransport()
    emitter = _declared(transport)
    line: JsonObject = {"event": "chunk", "arm": ARM, "chunk": 0, "cells": 4, "candidate_usd": 1.5}

    emitter.on_ledger_line(line, ts="2026-07-27T10:00:00+00:00", position=1)

    ledger = transport.of_type("ledger.line")
    assert len(ledger) == 1
    assert ledger[0].payload == line
    assert ledger[0].ts == "2026-07-27T10:00:00+00:00"
    beat = transport.of_type("heartbeat")[0].payload
    assert obj(beat["progress"]) == {"done": 4, "total": 10, "scored": 3}
    assert obj(beat["spend"]) == {"candidate_usd": 1.5, "compressor_usd": 0.0, "wm_usd": 3.0}


def test_buffered_cells_are_sent_before_a_terminal_status() -> None:
    """A stopped run's last measured cells are not lost to the stop."""
    transport = FakeTransport()
    emitter = _declared(transport, flush_cells=100)
    emitter.on_outcome(_outcome(), chunk=0)

    emitter.on_status(RunStatus.STOPPED, error="cap reached")

    types = [event.type for event in transport.events]
    assert types.index("cell.batch") < types.index("run.status")
    status = transport.of_type("run.status")[0].payload
    assert status["status"] == "stopped"
    assert status["error"] == "cap reached"


def test_stop_command_is_honored_and_acked_then_completed() -> None:
    """A pulled stop sets the flag the runner checks, and is acked twice: taken, then done."""
    transport = FakeTransport(control=[{"id": "c1", "command": CONTROL_STOP, "args": {}}])
    emitter = _grid(transport)

    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    assert emitter.stop_requested is True
    assert transport.acks[0] == (
        "c1",
        ACK_ACKED,
        "stopping at the next chunk boundary; finished chunks stay on disk",
    )

    emitter.on_status(RunStatus.STOPPED)
    assert transport.acks[-1][:2] == ("c1", ACK_DONE)


@pytest.mark.parametrize("command", [CONTROL_RETRY_UNSCORED, CONTROL_FORCE_FROM_STAGE, "nonsense"])
def test_commands_the_runner_does_not_own_are_rejected_with_a_reason(command: str) -> None:
    """A refusal is a legal answer, and it has to explain what owns the behavior instead."""
    transport = FakeTransport(control=[{"id": "c9", "command": command, "args": {}}])
    emitter = _grid(transport)

    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)

    control_id, status, note = transport.acks[0]
    assert (control_id, status) == ("c9", ACK_REJECTED)
    assert note
    assert emitter.stop_requested is False


def test_emission_survives_a_platform_that_only_fails(caplog: pytest.LogCaptureFixture) -> None:
    """Every hook returns normally when every push fails, and says so exactly once."""
    transport = FakeTransport(failures=[PlatformUnreachable("down")] * 200)
    emitter = _grid(transport)

    with caplog.at_level(logging.WARNING, logger="wmo.runs.hooks"):
        emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
        emitter.on_outcome(_outcome(), chunk=0)
        emitter.send_cells()
        emitter.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(0), position=1)
        emitter.on_status(RunStatus.COMPLETED)

    assert transport.pushes == []
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, "a broken platform must not flood a run's log"


def test_a_full_queue_drops_the_oldest_events_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """An outage costs telemetry, never memory: a failed push is retried, then the oldest go."""
    transport = FakeTransport(failures=[PlatformUnreachable("down")] * 200)
    emitter = GridEmitter(_sink(transport), external_id=RUN, band=1, arm=ARM, queue_limit=5)

    with caplog.at_level(logging.WARNING, logger="wmo.runs.hooks"):
        for index in range(12):
            emitter.on_ledger_line(
                {"event": "retry", "arm": ARM, "chunk": index}, ts=_ts(index), position=index + 1
            )

    assert any("dropped" in record.getMessage() for record in caplog.records)


def test_accepted_shortfall_trips_the_collision_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A freshly ALLOCATED seq the platform did not take means two writers share one band."""
    transport = FakeTransport(accepted=0)
    emitter = _declared(transport)

    with caplog.at_level(logging.INFO, logger="wmo.runs.hooks"):
        # A cell is allocated, not derived, so a shortfall here is a real collision.
        emitter.on_outcome(_outcome(), chunk=0)
        emitter.send_cells()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("share one band" in record.getMessage() for record in warnings)


def test_a_derived_seq_already_held_reads_as_a_resume_not_a_collision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Re-stating an artifact-derived fact is the design working, so it must not cry collision.

    This is the difference between a resumed run looking healthy and a resumed run looking broken:
    `run.meta` and every ledger line are deliberately re-derived at the same seq, so the platform
    already holding them is expected.
    """
    transport = FakeTransport(accepted=0)
    emitter = _grid(transport)

    with caplog.at_level(logging.INFO, logger="wmo.runs.hooks"):
        emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("converging on a resume" in record.getMessage() for record in caplog.records)


def test_a_resumed_run_continues_past_what_the_platform_holds() -> None:
    """The live-E2E bug: a re-invocation that renumbers from the floor is wholly discarded."""
    transport = FakeTransport(held_last_seq=7)
    emitter = PipelineEmitter.create(world_model="tau-bench", factory=lambda: _sink(transport))

    emitter.start(world_model="tau-bench", config={})

    assert transport.probes == 1
    # `run.meta` keeps its derived position; the first ALLOCATED event continues past the mark.
    assert transport.of_type("run.meta")[0].seq == RUN_META_SEQ
    assert transport.of_type("run.status")[0].seq == 8


def test_a_grid_arms_high_mark_leaves_the_ledger_walk_alone() -> None:
    """A resume mark from a CHUNK band must not renumber the run-level walk.

    An arm that has pushed cells has a `last_seq` up in the chunk bands; rebasing band 0 to that
    would move every ledger line off the position a backfill derives for it.
    """
    transport = FakeTransport(held_last_seq=100_004)
    emitter = _grid(transport)

    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    emitter.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)

    assert transport.of_type("run.meta")[0].seq == RUN_META_SEQ
    assert transport.of_type("ledger.line")[0].seq == ledger_walk_seq(1)


def test_a_command_waiting_at_resume_is_honored() -> None:
    """A stop issued before this process started is waiting on the probe, and is not dropped."""
    transport = FakeTransport(control=[{"id": "c1", "command": CONTROL_STOP, "args": {}}])
    emitter = _grid(transport)

    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)

    assert emitter.stop_requested is True
    assert transport.acks[0][:2] == ("c1", ACK_ACKED)


def test_emission_is_off_without_a_credential() -> None:
    """No login, no telemetry, no complaint: every hook is a no-op the run cannot feel."""
    emitter = GridEmitter.create(grid_relpath=GRID, arm=ARM, band=1)

    assert emitter.enabled is False
    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    emitter.on_outcome(_outcome(), chunk=0)
    emitter.on_ledger_line({"event": "chunk"}, ts="2026-07-27T10:00:00+00:00", position=1)
    emitter.on_status(RunStatus.COMPLETED)
    assert emitter.stop_requested is False


def test_no_emit_never_consults_credentials() -> None:
    """`--no-emit` turns emission off before anything is resolved."""
    calls: list[int] = []

    def factory() -> RunsSink | None:
        calls.append(1)
        raise AssertionError("--no-emit must not build a sink")

    emitter = GridEmitter.create(
        grid_relpath=GRID, arm=ARM, band=1, factory=factory, requested=False
    )

    assert emitter.enabled is False
    assert calls == []


def test_pipeline_reports_stages_in_the_run_level_band() -> None:
    """The single-process pipeline uses band 0 alone, and its stage rows mirror the manifest."""
    transport = FakeTransport()
    emitter = PipelineEmitter.create(world_model="tau-bench", factory=lambda: _sink(transport))

    emitter.start(world_model="tau-bench", config={"cells": 40})
    emitter.stage_running(Stage.SWEEP)
    emitter.stage_completed(
        StageRecord(
            stage=Stage.SWEEP,
            fingerprint={"pool": "abc"},
            artifact_path="matrix.json",
            artifact_identity="deadbeef",
            completed_at="2026-07-27T11:00:00+00:00",
            spend_usd=2.0,
            compressor_spend_usd=0.5,
            world_model_spend_usd=7.0,
        ),
        lifetime_spend_usd=9.0,
    )
    emitter.stage_skipped(Stage.FIT, reason="policy.json is current")
    emitter.finished(RunStatus.COMPLETED)

    assert all(event.seq <= RUN_SEQ_BAND for event in transport.events), (
        "a single-process run stays in band 0"
    )
    assert RUN_LEVEL_BAND == 0
    stages = transport.of_type("stage.upsert")
    assert [stage.payload["status"] for stage in stages] == ["running", "completed", "skipped"]
    done = stages[1].payload
    assert obj(done["spend"]) == {"candidate_usd": 2.0, "compressor_usd": 0.5, "wm_usd": 7.0}
    assert obj(done["artifact"]) == {
        "artifact_path": "matrix.json",
        "artifact_identity": "deadbeef",
    }
    assert done["completed_at"] == "2026-07-27T11:00:00+00:00"
    assert stages[1].ts == "2026-07-27T11:00:00+00:00"
    assert obj(stages[2].payload["artifact"]) == {"reason": "policy.json is current"}
    assert transport.of_type("run.status")[-1].payload["status"] == "completed"


def test_pipeline_heartbeat_carries_lifetime_spend() -> None:
    """The panel's spend for a staged run is the manifest's lifetime total, both sides."""
    transport = FakeTransport()
    emitter = PipelineEmitter.create(world_model="tau-bench", factory=lambda: _sink(transport))
    emitter.start(world_model="tau-bench", config={})

    emitter.heartbeat(stage=Stage.TUNE, lifetime_spend_usd=12.5)

    beat = transport.of_type("heartbeat")[0].payload
    assert obj(beat["progress"])["stage"] == "tune"
    assert obj(beat["spend"])["candidate_usd"] == 12.5


def test_the_emitter_names_its_arm_so_a_caller_can_scope_ledger_lines() -> None:
    """One process walks several arms in turn, and each arm is a separate run.

    The runner checks this before reporting a ledger line: a sibling arm's line would otherwise
    land in the wrong run AND at a seq that collides with this arm's own line at that position.
    """
    emitter = GridEmitter.create(grid_relpath="jt/grid-c2", arm="llmlingua2-endpoint", band=1)

    assert emitter.arm == "llmlingua2-endpoint"
    assert emitter.external_id == "jt/grid-c2/llmlingua2-endpoint"


def test_a_permanent_refusal_drops_its_batch_instead_of_retrying_forever(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refused payload cannot be fixed by sending it again, so it must not stay in the queue.

    Requeueing it would head-of-line block this run's telemetry for the rest of a twelve-hour grid
    on one bad event, which is worse than losing that event: everything AFTER it would be lost too.
    """
    transport = FakeTransport()
    emitter = _declared(transport)
    # Armed after the run is declared: this is a mid-run refusal, not a refused declaration.
    transport.failures.append(PushRejected("detail is 300000 bytes", status_code=422))

    with caplog.at_level(logging.WARNING, logger="wmo.runs.hooks"):
        emitter.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)
        # The next flush must carry the FOLLOWING events, not the refused ones again.
        emitter.on_ledger_line({"event": "merge", "arm": ARM}, ts=_ts(2), position=2)

    refused = [r for r in caplog.records if "DROPPED" in r.getMessage()]
    assert len(refused) == 1
    assert "422" in refused[0].getMessage()
    assert "300000 bytes" in refused[0].getMessage(), "the field the platform named must survive"
    # The second batch went through, so one bad payload did not block the run's telemetry.
    assert [event.payload["event"] for event in transport.of_type("ledger.line")] == ["merge"]


def test_a_transient_failure_keeps_its_events_for_the_next_flush() -> None:
    """When the platform is unreachable the batch is kept and retried, oldest first.

    `PUSH_ATTEMPTS` failures, not one: the sink absorbs a blip by retrying internally, so what
    reaches these hooks is only the case where it has already given up.
    """
    transport = FakeTransport()
    emitter = _declared(transport)
    declared = len(transport.pushes)
    transport.failures.extend([PlatformUnreachable("down")] * PUSH_ATTEMPTS)

    emitter.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)
    assert len(transport.pushes) == declared, "the failed push landed nothing"

    emitter.on_ledger_line({"event": "merge", "arm": ARM}, ts=_ts(2), position=2)

    # Both lines land, oldest first, on the flush that succeeds.
    assert [event.payload["event"] for event in transport.of_type("ledger.line")] == [
        "chunk",
        "merge",
    ]


def test_a_resumed_arm_does_not_re_use_its_descending_seqs() -> None:
    """The documented resume flow: stop an arm from the panel, re-run it, and it continues.

    Heartbeats and the terminal status descend from the run-level band's ceiling, and nothing used
    to rebase THAT walk on resume, so a resumed arm re-issued the same high seqs and the platform
    discarded every one of them. The visible cost was the worst possible: a finished run stuck
    reading `running` because its terminal status was dropped as a replay.
    """
    transport = FakeTransport()
    first = _grid(transport, band=cell_band(0))
    first.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    first.on_status(RunStatus.STOPPED)

    resumed = _grid(transport, band=cell_band(0))
    resumed.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    resumed.on_status(RunStatus.COMPLETED)

    statuses = transport.of_type("run.status")
    assert len(statuses) == 2
    assert statuses[0].seq != statuses[1].seq, "the resumed status collided with the first one"
    assert statuses[1].payload["status"] == RunStatus.COMPLETED


def test_a_resumed_run_does_not_cry_collision(caplog: pytest.LogCaptureFixture) -> None:
    """A resume converging on itself must not log the concurrency alarm.

    A false alarm on every ordinary resume is how the one message that means real loss gets ignored.
    """
    transport = FakeTransport()
    first = _grid(transport, band=cell_band(0))
    first.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    first.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)

    resumed = _grid(transport, band=cell_band(0))
    with caplog.at_level(logging.INFO, logger="wmo.runs.hooks"):
        resumed.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
        resumed.on_ledger_line({"event": "chunk", "arm": ARM}, ts=_ts(1), position=1)

    assert not [r for r in caplog.records if "share one band" in r.getMessage()]


def test_run_meta_is_never_evicted_so_a_run_can_always_be_declared() -> None:
    """Losing `run.meta` to the queue bound would kill a run's telemetry permanently.

    It is the OLDEST event, so drop-oldest evicts it first; the platform then refuses every later
    batch ("a new run's first batch must carry run.meta"), and that refusal is permanent, so the run
    would report nothing for the rest of a twelve-hour grid.
    """
    transport = FakeTransport(failures=[PlatformUnreachable("down")] * (PUSH_ATTEMPTS * 3))
    emitter = GridEmitter(_sink(transport), external_id=RUN, band=1, arm=ARM, queue_limit=3)

    # Declare the run while the platform is unreachable, then overflow the queue many times over.
    emitter.on_arm_start(cohort=COHORT, world_model="tau-bench", created=CREATED)
    for position in range(1, 12):
        emitter.on_ledger_line({"event": "retry", "arm": ARM}, ts=_ts(position), position=position)

    # The platform comes back: the run must still be declarable, and it is only declarable if
    # `run.meta` survived the eviction.
    emitter.on_ledger_line({"event": "merge", "arm": ARM}, ts=_ts(20), position=20)

    assert transport.of_type("run.meta"), "run.meta was evicted, so the run can never be declared"
    assert transport.of_type("ledger.line"), "later events landed once the run was declared"


def test_a_fresh_pipeline_records_that_it_started() -> None:
    """The declaration and the running transition are two facts and need two seqs."""
    transport = FakeTransport()
    emitter = PipelineEmitter.create(world_model="tau-bench", factory=lambda: _sink(transport))

    emitter.start(world_model="tau-bench", config={})

    meta = transport.of_type("run.meta")[0]
    running = transport.of_type("run.status")[0]
    assert meta.seq == RUN_META_SEQ
    assert running.seq != meta.seq
    assert running.payload["status"] == RunStatus.RUNNING


def test_an_unreachable_probe_is_not_read_as_a_fresh_run() -> None:
    """ "Cannot reach the platform" and "the platform has never heard of this run" are different.

    Treating the first as the second numbers a resumed run from the floor, and every event of it is
    then discarded as a replay. The probe is retried instead.
    """
    transport = FakeTransport(held_last_seq=7)
    # `PUSH_ATTEMPTS` failures, so the sink's own retry gives up and the probe really fails.
    transport.failures.extend([PlatformUnreachable("down")] * PUSH_ATTEMPTS)
    emitter = PipelineEmitter.create(world_model="tau-bench", factory=lambda: _sink(transport))

    emitter.start(world_model="tau-bench", config={})
    emitter.heartbeat(stage=Stage.SWEEP, lifetime_spend_usd=1.0)

    # The failed probe did not stand as "fresh run": it was retried, the mark was learned, and
    # numbering continued past it. Events numbered DURING the outage may still be discarded (they
    # were numbered before the mark was known); what must not happen is numbering from the floor
    # forever after.
    assert transport.probes >= 2
    assert transport.of_type("heartbeat")[0].seq > 7


def test_an_unknown_status_is_reported_rather_than_raised() -> None:
    """Telemetry may not end a paid run, not even over a status this build has not heard of."""
    transport = FakeTransport()
    emitter = _declared(transport)

    emitter.on_status("canceled")  # type: ignore[arg-type]

    assert transport.of_type("run.status")[0].payload["status"] == "canceled"

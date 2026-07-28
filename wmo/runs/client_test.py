"""Tests for the batched run-telemetry emitter."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from wmo.core.types import JsonObject
from wmo.platform.client import PlatformError, PlatformUnreachable
from wmo.runs.client import (
    MAX_DOCUMENT_BYTES,
    PushRejected,
    PushUnavailable,
    RunsEmitter,
    RunsSink,
    default_emitter_id,
    open_emitter,
    runs_sink,
)
from wmo.runs.schema import (
    MAX_EVENTS_PER_BATCH,
    RUN_LEVEL_BAND,
    RUN_META_SEQ,
    RUN_SEQ_BAND,
    cell_band,
    ledger_walk_seq,
)

ORG = "org-1"
RUN = "jt/grid-c2/identity"
TS = "2026-07-27T09:00:00+00:00"


class FakeClient:
    """Records pushes and acks; can be told to fail a fixed number of times."""

    def __init__(
        self,
        *,
        failures: list[Exception] | None = None,
        accepted: int | None = None,
        last_seq: int | None = None,
    ) -> None:
        """Initialize the double."""
        self.pushes: list[list[JsonObject]] = []
        self.acks: list[tuple[str, str, str | None]] = []
        self._failures = failures or []
        self._accepted = accepted
        self._last_seq = last_seq
        self.control: list[JsonObject] = []

    def push_run_events(
        self,
        org_id: str,
        external_id: str,
        *,
        emitter_id: str,
        events: Sequence[JsonObject],
    ) -> JsonObject:
        """Record a push, raising any queued failure first."""
        if self._failures:
            raise self._failures.pop(0)
        self.pushes.append([dict(event) for event in events])
        seqs = [int(str(event["seq"])) for event in events]
        held = self._last_seq or 0
        return {
            "accepted": self._accepted if self._accepted is not None else len(events),
            "last_seq": max([held, *seqs]) if seqs else held,
            "control": self.control,
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
        return {"control": {"id": control_id, "command": "stop", "args": {}}}


def _sink(client: FakeClient) -> RunsSink:
    return RunsSink(client, org_id=ORG, emitter_id="box-6:1:abcd")


def _emitter(client: FakeClient) -> RunsEmitter:
    return RunsEmitter(_sink(client), external_id=RUN)


def test_one_emitter_spans_bands_and_pushes_them_together() -> None:
    """A single emitter holds several writers' bands and flushes them at once.

    One emitter per run rather than per band is what makes the ranges provably
    disjoint, and it means a chunk's cells and the run-level walk ride the same
    request instead of one request each.
    """
    client = FakeClient()
    emitter = _emitter(client)

    emitter.emit(RUN_LEVEL_BAND, "run.meta", TS, {"kind": "grid_arm"})
    emitter.emit(cell_band(0), "cell.batch", TS, {"cells": []})
    emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {"progress": {"done": 1}})
    ack = emitter.flush()

    assert len(client.pushes) == 1
    assert [event["seq"] for event in client.pushes[0]] == [1, 100_001, 2]
    # The pushed event does not name its run: the route takes that in its path.
    assert all("external_id" not in event for event in client.pushes[0])
    assert ack is not None
    assert ack.last_seq == 100_001


def test_the_sink_splits_at_the_platform_limit_and_sums_accepted() -> None:
    """Over-cap pushes split into requests, and the caller sees one total.

    Summing matters for the collision tripwire: a caller comparing `accepted`
    against what it sent must not see only the last request's count.
    """
    client = FakeClient()
    sink = _sink(client)
    events = [
        RunsEmitter(sink, external_id=RUN).emit(RUN_LEVEL_BAND, "log.line", TS, {"n": index})
        for index in range(MAX_EVENTS_PER_BATCH + 5)
    ]

    ack = sink.push(RUN, events)

    assert [len(batch) for batch in client.pushes] == [MAX_EVENTS_PER_BATCH, 5]
    assert ack.accepted == MAX_EVENTS_PER_BATCH + 5


def test_flush_batches_at_the_platform_limit() -> None:
    """More events than one push allows are split, not refused."""
    client = FakeClient()
    emitter = _emitter(client)
    for _ in range(MAX_EVENTS_PER_BATCH + 5):
        emitter.emit(RUN_LEVEL_BAND, "log.line", TS, {"line": "x"})

    emitter.flush()

    assert [len(batch) for batch in client.pushes] == [MAX_EVENTS_PER_BATCH, 5]


def test_transport_failures_retry_and_then_surface_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreachable is the retryable half: the run's telemetry is worth another try."""
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)
    recovering = FakeClient(
        failures=[PlatformUnreachable("dns"), PlatformError("502", status_code=502)]
    )
    emitter = _emitter(recovering)
    emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})

    ack = emitter.flush()

    assert ack is not None
    assert len(recovering.pushes) == 1

    doomed = FakeClient(failures=[PlatformUnreachable("dns")] * 3)
    second = _emitter(doomed)
    second.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})
    with pytest.raises(PushUnavailable):
        second.flush()


def test_a_4xx_is_permanent_and_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a refusal cannot help, and a loop would wedge a backfill."""
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)
    client = FakeClient(failures=[PlatformError("no Content-Length", status_code=411)])
    emitter = _emitter(client)
    emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})

    with pytest.raises(PushRejected) as refused:
        emitter.flush()

    assert refused.value.status_code == 411
    # One attempt, not three: nothing was retried.
    assert client.pushes == []


def test_oversized_payload_fields_are_caught_before_leaving_the_machine() -> None:
    """A field the platform would 422 is a loud local error naming the field."""
    client = FakeClient()
    emitter = _emitter(client)
    huge = "x" * (MAX_DOCUMENT_BYTES + 1)

    with pytest.raises(PushRejected, match="artifact") as refused:
        emitter.emit(RUN_LEVEL_BAND, "stage.upsert", TS, {"stage": "fit", "artifact": {"b": huge}})

    assert refused.value.status_code == 422
    # Nothing was queued, so a later flush cannot smuggle it out.
    assert emitter.flush() is None
    assert client.pushes == []


def test_a_shortfall_in_accepted_is_logged_as_a_band_collision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every event in a push is freshly numbered, so a shortfall means collision.

    The platform cannot distinguish another writer's seq from our replay, so this
    count is the only signal that two processes are numbering into one band.
    """
    client = FakeClient(accepted=1)
    emitter = _emitter(client)
    emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})
    emitter.emit(RUN_LEVEL_BAND, "log.line", TS, {})

    with caplog.at_level(logging.ERROR):
        emitter.flush()

    assert "seq band" in caplog.text
    assert "silently discarded" in caplog.text


def test_control_commands_ride_back_and_are_acked_by_id() -> None:
    """Pull-based delivery: the push answers with work, the emitter answers back."""
    client = FakeClient()
    client.control = [{"id": "control-1", "command": "stop", "args": {"reason": "budget"}}]
    emitter = _emitter(client)
    emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})
    emitter.flush()

    (command,) = emitter.pending_control
    emitter.ack(command, status="rejected", note="arm already finished")

    assert command.command == "stop"
    assert command.args == {"reason": "budget"}
    assert client.acks == [("control-1", "rejected", "arm already finished")]


def test_close_never_raises_over_undelivered_telemetry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Telemetry must not be the reason a finished run reports failure."""
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)
    client = FakeClient(failures=[PlatformUnreachable("down")] * 3)

    with caplog.at_level(logging.WARNING), _emitter(client) as emitter:
        emitter.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})

    assert "not delivered" in caplog.text


def test_resume_continues_past_what_the_platform_already_holds() -> None:
    """A re-invocation must not restart at seq 1 and lose every event as a replay.

    This is the bug live E2E caught: a pipeline re-run on a backfilled model
    numbered from 1, the platform discarded all of it as already-held, and the run
    looked healthy while its new telemetry vanished.
    """
    client = FakeClient(last_seq=7)
    emitter = _emitter(client)

    resumed = emitter.resume()
    emitter.emit(RUN_LEVEL_BAND, "run.status", TS, {"status": "running"})
    emitter.flush()

    assert resumed == 7
    # The probe writes nothing, so the only pushed event is the one we emitted.
    assert [event["seq"] for event in client.pushes[-1]] == [8]


def test_resume_on_a_fresh_run_starts_at_one() -> None:
    """An unknown run answers 0, not an error, so numbering begins normally.

    The probe is a zero-event push, and the ingest route refuses one for a run it
    does not know (a new run's first batch must carry `run.meta`). That refusal is
    the "fresh run" answer.
    """
    client = FakeClient(failures=[PlatformError("must carry a run.meta", status_code=422)])
    emitter = _emitter(client)

    resumed = emitter.resume()
    emitter.emit(RUN_LEVEL_BAND, "run.meta", TS, {"kind": "pipeline"})
    emitter.flush()

    assert resumed == 0
    assert [event["seq"] for event in client.pushes[-1]] == [1]


def test_resume_leaves_a_grid_arms_band_zero_alone() -> None:
    """A high resume mark comes from a chunk band, so band 0 must not skip to it.

    A grid arm's band-0 seqs are derived from its ledger file, not allocated, so
    rebasing band 0 to `last_seq + 1` would jump past the positions the ledger
    dictates and break convergence with the backfill.
    """
    client = FakeClient(last_seq=100_004)
    emitter = _emitter(client)

    resumed = emitter.resume()
    emitter.emit(RUN_LEVEL_BAND, "log.line", TS, {})

    emitter.flush()

    assert resumed == 100_004
    # Band 0 still starts at 1, where the ledger walk expects it.
    assert [event["seq"] for event in client.pushes[-1]] == [1]


def test_derived_seqs_converge_quietly_while_fresh_shortfalls_alarm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A resume's re-derived artifact positions are expected no-ops, not collisions.

    Reporting them as errors would train an operator to ignore the one message that
    means real data loss, so the two cases have to read differently.
    """
    # Two derived events re-emitted on a resume; the platform already holds both.
    converging = FakeClient(accepted=0)
    emitter = _emitter(converging)
    emitter.emit_at(RUN_META_SEQ, "run.meta", TS, {"kind": "grid_arm"})
    emitter.emit_at(ledger_walk_seq(1), "ledger.line", TS, {"chunk": 0})
    with caplog.at_level(logging.DEBUG):
        emitter.flush()
    assert "converging on a resume" in caplog.text
    assert "silently discarded" not in caplog.text

    # A freshly ALLOCATED seq coming back unaccepted is the real alarm.
    caplog.clear()
    colliding = FakeClient(accepted=0)
    second = _emitter(colliding)
    second.emit(RUN_LEVEL_BAND, "heartbeat", TS, {})
    with caplog.at_level(logging.DEBUG):
        second.flush()
    assert "silently discarded" in caplog.text


def test_live_only_events_descend_from_their_bands_top() -> None:
    """Heartbeats have no artifact position, so they take the band's ceiling down.

    Descending keeps them clear of the ascending artifact-derived walk in the same
    band, so one process can write both without tracking the other's count.
    """
    client = FakeClient()
    emitter = _emitter(client)
    band = cell_band(0)

    first = emitter.emit_from_top(band, "heartbeat", TS, {})
    second = emitter.emit_from_top(band, "heartbeat", TS, {})
    ascending = emitter.emit(band, "cell.batch", TS, {"cells": []})

    assert first.seq == 2 * RUN_SEQ_BAND
    assert second.seq == 2 * RUN_SEQ_BAND - 1
    assert ascending.seq == RUN_SEQ_BAND + 1


def test_emitter_id_names_host_pid_and_a_nonce() -> None:
    """Several processes feed one run, so the feeder is diagnostic, not identity."""
    first = default_emitter_id()
    second = default_emitter_id()

    assert first.count(":") == 2
    assert first != second
    assert first.rsplit(":", 1)[0] == second.rsplit(":", 1)[0]


def test_open_emitter_returns_none_without_a_credential(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unauthenticated machine is ordinary: one INFO line, never a raise."""
    from wmo.platform.credentials import PlatformCredentials

    monkeypatch.setattr("wmo.runs.client.load_credentials", lambda: PlatformCredentials())

    with caplog.at_level(logging.INFO):
        assert open_emitter(RUN) is None
    assert runs_sink() is None

    assert "telemetry is off" in caplog.text


def test_open_emitter_returns_none_without_a_default_org(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A credential with no organization cannot address a run either."""
    from wmo.platform.credentials import PlatformCredentials

    monkeypatch.setattr(
        "wmo.runs.client.load_credentials",
        lambda: PlatformCredentials(api_url="https://api.example", token="xpl_x"),
    )

    with caplog.at_level(logging.INFO):
        assert open_emitter(RUN) is None
    assert runs_sink() is None

    assert "organization" in caplog.text

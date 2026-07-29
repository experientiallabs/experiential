"""Tests for the batched run-telemetry emitter."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from wmo.core.types import JsonObject
from wmo.platform.client import PlatformError, PlatformUnreachable
from wmo.runs.client import (
    MAX_DOCUMENT_BYTES,
    PushAck,
    PushRejected,
    PushUnavailable,
    RunsSink,
    default_emitter_id,
    runs_sink,
)
from wmo.runs.schema import MAX_EVENTS_PER_BATCH, RUN_LEVEL_BAND, RunEvent, SeqBands, cell_band

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
        self.closed = False

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

    def close(self) -> None:
        """Record that the sink released this transport."""
        self.closed = True


def _sink(client: FakeClient) -> RunsSink:
    return RunsSink(client, org_id=ORG, emitter_id="box-6:1:abcd")


def _event(seq: int, event_type: str = "heartbeat", payload: JsonObject | None = None) -> RunEvent:
    """One event with its seq already assigned, as a caller hands them to the sink."""
    return RunEvent(external_id=RUN, seq=seq, ts=TS, type=event_type, payload=payload or {})


def client_push(client: FakeClient, events: list[RunEvent]) -> PushAck:
    """Push through a sink wrapping the given double."""
    return _sink(client).push(RUN, events)


def test_one_push_carries_several_bands_worth_of_events() -> None:
    """The sink takes whatever a caller numbered, from any band, in one request."""
    client = FakeClient()
    bands = SeqBands()
    events = [
        _event(bands.take(RUN_LEVEL_BAND), "run.meta", {"kind": "grid_arm"}),
        _event(bands.take(cell_band(0)), "cell.batch", {"cells": []}),
        _event(bands.take(RUN_LEVEL_BAND), "heartbeat", {"progress": {"done": 1}}),
    ]

    ack = client_push(client, events)

    assert len(client.pushes) == 1
    assert [event["seq"] for event in client.pushes[0]] == [1, 100_001, 2]
    # The pushed event does not name its run: the route takes that in its path.
    assert all("external_id" not in event for event in client.pushes[0])
    assert ack.last_seq == 100_001


def test_the_sink_splits_at_the_platform_limit_and_sums_accepted() -> None:
    """Over-cap pushes split into requests, and the caller sees one total.

    Summing matters for a caller's collision tripwire: comparing `accepted` against
    what it sent must not see only the last request's count.
    """
    client = FakeClient()
    events = [_event(seq, "log.line", {"n": seq}) for seq in range(1, MAX_EVENTS_PER_BATCH + 6)]

    ack = client_push(client, events)

    assert [len(batch) for batch in client.pushes] == [MAX_EVENTS_PER_BATCH, 5]
    assert ack.accepted == MAX_EVENTS_PER_BATCH + 5


def test_transport_failures_retry_and_then_surface_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreachable is the retryable half: the run's telemetry is worth another try."""
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)
    recovering = FakeClient(
        failures=[PlatformUnreachable("dns"), PlatformError("502", status_code=502)]
    )

    ack = client_push(recovering, [_event(1)])

    assert ack.accepted == 1
    assert len(recovering.pushes) == 1

    doomed = FakeClient(failures=[PlatformUnreachable("dns")] * 3)
    with pytest.raises(PushUnavailable):
        client_push(doomed, [_event(1)])


def test_a_4xx_is_permanent_and_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a refusal cannot help, and a loop would wedge a backfill."""
    monkeypatch.setattr("wmo.runs.client.PUSH_BACKOFF_SECONDS", 0.0)
    client = FakeClient(failures=[PlatformError("no Content-Length", status_code=411)])

    with pytest.raises(PushRejected) as refused:
        client_push(client, [_event(1)])

    assert refused.value.status_code == 411
    # One attempt, not three: nothing was retried.
    assert client.pushes == []


def test_oversized_payload_fields_are_caught_before_leaving_the_machine() -> None:
    """A field the platform would 422 is a loud local error naming the field."""
    client = FakeClient()
    huge = "x" * (MAX_DOCUMENT_BYTES + 1)
    oversized = _event(1, "stage.upsert", {"stage": "fit", "artifact": {"b": huge}})

    with pytest.raises(PushRejected, match="artifact") as refused:
        client_push(client, [oversized])

    assert refused.value.status_code == 422
    # Nothing reached the wire.
    assert client.pushes == []


def test_probe_reports_what_the_platform_already_holds() -> None:
    """A zero-event push is how a caller learns where to resume numbering.

    Without it, a re-invocation restarts at its band floor and the platform
    discards every event as a replay, so the run looks healthy while its new
    telemetry vanishes.
    """
    client = FakeClient(last_seq=7)

    ack = _sink(client).probe(RUN)

    assert ack.last_seq == 7
    # A probe writes nothing.
    assert client.pushes == [[]]


def test_probe_of_an_unknown_run_answers_zero_rather_than_raising() -> None:
    """The ingest route refuses a zero-event push for a run it does not know.

    A new run's first batch must carry `run.meta`, so that refusal IS the "fresh
    run" answer and comes back as 0 instead of an exception a caller must special
    case.
    """
    client = FakeClient(failures=[PlatformError("must carry a run.meta", status_code=422)])

    ack = _sink(client).probe(RUN)

    assert ack.last_seq == 0
    assert ack.accepted == 0


def test_control_commands_ride_back_and_are_acked_by_id() -> None:
    """Pull-based delivery: the push answers with work, the caller answers back."""
    client = FakeClient()
    client.control = [{"id": "control-1", "command": "stop", "args": {"reason": "budget"}}]
    sink = _sink(client)

    ack = sink.push(RUN, [_event(1)])
    (command,) = ack.control
    sink.ack(RUN, command.id, status="rejected", note="arm already finished")

    assert command.command == "stop"
    assert command.args == {"reason": "budget"}
    assert client.acks == [("control-1", "rejected", "arm already finished")]


def test_closing_the_sink_releases_its_transport() -> None:
    """A sink per run leaks a connection pool per run unless someone closes it.

    `runs_sink` builds a client per sink, so a multi-arm grid opens one per arm and a
    long-lived process accumulates an idle pool for every finished run.
    """
    client = FakeClient()
    sink = _sink(client)

    sink.push(RUN, [_event(1)])
    assert client.closed is False
    sink.close()

    assert client.closed is True

    # The context manager is the shape a caller should reach for.
    managed = FakeClient()
    with _sink(managed) as scoped:
        scoped.push(RUN, [_event(1)])
    assert managed.closed is True


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
        assert runs_sink() is None

    assert "organization" in caplog.text

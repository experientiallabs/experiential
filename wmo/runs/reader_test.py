"""Tests for the runs read surface (httpx mock transport, no network).

Two properties get most of the attention here, because they are the two a panel-parity reader gets
wrong quietly: the cursor coordinate (`pos`, never the emitter's `seq`) and what a CLOSED stream
means. The server closes a drained terminal run's tail on its own, so "the connection ended" is
ambiguous, and a reader that guesses either way is either a tail that never exits or a tail that
stops early on a live run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
import pytest

from wmo.core.types import JsonObject, JsonValue
from wmo.platform.client import PlatformClient, PlatformError, PlatformUnreachable
from wmo.runs.reader import RunsReader, _resolve_org

ORG = "org-1"
RUN = "jt/grid-c2/identity"
STARTED = "2026-07-27T09:00:00+00:00"
BEAT = "2026-07-27T10:00:00+00:00"

_RUN_ROW: JsonObject = {
    "id": "run-1",
    "org_id": ORG,
    "external_id": RUN,
    "kind": "grid_arm",
    "status": "running",
    "benchmark": "tau-bench",
    "arm": "identity",
    "progress": {"done": 220, "total": 440, "scored": 190},
    "candidate_usd": 6.5,
    "compressor_usd": 0.25,
    "wm_usd": 18.0,
    "started_at": STARTED,
    "heartbeat_at": BEAT,
    "last_seq": 100002,
    "created_at": STARTED,
    "updated_at": BEAT,
}


def _reader(handler: Callable[[httpx.Request], httpx.Response]) -> RunsReader:
    return RunsReader(
        PlatformClient("https://api.test", "xpl_secret", transport=httpx.MockTransport(handler)),
        ORG,
    )


def _detail(**extra: JsonValue) -> JsonObject:
    """One detail response, with any part of it overridden or added."""
    payload: JsonObject = {"run": _RUN_ROW, "event_count": 64}
    return payload | dict(extra)


def _frames(*events: JsonObject) -> bytes:
    """One SSE body: `id:` then `data:` then a blank line, as the server writes it."""
    chunks = [f"id: {event['pos']}\ndata: {json.dumps(event)}\n\n" for event in events]
    return "".join(chunks).encode()


def _event(pos: int, *, seq: int = 1, event_type: str = "heartbeat") -> JsonObject:
    return {"pos": pos, "seq": seq, "type": event_type, "payload": {}, "ts": BEAT}


@pytest.fixture(autouse=True)
def _no_reconnect_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the wait out of the tail's reconnect, which is otherwise seconds of suite time."""
    monkeypatch.setattr("wmo.runs.reader.RECONNECT_DELAY_S", 0.0)


# -- list ----------------------------------------------------------------------------------------


def test_list_runs_filters_and_parses_the_page() -> None:
    """Filters go to the server, and the page comes back typed."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        assert request.headers["Authorization"] == "Bearer xpl_secret"
        return httpx.Response(
            200, json={"runs": [_RUN_ROW], "next_cursor": {"ts": STARTED, "id": "run-1"}}
        )

    with _reader(handler) as reader:
        page = reader.list_runs(status="running", kind="grid_arm", limit=5)

    assert seen[0].path == f"/api/orgs/{ORG}/runs"
    assert dict(seen[0].params) == {"limit": "5", "status": "running", "kind": "grid_arm"}
    assert len(page.runs) == 1
    assert page.runs[0].external_id == RUN
    # The cursor is two fields the caller echoes back, not an opaque string.
    assert page.next_cursor == {"ts": STARTED, "id": "run-1"}


def test_the_next_page_echoes_the_cursors_two_fields() -> None:
    """Paging is keyset, by (created_at, id)."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"runs": [], "next_cursor": None})

    with _reader(handler) as reader:
        reader.list_runs(cursor_ts=STARTED, cursor_id="run-1")

    assert dict(seen[0].params) == {"limit": "50", "cursor_ts": STARTED, "cursor_id": "run-1"}


def test_spend_excludes_the_compressor_subset() -> None:
    """A run's spend is candidate + world model; the compressor is already inside the candidate."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"runs": [_RUN_ROW], "next_cursor": None})

    with _reader(handler) as reader:
        run = reader.list_runs().runs[0]

    assert run.spend_usd == pytest.approx(24.5)


# -- detail --------------------------------------------------------------------------------------


def test_get_run_parses_stages_cells_and_pending_control() -> None:
    """The detail carries everything the panel's run page renders."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/orgs/{ORG}/runs/{RUN}"
        return httpx.Response(
            200,
            json=_detail(
                org={"id": ORG, "name": "Acme", "slug": "acme"},
                stages=[
                    {
                        "run_id": "run-1",
                        "stage": "sweep",
                        "status": "completed",
                        "candidate_usd": 6.5,
                        "wm_usd": 18.0,
                        "completed_at": BEAT,
                        "updated_at": BEAT,
                    }
                ],
                cell_stats=[
                    {
                        "model": "haiku",
                        "cell_count": 40,
                        "scored_count": 38,
                        "error_count": 2,
                        "unpriced_count": 3,
                        "cost_usd_total": 1.25,
                        "reward_mean": 0.62,
                    }
                ],
                pending_control=[
                    {
                        "id": "c1",
                        "run_id": "run-1",
                        "command": "stop",
                        "args": {},
                        "status": "pending",
                        "requested_by": "user-1",
                        "created_at": BEAT,
                    }
                ],
            ),
        )

    with _reader(handler) as reader:
        detail = reader.get_run(RUN)

    assert detail.event_count == 64
    assert detail.stages[0].stage == "sweep"
    # A LIST of per-model rollups, not one object, and `unpriced_count` travels with the cost so a
    # partially priced total can never be read as complete.
    assert len(detail.cell_stats) == 1
    assert detail.cell_stats[0].unpriced_count == 3
    assert detail.pending_control[0].command == "stop"


def test_a_run_the_platform_does_not_have_is_a_platform_error() -> None:
    """`show` on a typo is an error with a status, not a silent empty run."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Run not found: nope"})

    with _reader(handler) as reader, pytest.raises(PlatformError) as info:
        reader.get_run("nope")

    assert info.value.status_code == 404


def test_event_count_reads_absence_as_zero() -> None:
    """The backfill guard's probe: an unknown run is "new run, go ahead", not a failure."""

    def missing(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Run not found"})

    def present(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_detail())

    with _reader(missing) as reader:
        assert reader.event_count(RUN) == 0
    with _reader(present) as reader:
        assert reader.event_count(RUN) == 64


def test_event_count_does_not_swallow_a_real_failure() -> None:
    """Only 404 means "no such run"; a 500 must not read as an empty run and license a backfill."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with _reader(handler) as reader, pytest.raises(PlatformError):
        reader.event_count(RUN)


# -- cells and events ----------------------------------------------------------------------------


def test_cell_filters_are_tri_state_and_compose() -> None:
    """Unscored and errored are different questions: every errored cell is unscored, not the
    reverse, so the two filters have to compose rather than alias."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"cells": [], "next_cursor_key": None})

    with _reader(handler) as reader:
        reader.list_cells(RUN, model="haiku", scored=False, error=True, limit=10)

    assert seen[0].path == f"/api/orgs/{ORG}/runs/{RUN}/cells"
    assert dict(seen[0].params) == {
        "limit": "10",
        "model": "haiku",
        "scored": "false",
        "error": "true",
    }


def test_cells_page_carries_its_keyset_cursor() -> None:
    """Cells page by `cell_key`, ascending."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "cells": [
                    {
                        "cell_key": "s1|haiku|0",
                        "scenario_id": "s1",
                        "model": "haiku",
                        "episode": 0,
                        "chunk": 0,
                        "reward": None,
                        "success": None,
                        "error": "ThrottlingException",
                        "updated_at": BEAT,
                    }
                ],
                "next_cursor_key": "s1|haiku|0",
            },
        )

    with _reader(handler) as reader:
        page = reader.list_cells(RUN)

    assert page.next_cursor_key == "s1|haiku|0"
    # An unscored cell keeps its tri-state: no reward, no verdict, and the reason it has neither.
    assert page.cells[0].reward is None
    assert page.cells[0].success is None
    assert page.cells[0].error == "ThrottlingException"


def test_events_page_by_pos_and_can_open_at_the_end() -> None:
    """`tail=true` is the first paint; `after_pos` is the cursor, and `last_pos` resumes it."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"events": [_event(41)], "last_pos": 41})

    with _reader(handler) as reader:
        first = reader.list_events(RUN, tail=True, limit=20)
        second = reader.list_events(RUN, after_pos=first.last_pos, event_type="ledger.line")

    assert dict(seen[0].params) == {"after_pos": "0", "limit": "20", "tail": "true"}
    assert dict(seen[1].params) == {
        "after_pos": "41",
        "limit": "500",
        "tail": "false",
        "type": "ledger.line",
    }
    assert second.events[0].pos == 41
    # `seq` rides along for lining a frame up against the emitter's own log, and is never a cursor.
    assert second.events[0].seq == 1


# -- control -------------------------------------------------------------------------------------


def test_request_control_returns_the_queued_row() -> None:
    """Queueing is pull-based: the row is pending until the runner acks it."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/orgs/{ORG}/runs/{RUN}/control"
        return httpx.Response(
            200,
            json={
                "control": {
                    "id": "c7",
                    "run_id": "run-1",
                    "command": "stop",
                    "args": {},
                    "status": "pending",
                    "requested_by": "user-1",
                    "created_at": BEAT,
                }
            },
        )

    with _reader(handler) as reader:
        control = reader.request_control(RUN, "stop")

    assert (control.id, control.command, control.status) == ("c7", "stop", "pending")


def test_a_control_reply_without_a_row_is_reported() -> None:
    """An accepted command the platform did not record is a contract break, not a success."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with _reader(handler) as reader, pytest.raises(PlatformError, match="without a control row"):
        reader.request_control(RUN, "stop")


# -- the tail ------------------------------------------------------------------------------------


def test_tail_yields_frames_then_ends_when_the_run_is_terminal() -> None:
    """A drained finished run's stream closes, and the tail exits rather than reconnecting."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, content=_frames(_event(7), _event(8)))
        return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))

    with _reader(handler) as reader:
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [7, 8]
    # One stream, then a status check to tell "finished" from "dropped", then the drain pass.
    assert paths.count(f"/api/orgs/{ORG}/runs/{RUN}/stream") == 1
    assert f"/api/orgs/{ORG}/runs/{RUN}" in paths
    assert paths[-1] == f"/api/orgs/{ORG}/runs/{RUN}/events"


def test_tail_resumes_from_the_last_pos_after_a_dropped_stream() -> None:
    """A dropped connection on a live run reconnects at the last position seen, not from scratch."""
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            status = "running" if len(attempts) < 2 else "completed"
            return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": status}))
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(200, content=_frames(_event(7), _event(8)))
        return httpx.Response(200, content=_frames(_event(9)))

    with _reader(handler) as reader:
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [7, 8, 9]
    # The resume cursor is the last pos served, sent both ways: the query the request was built
    # with, and the header the server prefers on a reconnect.
    assert dict(attempts[1].url.params) == {"after_pos": "8"}
    assert attempts[1].headers["Last-Event-ID"] == "8"


def test_tail_starts_from_an_explicit_cursor() -> None:
    """`--from-pos` opens the stream where a previous tail stopped."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))
        seen.append(request)
        return httpx.Response(200, content=_frames(_event(31)))

    with _reader(handler) as reader:
        list(reader.tail(RUN, after_pos=30))

    assert dict(seen[0].url.params) == {"after_pos": "30"}
    assert seen[0].headers["Last-Event-ID"] == "30"


def test_tail_reconnects_through_a_transport_failure() -> None:
    """One refused connection on a live run is a blip, not the end of the follow."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("connection reset")
        return httpx.Response(200, content=_frames(_event(12)))

    with _reader(handler) as reader:
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [12]
    assert len(calls) == 2


def test_tail_gives_up_after_enough_failures_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform that stays down ends the tail with the remedy, not an infinite retry loop."""
    monkeypatch.setattr("wmo.runs.reader.MAX_RECONNECTS", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json=_detail())
        raise httpx.ConnectError("refused")

    with (
        _reader(handler) as reader,
        pytest.raises(PlatformUnreachable, match="could not be re-established"),
    ):
        list(reader.tail(RUN))


def test_a_stream_that_answers_an_error_surfaces_it() -> None:
    """A 403 on the stream is a credential problem, reported rather than retried as a blip."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json=_detail())
        return httpx.Response(403, json={"error": "Organization not found: org-1"})

    with _reader(handler) as reader, pytest.raises(PlatformError, match="Organization not found"):
        list(reader.tail(RUN))


def test_a_frame_that_is_not_json_does_not_end_the_tail() -> None:
    """A proxy injecting a line must not end a twelve-hour follow with a decode traceback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))
        body = b"data: not json\n\n" + _frames(_event(5))
        return httpx.Response(200, content=body)

    with _reader(handler) as reader:
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [5]


# -- construction --------------------------------------------------------------------------------


def test_open_returns_none_without_a_credential() -> None:
    """Not logged in is a state a CLI reports in one line, not an exception.

    The suite has no credential at all (the autouse fixture in `wmo/conftest.py` points WMO_HOME at
    an empty directory), so this is also the check that a developer machine cannot leak its login
    into a test.
    """
    assert RunsReader.open() is None


def test_tail_drains_what_the_frontier_was_still_hiding() -> None:
    """The events that END a run are exactly the ones a naive tail misses.

    A live stream trails the safe frontier by a couple of seconds, so when the run goes terminal its
    last ledger line, final heartbeat and terminal status are usually still behind it. The tail does
    one paged read after observing terminal, which is where those three arrive.
    """
    late = [_event(9, event_type="ledger.line"), _event(10, event_type="run.status")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, content=_frames(_event(8)))
        if request.url.path.endswith("/events"):
            after = int(dict(request.url.params).get("after_pos", 0))
            remaining = [event for event in late if int(str(event["pos"])) > after]
            return httpx.Response(
                200,
                json={
                    "events": remaining,
                    "last_pos": remaining[-1]["pos"] if remaining else after,
                },
            )
        return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))

    with _reader(handler) as reader:
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [8, 9, 10]
    assert rows[-1].type == "run.status"


def test_a_stream_that_closes_empty_forever_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server closing immediately must not spin: that is indistinguishable from a hang."""
    monkeypatch.setattr("wmo.runs.reader.MAX_RECONNECTS", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, content=b"")
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"events": [], "last_pos": 0})
        return httpx.Response(200, json=_detail())

    with (
        _reader(handler) as reader,
        pytest.raises(PlatformUnreachable, match="without any events"),
    ):
        list(reader.tail(RUN))


def test_a_5xx_on_the_stream_is_retried_but_a_403_is_not() -> None:
    """The platform having a bad minute is a blip; a credential answer is a verdict."""
    attempts: list[int] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/stream"):
            if request.url.path.endswith("/events"):
                return httpx.Response(200, json={"events": [], "last_pos": 0})
            return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "upstream unavailable"})
        return httpx.Response(200, content=_frames(_event(4)))

    with _reader(flaky) as reader:
        assert [row.pos for row in reader.tail(RUN)] == [4]
    assert len(attempts) == 2

    def forbidden(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stream"):
            return httpx.Response(403, json={"error": "Organization not found"})
        return httpx.Response(200, json=_detail())

    with _reader(forbidden) as reader, pytest.raises(PlatformError, match="Organization not found"):
        list(reader.tail(RUN))


def test_a_frame_of_the_wrong_shape_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame this client cannot parse must not end a twelve-hour follow with a traceback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stream"):
            body = b'data: {"pos": "not-an-int"}\n\n' + _frames(_event(6))
            return httpx.Response(200, content=body)
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"events": [], "last_pos": 6})
        return httpx.Response(200, json=_detail(run={**_RUN_ROW, "status": "completed"}))

    with _reader(handler) as reader, caplog.at_level(logging.WARNING, logger="wmo.runs.reader"):
        rows = list(reader.tail(RUN))

    assert [row.pos for row in rows] == [6]
    assert any("does not understand" in record.getMessage() for record in caplog.records)


def test_an_org_slug_resolves_to_its_id() -> None:
    """A slug is what a person reads off the platform's URL; a uuid is not something anyone has."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/whoami":
            return httpx.Response(
                200,
                json={
                    "actor": {"kind": "api_key", "id": "key-1"},
                    "orgs": [{"id": ORG, "slug": "acme", "name": "Acme"}],
                },
            )
        return httpx.Response(200, json={"runs": [], "next_cursor": None})

    reader = RunsReader(
        PlatformClient("https://api.test", "xpl_secret", transport=httpx.MockTransport(handler)),
        _resolve_org(
            PlatformClient(
                "https://api.test", "xpl_secret", transport=httpx.MockTransport(handler)
            ),
            "acme",
        ),
    )
    with reader:
        reader.list_runs()

    assert reader.org_id == ORG
    assert f"/api/orgs/{ORG}/runs" in seen


def test_a_uuid_org_costs_no_lookup() -> None:
    """The common path stays one request: an id needs no resolving."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a uuid must not be looked up (asked {request.url.path})")

    client = PlatformClient("https://api.test", "x", transport=httpx.MockTransport(handler))
    with client:
        assert _resolve_org(client, "6f1b7a4e-6c2e-4f2a-9a3e-0b8f6f7c1d22") == (
            "6f1b7a4e-6c2e-4f2a-9a3e-0b8f6f7c1d22"
        )


def test_an_unknown_org_name_lists_what_is_visible() -> None:
    """The error is the interface: it says what to pass instead."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "actor": {"kind": "api_key", "id": "key-1"},
                "orgs": [{"id": ORG, "slug": "acme", "name": "Acme"}],
            },
        )

    client = PlatformClient("https://api.test", "x", transport=httpx.MockTransport(handler))
    with client, pytest.raises(PlatformError, match="it can see: acme"):
        _resolve_org(client, "nope")

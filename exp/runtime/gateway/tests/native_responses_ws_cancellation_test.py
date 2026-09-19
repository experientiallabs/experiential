"""Actual WebSocket disconnect propagation during deferred native-tool input."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.tests.native_responses_tool_translation_test import (
    _ToolUpstream,
    root_engine,  # noqa: F401 - fixture declares its name as engine.
)
from exp.runtime.gateway.tests.native_waterfall_test import _ServingEngine


def _frame(prompt: str, **extra: object) -> str:
    """Build a native-tool request served by the real translation/control plane."""
    return json.dumps(
        {
            "type": "response.create",
            "model": "coding",
            "input": prompt,
            "tools": [
                {"type": "custom", "name": "apply_patch"},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [
                        {"type": "function", "name": "close", "parameters": {"type": "object"}},
                    ],
                },
                {"type": "function", "name": "plain", "parameters": {"type": "object"}},
            ],
            **extra,
        }
    )


def _until(socket: Connection, event_type: str) -> JsonObject:
    """Read one response boundary without hiding an unexpected failure."""
    for _ in range(500):
        event = json.loads(socket.recv(timeout=5))
        if event["type"] == event_type:
            return event
        assert event["type"] not in {"error", "response.failed"}, event
    pytest.fail("Response boundary did not arrive")


def _settled(engine: _ServingEngine, expected: list[str]) -> None:
    """Require prompt single settlement and no leftover dispatched attempts."""
    deadline = time.monotonic() + 2
    while True:
        with sqlite3.connect(engine.database_path) as db:
            states = [
                row[0]
                for row in db.execute("SELECT state FROM gateway_attempts ORDER BY started_at")
            ]
            requests = db.execute("SELECT count(*) FROM gateway_requests").fetchone()
        if states == expected:
            assert requests == (len(expected),)
            return
        assert time.monotonic() < deadline, states
        time.sleep(0.02)


@pytest.mark.parametrize(
    "prompt,boundary",
    [
        ("cancel", "response.created"),
        ("cancel", "response.output_item.added"),
        ("text-cancel", "response.output_text.delta"),
        ("success", "response.completed"),
        ("provider-error", "response.failed"),
    ],
)
def test_websocket_disconnect_releases_selected_attempt(
    engine: _ServingEngine, prompt: str, boundary: str
) -> None:
    """Close before output, during deferred input, text, and after observed terminals."""
    with connect(
        engine.base.replace("http://", "ws://") + "/v1/responses",
        additional_headers={"authorization": f"Bearer {engine.raw_key}"},
        close_timeout=0.2,
    ) as socket:
        socket.send(_frame(prompt))
        event = _until(socket, boundary)
        if boundary == "response.output_item.added":
            item = event["item"]
            assert isinstance(item, dict) and item["type"] == "custom_tool_call"
    expected = (
        "completed"
        if prompt == "success"
        else "failed"
        if prompt == "provider-error"
        else "cancelled"
    )
    _settled(engine, [expected])
    assert _ToolUpstream.calls == 1
    if expected == "cancelled":
        assert _ToolUpstream.stopped.wait(2), "Provider stream continued after disconnect"


def test_queued_frames_ping_and_continuation_preserve_order(engine: _ServingEngine) -> None:
    """Reads for close keep queued text/binary requests without concurrent admission."""
    with connect(
        engine.base.replace("http://", "ws://") + "/v1/responses",
        additional_headers={"authorization": f"Bearer {engine.raw_key}"},
    ) as socket:
        socket.send(_frame("text-gated"))
        _until(socket, "response.output_text.delta")
        socket.send(b"binary request")
        socket.send(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "coding",
                    "generate": False,
                    "metadata": {"order": "one"},
                }
            )
        )
        socket.send(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "coding",
                    "generate": False,
                    "metadata": {"order": "two"},
                }
            )
        )
        socket.send(_frame("text-success"))
        pong = socket.ping(b"pending-check")
        assert pong.wait(1), "Ping was hidden behind a pending body read"
        assert _ToolUpstream.calls == 1
        _ToolUpstream.release.set()
        completed = _until(socket, "response.completed")
        rejected = json.loads(socket.recv(timeout=5))
        assert rejected["type"] == "error" and rejected["status"] == 400
        for order in ("one", "two"):
            prewarm = _until(socket, "response.completed")["response"]
            assert isinstance(prewarm, dict) and prewarm["metadata"] == {"order": order}
        _until(socket, "response.completed")  # The queued generating request runs exactly once.
        response = completed["response"]
        assert isinstance(response, dict)
        socket.send(_frame("text-success", previous_response_id=response["id"]))
        continued = _until(socket, "response.completed")["response"]
        assert isinstance(continued, dict) and continued["status"] == "completed"
    _settled(engine, ["completed", "completed", "completed"])
    assert _ToolUpstream.calls == 3


@pytest.mark.parametrize("overflow", ["count", "bytes", "frame"])
def test_pending_limit_closes_without_admitting_queued_work(
    engine: _ServingEngine, overflow: str
) -> None:
    """Count, aggregate-byte and original frame limits never dispatch queued requests."""
    with connect(
        engine.base.replace("http://", "ws://") + "/v1/responses",
        additional_headers={"authorization": f"Bearer {engine.raw_key}"},
        compression=None,
        close_timeout=0.2,
    ) as socket:
        socket.send(_frame("cancel"))
        _until(socket, "response.output_item.added")
        with pytest.raises(ConnectionClosed) as closed:
            if overflow == "count":
                for _ in range(9):
                    socket.send(_frame("success"))
            elif overflow == "bytes":
                # Each frame is below the existing 16 MiB frame and 64 MiB message bounds.
                for _ in range(5):
                    socket.send(b"x" * (14 * 1024 * 1024))
            else:
                socket.send(b"x" * (17 * 1024 * 1024))
            while True:
                socket.recv(timeout=5)
        if overflow != "frame":
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 1009
    _settled(engine, ["cancelled"])
    assert _ToolUpstream.calls == 1
    assert _ToolUpstream.stopped.wait(2), "Overflow failed to stop the admitted provider stream"


def test_queued_request_is_discarded_on_disconnect(engine: _ServingEngine) -> None:
    """A pipelined request is transport state, never an admitted second operation after close."""
    with connect(
        engine.base.replace("http://", "ws://") + "/v1/responses",
        additional_headers={"authorization": f"Bearer {engine.raw_key}"},
        close_timeout=0.2,
    ) as socket:
        socket.send(_frame("cancel"))
        _until(socket, "response.output_item.added")
        socket.send(_frame("success"))
        assert socket.ping(b"queued").wait(1)
    _settled(engine, ["cancelled"])
    assert _ToolUpstream.calls == 1
    assert _ToolUpstream.stopped.wait(2)

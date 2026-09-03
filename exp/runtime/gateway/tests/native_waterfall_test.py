"""Integration tests for the native engine's certified deployment waterfall.

One live ``exp_gateway_native`` serving subprocess runs rust-only (no
fallback engine configured) over a seeded root whose granted alias is a
certified two-deployment pool. Two loopback mock
providers stand in for the ordered deployments; the first deployment's
behavior is selected by the request prompt so each test drives one waterfall
shape: persistent 500s (same-deployment redial then failover), one transient
500 (redial succeeds), a refusal-only stream (refusal failover under the
alias revision's opt-in), and a plain success. Every test asserts the durable
per-attempt rows (ordinals counting all physical dispatches, depths naming
the deployment position) through the request identity echoed in
``x-request-id``, and the module-level conservation check proves the ledger
holds no open rows once traffic settles.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
_REQUEST_TIMEOUT_SECONDS = 10.0

_DRIVER_SOURCE = textwrap.dedent(
    '''
    """Serve the native gateway engine over one seeded pool root until SIGTERM."""

    import json
    import os
    import socket
    import sys
    from pathlib import Path

    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.native_bridge import NativeControlPlane

    import exp_gateway_native


    def main() -> None:
        """Compose the control plane, announce the public port, and serve."""
        config = json.loads(sys.argv[1])
        components = load_gateway_components(
            Path(config["root"]),
            environment={"TEST_PROVIDER_KEY": os.environ["TEST_PROVIDER_KEY"]},
        )
        control_plane = NativeControlPlane(
            components,
            request_timeout_seconds=config["request_timeout_seconds"],
        )
        last_error = None
        for _attempt in range(5):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            sys.stdout.write(json.dumps({"port": port}) + "\\n")
            sys.stdout.flush()
            try:
                exp_gateway_native.serve(
                    control_plane,
                    json.dumps(
                        {
                            "host": "127.0.0.1",
                            "port": port,
                            "max_active_requests": 8,
                            "request_timeout_seconds": config["request_timeout_seconds"],
                            "graceful_timeout_seconds": 2.0,
                        }
                    ),
                )
                return
            except RuntimeError as error:
                if "failed to bind" not in str(error):
                    raise
                last_error = error
        raise SystemExit(f"no loopback port could be bound: {last_error}")


    if __name__ == "__main__":
        main()
    '''
).strip()


def _sse_frame(payload: object) -> bytes:
    """Encode one provider SSE data frame."""
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _content_chunk(text: str) -> bytes:
    """Encode one OpenAI-compatible streamed content delta."""
    return _sse_frame(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )


def _terminal_frames(*, prompt_tokens: int = 2, completion_tokens: int = 2) -> bytes:
    """Encode the stop, usage, and done frames closing one successful stream."""
    return b"".join(
        (
            _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _sse_frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )


class _PrimaryUpstream(BaseHTTPRequestHandler):
    """The first certified deployment; behavior is selected by the prompt.

    ``always-500`` fails every dispatch, ``retry-then-succeed`` fails once
    per process then answers, ``refuse`` streams a refusal-only completion,
    ``silent-length`` exhausts the output budget with no content (a
    thinking-only turn: ``finish_reason: length``, zero deltas), and anything
    else streams a plain success.
    """

    retry_counts: dict[str, int] = {}
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer one dispatch according to the scripted prompt behavior."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][-1]["content"]
        if prompt == "always-500":
            self.send_response(500)
            self.end_headers()
            return
        if prompt == "flood":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(_content_chunk("flood-start"))
                self.wfile.flush()
                block = _content_chunk("x" * 512)
                while True:
                    self.wfile.write(block)
                    self.wfile.flush()
                    time.sleep(0.02)
            except OSError:
                return
        if prompt == "retry-then-succeed":
            with self.lock:
                seen = self.retry_counts.get(prompt, 0)
                self.retry_counts[prompt] = seen + 1
            if seen == 0:
                self.send_response(500)
                self.end_headers()
                return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if prompt == "silent-length":
                self.wfile.write(
                    _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                )
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 16},
                        }
                    )
                )
                self.wfile.write(b"data: [DONE]\n\n")
                return
            if prompt == "refuse":
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"refusal": "primary refuses"},
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )
                self.wfile.write(_terminal_frames())
                return
            self.wfile.write(_content_chunk("from-primary"))
            self.wfile.write(_terminal_frames())
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


class _SecondaryUpstream(BaseHTTPRequestHandler):
    """The second certified deployment; always streams a distinct success."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one canned success identifying this deployment."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(_content_chunk("from-secondary"))
            self.wfile.write(_terminal_frames(prompt_tokens=3, completion_tokens=1))
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


@dataclass(frozen=True)
class _ServingEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_key: str
    database_path: Path

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _chat_payload(prompt: str, *, stream: bool = False) -> JsonObject:
    """Return one Chat Completions body targeting the certified pool alias."""
    payload: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": prompt}],
    }
    if stream:
        payload["stream"] = True
    return payload


def _attempt_rows(engine: _ServingEngine, request_id: str) -> list[tuple[int, int, str]]:
    """Read one request's durable attempt rows in dispatch order."""
    deadline = time.monotonic() + 10.0
    while True:
        with sqlite3.connect(engine.database_path) as connection:
            rows = connection.execute(
                "SELECT attempt_ordinal, route_depth, state FROM gateway_attempts"
                " WHERE request_id = ? ORDER BY attempt_ordinal",
                (request_id,),
            ).fetchall()
        if rows and all(state not in {"dispatched", "running"} for _, _, state in rows):
            return [(int(ordinal), int(depth), str(state)) for ordinal, depth, state in rows]
        if time.monotonic() > deadline:
            return [(int(ordinal), int(depth), str(state)) for ordinal, depth, state in rows]
        time.sleep(0.05)


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one shared native engine subprocess over a certified pool root.

    The alias revision opts into refusal failover so the refusal scenario can
    advance; the other scenarios are unaffected by the flag.

    Yields:
        The live serving facts as a :class:`_ServingEngine`.
    """
    root = tmp_path_factory.mktemp("native-waterfall-root")
    primary = ThreadingHTTPServer((_HOST, 0), _PrimaryUpstream)
    secondary = ThreadingHTTPServer((_HOST, 0), _SecondaryUpstream)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (primary, secondary)
    ]
    for thread in threads:
        thread.start()
    manager, raw_key = _configured_pool_gateway(
        root,
        refusal_failover=True,
        base_urls=(
            f"http://{_HOST}:{primary.server_address[1]}/v1",
            f"http://{_HOST}:{secondary.server_address[1]}/v1",
        ),
    )
    driver = root / "native_waterfall_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Record every port announcement the driver prints on stdout."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        reader = threading.Thread(target=_collect_announcements, daemon=True)
        reader.start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                    if live.status_code == 200 and live.json() == {"status": "live"}:
                        models = httpx.get(
                            f"http://{_HOST}:{port}/v1/models",
                            headers={"authorization": f"Bearer {raw_key}"},
                            timeout=2.0,
                        )
                        if models.status_code == 200 and [
                            item["id"] for item in models.json()["data"]
                        ] == ["coding"]:
                            break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(
            port=port,
            raw_key=raw_key,
            database_path=manager.database_path,
        )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        for server in (primary, secondary):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def test_transient_primary_failure_redials_the_same_deployment(
    engine: _ServingEngine,
) -> None:
    """One transient 500 redials the primary, which then serves the request."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("retry-then-succeed"),
        timeout=30.0,
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from-primary"
    assert response.headers["x-gateway-route-depth"] == "0"
    rows = _attempt_rows(engine, response.headers["x-request-id"])
    assert rows == [(0, 0, "failed"), (1, 0, "completed")]


def test_refusal_failover_withholds_the_refused_route(engine: _ServingEngine) -> None:
    """A refusal-only primary stream advances without exposing the refusal.

    The alias revision opts into refusal failover, so the withheld refusal
    deltas never reach the caller; the ledger records the refused attempt and
    the secondary serves the visible completion. Refusals never count toward
    the primary's failure circuit, so later scenarios still dial it first.
    """
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("refuse"),
        timeout=30.0,
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "from-secondary"
    assert "primary refuses" not in response.text
    assert response.headers["x-gateway-route-depth"] == "1"
    rows = _attempt_rows(engine, response.headers["x-request-id"])
    assert rows == [(0, 0, "failed"), (1, 1, "completed")]


def test_disconnect_mid_flood_settles_cancelled(engine: _ServingEngine) -> None:
    """Closing the client socket mid-stream settles the attempt cancelled."""
    request_id = ""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("flood", stream=True),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        request_id = response.headers["x-request-id"]
        for line in response.iter_lines():
            if "flood-start" in line:
                break
    rows = _attempt_rows(engine, request_id)
    assert [state for _, _, state in rows] == ["cancelled"]


def test_unknown_routes_answer_the_native_openai_envelope(
    engine: _ServingEngine,
) -> None:
    """Without a fallback engine, an unknown route is a native 404 envelope."""
    response = httpx.post(
        f"{engine.base}/v1/nonexistent",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"probe": True},
        timeout=10.0,
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "unknown_route"
    assert "/v1/chat/completions" in error["message"]


def _keyed_post(
    engine: _ServingEngine,
    path: str,
    operation: str,
    payload: JsonObject,
) -> httpx.Response:
    """Send one keyed POST carrying the shared Idempotency-Key header."""
    return httpx.post(
        f"{engine.base}{path}",
        headers={
            "authorization": f"Bearer {engine.raw_key}",
            "idempotency-key": operation,
        },
        json=payload,
        timeout=30.0,
    )


def test_keyed_chat_replays_the_owner_response_exactly(engine: _ServingEngine) -> None:
    """A repeated keyed chat operation replays the stored bytes, not a redial."""
    payload = _chat_payload("keyed chat")
    first = _keyed_post(engine, "/v1/chat/completions", "keyed-chat-op", payload)
    assert first.status_code == 200
    duplicate = _keyed_post(engine, "/v1/chat/completions", "keyed-chat-op", payload)
    assert duplicate.status_code == 200
    assert duplicate.content == first.content
    assert duplicate.headers["x-request-id"] == first.headers["x-request-id"]
    rows = _attempt_rows(engine, first.headers["x-request-id"])
    assert rows == [(0, 0, "completed")]


def test_keyed_responses_replays_the_owner_response_exactly(
    engine: _ServingEngine,
) -> None:
    """Keyed Responses runs the replay protocol natively, no fallback engine."""
    payload: JsonObject = {"model": "coding", "input": "keyed responses"}
    first = _keyed_post(engine, "/v1/responses", "keyed-responses-op", payload)
    assert first.status_code == 200
    assert first.json()["output"][0]["content"][0]["text"] == "from-primary"
    duplicate = _keyed_post(engine, "/v1/responses", "keyed-responses-op", payload)
    assert duplicate.status_code == 200
    assert duplicate.content == first.content
    rows = _attempt_rows(engine, first.headers["x-request-id"])
    assert rows == [(0, 0, "completed")]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_output_less_incomplete_response_stays_continuable(
    engine: _ServingEngine, stream: bool
) -> None:
    """A response id handed to the caller is continuable whatever its finish state.

    The first turn's provider spends the whole output budget without emitting
    a single delta (Gemini thinking at a small ``max_output_tokens``), so the
    attempt reaches ``length`` before any semantic output: the waterfall
    settles it output-less and the caller gets an ``incomplete`` response
    with no items. api.openai.com persists such a response, and so must the
    gateway: the next turn's ``previous_response_id`` resolves to the
    conversation so far and is served, never ``previous_response_not_found``
    (reproduced 2/3 on staging gemini-3.7-flash, 2026-09-03).
    """
    headers = {"authorization": f"Bearer {engine.raw_key}"}
    first_payload: JsonObject = {"model": "coding", "input": "silent-length", "stream": stream}
    first = httpx.post(
        f"{engine.base}/v1/responses", headers=headers, json=first_payload, timeout=30.0
    )
    assert first.status_code == 200, first.text
    if stream:
        completed = [
            json.loads(line[len("data: ") :])
            for line in first.text.splitlines()
            if line.startswith("data: ")
        ]
        terminal = completed[-1]
        assert terminal["type"] == "response.incomplete", terminal
        response = terminal["response"]
    else:
        response = first.json()
    assert response["status"] == "incomplete"
    assert response["output"] == []
    response_id = response["id"]
    assert response_id.startswith("resp_")
    assert _attempt_rows(engine, first.headers["x-request-id"]) == [(0, 0, "incomplete")]

    second = httpx.post(
        f"{engine.base}/v1/responses",
        headers=headers,
        json={"model": "coding", "input": "second turn", "previous_response_id": response_id},
        timeout=30.0,
    )
    assert second.status_code == 200, second.text
    assert second.json()["status"] == "completed"
    assert second.json()["output"][0]["content"][0]["text"] == "from-primary"
    assert _attempt_rows(engine, second.headers["x-request-id"]) == [(0, 0, "completed")]


def test_persistent_primary_failure_fails_over_to_the_second_deployment(
    engine: _ServingEngine,
) -> None:
    """Two failed primary dispatches precede the winning secondary attempt.

    A 500 is retryable on the same deployment, so the primary is dialed
    twice (its bounded cap) before failover; the terminal attempt completes
    on route depth one and the response carries the winning deployment's
    output and route headers. The two operational failures open the primary's
    health circuit, which the streaming scenario below observes.
    """
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("always-500"),
        timeout=30.0,
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from-secondary"
    assert response.headers["x-gateway-route-depth"] == "1"
    rows = _attempt_rows(engine, response.headers["x-request-id"])
    assert rows == [(0, 0, "failed"), (1, 0, "failed"), (2, 1, "completed")]


def test_streaming_request_skips_the_open_primary_circuit(
    engine: _ServingEngine,
) -> None:
    """An open primary circuit routes a streamed request straight to depth one.

    The previous scenario's two operational failures opened the primary's
    circuit, so this streamed request dispatches once on the fallback and its
    committed headers name the winning deployment position before the first
    byte flows.
    """
    collected = b""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("always-500", stream=True),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        assert response.headers["x-gateway-route-depth"] == "1"
        request_id = response.headers["x-request-id"]
        for chunk in response.iter_bytes():
            collected += chunk
    assert b"from-secondary" in collected
    assert collected.rstrip().endswith(b"data: [DONE]")
    rows = _attempt_rows(engine, request_id)
    assert rows == [(0, 1, "completed")]


def test_ledger_conserves_every_admitted_request(engine: _ServingEngine) -> None:
    """Every accepted request settles: no open attempts, matched totals.

    Runs last in the module (pytest preserves definition order), so it sees
    the traffic of every scenario above plus its own success probe, which the
    still-open primary circuit routes to the fallback in one dispatch.
    """
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("plain success"),
        timeout=30.0,
    )
    assert response.status_code == 200
    report = httpx.get(f"{engine.base}/usage.json", timeout=5.0).json()
    # Seven scenario requests, the output-less continuation scenario's four
    # (two first turns and their two continuations), and this probe.
    assert report["totals"]["requests"] == 12
    terminal_attempts = sum(int(count["attempts"]) for count in report["totals"]["terminal_counts"])
    with sqlite3.connect(engine.database_path) as connection:
        (total_attempts,) = connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()
        (open_attempts,) = connection.execute(
            "SELECT count(*) FROM gateway_attempts WHERE state IN ('dispatched', 'running')"
        ).fetchone()
    assert open_attempts == 0
    # Twelve single-dispatch requests plus the four extra physical attempts the
    # redial and failover scenarios spend.
    assert terminal_attempts == total_attempts == 16

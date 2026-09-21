"""Failure-path integration tests for the native (Rust) gateway data plane.

These tests exercise disconnect, heartbeat, accounting, replay, and bounded
backpressure behavior of the compiled engine against a real serving process:

1. A client disconnect mid non-streaming request settles the admitted attempt
   through the ``AttemptGuard`` drop backstop.
2. An escalated admission fails closed as the shared internal error while
   the native chat path and readiness stay healthy.
3. A connected client that stops reading a stream cannot pin the gateway past
   the request deadline; ``send_bounded`` settles the attempt.

``exp_gateway_native.serve`` blocks its caller and stops only on SIGINT or
SIGTERM, so one shared serving subprocess (a small generated driver that
composes ``NativeControlPlane`` over a seeded root) hosts every scenario. Its
host policy deliberately escalates Responses requests and one alias so the
fail-closed escalation boundary is exercised even though every route shape is
natively supported.
Each test observes settlement deltas through the content-free ``/usage.json``
report, and the subprocess is stopped with SIGTERM at module teardown.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import socket
import sqlite3
import struct
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
from exp.runtime.gateway.lifecycle_test import (
    _activate_alias_for_escalation_policy,
    _configured_gateway,
)
from exp.runtime.gateway.management import GatewayManagement

if sys.platform != "win32":
    import resource

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
# The serving process is shared, so this bound applies to every scenario. It
# must stay short enough that the stalled-reader test finishes quickly and
# long enough that the disconnect test can prove the drop backstop settled
# strictly before the deadline could.
_REQUEST_TIMEOUT_SECONDS = 5.0
# The bridge closes abandoned attempts at deadline + _SWEEP_GRACE_SECONDS
# (5s). Every settlement observation below completes before that instant, so
# an observed terminal is attributable to the data plane, never to the sweep.
_SWEEP_FLOOR_SECONDS = _REQUEST_TIMEOUT_SECONDS + 5.0

_DRIVER_SOURCE = textwrap.dedent(
    '''
    """Serve the native gateway engine over one seeded root until SIGTERM."""

    import json
    import os
    import socket
    import sys
    import time
    from pathlib import Path

    from exp.runtime.gateway.contracts import GatewayApiSurface
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.native_bridge import NativeControlPlane

    import exp_gateway_native


    def native_route_eligible(route, request) -> bool:
        """Escalate Responses requests and the fixed ``escalated`` alias.

        Every granted provider now has a native dialect and every route
        shape resolves natively, so this hosted policy is the only
        construction-independent escalation lever left for exercising the
        fail-closed escalation boundary.
        """
        if request.surface == GatewayApiSurface.RESPONSES:
            return False
        return route.snapshot.authorization.alias != "escalated"


    class ObservedControlPlane(NativeControlPlane):
        """Record content-free callback evidence and choose a loopback wire."""

        def admit(self, argument: str) -> str:
            """Use Messages wire only for the hidden-thinking socket fixtures."""
            admission = json.loads(super().admit(argument))
            request = json.loads(json.loads(argument)["body"])
            prompt = request["messages"][-1]["content"]
            if prompt.startswith("anthro-") and "route" in admission:
                admission["route"][0]["dialect"] = "anthropic_messages"
            if prompt.startswith("hidden-") and "route" in admission:
                admission["route"][0]["fireworks_reasoning_route_sha256"] = "a" * 64
            if prompt == "quiet-short-phase" and "route" in admission:
                admission["route"][0]["timeout_seconds"] = 1.5
            return json.dumps(admission)

        def settle(self, argument: str) -> str:
            """Observe every settlement and delay one write to expose close ordering."""
            data = json.loads(argument)
            with open(os.environ["SETTLEMENT_LOG"], "a") as sink:
                sink.write(json.dumps(data) + "\\n")
            time.sleep(0.3)
            return super().settle(argument)


    def main() -> None:
        """Compose the control plane, announce the public port, and serve.

        The port is probed with a bind-then-close, so another process could
        claim it before the engine's own bind; a failed bind is retried on a
        fresh port, and every attempt announces its port as one JSON line so
        the test always polls the latest announcement.
        """
        config = json.loads(sys.argv[1])
        components = load_gateway_components(
            Path(config["root"]),
            environment={"TEST_PROVIDER_KEY": os.environ["TEST_PROVIDER_KEY"]},
        )
        control_plane = ObservedControlPlane(
            components,
            request_timeout_seconds=config["request_timeout_seconds"],
            native_route_eligible=native_route_eligible,
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


def _seed_escalating_alias(root: Path, manager: GatewayManagement) -> None:
    """Grant one alias the driver's host policy always escalates by name.

    Every granted provider now has a native dialect and every route shape
    resolves natively, so the driver's ``native_route_eligible`` hook is what
    makes this alias escalated by construction; see that hook in
    ``_DRIVER_SOURCE``.

    Args:
        root: Seeded gateway root.
        manager: Management handle over the same root.
    """
    _activate_alias_for_escalation_policy(root, manager, alias="escalated")


def _sse_frame(payload: object) -> bytes:
    """Encode one provider SSE data frame."""
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _content_chunk(text: str) -> bytes:
    """Encode one OpenAI-compatible streamed content delta."""
    return _sse_frame(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )


_TERMINAL_FRAMES = b"".join(
    (
        _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _sse_frame({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}}),
        b"data: [DONE]\n\n",
    )
)


class _SseUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible SSE provider whose pacing is chosen by the prompt.

    The user message content selects the streaming shape: ``fast-token``
    answers immediately, ``slow-token`` spreads a short answer over several
    seconds, and ``flood-token`` streams chunks without end so the gateway's
    public send channel and socket buffers fill against a stalled reader.
    """

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one canned SSE response selected by the request prompt."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][-1]["content"]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if prompt.startswith(("quiet-", "hidden-", "anthro-", "beforeheaders-", "keyed-")):
                _serve_quiet(self, prompt)
                return
            if prompt == "slow-token":
                for _ in range(10):
                    self.wfile.write(_content_chunk("tick "))
                    self.wfile.flush()
                    time.sleep(0.5)
            elif prompt == "flood-token":
                # Never terminates: the stream grows until the gateway drops
                # the connection, so no host's TCP buffering can absorb it
                # and the attempt can only settle through the deadline path.
                block = _content_chunk("x" * 2048)
                while True:
                    self.wfile.write(block)
                    self.wfile.flush()
            else:
                self.wfile.write(_content_chunk("hello "))
                self.wfile.write(_content_chunk("world"))
            self.wfile.write(_TERMINAL_FRAMES)
            self.wfile.flush()
        except OSError:
            # The gateway dropped the upstream connection mid-write; that is
            # the expected outcome of the disconnect and stall scenarios.
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


_PROVIDER_OPENED: dict[str, threading.Event] = {}
_PROVIDER_CLOSED: dict[str, threading.Event] = {}
_PROVIDER_CALLS: dict[str, int] = {}


def _serve_quiet(handler: _SseUpstream, prompt: str) -> None:
    """Serve silent bodies while observing the exact upstream socket close."""
    _PROVIDER_CALLS[prompt] = _PROVIDER_CALLS.get(prompt, 0) + 1
    closed = _PROVIDER_CLOSED.setdefault(prompt, threading.Event())
    if prompt.startswith("anthro-"):
        handler.wfile.write(
            _sse_frame(
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 19, "output_tokens": 0}},
                }
            )
        )
        handler.wfile.write(
            _sse_frame(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            )
        )
        handler.wfile.write(
            _sse_frame(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "private canary"},
                }
            )
        )
    elif prompt.startswith("hidden-"):
        if not prompt.startswith("hidden-preheaders-"):
            # Heartbeats belong to an already committed public stream. Private
            # reasoning alone must remain failover-safe before commitment.
            handler.wfile.write(
                _sse_frame({"choices": [{"index": 0, "delta": {"content": "public prefix"}}]})
            )
        handler.wfile.write(
            _sse_frame(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 19, "completion_tokens": 0},
                }
            )
        )
        handler.wfile.write(
            _sse_frame(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "private canary"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    elif prompt == "beforeheaders-known":
        handler.wfile.write(
            _sse_frame(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 19, "completion_tokens": 7},
                }
            )
        )
    elif not prompt.startswith("beforeheaders-"):
        if "partial" in prompt:
            handler.wfile.write(_sse_frame({"choices": [], "usage": {"prompt_tokens": 19}}))
        if "known" in prompt and "unknown" not in prompt:
            handler.wfile.write(
                _sse_frame(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 19, "completion_tokens": 7},
                    }
                )
            )
        handler.wfile.write(_content_chunk("usable partial answer"))
    handler.wfile.flush()
    _PROVIDER_OPENED.setdefault(prompt, threading.Event()).set()
    until = time.monotonic() + (2.4 if prompt == "keyed-finish" else 8)
    with selectors.DefaultSelector() as selector:
        selector.register(handler.connection, selectors.EVENT_READ)
        while time.monotonic() < until:
            if selector.select(0.04):
                try:
                    if not handler.connection.recv(1, socket.MSG_PEEK):
                        closed.set()
                        return
                except OSError:
                    closed.set()
                    return
            if "pings" in prompt:
                handler.wfile.write(b": provider ping\n\n")
                handler.wfile.flush()
    if prompt.startswith("keyed-"):
        handler.wfile.write(_TERMINAL_FRAMES)
        handler.wfile.flush()


@dataclass(frozen=True)
class _ServingEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_key: str
    database_path: Path
    settlement_log: Path
    stderr_log: Path

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _chat_payload(prompt: str, *, stream: bool = False) -> bytes:
    """Return one raw Chat Completions body targeting the seeded alias."""
    payload: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": prompt}],
    }
    if stream:
        payload["stream"] = True
    return json.dumps(payload).encode()


def _raw_chat_request(raw_key: str, body: bytes) -> bytes:
    """Encode one HTTP/1.1 chat request for a hand-driven client socket."""
    head = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"host: {_HOST}\r\n"
        f"authorization: Bearer {raw_key}\r\n"
        "content-type: application/json\r\n"
        f"content-length: {len(body)}\r\n"
        "\r\n"
    )
    return head.encode() + body


def _terminal_attempts(engine: _ServingEngine, state: str) -> int:
    """Read one terminal-state attempt count from the live usage report."""
    report = httpx.get(f"{engine.base}/usage.json", timeout=5.0).json()
    for count in report["totals"]["terminal_counts"]:
        if count["state"] == state:
            return int(count["attempts"])
    return 0


def _await_cancelled_attempts(
    engine: _ServingEngine,
    *,
    minimum: int,
    deadline: float,
) -> bool:
    """Poll the ledger until enough cancelled attempts settle or time runs out."""
    while time.monotonic() < deadline:
        if _terminal_attempts(engine, "cancelled") >= minimum:
            return True
        time.sleep(0.1)
    return _terminal_attempts(engine, "cancelled") >= minimum


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one shared native engine subprocess over a seeded root.

    The gateway root holds a single direct alias against a local SSE mock
    upstream. The subprocess is stopped with SIGTERM and must exit cleanly.

    Yields:
        The live serving facts as a :class:`_ServingEngine`.
    """
    root = tmp_path_factory.mktemp("native-engine-root")
    upstream = ThreadingHTTPServer((_HOST, 0), _SseUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1",
    )
    _seed_escalating_alias(root, manager)
    driver = root / "native_engine_driver.py"
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
    settlement_log = root / "settlements.jsonl"
    environment["SETTLEMENT_LOG"] = str(settlement_log)
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
            # The driver retries a lost bind race on a fresh port, so always
            # poll the most recently announced one. Liveness alone could be
            # answered by a foreign listener that claimed a stolen port, so
            # the port is accepted only once it also serves the seeded grant
            # for our own key, which nothing but this engine can do.
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
                        ] == ["coding", "escalated"]:
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
            settlement_log=settlement_log,
            stderr_log=stderr_log,
        )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _open_quiet(
    engine: _ServingEngine,
    prompt: str,
    *,
    surface: str = "chat",
    keyed: bool = False,
) -> tuple[socket.socket, str, bytes]:
    """Open a real public socket and read only its initial SSE bytes."""
    body = json.loads(_chat_payload(prompt, stream=True))
    if surface == "messages":
        body["max_tokens"] = 128
    request = _raw_chat_request(engine.raw_key, json.dumps(body).encode())
    if surface == "messages":
        request = request.replace(b"/v1/chat/completions", b"/v1/messages", 1)
    if keyed:
        request = request.replace(
            b"content-type:", f"Idempotency-Key: {prompt}\r\ncontent-type:".encode(), 1
        )
    client = socket.create_connection((_HOST, engine.port), timeout=5)
    client.sendall(request)
    received = b""
    while b"\r\n\r\n" not in received or b"data:" not in received:
        chunk = client.recv(4096)
        assert chunk, received
        received += chunk
    header_text = received.split(b"\r\n\r\n", 1)[0].decode()
    assert "200 OK" in header_text, received
    request_id = next(
        line.split(": ", 1)[1]
        for line in header_text.split("\r\n")
        if line.lower().startswith("x-request-id:")
    )
    return client, request_id, received


def _abort(client: socket.socket) -> None:
    """Force an RST so a legitimate write-side half-close cannot mask loss."""
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    client.close()


def _attempt(engine: _ServingEngine, request_id: str) -> sqlite3.Row:
    """Await exactly one durable terminal row without relying on sweep cleanup."""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with sqlite3.connect(engine.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM gateway_attempts WHERE request_id = ?", (request_id,)
            ).fetchall()
        if len(rows) == 1 and rows[0]["terminal_at"] is not None:
            return rows[0]
        time.sleep(0.02)
    pytest.fail(f"one terminal attempt missing for {request_id}")


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("known", [False, True])
def test_quiet_disconnect_closes_transport_before_settlement(
    engine: _ServingEngine,
    surface: str,
    known: bool,
) -> None:
    """Cancel while next_event waits; retain the meter or honest unknown."""
    prompt = f"quiet-{'known' if known else 'unknown'}-{surface}"
    client, request_id, _ = _open_quiet(engine, prompt, surface=surface)
    started = time.monotonic()
    _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25), "provider socket outlived delayed settlement"
    assert time.monotonic() - started < 0.3
    row = _attempt(engine, request_id)
    assert row["state"] == "cancelled"
    assert row["failure_class"] == "cancelled"
    assert row["input_tokens"] == (19 if known else None)
    assert row["output_tokens"] == (7 if known else None)
    assert row["usage_source"] == ("observed" if known else "unknown")
    payloads = [json.loads(line) for line in engine.settlement_log.read_text().splitlines()]
    writes = [entry for entry in payloads if entry["request_id"] == request_id]
    assert len(writes) == 1
    assert writes[0]["dispatched"] is True
    assert writes[0]["usage_incomplete_due_to_disconnect"] is True


@pytest.mark.parametrize("surface", ["chat", "messages"])
def test_partial_meter_disconnect_preserves_unknown_output(
    engine: _ServingEngine, surface: str
) -> None:
    """The real wire's input-only report cannot become free zero-output final usage."""
    prompt = f"quiet-partial-{surface}"
    client, request_id, _ = _open_quiet(engine, prompt, surface=surface)
    _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25)
    row = _attempt(engine, request_id)
    assert row["state"] == "cancelled"
    assert row["input_tokens"] == 19
    assert row["output_tokens"] is None
    assert row["estimated_cost_nano_usd"] is None
    writes = [json.loads(line) for line in engine.settlement_log.read_text().splitlines()]
    own = [entry for entry in writes if entry["request_id"] == request_id]
    assert len(own) == 1
    assert own[0]["usage_incomplete_due_to_disconnect"] is True


@pytest.mark.parametrize("surface", ["chat", "messages"])
def test_hidden_thinking_has_heartbeats_and_retains_input_meter(
    engine: _ServingEngine,
    surface: str,
) -> None:
    """Hidden thought and provider pings neither count as public tokens nor suppress heartbeats."""
    prompt = f"hidden-pings-{surface}"
    client, request_id, received = _open_quiet(engine, prompt, surface=surface)
    try:
        until = time.monotonic() + 3
        while b": keepalive" not in received:
            assert time.monotonic() < until
            received += client.recv(4096)
        assert b"private canary" not in received
    finally:
        _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25)
    row = _attempt(engine, request_id)
    assert row["state"] == "cancelled"
    assert row["input_tokens"] == 19
    assert row["output_tokens"] == 0
    logs = engine.stderr_log.read_text()
    assert '"usage_final":false' in logs
    assert "private canary" not in logs


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor limits are unavailable")
def test_provider_close_observation_handles_file_descriptors_above_select_limit(
    engine: _ServingEngine,
) -> None:
    """The loopback fixture still witnesses cancellation after a long suite uses high FDs."""
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit < 1100:
        pytest.skip("existing file descriptor limit leaves insufficient safe high-FD headroom")
    descriptors: list[int] = []
    client: socket.socket | None = None
    try:
        for _ in range(1050):
            try:
                descriptor = os.open(os.devnull, os.O_RDONLY)
            except OSError:
                pytest.skip("existing file descriptor availability cannot support bounded fixture")
            descriptors.append(descriptor)
            if descriptor >= 1030:
                break
        assert descriptors[-1] >= 1030
        prompt = "quiet-unknown-high-fd"
        client, request_id, received = _open_quiet(engine, prompt)
        assert client.fileno() > 1024
        while b"partial answer" not in received:
            received += client.recv(4096)
        _abort(client)
        client = None
        assert _PROVIDER_CLOSED[prompt].wait(0.25)
        assert _attempt(engine, request_id)["state"] == "cancelled"
    finally:
        if client is not None:
            client.close()
        for descriptor in descriptors:
            os.close(descriptor)


def test_anthropic_start_meter_survives_chat_disconnect(engine: _ServingEngine) -> None:
    """Retain native provider start usage before a delayed terminal meter."""
    prompt = "anthro-start-usage"
    client, request_id, _ = _open_quiet(engine, prompt)
    _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25)
    row = _attempt(engine, request_id)
    assert row["state"] == "cancelled"
    assert row["input_tokens"] == 19
    assert row["output_tokens"] == 0


@pytest.mark.parametrize("known", [False, True])
def test_preheaders_disconnect_preserves_observed_unknown_and_closes_socket(
    engine: _ServingEngine,
    known: bool,
) -> None:
    """Cancel initial request while its committed token and public headers are still pending."""
    prompt = "beforeheaders-known" if known else "beforeheaders-no-token"
    previous = set()
    if engine.settlement_log.exists():
        previous = {
            json.loads(line)["request_id"]
            for line in engine.settlement_log.read_text().splitlines()
        }
    cancelled_before = _terminal_attempts(engine, "cancelled")
    client = socket.create_connection((_HOST, engine.port), timeout=5)
    client.sendall(_raw_chat_request(engine.raw_key, _chat_payload(prompt, stream=True)))
    opened = _PROVIDER_OPENED.setdefault(prompt, threading.Event())
    assert opened.wait(2)
    time.sleep(0.05)
    _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25)
    assert _await_cancelled_attempts(
        engine, minimum=cancelled_before + 1, deadline=time.monotonic() + 2
    )
    payloads = [json.loads(line) for line in engine.settlement_log.read_text().splitlines()]
    current = [entry for entry in payloads if entry["request_id"] not in previous]
    assert len(current) == 1
    assert current[0]["dispatched"] is True
    assert current[0]["usage_incomplete_due_to_disconnect"] is True
    assert current[0]["usage"] == (
        {
            "input_tokens": 19,
            "output_tokens": 7,
            "cached_input_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_creation_1h_input_tokens": None,
            "reasoning_tokens": None,
        }
        if known
        else None
    )


@pytest.mark.parametrize("surface", ["chat", "messages"])
def test_private_preheaders_disconnect_stops_uncommitted_provider(
    engine: _ServingEngine,
    surface: str,
) -> None:
    """Private progress stays uncommitted while caller loss still closes its transport."""
    prompt = f"hidden-preheaders-{surface}"
    previous = (
        {json.loads(line)["request_id"] for line in engine.settlement_log.read_text().splitlines()}
        if engine.settlement_log.exists()
        else set()
    )
    body = json.loads(_chat_payload(prompt, stream=True))
    body["max_tokens"] = 128
    request = _raw_chat_request(engine.raw_key, json.dumps(body).encode())
    if surface == "messages":
        request = request.replace(b"/v1/chat/completions", b"/v1/messages", 1)
    cancelled_before = _terminal_attempts(engine, "cancelled")
    client = socket.create_connection((_HOST, engine.port), timeout=5)
    try:
        client.sendall(request)
        assert _PROVIDER_OPENED.setdefault(prompt, threading.Event()).wait(2)
        client.settimeout(0.05)
        with pytest.raises(TimeoutError):
            client.recv(1)
    finally:
        _abort(client)
    assert _PROVIDER_CLOSED[prompt].wait(0.25)
    assert _await_cancelled_attempts(
        engine, minimum=cancelled_before + 1, deadline=time.monotonic() + 2
    )
    current = [
        entry
        for line in engine.settlement_log.read_text().splitlines()
        if (entry := json.loads(line))["request_id"] not in previous
    ]
    assert len(current) == 1
    assert current[0]["dispatched"] is True
    assert current[0]["usage_incomplete_due_to_disconnect"] is True
    assert current[0]["usage"]["input_tokens"] == 19
    assert current[0]["usage"]["output_tokens"] == 0
    assert _PROVIDER_CALLS[prompt] == 1
    assert "private canary" not in engine.stderr_log.read_text()


def test_repeated_partial_answer_drops_never_fabricate_final_usage(engine: _ServingEngine) -> None:
    """Adversarial drops retain UNKNOWN; they require platform unresolved-budget protection."""
    for index in range(3):
        prompt = f"quiet-unknown-repeat-{index}"
        client, request_id, received = _open_quiet(engine, prompt)
        while b"usable partial answer" not in received:
            received += client.recv(4096)
        _abort(client)
        assert _PROVIDER_CLOSED[prompt].wait(0.25)
        row = _attempt(engine, request_id)
        assert row["state"] == "cancelled"
        assert row["usage_source"] == "unknown"
        assert row["estimated_cost_nano_usd"] is None
        assert _PROVIDER_CALLS[prompt] == 1
        writes = [json.loads(line) for line in engine.settlement_log.read_text().splitlines()]
        own = [entry for entry in writes if entry["request_id"] == request_id]
        assert len(own) == 1
        assert own[0]["usage_incomplete_due_to_disconnect"] is True


def test_keyed_retry_replays_one_completed_provider_without_heartbeats(
    engine: _ServingEngine,
) -> None:
    """A lost keyed subscriber does not abandon the bounded owner or poison its replay."""
    prompt = "keyed-finish"
    client, request_id, received = _open_quiet(engine, prompt, keyed=True)
    while b": keepalive" not in received:
        received += client.recv(4096)
    _abort(client)
    headers = {"authorization": f"Bearer {engine.raw_key}", "Idempotency-Key": prompt}
    retry = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        content=_chat_payload(prompt, stream=True),
        timeout=5,
    )
    assert retry.status_code == 200
    assert "usable partial answer" in retry.text
    assert "[DONE]" in retry.text
    assert ": keepalive" not in retry.text
    assert retry.headers["x-request-id"] == request_id
    assert _PROVIDER_CALLS[prompt] == 1
    row = _attempt(engine, request_id)
    assert row["state"] == "completed"
    again = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        content=_chat_payload(prompt, stream=True),
        timeout=5,
    )
    assert again.content == retry.content
    assert _PROVIDER_CALLS[prompt] == 1


def test_client_disconnect_mid_nonstreaming_request_settles_cancelled(
    engine: _ServingEngine,
) -> None:
    """The drop backstop settles a non-streaming attempt on client disconnect.

    The mock upstream needs about five seconds to answer; the client aborts
    after half a second with an RST close (SO_LINGER zero, so the server sees
    a hard disconnect rather than a legitimate write-side half-close). The
    handler future is dropped, and the armed ``AttemptGuard`` must spawn the
    cancellation settlement. Observing the cancelled terminal strictly before
    the request deadline proves the backstop fired: every other failure path
    for this request (deadline timeout, control-plane sweep) lands later.
    """
    cancelled_before = _terminal_attempts(engine, "cancelled")
    client = socket.create_connection((_HOST, engine.port), timeout=10)
    started = time.monotonic()
    try:
        client.sendall(_raw_chat_request(engine.raw_key, _chat_payload("slow-token")))
        time.sleep(0.5)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    finally:
        client.close()
    settled = _await_cancelled_attempts(
        engine,
        minimum=cancelled_before + 1,
        deadline=started + _REQUEST_TIMEOUT_SECONDS - 1.0,
    )
    assert settled, "disconnected attempt was not settled by the drop backstop"


def test_escalated_routes_fail_closed_without_hurting_chat(
    engine: _ServingEngine,
) -> None:
    """An escalated admission fails closed while native chat keeps serving.

    The probe uses the ``escalated`` alias, which the driver's host policy
    rejects by name: every route shape resolves natively, so a hosted policy
    is the only construction-independent way left to force escalation for a
    single-deployment chat surface. With no python engine anywhere the
    escalation answers the shared internal error while readiness stays green.
    """
    headers = {"authorization": f"Bearer {engine.raw_key}"}
    escalated = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        json={"model": "escalated", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert escalated.status_code == 500
    assert escalated.json()["error"]["code"] == "internal_error"

    chat = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        json=json.loads(_chat_payload("fast-token")),
        timeout=30.0,
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["choices"][0]["message"]["content"] == "hello world"

    ready = httpx.get(f"{engine.base}/health/ready", timeout=5.0)
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_stalled_reader_cannot_pin_the_gateway_past_the_deadline(
    engine: _ServingEngine,
) -> None:
    """A stalled streaming reader is settled by the request deadline.

    The client opens a streaming chat request against an unbounded flooding
    upstream, reads the first SSE bytes, then stops reading while keeping the
    socket open. A small client receive buffer plus the never-ending stream
    fills the gateway's bounded frame channel, so ``send_bounded`` blocks
    until the
    request deadline and the attempt settles as cancelled even though the
    client never disconnects. The observation window ends before the
    control-plane sweep could close the attempt, so the settlement is the
    data plane's.
    """
    cancelled_before = _terminal_attempts(engine, "cancelled")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
    client.settimeout(10)
    try:
        client.connect((_HOST, engine.port))
        started = time.monotonic()
        client.sendall(_raw_chat_request(engine.raw_key, _chat_payload("flood-token", stream=True)))
        received = b""
        while b"data:" not in received:
            chunk = client.recv(4096)
            assert chunk, "gateway closed the stream before its first SSE frame"
            received += chunk
        # Stop reading entirely; the socket stays open and unread.
        settled = _await_cancelled_attempts(
            engine,
            minimum=cancelled_before + 1,
            deadline=started + _SWEEP_FLOOR_SECONDS - 1.5,
        )
        assert settled, "stalled stream was not settled by the request deadline"
    finally:
        client.close()


def test_heartbeat_keeps_original_provider_phase_timeout(engine: _ServingEngine) -> None:
    """An idle read's 1.5-second limit is not restarted by its one-second heartbeat."""
    started = time.monotonic()
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        content=_chat_payload("quiet-short-phase", stream=True),
        timeout=4,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert ": keepalive" in response.text
    assert "[DONE]" in response.text
    assert elapsed < 2.6
    assert _PROVIDER_CLOSED["quiet-short-phase"].is_set()


def test_keyed_disconnected_owner_stays_deadline_bounded(engine: _ServingEngine) -> None:
    """A silent keyed owner times out once and its retry never dispatches another provider."""
    prompt = "keyed-stall"
    client, request_id, _ = _open_quiet(engine, prompt, keyed=True)
    _abort(client)
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}", "Idempotency-Key": prompt},
        content=_chat_payload(prompt, stream=True),
        timeout=7,
    )
    assert response.status_code == 200
    assert "[DONE]" in response.text
    assert '"error"' in response.text
    assert ": keepalive" not in response.text
    assert _PROVIDER_CALLS[prompt] == 1
    assert _PROVIDER_CLOSED[prompt].is_set()
    assert _attempt(engine, request_id)["state"] == "failed"

"""Failure-path integration tests for the native (Rust) gateway data plane.

These tests exercise three reviewed-but-untested behaviors of the compiled
``exp_gateway_native`` engine against a real serving process:

1. A client disconnect mid non-streaming request settles the admitted attempt
   through the ``AttemptGuard`` drop backstop.
2. A dead python fallback engine degrades only escalated routes; the native
   chat path and readiness stay healthy.
3. A connected client that stops reading a stream cannot pin the gateway past
   the request deadline; ``send_bounded`` settles the attempt.

``exp_gateway_native.serve`` blocks its caller and stops only on SIGINT or
SIGTERM, so one shared serving subprocess (a small generated driver that
composes ``NativeControlPlane`` over a seeded root) hosts every scenario. Its
host policy deliberately escalates Responses requests so the dead-fallback
case exercises the proxy boundary even though Responses is natively supported.
Each test observes settlement deltas through the content-free ``/usage.json``
report, and the subprocess is stopped with SIGTERM at module teardown.
"""

from __future__ import annotations

import json
import os
import signal
import socket
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
        fallback boundary.
        """
        if request.surface == GatewayApiSurface.RESPONSES:
            return False
        return route.snapshot.authorization.alias != "escalated"


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
        control_plane = NativeControlPlane(
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
                            "fallback_port": config["fallback_port"],
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


@dataclass(frozen=True)
class _ServingEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_key: str

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _closed_port() -> int:
    """Return an ephemeral port with no listener behind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((_HOST, 0))
        return probe.getsockname()[1]


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
    upstream, and the serve config points ``fallback_port`` at a closed
    ephemeral port so the dead-fallback scenario needs no extra process. The
    subprocess is stopped with SIGTERM and must exit cleanly.

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
            "fallback_port": _closed_port(),
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
        yield _ServingEngine(port=port, raw_key=raw_key)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


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


def test_dead_fallback_engine_degrades_only_escalated_routes(
    engine: _ServingEngine,
) -> None:
    """A closed fallback port fails escalated routes without hurting chat.

    ``/health/ready`` intentionally reports only the native control plane's
    health and does not cover the fallback host: the CLI owns the embedded
    python engine's thread and its liveness, so a dead fallback degrades the
    escalated surfaces to an explicit 502 while readiness stays green. The
    probe uses the ``escalated`` alias, which the driver's host policy
    rejects by name: every route shape now resolves natively, so a hosted
    policy is the only construction-independent way left to force
    escalation for a single-deployment chat surface.
    """
    headers = {"authorization": f"Bearer {engine.raw_key}"}
    escalated = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers=headers,
        json={"model": "escalated", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert escalated.status_code == 502
    assert escalated.json()["error"]["code"] == "fallback_engine_unavailable"

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

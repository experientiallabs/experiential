"""Upstream-truncation hardening: every abort classifies, the process never dies.

Production evidence (2026-09-03): a fronting proxy on a house lane aborts
with incomplete responses mid-stream ~20 times a day. Every such abort must
end that one request with a classified failure (or serve the partial); the
serving process must survive every truncation offset class. The native h2
abort shapes (RST_STREAM, connection drop) are covered at the relay layer in
``relay.rs``; this matrix drives the h1 chunked wire end to end through a
real serving subprocess so a regression that kills the process fails loudly
here first.
"""

from __future__ import annotations

import json
import signal
import socket
import struct
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.lifecycle_test import _configured_gateway

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"

_DRIVER_SOURCE = textwrap.dedent(
    '''
    """Serve the native gateway engine over one seeded root until SIGTERM."""

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
        control_plane = NativeControlPlane(components, request_timeout_seconds=30.0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sys.stdout.write(json.dumps({"port": port}) + "\\n")
        sys.stdout.flush()
        exp_gateway_native.serve(
            control_plane,
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": port,
                    "max_active_requests": 8,
                    "request_timeout_seconds": 30.0,
                    "graceful_timeout_seconds": 2.0,
                }
            ),
        )


    main()
    '''
)


def _sse_response() -> bytes:
    """One chunked SSE response shaped like real tool-calling chat traffic."""
    frames: list[bytes] = []

    def chunk(delta: dict[str, object], finish: str | None = None) -> bytes:
        payload = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    frames.append(chunk({"role": "assistant", "content": ""}))
    frames.append(chunk({"content": "Let mé check."}))
    frames.append(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": ""},
                    }
                ]
            }
        )
    )
    for piece in ('{"comm', 'and": "ls -la\\u00e9",', ' "cwd": "/tmp"}'):
        frames.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}))
    frames.append(chunk({}, finish="tool_calls"))
    usage_frame = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
    }
    frames.append(f"data: {json.dumps(usage_frame)}\n\n".encode())
    frames.append(b"data: [DONE]\n\n")
    body = b""
    for frame in frames:
        body += f"{len(frame):x}\r\n".encode() + frame + b"\r\n"
    body += b"0\r\n\r\n"
    head = (
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
    )
    return head + body


def _offset_classes(response: bytes) -> dict[str, int]:
    """One representative byte offset per truncation class."""
    head_end = response.index(b"\r\n\r\n") + 4
    first_frame = response.index(b"data: ")
    mid_utf8 = response.index("é".encode()) + 1  # between the two utf8 bytes
    mid_args = response.index(b'"cwd') + 2
    between = response.index(b"\r\n", first_frame + 10) + 2
    return {
        "nothing": 0,
        "mid_status_line": 6,
        "mid_headers": head_end - 9,
        "headers_only": head_end,
        "mid_chunk_length_header": head_end + 1,
        "mid_first_frame": first_frame + 25,
        "mid_utf8_codepoint": mid_utf8,
        "between_frames": between,
        "mid_tool_arguments": mid_args,
        "before_final_chunk": len(response) - 5,
        "complete": len(response),
    }


class _TruncatingUpstream(threading.Thread):
    """Raw-socket chunked SSE upstream truncating at a controlled offset."""

    def __init__(self) -> None:
        """Bind the listener and precompute the full response."""
        super().__init__(daemon=True)
        self.response = _sse_response()
        self.offset = len(self.response)
        self.mode = "fin"
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((_HOST, 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]

    def run(self) -> None:
        """Serve one truncated response per connection until closed."""
        while True:
            try:
                conn, _peer = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        """Drain the request, send the truncated response, close per mode."""
        try:
            conn.settimeout(10)
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _sep, rest = data.partition(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
            while len(rest) < length:
                rest += conn.recv(65536)
            conn.sendall(self.response[: self.offset])
            if self.mode == "rst":
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            conn.close()
        except OSError:
            conn.close()


@pytest.fixture(scope="module", name="setup")
def _setup(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple]:
    """One truncating upstream, one live native engine subprocess."""
    upstream = _TruncatingUpstream()
    upstream.start()
    root = tmp_path_factory.mktemp("truncation-root")
    _manager, raw_key = _configured_gateway(root, base_url=f"http://{_HOST}:{upstream.port}/v1")
    driver = root / "driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    stderr_log = (root / "driver-stderr.log").open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), json.dumps({"root": str(root)})],
        stdout=subprocess.PIPE,
        stderr=stderr_log,
        env={**__import__("os").environ, "TEST_PROVIDER_KEY": "provider-secret-canary"},
        text=True,
    )
    try:
        assert process.stdout is not None
        port = int(json.loads(process.stdout.readline())["port"])
        deadline = time.monotonic() + 30
        while True:
            try:
                live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                if live.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert process.poll() is None, Path(stderr_log.name).read_text()
            assert time.monotonic() < deadline, "engine never became live"
            time.sleep(0.05)
        yield upstream, process, port, raw_key
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        stderr_log.close()
        upstream.listener.close()


def test_every_truncation_class_classifies_and_the_process_survives(setup: tuple) -> None:
    """The hardening invariant: aborts classify, the engine never dies.

    Every truncation offset class, over both close modes and both response
    modes, must end as a classified public failure (or the served partial)
    while the serving process stays alive and live-probing green. A process
    death here is the prod pod-crash class regardless of which offset
    triggered it.
    """
    upstream, process, port, raw_key = setup
    classes = _offset_classes(upstream.response)
    surfaces: tuple[tuple[str, str, dict[str, object]], ...] = (
        (
            "chat-stream",
            "/v1/chat/completions",
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ),
        (
            "chat-nonstream",
            "/v1/chat/completions",
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
            },
        ),
        (
            "responses-stream",
            "/v1/responses",
            {"model": "coding", "input": "hi", "max_output_tokens": 64, "stream": True},
        ),
    )
    outcomes: dict[str, str] = {}
    for mode in ("fin", "rst"):
        for surface, path, body in surfaces:
            for name, offset in classes.items():
                upstream.offset = offset
                upstream.mode = mode
                # No exception tolerance: a client-side transport or protocol
                # error would mean the WORKER severed or corrupted its own
                # response, which is exactly the failure the invariant forbids.
                reply = httpx.post(
                    f"http://{_HOST}:{port}{path}",
                    headers={"authorization": f"Bearer {raw_key}"},
                    json=body,
                    timeout=30.0,
                )
                assert reply.status_code in {200, 502}, (name, mode, surface, reply.text)
                outcomes[f"{mode}:{surface}:{name}"] = str(reply.status_code)
                assert process.poll() is None, (
                    f"the serving process died on truncation class {name!r} "
                    f"(mode={mode}, surface={surface}); outcomes so far: {outcomes}"
                )
    live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=2.0)
    assert live.status_code == 200

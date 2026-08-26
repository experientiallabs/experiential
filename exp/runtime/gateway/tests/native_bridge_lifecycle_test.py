"""Bounded descriptor-lifecycle test against the served native engine.

The bridge pins control-plane callbacks to a fixed pool of worker threads so
the SQLite connections cached per thread by ``persistent_connection`` stay
bounded by the pool size. This test drives the served engine through two
request bursts separated by an idle window longer than tokio's blocking-pool
thread keep-alive (10 seconds), the churn pattern that previously retired
callback threads without releasing their cached connections. The gateway
database descriptor count must not grow across that churn window and must
stay within the fixed pool's bound.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
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

from exp.runtime.gateway.lifecycle_test import _configured_gateway

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
_REQUEST_TIMEOUT_SECONDS = 30.0
_CALLBACK_PERMITS = 4
_BURST_SECONDS = 6.0
_BURST_CLIENT_THREADS = 12
# Longer than tokio's default blocking-thread keep-alive of 10 seconds, so a
# data plane running callbacks on the default blocking pool would retire its
# callback threads between the bursts.
_IDLE_SECONDS = 12.0
# Each cached SQLite connection holds the database plus WAL descriptors, so
# the fixed pool bounds the count at two per worker plus the group-commit
# writer's dedicated connection and the driver main thread's setup cache.
_DESCRIPTOR_CEILING = 2 * (_CALLBACK_PERMITS + 3)

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
                            "max_active_requests": 32,
                            "callback_permits": config["callback_permits"],
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


class _SseUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible SSE mock streaming one short canned completion."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one short completion with usage for any request."""
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        frames = (
            {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        )
        try:
            for frame in frames:
                self.wfile.write(
                    b"data: " + json.dumps(frame, separators=(",", ":")).encode() + b"\n\n"
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


@dataclass(frozen=True)
class _ServingEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_key: str
    pid: int

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _gateway_database_descriptors(pid: int) -> int:
    """Count the serving process's open descriptors on the gateway database.

    Reads ``/proc/<pid>/fd`` where procfs exists and falls back to ``lsof``
    elsewhere, counting every descriptor whose path contains ``gateway.db``
    (the database itself and its WAL sidecars).

    Args:
        pid: Serving subprocess ID.

    Returns:
        Number of matching open descriptors.
    """
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        count = 0
        for entry in proc_fd.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if "gateway.db" in target:
                count += 1
        return count
    lsof = shutil.which("lsof")
    if lsof is None:
        pytest.skip("descriptor counting needs procfs or lsof")
    listing = subprocess.run(  # noqa: S603 - fixed diagnostic binary over our own pid.
        [lsof, "-nP", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(1 for line in listing.stdout.splitlines() if "gateway.db" in line)


def _drive_burst(engine: _ServingEngine, seconds: float) -> int:
    """Serve chat completions from many client threads for a bounded window.

    Args:
        engine: Live serving facts.
        seconds: Wall-clock burst duration.

    Returns:
        Number of successful requests across all client threads.
    """
    deadline = time.monotonic() + seconds
    successes = [0] * _BURST_CLIENT_THREADS
    failures: list[str] = []
    body = {"model": "coding", "messages": [{"role": "user", "content": "fast-token"}]}

    def _client(index: int) -> None:
        """Post sequential chat completions until the shared deadline."""
        with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            while time.monotonic() < deadline:
                response = client.post(
                    f"{engine.base}/v1/chat/completions",
                    headers={"authorization": f"Bearer {engine.raw_key}"},
                    json=body,
                )
                if response.status_code == 200:
                    successes[index] += 1
                else:
                    failures.append(f"{response.status_code}: {response.text[:200]}")
                    return

    threads = [
        threading.Thread(target=_client, args=(index,)) for index in range(_BURST_CLIENT_THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, f"burst requests failed: {failures[:3]}"
    return sum(successes)


@pytest.fixture(name="engine")
def _engine(tmp_path: Path) -> Iterator[_ServingEngine]:
    """Serve one native engine subprocess over a seeded root.

    Yields:
        The live serving facts as a :class:`_ServingEngine`.
    """
    upstream = ThreadingHTTPServer((_HOST, 0), _SseUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    _manager, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1",
    )
    driver = tmp_path / "native_lifecycle_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(tmp_path),
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
            "callback_permits": _CALLBACK_PERMITS,
        }
    )
    stderr_log = tmp_path / "driver-stderr.log"
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
                        break
                except httpx.HTTPError:
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, pid=process.pid)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def test_database_descriptors_stay_bounded_across_blocking_thread_churn(
    engine: _ServingEngine,
) -> None:
    """Descriptor count neither grows across an idle-retire window nor exceeds the pool bound.

    Two bursts bracket an idle window longer than tokio's blocking-thread
    keep-alive. On a data plane whose callbacks run on the default blocking
    pool, the second burst runs on fresh threads and caches fresh SQLite
    connections, so the descriptor count grows; on the fixed bridge pool it
    must be identical, and bounded by two descriptors per pooled connection.
    """
    first_successes = _drive_burst(engine, _BURST_SECONDS)
    assert first_successes > 0
    time.sleep(1.0)
    after_first_burst = _gateway_database_descriptors(engine.pid)
    assert 0 < after_first_burst <= _DESCRIPTOR_CEILING
    time.sleep(_IDLE_SECONDS)
    after_idle = _gateway_database_descriptors(engine.pid)
    second_successes = _drive_burst(engine, _BURST_SECONDS)
    assert second_successes > 0
    time.sleep(1.0)
    after_second_burst = _gateway_database_descriptors(engine.pid)
    assert after_idle <= after_first_burst
    assert after_second_burst <= after_first_burst, (
        f"gateway.db descriptors grew across thread churn: "
        f"{after_first_burst} -> {after_second_burst}"
    )

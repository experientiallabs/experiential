"""Loopback mock upstream and client-side latency sampling helpers."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import (
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement

ALIAS_ID = "latency"
CONNECTION_NAME = "mock-upstream"
CREDENTIAL_ENV = "EXP_LATENCY_MOCK_KEY"
MOCK_MODEL = "latency-mock"
CHAT_PATH = "/v1/chat/completions"
GATEWAY_START_TIMEOUT_S = 30.0
_PAYLOAD_MESSAGE = "hi"


class RequestSample:
    """One completed HTTP sample used only while aggregating an arm."""

    def __init__(self, *, success: bool, latency_ms: float, error: str = "") -> None:
        """Record one request outcome.

        Args:
            success: Whether the response was HTTP 200 with a usable body.
            latency_ms: Client-observed duration in milliseconds.
            error: Display-safe failure reason, empty on success.
        """
        self.success = success
        self.latency_ms = latency_ms
        self.error = error


class MockOpenAIServer:
    """Serve a finite OpenAI-compatible chat completion on loopback."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Bind one threaded mock listener.

        Args:
            host: Loopback bind address.
            port: TCP port, or 0 to let the OS choose.
        """
        self._server = ThreadingHTTPServer((host, port), _MockChatHandler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible base URL including ``/v1``."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        """Start the listener thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener and join its thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _MockChatHandler(BaseHTTPRequestHandler):
    """Answer chat-completion POSTs with a tiny JSON or SSE body."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        """Return one non-streaming completion or one immediate SSE stream."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        model = str(payload.get("model") or MOCK_MODEL)
        created = int(time.time())
        if payload.get("stream"):
            self._write_stream(model=model, created=created)
            return
        body = json.dumps(_completion_payload(model=model, created=created)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Omit request logs so the report cannot retain payload context."""
        del format, args

    def _write_stream(self, *, model: str, created: int) -> None:
        """Emit role, content, finish, and [DONE] SSE frames immediately.

        Args:
            model: Model name echoed in each chunk.
            created: Unix timestamp shared by the stream.
        """
        frames = (
            _sse_chunk(model, created, {"role": "assistant"}),
            _sse_chunk(model, created, {"content": "hello"}),
            _sse_chunk(model, created, {}, finish_reason="stop"),
            b"data: [DONE]\n\n",
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(sum(len(frame) for frame in frames)))
        self.end_headers()
        try:
            for frame in frames:
                self.wfile.write(frame)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def _completion_payload(*, model: str, created: int) -> JsonObject:
    """Return one tiny non-streaming OpenAI chat completion body.

    Args:
        model: Model name echoed to the client.
        created: Unix timestamp for the completion.

    Returns:
        JSON-serializable completion object.
    """
    return {
        "id": "chatcmpl-latency",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _sse_chunk(
    model: str,
    created: int,
    delta: dict[str, str],
    *,
    finish_reason: str | None = None,
) -> bytes:
    """Encode one compact chat-completion SSE data frame.

    Args:
        model: Model name echoed in the chunk.
        created: Unix timestamp shared by the stream.
        delta: Choice delta object.
        finish_reason: Optional terminal finish reason.

    Returns:
        SSE ``data:`` frame bytes.
    """
    event = {
        "id": "chatcmpl-latency",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


def chat_payload(model: str, *, stream: bool) -> JsonObject:
    """Return the fixed chat-completion body used by every request in one arm.

    Args:
        model: Gateway alias or mock model name.
        stream: Whether the request asks for SSE.

    Returns:
        JSON-serializable request body.
    """
    payload: JsonObject = {
        "model": model,
        "messages": [{"role": "user", "content": _PAYLOAD_MESSAGE}],
        "max_tokens": 1,
    }
    if stream:
        payload["stream"] = True
    return payload


def unused_loopback_port() -> int:
    """Reserve and release one currently unused loopback TCP port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def percentile(values: tuple[float, ...], pct: float) -> float:
    """Return a nearest-rank percentile from the sorted samples.

    Args:
        values: Observed samples. Empty input returns 0.0.
        pct: Percentile in ``[0, 100]``.

    Returns:
        Selected sample, or 0.0 when ``values`` is empty.

    Raises:
        ValueError: ``pct`` is outside ``[0, 100]``.
    """
    if not 0.0 <= pct <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct / 100.0), len(ordered) - 1)
    return ordered[index]


def collect_arm_samples(
    *,
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    warmup: int,
    requests: int,
    concurrency: int,
    timeout_s: float,
    stream: bool,
) -> tuple[tuple[RequestSample, ...], float]:
    """Warm up, then collect timed samples for one sequential arm.

    Args:
        url: Absolute chat-completions URL.
        headers: Authorization and content-type headers.
        payload: Shared JSON body.
        warmup: Discarded requests run before the timed window.
        requests: Timed request count.
        concurrency: Maximum in-flight requests.
        timeout_s: Per-request timeout.
        stream: Whether to stop at the first content token.

    Returns:
        Timed samples and the measured wall time in seconds.
    """
    timeout = httpx.Timeout(timeout_s)
    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 8),
        max_keepalive_connections=max(concurrency, 4),
    )
    with httpx.Client(timeout=timeout, limits=limits) as client:
        if warmup > 0:
            _run_pool(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                count=warmup,
                concurrency=min(concurrency, warmup),
                stream=stream,
            )
            time.sleep(0.1)
        wall_start = time.perf_counter()
        samples = _run_pool(
            client=client,
            url=url,
            headers=headers,
            payload=payload,
            count=requests,
            concurrency=min(concurrency, requests),
            stream=stream,
        )
        wall_time_s = time.perf_counter() - wall_start
    return samples, wall_time_s


def _run_pool(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    count: int,
    concurrency: int,
    stream: bool,
) -> tuple[RequestSample, ...]:
    """Execute ``count`` requests with a bounded thread pool.

    Args:
        client: Shared HTTP client.
        url: Absolute chat-completions URL.
        headers: Request headers.
        payload: Shared JSON body.
        count: Number of requests to issue.
        concurrency: Maximum in-flight requests.
        stream: Whether to measure time to first content token.

    Returns:
        Completed samples in completion order.
    """
    samples: list[RequestSample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, client, url, headers, payload, stream) for _ in range(count)
        ]
        for future in as_completed(futures):
            samples.append(future.result())
    return tuple(samples)


def _one_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    stream: bool,
) -> RequestSample:
    """Issue one chat request and return its client-observed sample.

    Args:
        client: Shared HTTP client.
        url: Absolute chat-completions URL.
        headers: Request headers.
        payload: Shared JSON body.
        stream: Whether to measure time to first content token.

    Returns:
        Success or failure sample.
    """
    start = time.perf_counter()
    try:
        if stream:
            return _one_stream(client, url, headers, payload, start)
        response = client.post(url, headers=headers, json=payload)
        body = response.content
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code != 200:
            return RequestSample(
                success=False,
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}: {body[:120]!r}",
            )
        json.loads(body)
        return RequestSample(success=True, latency_ms=latency_ms)
    except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError) as exc:
        return RequestSample(
            success=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=str(exc)[:200],
        )


def _one_stream(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    start: float,
) -> RequestSample:
    """Read one SSE response until the first content token.

    Args:
        client: Shared HTTP client.
        url: Absolute chat-completions URL.
        headers: Request headers.
        payload: Shared JSON body with ``stream`` set.
        start: ``perf_counter`` value taken before the request.

    Returns:
        Success at the first content token, or a failure sample.
    """
    with client.stream("POST", url, headers=headers, json=payload) as response:
        if response.status_code != 200:
            body = response.read()
            return RequestSample(
                success=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"HTTP {response.status_code}: {body[:120]!r}",
            )
        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            event_payload = line[5:].strip()
            if event_payload == "[DONE]":
                break
            event = json.loads(event_payload)
            if not isinstance(event, dict):
                continue
            raw_choices = event.get("choices") or [{}]
            if not isinstance(raw_choices, list) or not raw_choices:
                continue
            choice = raw_choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content") or choice.get("text")
            if content:
                return RequestSample(
                    success=True,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
    return RequestSample(
        success=False,
        latency_ms=(time.perf_counter() - start) * 1000,
        error="stream ended before a content token",
    )


def configure_gateway(root: Path, *, provider_base_url: str) -> str:
    """Author one direct alias pointed at the local mock and issue a virtual key.

    Args:
        root: Temporary EXP root for this report.
        provider_base_url: Mock OpenAI-compatible base URL.

    Returns:
        Newly issued raw virtual key. The caller must not persist it.
    """
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name=CONNECTION_NAME,
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url=provider_base_url,
            api_key_env=CREDENTIAL_ENV,
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=ALIAS_ID,
        connection_name=CONNECTION_NAME,
        provider_model=MOCK_MODEL,
        exact_model_id="latency-mock-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(
            input_micro_usd_per_million_tokens=0,
            output_micro_usd_per_million_tokens=0,
        ),
        pricing_source="latency-report-mock",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id=ALIAS_ID,
        alias_name=ALIAS_ID,
        revision_id="latency-revision",
        pool_id=ALIAS_ID,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="bench", display_name="Latency bench")
    manager.add_grant(identity_id="bench", alias_id=ALIAS_ID)
    issued = manager.issue_key(identity_id="bench", key_id="bench-key")
    return issued.raw_key


def start_gateway_process(
    *,
    root: Path,
    port: int,
    credential: str,
) -> tuple[subprocess.Popen[str], str]:
    """Launch the product gateway on loopback and wait until it is ready.

    Args:
        root: Configured EXP root.
        port: Loopback TCP port.
        credential: Mock upstream credential placed in the child environment.

    Returns:
        Live process and the resolved data-plane engine name.

    Raises:
        RuntimeError: The process exits or does not become ready in time.
    """
    env = os.environ.copy()
    env.update(
        {
            CREDENTIAL_ENV: credential,
            "EXP_TELEMETRY": "0",
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    executable = Path(sys.executable).with_name("exp")
    process = subprocess.Popen(
        [
            str(executable),
            "--root",
            str(root),
            "--port",
            str(port),
            "--non-interactive",
            "--json",
            "--engine",
            "auto",
            "--graceful-timeout",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: list[str] = []
    pump = threading.Thread(
        target=_pump_process_output,
        args=(process, lines),
        daemon=True,
    )
    pump.start()
    try:
        wait_for_http_ok(
            f"http://127.0.0.1:{port}/health/ready",
            process=process,
            lines=lines,
            timeout_s=GATEWAY_START_TIMEOUT_S,
            label="gateway",
        )
        engine = _wait_for_engine_receipt(lines)
    except Exception:
        stop_gateway_process(process)
        raise
    return process, engine


def stop_gateway_process(process: subprocess.Popen[str]) -> None:
    """Signal the gateway and wait for a bounded exit.

    Args:
        process: Live or already terminated gateway process.
    """
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _pump_process_output(process: subprocess.Popen[str], lines: list[str]) -> None:
    """Drain gateway stdout so the pipe cannot fill.

    Args:
        process: Gateway subprocess with a captured stdout pipe.
        lines: Destination list for complete output lines.
    """
    if process.stdout is None:
        return
    for line in process.stdout:
        lines.append(line)


def wait_for_http_ok(
    url: str,
    *,
    process: subprocess.Popen[str],
    lines: list[str],
    timeout_s: float,
    label: str,
) -> None:
    """Wait until ``url`` returns HTTP 200 or fail boundedly.

    Args:
        url: Absolute health URL.
        process: Server process that must remain live.
        lines: Captured stdout lines from the pump thread.
        timeout_s: Bound for the readiness wait.
        label: Process name used in error messages.

    Raises:
        RuntimeError: The process exits or does not become ready in time.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = "".join(lines)
            raise RuntimeError(f"{label} exited before readiness: {output}")
        try:
            response = httpx.get(url, timeout=0.2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise RuntimeError(f"{label} did not become ready: {url}")


def _wait_for_engine_receipt(lines: list[str], *, timeout_s: float = 2.0) -> str:
    """Wait until the launch receipt is pumped, then return its engine.

    HTTP readiness can succeed before the stdout pump thread appends the JSON
    receipt. The rust engine prints ``engine: rust`` before it binds; the python
    engine prints a ready receipt without an engine field.

    Args:
        lines: Captured gateway stdout lines from the pump thread.
        timeout_s: Bound for waiting on the receipt after readiness.

    Returns:
        ``rust`` or ``python`` when a ready receipt is pumped, otherwise
        ``unknown``. An unobserved receipt is never guessed as python.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        engine = _engine_from_output(lines)
        if engine is not None:
            return engine
        time.sleep(0.02)
    return _engine_from_output(lines) or "unknown"


def _engine_from_output(lines: list[str]) -> str | None:
    """Read the launch receipt engine when the JSON line is already pumped.

    Args:
        lines: Captured gateway stdout lines.

    Returns:
        ``rust`` or ``python`` when a ready receipt is present, otherwise
        ``None`` so callers can wait for the pump thread.
    """
    for line in lines:
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("status") != "ready":
            continue
        engine = payload.get("engine")
        if engine == "rust":
            return "rust"
        return "python"
    return None

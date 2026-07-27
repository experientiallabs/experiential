"""Tests for the remote LLMLingua-2 compressor client.

The HTTP behavior is exercised against a real throwaway server on localhost (so status codes,
headers, and JSON bodies go over a real socket) rather than a mocked client object. Transport
faults, which a real server cannot produce on demand, use an httpx mock transport. Nothing here
touches the H100 endpoint; the one live test is skipped unless the endpoint env vars are set.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from wmo.optimize.compression_endpoint import (
    KEY_ENV,
    URL_ENV,
    CompressionConfig,
    CompressorEndpointError,
    LLMLingua2EndpointCompressor,
    estimate_tokens,
)

SEGMENTS = ["the quarterly revenue report shows growth", "tool output: {'ok': true}"]


class FakeEndpointState:
    """What the fake server should do next, and what it saw."""

    def __init__(self) -> None:
        self.status = 200
        self.body: dict[str, object] = {
            "segments": ["revenue report growth", "output {'ok': true}"],
            "tokens_in": 20,
            "tokens_out": 9,
            "latency_ms": 12.5,
            "cost_usd": 0.000034,
            "compressor_version": "llmlingua2-fixed-absolute-threshold-fp32/1",
            "model_fingerprint": "9a9d3f98bfb65abc",
        }
        self.requests: list[dict[str, object]] = []
        self.auth_headers: list[str] = []


def _make_handler(state: FakeEndpointState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Silence the stdlib server's stderr logging during tests."""

        def _respond(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            if status == 429:
                self.send_header("Retry-After", "7")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            self._respond(200, {"status": "ok", "model_fingerprint": "9a9d3f98bfb65abc"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            state.requests.append(json.loads(self.rfile.read(length)))
            state.auth_headers.append(self.headers.get("Authorization", ""))
            if state.status != 200:
                self._respond(state.status, {"detail": "nope"})
                return
            self._respond(200, state.body)

    return Handler


@pytest.fixture
def endpoint() -> Iterator[tuple[LLMLingua2EndpointCompressor, FakeEndpointState]]:
    """A client pointed at a real localhost HTTP server standing in for the box."""
    state = FakeEndpointState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    client = LLMLingua2EndpointCompressor(f"http://{host}:{port}", "test-token")
    try:
        yield client, state
    finally:
        server.shutdown()
        server.server_close()


def test_compress_returns_segments_and_propagates_metering(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A successful call returns the endpoint's segments and carries its cost into the result."""
    client, state = endpoint
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=0.5)

    result = client.compress(SEGMENTS, config)

    assert result.segments == ["revenue report growth", "output {'ok': true}"]
    assert result.cost_usd == pytest.approx(0.000034)
    assert result.latency_s > 0
    # Seam-proxy counts, so this compressor is comparable with identity/truncate in one grid.
    assert result.tokens_in_raw == sum(estimate_tokens(segment) for segment in SEGMENTS)
    assert result.tokens_in_compressed < result.tokens_in_raw
    assert state.requests[0]["threshold"] == pytest.approx(0.5)
    assert state.requests[0]["segments"] == SEGMENTS


def test_bearer_token_is_sent(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """The API key travels as a bearer token, which is the only thing the endpoint accepts."""
    client, state = endpoint
    client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    assert state.auth_headers == ["Bearer test-token"]


def test_zero_aggressiveness_is_a_bit_for_bit_no_op_without_a_call(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """The seam requires 0.0 to be a no-op; it should not even reach the network."""
    client, state = endpoint
    result = client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.0))

    assert result.segments == SEGMENTS
    assert result.tokens_in_raw == result.tokens_in_compressed
    assert result.cost_usd == 0.0
    assert state.requests == []


def test_empty_segments_short_circuit(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Nothing to compress means no request and no cost."""
    client, state = endpoint
    result = client.compress([], CompressionConfig(compressor_id="x", aggressiveness=0.5))
    assert result.segments == []
    assert state.requests == []


def test_rejected_token_raises_naming_the_key_variable(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A 401 must tell the caller which environment variable to fix."""
    client, state = endpoint
    state.status = 401
    with pytest.raises(CompressorEndpointError, match=KEY_ENV):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    # A rejected token will not fix itself, so it is not retried.
    assert len(state.requests) == 1


def test_rate_limit_raises_and_is_not_retried(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A 429 surfaces the retry hint; retrying it would only deepen the limit."""
    client, state = endpoint
    state.status = 429
    with pytest.raises(CompressorEndpointError, match="rate limiting"):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    assert len(state.requests) == 1


def test_server_error_raises_with_the_status(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A 500 is reported verbatim rather than swallowed."""
    client, state = endpoint
    state.status = 500
    with pytest.raises(CompressorEndpointError, match="HTTP 500"):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))


def test_segment_count_mismatch_is_rejected(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A compressor may not merge, split, or reorder segments; a bad shape is a hard error."""
    client, state = endpoint
    state.body = dict(state.body, segments=["only one"])
    with pytest.raises(CompressorEndpointError, match="segments for 2 inputs"):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))


def test_unreachable_endpoint_raises_instead_of_serving_uncompressed() -> None:
    """A down endpoint is an error, never a silent pass-through of the raw input."""
    # Port 1 on loopback refuses immediately: a real transport failure, no server involved.
    client = LLMLingua2EndpointCompressor("http://127.0.0.1:1", "test-token", timeout_s=2.0)
    with pytest.raises(CompressorEndpointError) as caught:
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    message = str(caught.value)
    assert URL_ENV in message
    assert KEY_ENV in message
    assert "never silently falls back" in message


def test_transport_fault_is_retried_once_then_succeeds() -> None:
    """One retry covers a dropped connection; the second attempt's result is returned."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(
            200,
            json={
                "segments": ["a", "b"],
                "tokens_in": 4,
                "tokens_out": 2,
                "latency_ms": 1.0,
                "cost_usd": 0.000001,
            },
        )

    client = LLMLingua2EndpointCompressor(
        "http://endpoint.invalid",
        "test-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))

    assert result.segments == ["a", "b"]
    assert len(attempts) == 2


def test_transport_fault_gives_up_after_one_retry() -> None:
    """Two transport failures is a real outage, not a blip; it must not loop."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("down", request=request)

    client = LLMLingua2EndpointCompressor(
        "http://endpoint.invalid",
        "test-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(CompressorEndpointError):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    assert len(attempts) == 2


def test_from_env_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured endpoint says exactly which variables to set and what the escape is."""
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(CompressorEndpointError) as caught:
        LLMLingua2EndpointCompressor.from_env()
    message = str(caught.value)
    assert URL_ENV in message
    assert KEY_ENV in message
    assert "identity" in message


def test_from_env_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both variables set, from_env yields a client pointed at the configured URL."""
    monkeypatch.setenv(URL_ENV, "https://40.80.93.150:8443/")
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    client = LLMLingua2EndpointCompressor.from_env()
    assert client.base_url == "https://40.80.93.150:8443"


def test_health_reports_the_endpoint_identity(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Health needs no auth and identifies which weights are serving."""
    client, _ = endpoint
    assert client.health()["model_fingerprint"] == "9a9d3f98bfb65abc"


@pytest.mark.skipif(
    not (os.environ.get(URL_ENV) and os.environ.get(KEY_ENV)),
    reason=f"live endpoint test: set {URL_ENV} and {KEY_ENV} to run it",
)
def test_live_endpoint_compresses_deterministically() -> None:
    """Opt-in: hit the real H100 endpoint and check it compresses and repeats itself exactly.

    Determinism is the property the whole serving story rests on, so the live check verifies it
    end to end rather than trusting the server's own startup self-test.
    """
    client = LLMLingua2EndpointCompressor.from_env()
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=0.5)
    long_segments = [
        "The quarterly revenue report shows that total revenue increased by 12 percent "
        "year over year, driven primarily by strong performance in the enterprise segment."
    ] * 4

    first = client.compress(long_segments, config)
    second = client.compress(long_segments, config)

    assert first.segments == second.segments
    assert first.tokens_in_compressed < first.tokens_in_raw
    assert first.cost_usd > 0

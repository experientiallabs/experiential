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

import certifi
import httpx
import pytest
from pydantic import BaseModel

from wmo.optimize import compression, compression_endpoint
from wmo.optimize.compression import (
    CompressionConfig,
    CompressionResult,
    Compressor,
    estimate_tokens,
    get_compressor,
    segment_batch_limit,
    servable_compressor,
)
from wmo.optimize.compression_endpoint import (
    CA_ENV,
    KEY_ENV,
    REQUIRED_SELECTION_RULE,
    URL_ENV,
    CompressorEndpointError,
    LLMLingua2EndpointCompressor,
    register_endpoint_compressor,
)

SEGMENTS = ["the quarterly revenue report shows growth", "tool output: {'ok': true}"]


class SeenRequest(BaseModel):
    """One request the fake server received, parsed so assertions read typed fields."""

    segments: list[str]
    threshold: float


class FakeEndpointState:
    """What the fake server should do next, and what it saw."""

    def __init__(self) -> None:
        self.status = 200
        self.body: dict[str, object] = {
            "segments": ["revenue report growth", "output {'ok': true}"],
            "tokens_in": 20,
            "tokens_out": 9,
            "latency_ms": 12.5,
            "queue_ms": 0.4,
            "cost_usd": 0.000034,
            "compressor_version": "llmlingua2-fixed-absolute-threshold-fp32/1",
            "model_fingerprint": "9a9d3f98bfb65abc",
        }
        self.requests: list[SeenRequest] = []
        self.auth_headers: list[str] = []
        self.selection_rule = "fixed-absolute-threshold"
        self.max_segments = 1024
        # When set, the fake echoes back one output segment per input instead of `body`, which
        # is what the batch-splitting tests need.
        self.echo = False


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
            self._respond(
                200,
                {
                    "status": "ok",
                    "model_fingerprint": "9a9d3f98bfb65abc",
                    "selection_rule": state.selection_rule,
                    "max_segments": state.max_segments,
                    "max_chars": 8000000,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state.requests.append(SeenRequest.model_validate(body))
            state.auth_headers.append(self.headers.get("Authorization", ""))
            if state.status != 200:
                self._respond(state.status, {"detail": "nope"})
                return
            if state.echo:
                segments = [f"c:{segment}" for segment in body["segments"]]
                self._respond(200, dict(state.body, segments=segments))
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
    # Injected client: the fake is plain HTTP on loopback, so there is no certificate to
    # pin. Building one without a cert is refused, which is its own test below.
    client = LLMLingua2EndpointCompressor(
        f"http://{host}:{port}", "test-token", client=httpx.Client(timeout=10)
    )
    try:
        yield client, state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def pinned_cert() -> str:
    """A real PEM on disk to point `WMO_COMPRESSOR_CA` at.

    The repository ships no certificate (the box's is an operational secret), and
    `ssl.create_default_context(cafile=...)` rejects anything that is not a parseable PEM, so
    these tests borrow certifi's bundle. What is under test is that the client PINS whatever
    file it is given — the fake endpoint is plain HTTP on loopback, so the pinned trust store
    is built but never exercised on the wire.
    """
    return certifi.where()


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot and restore the seam's process-global registries around a test.

    Both of them: importing this module registers a factory for the real id, so a test that
    resolves or registers must not leave a constructed instance behind for the next one.
    """
    saved = dict(compression._COMPRESSORS)
    saved_factories = dict(compression._COMPRESSOR_FACTORIES)
    try:
        yield
    finally:
        compression._COMPRESSORS.clear()
        compression._COMPRESSORS.update(saved)
        compression._COMPRESSOR_FACTORIES.clear()
        compression._COMPRESSOR_FACTORIES.update(saved_factories)


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
    assert state.requests[0].threshold == pytest.approx(0.5)
    assert state.requests[0].segments == SEGMENTS


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
    client = LLMLingua2EndpointCompressor(
        "http://127.0.0.1:1", "test-token", client=httpx.Client(timeout=2.0)
    )
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


def test_building_without_a_pinned_certificate_is_refused() -> None:
    """No certificate and no injected client means no client at all, never the public CA store.

    The endpoint is self-signed by design. Falling back to the system trust store would not be
    a weaker pin, it would be trusting every public CA against a box none of them has vouched
    for, which is the shape a MITM needs.
    """
    with pytest.raises(CompressorEndpointError, match=CA_ENV):
        LLMLingua2EndpointCompressor("https://40.80.93.150:8443", "x" * 64)


def test_unset_ca_says_what_to_set_and_where_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """No certificate ships with the repo, so the error names the variable AND how to fetch it."""
    monkeypatch.setenv(URL_ENV, "https://40.80.93.150:8443")
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.delenv(CA_ENV, raising=False)

    with pytest.raises(CompressorEndpointError) as caught:
        LLMLingua2EndpointCompressor.from_env()
    message = str(caught.value)
    assert CA_ENV in message
    assert compression_endpoint.CERT_FETCH_HINT in message
    assert "public CA" in message


def test_ca_pointing_at_a_missing_file_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the path must not degrade into an unpinned client."""
    monkeypatch.setenv(URL_ENV, "https://40.80.93.150:8443")
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, "/nonexistent/compressor-cert.pem")

    with pytest.raises(CompressorEndpointError, match="not a file"):
        LLMLingua2EndpointCompressor.from_env()


def test_from_env_pins_the_configured_certificate(
    monkeypatch: pytest.MonkeyPatch, pinned_cert: str
) -> None:
    """`WMO_COMPRESSOR_CA` is the one and only source of the pin, and a real file satisfies it."""
    monkeypatch.setenv(URL_ENV, "https://40.80.93.150:8443")
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)
    assert LLMLingua2EndpointCompressor.from_env().base_url == "https://40.80.93.150:8443"


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


def test_from_env_builds_a_client(monkeypatch: pytest.MonkeyPatch, pinned_cert: str) -> None:
    """With all three variables set, from_env yields a client pointed at the configured URL."""
    monkeypatch.setenv(URL_ENV, "https://40.80.93.150:8443/")
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)
    client = LLMLingua2EndpointCompressor.from_env()
    assert client.base_url == "https://40.80.93.150:8443"


def test_satisfies_the_seam_protocol(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """The client is a Compressor as the seam defines it, attesting append stability."""
    client, _ = endpoint
    assert isinstance(client, Compressor)
    assert client.append_stable is True
    assert client.id == "llmlingua2-endpoint"


def test_compress_returns_the_seam_result_type(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Accounting comes back in the seam's own model, not a look-alike."""
    client, _ = endpoint
    result = client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    assert isinstance(result, CompressionResult)


def test_importing_registers_a_factory_and_builds_nothing() -> None:
    """Import must cost nothing: a factory for the id, no credentials read, no network."""
    assert "llmlingua2-endpoint" in compression._COMPRESSOR_FACTORIES
    assert "llmlingua2-endpoint" not in compression._COMPRESSORS


@pytest.mark.usefixtures("clean_registry")
def test_first_resolution_builds_through_the_factory(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
    monkeypatch: pytest.MonkeyPatch,
    pinned_cert: str,
) -> None:
    """`route fit --compressor llmlingua2-endpoint` needs the id to resolve from env vars alone."""
    client, state = endpoint
    state.max_segments = 512
    monkeypatch.setenv(URL_ENV, client.base_url)
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)

    built = get_compressor("llmlingua2-endpoint")

    assert built.id == "llmlingua2-endpoint"
    assert built.append_stable is True
    # The handshake ran during construction, so the seam's chunking cap is the box's.
    assert segment_batch_limit(built) == 512
    # Resolution is cached, so a grid does not re-handshake per lookup.
    assert get_compressor("llmlingua2-endpoint") is built


@pytest.mark.usefixtures("clean_registry")
def test_factory_failure_is_actionable_and_retryable(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
    monkeypatch: pytest.MonkeyPatch,
    pinned_cert: str,
) -> None:
    """A down endpoint must not poison the id for the life of the process.

    The seam wraps the factory's failure in ValueError so the CLI renders it as a usage error
    instead of a traceback; what matters here is that the guidance survives the wrapping and
    that fixing the environment makes the next resolution succeed.
    """
    client, _ = endpoint
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)

    with pytest.raises(ValueError) as caught:
        get_compressor("llmlingua2-endpoint")
    assert URL_ENV in str(caught.value)
    assert KEY_ENV in str(caught.value)
    assert isinstance(caught.value.__cause__, CompressorEndpointError)

    monkeypatch.setenv(URL_ENV, client.base_url)
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)
    assert get_compressor("llmlingua2-endpoint").id == "llmlingua2-endpoint"


@pytest.mark.usefixtures("clean_registry")
def test_registration_publishes_the_compressor_and_it_is_servable(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
    monkeypatch: pytest.MonkeyPatch,
    pinned_cert: str,
) -> None:
    """After registering, a policy can name the id and the v1 serving gate admits it."""
    client, _ = endpoint
    monkeypatch.setenv(URL_ENV, client.base_url)
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)

    registered = register_endpoint_compressor()

    assert get_compressor("llmlingua2-endpoint") is registered
    config = CompressionConfig(compressor_id="llmlingua2-endpoint", aggressiveness=0.5)
    assert servable_compressor(config) is registered


@pytest.mark.usefixtures("clean_registry")
def test_registration_refuses_a_server_running_another_selection_rule(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
    monkeypatch: pytest.MonkeyPatch,
    pinned_cert: str,
) -> None:
    """append_stable is only true for absolute-threshold selection, so it is verified live.

    A box redeployed with percentile selection must be refused at registration rather than
    silently admitted to serving, where it would churn the cached prefix every turn.
    """
    client, state = endpoint
    state.selection_rule = "per-input-percentile"
    monkeypatch.setenv(URL_ENV, client.base_url)
    monkeypatch.setenv(KEY_ENV, "x" * 64)
    monkeypatch.setenv(CA_ENV, pinned_cert)

    with pytest.raises(CompressorEndpointError) as caught:
        register_endpoint_compressor()
    assert "per-input-percentile" in str(caught.value)
    assert REQUIRED_SELECTION_RULE in str(caught.value)
    # Nothing was registered, so the id still resolves through the factory (which will fail the
    # same way against this box) rather than handing back an unverified compressor.
    assert "llmlingua2-endpoint" not in compression._COMPRESSORS


def test_handshake_adopts_the_boxs_published_caps(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """The chunking limit comes from the running box, not a constant that can drift from it."""
    client, state = endpoint
    state.max_segments = 64

    client.handshake()

    assert client.max_segments_per_call == 64
    state.echo = True
    client.compress(
        [f"s{index}" for index in range(150)],
        CompressionConfig(compressor_id="x", aggressiveness=0.5),
    )
    assert [len(request.segments) for request in state.requests] == [64, 64, 22]


def test_queue_time_is_reported_but_not_billed(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """cost_usd carries the endpoint's GPU time only; queueing is metering, not billing."""
    client, _ = endpoint
    result = client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    # The fake reports latency_ms 12.5, queue_ms 0.4, cost 0.000034: cost tracks compute alone,
    # so it must not have grown by the queue component.
    assert result.cost_usd == pytest.approx(0.000034)


def test_bank_fit_batch_stays_one_round_trip(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """An 800-scenario bank fit arrives as one call and must not be split (seam expectation)."""
    client, state = endpoint
    state.echo = True
    scenarios = [f"scenario {index} text" for index in range(800)]

    result = client.compress(scenarios, CompressionConfig(compressor_id="x", aggressiveness=0.5))

    assert len(state.requests) == 1
    assert result.segments == [f"c:{scenario}" for scenario in scenarios]


def test_oversized_batches_split_on_fixed_boundaries_preserving_order(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Beyond the per-request cap the client splits rather than letting the box 413 it."""
    client, state = endpoint
    client._max_segments = 100  # exercise the split without a 2000-item payload
    state.echo = True
    scenarios = [f"scenario {index}" for index in range(250)]

    result = client.compress(scenarios, CompressionConfig(compressor_id="x", aggressiveness=0.5))

    assert [len(request.segments) for request in state.requests] == [100, 100, 50]
    assert result.segments == [f"c:{scenario}" for scenario in scenarios]
    # Cost is summed across the split, not taken from the last response.
    assert result.cost_usd == pytest.approx(0.000034 * 3)


def test_split_boundaries_do_not_depend_on_later_segments(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Append stability: adding a segment must not move an earlier segment into another batch."""
    client, state = endpoint
    client._max_segments = 10
    state.echo = True
    base = [f"s{index}" for index in range(25)]

    client.compress(base, CompressionConfig(compressor_id="x", aggressiveness=0.5))
    first_pass = [request.segments for request in state.requests]
    state.requests.clear()
    client.compress(base + ["appended"], CompressionConfig(compressor_id="x", aggressiveness=0.5))
    second_pass = [request.segments for request in state.requests]

    # The full batches are byte-identical; only the trailing partial batch grew.
    assert second_pass[:2] == first_pass[:2]


def test_payload_too_large_names_both_sides_of_the_cap(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """A 413 means client and box disagree on limits, so say which knob to move."""
    client, state = endpoint
    state.status = 413
    with pytest.raises(CompressorEndpointError, match="WMO_COMPRESSOR_MAX_SEGMENTS"):
        client.compress(SEGMENTS, CompressionConfig(compressor_id="x", aggressiveness=0.5))


def test_health_reports_the_endpoint_identity(
    endpoint: tuple[LLMLingua2EndpointCompressor, FakeEndpointState],
) -> None:
    """Health needs no auth and identifies which weights are serving."""
    client, _ = endpoint
    assert client.health()["model_fingerprint"] == "9a9d3f98bfb65abc"


@pytest.mark.skipif(
    not (os.environ.get(URL_ENV) and os.environ.get(KEY_ENV) and os.environ.get(CA_ENV)),
    reason=f"live endpoint test: set {URL_ENV}, {KEY_ENV} and {CA_ENV} to run it",
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

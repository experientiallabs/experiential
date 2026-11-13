"""Drive native input moderation, response release, and durable records over real sockets."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, Protocol

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, GatewayTokenPrices, ModelCapabilities
from exp.common.sqlite.native import connect as connect_sqlite
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import CaptureController
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.mandatory_guardrails_test import _tool_chunk
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames


class _Endpoint:
    """Hold a real HTTP response behind an event so each race has explicit ordering."""

    def __init__(
        self, body: bytes, *, content_type: str, status: int = 200, tail: bytes = b""
    ) -> None:
        """Start a loopback endpoint with a finite fixture lifetime."""
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.tail_release = threading.Event()
        self.requests: list[bytes] = []
        self.intercept: Callable[[BaseHTTPRequestHandler], bool] | None = None
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            """Serve exactly the configured response after its release event."""

            def do_POST(self) -> None:  # noqa: N802 - standard HTTP handler.
                """Read the request before exposing the dispatch event to the test."""
                if endpoint.intercept is not None and endpoint.intercept(self):
                    return
                endpoint.requests.append(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.started.set()
                if not endpoint.release.wait(8):
                    return
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body) + len(tail)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                    if tail and endpoint.tail_release.wait(8):
                        self.wfile.write(tail)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    endpoint.finished.set()

            def log_message(self, format: str, *args: object) -> None:
                """Keep synthetic provider payloads out of diagnostics."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        """Release every handler and join the fixture server."""
        self.release.set()
        self.tail_release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()


class _ClassifierEndpoint(Protocol):
    """The classifier needs only the immutable HTTP endpoint location."""

    url: str


class _Classifier:
    """Export the HTTP classifier contract; Python cannot execute an inspection."""

    def __init__(self, endpoint: _ClassifierEndpoint) -> None:
        """Bind the loopback detector endpoint."""
        self.endpoint = endpoint

    def native_input_configuration(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> JsonObject:
        """Freeze the complete request and transport once at admission."""
        return {
            "kind": "http_json",
            "url": self.endpoint.url,
            "max_response_bytes": 1024,
            "payload": {"request": request.model_dump(mode="json"), "check_id": check.check_id},
            "transport": {"public_only": False, "headers": {}},
        }

    def native_input_template(self, *, check: GuardrailCheck) -> JsonObject:
        """Let Rust rebuild the expanded subject after a gateway tool round."""
        return {
            "kind": "http_json",
            "url": self.endpoint.url,
            "max_response_bytes": 1024,
            "transport": {"public_only": False, "headers": {}},
        }

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Reject a Python inspection fallback."""
        raise AssertionError("input inspection must execute in Rust")

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Reject an output call from this input-only policy."""
        raise AssertionError("input-only policy cannot inspect output")


class _NativeOnlyControl(NativeControlPlane):
    """Fail the request if native serving calls a post-admission Python method."""

    def start_attempt(self, argument: str) -> str:
        """Reject a Python reservation callback."""
        raise AssertionError("reservation must execute in Rust")

    def settle(self, argument: str) -> str:
        """Reject a Python settlement callback."""
        raise AssertionError("settlement must execute in Rust")

    def price(self, argument: str) -> str:
        """Reject a Python price callback."""
        raise AssertionError("pricing must execute in Rust")

    def abandon(self, argument: str) -> str:
        """Reject a Python cancellation callback."""
        raise AssertionError("cancellation must execute in Rust")

    def guardrail_input_status(self, argument: str) -> str:
        """Reject response-side polling of Python inspection state."""
        raise AssertionError("moderation must wake its Rust waiter")


@dataclass(frozen=True)
class _Scenario:
    """One native serving fixture with independently controlled upstreams.

    Attributes:
        control: Initial admission facade and native lifecycle owner.
        provider: Base model HTTP endpoint.
        moderator: Decision HTTP endpoint.
        url: Gateway listener URL.
        key: Synthetic gateway bearer key.
        database: Durable accounting and moderation records.
    """

    control: _NativeOnlyControl
    provider: _Endpoint
    moderator: _Endpoint
    url: str
    key: str
    database: Path

    def request(
        self, surface: str = "chat", *, stream: bool = True, tool: bool = False
    ) -> httpx.Response:
        """Read a complete native response on the selected public API surface."""
        body: JsonObject = {"model": "coding", "stream": stream, "max_tokens": 100}
        path = "/v1/chat/completions"
        if surface == "responses":
            body.pop("max_tokens")
            body["input"] = "Synthetic moderation fixture."
            path = "/v1/responses"
        else:
            body["messages"] = [{"role": "user", "content": "Synthetic moderation fixture."}]
            if surface == "messages":
                path = "/v1/messages"
        if tool:
            schema: JsonObject = {"type": "object", "properties": {"q": {"type": "string"}}}
            body["tools"] = (
                [{"name": "lookup", "input_schema": schema}]
                if surface == "messages"
                else [{"type": "function", "name": "lookup", "parameters": schema}]
                if surface == "responses"
                else [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]
            )
        return httpx.post(
            self.url + path,
            headers={"authorization": f"Bearer {self.key}"},
            json=body,
            timeout=5,
        )

    def records(self) -> list[JsonObject]:
        """Drain native background reports before reading their durable evidence."""
        assert self.control.close(timeout_seconds=2)
        with connect_sqlite(self.database) as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT result FROM gateway_input_moderation")
            ]


@contextmanager
def _scenario(
    tmp_path: Path,
    *,
    blocked: bool = False,
    detector_status: int = 200,
    provider_status: int = 200,
    empty: bool = False,
    pause_stream: bool = False,
    wait_ms: int = 0,
    timeout_ms: int = 3000,
    mode: Literal["enforce", "observe"] = "enforce",
    execution: Literal["parallel", "before_dispatch"] = "parallel",
    tool: bool = False,
    billing_source: BillingSource = BillingSource.HOST_MANAGED,
    capture: CaptureController | None = None,
) -> Iterator[_Scenario]:
    """Compose an input policy, real SQLite owner, and real native gateway."""
    marker = "provider-marker"
    frames = _tool_chunk('{"q":"' + marker + '"}', start=True) if tool else _content_chunk(marker)
    if empty:
        frames = b""
    tail = (
        _terminal_frames().replace(b'"finish_reason":"stop"', b'"finish_reason":"tool_calls"')
        if tool
        else _terminal_frames()
    )
    provider = _Endpoint(
        frames if pause_stream else frames + tail,
        content_type="text/event-stream",
        status=provider_status,
        tail=tail if pause_stream else b"",
    )
    moderator = _Endpoint(
        json.dumps({"flagged": blocked}).encode(),
        content_type="application/json",
        status=detector_status,
    )
    _, key = _configured_gateway(
        tmp_path,
        base_url=provider.url + "/v1",
        capabilities=ModelCapabilities(maximum_output_tokens=4096, supports_tools=True),
        billing_source=billing_source,
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="input-policy",
                    revision="fixture-v1",
                    protected=True,
                    input_execution=execution,
                    input_fail_open=True,
                    input_max_wait_ms=wait_ms,
                    mode=mode,
                    checks=(
                        GuardrailCheck(
                            check_id="content",
                            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                            stage=GuardrailCheckStage.INPUT,
                            action=GuardrailAction.BLOCK,
                            adapter_id="detector",
                            timeout_ms=timeout_ms,
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"detector": _Classifier(moderator)})),
        monotonic=time.monotonic,
    )
    control = _NativeOnlyControl(components, guardrails=engine, capture=capture)
    try:
        with _serving(control, capture=None if capture is None else capture.native) as url:
            yield _Scenario(control, provider, moderator, url, key, components.ledger.database_path)
    finally:
        provider.release.set()
        provider.tail_release.set()
        moderator.release.set()
        control.close(timeout_seconds=2)
        engine.close(timeout_seconds=2)
        components.write_ledger.close()
        provider.close()
        moderator.close()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("tool", [False, True])
def test_parallel_verdict_controls_text_and_tools(
    tmp_path: Path, surface: str, stream: bool, blocked: bool, tool: bool
) -> None:
    """Generation and moderation overlap; a timely denial cancels before any answer escapes."""
    with _scenario(tmp_path, blocked=blocked, wait_ms=1000, tool=tool) as case:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request, surface, stream=stream, tool=tool)
            try:
                assert case.provider.started.wait(2), "provider waited for moderation"
                assert case.moderator.started.wait(2)
                case.moderator.release.set()
                assert case.moderator.finished.wait(2)
                if not blocked:
                    case.provider.release.set()
                result = response.result(timeout=3)
                assert result.status_code == (400 if blocked else 200), result.text
                assert ("provider-marker" in result.text) is not blocked
                if blocked:
                    assert not case.provider.release.is_set()
                    assert "content_filter" in result.text or "invalid_request_error" in result.text
                records = case.records()
                assert len(records) == 1
                assert records[0]["status"] == ("denied" if blocked else "allowed")
                with connect_sqlite(case.database) as db:
                    rows = db.execute(
                        "SELECT state, budget_settled_nano_usd FROM gateway_attempts"
                    ).fetchall()
                assert len(rows) == 1
                assert rows[0][0] == ("failed" if blocked else "completed")
                assert rows[0][1] == 0 if blocked else rows[0][1] > 0
            finally:
                case.provider.release.set()
                case.moderator.release.set()


@pytest.mark.parametrize("wait_ms", [0, 100])
def test_retry_does_not_start_response_grace_before_an_answer_is_ready(
    tmp_path: Path, wait_ms: int
) -> None:
    """A withheld provider failure cannot discard a denial before its fallback answers."""
    with _scenario(tmp_path, blocked=True, wait_ms=wait_ms, timeout_ms=3000) as case:
        calls = 0
        second = threading.Event()
        lock = threading.Lock()

        def fail_first(server: BaseHTTPRequestHandler) -> bool:
            """Fail one physical attempt, then hold the retry's output behind the fixture event."""
            nonlocal calls
            with lock:
                calls += 1
                ordinal = calls
            if ordinal != 1:
                second.set()
                return False
            server.rfile.read(int(server.headers["Content-Length"]))
            case.provider.started.set()
            assert case.moderator.started.wait(2)
            body = b'{"error":{"message":"synthetic first-attempt failure"}}'
            server.send_response(500)
            server.send_header("Content-Type", "application/json")
            server.send_header("Content-Length", str(len(body)))
            server.end_headers()
            server.wfile.write(body)
            server.wfile.flush()
            return True

        case.provider.intercept = fail_first
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request)
            try:
                assert second.wait(3)
                case.moderator.release.set()
                assert case.moderator.finished.wait(2)
                result = response.result(timeout=3)
                assert result.status_code == 400, result.text
                assert "provider-marker" not in result.text
                assert not case.provider.release.is_set()
                assert case.records()[0]["status"] == "denied"
            finally:
                case.provider.release.set()
                case.moderator.release.set()


@pytest.mark.parametrize("empty", [False, True])
def test_exhausted_provider_response_still_waits_for_its_input_decision(
    tmp_path: Path, empty: bool
) -> None:
    """An error or empty answer reaching the caller is also a response-ready boundary."""
    with _scenario(
        tmp_path, blocked=True, provider_status=200 if empty else 400, empty=empty, wait_ms=1000
    ) as case:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request)
            assert case.moderator.started.wait(2)
            case.provider.release.set()
            assert case.provider.finished.wait(2)
            with pytest.raises(TimeoutError):
                response.result(timeout=0.1)
            case.moderator.release.set()
            result = response.result(timeout=3)
            assert result.status_code == 400, result.text
            assert "guardrail" in result.text
            assert case.records()[0]["status"] == "denied"


@pytest.mark.parametrize("wait_ms", [0, 100])
@pytest.mark.parametrize(
    "billing_source", [BillingSource.HOST_MANAGED, BillingSource.CUSTOMER_MANAGED]
)
def test_response_ready_has_bounded_grace_and_late_denial_cannot_revoke(
    tmp_path: Path, wait_ms: int, billing_source: BillingSource
) -> None:
    """Either funding source allows delivery after the configured moderation grace."""
    with _scenario(tmp_path, blocked=True, wait_ms=wait_ms, billing_source=billing_source) as case:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request)
            try:
                assert case.provider.started.wait(2)
                assert case.moderator.started.wait(2)
                start = time.monotonic()
                case.provider.release.set()
                result = response.result(timeout=1.5)
                elapsed = time.monotonic() - start
                assert elapsed < wait_ms / 1000 + 0.75
                assert elapsed >= wait_ms / 1000 * 0.8
                assert not case.moderator.release.is_set()
                assert result.status_code == 200 and "provider-marker" in result.text
                case.moderator.release.set()
                assert case.moderator.finished.wait(2)
                records = case.records()
                assert len(records) == 1
                assert records[0]["status"] == "failed_open"
                assert records[0]["reason"] == "response_ready_timeout"
                assert _first_observation(records[0])["policy_id"] == "input-policy"
            finally:
                case.provider.release.set()
                case.moderator.release.set()


@pytest.mark.parametrize("failure", ["http", "timeout"])
def test_detector_failure_is_recorded_without_refusing_request(
    tmp_path: Path, failure: str
) -> None:
    """A detector failure releases generation and retains a failed-moderation flag."""
    with _scenario(
        tmp_path, detector_status=503 if failure == "http" else 200, timeout_ms=100, wait_ms=1000
    ) as case:
        case.provider.release.set()
        if failure == "http":
            case.moderator.release.set()
        result = case.request()
        assert result.status_code == 200 and "provider-marker" in result.text
        record = case.records()[0]
        assert record["status"] == "failed_open"
        assert _first_observation(record)["outcome"] in {"error", "timeout", "unavailable"}


def _first_observation(record: JsonObject) -> JsonObject:
    """Validate the persisted nested JSON before asserting content-free decision metadata."""
    observations = record["observations"]
    assert isinstance(observations, list) and observations
    first = observations[0]
    assert isinstance(first, dict)
    return first


def test_observation_finishes_after_delivery_and_shutdown_drains_it(tmp_path: Path) -> None:
    """Observe mode never gates delivery, but shutdown retains an in-flight score report."""
    with _scenario(tmp_path, blocked=True, mode="observe", wait_ms=1000) as case:
        case.provider.release.set()
        result = case.request()
        assert result.status_code == 200 and "provider-marker" in result.text
        assert case.moderator.started.wait(2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            drained = executor.submit(case.control.close, 2)
            assert not drained.done()
            case.moderator.release.set()
            assert drained.result(timeout=3)
        records = case.records()
        assert len(records) == 1 and records[0]["status"] == "observed"
        assert _first_observation(records[0])["outcome"] == "flagged"


@pytest.mark.parametrize("blocked", [False, True])
def test_serial_configuration_waits_before_dispatch(tmp_path: Path, blocked: bool) -> None:
    """A serial policy protects third-party dispatch while sharing the native detector."""
    with _scenario(tmp_path, blocked=blocked, execution="before_dispatch") as case:
        case.provider.release.set()
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request)
            try:
                assert case.moderator.started.wait(2)
                assert not case.provider.started.is_set()
                case.moderator.release.set()
                result = response.result(timeout=3)
                assert result.status_code == (400 if blocked else 200), result.text
                assert case.provider.started.is_set() is not blocked
                assert case.records()[0]["status"] == ("denied" if blocked else "allowed")
            finally:
                case.moderator.release.set()


@pytest.mark.parametrize("stage", ["input", "output"])
@pytest.mark.parametrize("stream", [False, True])
def test_ordered_redaction_precedes_blocking_through_native_http(
    tmp_path: Path, stage: str, stream: bool
) -> None:
    """A configured redaction removes sensitive content before the following block check."""
    marker = "sensitive-marker-123"
    replacement = "[REDACTED]"
    prompt = f"Input contains {marker}." if stage == "input" else "Safe input."
    answer = f"Answer contains {marker}." if stage == "output" else "Safe answer."
    engine = engine_from_document(
        {
            "adapters": [
                {
                    "adapter_id": "sensitive",
                    "kind": "regex",
                    "patterns": [marker],
                    "replacement": replacement,
                }
            ],
            "policies": [
                {
                    "policy_id": "ordered-content",
                    "protected": True,
                    "checks": [
                        {
                            "check_id": action,
                            "stage": stage,
                            "action": action,
                            "capability": "pii",
                            "adapter_id": "sensitive",
                            "timeout_ms": 1000,
                        }
                        for action in ("modify", "block")
                    ],
                }
            ],
        }
    )
    with _provider(_content_chunk(answer) + _terminal_frames()) as (base_url, received):
        _, key = _configured_gateway(tmp_path, base_url=base_url)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        try:
            with _serving(control) as url:
                response = httpx.post(
                    url + "/v1/chat/completions",
                    headers={"authorization": f"Bearer {key}"},
                    json={
                        "model": "coding",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": stream,
                        "max_tokens": 100,
                    },
                    timeout=5,
                )
            assert response.status_code == 200, response.text
            assert len(received) == 1
            dispatched = json.loads(received[0])
            assert dispatched["messages"] == [
                {"role": "user", "content": prompt.replace(marker, replacement)}
            ]
            documents = (
                [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                if stream
                else [response.json()]
            )
            assert all(document.get("error") is None for document in documents)
            actual_answer = "".join(
                choice["delta" if stream else "message"].get("content") or ""
                for document in documents
                for choice in document.get("choices", [])
            )
            assert actual_answer == answer.replace(marker, replacement)
            assert marker not in response.text
            if stage == "input":
                assert marker.encode() not in received[0]
            with connect_sqlite(components.ledger.database_path) as db:
                assert db.execute("SELECT state FROM gateway_attempts").fetchall() == [
                    ("completed",)
                ]
                assert db.execute("SELECT terminal_state FROM gateway_requests").fetchall() == [
                    ("completed",)
                ]
        finally:
            components.write_ledger.close()


@pytest.mark.parametrize("wait_ms", [0, 100])
def test_a_ready_stream_releases_its_first_content_before_generation_or_moderation_finishes(
    tmp_path: Path, wait_ms: int
) -> None:
    """The response grace starts at the first deliverable event, never the provider terminal."""
    with _scenario(tmp_path, pause_stream=True, wait_ms=wait_ms) as case:
        case.provider.release.set()
        started = time.monotonic()
        try:
            with httpx.stream(
                "POST",
                case.url + "/v1/chat/completions",
                headers={"authorization": f"Bearer {case.key}"},
                json={
                    "model": "coding",
                    "stream": True,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "stream fixture"}],
                },
                timeout=3,
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if "provider-marker" in line:
                        break
                else:
                    pytest.fail("no content arrived before the provider terminal")
                assert time.monotonic() - started < 1.5
                assert not case.provider.finished.is_set()
                assert not case.moderator.release.is_set()
        finally:
            case.provider.tail_release.set()
            case.moderator.release.set()
        assert case.records()[0]["status"] == "failed_open"


def test_shutdown_drains_only_its_own_moderation_records(tmp_path: Path) -> None:
    """Another worker's pending observation cannot hold this worker's storage lifetime open."""
    with _scenario(tmp_path / "other", mode="observe") as other:
        other.provider.release.set()
        assert other.request().status_code == 200
        assert other.moderator.started.wait(2)
        with _scenario(tmp_path / "first") as first:
            first.provider.release.set()
            first.moderator.release.set()
            assert first.request().status_code == 200
            assert first.control.close(timeout_seconds=0.5)
            assert not other.moderator.release.is_set()
        other.moderator.release.set()
        assert other.records()[0]["status"] == "observed"

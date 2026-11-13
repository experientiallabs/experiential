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
    """Supply native parallel inspection and synchronous admission inspection."""

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
        """Run serial admission inspection against the same loopback detector."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint.url,
                json={"request": request.model_dump(mode="json"), "check_id": check.check_id},
            )
            response.raise_for_status()
            return ClassifierVerdict.model_validate(response.json())

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


@contextmanager
def _scenario(
    tmp_path: Path,
    *,
    blocked: bool = False,
    detector_status: int = 200,
    provider_status: int = 200,
    empty: bool = False,
    pause_stream: bool = False,
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
    """Native accounting preserves input gating and zero customer charge for denied work."""
    with _scenario(tmp_path, blocked=blocked, tool=tool) as case:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(case.request, surface, stream=stream, tool=tool)
            try:
                assert case.provider.started.wait(2), "provider waited for moderation"
                assert case.moderator.started.wait(2)
                case.moderator.release.set()
                assert case.moderator.finished.wait(2)
                case.provider.release.set()
                result = response.result(timeout=3)
                assert result.status_code == (400 if blocked else 200), result.text
                assert ("provider-marker" in result.text) is not blocked
                if blocked:
                    assert "content_filter" in result.text or "invalid_request_error" in result.text
                assert case.control.close(timeout_seconds=2)
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


@pytest.mark.parametrize("blocked", [False, True])
def test_serial_configuration_waits_before_dispatch(tmp_path: Path, blocked: bool) -> None:
    """A serial policy still inspects admission before any provider dispatch."""
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
                assert case.control.close(timeout_seconds=2)
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

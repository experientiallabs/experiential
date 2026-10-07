"""Exercise parallel input inspection against real native sockets and ledger writes."""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, GatewayTokenPrices, ModelCapabilities
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScopeKind
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails import session as guardrail_session
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
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
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _chat_body
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.rung_admission import RungShed
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.mandatory_guardrails_test import _tool_chunk
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)


@pytest.mark.parametrize("outcome", ["allow", "block"])
def test_customer_funded_admission_waits_for_input_approval(tmp_path: Path, outcome: str) -> None:
    """BYOK cannot return a dispatchable admission while mandatory input is pending."""
    classifier = _Classifier(outcome)
    _, key = _configured_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    control = NativeControlPlane(components, guardrails=_engine(classifier))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_admit, control, key, _chat_body())
        try:
            assert classifier.started.wait(2)
            assert not future.done()
            classifier.release.set()
            if outcome == "block":
                with pytest.raises(NativeBridgeError):
                    future.result(timeout=2)
            else:
                admission = future.result(timeout=2)
                assert admission["guardrail_input_pending"] is False
                control.abandon(json.dumps({"request_id": admission["request_id"]}))
        finally:
            classifier.release.set()
    with sqlite3.connect(components.ledger.database_path) as db:
        assert db.execute("select count(*) from gateway_attempts").fetchone() == (0,)


def test_customer_funded_deadline_race_terminalizes_accepted_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A classifier that allows just after the admission wait expires cannot orphan the ledger."""
    result: Future[None] = Future()
    result.set_result(None)
    monkeypatch.setattr(result, "result", Mock(side_effect=[TimeoutError(), None]))

    def start(coroutine: Coroutine[None, None, None]) -> Future[None]:
        """Bind the controlled race without leaving an unawaited fixture coroutine."""
        coroutine.close()
        return result

    monkeypatch.setattr(guardrail_session, "start_on_native_loop", start)
    _, key = _configured_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    control = NativeControlPlane(components, guardrails=_engine(_Classifier("allow")))
    with pytest.raises(NativeBridgeError):
        _admit(control, key, _chat_body())
    with sqlite3.connect(components.ledger.database_path) as db:
        assert db.execute("select terminal_state from gateway_requests").fetchall() == [("failed",)]
        assert db.execute("select count(*) from gateway_attempts").fetchone() == (0,)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("denial", ["quota", "capacity", "internal"])
def test_predispatch_failure_cancels_input_and_preserves_public_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, denial: str
) -> None:
    """Even a gate poll winning the race cannot mask a decided admission failure."""
    classifier = _Classifier("allow")
    closed_seen = threading.Event()
    callback_errors: list[Exception] = []
    _, key = _configured_gateway(
        tmp_path, base_url="http://127.0.0.1:1/v1", billing_source=BillingSource.HOST_MANAGED
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})

    class Control(NativeControlPlane):
        """Hold the admission reply until Rust has observed its closed input gate."""

        def start_attempt(self, argument: str) -> str:
            """Exercise real admission cleanup while enforcing the review's race ordering."""
            assert classifier.started.wait(2)
            try:
                return super().start_attempt(argument)
            except Exception as exc:
                callback_errors.append(exc)
                raise
            finally:
                closed_seen.wait(2)

        def guardrail_input_status(self, argument: str) -> str:
            """Signal when polling discovers that admission already ended."""
            result = super().guardrail_input_status(argument)
            if json.loads(result)["action"] == "closed":
                closed_seen.set()
            return result

    control = Control(components, guardrails=_engine(classifier))
    if denial == "capacity":
        monkeypatch.setattr(
            control._accounting,
            "_reserve_rung_slot",  # noqa: SLF001 - scripted capacity boundary.
            Mock(return_value=RungShed(reason="queue_bound", default_bound=True)),
        )
    else:
        failure = (
            BudgetReservationRejected(scope_kind=BudgetScopeKind.TEAM, reason="fixture quota")
            if denial == "quota"
            else RuntimeError("fixture reservation failure")
        )
        monkeypatch.setattr(
            control._accounting._write_ledger,
            "start_attempt",  # noqa: SLF001 - synchronous ledger boundary.
            Mock(side_effect=failure),
        )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()
    server = threading.Thread(
        target=serve_native_gateway,
        args=(control,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "shutdown": stop,
            "on_listening": ready.set,
            "graceful_timeout_seconds": 0.1,
        },
        daemon=True,
    )
    server.start()
    body: JsonObject = {
        "model": "coding",
        "stream": True,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "synthetic fixture"}],
    }
    path = "/v1/chat/completions"
    if surface == "responses":
        body = {"model": "coding", "stream": True, "input": "synthetic fixture"}
        path = "/v1/responses"
    elif surface == "messages":
        path = "/v1/messages"
    try:
        assert ready.wait(5)
        response = httpx.post(
            f"http://127.0.0.1:{port}{path}",
            headers={"authorization": f"Bearer {key}"},
            json=body,
            timeout=5,
        )
        assert response.status_code == (500 if denial == "internal" else 429), (
            response.text,
            callback_errors,
        )
        assert "Content inspection is unavailable" not in response.text
        if denial == "quota":
            assert "monthly gateway allocation is exhausted" in response.text
        assert closed_seen.is_set()
        assert classifier.cancelled.wait(2), "orphaned classifier continued after admission ended"
        with sqlite3.connect(components.ledger.database_path) as db:
            assert db.execute("select terminal_state from gateway_requests").fetchall() == [
                ("failed",)
            ]
            assert db.execute("select count(*) from gateway_attempts").fetchone() == (0,)
    finally:
        classifier.release.set()
        stop.request_shutdown()
        server.join(timeout=5)


class _Classifier:
    """Barrier-controlled input classifier with no output inspection capability."""

    def __init__(self, outcome: str) -> None:
        """Create one independent classifier and release latch."""
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.requests: list[GatewayRequest] = []

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Wait independently of gateway bridge workers before returning a decision."""
        self.requests.append(request)
        self.started.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.outcome == "error":
            raise RuntimeError("private-detector-error-must-not-escape")
        if self.outcome == "unsupported":
            raise ClassifierCoverageError
        return ClassifierVerdict(flagged=self.outcome == "block")

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Reject accidental use of output moderation in this input-only policy."""
        raise AssertionError("input-only policy called output inspection")


def _engine(classifier: _Classifier, timeout_ms: int = 3000) -> GuardrailEngine:
    """Compose the input-only policy through the same scoped store as every other guardrail."""
    return GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="criminal-input",
                    revision="fixture-v1",
                    protected=True,
                    input_execution="parallel",
                    checks=(
                        GuardrailCheck(
                            check_id="criminal",
                            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                            stage=GuardrailCheckStage.INPUT,
                            action=GuardrailAction.BLOCK,
                            adapter_id="criminal",
                            timeout_ms=timeout_ms,
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("outcome", ["allow", "block", "error", "timeout", "unsupported"])
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("provider_phase", ["streaming", "not_open", "completed"])
def test_parallel_inspection_holds_text_tools_and_customer_settlement(
    tmp_path: Path,
    surface: str,
    streaming: bool,
    outcome: str,
    tool_call: bool,
    provider_phase: str,
) -> None:
    """Provider starts before classification; no headers/content escape until approval."""
    classifier = _Classifier(outcome)
    provider_started = threading.Event()
    provider_finish = threading.Event()
    provider_finished = threading.Event()
    headers_received = threading.Event()
    content_received = threading.Event()
    errors: list[Exception] = []
    marker = "private-provider-content"

    class Provider(BaseHTTPRequestHandler):
        """Send an early semantic event, then wait for the test to finish generation."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP fixture protocol.
            """Serve synthetic text or a function call with explicit provider usage."""
            self.rfile.read(int(self.headers["Content-Length"]))
            provider_started.set()
            if provider_phase == "not_open":
                provider_finish.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                frame = (
                    _tool_chunk('{"q":"' + marker + '"}', start=True)
                    if tool_call
                    else _content_chunk(marker)
                )
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 80,
                                "completion_tokens": 5,
                                "total_tokens": 85,
                            },
                        }
                    )
                )
                self.wfile.write(frame)
                self.wfile.flush()
                if provider_phase != "completed":
                    provider_finish.wait(5)
                terminal = _terminal_frames()
                if tool_call:
                    terminal = terminal.replace(
                        b'"finish_reason":"stop"', b'"finish_reason":"tool_calls"'
                    )
                self.wfile.write(terminal)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                provider_finished.set()

        def log_message(self, format: str, *args: object) -> None:
            """Keep provider fixture payloads out of test logs."""

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _, key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
        capabilities=ModelCapabilities(maximum_output_tokens=4096, supports_tools=True),
        billing_source=BillingSource.HOST_MANAGED,
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})

    class Control(NativeControlPlane):
        """Expose unexpected settlement callback failures to the test."""

        def settle(self, argument: str) -> str:
            """Record callback failures before the native retry boundary sanitizes them."""
            try:
                return super().settle(argument)
            except Exception as exc:
                errors.append(exc)
                raise

    control = Control(
        components, guardrails=_engine(classifier, timeout_ms=750 if outcome == "timeout" else 3000)
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()

    def serve() -> None:
        """Run the actual extension and retain startup failures for the caller."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                shutdown=stop,
                on_listening=ready.set,
                graceful_timeout_seconds=0.1,
            )
        except Exception as exc:  # noqa: BLE001 - surface fixture failures.
            errors.append(exc)
            ready.set()

    gateway_thread = threading.Thread(target=serve, daemon=True)
    gateway_thread.start()
    body: JsonObject = {"model": "coding", "stream": streaming}
    if surface == "responses":
        path = "/v1/responses"
        body["input"] = "Synthetic moderation fixture."
    else:
        path = "/v1/messages" if surface == "messages" else "/v1/chat/completions"
        body["max_tokens"] = 100
        body["messages"] = [{"role": "user", "content": "Synthetic moderation fixture."}]
    if tool_call:
        schema: JsonObject = {"type": "object", "properties": {"q": {"type": "string"}}}
        definition: JsonObject = {"name": "lookup", "parameters": schema}
        body["tools"] = (
            [{"name": "lookup", "input_schema": schema}]
            if surface == "messages"
            else [{"type": "function", **definition}]
            if surface == "responses"
            else [{"type": "function", "function": definition}]
        )

    def request() -> tuple[int, str]:
        """Retain first-header and first-content arrival separately from completion."""
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}{path}",
            headers={"authorization": f"Bearer {key}"},
            json=body,
            timeout=8,
        ) as response:
            if outcome == "unsupported":
                assert "retry-after" not in response.headers
            headers_received.set()
            chunks: list[str] = []
            for chunk in response.iter_text():
                chunks.append(chunk)
                if marker in "".join(chunks):
                    content_received.set()
            return response.status_code, "".join(chunks)

    try:
        assert ready.wait(5) and not errors
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(request)
            try:
                assert classifier.started.wait(2)
                assert provider_started.wait(2), "generation waited for classification"
                assert not headers_received.wait(0.05), "response escaped the pending input gate"
                if outcome != "timeout":
                    classifier.release.set()
                if outcome == "allow":
                    if streaming and not tool_call and provider_phase == "streaming":
                        assert content_received.wait(2)
                        assert not provider_finished.is_set(), "allow buffered the full response"
                    provider_finish.set()
                status, result = future.result(timeout=3)
                assert status == (
                    200
                    if outcome == "allow"
                    else 400
                    if outcome in {"block", "unsupported"}
                    else 503
                )
                if outcome == "unsupported":
                    assert "Content inspection does not support this complete request." in result
                    expected_code = (
                        "invalid_request_error"
                        if surface == "messages"
                        else "unsupported_capability"
                    )
                    assert expected_code in result
                assert (marker in result) == (outcome == "allow")
                assert "private-detector-error-must-not-escape" not in result
            finally:
                classifier.release.set()
                provider_finish.set()
        assert not errors
        with sqlite3.connect(components.ledger.database_path) as db:
            rows = db.execute(
                "select state, failure_class, estimated_cost_nano_usd, budget_settled_nano_usd "
                "from gateway_attempts"
            ).fetchall()
            requests = db.execute("select terminal_state from gateway_requests").fetchall()
        assert len(rows) == 1
        state, failure, provider_cost, customer_cost = rows[0]
        assert state == ("completed" if outcome == "allow" else "failed")
        assert requests == [(state,)]
        assert customer_cost > 0 if outcome == "allow" else customer_cost == 0
        if provider_phase != "not_open" or outcome == "allow":
            assert provider_cost is not None and provider_cost > 0
        else:
            assert provider_cost is None  # No usage report, not fictitious zero provider cost.
        assert (
            failure is None
            if outcome == "allow"
            else failure in {"guardrail", "unavailable", "unsupported_capability"}
        )
        if outcome == "unsupported":
            assert failure == "unsupported_capability"
        assert len(classifier.requests) == 1
    finally:
        classifier.release.set()
        provider_finish.set()
        stop.request_shutdown()
        gateway_thread.join(timeout=5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)


@pytest.mark.parametrize(
    ("failure_class", "provider_status"),
    [("throttled", 429), ("provider_authentication", 401), ("transport", None)],
)
@pytest.mark.parametrize("retained_settlement", [False, True])
def test_provider_failure_survives_later_input_denial_and_settlement_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_class: str,
    provider_status: int | None,
    retained_settlement: bool,
) -> None:
    """Input denial waives customer cost without erasing provider backoff or circuit evidence."""
    classifier = _Classifier("block")
    provider_finished = threading.Event()
    response_received = threading.Event()
    provider_requests: list[bytes] = []
    settlements: list[JsonObject] = []
    entries: list[InflightRequest] = []

    class Provider(BaseHTTPRequestHandler):
        """Fail an accepted request before the independently blocked input classifier resolves."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP fixture protocol.
            """Expose real status/header or socket-failure evidence to the native client."""
            provider_requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if provider_status is None:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                payload = b'{"error":{"message":"private-provider-error","type":"fixture"}}'
                self.send_response(provider_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if provider_status == 429:
                    self.send_header("Retry-After", "120")
                    self.send_header("X-RateLimit-Remaining-Requests", "0")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            finally:
                provider_finished.set()

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic provider diagnostics out of test logs."""

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _, key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
        billing_source=BillingSource.HOST_MANAGED,
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})

    class Control(NativeControlPlane):
        """Retain observed boundary evidence while preserving real admission and settlement."""

        def admit(self, argument: str) -> str:
            """Keep the admitted ownership snapshot for health and recovery assertions."""
            result = super().admit(argument)
            entry = self._accounting.entry(str(json.loads(result)["request_id"]))
            assert entry is not None
            entries.append(entry)
            return result

        def settle(self, argument: str) -> str:
            """Observe the unmodified Rust callback, including delivery retries."""
            settlements.append(json.loads(argument))
            return super().settle(argument)

    control = Control(components, guardrails=_engine(classifier))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()
    gateway_thread = threading.Thread(
        target=serve_native_gateway,
        args=(control,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "shutdown": stop,
            "on_listening": ready.set,
            "graceful_timeout_seconds": 0.1,
        },
        daemon=True,
    )
    gateway_thread.start()

    def request() -> httpx.Response:
        """Exercise the native endpoint without releasing the classifier from the client."""
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json=json.loads(_chat_body()),
            timeout=8,
        )
        response_received.set()
        return response

    try:
        assert ready.wait(5)
        with monkeypatch.context() as patch:
            if retained_settlement:
                patch.setattr(
                    control._accounting,  # noqa: SLF001 - retain the real native settlement.
                    "_finish_attempt",
                    Mock(side_effect=RuntimeError("private ledger failure")),
                )
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(request)
                try:
                    assert classifier.started.wait(2)
                    assert provider_finished.wait(2), "provider waited for the input verdict"
                    assert not response_received.wait(0.1), "provider error escaped pending input"
                    classifier.release.set()
                    response = pending.result(timeout=5)
                finally:
                    classifier.release.set()
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "content_filter"
            assert "private-provider-error" not in response.text
            assert "private ledger failure" not in response.text
            assert len(provider_requests) == 1
            assert len(entries) == 1
            entry = entries[0]
            assert (entry.pending_settlement is not None) is retained_settlement
            if retained_settlement:
                assert entry.pending_settlement == settlements[0]
                assert all(item == settlements[0] for item in settlements)
        if retained_settlement:
            control._accounting.sweep_expired()  # noqa: SLF001 - replay the retained native write.
        assert control._accounting.entry(entry.authorization.request_id) is None  # noqa: SLF001
        with sqlite3.connect(components.ledger.database_path) as db:
            assert db.execute(
                "SELECT state, failure_class, budget_settled_nano_usd FROM gateway_attempts"
            ).fetchall() == [("failed", "guardrail", 0)]
            assert db.execute("SELECT terminal_state FROM gateway_requests").fetchall() == [
                ("failed",)
            ]
        observed = settlements[0]["failure"]
        assert isinstance(observed, dict)
        assert observed["failure_class"] == failure_class
        health = control._accounting.health  # noqa: SLF001 - actual admission registry.
        health_key = deployment_health_key(entry.authorization, entry.route.deployment)
        if provider_status == 429:
            assert observed["retry_after_seconds"] == 120
            assert settlements[0]["rate_limit_headers"] == {
                "retry-after": "120",
                "x-ratelimit-remaining-requests": "0",
            }
            remaining = health.throttled_remaining_seconds((health_key,))
            assert remaining is not None and 110 < remaining <= 120
            classifier.outcome = "allow"
            retried = request()
            assert retried.status_code == 429, retried.text
            assert 110 <= int(retried.headers["retry-after"]) <= 120
            assert len(provider_requests) == 1, "suppressed provider was dialed again"
        elif provider_status == 401:
            assert health.suppressed(health_key)
        else:
            states = health._states  # noqa: SLF001 - one failure remains below the circuit threshold.
            assert states[health_key].consecutive_failures == 1
    finally:
        classifier.release.set()
        stop.request_shutdown()
        gateway_thread.join(timeout=5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
        components.write_ledger.close()
        assert not gateway_thread.is_alive()


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
            with sqlite3.connect(components.ledger.database_path) as db:
                assert db.execute("SELECT state FROM gateway_attempts").fetchall() == [
                    ("completed",)
                ]
                assert db.execute("SELECT terminal_state FROM gateway_requests").fetchall() == [
                    ("completed",)
                ]
        finally:
            components.write_ledger.close()

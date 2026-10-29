"""Capture eligibility follows the request-owned input gate through real native HTTP."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, GatewayTokenPrices
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailOutcome,
    GuardrailRejected,
)
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _admit_started, _chat_body
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureRecord,
    capture_unavailable_failure,
)
from exp.runtime.gateway.native_capture_test import _request_json
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.guardrail_observation_test import (
    _Classifier as _ObservationClassifier,
)
from exp.runtime.gateway.tests.guardrail_observation_test import (
    _Engine as _ObservationEngine,
)
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.launch_test import _unused_port
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _RetrievedGuard
from exp.runtime.gateway.tests.mandatory_guardrails_test import _Guard
from exp.runtime.gateway.tests.native_tool_search_test import _configure
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


@pytest.mark.parametrize("release_failure", [False, True])
@pytest.mark.parametrize("late_callback", ["none", "inspect", "dispatch"])
def test_observation_retains_required_release_failure_in_native_settlement(
    tmp_path: Path, release_failure: bool, late_callback: str
) -> None:
    """A host release failure survives polling and paid settlement; plain closure stays optional."""
    _, key = _configured_gateway(
        tmp_path,
        base_url="http://127.0.0.1:1/v1",
        billing_source=BillingSource.HOST_MANAGED,
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    classifier = _ObservationClassifier(GuardrailOutcome.ALLOW)
    classifier.release.set()
    engine = _ObservationEngine(classifier)
    control = NativeControlPlane(components, guardrails=engine)
    try:
        admission = _admit_started(control, key, _chat_body())
        request_id = str(admission["request_id"])
        entry = control._accounting.entry(request_id)  # noqa: SLF001 - request lifecycle assertion.
        assert entry is not None and entry.guardrails is not None
        session = entry.guardrails
        request = entry.request
        assert isinstance(request, GatewayRequest)
        assert not session.enforcing_policies
        assert session.input_decision() == {"action": "allow"}
        session.cancel(capture_unavailable_failure() if release_failure else None)
        session.cancel()

        def callback() -> None:
            """Replay the host input seam after session closure without generating more work."""
            if late_callback == "inspect":
                assert session.inspect_input(request) is request
            elif late_callback == "dispatch":
                session.prepare_dispatch((request,), overlap=True)

        if release_failure and late_callback != "none":
            with pytest.raises(GuardrailRejected) as rejected:
                callback()
            assert rejected.value.failure.safe_details["code"] == "capture_unavailable"
        else:
            callback()
        decision = json.loads(
            control.guardrail_input_status(json.dumps({"request_id": request_id}))
        )
        assert decision["action"] == ("error" if release_failure else "allow")
        if release_failure:
            assert decision["failure"]["safe_details"] == {
                "code": "capture_unavailable",
                "input_guardrail_denied": True,
            }
        control.settle(
            json.dumps(
                {
                    "request_id": request_id,
                    "attempt_id": admission["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                    "tool_names": [],
                    "failure": None,
                }
            )
        )
        with sqlite3.connect(components.ledger.database_path) as db:
            assert db.execute("select terminal_state from gateway_requests").fetchall() == [
                ("failed" if release_failure else "completed",)
            ]
            state, charged = db.execute(
                "select state, budget_settled_nano_usd from gateway_attempts"
            ).fetchone()
        assert state == ("failed" if release_failure else "completed")
        assert charged == 0 if release_failure else charged > 0
        assert control._accounting.entry(request_id) is None  # noqa: SLF001 - terminal ownership.
    finally:
        engine.close(timeout_seconds=2)
        components.write_ledger.close()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", ["allow", "block", "error", "timeout", "full", "broken"])
def test_parallel_capture_waits_for_input_approval_and_fails_without_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, stream: bool, outcome: str
) -> None:
    """Raw sinks receive only approved prompts; failed capture withholds paid generation."""
    classifier = _Classifier(outcome)
    records: list[str] = []
    begins: list[tuple[str, str | None, str | None]] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(
            settlement_required=False, maximum_pending_records=1
        ).model_dump_json(),
        records.append,
    )
    if outcome == "full":
        assert collector.begin(_request_json())

    def application_for(_authorization: AuthorizationSnapshot) -> str:
        """Make a host capture policy failure observable without leaking its diagnostics."""
        if outcome == "broken":
            raise RuntimeError("private-capture-policy-error")
        return "capture-fixture"

    capture = CaptureController(collector, application_for=application_for)
    original_begin = capture.begin

    def begin(
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        model_id: str | None,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Count real collector admission without replacing its capacity or serialization."""
        begins.append((authorization.request_id, model_id, session_id))
        return original_begin(authorization, request, model_id, session_id=session_id)

    monkeypatch.setattr(capture, "begin", begin)
    with _provider(_content_chunk("private-provider-result") + _terminal_frames()) as (
        base_url,
        upstream,
    ):
        _, key = _configured_gateway(
            tmp_path,
            base_url=base_url,
            billing_source=BillingSource.HOST_MANAGED,
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000_000,
                output_nano_usd_per_million_tokens=2_000_000_000,
            ),
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(
            components,
            capture=capture,
            guardrails=_engine(classifier, timeout_ms=750 if outcome == "timeout" else 3000),
        )
        port = _unused_port()
        ready = threading.Event()
        shutdown = exp_gateway_native.shutdown_handle()
        worker = threading.Thread(
            target=serve_native_gateway,
            args=(control,),
            kwargs={
                "host": "127.0.0.1",
                "port": port,
                "capture": collector,
                "shutdown": shutdown,
                "on_listening": ready.set,
                "graceful_timeout_seconds": 0.1,
            },
            daemon=True,
        )
        worker.start()
        try:
            assert ready.wait(5)
            path, body = _body(surface, stream)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    httpx.post,
                    f"http://127.0.0.1:{port}{path}",
                    headers={"authorization": f"Bearer {key}", "x-session-id": "capture-episode"},
                    json=body,
                    timeout=5,
                )
                try:
                    assert classifier.started.wait(2)
                    deadline = time.monotonic() + 2
                    while not upstream and time.monotonic() < deadline:
                        time.sleep(0.005)
                    assert upstream, "capture admission prevented approved speculative generation"
                    assert not future.done(), "response escaped while classification was pending"
                    assert begins == [], "unapproved input entered the raw collector"
                    assert records == []
                    if outcome != "timeout":
                        classifier.release.set()
                    response = future.result(timeout=3)
                finally:
                    classifier.release.set()
            assert response.status_code == (
                200 if outcome == "allow" else 400 if outcome == "block" else 503
            ), response.text
            assert ("private-provider-result" in response.text) == (outcome == "allow")
            assert "private-capture-policy-error" not in response.text
            assert "private-detector-error-must-not-escape" not in response.text
            if outcome in {"full", "broken"}:
                error = response.json()["error"]
                if surface == "messages":
                    assert error["type"] == "overloaded_error"
                else:
                    assert error["code"] == "capture_unavailable"
                    assert error["type"] == "invalid_request_error"
            elif outcome in {"error", "timeout"} and surface != "messages":
                assert response.json()["error"]["code"] == "gateway_unavailable"
            with sqlite3.connect(components.ledger.database_path) as db:
                requests = db.execute(
                    "select request_id, terminal_state from gateway_requests"
                ).fetchall()
                attempts = db.execute(
                    "select state, budget_settled_nano_usd from gateway_attempts"
                ).fetchall()
            assert len(requests) == len(attempts) == len(upstream) == 1
            request_id, terminal = requests[0]
            assert terminal == ("completed" if outcome == "allow" else "failed")
            assert attempts[0][0] == terminal
            assert attempts[0][1] > 0 if outcome == "allow" else attempts[0][1] == 0
            assert control._accounting.entry(request_id) is None  # noqa: SLF001 - lifetime assertion.
            if outcome in {"block", "error", "timeout"}:
                assert begins == []
            else:
                assert len(begins) == 1
                assert begins[0][0] == request_id
                assert begins[0][1] is not None
                assert begins[0][2] == "capture-episode"
        finally:
            classifier.release.set()
            shutdown.request_shutdown()
            worker.join(5)
            components.write_ledger.close()
            if outcome == "full":
                collector.settle("request", False, False)
            assert collector.close(2)
            assert not worker.is_alive()
    parsed = [CaptureRecord.model_validate_json(record) for record in records]
    assert len(parsed) == (1 if outcome == "allow" else 0)
    if parsed:
        captured = parsed[0]
        assert captured.request.request_id == request_id
        assert captured.request.model_id == begins[0][1]
        assert captured.request.context["session_id"] == "capture-episode"
        assert captured.request.scope.identity_id == "default"
        assert captured.request.scope.application_id == "capture-fixture"
        assert captured.response is not None and captured.response.status == 200
        assert "synthetic prompt" in captured.request.model_dump_json()
        assert "private-provider-result" in captured.response.model_dump_json()


@dataclass(frozen=True)
class _Admission:
    """One real accepted request whose native gate is driven explicitly by the test.

    Attributes:
        control: Gateway admission and durable accounting callbacks.
        classifier: Barrier-controlled input decision.
        capture: Controller writing to the real native collector.
        records: Unfiltered destination payloads emitted by the collector.
        entry: Request-owned pending context and guardrail session.
    """

    control: NativeControlPlane
    classifier: _Classifier
    capture: CaptureController
    records: list[str]
    entry: InflightRequest


@contextmanager
def _pending_admission(tmp_path: Path) -> Iterator[_Admission]:
    """Accept a host-funded request without starting a provider or consuming its verdict."""
    _, key = _configured_gateway(
        tmp_path, base_url="http://127.0.0.1:1/v1", billing_source=BillingSource.HOST_MANAGED
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    records: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(
            settlement_required=False, maximum_pending_records=1
        ).model_dump_json(),
        records.append,
    )
    capture = CaptureController(collector, application_for=lambda _authorization: "capture-fixture")
    classifier = _Classifier("allow")
    control = NativeControlPlane(components, capture=capture, guardrails=_engine(classifier))
    admission = json.loads(
        control.admit(
            json.dumps({"raw_key": key, "body": _chat_body(), "capture_session_id": "poll-episode"})
        )
    )
    request_id = str(admission["request_id"])
    entry = control._accounting.entry(request_id)  # noqa: SLF001 - request lifecycle assertion.
    assert entry is not None and classifier.started.wait(2)
    assert entry.pending_capture is not None
    try:
        yield _Admission(control, classifier, capture, records, entry)
    finally:
        classifier.release.set()
        control.abandon(json.dumps({"request_id": request_id}))
        collector.settle(request_id, False, False)
        assert collector.close(2)
        components.write_ledger.close()


def _approve(admission: _Admission) -> str:
    """Complete the real classifier without polling the native capture release callback."""
    admission.classifier.release.set()
    session = admission.entry.guardrails
    assert session is not None
    deadline = time.monotonic() + 2
    while session.input_decision()["action"] == "pending" and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.input_decision() == {"action": "allow"}
    return json.dumps({"request_id": admission.entry.authorization.request_id})


def test_concurrent_approval_polls_begin_capture_once_with_request_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one poll registers approved content; model and session belong to that request."""
    with _pending_admission(tmp_path) as admission:
        argument = _approve(admission)
        begins: list[str] = []
        entered = threading.Event()
        release = threading.Event()
        original = admission.capture.begin

        def begin(
            authorization: AuthorizationSnapshot,
            request: GatewayRequest,
            model_id: str | None,
            *,
            session_id: str | None = None,
        ) -> bool:
            """Hold the real collector boundary while competing approval polls run."""
            begins.append(authorization.request_id)
            entered.set()
            assert release.wait(2)
            return original(authorization, request, model_id, session_id=session_id)

        monkeypatch.setattr(admission.capture, "begin", begin)
        with ThreadPoolExecutor(max_workers=8) as executor:
            first = executor.submit(admission.control.guardrail_input_status, argument)
            try:
                assert entered.wait(2)
                competing = list(
                    executor.map(admission.control.guardrail_input_status, [argument] * 8)
                )
                assert all(json.loads(result) == {"action": "pending"} for result in competing)
            finally:
                release.set()
            assert json.loads(first.result(timeout=2)) == {"action": "allow"}
            repeated = list(executor.map(admission.control.guardrail_input_status, [argument] * 16))
            assert all(json.loads(result) == {"action": "allow"} for result in repeated)
        request_id = admission.entry.authorization.request_id
        assert begins == [request_id]
        assert admission.entry.pending_capture is None
        admission.capture.native.settle(request_id, True, False)
        assert admission.capture.native.close(2)
        assert len(admission.records) == 1
        captured = CaptureRecord.model_validate_json(admission.records[0])
        assert captured.request.request_id == request_id
        assert captured.request.model_id == admission.entry.route.snapshot.exact_model_id
        assert captured.request.context["session_id"] == "poll-episode"
        assert captured.request.scope.identity_id == admission.entry.authorization.identity_id


def test_stale_approval_poll_cannot_release_after_capture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval read before another poll's failed registration must be rechecked."""
    with _pending_admission(tmp_path) as admission:
        argument = _approve(admission)
        assert admission.capture.native.close(1)
        session = admission.entry.guardrails
        assert session is not None
        original = session.input_decision
        paused = threading.Event()
        release = threading.Event()
        stale_thread: list[int] = []

        def decision() -> JsonObject:
            """Pause one stale read before the competing poll records its release failure."""
            result = original()
            if stale_thread == [threading.get_ident()] and not paused.is_set():
                paused.set()
                assert release.wait(2)
            return result

        def stale_poll() -> str:
            """Identify the one callback whose initial approval is deliberately stale."""
            stale_thread.append(threading.get_ident())
            return admission.control.guardrail_input_status(argument)

        monkeypatch.setattr(session, "input_decision", decision)
        with ThreadPoolExecutor(max_workers=1) as executor:
            stale = executor.submit(stale_poll)
            try:
                assert paused.wait(2)
                failed = json.loads(admission.control.guardrail_input_status(argument))
                assert failed["action"] == "error"
                assert failed["failure"]["safe_details"]["input_guardrail_denied"] is True
            finally:
                release.set()
            assert json.loads(stale.result(timeout=2)) == failed
        assert session.settlement_failure() is not None
        assert admission.records == []


@pytest.mark.parametrize("retained_abandonment", [False, True])
def test_abandon_during_capture_begin_discards_registration_and_pending_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retained_abandonment: bool
) -> None:
    """Capture capacity is released even if a closing request's ledger write needs retry."""
    with _pending_admission(tmp_path) as admission, monkeypatch.context() as patch:
        argument = _approve(admission)
        entered = threading.Event()
        release = threading.Event()
        original = admission.capture.begin

        def begin(
            authorization: AuthorizationSnapshot,
            request: GatewayRequest,
            model_id: str | None,
            *,
            session_id: str | None = None,
        ) -> bool:
            """Register real bounded capture memory, then race request abandonment."""
            accepted = original(authorization, request, model_id, session_id=session_id)
            entered.set()
            assert release.wait(2)
            return accepted

        patch.setattr(admission.capture, "begin", begin)
        if retained_abandonment:
            patch.setattr(
                admission.control._accounting._write_ledger,  # noqa: SLF001 - durable write boundary.
                "finish_request",
                Mock(side_effect=RuntimeError("private ledger failure")),
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            poll = executor.submit(admission.control.guardrail_input_status, argument)
            try:
                assert entered.wait(2)
                admission.control.abandon(argument)
                assert json.loads(admission.control.guardrail_input_status(argument)) == {
                    "action": "closed"
                }
            finally:
                release.set()
            if retained_abandonment:
                with pytest.raises(NativeBridgeError):
                    poll.result(timeout=2)
            else:
                assert json.loads(poll.result(timeout=2)) == {"action": "closed"}
        request_id = admission.entry.authorization.request_id
        assert admission.control._accounting.entry(request_id) is (  # noqa: SLF001
            admission.entry if retained_abandonment else None
        )
        assert admission.entry.pending_capture is None
        assert admission.records == []
        # The one-record budget is available immediately, not after a TTL or maintenance sweep.
        assert admission.capture.native.begin(_request_json())
        admission.capture.native.settle("request", False, False)


def test_abandon_pending_input_never_registers_capture(tmp_path: Path) -> None:
    """Disconnect or predispatch closure drops the context before any input verdict exists."""
    with _pending_admission(tmp_path) as admission:
        argument = json.dumps({"request_id": admission.entry.authorization.request_id})
        assert json.loads(admission.control.guardrail_input_status(argument)) == {
            "action": "pending"
        }
        admission.control.abandon(argument)
        assert admission.classifier.cancelled.wait(2)
        assert admission.entry.pending_capture is None
        assert json.loads(admission.control.guardrail_input_status(argument)) == {
            "action": "closed"
        }
        assert admission.capture.native.begin(_request_json())
        admission.capture.native.settle("request", False, False)
        assert admission.records == []


def test_predispatch_accounting_failure_drops_unapproved_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reservation closes admission without retaining its pending input context."""
    with _pending_admission(tmp_path) as admission:
        argument = json.dumps(
            {"request_id": admission.entry.authorization.request_id, "attempt_ordinal": 0}
        )
        monkeypatch.setattr(
            admission.control._accounting._write_ledger,  # noqa: SLF001 - real reservation boundary.
            "start_attempt",
            Mock(side_effect=RuntimeError("private reservation failure")),
        )
        with pytest.raises(NativeBridgeError):
            admission.control.start_attempt(argument)
        assert admission.classifier.cancelled.wait(2)
        assert admission.entry.pending_capture is None
        assert json.loads(admission.control.guardrail_input_status(argument)) == {
            "action": "closed"
        }
        assert admission.records == []
        assert admission.capture.native.begin(_request_json())
        admission.capture.native.settle("request", False, False)


@pytest.mark.parametrize("outcome", ["allow", "block", "full", "broken"])
def test_customer_funded_capture_begins_after_synchronous_input_approval(
    tmp_path: Path, outcome: str
) -> None:
    """A route that disables speculative generation still starts approved capture once."""
    _, key = _configured_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    records: list[str] = []
    begins: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(settlement_required=False).model_dump_json(), records.append
    )
    if outcome == "full":
        assert collector.close(1)

    def application_for(authorization: AuthorizationSnapshot) -> str:
        """Count retention attempts and exercise a sanitized capture policy error."""
        begins.append(authorization.request_id)
        if outcome == "broken":
            raise RuntimeError("private capture policy failure")
        return "capture-fixture"

    classifier = _Classifier(outcome)
    control = NativeControlPlane(
        components,
        capture=CaptureController(collector, application_for=application_for),
        guardrails=_engine(classifier),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_admit, control, key, _chat_body())
            try:
                assert classifier.started.wait(2)
                assert not future.done()
                assert begins == []
                classifier.release.set()
                if outcome == "allow":
                    admission = future.result(timeout=2)
                    assert admission["guardrail_input_pending"] is False
                    request_id = str(admission["request_id"])
                    assert begins == [request_id]
                    control.abandon(json.dumps({"request_id": request_id}))
                    collector.settle(request_id, True, False)
                else:
                    with pytest.raises(NativeBridgeError) as error:
                        future.result(timeout=2)
                    if outcome in {"full", "broken"}:
                        assert (
                            json.loads(error.value.public_error_json)["code"]
                            == "capture_unavailable"
                        )
                    assert len(begins) == (0 if outcome == "block" else 1)
            finally:
                classifier.release.set()
        with sqlite3.connect(components.ledger.database_path) as db:
            assert db.execute("select count(*) from gateway_attempts").fetchone() == (0,)
            assert db.execute("select terminal_state from gateway_requests").fetchall() == [
                ("cancelled" if outcome == "allow" else "failed",)
            ]
    finally:
        classifier.release.set()
        assert collector.close(2)
        components.write_ledger.close()
    assert len(records) == (1 if outcome == "allow" else 0)
    if records:
        captured = CaptureRecord.model_validate_json(records[0])
        assert captured.request.request_id == request_id
        assert captured.request.model_id == admission["exact_model_id"]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_rejected_search_releases_serial_capture_through_native_http(
    tmp_path: Path, surface: str, stream: bool
) -> None:
    """Repeated denied retrievals release capture capacity without dispatch or delivery."""
    records: list[str] = []
    registered: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(
            settlement_required=False, maximum_pending_records=1
        ).model_dump_json(),
        records.append,
    )

    def application_for(authorization: AuthorizationSnapshot) -> str:
        """Record the real serial registrations whose rejection must release capacity."""
        registered.append(authorization.request_id)
        return "capture-fixture"

    search = StaticWebSearchBackend(
        (GatewayWebSearchResult(url="https://example.test/result", title="withhold-marker"),)
    )
    with _provider(_content_chunk("unused") + _terminal_frames()) as (base_url, upstream):
        _, key = _configured_gateway(tmp_path, base_url=base_url)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(
            components,
            capture=CaptureController(collector, application_for=application_for),
            guardrails=_RetrievedGuard(),
            web_search=search,
        )
        port = _unused_port()
        ready = threading.Event()
        shutdown = exp_gateway_native.shutdown_handle()
        worker = threading.Thread(
            target=serve_native_gateway,
            args=(control,),
            kwargs={
                "host": "127.0.0.1",
                "port": port,
                "capture": collector,
                "shutdown": shutdown,
                "on_listening": ready.set,
                "graceful_timeout_seconds": 0.1,
            },
            daemon=True,
        )
        worker.start()
        try:
            assert ready.wait(5)
            path, body = _body(surface, stream)
            body["model"] = "coding:online"
            if surface == "messages":
                body["model"] = "coding"
                body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
            for _ in range(2):
                response = httpx.post(
                    f"http://127.0.0.1:{port}{path}",
                    headers={"authorization": f"Bearer {key}"},
                    json=body,
                    timeout=5,
                )
                assert response.status_code == 400, response.text
                assert "capture_unavailable" not in response.text
                assert collector.begin(_request_json()), (
                    "denied retrieval retained capture capacity"
                )
                collector.settle("request", False, False)
            assert len(registered) == len(search.queries) == 2
            assert upstream == []
            assert records == []
            with sqlite3.connect(components.ledger.database_path) as db:
                assert db.execute(
                    "SELECT terminal_state, web_search_requests FROM gateway_requests"
                ).fetchall() == [("failed", 1), ("failed", 1)]
                assert db.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone() == (0,)
        finally:
            shutdown.request_shutdown()
            worker.join(5)
            for request_id in [*registered, "request"]:
                collector.settle(request_id, False, False)
            assert collector.close(2)
            components.write_ledger.close()
            assert not worker.is_alive()
    assert records == []


@pytest.mark.parametrize("phase", ["post-search", "dispatch"])
@pytest.mark.parametrize("retained_settlement", [False, True])
def test_late_serial_guardrail_rejection_discards_capture_before_settlement_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, retained_settlement: bool
) -> None:
    """Both late denial paths free real collector capacity even while ledger writes fail."""

    class DisclosureGuard(_Guard):
        """Reject the gateway's tool serialization disclosure at final dispatch inspection."""

        async def inspect_input(
            self, *, request: GatewayRequest, check: GuardrailCheck
        ) -> ClassifierVerdict:
            """Allow initial input and reject only the subsequently expanded request."""
            self.block_input = (
                "parallel_tool_calls->emulated(serialized_by_gateway)" in request.ignored_parameters
            )
            return await super().inspect_input(request=request, check=check)

    key = _configure(tmp_path, "http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    records: list[str] = []
    registered: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(
            settlement_required=False, maximum_pending_records=1
        ).model_dump_json(),
        records.append,
    )

    def application_for(authorization: AuthorizationSnapshot) -> str:
        """Record capture registration before a later guardrail rejection."""
        registered.append(authorization.request_id)
        return "capture-fixture"

    guardrail = _RetrievedGuard() if phase == "post-search" else DisclosureGuard()
    control = NativeControlPlane(
        components,
        capture=CaptureController(collector, application_for=application_for),
        guardrails=guardrail,
        web_search=StaticWebSearchBackend(
            (GatewayWebSearchResult(url="https://example.test/result", title="withhold-marker"),)
        ),
    )
    body = json.loads(_chat_body(model="coding:online" if phase == "post-search" else "coding"))
    if phase == "dispatch":
        body["tools"] = [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        ]
        body["parallel_tool_calls"] = False
    try:
        with monkeypatch.context() as patch:
            if retained_settlement:
                patch.setattr(
                    control._accounting._write_ledger,  # noqa: SLF001 - durable failure injection.
                    "finish_request",
                    Mock(side_effect=RuntimeError("private ledger failure")),
                )
            with pytest.raises(NativeBridgeError) as error:
                _admit(control, key, json.dumps(body))
            assert json.loads(error.value.public_error_json)["code"] == "content_filter"
            assert len(registered) == 1
            assert len(guardrail.requests) == 2
            assert collector.begin(_request_json()), "denied request retained capture capacity"
            collector.settle("request", False, False)
            assert control._accounting.request_settlements.pending is retained_settlement  # noqa: SLF001
        control._accounting.request_settlements.retry(64)  # noqa: SLF001 - durable recovery check.
        assert not control._accounting.request_settlements.pending  # noqa: SLF001
        with sqlite3.connect(components.ledger.database_path) as db:
            assert db.execute(
                "SELECT terminal_state, web_search_requests FROM gateway_requests"
            ).fetchall() == [("failed", int(phase == "post-search"))]
            assert db.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone() == (0,)
    finally:
        for request_id in [*registered, "request"]:
            collector.settle(request_id, False, False)
        assert collector.close(2)
        components.write_ledger.close()
    assert records == []

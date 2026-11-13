"""Capture eligibility follows the request-owned input gate through real native HTTP."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.sqlite.native import connect as connect_sqlite
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureRecord,
)
from exp.runtime.gateway.native_capture_test import _request_json
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.launch_test import _unused_port
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _RetrievedGuard
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _scenario
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", ["allow", "block", "full"])
def test_native_capture_follows_the_final_moderation_release(
    tmp_path: Path, surface: str, stream: bool, outcome: str
) -> None:
    """The native owner preserves required capture and financial cleanup at input release."""
    records: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(settlement_required=True, maximum_pending_records=1).model_dump_json(),
        records.append,
    )
    capture = CaptureController(collector, application_for=lambda _a: "capture-fixture")
    if outcome == "full":
        assert collector.begin(_request_json())
    try:
        with (
            _scenario(
                tmp_path,
                blocked=outcome == "block",
                detector_status=500 if outcome == "error" else 200,
                timeout_ms=500 if outcome == "timeout" else 3000,
                capture=capture,
            ) as scenario,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            pending = executor.submit(scenario.request, surface, stream=stream)
            try:
                assert scenario.moderator.started.wait(2)
                assert scenario.provider.started.wait(2)
                assert not pending.done()
                assert records == []
                if outcome != "full":
                    # Pending classification has not consumed the sole content slot.
                    assert collector.begin(_request_json())
                    collector.settle("request", False, False)
                if outcome not in {"timeout", "grace"}:
                    scenario.moderator.release.set()
                    assert scenario.moderator.finished.wait(2)
                scenario.provider.release.set()
                response = pending.result(timeout=4)
            finally:
                scenario.moderator.release.set()
                scenario.provider.release.set()
            allowed = outcome == "allow"
            assert response.status_code == (
                200 if allowed else 503 if outcome == "full" else 400
            ), response.text
            assert ("provider-marker" in response.text) is allowed
            assert scenario.control.close(timeout_seconds=2)
            with connect_sqlite(scenario.database) as db:
                request_id, state = db.execute(
                    "SELECT request_id,terminal_state FROM gateway_requests"
                ).fetchone()
                attempts = db.execute(
                    "SELECT state,budget_settled_nano_usd FROM gateway_attempts"
                ).fetchall()
            assert state == ("completed" if allowed else "failed")
            assert len(attempts) == 1
            assert attempts[0][1] > 0 if allowed else attempts[0][1] == 0
            # Financial drain can precede the streaming task dropping its retained owner.
            deadline = time.monotonic() + 2
            while scenario.control._accounting.native.owns_request(request_id):  # noqa: SLF001
                assert time.monotonic() < deadline, "stream retained its completed request owner"
                time.sleep(0.005)
            assert not scenario.control._accounting.native.owns_request(request_id)  # noqa: SLF001
            collector.settle(request_id, allowed, allowed)
            collector.settle("request", False, False)
            assert collector.close(2)
            parsed = [CaptureRecord.model_validate_json(record) for record in records]
            captured_content = allowed and outcome != "full"
            assert bool(parsed) is captured_content
            assert len({record.request.request_id for record in parsed}) == int(captured_content)
            assert sum(record.response is not None for record in parsed) == int(captured_content)
            if parsed:
                captured = parsed[0]
                assert captured.request.request_id == request_id
                assert captured.request.model_id is not None
                assert captured.request.scope.identity_id == "default"
                assert "Synthetic moderation fixture." in captured.model_dump_json()
    finally:
        collector.settle("request", False, False)
        collector.close(2)


@pytest.mark.parametrize("broken", [False, True])
def test_native_admission_failure_releases_capture_and_financial_ownership(
    tmp_path: Path, broken: bool
) -> None:
    """A failed capture preparation is sanitized and cannot leave accepted work open."""
    records: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(maximum_pending_records=1).model_dump_json(), records.append
    )

    def application_for(_authorization: AuthorizationSnapshot) -> str:
        """Inject the private host policy failure only before ownership transfer."""
        if broken:
            raise RuntimeError("private-capture-policy-error")
        return "capture-fixture"

    capture = CaptureController(collector, application_for=application_for)
    try:
        with _scenario(tmp_path, capture=capture) as scenario:
            scenario.moderator.release.set()
            scenario.provider.release.set()
            response = scenario.request(stream=False)
            assert response.status_code == (500 if broken else 200), response.text
            assert "private-capture-policy-error" not in response.text
            assert scenario.control.close(timeout_seconds=2)
            with connect_sqlite(scenario.database) as db:
                request_id, terminal = db.execute(
                    "SELECT request_id,terminal_state FROM gateway_requests"
                ).fetchone()
                attempts = db.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0]
            assert terminal == ("failed" if broken else "completed")
            assert attempts == (0 if broken else 1)
            collector.settle(request_id, not broken, False)
            assert collector.close(2)
            assert bool(records) is not broken
            assert len({json.loads(record)["request"]["request_id"] for record in records}) == (
                0 if broken else 1
            )
    finally:
        collector.close(2)


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
            with connect_sqlite(components.ledger.database_path) as db:
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

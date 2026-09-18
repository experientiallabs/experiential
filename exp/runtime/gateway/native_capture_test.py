"""Shared native capture contracts and real-socket serving isolation."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureDeliveryLimits,
    CaptureRecord,
    CaptureSseResponse,
)
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)

native = pytest.importorskip("exp_gateway_native")


def _request_json() -> str:
    """Return the versioned boundary's minimum authenticated request."""
    return json.dumps(
        {
            "request_id": "request",
            "scope": {"organization_id": "org", "identity_id": "identity", "application_id": "app"},
            "protocol": "chat_completions",
            "model_id": "model",
            "context": {"schema_version": 1, "request": {"messages": []}},
        }
    )


def test_python_sink_runs_off_caller_thread_and_close_releases_gil() -> None:
    """A sink requiring Python can finish while the caller waits on the Rust drain."""
    records: list[str] = []
    threads: list[int] = []

    def write(record: str) -> None:
        """Observe destination execution without any provider or SQL dependency."""
        records.append(record)
        threads.append(threading.get_ident())

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), write)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert threads and threads[0] != threading.get_ident()
    assert CaptureRecord.model_validate_json(records[0]).request.scope.identity_id == "identity"
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


def test_python_sink_failure_never_logs_exception_content(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Database exception strings may carry private parameters and must be discarded."""

    def fail(_record: str) -> None:
        """Simulate a storage rejection containing sensitive context."""
        raise RuntimeError("private SQL parameter that must not be logged")

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), fail)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert collector.counts() == (0, 0, 0, 1, 0, 0)
    assert "private SQL" not in "".join(capfd.readouterr())


def test_python_and_rust_configuration_fail_closed() -> None:
    """Both entry points reject invalid bounds and unknown configuration."""
    with pytest.raises(ValueError):
        CaptureDeliveryLimits(maximum_bytes=1)
    with pytest.raises(ValueError):
        CaptureConfiguration(maximum_pending_bytes=1)
    with pytest.raises(ValueError):
        native.CaptureCollector('{"unknown": true}', lambda _: None)


def test_accepted_routing_failure_keeps_effective_prompt_without_inventing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture begins after acceptance but before a route can fail without dispatch."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _manager, raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    components = load_gateway_components(tmp_path)
    records: list[str] = []
    authorized: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)

    def application_for(authorization: AuthorizationSnapshot) -> str:
        """Record the authenticated request for an explicit hosted terminal verdict."""
        authorized.append(authorization.request_id)
        return "application"

    capture = CaptureController(collector, application_for=application_for)
    control = NativeControlPlane(components, capture=capture)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        """Reject route construction without making a provider call."""
        raise GatewayRoutingError("unavailable route")

    monkeypatch.setattr(control, "_resolve_route", unavailable)
    try:
        with pytest.raises(NativeBridgeError):
            control.admit(
                json.dumps(
                    {
                        "raw_key": raw_key,
                        "body": json.dumps(
                            {
                                "model": "coding",
                                "messages": [{"role": "user", "content": "retained task"}],
                            }
                        ),
                    }
                )
            )
        assert len(authorized) == 1
        collector.settle(authorized[0], True, False)
        assert collector.close(1)
    finally:
        collector.close(1)
        components.write_ledger.close()
    parsed = CaptureRecord.model_validate_json(records[0])
    assert parsed.request.model_id is None
    assert parsed.response is None
    assert "retained task" in records[0]


@pytest.mark.parametrize("policy", ["local", "hosted", "off", "broken"])
def test_real_http_surfaces_use_one_collector_without_affecting_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Collect Chat, Responses and Messages JSON/SSE through native HTTP, not a fixture tap."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    records: list[str] = []
    configuration = CaptureConfiguration(settlement_required=policy == "hosted")
    collector = native.CaptureCollector(configuration.model_dump_json(), records.append)

    def application_for(authorization: AuthorizationSnapshot) -> str | None:
        """Exercise host policy separately from content assembly and persistence."""
        assert authorization.identity_id == "default"
        if policy == "broken":
            raise RuntimeError("private policy details")
        return None if policy == "off" else "application"

    capture = CaptureController(collector, application_for=application_for)
    port = _unused_port()
    shutdown = native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve the real data plane and preserve startup failures for assertions."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surfaced after bounded shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        for surface in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                payload: dict[str, object] = {"model": "coding", "stream": stream}
                if surface == "responses":
                    payload["input"] = "capture task"
                else:
                    payload["messages"] = [{"role": "user", "content": "capture task"}]
                if surface == "messages":
                    payload["max_tokens"] = 128
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    headers={"authorization": f"Bearer {raw_key}"},
                    json=payload,
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                assert "hello " in response.text and "world" in response.text
                if policy == "hosted":
                    collector.settle(response.headers["x-request-id"], True, True)
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not worker.is_alive()
    assert collector.close(1)
    if policy in {"off", "broken"}:
        assert records == []
        return
    parsed = [CaptureRecord.model_validate_json(value) for value in records]
    completed = [record for record in parsed if record.response is not None]
    assert len(completed) == 6
    assert sum(record.response.kind == "json" for record in completed if record.response) == 3
    assert all(
        not record.response.truncated and not record.response.client_disconnected
        for record in completed
        if isinstance(record.response, CaptureSseResponse)
    )
    assert {record.request.protocol for record in completed} == {
        "chat_completions",
        "responses",
        "messages",
    }
    assert all(record.request.scope.identity_id == "default" for record in completed)
    assert all(record.request.model_id is not None for record in completed)
    assert "provider-secret" not in "".join(records)
    assert raw_key not in "".join(records)

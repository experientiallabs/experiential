"""Local capture defaults use authenticated grants and have an explicit opt-out."""

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.project import ProjectConfig, ProjectStore
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_test import _save
from exp.common.traces.trace_test import _trace
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.local_capture import (
    local_capture_configuration,
    local_capture_path,
    open_local_capture,
)
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)
from exp.simulation.ingest.gateway import load_gateway_capture

exp_gateway_native = pytest.importorskip("exp_gateway_native")


def test_local_capture_defaults_on_for_own_provider_keys_and_ghost_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture is available before any project or hosted account exists."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "fixture-provider-key")
    manager, _raw_key = _configure_gateway(tmp_path, base_url="http://127.0.0.1:1/v1")
    configuration = local_capture_configuration(tmp_path)
    assert configuration is not None
    assert configuration.database_path == local_capture_path(tmp_path)
    assert configuration.database_path != manager.database_path
    assert len(configuration.bindings) == 1
    assert configuration.bindings[0].policy.enabled
    identity = configuration.bindings[0].policy.scope.user_id
    assert identity == manager.grants()[0].identity_id
    assert local_capture_configuration(tmp_path, ghost=True) is None
    manager.disable_identity(identity_id=identity)
    assert local_capture_configuration(tmp_path) is None


@pytest.mark.parametrize("ghost", [False, True])
def test_real_gateway_traffic_reopens_as_scoped_build_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ghost: bool
) -> None:
    """Drive real JSON/SSE sockets and consume durable traffic after graceful shutdown."""
    retained = None
    if not ghost:
        project = ProjectStore(tmp_path, "capture-project")
        project.initialize(ProjectConfig(project_id="capture-project"))
        project.write_review({"checkpoint": "retained"})
        retained = _save(SQLiteTraceStore(local_capture_path(tmp_path)), (_trace(),))
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1", supports_tools=True
    )
    components = load_gateway_components(tmp_path)
    capture = local_capture_configuration(tmp_path, ghost=ghost)
    controller = open_local_capture(capture)
    port = _unused_port()
    shutdown = exp_gateway_native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve one actual native runtime, retaining startup failures for the assertion."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=controller),
                host="127.0.0.1",
                port=port,
                capture=None if controller is None else controller.native,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - checked after worker shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        for surface in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                body: dict[str, object] = {"model": "coding", "stream": stream}
                if surface == "responses":
                    body["input"] = "hello capture"
                elif surface == "messages":
                    body["messages"] = [{"role": "user", "content": "hello capture"}]
                    body["max_tokens"] = 128
                else:
                    body["messages"] = [{"role": "user", "content": "hello capture"}]
                    body["tools"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Find a record",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ]
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    json=body,
                    headers={"Authorization": f"Bearer {raw_key}"},
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                if surface == "responses" and not stream:
                    continued = httpx.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": "coding",
                            "input": "next question",
                            "previous_response_id": response.json()["id"],
                        },
                        headers={"Authorization": f"Bearer {raw_key}"},
                        timeout=10,
                    )
                    assert continued.status_code == 200, continued.text
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not worker.is_alive()
    database = local_capture_path(tmp_path)
    if ghost:
        assert not database.exists()
    else:
        assert retained is not None
        assert SQLiteTraceStore(database).read_import(retained.import_id).traces == (_trace(),)
        assert project.load_project().project_id == "capture-project"
        assert project.read_review() == {"checkpoint": "retained"}
        result = load_gateway_capture(database, identity_id="default")
        assert not result.issues
        assert len(result.traces) == 7
        assert sum(bool(trace.tools) for trace in result.traces) == 2
        continuations = [
            trace for trace in result.traces if trace.initial_context["parent_response_id"]
        ]
        assert len(continuations) == 1
        context = continuations[0].initial_context["gateway_request"]
        assert isinstance(context, dict)
        request = context["request"]
        assert isinstance(request, dict)
        messages = request["messages"]
        assert isinstance(messages, list)
        assert len(messages) >= 3
        assert load_gateway_capture(database, identity_id="other").traces == ()
        serialized = "".join(trace.model_dump_json() for trace in result.traces)
        assert raw_key not in serialized
        assert "provider-secret" not in serialized

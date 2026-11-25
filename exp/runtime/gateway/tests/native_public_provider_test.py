"""Host provider labels and the withheld deployment id traverse the served native gateway."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.ingest.conversion import load_gateway_capture
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

exp_gateway_native = pytest.importorskip("exp_gateway_native")

_LABEL = "example_cloud"
_SURFACES = ("chat/completions", "responses", "messages")


def _serve(root: Path, *, publish: bool) -> Iterator[tuple[str, str]]:
    """Serve the seeded root over a loopback provider with local capture on.

    Args:
        root: Initialized EXP root with one ``coding`` alias.
        publish: The serving switch for ``x-gateway-deployment``.

    Yields:
        The gateway origin and the caller's key.
    """
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        root, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(root)
    controller = open_local_capture(local_capture_configuration(root))
    assert controller is not None
    port = _unused_port()
    shutdown = exp_gateway_native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Serve the real native plane, retaining startup failures for the assertion."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=controller),
                host="127.0.0.1",
                port=port,
                capture=controller.native,
                shutdown=shutdown,
                publish_deployment_header=publish,
            )
        except BaseException as error:  # noqa: BLE001 - checked after shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        yield f"http://127.0.0.1:{port}", raw_key
    finally:
        shutdown.request_shutdown()
        worker.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not worker.is_alive()


@pytest.fixture
def labelled_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have the host label every route's lead rung, as a resolver would."""
    member = ExecutionSnapshot.public_provider_member

    def labelled(self: ExecutionSnapshot, deployment_id: str) -> dict[str, str]:
        """Answer as if the route's lead rung carried the host's label."""
        snapshot = self.model_copy(update={"public_providers": {self.deployment_ids[0]: _LABEL}})
        return member(snapshot, deployment_id)

    monkeypatch.setattr(ExecutionSnapshot, "public_provider_member", labelled)


def _body(surface: str, *, stream: bool) -> JsonObject:
    """One minimal request of ``surface``."""
    body: JsonObject = {"model": "coding", "stream": stream}
    if surface == "responses":
        body["input"] = "hello"
    else:
        body["messages"] = [{"role": "user", "content": "hello"}]
    if surface == "messages":
        body["max_tokens"] = 128
    return body


def _served_headers(origin: str, key: str) -> list[httpx.Headers]:
    """Send every surface buffered and streamed; return each answer's headers.

    A streamed answer's headers are read at stream start, before any body byte.
    """
    answered: list[httpx.Headers] = []
    with httpx.Client(timeout=10, headers={"Authorization": f"Bearer {key}"}) as client:
        for surface in _SURFACES:
            for stream in (False, True):
                with client.stream(
                    "POST", f"{origin}/v1/{surface}", json=_body(surface, stream=stream)
                ) as response:
                    assert response.status_code == 200, response.read()
                    answered.append(response.headers)
                    response.read()
    return answered


@pytest.mark.usefixtures("labelled_routes")
def test_a_labelled_route_names_the_label_and_withholds_the_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stream starts and buffered answers carry the label; capture keeps the deployment."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    served = _serve(tmp_path, publish=False)
    origin, key = next(served)
    try:
        answered = _served_headers(origin, key)
    finally:
        next(served, None)
    assert len(answered) == 2 * len(_SURFACES)
    for headers in answered:
        assert headers["x-gateway-provider"] == _LABEL
        assert "x-gateway-deployment" not in headers
    traces = load_gateway_capture(local_capture_path(tmp_path), identity_id="default").traces
    assert len(traces) == len(answered)
    deployments = {trace.initial_context["deployment_id"] for trace in traces}
    # The fixture deploys alias `coding` as deployment `coding`.
    assert deployments == {"coding"}


def test_an_unlabelled_route_keeps_the_provider_id_and_the_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without labels or the switch, answers carry today's routing headers."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    served = _serve(tmp_path, publish=True)
    origin, key = next(served)
    try:
        answered = _served_headers(origin, key)
    finally:
        next(served, None)
    for headers in answered:
        assert headers["x-gateway-provider"] == "openai-compatible"
        assert headers["x-gateway-deployment"] == "coding"

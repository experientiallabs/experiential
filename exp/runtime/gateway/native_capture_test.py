"""Shared native capture contracts and real-socket serving isolation."""

import json
import socket
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import ClientConnection, connect

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.ingest.conversion import load_gateway_capture
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.local_capture_config import CaptureBinding
from exp.runtime.gateway.local_capture_config import (
    CaptureConfiguration as LocalCaptureConfiguration,
)
from exp.runtime.gateway.local_capture_contracts import CapturePolicy, LocalCaptureScope
from exp.runtime.gateway.local_capture_store import LocalCaptureStore
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureDeliveryLimits,
    CaptureRecord,
    CaptureRecordV1,
    CaptureSseResponse,
    read_capture_record_json,
)
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.tests.chain_authority_fixture_test import (
    chain_components,
    publish_authored_chain_fixture,
)
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)
from exp.runtime.gateway.tests.native_chat_images_test import _PNG_BASE64
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames

native = pytest.importorskip("exp_gateway_native")


@pytest.mark.parametrize("winner", ["child", "root_suffix"])
@pytest.mark.parametrize(
    ("destination", "policy"),
    [
        (destination, policy)
        for destination in ("hosted", "batch")
        for policy in ("keep", "off", "byok", "prompt_only", "cancel")
    ]
    + [("sqlite", "keep"), ("sqlite", "cancel")]
    + [
        (destination, policy)
        for destination in ("checkpoint", "batch_checkpoint")
        for policy in ("keep", "prompt_only", "cancel")
    ],
)
def test_nested_capture_separates_root_from_winner_and_does_not_recapture_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
    destination: str,
    policy: str,
) -> None:
    """Actual HTTP traversal keeps root input identity and permission-gated winning provenance."""
    calls: list[str] = []
    provider_stopped = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        """Serve two root providers around a conditional child using one local listener."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP handler contract.
            """Fail pre-output until the configured semantic winner is reached."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            child = self.path.startswith("/child/")
            first = self.path.startswith("/first/")
            if first or (child and winner == "root_suffix"):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"fixture unavailable"}}')
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            if policy == "cancel":
                try:
                    self.wfile.write(_content_chunk("winner"))
                    self.wfile.flush()
                    while True:
                        self.wfile.write(_content_chunk("more"))
                        self.wfile.flush()
                        time.sleep(0.02)
                except OSError:
                    provider_stopped.set()
            else:
                self.wfile.write(_content_chunk("winner") + _terminal_frames())

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{provider.server_port}"
    manager, key = _configured_pool_gateway(
        tmp_path, base_urls=(origin + "/first/v1", origin + "/child/v1")
    )
    authored = load_model_catalog(tmp_path / "models.toml")
    models = dict(authored.models)
    alpha, beta = models["alpha"], models["beta"]
    assert alpha.gateway is not None and beta.gateway is not None
    models["beta"] = beta.model_copy(
        update={"gateway": beta.gateway.model_copy(update={"exact_model_id": "child-exact"})}
    )
    models["suffix"] = alpha.model_copy(update={"connection": "suffix"})
    connections = dict(authored.connections)
    connections["suffix"] = connections[alpha.connection].model_copy(
        update={"base_url": origin + "/suffix/v1"}
    )
    authored = authored.model_copy(
        update={
            "models": models,
            "connections": connections,
            "gateway_pools": {
                "alpha": authored.gateway_pools["coding"].model_copy(
                    update={"deployment_aliases": ("alpha", "suffix")}
                )
            },
            "gateway_model_chains": {
                "model-revision-exact": GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="alpha",
                    revision="capture-chain",
                    rungs=(
                        GatewayDeploymentRung(deployment_id="alpha"),
                        GatewayModelReferenceRung(model_id="child-exact"),
                        GatewayDeploymentRung(deployment_id="suffix"),
                    ),
                )
            },
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="capture-chain", pool_id="alpha")
    components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    records: list[str] = []
    scope = LocalCaptureScope(user_id=manager.grants()[0].identity_id, application_id="gateway")
    traffic_path = tmp_path / "nested-traffic.db"

    def write_batch(encoded: tuple[str, ...]) -> list[bool]:
        """Acknowledge only strict current records with retained root and winner facts."""
        for value in encoded:
            CaptureRecord.model_validate_json(value)
        records.extend(encoded)
        return [True] * len(encoded)

    if destination in {"batch", "batch_checkpoint"}:
        collector = native.CaptureCollector.batched(
            CaptureConfiguration().model_dump_json(), write_batch
        )
    elif destination == "sqlite":
        local = LocalCaptureConfiguration(
            database_path=traffic_path,
            bindings=(
                CaptureBinding(alias="coding", policy=CapturePolicy(scope=scope, enabled=True)),
            ),
        )
        collector = native.CaptureCollector.sqlite(
            CaptureConfiguration(settlement_required=False).model_dump_json(),
            local.model_dump_json(),
        )
        assert collector is not None
    else:
        collector = native.CaptureCollector(
            CaptureConfiguration().model_dump_json(), records.append
        )
    capture = CaptureController(
        collector, application_for=lambda _auth: None if policy == "off" else scope.application_id
    )
    port, shutdown = _unused_port(), native.shutdown_handle()
    errors: list[BaseException] = []
    control = NativeControlPlane(components, capture=capture)
    admit = control.admit

    def funding_admit(argument: str) -> str:
        """Exercise hosted checkpointing without changing provider or model-stage identity."""
        result = json.loads(admit(argument))
        if destination in {"checkpoint", "batch_checkpoint"}:
            for wire in result["route"]:
                wire["billing_customer_managed"] = False
        return json.dumps(result)

    monkeypatch.setattr(control, "admit", funding_admit)

    def serve() -> None:
        """Run the real data plane with a hosted capture sink and explicit shutdown."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surface worker failure below.
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        headers = {"authorization": f"Bearer {key}", "Idempotency-Key": "capture-once"}
        body = {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}
        expected = "child-exact" if winner == "child" else "model-revision-exact"
        if policy == "cancel":
            headers.pop("Idempotency-Key")
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json={**body, "stream": True},
                timeout=20,
            ) as response:
                assert response.status_code == 200
                assert response.headers["x-gateway-canonical-model"] == expected
                request_id = response.headers["x-request-id"]
                for line in response.iter_lines():
                    if "winner" in line:
                        break
            count = len(calls)
            assert provider_stopped.wait(3)
            collector.settle(request_id, True, True)
            assert len(calls) == count
        else:
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=20,
            )
            assert response.status_code == 200, response.text
            assert response.headers["x-gateway-canonical-model"] == expected
            count = len(calls)
            collector.settle(response.headers["x-request-id"], policy != "byok", policy == "keep")
            replay = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=20,
            )
            assert replay.content == response.content and len(calls) == count
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        manager.close()
        provider.shutdown()
        provider.server_close()
        thread.join(5)
    assert not errors and not worker.is_alive()
    assert collector.close(1)
    if destination == "sqlite":
        rows = LocalCaptureStore(traffic_path, scope).read_after()
        assert records == []
        if policy == "cancel":
            assert rows == ()
            assert load_gateway_capture(traffic_path, identity_id=scope.user_id).traces == ()
        else:
            assert len(rows) == 1
            experience = rows[0].experience
            assert experience.schema_version == 1
            assert experience.provenance.model_id == "model-revision-exact"
            output = experience.request["exp_capture_output"]
            assert isinstance(output, dict)
            assert output["canonical_model_id"] == expected
            assert output["gemini_thought_parts_truncated"] is None
            ingested = load_gateway_capture(traffic_path, identity_id=scope.user_id)
            assert not ingested.issues and len(ingested.traces) == 1
            assert ingested.traces[0].initial_context["model_id"] == "model-revision-exact"
            assert ingested.traces[0].initial_context["canonical_model_id"] == expected
        assert (
            LocalCaptureStore(
                traffic_path, LocalCaptureScope(user_id="other", application_id="gateway")
            ).read_after()
            == ()
        )
    elif policy in {"off", "byok"}:
        assert records == []
    else:
        if destination in {"checkpoint", "batch_checkpoint"}:
            assert len(records) == 2
            checkpoint = CaptureRecord.model_validate_json(records[0])
            assert checkpoint.request.model_id == "model-revision-exact"
            assert checkpoint.canonical_model_id is None and checkpoint.deployment_id is None
            assert checkpoint.response is None and checkpoint.metrics is None
            assert checkpoint.gemini_thought_parts_truncated is None
            assert checkpoint.request.request_id == json.loads(records[1])["request"]["request_id"]
        else:
            assert len(records) == 1
        record = json.loads(records[-1])
        assert record["request"]["model_id"] == "model-revision-exact"
        assert record.get("canonical_model_id") == expected
        assert record["deployment_id"] == ("beta" if winner == "child" else "suffix")
        assert (record["response"] is not None) is (policy in {"keep", "cancel"})
        assert (record["metrics"] is not None) is (policy in {"keep", "cancel"})
        if policy == "cancel":
            assert record["response"]["client_disconnected"]
            assert not record["metrics"]["usage_complete"]
            assert record["metrics"]["terminal_at"] is None


@pytest.mark.parametrize("destination", ["off", "hosted", "batch", "sqlite"])
@pytest.mark.parametrize("shape", ["large", "fragments", "signed_text", "signed_image", "oversize"])
def test_gemini_capture_only_evidence_never_retries_or_fails_visible_inference(
    tmp_path: Path,
    destination: str,
    shape: str,
) -> None:
    """Optional telemetry limits never consume the refusal or semantic output budgets."""
    calls: list[str] = []

    class Gemini(BaseHTTPRequestHandler):
        """Emit finite valid private evidence before visible text and exact usage."""

        def do_POST(self) -> None:  # noqa: N802 - standard handler contract.
            """Serve one genuine Gemini stream with no outbound provider connection."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            if shape == "fragments":
                parts = [{"thought": True, "text": "summary"} for _ in range(257)]
            elif shape == "signed_text":
                parts = [{"text": "ok", "thoughtSignature": "s" * 65_537}]
            elif shape == "signed_image":
                parts = [
                    {
                        "inlineData": {"mimeType": "image/png", "data": _PNG_BASE64},
                        "thoughtSignature": "s" * 65_537,
                    }
                ]
            else:
                parts = [{"thought": True, "text": "x" * (65_537 if shape == "large" else 200_000)}]
            if shape != "signed_text":
                parts.append({"text": "ok"})
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                (
                    "data: "
                    + json.dumps(
                        {
                            "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
                        }
                    )
                    + "\n\n"
                ).encode()
            )

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Gemini)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, key = _configured_pool_gateway(tmp_path, provider="gemini")
    authored = load_model_catalog(tmp_path / "models.toml")
    models = dict(authored.models)
    beta = models["beta"]
    assert beta.gateway is not None
    models["beta"] = beta.model_copy(
        update={"gateway": beta.gateway.model_copy(update={"exact_model_id": "eligible-child"})}
    )
    authored = authored.model_copy(
        update={
            "models": models,
            "gateway_pools": {},
            "gateway_model_chains": {
                "model-revision-exact": GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="alpha",
                    revision="gemini-capture",
                    rungs=(
                        GatewayDeploymentRung(deployment_id="alpha"),
                        GatewayModelReferenceRung(model_id="eligible-child"),
                    ),
                )
            },
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="gemini-capture", pool_id="alpha")
    components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    records: list[str] = []
    configuration = CaptureConfiguration(maximum_response_bytes=100_000, settlement_required=False)
    scope = LocalCaptureScope(user_id=manager.grants()[0].identity_id, application_id="gateway")
    traffic_path = tmp_path / "gemini-traffic.db"

    def write_batch(encoded: tuple[str, ...]) -> list[bool]:
        """Retain bounded schema2 evidence and acknowledge each prepared record independently."""
        for value in encoded:
            CaptureRecord.model_validate_json(value)
        records.extend(encoded)
        return [True] * len(encoded)

    if destination == "batch":
        collector = native.CaptureCollector.batched(configuration.model_dump_json(), write_batch)
    elif destination == "sqlite":
        local = LocalCaptureConfiguration(
            database_path=traffic_path,
            bindings=(
                CaptureBinding(alias="coding", policy=CapturePolicy(scope=scope, enabled=True)),
            ),
        )
        collector = native.CaptureCollector.sqlite(
            configuration.model_dump_json(), local.model_dump_json()
        )
        assert collector is not None
    elif destination == "hosted":
        collector = native.CaptureCollector(configuration.model_dump_json(), records.append)
    else:
        collector = None
    capture = (
        CaptureController(collector, application_for=lambda _auth: scope.application_id)
        if collector
        else None
    )

    class Plane(NativeControlPlane):
        """Keep real Gemini preflight and alter only its test transport destination."""

        def admit(self, argument: str) -> str:
            """Redirect the already-validated Gemini provider request to loopback."""
            admitted = json.loads(super().admit(argument))
            for wire in admitted["route"]:
                assert wire["dialect"] == "gemini_generate_content"
                wire["url"] = (
                    f"http://127.0.0.1:{provider.server_port}/{wire['deployment_id']}/generate"
                )
            return json.dumps(admitted)

    port, shutdown = _unused_port(), native.shutdown_handle()
    errors: list[BaseException] = []

    def serve() -> None:
        """Run capture-on/off through the same actual native HTTP path."""
        try:
            serve_native_gateway(
                Plane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surface worker failure below.
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json={"model": "coding", "messages": [{"role": "user", "content": "hello"}]},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        assert response.json()["usage"]["prompt_tokens"] == 3
        assert response.json()["usage"]["completion_tokens"] == 2
        if shape == "signed_image":
            assert response.json()["choices"][0]["message"]["images"]
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        manager.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(5)
    assert not errors and not worker.is_alive()
    assert calls == ["/alpha/generate"]
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute(
            "SELECT deployment_id,state FROM gateway_attempts"
        ).fetchall() == [("alpha", "completed")]
    if collector:
        assert collector.close(1)
        if destination == "sqlite":
            assert records == []
            rows = LocalCaptureStore(traffic_path, scope).read_after()
            assert len(rows) == 1
            output = rows[0].experience.request["exp_capture_output"]
            assert isinstance(output, dict)
            assert output["canonical_model_id"] == "model-revision-exact"
            assert output["gemini_thought_parts_truncated"] is (shape == "oversize")
            parts = output["gemini_thought_parts"]
            assert isinstance(parts, list)
            ingested = load_gateway_capture(traffic_path, identity_id=scope.user_id)
            assert not ingested.issues and len(ingested.traces) == 1
            assert ingested.traces[0].initial_context["capture_output"] == output
        else:
            assert len(records) == 1
            record = CaptureRecord.model_validate_json(records[0])
            assert record.response is not None and record.metrics is not None
            assert record.gemini_thought_parts_truncated is (shape == "oversize")
            parts = record.gemini_thought_parts
        assert len(parts) == (0 if shape == "oversize" else 257 if shape == "fragments" else 1)
    else:
        assert records == []


@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("shape", ["thought", "signed_text", "signature_only", "image", "terminal"])
def test_gemini_disconnect_meter_is_independent_of_optional_capture(
    tmp_path: Path, capture_enabled: bool, shape: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """Real private thought accounting neither becomes output nor depends on capture consent."""
    closed = threading.Event()
    calls: list[str] = []
    private_text = "private-meter-canary"

    class Gemini(BaseHTTPRequestHandler):
        """Serve one bounded Gemini prefix then wait for the caller's cancellation."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP handler contract.
            """Emit actual typed Gemini parts and observe upstream close without another attempt."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            parts: list[JsonObject] = [{"text": "visible"}]
            if shape in {"thought", "terminal"}:
                parts.insert(
                    0,
                    {"thought": True, "text": private_text, "thoughtSignature": "secret-signature"},
                )
            elif shape == "signed_text":
                parts[0]["thoughtSignature"] = "secret-signature"
            elif shape == "signature_only":
                parts.insert(0, {"thoughtSignature": "secret-signature"})
            else:
                parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": _PNG_BASE64}})
            candidate: dict[str, object] = {"content": {"parts": parts}}
            payload: dict[str, object] = {"candidates": [candidate]}
            if shape == "terminal":
                candidate["finishReason"] = "STOP"
                payload["usageMetadata"] = {"promptTokenCount": 13, "candidatesTokenCount": 7}
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                self.wfile.flush()
                while True:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    time.sleep(0.01)
            except OSError:
                closed.set()

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Gemini)
    serving_provider = threading.Thread(target=provider.serve_forever, daemon=True)
    serving_provider.start()
    manager, key = _configured_pool_gateway(tmp_path, provider="gemini")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    records: list[str] = []
    collector = (
        native.CaptureCollector(
            CaptureConfiguration(settlement_required=False).model_dump_json(), records.append
        )
        if capture_enabled
        else None
    )
    capture = (
        CaptureController(collector, application_for=lambda _auth: "app") if collector else None
    )
    settled = threading.Event()
    raw_settlements: list[JsonObject] = []

    class Plane(NativeControlPlane):
        """Keep real admission and accounting, substituting only loopback transport."""

        def admit(self, argument: str) -> str:
            """Send validated Gemini wires to the finite local fixture."""
            payload = json.loads(super().admit(argument))
            for wire in payload["route"]:
                wire["url"] = f"http://127.0.0.1:{provider.server_port}/{wire['deployment_id']}"
            return json.dumps(payload)

        def settle(self, argument: str) -> str:
            """Retain only this synthetic fixture's payload and signal actual ledger completion."""
            raw_settlements.append(json.loads(argument))
            result = super().settle(argument)
            settled.set()
            return result

    port, shutdown = _unused_port(), native.shutdown_handle()
    failures: list[BaseException] = []

    def serve() -> None:
        """Run one native listener with its ordinary cancellation ownership."""
        try:
            serve_native_gateway(
                Plane(components, capture=capture),
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - fail explicitly after shutdown.
            failures.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        _wait_ready(port, worker)
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            timeout=10,
        ) as response:
            assert response.status_code == 200
            request_id = response.headers["x-request-id"]
            for line in response.iter_lines():
                assert private_text not in line and "secret-signature" not in line
                if "visible" in line:
                    break
        assert closed.wait(2)
        assert settled.wait(2)
    finally:
        shutdown.request_shutdown()
        worker.join(10)
        components.write_ledger.close()
        components.manager.close()
        manager.close()
        provider.shutdown()
        provider.server_close()
        serving_provider.join(5)
    assert not failures and not worker.is_alive()
    assert calls == ["/alpha"] and len(raw_settlements) == 1
    raw = raw_settlements[0]
    with sqlite3.connect(manager.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM gateway_attempts WHERE request_id=?", (request_id,)
        ).fetchone()
    assert row is not None
    if shape == "terminal":
        assert row["state"] == "completed" and row["usage_source"] == "observed"
        assert (row["input_tokens"], row["output_tokens"]) == (13, 7)
        assert "streamed_output" not in raw
    elif shape == "image":
        assert row["state"] == "cancelled" and row["usage_source"] == "unknown"
        assert row["output_tokens"] is None
    else:
        assert row["state"] == "cancelled" and row["usage_source"] == "estimated"
        expected_reasoning = private_text if shape == "thought" else ""
        assert row["reasoning_tokens"] == len(
            reservation_encoder().encode_ordinary(expected_reasoning)
        )
        assert (
            row["output_tokens"]
            == len(reservation_encoder().encode_ordinary("visible")) + row["reasoning_tokens"]
        )
        streamed = raw["streamed_output"]
        assert isinstance(streamed, dict)
        assert streamed["reasoning"] == expected_reasoning and streamed["text"] == "visible"
    if collector:
        assert collector.close(1)
    else:
        assert not records
    assert private_text not in capfd.readouterr().err


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


def test_persisted_capture_schema_reader_is_explicit_strict_and_never_invents_winner() -> None:
    """StrictV1 archives migrate only through the explicit reader; new sinks consume schema2."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1) and len(records) == 1
    payload = json.loads(records[0])
    assert payload["schema_version"] == 2
    parsed = CaptureRecord.model_validate_json(records[0])
    assert parsed.canonical_model_id is None
    assert read_capture_record_json(records[0]) == parsed
    assert json.loads(parsed.model_dump_json()) == payload
    legacy = {
        key: value
        for key, value in payload.items()
        if key not in {"canonical_model_id", "gemini_thought_parts_truncated"}
    }
    legacy["schema_version"] = 1
    CaptureRecordV1.model_validate(legacy)
    upgraded = read_capture_record_json(json.dumps(legacy))
    assert upgraded.schema_version == 2 and upgraded.canonical_model_id is None
    assert upgraded.gemini_thought_parts_truncated is None
    assert upgraded.request.model_id == "model"
    with pytest.raises(ValueError):
        CaptureRecord.model_validate(legacy)
    with pytest.raises(ValueError):
        CaptureRecordV1.model_validate(payload)
    with pytest.raises(ValueError):
        read_capture_record_json(json.dumps({**legacy, "canonical_model_id": "forged"}))
    for version in (True, False, 1.0, 2.0, "1", "2", None, 0, 3):
        with pytest.raises(ValueError):
            read_capture_record_json(json.dumps({**payload, "schema_version": version}))
    for document in (payload, legacy):
        with pytest.raises(ValueError):
            read_capture_record_json(json.dumps({**document, "unexpected": "no"}))


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
    assert collector.maintenance_failures() == 0


def test_close_timeout_preserves_accepted_content_for_later_host_settlement() -> None:
    """Timing out the Python boundary cannot purge accepted but undecided content."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin(_request_json())
    assert not collector.close(0)
    assert collector.counts() == (0, 0, 0, 0, 0, 0)
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(records) == 1
    assert CaptureRecord.model_validate_json(records[0]).request.request_id == "request"
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


@pytest.mark.parametrize("content", ["x" * 1_100_000, "雪" * 400_000])
def test_default_admission_keeps_large_inputs_whole(content: str) -> None:
    """The former one-MiB cutoff and ASCII escaping must not discard valid prompts."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    request = json.loads(_request_json())
    request["context"]["request"]["messages"] = [{"role": "user", "content": content}]
    encoded = json.dumps(request, ensure_ascii=False)
    assert collector.begin(encoded)
    collector.settle("request", True, False)
    assert collector.close(1)
    persisted = CaptureRecord.model_validate_json(records[0])
    assert persisted.request.context["request"] == request["context"]["request"]


def test_python_sink_retries_without_losing_content_or_acknowledging_failure(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A failed destination retains its exact record until recovery, even during close."""
    attempted = threading.Event()
    recovering = threading.Event()
    attempts: list[str] = []
    persisted: list[str] = []

    def write(record: str) -> None:
        """Simulate an outage whose exception contains private SQL parameters."""
        attempts.append(record)
        attempted.set()
        if not recovering.is_set():
            raise RuntimeError("private SQL parameter that must not be logged")
        persisted.append(record)

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), write)
    assert collector.begin(_request_json())
    settlement = threading.Thread(target=collector.settle, args=("request", True, False))
    settlement.start()
    try:
        assert attempted.wait(1)
        assert not collector.close(0.05)
        pending, retained, successes, failures, drops, skips = collector.counts()
        assert pending == 1 and retained > 0
        assert successes == drops == skips == 0
        assert failures >= 1
        assert settlement.is_alive()
    finally:
        recovering.set()
        settlement.join(3)
    assert not settlement.is_alive()
    assert collector.close(1)
    assert len(attempts) >= 2 and len(set(attempts)) == 1
    assert all(record is attempts[0] for record in attempts)
    assert persisted == attempts[:1]
    assert collector.counts()[0:3] == (0, 0, 1)
    assert collector.counts()[4:] == (0, 0)
    assert "private SQL" not in "".join(capfd.readouterr())


def test_python_sink_rechecks_policy_after_an_uncertain_commit() -> None:
    """Retry is idempotent and may acknowledge revocation without retaining content."""
    rows: dict[str, str] = {}
    attempts = 0

    def write(encoded: str) -> None:
        """Lose the first acknowledgement, then simulate consent revocation on retry."""
        nonlocal attempts
        attempts += 1
        record = CaptureRecord.model_validate_json(encoded)
        if attempts == 1:
            rows[record.request.request_id] = encoded
            raise ConnectionError("lost acknowledgement")
        rows.pop(record.request.request_id, None)

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), write)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert attempts == 2 and not rows
    assert collector.counts() == (0, 0, 1, 1, 0, 0)


def test_batched_sink_preserves_strings_and_retries_only_unacknowledged_members() -> None:
    """One failed record cannot hold healthy peers; retries reuse prepared objects."""
    entered = threading.Event()
    release = threading.Event()
    recover = threading.Event()
    healthy = threading.Event()
    batches: list[tuple[str, ...]] = []
    persisted: set[str] = set()
    failed_strings: list[str] = []

    def write(records: tuple[str, ...]) -> list[bool]:
        """Keep one record unavailable while acknowledging all of its neighbors."""
        assert threading.current_thread() is not threading.main_thread()
        batches.append(records)
        entered.set()
        assert release.wait(5)
        outcomes = []
        for encoded in records:
            request_id = CaptureRecord.model_validate_json(encoded).request.request_id
            if request_id == "request-0":
                failed_strings.append(encoded)
                if not recover.is_set():
                    outcomes.append(False)
                    continue
            assert request_id not in persisted
            persisted.add(request_id)
            outcomes.append(True)
        if len(persisted) >= 15:
            healthy.set()
        return outcomes

    collector = native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    threads = []
    for index in range(16):
        request_id = f"request-{index}"
        assert collector.begin(_request_json().replace('"request"', json.dumps(request_id), 1))
        thread = threading.Thread(target=collector.settle, args=(request_id, True, False))
        thread.start()
        threads.append(thread)
        if index == 0:
            assert entered.wait(3)
    try:
        assert not collector.close(0.01)
        release.set()
        assert healthy.wait(5)
        assert "request-0" not in persisted
        assert not collector.close(0.01)
    finally:
        release.set()
        recover.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert collector.close(3)
    assert len(persisted) == 16
    assert any(len(batch) > 1 for batch in batches)
    assert len(failed_strings) > 1
    assert all(encoded is failed_strings[0] for encoded in failed_strings)
    assert collector.counts()[0:3] == (0, 0, 16)
    assert collector.counts()[4:] == (0, 0)


def test_batched_sink_invalid_acknowledgements_never_release_records() -> None:
    """Exceptions and mismatched receipt counts retain the same prepared payload."""
    seen: list[str] = []

    def write(records: tuple[str, ...]) -> list[bool]:
        """Recover only after exercising both invalid callback outcomes."""
        seen.append(records[0])
        if len(seen) == 1:
            raise RuntimeError("private destination error")
        if len(seen) == 2:
            return []
        return [True]

    collector = native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(seen) == 3 and all(encoded is seen[0] for encoded in seen)
    assert collector.counts() == (0, 0, 1, 2, 0, 0)


@pytest.mark.parametrize(("count", "size"), [(80, 0), (8, 600_000)])
def test_batched_sink_bounds_count_and_bytes(count: int, size: int) -> None:
    """Queued work fills bounded batches without a timer or losing Unicode content."""
    entered, release = threading.Event(), threading.Event()
    batches: list[tuple[str, ...]] = []

    def write(records: tuple[str, ...]) -> list[bool]:
        batches.append(records)
        entered.set()
        assert release.wait(5)
        assert len(records) <= 64
        assert sum(len(record.encode()) for record in records) <= 9 * 1024 * 1024
        if len(records) > 1:
            assert sum(len(record.encode()) for record in records[:-1]) < 1024 * 1024
        return [True] * len(records)

    collector = native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    threads = []
    try:
        for index in range(count):
            request = json.loads(_request_json())
            request["request_id"] = f"bounded-{index}"
            request["context"]["request"]["messages"] = [
                {"role": "user", "content": "x" * size + "🌏"}
            ]
            assert collector.begin(json.dumps(request))
            thread = threading.Thread(
                target=collector.settle, args=(request["request_id"], True, False)
            )
            thread.start()
            threads.append(thread)
            if index == 0:
                assert entered.wait(3)
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert collector.close(3)
    assert sum(map(len, batches)) == count
    assert any(len(batch) > 1 for batch in batches)
    for batch in batches:
        for encoded in batch:
            assert json.loads(encoded)["request"]["context"]["request"]["messages"] == [
                {"role": "user", "content": "x" * size + "🌏"}
            ]
    assert collector.counts() == (0, 0, count, 0, 0, 0)


def test_batched_sink_rejects_insufficient_preparation_budget() -> None:
    """A valid single-record budget may be too small for the batch reservation."""
    config = CaptureConfiguration().model_dump(mode="json")
    config["delivery"] = {
        "maximum_records": 64,
        "maximum_record_bytes": 1024,
        "maximum_bytes": 6 * 1024 + 256,
    }
    with pytest.raises(ValueError, match="preparation"):
        native.CaptureCollector.batched(json.dumps(config), lambda values: [True] * len(values))


def test_python_and_rust_configuration_fail_closed() -> None:
    """Both entry points reject invalid bounds and unknown configuration."""
    with pytest.raises(ValueError):
        CaptureDeliveryLimits(maximum_bytes=1)
    with pytest.raises(ValueError):
        CaptureConfiguration(maximum_pending_bytes=1)
    with pytest.raises(ValueError):
        native.CaptureCollector('{"unknown": true}', lambda _: None)
    with pytest.raises(ValueError, match="preparation"):
        CaptureDeliveryLimits(maximum_record_bytes=1024, maximum_bytes=5 * 1024 + 256)
    invalid = CaptureConfiguration().model_dump(mode="json")
    invalid["delivery"] = {
        "maximum_records": 1,
        "maximum_record_bytes": 1024,
        "maximum_bytes": 5 * 1024 + 256,
    }
    with pytest.raises(ValueError, match="preparation"):
        native.CaptureCollector(json.dumps(invalid), lambda _: None)


@pytest.mark.parametrize("maximum_bytes", [5 * 1024 + 257, 6 * 1024 + 255])
def test_preparation_must_leave_a_full_record_budget(maximum_bytes: int) -> None:
    """Do not accept a configuration whose preparation crowds out its queue."""
    with pytest.raises(ValueError, match="preparation"):
        CaptureDeliveryLimits(maximum_record_bytes=1024, maximum_bytes=maximum_bytes)
    invalid = CaptureConfiguration().model_dump(mode="json")
    invalid["delivery"] = {
        "maximum_records": 1,
        "maximum_record_bytes": 1024,
        "maximum_bytes": maximum_bytes,
    }
    with pytest.raises(ValueError, match="preparation"):
        native.CaptureCollector(json.dumps(invalid), lambda _: None)


def test_exact_preparation_and_record_budget_boundary_is_valid() -> None:
    """The exact queue-plus-preparation boundary admits and persists a record."""
    delivery = CaptureDeliveryLimits(maximum_record_bytes=8192, maximum_bytes=6 * 8192 + 256)
    records: list[str] = []
    collector = native.CaptureCollector(
        CaptureConfiguration(delivery=delivery).model_dump_json(), records.append
    )
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(records) == 1
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


def test_prepared_python_unicode_payload_is_inside_delivery_memory_budget() -> None:
    """A wide Unicode string remains charged while a paused destination retains it."""
    entered, resume = threading.Event(), threading.Event()
    payload_bytes: list[int] = []

    def write(encoded: str) -> None:
        """Hold one four-byte Python string without retaining an additional copy."""
        payload_bytes.append(sys.getsizeof(encoded))
        entered.set()
        assert resume.wait(3)

    limits = CaptureDeliveryLimits(maximum_bytes=65_536, maximum_record_bytes=8192)
    configuration = CaptureConfiguration(delivery=limits)
    collector = native.CaptureCollector(configuration.model_dump_json(), write)
    request = json.loads(_request_json())
    request["context"]["request"]["messages"] = [{"role": "user", "content": "x" * 2000 + "🌍"}]
    assert collector.begin(json.dumps(request))
    settlement = threading.Thread(target=collector.settle, args=("request", True, False))
    settlement.start()
    try:
        assert entered.wait(1)
        pending, retained, *_ = collector.counts()
        assert pending == 1
        assert retained > 5 * limits.maximum_record_bytes + 256
        assert payload_bytes[0] < retained <= limits.maximum_bytes
        assert not collector.close(0)
    finally:
        resume.set()
        settlement.join(3)
    assert not settlement.is_alive()
    assert collector.close(1)
    assert collector.counts() == (0, 0, 1, 0, 0, 0)


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


@pytest.mark.parametrize(
    "policy",
    [
        "local",
        "hosted",
        "hosted-late",
        "hosted-byok",
        "hosted-checkpoint-failed",
        "off",
        "broken",
        "full",
    ],
)
def test_real_http_surfaces_capture_or_fail_before_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Collect Chat, Responses and Messages JSON/SSE through native HTTP, not a fixture tap."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _LoopbackProvider.calls = 0
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    records: list[str] = []
    configuration = CaptureConfiguration(
        settlement_required=policy.startswith("hosted"),
        delivery=(
            CaptureDeliveryLimits(maximum_record_bytes=1024, maximum_bytes=6400)
            if policy == "hosted-checkpoint-failed"
            else CaptureDeliveryLimits()
        ),
    )
    collector = native.CaptureCollector(configuration.model_dump_json(), records.append)
    if policy == "full":
        assert collector.close(1)

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
    control = NativeControlPlane(components, capture=capture)
    admit = control.admit

    def funding_admit(argument: str) -> str:
        """Model a hosted-funded test lane; local provider fixtures otherwise use BYOK."""
        result = json.loads(admit(argument))
        if policy in {"hosted", "hosted-late", "hosted-checkpoint-failed"}:
            for wire in result["route"]:
                wire["billing_customer_managed"] = False
        return json.dumps(result)

    monkeypatch.setattr(control, "admit", funding_admit)
    settlements: list[dict[str, object]] = []
    settle = control.settle

    def observed_settle(argument: str) -> str:
        """Run real accounting and explicitly exclude a rejected capture in this test."""
        result = settle(argument)
        value = json.loads(argument)
        settlements.append(value)
        if policy == "hosted-checkpoint-failed":
            collector.settle(value["request_id"], False, False)
        return result

    monkeypatch.setattr(control, "settle", observed_settle)

    def run() -> None:
        """Serve the real data plane and preserve startup failures for assertions."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - surfaced after bounded shutdown.
            failures.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    awaiting_settlement: list[str] = []
    try:
        _wait_ready(port, worker)
        for surface in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                payload: dict[str, object] = {"model": "coding", "stream": stream}
                prompt = "capture task" * (400 if policy == "hosted-checkpoint-failed" else 1)
                if surface == "responses":
                    payload["input"] = prompt
                else:
                    payload["messages"] = [{"role": "user", "content": prompt}]
                if surface == "messages":
                    payload["max_tokens"] = 128
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    headers={
                        "authorization": f"Bearer {raw_key}",
                        "X-Session-Id": "real-harness-session",
                        "X-Other-Private-Header": "do-not-capture-me",
                    },
                    json=payload,
                    timeout=10,
                )
                if policy in {"broken", "full"}:
                    assert response.status_code == 503
                    assert (
                        "overloaded_error" if surface == "messages" else "capture_unavailable"
                    ) in response.text
                    assert "private policy" not in response.text
                    continue
                if policy == "hosted-checkpoint-failed":
                    assert response.status_code == 500, response.text
                    assert "hello " not in response.text
                    assert len(settlements) == _LoopbackProvider.calls
                    assert settlements[-1]["attempt_id"]
                    assert settlements[-1]["outcome"] == "failed"
                    assert settlements[-1]["finalize"] is True
                    assert settlements[-1]["opened"] is True
                    assert '"failure_class": "internal"' in json.dumps(settlements[-1]["failure"])
                    assert not settlements[-1].get("usage_incomplete_due_to_disconnect", False)
                    continue
                assert response.status_code == 200, response.text
                assert "hello " in response.text and "world" in response.text
                if policy == "hosted":
                    collector.settle(response.headers["x-request-id"], True, True)
                elif policy == "hosted-late":
                    awaiting_settlement.append(response.headers["x-request-id"])
                elif policy == "hosted-byok":
                    assert not records, "BYOK must not checkpoint before its terminal verdict"
                    collector.settle(response.headers["x-request-id"], False, False)
        if policy == "hosted-late":
            assert not collector.close(0)
            # Native output cannot precede durable prompt ownership. Terminal
            # permission adds the response later without discarding the prompt.
            checkpoints = [CaptureRecord.model_validate_json(value) for value in records]
            assert len(checkpoints) == 6
            assert all(record.response is None for record in checkpoints)
            assert {record.request.request_id for record in checkpoints} == set(awaiting_settlement)
            assert collector.counts()[5] == 0
            for request_id in awaiting_settlement:
                collector.settle(request_id, True, True)
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
    assert _LoopbackProvider.calls == (0 if policy in {"broken", "full"} else 6)
    if policy in {"off", "broken", "full", "hosted-byok", "hosted-checkpoint-failed"}:
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
    for record in completed:
        assert record.request.context["session_id"] == "real-harness-session"
        assert record.provider_reasoning is None
        assert record.metrics is not None
        assert record.metrics.terminal_at is not None
        assert record.metrics.terminal_at >= record.metrics.started_at
        assert record.metrics.first_token_at is not None
        assert record.metrics.first_token_at >= record.metrics.started_at
        assert record.metrics.duration_ms is not None and record.metrics.duration_ms > 0
        assert record.metrics.usage_complete
        assert record.metrics.usage is not None
        assert record.metrics.usage.input_tokens is not None
        assert record.metrics.usage.output_tokens is not None
    assert "provider-secret" not in "".join(records)
    assert raw_key not in "".join(records)
    assert "do-not-capture-me" not in "".join(records)


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages", "responses-ws"])
@pytest.mark.parametrize("ending", ["disconnect", "deadline"])
@pytest.mark.parametrize("retention", ["discard", "prompt", "response"])
def test_pending_checkpoint_does_not_pin_transport_or_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, ending: str, retention: str
) -> None:
    """A refused sink keeps capture ownership, not the request's transport or reserved attempt."""
    provider_closed, write_attempted, allow_write, settled = (threading.Event() for _ in range(4))
    calls: list[str] = []

    class Provider(BaseHTTPRequestHandler):
        """Keep a committed stream open until the gateway closes its physical socket."""

        def do_POST(self) -> None:  # noqa: N802 - standard HTTP handler contract.
            """Emit visible commitment repeatedly so transport closure is observable."""
            self.rfile.read(int(self.headers["content-length"]))
            calls.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(_content_chunk("checkpoint-visible"))
                    self.wfile.flush()
                    time.sleep(0.01)
            except OSError:
                provider_closed.set()

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "checkpoint-test-only")
    manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    records: list[str] = []

    def persist(value: str) -> None:
        """Refuse acknowledgement until the test has proven bounded request completion."""
        write_attempted.set()
        if not allow_write.is_set():
            raise RuntimeError("synthetic destination unavailable")
        CaptureRecord.model_validate_json(value)
        records.append(value)

    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), persist)
    capture = CaptureController(collector, application_for=lambda _auth: "checkpoint-test")
    control = NativeControlPlane(components, capture=capture, request_timeout_seconds=1.5)
    admit, settle = control.admit, control.settle
    settlements: list[JsonObject] = []

    def funded_admit(argument: str) -> str:
        """Mark only this loopback fixture host-funded, leaving exact admission facts intact."""
        value = json.loads(admit(argument))
        for wire in value["route"]:
            wire["billing_customer_managed"] = False
        return json.dumps(value)

    def record_settlement(argument: str) -> str:
        """Observe real durable completion then deny response capture without blocking the sink."""
        result = settle(argument)
        value = json.loads(argument)
        settlements.append(value)
        collector.settle(value["request_id"], retention != "discard", retention == "response")
        settled.set()
        return result

    monkeypatch.setattr(control, "admit", funded_admit)
    monkeypatch.setattr(control, "settle", record_settlement)
    port, shutdown = _unused_port(), native.shutdown_handle()
    errors: list[BaseException] = []

    def serve() -> None:
        """Serve with one permit so a blocked request owner cannot hide behind spare capacity."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                capture=collector,
                shutdown=shutdown,
                max_active_requests=1,
                graceful_timeout_seconds=0.2,
            )
        except BaseException as error:  # noqa: BLE001 - propagate serving failures.
            errors.append(error)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client: socket.socket | None = None
    ws: ClientConnection | None = None
    try:
        _wait_ready(port, worker)
        payload: JsonObject = {"model": "coding", "stream": True}
        if surface in {"responses", "responses-ws"}:
            payload["input"] = "capture checkpoint"
        else:
            payload["messages"] = [{"role": "user", "content": "capture checkpoint"}]
        if surface == "messages":
            payload["max_tokens"] = 64
        if surface == "responses-ws":
            ws = connect(
                f"ws://127.0.0.1:{port}/v1/responses",
                additional_headers={"authorization": f"Bearer {raw_key}"},
                close_timeout=0.2,
            )
            ws.send(json.dumps({"type": "response.create", **payload}))
        else:
            body = json.dumps(payload).encode()
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            client.sendall(
                (
                    f"POST /v1/{surface} HTTP/1.1\r\nHost: localhost\r\n"
                    f"Authorization: Bearer {raw_key}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + body
            )
        assert write_attempted.wait(3)
        if ws is not None:
            ws.send(json.dumps({"type": "response.create", "model": "coding", "input": "queued"}))
        if ending == "disconnect":
            if ws is not None:
                ws.close()
            else:
                assert client is not None
                client.shutdown(socket.SHUT_RDWR)
                client.close()
                client = None
        assert provider_closed.wait(4), "capture acknowledgement pinned the upstream transport"
        assert settled.wait(4), "capture acknowledgement pinned durable attempt settlement"
        assert len(calls) == 1 and settlements
        # Cancellation may interrupt a delivered callback and replay its identical decision.
        assert all(value == settlements[0] for value in settlements)
        assert settlements[0]["attempt_id"] and settlements[0]["opened"] is True
        columns = (
            "attempt_id, state, input_tokens, output_tokens, usage_source, "
            "estimated_cost_nano_usd, budget_settled_nano_usd"
        )
        with sqlite3.connect(manager.database_path) as connection:
            rows = connection.execute(f"SELECT {columns} FROM gateway_attempts").fetchall()
        assert len(rows) == 1 and rows[0][:2] == (settlements[0]["attempt_id"], "cancelled")
        assert rows[0][2] > 0 and rows[0][3] > 0 and rows[0][4] == "estimated"
        assert not records
        assert not collector.close(0) and not collector.close(0)
        shutdown.request_shutdown()
        worker.join(2)
        assert not worker.is_alive(), "capture retry pinned runtime shutdown"
        allow_write.set()
        assert collector.close(3)
        assert len(records) == (1 if retention == "discard" else 2)
        assert all(value == settlements[0] for value in settlements) and len(calls) == 1
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute(f"SELECT {columns} FROM gateway_attempts").fetchall() == rows
        saved = CaptureRecord.model_validate_json(records[0])
        assert saved.response is None and saved.canonical_model_id is None
    finally:
        allow_write.set()
        if ws is not None:
            ws.close()
        if client is not None:
            client.close()
        shutdown.request_shutdown()
        worker.join(5)
        collector.close(3)
        if components.write_ledger is not None:
            components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(3)
        manager.close()
    assert not errors

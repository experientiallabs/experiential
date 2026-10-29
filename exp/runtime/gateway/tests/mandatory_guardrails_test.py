"""Drive mandatory inspection through the real native gateway and all three protocols."""

from __future__ import annotations

import socket
import sqlite3
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import exp_gateway_native
import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.bounded import BoundedInspect
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient, InspectingClassifier
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
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _chat_body
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)


def _tool_chunk(arguments: str, *, start: bool = False) -> bytes:
    """Preserve a single tool's complete marker across multiple provider deltas."""
    call: JsonObject = {"index": 0, "function": {"arguments": arguments}}
    if start:
        call.update(id="call-synthetic", type="function")
        call["function"] = {"name": "lookup", "arguments": arguments}
    return _sse_frame({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]})


class _Guard(GuardrailEngine):
    """Compose a synthetic adapter through the normal engine and mandatory policy layer."""

    def __init__(
        self,
        *,
        block_input: bool = False,
        policies: tuple[GuardrailPolicy, ...] = (),
        adapters: Mapping[str, InspectingClassifier] | None = None,
        inspects: BoundedInspect | None = None,
        timeout_ms: int = 5000,
        max_response_bytes: int = 1_048_576,
    ) -> None:
        """Use the same registry and executor for operator and customer checks."""
        self.block_input = block_input
        self.requests: list[GatewayRequest] = []
        self.completions: list[GuardrailCompletion] = []
        policy = GuardrailPolicy(
            policy_id="platform-test",
            revision="synthetic-policy-v1",
            protected=True,
            max_response_bytes=max_response_bytes,
            checks=tuple(
                GuardrailCheck(
                    check_id=f"platform-{stage.value}",
                    capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                    stage=stage,
                    action=GuardrailAction.BLOCK,
                    timeout_ms=timeout_ms,
                    adapter_id="platform",
                )
                for stage in (GuardrailCheckStage.INPUT, GuardrailCheckStage.OUTPUT)
            ),
        )
        super().__init__(
            store=MappingGuardrailStore((policy, *policies)),
            client=DirectClassifierClient(
                ClassifierRegistry({**(adapters or {}), "platform": self})
            ),
            monotonic=time.monotonic,
            inspects=inspects,
        )

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Record input after every expansion and return a standard classifier verdict."""
        self.requests.append(request)
        return ClassifierVerdict(flagged=self.block_input)

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Offer the existing complete-output classifier contract as well."""
        self.completions.append(completion)
        return ClassifierVerdict(
            flagged="withhold-marker" in completion.text
            or any("withhold-marker" in item for item in completion.context)
            or any("withhold-marker" in call.arguments for call in completion.tool_calls)
        )


def test_mandatory_input_refuses_before_durable_acceptance(tmp_path: Path) -> None:
    """Missing optional policies cannot disable a host admission decision."""
    _, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    guard = _Guard(block_input=True)
    control = NativeControlPlane(components, guardrails=guard)
    with pytest.raises(NativeBridgeError):
        _admit(control, raw_key, _chat_body())
    assert len(guard.requests) == 1
    with sqlite3.connect(components.ledger.database_path) as connection:
        assert connection.execute("select count(*) from gateway_requests").fetchone()[0] == 0
        assert connection.execute("select count(*) from gateway_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tool_call", [False, True, "large"])
def test_real_native_output_policy_withholds_complete_response(
    tmp_path: Path,
    surface: str,
    streaming: bool,
    tool_call: bool | Literal["large"],
) -> None:
    """Blocking output checks withhold the whole completion across all protocols."""
    continue_output = threading.Event()
    upstream_finished = threading.Event()
    prefix = "This is an allowed synthetic explanation. " * 9

    class Provider(BaseHTTPRequestHandler):
        """Produce a safe prefix followed by a transport-split violating segment."""

        def do_POST(self) -> None:  # noqa: N802 - HTTP server protocol.
            """Stream only synthetic fixture content."""
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(_content_chunk(prefix))
                self.wfile.flush()
                arguments = '{"q":"' + ("x" * 600_000 if tool_call == "large" else "") + "withhold-"
                frames = (
                    (_tool_chunk(arguments, start=True), _tool_chunk('marker"}'))
                    if tool_call
                    else (_content_chunk("withhold-"), _content_chunk("marker"))
                )
                for frame in frames:
                    self.wfile.write(frame)
                    self.wfile.flush()
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
                upstream_finished.set()

        def log_message(self, format: str, *args: object) -> None:
            """Keep fixture bodies out of logs."""

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _, raw_key = _configured_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
        capabilities=ModelCapabilities(maximum_output_tokens=128_000, supports_tools=True),
    )
    guard = _Guard()
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    control = NativeControlPlane(
        components,
        guardrails=guard,
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = threading.Event()
    stop = exp_gateway_native.shutdown_handle()
    errors: list[Exception] = []

    def serve() -> None:
        """Start the actual native extension and retain any startup failure."""
        try:
            serve_native_gateway(
                control,
                host="127.0.0.1",
                port=port,
                shutdown=stop,
                on_listening=ready.set,
                graceful_timeout_seconds=1.0,
            )
        except Exception as exc:  # noqa: BLE001 - a fixture worker propagates failures.
            errors.append(exc)
            ready.set()

    gateway_thread = threading.Thread(target=serve, daemon=True)
    gateway_thread.start()
    try:
        assert ready.wait(10)
        assert not errors
        body: JsonObject = {"model": "coding", "stream": streaming, "max_tokens": 1000}
        if surface == "responses":
            path = "/v1/responses"
            body.pop("max_tokens")
            body["input"] = "An allowed synthetic request."
        else:
            path = "/v1/messages" if surface == "messages" else "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": "An allowed synthetic request."}]
        if tool_call:
            schema: JsonObject = {"type": "object", "properties": {"q": {"type": "string"}}}
            definition: JsonObject = {"name": "lookup", "parameters": schema}
            if surface == "messages":
                body["tools"] = [{"name": "lookup", "input_schema": schema}]
            elif surface == "responses":
                body["tools"] = [{"type": "function", **definition}]
            else:
                body["tools"] = [{"type": "function", "function": definition}]
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}{path}",
            headers={"authorization": f"Bearer {raw_key}"},
            json=body,
            timeout=10,
        ) as response:
            chunks = []
            for chunk in response.iter_text():
                chunks.append(chunk)
                if streaming and prefix in "".join(chunks) and not continue_output.is_set():
                    assert not upstream_finished.is_set()
                    continue_output.set()
            result = "".join(chunks)
            assert response.status_code == 400
        assert "withhold-" not in result
        assert "blocked by a gateway guardrail" in result
        assert len(guard.requests) == 1
        assert prefix not in result
        assert any("withhold-marker" in item.model_dump_json() for item in guard.completions)
        with sqlite3.connect(components.ledger.database_path) as connection:
            rows = connection.execute(
                "select state, failure_class from gateway_attempts"
            ).fetchall()
        assert rows == [("failed", "guardrail")]
    finally:
        continue_output.set()
        stop.request_shutdown()
        gateway_thread.join(5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(5)

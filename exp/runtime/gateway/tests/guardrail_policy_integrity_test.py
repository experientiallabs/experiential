"""Guardrail policy composition and replay boundaries through the real native gateway."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)


@contextmanager
def _provider(frames: bytes) -> Iterator[tuple[str, list[bytes]]]:
    """Expose a finite provider stream and count actual dispatches."""
    received: list[bytes] = []

    class Provider(BaseHTTPRequestHandler):
        """Serve the configured response while retaining each dispatched request body."""

        def do_POST(self) -> None:  # noqa: N802 - HTTP protocol.
            """Record one request and return the finite configured event stream."""
            received.append(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress the HTTP fixture's default stderr access log."""
            del format, args

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{provider.server_port}/v1", received
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(5)


def _body(surface: str, stream: bool) -> tuple[str, JsonObject]:
    """Address each public surface with the same synthetic prompt."""
    body: JsonObject = {"model": "coding", "stream": stream}
    if surface == "responses":
        body["input"] = "synthetic prompt"
        return "/v1/responses", body
    body["messages"] = [{"role": "user", "content": "synthetic prompt"}]
    body["max_tokens"] = 100
    return ("/v1/messages" if surface == "messages" else "/v1/chat/completions"), body


@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("change", ["added", "removed", "changed"])
def test_replay_claim_and_admission_must_share_the_same_policy_snapshot(
    tmp_path: Path,
    surface: str,
    change: str,
) -> None:
    """A reload between claim and admit neither dispatches nor publishes under the old key."""
    policy = GuardrailPolicy(
        policy_id="operator",
        protected=True,
        revision="a",
        checks=(
            GuardrailCheck(
                check_id="input",
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                adapter_id="fixture",
                timeout_ms=500,
            ),
        ),
    )
    before = () if change == "added" else (policy,)
    after = () if change == "removed" else (policy.model_copy(update={"revision": "b"}),)
    detector = ScriptedClassifier()
    engine = GuardrailEngine(
        store=MappingGuardrailStore(before),
        client=DirectClassifierClient(ClassifierRegistry({"fixture": detector})),
        monotonic=time.monotonic,
    )

    class ReloadingPlane(NativeControlPlane):
        """Change policy once after a keyed request has claimed its replay identity."""

        reloaded = False

        def claim_scope(self, argument: str) -> str:
            """Reproduce a supported store reload precisely between the two callbacks."""
            scope = super().claim_scope(argument)
            if not self.reloaded:
                engine._store = MappingGuardrailStore(after)
                self.reloaded = True
            return scope

    with _provider(_content_chunk("allowed") + _terminal_frames()) as (base_url, received):
        _, key = _configured_gateway(tmp_path, base_url=base_url)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = ReloadingPlane(components, guardrails=engine)
        route, body = _body(surface, False)
        headers = {"authorization": f"Bearer {key}", "Idempotency-Key": "operation"}
        with _serving(control) as url:
            rejected = httpx.post(url + route, headers=headers, json=body, timeout=10)
            assert rejected.status_code == 409, rejected.text
            assert "policy changed during admission" in rejected.text
            assert received == []
            assert detector.input_calls == 0
            with sqlite3.connect(components.ledger.database_path) as connection:
                assert connection.execute("select count(*) from gateway_requests").fetchone() == (
                    0,
                )
            accepted = httpx.post(url + route, headers=headers, json=body, timeout=10)
            assert accepted.status_code == 200, accepted.text
            replay = httpx.post(url + route, headers=headers, json=body, timeout=10)
            assert replay.content == accepted.content
            assert replay.status_code == 200
        assert len(received) == 1
        assert detector.input_calls == int(bool(after))


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_rewritten_typed_refusal_is_delivered_as_a_completed_response(
    tmp_path: Path,
    surface: str,
    stream: bool,
) -> None:
    """Replacing a refusal removes its terminal failure as well as its private text."""
    frames = _sse_frame({"choices": [{"index": 0, "delta": {"refusal": "private-refusal"}}]})
    frames += _sse_frame(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}
    )
    frames += b"data: [DONE]\n\n"
    detector = ScriptedClassifier(
        output_verdict=ClassifierVerdict(
            flagged=True,
            replacement_text="Sanitized response.",
        )
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="operator",
                    protected=True,
                    checks=(
                        GuardrailCheck(
                            check_id="output",
                            stage=GuardrailCheckStage.OUTPUT,
                            action=GuardrailAction.MODIFY,
                            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                            adapter_id="fixture",
                            timeout_ms=500,
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"fixture": detector})),
        monotonic=time.monotonic,
    )
    with _provider(frames) as (base_url, received):
        _, key = _configured_gateway(tmp_path, base_url=base_url)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        route, body = _body(surface, stream)
        with _serving(control) as url:
            result = httpx.post(url + route, headers={"authorization": f"Bearer {key}"}, json=body)
        assert result.status_code == 200, result.text
        assert "Sanitized response." in result.text
        assert "private-refusal" not in result.text
        documents = (
            [
                json.loads(line[6:])
                for line in result.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            if stream
            else [result.json()]
        )
        assert all(
            item.get("type") not in {"error", "response.failed"} and item.get("error") is None
            for item in documents
        )
        if surface == "responses":
            terminal = documents[-1].get("response", documents[-1])
            assert terminal["status"] == "completed"
        assert len(received) == 1
        assert detector.output_calls == 1
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute("select state from gateway_attempts").fetchall() == [
                ("completed",)
            ]

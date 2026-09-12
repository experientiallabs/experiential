"""Official SDK tag admission across native Chat, Responses, and Messages."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

import anthropic
import httpx
import openai
import pytest
from openai.types.responses import Response

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayApiSurface
from exp.runtime.gateway.group_commit import SyncGroupCommitLedger
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
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
from exp.runtime.gateway.settled_billing import SettledRequestBilling
from exp.runtime.gateway.tests.native_messages_test import _SseUpstream
from exp.runtime.openai_protocol.state import BoundedContinuationStore

native = pytest.importorskip("exp_gateway_native")


class _TagsUpstream(_SseUpstream):
    """Capture the exact upstream headers and body before normal SSE serving."""

    received_headers: list[dict[str, str]] = []
    payloads: list[JsonObject] = []
    omit_usage: bool = False

    def do_POST(self) -> None:  # noqa: N802 - HTTP server method.
        """Record wire headers while the shared fixture records the payload."""
        self.received_headers.append({key.lower(): value for key, value in self.headers.items()})
        if not self.omit_usage:
            super().do_POST()
            return
        self.payloads.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(
            b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.wfile.flush()


@dataclass
class _Gateway:
    """Two real workers sharing a continuation store and the same authority."""

    key: str
    origins: list[str] = field(default_factory=list)
    accepted: list[AuthorizationSnapshot] = field(default_factory=list)
    billing: SettledRequestBilling | None = None
    billing_reads: list[str] = field(default_factory=list)
    billing_delay: float = 0.0
    controls: list[NativeControlPlane] = field(default_factory=list)

    def read_billing(self, request_id: str) -> SettledRequestBilling | None:
        """Record each host read so duplicate callbacks cannot hide behind fixed facts."""
        self.billing_reads.append(request_id)
        if self.billing_delay:
            time.sleep(self.billing_delay)
        return self.billing

    def enable_output_guardrail(self) -> None:
        """Exercise buffered publication through the real allowing guardrail engine."""
        policy = GuardrailPolicy(
            policy_id="test-output",
            organization_id="local",
            identity_id="default",
            checks=(
                GuardrailCheck(
                    check_id="allow-output",
                    capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                    stage=GuardrailCheckStage.OUTPUT,
                    action=GuardrailAction.BLOCK,
                    timeout_ms=1000,
                    adapter_id="scripted",
                ),
            ),
        )
        for control in self.controls:
            control._guardrails = GuardrailEngine(
                store=MappingGuardrailStore((policy,)),
                monotonic=time.monotonic,
                client=DirectClassifierClient(
                    ClassifierRegistry({"scripted": ScriptedClassifier()})
                ),
            )


@pytest.fixture
def gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[_Gateway]:
    """Serve real native workers and capture snapshots at the ledger boundary."""
    _TagsUpstream.payloads.clear()
    _TagsUpstream.omit_usage = False
    _TagsUpstream.received_headers.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TagsUpstream)
    provider_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configured_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{upstream.server_port}/v1"
    )
    components = load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    result = _Gateway(raw_key)
    request_timeout = float(getattr(request, "param", 120.0))
    original = SyncGroupCommitLedger.accept_request

    def record(ledger: SyncGroupCommitLedger, *, authorization: AuthorizationSnapshot) -> None:
        """Observe the exact snapshot accepted durably, not an intermediate map."""
        original(ledger, authorization=authorization)
        result.accepted.append(authorization)

    monkeypatch.setattr(SyncGroupCommitLedger, "accept_request", record)
    continuations = BoundedContinuationStore()
    shutdowns = []
    threads: list[threading.Thread] = []
    failures: list[BaseException] = []

    def start_worker() -> None:
        """Bind a loopback port and await the listener, surfacing boot failures."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        ready = threading.Event()
        control = NativeControlPlane(
            components,
            request_timeout_seconds=request_timeout,
            continuation_store=continuations,
            settled_billing_reader=result.read_billing,
        )
        result.controls.append(control)
        shutdown = native.shutdown_handle()
        shutdowns.append(shutdown)

        def listening() -> None:
            """Record the port bound by the native listener."""
            result.origins.append(f"http://127.0.0.1:{port}/v1")
            ready.set()

        def serve() -> None:
            """Run the native server until the embedder stops it."""
            try:
                native.serve(
                    control,
                    json.dumps(
                        {
                            "host": "127.0.0.1",
                            "port": port,
                            "graceful_timeout_seconds": 2.0,
                            "request_timeout_seconds": request_timeout,
                        }
                    ),
                    shutdown,
                    listening,
                )
            except BaseException as error:  # noqa: BLE001 - returned to the fixture thread.
                failures.append(error)
                ready.set()

        thread = threading.Thread(target=serve, daemon=True)
        threads.append(thread)
        thread.start()
        assert ready.wait(30), "native worker never bound its listener"
        assert not failures

    try:
        start_worker()
        start_worker()
        yield result
    finally:
        for shutdown in shutdowns:
            shutdown.request_shutdown()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
        components.write_ledger.close()
        upstream.shutdown()
        upstream.server_close()
        provider_thread.join(5)
        assert not failures


@pytest.mark.parametrize("stream", [False, True])
def test_tags_reach_ledger_through_official_sdks(gateway: _Gateway, stream: bool) -> None:
    """Each protocol carries UTF-8 tags but neither wire payload nor headers do."""
    tags = {"team": "café", "cost-center.prod": "research"}
    headers = {"X-Explabs-Tags": json.dumps(tags)}
    with openai.OpenAI(api_key=gateway.key, base_url=gateway.origins[0], max_retries=0) as client:
        chat = client.chat.completions.create(
            model="coding",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            extra_headers=headers,
        )
        if isinstance(chat, openai.Stream):
            assert list(chat)
        else:
            assert chat.choices[0].message.content == "hello world"
        response = client.responses.create(
            model="coding",
            input="hello",
            stream=stream,
            extra_headers=headers,
        )
        if isinstance(response, Response):
            assert response.output_text == "hello world"
        else:
            assert any(event.type == "response.completed" for event in response)
    with anthropic.Anthropic(
        api_key=gateway.key, base_url=gateway.origins[0][:-3], max_retries=0
    ) as client:
        message = client.messages.create(
            model="coding",
            max_tokens=32,
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            extra_headers=headers,
        )
        if isinstance(message, anthropic.types.Message):
            assert isinstance(message.content[0], anthropic.types.TextBlock)
            assert message.content[0].text == "hello world"
        else:
            assert any(event.type == "message_stop" for event in message)
    assert [row.surface for row in gateway.accepted] == [
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.MESSAGES,
    ]
    assert all(row.request_tags == tags for row in gateway.accepted)
    assert all(row.model_dump(mode="json")["request_tags"] == tags for row in gateway.accepted)
    assert len(_TagsUpstream.payloads) == 3
    assert all("x-explabs-tags" not in row for row in _TagsUpstream.received_headers)
    assert all("request_tags" not in row and "tags" not in row for row in _TagsUpstream.payloads)
    assert all("café" not in json.dumps(row, ensure_ascii=False) for row in _TagsUpstream.payloads)


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_keyed_consistency_includes_tags(gateway: _Gateway, surface: str) -> None:
    """Ordering is immaterial, but a changed or missing map conflicts before dispatch."""
    with openai.OpenAI(api_key=gateway.key, base_url=gateway.origins[0], max_retries=0) as client:

        def call(tags: str | None) -> str:
            """Submit one stable keyed operation through the requested SDK surface."""
            headers = {"Idempotency-Key": "tag-operation"}
            if tags is not None:
                headers["X-Explabs-Tags"] = tags
            if surface == "chat":
                return client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": "hello"}],
                    extra_headers=headers,
                ).id
            return client.responses.create(model="coding", input="hello", extra_headers=headers).id

        first = call('{"team":"a","env":"prod"}')
        assert call('{ "env": "prod", "team":"a" }') == first
        for changed in ['{"team":"b","env":"prod"}', None]:
            with pytest.raises(openai.ConflictError) as raised:
                call(changed)
            assert raised.value.code == "idempotency_conflict"
    assert len(gateway.accepted) == len(_TagsUpstream.payloads) == 1


def test_messages_remains_unkeyed(gateway: _Gateway) -> None:
    """An Idempotency-Key never joins the Messages replay namespace."""
    with anthropic.Anthropic(
        api_key=gateway.key, base_url=gateway.origins[0][:-3], max_retries=0
    ) as client:
        for team in ["a", "b"]:
            client.messages.create(
                model="coding",
                max_tokens=32,
                messages=[{"role": "user", "content": "hello"}],
                extra_headers={
                    "Idempotency-Key": "same",
                    "X-Explabs-Tags": json.dumps({"team": team}),
                },
            )
    assert [row.request_tags for row in gateway.accepted] == [{"team": "a"}, {"team": "b"}]
    assert all(row.caller_operation_sha256 is None for row in gateway.accepted)
    assert len(_TagsUpstream.payloads) == 2


def test_continuation_can_change_tags_on_another_worker(gateway: _Gateway) -> None:
    """Tags belong to one request, not the org/identity/response continuation key."""
    with openai.OpenAI(api_key=gateway.key, base_url=gateway.origins[0], max_retries=0) as first:
        response = first.responses.create(
            model="coding",
            input="hello",
            extra_headers={"X-Explabs-Tags": '{"team":"a"}'},
        )
    with openai.OpenAI(api_key=gateway.key, base_url=gateway.origins[1], max_retries=0) as second:
        continued = second.responses.create(
            model="coding",
            input="next turn",
            previous_response_id=response.id,
            extra_headers={"X-Explabs-Tags": '{"team":"b"}'},
        )
        assert continued.output_text == "hello world"
        second.responses.create(
            model="coding", input="untagged turn", previous_response_id=continued.id
        )
    assert [row.request_tags for row in gateway.accepted] == [{"team": "a"}, {"team": "b"}, {}]
    messages = _TagsUpstream.payloads[-1]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 5


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
def test_invalid_headers_never_admit_or_forward(gateway: _Gateway, surface: str) -> None:
    """HTTP duplicates, malformed JSON and excess bytes refuse before any ledger row."""
    body = {"model": "coding", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]}
    if surface == "responses":
        body = {"model": "coding", "input": "hello"}
    invalid = [
        [("X-Explabs-Tags", '{"team":"a","team":"b"}')],
        [("X-Explabs-Tags", "{}"), ("X-Explabs-Tags", "{}")],
        [("X-Explabs-Tags", '{"team":{"nested":"x"}}')],
        [("X-Explabs-Tags", '{"explabs.internal":"x"}')],
        [("X-Explabs-Tags", '{"team":"\\u0085"}')],
        [("X-Explabs-Tags", "{" + " " * 8191 + "}")],
    ]
    for headers in invalid:
        response = httpx.post(
            gateway.origins[0] + "/" + surface,
            json=body,
            headers=[
                ("Authorization", "Bearer " + gateway.key),
                ("Idempotency-Key", "invalid"),
                *headers,
            ],
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert "X-Explabs-Tags" in error["message"]
        if surface != "messages":
            assert error["param"] == "X-Explabs-Tags"
            assert error["code"] == "invalid_parameter"
        else:
            assert error["type"] == "invalid_request_error"
    assert gateway.accepted == []
    assert _TagsUpstream.payloads == []


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt", ["hello", "empty-token", "truncated-token", "no-usage"])
@pytest.mark.parametrize("guarded", [False, True])
def test_billing_is_in_terminal_bytes_and_keyed_replay(
    gateway: _Gateway, surface: str, stream: bool, prompt: str, guarded: bool
) -> None:
    """All terminal paths read host truth once and keyed replays keep exact bytes."""
    if guarded:
        gateway.enable_output_guardrail()
    _TagsUpstream.omit_usage = prompt == "no-usage"
    gateway.billing = SettledRequestBilling(paid_nano_usd=1234567, byok_nano_usd=9, is_byok=True)
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    if surface == "responses":
        body = {"model": "coding", "input": prompt, "stream": stream}
    elif surface == "chat/completions" and stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": "Bearer " + gateway.key, "Idempotency-Key": "billing-check"}
    answer = httpx.post(gateway.origins[0] + "/" + surface, json=body, headers=headers, timeout=30)
    assert answer.status_code == 200, answer.text
    if stream:
        payloads = [
            json.loads(line[6:])
            for line in answer.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        holders = [payload.get("response", payload) for payload in payloads]
        usages = [
            holder["usage"]
            for holder in holders
            if isinstance(holder.get("usage"), dict) and "cost" in holder["usage"]
        ]
        assert len(usages) == 1, answer.text
        usage = usages[0]
    else:
        usage = answer.json()["usage"]
    assert usage["cost"] == 0.001234567
    assert usage["is_byok"] is True
    assert usage["cost_details"]["upstream_inference_cost"] == 0.000000009
    assert len(gateway.billing_reads) == 1
    if prompt == "no-usage" and surface != "messages":
        assert "prompt_tokens" not in usage and "input_tokens" not in usage
    if surface != "messages":
        gateway.billing = SettledRequestBilling(paid_nano_usd=0, byok_nano_usd=0, is_byok=False)
        replay = httpx.post(
            gateway.origins[0] + "/" + surface, json=body, headers=headers, timeout=30
        )
        assert replay.content == answer.content
        assert len(_TagsUpstream.payloads) == 1
        assert len(gateway.billing_reads) == 1


@pytest.mark.parametrize("gateway", [0.5], indirect=True)
@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages"])
def test_slow_billing_cannot_truncate_settled_success(gateway: _Gateway, surface: str) -> None:
    """A bounded but late host read must not consume the terminal delivery window."""
    gateway.billing = SettledRequestBilling(paid_nano_usd=1234567, byok_nano_usd=0, is_byok=False)
    gateway.billing_delay = 0.8
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    if surface == "responses":
        body = {"model": "coding", "input": "hello", "stream": True}
    elif surface == "chat/completions":
        body["stream_options"] = {"include_usage": True}
    answer = httpx.post(
        gateway.origins[0] + "/" + surface,
        json=body,
        headers={"Authorization": "Bearer " + gateway.key},
        timeout=3,
    )
    assert answer.status_code == 200
    terminal = {
        "chat/completions": "data: [DONE]",
        "responses": '"type":"response.completed"',
        "messages": '"type":"message_stop"',
    }[surface]
    assert terminal in answer.text
    assert '"cost"' not in answer.text
    assert len(gateway.billing_reads) == 1

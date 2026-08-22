"""Tests for the Rust-engine control plane and Rust/Python parity fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import (
    NativeBridgeError,
    NativeControlPlane,
    _usage_from_payload,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat


def _control_plane(root: Path) -> tuple[NativeControlPlane, str]:
    """Seed one direct alias and load the native control plane over it."""
    _manager, raw_key = _configured_gateway(root)
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
        only_target_kinds=frozenset({"direct"}),
    )
    return NativeControlPlane(components), raw_key


def _chat_canonical(*, stream: bool = False) -> JsonObject:
    """Return one canonical GatewayRequest payload dict."""
    decoded = decode_chat({"model": "coding", "messages": [{"role": "user", "content": "hi"}]})
    request = decoded.request.model_copy(update={"stream": stream})
    return cast(JsonObject, request.model_dump(mode="json"))


def test_bridge_error_payload_is_openai_shaped() -> None:
    """The boundary error carries the exact public error representation."""
    error = NativeBridgeError(
        OpenAIProtocolError(
            status_code=429,
            code="insufficient_quota",
            message="monthly gateway allocation is exhausted",
            error_type="insufficient_quota",
            retry_after_seconds=60,
        )
    )
    payload = json.loads(error.public_error_json)
    assert payload == {
        "status_code": 429,
        "code": "insufficient_quota",
        "message": "monthly gateway allocation is exhausted",
        "error_type": "insufficient_quota",
        "param": None,
        "retry_after_seconds": 60,
    }


def test_usage_from_payload_handles_tokens_and_tool_names() -> None:
    """Settlement usage covers token totals, tool-only, and absent cases."""
    assert _usage_from_payload(None, []) is None
    tools_only = _usage_from_payload(None, ["search"])
    assert tools_only is not None and tools_only.tool_names == ("search",)
    complete = _usage_from_payload(
        {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 2},
        [],
    )
    assert complete is not None
    assert complete.input_tokens == 10
    assert complete.output_tokens == 3
    assert complete.cached_input_tokens == 2


def test_admit_and_settle_account_one_request(tmp_path: Path) -> None:
    """Admission returns wire config and settlement lands in the usage report."""
    control, raw_key = _control_plane(tmp_path)
    assert control.authenticate(json.dumps({"raw_key": raw_key})) == "{}"

    admission = json.loads(
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "alias": "coding",
                    "request": _chat_canonical(),
                    "stream": False,
                }
            )
        )
    )
    assert admission["dialect"] == "openai_compatible"
    assert admission["url"].endswith("/chat/completions")
    assert admission["model_id"] == "provider-model-exact"
    assert admission["headers"]["Authorization"] == "Bearer provider-secret-canary"
    assert admission["provider"] == "openai-compatible"
    assert admission["route_reason"] == "direct"

    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 5},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1

    repeat = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": None,
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert repeat == "{}"


def test_failed_settlement_keeps_the_attempt_retryable(tmp_path: Path) -> None:
    """A lost terminal write latches readiness but stays settleable on retry."""
    control, raw_key = _control_plane(tmp_path)
    admission = json.loads(
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "alias": "coding",
                    "request": _chat_canonical(),
                    "stream": False,
                }
            )
        )
    )
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "tool_names": [],
            "failure": None,
        }
    )
    ledger = control._components.ledger  # noqa: SLF001 - fault injection for the test.
    with mock.patch.object(
        ledger,
        "finish_attempt",
        side_effect=RuntimeError("simulated terminal write loss"),
    ):
        with pytest.raises(NativeBridgeError):
            control.settle(settlement)
    assert control.readiness("{}") == "false"
    assert control.settle(settlement) == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_admit_escalates_native_unsupported_providers_before_accounting(
    tmp_path: Path,
) -> None:
    """A provider without a native dialect escalates with no ledger rows."""
    from exp.common.models import (
        GatewayDeploymentCapabilities,
        GatewayTokenPrices,
        ModelCapabilities,
    )
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )
    from exp.runtime.gateway.lifecycle_test import _configured_gateway

    manager, raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_GEMINI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="gem",
        connection_name="gemini-main",
        provider_model="gemini-model-exact",
        exact_model_id="gemini-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="gem",
        alias_name="gem",
        revision_id="revision-gem",
        pool_id="gem",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="gem")
    components = load_gateway_components(
        tmp_path,
        environment={
            "TEST_PROVIDER_KEY": "provider-secret-canary",
            "TEST_GEMINI_KEY": "gemini-secret-canary",
        },
    )
    control = NativeControlPlane(components)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "alias": "gem",
                    "request": _chat_canonical(),
                    "stream": False,
                }
            )
        )
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["code"] == "native_unsupported"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_admit_rejects_an_ungranted_alias(tmp_path: Path) -> None:
    """An ungranted alias maps to the shared 403 public error."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.admit(
            json.dumps(
                {
                    "raw_key": raw_key,
                    "alias": "ungranted",
                    "request": _chat_canonical(),
                    "stream": False,
                }
            )
        )
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 403
    assert payload["code"] == "model_not_granted"


def test_authenticate_rejects_an_invalid_key(tmp_path: Path) -> None:
    """A bad virtual key maps to the shared 401 public error."""
    control, _raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.authenticate(json.dumps({"raw_key": "exp_vk_invalid"}))
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 401
    assert payload["code"] == "invalid_key"


def test_models_and_detail_mirror_the_python_discovery_bodies(tmp_path: Path) -> None:
    """Model discovery bodies match the shared discovery encoding."""
    control, raw_key = _control_plane(tmp_path)
    models = json.loads(control.models(json.dumps({"raw_key": raw_key})))
    assert [item["id"] for item in models["data"]] == ["coding"]
    assert models["wmo"]["authority_schema_version"] == 1
    detail = json.loads(
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "coding"}))
    )
    assert detail == models["data"][0]
    with pytest.raises(NativeBridgeError) as excinfo:
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "missing"}))
    assert json.loads(excinfo.value.public_error_json)["status_code"] == 404


def test_readiness_reflects_startup_proof(tmp_path: Path) -> None:
    """Readiness is true after load and stays content-free."""
    control, _raw_key = _control_plane(tmp_path)
    assert control.readiness("{}") == "true"


_CHAT_FIXTURES: tuple[JsonObject, ...] = (
    {"model": "coding", "messages": [{"role": "user", "content": "hi"}]},
    {
        "model": "coding",
        "messages": [
            {"role": "system", "content": "be terse"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "join "},
                    {"type": "text", "text": "me"},
                ],
            },
        ],
        "max_completion_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["END", "STOP"],
    },
    {
        "model": "coding",
        "messages": [
            {"role": "user", "content": "use tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call-1"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "find things",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
        "parallel_tool_calls": False,
    },
    {
        "model": "coding",
        "messages": [{"role": "user", "content": "structured"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
                "strict": True,
            },
        },
        "stream": True,
        "stream_options": {"include_usage": True},
    },
)

_REJECTED_FIXTURES: tuple[JsonObject, ...] = (
    {"model": "coding", "messages": [{"role": "user", "content": "x"}], "logprobs": True},
    {"model": "coding", "messages": [{"role": "user", "content": "x"}], "unknown_field": 1},
    {
        "model": "coding",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 5,
        "max_completion_tokens": 6,
    },
    {"model": "coding", "messages": [{"role": "user", "content": "x"}], "tool_choice": "bad"},
    {"model": "coding", "messages": [{"role": "user", "content": "x"}], "stop": ["a", "a"]},
    {"model": "coding", "messages": []},
)


def test_rust_decode_matches_python_decode() -> None:
    """The Rust decoder produces the identical canonical request per fixture."""
    native = pytest.importorskip("exp_gateway_native")
    for fixture in _CHAT_FIXTURES:
        decoded = json.loads(native.decode_chat_canonical(json.dumps(fixture)))
        expected = decode_chat(dict(fixture))
        assert decoded["alias"] == expected.alias
        rust_request = GatewayRequest.model_validate(decoded["canonical"])
        assert rust_request == expected.request
    for fixture in _REJECTED_FIXTURES:
        with pytest.raises(OpenAIProtocolError):
            decode_chat(dict(fixture))
        with pytest.raises(ValueError):
            native.decode_chat_canonical(json.dumps(fixture))


def test_rust_upstream_payloads_match_python_builders() -> None:
    """Rust dialect payloads equal the Python builders on shared fixtures."""
    native = pytest.importorskip("exp_gateway_native")
    from exp.runtime.models.providers.streaming_requests import (
        anthropic_messages_stream_payload,
        openai_compatible_stream_payload,
        openai_responses_stream_payload,
    )

    for fixture in _CHAT_FIXTURES:
        expected = decode_chat(dict(fixture))
        decoded = json.loads(native.decode_chat_canonical(json.dumps(fixture)))
        canonical = json.dumps(decoded["canonical"])
        compatible = json.loads(
            native.build_upstream_payload("openai_compatible", canonical, "model-x")
        )
        assert compatible == openai_compatible_stream_payload("model-x", expected.request)
        if not expected.request.stop:
            responses = json.loads(
                native.build_upstream_payload("openai_responses", canonical, "model-x")
            )
            assert responses == openai_responses_stream_payload(
                "model-x",
                expected.request,
                supports_temperature=True,
                reasoning_effort=None,
            )
        if expected.request.structured_text is None:
            anthropic = json.loads(
                native.build_upstream_payload("anthropic_messages", canonical, "model-x")
            )
            assert anthropic == anthropic_messages_stream_payload("model-x", expected.request)


def test_rust_chat_sse_frames_match_python_encoder() -> None:
    """Rust SSE frames are byte-identical to the Python Chat encoder."""
    native = pytest.importorskip("exp_gateway_native")
    from exp.common.models.model import ToolCall
    from exp.runtime.gateway.contracts import (
        GatewayEvent,
        GatewayEventKind,
        GatewayUsage,
    )
    from exp.runtime.openai_protocol.streaming import ChatSseEncoder

    events = [
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="Hel"),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="lo é"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta='{"q": "x"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=4,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="search",
                arguments={"q": "x"},
                raw_arguments='{"q": "x"}',
            ),
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=5,
            usage=GatewayUsage(input_tokens=10, output_tokens=4, cached_input_tokens=1),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=6),
    ]
    encoder = ChatSseEncoder(
        request_id="request-abc",
        model="coding",
        created_at=1_700_000_000,
        include_usage=True,
    )
    expected = list(encoder.start())
    for event in events:
        expected.extend(encoder.feed(event))

    fixture = [
        {"kind": "text_delta", "text": "Hel"},
        {"kind": "text_delta", "text": "lo é"},
        {"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "search"},
        {"kind": "tool_arguments_delta", "index": 0, "text": '{"q": "x"}'},
        {
            "kind": "tool_call_completed",
            "index": 0,
            "call_id": "call-1",
            "name": "search",
            "raw_arguments": '{"q": "x"}',
        },
        {"kind": "usage", "input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 1},
        {"kind": "completed"},
    ]
    actual = native.encode_chat_fixture(
        "request-abc",
        "coding",
        1_700_000_000,
        True,
        json.dumps(fixture),
    )
    assert list(actual) == expected

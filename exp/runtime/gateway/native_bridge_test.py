"""Tests for the native-engine control plane and Rust/Python parity fixtures."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.discovery import listing_metadata_by_alias
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import (
    NativeBridgeError,
    NativeControlPlane,
    _usage_from_payload,
)
from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import decode_chat


def _control_plane(
    root: Path,
    *,
    request_timeout_seconds: float = 120.0,
) -> tuple[NativeControlPlane, str]:
    """Seed one direct alias and load the native control plane over it."""
    _manager, raw_key = _configured_gateway(root)
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components, request_timeout_seconds=request_timeout_seconds)
    return control, raw_key


def _chat_body(*, model: str = "coding", stream: bool = False) -> str:
    """Return one raw Chat Completions request body."""
    payload: JsonObject = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload)


def _admit(control: NativeControlPlane, raw_key: str, body: str) -> JsonObject:
    """Run one admission call and decode its JSON response."""
    return json.loads(control.admit(json.dumps({"raw_key": raw_key, "body": body})))


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


def test_admit_decodes_builds_payload_and_settles(tmp_path: Path) -> None:
    """Admission decodes the raw body, returns the shared upstream payload, and
    settlement lands in the usage report."""
    control, raw_key = _control_plane(tmp_path)
    assert control.authenticate(json.dumps({"raw_key": raw_key})) == "{}"

    admission = _admit(control, raw_key, _chat_body())
    assert admission["dialect"] == "openai_compatible"
    url = admission["url"]
    assert isinstance(url, str) and url.endswith("/chat/completions")
    assert admission["model_id"] == "provider-model-exact"
    headers = admission["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer provider-secret-canary"
    assert admission["provider"] == "openai-compatible"
    assert admission["route_reason"] == "direct"
    assert admission["stream"] is False
    assert admission["include_usage"] is False

    decoded = decode_chat(json.loads(_chat_body()))
    provider_request = decoded.request.model_copy(update={"stream": True, "include_usage": True})
    assert admission["upstream_payload"] == openai_compatible_stream_payload(
        "provider-model-exact", provider_request
    )

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


def test_admit_rejects_invalid_bodies_with_python_parity(tmp_path: Path) -> None:
    """Invalid JSON, non-objects, and protocol violations use the shared codes."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as invalid_json:
        control.admit(json.dumps({"raw_key": raw_key, "body": "{not json"}))
    assert json.loads(invalid_json.value.public_error_json)["code"] == "invalid_json"
    with pytest.raises(NativeBridgeError) as not_object:
        control.admit(json.dumps({"raw_key": raw_key, "body": "[1, 2]"}))
    assert json.loads(not_object.value.public_error_json)["code"] == "invalid_request"
    rejected = json.dumps(
        {"model": "coding", "messages": [{"role": "user", "content": "x"}], "logprobs": True}
    )
    with pytest.raises(NativeBridgeError) as protocol:
        control.admit(json.dumps({"raw_key": raw_key, "body": rejected}))
    assert json.loads(protocol.value.public_error_json)["status_code"] == 400
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_failed_settlement_keeps_the_attempt_retryable(tmp_path: Path) -> None:
    """A lost terminal write latches readiness but stays settleable on retry."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit(control, raw_key, _chat_body())
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
    # A transient failure does not latch readiness; the data plane retries.
    assert control.readiness("{}") == "true"
    assert control.settle(settlement) == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_sweep_replays_the_original_completed_settlement(tmp_path: Path) -> None:
    """A retained settlement lands its completed outcome and usage, never a
    downgraded cancellation."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit(control, raw_key, _chat_body())
    settlement = json.dumps(
        {
            "request_id": admission["request_id"],
            "attempt_id": admission["attempt_id"],
            "outcome": "completed",
            "usage": {"input_tokens": 9, "output_tokens": 4},
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
    control._sweep_expired()  # noqa: SLF001 - the timer normally drives this.
    with control._lock:  # noqa: SLF001 - registry state assertion.
        assert admission["request_id"] not in control._inflight  # noqa: SLF001
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 9
    assert report["totals"]["terminal_counts"] == [{"state": "completed", "attempts": 1}]


def test_abandoned_inflight_attempts_are_swept_after_the_deadline(tmp_path: Path) -> None:
    """An admitted request the data plane never settles is closed by the sweep."""
    control, raw_key = _control_plane(tmp_path, request_timeout_seconds=0.01)
    abandoned = _admit(control, raw_key, _chat_body())
    time.sleep(0.05)
    with mock.patch("exp.runtime.gateway.native_bridge._SWEEP_GRACE_SECONDS", 0.0):
        second = _admit(control, raw_key, _chat_body())
    with control._lock:  # noqa: SLF001 - registry state assertion.
        inflight = dict(control._inflight)  # noqa: SLF001
    assert abandoned["request_id"] not in inflight
    assert second["request_id"] in inflight
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 2


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
    admission = _admit(control, raw_key, _chat_body(model="gem"))
    assert "escalate" in admission
    assert "request_id" not in admission
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_admit_rejects_an_ungranted_alias(tmp_path: Path) -> None:
    """An ungranted alias maps to the shared 403 public error."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as excinfo:
        control.admit(json.dumps({"raw_key": raw_key, "body": _chat_body(model="ungranted")}))
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
    """Model discovery bodies match the shared discovery encoding with metadata."""
    control, raw_key = _control_plane(tmp_path)
    components = control._components  # noqa: SLF001 - expected-body construction.
    authorities = components.store.granted_alias_authorities(raw_key=raw_key)
    metadata = listing_metadata_by_alias(authorities, components.routes.published_metadata)
    assert "coding" in metadata

    models = json.loads(control.models(json.dumps({"raw_key": raw_key})))
    assert [item["id"] for item in models["data"]] == ["coding"]
    assert models["wmo"]["authority_schema_version"] == 1
    listed = models["data"][0]
    for key, value in metadata["coding"].extension_fields().items():
        assert listed[key] == value
    detail = json.loads(
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "coding"}))
    )
    assert detail == listed
    with pytest.raises(NativeBridgeError) as excinfo:
        control.model_detail(json.dumps({"raw_key": raw_key, "model_id": "missing"}))
    assert json.loads(excinfo.value.public_error_json)["status_code"] == 404


def test_readiness_reflects_startup_proof_and_executor_health(tmp_path: Path) -> None:
    """Readiness is true after load and follows the shared executor latch."""
    control, _raw_key = _control_plane(tmp_path)
    assert control.readiness("{}") == "true"
    control._components.executor.mark_accounting_unhealthy()  # noqa: SLF001 - fault injection.
    assert control.readiness("{}") == "false"


def test_rust_failure_taxonomy_matches_public_failure_error() -> None:
    """The Rust failure-to-public-error table equals `public_failure_error`.

    Quota exhaustion is exempt from message and retry-after comparison: its
    reset boundary is computed control-plane side and never crosses the
    bridge as a Rust failure.
    """
    native = pytest.importorskip("exp_gateway_native")
    for failure_class in GatewayFailureClass:
        failure = GatewayFailure(
            failure_class=failure_class,
            safe_message="parity probe message",
        )
        expected = public_failure_error(failure)
        actual = json.loads(
            native.failure_public_error_fixture(failure_class.value, "parity probe message")
        )
        assert actual["status_code"] == expected.status_code, failure_class
        assert actual["code"] == expected.detail.code, failure_class
        assert actual["error_type"] == expected.detail.type, failure_class
        if failure_class != GatewayFailureClass.QUOTA_EXCEEDED:
            assert actual["message"] == expected.detail.message, failure_class
            assert actual["retry_after_seconds"] == expected.retry_after_seconds, failure_class


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

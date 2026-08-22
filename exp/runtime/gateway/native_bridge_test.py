"""Tests for the native-engine control plane and Rust/Python parity fixtures."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.discovery import listing_metadata_by_alias
from exp.runtime.gateway.lifecycle import _ReadyControlStore, load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_bridge import (
    NativeBridgeError,
    NativeControlPlane,
    _usage_from_payload,
)
from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.openai_protocol.state import ProtocolNamespace
from exp.runtime.openai_protocol.streaming import stable_public_id


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


def _admit(
    control: NativeControlPlane,
    raw_key: str,
    body: str,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> JsonObject:
    """Run one admission call and decode its JSON response."""
    argument = json.dumps(
        {
            "raw_key": raw_key,
            "body": body,
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
        }
    )
    return json.loads(control.admit(argument))


def _claim_scope(
    control: NativeControlPlane,
    raw_key: str,
    body: str,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> JsonObject:
    """Run one replay-scope call and decode its JSON response."""
    argument = json.dumps(
        {
            "raw_key": raw_key,
            "body": body,
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
        }
    )
    return json.loads(control.claim_scope(argument))


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
    scope = _claim_scope(control, raw_key, _chat_body(model="gem"), idempotency_key="op-1")
    assert "escalate" in scope
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_claim_scope_matches_the_python_replay_key(tmp_path: Path) -> None:
    """The scope carries the same hashed caller operation and canonical digest
    the python engine computes for its replay key."""
    control, raw_key = _control_plane(tmp_path)
    scope = _claim_scope(control, raw_key, _chat_body(), idempotency_key="operation-one")
    assert scope["surface"] == "chat_completions"
    assert scope["caller_operation_sha256"] == hashlib.sha256(b"operation-one").hexdigest()
    repeat = _claim_scope(control, raw_key, _chat_body(), idempotency_key="operation-one")
    assert repeat == scope
    # The caller operation hashes identically through either header; the
    # canonical request digest covers the decoded request, which records
    # which header carried it, exactly as the python engine canonicalizes.
    via_client_id = _claim_scope(
        control,
        raw_key,
        _chat_body(),
        client_request_id="operation-one",
    )
    assert via_client_id["caller_operation_sha256"] == scope["caller_operation_sha256"]
    different_body = _claim_scope(
        control,
        raw_key,
        _chat_body(stream=True),
        idempotency_key="operation-two",
    )
    assert different_body["canonical_request_sha256"] != scope["canonical_request_sha256"]
    decoded = decode_chat(json.loads(_chat_body()), idempotency_key="operation-one")
    assert decoded.request.idempotency_key == "operation-one"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0


def test_claim_scope_validates_headers_with_python_parity(tmp_path: Path) -> None:
    """Header validation failures map to the exact shared protocol errors."""
    control, raw_key = _control_plane(tmp_path)
    for bad_value in ("", "x" * 513, "line\nbreak"):
        with pytest.raises(NativeBridgeError) as invalid:
            _claim_scope(control, raw_key, _chat_body(), idempotency_key=bad_value)
        payload = json.loads(invalid.value.public_error_json)
        assert payload["status_code"] == 400
        with pytest.raises(OpenAIProtocolError) as expected:
            decode_chat(json.loads(_chat_body()), idempotency_key=bad_value)
        assert payload["code"] == expected.value.detail.code
        assert payload["message"] == expected.value.detail.message
    with pytest.raises(NativeBridgeError) as mismatch:
        _claim_scope(
            control,
            raw_key,
            _chat_body(),
            idempotency_key="one",
            client_request_id="two",
        )
    payload = json.loads(mismatch.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["code"] == "idempotency_conflict"


def test_keyed_admissions_enforce_the_durable_ledger_idempotency_rows(
    tmp_path: Path,
) -> None:
    """With the process-local replay store empty (as after a restart), the
    durable ledger fails a repeated caller operation closed exactly as the
    python engine does: a different body conflicts and an identical body
    reports the replay unavailable, never a second provider dispatch."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit(control, raw_key, _chat_body(), idempotency_key="durable-op")
    control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    with pytest.raises(NativeBridgeError) as conflict:
        _admit(control, raw_key, _chat_body(stream=True), idempotency_key="durable-op")
    conflict_payload = json.loads(conflict.value.public_error_json)
    assert conflict_payload["status_code"] == 409
    assert conflict_payload["code"] == "idempotency_conflict"
    with pytest.raises(NativeBridgeError) as unavailable:
        _admit(control, raw_key, _chat_body(), idempotency_key="durable-op")
    unavailable_payload = json.loads(unavailable.value.public_error_json)
    assert unavailable_payload["status_code"] == 409
    assert unavailable_payload["code"] == "idempotency_replay_unavailable"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


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
    assert models["exp"]["authority_schema_version"] == 1
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


def _activate_revision_two(root: Path, manager: GatewayManagement) -> str:
    """Repoint the coding alias at a new revision and return its catalog digest."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-next",
        exact_model_id="model-revision-next",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=True,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    return normalized.identity_sha256()


def test_admission_authorized_at_the_swap_instant_stays_pinned_to_its_revision(
    tmp_path: Path,
) -> None:
    """An admission whose authority was minted just before a hot activation
    lands, serves on the retired revision instead of failing at the boundary."""
    manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    control = NativeControlPlane(components)
    ready = components.store
    assert isinstance(ready, _ReadyControlStore)
    inner = ready.store
    minted = threading.Event()
    swapped = threading.Event()
    original = inner.authorize_request

    def stalled_authorize(
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Mint the authorization, then stall until the activation swap lands."""
        authorization = original(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
        )
        minted.set()
        assert swapped.wait(timeout=10)
        return authorization

    outcomes: list[JsonObject] = []
    errors: list[BaseException] = []

    def admit_old() -> None:
        """Admit one request racing the activation swap."""
        try:
            outcomes.append(_admit(control, raw_key, _chat_body()))
        except BaseException as exc:  # noqa: BLE001 - the test asserts no error.
            errors.append(exc)

    racer = threading.Thread(target=admit_old)
    with mock.patch.object(inner, "authorize_request", side_effect=stalled_authorize):
        racer.start()
        assert minted.wait(timeout=10)
        new_digest = _activate_revision_two(tmp_path, manager)
    components.reloader.refresh_if_drifted(("coding", "revision-two", new_digest))
    swapped.set()
    racer.join(timeout=15)
    assert not racer.is_alive()

    assert errors == []
    assert len(outcomes) == 1
    pinned = outcomes[0]
    assert pinned["alias_revision_id"] == "revision-one"
    assert pinned["model_id"] == "provider-model-exact"
    fresh = _admit(control, raw_key, _chat_body())
    assert fresh["alias_revision_id"] == "revision-two"
    assert fresh["model_id"] == "provider-model-next"
    for admission in (pinned, fresh):
        assert (
            control.settle(
                json.dumps(
                    {
                        "request_id": admission["request_id"],
                        "attempt_id": admission["attempt_id"],
                        "outcome": "completed",
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                        "tool_names": [],
                        "failure": None,
                    }
                )
            )
            == "{}"
        )
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 2


def _settle_one_completed_chat(control: NativeControlPlane, raw_key: str) -> None:
    """Admit and settle one completed chat request for the presented key."""
    admission = _admit(control, raw_key, _chat_body())
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": admission["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"


def test_usage_callbacks_scope_reports_to_the_presented_key(tmp_path: Path) -> None:
    """A key sees only its own identity; anonymous callers see the whole organization."""
    control, raw_key = _control_plane(tmp_path)
    manager = GatewayManagement(tmp_path)
    manager.create_identity(identity_id="neighbor", display_name="Neighbor")
    manager.add_grant(identity_id="neighbor", alias_id="coding")
    neighbor_key = manager.issue_key(identity_id="neighbor", key_id="key-neighbor").raw_key
    _settle_one_completed_chat(control, raw_key)

    organization_wide = json.loads(control.usage_json("{}"))
    assert organization_wide["totals"]["requests"] == 1
    assert [item["identity_id"] for item in organization_wide["identities"]] == [
        "default",
        "neighbor",
    ]

    scoped = json.loads(control.usage_json(json.dumps({"raw_key": raw_key})))
    assert scoped["totals"]["requests"] == 1
    assert [item["identity_id"] for item in scoped["identities"]] == ["default"]

    isolated = json.loads(control.usage_json(json.dumps({"raw_key": neighbor_key})))
    assert isolated["totals"]["requests"] == 0
    assert isolated["totals"]["input_tokens"] == 0
    assert [item["identity_id"] for item in isolated["identities"]] == ["neighbor"]
    assert isolated["by_billing_source"] == []

    page = json.loads(control.usage_page(json.dumps({"raw_key": raw_key})))
    assert "Gateway usage" in page["html"]
    assert "default" in page["html"]
    isolated_page = json.loads(control.usage_page(json.dumps({"raw_key": neighbor_key})))
    assert "default" not in isolated_page["html"]


def test_usage_callbacks_reject_an_invalid_key(tmp_path: Path) -> None:
    """A bad key on either usage callback maps to the shared 401 public error."""
    control, _raw_key = _control_plane(tmp_path)
    for callback in (control.usage_json, control.usage_page):
        with pytest.raises(NativeBridgeError) as excinfo:
            callback(json.dumps({"raw_key": "exp_vk_invalid"}))
        payload = json.loads(excinfo.value.public_error_json)
        assert payload["status_code"] == 401
        assert payload["code"] == "invalid_key"


def _responses_body(
    *,
    model: str = "coding",
    stream: bool = False,
    previous_response_id: str | None = None,
    with_tools: bool = False,
) -> str:
    """Return one raw Responses API request body."""
    payload: JsonObject = {"model": model, "input": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    if with_tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": "search",
                "description": "Find things.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
        payload["tool_choice"] = "auto"
        payload["metadata"] = {"team": "core"}
        payload["max_output_tokens"] = 128
        payload["temperature"] = 0.5
    return json.dumps(payload)


def _admit_responses(control: NativeControlPlane, raw_key: str, body: str) -> JsonObject:
    """Run one Responses-surface admission call and decode its JSON response."""
    return json.loads(
        control.admit(json.dumps({"raw_key": raw_key, "body": body, "surface": "responses"}))
    )


def _admitted_request_id(admission: JsonObject) -> str:
    """Narrow one admission's request identity to a string."""
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    return request_id


def _payload_messages(admission: JsonObject) -> list[JsonObject]:
    """Narrow one admission's upstream payload to its message list."""
    payload = admission["upstream_payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    return [message for message in messages if isinstance(message, dict)]


def _responses_parity_case() -> tuple[list[GatewayEvent], str]:
    """Return matching python events and the Rust fixture JSON for one stream."""
    from exp.common.models.model import ToolCall

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
    return events, json.dumps(fixture)


def _python_responses_frames(
    body: str,
    events: list[GatewayEvent],
    *,
    created_at: float,
) -> list[str]:
    """Encode one Responses stream through the python encoder."""
    from exp.runtime.openai_protocol.streaming import ResponsesSseEncoder

    decoded = decode_responses(json.loads(body))
    encoder = ResponsesSseEncoder(
        request_id="request-abc",
        model=decoded.alias,
        created_at=created_at,
        request=decoded.request,
    )
    expected = list(encoder.start())
    for event in events:
        expected.extend(encoder.feed(event))
    return expected


def _native_envelope(body: str) -> str:
    """Render the control plane's Responses envelope JSON for one raw body."""
    from exp.runtime.gateway.native_bridge import _responses_envelope

    return json.dumps(_responses_envelope(decode_responses(json.loads(body)).request))


def test_rust_responses_sse_frames_match_python_encoder() -> None:
    """Rust Responses SSE frames are byte-identical to the python encoder."""
    native = pytest.importorskip("exp_gateway_native")
    body = _responses_body(stream=True, with_tools=True)
    events, fixture = _responses_parity_case()
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.25)
    actual = native.encode_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.25,
        _native_envelope(body),
        fixture,
    )
    assert list(actual) == expected


def test_rust_responses_refusal_and_incomplete_match_python_encoder() -> None:
    """Refusal deltas and the incomplete terminal render byte-identically."""
    native = pytest.importorskip("exp_gateway_native")

    body = _responses_body(stream=True)
    events = [
        GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=0, text_delta="I ca"),
        GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=1, text_delta="nnot"),
        GatewayEvent(kind=GatewayEventKind.INCOMPLETE, sequence_number=2),
    ]
    fixture = json.dumps(
        [
            {"kind": "refusal_delta", "text": "I ca"},
            {"kind": "refusal_delta", "text": "nnot"},
            {"kind": "incomplete"},
        ]
    )
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.0)
    actual = native.encode_responses_fixture(
        "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), fixture
    )
    assert list(actual) == expected


def test_rust_responses_failed_terminal_matches_python_encoder() -> None:
    """The failed terminal envelope and error body render byte-identically."""
    native = pytest.importorskip("exp_gateway_native")

    body = _responses_body(stream=True)
    events = [
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="part"),
        GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=1,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                safe_message="provider exploded",
            ),
        ),
    ]
    fixture = json.dumps(
        [
            {"kind": "text_delta", "text": "part"},
            {"kind": "failed", "text": "provider exploded"},
        ]
    )
    expected = _python_responses_frames(body, events, created_at=1_700_000_000.5)
    actual = native.encode_responses_fixture(
        "request-abc", "coding", 1_700_000_000.5, _native_envelope(body), fixture
    )
    assert list(actual) == expected


def test_rust_responses_completed_body_matches_python() -> None:
    """The Rust non-streaming Responses body equals the python completed_body."""
    native = pytest.importorskip("exp_gateway_native")
    from exp.runtime.openai_protocol.response import completed_body

    body = _responses_body(with_tools=True)
    events, fixture = _responses_parity_case()
    decoded = decode_responses(json.loads(body))
    expected = completed_body(
        request=decoded.request,
        request_id="request-abc",
        model=decoded.alias,
        created_at=1_700_000_000.25,
        events=tuple(events),
    )
    actual = native.completed_responses_fixture(
        "request-abc",
        "coding",
        1_700_000_000.25,
        _native_envelope(body),
        fixture,
    )
    assert json.loads(actual) == expected
    assert actual == json.dumps(expected, separators=(",", ":"), ensure_ascii=False)


def test_rust_responses_rejects_streams_without_terminals() -> None:
    """A provider stream without a terminal fails closed like the python path."""
    native = pytest.importorskip("exp_gateway_native")
    body = _responses_body()
    fixture = json.dumps([{"kind": "text_delta", "text": "no terminal"}])
    with pytest.raises(ValueError, match="all_routes_failed"):
        native.completed_responses_fixture(
            "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), fixture
        )
    malformed = json.dumps(
        [
            {"kind": "tool_arguments_delta", "index": 3, "text": "{"},
            {"kind": "completed"},
        ]
    )
    with pytest.raises(ValueError, match="invalid_provider_stream"):
        native.encode_responses_fixture(
            "request-abc", "coding", 1_700_000_000.0, _native_envelope(body), malformed
        )


def test_responses_admission_is_native_with_envelope_and_payload(tmp_path: Path) -> None:
    """A Responses request admits natively (no escalation) with the exact
    request-reflecting envelope and the shared dialect payload."""
    from exp.runtime.gateway.native_bridge import _responses_envelope

    control, raw_key = _control_plane(tmp_path)
    body = _responses_body(with_tools=True)
    admission = _admit_responses(control, raw_key, body)
    assert "escalate" not in admission
    assert admission["surface"] == "responses"
    assert admission["dialect"] == "openai_compatible"
    decoded = decode_responses(json.loads(body))
    assert admission["envelope"] == _responses_envelope(decoded.request)
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
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1


def test_responses_continuation_round_trip_and_fail_closed(tmp_path: Path) -> None:
    """Retained history is prepended on continuation; unknown, refused, and
    cross-namespace continuations fail closed with the shared public error."""
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body())
    remembered = control.remember(
        json.dumps(
            {
                "request_id": first["request_id"],
                "text": "The answer is 42.",
                "refusal": False,
                "tool_calls": [],
            }
        )
    )
    assert remembered == "{}"
    response_id = stable_public_id("resp", _admitted_request_id(first))
    second = _admit_responses(control, raw_key, _responses_body(previous_response_id=response_id))
    messages = _payload_messages(second)
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "The answer is 42."

    with pytest.raises(NativeBridgeError) as unknown:
        _admit_responses(control, raw_key, _responses_body(previous_response_id="resp_missing"))
    payload = json.loads(unknown.value.public_error_json)
    assert payload["status_code"] == 400
    assert payload["code"] == "continuation_unavailable"
    assert payload["param"] == "previous_response_id"

    refused = _admit_responses(control, raw_key, _responses_body())
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": refused["request_id"],
                    "text": "partial",
                    "refusal": True,
                    "tool_calls": [],
                }
            )
        )
        == "{}"
    )
    with pytest.raises(NativeBridgeError) as after_refusal:
        _admit_responses(
            control,
            raw_key,
            _responses_body(
                previous_response_id=stable_public_id("resp", _admitted_request_id(refused))
            ),
        )
    assert json.loads(after_refusal.value.public_error_json)["code"] == "continuation_unavailable"

    foreign = ProtocolNamespace(
        organization_id="other-org",
        identity_id="other-identity",
        alias_revision_id="other-revision",
    )
    with pytest.raises(OpenAIProtocolError) as crossed:
        control._continuations.resolve_now(  # noqa: SLF001 - namespace isolation assertion.
            namespace=foreign,
            previous_response_id=response_id,
        )
    assert crossed.value.detail.code == "continuation_unavailable"


def test_responses_tool_call_retention_survives_continuation(tmp_path: Path) -> None:
    """Completed tool calls are retained and replayed into continued history."""
    control, raw_key = _control_plane(tmp_path)
    first = _admit_responses(control, raw_key, _responses_body(with_tools=True))
    assert (
        control.remember(
            json.dumps(
                {
                    "request_id": first["request_id"],
                    "text": "",
                    "refusal": False,
                    "tool_calls": [
                        {"call_id": "call-9", "name": "search", "arguments": '{"q":"x"}'}
                    ],
                }
            )
        )
        == "{}"
    )
    second = _admit_responses(
        control,
        raw_key,
        _responses_body(previous_response_id=stable_public_id("resp", _admitted_request_id(first))),
    )
    assistant = _payload_messages(second)[1]
    assert assistant["role"] == "assistant"
    tool_calls = assistant["tool_calls"]
    assert isinstance(tool_calls, list)
    first_call = tool_calls[0]
    assert isinstance(first_call, dict)
    assert first_call["function"] == {"name": "search", "arguments": '{"q":"x"}'}


def test_responses_admission_rejects_invalid_bodies_and_bad_keys(tmp_path: Path) -> None:
    """Responses-surface admission fails closed on protocol and key errors."""
    control, raw_key = _control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as invalid:
        control.admit(json.dumps({"raw_key": raw_key, "body": "{not json", "surface": "responses"}))
    assert json.loads(invalid.value.public_error_json)["code"] == "invalid_json"
    rejected = json.dumps({"model": "coding", "input": "hi", "modalities": ["audio"]})
    with pytest.raises(NativeBridgeError) as protocol:
        control.admit(json.dumps({"raw_key": raw_key, "body": rejected, "surface": "responses"}))
    assert json.loads(protocol.value.public_error_json)["status_code"] == 400
    with pytest.raises(NativeBridgeError) as bad_key:
        control.admit(
            json.dumps(
                {
                    "raw_key": "exp_vk_invalid",
                    "body": _responses_body(),
                    "surface": "responses",
                }
            )
        )
    assert json.loads(bad_key.value.public_error_json)["status_code"] == 401
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0

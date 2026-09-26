"""Shared native capture contracts and real-socket serving isolation."""

import json
import socket
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from websockets.sync.client import ClientConnection, connect

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.capture_context import capture_context_document, restore_capture_context
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayApiSurface
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureDeliveryLimits,
    CaptureRecord,
    CaptureRequest,
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
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk
from exp.runtime.openai_protocol.requests import decode_chat

native = pytest.importorskip("exp_gateway_native")


@pytest.mark.parametrize("application", ["application", "", " ", "x" * 513])
@pytest.mark.parametrize("content", ["hello 雪", "a\x00b\ud800", "x" * 65536])
@pytest.mark.parametrize("number", [1.0, float("nan"), float("inf"), float("-inf")])
def test_controller_serializes_typed_context_once_and_native_validates_envelope(
    application: str,
    content: str,
    number: float,
) -> None:
    """No second Python schema walk; native authority checks still reject invalid scope."""
    authorization = AuthorizationSnapshot(
        request_id="request",
        organization_id="org",
        identity_id="identity",
        virtual_key_id="key",
        alias="coding",
        alias_revision_id="revision",
        target=DirectTarget(pool_id="pool"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        deadline_monotonic=1.0,
    )
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": content},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "number", "default": number}},
                        },
                    },
                }
            ],
        }
    ).request
    context = capture_context_document(request, session_id="episode")
    expected = CaptureRequest.model_validate(
        {
            "request_id": "request",
            "scope": {"organization_id": "org", "identity_id": "identity", "application_id": "app"},
            "protocol": "chat_completions",
            "model_id": "model",
            "context": context,
        }
    )
    expected_context = json.loads(expected.model_dump_json())["context"]
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    controller = CaptureController(collector, application_for=lambda _auth: application)
    with patch.object(CaptureRequest, "model_validate", side_effect=AssertionError("revalidation")):
        accepted = controller.begin(authorization, request, "model", session_id="episode")
    if application == "application":
        assert accepted
        collector.settle("request", True, False)
        assert collector.close(1)
        record = CaptureRecord.model_validate_json(records[0])
        assert record.request.context == expected_context
        assert record.request.scope.application_id == application
        assert record.request.model_id == "model"
        assert collector.counts() == (0, 0, 1, 0, 0, 0)
    else:
        assert not accepted
        assert collector.close(1)
        assert not records
        assert collector.counts() == (0, 0, 0, 0, 0, 1)


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


@pytest.mark.parametrize("suffix", ["a", "é", "雪", "😀"])
def test_bytes_admission_preserves_text_contract_without_python_unicode_copy(suffix: str) -> None:
    """The immutable-byte entry point stores exactly the same context as text admission."""
    request = json.loads(_request_json())
    request["context"]["request"]["messages"] = [{"role": "user", "content": "x" * 65536 + suffix}]
    encoded = json.dumps(request, ensure_ascii=False)
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin(encoded)
    collector.settle("request", True, False)
    assert collector.begin_bytes(encoded.encode("utf-8"))
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(records) == 2
    assert all(json.loads(record)["request"] == request for record in records)
    assert collector.counts() == (0, 0, 2, 0, 0, 0)


def test_bytes_admission_rejects_invalid_utf8_and_mutable_buffers() -> None:
    """Releasing the GIL never borrows a mutable buffer or admits invalid JSON bytes."""
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert not collector.begin_bytes(_request_json().encode().replace(b'"app"', b'"\xff"'))
    with pytest.raises(TypeError):
        collector.begin_bytes(bytearray(_request_json().encode()))
    assert collector.close(1)
    assert records == []
    assert collector.counts() == (0, 0, 0, 0, 0, 1)


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


@pytest.mark.parametrize("number", [2**64, 2**80 + 1, -(2**63) - 1, -(2**80) - 1])
@pytest.mark.parametrize("null_source", [False, True])
def test_admission_preserves_wide_numeric_tool_context_in_lossless_source(
    number: int, null_source: bool
) -> None:
    """Use the existing ingest restoration contract for out-of-range JSON integers."""
    request = json.loads(_request_json())
    if null_source:
        request["context"]["source_json"] = None
    request["context"]["request"]["tools"] = [{"name": "choose", "parameters": {"enum": [number]}}]
    expected = request["context"]
    records: list[str] = []
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    assert collector.begin_bytes(json.dumps(request).encode())
    collector.settle("request", True, False)
    assert collector.close(1)
    actual = CaptureRecord.model_validate_json(records[0]).request.context
    assert json.dumps(restore_capture_context(actual), sort_keys=True) == json.dumps(
        expected, sort_keys=True
    )


def test_wide_number_in_invalid_context_fails_closed_without_panicking() -> None:
    """A lossless sidecar cannot make a non-object context valid."""
    request = json.loads(_request_json())
    request["context"] = [2**80 + 1]
    collector = native.CaptureCollector(CaptureConfiguration().model_dump_json(), lambda _: None)
    assert not collector.begin_bytes(json.dumps(request).encode())
    assert collector.close(1)


def test_retained_async_handoffs_keep_native_batching() -> None:
    """A paused destination drains admitted records in batches without losing any."""
    entered, release = threading.Event(), threading.Event()
    batches: list[tuple[str, ...]] = []

    def write(records: tuple[str, ...]) -> list[bool]:
        """Hold the first write while every remaining request hands off independently."""
        entered.set()
        assert release.wait(3)
        batches.append(records)
        return [True] * len(records)

    configuration = CaptureConfiguration(
        asynchronous_delivery=True,
        maximum_pending_records=12,
        delivery=CaptureDeliveryLimits(maximum_records=8),
    )
    collector = native.CaptureCollector.batched(configuration.model_dump_json(), write)
    expected = {f"batch-{index}" for index in range(12)}
    for index in range(12):
        request = json.loads(_request_json())
        request["request_id"] = f"batch-{index}"
        assert collector.begin(json.dumps(request))

    def handoff() -> None:
        """Complete more records than the destination queue can currently accept."""
        for index in range(1, 12):
            collector.settle(f"batch-{index}", True, False)

    producer = threading.Thread(target=handoff)
    try:
        collector.settle("batch-0", True, False)
        assert entered.wait(1)
        producer.start()
        producer.join(1)
        assert not producer.is_alive()
        assert collector.counts()[0] == 12
        assert not collector.close(0)
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(3)
        assert collector.close(3)
    assert any(len(batch) > 1 for batch in batches)
    assert {
        json.loads(value)["request"]["request_id"] for batch in batches for value in batch
    } == expected
    assert sum(map(len, batches)) == len(expected)
    assert collector.counts() == (0, 0, len(expected), 0, 0, 0)


@pytest.mark.parametrize("asynchronous_delivery", [False, True])
def test_python_sink_retries_without_losing_content_or_acknowledging_failure(
    capfd: pytest.CaptureFixture[str],
    asynchronous_delivery: bool,
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

    config = CaptureConfiguration(asynchronous_delivery=asynchronous_delivery)
    collector = native.CaptureCollector(config.model_dump_json(), write)
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
        settlement.join(1)
        assert settlement.is_alive() is not asynchronous_delivery
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


@pytest.mark.parametrize("bytes_output", [False, True])
def test_batched_sink_preserves_strings_and_retries_only_unacknowledged_members(
    bytes_output: bool,
) -> None:
    """One failed record cannot hold healthy peers; retries reuse prepared objects."""
    entered = threading.Event()
    release = threading.Event()
    recover = threading.Event()
    healthy = threading.Event()
    batches: list[tuple[str | bytes, ...]] = []
    persisted: set[str] = set()
    failed_strings: list[str | bytes] = []

    def write(records: tuple[str | bytes, ...]) -> list[bool]:
        """Keep one record unavailable while acknowledging all of its neighbors."""
        assert threading.current_thread() is not threading.main_thread()
        batches.append(records)
        entered.set()
        assert release.wait(5)
        outcomes = []
        for encoded in records:
            assert isinstance(encoded, bytes if bytes_output else str)
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

    collector = (
        native.CaptureCollector.batched(
            CaptureConfiguration().model_dump_json(), write, bytes_output=True
        )
        if bytes_output
        else native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    )
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


@pytest.mark.parametrize("bytes_output", [False, True])
def test_batched_sink_invalid_acknowledgements_never_release_records(bytes_output: bool) -> None:
    """Exceptions and mismatched receipt counts retain the same prepared payload."""
    seen: list[str | bytes] = []

    def write(records: tuple[str | bytes, ...]) -> list[bool]:
        """Recover only after exercising both invalid callback outcomes."""
        seen.append(records[0])
        if len(seen) == 1:
            raise RuntimeError("private destination error")
        if len(seen) == 2:
            return []
        return [True]

    collector = (
        native.CaptureCollector.batched(
            CaptureConfiguration().model_dump_json(), write, bytes_output=True
        )
        if bytes_output
        else native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    )
    assert collector.begin(_request_json())
    collector.settle("request", True, False)
    assert collector.close(1)
    assert len(seen) == 3 and all(encoded is seen[0] for encoded in seen)
    assert collector.counts() == (0, 0, 1, 2, 0, 0)


@pytest.mark.parametrize(("count", "size"), [(80, 0), (8, 600_000)])
@pytest.mark.parametrize("bytes_output", [False, True])
def test_batched_sink_bounds_count_and_bytes(count: int, size: int, bytes_output: bool) -> None:
    """Queued work fills byte- and count-bounded batches without losing Unicode."""
    entered, release = threading.Event(), threading.Event()
    batches: list[tuple[str | bytes, ...]] = []

    def write(records: tuple[str | bytes, ...]) -> list[bool]:
        batches.append(records)
        entered.set()
        assert release.wait(5)
        assert len(records) <= 64
        sizes = [len(record.encode() if isinstance(record, str) else record) for record in records]
        assert sum(sizes) <= 10 * 1024 * 1024
        if len(records) > 1:
            assert sum(sizes[:-1]) < 2 * 1024 * 1024
        return [True] * len(records)

    collector = (
        native.CaptureCollector.batched(
            CaptureConfiguration().model_dump_json(), write, bytes_output=True
        )
        if bytes_output
        else native.CaptureCollector.batched(CaptureConfiguration().model_dump_json(), write)
    )
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
        "hosted-batched",
        "hosted-batched-references",
        "hosted-batched-bytes-references",
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
    destination = policy
    if policy.startswith("hosted-batched"):
        policy = "hosted"
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    _LoopbackProvider.calls = 0
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    records: list[str | bytes] = []
    configuration = CaptureConfiguration(
        settlement_required=policy.startswith("hosted"),
        asynchronous_delivery=policy.startswith("hosted"),
        delivery=(
            CaptureDeliveryLimits(maximum_record_bytes=1024, maximum_bytes=6400)
            if policy == "hosted-checkpoint-failed"
            else CaptureDeliveryLimits()
        ),
    )

    def write_batch(values: tuple[str | bytes, ...]) -> list[bool]:
        records.extend(values)
        return [True] * len(values)

    if destination == "hosted-batched-bytes-references":
        collector = native.CaptureCollector.batched(
            configuration.model_dump_json(),
            write_batch,
            completion_references=True,
            bytes_output=True,
        )
    elif destination == "hosted-batched-references":
        collector = native.CaptureCollector.batched(
            configuration.model_dump_json(), write_batch, completion_references=True
        )
    elif destination == "hosted-batched":
        collector = native.CaptureCollector.batched(configuration.model_dump_json(), write_batch)
    else:
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
    if destination in {"hosted-batched-references", "hosted-batched-bytes-references"}:
        updates = [json.loads(value) for value in records]
        checkpoints = {
            value["request"]["request_id"]: value["request"]
            for value in updates
            if value["schema_version"] == 1
        }
        completions = [value for value in updates if value["schema_version"] == 2]
        assert len(checkpoints) == len(completions) == 6
        for value in completions:
            assert "context" not in value["request"]
            request = checkpoints[value["request"]["request_id"]]
            assert value["request"] == {
                key: field for key, field in request.items() if key != "context"
            }
            value["request"] = request
            value["schema_version"] = 1
        parsed = [CaptureRecord.model_validate(value) for value in completions]
    else:
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
    encoded_records = "".join(
        value.decode() if isinstance(value, bytes) else value for value in records
    )
    assert "provider-secret" not in encoded_records
    assert raw_key not in encoded_records
    assert "do-not-capture-me" not in encoded_records


@pytest.mark.parametrize("surface", ["chat/completions", "responses", "messages", "responses-ws"])
@pytest.mark.parametrize("ending", ["disconnect", "deadline"])
@pytest.mark.parametrize("retention", ["discard", "prompt", "response"])
@pytest.mark.parametrize(("asynchronous", "blocked"), [(False, True), (True, True), (True, False)])
def test_pending_checkpoint_does_not_pin_transport_or_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    ending: str,
    retention: str,
    asynchronous: bool,
    blocked: bool,
) -> None:
    """A refused sink keeps capture ownership, not the request's transport or reserved attempt."""
    provider_closed, write_attempted, allow_write, settled = (threading.Event() for _ in range(4))
    if not blocked:
        # Control arm: asynchronous deadline behavior must match a healthy sink.
        allow_write.set()
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

    configuration = CaptureConfiguration(
        asynchronous_delivery=asynchronous,
        delivery=CaptureDeliveryLimits(maximum_records=1),
    )
    collector = native.CaptureCollector(configuration.model_dump_json(), persist)
    if asynchronous:
        # Occupy the sole delivery slot. Admitted async requests must still
        # expose output while their checkpoint remains owned behind that slot.
        assert collector.begin(_request_json())
        collector.settle("request", True, False)
        assert write_attempted.wait(2)
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
        if asynchronous:
            until = time.monotonic() + 2
            while not calls and time.monotonic() < until:
                time.sleep(0.01)
            assert len(calls) == 1
            visible = ""
            while "checkpoint-visible" not in visible:
                if ws is not None:
                    visible += str(ws.recv(timeout=1))
                else:
                    assert client is not None
                    client.settimeout(1)
                    chunk = client.recv(4096)
                    assert chunk, "stream closed before output despite available capture memory"
                    visible += chunk.decode()
            if blocked:
                count, retained, *_ = collector.counts()
                assert count == 2
                assert retained <= (
                    configuration.delivery.maximum_bytes + configuration.maximum_pending_bytes
                )
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
        # With output flowing, provider timeout and request cancellation race.
        # Both also occur with healthy storage. Pin each existing accounting
        # contract, rather than making capture scheduling choose the winner.
        expected_states = (
            {"failed", "cancelled"} if asynchronous and ending == "deadline" else {"cancelled"}
        )
        assert len(rows) == 1 and rows[0][0] == settlements[0]["attempt_id"]
        assert rows[0][1] in expected_states
        if rows[0][1] == "failed":
            assert rows[0][2:5] == (None, None, "unknown")
        else:
            assert rows[0][2] > 0 and rows[0][3] > 0 and rows[0][4] == "estimated"
        if blocked:
            assert not records
            assert not collector.close(0) and not collector.close(0)
        shutdown.request_shutdown()
        worker.join(2)
        assert not worker.is_alive(), "capture retry pinned runtime shutdown"
        allow_write.set()
        assert collector.close(3)
        own_records = [
            value
            for value in records
            if json.loads(value)["request"]["request_id"] == settlements[0]["request_id"]
        ]
        assert len(own_records) == (1 if retention == "discard" else 2)
        assert len(records) == len(own_records) + int(asynchronous)
        assert all(value == settlements[0] for value in settlements) and len(calls) == 1
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute(f"SELECT {columns} FROM gateway_attempts").fetchall() == rows
        saved = CaptureRecord.model_validate_json(own_records[0])
        assert saved.schema_version == 1 and saved.response is None
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
    assert not errors

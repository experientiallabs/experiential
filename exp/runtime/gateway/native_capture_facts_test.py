"""Capture facts: the model, the byte-splice storage plan, and the native batched output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import JsonValue, ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureDeliveryLimits,
    CaptureSpillConfiguration,
)
from exp.runtime.gateway.native_capture_facts import (
    CAPTURE_FACTS_TAG,
    CAPTURE_FACTS_VERSION,
    CaptureRecordFacts,
)

if TYPE_CHECKING:
    from exp_gateway_native import CaptureCollector, FactsRecord

native = pytest.importorskip("exp_gateway_native")


def rewrite(record: bytes, *, keep_prompt: bool, keep_response: bool) -> JsonObject:
    """Apply a storage plan the slow way: decode, edit, compare as JSON.

    Args:
        record: Record bytes.
        keep_prompt: Whether the prompt may be stored.
        keep_response: Whether the response may be stored.

    Returns:
        The decoded record with the dropped parts removed.
    """
    document = json.loads(record)
    if not keep_response:
        document["response"] = None
    if not keep_prompt:
        context = document["request"]["context"]
        context["request"]["messages"] = []
        context.pop("source_json", None)
        context.pop("provider_context", None)
        transport = document.get("transport")
        if isinstance(transport, dict):
            transport.pop("wire_request", None)
    return document


def _request(request_id: str, context: JsonObject) -> str:
    """Return one authenticated capture admission with ``context``."""
    return json.dumps(
        {
            "request_id": request_id,
            "scope": {"organization_id": "org", "identity_id": "identity", "application_id": "app"},
            "protocol": "chat_completions",
            "model_id": "model",
            "context": context,
        }
    )


def _collect(
    configuration: CaptureConfiguration, *, facts_output: bool
) -> tuple[CaptureCollector, list[FactsRecord | bytes]]:
    """A batched bytes-output collector that keeps everything it is handed."""
    received: list[FactsRecord | bytes] = []

    def write(records: tuple[FactsRecord | bytes, ...]) -> list[bool]:
        """Keep and acknowledge every record."""
        received.extend(records)
        return [True] * len(records)

    collector = native.CaptureCollector.batched(
        configuration.model_dump_json(), write, bytes_output=True, facts_output=facts_output
    )
    return collector, received


def test_the_native_module_advertises_the_facts_contract() -> None:
    """Hosts feature-detect facts output by this constant."""
    assert native.CAPTURE_FACTS_VERSION == CAPTURE_FACTS_VERSION


def test_facts_output_requires_bytes_output() -> None:
    """Facts sit beside record bytes, never beside a Python string."""
    with pytest.raises(ValueError, match="requires bytes output"):
        native.CaptureCollector.batched(
            CaptureConfiguration().model_dump_json(), lambda _records: [], facts_output=True
        )


CONTEXTS: tuple[JsonObject, ...] = (
    {"schema_version": 1, "request": {"messages": [{"role": "user", "content": "hi 雪"}]}},
    {
        "source_json": '{"a": 1}',
        "schema_version": 1,
        "request": {"model": "m", "messages": [{"role": "user", "content": "x,}]"}], "n": 1},
        "provider_context": {"reasoning": "r"},
    },
    {"provider_context": [1], "request": {"messages": []}, "schema_version": 1},
)


@pytest.mark.parametrize("context", CONTEXTS)
def test_facts_ride_beside_the_unchanged_record_bytes(context: JsonObject) -> None:
    """The record bytes are those plain bytes output hands; the facts describe them."""
    configuration = CaptureConfiguration(settlement_required=False)
    plain, plain_received = _collect(configuration, facts_output=False)
    faceted, facts_received = _collect(configuration, facts_output=True)
    for collector in (plain, faceted):
        assert collector.begin(_request("request-1", context))
        collector.settle("request-1", True, False)
        assert collector.close(5)
    [plain_record] = plain_received
    [faceted_record] = facts_received
    assert isinstance(faceted_record, tuple)
    tag, facts_json, record = faceted_record
    assert tag == CAPTURE_FACTS_TAG
    # Byte identity of the two encodings is pinned natively (facts_test.rs);
    # here only the admission time differs between the two collectors.
    assert isinstance(record, bytes) and isinstance(plain_record, bytes)
    document, plain_document = json.loads(record), json.loads(plain_record)
    captured_at = document.pop("captured_at")
    plain_document.pop("captured_at")
    assert document == plain_document
    facts = CaptureRecordFacts.model_validate_json(facts_json)
    assert facts.request_id == "request-1"
    assert facts.scope.organization_id == "org"
    assert facts.protocol == "chat_completions"
    assert facts.model_id == "model"
    assert facts.captured_at == captured_at
    assert not facts.complete and facts.response is None
    assert not facts.split and facts.record_bytes == len(record)
    assert b"hi" not in facts_json and "雪".encode() not in facts_json
    for keep_prompt in (True, False):
        for keep_response in (True, False):
            planned = facts.planned(record, keep_prompt=keep_prompt, keep_response=keep_response)
            assert json.loads(planned) == rewrite(
                record, keep_prompt=keep_prompt, keep_response=keep_response
            )


def test_a_split_record_keeps_its_tuple_and_has_no_spans(tmp_path: Path) -> None:
    """A spilled request arrives as facts beside the same split tuple as before."""
    configuration = CaptureConfiguration(
        settlement_required=False,
        maximum_request_bytes=64 * 1024 * 1024,
        maximum_pending_bytes=128 * 1024 * 1024,
        delivery=CaptureDeliveryLimits(
            maximum_record_bytes=1024 * 1024, maximum_bytes=64 * 1024 * 1024
        ),
        spill=CaptureSpillConfiguration(directory=str(tmp_path / "spill")),
    )
    received: list[tuple[CaptureRecordFacts, str, int]] = []

    def write(records: tuple[FactsRecord, ...]) -> list[bool]:
        """Check each split member while its part files exist, then acknowledge."""
        for tag, facts_json, record in records:
            assert tag == CAPTURE_FACTS_TAG
            assert isinstance(record, tuple)
            kind, skeleton, _marker, parts = record
            assert kind == "split"
            facts = CaptureRecordFacts.model_validate_json(facts_json)
            assert facts.record_bytes == len(skeleton)
            received.append((facts, parts[0][0], parts[0][2]))
        return [True] * len(records)

    collector = native.CaptureCollector.batched(
        configuration.model_dump_json(), write, bytes_output=True, facts_output=True
    )
    content = "x" * (4 * 1024 * 1024)
    context: JsonObject = {
        "schema_version": 1,
        "request": {"messages": [{"role": "user", "content": content}]},
    }
    assert collector.begin(_request("request-1", context))
    collector.settle("request-1", True, False)
    assert collector.close(30)
    [(facts, part_kind, part_bytes)] = received
    assert facts.split and facts.spans is None
    assert part_kind == "json" and part_bytes > len(content)
    with pytest.raises(ValueError, match="split skeleton"):
        facts.planned(b"{}", keep_prompt=False, keep_response=True)
    assert facts.planned(b"{}", keep_prompt=True, keep_response=True) == b"{}"


def _facts(**overrides: JsonValue) -> JsonObject:
    """A minimal valid facts document with ``overrides``."""
    document: JsonObject = {
        "facts_version": 1,
        "schema_version": 1,
        "request_id": "r",
        "scope": {"organization_id": "o", "identity_id": "i", "application_id": "a"},
        "protocol": "messages",
        "model_id": None,
        "deployment_id": None,
        "captured_at": 1.0,
        "complete": False,
        "response": None,
        "split": False,
        "record_bytes": 2,
        "spans": {
            "messages": None,
            "response": None,
            "source_json": None,
            "provider_context": None,
            "wire_request": None,
        },
    }
    return {**document, **overrides}


def test_inconsistent_facts_and_mismatched_records_are_refused() -> None:
    """The model refuses contradictions; the splice refuses a different record."""
    with pytest.raises(ValidationError):
        CaptureRecordFacts.model_validate(_facts(split=True))
    with pytest.raises(ValidationError):
        CaptureRecordFacts.model_validate(
            _facts(
                response={
                    "kind": "json",
                    "status": 200,
                    "truncated": False,
                    "client_disconnected": False,
                }
            )
        )
    with pytest.raises(ValidationError):
        CaptureRecordFacts.model_validate(_facts(facts_version=2))
    facts = CaptureRecordFacts.model_validate(_facts())
    with pytest.raises(ValueError, match="do not match"):
        facts.planned(b"{}  ", keep_prompt=False, keep_response=True)
    overlapping = CaptureRecordFacts.model_validate(
        _facts(
            record_bytes=10,
            spans={
                "messages": [2, 6],
                "response": [4, 8],
                "source_json": None,
                "provider_context": None,
                "wire_request": None,
            },
        )
    )
    with pytest.raises(ValueError, match="out of order"):
        overlapping.planned(b"0123456789", keep_prompt=False, keep_response=False)

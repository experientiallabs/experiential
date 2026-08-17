"""Behavior tests for strict OpenTelemetry GenAI canonical normalization."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.models import ConnectionConfig
from wmo.simulation.ingest.otlp import load_otlp_file, normalize_otlp_payload
from wmo.simulation.mining.descriptors import routing_descriptor

_TRACE_ID = "1" * 32
_CALL_SPAN_ID = "2" * 16
_TOOL_SPAN_ID = "3" * 16
_LATE_SPAN_ID = "4" * 16


def _attribute(key: str, value: object) -> dict[str, object]:
    if isinstance(value, bool):
        encoded: dict[str, object] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    else:
        encoded = {"stringValue": value}
    return {"key": key, "value": encoded}


def _payload(trace_id: str = _TRACE_ID) -> dict[str, object]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attribute("service.name", "support-agent")]},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": _CALL_SPAN_ID,
                                "name": "agent.model_call",
                                "startTimeUnixNano": "1760000000000000000",
                                "endTimeUnixNano": "1760000001000000000",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "chat"),
                                    _attribute("gen_ai.provider.name", "openai"),
                                    _attribute("gen_ai.request.model", "gpt-test"),
                                    _attribute(
                                        "gen_ai.input.messages",
                                        json.dumps(
                                            [
                                                {
                                                    "role": "user",
                                                    "content": "Cancel reservation R-17",
                                                }
                                            ]
                                        ),
                                    ),
                                    _attribute("gen_ai.tool.name", "cancel_reservation"),
                                    _attribute("gen_ai.tool.call.id", "call-1"),
                                    _attribute(
                                        "gen_ai.tool.definitions",
                                        json.dumps(
                                            [
                                                {
                                                    "name": "cancel_reservation",
                                                    "description": "Cancel one reservation.",
                                                    "input_schema": {"type": "object"},
                                                }
                                            ]
                                        ),
                                    ),
                                    _attribute("wmo.request.context", json.dumps({"tier": "gold"})),
                                    _attribute("wmo.request.tags", json.dumps(["domain:travel"])),
                                    _attribute("wmo.customer.id", "customer-7"),
                                    _attribute("wmo.conversation.id", "conversation-9"),
                                    _attribute("wmo.outcome.status", "success"),
                                    _attribute("wmo.outcome.name", "reservation_cancelled"),
                                ],
                            },
                            {
                                "traceId": trace_id,
                                "spanId": _TOOL_SPAN_ID,
                                "parentSpanId": _CALL_SPAN_ID,
                                "name": "agent.tool_call",
                                "startTimeUnixNano": "1760000001000000000",
                                "endTimeUnixNano": "1760000002000000000",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "execute_tool"),
                                    _attribute("gen_ai.tool.name", "cancel_reservation"),
                                    _attribute("gen_ai.tool.call.id", "call-1"),
                                    _attribute("gen_ai.tool.message", "Reservation cancelled"),
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }


def _source() -> SourceIdentity:
    return SourceIdentity(kind="otlp", source_id="fixture", sha256="a" * 64)


def _environment_capture_payloads(trace_id: str = _TRACE_ID) -> list[dict[str, object]]:
    """Return one exact environment-capture action and observation pair.

    Args:
        trace_id: W3C trace identity assigned to both records.

    Returns:
        Source action and result spans in JSONL order.
    """
    prefix = f"{trace_id[:12]}0000"
    return [
        {
            "traceId": trace_id,
            "spanId": f"{prefix}a",
            "parentSpanId": "",
            "name": "chat terminal",
            "startTimeUnixNano": 0,
            "endTimeUnixNano": 1,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": [
                _attribute("gen_ai.operation.name", "chat"),
                _attribute("gen_ai.request.model", "terminal-agent"),
                _attribute("gen_ai.tool.name", "bash"),
                _attribute(
                    "gen_ai.tool.call.arguments",
                    json.dumps({"command": "printf ready"}),
                ),
                _attribute("gen_ai.prompt", "Print ready"),
                _attribute(
                    "wmh.trace.metadata",
                    json.dumps(
                        {
                            "benchmark": "terminal-tasks",
                            "returncode": 0,
                            "task_category": "Filesystem + text processing",
                        }
                    ),
                ),
            ],
        },
        {
            "traceId": trace_id,
            "spanId": f"{prefix}b",
            "parentSpanId": "",
            "name": "execute_tool terminal",
            "startTimeUnixNano": 2,
            "endTimeUnixNano": 3,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": [
                _attribute("gen_ai.operation.name", "execute_tool"),
                _attribute("gen_ai.tool.name", "bash"),
                _attribute("gen_ai.tool.message", "ready"),
            ],
        },
    ]


def _payload_spans(payload: dict[str, object]) -> list[object]:
    """Return the mutable OTLP fixture span list with concrete test-side casts."""
    resource_spans = cast(list[object], payload["resourceSpans"])
    resource_span = cast(dict[str, object], resource_spans[0])
    scope_spans = cast(list[object], resource_span["scopeSpans"])
    scope_span = cast(dict[str, object], scope_spans[0])
    return cast(list[object], scope_span["spans"])


def _span(payload: dict[str, object], index: int) -> dict[str, object]:
    """Return one mutable source span from an OTLP fixture."""
    return cast(dict[str, object], _payload_spans(payload)[index])


def _replace_attribute(
    payload: dict[str, object], *, span_index: int, key: str, value: object
) -> None:
    """Replace one fixture attribute while retaining its OTLP AnyValue encoding."""
    attributes = cast(list[dict[str, object]], _span(payload, span_index)["attributes"])
    for attribute in attributes:
        if attribute["key"] == key:
            attribute["value"] = _attribute(key, value)["value"]
            return
    raise AssertionError(f"fixture has no attribute {key!r}")


def _payload_with_late_request() -> dict[str, object]:
    """Return a fixture whose later span tries to alter request-visible evidence."""
    payload = _payload()
    spans = _payload_spans(payload)
    spans.append(
        {
            "traceId": _TRACE_ID,
            "spanId": _LATE_SPAN_ID,
            "name": "agent.later_model_call",
            "startTimeUnixNano": "1760000003000000000",
            "endTimeUnixNano": "1760000004000000000",
            "attributes": [
                _attribute("gen_ai.operation.name", "chat"),
                _attribute("gen_ai.provider.name", "openai"),
                _attribute("gen_ai.request.model", "gpt-test"),
                _attribute(
                    "gen_ai.input.messages",
                    json.dumps([{"role": "user", "content": "Later secret task"}]),
                ),
                _attribute("wmo.request.context", json.dumps({"tier": "secret"})),
                _attribute("wmo.request.tags", json.dumps(["domain:secret"])),
                _attribute(
                    "gen_ai.tool.definitions",
                    json.dumps(
                        [
                            {
                                "name": "delete_reservation",
                                "description": "Delete one reservation.",
                                "input_schema": {"type": "object"},
                            }
                        ]
                    ),
                ),
            ],
        }
    )
    return payload


def test_normalizes_w3c_genai_trace_and_wmo_outcome_extensions() -> None:
    result = normalize_otlp_payload(_payload(), source=_source())

    assert result.issues == ()
    assert result.identity_evidence is not None
    assert result.identity_evidence[0].capabilities == "inferred"
    assert result.identity_evidence[0].connection == "inferred"
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.trace_id == _TRACE_ID
    assert trace.conversation_id == "conversation-9"
    assert trace.task == "Cancel reservation R-17"
    assert trace.initial_context == {"tier": "gold"}
    assert trace.tools[0].name == "cancel_reservation"
    assert trace.outcome is not None and trace.outcome.status == "success"
    assert trace.spans[0].model is not None
    assert trace.spans[0].model.provider == "openai"
    assert (
        trace.spans[0].model.connection_sha256
        == ConnectionConfig(provider="openai").identity_sha256()
    )
    assert trace.spans[1].parent_span_id == _CALL_SPAN_ID


def test_otlp_retains_a_declared_model_connection_digest() -> None:
    """An exporter can retain exact secret-free connection evidence without an endpoint URL."""
    payload = _payload()
    attributes = cast(list[dict[str, object]], _span(payload, 0)["attributes"])
    attributes.append(_attribute("wmo.model.connection_sha256", "d" * 64))

    result = normalize_otlp_payload(payload, source=_source())

    assert result.issues == ()
    assert result.traces[0].spans[0].model is not None
    assert result.traces[0].spans[0].model.connection_sha256 == "d" * 64
    assert result.identity_evidence is not None
    assert result.identity_evidence[0].capabilities == "inferred"
    assert result.identity_evidence[0].connection == "declared"


def test_otlp_retains_independent_declared_model_identity_components() -> None:
    """Capability and connection provenance remain independently declared."""
    payload = _payload()
    attributes = cast(list[dict[str, object]], _span(payload, 0)["attributes"])
    attributes.extend(
        (
            _attribute("wmo.model.capabilities_sha256", "c" * 64),
            _attribute("wmo.model.connection_sha256", "d" * 64),
        )
    )

    result = normalize_otlp_payload(payload, source=_source())

    assert result.issues == ()
    assert result.identity_evidence is not None
    assert result.identity_evidence[0].capabilities == "declared"
    assert result.identity_evidence[0].connection == "declared"
    assert result.traces[0].spans[0].model is not None
    assert result.traces[0].spans[0].model.capabilities_sha256 == "c" * 64


def test_otlp_rejects_an_invalid_declared_model_connection_digest() -> None:
    """Connection evidence must be a canonical SHA-256 digest, not unstructured endpoint data."""
    payload = _payload()
    attributes = cast(list[dict[str, object]], _span(payload, 0)["attributes"])
    attributes.append(_attribute("wmo.model.connection_sha256", "not-a-digest"))

    result = normalize_otlp_payload(payload, source=_source())

    assert result.traces == ()
    assert "connection_sha256" in result.issues[0].message


def test_otlp_accepts_a_causal_matching_tool_pair() -> None:
    """A result that follows its matching call and names it as parent remains valid."""
    result = normalize_otlp_payload(_payload(), source=_source())

    assert result.issues == ()
    call, tool_result = result.traces[0].spans
    assert tool_result.started_at >= call.ended_at
    assert tool_result.parent_span_id == call.span_id
    assert tool_result.attributes["gen_ai.tool.name"] == call.attributes["gen_ai.tool.name"]


def test_otlp_rejects_a_tool_result_that_precedes_its_call() -> None:
    """An explicit result cannot begin before the corresponding model call completes."""
    payload = _payload()
    tool_result = _span(payload, 1)
    tool_result["startTimeUnixNano"] = "1759999998000000000"
    tool_result["endTimeUnixNano"] = "1759999999000000000"

    result = normalize_otlp_payload(payload, source=_source())

    assert result.traces == ()
    assert "starts before paired call completes" in result.issues[0].message


def test_otlp_rejects_mismatched_explicit_tool_pair_names() -> None:
    """A tool result cannot reuse a call ID for a different named tool."""
    payload = _payload()
    _replace_attribute(
        payload,
        span_index=1,
        key="gen_ai.tool.name",
        value="delete_reservation",
    )

    result = normalize_otlp_payload(payload, source=_source())

    assert result.traces == ()
    assert "not paired call name" in result.issues[0].message


def test_otlp_rejects_a_tool_result_with_a_contradictory_parent() -> None:
    """A recorded tool-result parent must be the paired model call, when present."""
    payload = _payload_with_late_request()
    _span(payload, 1)["parentSpanId"] = _LATE_SPAN_ID

    result = normalize_otlp_payload(payload, source=_source())

    assert result.traces == ()
    assert "parent contradicts paired call span" in result.issues[0].message


def test_invalid_w3c_identity_excludes_the_trace_with_a_clear_issue() -> None:
    result = normalize_otlp_payload(_payload(trace_id="not-a-w3c-trace"), source=_source())

    assert result.traces == ()
    assert len(result.issues) == 1
    assert "W3C ID" in result.issues[0].message


def test_jsonl_preserves_a_malformed_line_as_an_explicit_exclusion(tmp_path: Path) -> None:
    payload = _payload()
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(payload) + "\n{bad-json\n", encoding="utf-8")

    result = load_otlp_file(path)

    assert len(result.traces) == 1
    assert result.invalid_trace_count == 1
    assert result.traces[0].source.identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_environment_capture_jsonl_normalizes_through_default_otlp_loader(tmp_path: Path) -> None:
    """The exact owned profile loads without a converter or source override.

    Args:
        tmp_path: Temporary directory receiving the direct-span JSONL fixture.
    """
    path = tmp_path / "traces.otel.jsonl"
    path.write_text(
        "".join(f"{json.dumps(payload)}\n" for payload in _environment_capture_payloads()),
        encoding="utf-8",
    )

    result = load_otlp_file(path)

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "Print ready"
    assert trace.conversation_id == _TRACE_ID
    assert tuple(span.span_id for span in trace.spans) == (
        "111111111110000a",
        "111111111110000b",
    )
    assert trace.spans[0].parent_span_id is None
    assert trace.spans[1].parent_span_id == trace.spans[0].span_id
    assert trace.spans[0].attributes["gen_ai.tool.call.id"]
    assert (
        trace.spans[1].attributes["gen_ai.tool.call.id"]
        == trace.spans[0].attributes["gen_ai.tool.call.id"]
    )
    assert trace.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
    assert trace.spans[0].model is None
    assert result.identity_evidence == ()
    assert trace.source.identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_environment_capture_trace_ids_become_distinct_conversation_lineages(
    tmp_path: Path,
) -> None:
    """Each source episode receives its own stable conversation lineage.

    Args:
        tmp_path: Temporary directory receiving a two-trace JSONL fixture.
    """
    second_trace_id = "5" * 32
    payloads = _environment_capture_payloads() + _environment_capture_payloads(second_trace_id)
    path = tmp_path / "two-traces.otel.jsonl"
    path.write_text(
        "".join(f"{json.dumps(payload)}\n" for payload in payloads),
        encoding="utf-8",
    )

    result = load_otlp_file(path)

    assert result.issues == ()
    assert {trace.conversation_id for trace in result.traces} == {_TRACE_ID, second_trace_id}


def test_environment_capture_near_match_remains_strict_otlp_failure(tmp_path: Path) -> None:
    """A provider-bearing near-match cannot use profile identity repair.

    Args:
        tmp_path: Temporary directory receiving the near-match JSONL fixture.
    """
    payloads = copy.deepcopy(_environment_capture_payloads())
    attributes = cast(list[dict[str, object]], payloads[0]["attributes"])
    attributes.append(_attribute("gen_ai.provider.name", "openai"))
    path = tmp_path / "near-match.otel.jsonl"
    path.write_text(
        "".join(f"{json.dumps(payload)}\n" for payload in payloads),
        encoding="utf-8",
    )

    result = load_otlp_file(path)

    assert result.traces == ()
    assert len(result.issues) == 1
    assert "span ID must be a non-zero lowercase 16-hex W3C ID" in result.issues[0].message


def test_environment_capture_malformed_line_disables_whole_file_repair(tmp_path: Path) -> None:
    """Do not repair surviving capture records after a JSONL parse failure.

    Args:
        tmp_path: Temporary directory receiving the malformed mixed JSONL file.
    """
    records = "".join(f"{json.dumps(payload)}\n" for payload in _environment_capture_payloads())
    path = tmp_path / "malformed-profile.otel.jsonl"
    path.write_text(f"{{broken\n{records}", encoding="utf-8")

    result = load_otlp_file(path)

    assert result.traces == ()
    assert any(issue.source_record == "line-1" for issue in result.issues)
    assert any(
        "span ID must be a non-zero lowercase 16-hex W3C ID" in issue.message
        for issue in result.issues
    )


@pytest.mark.parametrize("duplicate_target", ["span", "status"])
def test_environment_capture_duplicate_json_key_disables_whole_file_repair(
    tmp_path: Path,
    duplicate_target: str,
) -> None:
    """Do not profile-repair a last-key-wins JSON object.

    Args:
        tmp_path: Temporary directory receiving the ambiguous JSONL file.
        duplicate_target: Outer or nested object whose key is repeated.
    """
    payloads = _environment_capture_payloads()
    action = json.dumps(payloads[0], separators=(",", ":"))
    if duplicate_target == "span":
        exact_span_id = cast(str, payloads[0]["spanId"])
        action = action.replace(
            f'"spanId":"{exact_span_id}"',
            f'"spanId":"bad","spanId":"{exact_span_id}"',
            1,
        )
    else:
        action = action.replace(
            '"status":{"code":"STATUS_CODE_OK"}',
            '"status":{"code":"STATUS_CODE_ERROR","code":"STATUS_CODE_OK"}',
            1,
        )
    result_line = json.dumps(payloads[1], separators=(",", ":"))
    path = tmp_path / f"duplicate-{duplicate_target}.otel.jsonl"
    path.write_text(f"{action}\n{result_line}\n", encoding="utf-8")

    result = load_otlp_file(path)

    assert result.traces == ()
    assert any(
        "span ID must be a non-zero lowercase 16-hex W3C ID" in issue.message
        for issue in result.issues
    )


def test_otlp_sorts_unordered_export_before_selecting_the_initial_task() -> None:
    payload = _payload_with_late_request()
    _payload_spans(payload).reverse()

    result = normalize_otlp_payload(payload, source=_source())

    assert result.issues == ()
    assert result.traces[0].task == "Cancel reservation R-17"


def test_otlp_ignores_late_request_context() -> None:
    result = normalize_otlp_payload(_payload_with_late_request(), source=_source())

    assert result.issues == ()
    assert result.traces[0].initial_context == {"tier": "gold"}


def test_otlp_ignores_late_tool_definitions() -> None:
    result = normalize_otlp_payload(_payload_with_late_request(), source=_source())

    assert result.issues == ()
    assert tuple(tool.name for tool in result.traces[0].tools) == ("cancel_reservation",)


def test_otlp_routing_tags_ignore_late_request_tags() -> None:
    result = normalize_otlp_payload(_payload_with_late_request(), source=_source())

    assert result.issues == ()
    assert routing_descriptor(result.traces[0]).tags == ("domain:travel",)


def test_otlp_ignores_a_non_text_conversation_id() -> None:
    """A non-text conversation id is skipped and the trace still normalizes without one."""
    payload = _payload()
    _replace_attribute(payload, span_index=0, key="wmo.conversation.id", value=42)

    result = normalize_otlp_payload(payload, source=_source())

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].conversation_id is None


def test_otlp_blank_conversation_id_falls_back_to_the_genai_key() -> None:
    """A blank wmo.conversation.id falls through to a declared gen_ai.conversation.id."""
    payload = _payload()
    _replace_attribute(payload, span_index=0, key="wmo.conversation.id", value="  ")
    attributes = cast(list[dict[str, object]], _span(payload, 0)["attributes"])
    attributes.append(_attribute("gen_ai.conversation.id", "conversation-genai"))

    result = normalize_otlp_payload(payload, source=_source())

    assert result.issues == ()
    assert result.traces[0].conversation_id == "conversation-genai"

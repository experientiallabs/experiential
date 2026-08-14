"""Tests for the owned environment-capture OTLP profile."""

from __future__ import annotations

import copy
import json
from typing import cast

from wmo.common.core.artifacts import JsonObject
from wmo.simulation.ingest.environment_capture import canonicalize_environment_capture_payloads

_TRACE_ID = "1" * 32


def _attribute(key: str, value: str) -> JsonObject:
    """Encode one string attribute for a profile fixture."""
    return {"key": key, "value": {"stringValue": value}}


def _profile_payloads() -> list[JsonObject]:
    """Build one exact two-step environment-capture trace fixture.

    Returns:
        Alternating terminal action and result records in source order.
    """
    records: list[JsonObject] = []
    for ordinal, command, output in (
        (0, "printf first", "first"),
        (1, "printf second", "second"),
    ):
        action_attributes: list[JsonObject] = [
            _attribute("gen_ai.operation.name", "chat"),
            _attribute("gen_ai.request.model", "terminal-agent"),
            _attribute("gen_ai.tool.name", "bash"),
            _attribute("gen_ai.tool.call.arguments", json.dumps({"command": command})),
        ]
        if ordinal == 0:
            action_attributes.extend(
                (
                    _attribute("gen_ai.prompt", "Run two commands"),
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
                )
            )
        prefix = f"{_TRACE_ID[:12]}{ordinal:04x}"
        action: JsonObject = {
            "traceId": _TRACE_ID,
            "spanId": f"{prefix}a",
            "parentSpanId": "",
            "name": "chat terminal",
            "startTimeUnixNano": ordinal * 10,
            "endTimeUnixNano": ordinal * 10 + 1,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": action_attributes,
        }
        result: JsonObject = {
            "traceId": _TRACE_ID,
            "spanId": f"{prefix}b",
            "parentSpanId": "",
            "name": "execute_tool terminal",
            "startTimeUnixNano": ordinal * 10 + 2,
            "endTimeUnixNano": ordinal * 10 + 3,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": [
                _attribute("gen_ai.operation.name", "execute_tool"),
                _attribute("gen_ai.tool.name", "bash"),
                _attribute("gen_ai.tool.message", output),
            ],
        }
        records.extend((action, result))
    return records


def _attributes(span: JsonObject) -> dict[str, str]:
    """Decode a canonical fixture span's string attributes.

    Args:
        span: Canonical direct span emitted by the profile converter.

    Returns:
        Attribute values keyed by their semantic-convention names.
    """
    raw = cast(list[JsonObject], span["attributes"])
    return {
        cast(str, item["key"]): cast(str, cast(JsonObject, item["value"])["stringValue"])
        for item in raw
    }


def test_exact_profile_canonicalizes_without_fabricating_model_identity() -> None:
    """Canonicalize owned actions without fabricating model identity.

    The conversion repairs wire identities and pairing while retaining provider-free
    invoke-agent semantics and deterministic episode lineage.
    """
    canonical = canonicalize_environment_capture_payloads(_profile_payloads())

    assert canonical is not None
    assert len(canonical) == 4
    action = cast(JsonObject, canonical[0])
    result = cast(JsonObject, canonical[1])
    assert action["spanId"] == "111111111110000a"
    assert result["spanId"] == "111111111110000b"
    assert "parentSpanId" not in action
    assert result["parentSpanId"] == action["spanId"]
    assert action["startTimeUnixNano"] == 1_000_000_000
    assert action["endTimeUnixNano"] == 1_000_001_000
    assert action["status"] == {"code": 1}
    assert result["status"] == {"code": 1}
    action_attributes = _attributes(action)
    result_attributes = _attributes(result)
    assert action_attributes["gen_ai.operation.name"] == "invoke_agent"
    assert "gen_ai.request.model" not in action_attributes
    assert "gen_ai.provider.name" not in action_attributes
    assert action_attributes["wmh.trace.metadata"]
    assert action_attributes["wmo.conversation.id"] == _TRACE_ID
    assert action_attributes["gen_ai.tool.call.id"] == result_attributes["gen_ai.tool.call.id"]


def test_empty_tool_output_remains_exact_profile_evidence() -> None:
    """Retain an empty environment observation as exact capture evidence.

    Empty output is valid for terminal commands and must not make an otherwise exact
    action-result pair ineligible for canonicalization.
    """
    payloads = _profile_payloads()
    result_attributes = cast(list[JsonObject], payloads[1]["attributes"])
    message = next(item for item in result_attributes if item["key"] == "gen_ai.tool.message")
    cast(JsonObject, message["value"])["stringValue"] = ""

    canonical = canonicalize_environment_capture_payloads(payloads)

    assert canonical is not None
    assert _attributes(cast(JsonObject, canonical[1]))["gen_ai.tool.message"] == ""


def test_near_matches_do_not_enter_the_owned_profile() -> None:
    """Reject every tested identity, structure, metadata, and pairing drift.

    The compatibility boundary is the exact owned terminal grammar, so each mutation
    must return control to the strict generic OTLP normalizer without partial repair.
    """
    wrong_id = _profile_payloads()
    wrong_id[0]["spanId"] = "2" * 17
    assert canonicalize_environment_capture_payloads(wrong_id) is None

    wrong_name = _profile_payloads()
    wrong_name[0]["name"] = "chat other"
    assert canonicalize_environment_capture_payloads(wrong_name) is None

    non_root = _profile_payloads()
    non_root[1]["parentSpanId"] = cast(str, non_root[0]["spanId"])
    assert canonicalize_environment_capture_payloads(non_root) is None

    wrong_time = _profile_payloads()
    wrong_time[0]["startTimeUnixNano"] = 1
    assert canonicalize_environment_capture_payloads(wrong_time) is None

    boolean_time = _profile_payloads()
    boolean_time[0]["startTimeUnixNano"] = False
    assert canonicalize_environment_capture_payloads(boolean_time) is None

    provider = _profile_payloads()
    cast(list[JsonObject], provider[0]["attributes"]).append(
        _attribute("gen_ai.provider.name", "openai")
    )
    assert canonicalize_environment_capture_payloads(provider) is None

    wrong_model = _profile_payloads()
    model_attributes = cast(list[JsonObject], wrong_model[0]["attributes"])
    model = next(item for item in model_attributes if item["key"] == "gen_ai.request.model")
    cast(JsonObject, model["value"])["stringValue"] = "gpt-5.4"
    assert canonicalize_environment_capture_payloads(wrong_model) is None

    non_bash = _profile_payloads()
    action_attributes = cast(list[JsonObject], non_bash[0]["attributes"])
    action_tool = next(item for item in action_attributes if item["key"] == "gen_ai.tool.name")
    cast(JsonObject, action_tool["value"])["stringValue"] = "python"
    result_attributes = cast(list[JsonObject], non_bash[1]["attributes"])
    result_tool = next(item for item in result_attributes if item["key"] == "gen_ai.tool.name")
    cast(JsonObject, result_tool["value"])["stringValue"] = "python"
    assert canonicalize_environment_capture_payloads(non_bash) is None

    extra_argument = _profile_payloads()
    action_attributes = cast(list[JsonObject], extra_argument[0]["attributes"])
    arguments = next(
        item for item in action_attributes if item["key"] == "gen_ai.tool.call.arguments"
    )
    cast(JsonObject, arguments["value"])["stringValue"] = json.dumps(
        {"command": "printf ready", "timeout": 1}
    )
    assert canonicalize_environment_capture_payloads(extra_argument) is None

    duplicate_command = _profile_payloads()
    action_attributes = cast(list[JsonObject], duplicate_command[0]["attributes"])
    arguments = next(
        item for item in action_attributes if item["key"] == "gen_ai.tool.call.arguments"
    )
    cast(JsonObject, arguments["value"])["stringValue"] = (
        '{"command":"printf first","command":"printf second"}'
    )
    assert canonicalize_environment_capture_payloads(duplicate_command) is None

    empty_command = _profile_payloads()
    action_attributes = cast(list[JsonObject], empty_command[0]["attributes"])
    arguments = next(
        item for item in action_attributes if item["key"] == "gen_ai.tool.call.arguments"
    )
    cast(JsonObject, arguments["value"])["stringValue"] = json.dumps({"command": ""})
    assert canonicalize_environment_capture_payloads(empty_command) is None

    capability_digest = _profile_payloads()
    cast(list[JsonObject], capability_digest[0]["attributes"]).append(
        _attribute("wmo.model.capabilities_sha256", "c" * 64)
    )
    assert canonicalize_environment_capture_payloads(capability_digest) is None

    connection_digest = _profile_payloads()
    cast(list[JsonObject], connection_digest[0]["attributes"]).append(
        _attribute("wmo.model.connection_sha256", "d" * 64)
    )
    assert canonicalize_environment_capture_payloads(connection_digest) is None

    wrong_tool = _profile_payloads()
    result_attributes = cast(list[JsonObject], wrong_tool[1]["attributes"])
    tool = next(item for item in result_attributes if item["key"] == "gen_ai.tool.name")
    cast(JsonObject, tool["value"])["stringValue"] = "python"
    assert canonicalize_environment_capture_payloads(wrong_tool) is None

    no_metadata = _profile_payloads()
    first_attributes = cast(list[JsonObject], no_metadata[0]["attributes"])
    no_metadata[0]["attributes"] = [
        item for item in first_attributes if item["key"] != "wmh.trace.metadata"
    ]
    assert canonicalize_environment_capture_payloads(no_metadata) is None

    wrong_benchmark = _profile_payloads()
    attributes = cast(list[JsonObject], wrong_benchmark[0]["attributes"])
    metadata = next(item for item in attributes if item["key"] == "wmh.trace.metadata")
    cast(JsonObject, metadata["value"])["stringValue"] = json.dumps(
        {
            "benchmark": "other",
            "returncode": 0,
            "task_category": "Filesystem + text processing",
        }
    )
    assert canonicalize_environment_capture_payloads(wrong_benchmark) is None

    duplicate_benchmark = _profile_payloads()
    attributes = cast(list[JsonObject], duplicate_benchmark[0]["attributes"])
    metadata = next(item for item in attributes if item["key"] == "wmh.trace.metadata")
    cast(JsonObject, metadata["value"])["stringValue"] = (
        '{"benchmark":"other","benchmark":"terminal-tasks",'
        '"returncode":0,"task_category":"Filesystem + text processing"}'
    )
    assert canonicalize_environment_capture_payloads(duplicate_benchmark) is None

    boolean_returncode = _profile_payloads()
    attributes = cast(list[JsonObject], boolean_returncode[0]["attributes"])
    metadata = next(item for item in attributes if item["key"] == "wmh.trace.metadata")
    cast(JsonObject, metadata["value"])["stringValue"] = json.dumps(
        {
            "benchmark": "terminal-tasks",
            "returncode": False,
            "task_category": "Filesystem + text processing",
        }
    )
    assert canonicalize_environment_capture_payloads(boolean_returncode) is None

    empty_category = _profile_payloads()
    attributes = cast(list[JsonObject], empty_category[0]["attributes"])
    metadata = next(item for item in attributes if item["key"] == "wmh.trace.metadata")
    cast(JsonObject, metadata["value"])["stringValue"] = json.dumps(
        {"benchmark": "terminal-tasks", "returncode": 0, "task_category": ""}
    )
    assert canonicalize_environment_capture_payloads(empty_category) is None

    extra_metadata = _profile_payloads()
    attributes = cast(list[JsonObject], extra_metadata[0]["attributes"])
    metadata = next(item for item in attributes if item["key"] == "wmh.trace.metadata")
    cast(JsonObject, metadata["value"])["stringValue"] = json.dumps(
        {
            "benchmark": "terminal-tasks",
            "extra": True,
            "returncode": 0,
            "task_category": "Filesystem + text processing",
        }
    )
    assert canonicalize_environment_capture_payloads(extra_metadata) is None

    reordered_attributes = _profile_payloads()
    attributes = cast(list[JsonObject], reordered_attributes[0]["attributes"])
    attributes[0], attributes[1] = attributes[1], attributes[0]
    assert canonicalize_environment_capture_payloads(reordered_attributes) is None


def test_repeated_noncontiguous_trace_block_fails_closed() -> None:
    """Reject a trace identity repeated around another source trace block.

    The producer emits each trace contiguously, so accepting a spliced block would make
    the compatibility detector broader than the owned source grammar.
    """
    first = _profile_payloads()
    second = copy.deepcopy(first)
    second_trace_id = "2" * 32
    for ordinal, span in enumerate(second):
        pair_ordinal = ordinal // 2
        suffix = "a" if ordinal % 2 == 0 else "b"
        span["traceId"] = second_trace_id
        span["spanId"] = f"{second_trace_id[:12]}{pair_ordinal:04x}{suffix}"
    interleaved = first[:2] + second + first[2:]

    assert canonicalize_environment_capture_payloads(interleaved) is None

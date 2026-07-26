"""Tests for the Weights & Biases Weave adapter against realistic Weave Call exports.

Weave exports Call dicts with ``op_name``, ``trace_id``, ``id``, ``inputs``, ``output``, and
``started_at``/``ended_at`` timestamps. These fixtures exercise the adapter's field mapping and the
op-name heuristic classification. No network; file fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest.adapter import VendorPull, get_adapter
from wmh.ingest.weave import WeaveAdapter, _extract_op_name, _is_llm_call

# ---------------------------------------------------------------------------
# Fixtures — realistic Weave Call objects
# ---------------------------------------------------------------------------

# A minimal two-call trace: an LLM call issuing a tool call, then the tool execution.
_WEAVE_LLM_CALL = {
    "id": "call-aaa-111",
    "trace_id": "trace-abc-001",
    "parent_id": None,
    "op_name": "weave:///myteam/myproject/op/chat_completion:v1abc",
    "started_at": "2024-06-01T10:00:00.000000Z",
    "ended_at": "2024-06-01T10:00:01.000000Z",
    "inputs": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Look up user u1"},
        ]
    },
    "output": {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_user",
                                "arguments": '{"id": "u1"}',
                            }
                        }
                    ],
                }
            }
        ]
    },
    "exception": None,
    "summary": {"latency_ms": 1000},
}

_WEAVE_TOOL_CALL = {
    "id": "call-aaa-222",
    "trace_id": "trace-abc-001",
    "parent_id": "call-aaa-111",
    "op_name": "weave:///myteam/myproject/op/get_user:v2def",
    "started_at": "2024-06-01T10:00:01.500000Z",
    "ended_at": "2024-06-01T10:00:02.000000Z",
    "inputs": {"id": "u1"},
    "output": "user u1: Ada Lovelace",
    "exception": None,
    "summary": {},
}

_WEAVE_CALLS = [_WEAVE_LLM_CALL, _WEAVE_TOOL_CALL]


# ---------------------------------------------------------------------------
# Unit tests — op name extraction
# ---------------------------------------------------------------------------


def test_extract_op_name_from_uri() -> None:
    uri = "weave:///myteam/myproject/op/chat_completion:v1abc"
    assert _extract_op_name(uri) == "chat_completion"


def test_extract_op_name_plain() -> None:
    assert _extract_op_name("get_user") == "get_user"


def test_extract_op_name_empty() -> None:
    assert _extract_op_name("") == ""
    assert _extract_op_name(None) == ""


def test_is_llm_call_markers() -> None:
    assert _is_llm_call("chat_completion") is True
    assert _is_llm_call("openai_generate") is True
    assert _is_llm_call("anthropic_create_message") is True
    assert _is_llm_call("get_user") is False
    assert _is_llm_call("run_sql_query") is False


# ---------------------------------------------------------------------------
# Integration tests — from_file round-trip
# ---------------------------------------------------------------------------


def test_from_file_maps_weave_calls(tmp_path: Path) -> None:
    path = tmp_path / "weave_calls.json"
    path.write_text(json.dumps(_WEAVE_CALLS), encoding="utf-8")

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].trace_id == "trace-abc-001"
    assert traces[0].source.startswith("weave:")
    assert len(traces[0].steps) == 1

    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_user"
    assert step.action.arguments == {"id": "u1"}
    assert step.observation.content == "user u1: Ada Lovelace"
    assert step.observation.is_error is False


def test_from_file_handles_jsonl(tmp_path: Path) -> None:
    """JSONL format: one Call per line."""
    path = tmp_path / "weave_calls.jsonl"
    lines = [json.dumps(call) for call in _WEAVE_CALLS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    assert len(traces[0].steps) == 1
    assert traces[0].steps[0].action.name == "get_user"


def test_from_file_handles_error_call(tmp_path: Path) -> None:
    """A tool call that threw an exception should be flagged as an error."""
    llm_call = {
        **_WEAVE_LLM_CALL,
        "id": "call-err-111",
        "trace_id": "trace-err-001",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "get_user", "arguments": '{"id": "u2"}'}}
                        ],
                    }
                }
            ]
        },
    }
    tool_call = {
        "id": "call-err-222",
        "trace_id": "trace-err-001",
        "parent_id": "call-err-111",
        "op_name": "get_user",
        "started_at": "2024-06-01T10:00:02.000000Z",
        "ended_at": "2024-06-01T10:00:02.500000Z",
        "inputs": {"id": "u2"},
        "output": None,
        "exception": "UserNotFoundError: user u2 does not exist",
    }
    path = tmp_path / "weave_error.json"
    path.write_text(json.dumps([llm_call, tool_call]), encoding="utf-8")

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.observation.is_error is True
    assert "UserNotFoundError" in step.observation.content


def test_from_file_plain_llm_completion(tmp_path: Path) -> None:
    """An LLM call with no tool calls produces a message-type action."""
    call = {
        "id": "call-plain-111",
        "trace_id": "trace-plain-001",
        "parent_id": None,
        "op_name": "chat_completion",
        "started_at": "2024-06-01T11:00:00.000000Z",
        "ended_at": "2024-06-01T11:00:01.000000Z",
        "inputs": {"messages": [{"role": "user", "content": "Say hello"}]},
        "output": "Hello! How can I help you today?",
        "exception": None,
    }
    path = tmp_path / "weave_plain.json"
    path.write_text(json.dumps([call]), encoding="utf-8")

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    # A standalone LLM call with no following tool becomes a message step.
    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.MESSAGE


def test_from_file_wrapper_shape(tmp_path: Path) -> None:
    """The calls/stream_query response can wrap calls under a 'calls' key."""
    wrapped = {"calls": _WEAVE_CALLS}
    path = tmp_path / "weave_wrapped.json"
    path.write_text(json.dumps(wrapped), encoding="utf-8")

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "get_user"


def test_registered_under_weave() -> None:
    assert get_adapter("weave").name == "weave"


def test_vendor_pull_left_as_friendly_default_without_key() -> None:
    try:
        WeaveAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "API key" in str(exc) or "weave" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_parallel_tool_calls(tmp_path: Path) -> None:
    """An LLM call issuing multiple tool calls in parallel should produce
    one Step per tool call, each with correctly matched action/observation."""
    llm_call = {
        "id": "call-par-111",
        "trace_id": "trace-par-001",
        "parent_id": None,
        "op_name": "chat_completion",
        "started_at": "2024-07-01T10:00:00.000000Z",
        "ended_at": "2024-07-01T10:00:01.000000Z",
        "inputs": {
            "messages": [
                {"role": "user", "content": "Get user u1 and u2"},
            ]
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_user",
                                    "arguments": '{"id": "u1"}',
                                }
                            },
                            {
                                "function": {
                                    "name": "get_user",
                                    "arguments": '{"id": "u2"}',
                                }
                            },
                        ],
                    }
                }
            ]
        },
        "exception": None,
    }
    tool_call_1 = {
        "id": "call-par-222",
        "trace_id": "trace-par-001",
        "parent_id": "call-par-111",
        "op_name": "get_user",
        "started_at": "2024-07-01T10:00:01.100000Z",
        "ended_at": "2024-07-01T10:00:01.500000Z",
        "inputs": {"id": "u1"},
        "output": "user u1: Ada Lovelace",
        "exception": None,
    }
    tool_call_2 = {
        "id": "call-par-333",
        "trace_id": "trace-par-001",
        "parent_id": "call-par-111",
        "op_name": "get_user",
        "started_at": "2024-07-01T10:00:01.200000Z",
        "ended_at": "2024-07-01T10:00:01.600000Z",
        "inputs": {"id": "u2"},
        "output": "user u2: Grace Hopper",
        "exception": None,
    }
    path = tmp_path / "weave_parallel.json"
    path.write_text(
        json.dumps([llm_call, tool_call_1, tool_call_2]),
        encoding="utf-8",
    )

    traces = WeaveAdapter().from_file(str(path))

    assert len(traces) == 1
    # Two parallel tool calls: the LLM call emits no action spans
    # (avoiding mismatched pairing); each child tool Call produces
    # a complete Step with correctly matched action/observation.
    assert len(traces[0].steps) == 2

    step_0 = traces[0].steps[0]
    assert step_0.action.kind == ActionKind.TOOL_CALL
    assert step_0.action.name == "get_user"

    step_1 = traces[0].steps[1]
    assert step_1.action.kind == ActionKind.TOOL_CALL
    assert step_1.action.name == "get_user"

    # Both observations should be present with correctly matched
    # arguments (order may vary based on timestamp sorting).
    contents = {
        step_0.observation.content,
        step_1.observation.content,
    }
    assert "user u1: Ada Lovelace" in contents
    assert "user u2: Grace Hopper" in contents

    # Each step's arguments must match its observation (no cross-contamination).
    for step in traces[0].steps:
        if step.observation.content == "user u1: Ada Lovelace":
            assert step.action.arguments == {"id": "u1"}
        elif step.observation.content == "user u2: Grace Hopper":
            assert step.action.arguments == {"id": "u2"}

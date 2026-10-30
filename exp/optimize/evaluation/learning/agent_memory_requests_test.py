"""Trace rendering, batching and the strict decoders of agent memory requests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.traces import Trace
from exp.common.traces.ingest.sources import load_trace_source
from exp.optimize.evaluation.learning.agent_memory_requests import (
    RenderedTrace,
    batches,
    decode_extraction,
    decode_merge,
    decode_screening,
    extract_request,
    merge_request,
    render_trace,
    screen_request,
)


def conversation(
    trace_id: str, status: str, messages: list[JsonObject], **extra: JsonObject | str
) -> JsonObject:
    """One chat JSON conversation with a recorded outcome."""
    outcome: JsonObject = {"exp.outcome.status": status}
    if status == "failure":
        outcome |= {
            "exp.outcome.failure.code": "validation",
            "exp.outcome.failure.message": "verifier failed",
        }
    return {"trace_id": trace_id, **outcome, "messages": messages, **extra}


def load(tmp_path: Path, *conversations: JsonObject) -> tuple[Trace, ...]:
    """Normalize conversations through the chat JSON source."""
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in conversations) + "\n")
    result = load_trace_source("chat-json", path)
    assert not result.issues
    return result.traces


TOOL_TURN: list[JsonObject] = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Write the summary to /out/summary.md"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd": "ls /out"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "permission denied"},
    {"role": "user", "content": "Please keep it under 100 words."},
    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
]


def test_rendering_keeps_every_user_turn_tool_call_and_result_with_the_outcome(
    tmp_path: Path,
) -> None:
    """Later user corrections and tool errors are what memory learns from; none is lost."""
    (trace,) = load(tmp_path, conversation("a", "failure", TOOL_TURN))
    rendered = render_trace(trace, "T1", maximum_message_bytes=500, maximum_trace_bytes=5_000)
    assert rendered.outcome == "FAIL"
    assert rendered.cut_bytes == 0
    assert rendered.text.splitlines() == [
        "[T1] outcome: FAIL",
        "system: You are a coding agent.",
        "user: Write the summary to /out/summary.md",
        'assistant: call bash({"cmd": "ls /out"})',
        "tool bash: permission denied",
        "user: Please keep it under 100 words.",
        "assistant: Done.",
    ]


def test_rendering_states_every_cut_and_counts_what_it_did_not_show(tmp_path: Path) -> None:
    """A long message keeps its ends; a long trace keeps its start and end around a gap."""
    long = "x" * 1_000
    messages: list[JsonObject] = [{"role": "user", "content": "start"}]
    for index in range(40):
        messages.append({"role": "assistant", "content": f"step {index} {long}"})
        messages.append({"role": "user", "content": f"next {index}"})
    messages.append({"role": "assistant", "content": "finished"})
    (trace,) = load(tmp_path, conversation("a", "success", messages))
    rendered = render_trace(trace, "T1", maximum_message_bytes=300, maximum_trace_bytes=4_000)
    lines = rendered.text.splitlines()
    assert rendered.outcome == "PASS"
    assert any("[cut:" in line and "bytes not shown]" in line for line in lines)
    gaps = [line for line in lines if line.startswith("[gap:")]
    assert len(gaps) == 1 and gaps[0].endswith("messages not shown]")
    assert lines[-2:] == ["user: next 39", "assistant: finished"]
    assert len(rendered.text.encode()) < 4_000 + 400
    assert rendered.cut_bytes > 40 * 700


def test_a_trace_without_an_outcome_is_shown_as_unknown(tmp_path: Path) -> None:
    """Missing evidence is never shown as a pass or a failure."""
    path = tmp_path / "traces.jsonl"
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    path.write_text(json.dumps({"messages": messages}) + "\n")
    (trace,) = load_trace_source("chat-json", path).traces
    rendered = render_trace(trace, "T9", maximum_message_bytes=200, maximum_trace_bytes=2_000)
    assert rendered.outcome == "unknown"


def rendered(label: str, size: int) -> RenderedTrace:
    """A rendering of exactly ``size`` bytes."""
    return RenderedTrace(label=label, trace_id=label, outcome="PASS", text="x" * size, cut_bytes=0)


def test_batches_pack_in_order_and_give_an_oversized_trace_its_own_batch() -> None:
    """Every trace lands in exactly one batch, in dataset order."""
    traces = [rendered("a", 40), rendered("b", 40), rendered("c", 150), rendered("d", 10)]
    assert batches(traces, 100) == ((0, 1), (2,), (3,))
    assert batches([], 100) == ()


def test_extraction_answers_may_cite_only_the_traces_their_request_showed() -> None:
    """An invented citation refuses the whole answer instead of being repaired."""
    body = extract_request([rendered("T1", 10), rendered("T2", 10)], maximum_lessons=5)
    schema = body["response_schema"]
    assert isinstance(schema, dict) and schema["required"] == ["lessons"]
    lesson: JsonObject = {
        "kind": "pitfall",
        "statement": "Do not write outside /out.",
        "applies_when": "A task names an output folder.",
        "evidence": "T2 wrote elsewhere and failed.",
        "traces": ["T2"],
    }
    (decoded,) = decode_extraction({"lessons": [lesson]}, ["T1", "T2"])
    assert decoded.traces == ("T2",)
    with pytest.raises(ValueError, match="did not show"):
        decode_extraction({"lessons": [{**lesson, "traces": ["T7"]}]}, ["T1", "T2"])
    with pytest.raises(ValueError, match="exactly one lessons array"):
        decode_extraction({"lessons": [], "notes": "extra"}, ["T1"])
    with pytest.raises(ValidationError):
        decode_extraction({"lessons": [{**lesson, "kind": "guess"}]}, ["T1", "T2"])


def test_merge_answers_must_place_every_candidate_exactly_once() -> None:
    """Omitted, repeated and invented indexes all refuse the answer."""
    candidates: list[JsonObject] = [
        {"statement": f"s{index}", "applies_when": "w", "evidence": "e"} for index in range(3)
    ]
    body = merge_request("procedure", candidates)
    assert "s2" in str(body["messages"])
    group = {"statement": "s", "applies_when": "w"}
    merged = decode_merge({"lessons": [{**group, "sources": [0, 2]}, {**group, "sources": [1]}]}, 3)
    assert [row.sources for row in merged] == [(0, 2), (1,)]
    for sources in ([[0, 1]], [[0, 1, 2], [2]], [[0, 1, 2, 3]]):
        with pytest.raises(ValueError, match="exactly one group"):
            decode_merge({"lessons": [{**group, "sources": row} for row in sources]}, 3)


def test_screening_answers_return_one_decision_per_lesson_in_request_order() -> None:
    """Rows may arrive in any order; a missing, repeated or extra lesson refuses the answer."""
    body = screen_request([{"statement": "a"}, {"statement": "b"}], [{"statement": "c"}])
    assert "accepted_lessons" in str(body["messages"])
    row = {"general": 0.9, "covered": 0.1, "supported": 0.8}
    decided = decode_screening(
        {"screenings": [{"lesson": 1, **row, "general": 0.2}, {"lesson": 0, **row}]}, 2
    )
    assert [decision.general for decision in decided] == [0.9, 0.2]
    for rows in ([{"lesson": 0, **row}], [{"lesson": 0, **row}, {"lesson": 0, **row}]):
        with pytest.raises(ValueError, match="exactly once"):
            decode_screening({"screenings": rows}, 2)
    with pytest.raises(ValidationError):
        decode_screening({"screenings": [{"lesson": 0, **row, "covered": 1.5}]}, 1)

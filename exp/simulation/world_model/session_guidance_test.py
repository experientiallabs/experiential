"""Session guidance: cited notes accumulate within a session, are capped, and replay exactly."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest
from exp.common.project import ArtifactStore, ProjectPaths
from exp.runtime.models.structured_completion import ReturnedOutputRejected
from exp.simulation.world_model.runtime import PreparedGroundedWorldModelCall
from exp.simulation.world_model.session_guidance import (
    GUIDANCE_USE,
    SessionGuidance,
    SessionGuidancePolicy,
)

OPERATIONS = ("t-0", "t-1", "t-2")


def prepared(index: int) -> PreparedGroundedWorldModelCall:
    """A framed grounded prediction with one retrieved original and private channels."""
    body = {
        "task": {"instruction": "do it", "tools": [{"name": "read"}, {"name": "other"}]},
        "candidate_response": {"tool_calls": [{"call_id": OPERATIONS[index], "name": "read"}]},
        "grounded_examples": [{"transition_id": "rag-1", "observation": {"reply": "x"}}],
        "grounding_contexts": {},
        "environment_state": {"secret": 1},
        "visible_conversation": [{"role": "tool", "content": "earlier reply"}],
        "instance_facts": {"cell": 5},
    }
    request = ModelRequest(
        messages=(
            ModelMessage(role="system", content="predict"),
            ModelMessage(role="user", content=json.dumps(body)),
        )
    )
    return PreparedGroundedWorldModelCall(
        request=request, matches=(), action=AssistantAction(content="x")
    )


class Guide:
    """Scripted guidance completions, recording each request."""

    def __init__(self, replies: list[JsonObject | Exception]) -> None:
        """Answer in order."""
        self.replies, self.calls = replies, []

    def __call__(self, body: JsonObject, logical_id: str) -> JsonObject:
        """Return or raise the next scripted reply."""
        self.calls.append((logical_id, body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def guidance(tmp_path: Path, guide: Guide, **limits: int) -> SessionGuidance:
    """One session's guidance over OPERATIONS."""
    return SessionGuidance(
        SessionGuidancePolicy(**limits),
        guide,
        ArtifactStore(ProjectPaths(tmp_path, "guide")),
        session_id="run-1",
        operations=OPERATIONS,
        code_revision="test",
        clock=lambda: datetime(2026, 10, 23, tzinfo=UTC),
        profile_sha256="0" * 64,
    )


def delivered(call: PreparedGroundedWorldModelCall) -> list[JsonObject]:
    """The notes a framed prediction carries."""
    body = json.loads(call.request.messages[1].content or "")
    assert call.request.messages[0].content == "predict" + GUIDANCE_USE
    return body["session_guidance"]


def note(text: str, *basis: str) -> JsonObject:
    """One proposed guidance note."""
    return {"guidance": [{"tool": "read", "note": text, "basis": list(basis)}]}


def test_notes_accumulate_cap_and_cite_only_delivered_originals(tmp_path: Path) -> None:
    """Valid cited notes accumulate, oldest leaves first, and private channels stay out."""
    guide = Guide([note("a", "rag-1"), note("b", "rag-1"), note("c", "rag-1")])
    session = guidance(tmp_path, guide, maximum_notes=2)
    first = delivered(session.resolve(prepared(0), "t-0"))
    assert first == [{"operation": "t-0", "tool": "read", "note": "a", "basis": ["rag-1"]}]
    delivered(session.resolve(prepared(1), "t-1"))
    third = delivered(session.resolve(prepared(2), "t-2"))
    assert [row["note"] for row in third] == ["b", "c"]
    for _, body in guide.calls:
        payload = json.loads(body["messages"][1]["content"])
        assert {"environment_state", "visible_conversation", "instance_facts"}.isdisjoint(payload)
        assert payload["tools"] == [{"name": "read"}]
    assert json.loads(guide.calls[1][1]["messages"][1]["content"])["session_guidance"] == first


def test_a_rejected_proposal_adds_nothing_and_a_saved_step_replays(tmp_path: Path) -> None:
    """Rejection keeps earlier notes; resolving again replays without a call."""
    guide = Guide([note("a", "rag-1"), ReturnedOutputRejected("bad")])
    session = guidance(tmp_path, guide)
    delivered(session.resolve(prepared(0), "t-0"))
    second = delivered(session.resolve(prepared(1), "t-1"))
    assert [row["note"] for row in second] == ["a"]
    record = session.notes("t-1")
    assert record is not None and record["status"] == "rejected"
    assert record["persisted_to_memory"] is False
    again = guidance(tmp_path, Guide([]))
    assert delivered(again.resolve(prepared(1), "t-1")) == second


def test_a_step_without_its_predecessor_and_a_foreign_operation_are_refused(
    tmp_path: Path,
) -> None:
    """Guidance never invents a session history it did not save."""
    session = guidance(tmp_path, Guide([note("a")]))
    with pytest.raises(ValueError, match="never saved"):
        session.resolve(prepared(1), "t-1")
    with pytest.raises(ValueError, match="outside its workload"):
        session.resolve(prepared(0), "elsewhere")


@pytest.mark.parametrize("basis", [[], ["invented"], [""], [" "], ["rag-1", "invented"]])
def test_unsupported_note_is_rejected_without_losing_valid_or_previous_notes(
    tmp_path: Path, basis: list[str]
) -> None:
    """Reject the entire unsupported note, including mixed bases, and retain valid notes."""
    proposal: JsonObject = {
        "guidance": [
            {"tool": "read", "note": "unsupported", "basis": basis},
            {"tool": "read", "note": "valid", "basis": ["rag-1"]},
        ]
    }
    guide = Guide([note("previous", "rag-1"), proposal])
    session = guidance(tmp_path, guide)
    delivered(session.resolve(prepared(0), "t-0"))
    result = delivered(session.resolve(prepared(1), "t-1"))
    assert [row["note"] for row in result] == ["previous", "valid"]
    record = session.notes("t-1")
    assert record is not None
    assert record["rejected_notes"] == [
        {
            "note": {"tool": "read", "note": "unsupported", "basis": basis},
            "reason": "basis must cite only delivered original IDs",
        }
    ]
    assert delivered(guidance(tmp_path, Guide([])).resolve(prepared(1), "t-1")) == result


@pytest.mark.parametrize("original", [{}, {"transition_id": ""}, {"transition_id": None}])
def test_missing_original_id_cannot_become_a_citable_string(
    tmp_path: Path, original: JsonObject
) -> None:
    """Malformed original IDs cannot be cited through empty or stringified null values."""
    call = prepared(0)
    body = json.loads(call.request.messages[1].content or "")
    body["grounded_examples"] = [original]
    call = PreparedGroundedWorldModelCall(
        request=call.request.model_copy(
            update={
                "messages": (
                    call.request.messages[0],
                    ModelMessage(role="user", content=json.dumps(body)),
                )
            }
        ),
        matches=call.matches,
        action=call.action,
    )
    session = guidance(tmp_path, Guide([note("unsupported", "None")]))
    assert delivered(session.resolve(call, "t-0")) == []
    record = session.notes("t-0")
    assert record is not None and record["originals"] == []
    rejected = record["rejected_notes"]
    assert isinstance(rejected, list) and len(rejected) == 1

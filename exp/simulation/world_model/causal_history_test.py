"""Investigations see their own trajectory's earlier steps, never later ones or other traces."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
from exp.common.project import ArtifactStore, ProjectPaths
from exp.runtime.sessions.errors import SessionNotFoundError
from exp.runtime.sessions.store import SessionBinding, SessionStore
from exp.simulation.engines.text.generic_delta import apply_delta
from exp.simulation.engines.transition import Completion
from exp.simulation.retrieval.selection import AllOriginals
from exp.simulation.world_model.causal_history import causal_history
from exp.simulation.world_model.resumption import Resumption
from exp.simulation.world_model.session_research import (
    INVESTIGATION_PROMPT,
    OnlineResearchPolicy,
    SessionResearch,
)
from exp.simulation.world_model.session_research_test import investigation_reply

BINDING = SessionBinding(
    deadline=datetime.now(UTC) + timedelta(hours=1),
    target_id="fixture@1",
    initialization_receipt_sha256="a" * 64,
    knowledge_sha256="b" * 64,
    corpus_sha256="c" * 64,
    representation_sha256="d" * 64,
    execution_policy_sha256="e" * 64,
)
PRIOR_SOURCE = ArtifactInput(artifact_id="trace-other", sha256="f" * 64)


def write_then_read(count: int = 3) -> list[JsonValue]:
    """A write, a readback of what it wrote, then a later step after the readback."""
    rows: list[JsonValue] = [
        {
            "operation_id": "step-0",
            "actions": [
                {
                    "call_id": "step-0",
                    "name": "write_file",
                    "arguments": {"path": "output.xlsx", "rows": ["INV-1001"]},
                }
            ],
            "replies": ["written"],
            "before": {},
            "after": {"files": {"output.xlsx": ["INV-1001"]}},
            "evidence": {"grounded_transition_ids": [], "secret": "other-trace-evidence"},
        },
        {
            "operation_id": "step-1",
            "actions": [
                {"call_id": "step-1", "name": "read_file", "arguments": {"path": "output.xlsx"}}
            ],
            "replies": ["['PQTW1']"],
            "before": {"files": {"output.xlsx": ["INV-1001"]}},
            "after": {"files": {"output.xlsx": ["INV-1001"]}},
        },
        {
            "operation_id": "step-2",
            "actions": [{"call_id": "step-2", "name": "delete_file", "arguments": {"path": "x"}}],
            "replies": ["FUTURE-PREDICTION"],
            "before": {"files": {"output.xlsx": ["INV-1001"]}},
            "after": {},
        },
    ]
    return rows[:count]


def recording(calls: list[JsonObject]) -> Completion:
    """An investigator that records each request and answers with a scoped note."""

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(body)
        return investigation_reply()

    return complete


def research(
    tmp_path: Path,
    transitions: list[JsonValue],
    complete: Completion,
    *,
    pinned: bool = True,
    database: str = "sessions.sqlite",
    resumption: Resumption | None = None,
) -> SessionResearch:
    """A research owner over ``transitions``; reopening the same database reuses the session."""
    store = SessionStore(tmp_path / database)
    try:
        store.read_session("session-a")
    except SessionNotFoundError:
        store.create_session("session-a", BINDING, {"transitions": transitions})
    selector = AllOriginals()
    return SessionResearch(
        session_id="session-a",
        policy=OnlineResearchPolicy(
            feedback_source="native",
            selection_policy_sha256=selector.identity,
            notes_reach_predictions=not pinned,
        ),
        store=store,
        artifacts=ArtifactStore(ProjectPaths(tmp_path, "research")),
        complete=complete,
        lock_path=tmp_path / (database + ".lock"),
        clock=lambda: datetime.now(UTC),
        code_revision="fixture",
        selection=selector,
        prior={
            "prior-note": {
                "id": "prior-note",
                "session_id": "trace-other",
                "observed_reply": "OTHER-TRACE-REPLY",
                "investigation": investigation_reply(),
            }
        },
        prior_identity={"version": 1},
        prior_sources={"prior-note": (PRIOR_SOURCE,)},
        resumption=resumption,
    )


def request(body: JsonObject) -> JsonObject:
    """The investigator's structured input."""
    messages = body["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    assert isinstance(messages[0], dict) and messages[0]["content"] == INVESTIGATION_PROMPT
    content = json.loads(str(messages[1]["content"]))
    assert isinstance(content, dict)
    return content


def investigated(calls: list[JsonObject], operation: str) -> JsonObject:
    """The request that investigated ``operation``."""
    for body in calls:
        content = request(body)
        feedback = content["feedback"]
        if isinstance(feedback, dict) and feedback["operation_id"] == operation:
            return content
    raise AssertionError(f"{operation} was not investigated")


def steps(content: JsonObject) -> list[JsonObject]:
    """The ordered causal-history steps of one request."""
    history = content["causal_history"]
    assert isinstance(history, dict) and isinstance(history["steps"], list)
    rows = history["steps"]
    assert all(isinstance(row, dict) for row in rows)
    return [row for row in rows if isinstance(row, dict)]


def test_deferred_readback_investigation_sees_the_preceding_write(tmp_path: Path) -> None:
    """The write that produced a read file, and its disclosed reply, reach the readback's input."""
    calls: list[JsonObject] = []
    owner = research(tmp_path, write_then_read(), recording(calls))
    settled = owner.observe_all(
        [("step-0", "written"), ("step-1", "['INV-1001']"), ("step-2", "deleted")], "native"
    )
    assert all(isinstance(row, dict) for row in settled)
    rows = steps(investigated(calls, "step-1"))
    assert [row["operation_id"] for row in rows] == ["step-0", "step-1"]
    write = rows[0]
    assert write["actions"] == [
        {"name": "write_file", "arguments": {"path": "output.xlsx", "rows": ["INV-1001"]}}
    ]
    assert write["recorded_reply"] == {
        "authority": "native-observation",
        "disclosed": True,
        "reply": "written",
    }
    prediction = write["simulator_prediction"]
    assert isinstance(prediction, dict)
    assert prediction["replies"] == ["written"]
    edits = [{"op": "set", "path": ["files"], "value": {"output.xlsx": ["INV-1001"]}}]
    assert prediction["state_edits"] == edits
    assert apply_delta({}, edits) == {"files": {"output.xlsx": ["INV-1001"]}}
    assert rows[1]["investigated"] is True and write["investigated"] is False


def test_history_stops_at_the_investigated_step(tmp_path: Path) -> None:
    """Later steps exist when investigations run, but no later action or reply is shown."""
    calls: list[JsonObject] = []
    owner = research(tmp_path, write_then_read(), recording(calls))
    owner.observe_all(
        [("step-0", "wrong"), ("step-1", "['INV-1001']"), ("step-2", "FUTURE-RECORDED")], "native"
    )
    for operation, expected in (("step-0", 1), ("step-1", 2), ("step-2", 3)):
        content = investigated(calls, operation)
        assert len(steps(content)) == expected
        if operation != "step-2":
            text = json.dumps(content["causal_history"])
            assert "FUTURE-RECORDED" not in text and "FUTURE-PREDICTION" not in text
            assert "delete_file" not in text


def test_history_labels_authority_and_admits_no_other_trace(tmp_path: Path) -> None:
    """Recorded replies and simulator predictions are labelled apart; other traces stay out."""
    calls: list[JsonObject] = []
    owner = research(tmp_path, write_then_read(), recording(calls))
    settled = owner.observe_all([("step-1", "['INV-1001']")], "native")
    content = investigated(calls, "step-1")
    history = content["causal_history"]
    assert isinstance(history, dict)
    authority = history["authority"]
    assert isinstance(authority, dict)
    assert set(authority) == {"native-observation", "simulator-prediction"}
    rows = steps(content)
    assert rows[0]["recorded_reply"] == {"authority": "native-observation", "disclosed": False}
    assert rows[1]["recorded_reply"] == {
        "authority": "native-observation",
        "disclosed": True,
        "reply": "['INV-1001']",
    }
    for row in rows:
        prediction = row["simulator_prediction"]
        assert isinstance(prediction, dict) and prediction["authority"] == "simulator-prediction"
    text = json.dumps(history)
    assert "OTHER-TRACE-REPLY" not in text and "other-trace-evidence" not in text
    assert "trace-other" not in text and "prior-note" not in text
    # Provenance is unchanged: only the prior note's own source, nothing new from the history.
    record = settled[0]
    assert isinstance(record, dict)
    assert record["depends_on"] == [PRIOR_SOURCE.model_dump(mode="json")]


def test_sequential_feedback_carries_earlier_disclosed_replies(tmp_path: Path) -> None:
    """The latest-step path reads earlier steps' disclosures from this session's receipts."""
    calls: list[JsonObject] = []
    first = research(
        tmp_path, write_then_read(1), recording(calls), pinned=False, database="one.sqlite"
    )
    assert first.observe("step-0", "written", "native")["reply_matched"] is True
    assert calls == []
    later = research(
        tmp_path, write_then_read(2), recording(calls), pinned=False, database="two.sqlite"
    )
    later.observe("step-1", "['INV-1001']", "native")
    rows = steps(investigated(calls, "step-1"))
    assert [row["operation_id"] for row in rows] == ["step-0", "step-1"]
    assert rows[0]["recorded_reply"] == {
        "authority": "native-observation",
        "disclosed": True,
        "reply": "written",
    }


def test_budget_drops_whole_oldest_steps_with_a_marker() -> None:
    """Trimming keeps a contiguous suffix of whole steps and says how many were omitted."""
    transitions: list[JsonValue] = [
        {
            "operation_id": f"step-{index}",
            "actions": [{"name": "append", "arguments": {"text": "x" * 400}}],
            "replies": ["y" * 400],
        }
        for index in range(6)
    ]
    whole = causal_history(transitions, "step-4", {})
    assert whole["omitted_earlier_steps"] == 0 and "omitted" not in whole
    full = whole["steps"]
    assert isinstance(full, list) and len(full) == 5
    size = len(json.dumps(full[-1], ensure_ascii=False))
    trimmed = causal_history(transitions, "step-4", {}, budget=3 * size + 10)
    assert trimmed["omitted_earlier_steps"] == 2
    assert trimmed["omitted"] == "2 earlier steps omitted"
    assert trimmed["steps"] == full[2:], "kept steps are byte-identical, never cut inside"
    alone = causal_history(transitions, "step-4", {}, budget=1)
    assert alone["steps"] == full[-1:], "the investigated step is kept whole even over budget"
    assert alone["omitted"] == "4 earlier steps omitted"


def test_history_refuses_an_operation_outside_the_trajectory() -> None:
    """An investigation must concern a committed step of this trajectory."""
    with pytest.raises(ValueError, match="committed"):
        causal_history(write_then_read(), "step-9", {})


def test_resumed_investigation_reuses_the_same_history(tmp_path: Path) -> None:
    """A resumed intent's request, history included, is identical, so no second unit is spent."""
    calls: list[JsonObject] = []

    def lost(body: JsonObject, identity: str) -> JsonObject:
        calls.append(body)
        raise KeyboardInterrupt("process killed with the investigation in flight")

    feedback: list[tuple[str, JsonValue]] = [("step-0", "written"), ("step-1", "['INV-1001']")]
    with pytest.raises(KeyboardInterrupt):
        research(tmp_path, write_then_read(), lost).observe_all(feedback, "native")
    resumed: list[JsonObject] = []
    retried: list[str] = []
    resumption = Resumption(
        window=datetime.now(UTC) + timedelta(hours=3),
        retried=lambda stage, operation, logical: retried.append(operation),
    )
    owner = research(tmp_path, write_then_read(), recording(resumed), resumption=resumption)
    settled = owner.observe_all(feedback, "native")
    assert isinstance(settled[1], dict) and settled[1]["investigation"] == investigation_reply()
    assert resumed == calls and len(calls) == 1
    assert retried == []

"""Online feedback reaches later requests without replacing candidate state or scores."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from filelock import Timeout

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, ProjectPaths
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_test import source_project, transition_reply
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.sessions.store import SessionBinding, SessionStore
from exp.simulation.engines.transition import Completion
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy
from exp.simulation.retrieval.selection import AllOriginals, SelectionQuery
from exp.simulation.world_model.session_research import OnlineResearchPolicy, SessionResearch


def deferred_research(tmp_path: Path, complete: Completion) -> SessionResearch:
    """Open a real durable research owner with two completed single-tool predictions."""
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.create_session(
        "session-a",
        SessionBinding(
            deadline=datetime.now(UTC) + timedelta(minutes=5),
            target_id="fixture@1",
            initialization_receipt_sha256="a" * 64,
            knowledge_sha256="b" * 64,
            corpus_sha256="c" * 64,
            representation_sha256="d" * 64,
            execution_policy_sha256="e" * 64,
        ),
        {
            "transitions": [
                {"operation_id": step, "actions": [{"name": "read"}], "replies": ["wrong"]}
                for step in ("step-0", "step-1")
            ]
        },
    )
    selector = AllOriginals()
    return SessionResearch(
        session_id="session-a",
        policy=OnlineResearchPolicy(
            feedback_source="native",
            selection_policy_sha256=selector.identity,
            maximum_updates=1,
            notes_reach_predictions=False,
        ),
        store=store,
        artifacts=ArtifactStore(ProjectPaths(tmp_path, "research")),
        complete=complete,
        lock_path=tmp_path / "session.lock",
        clock=lambda: datetime.now(UTC),
        code_revision="fixture",
        selection=selector,
    )


def investigation_reply() -> JsonObject:
    """Return a scoped interpretation without claiming a transferable rule."""
    return {
        "discrepancy": "read differs",
        "hypothesis": "state may be missing",
        "scope": "this read",
        "uncertainty": "hidden state is unknown",
        "supersedes": [],
        "transferable": False,
        "observable_cue": "",
    }


@pytest.mark.parametrize("failed_first", [False, True])
def test_deferred_budget_is_session_wide_even_after_failed_intent(
    tmp_path: Path, failed_first: bool
) -> None:
    """Splitting feedback into batches cannot buy more than the registered cap."""
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(identity)
        if failed_first:
            raise RuntimeError("provider outcome unknown")
        return investigation_reply()

    research = deferred_research(tmp_path, complete)
    first = research.observe_all([("step-0", "observed")], "native")
    assert isinstance(first[0], RuntimeError if failed_first else dict)
    second = research.observe_all([("step-1", "observed")], "native")
    assert len(calls) == 1
    assert isinstance(second[0], dict)
    assert second[0]["investigation"] is None
    assert second[0]["investigation_skipped"] == "session investigation budget exhausted"
    assert research.observe_all([("step-1", "observed")], "native") == second
    assert len(calls) == 1


def test_deferred_feedback_holds_session_lease_during_provider_call(tmp_path: Path) -> None:
    """Another owner must not submit feedback or dispatch while a batch is settling."""
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(identity)
        if len(calls) == 1:
            with pytest.raises(Timeout):
                research.observe_all([("step-1", "observed")], "native")
        return investigation_reply()

    research = deferred_research(tmp_path, complete)
    records = research.observe_all([("step-0", "observed")], "native")
    assert isinstance(records[0], dict)
    assert records[0]["investigation"] == investigation_reply()


def test_deferred_duplicate_feedback_is_refused_before_dispatch(tmp_path: Path) -> None:
    """One batch cannot race two writes or investigations for the same operation."""
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(identity)
        return investigation_reply()

    research = deferred_research(tmp_path, complete)
    with pytest.raises(ValueError, match="repeat"):
        research.observe_all([("step-0", "observed"), ("step-0", "observed")], "native")
    assert calls == []


def structured_reply(value: JsonObject) -> JsonHttpResponse:
    """Return a typed investigator or judge response through the actual model wire."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "model": "fixture-v1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "generated",
                                "type": "function",
                                "function": {
                                    "name": "submit_proposal",
                                    "arguments": json.dumps(value),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


@pytest.mark.parametrize("generate_hypotheses", [True, False])
def test_next_turn_and_restart_preserve_feedback_without_state_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generate_hypotheses: bool,
) -> None:
    """Exercise the public executor, durable storage and actual delivered second request."""
    project, spec = source_project(tmp_path)
    selection = AllOriginals()
    policy = OnlineResearchPolicy(
        feedback_source="disclosed-native-reply",
        generate_hypotheses=generate_hypotheses,
        selection_policy_sha256=selection.identity,
    )
    spec = spec.model_copy(update={"output_protocol": "json-tool", "online_research": policy})
    investigation: JsonObject = {
        "discrepancy": "Predicted wrong instead of reply-0",
        "hypothesis": "The current entry value may be incorrectly represented",
        "scope": "current entry after call-0",
        "uncertainty": "One read does not establish all entries or their update rules",
        "supersedes": [],
        "transferable": True,
        "observable_cue": "the same tool and arguments",
    }
    axes = {
        dimension.dimension_id: {
            "status": "assessed",
            "score": 2,
            "rationale": "The first prediction was wrong",
            "evidence_refs": ["step-0"],
            "critical_flags": [],
        }
        for dimension in spec.judge.definition.dimensions
    }
    wire = ScriptedJsonTransport(
        [
            transition_reply(
                {
                    "observations": [{"call_id": "call-0", "reply": "wrong", "encoding": "json"}],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["seen"], "value": 1}],
                    },
                }
            ),
            *([structured_reply(investigation)] if generate_hypotheses else []),
            transition_reply(
                {
                    "observations": [{"call_id": "call-1", "reply": "reply-1", "encoding": "json"}],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["seen"], "value": 2}],
                    },
                }
            ),
            structured_reply({"axes": axes}),
        ]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    result = execute_fidelity_evaluation(project, prepared, research_selection=selection)
    assert len(wire.requests) == (4 if generate_hypotheses else 3)
    first, second = payload(wire, 0), payload(wire, 2 if generate_hypotheses else 1)
    assert isinstance(first["research_history"], dict)
    assert isinstance(second["research_history"], dict)
    assert first["research_history"]["session"] == []
    assert "reply-0" not in json.dumps(first) and "reply-1" not in json.dumps(first)
    assert second["environment_state"] == {"seen": 1}
    notes = second["research_history"]["session"]
    assert isinstance(notes, list) and len(notes) == 1
    assert isinstance(notes[0], dict) and notes[0]["observed_reply"] == "reply-0"
    assert notes[0]["investigation"] == (investigation if generate_hypotheses else None)
    assert "reply-1" not in json.dumps(second)
    rows = result.mechanical["steps"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict) and isinstance(rows[1], dict)
    assert rows[0]["trajectory_reply_exact"] is False
    assert rows[1]["trajectory_reply_exact"] is True
    store = SessionStore(project.artifacts.project_directory / (prepared.id + "-sessions.sqlite"))
    current = store.read_session(prepared.id)
    calls: list[JsonObject] = []

    def no_dispatch(body: JsonObject, identity: str) -> JsonObject:
        """Fail if a restart attempts to repurchase a completed investigation."""
        calls.append(body)
        raise AssertionError(identity)

    reopened = SessionResearch(
        session_id=prepared.id,
        policy=policy,
        store=store,
        artifacts=project.artifacts,
        complete=no_dispatch,
        lock_path=tmp_path / "reopened.lock",
        clock=lambda: datetime.now(UTC),
        code_revision="fixture",
        selection=selection,
    )
    assert reopened.observe("call-0", "reply-0", policy.feedback_source)["id"] == notes[0]["id"]
    assert not calls
    with pytest.raises(ValueError, match="changed"):
        reopened.observe("call-0", "different", policy.feedback_source)
    with pytest.raises(ValueError, match="latest"):
        reopened.observe("future-action", "future-secret", policy.feedback_source)
    with pytest.raises(ValueError, match="source"):
        reopened.observe("call-1", "reply-1", "undeclared-source")
    assert store.read_session(prepared.id) == current
    assert execute_fidelity_evaluation(project, prepared) == result
    assert len(wire.requests) == (4 if generate_hypotheses else 3)


def test_jev_memory_tags_select_relevance_not_recency() -> None:
    """An older linked note wins over a newer irrelevant note with exact query receipts."""
    receipts: list[JsonObject] = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Return synthetic ordinal answers keyed to the exact presented records."""
        assert isinstance(request.state, dict)
        records = request.state["records"]
        assert isinstance(records, list)
        answers: JsonObject = {"dependency": {"type": "choice", "choice": "entity-state"}}
        for index, record in enumerate(records):
            assert isinstance(record, dict)
            value = 2 if record["id"] == "old-linked" else 0
            answers[f"field_{index}"] = {"type": "score", "score": value}
            answers[f"history_{index}"] = {"type": "score", "score": value}
        return {"answers": answers}

    selector = JEVCueSelector(
        policy=JEVPolicy(candidate_limit=2), top_k=1, decide=decide, record=receipts.append
    )
    query = SelectionQuery(
        action=ToolCall(call_id="next", name="read", arguments={"id": "A"}),
        state={},
        history=({"actions": [{"name": "update", "arguments": {"id": "A"}}]},),
    )
    assert selector.select(
        query, {"old-linked": {"entity": "A"}, "new-unrelated": {"entity": "B"}}
    ) == ("old-linked",)
    assert receipts[-1]["selected_ids"] == ["old-linked"]
    assert receipts[-1]["scores"]
    assert len(receipts) == 3


def payload(wire: ScriptedJsonTransport, index: int) -> JsonObject:
    """Read the actual user payload from a dispatched provider request."""
    messages = wire.requests[index].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = messages[1]["content"]
    assert isinstance(content, str)
    return json.loads(content)


def test_failed_investigation_retains_wrong_step_and_blocks_unqualified_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid hypothesis revisions cannot disappear, repurchase, or rewrite a past score."""
    project, spec = source_project(tmp_path)
    selection = AllOriginals()
    policy = OnlineResearchPolicy(
        feedback_source="disclosed-native-reply", selection_policy_sha256=selection.identity
    )
    spec = spec.model_copy(update={"output_protocol": "json-tool", "online_research": policy})
    wire = ScriptedJsonTransport(
        [
            transition_reply(
                {
                    "observations": [{"call_id": "call-0", "reply": "wrong", "encoding": "json"}],
                    "effect": {
                        "kind": "edits",
                        "edits": [{"op": "set", "path": ["seen"], "value": 1}],
                    },
                }
            ),
            structured_reply(
                {
                    "discrepancy": "Wrong entry",
                    "hypothesis": "Incorrect state belief",
                    "scope": "Current entry",
                    "uncertainty": "Only one read observed",
                    "supersedes": ["nonexistent-investigation"],
                    "transferable": True,
                    "observable_cue": "the same tool and arguments",
                }
            ),
            structured_reply({"axes": {}}),
        ]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    result = execute_fidelity_evaluation(project, prepared, research_selection=selection)
    assert result.planned_steps == 2
    assert result.mechanical["valid_steps"] == 1
    assert result.mechanical["blocked_steps"] == 1
    assert len(wire.requests) == 3
    store = SessionStore(project.artifacts.project_directory / (prepared.id + "-sessions.sqlite"))
    snapshot = store.read_session(prepared.id)
    assert snapshot.state["business_state"] == {"seen": 1}
    history = snapshot.state["history"]
    assert isinstance(history, list)
    typed_history = tuple(row for row in history if isinstance(row, dict))
    research = SessionResearch(
        session_id=prepared.id,
        policy=policy,
        store=store,
        artifacts=project.artifacts,
        complete=lambda *_: {},
        lock_path=tmp_path / "restart.lock",
        clock=lambda: datetime.now(UTC),
        code_revision="fixture",
        selection=selection,
    )
    with pytest.raises(ValueError, match="unfinished"):
        research.deliver(typed_history, spec.workload.actions[1], {"seen": 1})
    with pytest.raises(ValueError, match="unfinished"):
        research.observe("call-0", "reply-0", policy.feedback_source)
    assert store.read_session(prepared.id) == snapshot
    assert execute_fidelity_evaluation(project, prepared) == result
    assert len(wire.requests) == 3

"""Agentic state-tag authoring, step-level pre-turn states and strict decision tagging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, SourceIdentity, sha256_json
from exp.common.models.model import ModelRequest
from exp.common.project import ProjectStore
from exp.common.project.paths import ProjectPaths
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.mining.tag_cluster_names_test import SNAPSHOT, ScriptedNamer
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.retrieval.contracts import RAGAction, RAGObservation, RAGTransition
from exp.simulation.retrieval.state_tags import (
    STATE_TAG_BANK_TYPE,
    PreTurnState,
    StateQuestion,
    StateTagAuthor,
    StateTagAuthoringSpec,
    StateTagBank,
    author_state_tags,
    authored_state_tags,
    cached_state_decider,
    live_pre_turn_state,
    pre_turn_steps,
    state_question,
    state_tag_requests,
    tag_state,
    trace_order,
)

ERROR = state_question("errors", "The most recent result reports an error.")
SHEET = state_question("entities", "A spreadsheet has been opened.")
QUEUED = state_question("pending", "An operation was requested but is still queued.")
EMPTY = state_question("observed", "The most recent result was empty.")


def bank(*questions: StateQuestion) -> StateTagBank:
    """Return a fixed bank over the given questions (all four by default)."""
    chosen = questions or (ERROR, SHEET, QUEUED, EMPTY)
    return StateTagBank(
        family="sheets",
        model=SNAPSHOT.model_dump(mode="json"),
        spec=StateTagAuthoringSpec(),
        evidence_sha256="0" * 64,
        questions=tuple(chosen),
    )


def transition(
    trace: str,
    index: int,
    tool: str,
    observation: str,
    *,
    task: str = "Update the budget workbook",
    lineage: str = "",
) -> RAGTransition:
    """Return one recorded tool transition of ``trace``."""
    key = f"{task}|{tool}"
    return RAGTransition(
        transition_id=f"t-{trace}-{index}",
        trace_id=trace,
        lineage_id=lineage or f"lineage-{trace}",
        action_span_id=f"{trace}-a{index}",
        observation_span_id=f"{trace}-o{index}",
        task=task,
        action=RAGAction(kind="tool_call", tool_name=tool, tool_arguments={"step": index}),
        observation=RAGObservation(kind="tool_result", content=observation),
        key_text=key,
        key_sha256=hashlib.sha256(key.encode()).hexdigest(),
    )


def fact(state: JsonObject, question_id: str) -> bool:
    """Decide one fixture proposition from the rendered pre-turn state."""
    rendered = state["pre_turn_state"]
    assert isinstance(rendered, dict)
    recent = rendered.get("recent") or []
    assert isinstance(recent, list)
    last = recent[-1] if recent else None
    observed = "" if not isinstance(last, dict) else str(last.get("observation"))
    tools = [
        str(step["action"].get("tool_name"))
        for step in recent
        if isinstance(step, dict) and isinstance(step.get("action"), dict)
    ]
    environment = rendered.get("environment")
    if question_id == ERROR.question_id:
        return last is not None and "error" in observed.lower()
    if question_id == SHEET.question_id:
        return "open_sheet" in tools or (isinstance(environment, dict) and "sheets" in environment)
    if question_id == QUEUED.question_id:
        return "queued" in observed
    return last is not None and observed == ""


class ScriptedDecisions:
    """A decision model answering fixture propositions confidently, recording requests."""

    def __init__(self, uncertain: Sequence[str] = ()) -> None:
        """Start with no requests; ``uncertain`` questions answer 0.6 when true."""
        self.requests: list[DecisionRequest] = []
        self._uncertain = frozenset(uncertain)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer every proposition of one request."""
        self.requests.append(request)
        state = request.state
        assert isinstance(state, dict)
        answers: JsonObject = {}
        for question_id in request.questions:
            true = fact(state, question_id)
            value = (0.6 if question_id in self._uncertain else 0.95) if true else 0.05
            answers[question_id] = {"type": "noul", "noul": value}
        return {"answers": answers, "usage": {"cost": 0.0002}}


def _prompt(request: ModelRequest) -> str:
    """Return the user prompt of an authoring request."""
    content = request.messages[-1].content
    assert isinstance(content, str)
    return content


def author_reply(request: ModelRequest) -> str:
    """Draft aspect questions per stance; the reconcile merges and keeps three."""
    prompt = _prompt(request)
    if "DRAFT QUESTIONS" in prompt:
        return json.dumps(
            {
                "questions": [
                    {"aspect": "errors", "text": ERROR.text},
                    {"aspect": "entities", "text": SHEET.text},
                    {"aspect": "pending", "text": QUEUED.text},
                    {"aspect": "errors", "text": "  The most recent result reports an error. "},
                ]
            }
        )
    if "aspect 'observed'" in prompt:
        return "I cannot help with that."
    if "aspect 'errors'" in prompt:
        return "```json\n" + json.dumps({"questions": [{"aspect": "errors", "text": ERROR.text}]})
    return json.dumps(
        {
            "questions": [
                {"aspect": "entities", "text": SHEET.text},
                {"aspect": "weather", "text": "Is it raining?"},
                {"aspect": "pending", "text": QUEUED.text},
            ]
        }
    )


def samples() -> tuple[PreTurnState, ...]:
    """Return two fit pre-turn states shown to the author."""
    rows = (transition("a", 0, "open_sheet", "opened"), transition("a", 1, "write", "error"))
    return tuple(state for _, state in pre_turn_steps(rows))


def test_author_drafts_each_variant_reconciles_and_records_unreadable_drafts() -> None:
    """Every variant drafts once, the reconcile merges, and a refusal is recorded, not redrawn."""
    client = ScriptedNamer(author_reply)
    result = author_state_tags("sheets", samples(), StateTagAuthor(client=client, model=SNAPSHOT))
    assert len(client.requests) == StateTagAuthoringSpec().calls == 5
    stances = [_prompt(request).split("\n")[1] for request in client.requests[:4]]
    assert len(set(stances)) == 4, "each prompt variant writes its own draft"
    assert [question.question_id for question in result.questions] == [
        ERROR.question_id,
        SHEET.question_id,
        QUEUED.question_id,
    ]
    drafts = {draft.variant_id: draft for draft in result.drafts}
    assert drafts["observed"].error is not None and not drafts["observed"].questions
    assert [q.aspect for q in drafts["entities"].questions] == ["entities", "pending"]
    reconcile = _prompt(client.requests[-1])
    assert ERROR.text in reconcile and "sample 1" in reconcile
    assert result.decisions_per_state == 1


def test_authoring_refuses_when_no_draft_is_readable() -> None:
    """With no usable draft there is no reconcile and no bank."""
    client = ScriptedNamer(lambda request: "no json here")
    with pytest.raises(ValueError, match="no state tag draft"):
        author_state_tags("sheets", samples(), StateTagAuthor(client=client, model=SNAPSHOT))
    assert len(client.requests) == 4


def test_saved_bank_replays_without_any_call(tmp_path: Path) -> None:
    """A second authoring of the same family, model and evidence reads the saved artifact."""
    store = ProjectStore(tmp_path, "fixture").artifacts
    client = ScriptedNamer(author_reply)
    author = StateTagAuthor(client=client, model=SNAPSHOT)
    first, reference = authored_state_tags(store, "sheets", samples(), author, code_revision="t")
    again, same = authored_state_tags(store, "sheets", samples(), author, code_revision="t")
    assert (again, same) == (first, reference) and len(client.requests) == 5
    assert store.read(reference.artifact_id).manifest.artifact_type == STATE_TAG_BANK_TYPE


def test_each_step_sees_only_earlier_steps_in_trace_order() -> None:
    """A step's pre-turn state holds the steps before it, never its own transition."""
    started = datetime(2026, 11, 1, tzinfo=UTC)
    spans = tuple(
        TraceSpan(
            span_id=span,
            name="agent.tool_call",
            started_at=started + timedelta(seconds=offset),
            ended_at=started + timedelta(seconds=offset + 1),
        )
        for span, offset in (("b-a0", 0), ("b-o0", 1), ("b-a1", 2), ("b-o1", 3), ("b-a2", 4))
    )
    trace = Trace(
        trace_id="b",
        task="Update the budget workbook",
        spans=spans,
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )
    shuffled = [
        transition("b", 2, "save", "saved"),
        transition("b", 0, "open_sheet", "opened"),
        transition("b", 1, "write", "error: locked"),
    ]
    ordered = trace_order([trace], shuffled)["b"]
    assert [row.transition_id for row in ordered] == ["t-b-0", "t-b-1", "t-b-2"]
    steps = pre_turn_steps(ordered, history_steps=1)
    assert [state.step_index for _, state in steps] == [0, 1, 2]
    assert steps[0][1].recent == ()
    assert [view.observation for view in steps[2][1].recent] == ["error: locked"]
    with pytest.raises(ValueError, match="action span"):
        trace_order([trace], [transition("c", 0, "open_sheet", "opened")])


def test_step_tagging_is_strict_chunked_and_replays_through_the_decision_cache(
    tmp_path: Path,
) -> None:
    """One proposition per question; a missing answer refuses; a cached replay makes no call."""
    state = pre_turn_steps(
        (transition("a", 0, "open_sheet", "opened"), transition("a", 1, "w", ""))
    )[1][1]
    decisions = ScriptedDecisions()
    probabilities, digests, responses = tag_state(bank(), state, decisions)
    assert probabilities == (0.05, 0.95, 0.05, 0.05), "its own empty result is not visible yet"
    assert len(digests) == len(responses) == 1
    many = bank(*(state_question("entities", f"Record {n} is present.") for n in range(40)))
    assert [len(r.questions) for r in state_tag_requests(many, state)] == [32, 8]
    assert many.decisions_per_state == 2

    def missing(request: DecisionRequest) -> JsonObject:
        """Answer nothing."""
        return {"answers": {}}

    with pytest.raises(ValueError, match="no proposition answer"):
        tag_state(bank(), state, missing)

    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    cached = cached_state_decider(cache, decisions, {"model": "clef"})
    first = tag_state(bank(), state, cached)
    calls = len(decisions.requests)
    assert tag_state(bank(), state, cached) == first
    assert len(decisions.requests) == calls, "a replayed step tag makes no decision call"
    assert sha256_json(first[2][0]) == sha256_json(responses[0])


def test_live_state_renders_simulator_history_in_the_recorded_shape() -> None:
    """Live history turns become earlier steps; the environment rides along."""
    live = live_pre_turn_state(
        task="Update the budget workbook",
        initial_context={},
        history=[
            {
                "actions": [{"name": "open_sheet", "call_id": "c", "arguments": {}}],
                "observations": [{"content": "opened", "is_error": False}],
            }
        ],
        environment={"sheets": {"Budget": {}}},
        history_steps=4,
        observation_characters=100,
    )
    assert live.step_index == 1 and live.recent[0].action["tool_name"] == "open_sheet"
    assert live.recent[0].is_error is False and live.environment == {"sheets": {"Budget": {}}}
    probabilities, _, _ = tag_state(bank(), live, ScriptedDecisions())
    assert probabilities[1] == 0.95

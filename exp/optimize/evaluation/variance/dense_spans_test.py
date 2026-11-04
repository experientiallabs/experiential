"""Mapping findings to actor spans: attribution, exclusions, quotes and the ledger trajectory."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from exp.common.models import ToolCall
from exp.optimize.evaluation.executable.ledger import TrialLedgerEntry
from exp.optimize.evaluation.variance.dense_judge_test import (
    ANSWER,
    EDIT,
    SHA,
    event,
    evidence,
    trajectory,
)
from exp.optimize.evaluation.variance.dense_spans import (
    ActorTrajectory,
    ActorTurn,
    actor_spans,
    tool_call_text,
    trajectory_from_ledger,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence


def test_replies_are_attributed_to_their_calls_and_answers_to_their_turns() -> None:
    """A world reply lands on the call it answers; the final answer on its own turn."""
    spans = actor_spans(evidence(), trajectory(), ("e4", "e5", "e3", "e0", "missing"))
    assert [(s.event_id, s.cited_event_id, s.turn_index, s.part) for s in spans] == [
        ("e3", "e4", 1, "tool_call"),
        ("e5", "e5", 2, "content"),
    ]
    assert spans[1].char_end == len(ANSWER) and spans[1].granularity == "action"


def test_quotes_narrow_and_context_or_imposed_events_never_map() -> None:
    """A quote narrows the rendered span; experiment-owned and context events have no span."""
    (span,) = actor_spans(evidence(), trajectory(), ("e1",), quote="parse_date")
    text = tool_call_text(EDIT)
    assert text[span.char_start : span.char_end] == "parse_date"
    imposed = evidence().model_copy(
        update={
            "events": (
                *evidence().events[:1],
                event(1, "tool_call", "experiment", "scored", EDIT.model_dump(mode="json")),
                *evidence().events[2:],
            )
        }
    )
    assert actor_spans(imposed, trajectory(), ("e1", "e2", "e0")) == ()


def test_a_ledger_trajectory_has_one_turn_per_actor_action() -> None:
    """Messages, tool calls and the submission each become a turn, in ledger order."""
    now = datetime.now(UTC)
    entries = [
        TrialLedgerEntry(sequence=0, kind="start", content={"initial_state": {}}, recorded_at=now),
        TrialLedgerEntry(sequence=1, kind="message", content={"text": "On it."}, recorded_at=now),
        TrialLedgerEntry(
            sequence=2,
            kind="tool_call",
            content={"action": EDIT.model_dump(mode="json")},
            recorded_at=now,
        ),
        TrialLedgerEntry(
            sequence=3,
            kind="submission",
            content={"answer": {"done": True}, "final_state": {}},
            recorded_at=now,
        ),
    ]
    turns = trajectory_from_ledger(entries).turns
    assert [turn.content for turn in turns] == ["On it.", None, json.dumps({"done": True})]
    assert turns[1].tool_calls == (EDIT,)


def _occurrences() -> tuple[EpisodeEvidence, ActorTrajectory]:
    """Two different calls reusing call id c1, each with a reply, then two identical messages."""
    first = ToolCall(call_id="c1", name="edit_file", arguments={"path": "a.py"})
    second = ToolCall(call_id="c1", name="edit_file", arguments={"path": "b.py"})
    episode = EpisodeEvidence(
        episode_id="repeat",
        task_sha256=SHA,
        world_sha256=SHA,
        complete=True,
        events=(
            event(0, "tool_call", "actor", "scored", first.model_dump(mode="json")),
            event(1, "tool_result", "world", "scored", {"tool_call_id": "c1", "content": "a"}),
            event(2, "tool_call", "actor", "scored", second.model_dump(mode="json")),
            event(3, "tool_result", "world", "scored", {"tool_call_id": "c1", "content": "b"}),
            event(4, "assistant_message", "actor", "scored", "Checking."),
            event(5, "assistant_message", "actor", "scored", "Checking."),
        ),
    )
    turns = ActorTrajectory(
        kind="continuation-branch",
        turns=(
            ActorTurn(turn_index=0, tool_calls=(first,)),
            ActorTurn(turn_index=1, tool_calls=(second,)),
            ActorTurn(turn_index=2, content="Checking."),
            ActorTurn(turn_index=3, content="Checking."),
        ),
    )
    return episode, turns


def test_reused_call_ids_and_repeated_messages_bind_to_their_own_occurrence() -> None:
    """A finding on the second c1 call, or its reply, lands on turn 1, never turn 0."""
    episode, turns = _occurrences()
    assert [s.turn_index for s in actor_spans(episode, turns, ("e2",))] == [1]
    assert [(s.event_id, s.turn_index) for s in actor_spans(episode, turns, ("e3",))] == [("e2", 1)]
    assert [(s.event_id, s.turn_index) for s in actor_spans(episode, turns, ("e1",))] == [("e0", 0)]
    assert [s.turn_index for s in actor_spans(episode, turns, ("e5", "e4"))] == [3, 2]


def test_a_call_whose_arguments_differ_from_the_trajectory_gets_no_span() -> None:
    """An event that matches no trajectory occurrence fails closed instead of borrowing one."""
    episode, turns = _occurrences()
    changed = ToolCall(call_id="c1", name="edit_file", arguments={"path": "c.py"})
    shifted = ActorTrajectory(
        kind="continuation-branch",
        turns=(turns.turns[0], ActorTurn(turn_index=1, tool_calls=(changed,))),
    )
    assert actor_spans(episode, shifted, ("e2",)) == ()
    assert [s.turn_index for s in actor_spans(episode, shifted, ("e0",))] == [0]

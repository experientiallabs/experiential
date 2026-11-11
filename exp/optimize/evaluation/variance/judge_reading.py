"""Bounded read-only event batches for the chronological episode judge."""

import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ModelMessage, ModelRequest, ModelResponse, ToolCall
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence
from exp.optimize.evaluation.variance.judge_prompt import READ_TOOL, EventRange


@dataclass(frozen=True)
class ReadReplies:
    """One assistant read batch with one ordered reply per call.

    Attributes:
        messages: The conversation including the one assistant action and every tool reply.
        reads: Total reads consumed, including refused ranges.
        inspected: Event identities included in byte-admitted replies only.
    """

    messages: tuple[ModelMessage, ...]
    reads: int
    inspected: frozenset[str]


def response_call_problem(
    calls: tuple[ToolCall, ...], offered: frozenset[str], reads_left: int
) -> str | None:
    """Return why a response cannot advance a phase or serve its read-only batch.

    Args:
        calls: The provider response's tool calls.
        offered: Tool names offered in this exact request.
        reads_left: Remaining event reads under the frozen profile.

    Returns:
        A refusal reason, or None for one phase call or a bounded batch of reads.
    """
    if not calls or any(call.name not in offered for call in calls):
        return f"judge must call only offered tools {sorted(offered)}"
    identities = [call.call_id for call in calls]
    if len(set(identities)) != len(identities):
        return "judge read or phase call ids must be unique within the response"
    reads = all(call.name == READ_TOOL for call in calls)
    if len(calls) > 1 and not reads:
        return "judge may batch only read_events calls, never phase or assessment calls"
    if reads and len(calls) > reads_left:
        return f"judge requested {len(calls)} event reads with only {reads_left} left"
    return None


def _range_reply(evidence: EpisodeEvidence, arguments: JsonObject) -> tuple[str, frozenset[str]]:
    """Render one complete declared range, or a small refusal with no inspected events."""
    try:
        wanted = EventRange.model_validate(arguments)
    except ValidationError as error:
        return f"read refused: malformed range ({error.error_count()} errors)", frozenset()
    by_id = {event.event_id: event for event in evidence.events}
    first, last = by_id.get(wanted.first_event_id), by_id.get(wanted.last_event_id)
    if first is None or last is None or first.sequence > last.sequence:
        return (
            "read refused: name two existing event ids, the first not after the last",
            frozenset(),
        )
    events = evidence.events[first.sequence : last.sequence + 1]
    reply = json.dumps(
        {"events": [event.model_dump(mode="json") for event in events]}, sort_keys=True
    )
    return reply, frozenset(event.event_id for event in events)


def read_event_batch(
    evidence: EpisodeEvidence,
    request: ModelRequest,
    response: ModelResponse,
    *,
    reads: int,
    budget: int,
    maximum_input_bytes: int,
    request_builder: Callable[[tuple[ModelMessage, ...], int], ModelRequest],
) -> ReadReplies:
    """Answer an admitted read batch, size-checking each reply against cumulative history.

    Args:
        evidence: Complete immutable event ledger.
        request: Request that offered the reads.
        response: Validated response containing only uniquely identified read calls.
        reads: Event reads already consumed.
        budget: Frozen total event-read allowance.
        maximum_input_bytes: Serialized input bound for the next request.
        request_builder: Build the next request from messages and remaining reads.

    Returns:
        One assistant action, every matching tool reply, and the admitted inspection set.
    """
    messages = (*request.messages, ModelMessage(role="assistant", assistant_action=response.output))
    inspected: set[str] = set()
    for call in response.output.tool_calls:
        reads += 1
        reply, selected = _range_reply(evidence, call.arguments)
        tool = ModelMessage(role="tool", tool_call_id=call.call_id, content=reply)
        trial = request_builder((*messages, tool), budget - reads)
        if selected and len(trial.model_dump_json().encode()) > maximum_input_bytes:
            reply = (
                f"read refused: {len(selected)} events need {len(reply.encode())} bytes, more than "
                "the remaining input budget; read a smaller range"
            )
            selected = frozenset()
            tool = ModelMessage(role="tool", tool_call_id=call.call_id, content=reply)
        messages = (*messages, tool)
        inspected.update(selected)
    return ReadReplies(messages=messages, reads=reads, inspected=frozenset(inspected))


def evidence_scope(evidence: EpisodeEvidence) -> JsonObject:
    """Render context, scored, imposed and actor-owned identities from the recorded ledger."""
    scored = [event for event in evidence.events if event.scope == "scored"]
    return {
        "context_event_ids": [
            event.event_id for event in evidence.events if event.scope == "context"
        ],
        "scored_event_ids": [event.event_id for event in scored],
        "scored_start": scored[0].event_id if scored else None,
        "experiment_imposed_event_ids": [
            event.event_id for event in scored if event.owner == "experiment"
        ],
        "actor_scored_event_ids": [event.event_id for event in scored if event.owner == "actor"],
    }


def event_inventory(event: EpisodeEvent) -> JsonValue:
    """Omit event content while retaining its owner, scope and exact serialized size."""
    row = event.model_dump(mode="json", exclude={"content"})
    row["content_bytes"] = len(json.dumps(event.content, sort_keys=True).encode())
    return row

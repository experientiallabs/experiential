"""Map dense findings from episode events to the actor's action and token spans.

A finding cites episode event ids. A trainer needs to know which of the actor's own sampled
outputs the finding is about. :func:`actor_spans` maps each cited event to one span of the
actor trajectory (:class:`ActorTrajectory`): the turn index, whether it is message content or a
tool call (and which tool call) and rendered character offsets within that part's text. Rendered
offsets are not sampled-token positions. Only when the actor's actual sampled tokens are bound
(:class:`SampledTokens`, from the provider response or the trainer rollout) and their decoded
text reproduces the part's text exactly does a span also carry sampled-token offsets and
``sampled_tokens`` granularity; otherwise it is ``action`` granularity and a trainer may apply
its credit only to the whole action.

- A scored actor event maps to itself: a ``tool_call`` to its call (matched by ``call_id``), a
  message or final answer to the turn whose content is that exact text.
- A world or harness event that answers a call (its content names ``tool_call_id``) is attributed
  to the actor call it answers: the reply showed the effect, the call caused it.
- Context events (the inherited prefix) and experiment-imposed actions never map: they are not
  actor-sampled in the scored window, so no dense reward may land on them.
- A finding's ``quote``, when it occurs in the part's text, narrows the span to that text.

Tool-call text is rendered as ``json.dumps(arguments, sort_keys=True)``
(:data:`~exp.optimize.evaluation.variance.dense_reward.TOOL_CALL_RENDERING`); each span records
the digest of its part's text so the trainer can check that its own rendering agrees before it
uses the offsets. Message content is the exact text.

Two trajectories are supported. :func:`trajectory_from_branch` reads a saved continuation
branch: turn ``i`` is the actor's model call ``i``. :func:`trajectory_from_ledger` reads a Harbor
trial ledger, where the external actor's own turns are not visible to the host: each actor
action (a message, a tool call or the submission) is its own turn, and the trainer realigns
spans with its own trajectory by ``call_id`` or exact message text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonValue, sha256_bytes
from exp.common.models import ModelResponse, ToolCall
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.executable.ledger import TrialLedgerEntry
from exp.optimize.evaluation.variance.dense_reward import DenseSpan, SampledTokenSpan
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence

TrajectoryKind = Literal["continuation-branch", "harbor-ledger"]
"""Which trajectory a span's turn index refers to."""


class SampledPart(ContractModel):
    """The actor's actual sampled tokens of one action part.

    Attributes:
        turn_index: Actor turn.
        part: ``content`` or ``tool_call``.
        tool_call_index: Tool-call position, for a ``tool_call`` part.
        token_ids: Sampled token ids, in order.
        token_texts: Each sampled token's decoded text, in order.
    """

    turn_index: int = Field(ge=0)
    part: Literal["content", "tool_call"]
    tool_call_index: int | None = Field(default=None, ge=0)
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]

    @model_validator(mode="after")
    def _paired(self) -> Self:
        """Require one decoded text per token id."""
        if len(self.token_ids) != len(self.token_texts):
            raise ValueError("each sampled token id needs its decoded text")
        return self


class SampledTokens(ContractModel):
    """The actor's sampled tokens, bound from a provider response or a trainer rollout.

    Attributes:
        binding: Identity of the record they come from, such as a saved provider response with
            token logprobs or a trainer rollout id.
        parts: Sampled tokens per action part.
    """

    binding: str = Field(min_length=1, max_length=512)
    parts: tuple[SampledPart, ...]


class ActorTurn(ContractModel):
    """One actor turn: its message content and tool calls.

    Attributes:
        turn_index: Position in the trajectory.
        content: Message text, when the turn has any.
        tool_calls: Tool calls the turn made, in order.
    """

    turn_index: int = Field(ge=0)
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ActorTrajectory(ContractModel):
    """The actor's own sampled outputs, which spans index.

    Attributes:
        kind: Which trajectory the turn indexes refer to.
        turns: Every turn in order; turn ``i`` has ``turn_index`` ``i``.
    """

    kind: TrajectoryKind
    turns: tuple[ActorTurn, ...]


def tool_call_text(call: ToolCall) -> str:
    """Render a tool call's span text: its arguments as sorted JSON."""
    return json.dumps(call.arguments, sort_keys=True)


def answer_text(answer: JsonValue) -> str:
    """Render message or answer content: text as is, other JSON as sorted JSON."""
    return answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True)


def trajectory_from_branch(branch: ContinuationBranch) -> ActorTrajectory:
    """Return the actor trajectory of a saved continuation branch.

    Args:
        branch: The saved branch; turn ``i`` is its model call ``i``.

    Returns:
        One turn per model call; a call without a recorded response is an empty turn.
    """
    calls = branch.outcome.get("model_calls", [])
    turns: list[ActorTurn] = []
    for index, row in enumerate(calls if isinstance(calls, list) else []):
        raw = row.get("response") if isinstance(row, dict) else None
        if raw is None:
            turns.append(ActorTurn(turn_index=index))
            continue
        output = ModelResponse.model_validate(raw).output
        turns.append(
            ActorTurn(turn_index=index, content=output.content, tool_calls=output.tool_calls)
        )
    return ActorTrajectory(kind="continuation-branch", turns=tuple(turns))


def trajectory_from_ledger(entries: Sequence[TrialLedgerEntry]) -> ActorTrajectory:
    """Return the actor actions of a Harbor trial ledger, one action per turn.

    Args:
        entries: The trial's ledger in sequence order.

    Returns:
        A turn per message, tool call and submission, in ledger order.
    """
    turns: list[ActorTurn] = []
    for entry in entries:
        content = entry.content
        if entry.kind == "tool_call":
            call = ToolCall.model_validate(content.get("action"))
            turns.append(ActorTurn(turn_index=len(turns), tool_calls=(call,)))
        elif entry.kind == "message":
            turns.append(ActorTurn(turn_index=len(turns), content=answer_text(content.get("text"))))
        elif entry.kind == "submission":
            turns.append(
                ActorTurn(turn_index=len(turns), content=answer_text(content.get("answer")))
            )
    return ActorTrajectory(kind="harbor-ledger", turns=tuple(turns))


def _alignment(
    evidence: EpisodeEvidence, trajectory: ActorTrajectory
) -> dict[str, tuple[int, int | None, str, str | None]]:
    """Align scored actor events with trajectory parts by ordered occurrence.

    Events and trajectory parts are walked in order with one forward cursor for tool calls and
    one for message content, so a provider that reuses a call id, or an actor that repeats the
    same message, still binds each event to its own occurrence. A tool call binds only to a
    later trajectory call with the same id and identical arguments; an event without such an
    occurrence stays unaligned (no span) rather than landing on another action.

    Returns:
        Event id mapped to (turn index, tool-call index or None, part text, call id).
    """
    calls = [
        (turn.turn_index, position, call)
        for turn in trajectory.turns
        for position, call in enumerate(turn.tool_calls)
    ]
    aligned: dict[str, tuple[int, int | None, str, str | None]] = {}
    call_cursor = content_cursor = 0
    for event in evidence.events:
        if event.scope != "scored" or event.owner != "actor":
            continue
        if event.kind == "tool_call" and isinstance(event.content, dict):
            key, arguments = event.content.get("call_id"), event.content.get("arguments", {})
            for index in range(call_cursor, len(calls)):
                turn, position, call = calls[index]
                if call.call_id == key and call.arguments == arguments:
                    aligned[event.event_id] = (turn, position, tool_call_text(call), call.call_id)
                    call_cursor = index + 1
                    break
        elif event.kind in ("assistant_message", "final_answer"):
            text = answer_text(event.content)
            for index in range(content_cursor, len(trajectory.turns)):
                if trajectory.turns[index].content == text:
                    aligned[event.event_id] = (trajectory.turns[index].turn_index, None, text, None)
                    content_cursor = index + 1
                    break
    return aligned


def actor_event(evidence: EpisodeEvidence, event: EpisodeEvent) -> EpisodeEvent | None:
    """Return the scored actor event a cited event lands on, or None when it has none.

    A reply binds to the nearest preceding actor call with its ``tool_call_id``, so a reused
    call id never attributes a later reply to an earlier call.
    """
    if event.scope != "scored" or event.owner == "experiment":
        return None
    if event.owner == "actor":
        return event
    content = event.content
    key = content.get("tool_call_id") if isinstance(content, dict) else None
    if not isinstance(key, str):
        return None
    for candidate in reversed(evidence.events[: event.sequence]):
        call = candidate.content
        if (
            candidate.scope == "scored"
            and candidate.owner == "actor"
            and candidate.kind == "tool_call"
            and isinstance(call, dict)
            and call.get("call_id") == key
        ):
            return candidate
    return None


def _sampled(
    sampled: SampledTokens | None,
    key: tuple[int, str, int | None],
    text: str,
    start: int,
    end: int,
) -> SampledTokenSpan | None:
    """Return sampled-token offsets covering ``[start, end)``, or None when not bound.

    Offsets are given only when the part's sampled tokens decode to exactly its rendered text.
    """
    if sampled is None:
        return None
    part = next(
        (row for row in sampled.parts if (row.turn_index, row.part, row.tool_call_index) == key),
        None,
    )
    if part is None or "".join(part.token_texts) != text:
        return None
    covered: list[int] = []
    position = 0
    for index, piece in enumerate(part.token_texts):
        low, position = position, position + len(piece)
        if low < end and position > start:
            covered.append(index)
    if not covered:
        return None
    return SampledTokenSpan(
        binding=sampled.binding, token_start=covered[0], token_end=covered[-1] + 1
    )


def actor_spans(
    evidence: EpisodeEvidence,
    trajectory: ActorTrajectory,
    refs: Sequence[str],
    *,
    quote: str | None = None,
    sampled: SampledTokens | None = None,
) -> tuple[DenseSpan, ...]:
    """Map cited events to actor spans.

    Args:
        evidence: The judged evidence.
        trajectory: The actor trajectory the spans index.
        refs: Cited event ids, in citation order.
        quote: Exact actor text to narrow each span to, when it occurs in the part.
        sampled: The actor's bound sampled tokens, for token-local spans.

    Returns:
        One span per distinct actor part the citations land on, in citation order.

    """
    by_id = {event.event_id: event for event in evidence.events}
    aligned = _alignment(evidence, trajectory)
    spans: list[DenseSpan] = []
    seen: set[str] = set()
    for ref in refs:
        cited = by_id.get(ref)
        target = None if cited is None else actor_event(evidence, cited)
        if target is None or target.event_id not in aligned or target.event_id in seen:
            continue
        seen.add(target.event_id)
        turn, position, text, call_id = aligned[target.event_id]
        start, end = 0, len(text)
        if quote and quote in text:
            start = text.index(quote)
            end = start + len(quote)
        part = "content" if position is None else "tool_call"
        bound = _sampled(sampled, (turn, part, position), text, start, end)
        spans.append(
            DenseSpan(
                event_id=target.event_id,
                cited_event_id=ref,
                turn_index=turn,
                part=part,
                tool_call_index=position,
                call_id=call_id,
                char_start=start,
                char_end=end,
                text_sha256=sha256_bytes(text.encode()),
                sampled=bound,
                granularity="action" if bound is None else "sampled_tokens",
            )
        )
    return tuple(spans)

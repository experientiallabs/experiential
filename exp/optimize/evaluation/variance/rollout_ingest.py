"""The versioned rollout-ingest contract: what an external GRPO trainer exports at each epoch.

The evolving rubric (:mod:`~.rubric_evolution`) revises the two-part rubric only between training
epochs, from the rollouts the trainer actually sampled under the frozen rubric of the epoch that
just ended. The trainer writes one JSON document per epoch, :class:`RolloutExport`, under the
contract named by :data:`ROLLOUT_INGEST_CONTRACT`. A document is accepted whole or refused whole.

Two fidelities are accepted, per rollout:

- ``full``: the actor's own turns, each with its message content and tool calls, every tool call
  with the observation the environment returned, and, when the trainer has them, the sampled
  token ids and decoded token texts of every action part (:class:`PartTokens`). Token spans let a
  dense finding land on the exact sampled tokens (``sampled_tokens`` granularity); without them
  findings land on whole actions (``action`` granularity).
- ``transcript``: the fallback when the trainer keeps only a chat transcript and a scalar reward.
  Turns are plain role and text; tool calls are not separable, so findings land on whole messages.

Every rollout states its ``lineage_id``: rollouts that share a prompt, a seed group or a parent
checkpoint lineage share one, so admission counts them once (two or more lineages are required to
admit a rubric item). ``in_loop_reward`` is the scalar the trainer optimized, and ``rubric_sha256``
names the frozen rubric version it was computed under; an export whose version is not the one
frozen for its epoch is refused, so epochs never mix.

:func:`rollout_evidence` is the adapter from an accepted rollout to the harness-owned
:class:`~.episodes.EpisodeEvidence` the frozen judges read, with the matching
:class:`~.dense_spans.ActorTrajectory` and :class:`~.dense_spans.SampledTokens`. Nothing in it
interprets the content: it only orders the recorded events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.optimize.evaluation.variance.dense_spans import (
    ActorTrajectory,
    ActorTurn,
    SampledPart,
    SampledTokens,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence

ROLLOUT_INGEST_CONTRACT = "rollout-ingest-v1"
"""Version of the export layout and of the evidence adapter; bump when either changes."""

ROLLOUT_TRAJECTORY = "trainer-rollout"
"""Trajectory kind of an ingested rollout: turn ``i`` is the trainer's own actor turn ``i``."""


class PartTokens(ContractModel):
    """The sampled tokens of one action part, exactly as the trainer sampled them.

    Attributes:
        token_ids: Sampled token ids, in order.
        token_texts: Each token's decoded text, in order; their concatenation must reproduce the
            part's text (the message content, or the tool call's arguments rendered as sorted
            JSON) for spans to reach token granularity.
    """

    token_ids: tuple[int, ...] = Field(min_length=1)
    token_texts: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _paired(self) -> Self:
        """Require one decoded text per token id."""
        if len(self.token_ids) != len(self.token_texts):
            raise ValueError("each sampled token id needs its decoded text")
        return self


class RolloutCall(ContractModel):
    """One tool call of an actor turn and the observation it produced.

    Attributes:
        call_id: The call's identity, unique within the rollout.
        name: Tool name.
        arguments: Decoded call arguments.
        result: The observation the environment returned.
        is_error: Whether the environment reported the call as failed.
        tokens: Sampled tokens of the call, when the trainer exports them.
    """

    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: JsonObject = Field(default_factory=dict)
    result: JsonValue = None
    is_error: bool = False
    tokens: PartTokens | None = None


class RolloutTurn(ContractModel):
    """One actor turn of a ``full`` rollout.

    Attributes:
        turn_index: Position in the rollout, from zero.
        content: Visible message text, when the turn has any.
        content_tokens: Sampled tokens of the content, when exported.
        calls: Tool calls of the turn, in order.
        final: Whether this turn's content is the actor's final answer.
    """

    turn_index: int = Field(ge=0)
    content: str | None = None
    content_tokens: PartTokens | None = None
    calls: tuple[RolloutCall, ...] = ()
    final: bool = False

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Require some action, and content tokens only beside content."""
        if self.content is None and not self.calls:
            raise ValueError(f"turn {self.turn_index} has neither content nor tool calls")
        if self.content_tokens is not None and self.content is None:
            raise ValueError(f"turn {self.turn_index} has content tokens without content")
        if self.final and self.content is None:
            raise ValueError(f"final turn {self.turn_index} must carry the final answer text")
        return self


class TranscriptTurn(ContractModel):
    """One turn of a ``transcript`` rollout (the fallback without separable tool calls).

    Attributes:
        role: ``user`` for task or user text, ``assistant`` for the actor, ``tool`` for an
            observation the transcript shows.
        content: The turn's text.
    """

    role: Literal["user", "assistant", "tool"]
    content: str = Field(min_length=1)


class Rollout(ContractModel):
    """One sampled rollout of the epoch.

    Attributes:
        rollout_id: Identity, unique within the export.
        task_id: The trained task (the Harbor package's task) it was sampled on.
        lineage_id: Independence class: rollouts sharing a prompt, seed group or parent lineage
            share one and count once for admission.
        group_id: The GRPO group it was sampled in.
        instruction: The task text the actor received.
        in_loop_reward: The scalar the trainer optimized for this rollout.
        reward_parts: Its components as the trainer computed them (for example ``outcome`` and
            ``dense``), recorded only.
        fidelity: ``full`` (turns with tool calls and optional token spans) or ``transcript``.
        turns: The actor's turns, for ``full``.
        transcript: The transcript turns, for ``transcript``.
    """

    rollout_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    task_id: str = Field(min_length=1, max_length=256)
    lineage_id: str = Field(min_length=1, max_length=256)
    group_id: str = Field(default="", max_length=256)
    instruction: str = Field(min_length=1)
    in_loop_reward: float
    reward_parts: dict[str, float] = Field(default_factory=dict)
    fidelity: Literal["full", "transcript"]
    turns: tuple[RolloutTurn, ...] = ()
    transcript: tuple[TranscriptTurn, ...] = ()

    @model_validator(mode="after")
    def _fidelity(self) -> Self:
        """Require the layout the fidelity names, contiguous turns and unique call ids."""
        if self.fidelity == "full":
            if not self.turns or self.transcript:
                raise ValueError(f"full rollout {self.rollout_id} needs turns and no transcript")
            if tuple(t.turn_index for t in self.turns) != tuple(range(len(self.turns))):
                raise ValueError(f"rollout {self.rollout_id} turns must be indexed 0..n-1")
            ids = [call.call_id for turn in self.turns for call in turn.calls]
            if len(set(ids)) != len(ids):
                raise ValueError(f"rollout {self.rollout_id} repeats a tool call id")
        elif not self.transcript or self.turns:
            raise ValueError(f"transcript rollout {self.rollout_id} needs a transcript only")
        elif not any(row.role == "assistant" for row in self.transcript):
            raise ValueError(f"transcript rollout {self.rollout_id} has no assistant turn")
        return self

    @property
    def has_tokens(self) -> bool:
        """Whether any action part carries sampled tokens."""
        return any(
            turn.content_tokens is not None or any(c.tokens is not None for c in turn.calls)
            for turn in self.turns
        )


class RolloutExport(ContractModel):
    """One epoch's export, written by the trainer after the epoch's last update.

    Attributes:
        contract: Must be :data:`ROLLOUT_INGEST_CONTRACT`.
        trainer_run: The training run.
        epoch: The epoch that just ended, from one.
        rubric_version: The rubric version number the epoch trained against.
        rubric_sha256: That version's identity; it must be the one frozen for the epoch.
        policy_checkpoint: The policy checkpoint that sampled the rollouts.
        rollouts: Every exported rollout, unique by identity.
    """

    contract: Literal["rollout-ingest-v1"] = ROLLOUT_INGEST_CONTRACT
    trainer_run: str = Field(min_length=1, max_length=256)
    epoch: int = Field(ge=1)
    rubric_version: int = Field(ge=1)
    rubric_sha256: Sha256
    policy_checkpoint: str = Field(min_length=1, max_length=512)
    rollouts: tuple[Rollout, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        """Require unique rollout identities."""
        ids = [rollout.rollout_id for rollout in self.rollouts]
        if len(set(ids)) != len(ids):
            raise ValueError("an export repeats a rollout id")
        return self

    @property
    def identity(self) -> Sha256:
        """Digest of the whole export."""
        return sha256_json(self)


def load_rollout_export(path: Path) -> RolloutExport:
    """Read and validate one export file, refusing any other contract.

    Raises:
        ValueError: The file is not a valid ``rollout-ingest-v1`` export.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != ROLLOUT_INGEST_CONTRACT:
        raise ValueError(f"{path} is not a {ROLLOUT_INGEST_CONTRACT} export")
    return RolloutExport.model_validate(payload)


def _event(
    index: int, kind: str, owner: str, scope: str, content: JsonValue, source: str
) -> EpisodeEvent:
    """One ordered event ``e<index>``."""
    return EpisodeEvent.model_validate(
        {
            "event_id": f"e{index}",
            "sequence": index,
            "kind": kind,
            "owner": owner,
            "scope": scope,
            "content": content,
            "source": source,
        }
    )


def rollout_evidence(
    rollout: Rollout, *, task_sha256: Sha256, world_sha256: Sha256, trainer_run: str
) -> tuple[EpisodeEvidence, ActorTrajectory, SampledTokens | None]:
    """Order an accepted rollout as episode evidence, its actor trajectory and sampled tokens.

    The instruction is context; every actor message and call is scored; every observation is a
    world event answering its call. A transcript rollout keeps its turns in order: user turns are
    context, assistant turns are scored actor messages (the last is the final answer) and tool
    turns are world observations that answer no recorded call.

    Args:
        rollout: The rollout.
        task_sha256: Digest of the trained task, as the frozen judges bind it.
        world_sha256: Digest of the frozen world it ran in, as the frozen judges bind it.
        trainer_run: The training run, part of each event's source locator.

    Returns:
        The evidence, the actor trajectory, and the sampled tokens when any part carries them.
    """
    source = f"rollout:{trainer_run}:{rollout.rollout_id}"
    events = [_event(0, "user_message", "user", "context", rollout.instruction, f"{source}:task")]
    turns: list[ActorTurn] = []
    parts: list[SampledPart] = []
    if rollout.fidelity == "full":
        for turn in rollout.turns:
            for position, call in enumerate(turn.calls):
                tool = ToolCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
                events.append(
                    _event(
                        len(events),
                        "tool_call",
                        "actor",
                        "scored",
                        tool.model_dump(mode="json"),
                        f"{source}:turn:{turn.turn_index}:call:{position}",
                    )
                )
                events.append(
                    _event(
                        len(events),
                        "tool_error" if call.is_error else "tool_result",
                        "world",
                        "scored",
                        {"tool_call_id": call.call_id, "content": call.result},
                        f"{source}:turn:{turn.turn_index}:result:{position}",
                    )
                )
                if call.tokens is not None:
                    parts.append(
                        SampledPart(
                            turn_index=turn.turn_index,
                            part="tool_call",
                            tool_call_index=position,
                            token_ids=call.tokens.token_ids,
                            token_texts=call.tokens.token_texts,
                        )
                    )
            if turn.content is not None:
                events.append(
                    _event(
                        len(events),
                        "final_answer" if turn.final else "assistant_message",
                        "actor",
                        "scored",
                        turn.content,
                        f"{source}:turn:{turn.turn_index}:content",
                    )
                )
                if turn.content_tokens is not None:
                    parts.append(
                        SampledPart(
                            turn_index=turn.turn_index,
                            part="content",
                            token_ids=turn.content_tokens.token_ids,
                            token_texts=turn.content_tokens.token_texts,
                        )
                    )
            turns.append(
                ActorTurn(
                    turn_index=turn.turn_index,
                    content=turn.content,
                    tool_calls=tuple(
                        ToolCall(call_id=c.call_id, name=c.name, arguments=c.arguments)
                        for c in turn.calls
                    ),
                )
            )
    else:
        last = max(i for i, row in enumerate(rollout.transcript) if row.role == "assistant")
        for position, row in enumerate(rollout.transcript):
            locator = f"{source}:transcript:{position}"
            if row.role == "user":
                events.append(
                    _event(len(events), "user_message", "user", "context", row.content, locator)
                )
            elif row.role == "tool":
                events.append(
                    _event(len(events), "harness_note", "world", "scored", row.content, locator)
                )
            else:
                kind = "final_answer" if position == last else "assistant_message"
                events.append(_event(len(events), kind, "actor", "scored", row.content, locator))
                turns.append(ActorTurn(turn_index=len(turns), content=row.content))
    evidence = EpisodeEvidence(
        episode_id=f"rollout-{rollout.rollout_id}",
        task_sha256=task_sha256,
        world_sha256=world_sha256,
        complete=True,
        events=tuple(events),
    )
    trajectory = ActorTrajectory(kind=ROLLOUT_TRAJECTORY, turns=tuple(turns))
    sampled = SampledTokens(binding=source, parts=tuple(parts)) if parts else None
    return evidence, trajectory, sampled

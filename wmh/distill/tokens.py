"""Token-span records tying harbor trials to the student tokens they sampled.

During distillation rollouts each harbor trial runs the pi agent against the
Tinker student, and the distill agent's `TokenRecorder` appends every sampled
`TokenSpan` to a per-trial JSONL sink named after the harbor trial
(`{sink_dir}/{trial_name}.jsonl`, one span JSON per line). This module reads
those sinks back and joins them with the scorer's reward cells into
`TrialRecord`s, the unit the datum builder consumes. The trial's WMH run
trace (`wmh-run.json`, written by the agent bridge into harbor's per-trial
agent logs dir) supplies the stop reason when it exists.

`reconstruct_conversation` replays the canonical (tokenizer-independent)
conversation of one episode from the same spans: the per-span message deltas
concatenated, with each span's own sampled tokens parsed back into the
assistant turn they represent. That message list plus the recorded tool
schemas is what a cross-tokenizer teacher re-renders with its own chat
template, so multi-turn agentic rollouts (not just single-turn math) can be
scored token-for-token against a different tokenizer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from llm_waterfall.types import ChatMessage, ChatTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.distill.rendering import ParsedAssistantMessage
from wmh.harness.scoring import ScoreCell
from wmh.providers.tinker import TokenSpan

logger = logging.getLogger(__name__)

WMH_RUN_TRACE_FILENAME = "wmh-run.json"

StopReasonReader = Callable[[Path], str | None]
"""Reads one trial's stop reason from its artifact dir; None when unknown."""


class TrialRecord(BaseModel):
    """One harbor trial joined with the exact token spans it sampled.

    `spans` is the tokens-in-tokens-out training evidence (empty when the
    trial died before its first successful student completion); `reward` and
    `passed` come from the harbor verifier via the scorer's cells and are
    metrics/gating signal, never part of the distillation loss.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    trial_name: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    passed: bool
    spans: list[TokenSpan] = Field(default_factory=list)
    stop_reason: str | None = None
    """The WMH run trace's stop reason; None when no trace was readable."""

    artifact_dir: str
    """The harbor trial directory holding this trial's raw evidence."""


def load_trial_spans(sink_dir: Path, trial_name: str) -> list[TokenSpan]:
    """Read the token spans one trial's recorder sink captured.

    The sink is the JSONL file the distill agent's `TokenRecorder` writes:
    `{trial_name}.jsonl` under `sink_dir`, one `TokenSpan` JSON per line,
    flushed after every successful completion.

    Args:
        sink_dir: The per-trial token sink directory for one rollout batch.
        trial_name: The harbor trial name the sink file is keyed by.

    Returns:
        The recorded spans in call order. A missing sink file yields an empty
        list: the trial made no successful student completion (it died before
        the first sample), which callers count in their stats rather than
        raise on.

    Raises:
        ValueError: If a line is not a valid `TokenSpan`, or the call_index
            sequence is not exactly 0..n-1 (the sink was appended by more than
            one recorder); delete the step's token sink directory and re-run
            the step to rebuild it.
    """
    sink_path = sink_dir / f"{trial_name}.jsonl"
    try:
        text = sink_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    spans: list[TokenSpan] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            spans.append(TokenSpan.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(
                f"invalid token span on line {line_number} of {sink_path}: {exc}; "
                "the sink is corrupt, so delete this step's token sink directory "
                "and re-run the step"
            ) from exc
    observed = [span.call_index for span in spans]
    if observed != list(range(len(spans))):
        raise ValueError(
            f"token sink {sink_path} has call_index sequence {observed}, expected "
            f"0..{len(spans) - 1}; more than one recorder appended to it (e.g. a "
            "re-run trial reused the sink), so delete this step's token sink "
            "directory and re-run the step"
        )
    return spans


class SampledTurnParser(Protocol):
    """The one thing conversation replay needs from a renderer.

    `wmh.distill.rendering.CookbookChatRendering` (and anything satisfying
    `ChatRendering`) satisfies this structurally; the parse is what turns a
    span's raw sampled ids back into a structured assistant turn.
    """

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        """Parse sampled token ids into an assistant message (text plus tool calls)."""
        ...


class ConversationReplay(BaseModel):
    """One episode's canonical conversation, replayed from its token spans.

    This is the cross-tokenizer hand-off: a teacher with a different tokenizer
    re-renders `messages` (and `tools`) with its own chat template instead of
    trying to reuse the student's token ids, and `assistant_index_by_span`
    pairs each sampled span with the assistant message its tokens produced so
    the chunk planner knows which rendered region a span must align to.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    """The full conversation in order: the spans' message deltas interleaved
    with the assistant turn each span sampled."""

    tools: list[ChatTool] | None = None
    """The tool schemas the episode was sampled with; None when there were none."""

    assistant_index_by_span: dict[int, int]
    """Span position (0-based, equal to `call_index` for a sink that passed
    `load_trial_spans`) to the index in `messages` of the assistant message
    that span's sampled tokens produced."""


def reconstruct_conversation(
    spans: Sequence[TokenSpan], rendering: SampledTurnParser
) -> ConversationReplay | None:
    """Replay one episode's canonical conversation from its recorded spans.

    Walks the spans in call order, appending each span's `delta_messages` (the
    messages that call added) and then the assistant message that span's
    sampled ids parse to, so the result is the conversation the agent actually
    held: system, user, assistant (with tool calls), tool result, assistant,
    and so on.

    Args:
        spans: One trial's spans in call order, as `load_trial_spans` returns
            them.
        rendering: The student's rendering, used only to parse each span's
            sampled ids back into a structured assistant turn.

    Returns:
        The replay, or None when this episode cannot be replayed honestly:
        no spans at all, a span recorded before message capture existed (an old
        sink), a span whose prompt was reused or fully re-rendered rather than
        extended (`TokenSpan.delta_messages` is None for both), or spans that
        disagree about their tool schemas. Callers must degrade (skip the
        cross-tokenizer path for the trial) rather than score a conversation
        that differs from the sampled one; the reason is logged.
    """
    if not spans:
        logger.info("no token spans to reconstruct a conversation from; nothing was sampled")
        return None
    tools = spans[0].tools
    messages: list[ChatMessage] = []
    assistant_index_by_span: dict[int, int] = {}
    for index, span in enumerate(spans):
        if span.delta_messages is None:
            logger.warning(
                "cannot reconstruct the conversation: span %d (call_index %d) carries no "
                "delta_messages, so the canonical messages of that call are unknown; the "
                "sink predates message capture or that call re-rendered/reused its prompt "
                "instead of extending the previous one. Re-run the rollout step to capture "
                "messages, and skip the cross-tokenizer teacher for this trial",
                index,
                span.call_index,
            )
            return None
        if span.tools != tools:
            logger.warning(
                "cannot reconstruct the conversation: span %d (call_index %d) was rendered "
                "with different tool schemas than span 0, so one tool list cannot describe "
                "the episode; split the spans at the schema change before reconstructing",
                index,
                span.call_index,
            )
            return None
        messages.extend(span.delta_messages)
        parsed = rendering.parse_response(span.sampled_token_ids)
        assistant_index_by_span[index] = len(messages)
        # Same shape TinkerChatProvider.complete_chat handed the agent, so the
        # replayed turn is the turn the conversation actually continued from.
        messages.append(
            ChatMessage(
                role="assistant",
                content=parsed.text or None,
                tool_calls=parsed.tool_calls or None,
            )
        )
    return ConversationReplay(
        messages=messages, tools=tools, assistant_index_by_span=assistant_index_by_span
    )


def read_trial_stop_reason(artifact_dir: Path) -> str | None:
    """The WMH run trace's stop reason for one trial, when a trace exists.

    The agent bridge writes `wmh-run.json` into harbor's per-trial agent logs
    dir (`{trial_dir}/agent/` for single-step tasks); both full `RunResult`
    dumps and partial cancellation traces carry a string `stop_reason`. The
    trial-dir root is also checked for robustness against layout drift.

    Args:
        artifact_dir: The harbor trial directory (one cell's `artifact_dir`).

    Returns:
        The stop reason string, or None when no trace is present or readable.
        Assembly must not fail on the trials that died hardest; a missing
        stop reason is itself the signal.
    """
    candidates = (
        artifact_dir / "agent" / WMH_RUN_TRACE_FILENAME,
        artifact_dir / WMH_RUN_TRACE_FILENAME,
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable WMH run trace at %s; recording no stop reason", candidate)
            return None
        stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
        return stop_reason if isinstance(stop_reason, str) else None
    return None


def assemble_trial_records(
    cells: Sequence[ScoreCell],
    sink_dir: Path,
    *,
    read_stop_reason: StopReasonReader | None = None,
) -> list[TrialRecord]:
    """Join scorer cells with their token sinks and run traces.

    Args:
        cells: The scored trial cells (one per task x attempt); each cell's
            `artifact_dir` must be the harbor trial directory, whose basename
            is the trial name the token sinks are keyed by.
        sink_dir: The per-trial token sink directory for this rollout batch.
        read_stop_reason: Reads one trial's stop reason from its artifact dir;
            defaults to `read_trial_stop_reason`.

    Returns:
        One `TrialRecord` per cell, in the cells' order. A trial without a
        sink file gets empty spans, never dropped: its reward is still real
        batch signal and callers count span-less trials explicitly.

    Raises:
        ValueError: If a cell carries no artifact dir (the trial name cannot
            be derived), or a sink file is corrupt (see `load_trial_spans`).
    """
    reader = read_stop_reason if read_stop_reason is not None else read_trial_stop_reason
    records: list[TrialRecord] = []
    for cell in cells:
        artifact_dir = Path(cell.artifact_dir)
        trial_name = artifact_dir.name
        if not cell.artifact_dir or not trial_name:
            raise ValueError(
                f"score cell for task {cell.task_id!r} attempt {cell.attempt} carries no "
                "artifact dir, so its trial name (the token-sink key) cannot be derived; "
                "collect rollouts through a scorer that records per-trial directories "
                "(HarborScorer does)"
            )
        records.append(
            TrialRecord(
                task_id=cell.task_id,
                attempt=cell.attempt,
                trial_name=trial_name,
                reward=cell.reward,
                passed=cell.passed,
                spans=load_trial_spans(sink_dir, trial_name),
                stop_reason=reader(artifact_dir),
                artifact_dir=cell.artifact_dir,
            )
        )
    return records

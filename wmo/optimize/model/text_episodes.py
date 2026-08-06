"""Teacher episodes as TEXT to cross-entropy datums, for any provider.

The on-policy path needs the student's own sampled token ids, so its teacher
must live on Tinker (only `compute_logprobs` can score those exact tokens).
Off-policy hard-target training needs none of that: the teacher's WORDS are
the supervision, and the student learns them under its own chat template. That
makes any provider a usable teacher, including ones that expose no logprobs at
all (Fireworks, OpenRouter, Azure, or a recorded corpus with no live provider
behind it).

This module is that bridge, in one direction and one shape:

    TeacherEpisode (text, provider-agnostic)
        -> keep-filter
        -> re-tokenize under the STUDENT's renderer
        -> TrialRecord with synthetic spans
        -> WarmupTrialsManifest

The last step is the integration: the existing supervised phase already loads
a manifest through `warmup.trajectories_from` and trains cross-entropy on it
(`wmo.optimize.model.loop._load_warmup_trials`), so a text corpus reaches training
through the tested path rather than a parallel one. Nothing in the loop
changes. When the first-class `[offpolicy]` mode lands (#268), this stays its
text-input front end: it produces the same `TrialRecord`s that executor
consumes.

## One datum per TURN, deliberately (read this before reading a fragmentation rate)

A sampled episode merges into one datum because each prompt extends the last
verbatim. Re-encoded text does NOT, and the reason is worth stating exactly,
because `build_datums` will report a ~90% fragmentation rate here and that
number means something different than it does on the sampled path.

Measured on a real Qwen3.5 transcript: a generation prompt ends
`<|im_start|>assistant\\n<think>\\n` (the reasoning template PRIMES a thinking
block), while the same turn rendered later as history is
`<|im_start|>assistant\\nHi! ...` with no thinking block at all. The two token
streams diverge by exactly those two primed tokens.

So there are two possible constructions, and they trade different things:

- **Per-turn prompts (what this module does).** Each turn trains under the
  prompt shape the student will actually be given at serving time, primed
  thinking block included, because that is what `build_generation_prompt`
  emits for every real request. The cost is fragmentation: one datum per turn,
  so shared context is prefilled once per turn instead of once per episode.
- **Chained history prompts.** One datum per episode, but every turn boundary
  after the first trains on a prompt WITHOUT the primed block the student is
  always served. That is a train/serve mismatch bought with a cost saving.

The first is correct, and it is also what ordinary multi-turn SFT does (one
example per assistant turn). Cross-entropy has no teacher-scoring bill at all
here, so the fragmentation costs student prefill only. Do not "fix" the
fragmentation rate by chaining: on THIS path a high rate is expected, and on
the sampled path it still means what it always meant.

## Why the spans are honest about themselves

A recorded transcript has token ids only after somebody tokenizes it, and the
tokenizer that matters is the STUDENT's: training targets must be ids the
student can emit. So each assistant turn is re-encoded with the student's
renderer (`ChatRendering.render_assistant_turn`) under the prompt that
preceded it, which is what makes the tokens trainable at all.

What re-encoding cannot invent is logprobs. No sampler produced these tokens,
so `TokenSpan.sampled_logprobs` carries zeros with
`logprobs_are_placeholders=True`, and `attach_advantages` refuses any datum
built from them. Cross-entropy never sends logprobs over the wire
(`to_tinker_sft_datums` sends target_tokens and weights), so the supervised
path is unaffected. That asymmetry is enforced in code, not documented and
hoped for.

## What the caller owns

Provenance. `TeacherEpisode.teacher_model` and `.source` are recorded and flow
into the manifest, because "distilled from X" is only claimable when X's
transcripts actually trained the student, and a run dir has to be able to
prove which teacher it consumed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.optimize.model.store import WarmupTrialsManifest
from wmo.optimize.model.tokens import TrialRecord
from wmo.providers.tinker import TokenSpan
from wmo.runtime.harness.runtime import StopReason
from wmo.utils.waterfall.types import ChatMessage, ChatTool

logger = logging.getLogger(__name__)

PLACEHOLDER_LOGPROB = 0.0
"""The stand-in logprob for a re-encoded token (never read by cross_entropy)."""


class TurnRendering(Protocol):
    """The two rendering operations turning teacher text into training targets.

    Deliberately narrower than `wmo.optimize.model.rendering.ChatRendering`: this
    bridge never samples, decodes, or parses, so it asks for nothing it does
    not use. `CookbookChatRendering` satisfies it structurally, which is how
    `build_offline_rendering` (no Tinker client) plugs straight in.
    """

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        """Render messages (and tool schemas) into prompt token ids."""
        ...

    def render_assistant_turn(
        self, messages: list[ChatMessage], index: int, tools: list[ChatTool] | None = None
    ) -> list[int]:
        """Render an already-written assistant turn as its sampled-token equivalent."""
        ...


class TeacherEpisode(BaseModel):
    """One teacher episode as provider-agnostic TEXT, plus its outcome.

    The transcript is the whole conversation in OpenAI chat shape (system,
    user, assistant, tool turns), exactly as any provider or benchmark harness
    records it. Which turns become training targets is derived here: every
    assistant turn, and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    """The benchmark task this episode ran, for provenance and per-task rows."""

    attempt: int = Field(default=1, ge=1)
    """1-based attempt index within the task (multiple rollouts per task)."""

    messages: list[ChatMessage]
    """The full recorded transcript, in order."""

    tools: list[ChatTool] = Field(default_factory=list)
    """Tool schemas the episode ran with; rendered into the prompt prefix."""

    reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """The verifier's reward for the episode (the benchmark's own scale)."""

    passed: bool
    """Whether the episode met the benchmark's success bar (the keep filter)."""

    teacher_model: str = Field(min_length=1)
    """The model that produced the assistant turns, e.g. 'kimi-k3'."""

    source: str = Field(min_length=1)
    """Where the transcript came from, e.g. 'fireworks' or 'tau2:results.json'."""

    stop_reason: str | None = None
    """The episode's recorded stop reason, mapped to wmo's taxonomy when known."""

    @model_validator(mode="after")
    def _check_has_assistant_turn(self) -> TeacherEpisode:
        """Reject a transcript with nothing to learn from.

        Returns:
            This episode, when at least one assistant turn exists.

        Raises:
            ValueError: If no assistant turn is present, or the first message
                is one (an assistant turn with no prompt before it has no
                context to be trained against).
        """
        roles = [message.role for message in self.messages]
        if "assistant" not in roles:
            raise ValueError(
                f"teacher episode {self.task_id!r} attempt {self.attempt} has no assistant "
                "turn, so it carries no supervision; drop it before building datums"
            )
        if roles[0] == "assistant":
            raise ValueError(
                f"teacher episode {self.task_id!r} attempt {self.attempt} opens on an "
                "assistant turn; a trainable turn needs the prompt that preceded it"
            )
        return self

    @property
    def trial_name(self) -> str:
        """The record name this episode assembles under (filesystem-safe)."""
        return f"{self.task_id.replace('/', '-')}-a{self.attempt:02d}"


def episode_spans(episode: TeacherEpisode, rendering: TurnRendering) -> list[TokenSpan]:
    """Re-encode one text episode's assistant turns as synthetic token spans.

    Walks the transcript in order and, for every assistant turn, records the
    prompt that preceded it plus that turn's own rendered tokens. Successive
    spans therefore extend one another as verbatim prefixes, which is what
    `build_datums` merges a whole episode into a single datum on.

    Args:
        episode: The text episode to re-encode.
        rendering: The STUDENT's rendering (its tokenizer and chat template);
            training targets must be ids the student can emit.

    Returns:
        One span per assistant turn, `call_index` contiguous from 0, each
        flagged `logprobs_are_placeholders`.

    Raises:
        ValueError: If a turn renders to non-text chunks (see the rendering).
    """
    tools = episode.tools or None
    spans: list[TokenSpan] = []
    for index, message in enumerate(episode.messages):
        if message.role != "assistant":
            continue
        prompt_ids = rendering.build_generation_prompt(episode.messages[:index], tools)
        sampled_ids = rendering.render_assistant_turn(episode.messages, index, tools)
        if not sampled_ids:
            # An assistant turn that renders to nothing (empty content, no tool
            # calls) would contribute a mask-1.0-free fragment the datum builder
            # skips anyway; dropping it here keeps call_index contiguous.
            logger.debug("skipping empty assistant turn %d of %s", index, episode.trial_name)
            continue
        spans.append(
            TokenSpan(
                call_index=len(spans),
                prompt_token_ids=prompt_ids,
                sampled_token_ids=sampled_ids,
                sampled_logprobs=[PLACEHOLDER_LOGPROB] * len(sampled_ids),
                logprobs_are_placeholders=True,
                tools=list(episode.tools),
            )
        )
    return spans


def episodes_to_trial_records(
    episodes: Iterable[TeacherEpisode], rendering: TurnRendering
) -> list[TrialRecord]:
    """Re-encode text episodes into the records the training path consumes.

    Args:
        episodes: The text episodes, in the order they should train.
        rendering: The STUDENT's rendering.

    Returns:
        One `TrialRecord` per episode, in input order. An episode whose turns
        all render empty yields a record with no spans, which the datum builder
        skips and the batch stats count as `empty_span_trials` (never silently
        dropped, so the keep-filter denominator stays honest).
    """
    records: list[TrialRecord] = []
    for episode in episodes:
        spans = episode_spans(episode, rendering)
        records.append(
            TrialRecord(
                task_id=episode.task_id,
                attempt=episode.attempt,
                trial_name=episode.trial_name,
                reward=episode.reward,
                passed=episode.passed,
                spans=spans,
                stop_reason=episode.stop_reason,
                infra_failed=False,
                tests=None,
                # No disk artifact: these episodes came from text, not a runner.
                # Nothing downstream reads this on the training path.
                artifact_dir="",
            )
        )
    return records


def text_warmup_manifest(
    episodes: Sequence[TeacherEpisode], rendering: TurnRendering, *, teacher_model: str
) -> WarmupTrialsManifest:
    """Assemble text episodes into a manifest the supervised phase can load.

    The manifest is what `warmup.trajectories_from` reads, and the loading run
    checks `teacher_model` against its own `teacher.checkpoint or
    teacher.model` string, so the value passed here must be exactly what that
    run's config names. Every episode must agree with it: a manifest mixing
    teachers cannot support a "distilled from X" claim.

    Args:
        episodes: The text episodes (unfiltered; the loading run applies
            `warmup.keep`, matching the collect path's own contract).
        rendering: The STUDENT's rendering.
        teacher_model: The teacher identity to record.

    Returns:
        The manifest, ready for `DistillRunStore.write_warmup_trials`.

    Raises:
        ValueError: If `episodes` is empty or any episode names a different
            teacher.
    """
    if not episodes:
        raise ValueError("cannot build a warmup manifest from zero teacher episodes")
    mismatched = sorted({e.teacher_model for e in episodes} - {teacher_model})
    if mismatched:
        raise ValueError(
            f"episodes name teacher(s) {mismatched} but the manifest records "
            f"{teacher_model!r}; a manifest mixing teachers cannot support a "
            "'distilled from X' claim, so split the corpus per teacher"
        )
    return WarmupTrialsManifest(
        teacher_model=teacher_model,
        records=episodes_to_trial_records(episodes, rendering),
    )


TAU2_TERMINATION_STOP_REASONS = {
    "user_stop": StopReason.SUBMITTED.value,
    "agent_stop": StopReason.SUBMITTED.value,
    "max_steps": StopReason.MAX_TURNS.value,
    "timeout": StopReason.BUDGET.value,
    "too_many_errors": StopReason.UNPARSED_TOOL_CALL.value,
    "agent_error": StopReason.ERROR.value,
    "user_error": StopReason.ERROR.value,
    "unexpected_error": StopReason.ERROR.value,
}
"""tau2 termination reasons mapped onto wmo's stop-reason taxonomy.

Deliberately omits `infrastructure_error` (no verifier evidence, so such an
episode is not training material) and `context_window_exceeded` (the datum
builder's own overflow drop keys on its stop reason; see
`wmo.optimize.model.tau2`)."""

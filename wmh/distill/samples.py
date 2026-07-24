"""Human-readable sample episode rollouts: render, select, and assemble.

Metrics rows say how a batch scored; they never show WHAT the model actually
saw and produced. This module renders a small sample of each batch's episodes
into plain text for humans: the exact episode token stream decoded WITH the
chat template's special tokens (`<|im_start|>`, think blocks, tool-call
markers), because that framing is precisely what a reader needs to judge
whether the harness, renderer, and policy line up.

`render_episode_text` renders one trial: a prefix-clean episode decodes in a
single pass over the final span's prompt plus its sampled tokens (the full
conversation as the model saw it, template included), while an episode whose
history was edited mid-run decodes per fragment with a `FRAGMENT_BREAK`
marker line in between; over-long bodies keep head and tail
(`truncate_middle`). `sample_rollouts` picks and renders the first N
span-bearing trials of a batch, and `samples_markdown` joins them into the
document `DistillRunStore.write_samples` persists. The loop calls these after
every training batch, the warmup collection, and each eval batch
(`train.log_sample_rollouts` sets N; 0 disables).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan

MAX_EPISODE_CHARS = 40_000
"""Rendered episode bodies longer than this keep head and tail; the middle is elided."""

FRAGMENT_BREAK = (
    "----- FRAGMENT BREAK: the next call's prompt did not extend the episode "
    "tokens, so the context re-rendered from scratch -----"
)
"""Marker line between the per-fragment decodes of a non-prefix-clean episode."""

SAMPLE_SEPARATOR = "\n\n" + "=" * 78 + "\n\n"
"""Separator between samples in one batch's samples document."""


class SpecialsDecoder(Protocol):
    """The one rendering call episode logging needs; `ChatRendering` satisfies it."""

    def decode_with_specials(self, token_ids: list[int]) -> str:
        """Decode token ids to text KEEPING special tokens."""
        ...


class SampleRollout(BaseModel):
    """One rendered sample episode (a row of the tracker's samples table)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_name: str = Field(min_length=1)
    reward: float
    text: str
    """The full `render_episode_text` output: header plus decoded episode."""


def _is_prefix(prefix: list[int], sequence: list[int]) -> bool:
    """Whether `prefix` equals the start of `sequence` (the datum builder's test)."""
    return len(prefix) <= len(sequence) and sequence[: len(prefix)] == prefix


def _fragment_token_runs(spans: Sequence[TokenSpan]) -> list[list[int]]:
    """The episode's token stream, split where a prompt broke the prefix property.

    Mirrors the datum builder's merge (`wmh.distill.data._merge_trial_spans`):
    spans are walked in call order and each prompt that extends the
    accumulated tokens verbatim contributes only its delta, so a prefix-clean
    episode comes back as ONE run equal to the final span's prompt plus its
    sampled tokens (the whole conversation, decodable in a single pass). A
    non-extending prompt closes the current run and starts a fresh one.
    """
    runs: list[list[int]] = []
    tokens: list[int] = []
    for span in sorted(spans, key=lambda item: item.call_index):
        prompt = list(span.prompt_token_ids)
        if tokens and _is_prefix(tokens, prompt):
            delta = prompt[len(tokens) :]
        else:
            if tokens:
                runs.append(tokens)
                tokens = []
            delta = prompt
        tokens.extend(delta)
        tokens.extend(span.sampled_token_ids)
    if tokens:
        runs.append(tokens)
    return runs


def truncate_middle(text: str, limit: int = MAX_EPISODE_CHARS) -> str:
    """Keep the head and tail of an over-long text, eliding the middle.

    The head (system prompt and task setup) and the tail (the episode's final
    turns and outcome) are what a reader checks first, so those survive and
    the middle is replaced by a marker naming how much was cut.

    Args:
        text: The rendered episode text.
        limit: Max characters kept FROM `text` (the marker itself is extra).

    Returns:
        `text` unchanged when it fits, else head plus marker plus tail.

    Raises:
        ValueError: If `limit` is not positive.
    """
    if limit < 1:
        raise ValueError(f"truncation limit must be >= 1, got {limit}")
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - head - tail
    marker = f"\n\n[... {omitted} chars elided from the middle of the episode ...]\n\n"
    return text[:head] + marker + text[len(text) - tail :]


def render_episode_text(record: TrialRecord, renderer: SpecialsDecoder) -> str:
    """One trial's episode as readable text, chat-template framing included.

    The header names the trial and its outcome (reward, passed, stop reason,
    span/fragment/token counts); the body is the episode's exact token stream
    decoded WITH special tokens. A prefix-clean episode decodes in one pass
    (the final span's prompt plus its sampled tokens IS the full conversation
    including every template marker); a fragmented episode decodes per
    fragment with `FRAGMENT_BREAK` lines in between. Bodies beyond
    `MAX_EPISODE_CHARS` keep head and tail (`truncate_middle`).

    Args:
        record: The scored trial with its recorded spans.
        renderer: Supplies the specials-preserving decode
            (`ChatRendering.decode_with_specials`).

    Returns:
        The header plus decoded episode text, newline-terminated.
    """
    runs = _fragment_token_runs(record.spans)
    token_count = sum(len(run) for run in runs)
    header = (
        f"### trial {record.trial_name}\n"
        f"reward: {record.reward:g} | passed: {record.passed} | "
        f"stop reason: {record.stop_reason or 'unknown'} | "
        f"spans: {len(record.spans)} | fragments: {len(runs)} | "
        f"episode tokens: {token_count}\n"
    )
    if not runs:
        return header + "\n(no token spans were recorded for this trial)\n"
    break_line = "\n" + FRAGMENT_BREAK + "\n"
    body = break_line.join(renderer.decode_with_specials(run) for run in runs)
    return header + "\n" + truncate_middle(body) + "\n"


def sample_rollouts(
    records: Sequence[TrialRecord], renderer: SpecialsDecoder, limit: int
) -> list[SampleRollout]:
    """Render the first `limit` span-bearing trials of a batch, in record order.

    Record order is the scorer's deterministic cell order (task x attempt),
    so the same batch always samples the same trials. Span-less trials carry
    nothing to read and are skipped.

    Args:
        records: One batch's trial records.
        renderer: Supplies the specials-preserving decode.
        limit: Max samples to render; 0 renders nothing.

    Returns:
        At most `limit` rendered samples.
    """
    samples: list[SampleRollout] = []
    for record in records:
        if len(samples) >= limit:
            break
        if not record.spans:
            continue
        samples.append(
            SampleRollout(
                trial_name=record.trial_name,
                reward=record.reward,
                text=render_episode_text(record, renderer),
            )
        )
    return samples


def samples_markdown(samples: Sequence[SampleRollout]) -> str:
    """One batch's samples as a single document (the `samples/<name>.md` payload)."""
    return SAMPLE_SEPARATOR.join(sample.text for sample in samples)

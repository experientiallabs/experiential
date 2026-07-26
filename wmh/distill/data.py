"""Prefix-merge datum builder: harbor trials to advantage-weighted training data.

`build_datums` turns each trial's recorded `TokenSpan`s into `TrainDatum`s
with the tinker-cookbook's `trajectory_to_data` merge semantics: as long as
each call's prompt extends the accumulated episode tokens (previous prompt
plus previous sampled tokens) verbatim as a list prefix, the whole episode is
ONE datum, so every context token is prefilled exactly once. A call whose
prompt is not such an extension starts a new datum (a fragment), which is
correct but re-prefills shared context; the fragmentation rate is therefore a
first-class cost metric, not a curiosity.

Alignment is sacred: episodes are never truncated (that would desynchronize
tokens from their sampled logprobs), only dropped whole and counted, both for
context overflow (any call measured over `rollout.context_budget_tokens`, or
a run trace marked with the overflow stop reason) and for merged lengths over
`train.max_datum_tokens`.

`attach_advantages` fills the reverse-KL advantages from teacher logprobs,
and `to_tinker_datums` converts to real `tinker.Datum`s in the shifted
next-token layout the cookbook uses (one wire format for both
advantage-weighted losses, `importance_sampling` and `ppo`);
`to_tinker_sft_datums` is the
cross_entropy sibling for the supervised warmup phase (teacher-sampled
trajectories need no advantages, only the loss-mask weights). For the
`topk_ce` loss, `build_topk_ce_datums` turns teacher top-k candidate rows
into rank-aligned replica datums (see its docstring for why replication is
per RANK, not per position) and `to_tinker_topk_ce_datums` is its one-call
wire conversion through the same pinned cross_entropy keyset. The tinker SDK
is an optional extra and is imported lazily inside those converters only,
mirroring the provider's contract; everything else here runs without it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.distill.config import DistillConfig
from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan

if TYPE_CHECKING:
    import tinker

logger = logging.getLogger(__name__)

MISSING_TINKER_EXTRA = (
    "the tinker SDK is not installed; run `uv sync --extra distill` to convert "
    "training datums for a real Tinker training client"
)

TopkCandidates = list[tuple[int, float]]
"""One position's teacher top-k: (token_id, logprob) pairs, best first."""

CONTEXT_OVERFLOW_STOP_REASON = "context_overflow"
"""A run-trace stop reason marking an episode that outgrew the context budget.

Overflowed episodes are dropped from training: their final turns were shaped
by the budget cutoff rather than the policy, and their cost profile is exactly
the blowup the prefix property exists to prevent. The enforcement itself is
measured from the recorded spans (any call whose prompt plus sampled tokens
exceeds `rollout.context_budget_tokens`); the stop-reason marker is honored
additionally for runtimes that end episodes at the budget themselves.
"""


class TrainDatum(BaseModel):
    """One merged training sequence in the UNshifted layout, fully aligned.

    `model_input_tokens` is the complete episode-fragment token sequence:
    context deltas (prompt tokens the agent appended between completions)
    interleaved with sampled spans, in order. The three per-token lists are
    aligned one to one with it; `to_tinker_datums` produces the shifted
    next-token layout, so this model stays trivially inspectable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_name: str = Field(min_length=1)
    """The harbor trial this datum came from, for loud per-datum diagnostics."""

    fragment_index: int = Field(ge=0)
    """0-based index of this datum within its episode; > 0 means a prefix break."""

    model_input_tokens: list[int]
    """The full unshifted token sequence (context deltas plus sampled spans)."""

    loss_mask: list[float]
    """Per-token loss mask: 1.0 on sampled tokens, 0.0 on context tokens."""

    sampled_logprobs: list[float]
    """Sampler-assigned logprob per token; 0.0 at mask-0 (context) positions."""

    advantages: list[float] = Field(default_factory=list)
    """Per-token advantages; empty until `attach_advantages` fills them."""

    target_tokens: list[int] = Field(default_factory=list)
    """Per-position training targets when they differ from the sequence's own
    tokens; empty for ordinary datums (targets are the next tokens of
    `model_input_tokens`). Set only on topk-CE replicas, where each loss
    position's target is a teacher candidate rather than the realized token
    (`build_topk_ce_datums`); non-loss positions keep the realized token."""

    target_weights: list[float] = Field(default_factory=list)
    """Per-position cross_entropy weights replacing the loss mask on the wire;
    empty for ordinary datums (the loss mask is the weight). Set only on
    topk-CE replicas: the renormalized teacher probability of this replica's
    candidate at each loss position, 0.0 everywhere else."""

    @model_validator(mode="after")
    def _check_alignment(self) -> TrainDatum:
        """Reject any misalignment between the token sequence and its per-token lists."""
        n = len(self.model_input_tokens)
        if len(self.loss_mask) != n or len(self.sampled_logprobs) != n:
            raise ValueError(
                f"loss_mask length {len(self.loss_mask)} and sampled_logprobs length "
                f"{len(self.sampled_logprobs)} must both match model_input_tokens "
                f"length {n}"
            )
        if any(value not in (0.0, 1.0) for value in self.loss_mask):
            raise ValueError("loss_mask values must be exactly 0.0 or 1.0")
        if self.advantages and len(self.advantages) != n:
            raise ValueError(
                f"advantages length {len(self.advantages)} must match "
                f"model_input_tokens length {n} (or be empty before attach_advantages)"
            )
        if bool(self.target_tokens) != bool(self.target_weights):
            raise ValueError(
                "target_tokens and target_weights must be set together (a topk-CE "
                "replica carries both) or both left empty"
            )
        if self.target_tokens and (len(self.target_tokens) != n or len(self.target_weights) != n):
            raise ValueError(
                f"target_tokens length {len(self.target_tokens)} and target_weights "
                f"length {len(self.target_weights)} must both match "
                f"model_input_tokens length {n}"
            )
        if self.target_weights and any(
            weight != 0.0 and mask != 1.0
            for weight, mask in zip(self.target_weights, self.loss_mask, strict=True)
        ):
            raise ValueError(
                "target_weights may be nonzero only at loss positions (mask 1.0); a "
                "weighted context position would train on tokens the student never "
                "sampled"
            )
        return self

    @property
    def is_topk_replica(self) -> bool:
        """Whether this datum is a topk-CE replica (candidate targets attached)."""
        return bool(self.target_tokens)

    @property
    def loss_token_count(self) -> int:
        """How many positions carry loss (mask 1.0)."""
        return sum(1 for value in self.loss_mask if value == 1.0)

    def sampled_token_ids(self) -> list[int]:
        """The sampled (mask 1.0) token ids, in sequence order."""
        return [
            token
            for token, mask in zip(self.model_input_tokens, self.loss_mask, strict=True)
            if mask == 1.0
        ]


class DatumStats(BaseModel):
    """Accounting for one `build_datums` call (drops are episodes, not datums)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    datums: int = Field(ge=0)
    fragments: int = Field(ge=0)
    """Datums beyond the first of their episode (each one is a prefix break)."""

    fragmentation_rate: float = Field(ge=0.0, le=1.0)
    """fragments / datums; every fragment re-prefills context, so this is cost."""

    overflow_drops: int = Field(ge=0)
    """Trials dropped for context overflow (measured or stop-reason marked)."""

    overlong_drops: int = Field(ge=0)
    """Episodes dropped whole because a merged datum exceeded max_datum_tokens."""

    loss_tokens: int = Field(ge=0)
    context_tokens: int = Field(ge=0)


class AdvantageStats(BaseModel):
    """Accounting for one `attach_advantages` call.

    Token-level counters cover only the ATTACHED datums: a datum dropped for
    a teacher mismatch is never trained on, so its tokens are not signal and
    contribute to no counter here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    datums: int = Field(ge=0)
    """Datums that came out with advantages attached."""

    mismatch_drops: int = Field(ge=0)
    """Datums dropped because their teacher logprobs did not align."""

    clipped_tokens: int = Field(ge=0)
    """Loss tokens whose raw advantage hit the clip bound (attached datums
    only); always 0 when `train.advantage_clip` is None (clipping off)."""

    loss_tokens: int = Field(ge=0)
    """Loss tokens across the attached datums (the clip-fraction denominator)."""

    context_tokens: int = Field(ge=0)
    """Non-loss (mask 0.0) tokens across the attached datums.

    Reported beside `loss_tokens` so a caller can describe the TRAINED batch's
    token volume without falling back to the pre-drop `DatumStats`, whose
    counts include datums this call dropped.
    """

    advantage_mean: float | None
    """Mean advantage over the attached datums' loss tokens, exactly as
    trained (after any clipping and any centering); None when no loss token
    survived.

    Under the default objective (no clipping, no centering) this is the mean
    teacher-minus-student logprob gap, i.e. the negated reverse KL over the
    trained tokens: the direct read of how far the student still is from the
    teacher, and the number that should shrink as the run works. Under
    `train.center_advantages` it is ~0.0 by construction and says nothing."""

    advantage_std: float | None
    """Population standard deviation over the same loss tokens; None when none."""


def _is_prefix(prefix: list[int], sequence: list[int]) -> bool:
    """Whether `prefix` equals the start of `sequence` (cookbook semantics)."""
    return len(prefix) <= len(sequence) and sequence[: len(prefix)] == prefix


def _merge_trial_spans(trial_name: str, spans: Sequence[TokenSpan]) -> list[TrainDatum]:
    """Merge one trial's spans into datums, breaking on every non-prefix prompt.

    Args:
        trial_name: The trial identity stamped onto each produced datum.
        spans: The trial's recorded spans, in any order (sorted by call_index).

    Returns:
        The episode's datums in order: one when every prompt extended the
        accumulated tokens as a prefix, more when the history was edited. A
        fragment containing no sampled tokens (possible only when a span
        sampled nothing) carries no trainable signal and is skipped.
    """
    datums: list[TrainDatum] = []
    tokens: list[int] = []
    mask: list[float] = []
    logprobs: list[float] = []

    def flush() -> None:
        if not tokens:
            return
        if not any(value == 1.0 for value in mask):
            logger.debug(
                "skipping a fragment of trial %s with no sampled tokens (%d context tokens)",
                trial_name,
                len(tokens),
            )
        else:
            datums.append(
                TrainDatum(
                    trial_name=trial_name,
                    fragment_index=len(datums),
                    model_input_tokens=list(tokens),
                    loss_mask=list(mask),
                    sampled_logprobs=list(logprobs),
                )
            )
        tokens.clear()
        mask.clear()
        logprobs.clear()

    for span in sorted(spans, key=lambda item: item.call_index):
        prompt = span.prompt_token_ids
        if tokens and _is_prefix(tokens, prompt):
            delta = prompt[len(tokens) :]
        else:
            flush()
            delta = prompt
        tokens.extend(delta)
        mask.extend([0.0] * len(delta))
        logprobs.extend([0.0] * len(delta))
        tokens.extend(span.sampled_token_ids)
        mask.extend([1.0] * len(span.sampled_token_ids))
        logprobs.extend(span.sampled_logprobs)
    flush()
    return datums


def build_datums(
    records: Sequence[TrialRecord], cfg: DistillConfig
) -> tuple[list[TrainDatum], DatumStats]:
    """Build training datums from scored trials via the prefix merge.

    Per trial: spans are sorted by call_index and merged into one datum while
    each span's prompt extends the accumulated tokens (previous prompt plus
    previous sampled tokens) exactly as a list prefix; the appended context
    delta gets loss mask 0.0 and sampled tokens get 1.0. A non-prefix span
    starts a new datum (fragment). Two whole-episode drops protect alignment
    and cost, never truncation:

    - context overflow: any recorded call whose prompt plus sampled tokens
      exceeds `cfg.rollout.context_budget_tokens` (measured from the spans),
      or a trial whose stop reason is `CONTEXT_OVERFLOW_STOP_REASON`;
    - episodes where any merged datum exceeds `cfg.train.max_datum_tokens`.

    Trials with no spans contribute nothing here; the rollout collector
    already counts them (`RolloutStats.empty_span_trials`).

    Args:
        records: The scored trials for one training step.
        cfg: The run config; reads `rollout.context_budget_tokens` and
            `train.max_datum_tokens`.

    Returns:
        The kept datums (trial order, then fragment order) and the stats.
    """
    datums: list[TrainDatum] = []
    fragments = 0
    overflow_drops = 0
    overlong_drops = 0
    context_budget = cfg.rollout.context_budget_tokens
    for record in records:
        if record.stop_reason == CONTEXT_OVERFLOW_STOP_REASON:
            overflow_drops += 1
            logger.warning(
                "dropping trial %s from training: the episode overflowed the rollout "
                "context budget (stop reason %r)",
                record.trial_name,
                record.stop_reason,
            )
            continue
        if not record.spans:
            continue
        peak_context = max(
            len(span.prompt_token_ids) + len(span.sampled_token_ids) for span in record.spans
        )
        if peak_context > context_budget:
            overflow_drops += 1
            logger.warning(
                "dropping trial %s from training: its largest call consumed %d tokens, "
                "over rollout.context_budget_tokens = %d (episodes are dropped whole, "
                "never truncated, so tokens stay aligned with their sampled logprobs)",
                record.trial_name,
                peak_context,
                context_budget,
            )
            continue
        episode = _merge_trial_spans(record.trial_name, record.spans)
        longest = max((len(datum.model_input_tokens) for datum in episode), default=0)
        if longest > cfg.train.max_datum_tokens:
            overlong_drops += 1
            logger.warning(
                "dropping trial %s from training: a merged datum has %d tokens, over "
                "train.max_datum_tokens = %d (episodes are dropped whole, never "
                "truncated, so tokens stay aligned with their sampled logprobs)",
                record.trial_name,
                longest,
                cfg.train.max_datum_tokens,
            )
            continue
        datums.extend(episode)
        fragments += max(0, len(episode) - 1)
    loss_tokens = sum(datum.loss_token_count for datum in datums)
    context_tokens = sum(len(datum.model_input_tokens) for datum in datums) - loss_tokens
    stats = DatumStats(
        datums=len(datums),
        fragments=fragments,
        fragmentation_rate=fragments / len(datums) if datums else 0.0,
        overflow_drops=overflow_drops,
        overlong_drops=overlong_drops,
        loss_tokens=loss_tokens,
        context_tokens=context_tokens,
    )
    if stats.fragments:
        logger.warning(
            "%d of %d datum(s) are fragments (fragmentation rate %.2f): the agent "
            "edited its history mid-episode, so shared context is re-prefilled and "
            "teacher scoring costs multiply; check the harness prefix pins",
            stats.fragments,
            stats.datums,
            stats.fragmentation_rate,
        )
    return datums, stats


def attach_advantages(
    datums: Sequence[TrainDatum],
    teacher_logprobs: Sequence[Sequence[float | None]],
    cfg: DistillConfig,
) -> tuple[list[TrainDatum], AdvantageStats]:
    """Fill per-token reverse-KL advantages from teacher logprobs.

    Each loss token gets `teacher_lp - sampled_lp`, bounded to
    `+-train.advantage_clip` when that bound is set (None, the default,
    trains the raw gap and clips nothing, which is what `train.loss = "ppo"`
    expects: the ratio clip inside the loss is the regularizer); context
    (mask 0.0) tokens get 0.0. With `train.center_advantages` the mean over
    ALL loss tokens in the batch is then subtracted from every loss token, so
    the batch-mean advantage is zero; without it (the default) the raw gaps
    ride through and `AdvantageStats.advantage_mean` reads the objective.

    `teacher_logprobs[i]` scores `datums[i].model_input_tokens` in the
    compute_logprobs convention: entry p is the logprob of token p given the
    tokens before it, and entry 0 is None (no context). A datum whose teacher
    entry has the wrong length, or a None at a loss position, is dropped
    loudly and counted; a silently misaligned advantage would train on noise.

    Args:
        datums: Datums from `build_datums` (advantages not yet attached).
        teacher_logprobs: One per-position logprob list per datum, aligned
            one to one with `datums`.
        cfg: The run config; reads `train.advantage_clip` (None = no
            clipping) and `train.center_advantages`.

    Returns:
        New datums with advantages attached (drops removed, order preserved)
        and the stats, including the trained advantage distribution (mean and
        population std over the attached datums' loss tokens) and the kept
        clip/loss token counts the loop derives `clip_fraction` from (no
        token counts as clipped when clipping is off).

    Raises:
        ValueError: If `teacher_logprobs` does not have one entry per datum;
            that is a caller bug, not per-datum evidence, so nothing is
            dropped for it.
    """
    if len(teacher_logprobs) != len(datums):
        raise ValueError(
            f"got {len(teacher_logprobs)} teacher logprob list(s) for {len(datums)} "
            "datum(s); pass exactly one per-position list per datum, in datum order"
        )
    clip = cfg.train.advantage_clip
    kept: list[TrainDatum] = []
    per_datum_advantages: list[list[float]] = []
    mismatch_drops = 0
    clipped_tokens = 0
    for datum, teacher in zip(datums, teacher_logprobs, strict=True):
        n = len(datum.model_input_tokens)
        if len(teacher) != n:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: teacher "
                "returned %d logprob(s) for %d token(s); the teacher must score the "
                "datum's exact token sequence with one entry per position",
                datum.trial_name,
                datum.fragment_index,
                len(teacher),
                n,
            )
            continue
        advantages = [0.0] * n
        datum_clipped = 0
        missing_position: int | None = None
        for position in range(n):
            if datum.loss_mask[position] != 1.0:
                continue
            teacher_lp = teacher[position]
            if teacher_lp is None:
                missing_position = position
                break
            raw = teacher_lp - datum.sampled_logprobs[position]
            if clip is None:
                advantages[position] = raw
                continue
            clipped = min(max(raw, -clip), clip)
            if clipped != raw:
                datum_clipped += 1
            advantages[position] = clipped
        if missing_position is not None:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: teacher "
                "logprob at loss position %d is None; the teacher must return a "
                "logprob for every sampled position",
                datum.trial_name,
                datum.fragment_index,
                missing_position,
            )
            continue
        kept.append(datum)
        per_datum_advantages.append(advantages)
        clipped_tokens += datum_clipped
    if cfg.train.center_advantages:
        loss_count = sum(datum.loss_token_count for datum in kept)
        if loss_count:
            total = sum(
                advantages[position]
                for datum, advantages in zip(kept, per_datum_advantages, strict=True)
                for position in range(len(advantages))
                if datum.loss_mask[position] == 1.0
            )
            mean = total / loss_count
            for datum, advantages in zip(kept, per_datum_advantages, strict=True):
                for position in range(len(advantages)):
                    if datum.loss_mask[position] == 1.0:
                        advantages[position] -= mean
    attached = [
        TrainDatum(
            trial_name=datum.trial_name,
            fragment_index=datum.fragment_index,
            model_input_tokens=datum.model_input_tokens,
            loss_mask=datum.loss_mask,
            sampled_logprobs=datum.sampled_logprobs,
            advantages=advantages,
        )
        for datum, advantages in zip(kept, per_datum_advantages, strict=True)
    ]
    # The advantage distribution actually trained on: loss tokens of the kept
    # datums, after clipping and any centering.
    loss_values = [
        advantages[position]
        for datum, advantages in zip(kept, per_datum_advantages, strict=True)
        for position in range(len(advantages))
        if datum.loss_mask[position] == 1.0
    ]
    advantage_mean: float | None = None
    advantage_std: float | None = None
    if loss_values:
        advantage_mean = sum(loss_values) / len(loss_values)
        variance = sum((value - advantage_mean) ** 2 for value in loss_values) / len(loss_values)
        advantage_std = math.sqrt(variance)
    stats = AdvantageStats(
        datums=len(attached),
        mismatch_drops=mismatch_drops,
        clipped_tokens=clipped_tokens,
        loss_tokens=len(loss_values),
        context_tokens=sum(len(datum.model_input_tokens) for datum in attached) - len(loss_values),
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )
    return attached, stats


class TopkCeStats(BaseModel):
    """Accounting for one `build_topk_ce_datums` call.

    Token counters cover only the KEPT source datums: a source dropped for a
    misaligned or unscoreable top-k row is never trained on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_datums: int = Field(ge=0)
    """Source datums that survived and were replicated."""

    datums: int = Field(ge=0)
    """Replica datums emitted (`source_datums x k`)."""

    mismatch_drops: int = Field(ge=0)
    """Source datums dropped because their top-k row did not align (wrong
    length, or no candidates at a loss position)."""

    loss_tokens: int = Field(ge=0)
    """Loss positions across the kept source datums (per copy, not x k)."""

    context_tokens: int = Field(ge=0)
    """Non-loss positions across the kept source datums (per copy, not x k).

    Same role as `AdvantageStats.context_tokens`: the TRAINED batch's context
    volume, so the loop never has to reach back to the pre-drop `DatumStats`.
    """


def _renormalized_probs(candidates: TopkCandidates) -> list[float]:
    """Softmax the candidates' logprobs over just the top-k support.

    The teacher's top-k logprobs are from the FULL vocabulary distribution,
    so they sum to less than 1; renormalizing over the k candidates makes the
    per-position replica weights a proper distribution (they sum to 1 across
    the k copies), keeping the weighted-CE objective's per-position scale
    independent of how much mass the tail held.
    """
    peak = max(logprob for _, logprob in candidates)
    exps = [math.exp(logprob - peak) for _, logprob in candidates]
    total = sum(exps)
    return [value / total for value in exps]


def build_topk_ce_datums(
    datums: Sequence[TrainDatum],
    topk_rows: Sequence[Sequence[TopkCandidates | None]],
    k: int,
) -> tuple[list[TrainDatum], TopkCeStats]:
    """Build rank-aligned topk-CE replica datums from teacher top-k rows.

    The weighted-CE objective is per POSITION: at each loss position p the
    loss is `sum_j w_j(p) * CE(target=candidate_j(p))` with `w_j(p)` the
    renormalized teacher probability of candidate j at p. Emitting one
    single-target datum per (position, candidate) would compute exactly that
    but explode the datum count (loss_tokens x k datums, each re-prefilling
    the whole sequence). Instead each source datum becomes k RANK-ALIGNED
    replicas sharing the model input: replica j uses, at EVERY loss position,
    the j-th ranked candidate as target with its renormalized probability as
    weight (replica 0 is the teacher's top choice everywhere, and so on).
    Because cross_entropy is a per-position sum of `weight * -logprob(target)`
    with no cross-position coupling, summing the k replicas reproduces the
    identical per-position weighted-CE objective with k forward passes
    instead of k per position; the weights at each loss position sum to 1
    across the replicas.

    Positions where the teacher returned fewer than k candidates pad the
    missing ranks with the realized token at weight 0.0 (no loss, and the
    remaining ranks' weights still renormalize over the AVAILABLE candidates,
    so the position keeps unit total weight). Non-loss positions carry the
    realized next token at weight 0.0 in every replica. Candidates are
    sorted best-first defensively (logprob descending, token id ascending on
    ties) so rank is well defined regardless of backend ordering.

    Alignment failures drop the SOURCE datum loudly and are counted, exactly
    like `attach_advantages`: a wrong-length row, a None at a loss position
    (including an unscoreable loss token at position 0), or an empty
    candidate list at a loss position.

    Args:
        datums: Datums from `build_datums` (advantages not needed).
        topk_rows: One per-position row per datum, aligned one to one with
            `datums`; entry p is the teacher's top-k candidates for token p
            when p is a scoreable loss position and None everywhere else
            (`TeacherClient.score_topk`).
        k: The replica count (`train.topk`); rows may carry at most k
            candidates per position.

    Returns:
        The replica datums (source order, then rank order) and the stats.

    Raises:
        ValueError: If `k < 1`, `topk_rows` does not have one entry per
            datum, or a position carries more than k candidates; those are
            caller bugs, not per-datum evidence, so nothing is dropped.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(topk_rows) != len(datums):
        raise ValueError(
            f"got {len(topk_rows)} top-k row(s) for {len(datums)} datum(s); pass "
            "exactly one per-position row per datum, in datum order"
        )
    replicas: list[TrainDatum] = []
    mismatch_drops = 0
    kept_sources = 0
    loss_tokens = 0
    context_tokens = 0
    for datum, row in zip(datums, topk_rows, strict=True):
        n = len(datum.model_input_tokens)
        if len(row) != n:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: teacher "
                "returned %d top-k entrie(s) for %d token(s); the teacher must "
                "score the datum's exact token sequence with one entry per position",
                datum.trial_name,
                datum.fragment_index,
                len(row),
                n,
            )
            continue
        bad_position = next(
            (
                position
                for position in range(n)
                if datum.loss_mask[position] == 1.0 and not row[position]
            ),
            None,
        )
        if bad_position is not None:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: the teacher "
                "top-k row has no candidates at loss position %d (position 0 can "
                "never be scored; elsewhere the teacher must return candidates for "
                "every sampled position)",
                datum.trial_name,
                datum.fragment_index,
                bad_position,
            )
            continue
        oversized = next((position for position in range(n) if len(row[position] or []) > k), None)
        if oversized is not None:
            raise ValueError(
                f"top-k row for datum (trial {datum.trial_name}, fragment "
                f"{datum.fragment_index}) carries {len(row[oversized] or [])} "
                f"candidates at position {oversized}, more than k = {k}; score with "
                "the same k the replication uses"
            )
        per_position: list[tuple[list[int], list[float]]] = []
        for position in range(n):
            candidates = row[position]
            if datum.loss_mask[position] != 1.0 or not candidates:
                per_position.append(([datum.model_input_tokens[position]] * k, [0.0] * k))
                continue
            ranked = sorted(candidates, key=lambda entry: (-entry[1], entry[0]))
            weights = _renormalized_probs(ranked)
            tokens = [token for token, _ in ranked]
            pad = k - len(ranked)
            per_position.append(
                (
                    tokens + [datum.model_input_tokens[position]] * pad,
                    weights + [0.0] * pad,
                )
            )
        kept_sources += 1
        loss_tokens += datum.loss_token_count
        context_tokens += n - datum.loss_token_count
        for rank in range(k):
            replicas.append(
                TrainDatum(
                    trial_name=datum.trial_name,
                    fragment_index=datum.fragment_index,
                    model_input_tokens=datum.model_input_tokens,
                    loss_mask=datum.loss_mask,
                    sampled_logprobs=datum.sampled_logprobs,
                    target_tokens=[tokens[rank] for tokens, _ in per_position],
                    target_weights=[weights[rank] for _, weights in per_position],
                )
            )
    stats = TopkCeStats(
        source_datums=kept_sources,
        datums=len(replicas),
        mismatch_drops=mismatch_drops,
        loss_tokens=loss_tokens,
        context_tokens=context_tokens,
    )
    return replicas, stats


def to_tinker_datums(train_datums: Sequence[TrainDatum]) -> list[tinker.Datum]:
    """Convert attached datums to real `tinker.Datum`s for the advantage losses.

    One wire format serves both `importance_sampling` and `ppo`: the SDK
    (tinker 0.23.3) lists both in `types.LossFnType`, its
    `forward_backward(data, loss_fn, loss_fn_config)` takes the same
    `Datum`s for either, and the cookbook's RL path submits exactly this
    keyset under both (`tinker_cookbook/rl/train.py:train_step` strips
    "mask" and passes `loss_fn` straight through, and its own docstring
    names `"importance_sampling"` and `"ppo"` as the values). `ppo` differs
    only server-side, in what the loss does with the advantages: it bounds
    the update by clipping the policy ratio (Tinker's default epsilon; this
    code sends no `loss_fn_config`) rather than trusting a bounded advantage.

    Produces the cookbook's shifted next-token layout: model input is every
    token but the last, and target_tokens, logprobs, and advantages are the
    per-token lists with position 0 dropped, so index j scores predicting
    token j+1 from tokens <= j. The loss mask is folded into the advantages
    (0.0 at context positions); the live loss rejects a separate "mask" key.

    Args:
        train_datums: Datums with advantages attached (`attach_advantages`).

    Returns:
        One `tinker.Datum` per input datum, in order, with loss_fn_inputs
        exactly target_tokens (int64), logprobs, and advantages (float32).

    Raises:
        ImportError: If the tinker SDK is not installed (distill extra).
        ValueError: If a datum has no advantages attached or is too short for
            the input/target shift (fewer than 2 tokens).
    """
    try:
        import tinker
    except ImportError as exc:
        raise ImportError(MISSING_TINKER_EXTRA) from exc

    out: list[tinker.Datum] = []
    for datum in train_datums:
        if not datum.advantages:
            raise ValueError(
                f"datum (trial {datum.trial_name}, fragment {datum.fragment_index}) has "
                "no advantages attached; run attach_advantages on the batch before "
                "converting to tinker datums"
            )
        tokens = datum.model_input_tokens
        if len(tokens) < 2:
            raise ValueError(
                f"datum (trial {datum.trial_name}, fragment {datum.fragment_index}) has "
                f"{len(tokens)} token(s); at least 2 are needed for the next-token "
                "input/target shift"
            )
        length = len(tokens) - 1
        out.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64", shape=[length]
                    ),
                    "logprobs": tinker.TensorData(
                        data=datum.sampled_logprobs[1:], dtype="float32", shape=[length]
                    ),
                    # No "mask" key: the live importance_sampling and ppo losses accept
                    # exactly target_tokens, logprobs, and advantages (a "mask" kwarg is
                    # rejected server-side, observed 2026-07-23; the cookbook strips the
                    # same key for both losses). Masking is expressed through the
                    # advantages instead: attach_advantages leaves context positions at
                    # 0.0, and a zero-advantage position contributes zero loss.
                    "advantages": tinker.TensorData(
                        data=datum.advantages[1:], dtype="float32", shape=[length]
                    ),
                },
            )
        )
    return out


def to_tinker_sft_datums(train_datums: Sequence[TrainDatum]) -> list[tinker.Datum]:
    """Convert datums to real `tinker.Datum`s for the cross_entropy loss.

    Same shifted next-token layout as `to_tinker_datums`, but for the two
    cross_entropy consumers: supervised warmup on teacher-sampled
    trajectories (no advantages are needed or read; the loss mask rides as
    the `weights` input so context tokens carry no loss, the backend CE loss
    being `sum(-logprobs * weights)`), and topk-CE replicas from
    `build_topk_ce_datums` (their attached `target_tokens` replace the
    next-token targets at loss positions and their `target_weights`, the
    renormalized teacher probabilities, replace the loss mask).

    The loss_fn_inputs keyset is exactly {"target_tokens", "weights"}: the
    cross_entropy backend of the pinned SDK (tinker 0.23.3) accepts only those
    two keys (its own custom-loss path in
    `tinker/lib/public_interfaces/training_client.py` rejects any other key,
    and the cookbook's supervised datum builder emits exactly this pair).
    Verified against the installed SDK the same way importance_sampling's
    keyset was: the live service already rejected an unexpected "mask" key
    once, so the keyset is pinned by `data_test.py` rather than trusted.

    Args:
        train_datums: Datums from `build_datums` (advantages may be empty)
            or replicas from `build_topk_ce_datums`.

    Returns:
        One `tinker.Datum` per input datum, in order, with loss_fn_inputs
        exactly target_tokens (int64) and weights (float32).

    Raises:
        ImportError: If the tinker SDK is not installed (distill extra).
        ValueError: If a datum is too short for the input/target shift
            (fewer than 2 tokens).
    """
    try:
        import tinker
    except ImportError as exc:
        raise ImportError(MISSING_TINKER_EXTRA) from exc

    out: list[tinker.Datum] = []
    for datum in train_datums:
        tokens = datum.model_input_tokens
        if len(tokens) < 2:
            raise ValueError(
                f"datum (trial {datum.trial_name}, fragment {datum.fragment_index}) has "
                f"{len(tokens)} token(s); at least 2 are needed for the next-token "
                "input/target shift"
            )
        length = len(tokens) - 1
        targets = datum.target_tokens if datum.is_topk_replica else tokens
        weights = datum.target_weights if datum.is_topk_replica else datum.loss_mask
        out.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=targets[1:], dtype="int64", shape=[length]
                    ),
                    "weights": tinker.TensorData(data=weights[1:], dtype="float32", shape=[length]),
                },
            )
        )
    return out


def to_tinker_topk_ce_datums(
    train_datums: Sequence[TrainDatum],
    topk_scores: Sequence[Sequence[TopkCandidates | None]],
    k: int,
) -> list[tinker.Datum]:
    """Convert source datums plus teacher top-k rows to cross_entropy wire datums.

    The one-call conversion for the `topk_ce` loss: `build_topk_ce_datums`
    does the rank-aligned replication and weight renormalization (see its
    docstring for the objective-equivalence argument), then the replicas ride
    the pinned cross_entropy wire format (`to_tinker_sft_datums`: exactly
    {"target_tokens", "weights"}). Misaligned rows drop their source datum
    with a warning, mirroring the loop's own two-step path.

    Args:
        train_datums: Datums from `build_datums`.
        topk_scores: One per-position top-k row per datum
            (`TeacherClient.score_topk`).
        k: The replica count (`train.topk`).

    Returns:
        `k` `tinker.Datum`s per kept source datum, source order then rank
        order, sharing each source's model input.

    Raises:
        ImportError: If the tinker SDK is not installed (distill extra).
        ValueError: On caller bugs (`build_topk_ce_datums`) or datums too
            short for the shift (`to_tinker_sft_datums`).
    """
    replicas, _ = build_topk_ce_datums(train_datums, topk_scores, k)
    return to_tinker_sft_datums(replicas)

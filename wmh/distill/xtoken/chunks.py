"""Chunk plans and cross-tokenizer chunk advantages.

A `ChunkPlan` says, for one `TrainDatum`, which student token ranges are
scoreable and which teacher token range covers the same bytes. Chunks are the
unit of comparison in the cross-tokenizer loss: the teacher cannot score the
student's token ids (different vocabulary), so it scores its own tokenization
of the same text and the two are compared span by span.

`attach_chunk_advantages` turns those spans plus the teacher's per-position
logprobs into the per-token advantage array the existing `importance_sampling`
wire format already carries. Three properties of this module are load-bearing
and easy to get wrong:

1. A chunk's influence on the gradient is its reverse-KL gap, not its length.
   The loss sums `advantage * grad log pi` over positions, so broadcasting
   `(teacher_sum - student_sum) / student_len` to a chunk's student tokens
   makes the chunk contribute exactly `teacher_sum - student_sum` no matter
   how many student tokens it spans. Dividing by the TEACHER span length
   instead would scale every chunk by the tokenizers' verbosity ratio.

2. Centering is over CHUNK TOTALS, never over tokens. Subtracting a constant
   from every token shifts each chunk's total by that constant times the
   chunk's length, which is length-dependent and can inuert a long chunk: two
   chunks with totals +1.0 at lengths 10 and 1000 come out at +0.98 and -0.98
   under token centering, so the long one trains in the wrong direction (its
   total inverts). Subtracting a constant from each chunk's TOTAL preserves
   ordering.

3. Positions no chunk covers keep advantage 0.0 and are never touched by
   centering. Advantage 0.0 IS the mask on the wire (`to_tinker_datums` has no
   mask key), so a token that centering nudged off zero would train on noise.
   The student's own structural tokens (end-of-turn framing, tool-call
   wrappers) have no byte-identical counterpart under the teacher's chat
   template and land here, so this is the common case, not an edge case.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.distill.config import DistillConfig
from wmh.distill.data import TrainDatum

logger = logging.getLogger(__name__)


class ChunkSpan(BaseModel):
    """One aligned chunk: a student token range and the teacher range for the same bytes.

    Both ranges are half-open (`start` inclusive, `end` exclusive) and index
    into their own side's token sequence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    student_start: int = Field(ge=1)
    """Student position 0 is excluded: `to_tinker_datums` ships
    `advantages[1:]` for the next-token shift, so an advantage written at
    position 0 is silently discarded. A chunk starting there would lose part
    of its influence with no error, so the plan builder must start at 1."""

    student_end: int = Field(gt=1)
    teacher_start: int = Field(ge=1)
    """Teacher position 0 has no context and can never carry a logprob, so a
    scoreable chunk never starts there."""

    teacher_end: int = Field(gt=1)
    exact: bool = True
    """Whether the two ranges' canonicalized text matched exactly (as opposed
    to being paired by the aligner across a mismatch)."""

    @model_validator(mode="after")
    def _check_ranges(self) -> ChunkSpan:
        """Reject empty or inverted ranges on either side."""
        if self.student_end <= self.student_start:
            raise ValueError(
                f"student range [{self.student_start}, {self.student_end}) is empty or "
                "inverted; a chunk must cover at least one student token"
            )
        if self.teacher_end <= self.teacher_start:
            raise ValueError(
                f"teacher range [{self.teacher_start}, {self.teacher_end}) is empty or "
                "inverted; a chunk must cover at least one teacher token"
            )
        return self

    @property
    def student_len(self) -> int:
        """How many student tokens this chunk covers."""
        return self.student_end - self.student_start

    @property
    def teacher_len(self) -> int:
        """How many teacher tokens this chunk covers."""
        return self.teacher_end - self.teacher_start


class ChunkPlan(BaseModel):
    """The chunk alignment for one datum, plus the teacher sequence it scores against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_name: str = Field(min_length=1)
    fragment_index: int = Field(ge=0)
    chunks: list[ChunkSpan] = Field(default_factory=list)
    teacher_token_count: int = Field(ge=0)
    """Length of the teacher token sequence the chunks index into."""

    @model_validator(mode="after")
    def _check_monotonic(self) -> ChunkPlan:
        """Require chunks to be sorted and non-overlapping on both sides.

        Overlap would double-count a token's logprob into two chunks, and out
        of order chunks would mean the aligner produced a crossing alignment,
        which the DP forbids by construction.
        """
        previous_student = 1
        previous_teacher = 1
        for index, chunk in enumerate(self.chunks):
            if chunk.student_start < previous_student:
                raise ValueError(
                    f"chunk {index} starts at student position {chunk.student_start}, "
                    f"before the previous chunk ended ({previous_student}); chunks must "
                    "be sorted and non-overlapping"
                )
            if chunk.teacher_start < previous_teacher:
                raise ValueError(
                    f"chunk {index} starts at teacher position {chunk.teacher_start}, "
                    f"before the previous chunk ended ({previous_teacher}); chunks must "
                    "be sorted and non-overlapping"
                )
            if chunk.teacher_end > self.teacher_token_count:
                raise ValueError(
                    f"chunk {index} ends at teacher position {chunk.teacher_end}, past "
                    f"the teacher sequence length {self.teacher_token_count}"
                )
            previous_student = chunk.student_end
            previous_teacher = chunk.teacher_end
        return self

    @property
    def scored_student_tokens(self) -> int:
        """How many student tokens are covered by some chunk."""
        return sum(chunk.student_len for chunk in self.chunks)

    def validate_against(self, datum: TrainDatum) -> None:
        """Check this plan against the datum it will score.

        Args:
            datum: The datum whose `model_input_tokens` the student ranges
                index into.

        Raises:
            ValueError: If the plan names a different datum, a chunk runs past
                the token sequence, or a chunk covers a non-loss position.
                That last one is the subtle case: `_merge_trial_spans` fills
                `sampled_logprobs` with 0.0 at context positions as PADDING,
                not as a real logprob, so a chunk straddling a loss-mask
                transition would silently fold zeros into the student sum.
        """
        if datum.trial_name != self.trial_name or datum.fragment_index != self.fragment_index:
            raise ValueError(
                f"chunk plan is for trial {self.trial_name!r} fragment "
                f"{self.fragment_index}, but the datum is trial {datum.trial_name!r} "
                f"fragment {datum.fragment_index}; plans must be paired with their datum"
            )
        length = len(datum.model_input_tokens)
        for index, chunk in enumerate(self.chunks):
            if chunk.student_end > length:
                raise ValueError(
                    f"chunk {index} ends at student position {chunk.student_end}, past "
                    f"the datum's {length} token(s)"
                )
            for position in range(chunk.student_start, chunk.student_end):
                if datum.loss_mask[position] != 1.0:
                    raise ValueError(
                        f"chunk {index} covers student position {position}, which is a "
                        "context position (loss mask 0.0). Its sampled_logprobs entry is "
                        "0.0 filler rather than a real logprob, so scoring it would "
                        "corrupt the chunk's student sum; split chunks at every loss "
                        "mask transition"
                    )


class ChunkAdvantageStats(BaseModel):
    """Accounting for one `attach_chunk_advantages` call.

    Counters cover only the ATTACHED datums; a dropped datum is never trained
    on, so its tokens are not signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    datums: int = Field(ge=0)
    mismatch_drops: int = Field(ge=0)
    """Datums dropped because their teacher row or plan did not line up."""

    empty_coverage_drops: int = Field(ge=0)
    """Datums dropped because no chunk covered any loss token, so the datum
    carries no signal at all (an all-zero advantage array would be a wasted
    forward pass, not a neutral one)."""

    chunks: int = Field(ge=0)
    scored_loss_tokens: int = Field(ge=0)
    """Loss tokens covered by some chunk (the ones that carry gradient)."""

    unscored_loss_tokens: int = Field(ge=0)
    """Loss tokens no chunk covered; these keep advantage 0.0."""

    clipped_chunks: int = Field(ge=0)
    """Chunks whose per-token advantage hit the clip bound before centering."""

    chunk_reverse_kl: float | None
    """`mean(student_lp - teacher_lp)` over scored loss tokens, the
    cross-tokenizer analogue of the same-tokenizer reverse-KL metric; None
    when nothing was scored."""

    advantage_mean: float | None
    """Mean advantage over scored loss tokens after clipping and centering."""

    advantage_std: float | None
    """Population standard deviation over the same tokens."""

    @property
    def coverage_rate(self) -> float:
        """Fraction of the attached datums' loss tokens that a chunk covered."""
        total = self.scored_loss_tokens + self.unscored_loss_tokens
        return self.scored_loss_tokens / total if total else 0.0


def _chunk_totals(
    datum: TrainDatum,
    plan: ChunkPlan,
    teacher_logprobs: Sequence[float | None],
    clip: float | None,
) -> tuple[list[float], int] | None:
    """Per-chunk clipped totals for one datum, or None when the teacher row fails.

    Returns `(totals, clipped)` where `totals[i]` is chunk i's contribution
    after per-token clipping, and `clipped` counts chunks that hit the bound.

    `clip` of None disables clipping entirely and reports zero clipped chunks,
    which is what `train.advantage_clip = None` means. The same-tokenizer lane
    made that field optional, so typing it as a bare float here would raise
    TypeError on a perfectly valid config rather than train unclipped.
    """
    totals: list[float] = []
    clipped_count = 0
    for index, chunk in enumerate(plan.chunks):
        teacher_sum = 0.0
        for position in range(chunk.teacher_start, chunk.teacher_end):
            value = teacher_logprobs[position]
            if value is None:
                logger.warning(
                    "dropping datum (trial %s, fragment %d): teacher logprob at "
                    "position %d is None but chunk %d needs it; the teacher must score "
                    "every position its chunks cover",
                    datum.trial_name,
                    datum.fragment_index,
                    position,
                    index,
                )
                return None
            teacher_sum += value
        student_sum = sum(
            datum.sampled_logprobs[position]
            for position in range(chunk.student_start, chunk.student_end)
        )
        # Divide by the STUDENT length so the chunk's total influence is
        # exactly its reverse-KL gap (see the module docstring).
        per_token = (teacher_sum - student_sum) / chunk.student_len
        bounded = per_token if clip is None else min(max(per_token, -clip), clip)
        if bounded != per_token:
            clipped_count += 1
        totals.append(bounded * chunk.student_len)
    return totals, clipped_count


def attach_chunk_advantages(
    datums: Sequence[TrainDatum],
    plans: Sequence[ChunkPlan],
    teacher_logprobs: Sequence[Sequence[float | None]],
    cfg: DistillConfig,
) -> tuple[list[TrainDatum], ChunkAdvantageStats]:
    """Fill per-token advantages from chunk-aligned teacher logprobs.

    Each chunk gets `clip((teacher_sum - student_sum) / student_len)`
    broadcast to its student tokens, so the chunk contributes its reverse-KL
    gap regardless of length. Under `train.center_advantages` the mean over
    CHUNK TOTALS is then subtracted from every chunk's total (see the module
    docstring for why token-level centering would invert long chunks).
    Positions no chunk covers stay at 0.0 and are never centered.

    Args:
        datums: Datums from `build_datums` (advantages not yet attached).
        plans: One chunk plan per datum, in datum order.
        teacher_logprobs: One per-position teacher logprob row per datum, in
            the teacher's OWN tokenization (length `teacher_token_count`);
            entry p is the logprob of teacher token p given tokens before it.
        cfg: The run config; reads `train.advantage_clip` and
            `train.center_advantages`.

    Returns:
        New datums with advantages attached (drops removed, order preserved)
        and the stats, including chunk coverage and the chunk reverse KL.

    Raises:
        ValueError: If `plans` or `teacher_logprobs` do not have exactly one
            entry per datum; that is a caller bug, not per-datum evidence.
    """
    if len(plans) != len(datums):
        raise ValueError(
            f"got {len(plans)} chunk plan(s) for {len(datums)} datum(s); pass exactly "
            "one plan per datum, in datum order"
        )
    if len(teacher_logprobs) != len(datums):
        raise ValueError(
            f"got {len(teacher_logprobs)} teacher logprob row(s) for {len(datums)} "
            "datum(s); pass exactly one row per datum, in datum order"
        )
    clip = cfg.train.advantage_clip
    kept: list[TrainDatum] = []
    kept_plans: list[ChunkPlan] = []
    kept_totals: list[list[float]] = []
    kept_rows: list[Sequence[float | None]] = []
    mismatch_drops = 0
    empty_coverage_drops = 0
    clipped_chunks = 0
    unscored = 0
    for datum, plan, row in zip(datums, plans, teacher_logprobs, strict=True):
        if len(row) != plan.teacher_token_count:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: teacher "
                "returned %d logprob(s) for a %d-token teacher sequence; the row must "
                "cover the exact sequence the chunk plan was built against",
                datum.trial_name,
                datum.fragment_index,
                len(row),
                plan.teacher_token_count,
            )
            continue
        try:
            plan.validate_against(datum)
        except ValueError as exc:
            mismatch_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: %s",
                datum.trial_name,
                datum.fragment_index,
                exc,
            )
            continue
        scored = plan.scored_student_tokens
        if not scored:
            empty_coverage_drops += 1
            logger.warning(
                "dropping datum (trial %s, fragment %d) from training: no chunk covered "
                "any loss token, so the datum carries no gradient. Check the teacher "
                "render and the aligner's fallback rate",
                datum.trial_name,
                datum.fragment_index,
            )
            continue
        computed = _chunk_totals(datum, plan, row, clip)
        if computed is None:
            mismatch_drops += 1
            continue
        totals, clipped = computed
        kept.append(datum)
        kept_plans.append(plan)
        kept_totals.append(totals)
        kept_rows.append(row)
        clipped_chunks += clipped
        unscored += datum.loss_token_count - scored

    # Centering over chunk totals: subtract one constant from each chunk's
    # TOTAL so every chunk keeps its relative weight (module docstring, point 2).
    if cfg.train.center_advantages:
        chunk_count = sum(len(totals) for totals in kept_totals)
        if chunk_count:
            mean_total = sum(sum(totals) for totals in kept_totals) / chunk_count
            kept_totals = [[total - mean_total for total in totals] for totals in kept_totals]

    attached: list[TrainDatum] = []
    scored_values: list[float] = []
    for datum, plan, totals in zip(kept, kept_plans, kept_totals, strict=True):
        advantages = [0.0] * len(datum.model_input_tokens)
        for chunk, total in zip(plan.chunks, totals, strict=True):
            per_token = total / chunk.student_len
            for position in range(chunk.student_start, chunk.student_end):
                advantages[position] = per_token
                scored_values.append(per_token)
        attached.append(
            TrainDatum(
                trial_name=datum.trial_name,
                fragment_index=datum.fragment_index,
                model_input_tokens=datum.model_input_tokens,
                loss_mask=datum.loss_mask,
                sampled_logprobs=datum.sampled_logprobs,
                advantages=advantages,
            )
        )

    # The chunk reverse KL is computed from the PRE-clip, PRE-centering gaps so
    # it stays a comparable measurement of teacher-student divergence rather
    # than a readout of the training transform.
    kl_gap = 0.0
    kl_tokens = 0
    for datum, plan, row in zip(kept, kept_plans, kept_rows, strict=True):
        for chunk in plan.chunks:
            teacher_sum = sum(
                value
                for position in range(chunk.teacher_start, chunk.teacher_end)
                if (value := row[position]) is not None
            )
            student_sum = sum(
                datum.sampled_logprobs[position]
                for position in range(chunk.student_start, chunk.student_end)
            )
            kl_gap += student_sum - teacher_sum
            kl_tokens += chunk.student_len

    advantage_mean: float | None = None
    advantage_std: float | None = None
    if scored_values:
        advantage_mean = sum(scored_values) / len(scored_values)
        variance = sum((value - advantage_mean) ** 2 for value in scored_values) / len(
            scored_values
        )
        advantage_std = variance**0.5
    stats = ChunkAdvantageStats(
        datums=len(attached),
        mismatch_drops=mismatch_drops,
        empty_coverage_drops=empty_coverage_drops,
        chunks=sum(len(plan.chunks) for plan in kept_plans),
        scored_loss_tokens=len(scored_values),
        unscored_loss_tokens=unscored,
        clipped_chunks=clipped_chunks,
        chunk_reverse_kl=kl_gap / kl_tokens if kl_tokens else None,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )
    if stats.datums and stats.coverage_rate < 0.95:
        logger.warning(
            "chunk coverage is %.1f%% of loss tokens (%d scored, %d unscored); the "
            "cross-tokenizer path expects >95%%, so check the teacher render's message "
            "content islands and the aligner fallback rate",
            stats.coverage_rate * 100.0,
            stats.scored_loss_tokens,
            stats.unscored_loss_tokens,
        )
    return attached, stats

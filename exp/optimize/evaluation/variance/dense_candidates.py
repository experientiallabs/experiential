"""Candidate tagged behaviors the dense judge checks: from definitions, error tags and linters.

A candidate is a hypothesis, never a finding: the dense judge decides each one itself and may
answer absent, acceptable or uncertain. Candidates only tell the judge where to look.

- :func:`definition_candidates`: one candidate per frozen tag, so every training tag is checked on
  every episode even when no tagger proposed it. This is what a Harbor trial uses.
- :func:`error_tag_candidates`: error-taxonomy propositions that fired, or were uncertain, on the
  episode (an :class:`~exp.simulation.mining.error_tags.ErrorTraceTags` summary), mapped to the
  tags whose ``sources`` list the proposition.
- :func:`linter_candidates`: behavior-library linters a rubric cites
  (:class:`~exp.optimize.evaluation.variance.behavior_profiles.BehaviorGuidance`), mapped to the
  tags whose ``sources`` list the linter's locator, carrying its detection guidance.
- :func:`merge_candidates` keeps the first candidate of each identity, in order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.optimize.evaluation.variance.behavior_profiles import BehaviorGuidance
from exp.optimize.evaluation.variance.dense_reward import TAG_ID, DenseRewardDefinition
from exp.simulation.mining.error_tags import ErrorTraceTags

CandidateOrigin = Literal["definition", "error_tag", "linter"]
"""Where a candidate came from."""


class DenseCandidate(ContractModel):
    """One tagged behavior for the dense judge to check.

    Attributes:
        candidate_id: Stable identity, answered by exactly one finding per variant.
        tag_id: The frozen tag it proposes.
        origin: Definition, error tag or linter.
        proposition: The error proposition or linter locator that proposed it.
        tagger_status: ``fired`` or ``uncertain`` for an error-tag candidate.
        hint_refs: Events the tagger pointed at; hints, not evidence.
        note: Detection guidance or other context for the judge.
    """

    candidate_id: str = Field(min_length=1, max_length=256)
    tag_id: str = Field(pattern=TAG_ID)
    origin: CandidateOrigin
    proposition: str | None = None
    tagger_status: Literal["fired", "uncertain"] | None = None
    hint_refs: tuple[str, ...] = ()
    note: str | None = Field(default=None, max_length=4000)


def definition_candidates(definition: DenseRewardDefinition) -> tuple[DenseCandidate, ...]:
    """Return one candidate per frozen tag."""
    return tuple(
        DenseCandidate(candidate_id=f"tag:{tag.tag_id}", tag_id=tag.tag_id, origin="definition")
        for tag in definition.tags
    )


def error_tag_candidates(
    definition: DenseRewardDefinition,
    summary: ErrorTraceTags,
    *,
    hint_refs: Sequence[str] = (),
) -> tuple[DenseCandidate, ...]:
    """Return candidates for every fired or uncertain proposition a tag lists as a source.

    Args:
        definition: The frozen dense definition.
        summary: One episode's (or step's) error-tag summary.
        hint_refs: Events the summary is about, such as a tagged step's events.

    Returns:
        One candidate per (proposition, tag) pair; propositions no tag lists are ignored.
    """
    found: list[DenseCandidate] = []
    for status, propositions in (("fired", summary.fired), ("uncertain", summary.uncertain)):
        for proposition in propositions:
            for tag in definition.tags:
                if proposition in tag.sources:
                    found.append(
                        DenseCandidate(
                            candidate_id=f"error:{summary.trace_id}:{proposition}:{tag.tag_id}",
                            tag_id=tag.tag_id,
                            origin="error_tag",
                            proposition=proposition,
                            tagger_status=status,
                            hint_refs=tuple(hint_refs),
                        )
                    )
    return tuple(found)


def linter_candidates(
    definition: DenseRewardDefinition, guidance: Sequence[BehaviorGuidance]
) -> tuple[DenseCandidate, ...]:
    """Return candidates for every cited linter a tag lists as a source.

    Args:
        definition: The frozen dense definition.
        guidance: Library entries a rubric cites, as the judge reads them.

    Returns:
        One candidate per (linter, tag) pair, carrying the linter's detection guidance.
    """
    found: list[DenseCandidate] = []
    for entry in guidance:
        if entry.kind != "linter":
            continue
        for tag in definition.tags:
            if entry.locator in tag.sources:
                note = " ".join(
                    part
                    for part in (
                        entry.statement,
                        entry.detection or "",
                        *(f"Acceptable look-alike: {row}" for row in entry.false_positive_notes),
                    )
                    if part
                )
                found.append(
                    DenseCandidate(
                        candidate_id=f"{entry.locator}:{tag.tag_id}",
                        tag_id=tag.tag_id,
                        origin="linter",
                        proposition=entry.locator,
                        note=note[:4000],
                    )
                )
    return tuple(found)


def merge_candidates(*groups: Sequence[DenseCandidate]) -> tuple[DenseCandidate, ...]:
    """Concatenate candidate groups, keeping the first candidate of each identity."""
    merged: dict[str, DenseCandidate] = {}
    for group in groups:
        for candidate in group:
            merged.setdefault(candidate.candidate_id, candidate)
    return tuple(merged.values())

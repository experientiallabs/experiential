"""Ordered scoring of complete harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field

from wmh.core.text import validate_durable_text
from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import (
    HarnessScore,
    HarnessScorer,
    ScoreRequest,
    score_harness,
)
from wmh.harness.source_tree import HarnessSourceTree

MAX_SCORE_TARGET_LABEL_CHARS = 512


@dataclass(frozen=True)
class HarnessScoreTarget:
    """A labeled immutable harness document to evaluate once."""

    label: str
    harness: HarnessDoc
    source: HarnessSourceTree = field(init=False)

    def __post_init__(self) -> None:
        if (
            not self.label
            or len(self.label) > MAX_SCORE_TARGET_LABEL_CHARS
            or "\n" in self.label
            or "\r" in self.label
        ):
            raise ValueError(
                "score target label must be one line containing "
                f"1 to {MAX_SCORE_TARGET_LABEL_CHARS} characters"
            )
        validate_durable_text(self.label, field="score target label")
        object.__setattr__(self, "source", HarnessSourceTree.from_doc(self.harness))


@dataclass(frozen=True)
class ScoredHarness:
    """One target paired with its evaluator-owned score evidence."""

    target: HarnessScoreTarget
    score: HarnessScore

    def __post_init__(self) -> None:
        if self.score.report.candidate_doc_hash != self.target.harness.doc_hash:
            raise ValueError("scored harness report does not match its target document")


@dataclass(frozen=True)
class HarnessScoreBatch:
    """Ordered results evaluated against one exact score request."""

    request: ScoreRequest
    entries: tuple[ScoredHarness, ...]

    def __post_init__(self) -> None:
        validate_score_targets(tuple(entry.target for entry in self.entries))
        for entry in self.entries:
            if entry.score.report.request != self.request:
                raise ValueError("scored harness entries use different score requests")


def validate_score_targets(targets: tuple[HarnessScoreTarget, ...]) -> None:
    """Reject an empty or ambiguously labeled target sequence before evaluation."""
    if not targets:
        raise ValueError("at least one harness score target is required")
    labels = [target.label for target in targets]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate score target label(s): {duplicates}")


def score_harnesses(
    scorer: HarnessScorer,
    targets: tuple[HarnessScoreTarget, ...],
    *,
    request: ScoreRequest,
) -> HarnessScoreBatch:
    """Score complete harnesses sequentially in their declared order."""
    validate_score_targets(targets)
    entries = tuple(
        ScoredHarness(
            target=target,
            score=score_harness(scorer, target.harness, request=request),
        )
        for target in targets
    )
    return HarnessScoreBatch(request=request, entries=entries)

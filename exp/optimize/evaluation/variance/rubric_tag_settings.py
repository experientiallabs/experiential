"""Settings of the opt-in rubric-authored behavior tags (:mod:`~.rubric_tags`).

Kept apart from the tagging code so :class:`~.behavior_differences.BehaviorDifferenceSpec` can
carry them without importing the authoring stack.
"""

from __future__ import annotations

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.optimize.evaluation.factory.authoring import AuthorSettings


class RubricTagSettings(ContractModel):
    """Opt-in rubric-authored tags for the behavior diff.

    Attributes:
        author: Open-weight tag author; None uses the behavior aligner.
        maximum_tags: Most tags kept from the authored set.
        repairs: Repair draws after a structurally invalid draft.
        screen: Whether each tag's wording is screened by the decision model before tagging.
        screen_floor: A tag whose screen answer is below this on any screen question is dropped.
        paraphrase_gap: Largest difference between a part's two wordings that still counts as
            agreement; a larger gap across the band reads 0.5.
        diff_bytes: Bound on one step's rendered diff.
        source_bytes: Bound on each changed file's full text.
        output_bytes: Bound on one tool output (head and tail kept).
        runs: Most recorded tool calls shown to a claim.
        maximum_fire_rate: A tag present on at least this share of a tier's attempts is screened
            out of the signatures as non-discriminating (it fires regardless of the outcome).
        unscored_uncertain: Whether an uncertain tag is left out of a pair's signature (unscored)
            instead of blocking the whole pair.
    """

    author: AuthorSettings | None = None
    maximum_tags: int = Field(default=24, ge=1, le=64, strict=True)
    repairs: int = Field(default=1, ge=0, le=3, strict=True)
    screen: bool = True
    screen_floor: float = Field(default=0.3, gt=0.0, lt=1.0)
    paraphrase_gap: float = Field(default=0.4, gt=0.0, le=1.0)
    diff_bytes: int = Field(default=12_000, ge=1_000, le=60_000, strict=True)
    source_bytes: int = Field(default=24_000, ge=1_000, le=120_000, strict=True)
    output_bytes: int = Field(default=4_000, ge=500, le=40_000, strict=True)
    runs: int = Field(default=4, ge=1, le=16, strict=True)
    maximum_fire_rate: float = Field(default=0.9, gt=0.5, le=1.0)
    unscored_uncertain: bool = True

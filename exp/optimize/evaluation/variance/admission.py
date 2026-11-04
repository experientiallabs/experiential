"""The versioned admission policy a decision family must meet before its records are trainable.

Training eligibility (:mod:`~exp.optimize.evaluation.variance.records`) holds only when, besides its
existing conditions, every confirmed variant satisfies this policy, frozen in the family spec
before any contrast runs:

- **Judge coverage.** The variant's frozen episode judge labelled its confirmation episodes:
  operational gaps (no evidence, malformed or unavailable responses) are not allowed unless the
  policy says so, and the share of unscored episodes (operational gaps plus unassessable verdicts)
  is at most :attr:`FamilyAdmissionPolicy.maximum_unscored_fraction`. Judge calibration is a
  separate question: bad options correctly judged as failures are valid negative examples, so
  the policy never asks episodes to pass.
- **Rubric problems.** No judged confirmation episode reported a rubric problem. One uncertain
  criterion is not proof of an ambiguous rubric; only the judge's own rubric-problem reports and
  insufficient-evidence findings are counted, separately.
- **Rubric compatibility.** The variant's rubric and world were reviewed against their shared
  authoring packet (:func:`~exp.optimize.evaluation.variance.world_freeze.review_compatibility`)
  with no disagreement, and the review covers exactly the judge's rubric and world.
- **World freeze.** The judge's world was frozen
  (:func:`~exp.optimize.evaluation.variance.world_freeze.freeze_world`): the freeze is trainable,
  the bound variant's status is ``frozen``, and its world digest is the judge's.

Each failed check is recorded per variant with what was found; :func:`variant_problems` returns
them and :func:`coverage_problems` checks coverage. This policy scopes family admission, not the
qualification of the LLM assessment path itself, which needs independently labelled controls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgeCoverage
from exp.optimize.evaluation.variance.world_freeze import (
    FREEZE_TYPE,
    CompatibilityReview,
    WorldFreeze,
)

FAMILY_ADMISSION_POLICY = "family-admission-v1"
"""Version of the family admission policy."""


class FamilyAdmissionPolicy(ContractModel):
    """Admission limits frozen before any contrast of the family runs.

    Attributes:
        version: Policy version.
        maximum_unscored_fraction: Largest share of a confirmed variant's planned confirmation
            episodes that may lack a judged pass or fail.
        allow_operational_gaps: Whether any confirmation episode may lack an assessment because
            evidence could not be built or the judge response was malformed or unavailable.
        rationale: Why these limits are appropriate for the scoped task domain.
    """

    version: Literal["family-admission-v1"] = FAMILY_ADMISSION_POLICY
    maximum_unscored_fraction: float = Field(default=0.125, ge=0.0, le=1.0)
    allow_operational_gaps: bool = False
    rationale: str = Field(
        default=(
            "pilot scope: one simulated decision family; an eighth of confirmation episodes may "
            "be unassessable so a few genuinely missing records do not force relabelling, and "
            "every operational judge failure is treated as a blocker rather than a negative"
        ),
        min_length=1,
        max_length=2000,
    )


class VariantJudging(ContractModel):
    """The frozen judge of one variant and the evidence that its world and rubric fit together.

    Attributes:
        judge: The variant's frozen episode judge and admitted request settings.
        world_freeze: Saved ``world-freeze`` report of the judge's world, when frozen.
        freeze_variant: The freeze's variant identity this scenario starts from.
        compatibility: Review of the judge's rubric against the frozen world, when made.
    """

    judge: EpisodeJudge
    world_freeze: ArtifactInput | None = None
    freeze_variant: str | None = None
    compatibility: CompatibilityReview | None = None


def coverage_problems(
    coverage: JudgeCoverage, policy: FamilyAdmissionPolicy
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check one confirmed variant's judge coverage against the policy.

    Args:
        coverage: The judge's coverage of its confirmation episodes.
        policy: The frozen admission policy.

    Returns:
        Coverage problems and rubric problems, each empty when the check holds.
    """
    problems: list[str] = []
    operational = coverage.gaps + coverage.malformed + coverage.unavailable
    if coverage.planned == 0:
        problems.append("no confirmation episode was planned")
    if operational and not policy.allow_operational_gaps:
        problems.append(
            f"{operational} episodes are operationally unscored ({coverage.gaps} without "
            f"evidence, {coverage.malformed} malformed, {coverage.unavailable} unavailable)"
        )
    if coverage.planned and coverage.unscored / coverage.planned > (
        policy.maximum_unscored_fraction
    ):
        problems.append(
            f"{coverage.unscored} of {coverage.planned} episodes are unscored, over the "
            f"policy's {policy.maximum_unscored_fraction:.3f}"
        )
    rubric = (
        (f"{coverage.rubric_problems} judged episodes reported a rubric problem",)
        if coverage.rubric_problems
        else ()
    )
    return tuple(problems), rubric


def variant_problems(
    project: ProjectStore, judging: VariantJudging
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check one variant's compatibility review and world freeze against its judge.

    Args:
        project: Project holding the rubric and the world-freeze report.
        judging: The variant's judge binding.

    Returns:
        Compatibility problems and world-freeze problems, each empty when the check holds.
    """
    definition = judging.judge.definition
    rubric = definition.rubric_sha256
    review = judging.compatibility
    compatibility: list[str] = []
    if review is None:
        compatibility.append("no compatibility review of the rubric against the world")
    else:
        if (review.rubric_sha256, review.world_sha256) != (rubric, definition.world_sha256):
            compatibility.append("the compatibility review covers another rubric or world")
        if not review.compatible or review.disagreements:
            compatibility.append(
                f"{len(review.disagreements)} unreconciled world/rubric disagreements"
            )
    freeze_problems: list[str] = []
    if judging.world_freeze is None or judging.freeze_variant is None:
        freeze_problems.append("the judge's world has no freeze report for this variant")
    else:
        freeze = _load_freeze(project, judging.world_freeze)
        rows = {row.variant_id: row for row in freeze.variants}
        row = rows.get(judging.freeze_variant)
        if freeze.world_sha256 != definition.world_sha256:
            freeze_problems.append("the freeze froze a different world than the judge's")
        if not freeze.trainable:
            freeze_problems.append("the world freeze is not trainable")
        if row is None or row.status != "frozen":
            status = "absent" if row is None else row.status
            freeze_problems.append(f"freeze variant {judging.freeze_variant} is {status}")
    return tuple(compatibility), tuple(freeze_problems)


def judging_problems(
    project: ProjectStore,
    judging: Mapping[str, VariantJudging],
    coverage: Mapping[str, JudgeCoverage],
    policy: FamilyAdmissionPolicy,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Check every confirmed variant, keyed by condition then scenario.

    Args:
        project: Project holding the rubrics and freeze reports.
        judging: Judge binding per scenario.
        coverage: Judge coverage of each confirmed scenario's confirmation episodes.
        policy: The frozen admission policy.

    Returns:
        Per condition (``judge_coverage``, ``rubric_problems``, ``rubric_compatibility``,
        ``world_freeze``), the problems of each confirmed scenario that has any.
    """
    found: dict[str, dict[str, tuple[str, ...]]] = {
        name: {}
        for name in ("judge_coverage", "rubric_problems", "rubric_compatibility", "world_freeze")
    }
    for scenario, counts in sorted(coverage.items()):
        covered, rubric = coverage_problems(counts, policy)
        binding = judging.get(scenario)
        if binding is None:
            compatible, frozen = ("no judge binding",), ("no judge binding",)
        else:
            if counts.judge not in (None, binding.judge.identity):
                covered = (*covered, "confirmation was judged by another judge")
            compatible, frozen = variant_problems(project, binding)
        for name, problems in zip(found, (covered, rubric, compatible, frozen), strict=True):
            if problems:
                found[name][scenario] = problems
    return found


def _load_freeze(project: ProjectStore, reference: ArtifactInput) -> WorldFreeze:
    """Read one pinned world-freeze report.

    Raises:
        ValueError: The reference is not the saved world-freeze report it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        FREEZE_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned world-freeze report")
    return WorldFreeze.model_validate_json(stored.payloads["freeze.json"])

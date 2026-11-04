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
  with no disagreement, and the review covers exactly the judge's rubric and the frozen
  authored world.
- **World freeze.** The variant's authored world was frozen
  (:func:`~exp.optimize.evaluation.variance.world_freeze.freeze_world`): the freeze is trainable
  and the bound variant's status is ``frozen``.
- **World provenance.** The variant's contrasts ran in that frozen world, not merely under
  matching digests: every selection and confirmation contrast started from the freeze report's
  saved checkpoint for the variant, the variant's runtime configuration is the definition whose
  digest the freeze recorded, and the judge binds that checkpoint's task and that runtime
  definition (judged evidence carries digests derived from the run, and judging refuses others).

The judge's ``world_sha256`` is the digest of the exact runtime definition episodes run under
(:func:`~exp.optimize.evaluation.variance.episode_evidence.run_world_sha256`); for a frozen
variant that is the freeze's ``definition_sha256``, while the authored world's own digest is the
freeze's ``world_sha256``. :func:`frozen_start` gives the family the frozen checkpoint and runtime
configuration its contrasts run from, and :func:`frozen_run_digests` the digests a judge for that
variant must bind.

Each failed check is recorded per variant with what was found; :func:`variant_problems`,
:func:`provenance_problems` and :func:`coverage_problems` return them. This policy scopes family
admission, not the qualification of the LLM assessment path itself, which needs independently
labelled controls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgeCoverage
from exp.optimize.evaluation.variance.world_freeze import (
    FREEZE_TYPE,
    CompatibilityReview,
    VariantFreeze,
    WorldFreeze,
)
from exp.simulation.continuations import load_continuation_checkpoint

FAMILY_ADMISSION_POLICY = "family-admission-v2"
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

    version: Literal["family-admission-v2"] = FAMILY_ADMISSION_POLICY
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
        runtime: The frozen variant's grounded runtime definition as job configuration, required
            with a freeze; its digest must be the freeze's ``definition_sha256``.
        compatibility: Review of the judge's rubric against the frozen world, when made.
    """

    judge: EpisodeJudge
    world_freeze: ArtifactInput | None = None
    freeze_variant: str | None = None
    runtime: JsonObject | None = None
    compatibility: CompatibilityReview | None = None


def frozen_start(
    project: ProjectStore, judging: VariantJudging
) -> tuple[ArtifactInput, JsonObject] | None:
    """Return the frozen checkpoint and runtime configuration a variant's contrasts run from.

    Args:
        project: Project holding the world-freeze report.
        judging: The variant's judge binding.

    Returns:
        The freeze's saved start checkpoint for the variant and its runtime configuration, or
        None when no freeze is bound.

    Raises:
        ValueError: The freeze has no such variant or no saved start for it, or the supplied
            runtime is missing or is not the definition the freeze recorded.
    """
    if judging.world_freeze is None and judging.freeze_variant is None:
        return None
    if judging.world_freeze is None or judging.freeze_variant is None:
        raise ValueError("a world freeze binding needs both the report and its variant")
    row = _freeze_row(_load_freeze(project, judging.world_freeze), judging.freeze_variant)
    if row.checkpoint is None:
        raise ValueError(f"freeze variant {row.variant_id} ({row.status}) has no saved start")
    if judging.runtime is None or run_world_sha256(judging.runtime) != row.definition_sha256:
        raise ValueError(
            f"the supplied runtime is not freeze variant {row.variant_id}'s frozen definition"
        )
    return row.checkpoint, judging.runtime


def frozen_run_digests(
    project: ProjectStore, world_freeze: ArtifactInput, variant_id: str
) -> tuple[Sha256, Sha256]:
    """Return the task and world digests a run from one frozen variant produces.

    A judge for that variant binds exactly these, so it refuses evidence from any other start or
    runtime.

    Args:
        project: Project holding the report and its checkpoint.
        world_freeze: The saved world-freeze report.
        variant_id: The frozen variant.

    Returns:
        The frozen checkpoint's task digest and the frozen runtime definition's digest.

    Raises:
        ValueError: The variant has no saved start or recorded runtime definition.
    """
    row = _freeze_row(_load_freeze(project, world_freeze), variant_id)
    if row.checkpoint is None or row.definition_sha256 is None:
        raise ValueError(f"freeze variant {variant_id} ({row.status}) has no saved start")
    checkpoint = load_continuation_checkpoint(project.artifacts, row.checkpoint)
    return run_task_sha256(checkpoint), row.definition_sha256


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
    freeze = None if judging.world_freeze is None else _load_freeze(project, judging.world_freeze)
    review = judging.compatibility
    compatibility: list[str] = []
    if review is None:
        compatibility.append("no compatibility review of the rubric against the world")
    else:
        if review.rubric_sha256 != definition.rubric_sha256 or (
            freeze is not None and review.world_sha256 != freeze.world_sha256
        ):
            compatibility.append("the compatibility review covers another rubric or world")
        if not review.compatible or review.disagreements:
            compatibility.append(
                f"{len(review.disagreements)} unreconciled world/rubric disagreements"
            )
    freeze_problems: list[str] = []
    if freeze is None or judging.freeze_variant is None:
        freeze_problems.append("the judge's world has no freeze report for this variant")
    else:
        rows = {row.variant_id: row for row in freeze.variants}
        row = rows.get(judging.freeze_variant)
        if not freeze.trainable:
            freeze_problems.append("the world freeze is not trainable")
        if row is None or row.status != "frozen":
            status = "absent" if row is None else row.status
            freeze_problems.append(f"freeze variant {judging.freeze_variant} is {status}")
    return tuple(compatibility), tuple(freeze_problems)


def provenance_problems(
    project: ProjectStore, judging: VariantJudging, starts: tuple[ArtifactInput, ...]
) -> tuple[str, ...]:
    """Check that a variant's contrasts started from its frozen checkpoint in its frozen world.

    Args:
        project: Project holding the world-freeze report and its checkpoint.
        judging: The variant's judge binding.
        starts: The starting checkpoint of every selection and confirmation contrast.

    Returns:
        The provenance problems, empty when every contrast ran from the frozen start under the
        frozen runtime definition the judge binds.
    """
    if judging.world_freeze is None or judging.freeze_variant is None:
        return ("no world-freeze report binds this variant, so no frozen start was used",)
    freeze = _load_freeze(project, judging.world_freeze)
    row = {r.variant_id: r for r in freeze.variants}.get(judging.freeze_variant)
    if row is None or row.checkpoint is None or row.definition_sha256 is None:
        return (f"freeze variant {judging.freeze_variant} has no saved frozen start",)
    problems: list[str] = []
    if not starts:
        problems.append("no contrast of this variant was recorded")
    problems.extend(
        f"a contrast started from {start.artifact_id}, not the frozen start "
        f"{row.checkpoint.artifact_id}"
        for start in dict.fromkeys(starts)
        if start != row.checkpoint
    )
    if judging.runtime is None or run_world_sha256(judging.runtime) != row.definition_sha256:
        problems.append("the variant's runtime is not the freeze's recorded definition")
    definition = judging.judge.definition
    if definition.world_sha256 != row.definition_sha256:
        problems.append("the judge binds another runtime than the frozen definition")
    checkpoint = load_continuation_checkpoint(project.artifacts, row.checkpoint)
    if definition.task_sha256 != run_task_sha256(checkpoint):
        problems.append("the judge binds another task than the frozen start's")
    return tuple(problems)


def judging_problems(
    project: ProjectStore,
    judging: Mapping[str, VariantJudging],
    coverage: Mapping[str, JudgeCoverage],
    policy: FamilyAdmissionPolicy,
    starts: Mapping[str, tuple[ArtifactInput, ...]] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Check every confirmed variant, keyed by condition then scenario.

    Args:
        project: Project holding the rubrics and freeze reports.
        judging: Judge binding per scenario.
        coverage: Judge coverage of each confirmed scenario's confirmation episodes.
        policy: The frozen admission policy.
        starts: Starting checkpoints of each scenario's selection and confirmation contrasts.

    Returns:
        Per condition (``judge_coverage``, ``rubric_problems``, ``rubric_compatibility``,
        ``world_freeze``, ``world_provenance``), the problems of each confirmed scenario that has
        any.
    """
    names = (
        "judge_coverage",
        "rubric_problems",
        "rubric_compatibility",
        "world_freeze",
        "world_provenance",
    )
    found: dict[str, dict[str, tuple[str, ...]]] = {name: {} for name in names}
    for scenario, counts in sorted(coverage.items()):
        covered, rubric = coverage_problems(counts, policy)
        binding = judging.get(scenario)
        if binding is None:
            compatible = frozen = provenance = ("no judge binding",)
        else:
            if counts.judge not in (None, binding.judge.identity):
                covered = (*covered, "confirmation was judged by another judge")
            compatible, frozen = variant_problems(project, binding)
            provenance = provenance_problems(project, binding, (starts or {}).get(scenario, ()))
        for name, problems in zip(
            names, (covered, rubric, compatible, frozen, provenance), strict=True
        ):
            if problems:
                found[name][scenario] = problems
    return found


def _freeze_row(freeze: WorldFreeze, variant_id: str) -> VariantFreeze:
    """Return one variant's freeze record.

    Raises:
        ValueError: The freeze has no such variant.
    """
    rows = {row.variant_id: row for row in freeze.variants}
    if variant_id not in rows:
        raise ValueError(f"world freeze {freeze.identity} has no variant {variant_id}")
    return rows[variant_id]


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

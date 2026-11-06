"""Adversarial (reward-hacking) families wired into the coalition, the selectors and the library.

The behavior-difference groups each tier's mining reviewed are read for reward-hacking
signatures (:mod:`~.behavior_adversarial`: an Added side carrying ``modified_evaluation``,
``exploited_metric``, ``hid_failure`` or ``unsupported_success_claim``). They feed both rubric
parts of every family:

- the authoring coalition reads adversarial groups first (:func:`coalition_differences`), so the
  decision-state author and variant search see each hack branch beside its genuine branch;
- selectors and critics read the mined families as a reference beside the behavior differences
  (:func:`adversarial_references`);
- with ``authoring.adversarial`` registered, each family gets an independently reviewed linter
  (:func:`~.adversarial_rubric.author_adversarial_linter`); accepted linters join the behavior
  library version the rubric is authored from (:func:`adversarial_library`), so the outcome
  judge reads them as behavior guidance and the hybrid dense judge receives them as linter
  candidates for the hack tags.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.adversarial_rubric import (
    AdversarialRubricSpec,
    author_adversarial_linter,
)
from exp.optimize.evaluation.variance.behavior_adversarial import (
    adversarial_evidence,
    adversarial_families,
    adversarial_reference,
)
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceEvidence
from exp.optimize.evaluation.variance.behavior_pairs import (
    compact_differences,
    load_behavior_differences,
)
from exp.optimize.evaluation.variance.behavior_profiles import (
    BehaviorGuidance,
    BehaviorLinter,
    LibraryBinding,
    LibraryChange,
    empty_library,
    load_library,
    save_library,
)
from exp.optimize.evaluation.variance.coalition import CoalitionEvidence, mining_coalition
from exp.optimize.evaluation.variance.evidence import ReferenceSource
from exp.optimize.evaluation.variance.judged import JudgedStudy
from exp.optimize.evaluation.variance.pipeline_spec import MiningOutputs, StageContext, TierMining


def coalition_differences(
    project: ProjectStore, tiers: Sequence[TierMining]
) -> tuple[BehaviorDifferenceEvidence, ...]:
    """Read each tier's reviewed differences, adversarial groups first, never pooling tiers.

    Raises:
        ValueError: A tier's differences belong to another tier or judge.
    """
    groups: list[BehaviorDifferenceEvidence] = []
    for tier in tiers:
        if tier.behavior_differences is None:
            continue
        report = load_behavior_differences(project, tier.behavior_differences)
        if (
            report.tier != tier.tier
            or report.judged != tier.judged_group
            or report.judge != tier.judge.identity
        ):
            raise ValueError("behavior differences belong to another mining tier or judge")
        hacks = adversarial_evidence(project, tier.behavior_differences)
        seen = {row.group_id for row in hacks}
        rest = compact_differences(project, tier.behavior_differences)
        groups.extend((*hacks, *(row for row in rest if row.group_id not in seen)))
    return tuple(groups)


def adversarial_references(
    project: ProjectStore, behavior: ArtifactInput | None
) -> tuple[ReferenceSource, ...]:
    """Return the mined adversarial families as a selector and critic reference, when any."""
    if behavior is None:
        return ()
    report = load_behavior_differences(project, behavior)
    if not adversarial_families(report, source=behavior):
        return ()
    return (adversarial_reference(project, behavior),)


def _linter(guidance: BehaviorGuidance, family_id: str) -> BehaviorLinter:
    """Return an accepted adversarial guidance entry as a library linter."""
    return BehaviorLinter(
        linter_id=guidance.locator.removeprefix("linter:"),
        revision=guidance.revision,
        title=guidance.title,
        description=guidance.statement,
        detection=guidance.detection or guidance.statement,
        evidence_needed=guidance.evidence_needed,
        applies_when=guidance.applies_when or "an evaluation or its evidence could be edited",
        required_when=guidance.required_when or "the task obliges a genuine passing result",
        examples=guidance.examples,
        false_positive_notes=guidance.false_positive_notes,
        source_evidence=(f"adversarial:{family_id}",),
    )


def adversarial_library(
    project: ProjectStore,
    spec: AdversarialRubricSpec,
    tiers: Sequence[TierMining],
    base: LibraryBinding | None,
    *,
    library_id: str,
    family: str,
    code_revision: str,
) -> LibraryBinding | None:
    """Author reviewed linters for every mined adversarial family and add them to the library.

    Args:
        project: Project holding every draw.
        spec: Linter author and independent reviewer (open-weight).
        tiers: The mining tiers whose reviewed differences are mined.
        base: The library version the family would author from.
        library_id: The library identity.
        family: The decision family whose rubric the version serves.
        code_revision: Producing source identity.

    Returns:
        A new library version holding every accepted adversarial linter, or ``base`` when no
        linter was accepted or every one is already in it.
    """
    added: list[BehaviorLinter] = []
    sources = []
    held = set() if base is None else {row.linter_id for row in base.library.linters}
    for tier in tiers:
        if tier.behavior_differences is None:
            continue
        report = load_behavior_differences(project, tier.behavior_differences)
        for mined in adversarial_families(report, source=tier.behavior_differences):
            linter = author_adversarial_linter(
                project,
                spec,
                mined,
                sources=(tier.behavior_differences,),
                code_revision=code_revision,
            )
            sources.extend(linter.transcripts)
            if linter.guidance is None:
                continue
            entry = _linter(linter.guidance, mined.family_id)
            if entry.linter_id not in held:
                held.add(entry.linter_id)
                added.append(entry)
    if not added:
        return base
    parent = empty_library(library_id) if base is None else base.library
    version = parent.model_copy(
        update={
            "version": parent.version + 1,
            "parent_sha256": parent.identity,
            "linters": (*parent.linters, *added),
            "changes": tuple(
                LibraryChange(
                    operation="add_linter",
                    subject=f"linter:{row.linter_id}",
                    evidence=row.source_evidence,
                    families=(family,),
                    rationale="reviewed adversarial linter mined from a reward-hacking signature",
                )
                for row in added
            ),
            "families": tuple(dict.fromkeys((*parent.families, family))),
        }
    )
    reference = save_library(
        project,
        version,
        parent=None if base is None else base.reference,
        sources=sources,
        code_revision=code_revision,
    )
    return LibraryBinding(reference=reference, library=load_library(project, reference))


def behavior_authoring_coalition(context: StageContext) -> CoalitionEvidence | None:
    """Give the early decision-state author the reviewed contrasts, adversarial ones first."""
    mining = MiningOutputs.model_validate(context.upstream["error-mining"].outputs)
    if not any(tier.behavior_differences is not None for tier in mining.tiers):
        return None
    project = context.project
    reports = tuple(
        ActorCriticReport.model_validate_json(
            project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
        )
        for tier in mining.tiers
    )
    judged = {
        tier.tier: JudgedStudy.model_validate_json(
            project.artifacts.read_bytes(tier.judged_group.artifact_id, "judged.json")
        )
        for tier in mining.tiers
    }
    return mining_coalition(
        project,
        reports,
        judged_groups=judged,
        behavior_differences=coalition_differences(project, mining.tiers),
    )

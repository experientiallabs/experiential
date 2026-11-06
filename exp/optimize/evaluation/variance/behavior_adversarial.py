"""Adversarial situation mining: reward-hacking signatures over the symbolic behavior diff.

The behavior diff (:mod:`~.behavior_differences`) pairs a judged pass with a judged failure of
the same task, aligns the decision where they diverge, and groups accepted pairs by identical
Added/Removed tag signatures (poor minus good). A group is *adversarial* when its poor side added
a reward-hacking behavior (:data:`HACK_ADDED`: ``modified_evaluation``, ``exploited_metric``,
``hid_failure``, ``unsupported_success_claim``); removing genuine work (:data:`GENUINE_REMOVED`:
``verified_result``, ``changed_implementation``) strengthens the signature. For example
``y = {changed_implementation, verified_result}`` against
``y' = {modified_evaluation, unsupported_success_claim}`` is the signature Added
``{modified_evaluation, unsupported_success_claim}``, Removed
``{changed_implementation, verified_result}``.

The tags are the decision model's observations; whether a change was a weakening was decided by
the frozen episode judge (the poor side failed) and by the independent pair audit. Nothing here
re-decides it, and a group is analysis for authors, never a reward.

- :func:`adversarial_families` keeps the adversarial groups and, for every pair, the
  :class:`DivergencePoint`: the poor attempt's pre-action checkpoint where the hack branch
  leaves the genuine branch, both windows' steps and events, and whether the hack scored high on
  the historical verifier (a judged failure the mechanical verifier passed).
- :func:`adversarial_evidence` gives the same groups to the authoring coalition (decision mining,
  then variant search through the authored families) in the existing
  :class:`~.behavior_differences.BehaviorDifferenceEvidence` shape.
- :func:`adversarial_reference` registers them as a read-only selector reference.
- :mod:`~.adversarial_rubric` turns each family into a reviewed behavior-library linter and
  checks the three-case acceptance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.behavior_differences import (
    BEHAVIOR_DIFFERENCES_TYPE,
    BehaviorDifferenceEvidence,
    BehaviorDifferenceGroup,
    BehaviorDifferenceReport,
)
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.evidence import ReferenceSource
from exp.optimize.evaluation.variance.pipeline_behaviors import compact_differences
from exp.optimize.evaluation.variance.reward_states import (
    FAMILY_MACHINE,
    FamilyState,
    Lifecycle,
    Transition,
)

ADVERSARIAL_PROTOCOL = "adversarial-signatures-v1"
"""Version of the hack signature rule and the divergence record."""

HACK_ADDED: tuple[str, ...] = (
    "modified_evaluation",
    "exploited_metric",
    "hid_failure",
    "unsupported_success_claim",
)
"""Behaviors whose addition on the failing side marks a reward-hacking signature."""

GENUINE_REMOVED: tuple[str, ...] = ("verified_result", "changed_implementation")
"""Genuine behaviors whose removal on the failing side strengthens the signature."""


class HackSignature(ContractModel):
    """The reward-hacking part of one Added/Removed signature.

    Attributes:
        added: Hack behaviors the failing side added, sorted.
        removed: Genuine behaviors it dropped, sorted.
    """

    added: tuple[str, ...] = Field(min_length=1)
    removed: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """A readable signature name, such as ``+modified_evaluation -verified_result``."""
        return " ".join((*(f"+{t}" for t in self.added), *(f"-{t}" for t in self.removed)))


def hack_signature(
    added: Sequence[str],
    removed: Sequence[str],
    *,
    hacks: Sequence[str] = HACK_ADDED,
    genuine: Sequence[str] = GENUINE_REMOVED,
) -> HackSignature | None:
    """Return the hack part of a signature, or None when it adds no hack behavior."""
    found = sorted(set(added) & set(hacks))
    if not found:
        return None
    return HackSignature(added=tuple(found), removed=tuple(sorted(set(removed) & set(genuine))))


class DivergencePoint(ContractModel):
    """Where one hack branch leaves its genuine branch.

    Attributes:
        pair_id: The reviewed pair.
        genuine_attempt: The judged pass.
        hack_attempt: The judged failure.
        point: The hack attempt's decision boundary, before its consequential action.
        checkpoint: The saved checkpoint at that boundary; decision mining starts here.
        genuine_steps: Step locators of the genuine window.
        hack_steps: Step locators of the hack window.
        genuine_events: Audited genuine-side events.
        hack_events: Audited hack-side events.
        high_scoring: Whether the historical verifier passed the hack (None when unknown).
        mechanism: The aligner's mechanism, as reviewed.
    """

    pair_id: str
    genuine_attempt: int
    hack_attempt: int
    point: int = Field(ge=0)
    checkpoint: ArtifactInput
    genuine_steps: tuple[str, ...]
    hack_steps: tuple[str, ...]
    genuine_events: tuple[str, ...]
    hack_events: tuple[str, ...]
    high_scoring: bool | None = None
    mechanism: str


class AdversarialFamily(ContractModel):
    """One adversarial signature group and its divergence points.

    Attributes:
        family_id: Stable identity (the diff group's identity under this protocol).
        group_id: The behavior-diff group.
        tier: Actor tier of the report.
        signature: The hack part of the signature.
        added: The group's full Added set.
        removed: The group's full Removed set.
        points: One divergence point per accepted pair.
        high_scoring: Pairs whose hack the historical verifier passed.
        source: The behavior-differences report.
    """

    family_id: str
    group_id: str
    tier: str
    signature: HackSignature
    added: tuple[str, ...]
    removed: tuple[str, ...]
    points: tuple[DivergencePoint, ...] = Field(min_length=1)
    high_scoring: int = Field(ge=0)
    source: ArtifactInput | None = None
    transitions: tuple[Transition, ...] = ()


def _divergence(report: BehaviorDifferenceReport, pair_id: str) -> DivergencePoint:
    """Read one accepted pair's divergence point from the report."""
    pair = next(row for row in report.pairs if row.pair_id == pair_id)
    alignment, audit = pair.alignment, pair.audit
    if (
        not pair.accepted
        or alignment is None
        or alignment.good_window is None
        or alignment.poor_window is None
        or audit is None
        or pair.point is None
        or pair.checkpoint is None
    ):
        raise ValueError(f"behavior pair {pair_id} is not an accepted, aligned contrast")
    attempts = {attempt.attempt: attempt for attempt in report.attempts}
    good, poor = attempts[pair.good], attempts[pair.poor]
    return DivergencePoint(
        pair_id=pair_id,
        genuine_attempt=pair.good,
        hack_attempt=pair.poor,
        point=pair.point,
        checkpoint=pair.checkpoint,
        genuine_steps=tuple(
            s.locator
            for s in good.trajectory.steps[alignment.good_window.start : alignment.good_window.stop]
        ),
        hack_steps=tuple(
            s.locator
            for s in poor.trajectory.steps[alignment.poor_window.start : alignment.poor_window.stop]
        ),
        genuine_events=audit.good_events,
        hack_events=audit.poor_events,
        high_scoring=poor.historical_verifier_pass,
        mechanism=alignment.mechanism,
    )


def adversarial_groups(
    groups: Sequence[BehaviorDifferenceGroup],
    *,
    hacks: Sequence[str] = HACK_ADDED,
    genuine: Sequence[str] = GENUINE_REMOVED,
) -> tuple[tuple[BehaviorDifferenceGroup, HackSignature], ...]:
    """Keep the groups whose signature adds a hack behavior, with that hack signature."""
    found: list[tuple[BehaviorDifferenceGroup, HackSignature]] = []
    for group in groups:
        signature = hack_signature(group.added, group.removed, hacks=hacks, genuine=genuine)
        if signature is not None:
            found.append((group, signature))
    return tuple(found)


def mining_lifecycle(
    report: BehaviorDifferenceReport,
    *,
    hacks: Sequence[str] = HACK_ADDED,
    genuine: Sequence[str] = GENUINE_REMOVED,
) -> Lifecycle[FamilyState]:
    """Move every diff group through :data:`~.reward_states.FAMILY_MACHINE` up to ``mined``.

    A group is adversarial when its signature adds a hack behavior, else not adversarial
    (terminal); an adversarial group is mined once its divergence points are read.
    """
    states = FAMILY_MACHINE.lifecycle()
    for group in report.groups:
        states.start(group.group_id)
        signature = hack_signature(group.added, group.removed, hacks=hacks, genuine=genuine)
        if signature is None:
            states.move(group.group_id, FamilyState.NOT_ADVERSARIAL, "adds no hack behavior")
            continue
        states.move(group.group_id, FamilyState.ADVERSARIAL, signature.name)
        states.move(group.group_id, FamilyState.MINED, f"{len(group.pairs)} divergence points")
    return states


def adversarial_families(
    report: BehaviorDifferenceReport,
    *,
    source: ArtifactInput | None = None,
    hacks: Sequence[str] = HACK_ADDED,
    genuine: Sequence[str] = GENUINE_REMOVED,
) -> tuple[AdversarialFamily, ...]:
    """Mine the adversarial families of one behavior-differences report.

    Groups keep the report's identical Added/Removed signatures. Families are ordered by how many
    hacks scored high on the historical verifier, then by pair count, then by group identity.

    Args:
        report: A loaded behavior-differences report.
        source: Its saved reference, recorded on every family.
        hacks: Behaviors whose addition marks a hack.
        genuine: Behaviors whose removal strengthens one.
    """
    states = mining_lifecycle(report, hacks=hacks, genuine=genuine)
    families: list[AdversarialFamily] = []
    for group, signature in adversarial_groups(report.groups, hacks=hacks, genuine=genuine):
        points = tuple(_divergence(report, pair_id) for pair_id in group.pairs)
        families.append(
            AdversarialFamily(
                family_id=stable_id(
                    "adversarial-family",
                    {"protocol": ADVERSARIAL_PROTOCOL, "group": group.group_id},
                ),
                group_id=group.group_id,
                tier=report.tier,
                signature=signature,
                added=group.added,
                removed=group.removed,
                points=points,
                high_scoring=sum(point.high_scoring is True for point in points),
                source=source,
                transitions=tuple(t for t in states.transitions if t.subject == group.group_id),
            )
        )
    families.sort(key=lambda row: (-row.high_scoring, -len(row.points), row.group_id))
    return tuple(families)


def adversarial_evidence(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    hacks: Sequence[str] = HACK_ADDED,
    genuine: Sequence[str] = GENUINE_REMOVED,
) -> tuple[BehaviorDifferenceEvidence, ...]:
    """Return the adversarial groups as coalition evidence for decision mining and variants.

    The shape is the reviewed contrast evidence the authoring coalition already reads, so the
    decision-state author and variant search see each hack branch, its genuine branch and the
    divergence checkpoint.
    """
    report = load_behavior_differences(project, reference)
    keep = {
        group.group_id
        for group, _ in adversarial_groups(report.groups, hacks=hacks, genuine=genuine)
    }
    return tuple(row for row in compact_differences(project, reference) if row.group_id in keep)


def adversarial_reference(project: ProjectStore, reference: ArtifactInput) -> ReferenceSource:
    """Register a report's adversarial families as a read-only selector reference."""
    report = load_behavior_differences(project, reference)
    families = adversarial_families(report, source=reference)
    record: JsonObject = {
        "protocol": ADVERSARIAL_PROTOCOL,
        "source": reference.model_dump(mode="json"),
        "families": [
            {**family.model_dump(mode="json"), "signature_name": family.signature.name}
            for family in families
        ],
        "transitions": [t.model_dump(mode="json") for t in mining_lifecycle(report).transitions],
        "authority": "reviewed reward-hacking hypotheses, never episode rewards",
    }
    identity = stable_id("adversarial-differences", {"record": sha256_json(record)})
    store = project.artifacts
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=BEHAVIOR_DIFFERENCES_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=store.read(reference.artifact_id).manifest.code_revision,
                inputs=(reference,),
            ),
            files={"adversarial.json": record},
        )
    return ReferenceSource(
        kind="behavior_differences",
        name="adversarial-differences",
        artifact=artifact_input(store.read(identity).manifest),
        file="adversarial.json",
    )

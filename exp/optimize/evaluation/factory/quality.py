"""Independent task-quality review, separate from mechanical verifier qualification."""

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectStore, artifact_input
from exp.simulation.evaluation.tasks import LabelAuthority

QualityAxis = Literal[
    "prompt_verifier_alignment",
    "verifier_intent_alignment",
    "prompt_distribution_alignment",
    "world_sufficiency_and_causality",
    "failure_attribution",
    "meaningful_difficulty",
]
QUALITY_AXES: tuple[QualityAxis, ...] = (
    "prompt_verifier_alignment",
    "verifier_intent_alignment",
    "prompt_distribution_alignment",
    "world_sufficiency_and_causality",
    "failure_attribution",
    "meaningful_difficulty",
)
QualityDisposition = Literal["accepted", "needs_review", "rejected"]


class QualityFinding(ContractModel):
    """One evidence-backed judgment over a protected task-quality axis.

    Attributes:
        axis: Exact criterion, independent of actor success or model tier.
        verdict: Pass, revise, drop or explicitly unavailable evidence.
        rationale: Why the evidence supports this judgment and its limits.
        sources: Exact supporting observations or authored requirements, never a bare score.
    """

    axis: QualityAxis
    verdict: Literal["pass", "revise", "drop", "unassessable"]
    rationale: str = Field(min_length=1)
    sources: tuple[ArtifactInput, ...] = Field(min_length=1)


class TaskQualityReview(ContractModel):
    """Operator-attested review of one exact task and its generation recipe registration.

    Attributes:
        task: Exact executable task, binding prompt, world and verifier.
        proposals: Exact proposal artifact, binding the author recipe and its original draw.
        reviewer: Accountable reviewer identity, not a claim of authenticated identity.
        authority: Existing label authority; generated proposals cannot approve themselves.
        approved: Operator attestation of a human review, never inferred from model output.
        findings: Every protected axis exactly once, with original evidence references.

    Human identity and expertise remain the operator's trust boundary. This is a scoped
    review attestation, not an independent proof of realism or a calibrated proxy model.
    """

    task: ArtifactInput
    proposals: ArtifactInput
    reviewer: str = Field(min_length=1)
    authority: LabelAuthority
    approved: bool = Field(default=False, strict=True)
    findings: tuple[QualityFinding, ...] = Field(min_length=len(QUALITY_AXES))

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Require complete criteria and refuse generated or witnessed self-approval."""
        axes = tuple(row.axis for row in self.findings)
        if len(set(axes)) != len(axes) or set(axes) != set(QUALITY_AXES):
            raise ValueError("task quality requires every protected axis exactly once")
        if self.approved and self.authority != "human":
            raise ValueError("task-quality approval requires an operator-attested human review")
        return self

    def disposition(self) -> QualityDisposition:
        """Do not equate mechanical success or an unapproved model review with good tasks."""
        if not self.approved or self.authority != "human":
            return "needs_review"
        if any(row.verdict == "drop" for row in self.findings):
            return "rejected"
        return "accepted" if all(row.verdict == "pass" for row in self.findings) else "needs_review"

    def sources(self) -> set[ArtifactInput]:
        """Return all immutable dependencies retained by the batch registration."""
        return {self.task, self.proposals, *(ref for row in self.findings for ref in row.sources)}


def validate_quality_reviews(
    project: ProjectStore,
    reviews: tuple[TaskQualityReview, ...],
    *,
    proposals: ArtifactInput,
    tasks: frozenset[ArtifactInput],
) -> None:
    """Verify population binding and evidence before dispatch or saved-result replay."""
    seen = set()
    for review in reviews:
        if review.proposals != proposals or review.task not in tasks:
            raise ValueError("task-quality review belongs to another task or generation recipe")
        if review.task in seen:
            raise ValueError("task-quality reviews must name distinct candidates")
        seen.add(review.task)
        for source in review.sources():
            if artifact_input(project.artifacts.read(source.artifact_id).manifest) != source:
                raise ValueError("task-quality source differs from its exact saved evidence")


def quality_disposition(
    reviews: tuple[TaskQualityReview, ...], task: ArtifactInput
) -> QualityDisposition:
    """Missing review is missing evidence, never implicit acceptance or rejection."""
    return next((row.disposition() for row in reviews if row.task == task), "needs_review")

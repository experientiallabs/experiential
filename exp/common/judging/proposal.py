"""Evidence-cited rubric proposal contracts and immutable persistence."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    stable_id,
)
from exp.common.judging.provenance import JudgingProvenanceError, resolve_artifact
from exp.common.judging.rubric import RubricDimension
from exp.common.models import ModelSnapshot
from exp.common.project import ArtifactCorruptionError, ProjectStore


class RubricProposalError(ValueError):
    """Raised when rubric proposal evidence is unsafe to persist or resume."""


class ProposedRubricDimension(ContractModel):
    """A rubric-card candidate with its source citations and possible overlap."""

    dimension: RubricDimension
    source_rollout_ids: tuple[ArtifactId, ...]
    evidence_span_ids: tuple[str, ...]
    overlap_with_dimension_ids: tuple[ArtifactId, ...] = ()

    @field_validator("source_rollout_ids", "evidence_span_ids")
    @classmethod
    def _require_nonempty_unique_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("proposed rubric dimensions require source citations")
        if len(set(value)) != len(value):
            raise ValueError("proposed rubric dimension citations must not repeat")
        return value

    @field_validator("overlap_with_dimension_ids")
    @classmethod
    def _require_unique_overlap_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("proposed rubric overlap IDs must not repeat")
        return value

    @model_validator(mode="after")
    def _reject_self_overlap(self) -> ProposedRubricDimension:
        if self.dimension.dimension_id in self.overlap_with_dimension_ids:
            raise ValueError("a proposed rubric dimension cannot overlap itself")
        return self


class RubricProposal(ContractModel):
    """One model-proposed set of diverse rubric cards grounded in rollout evidence."""

    proposal_id: ArtifactId
    source_task_set_id: ArtifactId
    proposer_model: ModelSnapshot
    prompt_id: str = Field(min_length=1, max_length=256)
    prompt_sha256: Sha256
    dimensions: tuple[ProposedRubricDimension, ...]
    successful_rollout_ids: tuple[ArtifactId, ...]
    failed_rollout_ids: tuple[ArtifactId, ...]
    source_lineage_ids: tuple[ArtifactId, ...]
    excluded_router_held_out_lineage_ids: tuple[ArtifactId, ...]

    @field_validator("dimensions")
    @classmethod
    def _require_unique_dimensions(
        cls, value: tuple[ProposedRubricDimension, ...]
    ) -> tuple[ProposedRubricDimension, ...]:
        if not value:
            raise ValueError("a rubric proposal needs at least one dimension")
        dimension_ids = tuple(item.dimension.dimension_id for item in value)
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("rubric proposal dimensions must have unique IDs")
        return value

    @field_validator(
        "successful_rollout_ids",
        "failed_rollout_ids",
        "source_lineage_ids",
        "excluded_router_held_out_lineage_ids",
    )
    @classmethod
    def _require_unique_source_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("rubric proposal source IDs must not repeat")
        return value

    @model_validator(mode="after")
    def _require_success_and_failure_evidence(self) -> RubricProposal:
        if not self.successful_rollout_ids or not self.failed_rollout_ids:
            raise ValueError("rubric proposals require successful and failed rollout evidence")
        if set(self.successful_rollout_ids).intersection(self.failed_rollout_ids):
            raise ValueError("a rollout cannot be both successful and failed rubric evidence")
        if not self.source_lineage_ids:
            raise ValueError("rubric proposals require source fit lineages")
        cited_rollouts = {
            rollout_id
            for dimension in self.dimensions
            for rollout_id in dimension.source_rollout_ids
        }
        known_rollouts = set(self.successful_rollout_ids).union(self.failed_rollout_ids)
        unknown_rollouts = cited_rollouts - known_rollouts
        if unknown_rollouts:
            raise ValueError("rubric proposal citations must name supplied representative rollouts")
        if not cited_rollouts.intersection(self.successful_rollout_ids):
            raise ValueError("rubric proposals must cite a successful rollout")
        if not cited_rollouts.intersection(self.failed_rollout_ids):
            raise ValueError("rubric proposals must cite a failed rollout")
        if set(self.source_lineage_ids).intersection(self.excluded_router_held_out_lineage_ids):
            raise ValueError(
                "rubric proposal source lineages must exclude router-held-out lineages"
            )
        dimension_ids = {item.dimension.dimension_id for item in self.dimensions}
        if any(set(item.overlap_with_dimension_ids) - dimension_ids for item in self.dimensions):
            raise ValueError("rubric proposal overlap IDs must name another proposed dimension")
        return self


class RubricProposalEvidence(ArtifactEnvelope):
    """One immutable persisted proposal that may become accepted rubric evidence."""

    proposal_evidence_id: ArtifactId
    source_task_set_id: ArtifactId
    proposal: RubricProposal

    @model_validator(mode="after")
    def _require_task_set_input(self) -> RubricProposalEvidence:
        if self.proposal.source_task_set_id != self.source_task_set_id:
            raise ValueError("proposal evidence must retain its proposal task-set identity")
        if tuple(item.artifact_id for item in self.inputs) != (self.source_task_set_id,):
            raise ValueError("proposal evidence must hash exactly its source task set")
        return self


def write_rubric_proposal_evidence(
    store: ProjectStore,
    *,
    proposal: RubricProposal,
    source_task_set_input: ArtifactInput,
    code_revision: str,
    created_at: datetime,
) -> RubricProposalEvidence:
    """Persist one reviewed-proposal input after its task-set manifest is verified.

    Args:
        store: Project store that owns the immutable proposal-evidence artifact.
        proposal: Structured model proposal whose citations seeded a human review.
        source_task_set_input: Manifest-derived input for the proposal's source task set.
        code_revision: Exact revision that froze the evidence artifact.
        created_at: Time the evidence artifact is materialized.

    Returns:
        The stored immutable proposal evidence, including safe idempotent retry recovery.

    Raises:
        RubricProposalError: The proposal conflicts with its task-set evidence or stored artifact.
    """
    try:
        _task_set, verified_task_set_input = resolve_artifact(
            store,
            artifact_id=source_task_set_input.artifact_id,
            expected_artifact_type="task-set",
            expected_input=source_task_set_input,
        )
    except JudgingProvenanceError as exc:
        raise RubricProposalError(
            "proposal evidence requires a completed immutable source task set"
        ) from exc
    if proposal.source_task_set_id != verified_task_set_input.artifact_id:
        raise RubricProposalError(
            "proposal source task set does not match verified task-set evidence"
        )
    evidence = RubricProposalEvidence(
        schema_version=1,
        created_at=created_at,
        inputs=(verified_task_set_input,),
        code_revision=code_revision,
        proposal_evidence_id=stable_id(
            "rubric-proposal-evidence",
            {
                "proposal": proposal.model_dump(mode="json"),
                "source_task_set_input": verified_task_set_input.model_dump(mode="json"),
                "code_revision": code_revision,
            },
        ),
        source_task_set_id=proposal.source_task_set_id,
        proposal=proposal,
    )
    try:
        stored, _ = store.artifacts.write_or_replay(
            artifact_id=evidence.proposal_evidence_id,
            artifact_type="rubric-proposal-evidence",
            envelope=evidence,
            envelope_path="proposal.json",
            envelope_type=RubricProposalEvidence,
            files={"proposal.json": canonical_json_bytes(evidence)},
        )
    except ArtifactCorruptionError as exc:
        raise RubricProposalError(
            "existing rubric proposal evidence cannot be resumed safely"
        ) from exc
    except ValueError as exc:
        raise RubricProposalError(
            "existing rubric proposal evidence conflicts with this proposal"
        ) from exc
    return stored

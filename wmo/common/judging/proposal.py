"""Model-assisted, evidence-cited rubric proposal services."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, Sha256, stable_id
from wmo.common.judging.prompts import PromptDefinition
from wmo.common.judging.rubric import RubricDimension
from wmo.common.models import ModelClient, ModelMessage, ModelRequest, ModelSnapshot, ToolCall
from wmo.common.rollouts import RolloutArtifact


class RubricProposalError(ValueError):
    """Raised when a rubric proposer returns unsafe or unsupported structured output."""


class RepresentativeRollout(ContractModel):
    """One fit-lineage rollout selected as successful or failed rubric evidence."""

    rollout: RolloutArtifact
    lineage_id: ArtifactId
    outcome: Literal["successful", "failed"]


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


class _RawProposal(ContractModel):
    """Strict structured response accepted from the configured rubric proposer."""

    dimensions: tuple[ProposedRubricDimension, ...]


class LMRubricProposer:
    """Propose diverse zero-to-five rubric cards from representative fit rollouts."""

    def __init__(self, model: ModelClient, prompt: PromptDefinition) -> None:
        """Bind one injected model client and immutable proposer prompt."""
        self._model = model
        self._prompt = prompt

    def propose(
        self,
        *,
        source_task_set_id: ArtifactId,
        representatives: Sequence[RepresentativeRollout],
        router_held_out_lineage_ids: Sequence[ArtifactId] = (),
    ) -> RubricProposal:
        """Propose rubric cards from successful and failed router-fit evidence.

        Args:
            source_task_set_id: Immutable representative task-set identifier.
            representatives: Selected production rollouts with fit-lineage outcome labels.
            router_held_out_lineage_ids: Frozen router-held-out lineages that must not contribute.

        Returns:
            Structured proposals that cite source rollouts and rollout spans.

        Raises:
            RubricProposalError: Input evidence or the model response is not safe to use.
        """
        held_out_lineages = set(router_held_out_lineage_ids)
        if not representatives:
            raise RubricProposalError("rubric proposal requires representative rollouts")
        if any(item.lineage_id in held_out_lineages for item in representatives):
            raise RubricProposalError(
                "router-held-out lineages cannot contribute to rubric proposals"
            )
        successful = tuple(
            sorted(
                {
                    item.rollout.rollout_id
                    for item in representatives
                    if item.outcome == "successful"
                }
            )
        )
        failed = tuple(
            sorted(
                {item.rollout.rollout_id for item in representatives if item.outcome == "failed"}
            )
        )
        if not successful or not failed:
            raise RubricProposalError(
                "rubric proposal requires at least one successful and one failed rollout"
            )
        response = self._model.complete(
            ModelRequest(
                messages=(
                    ModelMessage(role="system", content=self._prompt.text),
                    ModelMessage(role="user", content=_render_representatives(representatives)),
                ),
                temperature=0.0,
                maximum_output_tokens=4_096,
            )
        )
        parsed = _parse_proposal_response(response.output.content, response.output.tool_calls)
        _validate_proposal_citations(parsed, representatives)
        _validate_proposal_overlap(parsed)
        source_lineage_ids = tuple(sorted({item.lineage_id for item in representatives}))
        excluded_lineage_ids = tuple(sorted(set(router_held_out_lineage_ids)))
        return RubricProposal(
            proposal_id=stable_id(
                "rubric-proposal",
                {
                    "source_task_set_id": source_task_set_id,
                    "proposer_model": response.model.model_dump(mode="json"),
                    "prompt_id": self._prompt.prompt_id,
                    "prompt_sha256": self._prompt.sha256,
                    "dimensions": [item.model_dump(mode="json") for item in parsed.dimensions],
                    "successful_rollout_ids": successful,
                    "failed_rollout_ids": failed,
                    "source_lineage_ids": source_lineage_ids,
                    "excluded_router_held_out_lineage_ids": excluded_lineage_ids,
                },
            ),
            source_task_set_id=source_task_set_id,
            proposer_model=response.model,
            prompt_id=self._prompt.prompt_id,
            prompt_sha256=self._prompt.sha256,
            dimensions=parsed.dimensions,
            successful_rollout_ids=successful,
            failed_rollout_ids=failed,
            source_lineage_ids=source_lineage_ids,
            excluded_router_held_out_lineage_ids=excluded_lineage_ids,
        )


def _parse_proposal_response(content: str | None, tool_calls: tuple[ToolCall, ...]) -> _RawProposal:
    """Parse a strict JSON response without accepting unstructured model output."""
    if tool_calls:
        raise RubricProposalError("rubric proposer must return JSON text, not tool calls")
    if content is None:
        raise RubricProposalError("rubric proposer returned no structured JSON text")
    try:
        return _RawProposal.model_validate_json(content)
    except ValueError as exc:
        raise RubricProposalError("rubric proposer returned malformed structured output") from exc


def _validate_proposal_citations(
    proposal: _RawProposal, representatives: Sequence[RepresentativeRollout]
) -> None:
    """Require every cited rollout and span to exist in the supplied evidence."""
    spans_by_rollout = {
        item.rollout.rollout_id: {span.span_id for span in item.rollout.spans}
        for item in representatives
    }
    for proposed_dimension in proposal.dimensions:
        cited_rollouts = set(proposed_dimension.source_rollout_ids)
        unknown_rollouts = cited_rollouts - set(spans_by_rollout)
        if unknown_rollouts:
            raise RubricProposalError(
                "rubric proposer cited unknown rollout IDs: " + ", ".join(sorted(unknown_rollouts))
            )
        cited_spans = set().union(*(spans_by_rollout[item] for item in cited_rollouts))
        unknown_spans = set(proposed_dimension.evidence_span_ids) - cited_spans
        if unknown_spans:
            raise RubricProposalError(
                "rubric proposer cited unknown rollout span IDs: "
                + ", ".join(sorted(unknown_spans))
            )


def _validate_proposal_overlap(proposal: _RawProposal) -> None:
    """Require overlap references to point at another card in the same response."""
    dimension_ids = {item.dimension.dimension_id for item in proposal.dimensions}
    for proposed_dimension in proposal.dimensions:
        unknown = set(proposed_dimension.overlap_with_dimension_ids) - dimension_ids
        if unknown:
            raise RubricProposalError(
                "rubric proposer named unknown overlapping dimensions: "
                + ", ".join(sorted(unknown))
            )


def _render_representatives(representatives: Sequence[RepresentativeRollout]) -> str:
    """Render source rollout evidence in a structured, citation-friendly form."""
    rendered = []
    for item in representatives:
        rendered.append(
            {
                "rollout_id": item.rollout.rollout_id,
                "lineage_id": item.lineage_id,
                "outcome": item.outcome,
                "task_id": item.rollout.task_id,
                "final_output": item.rollout.final_output.model_dump(mode="json")
                if item.rollout.final_output is not None
                else None,
                "spans": [
                    {
                        "span_id": span.span_id,
                        "kind": span.kind.value,
                        "payload": span.payload,
                        "failure": span.failure.model_dump(mode="json")
                        if span.failure is not None
                        else None,
                    }
                    for span in item.rollout.spans
                ],
            }
        )
    return (
        "Propose deliberately diverse zero-to-five rubric dimensions. Return only a JSON object "
        "with a dimensions array. Each item must include dimension {dimension_id, name, "
        "description, anchors}, source_rollout_ids, evidence_span_ids, and "
        "overlap_with_dimension_ids. Each dimension needs exactly anchors zero through five. "
        "Cite both successful and failed evidence across the proposal.\n\n"
        + json.dumps(rendered, ensure_ascii=False, sort_keys=True)
    )

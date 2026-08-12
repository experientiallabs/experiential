"""Tests for evidence-cited model-assisted rubric proposal."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.judging import (
    LMRubricProposer,
    PromptDefinition,
    RepresentativeRollout,
    RubricProposalError,
)
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.rollouts import (
    ProductionSimulatorSnapshot,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationMode,
    StopReason,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


class _FakeProposer:
    """Deterministic common-model fake that returns one structured rubric proposal."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self.content),
            model=_model(),
            economics=_economics(),
        )


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="fake",
        model_id="rubric-proposer",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _economics() -> OperationEconomics:
    """Return an empty operation-economics record for a deterministic fake client."""
    return OperationEconomics()


def _rollout(rollout_id: str, span_id: str) -> RolloutArtifact:
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
        artifact_id=f"artifact-{rollout_id}",
        simulation_id="simulation-1",
        cell_id=f"cell-{rollout_id}",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id=rollout_id,
        trace_id=f"trace-{rollout_id}",
        evidence_source="production",
        source_run_id="production-run-1",
        task_id="task-1",
        candidate=_model(),
        agent_id="support-agent",
        simulator=ProductionSimulatorSnapshot(
            source=SourceIdentity(kind="production", source_id="trace-source", sha256=_DIGEST)
        ),
        spans=(
            RolloutSpan(
                span_id=span_id,
                kind=RolloutEventKind.MESSAGE,
                started_at=_TIME,
                ended_at=_TIME,
                payload={"text": f"evidence for {rollout_id}"},
            ),
        ),
        repeat=0,
        final_output=AssistantAction(content="Final response."),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=_economics(),
    )


def _proposal_response() -> str:
    anchors = [{"score": score, "description": f"Anchor {score}."} for score in range(6)]
    return json.dumps(
        {
            "dimensions": [
                {
                    "dimension": {
                        "dimension_id": "task-success",
                        "name": "Task success",
                        "description": "Whether the customer received the needed outcome.",
                        "anchors": anchors,
                    },
                    "source_rollout_ids": ["rollout-success", "rollout-failed"],
                    "evidence_span_ids": ["span-success", "span-failed"],
                    "overlap_with_dimension_ids": ["policy-compliance"],
                },
                {
                    "dimension": {
                        "dimension_id": "policy-compliance",
                        "name": "Policy compliance",
                        "description": "Whether the response follows support policy.",
                        "anchors": anchors,
                    },
                    "source_rollout_ids": ["rollout-success"],
                    "evidence_span_ids": ["span-success"],
                    "overlap_with_dimension_ids": ["task-success"],
                },
            ]
        }
    )


def test_proposer_uses_successful_and_failed_fit_rollouts_with_citations() -> None:
    """Every proposal card has complete anchors, valid citations, and explicit overlap links."""
    client = _FakeProposer(_proposal_response())
    proposer = LMRubricProposer(client, PromptDefinition.from_text("rubric-v1", "Propose scales."))

    proposal = proposer.propose(
        source_task_set_id="task-set-1",
        representatives=(
            RepresentativeRollout(
                rollout=_rollout("rollout-success", "span-success"),
                lineage_id="lineage-fit-success",
                outcome="successful",
            ),
            RepresentativeRollout(
                rollout=_rollout("rollout-failed", "span-failed"),
                lineage_id="lineage-fit-failed",
                outcome="failed",
            ),
        ),
        router_held_out_lineage_ids=("lineage-held-out",),
    )

    assert isinstance(client, ModelClient)
    assert proposal.successful_rollout_ids == ("rollout-success",)
    assert proposal.failed_rollout_ids == ("rollout-failed",)
    assert len(proposal.dimensions[0].dimension.anchors) == 6
    assert proposal.dimensions[0].overlap_with_dimension_ids == ("policy-compliance",)
    request_content = client.requests[0].messages[1].content
    assert request_content is not None
    assert "rollout-failed" in request_content


def test_proposer_rejects_held_out_or_unknown_evidence() -> None:
    """Router-held-out lineages and fabricated source spans cannot influence rubric drafts."""
    client = _FakeProposer(_proposal_response())
    proposer = LMRubricProposer(client, PromptDefinition.from_text("rubric-v1", "Propose scales."))
    held_out = RepresentativeRollout(
        rollout=_rollout("rollout-success", "span-success"),
        lineage_id="lineage-held-out",
        outcome="successful",
    )

    with pytest.raises(RubricProposalError, match="held-out"):
        proposer.propose(
            source_task_set_id="task-set-1",
            representatives=(
                held_out,
                RepresentativeRollout(
                    rollout=_rollout("rollout-failed", "span-failed"),
                    lineage_id="lineage-fit-failed",
                    outcome="failed",
                ),
            ),
            router_held_out_lineage_ids=("lineage-held-out",),
        )
    assert not client.requests

    invalid = _proposal_response().replace(
        '"evidence_span_ids": ["span-success", "span-failed"]',
        '"evidence_span_ids": ["fabricated-span"]',
    )
    client = _FakeProposer(invalid)
    proposer = LMRubricProposer(client, PromptDefinition.from_text("rubric-v1", "Propose scales."))
    with pytest.raises(RubricProposalError, match="unknown rollout span"):
        proposer.propose(
            source_task_set_id="task-set-1",
            representatives=(
                RepresentativeRollout(
                    rollout=_rollout("rollout-success", "span-success"),
                    lineage_id="lineage-fit-success",
                    outcome="successful",
                ),
                RepresentativeRollout(
                    rollout=_rollout("rollout-failed", "span-failed"),
                    lineage_id="lineage-fit-failed",
                    outcome="failed",
                ),
            ),
        )

"""Tests for canonical simulation artifact and rollout subtype contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import FailureCode, StructuredFailure
from wmo.common.models import ModelSnapshot, OperationEconomics
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationArtifactSet,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)

_DIGEST = "a" * 64


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="openai",
        model_id="gpt-5.4",
        capabilities_sha256=_DIGEST,
    )


def _rollout() -> RolloutArtifact:
    started_at = datetime(2026, 8, 11, tzinfo=UTC)
    return RolloutArtifact(
        schema_version=1,
        created_at=started_at,
        code_revision="e7aad17",
        artifact_id="rollout-artifact-1",
        simulation_id="simulation-1",
        cell_id="cell-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-1",
        trace_id="0123456789abcdef0123456789abcdef",
        evidence_source="world_model",
        source_run_id="run-1",
        task_id="task-1",
        candidate=_model(),
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            world_model=_model(),
        ),
        world_model=_model(),
        seed=7,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id="span-1",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                model=_model(),
            ),
        ),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
    )


def test_rollout_subtype_and_artifact_set_round_trip() -> None:
    """A completed rollout remains a simulation artifact with complete provenance."""
    rollout = _rollout()
    artifact_set = SimulationArtifactSet(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        artifact_set_id="rollout-set-1",
        simulation_id="simulation-1",
        artifact_ids=(rollout.artifact_id,),
        artifacts_path="rollouts.jsonl",
        artifacts_sha256=_DIGEST,
    )

    assert RolloutArtifact.model_validate_json(rollout.model_dump_json()) == rollout
    assert SimulationArtifactSet.model_validate_json(artifact_set.model_dump_json()) == artifact_set


def test_rollout_rejects_missing_simulation_digest_and_failure_details() -> None:
    """Simulated and failed episodes preserve exactly the required provenance evidence."""
    rollout = _rollout()
    with pytest.raises(ValidationError, match="require a simulation specification"):
        RolloutArtifact.model_validate({**rollout.model_dump(), "simulation_spec_sha256": None})
    with pytest.raises(ValidationError, match="require a structured failure"):
        RolloutArtifact.model_validate(
            {**rollout.model_dump(), "stop_reason": "failure", "failure": None}
        )
    production = RolloutArtifact.model_validate(
        {
            **rollout.model_dump(),
            "evidence_source": "production",
            "simulation_spec_sha256": None,
            "failure": StructuredFailure(
                code=FailureCode.PROVIDER,
                message="captured failure",
            ).model_dump(),
        }
    )

    assert production.evidence_source == "production"

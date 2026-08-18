"""Tests for canonical simulation artifact and rollout subtype contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureCode,
    JsonObject,
    SourceIdentity,
    StructuredFailure,
)
from wmo.common.models import (
    BillingSource,
    EmbeddingCostReservation,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.rollouts import (
    ProductionSimulatorSnapshot,
    ProviderFreeSourceProvenance,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SandboxSimulatorSnapshot,
    SimulationArtifactSet,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)

_DIGEST = "a" * 64


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-5.4",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _rollout() -> RolloutArtifact:
    """Build one complete grounded rollout fixture.

    Returns:
        Canonical world-model rollout with retrieval economics.
    """
    started_at = datetime(2026, 8, 11, tzinfo=UTC)
    plan_input = ArtifactInput(artifact_id="evaluation-plan", sha256=_DIGEST)
    spec_input = ArtifactInput(artifact_id="simulation-spec", sha256=_DIGEST)
    task_set_input = ArtifactInput(artifact_id="task-set", sha256=_DIGEST)
    fit_rag_input = ArtifactInput(artifact_id="fit-rag", sha256=_DIGEST)
    grounded_world_model_input = ArtifactInput(artifact_id="grounded-world-model", sha256=_DIGEST)
    binding = SimulationCellBinding(
        evaluation_plan_input=plan_input,
        task_set_input=task_set_input,
        fit_rag_input=fit_rag_input,
        grounded_world_model_input=grounded_world_model_input,
        task_set_tasks_sha256=_DIGEST,
        task_sha256=_DIGEST,
        candidate_alias="candidate-a",
        candidate=_model(),
        agent_id="customer-agent",
        repeat=0,
        world_model_alias="world-model-a",
        world_model=_model(),
        simulator_id="world-model-v1",
        prompt_id="world-prompt-v1",
        prompt_version="v1",
        prompt_sha256=_DIGEST,
        query_embedding=EmbeddingCostReservation(
            model=_model(),
            input_usd_per_million_tokens=0.0,
            maximum_attempts=1,
            maximum_input_tokens=1,
        ),
        simulation_spec_input=spec_input,
        simulation_spec_sha256=_DIGEST,
        simulation_inputs_sha256=_DIGEST,
    )
    return RolloutArtifact(
        schema_version=1,
        created_at=started_at,
        inputs=(
            plan_input,
            fit_rag_input,
            grounded_world_model_input,
            spec_input,
            task_set_input,
        ),
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
            prompt_version="v1",
            prompt_sha256=_DIGEST,
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
        retrieval_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
        simulation_binding=binding,
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
    """Require simulation digests and structured failure provenance.

    Raises:
        AssertionError: Invalid provenance is unexpectedly accepted.
    """
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
            "world_model": None,
            "simulator": ProductionSimulatorSnapshot(
                source=SourceIdentity(kind="production", source_id="trace-1", sha256=_DIGEST)
            ).model_dump(),
            "retrieval_economics": None,
            "failure": StructuredFailure(
                code=FailureCode.PROVIDER,
                message="captured failure",
            ).model_dump(),
        }
    )

    assert production.evidence_source == "production"
    assert rollout.world_model is not None
    with pytest.raises(ValidationError, match="identity must match"):
        RolloutArtifact.model_validate(
            {
                **rollout.model_dump(),
                "world_model": {
                    **rollout.world_model.model_dump(),
                    "model_id": "different-world-model",
                },
            }
        )


def _provider_free_production() -> JsonObject:
    """Return one provider-free production rollout payload for validator coverage.

    Returns:
        Payload of a historical production rollout whose spans carry no model identity.
    """
    rollout = _rollout()
    spans = tuple(
        {**span.model_dump(mode="json"), "model": None, "kind": "message"} for span in rollout.spans
    )
    return {
        **rollout.model_dump(mode="json"),
        "evidence_source": "production",
        "candidate": None,
        "provider_free_source": ProviderFreeSourceProvenance(
            checked_span_count=len(spans)
        ).model_dump(mode="json"),
        "spans": list(spans),
        "simulation_spec_sha256": None,
        "simulation_binding": None,
        "world_model": None,
        "simulator": ProductionSimulatorSnapshot(
            source=SourceIdentity(kind="production", source_id="trace-1", sha256=_DIGEST)
        ).model_dump(mode="json"),
        "retrieval_economics": None,
    }


def test_provider_free_production_rollout_states_absent_model_identity() -> None:
    """Accept historical production evidence that records no generator identity at all."""
    rollout = RolloutArtifact.model_validate(_provider_free_production())

    assert rollout.candidate is None
    assert rollout.provider_free_source is not None
    assert rollout.provider_free_source.reason == "source_trace_records_no_model_identity"
    assert rollout.provider_free_source.checked_span_count == len(rollout.spans)


def test_rollout_requires_exactly_one_generator_provenance() -> None:
    """Reject a rollout with neither, or both, generator identity and provider-free evidence.

    Raises:
        AssertionError: Ambiguous generator provenance is unexpectedly accepted.
    """
    payload = _provider_free_production()
    with pytest.raises(ValidationError, match="either a candidate model snapshot"):
        RolloutArtifact.model_validate({**payload, "provider_free_source": None})
    with pytest.raises(ValidationError, match="either a candidate model snapshot"):
        RolloutArtifact.model_validate({**payload, "candidate": _model().model_dump(mode="json")})


def test_provider_free_evidence_stays_production_and_span_exact() -> None:
    """Reject provider-free evidence outside production, or contradicted by span identity.

    Raises:
        AssertionError: Inconsistent provider-free evidence is unexpectedly accepted.
    """
    payload = _provider_free_production()
    with pytest.raises(ValidationError, match="only production rollouts"):
        RolloutArtifact.model_validate(
            {
                **payload,
                "evidence_source": "sandbox",
                "simulator": SandboxSimulatorSnapshot(
                    simulator_id="sandbox-v1",
                    environment_id="local-process",
                    environment_sha256=_DIGEST,
                ).model_dump(mode="json"),
            }
        )
    spans = payload["spans"]
    assert isinstance(spans, list)
    first = spans[0]
    assert isinstance(first, dict)
    with pytest.raises(ValidationError, match="must not contain span model identity"):
        RolloutArtifact.model_validate(
            {
                **payload,
                "spans": [{**first, "model": _model().model_dump(mode="json")}, *spans[1:]],
            }
        )
    with pytest.raises(ValidationError, match="must cover every rollout span"):
        RolloutArtifact.model_validate(
            {
                **payload,
                "provider_free_source": ProviderFreeSourceProvenance(
                    checked_span_count=len(spans) + 1
                ).model_dump(mode="json"),
            }
        )


def test_world_model_rollout_requires_retrieval_economics() -> None:
    """Reject world-model evidence that omits retrieval dispatch economics.

    Raises:
        AssertionError: The rollout unexpectedly accepts missing retrieval economics.
    """
    rollout = _rollout()
    with pytest.raises(ValidationError, match="require retrieval economics"):
        RolloutArtifact.model_validate({**rollout.model_dump(), "retrieval_economics": None})

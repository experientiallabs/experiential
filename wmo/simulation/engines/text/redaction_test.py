"""Tests for persistence-time secret redaction of simulated rollout evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from wmo.common.core.artifacts import (
    SECRET_REDACTION_PLACEHOLDER,
    ArtifactInput,
    FailureCode,
    StructuredFailure,
    assert_secret_free,
)
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    EmbeddingCostReservation,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.simulation.engines.text.redaction import redact_rollout_secrets

_DIGEST = "a" * 64
_OPENAI_SECRET = "sk-abcdefghijklmnopqrstuvwxyz123456"
_BEARER_SECRET = "Bearer abcdefghijklmnop"


def _model() -> ModelSnapshot:
    """Return one deterministic model snapshot fixture.

    Returns:
        Snapshot bound to fixed capability and connection digests.
    """
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-5.4",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _rollout(
    *,
    spans: tuple[RolloutSpan, ...],
    final_output: AssistantAction | None,
    stop_reason: StopReason = StopReason.COMPLETED,
    failure: StructuredFailure | None = None,
) -> RolloutArtifact:
    """Build one complete world-model rollout around the given evidence.

    Args:
        spans: Execution spans to persist.
        final_output: Optional visible terminal action.
        stop_reason: Terminal outcome of the episode.
        failure: Optional structured terminal failure.

    Returns:
        Fully bound world-model rollout artifact.
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
        spans=spans,
        final_output=final_output,
        stop_reason=stop_reason,
        failure=failure,
        candidate_economics=OperationEconomics(),
        retrieval_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
        simulation_binding=binding,
    )


def _span(payload_content: str) -> RolloutSpan:
    """Return one agent model-call span carrying the given response content.

    Args:
        payload_content: Simulated model output text placed in the span payload.

    Returns:
        Span whose payload nests the content under a response object.
    """
    started_at = datetime(2026, 8, 11, tzinfo=UTC)
    return RolloutSpan(
        span_id="span-1",
        kind=RolloutEventKind.AGENT_MODEL_CALL,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        payload={"response": {"output": {"content": payload_content}}},
        model=_model(),
    )


def test_rollout_secret_redaction_covers_spans_output_and_failure() -> None:
    """Generated secrets in spans, final output, and failures are replaced and counted."""
    rollout = _rollout(
        spans=(_span(f"run export API={_OPENAI_SECRET}"),),
        final_output=AssistantAction(
            content=f"use {_BEARER_SECRET} for auth",
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="execute",
                    arguments={"command": f"curl -H 'Authorization: {_BEARER_SECRET}'"},
                ),
            ),
        ),
        stop_reason=StopReason.FAILURE,
        failure=StructuredFailure(
            code=FailureCode.PROVIDER,
            message=f"provider rejected {_OPENAI_SECRET}",
            details={"hint": f"retry without {_BEARER_SECRET}"},
        ),
    )

    redacted = redact_rollout_secrets(rollout)

    assert redacted.secret_redaction_count == 5
    assert _OPENAI_SECRET not in redacted.model_dump_json()
    assert _BEARER_SECRET not in redacted.model_dump_json()
    assert redacted.spans[0].payload == {
        "response": {"output": {"content": f"run export API={SECRET_REDACTION_PLACEHOLDER}"}}
    }
    assert redacted.final_output is not None
    assert redacted.final_output.content == f"use {SECRET_REDACTION_PLACEHOLDER} for auth"
    assert redacted.final_output.tool_calls[0].arguments == {
        "command": f"curl -H 'Authorization: {SECRET_REDACTION_PLACEHOLDER}'"
    }
    assert redacted.failure is not None
    assert redacted.failure.message == f"provider rejected {SECRET_REDACTION_PLACEHOLDER}"
    assert redacted.failure.details == {"hint": f"retry without {SECRET_REDACTION_PLACEHOLDER}"}
    assert_secret_free(redacted)


def test_rollout_secret_redaction_redacts_span_failures() -> None:
    """Secrets inside per-span structured failures are replaced and counted."""
    started_at = datetime(2026, 8, 11, tzinfo=UTC)
    span = RolloutSpan(
        span_id="span-1",
        kind=RolloutEventKind.AGENT_MODEL_CALL,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        model=_model(),
        failure=StructuredFailure(
            code=FailureCode.PROVIDER,
            message=f"call leaked {_OPENAI_SECRET}",
        ),
    )
    rollout = _rollout(spans=(span,), final_output=AssistantAction(content="done"))

    redacted = redact_rollout_secrets(rollout)

    assert redacted.secret_redaction_count == 1
    assert redacted.spans[0].failure is not None
    assert redacted.spans[0].failure.message == f"call leaked {SECRET_REDACTION_PLACEHOLDER}"
    assert_secret_free(redacted)


def test_rollout_without_secrets_is_returned_unchanged() -> None:
    """Clean rollouts keep their exact identity and a zero audit count."""
    rollout = _rollout(
        spans=(_span("plain terminal output"),),
        final_output=AssistantAction(content="all done"),
    )

    redacted = redact_rollout_secrets(rollout)

    assert redacted is rollout
    assert redacted.secret_redaction_count == 0

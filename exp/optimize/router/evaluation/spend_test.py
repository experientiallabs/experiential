"""Tests for conservative finite-cost reconciliation of persisted simulation evidence."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    JsonValue,
    StructuredFailure,
)
from exp.common.models import (
    BillingSource,
    EmbeddingCostReservation,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from exp.common.project import ProjectStore
from exp.common.project.request_budget import RequestBudgetStore
from exp.common.rollouts import (
    UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY,
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from exp.optimize.evaluation.simulation import run_or_load_simulation
from exp.optimize.evaluation.simulation_test import _DurableMeteredClient, _TruncatedTransport
from exp.optimize.evaluation.spending import BudgetedCompletion
from exp.optimize.router.errors import RouterCompositionError
from exp.optimize.router.evaluation.spend import (
    _observed_rollout_spend,
    observed_rollout_spend,
    verified_simulation_spend,
)
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.engines.text.bindings import rollout_id_for_binding
from exp.simulation.engines.text.lineage_spend import lineage_spend
from exp.simulation.engines.text.rollout_support import rollout_spend
from exp.simulation.engines.text.simulator_test import (
    _cell,
    _completion_reservation,
    _persist_completion_contract,
    _persist_plan,
    _persist_task_set,
    _plan,
    _response,
    _ScriptedClient,
    _simulator,
    _snapshot,
    _spec,
    _task,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _model() -> ModelSnapshot:
    """Return one pinned provider model snapshot fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-5.4",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _binding() -> SimulationCellBinding:
    """Return one complete immutable cell binding fixture."""
    return SimulationCellBinding(
        evaluation_plan_input=ArtifactInput(artifact_id="evaluation-plan", sha256=_DIGEST),
        task_set_input=ArtifactInput(artifact_id="task-set", sha256=_DIGEST),
        fit_rag_input=ArtifactInput(artifact_id="fit-rag", sha256=_DIGEST),
        grounded_world_model_input=ArtifactInput(
            artifact_id="grounded-world-model", sha256=_DIGEST
        ),
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
        simulation_spec_input=ArtifactInput(artifact_id="simulation-spec", sha256=_DIGEST),
        simulation_spec_sha256=_DIGEST,
        simulation_inputs_sha256=_DIGEST,
    )


def _rollout(
    *,
    candidate_economics: OperationEconomics,
    stop_reason: StopReason = StopReason.COMPLETED,
    failure: StructuredFailure | None = None,
) -> RolloutArtifact:
    """Build one world-model rollout with a single dispatched candidate call.

    Args:
        candidate_economics: Combined candidate operation economics to persist.
        stop_reason: Terminal reason recorded for the episode.
        failure: Optional structured failure recorded with the evidence.

    Returns:
        Canonical rollout fixture bound to one dispatched candidate call span.
    """
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        inputs=(
            ArtifactInput(artifact_id="evaluation-plan", sha256=_DIGEST),
            ArtifactInput(artifact_id="fit-rag", sha256=_DIGEST),
            ArtifactInput(artifact_id="grounded-world-model", sha256=_DIGEST),
            ArtifactInput(artifact_id="simulation-spec", sha256=_DIGEST),
            ArtifactInput(artifact_id="task-set", sha256=_DIGEST),
        ),
        code_revision="test-revision",
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
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                model=_model(),
            ),
        ),
        stop_reason=stop_reason,
        failure=failure,
        candidate_economics=candidate_economics,
        retrieval_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
        simulation_binding=_binding(),
    )


def _unknown_spend_failure(
    *,
    reserved: float | None,
) -> StructuredFailure:
    """Return one persisted provider dispatch failure with permanently ambiguous spend.

    Args:
        reserved: Optional durable worst-case reservation persisted with the failure.

    Returns:
        Structured provider failure marking an unknown-spend dispatch window.
    """
    details: dict[str, JsonValue] = {
        "phase": "candidate_or_world_model",
        "provider_dispatch_unknown_spend": True,
        "retry_classification": "transport",
    }
    if reserved is not None:
        details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved
    return StructuredFailure(
        code=FailureCode.PROVIDER,
        message="text simulation provider call failed with ProviderTransportError",
        retryable=True,
        exception_type="ProviderTransportError",
        attribution=FailureAttribution.MODEL,
        details=details,
    )


def _observed(cost: float) -> OperationEconomics:
    """Return complete observed economics for one priced operation."""
    return OperationEconomics(
        usage=Usage(input_tokens=8, output_tokens=4),
        cost_usd=NumericMeasurement(value=cost, provenance="observed"),
    )


def test_mixed_success_and_unknown_spend_rollouts_reconcile_conservatively() -> None:
    """One unknown-spend failure charges its reservation without aborting priced peers."""
    succeeded = _rollout(candidate_economics=_observed(0.10))
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=_unknown_spend_failure(reserved=0.25),
    )

    total = math.fsum(observed_rollout_spend(item) for item in (succeeded, failed))

    assert observed_rollout_spend(succeeded) == 0.10
    assert observed_rollout_spend(failed) == 0.25
    assert total == pytest.approx(0.35)


def test_unknown_spend_failure_without_reservation_stays_fail_closed() -> None:
    """Ambiguous dispatch spend with no durable worst-case reservation cannot reconcile."""
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=_unknown_spend_failure(reserved=None),
    )

    with pytest.raises(RouterCompositionError, match="no persisted reservation"):
        observed_rollout_spend(failed)


def test_ordinary_unpriced_evidence_still_fails_closed() -> None:
    """A dispatched call without unknown-spend classification must stay fully priced."""
    unpriced = _rollout(candidate_economics=OperationEconomics())

    with pytest.raises(RouterCompositionError, match="not fully observed"):
        observed_rollout_spend(unpriced)


def test_stale_lease_failure_charges_its_persisted_whole_ceiling_barrier() -> None:
    """A stale paid-cell tombstone reconciles to its exact durable reservation."""
    details: dict[str, JsonValue] = {
        "phase": "paid_cell_stale_lease",
        "lease_id": "lease-1",
        UNKNOWN_DISPATCH_RESERVED_COST_KEY: 1.0,
    }
    failed = _rollout(
        candidate_economics=OperationEconomics(),
        stop_reason=StopReason.FAILURE,
        failure=StructuredFailure(
            code=FailureCode.BUDGET,
            message="a prior paid-cell claim expired after its owner exited",
            attribution=FailureAttribution.MODEL,
            details=details,
        ),
    )

    assert observed_rollout_spend(failed) == 1.0


@pytest.mark.parametrize("kind", ["pricing", "truncated", "invalid_usage"])
def test_unbounded_attempt_stays_unknown_after_retry_and_blocks_finite_judging(
    tmp_path: Path, kind: str
) -> None:
    """A paid failed ancestor stays unknown after success and cannot fund a finite judge.

    Args:
        tmp_path: Isolated project root retaining failed ancestors, receipts, and finite replay.
        kind: Pricing, truncated-response, or nonretryable invalid-usage failure to exercise.
    """
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(project, identity="unknown-ancestor", maximum_cost_usd=None)
    if kind == "pricing":
        candidate = _DurableMeteredClient(budget, role="candidate", failures=1)
    elif kind == "truncated":
        transport = _TruncatedTransport(1)
        candidate = BudgetedCompletion(
            OpenAICompatibleClient(
                model=_snapshot("candidate-a"),
                base_url="https://example.test/v1",
                api_key="fake-key",
                transport=transport,
                retry_policy=RetryPolicy(maximum_attempts=1, initial_delay_seconds=0),
            ),
            budget,
            _completion_reservation("candidate-a"),
            role="assistant:candidate-a",
        )
    else:
        response = _response("saved invalid usage", snapshot=_snapshot("candidate-a"))
        candidate = BudgetedCompletion(
            _ScriptedClient(
                [
                    response.model_copy(
                        update={
                            "economics": OperationEconomics(
                                usage=Usage(input_tokens=80_001, output_tokens=1),
                            )
                        }
                    )
                ]
            ),
            budget,
            _completion_reservation("candidate-a"),
            role="assistant:candidate-a",
        )
    world = _ScriptedClient(
        [_response('{"message":"done","terminal":true}', snapshot=_snapshot("world-model-a"))]
    )
    plan = _plan((_cell("cell-a", "task-a"),))
    plan_input = _persist_plan(project.artifacts, plan)
    tasks = _persist_task_set(project.artifacts, {"task-a": _task("task-a")})
    completion = _persist_completion_contract(project.artifacts)
    spec = _spec(plan_input, tasks, ("cell-a",), completion_contract_input=completion)
    simulator = _simulator(
        project.artifacts,
        plan,
        plan_input,
        tasks,
        candidate,
        world,
        completion_contract_input=completion,
        request_budget=budget,
    )
    first_set = simulator.run(spec)
    first = simulator._load_rollout(first_set.artifact_ids[0])
    original = project.artifacts.read_bytes(first.rollout_id, "rollout.json")
    assert first.failure is not None
    reservation = first.failure.details[UNKNOWN_DISPATCH_RESERVED_COST_KEY]
    assert isinstance(reservation, (int, float)) and reservation > 0
    assert first.failure.details[UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY] is False
    assert RequestBudgetStore(project, "unknown-ancestor").has_unbounded_liability()
    final_set = run_or_load_simulation(project, plan, spec, lambda project, plan: simulator)
    final = simulator._load_rollout(final_set.artifact_ids[0])
    if kind == "invalid_usage":
        assert final == first and final.retry_attempt == 0
        assert final.failure is not None and final.failure.exception_type == "ValueError"
        assert not final.failure.retryable
    else:
        assert final.retry_attempt == 1 and final.stop_reason == StopReason.COMPLETED
        assert observed_rollout_spend(final) > 0
    assert rollout_spend(first) is None
    assert (
        verified_simulation_spend(project, final_set, completion, allow_unknown_interrupted=True)
        is None
    )
    with pytest.raises(RouterCompositionError, match="lineage spend is unknown"):
        verified_simulation_spend(project, final_set, completion)
    finite = RequestBudget(project, identity="unknown-ancestor", maximum_cost_usd=100)

    def judge() -> str:
        """Reject any dispatch past the unresolved-liability admission fence."""
        pytest.fail("finite judge dispatched with an unbounded failed ancestor")

    with (
        finite.scope("new-judgment"),
        pytest.raises(ValueError, match="resolved earlier charges"),
    ):
        finite.call(
            role="judge",
            fingerprint=_DIGEST,
            maximum_cost_usd=0.01,
            operation=judge,
            encode=lambda value: value,
            decode=lambda value: value,
            charge=lambda _: 0.01,
        )
    assert project.artifacts.read_bytes(first.rollout_id, "rollout.json") == original


@pytest.mark.parametrize("component", ["candidate", "retrieval", "orchestration"])
def test_unknown_reservation_never_skips_recorded_economics_validation(component: str) -> None:
    """Unbounded failure evidence cannot hide negative prices or forged cost provenance.

    Args:
        component: Candidate, retrieval, or orchestration economics containing a negative cost.
    """
    failure = _unknown_spend_failure(reserved=0.25)
    failure = failure.model_copy(
        update={
            "details": {
                **failure.details,
                UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY: False,
            }
        }
    )
    rollout = _rollout(candidate_economics=_observed(0.1), failure=failure)
    rollout = rollout.model_copy(update={f"{component}_economics": _observed(-0.1)})
    with pytest.raises(RouterCompositionError, match="spend"):
        _observed_rollout_spend(rollout)


def test_unknown_current_attempt_still_validates_its_priced_failed_ancestor(tmp_path: Path) -> None:
    """A current unknown charge does not short-circuit validation of a prior retry's cost.

    Args:
        tmp_path: Isolated artifact root containing the malformed priced ancestor.
    """
    project = ProjectStore(tmp_path, "project-a")
    binding = _binding()
    parent_id = rollout_id_for_binding(binding, attempt=0)
    parent = _rollout(candidate_economics=_observed(-1)).model_copy(
        update={
            "artifact_id": parent_id,
            "rollout_id": parent_id,
        }
    )
    project.artifacts.write_json(
        artifact_id=parent_id,
        artifact_type="rollout",
        envelope=parent,
        files={"rollout.json": parent},
    )
    failure = _unknown_spend_failure(reserved=0.25)
    failure = failure.model_copy(
        update={
            "details": {
                **failure.details,
                UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY: False,
            }
        }
    )
    current = _rollout(candidate_economics=OperationEconomics(), failure=failure).model_copy(
        update={"retry_attempt": 1}
    )
    with pytest.raises(RouterCompositionError, match="not fully observed"):
        lineage_spend(project.artifacts, (current,), measure=_observed_rollout_spend)

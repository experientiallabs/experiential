"""Read-only price planning for a frozen worker/scenario evaluation matrix."""

from __future__ import annotations

import math

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.judging import verify_persisted_calibration
from exp.common.models import CompletionCostReservation
from exp.common.project import ProjectStore
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.contracts import EvaluationSetup
from exp.optimize.router.evaluation.build import completed_project_build
from exp.optimize.router.evaluation.setup import verify_router_evaluation_setup
from exp.simulation.engines.text.grounding import maximum_query_reservation
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from exp.simulation.specs import load_simulation_completion_contract


class EvaluationCostComponent(ContractModel):
    """Planning estimate and hard context/retry bound for one already prepared stage."""

    estimated_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    maximum_cost_usd: float = Field(ge=0, allow_inf_nan=False)


class EvaluationCostPlan(ContractModel):
    """USD estimate bound to immutable execution and judge request settings.

    Mining and grounding are already completed inputs to this API and are not included.
    Hosting must add their quote before presenting the entire trace-to-report price.
    Credit conversion and promotions are hosting concerns, never engine price inputs.
    """

    quote_sha256: Sha256
    scenario_count: int = Field(gt=0)
    worker_count: int = Field(gt=0)
    judgment_count: int = Field(gt=0)
    workers: EvaluationCostComponent
    simulation: EvaluationCostComponent
    retrieval: EvaluationCostComponent
    judge: EvaluationCostComponent
    estimated_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    maximum_cost_usd: float = Field(ge=0, allow_inf_nan=False)


def estimate_model_evaluation(
    project: ProjectStore,
    setup: EvaluationSetup,
    *,
    judge_request: CompletionCostReservation,
    judge_calls_per_rollout: int = 1,
) -> EvaluationCostPlan:
    """Price all workers on the same scenarios without any writes or provider requests.

    Args:
        project: Completed grounded project with immutable task and model evidence.
        setup: Exact settings that will be passed to ``evaluate_models``.
        judge_request: Runtime-enforced request ceiling for the persisted LM judge.
        judge_calls_per_rollout: One for scalar judging, two for counterbalanced pairwise.

    Returns:
        A content-bound planning estimate plus an absolute token/retry bound. The maximum
        includes every permitted cell attempt, simulation step, retrieval and judge request.

    Raises:
        ValueError: A price, identity, completion bound or required input is unavailable.
    """
    if judge_calls_per_rollout not in (1, 2):
        raise ValueError("judge calls per rollout must be one or two")
    if setup.observed_cells:
        raise ValueError("model evaluation requires fresh rollouts for every selected worker")
    if setup.simulation_completion_input is None:
        raise ValueError("evaluation estimates require frozen completion reservations")
    completed = completed_project_build(project)
    tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
    calibration, calibration_input = verify_persisted_calibration(
        project, setup.simulation_protocol.judge_calibration_id
    )
    if (
        calibration.status != setup.judgment_status
        or calibration.rubric_id != setup.simulation_protocol.rubric_id
    ):
        raise ValueError("evaluation judge status or rubric differs from persisted evidence")
    verify_router_evaluation_setup(
        completed=completed,
        fit_rag_input=setup.fit_rag_input,
        grounded_world_model_input=setup.world_model_settings.grounded_world_model_input,
        production_protocol=setup.production_protocol,
        simulation_protocol=setup.simulation_protocol,
        rubric_id=calibration.rubric_id,
        calibration_id=calibration.calibration_id,
    )
    if calibration.judge_model != judge_request.model:
        raise ValueError("judge reservation differs from the persisted judge model")
    contract, pointer = load_simulation_completion_contract(
        project.artifacts, setup.simulation_completion_input.artifact_id
    )
    if pointer != setup.simulation_completion_input:
        raise ValueError("evaluation completion reservation pointer changed")
    requests = {item.candidate_alias: item.request for item in contract.candidate_requests}
    if set(requests) != {item.alias for item in setup.candidates} or any(
        requests[item.alias].model != item.model for item in setup.candidates
    ):
        raise ValueError("completion reservations differ from selected worker models")
    if (
        contract.world_model_alias != setup.world_model_settings.world_model_alias
        or contract.world_model_request.model != setup.simulation_protocol.world_model
    ):
        raise ValueError("completion reservation differs from the simulation model")
    retrieval = setup.world_model_settings.query_embedding
    if retrieval is None:
        raise ValueError("evaluation estimates require an explicit query embedding reservation")
    count = len(tasks)
    workers = len(setup.candidates)
    steps = count * setup.maximum_steps
    worker_cost = _sum_completion(tuple(requests.values()), steps)
    world_cost = _sum_completion((contract.world_model_request,), steps * workers)
    retrieval_reservation = maximum_query_reservation(retrieval).cost_usd
    assert retrieval_reservation is not None
    retrieval_cost = steps * workers * retrieval_reservation.value
    retrieval_component = EvaluationCostComponent(
        estimated_cost_usd=retrieval_cost,
        maximum_cost_usd=retrieval_cost * MAXIMUM_CELL_ATTEMPTS,
    )
    judge_count = count * workers
    judge_component = _sum_completion(
        (judge_request,), judge_count * judge_calls_per_rollout, cell_attempts=1
    )
    components = (worker_cost, world_cost, retrieval_component, judge_component)
    return EvaluationCostPlan(
        quote_sha256=sha256_json(
            {
                "version": 1,
                "task_set": completed.task_set.model_dump(mode="json"),
                "calibration": calibration_input.model_dump(mode="json"),
                "setup": setup.model_dump(mode="json"),
                "judge_request": judge_request.model_dump(mode="json"),
                "judge_calls_per_rollout": judge_calls_per_rollout,
            }
        ),
        scenario_count=count,
        worker_count=workers,
        judgment_count=judge_count,
        workers=worker_cost,
        simulation=world_cost,
        retrieval=retrieval_component,
        judge=judge_component,
        estimated_cost_usd=math.fsum(item.estimated_cost_usd for item in components),
        maximum_cost_usd=math.fsum(item.maximum_cost_usd for item in components),
    )


def _sum_completion(
    requests: tuple[CompletionCostReservation, ...],
    calls: int,
    *,
    cell_attempts: int = MAXIMUM_CELL_ATTEMPTS,
) -> EvaluationCostComponent:
    """Use canonical cache-aware request pricing, distinguishing estimates from hard limits."""
    return EvaluationCostComponent(
        estimated_cost_usd=calls
        * math.fsum(request.expected_maximum_call_cost_usd() for request in requests),
        maximum_cost_usd=calls
        * cell_attempts
        * math.fsum(request.absolute_maximum_call_cost_usd() for request in requests),
    )

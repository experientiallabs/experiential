"""Read-only price planning for a frozen worker/scenario evaluation matrix."""

from __future__ import annotations

import math

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.judging import verify_persisted_calibration
from exp.common.models import CompletionCostReservation
from exp.common.project import ProjectStore
from exp.common.tasks import load_task_set
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.contracts import EvaluationSetup
from exp.optimize.evaluation.usage_estimate import expected_completion_cost, task_usage
from exp.optimize.router.evaluation.build import completed_project_build
from exp.optimize.router.evaluation.setup import verify_router_evaluation_setup
from exp.simulation.engines.text.grounding import maximum_query_reservation
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.specs import load_simulation_completion_contract


class EvaluationCostComponent(ContractModel):
    """Planning estimate and hard context/retry bound for one already prepared stage.

    Attributes:
        estimated_cost_usd: Nonnegative finite planning estimate.
        maximum_cost_usd: Nonnegative finite token and retry ceiling.
    """

    estimated_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    maximum_cost_usd: float = Field(ge=0, allow_inf_nan=False)


class EvaluationCostPlan(ContractModel):
    """USD estimate bound to immutable execution and judge request settings.

    Mining and grounding are already completed inputs to this API and are not included.
    Hosting must add their quote before presenting the entire trace-to-report price.
    Credit conversion and promotions are hosting concerns, never engine price inputs.

    Attributes:
        quote_sha256: Digest of the immutable inputs and judge reservation.
        scenario_count: Positive number of distinct scenarios.
        worker_count: Positive number of selected worker models.
        judgment_count: Positive number of cell judgments.
        workers: Worker-model estimates and maximum spend.
        simulation: World-model estimates and maximum spend.
        retrieval: Retrieval-embedding estimates and maximum spend.
        judge: Judge-model estimates and maximum spend.
        estimated_cost_usd: Nonnegative finite sum of stage estimates.
        maximum_cost_usd: Nonnegative finite sum of stage ceilings.
        captured_turns: Source-sized assistant turns per scenario matrix, before model repeats.
        measured_turns: Those turns with recorded provider token counts.
        estimate_basis: Human-readable assumptions for the expected workload.
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
    captured_turns: float = Field(ge=0, allow_inf_nan=False)
    measured_turns: float = Field(ge=0, allow_inf_nan=False)
    estimate_basis: str


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
    traces = load_trace_dataset(project.artifacts, completed.trace_dataset.artifact_id).traces
    by_id = {trace.trace_id: trace for trace in traces}
    rag = load_rag_index(project.artifacts, setup.fit_rag_input.artifact_id)
    usage = tuple(
        task_usage(
            task,
            tuple(by_id[identity] for identity in task.source_trace_ids),
            rag.transitions,
            top_k=rag.index.default_top_k,
            maximum_steps=setup.maximum_steps,
            maximum_query_tokens=retrieval.maximum_input_tokens,
        )
        for task in tasks
    )
    worker_estimate = setup.repeats * math.fsum(
        expected_completion_cost(request, item.assistant_input, item.assistant_output)
        for request in requests.values()
        for item in usage
    )
    worker_maximum = (
        count
        * setup.repeats
        * MAXIMUM_CELL_ATTEMPTS
        * math.fsum(
            (
                request.maximum_attempts
                * setup.maximum_steps
                * request.maximum_input_tokens
                * max(
                    request.input_usd_per_million_tokens,
                    request.cached_input_usd_per_million_tokens,
                    request.cache_write_usd_per_million_tokens,
                )
                + (
                    min(
                        setup.maximum_steps * request.maximum_output_tokens,
                        setup.maximum_rollout_output_tokens,
                    )
                    + setup.maximum_steps
                    * (request.maximum_attempts - 1)
                    * min(request.maximum_output_tokens, setup.maximum_rollout_output_tokens)
                )
                * request.output_usd_per_million_tokens
            )
            / 1_000_000
            for request in requests.values()
        )
    )
    worker_cost = EvaluationCostComponent(
        estimated_cost_usd=min(worker_estimate, worker_maximum),
        maximum_cost_usd=worker_maximum,
    )
    world_maximum = (
        count
        * setup.repeats
        * setup.maximum_steps
        * workers
        * MAXIMUM_CELL_ATTEMPTS
        * contract.world_model_request.absolute_maximum_call_cost_usd()
    )
    world_cost = EvaluationCostComponent(
        estimated_cost_usd=min(
            world_maximum,
            setup.repeats
            * workers
            * math.fsum(
                expected_completion_cost(
                    contract.world_model_request, item.world_input, item.world_output
                )
                for item in usage
            ),
        ),
        maximum_cost_usd=world_maximum,
    )
    retrieval_reservation = maximum_query_reservation(retrieval).cost_usd
    assert retrieval_reservation is not None
    tool_tasks = sum(bool(task.tools) for task in tasks)
    # Every tool action consumes generated tokens. The cumulative output budget applies
    # once per rollout, not once per step; this bound never imposes a new tool-call limit.
    query_count = setup.repeats * sum(
        tool_tasks
        * min(
            setup.maximum_steps * request.maximum_output_tokens, setup.maximum_rollout_output_tokens
        )
        + (count - tool_tasks) * setup.maximum_steps
        for request in requests.values()
    )
    retrieval_maximum = query_count * retrieval_reservation.value * MAXIMUM_CELL_ATTEMPTS
    retrieval_component = EvaluationCostComponent(
        estimated_cost_usd=min(
            retrieval_maximum,
            setup.repeats
            * workers
            * math.fsum(
                item.query_input * retrieval.input_usd_per_million_tokens / 1_000_000
                for item in usage
            ),
        ),
        maximum_cost_usd=retrieval_maximum,
    )
    judge_count = count * workers * setup.repeats
    judge_maximum = (
        judge_count * judge_calls_per_rollout * judge_request.absolute_maximum_call_cost_usd()
    )
    judge_component = EvaluationCostComponent(
        estimated_cost_usd=min(
            judge_maximum,
            workers
            * setup.repeats
            * judge_calls_per_rollout
            * math.fsum(
                expected_completion_cost(
                    judge_request,
                    min(item.judge_input, judge_request.maximum_input_tokens),
                    min(512, judge_request.maximum_output_tokens),
                )
                for item in usage
            ),
        ),
        maximum_cost_usd=judge_maximum,
    )
    components = (worker_cost, world_cost, retrieval_component, judge_component)
    return EvaluationCostPlan(
        quote_sha256=sha256_json(
            {
                "version": 2,
                "trace_dataset": completed.trace_dataset.model_dump(mode="json"),
                "task_set": completed.task_set.model_dump(mode="json"),
                "calibration": calibration_input.model_dump(mode="json"),
                "setup": setup.model_dump(mode="json"),
                "judge_request": judge_request.model_dump(mode="json"),
                "judge_calls_per_rollout": judge_calls_per_rollout,
            }
        ),
        captured_turns=math.fsum(item.turns for item in usage),
        measured_turns=math.fsum(item.measured_turns for item in usage),
        estimate_basis=(
            "Captured requests and tool responses; missing usage uses approximately four "
            "UTF-8 bytes/token. World prompts include average eligible RAG examples. "
            "Judge assumes a transcript plus 1,024 framing tokens and 512 output tokens. "
            "No assumed cache savings or retries. Future reasoning, world state and "
            "model behavior may change usage."
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

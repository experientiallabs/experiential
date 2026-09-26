"""Credential-free preparation of a catalog-backed standalone model evaluation."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    sorted_unique_inputs,
    stable_id,
)
from exp.common.evaluations import EvaluationProtocol
from exp.common.judging import verify_persisted_calibration
from exp.common.judging.provenance import read_artifact_json
from exp.common.models import (
    CandidateTokenPrice,
    CompletionCostReservation,
    ModelCatalog,
    RoutedCandidateSnapshot,
    SetupRole,
    persist_pricing_snapshot,
    serves_role,
)
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.contracts import EvaluationSetup
from exp.optimize.evaluation.judge_selection import select_judge_model
from exp.optimize.evaluation.planning import EvaluationCostPlan, estimate_model_evaluation
from exp.optimize.router.automatic.provisional import (
    _judge_request_reservation,
    prepare_hosted_provisional_judge,
)
from exp.optimize.router.automatic.reservations import (
    retrieval_embedding_reservation,
    simulation_completion_reservations,
    simulation_input_token_estimate,
)
from exp.optimize.router.evaluation.build import completed_project_build
from exp.optimize.router.judging.contracts import (
    JudgeSetupArtifact,
    ManualJudgeSetupArtifact,
    ProvisionalJudgeSetupArtifact,
)
from exp.runtime.agents import agent_factory_sha256
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.engines.text import WORLD_MODEL_TEXT_PROMPT_ID, WORLD_MODEL_TEXT_PROMPT_VERSION
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.specs import WorldModelSettings, persist_simulation_completion_contract
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


class ModelEvaluationOptions(ContractModel):
    """Bounded execution controls independent of router fitting and activation.

    Attributes:
        maximum_steps: Positive turn ceiling, default 100.
        maximum_rollout_output_tokens: Total generation ceiling, default one million tokens.
        maximum_concurrency: Positive simultaneous-rollout limit, default eight.
        repeats: Positive independent repeats per scenario/model pair, default one.
        maximum_output_tokens: Optional per-call limit; omission uses published model limits
            or the rollout budget within the context window when no output limit is published.
        maximum_judge_input_tokens: Optional input ceiling; omission uses the judge context
            capacity minus its output reservation.
        maximum_judge_output_tokens: Positive judge output reservation, default 8,192.
        maximum_retrieval_query_tokens: Positive per-query embedding limit, default 32,768.
        seed: Reproducible scenario seed, default zero.
    """

    maximum_steps: int = Field(default=100, ge=1)
    maximum_rollout_output_tokens: int = Field(default=1_000_000, gt=0)
    maximum_concurrency: int = Field(default=8, ge=1)
    repeats: int = Field(default=1, ge=1)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    maximum_judge_input_tokens: int | None = Field(default=None, gt=0)
    maximum_judge_output_tokens: int = Field(default=8_192, gt=0)
    maximum_retrieval_query_tokens: int = Field(default=32_768, gt=0)
    seed: int = 0


class PreparedModelEvaluation(ContractModel):
    """Serializable frozen execution inputs and engine-owned price plan for hosting.

    Attributes:
        setup: Frozen model, environment, judge and execution inputs.
        judge_setup: Immutable authored or default judge setup.
        judge_request: Runtime-enforced judge request reservation.
        embedder_alias: Nonempty alias of the retrieval embedding model.
        agent_factory_sha256: Digest of the exact worker runtime configuration.
        redacted_field_names: Project privacy fields pinned before execution.
        cost: Engine-owned immutable stage estimates and maximum provider spend.
    """

    setup: EvaluationSetup
    judge_setup: ArtifactInput
    judge_request: CompletionCostReservation
    embedder_alias: str = Field(min_length=1)
    agent_factory_sha256: Sha256
    redacted_field_names: tuple[str, ...]
    cost: EvaluationCostPlan


def read_evaluation_judge(project: ProjectStore, pointer: ArtifactInput) -> JudgeSetupArtifact:
    """Load an exact authored or default judge setup without manufacturing calibration.

    Args:
        project: Owner of the immutable judge graph.
        pointer: Exact setup manifest selected by the caller.

    Returns:
        The verified typed setup envelope.

    Raises:
        ValueError: The manifest, type, project or envelope binding differs.
    """
    stored = project.artifacts.read(pointer.artifact_id)
    if artifact_input(stored.manifest) != pointer:
        raise ValueError("evaluation judge setup manifest changed; prepare a new evaluation")
    types = {
        "manual-judge-setup": ManualJudgeSetupArtifact,
        "provisional-judge-setup": ProvisionalJudgeSetupArtifact,
    }
    model_type = types.get(stored.manifest.artifact_type)
    if model_type is None:
        raise ValueError("evaluation requires an authored or default judge setup artifact")
    setup, verified_pointer = read_artifact_json(
        project,
        artifact_id=pointer.artifact_id,
        expected_artifact_type=stored.manifest.artifact_type,
        relative_path="setup.json",
        model_type=model_type,
    )
    if (
        verified_pointer != pointer
        or setup.setup_id != pointer.artifact_id
        or setup.project_id != project.paths.project_id
    ):
        raise ValueError("evaluation judge setup belongs to another project or artifact")
    return setup


def prepare_model_evaluation(
    project: ProjectStore,
    catalog: ModelCatalog,
    worker_aliases: tuple[str, ...],
    *,
    continuation_of: str | None = None,
    run_id: str | None = None,
    judge_setup: ArtifactInput | None = None,
    judge_alias: str | None = None,
    calibration_id: str | None = None,
    embedder_alias: str,
    options: ModelEvaluationOptions,
    created_at: datetime,
    code_revision: str,
    progress: ProgressHook | None = None,
) -> PreparedModelEvaluation:
    """Freeze a completed project's worker matrix and estimate it without provider calls.

    Args:
        project: Completed grounded project containing the selected judge evidence.
        catalog: Secret-free catalog with explicit model capabilities and prices.
        worker_aliases: Two or more distinct worker aliases, never an incumbent or router.
        continuation_of: Prior simulation ID to continue under increased budgets.
        run_id: Optional distinct experiment identity; identical settings can run fresh evidence.
        judge_setup: Authored judge setup manifest, or omit both judge arguments for task success.
        judge_alias: Optional model override retaining the syllabus with provisional calibration.
        calibration_id: Verified calibration identity paired with an explicit judge setup.
        embedder_alias: Catalog alias matching the completed fit-RAG embedder.
        options: Bounded execution controls.
        created_at: Timestamp for newly persisted immutable pricing/contracts.
        code_revision: Exact engine producer revision.
        progress: Optional observer of verification, cost estimation, and artifact stages.

    Returns:
        Frozen setup and prepared-project quote suitable for credit admission by a host.

    Raises:
        ValueError: Evidence, identity, capacity or pricing is unavailable or inconsistent.
    """
    if len(worker_aliases) < 2 or len(set(worker_aliases)) != len(worker_aliases):
        raise ValueError("select at least two distinct worker models")
    report(progress, "Verifying built project")
    completed = completed_project_build(project)
    if (judge_setup is None) != (calibration_id is None):
        raise ValueError("supply both judge setup and calibration, or omit both for task success")
    report(progress, "Preparing judge")
    if judge_setup is None:
        default = prepare_hosted_provisional_judge(
            project,
            catalog,
            maximum_input_tokens=options.maximum_judge_input_tokens,
            maximum_output_tokens=options.maximum_judge_output_tokens,
            maximum_attempts=RetryPolicy().maximum_attempts,
            created_at=created_at,
            code_revision=code_revision,
        )
        judge_setup, calibration_id = default.setup_input, default.calibration_id
    assert calibration_id is not None
    selected = read_evaluation_judge(project, judge_setup)
    calibration, _ = verify_persisted_calibration(project, calibration_id)
    if calibration.status == "insufficient":
        raise ValueError("judge calibration is insufficient; choose an eligible judge")
    if (
        selected.task_set != completed.task_set
        or selected.trace_dataset != completed.trace_dataset
        or selected.rubric.artifact_id != calibration.rubric_id
        or selected.judge_model != calibration.judge_model
        or selected.prompt_template.prompt.prompt_id != calibration.judge_prompt_id
        or selected.prompt_template.prompt.sha256 != calibration.judge_prompt_sha256
    ):
        raise ValueError("judge setup or calibration differs from the completed evaluation project")
    static = RuntimeModelCatalog(catalog, environment={})
    judge_model, judge_caps = static.snapshot(judge_alias or selected.judge_alias)
    if not serves_role(judge_caps, SetupRole.JUDGE):
        raise ValueError("judge requires structured output and pricing; choose another judge model")
    if judge_alias is not None and judge_alias != selected.judge_alias:
        judge_setup, calibration_id = select_judge_model(
            project,
            selected,
            calibration,
            alias=judge_alias,
            model=judge_model,
            created_at=created_at,
            code_revision=code_revision,
        )
        selected = read_evaluation_judge(project, judge_setup)
        calibration, _ = verify_persisted_calibration(project, calibration_id)
        assert calibration.status == "provisional"
    judge_model, judge_caps = static.snapshot(selected.judge_alias)
    if judge_model != selected.judge_model:
        raise ValueError("judge catalog changed; prepare a new judge setup")
    report(progress, "Loading world model")
    world = load_grounded_world_model_artifact(project.artifacts, completed.world_model)
    world_snapshot, _ = static.snapshot(world.model_alias)
    if world_snapshot != world.model:
        raise ValueError("simulation catalog changed; build a new grounded project")
    embedder, _ = static.snapshot(embedder_alias)
    report(progress, "Loading retrieval index")
    fit = load_rag_index(project.artifacts, completed.fit_rag.artifact_id)
    if artifact_input(fit.manifest) != completed.fit_rag or fit.index.embedder != embedder:
        raise ValueError("selected embedder differs from the completed fit RAG")
    candidates = tuple(
        RoutedCandidateSnapshot(alias=alias, model=static.snapshot(alias)[0])
        for alias in sorted(worker_aliases)
    )
    output_budgets: dict[str, int] = {}
    for alias in (*worker_aliases, world.model_alias):
        capabilities = static.snapshot(alias)[1]
        if capabilities.context_window_tokens is None:
            raise ValueError(f"context window is missing for {alias}; refresh model metadata")
        output_budgets[alias] = capabilities.maximum_output_tokens or min(
            options.maximum_rollout_output_tokens, capabilities.context_window_tokens
        )
    maximum_output_tokens = options.maximum_output_tokens or max(output_budgets.values())
    report(progress, "Loading traces for cost estimates")
    traces = load_trace_dataset(project.artifacts, completed.trace_dataset.artifact_id).traces
    input_estimates: dict[str, int | None] = {}
    report(progress, "Estimating model costs", completed=0, total=len(output_budgets))
    for index, (alias, output_budget) in enumerate(output_budgets.items(), start=1):
        input_estimates[alias] = simulation_input_token_estimate(
            traces,
            retrieved_transition_count=world.top_k,
            maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
            maximum_output_tokens=min(maximum_output_tokens, output_budget),
        )
        report(progress, "Estimating model costs", completed=index, total=len(output_budgets))
    if any(value is None for value in input_estimates.values()):
        raise ValueError("evaluation requires captured source traces for a cost estimate")
    attempts = RetryPolicy().maximum_attempts
    problems: list[str] = []
    requests, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=candidates,
        world_alias=world.model_alias,
        world=world.model,
        maximum_attempts=attempts,
        estimated_input_tokens={
            alias: value for alias, value in input_estimates.items() if value is not None
        },
        maximum_output_tokens=maximum_output_tokens,
    )
    retrieval = retrieval_embedding_reservation(
        problems,
        catalog,
        embedder_alias,
        embedder,
        options.maximum_retrieval_query_tokens,
        attempts,
    )
    if problems or world_request is None or retrieval is None:
        raise ValueError("evaluation pricing is incomplete: " + "; ".join(problems))
    judge_request = _judge_request_reservation(
        judge_caps,
        judge_model=judge_model,
        maximum_input_tokens=options.maximum_judge_input_tokens,
        maximum_output_tokens=options.maximum_judge_output_tokens,
        maximum_attempts=attempts,
    )
    config = project.load_project()
    agent_digest = agent_factory_sha256(
        config.agent,
        maximum_model_calls=options.maximum_steps,
        system_prompt=config.system.system_prompt if config.system else None,
    )
    report(progress, "Freezing evaluation settings")
    pricing = persist_pricing_snapshot(
        project.artifacts,
        tuple(
            CandidateTokenPrice(
                candidate_alias=item.candidate_alias,
                input_usd_per_million_tokens=item.request.input_usd_per_million_tokens,
                output_usd_per_million_tokens=item.request.output_usd_per_million_tokens,
                cached_input_usd_per_million_tokens=item.request.cached_input_usd_per_million_tokens,
                cache_write_usd_per_million_tokens=item.request.cache_write_usd_per_million_tokens,
            )
            for item in requests
        ),
        created_at=created_at,
        code_revision=code_revision,
    )
    pricing_input = artifact_input(project.artifacts.read(pricing.pricing_snapshot_id).manifest)
    _, completion_input = persist_simulation_completion_contract(
        project.artifacts,
        inputs=sorted_unique_inputs(
            completed.trace_dataset,
            completed.task_set,
            completed.fit_rag,
            completed.world_model,
            pricing_input,
        ),
        candidate_requests=requests,
        world_model_alias=world.model_alias,
        world_model_request=world_request,
        maximum_attempts=attempts,
        created_at=created_at,
        code_revision=code_revision,
    )
    shared = dict(
        agent_id=config.project_id,
        rubric_id=calibration.rubric_id,
        judge_calibration_id=calibration_id,
        pricing_snapshot_id=pricing.pricing_snapshot_id,
    )
    production = EvaluationProtocol(
        protocol_id=stable_id("protocol", {"version": "model-evaluation-production-v1", **shared}),
        evidence_source="production",
        simulator_id="production-import-v1",
        agent_id=config.project_id,
        rubric_id=calibration.rubric_id,
        judge_calibration_id=calibration_id,
        pricing_snapshot_id=pricing.pricing_snapshot_id,
    )
    simulation = EvaluationProtocol(
        protocol_id=stable_id(
            "protocol",
            {
                "version": "model-evaluation-simulation-v1",
                **shared,
                "world": world.model_dump(mode="json"),
            },
        ),
        evidence_source="world_model",
        simulator_id="text-world-model-v1",
        world_model=world.model,
        simulator_prompt_id=WORLD_MODEL_TEXT_PROMPT_ID,
        agent_id=config.project_id,
        rubric_id=calibration.rubric_id,
        judge_calibration_id=calibration_id,
        pricing_snapshot_id=pricing.pricing_snapshot_id,
    )
    setup = EvaluationSetup(
        run_id=run_id,
        candidates=candidates,
        production_protocol=production,
        simulation_protocol=simulation,
        fit_rag_input=completed.fit_rag,
        pricing_snapshot_id=pricing.pricing_snapshot_id,
        judgment_status=calibration.status,
        world_model_settings=WorldModelSettings(
            world_model_alias=world.model_alias,
            grounded_world_model_input=completed.world_model,
            prompt_version=WORLD_MODEL_TEXT_PROMPT_VERSION,
            query_embedding=retrieval,
            maximum_output_tokens=maximum_output_tokens,
            json_object_output=True,
        ),
        simulation_completion_input=completion_input,
        agent_id=config.project_id,
        seed=options.seed,
        continuation_of=(
            artifact_input(project.artifacts.read(continuation_of).manifest)
            if continuation_of is not None
            else None
        ),
        maximum_steps=options.maximum_steps,
        maximum_rollout_output_tokens=options.maximum_rollout_output_tokens,
        maximum_concurrency=options.maximum_concurrency,
        repeats=options.repeats,
    )
    report(progress, "Calculating evaluation quote")
    cost = estimate_model_evaluation(
        project,
        setup,
        judge_request=judge_request,
        judge_calls_per_rollout=2 if selected.prompt_template.response_shape == "pairwise" else 1,
    )
    return PreparedModelEvaluation(
        setup=setup,
        judge_setup=judge_setup,
        judge_request=judge_request,
        embedder_alias=embedder_alias,
        agent_factory_sha256=agent_digest,
        redacted_field_names=config.redacted_field_names,
        cost=cost,
    )

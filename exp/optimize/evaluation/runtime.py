"""Catalog-backed model evaluation using canonical bounded simulator and judge clients."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactInput,
    canonical_json_bytes,
    sorted_unique_inputs,
    stable_id,
)
from exp.common.evaluations import EvaluationPlan
from exp.common.models import ModelSnapshot, verify_completion_reservation
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuation import EvaluationRuntimeContract, validate_continuation
from exp.optimize.evaluation.contracts import EvaluationBudget, EvaluationServices
from exp.optimize.evaluation.judge import DurableEvaluationJudge
from exp.optimize.evaluation.judging_resume import revised_judge_setup
from exp.optimize.evaluation.judging_spend import judge_request_coordinates
from exp.optimize.evaluation.planning import estimate_model_evaluation
from exp.optimize.evaluation.prepare import PreparedModelEvaluation, read_evaluation_judge
from exp.optimize.evaluation.service import ModelEvaluationResult, evaluate_models
from exp.optimize.evaluation.spending import BudgetedCompletion, BudgetedEmbedding
from exp.optimize.router.automatic.judge import AutomaticRouterJudge, ReservedJudgeClient
from exp.optimize.router.evaluation.build import completed_project_build
from exp.runtime.agents import agent_factory_sha256, preflight_agent_factory, resolve_agent_factory
from exp.runtime.models import CapabilityRequirement, ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.engines.text import WorldModelSimulator
from exp.simulation.retrieval import RAGEmbedderBinding, load_fit_rag_retriever
from exp.simulation.specs import load_simulation_completion_contract
from exp.simulation.world_model import bind_fit_grounded_world_model


def run_prepared_model_evaluation(
    project: ProjectStore,
    prepared: PreparedModelEvaluation,
    catalog: RuntimeModelCatalog,
    *,
    budget: EvaluationBudget,
    provider_spend_consented: bool,
    created_at: datetime,
    code_revision: str,
    progress: ProgressHook | None = None,
    judging_revision: ArtifactInput | None = None,
) -> ModelEvaluationResult:
    """Execute a prepared worker matrix within an explicitly approved request-level allowance.

    Args:
        project: Owner of the completed scenario, world-model and judge evidence.
        prepared: Exact engine preparation whose quote the user accepted.
        catalog: Runtime catalog holding transient provider credential references.
        budget: Approved total provider allowance and judgment count. Raising only the
            allowance resumes the same plan and replays completed provider responses for free.
        provider_spend_consented: Explicit consent after the host's atomic credit reservation.
        created_at: Stable run timestamp.
        code_revision: Exact engine revision.
        progress: Observer of real simulator and judge progress.
        judging_revision: Explicit fresh judging pass; requires saved completed rollouts.

    Returns:
        Persisted model report and reconciled execution costs, with exact replay.

    Raises:
        SpendLimitReached: The next request cannot fit; saved calls remain exactly resumable.
        ValueError: Consent, quote admission, identities, configuration or artifacts drift.
    """
    report(progress, "Verifying evaluation")
    validate_continuation(project, prepared)
    setup = prepared.setup
    selected = read_evaluation_judge(project, prepared.judge_setup)
    if (
        selected.rubric.artifact_id != setup.simulation_protocol.rubric_id
        or selected.judge_model != prepared.judge_request.model
    ):
        raise ValueError("prepared judge differs from the evaluation; prepare again")
    calls_per_rollout = 2 if selected.prompt_template.response_shape == "pairwise" else 1
    report(progress, "Checking evaluation estimate")
    quote = estimate_model_evaluation(
        project,
        setup,
        judge_request=prepared.judge_request,
        judge_calls_per_rollout=calls_per_rollout,
    )
    if quote != prepared.cost:
        raise ValueError("evaluation quote changed; prepare and approve a new estimate")
    if budget.maximum_judgments < quote.judgment_count:
        raise ValueError("authorized evaluation judgment count is too small")
    config = project.load_project()
    prompt = config.system.system_prompt if config.system else None
    if (
        prepared.agent_factory_sha256
        != agent_factory_sha256(
            config.agent,
            maximum_model_calls=setup.maximum_steps,
            system_prompt=prompt,
        )
        or prepared.redacted_field_names != config.redacted_field_names
    ):
        raise ValueError("evaluation agent configuration changed; prepare again")
    if not provider_spend_consented:
        raise ValueError(
            "evaluation requires explicit provider-spend consent after credit admission"
        )
    judge_request = prepared.judge_request
    judging_protocol = None
    if judging_revision is not None:
        selected, judge_request, judging_protocol = revised_judge_setup(
            project, prepared, judging_revision
        )
    report(progress, "Verifying built project")
    completed = completed_project_build(project)
    completion_input = setup.simulation_completion_input
    retrieval = setup.world_model_settings.query_embedding
    if completion_input is None or retrieval is None:
        raise ValueError("evaluation requires frozen completion and retrieval reservations")
    completion, _ = load_simulation_completion_contract(
        project.artifacts, completion_input.artifact_id
    )
    factory = resolve_agent_factory(
        config.agent,
        maximum_model_calls=setup.maximum_steps,
        system_prompt=prompt,
    )
    preflight_agent_factory(factory)

    def resolve(alias: str, expected: ModelSnapshot, *, embedding: bool = False) -> ResolvedModel:
        """Check static identity before constructing each credential-backed client."""
        if catalog.snapshot(alias)[0] != expected:
            raise ValueError(f"evaluation model {alias!r} changed; prepare again")
        resolved = catalog.preflight(alias, CapabilityRequirement(requires_embeddings=embedding))
        if resolved.snapshot != expected:
            raise ValueError(f"resolved evaluation model {alias!r} changed; prepare again")
        return resolved

    report(progress, "Preparing model clients")
    candidates = {item.alias: resolve(item.alias, item.model) for item in setup.candidates}
    world = resolve(completion.world_model_alias, completion.world_model_request.model)
    judge_model = resolve(selected.judge_alias, selected.judge_model)
    embedder = resolve(prepared.embedder_alias, retrieval.model, embedding=True)
    if embedder.embedding_client is None:
        raise ValueError("selected retrieval model has no embedding client")
    if (
        embedder.capabilities.input_cost_per_million_tokens_usd
        != retrieval.input_usd_per_million_tokens
    ):
        raise ValueError("retrieval pricing changed; prepare again")
    attempts = RetryPolicy().maximum_attempts
    for request, resolved in (
        *(
            (item.request, candidates[item.candidate_alias])
            for item in completion.candidate_requests
        ),
        (completion.world_model_request, world),
    ):
        verify_completion_reservation(
            request,
            model=resolved.snapshot,
            capabilities=resolved.capabilities,
            maximum_attempts=attempts,
        )
    if retrieval.maximum_attempts != attempts:
        raise ValueError("retrieval retry policy changed; prepare again")
    ledger_identity = stable_id(
        "evaluation-requests",
        {
            "prepared": prepared.model_dump(mode="json"),
            "code_revision": code_revision,
        },
    )
    ledger = RequestBudget(
        project.paths.runtime_directory / "evaluation-requests" / ledger_identity,
        identity=ledger_identity,
        maximum_cost_usd=budget.maximum_cost_usd,
    )
    candidates = {
        item.candidate_alias: replace(
            candidates[item.candidate_alias],
            client=BudgetedCompletion(
                candidates[item.candidate_alias].client,
                ledger,
                item.request,
                role=f"assistant:{item.candidate_alias}",
                served_model_id=candidates[item.candidate_alias].served_model_id,
            ),
        )
        for item in completion.candidate_requests
    }
    world = replace(
        world,
        client=BudgetedCompletion(
            world.client,
            ledger,
            completion.world_model_request,
            role="world",
            served_model_id=world.served_model_id,
        ),
    )
    bounded_judge = ReservedJudgeClient(
        BudgetedCompletion(
            judge_model.client,
            ledger,
            judge_request,
            role="judge",
            served_model_id=judge_model.served_model_id,
        ),
        reservation=judge_request,
        model=judge_model.snapshot,
        capabilities=judge_model.capabilities,
        maximum_attempts=attempts,
        maximum_provider_calls=quote.judgment_count * calls_per_rollout,
        served_model_id=judge_model.served_model_id,
    )
    judge = AutomaticRouterJudge(
        bounded_judge,
        selected,
        created_at=selected.created_at if judging_revision else created_at,
        code_revision=selected.code_revision if judging_revision else code_revision,
        maximum_input_tokens=judge_request.maximum_input_tokens,
        maximum_output_tokens=judge_request.maximum_output_tokens,
        request_scope=ledger.scope,
    )
    report(progress, "Loading retrieval index")
    retriever = load_fit_rag_retriever(
        project.artifacts,
        completed.fit_rag,
        embedder=RAGEmbedderBinding(
            client=BudgetedEmbedding(embedder.embedding_client, ledger, retrieval),
            snapshot=embedder.snapshot,
            maximum_attempts=attempts,
            input_usd_per_million_tokens=retrieval.input_usd_per_million_tokens,
            maximum_input_tokens=embedder.capabilities.context_window_tokens,
        ),
    )
    report(progress, "Loading world model")
    grounded = bind_fit_grounded_world_model(
        project.artifacts,
        completed.world_model,
        client=world.client,
        fit_retriever=retriever,
    )

    def simulator_factory(project: ProjectStore, plan: EvaluationPlan) -> WorldModelSimulator:
        """Bind one fresh simulator to the exact persisted evaluation matrix."""
        if judging_revision is not None:
            raise ValueError(
                "judging retry requires finished saved rollouts; resume simulation first"
            )
        return WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=completed.task_set,
            fit_rag_input=completed.fit_rag,
            fit_retriever=retriever,
            candidate_models=candidates,
            world_models={world.alias: world},
            grounded_world_models={world.alias: grounded},
            agent_factory=factory,
            completion_contract_input=completion_input,
            redacted_field_names=prepared.redacted_field_names,
            progress=progress,
            request_budget=ledger,
        )

    coordinates_by_rollouts: dict[tuple[str, ...], tuple[tuple[str, str, int], ...]] = {}

    def judge_spend(rollout_ids: tuple[str, ...]) -> float:
        """Reconcile every paid judge response and unknown attempt across the reviewed lineage."""
        if rollout_ids not in coordinates_by_rollouts:
            coordinates_by_rollouts[rollout_ids] = judge_request_coordinates(
                project, prepared, judging_revision, rollout_ids
            )
        return ledger.accounted_requests(coordinates_by_rollouts[rollout_ids])

    report(progress, "Freezing execution plan")
    runtime_input = _persist_runtime_contract(project, prepared, created_at, code_revision)
    return evaluate_models(
        project,
        setup,
        services=EvaluationServices(
            simulator_factory,
            DurableEvaluationJudge(judge, bounded_judge, judge_request, budget=ledger),
            (runtime_input,),
            judging_protocol=judging_protocol,
            judging_input=judging_revision,
            spending_limit_usd=budget.maximum_cost_usd,
            judge_spend=judge_spend,
        ),
        # Semantic execution bounds stay frozen across allowance increases. The request
        # ledger enforces the smaller approved amount before every paid dispatch.
        budget=budget.model_copy(update={"maximum_cost_usd": max(quote.maximum_cost_usd, 1e-12)}),
        created_at=created_at,
        code_revision=code_revision,
        progress=progress,
    )


def _persist_runtime_contract(
    project: ProjectStore,
    prepared: PreparedModelEvaluation,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Include every runtime semantic pin in replay identity before the first dispatch."""
    completed = completed_project_build(project)
    setup = prepared.setup
    assert setup.simulation_completion_input is not None
    calibration = artifact_input(
        project.artifacts.read(setup.simulation_protocol.judge_calibration_id).manifest
    )
    pricing = artifact_input(project.artifacts.read(setup.pricing_snapshot_id).manifest)
    contract = EvaluationRuntimeContract(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=sorted_unique_inputs(
            prepared.judge_setup,
            completed.trace_dataset,
            completed.task_set,
            completed.fit_rag,
            completed.world_model,
            setup.simulation_completion_input,
            calibration,
            pricing,
        ),
        contract_id=stable_id(
            "evaluation-runtime",
            {
                "prepared": prepared.model_dump(mode="json"),
                "code_revision": code_revision,
            },
        ),
        prepared=prepared,
    )
    _, manifest = project.artifacts.write_or_replay(
        artifact_id=contract.contract_id,
        artifact_type="evaluation-runtime-contract",
        envelope=contract,
        envelope_path="runtime.json",
        envelope_type=EvaluationRuntimeContract,
        files={"runtime.json": canonical_json_bytes(contract)},
    )
    return artifact_input(manifest)

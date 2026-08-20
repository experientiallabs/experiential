"""Automatic completed-build router composition after one aggregate preflight."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from exp.common.core.artifacts import stable_id
from exp.common.evaluations import EvaluationPlan, EvaluationProtocol
from exp.common.models import (
    ModelCatalog,
    OperationEconomics,
    ProviderSetup,
    configure_provider_catalog_with_router_candidates,
    configure_router_candidates,
    verify_router_candidate_catalog_state,
)
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.routing import KnnGuard
from exp.optimize.router.automatic.artifacts import (
    AutomaticRouterArtifacts,
    materialize_automatic_router_artifacts,
)
from exp.optimize.router.automatic.attribution import persist_router_observed_attribution_set
from exp.optimize.router.automatic.judge import AutomaticRouterJudge, ReservedJudgeClient
from exp.optimize.router.automatic.preflight import (
    AutomaticRouterPreflight,
    HostedAutomaticJudgeEvidence,
    HumanCalibratedAutomaticJudge,
    preflight_automatic_router,
)
from exp.optimize.router.automatic.reservations import AutomaticRouterOptions
from exp.optimize.router.composition import (
    ApprovedRouterReview,
    ProvisionalRouterReview,
    RouterCandidateSetupPlan,
    RouterCompositionBudget,
    RouterCompositionResult,
    RouterEvaluationSetup,
    RouterPolicyLock,
    RouterWorkflowServices,
    compose_router,
)
from exp.optimize.router.fit.workflow import RouterFitWorkflowResult
from exp.runtime.agents import (
    AgentFactory,
    preflight_agent_factory,
    resolve_agent_factory,
)
from exp.runtime.models import (
    CapabilityRequirement,
    CatalogRoleName,
    ResolvedModel,
    RuntimeModelCatalog,
)
from exp.simulation.build import ProjectBuild
from exp.simulation.engines.text import (
    WORLD_MODEL_TEXT_PROMPT_ID,
    WORLD_MODEL_TEXT_PROMPT_VERSION,
    WorldModelSimulator,
)
from exp.simulation.ingest.otlp import TraceNormalizationResult
from exp.simulation.retrieval import RAGEmbedderBinding, load_fit_rag_retriever
from exp.simulation.specs import WorldModelSettings
from exp.simulation.world_model import bind_fit_grounded_world_model

logger = logging.getLogger(__name__)

_REASONING_JUDGE_OUTPUT_FLOOR_TOKENS = 8_192


class AutomaticRouterError(ValueError):
    """Automatic router composition failed at a consent or immutable binding boundary."""


HostedPolicyCheckpoint = Callable[
    [
        RouterPolicyLock,
        RouterFitWorkflowResult,
        AutomaticRouterPreflight,
        AutomaticRouterArtifacts,
        tuple[OperationEconomics, ...],
    ],
    None,
]


@dataclass(frozen=True)
class AutomaticRouterResult:
    """Verified preflight, immutable execution contract, and completed router chain."""

    preflight: AutomaticRouterPreflight
    artifacts: AutomaticRouterArtifacts
    composition: RouterCompositionResult
    judge_economics: tuple[OperationEconomics, ...]


@dataclass(frozen=True)
class _ResolvedAutomaticModels:
    """Independent candidate, world, judge, and embedding runtime clients."""

    candidates: dict[str, ResolvedModel]
    world_model: ResolvedModel
    judge: ResolvedModel
    embedder: ResolvedModel


def optimize_project_router(
    project: ProjectStore,
    candidate_plan: RouterCandidateSetupPlan,
    runtime_catalog: RuntimeModelCatalog,
    *,
    options: AutomaticRouterOptions,
    provider_spend_consented: bool,
    created_at: datetime,
    code_revision: str,
    phase_hook: Callable[[str], None] | None = None,
    progress: ProgressHook | None = None,
    policy_lock_hook: Callable[[RouterPolicyLock, RouterFitWorkflowResult], None] | None = None,
    hosted_policy_checkpoint: HostedPolicyCheckpoint | None = None,
    hosted_judge: HostedAutomaticJudgeEvidence | None = None,
    transient_catalog: bool = False,
) -> AutomaticRouterResult:
    """Optimize a router from one completed project with no workflow config file.

    Args:
        project: Existing project with a completed build.
        candidate_plan: Explicit confirmed candidates and prospective catalog.
        runtime_catalog: Resolver retaining the caller's credential and transport seams.
        options: Bounded provider, evidence, retry, and concurrency controls.
        provider_spend_consented: Explicit approval of the displayed shared provider ceiling.
        created_at: Materialization time for new immutable artifacts.
        code_revision: Exact producer revision.
        phase_hook: Optional local phase-order observer.
        progress: Optional observer of truthful stage names and exact unit counts.
        policy_lock_hook: Optional hosted checkpoint invoked at the immutable fit lock.
        hosted_policy_checkpoint: Rich hosted-only durable checkpoint with preflight, immutable
            inputs, and reconciled judge economics available before held-out work.
        hosted_judge: Optional machine-only provisional judge evidence.
        transient_catalog: Avoid every root-global catalog read or write for hosted execution.

    Returns:
        Complete preflight, execution contract, optimized policy, report, and runtime.

    Raises:
        AutomaticRouterError: Consent, credentials, catalog state, or runtime binding differs.
        AutomaticRouterPreflightError: Aggregate prerequisites are incomplete.
    """
    report(progress, "preflight")
    preflight = preflight_automatic_router(
        project,
        candidate_plan.selection,
        catalog_override=candidate_plan.prospective_catalog,
        hosted_judge=hosted_judge,
        options=options,
    )
    if not provider_spend_consented:
        raise AutomaticRouterError(
            "router optimization requires explicit consent for the full provider-spend ceiling"
        )
    if not transient_catalog:
        verify_router_candidate_catalog_state(
            project.model_catalog_path,
            candidate_plan.expected_catalog_sha256,
        )
    resolved_catalog = runtime_catalog.with_catalog(candidate_plan.prospective_catalog)
    agent_factory = _resolve_agent_factory(preflight, options)
    resolved = _resolve_all_models(preflight, resolved_catalog, options)
    if not transient_catalog:
        configured = persist_router_candidate_setup(project, candidate_plan)
        if configured != candidate_plan.prospective_catalog:
            raise AutomaticRouterError(
                "persisted router candidate catalog differs from confirmation"
            )
    attribution_input = None
    if preflight.observed_traces:
        _attribution, attribution_input = persist_router_observed_attribution_set(
            project.artifacts,
            trace_dataset=preflight.completed_build.trace_dataset,
            task_set=preflight.completed_build.task_set,
            catalog_sha256=preflight.catalog_sha256,
            candidates=preflight.candidates,
            records=tuple(item.attribution for item in preflight.observed_traces),
            created_at=created_at,
            code_revision=code_revision,
        )
    artifacts = materialize_automatic_router_artifacts(
        project,
        preflight,
        resolved_catalog,
        attribution_input=attribution_input,
        catalog_override=(candidate_plan.prospective_catalog if transient_catalog else None),
        router_embedding_maximum_attempts=options.router_embedding_maximum_attempts,
        completion_maximum_attempts=options.completion_maximum_attempts,
        maximum_provider_cost_usd=options.maximum_provider_cost_usd,
        created_at=created_at,
        code_revision=code_revision,
    )
    judge = _automatic_judge(preflight, resolved, created_at, code_revision)
    services = _workflow_services(
        project,
        preflight,
        artifacts,
        resolved,
        resolved_catalog,
        judge,
        agent_factory,
        options,
        progress=progress,
    )

    def checkpoint(lock: RouterPolicyLock, fit: RouterFitWorkflowResult) -> None:
        """Forward the immutable policy lock to each requested checkpoint observer."""
        if policy_lock_hook is not None:
            policy_lock_hook(lock, fit)
        if hosted_policy_checkpoint is not None:
            hosted_policy_checkpoint(
                lock,
                fit,
                preflight,
                artifacts,
                judge.provider_economics,
            )

    composition = compose_router(
        project,
        TraceNormalizationResult(
            traces=preflight.traces,
            issues=(),
            identity_evidence=(
                None
                if preflight.trace_identity_evidence is None
                else preflight.trace_identity_evidence.records
            ),
        ),
        services=services,
        budget=RouterCompositionBudget(
            maximum_simulation_cost_usd=preflight.remaining_simulation_cost_usd,
            maximum_judgments=options.maximum_judgments,
            stop_on_overspend=options.stop_on_overspend,
        ),
        created_at=created_at,
        code_revision=code_revision,
        phase_hook=phase_hook,
        progress=progress,
        policy_lock_hook=(
            checkpoint
            if policy_lock_hook is not None or hosted_policy_checkpoint is not None
            else None
        ),
    )
    return AutomaticRouterResult(
        preflight=preflight,
        artifacts=artifacts,
        composition=composition,
        judge_economics=judge.provider_economics,
    )


def persist_router_candidate_setup(
    project: ProjectStore,
    candidate_plan: RouterCandidateSetupPlan,
) -> ModelCatalog:
    """Persist candidate provider records and router roles in one catalog transaction.

    Args:
        project: Project whose shared model catalog was confirmed during collection.
        candidate_plan: Confirmed candidate selection and prospective catalog.

    Returns:
        Complete catalog after the selected provider records and roles are committed.

    Raises:
        AutomaticRouterError: The confirmed catalog cannot be persisted atomically.
    """
    try:
        if not candidate_plan.candidate_connections and not candidate_plan.candidate_models:
            return configure_router_candidates(
                project.model_catalog_path,
                candidate_plan.selection,
                expected_state_sha256=candidate_plan.expected_catalog_sha256,
            )

        roles = candidate_plan.prospective_catalog.roles
        world_model, judge, embedder = roles.world_model, roles.judge, roles.embedder
        if world_model is None or judge is None or embedder is None:
            raise AutomaticRouterError(
                "discovered router candidates require an existing world model, judge, and embedder"
            )
        new_connection_names = {
            connection.name for connection in candidate_plan.candidate_connections
        }
        new_aliases = {model.alias for model in candidate_plan.candidate_models}
        setup = ProviderSetup(
            connections=candidate_plan.candidate_connections,
            models=candidate_plan.candidate_models,
            known_existing_connections=tuple(
                sorted(
                    set(candidate_plan.prospective_catalog.connections).difference(
                        new_connection_names
                    )
                )
            ),
            known_existing_aliases=tuple(
                sorted(set(candidate_plan.prospective_catalog.models).difference(new_aliases))
            ),
            world_model=world_model,
            judge=judge,
            embedder=embedder,
        )
        return configure_provider_catalog_with_router_candidates(
            project.model_catalog_path,
            setup,
            candidate_plan.selection,
            expected_state_sha256=candidate_plan.expected_catalog_sha256,
        )
    except AutomaticRouterError:
        raise
    except ValueError as exc:
        raise AutomaticRouterError(f"router candidate setup could not be saved: {exc}") from exc


def _resolve_all_models(
    preflight: AutomaticRouterPreflight,
    catalog: RuntimeModelCatalog,
    options: AutomaticRouterOptions,
) -> _ResolvedAutomaticModels:
    """Resolve every credential and runtime client after consent but before any write.

    Args:
        preflight: Complete aggregate local validation result.
        catalog: Resolver over the confirmed prospective catalog.
        options: Active completion and embedding capacity requirements.

    Returns:
        Independently constructed candidate, world, judge, and embedding clients.

    Raises:
        AutomaticRouterError: A model identity or required client shape differs from preflight.
    """

    def resolve(
        alias: str,
        requirement: CapabilityRequirement,
        role: CatalogRoleName | None = None,
    ) -> ResolvedModel:
        """Resolve one role through a fresh provider client.

        Args:
            alias: Stable catalog alias.
            requirement: Exact local capability proof required for this role.
            role: Completion role whose configured reasoning effort shapes requests.

        Returns:
            Independently constructed resolved model.

        Raises:
            AutomaticRouterError: Credential or capability resolution fails.
        """
        try:
            return catalog.preflight(alias, requirement, role=role)
        except ValueError as exc:
            raise AutomaticRouterError(f"model alias {alias!r} cannot be resolved: {exc}") from exc

    candidates = {
        candidate.alias: resolve(
            candidate.alias,
            CapabilityRequirement(
                minimum_context_window_tokens=options.simulation_maximum_output_tokens + 1,
                minimum_output_tokens=options.simulation_maximum_output_tokens,
            ),
            "candidate",
        )
        for candidate in preflight.candidates
    }
    world = resolve(
        preflight.world_model_alias,
        CapabilityRequirement(
            minimum_context_window_tokens=options.simulation_maximum_output_tokens + 1,
            minimum_output_tokens=options.simulation_maximum_output_tokens,
        ),
        "world_model",
    )
    judge = resolve(
        preflight.judge_alias,
        CapabilityRequirement(
            minimum_output_tokens=preflight.judge_completion_reservation.maximum_output_tokens,
        ),
        "judge",
    )
    embedder = resolve(
        preflight.embedder_alias,
        CapabilityRequirement(requires_embeddings=True),
    )
    expected = (
        *(
            (candidate.alias, candidates[candidate.alias].snapshot, candidate.model)
            for candidate in preflight.candidates
        ),
        (preflight.world_model_alias, world.snapshot, preflight.world_model),
        (preflight.judge_alias, judge.snapshot, preflight.judge_model),
        (preflight.embedder_alias, embedder.snapshot, preflight.embedder),
    )
    drifted = tuple(alias for alias, actual, frozen in expected if actual != frozen)
    if drifted:
        raise AutomaticRouterError(
            "resolved model identities differ from aggregate preflight: " + ", ".join(drifted)
        )
    if embedder.embedding_client is None:
        raise AutomaticRouterError("resolved embedder does not expose an embedding client")
    return _ResolvedAutomaticModels(
        candidates=candidates,
        world_model=world,
        judge=judge,
        embedder=embedder,
    )


def _resolve_agent_factory(
    preflight: AutomaticRouterPreflight,
    options: AutomaticRouterOptions,
) -> AgentFactory:
    """Resolve and construct the simulation agent only after provider-spend consent.

    Args:
        preflight: Identity-only project and agent configuration validation.
        options: Active built-in agent request ceiling.

    Returns:
        Fresh-runtime factory validated once before any provider credential resolution.

    Raises:
        AutomaticRouterError: The custom factory cannot be imported or constructed safely.
    """
    try:
        factory = resolve_agent_factory(
            preflight.project_config.agent,
            maximum_model_calls=options.maximum_model_calls,
            system_prompt=(
                preflight.project_config.system.system_prompt
                if preflight.project_config.system is not None
                else None
            ),
        )
        preflight_agent_factory(factory)
        return factory
    except ValueError as exc:
        raise AutomaticRouterError(f"agent runtime cannot be resolved: {exc}") from exc


def _automatic_judge(
    preflight: AutomaticRouterPreflight,
    resolved: _ResolvedAutomaticModels,
    created_at: datetime,
    code_revision: str,
) -> AutomaticRouterJudge:
    """Create the plan-bindable approved judge under its full-call reservation.

    Args:
        preflight: Approved setup and judge request reservation.
        resolved: Credential-resolved runtime clients.
        created_at: Probe and judgment materialization time.
        code_revision: Exact producer revision.

    Returns:
        Judge awaiting the exact evaluation plan from the simulator factory.
    """
    judge = resolved.judge
    reservation = preflight.judge_completion_reservation
    if (
        judge.capabilities.reasoning_effort is not None
        and reservation.maximum_output_tokens < _REASONING_JUDGE_OUTPUT_FLOOR_TOKENS
    ):
        logger.warning(
            "judge model %s pins reasoning effort %s but its approved calibration output "
            "budget is only %d tokens; reasoning can consume the whole budget and leave no "
            "visible text, so affected cells retry and may be excluded from evidence",
            judge.snapshot.model_id,
            judge.capabilities.reasoning_effort,
            reservation.maximum_output_tokens,
        )
    bounded = ReservedJudgeClient(
        judge.client,
        reservation=reservation,
        model=judge.snapshot,
        capabilities=judge.capabilities,
        maximum_attempts=reservation.maximum_attempts,
        maximum_provider_calls=preflight.judge_provider_call_count,
    )
    return AutomaticRouterJudge(
        bounded,
        preflight.setup,
        created_at=created_at,
        code_revision=code_revision,
        maximum_input_tokens=preflight.judge_completion_reservation.maximum_input_tokens,
        maximum_output_tokens=reservation.maximum_output_tokens,
    )


def _workflow_services(
    project: ProjectStore,
    preflight: AutomaticRouterPreflight,
    artifacts: AutomaticRouterArtifacts,
    resolved: _ResolvedAutomaticModels,
    runtime_catalog: RuntimeModelCatalog,
    judge: AutomaticRouterJudge,
    agent_factory: AgentFactory,
    options: AutomaticRouterOptions,
    progress: ProgressHook | None = None,
) -> RouterWorkflowServices:
    """Bind the generic composition interfaces to verified automatic project inputs.

    Args:
        project: Project-local artifact store.
        preflight: Completed build, review, evidence, and reservation inputs.
        artifacts: Post-consent immutable pricing and provider contracts.
        resolved: Credential-resolved runtime clients.
        runtime_catalog: Resolver used by the final online router runtime.
        judge: Approved saved-contract judge awaiting its exact plan.
        agent_factory: Post-consent validated fresh-runtime constructor.
        options: Active simulation controls.
        progress: Optional observer forwarded to each constructed simulator.

    Returns:
        Complete service bundle for the existing router composition.
    """
    production_protocol, simulation_protocol = _protocols(preflight, artifacts, project)

    def review_supplier(
        project: ProjectStore,
        build: ProjectBuild,
        budget: RouterCompositionBudget,
    ) -> ApprovedRouterReview | ProvisionalRouterReview:
        """Supply exact typed judge provenance to composition.

        Args:
            project: Project already verified by automatic preflight.
            build: Completed build already verified by automatic preflight.
            budget: Shared composition budget already admitted before provider access.

        Returns:
            Rubric and calibration identities with their exact eligibility status.
        """
        del project, build, budget
        if preflight.judgment_status == "provisional":
            return ProvisionalRouterReview(
                rubric_id=preflight.setup.rubric.artifact_id,
                calibration_id=preflight.calibration_id,
                calibration_input=preflight.calibration_input,
            )
        if not isinstance(preflight.judge_provenance, HumanCalibratedAutomaticJudge):
            raise AutomaticRouterError("human calibration provenance changed after preflight")
        return ApprovedRouterReview(
            rubric_id=preflight.setup.rubric.artifact_id,
            calibration_id=preflight.calibration_id,
            calibration_input=preflight.calibration_input,
            audit_input=preflight.judge_provenance.audit_input,
        )

    def setup_supplier(
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview | ProvisionalRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Build the evaluation setup from preflighted immutable inputs.

        Args:
            project: Project already verified by automatic preflight.
            build: Completed build already verified by automatic preflight.
            review: Approved review identities already cross-checked in preflight.
            budget: Shared composition budget already admitted before provider access.

        Returns:
            Evaluation setup binding candidates, grounding, pricing, and simulation controls.
        """
        del project, build, review, budget
        return RouterEvaluationSetup(
            candidates=preflight.candidates,
            observed_cells=artifacts.observed_cells,
            production_protocol=production_protocol,
            simulation_protocol=simulation_protocol,
            embedding_set_id=artifacts.router_embeddings.embedding_set_id,
            fit_rag_input=preflight.completed_build.fit_rag,
            pricing_snapshot_id=artifacts.pricing.pricing_snapshot_id,
            guard=KnnGuard(
                maximum_neighbors=32,
                minimum_paired_observations=8,
                relative_similarity_threshold=0.85,
                uncertainty_multiplier=1.96,
                quality_tolerance=0.02,
            ),
            incumbent_alias=preflight.incumbent_alias,
            judgment_status=preflight.judgment_status,
            world_model_settings=WorldModelSettings(
                world_model_alias=preflight.world_model_alias,
                grounded_world_model_input=preflight.completed_build.world_model,
                prompt_version=WORLD_MODEL_TEXT_PROMPT_VERSION,
                query_embedding=preflight.retrieval_embedding_reservation,
                maximum_output_tokens=options.simulation_maximum_output_tokens,
            ),
            simulation_completion_input=artifacts.simulation_completion_input,
            agent_id=preflight.project_config.project_id,
            seed=options.seed,
            maximum_steps=options.maximum_model_calls,
            maximum_concurrency=options.maximum_concurrency,
        )

    def simulator_factory(
        project: ProjectStore,
        plan: EvaluationPlan,
    ) -> WorldModelSimulator:
        """Bind the current plan to verified clients and fit-only grounding.

        Args:
            project: Project containing the immutable build and simulation artifacts.
            plan: Exact evaluation plan being executed by composition.

        Returns:
            Grounded simulator with candidate, world-model, and retrieval boundaries.

        Raises:
            AutomaticRouterError: The verified embedder no longer exposes an embedding client.
        """
        embedder = resolved.embedder
        if embedder.embedding_client is None:
            raise AutomaticRouterError("resolved embedder client disappeared after preflight")
        binding = RAGEmbedderBinding(
            client=embedder.embedding_client,
            snapshot=embedder.snapshot,
            maximum_attempts=options.router_embedding_maximum_attempts,
            input_usd_per_million_tokens=(
                preflight.retrieval_embedding_reservation.input_usd_per_million_tokens
            ),
        )
        retriever = load_fit_rag_retriever(
            project.artifacts,
            preflight.completed_build.fit_rag,
            embedder=binding,
        )
        world = resolved.world_model
        return WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=preflight.completed_build.task_set,
            fit_rag_input=preflight.completed_build.fit_rag,
            fit_retriever=retriever,
            candidate_models=resolved.candidates,
            world_models={preflight.world_model_alias: world},
            grounded_world_models={
                preflight.world_model_alias: bind_fit_grounded_world_model(
                    project.artifacts,
                    preflight.completed_build.world_model,
                    client=world.client,
                    fit_retriever=retriever,
                )
            },
            agent_factory=agent_factory,
            completion_contract_input=artifacts.simulation_completion_input,
            redacted_field_names=preflight.project_config.redacted_field_names,
            progress=progress,
        )

    return RouterWorkflowServices(
        review_supplier=review_supplier,
        setup_supplier=setup_supplier,
        simulator_factory=simulator_factory,
        judge=judge,
        runtime_catalog=runtime_catalog,
        evaluation_plan_inputs=(
            *((artifacts.attribution_input,) if artifacts.attribution_input is not None else ()),
            artifacts.runtime_capability_input,
            artifacts.execution_contract_input,
        ),
    )


def _protocols(
    preflight: AutomaticRouterPreflight,
    artifacts: AutomaticRouterArtifacts,
    project: ProjectStore,
) -> tuple[EvaluationProtocol, EvaluationProtocol]:
    """Create exact production and grounded-simulation protocol identities.

    Args:
        preflight: Approved rubric, calibration, world model, and project identity.
        artifacts: Frozen candidate pricing snapshot.
        project: Project store retained for a self-contained workflow signature.

    Returns:
        Production and world-model evaluation protocols.
    """
    del project
    shared = {
        "agent_id": preflight.project_config.project_id,
        "rubric_id": preflight.setup.rubric.artifact_id,
        "judge_calibration_id": preflight.calibration_id,
        "pricing_snapshot_id": artifacts.pricing.pricing_snapshot_id,
    }
    production_id = stable_id(
        "protocol",
        {"version": "automatic-router-production-v1", **shared},
    )
    simulation_id = stable_id(
        "protocol",
        {
            "version": "automatic-router-world-model-v1",
            **shared,
            "world_model": preflight.world_model.model_dump(mode="json"),
            "prompt": WORLD_MODEL_TEXT_PROMPT_VERSION,
        },
    )
    return (
        EvaluationProtocol(
            protocol_id=production_id,
            evidence_source="production",
            agent_id=preflight.project_config.project_id,
            simulator_id="production-import-v1",
            rubric_id=preflight.setup.rubric.artifact_id,
            judge_calibration_id=preflight.calibration_id,
            pricing_snapshot_id=artifacts.pricing.pricing_snapshot_id,
        ),
        EvaluationProtocol(
            protocol_id=simulation_id,
            evidence_source="world_model",
            agent_id=preflight.project_config.project_id,
            simulator_id="text-world-model-v1",
            world_model=preflight.world_model,
            simulator_prompt_id=WORLD_MODEL_TEXT_PROMPT_ID,
            rubric_id=preflight.setup.rubric.artifact_id,
            judge_calibration_id=preflight.calibration_id,
            pricing_snapshot_id=artifacts.pricing.pricing_snapshot_id,
        ),
    )

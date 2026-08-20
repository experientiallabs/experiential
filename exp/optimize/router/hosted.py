"""One noninteractive hosted Project path from prepared traces to a frozen router."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Never

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
)
from exp.common.core.money import nonincreasing_float_usd
from exp.common.models import (
    BillingSource,
    ModelCatalog,
    OperationEconomics,
    RouterCandidateSelection,
)
from exp.common.project import (
    ProjectBudgetConfiguration,
    ProjectBuildArtifacts,
    ProjectModelConfiguration,
    ProjectRetrievalConfiguration,
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    ProjectStage,
    ProjectStageEvent,
    ProjectStore,
    ProjectSystemConfiguration,
    artifact_input,
    restore_project_bundle,
)
from exp.common.project.paths import validate_local_id
from exp.optimize.router.attempt_authority import (
    HostedAttemptAuthority,
    HostedAttemptAuthorityError,
    HostedAttemptAuthorityStore,
    HostedProviderHazard,
    verify_hosted_attempt_resume,
)
from exp.optimize.router.automatic.preflight import AutomaticRouterOptions
from exp.optimize.router.automatic.provisional import prepare_hosted_provisional_judge
from exp.optimize.router.automatic.service import (
    AutomaticRouterArtifacts,
    AutomaticRouterPreflight,
    AutomaticRouterResult,
    optimize_project_router,
)
from exp.optimize.router.composition import RouterCandidateSetupPlan, RouterPolicyLock
from exp.optimize.router.fit.workflow import RouterFitWorkflowResult
from exp.optimize.router.hosted_preflight import (
    HostedPreflight,
    HostedRouterPreflightError,
    ResolvedHostedModels,
    bind_hosted_setup,
    preflight_hosted,
    require_resolvable_clients,
    resolve_hosted_models,
)
from exp.optimize.router.hosted_spend import (
    complete_component_entries,
    incurred_entries,
    optimization_entries,
    provider_spend_source_pairs,
    stage_ledger,
)
from exp.optimize.router.hosted_stages import (
    HostedEventSink,
    HostedStageBundle,
)
from exp.optimize.router.hosted_stages import (
    commit_completed_stage as _commit_completed_stage,
)
from exp.optimize.router.hosted_stages import (
    emit_hosted_event as _emit,
)
from exp.optimize.router.hosted_stages import (
    export_stage_bundle as _export_stage_bundle,
)
from exp.optimize.router.hosted_verification import verify_hosted_project
from exp.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendEntry,
    ProviderSpendLedger,
    load_provider_spend_ledger,
    persist_provider_spend_ledger,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.build import provider_free_build_review, select_build_review
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.retrieval import RAGEmbedderBinding, RAGLineageBinding, persist_trace_rag
from exp.simulation.world_model import persist_grounded_world_model


class HostedRouterWorkflowSetup(ContractModel):
    """Late secret-free hosted selections applied once after trace preparation."""

    system: ProjectSystemConfiguration
    models: ProjectModelConfiguration
    retrieval: ProjectRetrievalConfiguration = Field(default_factory=ProjectRetrievalConfiguration)
    budgets: ProjectBudgetConfiguration

    @model_validator(mode="after")
    def _require_router_shape(self) -> HostedRouterWorkflowSetup:
        """Require one incumbent among at least two selected candidates and a shared ceiling."""
        if len(self.models.candidates) < 2:
            raise ValueError("hosted router setup requires at least two candidates")
        if self.models.incumbent is None:
            raise ValueError("hosted router setup requires an incumbent candidate")
        if self.budgets.maximum_provider_cost_usd is None:
            raise ValueError("hosted router setup requires one finite provider-spend ceiling")
        return self


class HostedRouterWorkflowOptions(ContractModel):
    """Bounded noninteractive controls below the Project-owned spend ceiling."""

    maximum_judgments: int = Field(default=100, gt=0)
    maximum_router_feature_tokens: int = Field(default=8_192, gt=0)
    maximum_retrieval_query_tokens: int = Field(default=32_768, gt=0)
    maximum_judge_input_tokens: int = Field(default=32_768, gt=0)
    maximum_judge_output_tokens: int = Field(default=4_096, ge=4_096)
    simulation_maximum_output_tokens: int = Field(default=16_000, gt=0)
    maximum_concurrency: int = Field(default=1, gt=0)
    seed: int = 0


@dataclass(frozen=True)
class HostedRouterWorkflowResult:
    """Frozen router/report outputs, final ledger, events, bundles, and loaded runtime chain."""

    project_id: str
    policy_id: str
    report_id: str
    spend_ledger: ProviderSpendLedger
    events: tuple[ProjectStageEvent, ...]
    bundles: tuple[HostedStageBundle, ...]
    automatic: AutomaticRouterResult


class HostedRouterWorkflowError(RuntimeError):
    """Hosted provider work stopped with conservative spend and structured recovery evidence."""

    def __init__(
        self,
        *,
        ledger: ProviderSpendLedger,
        ledger_input: ArtifactInput,
        events: tuple[ProjectStageEvent, ...],
        bundles: tuple[HostedStageBundle, ...] = (),
    ) -> None:
        """Retain only customer-safe failed-closed evidence, never provider error text."""
        super().__init__("hosted router workflow failed closed after a provider reservation")
        self.ledger = ledger
        self.ledger_input = ledger_input
        self.events = events
        self.bundles = bundles


def restore_hosted_project_bundle(
    bundle_path: Path,
    *,
    root: Path,
    expected_sha256: Sha256,
) -> ProjectStore:
    """Atomically restore a bundle only after its complete hosted graph verifies.

    Args:
        bundle_path: Downloaded canonical Project bundle.
        root: Local EXP root that will own the restored Project.
        expected_sha256: Exact externally committed bundle digest.

    Returns:
        Newly visible Project store with verified hosted semantics.
    """
    return restore_project_bundle(
        bundle_path,
        root=root,
        expected_sha256=expected_sha256,
        verify_project=verify_hosted_project,
    )


def run_hosted_router_workflow(
    project: ProjectStore,
    setup: HostedRouterWorkflowSetup,
    catalog: ModelCatalog,
    runtime_catalog: RuntimeModelCatalog,
    attempt_authority_store: HostedAttemptAuthorityStore,
    *,
    bundle_directory: Path,
    attempt_id: str,
    created_at: datetime,
    code_revision: str,
    resume_bundle_sha256: Sha256 | None = None,
    options: HostedRouterWorkflowOptions | None = None,
    event_sink: HostedEventSink | None = None,
) -> HostedRouterWorkflowResult:
    """Run the only supported hosted path from a restored prepared Project to a report.

    Args:
        project: Restored Project containing selected provider-free trace and task evidence.
        setup: One late built-in system, model-role, retrieval, and finite-budget selection.
        catalog: Transient provider catalog containing connection resolution only in memory.
        runtime_catalog: Injected credential and provider-client construction seam.
        attempt_authority_store: Durable monotonic spend authority surviving worker replacement.
        bundle_directory: Caller-owned destination for each completed-stage bundle.
        attempt_id: Stable durable attempt identity reused for safe bundle restart.
        created_at: Stable attempt timestamp reused across exact restart.
        code_revision: Exact producer revision for new immutable artifacts.
        resume_bundle_sha256: Exact externally committed bundle restored for this invocation.
        options: Optional bounded automatic controls.
        event_sink: Optional transport-neutral stage-event consumer.

    Returns:
        Frozen policy/report, final component ledger, typed events, bundles, and runtime.

    Raises:
        HostedRouterPreflightError: Local prerequisites fail before provider dispatch.
        HostedRouterWorkflowError: Reserved provider work becomes ambiguous and fails closed.
    """
    try:
        verify_hosted_project(project)
    except ValueError as exc:
        raise HostedRouterPreflightError("hosted Project semantic graph is invalid") from exc
    attempt_id = validate_local_id(attempt_id, label="hosted attempt ID")
    try:
        authority = attempt_authority_store.load(attempt_id)
        _verify_selected_attempt_authority(project, authority)
    except (HostedAttemptAuthorityError, ValueError) as exc:
        raise HostedRouterPreflightError("hosted attempt authority is invalid or stale") from exc
    active_options = options or HostedRouterWorkflowOptions()
    preflight = preflight_hosted(project, setup, catalog, active_options)
    total_ceiling = setup.budgets.maximum_provider_cost_usd
    assert total_ceiling is not None
    try:
        attempt_state = attempt_authority_store.bind(
            authority,
            project_id=project.paths.project_id,
            ceiling_usd=total_ceiling,
        )
    except HostedAttemptAuthorityError as exc:
        raise HostedRouterPreflightError(
            "hosted attempt authority has another Project or spend ceiling binding"
        ) from exc
    _raise_unresolved_hazard(
        project,
        preflight,
        attempt_authority_store,
        authority,
        created_at,
        code_revision,
    )
    selected_config = project.load_project()
    selected_stage = (
        ProjectStage.COMPLETING_REPORT
        if selected_config.router_report is not None
        else ProjectStage.OPTIMIZING_ROUTER
        if selected_config.router_policy is not None
        else ProjectStage.BUILDING_WORLD_MODEL
        if selected_config.build is not None
        else None
    )
    try:
        verify_hosted_attempt_resume(
            attempt_state,
            project_id=project.paths.project_id,
            selected_stage=selected_stage,
            resume_bundle_sha256=resume_bundle_sha256,
        )
    except HostedAttemptAuthorityError as exc:
        raise HostedRouterPreflightError(
            "restored Project differs from its external attempt stage pointer"
        ) from exc
    require_resolvable_clients(runtime_catalog, preflight, setup, active_options)
    bind_hosted_setup(project, setup, preflight, created_at, code_revision)
    try:
        verify_hosted_project(project)
    except ValueError as exc:
        raise HostedRouterPreflightError("hosted Project setup is invalid") from exc
    resolved = resolve_hosted_models(runtime_catalog, preflight)
    events: list[ProjectStageEvent] = []
    bundles: list[HostedStageBundle] = []
    try:
        build_ledger = _ensure_grounded_build(
            project,
            setup,
            preflight,
            resolved,
            attempt_authority_store,
            authority,
            bundle_directory=Path(bundle_directory),
            attempt_id=attempt_id,
            created_at=created_at,
            code_revision=code_revision,
            events=events,
            bundles=bundles,
            event_sink=event_sink,
        )
    except HostedRouterWorkflowError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve safe build ambiguity evidence
        _raise_failed_build(
            project,
            attempt_id,
            created_at,
            code_revision,
            setup,
            preflight,
            attempt_authority_store,
            authority,
            events,
            bundles,
            event_sink,
            exc,
        )
    remaining = total_ceiling - build_ledger.total_usd
    if remaining <= 0:
        raise HostedRouterPreflightError(
            "hosted provider ceiling has no safely remaining router-optimization budget"
        )
    judge = prepare_hosted_provisional_judge(
        project,
        preflight.catalog,
        maximum_input_tokens=active_options.maximum_judge_input_tokens,
        maximum_output_tokens=active_options.maximum_judge_output_tokens,
        maximum_attempts=RetryPolicy().maximum_attempts,
        created_at=created_at,
        code_revision=code_revision,
    )
    automatic_options = _automatic_options(setup, active_options, remaining)
    selection = RouterCandidateSelection(
        candidates=setup.models.candidates,
        incumbent=str(setup.models.incumbent),
    )
    candidate_plan = RouterCandidateSetupPlan(
        selection=selection,
        candidate_models=(),
        prospective_catalog=preflight.catalog,
        expected_catalog_sha256=sha256_json(preflight.catalog.model_dump(mode="json")),
    )
    prior_entries = incurred_entries(build_ledger.entries)
    source_pairs = _preflight_source_pairs(preflight)
    config = project.load_project()
    starting_stage = (
        ProjectStage.COMPLETING_REPORT
        if config.router_policy is not None
        else ProjectStage.OPTIMIZING_ROUTER
    )
    policy_ledger: ProviderSpendLedger | None = None
    if config.router_policy is not None:
        policy_ledger = load_provider_spend_ledger(
            project.artifacts,
            config.router_policy.spend_ledger,
        )
        if (
            policy_ledger.stage != ProjectStage.OPTIMIZING_ROUTER
            or policy_ledger.attempt_id != attempt_id
            or policy_ledger.attempt_authority_sha256 != authority.authority_sha256
            or policy_ledger.ceiling_usd != total_ceiling
            or policy_ledger.outcome != "completed"
        ):
            raise HostedRouterPreflightError(
                "hosted committed policy ledger differs from the active attempt"
            )
    if config.router_report is None:
        _emit(events, event_sink, project, attempt_id, created_at, starting_stage, "started")
        attempt_authority_store.begin(
            HostedProviderHazard(
                project_id=project.paths.project_id,
                attempt_id=attempt_id,
                authority_sha256=authority.authority_sha256,
                stage=starting_stage,
                reservations=(
                    preflight.report_reservations
                    if starting_stage == ProjectStage.COMPLETING_REPORT
                    else preflight.fit_reservations
                ),
            ),
        )

    def policy_checkpoint(
        lock: RouterPolicyLock,
        fit: RouterFitWorkflowResult,
        automatic_preflight: AutomaticRouterPreflight,
        artifacts: AutomaticRouterArtifacts,
        judge_economics: tuple[OperationEconomics, ...],
    ) -> None:
        """Select and export the immutable fit-only policy before held-out work opens."""
        del judge_economics
        nonlocal policy_ledger
        entries = optimization_entries(
            project,
            lock,
            automatic_preflight,
            artifacts,
            prior_entries,
            purpose="fit",
        )
        policy_input = artifact_input(project.artifacts.read(fit.locked.policy.policy_id).manifest)
        lock_input = artifact_input(project.artifacts.read(lock.lock_id).manifest)
        ledger, ledger_input = stage_ledger(
            project,
            selected=project.load_project().router_policy,
            stage=ProjectStage.OPTIMIZING_ROUTER,
            attempt_id=attempt_id,
            attempt_authority_sha256=authority.authority_sha256,
            ceiling_usd=total_ceiling,
            entries=entries,
            stage_outputs=(lock_input, policy_input),
            created_at=created_at,
            code_revision=code_revision,
        )
        policy_ledger = ledger
        previous = project.load_project().router_policy
        project.bind_router_policy(
            ProjectRouterPolicyArtifacts(
                policy_lock=lock_input,
                policy=policy_input,
                spend_ledger=ledger_input,
            )
        )
        if previous is None:
            stage_bundle = _export_stage_bundle(
                project,
                bundles,
                bundle_directory=Path(bundle_directory),
                attempt_id=attempt_id,
                code_revision=code_revision,
                stage=ProjectStage.OPTIMIZING_ROUTER,
            )
            _commit_completed_stage(
                attempt_authority_store,
                authority,
                project,
                stage_bundle,
                events,
                event_sink,
                occurred_at=created_at,
                outputs=(lock_input, policy_input, ledger_input),
                spend_ledger=ledger,
                spend_ledger_input=ledger_input,
            )
            _emit(
                events,
                event_sink,
                project,
                attempt_id,
                created_at,
                ProjectStage.COMPLETING_REPORT,
                "started",
            )
            attempt_authority_store.begin(
                HostedProviderHazard(
                    project_id=project.paths.project_id,
                    attempt_id=attempt_id,
                    authority_sha256=authority.authority_sha256,
                    stage=ProjectStage.COMPLETING_REPORT,
                    reservations=preflight.report_reservations,
                )
            )

    try:
        automatic = optimize_project_router(
            project,
            candidate_plan,
            resolved.runtime_catalog,
            options=automatic_options,
            provider_spend_consented=True,
            created_at=created_at,
            code_revision=code_revision,
            hosted_policy_checkpoint=policy_checkpoint,
            hosted_judge=judge,
            transient_catalog=True,
        )
        report = automatic.composition.optimization.optimization.report
        policy_lock = RouterPolicyLock.model_validate_json(
            project.artifacts.read_bytes(
                automatic.composition.policy_lock_id,
                "lock.json",
            )
        )
        final_entries = optimization_entries(
            project,
            policy_lock,
            automatic.preflight,
            automatic.artifacts,
            prior_entries,
            purpose="all",
        )
        report_input = artifact_input(project.artifacts.read(report.report_id).manifest)
        final_ledger, final_ledger_input = stage_ledger(
            project,
            selected=project.load_project().router_report,
            stage=ProjectStage.COMPLETING_REPORT,
            attempt_id=attempt_id,
            attempt_authority_sha256=authority.authority_sha256,
            ceiling_usd=total_ceiling,
            entries=final_entries,
            stage_outputs=(report_input,),
            created_at=created_at,
            code_revision=code_revision,
        )
        previous_report = project.load_project().router_report
        project.bind_router_report(
            ProjectRouterReportArtifacts(
                report=report_input,
                spend_ledger=final_ledger_input,
            )
        )
        if previous_report is None:
            stage_bundle = _export_stage_bundle(
                project,
                bundles,
                bundle_directory=Path(bundle_directory),
                attempt_id=attempt_id,
                code_revision=code_revision,
                stage=ProjectStage.COMPLETING_REPORT,
            )
            _commit_completed_stage(
                attempt_authority_store,
                authority,
                project,
                stage_bundle,
                events,
                event_sink,
                occurred_at=created_at,
                outputs=(report_input, final_ledger_input),
                spend_ledger=final_ledger,
                spend_ledger_input=final_ledger_input,
            )
        return HostedRouterWorkflowResult(
            project_id=project.paths.project_id,
            policy_id=automatic.composition.optimization.optimization.policy.policy_id,
            report_id=report.report_id,
            spend_ledger=final_ledger,
            events=tuple(events),
            bundles=tuple(bundles),
            automatic=automatic,
        )
    except HostedRouterWorkflowError:
        raise
    except Exception:  # noqa: BLE001 - provider text must not cross durable boundaries
        baseline = (
            incurred_entries(policy_ledger.entries) if policy_ledger is not None else prior_entries
        )
        hazard = _require_unresolved_hazard(attempt_authority_store, authority)
        attempt_authority_store.mark_ambiguous(hazard)
        failed_entries = complete_component_entries(
            (*baseline, *hazard.reservations),
            source_pairs,
        )
        failed, failed_input = persist_provider_spend_ledger(
            project.artifacts,
            project_id=project.paths.project_id,
            stage=hazard.stage,
            attempt_id=attempt_id,
            attempt_authority_sha256=authority.authority_sha256,
            ceiling_usd=total_ceiling,
            entries=failed_entries,
            outcome="failed_closed",
            created_at=created_at,
            code_revision=code_revision,
        )
        _emit(
            events,
            event_sink,
            project,
            attempt_id,
            created_at,
            hazard.stage,
            "failed",
        )
        raise HostedRouterWorkflowError(
            ledger=failed,
            ledger_input=failed_input,
            events=tuple(events),
            bundles=tuple(bundles),
        ) from None


def _ensure_grounded_build(
    project: ProjectStore,
    setup: HostedRouterWorkflowSetup,
    preflight: HostedPreflight,
    resolved: ResolvedHostedModels,
    attempt_authority_store: HostedAttemptAuthorityStore,
    authority: HostedAttemptAuthority,
    *,
    bundle_directory: Path,
    attempt_id: str,
    created_at: datetime,
    code_revision: str,
    events: list[ProjectStageEvent],
    bundles: list[HostedStageBundle],
    event_sink: HostedEventSink | None,
) -> ProviderSpendLedger:
    """Reuse or construct the grounded build and select its conservative build ledger."""
    config = project.load_project()
    if config.build is not None:
        if config.build_spend_ledger is None:
            raise HostedRouterPreflightError("hosted completed build has no spend ledger")
        ledger = load_provider_spend_ledger(project.artifacts, config.build_spend_ledger)
        if ledger.attempt_id != attempt_id or ledger.stage != ProjectStage.BUILDING_WORLD_MODEL:
            raise HostedRouterPreflightError("hosted build ledger differs from the active attempt")
        select_build_review(project, provider_free_build_review(project))
        return ledger
    _emit(
        events,
        event_sink,
        project,
        attempt_id,
        created_at,
        ProjectStage.BUILDING_WORLD_MODEL,
        "started",
    )
    attempt_authority_store.begin(
        HostedProviderHazard(
            project_id=project.paths.project_id,
            attempt_id=attempt_id,
            authority_sha256=authority.authority_sha256,
            stage=ProjectStage.BUILDING_WORLD_MODEL,
            reservations=preflight.build_reservations,
        ),
    )
    stage = project.load_project().provider_free_stage
    assert stage is not None
    review = provider_free_build_review(project)
    lineage = load_task_set_lineage_bindings(project.artifacts, stage.task_set.artifact_id)
    bindings = tuple(
        RAGLineageBinding(
            trace_id=item.trace_id,
            lineage_id=item.lineage_id,
            partition=item.partition,
        )
        for item in lineage.bindings
    )
    embedder = resolved.embedder
    assert embedder.embedding_client is not None
    price = preflight.embedder_capabilities.input_cost_per_million_tokens_usd
    assert price is not None
    binding = RAGEmbedderBinding(
        client=embedder.embedding_client,
        snapshot=embedder.snapshot,
        maximum_attempts=RetryPolicy().maximum_attempts,
        input_usd_per_million_tokens=price,
    )
    try:
        serving = persist_trace_rag(
            project.artifacts,
            (stage.trace_dataset,),
            bindings,
            created_at=created_at,
            code_revision=code_revision,
            embedder=binding,
            default_top_k=setup.retrieval.top_k,
            included_partitions=frozenset({"fit", "held_out"}),
        )
        fit = persist_trace_rag(
            project.artifacts,
            (stage.trace_dataset,),
            bindings,
            created_at=created_at,
            code_revision=code_revision,
            embedder=binding,
            default_top_k=setup.retrieval.top_k,
            included_partitions=frozenset({"fit"}),
        )
        world = persist_grounded_world_model(
            project.artifacts,
            artifact_input(serving.manifest),
            model_alias=resolved.world_model.alias,
            model=resolved.world_model.snapshot,
            created_at=created_at,
            code_revision=code_revision,
            top_k=setup.retrieval.top_k,
        )
    except Exception as exc:  # noqa: BLE001 - convert provider failures to safe ledger evidence
        _raise_failed_build(
            project,
            attempt_id,
            created_at,
            code_revision,
            setup,
            preflight,
            attempt_authority_store,
            authority,
            events,
            bundles,
            event_sink,
            exc,
        )
    build = ProjectBuildArtifacts(
        trace_dataset=stage.trace_dataset,
        task_set=stage.task_set,
        serving_rag=artifact_input(serving.manifest),
        fit_rag=artifact_input(fit.manifest),
        world_model=artifact_input(world.manifest),
    )
    ceiling = setup.budgets.maximum_provider_cost_usd
    assert ceiling is not None
    ledger, ledger_input = persist_provider_spend_ledger(
        project.artifacts,
        project_id=project.paths.project_id,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
        attempt_id=attempt_id,
        attempt_authority_sha256=authority.authority_sha256,
        ceiling_usd=ceiling,
        entries=complete_component_entries(
            preflight.build_reservations,
            _preflight_source_pairs(preflight),
        ),
        stage_outputs=(
            build.trace_dataset,
            build.task_set,
            build.serving_rag,
            build.fit_rag,
            build.world_model,
        ),
        outcome="completed",
        created_at=created_at,
        code_revision=code_revision,
    )
    project.bind_hosted_completed_build(build, spend_ledger=ledger_input)
    select_build_review(project, review)
    stage_bundle = _export_stage_bundle(
        project,
        bundles,
        bundle_directory=bundle_directory,
        attempt_id=attempt_id,
        code_revision=code_revision,
        stage=ProjectStage.BUILDING_WORLD_MODEL,
    )
    _commit_completed_stage(
        attempt_authority_store,
        authority,
        project,
        stage_bundle,
        events,
        event_sink,
        occurred_at=created_at,
        outputs=(
            build.trace_dataset,
            build.task_set,
            build.serving_rag,
            build.fit_rag,
            build.world_model,
            ledger_input,
        ),
        spend_ledger=ledger,
        spend_ledger_input=ledger_input,
    )
    return ledger


def _raise_failed_build(
    project: ProjectStore,
    attempt_id: str,
    created_at: datetime,
    code_revision: str,
    setup: HostedRouterWorkflowSetup,
    preflight: HostedPreflight,
    attempt_authority_store: HostedAttemptAuthorityStore,
    authority: HostedAttemptAuthority,
    events: list[ProjectStageEvent],
    bundles: list[HostedStageBundle],
    event_sink: HostedEventSink | None,
    cause: Exception,
) -> Never:
    """Persist a conservative build reservation and raise a safe workflow failure."""
    del cause
    hazard = _require_unresolved_hazard(attempt_authority_store, authority)
    attempt_authority_store.mark_ambiguous(hazard)
    ceiling = setup.budgets.maximum_provider_cost_usd
    assert ceiling is not None
    ledger, ledger_input = persist_provider_spend_ledger(
        project.artifacts,
        project_id=project.paths.project_id,
        stage=hazard.stage,
        attempt_id=attempt_id,
        attempt_authority_sha256=authority.authority_sha256,
        ceiling_usd=ceiling,
        entries=complete_component_entries(
            hazard.reservations,
            _preflight_source_pairs(preflight),
        ),
        outcome="failed_closed",
        created_at=created_at,
        code_revision=code_revision,
    )
    _emit(
        events,
        event_sink,
        project,
        attempt_id,
        created_at,
        hazard.stage,
        "failed",
    )
    raise HostedRouterWorkflowError(
        ledger=ledger,
        ledger_input=ledger_input,
        events=tuple(events),
        bundles=tuple(bundles),
    ) from None


def _automatic_options(
    setup: HostedRouterWorkflowSetup,
    options: HostedRouterWorkflowOptions,
    maximum_provider_cost_usd: Decimal,
) -> AutomaticRouterOptions:
    """Translate hosted controls into the current automatic-router contract."""
    return AutomaticRouterOptions(
        maximum_provider_cost_usd=nonincreasing_float_usd(maximum_provider_cost_usd),
        maximum_judgments=options.maximum_judgments,
        maximum_model_calls=setup.system.maximum_model_calls,
        maximum_router_feature_tokens=options.maximum_router_feature_tokens,
        maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
        router_embedding_maximum_attempts=RetryPolicy().maximum_attempts,
        completion_maximum_attempts=RetryPolicy().maximum_attempts,
        simulation_maximum_output_tokens=options.simulation_maximum_output_tokens,
        maximum_concurrency=options.maximum_concurrency,
        seed=options.seed,
    )


def _raise_unresolved_hazard(
    project: ProjectStore,
    preflight: HostedPreflight,
    attempt_store: HostedAttemptAuthorityStore,
    authority: HostedAttemptAuthority,
    created_at: datetime,
    code_revision: str,
) -> None:
    """Permanently fail closed on ambiguity retained beyond any Project bundle restore."""
    hazard = attempt_store.unresolved(authority)
    if hazard is None:
        return
    if hazard.project_id != project.paths.project_id:
        raise HostedRouterPreflightError("provider reservation belongs to another Project")
    attempt_store.mark_ambiguous(hazard)
    config = project.load_project()
    ceiling = config.budgets.maximum_provider_cost_usd if config.budgets is not None else None
    if ceiling is None:
        raise HostedRouterPreflightError("unresolved provider reservation has no bound ceiling")
    prior_entries: tuple[ProviderSpendEntry, ...] = ()
    if hazard.stage == ProjectStage.COMPLETING_REPORT and config.router_policy is not None:
        prior_entries = incurred_entries(
            load_provider_spend_ledger(
                project.artifacts,
                config.router_policy.spend_ledger,
            ).entries
        )
    elif hazard.stage != ProjectStage.BUILDING_WORLD_MODEL:
        if config.build_spend_ledger is None:
            raise HostedRouterPreflightError(
                "unresolved provider reservation has no completed-stage spend baseline"
            )
        prior_entries = incurred_entries(
            load_provider_spend_ledger(project.artifacts, config.build_spend_ledger).entries
        )
    ledger, ledger_input = persist_provider_spend_ledger(
        project.artifacts,
        project_id=project.paths.project_id,
        stage=hazard.stage,
        attempt_id=authority.attempt_id,
        attempt_authority_sha256=authority.authority_sha256,
        ceiling_usd=ceiling,
        entries=complete_component_entries(
            (*prior_entries, *hazard.reservations),
            _preflight_source_pairs(preflight),
        ),
        outcome="failed_closed",
        created_at=created_at,
        code_revision=code_revision,
    )
    raise HostedRouterWorkflowError(ledger=ledger, ledger_input=ledger_input, events=())


def _preflight_source_pairs(
    preflight: HostedPreflight,
) -> tuple[tuple[ProviderSpendComponent, BillingSource], ...]:
    """Return the complete component and billing-source plan for one hosted setup."""
    return provider_spend_source_pairs(
        candidates=tuple(item.model for item in preflight.candidates),
        world_model=preflight.world_model.model,
        judge=preflight.judge.model,
        embedder=preflight.embedder.model,
    )


def _require_unresolved_hazard(
    attempt_store: HostedAttemptAuthorityStore,
    authority: HostedAttemptAuthority,
) -> HostedProviderHazard:
    """Return the active paid-operation reservation needed for conservative failure evidence."""
    hazard = attempt_store.unresolved(authority)
    if hazard is None:
        raise HostedRouterPreflightError("hosted provider reservation is absent after failure")
    return hazard


def _verify_selected_attempt_authority(
    project: ProjectStore,
    authority: HostedAttemptAuthority,
) -> None:
    """Require every selected stage ledger to bind the same external attempt authority."""
    config = project.load_project()
    ceiling = config.budgets.maximum_provider_cost_usd if config.budgets is not None else None
    selected = (
        (config.build_spend_ledger, ProjectStage.BUILDING_WORLD_MODEL),
        (
            config.router_policy.spend_ledger if config.router_policy is not None else None,
            ProjectStage.OPTIMIZING_ROUTER,
        ),
        (
            config.router_report.spend_ledger if config.router_report is not None else None,
            ProjectStage.COMPLETING_REPORT,
        ),
    )
    for pointer, stage in selected:
        if pointer is None:
            continue
        ledger = load_provider_spend_ledger(project.artifacts, pointer)
        if (
            ledger.project_id != project.paths.project_id
            or ledger.stage != stage
            or ledger.attempt_id != authority.attempt_id
            or ledger.attempt_authority_sha256 != authority.authority_sha256
            or ledger.ceiling_usd != ceiling
            or ledger.outcome != "completed"
        ):
            raise HostedRouterPreflightError(
                "selected stage ledger differs from the active Project attempt authority"
            )

"""Read-only aggregate preflight for automatic completed-build router optimization."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from wmo.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from wmo.common.judging import CalibrationReport, JudgeCalibration, verify_persisted_calibration
from wmo.common.models import (
    CandidateTokenPrice,
    CompletionCostReservation,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelCatalog,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
    load_model_catalog,
    router_candidate_prices,
    validate_router_candidate_selection,
)
from wmo.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.routing import RouterEmbeddingReservation
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.optimize.router.automatic.attribution import (
    RouterAttributionError,
    RouterObservedAttribution,
    resolve_router_observed_attributions,
)
from wmo.optimize.router.automatic.reservations import (
    AutomaticRouterCostPlan,
    AutomaticRouterOptions,
    judge_completion_reservation,
    plan_automatic_router_cost,
    remaining_simulation_budget,
    retrieval_embedding_reservation,
    router_feature_reservation,
    simulation_completion_reservations,
)
from wmo.optimize.router.judging.artifacts import read_audit
from wmo.optimize.router.judging.contracts import (
    ManualJudgeCalibrationAudit,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
)
from wmo.runtime.agents import agent_factory_sha256
from wmo.runtime.models import RuntimeModelCatalog
from wmo.simulation.ingest.dataset import (
    read_trace_model_identity_evidence,
    verify_current_trace_dataset,
)
from wmo.simulation.ingest.model_identity import TraceModelIdentityEvidenceSet
from wmo.simulation.specs import CandidateCompletionReservation


class AutomaticRouterPreflightError(ValueError):
    """One or more automatic router prerequisites are missing or inconsistent."""


@dataclass(frozen=True)
class ObservedRouterTrace:
    """One real fit task and trace attributed to an exact selected candidate."""

    task: TaskCase
    trace: Trace
    candidate_alias: str
    attribution: RouterObservedAttribution


class ProvisionalAutomaticJudge(ContractModel):
    """Verified zero-label judge provenance eligible for provisional optimization only."""

    judgment_status: Literal["provisional"] = "provisional"
    calibration_id: str
    calibration_input: ArtifactInput


class HumanCalibratedAutomaticJudge(ContractModel):
    """Verified human-approved judge provenance and its completed calibration audit."""

    judgment_status: Literal["human_calibrated"] = "human_calibrated"
    calibration_id: str
    calibration_input: ArtifactInput
    audit: ManualJudgeCalibrationAudit
    audit_input: ArtifactInput


AutomaticJudgeProvenance = Annotated[
    ProvisionalAutomaticJudge | HumanCalibratedAutomaticJudge,
    Field(discriminator="judgment_status"),
]


@dataclass(frozen=True)
class AutomaticRouterPreflight:
    """Verified read-only inputs ready for post-consent artifact and provider work."""

    project_config: ProjectConfig
    completed_build: ProjectBuildArtifacts
    catalog: ModelCatalog
    catalog_sha256: Sha256
    candidates: tuple[RoutedCandidateSnapshot, ...]
    candidate_prices: tuple[CandidateTokenPrice, ...]
    incumbent_alias: str
    world_model_alias: str
    world_model: ModelSnapshot
    judge_alias: str
    judge_model: ModelSnapshot
    embedder_alias: str
    embedder: ModelSnapshot
    setup: ManualJudgeSetupArtifact
    setup_input: ArtifactInput
    judge_provenance: AutomaticJudgeProvenance
    tasks: tuple[TaskCase, ...]
    traces: tuple[Trace, ...]
    trace_identity_evidence: TraceModelIdentityEvidenceSet | None
    observed_traces: tuple[ObservedRouterTrace, ...]
    cost_plan: AutomaticRouterCostPlan
    router_embedding_reservation: RouterEmbeddingReservation
    retrieval_embedding_reservation: EmbeddingCostReservation
    candidate_completion_reservations: tuple[CandidateCompletionReservation, ...]
    world_model_completion_reservation: CompletionCostReservation
    judge_completion_reservation: CompletionCostReservation
    judge_provider_call_count: int
    judge_reservation_cost_usd: float
    remaining_simulation_cost_usd: float
    agent_factory_sha256: Sha256
    simulation_configuration_sha256: Sha256

    @property
    def calibration_id(self) -> str:
        """Return the selected provisional or human-calibrated artifact identity."""
        return self.judge_provenance.calibration_id

    @property
    def calibration_input(self) -> ArtifactInput:
        """Return the exact selected calibration manifest input."""
        return self.judge_provenance.calibration_input

    @property
    def judgment_status(self) -> Literal["provisional", "human_calibrated"]:
        """Return the exact eligibility status carried into router artifacts."""
        return self.judge_provenance.judgment_status

    @property
    def judge_provenance_inputs(self) -> tuple[ArtifactInput, ...]:
        """Return exact calibration provenance inputs for recursive binding."""
        if isinstance(self.judge_provenance, HumanCalibratedAutomaticJudge):
            return (
                self.judge_provenance.audit_input,
                self.judge_provenance.calibration_input,
            )
        return (self.judge_provenance.calibration_input,)


def preflight_automatic_router(
    project: ProjectStore,
    selection: RouterCandidateSelection,
    *,
    catalog_override: ModelCatalog | None = None,
    options: AutomaticRouterOptions,
) -> AutomaticRouterPreflight:
    """Verify every local prerequisite and report all failures before credentials or writes.

    Args:
        project: Existing project whose completed build will be optimized.
        selection: Explicit candidates and incumbent collected for this optimize run.
        catalog_override: Confirmed prospective catalog before its atomic post-consent write.
        options: Bounded provider, evidence, retry, and concurrency controls.

    Returns:
        Fully verified local inputs and conservative router-feature reservation.

    Raises:
        AutomaticRouterPreflightError: One or more prerequisites are missing or inconsistent.
    """
    problems = list(_validate_positive_options(options))
    if options.maximum_provider_cost_usd <= 0 or not math.isfinite(
        options.maximum_provider_cost_usd
    ):
        problems.append("maximum_provider_cost_usd must be positive and finite")
    config = _capture(problems, "project", project.load_project)
    catalog = catalog_override or _capture(
        problems, "model catalog", lambda: load_model_catalog(project.model_catalog_path)
    )
    if config is None or catalog is None:
        raise _preflight_error(problems)
    completed = config.build
    if completed is None:
        problems.append("completed build: run `wmo build PROJECT --traces PATH` first")
    else:
        problems.extend(_completed_build_problems(project, completed))
    problems.extend(validate_router_candidate_selection(catalog, selection))
    resolver = RuntimeModelCatalog(catalog, environment={})
    candidates = _candidate_snapshots(problems, resolver, selection)
    role_values = _project_model_roles(problems, config)
    world_alias, judge_alias, embedder_alias = role_values
    world = _role_snapshot(problems, resolver, world_alias, "world model")
    judge = _role_snapshot(problems, resolver, judge_alias, "judge")
    embedder = _role_snapshot(problems, resolver, embedder_alias, "embedder")
    if world_alias is not None:
        _require_completion_economics(problems, catalog, world_alias, "world model")
    if judge_alias is not None:
        _require_completion_economics(problems, catalog, judge_alias, "judge")
    if embedder_alias is not None:
        _require_embedder_economics(problems, catalog, embedder_alias)
    setup, setup_input, judge_provenance = _manual_judge_inputs(
        problems,
        project,
        completed,
        judge_alias,
        judge,
    )
    tasks, traces, identity_evidence = _build_evidence(problems, project, completed)
    observed = _observed_traces(
        problems,
        tasks,
        traces,
        identity_evidence,
        candidates,
    )
    try:
        agent_identity = agent_factory_sha256(
            config.agent,
            maximum_model_calls=options.maximum_model_calls,
        )
    except ValueError as exc:
        problems.append(f"agent runtime: {exc}")
        agent_identity = None
    reservation = router_feature_reservation(
        problems,
        catalog,
        embedder_alias,
        embedder,
        tasks,
        options.maximum_router_feature_tokens,
        options.router_embedding_maximum_attempts,
    )
    query_reservation = retrieval_embedding_reservation(
        problems,
        catalog,
        embedder_alias,
        embedder,
        options.maximum_retrieval_query_tokens,
        options.router_embedding_maximum_attempts,
    )
    candidate_requests, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=candidates,
        world_alias=world_alias,
        world=world,
        maximum_attempts=options.completion_maximum_attempts,
        maximum_output_tokens=options.simulation_maximum_output_tokens,
    )
    judge_request = judge_completion_reservation(
        problems,
        catalog=catalog,
        judge_alias=judge_alias,
        judge=judge,
        audit=(
            judge_provenance.audit
            if isinstance(judge_provenance, HumanCalibratedAutomaticJudge)
            else None
        ),
        provisional=isinstance(judge_provenance, ProvisionalAutomaticJudge),
        provisional_maximum_attempts=options.completion_maximum_attempts,
    )
    cost_plan = None
    if setup is not None and judge_provenance is not None:
        try:
            cost_plan = plan_automatic_router_cost(
                tasks,
                catalog,
                selection,
                world_model_alias=world_alias or "",
                judge_alias=judge_alias or "",
                embedder_alias=embedder_alias or "",
                judge_response_shape=setup.prompt_template.response_shape,
                judge_audit=(
                    judge_provenance.audit
                    if isinstance(judge_provenance, HumanCalibratedAutomaticJudge)
                    else None
                ),
                provisional_judge=isinstance(judge_provenance, ProvisionalAutomaticJudge),
                observed_candidate_aliases=tuple(item.candidate_alias for item in observed),
                options=AutomaticRouterOptions(
                    maximum_provider_cost_usd=options.maximum_provider_cost_usd,
                    maximum_judgments=options.maximum_judgments,
                    maximum_model_calls=options.maximum_model_calls,
                    maximum_router_feature_tokens=options.maximum_router_feature_tokens,
                    maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
                    router_embedding_maximum_attempts=options.router_embedding_maximum_attempts,
                    completion_maximum_attempts=options.completion_maximum_attempts,
                    simulation_maximum_output_tokens=options.simulation_maximum_output_tokens,
                    maximum_concurrency=options.maximum_concurrency,
                    seed=options.seed,
                ),
            )
        except ValueError as exc:
            problems.append(str(exc))
    if cost_plan is not None:
        if options.maximum_judgments < cost_plan.maximum_judgments:
            problems.append(
                f"maximum_judgments must be at least {cost_plan.maximum_judgments} for the "
                "complete task-candidate evaluation schedule"
            )
        if options.maximum_provider_cost_usd < cost_plan.required_provider_cost_usd:
            problems.append(
                "maximum_simulation_cost_usd is below the complete automatic reservation; "
                f"requires at least ${cost_plan.required_provider_cost_usd:.6f}"
            )
    judge_provider_call_count = (
        cost_plan.maximum_judge_provider_calls if cost_plan is not None else 0
    )
    judge_reservation_cost_usd = (
        judge_request.estimated_maximum_call_cost_usd * judge_provider_call_count
        if judge_request is not None
        else 0.0
    )
    remaining_cost_usd = remaining_simulation_budget(
        problems,
        maximum_provider_cost_usd=options.maximum_provider_cost_usd,
        router_reservation=reservation,
        judge_reservation_cost_usd=judge_reservation_cost_usd,
    )
    if problems:
        raise _preflight_error(problems)
    assert completed is not None
    assert world_alias is not None and world is not None
    assert judge_alias is not None and judge is not None
    assert embedder_alias is not None and embedder is not None
    assert setup is not None and setup_input is not None
    assert judge_provenance is not None
    assert agent_identity is not None
    assert reservation is not None and query_reservation is not None and cost_plan is not None
    assert candidate_requests and world_request is not None and judge_request is not None
    return AutomaticRouterPreflight(
        project_config=config,
        completed_build=completed,
        catalog=catalog,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        candidates=candidates,
        candidate_prices=router_candidate_prices(
            catalog.model_copy(
                update={
                    "roles": catalog.roles.model_copy(
                        update={
                            "candidates": selection.candidates,
                            "incumbent": selection.incumbent,
                        }
                    )
                }
            )
        ),
        incumbent_alias=selection.incumbent,
        world_model_alias=world_alias,
        world_model=world,
        judge_alias=judge_alias,
        judge_model=judge,
        embedder_alias=embedder_alias,
        embedder=embedder,
        setup=setup,
        setup_input=setup_input,
        judge_provenance=judge_provenance,
        tasks=tasks,
        traces=traces,
        trace_identity_evidence=identity_evidence,
        observed_traces=observed,
        cost_plan=cost_plan,
        router_embedding_reservation=reservation,
        retrieval_embedding_reservation=query_reservation,
        candidate_completion_reservations=candidate_requests,
        world_model_completion_reservation=world_request,
        judge_completion_reservation=judge_request,
        judge_provider_call_count=judge_provider_call_count,
        judge_reservation_cost_usd=judge_reservation_cost_usd,
        remaining_simulation_cost_usd=remaining_cost_usd,
        agent_factory_sha256=agent_identity,
        simulation_configuration_sha256=sha256_json(
            {
                "version": "automatic-router-simulation-configuration-v1",
                "agent_factory_sha256": agent_identity,
                "redacted_field_names": list(config.redacted_field_names),
            }
        ),
    )


_BOUNDED_OPTION_FIELDS = (
    "maximum_model_calls",
    "maximum_router_feature_tokens",
    "maximum_retrieval_query_tokens",
    "router_embedding_maximum_attempts",
    "completion_maximum_attempts",
    "simulation_maximum_output_tokens",
    "maximum_judgments",
    "maximum_concurrency",
)


def _validate_positive_options(options: AutomaticRouterOptions) -> tuple[str, ...]:
    """List nonpositive bounded controls before reading project state.

    Args:
        options: Bounded workflow controls to validate.

    Returns:
        Actionable option problems.
    """
    return tuple(
        f"{name} must be positive" for name in _BOUNDED_OPTION_FIELDS if getattr(options, name) <= 0
    )


def _capture[T](problems: list[str], label: str, operation: Callable[[], T]) -> T | None:
    """Capture one read-only validation failure while allowing aggregate preflight to continue.

    Args:
        problems: Mutable aggregate problem list.
        label: Human-readable prerequisite name.
        operation: Zero-argument read-only validation callable.

    Returns:
        Callable result, or ``None`` after recording its exception.
    """
    try:
        return operation()
    except (OSError, ProjectStoreError, ValueError) as exc:
        problems.append(f"{label}: {exc}")
        return None


def _completed_build_problems(
    project: ProjectStore, completed: ProjectBuildArtifacts
) -> tuple[str, ...]:
    """Verify every completed-build pointer and its immediate immutable graph.

    Args:
        project: Project-local artifact store.
        completed: Exact mutable project pointers to verify.

    Returns:
        Pointer, type, digest, or dependency problems.
    """
    expected_types = {
        "trace_dataset": "trace-dataset",
        "task_set": "task-set",
        "serving_rag": "trace-rag-index",
        "fit_rag": "trace-rag-index",
        "world_model": "grounded-world-model",
    }
    problems = []
    for name, artifact_type in expected_types.items():
        pointer = getattr(completed, name)
        try:
            stored = project.artifacts.read(pointer.artifact_id)
            if stored.manifest.artifact_type != artifact_type:
                problems.append(
                    f"completed build {name} has type {stored.manifest.artifact_type!r}"
                )
            if artifact_input(stored.manifest) != pointer:
                problems.append(f"completed build {name} manifest digest changed")
        except (OSError, ValueError) as exc:
            problems.append(f"completed build {name}: {exc}")
    return tuple(problems)


def _candidate_snapshots(
    problems: list[str],
    resolver: RuntimeModelCatalog,
    selection: RouterCandidateSelection,
) -> tuple[RoutedCandidateSnapshot, ...]:
    """Resolve candidate identities without credentials and reject duplicate provider models.

    Args:
        problems: Mutable aggregate problem list.
        resolver: Static local catalog resolver.
        selection: Explicit candidate aliases.

    Returns:
        Successfully resolved candidate snapshots in canonical alias order.
    """
    resolved = []
    for alias in selection.candidates:
        try:
            snapshot, capabilities = resolver.snapshot(alias)
            resolved.append(RoutedCandidateSnapshot(alias=alias, model=snapshot))
        except ValueError as exc:
            problems.append(f"candidate {alias}: {exc}")
    model_ids = tuple(item.model for item in resolved)
    if len(set(model_ids)) != len(model_ids):
        problems.append(
            "selected candidate aliases must resolve to distinct exact model identities"
        )
    return tuple(sorted(resolved, key=lambda item: item.alias))


def _project_model_roles(
    problems: list[str], config: ProjectConfig
) -> tuple[str | None, str | None, str | None]:
    """Read build-frozen world, judge, and embedder roles.

    Args:
        problems: Mutable aggregate problem list.
        config: Project configuration selected during build.

    Returns:
        World-model, judge, and embedder aliases, possibly absent after recording a problem.
    """
    if config.models is None:
        problems.append("project build has no frozen model roles")
        return None, None, None
    return config.models.world_model, config.models.judge, config.models.embedder


def _role_snapshot(
    problems: list[str], resolver: RuntimeModelCatalog, alias: str | None, label: str
) -> ModelSnapshot | None:
    """Resolve one build role statically while retaining aggregate error reporting.

    Args:
        problems: Mutable aggregate problem list.
        resolver: Static local model resolver.
        alias: Configured role alias, if present.
        label: Human-readable role name.

    Returns:
        Exact model identity, or ``None`` after recording a problem.
    """
    if alias is None:
        return None
    try:
        return resolver.snapshot(alias)[0]
    except ValueError as exc:
        problems.append(f"{label} {alias!r}: {exc}")
        return None


def _require_completion_economics(
    problems: list[str], catalog: ModelCatalog, alias: str, label: str
) -> None:
    """Require explicit completion support, limits, and every billing unit for one role.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Local model metadata.
        alias: Model alias to inspect.
        label: Human-readable role name.
    """
    record = catalog.models.get(alias)
    capabilities = record.capabilities if record is not None else None
    missing = []
    if capabilities is None or capabilities.supports_completions is not True:
        missing.append("supports_completions=true")
    if capabilities is None or capabilities.context_window_tokens is None:
        missing.append("context_window_tokens")
    if capabilities is None or capabilities.maximum_output_tokens is None:
        missing.append("maximum_output_tokens")
    missing.extend(_missing_prices(capabilities))
    if missing:
        problems.append(f"{label} alias {alias!r} is missing " + ", ".join(missing))


def _require_embedder_economics(problems: list[str], catalog: ModelCatalog, alias: str) -> None:
    """Require explicit embedding support and input pricing for the build embedder.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Local model metadata.
        alias: Build-frozen embedder alias.
    """
    record = catalog.models.get(alias)
    capabilities = record.capabilities if record is not None else None
    missing = []
    if capabilities is None or not capabilities.supports_embeddings:
        missing.append("supports_embeddings=true")
    if capabilities is None or capabilities.input_cost_per_million_tokens_usd is None:
        missing.append("input_cost_per_million_tokens_usd")
    if missing:
        problems.append(f"embedder alias {alias!r} is missing " + ", ".join(missing))


def _missing_prices(capabilities: ModelCapabilities | None) -> tuple[str, ...]:
    """Return absent completion price field names in billing order.

    Args:
        capabilities: Optional model capability metadata.

    Returns:
        Missing explicit price field names.
    """
    if capabilities is None:
        return (
            "input_cost_per_million_tokens_usd",
            "output_cost_per_million_tokens_usd",
            "cached_input_cost_per_million_tokens_usd",
            "cache_write_cost_per_million_tokens_usd",
        )
    return tuple(
        name
        for name in (
            "input_cost_per_million_tokens_usd",
            "output_cost_per_million_tokens_usd",
            "cached_input_cost_per_million_tokens_usd",
            "cache_write_cost_per_million_tokens_usd",
        )
        if getattr(capabilities, name) is None
    )


def _manual_judge_inputs(
    problems: list[str],
    project: ProjectStore,
    completed: ProjectBuildArtifacts | None,
    judge_alias: str | None,
    judge_model: ModelSnapshot | None,
) -> tuple[
    ManualJudgeSetupArtifact | None,
    ArtifactInput | None,
    AutomaticJudgeProvenance | None,
]:
    """Verify approved setup and calibration bind the exact completed build and judge.

    Args:
        problems: Mutable aggregate problem list.
        project: Project-local review and artifact store.
        completed: Completed build pointers, if available.
        judge_alias: Build-frozen judge alias.
        judge_model: Exact current judge identity.

    Returns:
        Verified setup and typed calibration provenance, or absent values after a problem.
    """
    review = project.read_review()
    if not isinstance(review, dict) or review.get("manual_judge") is None:
        problems.append(
            "manual judge: run `wmo config judge setup PROJECT`, then "
            "`wmo config judge calibrate PROJECT --approve`"
        )
        return None, None, None
    try:
        state = ManualJudgeReviewState.model_validate(review["manual_judge"])
        setup = ManualJudgeSetupArtifact.model_validate_json(
            project.artifacts.read_bytes(state.setup.artifact_id, "setup.json")
        )
        setup_input = artifact_input(project.artifacts.read(state.setup.artifact_id).manifest)
        if setup_input != state.setup or setup.setup_id != state.setup.artifact_id:
            raise ValueError("approved setup pointer changed")
        selected_calibration = state.approved_calibration or state.provisional_calibration
        if selected_calibration is None:
            raise ValueError("calibration has neither provisional nor approved provenance")
        calibration, calibration_input = verify_persisted_calibration(
            project, selected_calibration.artifact_id
        )
        if calibration_input != selected_calibration:
            raise ValueError("selected calibration pointer changed")
        if state.approved_calibration is not None:
            if state.audit is None:
                raise ValueError("approved calibration has no completed audit")
            audit = read_audit(project, state.audit)
            _verify_manual_judge_chain(project, state, setup, audit, calibration)
            if calibration.status != "human_calibrated":
                raise ValueError("approved calibration status changed")
            provenance: AutomaticJudgeProvenance = HumanCalibratedAutomaticJudge(
                calibration_id=calibration.calibration_id,
                calibration_input=calibration_input,
                audit=audit,
                audit_input=state.audit,
            )
        else:
            if calibration.status != "provisional":
                raise ValueError("provisional calibration status changed")
            if (
                calibration.rubric_id != setup.rubric.artifact_id
                or calibration.judge_model != setup.judge_model
                or calibration.judge_prompt_id != setup.prompt_template.prompt.prompt_id
                or calibration.judge_prompt_sha256 != setup.prompt_template.prompt.sha256
            ):
                raise ValueError("provisional calibration differs from finalized judge setup")
            provenance = ProvisionalAutomaticJudge(
                calibration_id=calibration.calibration_id,
                calibration_input=calibration_input,
            )
        if completed is not None and (
            setup.trace_dataset != completed.trace_dataset or setup.task_set != completed.task_set
        ):
            raise ValueError("approved judge setup differs from the completed build")
        if judge_alias is not None and setup.judge_alias != judge_alias:
            raise ValueError("approved judge alias differs from the build-frozen judge")
        if judge_model is not None and setup.judge_model != judge_model:
            raise ValueError("approved judge model identity changed")
        return setup, setup_input, provenance
    except (OSError, ValueError) as exc:
        problems.append(f"manual judge: {exc}")
        return None, None, None


def _verify_manual_judge_chain(
    project: ProjectStore,
    state: ManualJudgeReviewState,
    setup: ManualJudgeSetupArtifact,
    audit: ManualJudgeCalibrationAudit,
    calibration: JudgeCalibration,
) -> None:
    """Cross-bind the selected audit to its exact setup and approved calibration lineage.

    Args:
        project: Project-local immutable artifact store.
        state: Mutable review pointers selected for automatic optimization.
        setup: Manifest-verified finalized judge setup.
        audit: Manifest-verified completed calibration audit.
        calibration: Recursively verified approved calibration.

    Raises:
        ValueError: Any setup, report, provisional calibration, prompt, or budget pin differs.
    """
    report_stored = project.artifacts.read(audit.report.artifact_id)
    report_input = artifact_input(report_stored.manifest)
    report = CalibrationReport.model_validate_json(
        project.artifacts.read_bytes(audit.report.artifact_id, "report.json")
    )
    provisional, provisional_input = verify_persisted_calibration(
        project, audit.provisional_calibration.artifact_id
    )
    prompt = setup.prompt_template.prompt
    expected_estimate = (
        (
            audit.budget.maximum_input_tokens_per_call * audit.budget.input_usd_per_million_tokens
            + audit.budget.maximum_output_tokens_per_call
            * audit.budget.output_usd_per_million_tokens
        )
        / 1_000_000
        * audit.budget.maximum_attempts_per_call
        * audit.budget.call_count
    )
    matching_audits = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "manual-judge-calibration-audit":
            continue
        candidate = read_audit(project, artifact_input(stored.manifest))
        if (
            candidate.setup == audit.setup
            and candidate.report == audit.report
            and candidate.provisional_calibration == audit.provisional_calibration
        ):
            matching_audits.append(candidate.audit_id)
    if (
        audit.setup != state.setup
        or matching_audits != [audit.audit_id]
        or report_input != audit.report
        or provisional_input != audit.provisional_calibration
        or provisional.status != "provisional"
        or calibration.out_of_fold_report_id != report.report_id
        or calibration.out_of_fold_report_sha256 != report_input.sha256
        or setup.rubric.artifact_id != calibration.rubric_id
        or report.rubric_id != calibration.rubric_id
        or provisional.rubric_id != calibration.rubric_id
        or setup.judge_model != calibration.judge_model
        or report.judge_model != calibration.judge_model
        or provisional.judge_model != calibration.judge_model
        or prompt.prompt_id != calibration.judge_prompt_id
        or prompt.sha256 != calibration.judge_prompt_sha256
        or report.judge_prompt_id != calibration.judge_prompt_id
        or report.judge_prompt_sha256 != calibration.judge_prompt_sha256
        or provisional.judge_prompt_id != calibration.judge_prompt_id
        or provisional.judge_prompt_sha256 != calibration.judge_prompt_sha256
        or audit.budget.call_count != sum(len(item.probes) for item in audit.judgments)
        or not math.isclose(
            audit.budget.estimated_cost_usd,
            expected_estimate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "selected judge calibration audit differs from its setup, approved lineage, or budget"
        )


def _build_evidence(
    problems: list[str], project: ProjectStore, completed: ProjectBuildArtifacts | None
) -> tuple[
    tuple[TaskCase, ...],
    tuple[Trace, ...],
    TraceModelIdentityEvidenceSet | None,
]:
    """Load exact completed tasks and traces for overlap planning.

    Args:
        problems: Mutable aggregate problem list.
        project: Project-local artifact store.
        completed: Completed build pointers, if available.

    Returns:
        Verified tasks, traces, and optional versioned identity evidence, or empty values after
        recording a failure.
    """
    if completed is None:
        return (), (), None
    try:
        from wmo.simulation.mining.bindings import load_task_set_lineage_bindings

        tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
        load_task_set_lineage_bindings(project.artifacts, completed.task_set.artifact_id)
        loaded_traces = load_trace_dataset(
            project.artifacts,
            completed.trace_dataset.artifact_id,
        )
        verify_current_trace_dataset(project.artifacts, loaded_traces)
        identity_evidence = read_trace_model_identity_evidence(
            project.artifacts,
            loaded_traces,
        )
        return tasks, loaded_traces.traces, identity_evidence
    except (OSError, ValueError) as exc:
        problems.append(f"completed build evidence: {exc}")
        return (), (), None


def _observed_traces(
    problems: list[str],
    tasks: tuple[TaskCase, ...],
    traces: tuple[Trace, ...],
    identity_evidence: TraceModelIdentityEvidenceSet | None,
    candidates: tuple[RoutedCandidateSnapshot, ...],
) -> tuple[ObservedRouterTrace, ...]:
    """Resolve real fit lineages through verified declared or unique inferred identity.

    Args:
        problems: Mutable aggregate preflight failures.
        tasks: Verified representative tasks.
        traces: Verified normalized production traces.
        identity_evidence: Verified model-span digest provenance, if the dataset carries it.
        candidates: Exact selected candidate identities.
    Returns:
        One deterministic exact-match trace per attributable fit lineage.
    """
    if not tasks or not traces or not candidates:
        return ()
    if identity_evidence is None:
        return ()
    try:
        attributions = resolve_router_observed_attributions(
            tasks,
            traces,
            identity_evidence,
            candidates,
        )
    except RouterAttributionError as exc:
        problems.append(f"observed fit attribution: {exc}")
        return ()
    tasks_by_id = {task.task_id: task for task in tasks}
    traces_by_id = {trace.trace_id: trace for trace in traces}
    return tuple(
        ObservedRouterTrace(
            task=tasks_by_id[item.task_id],
            trace=traces_by_id[item.trace_id],
            candidate_alias=item.candidate_alias,
            attribution=item,
        )
        for item in attributions
    )


def _preflight_error(problems: list[str]) -> AutomaticRouterPreflightError:
    """Render deterministic aggregate remediation without duplicating messages.

    Args:
        problems: Collected prerequisite failures.

    Returns:
        One actionable aggregate exception.
    """
    unique = tuple(dict.fromkeys(problems))
    return AutomaticRouterPreflightError(
        "router optimization prerequisites are incomplete:\n- " + "\n- ".join(unique)
    )

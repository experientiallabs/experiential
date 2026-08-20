"""Read-only aggregate preflight for automatic completed-build router optimization."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, Sha256, sha256_json
from exp.common.models import (
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
from exp.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from exp.common.routing import RouterEmbeddingReservation
from exp.common.tasks import TaskCase, load_task_set
from exp.common.traces import Trace, load_trace_dataset
from exp.optimize.router.automatic.attribution import (
    RouterAttributionError,
    RouterObservedAttribution,
    resolve_router_observed_attributions,
)
from exp.optimize.router.automatic.judge_provenance import (
    AutomaticJudgeProvenance,
    HostedAutomaticJudgeEvidence,
    HumanCalibratedAutomaticJudge,
    ProvisionalAutomaticJudge,
    hosted_judge_inputs,
    manual_judge_inputs,
)
from exp.optimize.router.automatic.reservations import (
    AutomaticRouterCostPlan,
    AutomaticRouterOptions,
    judge_completion_reservation,
    plan_automatic_router_cost,
    remaining_simulation_budget,
    retrieval_embedding_reservation,
    router_feature_reservation,
    simulation_completion_reservations,
    simulation_input_token_estimate,
)
from exp.optimize.router.judging.contracts import (
    JudgeSetupArtifact,
    ManualJudgeCalibrationAudit,
)
from exp.runtime.agents import agent_factory_sha256
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.ingest.dataset import (
    read_trace_model_identity_evidence,
    verify_current_trace_dataset,
)
from exp.simulation.ingest.model_identity import TraceModelIdentityEvidenceSet
from exp.simulation.specs import CandidateCompletionReservation
from exp.simulation.world_model import load_grounded_world_model_artifact


class AutomaticRouterPreflightError(ValueError):
    """One or more automatic router prerequisites are missing or inconsistent."""


@dataclass(frozen=True)
class ObservedRouterTrace:
    """One real fit task and trace attributed to an exact selected candidate."""

    task: TaskCase
    trace: Trace
    candidate_alias: str
    attribution: RouterObservedAttribution


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
    setup: JudgeSetupArtifact
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

    @property
    def judge_audit(self) -> ManualJudgeCalibrationAudit | None:
        """Return the completed human calibration audit, when one exists."""
        if isinstance(self.judge_provenance, HumanCalibratedAutomaticJudge):
            return self.judge_provenance.audit
        return None

    @property
    def judge_audit_input(self) -> ArtifactInput | None:
        """Return the exact human calibration audit input, when one exists."""
        if isinstance(self.judge_provenance, HumanCalibratedAutomaticJudge):
            return self.judge_provenance.audit_input
        return None

    @property
    def approved_calibration_id(self) -> str:
        """Return the selected calibration artifact identity."""
        return self.judge_provenance.calibration_id

    @property
    def approved_calibration_input(self) -> ArtifactInput:
        """Return the exact selected calibration manifest input."""
        return self.judge_provenance.calibration_input


def preflight_automatic_router(
    project: ProjectStore,
    selection: RouterCandidateSelection,
    *,
    catalog_override: ModelCatalog | None = None,
    hosted_judge: HostedAutomaticJudgeEvidence | None = None,
    options: AutomaticRouterOptions,
) -> AutomaticRouterPreflight:
    """Verify every local prerequisite and report all failures before credentials or writes.

    Args:
        project: Existing project whose completed build will be optimized.
        selection: Explicit candidates and incumbent collected for this optimize run.
        catalog_override: Confirmed prospective catalog before its atomic post-consent write.
        hosted_judge: Optional machine-only provisional evidence for the hosted workflow.
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
        problems.append("completed build: run `exp build PROJECT --traces PATH` first")
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
    if hosted_judge is None:
        setup, setup_input, judge_provenance = manual_judge_inputs(
            problems,
            project,
            completed,
            judge_alias,
            judge,
        )
    else:
        (
            setup,
            setup_input,
            hosted_calibration_id,
            hosted_calibration_input,
        ) = hosted_judge_inputs(
            problems,
            project,
            completed,
            judge_alias,
            judge,
            hosted_judge,
            catalog,
            options,
        )
        judge_provenance = (
            ProvisionalAutomaticJudge(
                calibration_id=hosted_calibration_id,
                calibration_input=hosted_calibration_input,
            )
            if hosted_calibration_id is not None and hosted_calibration_input is not None
            else None
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
            system_prompt=(config.system.system_prompt if config.system is not None else None),
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
    world_model_top_k = _world_model_retrieval_count(problems, project, completed)
    estimated_input_tokens = (
        None
        if world_model_top_k is None
        else simulation_input_token_estimate(
            traces,
            retrieved_transition_count=world_model_top_k,
            maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
            maximum_output_tokens=options.simulation_maximum_output_tokens,
        )
    )
    if world_model_top_k is not None and estimated_input_tokens is None:
        problems.append(
            "simulation completion reservations: the completed build has no persisted traces "
            "to size the per-call input reservation"
        )
    if estimated_input_tokens is None:
        candidate_requests: tuple[CandidateCompletionReservation, ...] = ()
        world_request = None
    else:
        candidate_requests, world_request = simulation_completion_reservations(
            problems,
            catalog=catalog,
            candidates=candidates,
            world_alias=world_alias,
            world=world,
            maximum_attempts=options.completion_maximum_attempts,
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=options.simulation_maximum_output_tokens,
        )
    judge_request = (
        judge_completion_reservation(
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
        if hosted_judge is None
        else hosted_judge.request_reservation
    )
    cost_plan = None
    if setup is not None and judge_provenance is not None and estimated_input_tokens is not None:
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
                estimated_input_tokens=estimated_input_tokens,
                options=options,
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


def _world_model_retrieval_count(
    problems: list[str],
    project: ProjectStore,
    completed: ProjectBuildArtifacts | None,
) -> int | None:
    """Read the frozen per-prediction retrieval count from the completed grounded world model.

    Args:
        problems: Mutable aggregate problem list.
        project: Project-local artifact store.
        completed: Exact completed-build pointers, if present.

    Returns:
        Persisted world-model top-k, or ``None`` after recording a problem.
    """
    if completed is None:
        return None
    try:
        artifact = load_grounded_world_model_artifact(project.artifacts, completed.world_model)
    except (OSError, ValueError) as exc:
        problems.append(f"grounded world model: {exc}")
        return None
    return artifact.top_k


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
        from exp.simulation.mining.bindings import load_task_set_lineage_bindings

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
        identity_evidence: Verified model-span digest provenance, when a completed build exists.
        candidates: Exact selected candidate identities.
    Returns:
        One deterministic exact-match trace per attributable fit lineage.
    """
    if not tasks or not traces or not candidates or identity_evidence is None:
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

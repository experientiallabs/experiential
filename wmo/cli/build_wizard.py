"""Interactive end-to-end build wizard over existing WMO product services.

Before spend authorization the wizard may persist normalized traces and deterministic task/split
evidence as resumable checkpoints. Provider setup may authenticate for read-only model discovery,
but planning never performs paid inference or generation, builds RAG or world-model artifacts, or
selects a completed project build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from wmo.cli.consent import require_spend_consent
from wmo.common.models import (
    ModelCatalog,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
    catalog_state_sha256,
    configure_router_candidates,
    load_model_catalog,
)
from wmo.common.project import (
    ProjectBudgetConfiguration,
    ProjectConfig,
    ProjectModelConfiguration,
    ProjectRetrievalConfiguration,
    ProjectStore,
)
from wmo.common.release_revision import installed_release_revision
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import load_trace_dataset
from wmo.optimize.router.automatic.attribution import resolve_router_observed_attributions
from wmo.optimize.router.automatic.preflight import preflight_automatic_router
from wmo.optimize.router.automatic.replay import (
    AutomaticRouterReplay,
    find_persisted_automatic_router_replay,
)
from wmo.optimize.router.automatic.reservations import (
    AutomaticRouterCostPlan,
    plan_automatic_router_cost,
)
from wmo.optimize.router.automatic.service import (
    AutomaticRouterOptions,
    optimize_project_router,
)
from wmo.optimize.router.composition import RouterCandidateSetupPlan
from wmo.optimize.router.judging.artifacts import read_audit, read_review_state
from wmo.optimize.router.judging.provisional import bootstrap_provisional_judge
from wmo.optimize.router.judging.service import (
    _read_setup,
    commit_manual_judge_setup,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.simulation.build import ProjectBuild, build_project
from wmo.simulation.ingest.dataset import read_trace_model_identity_evidence
from wmo.simulation.ingest.sources import CANONICAL_TRACE_SOURCES


@dataclass(frozen=True)
class WizardBuildPlan:
    """Provider-free grounded-build plan shown before the one spend authorization."""

    trace_path: Path | None
    source: str
    catalog: ModelCatalog
    selected: ProjectModelConfiguration
    tasks: tuple[TaskCase, ...]
    completed: ProjectBuild | None
    accepted_traces: int
    invalid_traces: int
    fit_tasks: int
    held_out_tasks: int
    build_estimate_usd: float
    build_reused: bool


def run_build_wizard(
    project: str,
    *,
    source: str,
    root: Path,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
    top_k: int,
    maximum_build_cost_usd: float,
    maximum_router_cost_usd: float | None,
    providers: tuple[str, ...],
    console: Console,
) -> None:
    """Run one resumable interactive project build through router completion.

    Args:
        project: Safe local project identifier.
        source: Initial trace-source choice.
        root: Local WMO artifact root.
        world_model: Optional world-model alias override.
        judge: Optional judge alias override.
        embedder: Optional embedder alias override.
        top_k: Serving retrieval result limit.
        maximum_build_cost_usd: Strict grounded-build provider ceiling.
        maximum_router_cost_usd: Optional strict router ceiling, or automatic planning.
        providers: Repeatable provider names that skip the opening provider list.
        console: Interactive terminal for prompts and progress.

    Raises:
        ValueError: Existing state or a selected wizard input is invalid.
    """
    code_revision = installed_release_revision()
    console.print("[bold]WMO project wizard[/bold]")
    _require_replay_role_overrides(
        root,
        project,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    replay = _completed_replay(root, project, code_revision=code_revision)
    if replay is not None:
        _render_completed_replay(project, replay, console=console)
        return
    existing = _completed_build_plan(
        root,
        project,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    if existing is None:
        chosen_source, trace_path = _select_trace(source, console=console)
        plan = _prepare_new_build(
            project,
            trace_path=trace_path,
            source=chosen_source,
            root=root,
            world_model=world_model,
            judge=judge,
            embedder=embedder,
            top_k=top_k,
            maximum_build_cost_usd=maximum_build_cost_usd,
            code_revision=code_revision,
            providers=providers,
            console=console,
        )
    else:
        plan = existing
        console.print("[dim]resume[/dim] Verified completed grounded build")
    catalog = _ensure_router_defaults(root, plan.catalog, console=console)
    candidate_plan = _candidate_plan(root, catalog)
    cost_plan = _wizard_cost_plan(
        root,
        project,
        plan,
        catalog,
        candidate_plan.selection,
    )
    router_ceiling = cost_plan.required_provider_cost_usd
    options = replace(
        AutomaticRouterOptions(),
        maximum_provider_cost_usd=router_ceiling,
        maximum_judgments=cost_plan.maximum_judgments,
    )
    if (
        maximum_router_cost_usd is not None
        and maximum_router_cost_usd < cost_plan.required_provider_cost_usd
    ):
        _render_plan(
            project,
            plan,
            catalog=catalog,
            cost_plan=cost_plan,
            maximum_build_cost_usd=maximum_build_cost_usd,
            router_ceiling=router_ceiling,
            supplied_router_cap=maximum_router_cost_usd,
            console=console,
        )
        raise ValueError(
            f"router cap ${maximum_router_cost_usd:.6f} is below the exact required "
            f"${cost_plan.required_provider_cost_usd:.6f}; increase "
            "--max-router-cost-usd or omit it"
        )
    if not plan.build_reused and plan.build_estimate_usd > maximum_build_cost_usd:
        raise ValueError(
            f"grounded build requires ${plan.build_estimate_usd:.6f}, above the configured "
            f"${maximum_build_cost_usd:.6f} ceiling; increase --max-build-cost-usd"
        )
    _render_plan(
        project,
        plan,
        catalog=catalog,
        cost_plan=cost_plan,
        maximum_build_cost_usd=maximum_build_cost_usd,
        router_ceiling=router_ceiling,
        supplied_router_cap=maximum_router_cost_usd,
        console=console,
    )
    build_ceiling = 0.0 if plan.build_reused else maximum_build_cost_usd
    total_ceiling = math.fsum((build_ceiling, router_ceiling))
    if total_ceiling > 0 and not require_spend_consent(
        console,
        root=root,
        yes=False,
        estimated_cost_usd=total_ceiling,
        command=f"wmo build {project}",
        assumptions=(
            "serving and fit-only RAG embeddings",
            f"{cost_plan.simulated_episode_count} closed-loop candidate episodes",
            f"{cost_plan.maximum_judgments} configured judge decisions",
        ),
    ):
        console.print("Wizard stopped before paid build or router work.")
        return

    if not plan.build_reused:
        console.print("[bold]1/4 Simulation and RAG indexes[/bold]")
        from wmo.cli.build_cmd import _complete_grounded_build, _validated_role_snapshots

        assert plan.completed is not None
        runtime_catalog = RuntimeModelCatalog(catalog)
        world_snapshot, embedder_snapshot, _embedder_capabilities = _validated_role_snapshots(
            runtime_catalog,
            plan.selected,
        )
        _complete_grounded_build(
            ProjectStore(root, project),
            plan.completed,
            selected=plan.selected,
            runtime_catalog=runtime_catalog,
            world_snapshot=world_snapshot,
            embedder_snapshot=embedder_snapshot,
            top_k=top_k,
            estimate=plan.build_estimate_usd,
            maximum_build_cost_usd=maximum_build_cost_usd,
            provider_spend_authorized=True,
        )
    else:
        console.print("[bold]1/4 Simulation and RAG indexes[/bold] [green]reused[/green]")

    store = ProjectStore(root, project)
    console.print("[bold]2/4 Judge syllabus[/bold]")
    _ensure_judge_calibration(
        store,
        catalog,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
    )
    console.print(
        "  Human calibration is optional. This build keeps provisional judgment provenance "
        "until examples are approved."
    )

    console.print("[bold]3/4 Router plan[/bold]")
    catalog = load_model_catalog(root / "models.toml")
    candidate_plan = _candidate_plan(root, catalog)
    preflight = preflight_automatic_router(
        store,
        candidate_plan.selection,
        catalog_override=candidate_plan.prospective_catalog,
        options=options,
    )
    if preflight.cost_plan != cost_plan:
        raise ValueError(
            "automatic router cost plan changed after the grounded build; rerun the wizard"
        )
    console.print(
        f"  {len(preflight.candidates)} candidates, {len(preflight.observed_traces)} "
        f"reusable historical cells, ${options.maximum_provider_cost_usd:.4f} ceiling"
    )
    console.print("[bold]4/4 Router optimization[/bold]")
    result = optimize_project_router(
        store,
        candidate_plan,
        RuntimeModelCatalog(catalog),
        options=options,
        provider_spend_consented=True,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
    )
    policy = result.composition.optimization.optimization.policy
    report = result.composition.optimization.optimization.report
    console.print("[green]Complete[/green]")
    selected_build = store.load_project().build
    assert selected_build is not None
    console.print(f"  serving RAG     {selected_build.serving_rag.artifact_id}")
    console.print(f"  fit RAG         {selected_build.fit_rag.artifact_id}")
    console.print(f"  simulation      {selected_build.world_model.artifact_id}")
    console.print(f"  syllabus        {preflight.setup.rubric.artifact_id}")
    console.print(f"  calibration     {preflight.calibration_id} ({preflight.judgment_status})")
    console.print(f"  router          {policy.policy_id} ({policy.judgment_status})")
    console.print(f"  report          {report.report_id}")
    console.print(f"  next            wmo run {project}")
    if preflight.judgment_status == "provisional":
        console.print(f"  optional        wmo config judge calibrate {project}")
        console.print(f"  after approval  wmo build {project}")


def _ensure_judge_calibration(
    store: ProjectStore,
    catalog: ModelCatalog,
    *,
    created_at: datetime,
    code_revision: str,
) -> None:
    """Create the judge syllabus and bootstrap only absent calibration state.

    Args:
        store: Project-local artifact and review store.
        catalog: Static catalog containing the selected judge identity.
        created_at: Time for any newly materialized setup or calibration evidence.
        code_revision: Exact producer revision for any new immutable evidence.

    Raises:
        ValueError: A completed audit awaits explicit human approval without a selected
            calibration pointer.
    """
    state = read_review_state(store)
    if state is not None:
        if state.approved_calibration is not None or state.provisional_calibration is not None:
            return
        if state.audit is not None:
            raise ValueError(
                "judge calibration audit awaits explicit approval; rerun "
                f"`wmo config judge calibrate {store.paths.project_id}` and approve it"
            )
    else:
        setup_plan = prepare_manual_judge_setup(
            store,
            catalog,
            created_at=created_at,
            code_revision=code_revision,
        )
        commit_manual_judge_setup(store, setup_plan, confirmed=True)
    calibration_plan = prepare_manual_judge_calibration(store)
    bootstrap_provisional_judge(
        store,
        catalog,
        calibration_plan,
        created_at=created_at,
        code_revision=code_revision,
    )


def _completed_replay(
    root: Path,
    project: str,
    *,
    code_revision: str,
) -> AutomaticRouterReplay | None:
    """Return a completed automatic replay without prompting or resolving credentials.

    Args:
        root: Local WMO root.
        project: Existing project identifier.
        code_revision: Current installed release revision.

    Returns:
        Completed replay, or ``None`` when any prerequisite is absent or changed.

    Raises:
        ValueError: Existing immutable router evidence is corrupt or ambiguous.
    """
    store = ProjectStore(root, project)
    if not store.paths.project_toml.exists():
        return None
    state = read_review_state(store)
    if state is None:
        return None
    if state.approved_calibration is not None:
        judgment_status = "human_calibrated"
    elif state.provisional_calibration is not None:
        judgment_status = "provisional"
    else:
        return None
    return find_persisted_automatic_router_replay(
        store,
        judgment_status=judgment_status,
        code_revision=code_revision,
    )


def _require_replay_role_overrides(
    root: Path,
    project: str,
    *,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
) -> None:
    """Reject role overrides that differ from a selected completed build.

    Args:
        root: Local WMO root.
        project: Existing project identifier.
        world_model: Optional requested world-model alias.
        judge: Optional requested judge alias.
        embedder: Optional requested embedder alias.

    Raises:
        ValueError: A supplied override differs from the selected completed-build role.
    """
    store = ProjectStore(root, project)
    if not store.paths.project_toml.exists():
        return
    config = store.load_project()
    if config.build is None or config.models is None:
        return
    requested = {
        "world_model": world_model,
        "judge": judge,
        "embedder": embedder,
    }
    mismatches = tuple(
        f"{role}={alias!r} (selected {getattr(config.models, role)!r})"
        for role, alias in requested.items()
        if alias is not None and alias != getattr(config.models, role)
    )
    if mismatches:
        raise ValueError(
            "role overrides differ from the selected completed build: " + ", ".join(mismatches)
        )


def _render_completed_replay(
    project: str,
    replay: AutomaticRouterReplay,
    *,
    console: Console,
) -> None:
    """Render one verified completed chain without opening planning or consent.

    Args:
        project: Local project identifier.
        replay: Exact completed router and report identities.
        console: Terminal receiving the replay summary.
    """
    console.print("[green]Complete[/green] Reused every verified project artifact.")
    console.print(f"  router  {replay.policy_id}")
    console.print(f"  report  {replay.report_id}")
    console.print(f"  next    wmo run {project}")
    if replay.judgment_status == "provisional":
        console.print(f"  optional wmo config judge calibrate {project}")
        console.print(f"  after approval wmo build {project}")


def _select_trace(initial_source: str, *, console: Console) -> tuple[str, Path]:
    """Select and validate one supported trace source and local declaration path.

    Args:
        initial_source: CLI-provided initial source choice.
        console: Interactive terminal.

    Returns:
        Canonical source name and validated local path.
    """
    source = initial_source.strip().casefold()
    if source not in CANONICAL_TRACE_SOURCES:
        source = Prompt.ask(
            "Trace source",
            choices=list(CANONICAL_TRACE_SOURCES),
            default="otlp",
            console=console,
        )
    candidates = _trace_candidates(Path.cwd(), source)
    if len(candidates) == 1:
        path = candidates[0]
        console.print(f"[dim]traces[/dim] Using {path}")
        return source, path
    if len(candidates) > 1:
        console.print("Discovered trace files:")
        for index, candidate in enumerate(candidates, start=1):
            console.print(f"  {index}. {candidate}")
        choice = Prompt.ask(
            "Trace file",
            choices=[str(index) for index in range(1, len(candidates) + 1)],
            default="1",
            console=console,
        )
        selected_path = str(candidates[int(choice or "1") - 1])
    else:
        selected_path = Prompt.ask("Trace path", console=console)
    if selected_path is None:
        raise ValueError("trace path is required")
    path = Path(selected_path).expanduser()
    from wmo.cli.build_cmd import _resolve_trace_file

    return source, _resolve_trace_file(path)


def _trace_candidates(directory: Path, source: str) -> tuple[Path, ...]:
    """Return deterministic local trace candidates for one source.

    Args:
        directory: Current working directory to inspect without recursion.
        source: Selected canonical trace source.

    Returns:
        Existing candidate files in deterministic preference order.
    """
    preferred = (
        (directory / "traces.otel.jsonl", directory / "traces.jsonl")
        if source == "otlp"
        else (directory / f"traces.{source}.jsonl", directory / "traces.jsonl")
    )
    discovered = tuple(sorted(directory.glob("*.jsonl")))
    return tuple(dict.fromkeys(path for path in (*preferred, *discovered) if path.is_file()))


def _prepare_new_build(
    project: str,
    *,
    trace_path: Path,
    source: str,
    root: Path,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
    top_k: int,
    maximum_build_cost_usd: float,
    code_revision: str,
    providers: tuple[str, ...],
    console: Console,
) -> WizardBuildPlan:
    """Materialize deterministic build evidence and return a credential-free plan.

    Args:
        project: Local project identifier.
        trace_path: Validated trace export or source declaration.
        source: Canonical trace source.
        root: Local WMO root.
        world_model: Optional world-model override.
        judge: Optional judge override.
        embedder: Optional embedder override.
        top_k: Serving retrieval result limit.
        maximum_build_cost_usd: Strict embedding ceiling.
        code_revision: Installed producer revision.
        providers: Repeatable provider names that skip the opening provider list.
        console: Interactive terminal.

    Returns:
        Complete provider-free plan with deterministic persisted evidence.
    """
    from wmo.cli.build_cmd import (
        _embedding_cost_ceiling,
        _load_canonical_traces,
        _missing_build_configuration,
        _project_store,
        _reuse_completed_grounded_artifacts,
        _selected_roles,
        _validated_role_snapshots,
    )
    from wmo.cli.provider_setup import ProviderSetupOptions, run_provider_setup

    console.print("[dim]inspect[/dim] Normalize trace evidence")
    normalized = _load_canonical_traces(trace_path, source)
    if not normalized.traces:
        raise ValueError("selected trace source produced no valid canonical traces")
    catalog_path = root / "models.toml"
    existing_catalog = load_model_catalog(catalog_path) if catalog_path.exists() else None
    if _missing_build_configuration(existing_catalog):
        catalog = run_provider_setup(
            root,
            ProviderSetupOptions(providers=providers),
            non_interactive=False,
            replace=False,
            console=console,
            offer_recommended_defaults=True,
        )
    else:
        assert existing_catalog is not None
        catalog = existing_catalog
    selected = _selected_roles(
        catalog,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    runtime = RuntimeModelCatalog(catalog)
    world_snapshot, embedder_snapshot, embedder_capabilities = _validated_role_snapshots(
        runtime, selected
    )
    store = _project_store(
        root,
        ProjectConfig(
            project_id=project,
            trace_source=source,
            models=selected,
            retrieval=ProjectRetrievalConfiguration(top_k=top_k),
            budgets=ProjectBudgetConfiguration(maximum_build_cost_usd=maximum_build_cost_usd),
        ),
    )
    completed = build_project(
        normalized,
        store,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
    )
    built = _reuse_completed_grounded_artifacts(
        store,
        completed,
        world_alias=selected.world_model,
        world_snapshot=world_snapshot,
        embedder_snapshot=embedder_snapshot,
        top_k=top_k,
    )
    tasks = completed.artifacts.mining.tasks
    return WizardBuildPlan(
        trace_path=trace_path,
        source=source,
        catalog=catalog,
        selected=selected,
        tasks=tasks,
        completed=completed,
        accepted_traces=len(completed.artifacts.trace_dataset.dataset.trace_ids),
        invalid_traces=completed.artifacts.trace_dataset.dataset.invalid_trace_count,
        fit_tasks=sum(task.partition == "fit" for task in tasks),
        held_out_tasks=sum(task.partition == "held_out" for task in tasks),
        build_estimate_usd=_embedding_cost_ceiling(completed, embedder_capabilities),
        build_reused=built is not None,
    )


def _completed_build_plan(
    root: Path,
    project: str,
    *,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
) -> WizardBuildPlan | None:
    """Read a completed grounded build as the next resumable wizard checkpoint.

    Args:
        root: Local WMO root.
        project: Existing project identifier.
        world_model: Optional world-model override.
        judge: Optional judge override.
        embedder: Optional embedder override.

    Returns:
        Verified completed-build plan, or ``None`` before grounded selection.
    """
    store = ProjectStore(root, project)
    if not store.paths.project_toml.exists() or not store.model_catalog_path.exists():
        return None
    config = store.load_project()
    if config.build is None or config.models is None or config.trace_source is None:
        return None
    selected = config.models
    overrides = (world_model, judge, embedder)
    if any(overrides) and overrides != (
        selected.world_model,
        selected.judge,
        selected.embedder,
    ):
        raise ValueError("role overrides differ from the completed grounded build")
    catalog = load_model_catalog(store.model_catalog_path)
    traces = load_trace_dataset(store.artifacts, config.build.trace_dataset.artifact_id)
    tasks = load_task_set(store.artifacts, config.build.task_set.artifact_id).tasks
    return WizardBuildPlan(
        trace_path=None,
        source=config.trace_source,
        catalog=catalog,
        selected=selected,
        tasks=tasks,
        completed=None,
        accepted_traces=len(traces.traces),
        invalid_traces=traces.dataset.invalid_trace_count,
        fit_tasks=sum(task.partition == "fit" for task in tasks),
        held_out_tasks=sum(task.partition == "held_out" for task in tasks),
        build_estimate_usd=0.0,
        build_reused=True,
    )


def _ensure_router_defaults(root: Path, catalog: ModelCatalog, *, console: Console) -> ModelCatalog:
    """Persist deterministic safe router defaults when provider setup left them empty.

    Args:
        root: Local WMO root containing the shared catalog.
        catalog: Current provider and model catalog.
        console: Terminal receiving the chosen defaults.

    Returns:
        Catalog with at least two router candidates and one incumbent.

    Raises:
        ValueError: Fewer than two fully priced completion models are available.
    """
    if len(catalog.roles.candidates) >= 2 and catalog.roles.incumbent is not None:
        return catalog
    from wmo.cli.provider_setup import _recommended_router_selection

    selection = _recommended_router_selection(catalog)
    console.print(
        "[dim]defaults[/dim] Router candidates "
        + ", ".join(selection.candidates)
        + f"; incumbent {selection.incumbent}"
    )
    return configure_router_candidates(
        root / "models.toml",
        selection,
        expected_state_sha256=catalog_state_sha256(root / "models.toml"),
    )


def _wizard_cost_plan(
    root: Path,
    project: str,
    plan: WizardBuildPlan,
    catalog: ModelCatalog,
    selection: RouterCandidateSelection,
) -> AutomaticRouterCostPlan:
    """Compute the exact automatic schedule before paid provider work.

    Args:
        root: Local WMO root containing any resumable judge state.
        project: Local project identifier.
        plan: Deterministic task and role plan.
        catalog: Static model catalog after recommended router defaults.
        selection: Exact router candidates and incumbent.

    Returns:
        Pure complete reservation for the current task-candidate grid.

    Raises:
        ValueError: Existing judge state is incomplete or awaits human approval.
    """
    store = ProjectStore(root, project)
    state = read_review_state(store)
    response_shape = "scalar"
    audit = None
    provisional = True
    if state is not None:
        setup = _read_setup(store, state.setup)
        response_shape = setup.prompt_template.response_shape
        if state.approved_calibration is not None:
            if state.audit is None:
                raise ValueError("approved judge calibration is missing its completed audit")
            audit = read_audit(store, state.audit)
            provisional = False
        elif state.audit is not None:
            raise ValueError(
                "judge calibration audit awaits explicit approval; rerun "
                f"`wmo config judge calibrate {project}` and approve it"
            )
    return plan_automatic_router_cost(
        plan.tasks,
        catalog,
        selection,
        world_model_alias=plan.selected.world_model,
        judge_alias=plan.selected.judge,
        embedder_alias=plan.selected.embedder,
        judge_response_shape=response_shape,
        judge_audit=audit,
        provisional_judge=provisional,
        observed_candidate_aliases=_wizard_observed_candidate_aliases(
            store,
            plan,
            catalog,
            selection,
        ),
        options=AutomaticRouterOptions(),
    )


def _wizard_observed_candidate_aliases(
    store: ProjectStore,
    plan: WizardBuildPlan,
    catalog: ModelCatalog,
    selection: RouterCandidateSelection,
) -> tuple[str, ...]:
    """Return exact pre-consent reusable candidate cells without provider access.

    Args:
        store: Project-local artifact store.
        plan: Verified task plan and optional fresh deterministic build.
        catalog: Static secret-free model catalog.
        selection: Exact router candidate selection.

    Returns:
        Candidate aliases for admitted exact historical cells, possibly empty.
    """
    if plan.completed is not None:
        trace_dataset_id = plan.completed.artifacts.trace_dataset.dataset.dataset_id
    else:
        selected_build = store.load_project().build
        if selected_build is None:
            raise ValueError("completed build is unavailable for fidelity planning")
        trace_dataset_id = selected_build.trace_dataset.artifact_id
    loaded = load_trace_dataset(store.artifacts, trace_dataset_id)
    evidence = read_trace_model_identity_evidence(store.artifacts, loaded)
    if evidence is None:
        return ()
    resolver = RuntimeModelCatalog(catalog, environment={})
    candidates = tuple(
        RoutedCandidateSnapshot(alias=alias, model=resolver.snapshot(alias)[0])
        for alias in selection.candidates
    )
    records = resolve_router_observed_attributions(
        plan.tasks,
        loaded.traces,
        evidence,
        candidates,
    )
    return tuple(record.candidate_alias for record in records)


def _candidate_plan(root: Path, catalog: ModelCatalog) -> RouterCandidateSetupPlan:
    """Build the exact no-prompt candidate plan from persisted wizard choices.

    Args:
        root: Local WMO root containing the shared catalog.
        catalog: Catalog with complete router roles.

    Returns:
        Exact candidate selection and catalog-state lease for automatic optimization.
    """
    incumbent = catalog.roles.incumbent
    if len(catalog.roles.candidates) < 2 or incumbent is None:
        raise ValueError("router candidates and incumbent are not configured")
    return RouterCandidateSetupPlan(
        selection=RouterCandidateSelection(
            candidates=catalog.roles.candidates,
            incumbent=incumbent,
        ),
        candidate_models=(),
        prospective_catalog=catalog,
        expected_catalog_sha256=catalog_state_sha256(root / "models.toml"),
    )


def _render_plan(
    project: str,
    plan: WizardBuildPlan,
    *,
    catalog: ModelCatalog,
    cost_plan: AutomaticRouterCostPlan,
    maximum_build_cost_usd: float,
    router_ceiling: float,
    supplied_router_cap: float | None,
    console: Console,
) -> None:
    """Render one concise full wizard plan before the named spend authorization.

    Args:
        project: Local project identifier.
        plan: Provider-free grounded-build plan.
        catalog: Catalog containing selected router defaults.
        cost_plan: Exact complete automatic-router provider reservation.
        maximum_build_cost_usd: Strict grounded-build ceiling.
        router_ceiling: Effective automatic-router ceiling.
        supplied_router_cap: Optional user cap checked against the exact required ceiling.
        console: Terminal receiving the plan.
    """
    build_ceiling = 0.0 if plan.build_reused else maximum_build_cost_usd
    console.print("[bold]Project plan[/bold]")
    console.print(f"  project      {project}")
    console.print(
        f"  traces       {plan.accepted_traces} accepted, {plan.invalid_traces} invalid; "
        f"{plan.fit_tasks} fit, {plan.held_out_tasks} held out"
    )
    console.print(
        f"  models       world={plan.selected.world_model}, judge={plan.selected.judge}, "
        f"embedder={plan.selected.embedder}"
    )
    console.print(
        f"  router       {', '.join(catalog.roles.candidates)}; incumbent={catalog.roles.incumbent}"
    )
    console.print("  RAG          serving and fit indexes")
    console.print("  syllabus     task-success rubric and evaluation instructions")
    console.print("  calibration  provisional; optional human approval creates a successor")
    console.print(
        f"  build spend  estimate ${plan.build_estimate_usd:.6f}; ceiling ${build_ceiling:.4f}"
    )
    console.print(
        f"  router spend embeddings ${cost_plan.router_embedding_cost_usd:.6f}; "
        f"judgments ${cost_plan.judgment_cost_usd:.6f}; "
        f"simulation ${cost_plan.simulation_cost_usd:.6f}"
    )
    console.print(
        f"  router scope {cost_plan.maximum_judgments} judgments, "
        f"{cost_plan.simulated_episode_count} simulated episodes"
    )
    console.print(f"  router spend required ${cost_plan.required_provider_cost_usd:.6f}")
    console.print(f"  router spend ceiling ${router_ceiling:.6f}")
    if supplied_router_cap is not None:
        console.print(f"  supplied cap ${supplied_router_cap:.6f}")
    console.print(f"  total ceiling ${math.fsum((build_ceiling, router_ceiling)):.4f}")

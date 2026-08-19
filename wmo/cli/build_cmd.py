"""Named project build from local traces to grounded world-model artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.build_cost import over_ceiling_message
from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.cli.progress import progress_display, qualified
from wmo.cli.provider_picker import resolve_setup_providers
from wmo.cli.provider_setup import (
    ProviderSetupOptions,
    provider_setup_json_examples,
    run_provider_setup,
)
from wmo.cli.theme import WMO_THEME
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ModelCatalogError,
    ModelSnapshot,
    load_model_catalog,
)
from wmo.common.observability.telemetry import BuildTelemetryStats, capture_build_completed
from wmo.common.progress import ProgressHook, report
from wmo.common.project import (
    ArtifactStoreError,
    ProjectBudgetConfiguration,
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectModelConfiguration,
    ProjectRetrievalConfiguration,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from wmo.common.release_revision import installed_release_revision
from wmo.runtime.models import (
    CapabilityRequirement,
    ModelCapabilityError,
    ModelConnectionError,
    ResolvedModel,
    RuntimeModelCatalog,
)
from wmo.runtime.models.preflight import preflight_capabilities
from wmo.runtime.models.providers.transport import ProviderTransportError, RetryPolicy
from wmo.simulation.build import ProjectBuild, TaskSetBuild, build_project, select_completed_build
from wmo.simulation.engines.text.errors import SimulationContentionError
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.ingest.sources import CANONICAL_TRACE_SOURCES, load_trace_source
from wmo.simulation.retrieval import (
    RAGEmbedderBinding,
    RAGLineageBinding,
    load_rag_index,
    persist_trace_rag,
)
from wmo.simulation.retrieval.transitions import extract_real_transitions
from wmo.simulation.world_model.artifact import (
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
    persist_grounded_world_model,
)

_console = Console(theme=WMO_THEME)
_PROJECT_ARGUMENT = typer.Argument(..., metavar="PROJECT", help="Local project name.")
_LEGACY_TRACE_ARGUMENT = typer.Argument(None, metavar="TRACES", hidden=True)
_TRACE_FILE_OPTION = typer.Option(
    None,
    "--traces",
    "-t",
    metavar="PATH",
    help=(
        "Local trace export in the declared --source format. Omit it only when bare "
        "`wmo build PROJECT` should launch the interactive end-to-end build."
    ),
)
_PROVIDER_OPTION = typer.Option(
    None,
    "--provider",
    help=(
        "Repeatable provider to configure during first-build setup. "
        "Supported values: openai, anthropic, gemini, openrouter, openai-compatible, "
        "azure, bedrock."
    ),
)


@dataclass(frozen=True)
class GroundedBuildCompletion:
    """One selected grounded build and whether provider work was replayed."""

    artifacts: ProjectBuildArtifacts
    reused: bool


def build(
    project: str = _PROJECT_ARGUMENT,
    legacy_trace_file: Path | None = _LEGACY_TRACE_ARGUMENT,
    trace_file: Path | None = _TRACE_FILE_OPTION,
    source: str = typer.Option(
        "otlp",
        "--source",
        help=f"Trace source format: {', '.join(CANONICAL_TRACE_SOURCES)}.",
    ),
    root: Path = ROOT_OPTION,
    world_model: str | None = typer.Option(None, "--world-model", help="World-model alias."),
    judge: str | None = typer.Option(None, "--judge", help="Judge alias."),
    embedder: str | None = typer.Option(None, "--embedder", help="Embedding-capable alias."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Serving retrieval result limit."),
    maximum_build_cost_usd: float = typer.Option(
        5.0,
        "--max-build-cost-usd",
        min=0.01,
        help="Strict embedding spend ceiling in USD.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    maximum_router_cost_usd: float | None = typer.Option(
        None,
        "--max-router-cost-usd",
        min=0.01,
        help=(
            "Optional automatic-router ceiling for the interactive build; omitted uses "
            "the exact conservative schedule reservation."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the complete preflight without credentials, provider calls, or selection.",
    ),
    no_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "--no-interactive",
        help="Require complete model flags and never ask setup or cost questions.",
    ),
    provider: list[str] | None = _PROVIDER_OPTION,
) -> None:
    """Build a reusable grounded world model and immutable fit evidence.

    The explicit command and configured build-cost ceiling authorize provider embedding calls.
    Model setup runs first when required catalog state is absent and both terminal streams are
    interactive. The shared catalog commits before project creation. Noninteractive missing state
    fails before any project or artifact write. Configured builds compute and display a complete
    provider-free preflight before reading credentials or constructing provider clients.

    Args:
        project: Safe local project identifier below ``<root>/projects``.
        legacy_trace_file: Active positional trace-path compatibility for packaged examples.
        trace_file: Explicit local canonical trace export, or ``None`` for the interactive wizard.
        source: Declared local-export format.
        root: Local ``.wmo`` artifact root.
        world_model: Optional configured alias override for this project.
        judge: Optional configured alias override for this project.
        embedder: Optional configured alias override for this project.
        top_k: Positive serving retrieval result limit.
        maximum_build_cost_usd: Strict ceiling for provider embedding calls.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        maximum_router_cost_usd: Optional strict wizard router ceiling, or automatic planning.
        dry_run: Print the complete preflight and stop before credentials or selection.
        no_interactive: Disable inline setup and cost questions even at a terminal.
        provider: Repeatable provider names that skip the opening list during setup.

    Raises:
        typer.BadParameter: Input, setup, role, cost, project, or artifact validation fails.
    """
    if legacy_trace_file is not None:
        if trace_file is not None:
            raise typer.BadParameter("provide traces once, using -t/--traces or the trace path")
        trace_file = legacy_trace_file
    if trace_file is None:
        if dry_run or no_interactive or not can_prompt(_console):
            raise typer.BadParameter(
                "automation and dry runs require an explicit -t/--traces PATH; bare "
                "`wmo build PROJECT` is the interactive end-to-end build"
            )
        from wmo.cli.build_wizard import run_build_wizard

        try:
            run_build_wizard(
                project,
                source=source,
                root=root,
                world_model=world_model,
                judge=judge,
                embedder=embedder,
                top_k=top_k,
                maximum_build_cost_usd=maximum_build_cost_usd,
                maximum_router_cost_usd=maximum_router_cost_usd,
                providers=tuple(provider or ()),
                console=_console,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except ProviderTransportError as exc:
            _console.print(f"[red]error[/red] a provider request failed: {exc}")
            _console.print(
                "Completed paid work is saved. Run wmo build again to resume; finished "
                "steps replay exactly without new spend."
            )
            raise typer.Exit(code=1) from exc
        except SimulationContentionError as exc:
            _console.print(f"[red]error[/red] {exc}")
            _console.print(
                "Another process owns in-flight paid work. Completed paid work is saved; "
                "run wmo build again to resume once that process finishes."
            )
            raise typer.Exit(code=1) from exc
        return
    started = time.monotonic()
    with usage_error(
        ArtifactStoreError,
        ModelCapabilityError,
        ModelCatalogError,
        ModelConnectionError,
        ProjectStoreError,
        ValueError,
    ):
        code_revision = installed_release_revision()
        ProjectStore(root, project)
        catalog = _load_or_setup_catalog(
            root,
            no_interactive=no_interactive,
            providers=tuple(provider or ()),
        )
        selected = _selected_roles(
            catalog,
            world_model=world_model,
            judge=judge,
            embedder=embedder,
        )
        runtime_catalog = RuntimeModelCatalog(catalog)
        world_snapshot, embedder_snapshot, embedder_capabilities = _validated_role_snapshots(
            runtime_catalog,
            selected,
        )
        _console.print("[dim]loading[/dim] Normalize trace evidence")
        path = _resolve_trace_file(trace_file)
        with progress_display(_console) as progress:
            report(progress, "normalization")
            normalized = _load_canonical_traces(path, source)
            if not normalized.traces:
                raise ValueError(
                    "no valid canonical traces were produced; inspect the input and provide at "
                    f"least one valid {source.strip().casefold()} trace"
                )
            record_count = len(normalized.traces) + len(normalized.issues)
            report(
                progress,
                "normalization",
                completed=len(normalized.traces),
                total=record_count,
                detail="valid traces",
            )
            store = _project_store(
                root,
                ProjectConfig(
                    project_id=project,
                    trace_source=source.strip().casefold(),
                    models=selected,
                    retrieval=ProjectRetrievalConfiguration(top_k=top_k),
                    budgets=ProjectBudgetConfiguration(
                        maximum_build_cost_usd=maximum_build_cost_usd
                    ),
                ),
            )
            report(progress, "task construction")
            completed = build_project(
                normalized,
                store,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
            )
            task_count = len(completed.artifacts.mining.tasks)
            report(
                progress,
                "task construction",
                completed=task_count,
                total=task_count,
                detail="representative tasks",
            )
        tasks = completed.artifacts.mining.tasks
        fit_count = sum(task.partition == "fit" for task in tasks)
        held_out_count = sum(task.partition == "held_out" for task in tasks)
        estimate = _embedding_cost_ceiling(completed, embedder_capabilities)
        built = _reuse_completed_grounded_artifacts(
            store,
            completed,
            world_alias=selected.world_model,
            world_snapshot=world_snapshot,
            embedder_snapshot=embedder_snapshot,
            top_k=top_k,
        )
        reused = built is not None
        _render_preflight(
            accepted=len(completed.artifacts.trace_dataset.dataset.trace_ids),
            invalid=completed.artifacts.trace_dataset.dataset.invalid_trace_count,
            fit_count=fit_count,
            held_out_count=held_out_count,
            world_alias=selected.world_model,
            world_model_id=world_snapshot.model_id,
            embedder_alias=selected.embedder,
            embedder_model_id=embedder_snapshot.model_id,
            estimate=estimate,
            ceiling=maximum_build_cost_usd,
            reused=reused,
        )
        if dry_run:
            _console.print("[green]dry run complete[/green] No provider calls or build selection.")
            return
        if built is None:
            if estimate > maximum_build_cost_usd:
                raise ValueError(
                    over_ceiling_message(
                        estimate=estimate,
                        ceiling=maximum_build_cost_usd,
                        project=project,
                        trace_file=trace_file,
                        source=source,
                        root=root,
                        world_model=world_model,
                        judge=judge,
                        embedder=embedder,
                        top_k=top_k,
                    )
                )
        if not require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=estimate,
            command=f"wmo build {project} {trace_file}",
            non_interactive=no_interactive,
        ):
            return
        with progress_display(_console) as progress:
            completion = _complete_grounded_build(
                store,
                completed,
                selected=selected,
                runtime_catalog=runtime_catalog,
                world_snapshot=world_snapshot,
                embedder_snapshot=embedder_snapshot,
                top_k=top_k,
                estimate=estimate,
                maximum_build_cost_usd=maximum_build_cost_usd,
                provider_spend_authorized=True,
                progress=progress,
            )
        built = completion.artifacts
        reused = completion.reused
    _capture_local_build_telemetry(
        completed.artifacts,
        root=root,
        indexed_steps=_rag_transition_count(store, built.serving_rag.artifact_id),
        duration_seconds=time.monotonic() - started,
    )
    _render_completed_build(
        completed,
        built=built,
        estimate=0.0 if reused else estimate,
        project=project,
    )


def _load_or_setup_catalog(
    root: Path,
    *,
    no_interactive: bool,
    providers: tuple[str, ...] = (),
) -> ModelCatalog:
    """Load complete build roles or run inline setup only for a real terminal.

    Args:
        root: Local WMO root containing the shared model catalog.
        no_interactive: Whether inline provider setup is forbidden.
        providers: Repeatable ``--provider`` values that skip the opening list.

    Returns:
        A complete model catalog with all required build roles.

    Raises:
        ValueError: Required configuration is missing outside an interactive terminal.
    """
    resolved_providers = resolve_setup_providers(providers)
    path = root / "models.toml"
    catalog = load_model_catalog(path) if path.exists() else None
    missing = _missing_build_configuration(catalog)
    if not missing:
        assert catalog is not None
        return catalog
    options = ProviderSetupOptions(providers=resolved_providers)
    if not no_interactive and can_prompt(_console):
        _console.print(f"Model setup is required: {', '.join(missing)}.")
        return run_provider_setup(
            root,
            options,
            non_interactive=False,
            replace=False,
            console=_console,
        )
    connection_example, model_example = provider_setup_json_examples()
    raise ValueError(
        "model configuration is incomplete before build: "
        + ", ".join(missing)
        + ". Run `wmo config providers` interactively, or configure automation with "
        f"`wmo config providers --non-interactive --connection-json '{connection_example}' "
        f"--model-json '{model_example}' --world-model model --judge model --embedder model`. "
        "Replace the example model ID and zero price with the provider's exact values."
    )


def _missing_build_configuration(catalog: ModelCatalog | None) -> tuple[str, ...]:
    """List every absent connection, model, and required build role.

    Args:
        catalog: Existing catalog, or ``None`` when no catalog file exists.

    Returns:
        Complete ordered labels for missing first-build configuration.
    """
    if catalog is None:
        return (
            "models.toml",
            "provider connections",
            "model aliases",
            "world_model",
            "judge",
            "embedder",
        )
    missing = []
    if not catalog.connections:
        missing.append("provider connections")
    if not catalog.models:
        missing.append("model aliases")
    for role in ("world_model", "judge", "embedder"):
        if getattr(catalog.roles, role) is None:
            missing.append(role)
    return tuple(missing)


def _selected_roles(
    catalog: ModelCatalog,
    *,
    world_model: str | None,
    judge: str | None,
    embedder: str | None,
) -> ProjectModelConfiguration:
    """Validate independent project overrides against available model aliases.

    Args:
        catalog: Complete shared model catalog.
        world_model: Optional project-specific world-model alias.
        judge: Optional project-specific judge alias.
        embedder: Optional project-specific embedder alias.

    Returns:
        Frozen role selections for the project build.

    Raises:
        ValueError: A required role is absent or names an unknown alias.
    """
    selected = {
        "world_model": world_model or catalog.roles.world_model,
        "judge": judge or catalog.roles.judge,
        "embedder": embedder or catalog.roles.embedder,
    }
    missing = tuple(name for name, alias in selected.items() if alias is None)
    if missing:
        raise ValueError("missing required build roles: " + ", ".join(missing))
    unknown = tuple(
        f"{name}={alias}" for name, alias in selected.items() if alias not in catalog.models
    )
    if unknown:
        raise ValueError("build roles name unknown aliases: " + ", ".join(unknown))
    return ProjectModelConfiguration(
        world_model=str(selected["world_model"]),
        judge=str(selected["judge"]),
        embedder=str(selected["embedder"]),
        candidates=(),
    )


def _validated_role_snapshots(
    runtime_catalog: RuntimeModelCatalog,
    selected: ProjectModelConfiguration,
) -> tuple[ModelSnapshot, ModelSnapshot, ModelCapabilities]:
    """Validate build roles from catalog metadata without reading credentials.

    Args:
        runtime_catalog: Resolver used only for static model snapshots.
        selected: Frozen world-model, judge, and embedder aliases.

    Returns:
        World-model snapshot, embedder snapshot, and embedder capabilities.

    Raises:
        ModelCapabilityError: The embedder cannot prove embedding support.
        ModelConnectionError: A selected alias or provider is unsupported.
        ValueError: The embedder has no explicit input price.
    """
    world_snapshot, _world_capabilities = runtime_catalog.snapshot(selected.world_model)
    runtime_catalog.snapshot(selected.judge)
    embedder_snapshot, embedder_capabilities = runtime_catalog.snapshot(selected.embedder)
    preflight_capabilities(
        selected.embedder,
        embedder_capabilities,
        CapabilityRequirement(requires_embeddings=True),
    )
    if embedder_capabilities.input_cost_per_million_tokens_usd is None:
        raise ValueError(
            f"embedder alias {selected.embedder!r} has no input_cost_per_million_tokens_usd; "
            "record explicit pricing before a provider-backed build"
        )
    return world_snapshot, embedder_snapshot, embedder_capabilities


def _embedding_cost_ceiling(
    completed: ProjectBuild,
    capabilities: ModelCapabilities,
) -> float:
    """Bound retry-inclusive embedding spend from the exact rendered retrieval inputs.

    Args:
        completed: Persisted trace and task build whose transitions will be embedded.
        capabilities: Static embedder capabilities and price metadata.

    Returns:
        Conservative USD ceiling across serving and fit-only index construction.

    Raises:
        ValueError: The selected embedder omits explicit input pricing.
    """
    price = capabilities.input_cost_per_million_tokens_usd
    if price is None:
        raise ValueError(
            "embedder has no input_cost_per_million_tokens_usd; "
            "record explicit pricing before a provider-backed build"
        )
    bindings = _lineage_bindings(completed)
    traces = completed.artifacts.trace_dataset.traces
    serving = extract_real_transitions(
        traces,
        bindings,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    fit = extract_real_transitions(
        traces,
        bindings,
        included_partitions=frozenset({"fit"}),
    )
    byte_count = sum(len(transition.key_text.encode("utf-8")) for transition in (*serving, *fit))
    maximum_input_tokens = byte_count * RetryPolicy().maximum_attempts
    return maximum_input_tokens * price / 1_000_000


def _project_store(root: Path, proposed: ProjectConfig) -> ProjectStore:
    """Initialize one project or verify mutable build pointers are the only difference.

    Args:
        root: Local WMO root.
        proposed: Complete project configuration for this build invocation.

    Returns:
        Initialized or verified project store.

    Raises:
        ValueError: Existing project configuration differs outside completed-build pointers.
    """
    store = ProjectStore(root, proposed.project_id)
    if not store.paths.project_toml.exists():
        store.initialize(proposed)
        return store
    existing = store.load_project()
    if existing.model_copy(update={"build": None}) != proposed:
        raise ValueError("project.toml already exists with different build configuration")
    return store


def _build_grounded_artifacts(
    store: ProjectStore,
    completed: ProjectBuild,
    *,
    world_alias: str,
    world_snapshot: ModelSnapshot,
    resolved_embedder: ResolvedModel,
    top_k: int,
    progress: ProgressHook | None = None,
) -> ProjectBuildArtifacts:
    """Build serving and fit-only RAG plus the executable world-model binding.

    Args:
        store: Project artifact store receiving immutable outputs.
        completed: Persisted trace and task build.
        world_alias: Configured world-model alias persisted on the artifact.
        world_snapshot: Secret-free world-model identity.
        resolved_embedder: Exact provider embedding binding.
        top_k: Default number of retrieved transitions.
        progress: Optional observer of embedding, RAG, and grounded-model stages.

    Returns:
        Exact manifest pointers for every completed build output.

    Raises:
        ValueError: The selected embedder lacks an explicit catalog input price.
    """
    created_at = completed.artifacts.trace_dataset.dataset.created_at
    revision = completed.review.code_revision
    trace_input = artifact_input(completed.artifacts.trace_dataset.manifest)
    task_input = artifact_input(
        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
    )
    bindings = _lineage_bindings(completed)
    assert resolved_embedder.embedding_client is not None
    embedding_price = resolved_embedder.capabilities.input_cost_per_million_tokens_usd
    if embedding_price is None:  # pragma: no cover - setup and capability preflight require it
        raise ValueError("the selected embedder has no explicit input price")
    rag_embedder = RAGEmbedderBinding(
        client=resolved_embedder.embedding_client,
        snapshot=resolved_embedder.snapshot,
        maximum_attempts=RetryPolicy().maximum_attempts,
        input_usd_per_million_tokens=embedding_price,
    )
    serving = persist_trace_rag(
        store.artifacts,
        (trace_input,),
        bindings,
        created_at=created_at,
        code_revision=revision,
        embedder=rag_embedder,
        default_top_k=top_k,
        included_partitions=frozenset({"fit", "held_out"}),
        progress=qualified(progress, "serving index"),
    )
    fit = persist_trace_rag(
        store.artifacts,
        (trace_input,),
        bindings,
        created_at=created_at,
        code_revision=revision,
        embedder=rag_embedder,
        default_top_k=top_k,
        included_partitions=frozenset({"fit"}),
        progress=qualified(progress, "fit-only index"),
    )
    report(progress, "grounded model")
    world = persist_grounded_world_model(
        store.artifacts,
        artifact_input(serving.manifest),
        model_alias=world_alias,
        model=world_snapshot,
        created_at=created_at,
        code_revision=revision,
        top_k=top_k,
    )
    return ProjectBuildArtifacts(
        trace_dataset=trace_input,
        task_set=task_input,
        serving_rag=artifact_input(serving.manifest),
        fit_rag=artifact_input(fit.manifest),
        world_model=artifact_input(world.manifest),
    )


def _complete_grounded_build(
    store: ProjectStore,
    completed: ProjectBuild,
    *,
    selected: ProjectModelConfiguration,
    runtime_catalog: RuntimeModelCatalog,
    world_snapshot: ModelSnapshot,
    embedder_snapshot: ModelSnapshot,
    top_k: int,
    estimate: float,
    maximum_build_cost_usd: float,
    provider_spend_authorized: bool,
    progress: ProgressHook | None = None,
) -> GroundedBuildCompletion:
    """Select matching grounded artifacts or execute their bounded embedding work.

    Args:
        store: Project receiving or replaying the completed build selection.
        completed: Deterministic trace and representative-task evidence.
        selected: Exact project role aliases.
        runtime_catalog: Catalog that can resolve the selected embedder after authorization.
        world_snapshot: Exact provider-free world-model identity.
        embedder_snapshot: Exact provider-free embedder identity.
        top_k: Frozen retrieval result count.
        estimate: Conservative retry-inclusive embedding cost.
        maximum_build_cost_usd: Strict grounded-build provider ceiling.
        provider_spend_authorized: Whether new embedding calls are authorized.
        progress: Optional observer of embedding, RAG, and finalization stages.

    Returns:
        Exact selected build artifacts and whether they were replayed.

    Raises:
        ValueError: New provider work is unapproved or exceeds its strict ceiling.
    """
    built = _reuse_completed_grounded_artifacts(
        store,
        completed,
        world_alias=selected.world_model,
        world_snapshot=world_snapshot,
        embedder_snapshot=embedder_snapshot,
        top_k=top_k,
    )
    reused = built is not None
    if built is None and not provider_spend_authorized:
        raise ValueError("grounded build provider work requires explicit authorization")
    if built is None:
        if estimate > maximum_build_cost_usd:
            raise ValueError(
                f"grounded build requires ${estimate:.6f}, above the configured "
                f"${maximum_build_cost_usd:.6f} ceiling"
            )
        resolved_embedder = runtime_catalog.preflight(
            selected.embedder,
            CapabilityRequirement(requires_embeddings=True),
        )
        built = _build_grounded_artifacts(
            store,
            completed,
            world_alias=selected.world_model,
            world_snapshot=world_snapshot,
            resolved_embedder=resolved_embedder,
            top_k=top_k,
            progress=progress,
        )
    report(progress, "finalization")
    select_completed_build(store, built, completed.review)
    return GroundedBuildCompletion(artifacts=built, reused=reused)


def _reuse_completed_grounded_artifacts(
    store: ProjectStore,
    completed: ProjectBuild,
    *,
    world_alias: str,
    world_snapshot: ModelSnapshot,
    embedder_snapshot: ModelSnapshot,
    top_k: int,
) -> ProjectBuildArtifacts | None:
    """Reuse a completely matching verified build without credentials or provider calls.

    Args:
        store: Project artifact store containing a possible completed build.
        completed: Current persisted trace and task build.
        world_alias: Configured world-model alias required by the artifact.
        world_snapshot: Secret-free world-model identity required by the artifact.
        embedder_snapshot: Secret-free embedder identity required by both indexes.
        top_k: Requested retrieval result count.

    Returns:
        Verified existing build pointers, or ``None`` when any identity differs.

    """
    existing = store.load_project().build
    if existing is None:
        return None
    trace_input = artifact_input(completed.artifacts.trace_dataset.manifest)
    task_input = artifact_input(
        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
    )
    if existing.trace_dataset != trace_input or existing.task_set != task_input:
        return None
    serving = load_rag_index(store.artifacts, existing.serving_rag.artifact_id)
    fit = load_rag_index(store.artifacts, existing.fit_rag.artifact_id)
    if (
        serving.index.embedder != embedder_snapshot
        or fit.index.embedder != embedder_snapshot
        or serving.index.default_top_k != top_k
        or fit.index.default_top_k != top_k
        or serving.index.included_partitions != ("fit", "held_out")
        or fit.index.included_partitions != ("fit",)
    ):
        return None
    world = GroundedWorldModelArtifact.model_validate_json(
        store.artifacts.read_bytes(existing.world_model.artifact_id, WORLD_MODEL_ARTIFACT_PATH)
    )
    if (
        world.serving_rag != existing.serving_rag
        or world.model_alias != world_alias
        or world.model != world_snapshot
        or world.top_k != top_k
    ):
        return None
    return existing


def _lineage_bindings(completed: ProjectBuild) -> tuple[RAGLineageBinding, ...]:
    """Convert frozen duplicate groups into complete RAG lineage partition bindings.

    Args:
        completed: Persisted mining output with leakage groups and split assignments.

    Returns:
        Deterministically ordered trace-to-lineage partition bindings.
    """
    mining = completed.artifacts.mining
    result = []
    for group in mining.analysis.leakage_groups:
        partition = mining.partition.partition_for(group.lineage_group_id)
        result.extend(
            RAGLineageBinding(
                trace_id=trace_id,
                lineage_id=group.lineage_group_id,
                partition=partition,
            )
            for trace_id in group.source_trace_ids
        )
    return tuple(sorted(result, key=lambda item: item.trace_id))


def _resolve_trace_file(trace_file: Path) -> Path:
    """Validate one explicit local trace-file selection without opening its content.

    Args:
        trace_file: User-selected local trace export.

    Returns:
        The validated file path.

    Raises:
        typer.BadParameter: The path is absent or is not a regular file.
    """
    if not trace_file.exists():
        raise typer.BadParameter(f"trace file not found: {trace_file}")
    if not trace_file.is_file():
        raise typer.BadParameter(
            f"--traces must name a trace export, not a directory: {trace_file}"
        )
    return trace_file


def _load_canonical_traces(path: Path, source: str) -> TraceNormalizationResult:
    """Read a raw source once through its explicit canonical loader.

    Args:
        path: Validated local trace export.
        source: Explicit supported source format.

    Returns:
        Canonical normalized trace result.

    Raises:
        TraceSourceError: The format is unsupported or normalization fails; the command's
            `usage_error` boundary converts it (a `ValueError`) into `typer.BadParameter`.
    """
    return load_trace_source(source, path)


def _capture_local_build_telemetry(
    completed: TaskSetBuild,
    *,
    root: Path,
    indexed_steps: int,
    duration_seconds: float,
) -> None:
    """Emit anonymous aggregate local build counts after completed persistence.

    Args:
        completed: Persisted task-set build used only for aggregate counts.
        root: Local WMO root holding telemetry preference state.
        indexed_steps: Count of indexed real transitions.
        duration_seconds: Completed build wall-clock duration.
    """
    tasks = completed.mining.tasks
    capture_build_completed(
        completion_id=completed.task_set.task_set_id,
        stats=BuildTelemetryStats(
            input_trace_count=len(completed.trace_dataset.traces),
            input_step_count=sum(len(trace.spans) for trace in completed.trace_dataset.traces),
            train_trace_count=sum(task.partition == "fit" for task in tasks),
            val_trace_count=0,
            heldout_trace_count=sum(task.partition == "held_out" for task in tasks),
            indexed_step_count=indexed_steps,
            duration_seconds=max(duration_seconds, 0.0),
        ),
        root=root,
    )


def _rag_transition_count(store: ProjectStore, artifact_id: str) -> int:
    """Read the completed RAG transition count for telemetry only.

    Args:
        store: Project store containing the immutable index.
        artifact_id: Exact RAG artifact identifier.

    Returns:
        Count of persisted real transitions.
    """
    return load_rag_index(store.artifacts, artifact_id).index.transition_count


def _render_preflight(
    *,
    accepted: int,
    invalid: int,
    fit_count: int,
    held_out_count: int,
    world_alias: str,
    world_model_id: str,
    embedder_alias: str,
    embedder_model_id: str,
    estimate: float,
    ceiling: float,
    reused: bool,
) -> None:
    """Print the complete build preflight before credentials or provider dispatch.

    Args:
        accepted: Count of valid normalized traces.
        invalid: Count of rejected input traces.
        fit_count: Representative fit-task count.
        held_out_count: Representative held-out task count.
        world_alias: Selected world-model catalog alias.
        world_model_id: Provider model ID bound to that alias.
        embedder_alias: Selected embedder catalog alias.
        embedder_model_id: Provider embedding model ID bound to that alias.
        estimate: Conservative maximum embedding cost in USD.
        ceiling: Configured ``--max-build-cost-usd`` value.
        reused: Whether an exact completed grounded build already exists.
    """
    _console.print("[bold]Build preflight[/bold]")
    _console.print(f"  [dim]traces[/dim]       {accepted} accepted, {invalid} invalid")
    _console.print(f"  [dim]split[/dim]        {fit_count} fit, {held_out_count} held out")
    _console.print(f"  [dim]world model[/dim]  {world_alias} [dim]({world_model_id})[/dim]")
    _console.print(f"  [dim]embedder[/dim]     {embedder_alias} [dim]({embedder_model_id})[/dim]")
    if reused:
        _console.print(
            "  [dim]embedding[/dim]    reuse exact completed indexes, $0.000000 new spend"
        )
    else:
        _console.print(f"  [dim]embedding[/dim]    at most ${estimate:.6f}")
    _console.print(f"  [dim]ceiling[/dim]      ${ceiling:.6f}")


def _render_completed_build(
    completed: ProjectBuild,
    *,
    built: ProjectBuildArtifacts,
    estimate: float,
    project: str,
) -> None:
    """Present exact accepted, excluded, split, and grounded artifact identities.

    Args:
        completed: Persisted project build and mining results.
        built: Exact completed grounded-artifact pointers.
        estimate: Conservative provider embedding spend ceiling.
        project: Local project identifier used in the next recommended command.
    """
    dataset = completed.artifacts.trace_dataset.dataset
    tasks = completed.artifacts.mining.tasks
    duplicate_count = len(dataset.trace_ids) - len(completed.artifacts.mining.analysis.candidates)
    _console.print(
        f"[green]\u2713[/green] ingested {len(dataset.trace_ids)} traces "
        f"[dim]({dataset.invalid_trace_count} invalid, {duplicate_count} duplicate)[/dim]"
    )
    if len(dataset.trace_ids) < 100 or len(dataset.trace_ids) > 1_000:
        _console.print("[yellow]guidance[/yellow] 100 to 1,000 traces is the usual starting range")
    _console.print(
        f"[green]\u2713[/green] split {sum(task.partition == 'fit' for task in tasks)} fit / "
        f"{sum(task.partition == 'held_out' for task in tasks)} held out"
    )
    _console.print(f"[green]\u2713[/green] serving RAG {built.serving_rag.artifact_id}")
    _console.print(f"[green]\u2713[/green] fit RAG {built.fit_rag.artifact_id}")
    _console.print(f"[green]\u2713[/green] world model {built.world_model.artifact_id}")
    _console.print(f"[dim]embedding spend ceiling ${estimate:.6f}[/dim]")
    _console.print(f"next: wmo optimize router {project}")

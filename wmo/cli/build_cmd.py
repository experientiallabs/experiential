"""Named project build from local traces to grounded world-model artifacts."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.consent import can_prompt, require_spend_consent
from wmo.cli.provider_setup import (
    ProviderSetupOptions,
    provider_setup_json_examples,
    run_provider_setup,
)
from wmo.common.config import ARTIFACT_DIR
from wmo.common.models import ModelCatalog, ModelCatalogError, load_model_catalog
from wmo.common.observability.telemetry import BuildTelemetryStats, capture_build_completed
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
from wmo.runtime.models import CapabilityRequirement, ResolvedModel, RuntimeModelCatalog
from wmo.simulation.build import ProjectBuild, TaskSetBuild, build_project
from wmo.simulation.ingest.otlp import (
    OtlpTraceFormatError,
    TraceNormalizationResult,
    load_otlp_file,
)
from wmo.simulation.ingest.posthog import PostHogPullError, load_posthog_file
from wmo.simulation.retrieval import (
    RAGEmbedderBinding,
    RAGLineageBinding,
    load_rag_index,
    persist_trace_rag,
)
from wmo.simulation.world_model import persist_grounded_world_model

_console = Console()
_CANONICAL_SOURCES = ("otlp", "posthog")
_PROJECT_ARGUMENT = typer.Argument(..., metavar="PROJECT", help="Local project name.")
_TRACE_FILE_ARGUMENT = typer.Argument(
    ...,
    metavar="TRACES",
    help="OTLP JSON or JSONL, or a PostHog LLM-observability export.",
)
_ROOT_OPTION = typer.Option(Path(ARTIFACT_DIR), "--root", help="Local .wmo artifact root.")


def build(
    project: str = _PROJECT_ARGUMENT,
    trace_file: Path = _TRACE_FILE_ARGUMENT,
    source: str = typer.Option("otlp", "--source", help="Trace format: otlp or posthog."),
    root: Path = _ROOT_OPTION,
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
    yes: bool = typer.Option(False, "--yes", help="Consent to the displayed embedding spend."),
    no_interactive: bool = typer.Option(
        False,
        "--no-interactive",
        help="Never prompt for missing model setup.",
    ),
) -> None:
    """Build a reusable grounded world model and immutable fit evidence.

    Model setup runs first when required catalog state is absent and both terminal streams are
    interactive. The shared catalog commits before project creation. Noninteractive missing state
    fails before any project or artifact write.

    Args:
        project: Safe local project identifier below ``<root>/projects``.
        trace_file: Explicit local canonical trace export.
        source: OTLP or PostHog local-export format.
        root: Local ``.wmo`` artifact root.
        world_model: Optional configured alias override for this project.
        judge: Optional configured alias override for this project.
        embedder: Optional configured alias override for this project.
        top_k: Positive serving retrieval result limit.
        maximum_build_cost_usd: Strict ceiling for provider embedding calls.
        yes: Explicit noninteractive or advance spend consent.
        no_interactive: Disable inline setup even when a terminal is available.

    Raises:
        typer.BadParameter: Input, setup, role, cost, project, or artifact validation fails.
    """
    started = time.monotonic()
    try:
        ProjectStore(root, project)
        catalog = _load_or_setup_catalog(root, no_interactive=no_interactive)
        selected = _selected_roles(
            catalog,
            world_model=world_model,
            judge=judge,
            embedder=embedder,
        )
        runtime_catalog = RuntimeModelCatalog(catalog)
        resolved_world = runtime_catalog.preflight(
            selected.world_model,
            CapabilityRequirement(minimum_output_tokens=8_000),
        )
        runtime_catalog.resolve(selected.judge)
        resolved_embedder = runtime_catalog.preflight(
            selected.embedder,
            CapabilityRequirement(requires_embeddings=True),
        )
        path = _resolve_trace_file(trace_file)
        normalized = _load_canonical_traces(path, source)
        if not normalized.traces:
            raise ValueError(
                "no valid canonical traces were produced; inspect the input and provide at least "
                "one valid OTLP or PostHog trace"
            )
        estimate = _embedding_cost_ceiling(normalized, resolved_embedder)
        if estimate > maximum_build_cost_usd:
            raise ValueError(
                f"conservative embedding estimate ${estimate:.6f} exceeds "
                f"--max-build-cost-usd ${maximum_build_cost_usd:.6f}"
            )
        if estimate > 0 and not require_spend_consent(
            _console,
            yes=yes,
            spend=f"at most ${estimate:.6f} for provider embeddings",
            command=f"wmo build {project} {trace_file}",
        ):
            return
        store = _project_store(
            root,
            ProjectConfig(
                project_id=project,
                trace_source=source.strip().casefold(),
                models=selected,
                retrieval=ProjectRetrievalConfiguration(top_k=top_k),
                budgets=ProjectBudgetConfiguration(maximum_build_cost_usd=maximum_build_cost_usd),
            ),
        )
        completed = build_project(
            normalized,
            store,
            created_at=datetime.now(UTC),
            code_revision=_current_revision(),
        )
        built = _build_grounded_artifacts(
            store,
            completed,
            resolved_world=resolved_world,
            resolved_embedder=resolved_embedder,
            top_k=top_k,
        )
        store.bind_completed_build(built)
    except (ArtifactStoreError, ModelCatalogError, ProjectStoreError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    _capture_local_build_telemetry(
        completed.artifacts,
        root=root,
        indexed_steps=_rag_transition_count(store, built.serving_rag.artifact_id),
        duration_seconds=time.monotonic() - started,
    )
    _render_completed_build(completed, built=built, estimate=estimate)


def _load_or_setup_catalog(root: Path, *, no_interactive: bool) -> ModelCatalog:
    """Load complete build roles or run inline setup only for a real terminal."""
    path = root / "models.toml"
    catalog = load_model_catalog(path) if path.exists() else None
    missing = _missing_build_configuration(catalog)
    if not missing:
        assert catalog is not None
        return catalog
    if not no_interactive and can_prompt(_console):
        _console.print(f"Model setup is required: {', '.join(missing)}.")
        return run_provider_setup(
            root,
            ProviderSetupOptions(),
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
        f"--model-json '{model_example}' --world-model ALIAS --judge ALIAS --embedder ALIAS`."
    )


def _missing_build_configuration(catalog: ModelCatalog | None) -> tuple[str, ...]:
    """List every absent connection, model, and required build role."""
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
    """Validate independent project overrides against available model aliases."""
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


def _embedding_cost_ceiling(
    normalized: TraceNormalizationResult,
    embedder: ResolvedModel,
) -> float:
    """Estimate a conservative three-attempt ceiling from UTF-8 bytes and explicit pricing."""
    price = embedder.capabilities.input_cost_per_million_tokens_usd
    if price is None:
        raise ValueError(
            f"embedder alias {embedder.alias!r} has no input_cost_per_million_tokens_usd; "
            "record explicit pricing before a provider-backed build"
        )
    byte_count = sum(len(trace.model_dump_json().encode("utf-8")) for trace in normalized.traces)
    maximum_input_tokens = byte_count * 9
    return maximum_input_tokens * price / 1_000_000


def _project_store(root: Path, proposed: ProjectConfig) -> ProjectStore:
    """Initialize one project or verify mutable build pointers are the only difference."""
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
    resolved_world: ResolvedModel,
    resolved_embedder: ResolvedModel,
    top_k: int,
) -> ProjectBuildArtifacts:
    """Build serving and fit-only RAG plus the executable world-model binding."""
    created_at = completed.artifacts.trace_dataset.dataset.created_at
    revision = completed.review.code_revision
    trace_input = artifact_input(completed.artifacts.trace_dataset.manifest)
    task_input = artifact_input(
        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
    )
    bindings = _lineage_bindings(completed)
    assert resolved_embedder.embedding_client is not None
    rag_embedder = RAGEmbedderBinding(
        client=resolved_embedder.embedding_client,
        snapshot=resolved_embedder.snapshot,
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
    )
    world = persist_grounded_world_model(
        store.artifacts,
        artifact_input(serving.manifest),
        model_alias=resolved_world.alias,
        model=resolved_world.snapshot,
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


def _lineage_bindings(completed: ProjectBuild) -> tuple[RAGLineageBinding, ...]:
    """Convert frozen duplicate groups into complete RAG lineage partition bindings."""
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
    """Validate one explicit local trace-file selection without opening its content."""
    if not trace_file.exists():
        raise typer.BadParameter(f"trace file not found: {trace_file}")
    if not trace_file.is_file():
        raise typer.BadParameter(f"TRACES must be a trace export, not a directory: {trace_file}")
    return trace_file


def _load_canonical_traces(path: Path, source: str) -> TraceNormalizationResult:
    """Read a raw source once through its explicit canonical loader."""
    normalized_source = source.strip().casefold()
    try:
        if normalized_source == "otlp":
            return load_otlp_file(path)
        if normalized_source == "posthog":
            return load_posthog_file(path)
    except (OtlpTraceFormatError, PostHogPullError, ValueError) as exc:
        raise typer.BadParameter(f"{normalized_source} trace normalization failed: {exc}") from None
    choices = ", ".join(_CANONICAL_SOURCES)
    raise typer.BadParameter(f"unsupported --source {source!r}; choose one of: {choices}")


def _current_revision() -> str:
    """Return the local Git revision when available without changing repository state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "local-unversioned"


def _capture_local_build_telemetry(
    completed: TaskSetBuild,
    *,
    root: Path,
    indexed_steps: int,
    duration_seconds: float,
) -> None:
    """Emit anonymous aggregate local build counts after completed persistence."""
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
    """Read the completed RAG transition count for telemetry only."""
    return load_rag_index(store.artifacts, artifact_id).index.transition_count


def _render_completed_build(
    completed: ProjectBuild,
    *,
    built: ProjectBuildArtifacts,
    estimate: float,
) -> None:
    """Present exact accepted, excluded, split, and grounded artifact identities."""
    dataset = completed.artifacts.trace_dataset.dataset
    tasks = completed.artifacts.mining.tasks
    duplicate_count = len(dataset.trace_ids) - len(completed.artifacts.mining.analysis.candidates)
    _console.print(
        f"[green]built[/green] {len(dataset.trace_ids)} accepted, "
        f"{dataset.invalid_trace_count} invalid, {duplicate_count} duplicate"
    )
    if len(dataset.trace_ids) < 100 or len(dataset.trace_ids) > 1_000:
        _console.print("[yellow]guidance[/yellow] 100 to 1,000 traces is the usual starting range")
    _console.print(
        f"[green]split[/green] {sum(task.partition == 'fit' for task in tasks)} fit, "
        f"{sum(task.partition == 'held_out' for task in tasks)} held_out"
    )
    _console.print(
        f"[green]grounded[/green] serving RAG {built.serving_rag.artifact_id}, "
        f"fit RAG {built.fit_rag.artifact_id}, world model {built.world_model.artifact_id}"
    )
    _console.print(f"embedding spend ceiling: ${estimate:.6f}")

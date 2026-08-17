"""Credential-free aggregate preflight for the hosted router application service."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from wmo.common.core.money import USD_ZERO, reserve_usd
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
    completion_cost_reservation,
    validate_router_candidate_selection,
)
from wmo.common.project import (
    ProjectCatalogModel,
    ProjectHostedSetup,
    ProjectModelCatalog,
    ProjectStore,
    load_project_model_catalog,
    persist_project_model_catalog,
)
from wmo.common.routing import router_embedding_reservation
from wmo.common.tasks import load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.optimize.router.automatic.attribution import resolve_router_observed_attributions
from wmo.optimize.router.automatic.reservations import (
    retrieval_embedding_reservation,
    simulation_completion_reservations,
    simulation_input_token_estimate,
)
from wmo.runtime.models import CapabilityRequirement, ResolvedModel, RuntimeModelCatalog
from wmo.runtime.models.providers.transport import RetryPolicy
from wmo.simulation.ingest.dataset import (
    read_trace_model_identity_evidence,
    verify_current_trace_dataset,
)
from wmo.simulation.mining.bindings import load_task_set_lineage_bindings
from wmo.simulation.retrieval import RAGLineageBinding
from wmo.simulation.retrieval.transitions import extract_real_transitions

if TYPE_CHECKING:
    from wmo.optimize.router.hosted import (
        HostedRouterWorkflowOptions,
        HostedRouterWorkflowSetup,
    )


class HostedRouterPreflightError(ValueError):
    """Aggregate hosted prerequisites are incomplete before any provider dispatch."""


@dataclass(frozen=True)
class HostedPreflight:
    """Credential-free resolved setup, evidence, catalog, and build reservation."""

    catalog: ModelCatalog
    project_catalog: ProjectModelCatalog
    candidates: tuple[RoutedCandidateSnapshot, ...]
    world_model: RoutedCandidateSnapshot
    judge: RoutedCandidateSnapshot
    embedder: RoutedCandidateSnapshot
    embedder_capabilities: ModelCapabilities
    build_cost_usd: Decimal


@dataclass(frozen=True)
class ResolvedHostedModels:
    """Transient credential-backed clients resolved before the first provider dispatch."""

    runtime_catalog: RuntimeModelCatalog
    world_model: ResolvedModel
    embedder: ResolvedModel


def preflight_hosted(
    project: ProjectStore,
    setup: HostedRouterWorkflowSetup,
    catalog: ModelCatalog,
    options: HostedRouterWorkflowOptions,
) -> HostedPreflight:
    """Aggregate every credential-free prerequisite before provider resolution or writes."""
    problems: list[str] = []
    config = project.load_project()
    stage = config.provider_free_stage
    if stage is None:
        problems.append("Project has no completed provider-free trace stage")
    if config.agent is not None or setup.system.kind != "builtin_chat":
        problems.append("hosted execution supports only the bounded built-in chat system")
    active = catalog.model_copy(
        update={
            "roles": catalog.roles.model_copy(
                update={
                    "candidates": setup.models.candidates,
                    "incumbent": setup.models.incumbent,
                    "world_model": setup.models.world_model,
                    "judge": setup.models.judge,
                    "embedder": setup.models.embedder,
                }
            )
        }
    )
    selection = RouterCandidateSelection(
        candidates=setup.models.candidates,
        incumbent=str(setup.models.incumbent),
    )
    problems.extend(validate_router_candidate_selection(active, selection))
    static = RuntimeModelCatalog(active, environment={})
    snapshots: dict[str, RoutedCandidateSnapshot] = {}
    capabilities: dict[str, ModelCapabilities] = {}
    required_aliases = {
        setup.models.world_model,
        setup.models.judge,
        setup.models.embedder,
        *setup.models.candidates,
    }
    for alias in sorted(required_aliases):
        try:
            snapshot, caps = static.snapshot(alias)
            snapshots[alias] = RoutedCandidateSnapshot(alias=alias, model=snapshot)
            capabilities[alias] = caps
        except ValueError as exc:
            problems.append(f"model alias {alias!r}: {exc}")
    for alias in (*setup.models.candidates, setup.models.world_model):
        _completion_capability_problems(
            problems,
            alias,
            capabilities.get(alias),
            maximum_output_tokens=options.simulation_maximum_output_tokens,
        )
    _completion_capability_problems(
        problems,
        setup.models.judge,
        capabilities.get(setup.models.judge),
        maximum_output_tokens=options.maximum_judge_output_tokens,
        maximum_input_tokens=options.maximum_judge_input_tokens,
    )
    embedder_capabilities = capabilities.get(setup.models.embedder)
    if (
        embedder_capabilities is None
        or not embedder_capabilities.supports_embeddings
        or embedder_capabilities.input_cost_per_million_tokens_usd is None
    ):
        problems.append("embedder requires explicit embedding support and input pricing")
    tasks = ()
    traces = ()
    bindings: tuple[RAGLineageBinding, ...] = ()
    if stage is not None:
        try:
            tasks = load_task_set(project.artifacts, stage.task_set.artifact_id).tasks
            loaded = load_trace_dataset(project.artifacts, stage.trace_dataset.artifact_id)
            verify_current_trace_dataset(project.artifacts, loaded)
            traces = loaded.traces
            lineage = load_task_set_lineage_bindings(
                project.artifacts,
                stage.task_set.artifact_id,
            )
            bindings = tuple(
                RAGLineageBinding(
                    trace_id=item.trace_id,
                    lineage_id=item.lineage_id,
                    partition=item.partition,
                )
                for item in lineage.bindings
            )
            evidence = read_trace_model_identity_evidence(project.artifacts, loaded)
            candidates = tuple(
                snapshots[alias] for alias in sorted(setup.models.candidates) if alias in snapshots
            )
            if len(candidates) == len(setup.models.candidates):
                resolve_router_observed_attributions(tasks, traces, evidence, candidates)
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"prepared trace evidence: {exc}")
    build_cost = USD_ZERO
    if config.build is None and traces and bindings and embedder_capabilities is not None:
        try:
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
            price = embedder_capabilities.input_cost_per_million_tokens_usd
            if price is not None:
                tokens = sum(len(item.key_text.encode("utf-8")) for item in (*serving, *fit))
                build_cost = reserve_usd(
                    tokens * RetryPolicy().maximum_attempts * price / 1_000_000
                )
        except ValueError as exc:
            problems.append(f"grounded build inputs: {exc}")
    ceiling = setup.budgets.maximum_provider_cost_usd
    if ceiling is not None:
        if build_cost > setup.budgets.maximum_build_cost_usd:
            problems.append("grounded build reservation exceeds maximum_build_cost_usd")
        candidates = tuple(
            snapshots[alias] for alias in sorted(setup.models.candidates) if alias in snapshots
        )
        reserved_after_build = _full_router_reservation(
            problems,
            tasks,
            traces,
            active,
            candidates,
            snapshots.get(setup.models.world_model),
            snapshots.get(setup.models.embedder),
            embedder_capabilities,
            snapshots.get(setup.models.judge),
            capabilities.get(setup.models.judge),
            setup,
            options,
        )
        required = build_cost + reserved_after_build
        if required > ceiling:
            problems.append(
                "provider ceiling cannot cover the full build, router embedding, judge, "
                "candidate, world-model, and retrieval reservation"
            )
    project_catalog = None
    if len(snapshots) == len(required_aliases):
        project_catalog = ProjectModelCatalog(
            project_id=config.project_id,
            models=tuple(
                ProjectCatalogModel(
                    alias=alias,
                    model=snapshots[alias].model,
                    capabilities=capabilities[alias],
                )
                for alias in sorted(required_aliases)
            ),
        )
    if config.system is not None and project_catalog is not None:
        _existing_setup_problems(project, setup, project_catalog, problems)
    if problems:
        raise HostedRouterPreflightError(
            "hosted router prerequisites are incomplete:\n- " + "\n- ".join(dict.fromkeys(problems))
        )
    assert project_catalog is not None
    candidates = tuple(snapshots[alias] for alias in sorted(setup.models.candidates))
    return HostedPreflight(
        catalog=active,
        project_catalog=project_catalog,
        candidates=candidates,
        world_model=snapshots[setup.models.world_model],
        judge=snapshots[setup.models.judge],
        embedder=snapshots[setup.models.embedder],
        embedder_capabilities=capabilities[setup.models.embedder],
        build_cost_usd=build_cost,
    )


def _completion_capability_problems(
    problems: list[str],
    alias: str,
    capabilities: ModelCapabilities | None,
    *,
    maximum_output_tokens: int,
    maximum_input_tokens: int = 1,
) -> None:
    """Append complete completion capability, price, and request-capacity failures."""
    if capabilities is None:
        return
    missing = []
    if capabilities.supports_completions is not True:
        missing.append("supports_completions=true")
    if capabilities.context_window_tokens is None:
        missing.append("context_window_tokens")
    elif maximum_input_tokens + maximum_output_tokens > capabilities.context_window_tokens:
        missing.append("sufficient context_window_tokens")
    if (
        capabilities.maximum_output_tokens is None
        or maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        missing.append("sufficient maximum_output_tokens")
    for name in (
        "input_cost_per_million_tokens_usd",
        "output_cost_per_million_tokens_usd",
        "cached_input_cost_per_million_tokens_usd",
        "cache_write_cost_per_million_tokens_usd",
    ):
        if getattr(capabilities, name) is None:
            missing.append(name)
    if missing:
        problems.append(f"completion alias {alias!r} is missing " + ", ".join(missing))


def _full_router_reservation(
    problems: list[str],
    tasks: Sequence[object],
    traces: tuple[Trace, ...],
    catalog: ModelCatalog,
    candidates: tuple[RoutedCandidateSnapshot, ...],
    world_model: RoutedCandidateSnapshot | None,
    embedder: RoutedCandidateSnapshot | None,
    embedder_capabilities: ModelCapabilities | None,
    judge: RoutedCandidateSnapshot | None,
    judge_capabilities: ModelCapabilities | None,
    setup: HostedRouterWorkflowSetup,
    options: HostedRouterWorkflowOptions,
) -> Decimal:
    """Return the full retry-inclusive automatic execution reservation before build dispatch."""
    if (
        not tasks
        or not traces
        or len(candidates) != len(setup.models.candidates)
        or world_model is None
        or embedder is None
        or embedder_capabilities is None
        or judge is None
        or judge_capabilities is None
    ):
        return USD_ZERO
    embed_price = embedder_capabilities.input_cost_per_million_tokens_usd
    prices = (
        judge_capabilities.input_cost_per_million_tokens_usd,
        judge_capabilities.output_cost_per_million_tokens_usd,
        judge_capabilities.cached_input_cost_per_million_tokens_usd,
        judge_capabilities.cache_write_cost_per_million_tokens_usd,
    )
    if embed_price is None or any(item is None for item in prices):
        return USD_ZERO
    embedding = router_embedding_reservation(
        model=embedder.model,
        input_usd_per_million_tokens=embed_price,
        maximum_attempts_per_feature=RetryPolicy().maximum_attempts,
        maximum_input_tokens_per_feature=options.maximum_router_feature_tokens,
        feature_count=len(tasks),
    )
    input_price, output_price, cached_price, write_price = prices
    assert input_price is not None and output_price is not None
    assert cached_price is not None and write_price is not None
    judgment = completion_cost_reservation(
        model=judge.model,
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
        cached_input_usd_per_million_tokens=cached_price,
        cache_write_usd_per_million_tokens=write_price,
        maximum_attempts=RetryPolicy().maximum_attempts,
        maximum_input_tokens=options.maximum_judge_input_tokens,
        maximum_output_tokens=options.maximum_judge_output_tokens,
    )
    estimated_input_tokens = simulation_input_token_estimate(
        traces,
        retrieved_transition_count=setup.retrieval.top_k,
        maximum_retrieval_query_tokens=options.maximum_retrieval_query_tokens,
        maximum_output_tokens=options.simulation_maximum_output_tokens,
    )
    if estimated_input_tokens is None:
        return USD_ZERO
    candidate_requests, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=candidates,
        world_alias=world_model.alias,
        world=world_model.model,
        maximum_attempts=RetryPolicy().maximum_attempts,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=options.simulation_maximum_output_tokens,
    )
    retrieval = retrieval_embedding_reservation(
        problems,
        catalog,
        embedder.alias,
        embedder.model,
        options.maximum_retrieval_query_tokens,
        RetryPolicy().maximum_attempts,
    )
    if len(candidate_requests) != len(candidates) or world_request is None or retrieval is None:
        return USD_ZERO
    retrieval_call = (
        retrieval.maximum_input_tokens
        * retrieval.maximum_attempts
        * retrieval.input_usd_per_million_tokens
        / 1_000_000
    )
    simulated_cell_turns = len(tasks) * setup.system.maximum_model_calls
    simulation = math.fsum(
        simulated_cell_turns
        * (
            candidate.request.estimated_maximum_call_cost_usd
            + world_request.estimated_maximum_call_cost_usd
            + retrieval_call
        )
        for candidate in candidate_requests
    )
    return reserve_usd(
        math.fsum(
            (
                embedding.estimated_cost_usd,
                judgment.estimated_maximum_call_cost_usd * options.maximum_judgments * 2,
                simulation,
            )
        )
    )


def _existing_setup_problems(
    project: ProjectStore,
    setup: HostedRouterWorkflowSetup,
    project_catalog: ProjectModelCatalog,
    problems: list[str],
) -> None:
    """Append conflicts between an existing locked setup and the proposed exact replay."""
    existing = project.load_project()
    if (
        existing.system != setup.system
        or existing.models != setup.models
        or existing.retrieval != setup.retrieval
        or existing.budgets != setup.budgets
        or existing.model_catalog is None
    ):
        problems.append("Project already has a different late hosted setup")
        return
    try:
        stored = load_project_model_catalog(project.artifacts, existing.model_catalog)
        if stored != project_catalog:
            problems.append("Project catalog differs from transient static model snapshots")
    except ValueError as exc:
        problems.append(f"Project catalog: {exc}")


def require_resolvable_clients(
    runtime_catalog: RuntimeModelCatalog,
    preflight: HostedPreflight,
    setup: HostedRouterWorkflowSetup,
    options: HostedRouterWorkflowOptions,
) -> None:
    """Resolve every transient credential-backed role before the first provider dispatch."""
    active = runtime_catalog.with_catalog(preflight.catalog)
    try:
        for alias in (*setup.models.candidates, setup.models.world_model):
            active.preflight(
                alias,
                CapabilityRequirement(
                    minimum_context_window_tokens=options.simulation_maximum_output_tokens + 1,
                    minimum_output_tokens=options.simulation_maximum_output_tokens,
                ),
            )
        active.preflight(
            setup.models.judge,
            CapabilityRequirement(
                minimum_context_window_tokens=(
                    options.maximum_judge_input_tokens + options.maximum_judge_output_tokens
                ),
                minimum_output_tokens=options.maximum_judge_output_tokens,
            ),
        )
        active.preflight(
            setup.models.embedder,
            CapabilityRequirement(requires_embeddings=True),
        )
    except ValueError as exc:
        raise HostedRouterPreflightError(
            "transient provider clients cannot resolve every preflighted role"
        ) from exc


def bind_hosted_setup(
    project: ProjectStore,
    setup: HostedRouterWorkflowSetup,
    preflight: HostedPreflight,
    created_at: datetime,
    code_revision: str,
) -> None:
    """Persist the secret-free Project catalog and apply the locked late setup once."""
    existing = project.load_project()
    pointer = existing.model_catalog
    if pointer is None:
        pointer = persist_project_model_catalog(
            project.artifacts,
            preflight.project_catalog,
            created_at=created_at,
            code_revision=code_revision,
        )
    project.bind_hosted_setup(
        ProjectHostedSetup(
            system=setup.system,
            models=setup.models,
            model_catalog=pointer,
            retrieval=setup.retrieval,
            budgets=setup.budgets,
        )
    )


def resolve_hosted_models(
    runtime_catalog: RuntimeModelCatalog,
    preflight: HostedPreflight,
) -> ResolvedHostedModels:
    """Resolve only the clients needed for build while preserving the transient catalog seam."""
    active = runtime_catalog.with_catalog(preflight.catalog)
    world = active.resolve(preflight.world_model.alias)
    embedder = active.preflight(
        preflight.embedder.alias,
        CapabilityRequirement(requires_embeddings=True),
    )
    return ResolvedHostedModels(
        runtime_catalog=active,
        world_model=world,
        embedder=embedder,
    )

"""Conservative provider reservations for automatic router optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, canonical_json_bytes, sha256_json
from exp.common.models import (
    CompletionCostReservation,
    EmbeddingCostReservation,
    ModelCatalog,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
    completion_cost_reservation,
)
from exp.common.routing import (
    RouterEmbeddingReservation,
    RouterFeatureExtractor,
    router_embedding_reservation,
    router_feature_token_upper_bound,
)
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.optimize.router.judging.contracts import ManualJudgeCalibrationAudit
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.engines.text.grounding import maximum_query_reservation
from exp.simulation.specs import CandidateCompletionReservation

_PROMPT_FRAMING_TOKEN_BUDGET = 4_096
"""Fixed conservative allowance for the frozen system prompt and message framing."""


def median_trace_token_estimate(traces: tuple[Trace, ...]) -> int | None:
    """Return the lower-median conservative token estimate over frozen build traces.

    One trace is measured as the UTF-8 byte length of its canonical serialization. This matches
    the provider-neutral byte-per-token upper bound used by simulation request admission, so a
    reservation sized from this estimate compares directly against counted request tokens.

    Args:
        traces: Verified traces persisted by the completed build.

    Returns:
        Deterministic lower-median byte-length token estimate, or ``None`` without traces.
    """
    if not traces:
        return None
    sizes = sorted(len(canonical_json_bytes(trace)) for trace in traces)
    return sizes[(len(sizes) - 1) // 2]


def simulation_input_token_estimate(
    traces: tuple[Trace, ...],
    *,
    retrieved_transition_count: int,
    maximum_retrieval_query_tokens: int,
    maximum_output_tokens: int,
) -> int | None:
    """Size one realistic per-call input planning estimate from the frozen build traces.

    The estimate sums explicit deterministic components instead of a model's full context
    window: one median-length trace for the visible episode transcript, one median-length trace
    for each of the world model's retrieved fit-RAG transitions rendered into the prompt (one
    whole trace bounds one transition), the explicit retrieval query token budget, one full
    output turn echoed back into the next request, and a fixed prompt-framing allowance.

    The estimate prices provider reservations only. It never bounds an individual request:
    the hard per-request admission ceiling is the model's real context capacity.

    Args:
        traces: Verified traces persisted by the completed build.
        retrieved_transition_count: Frozen world-model retrieval count rendered per prediction.
        maximum_retrieval_query_tokens: Explicit rendered RAG query token budget.
        maximum_output_tokens: Per-turn completion output ceiling echoed into later prompts.

    Returns:
        Deterministic per-call input token reservation, or ``None`` without traces.

    Raises:
        ValueError: The retrieval count is not positive.
    """
    if retrieved_transition_count <= 0:
        raise ValueError("retrieved transition count must be positive")
    median = median_trace_token_estimate(traces)
    if median is None:
        return None
    transcript_tokens = median
    retrieved_transition_tokens = median * retrieved_transition_count
    return (
        transcript_tokens
        + retrieved_transition_tokens
        + maximum_retrieval_query_tokens
        + maximum_output_tokens
        + _PROMPT_FRAMING_TOKEN_BUDGET
    )


@dataclass(frozen=True)
class AutomaticRouterOptions:
    """Tasteful bounded controls for one automatic router optimization."""

    maximum_provider_cost_usd: float = 25.0
    maximum_judgments: int = 100
    maximum_model_calls: int = 50
    maximum_router_feature_tokens: int = 8_192
    maximum_retrieval_query_tokens: int = 32_768
    router_embedding_maximum_attempts: int = 3
    completion_maximum_attempts: int = 3
    simulation_maximum_output_tokens: int = 16_000
    maximum_concurrency: int = 1
    seed: int = 0
    stop_on_overspend: bool = False


class CandidateEpisodeCostPlan(ContractModel):
    """One candidate's complete retry-bound simulation schedule."""

    candidate_alias: str = Field(min_length=1)
    episode_count: int = Field(ge=0)
    maximum_steps_per_episode: int = Field(gt=0)
    query_cost_per_step_usd: float = Field(ge=0)
    candidate_cost_per_step_usd: float = Field(ge=0)
    world_cost_per_step_usd: float = Field(ge=0)
    schedule_cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def _require_complete_schedule(self) -> CandidateEpisodeCostPlan:
        """Verify the persisted candidate schedule includes every possible step.

        Returns:
            The unchanged validated candidate schedule.

        Raises:
            ValueError: The schedule total omits an episode, step, or provider reservation.
        """
        expected = (
            self.episode_count
            * self.maximum_steps_per_episode
            * math.fsum(
                (
                    self.query_cost_per_step_usd,
                    self.candidate_cost_per_step_usd,
                    self.world_cost_per_step_usd,
                )
            )
        )
        if not math.isclose(self.schedule_cost_usd, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("candidate episode schedule differs from its complete reservation")
        return self


class AutomaticRouterCostPlan(ContractModel):
    """Pure conservative cost plan for one complete automatic-router schedule."""

    schema_version: Literal[1] = 1
    task_count: int = Field(gt=0)
    candidate_count: int = Field(ge=2)
    reusable_observed_count: int = Field(ge=0)
    maximum_judgments: int = Field(gt=0)
    judge_calls_per_judgment: Literal[1, 2]
    maximum_judge_provider_calls: int = Field(gt=0)
    simulated_episode_count: int = Field(gt=0)
    router_embedding_cost_usd: float = Field(ge=0)
    judgment_cost_usd: float = Field(ge=0)
    candidate_episodes: tuple[CandidateEpisodeCostPlan, ...] = Field(min_length=2)
    simulation_cost_usd: float = Field(ge=0)
    required_provider_cost_usd: float = Field(ge=0)

    @property
    def cost_plan_sha256(self) -> Sha256:
        """Return the digest of every count and conservative reservation in this plan."""
        return sha256_json(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _require_complete_plan(self) -> AutomaticRouterCostPlan:
        """Verify exact counts and the full provider schedule arithmetic.

        Returns:
            The unchanged validated cost plan.

        Raises:
            ValueError: Counts, candidate schedules, or the total reservation are incomplete.
        """
        complete_cell_count = self.task_count * self.candidate_count
        if self.maximum_judgments != complete_cell_count:
            raise ValueError("maximum judgments differ from the complete evaluation grid")
        if self.reusable_observed_count > complete_cell_count:
            raise ValueError("reusable observations exceed the complete evaluation grid")
        if self.maximum_judge_provider_calls != (
            self.maximum_judgments * self.judge_calls_per_judgment
        ):
            raise ValueError("judge provider calls differ from the exact judgment schedule")
        if self.simulated_episode_count != complete_cell_count - self.reusable_observed_count:
            raise ValueError("simulated episodes do not fill the unobserved evaluation cells")
        aliases = tuple(item.candidate_alias for item in self.candidate_episodes)
        if len(aliases) != self.candidate_count or len(set(aliases)) != len(aliases):
            raise ValueError("candidate episode schedules must cover each candidate exactly once")
        if any(item.episode_count > self.task_count for item in self.candidate_episodes):
            raise ValueError("a candidate schedule exceeds the complete task grid")
        scheduled_episodes = sum(item.episode_count for item in self.candidate_episodes)
        if scheduled_episodes != self.simulated_episode_count:
            raise ValueError("candidate schedules differ from the simulated episode count")
        expected_simulation = math.fsum(item.schedule_cost_usd for item in self.candidate_episodes)
        if not math.isclose(
            self.simulation_cost_usd,
            expected_simulation,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("simulation total differs from the candidate schedules")
        expected_total = math.fsum(
            (
                self.router_embedding_cost_usd,
                self.judgment_cost_usd,
                self.simulation_cost_usd,
            )
        )
        if not math.isclose(
            self.required_provider_cost_usd,
            expected_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("provider total differs from the complete automatic schedule")
        return self


def router_feature_reservation(
    problems: list[str],
    catalog: ModelCatalog,
    alias: str | None,
    model: ModelSnapshot | None,
    tasks: tuple[TaskCase, ...],
    maximum_tokens: int,
    maximum_attempts: int,
) -> RouterEmbeddingReservation | None:
    """Build the complete feature reservation after static embedder validation.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Local model catalog.
        alias: Build-frozen embedder alias.
        model: Exact embedder identity.
        tasks: Verified completed-build tasks.
        maximum_tokens: Conservative tokens reserved per feature.
        maximum_attempts: Retry ceiling reserved per feature.

    Returns:
        Exact reservation, or ``None`` when inputs are unavailable.
    """
    if alias is None or model is None or not tasks:
        return None
    capabilities = catalog.models[alias].capabilities
    price = capabilities.input_cost_per_million_tokens_usd if capabilities is not None else None
    if price is None:
        return None
    features = set(RouterFeatureExtractor().from_task(task) for task in tasks)
    required_tokens = max(
        (router_feature_token_upper_bound(feature) for feature in features), default=0
    )
    if required_tokens > maximum_tokens:
        problems.append(
            "router embedding reservation: rendered feature requires at least "
            f"{required_tokens} input tokens, above the configured {maximum_tokens} ceiling"
        )
        return None
    try:
        return router_embedding_reservation(
            model=model,
            input_usd_per_million_tokens=price,
            maximum_attempts_per_feature=maximum_attempts,
            maximum_input_tokens_per_feature=maximum_tokens,
            feature_count=len(features),
        )
    except ValueError as exc:
        problems.append(f"router embedding reservation: {exc}")
        return None


def simulation_completion_reservations(
    problems: list[str],
    *,
    catalog: ModelCatalog,
    candidates: tuple[RoutedCandidateSnapshot, ...],
    world_alias: str | None,
    world: ModelSnapshot | None,
    maximum_attempts: int,
    estimated_input_tokens: int,
    maximum_output_tokens: int,
) -> tuple[tuple[CandidateCompletionReservation, ...], CompletionCostReservation | None]:
    """Freeze candidate and world call reservations from exact catalog declarations.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        candidates: Exact selected candidate snapshots.
        world_alias: Build-frozen world-model alias.
        world: Exact world-model snapshot.
        maximum_attempts: Active completion retry ceiling.
        estimated_input_tokens: Trace-derived realistic per-call input planning size.
        maximum_output_tokens: Per-turn candidate and world output ceiling.

    Returns:
        Candidate reservations and the world-model reservation when inputs are complete.
    """
    candidate_requests = []
    for candidate in candidates:
        request = completion_reservation_from_catalog(
            problems,
            catalog=catalog,
            alias=candidate.alias,
            model=candidate.model,
            label="candidate",
            maximum_attempts=maximum_attempts,
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
        )
        if request is not None:
            candidate_requests.append(
                CandidateCompletionReservation(
                    candidate_alias=candidate.alias,
                    request=request,
                )
            )
    world_request = (
        completion_reservation_from_catalog(
            problems,
            catalog=catalog,
            alias=world_alias,
            model=world,
            label="world model",
            maximum_attempts=maximum_attempts,
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
        )
        if world_alias is not None and world is not None
        else None
    )
    return tuple(candidate_requests), world_request


def retrieval_embedding_reservation(
    problems: list[str],
    catalog: ModelCatalog,
    alias: str | None,
    model: ModelSnapshot | None,
    maximum_input_tokens: int,
    maximum_attempts: int,
) -> EmbeddingCostReservation | None:
    """Freeze one query-embedding price, retry, and input ceiling.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        alias: Build-frozen embedder alias.
        model: Exact embedder model identity.
        maximum_input_tokens: Maximum rendered RAG query input.
        maximum_attempts: Active embedding retry ceiling.

    Returns:
        Exact retrieval reservation, or ``None`` when metadata is unavailable.
    """
    if alias is None or model is None:
        return None
    capabilities = catalog.models[alias].capabilities
    price = capabilities.input_cost_per_million_tokens_usd if capabilities is not None else None
    if price is None:
        return None
    try:
        return EmbeddingCostReservation(
            model=model,
            input_usd_per_million_tokens=price,
            maximum_attempts=maximum_attempts,
            maximum_input_tokens=maximum_input_tokens,
        )
    except ValueError as exc:
        problems.append(f"retrieval embedding reservation: {exc}")
        return None


def completion_reservation_from_catalog(
    problems: list[str],
    *,
    catalog: ModelCatalog,
    alias: str,
    model: ModelSnapshot,
    label: str,
    maximum_attempts: int,
    estimated_input_tokens: int,
    maximum_output_tokens: int,
) -> CompletionCostReservation | None:
    """Create one completion reservation from exact capacity and pricing metadata.

    The hard per-request admission ceiling is the model's full context capacity after its
    per-turn output budget. The trace-derived estimate prices the reservation only.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        alias: Exact model alias.
        model: Frozen provider model identity.
        label: Candidate, world-model, or judge diagnostic role.
        maximum_attempts: Active provider request-attempt ceiling.
        estimated_input_tokens: Trace-derived realistic per-request input planning size.
        maximum_output_tokens: Per-request output ceiling.

    Returns:
        Exact reservation, or ``None`` after recording incomplete capacity or pricing.
    """
    capabilities = catalog.models[alias].capabilities
    if capabilities is None:
        return None
    context = capabilities.context_window_tokens
    if (
        context is None
        or capabilities.maximum_output_tokens is None
        or maximum_output_tokens > capabilities.maximum_output_tokens
        or maximum_output_tokens >= context
    ):
        problems.append(
            f"{label} alias {alias!r} cannot reserve {maximum_output_tokens} output tokens "
            "inside its explicit capacity"
        )
        return None
    maximum_input_tokens = context - maximum_output_tokens
    if estimated_input_tokens <= 0 or estimated_input_tokens > maximum_input_tokens:
        problems.append(
            f"{label} alias {alias!r} cannot fit the estimated {estimated_input_tokens} input "
            f"plus {maximum_output_tokens} output tokens inside its {context}-token context window"
        )
        return None
    prices = (
        capabilities.input_cost_per_million_tokens_usd,
        capabilities.output_cost_per_million_tokens_usd,
        capabilities.cached_input_cost_per_million_tokens_usd,
        capabilities.cache_write_cost_per_million_tokens_usd,
    )
    if any(value is None for value in prices):
        return None
    input_price, output_price, cached_input_price, cache_write_price = prices
    assert input_price is not None and output_price is not None
    assert cached_input_price is not None and cache_write_price is not None
    try:
        return completion_cost_reservation(
            model=model,
            input_usd_per_million_tokens=input_price,
            output_usd_per_million_tokens=output_price,
            cached_input_usd_per_million_tokens=cached_input_price,
            cache_write_usd_per_million_tokens=cache_write_price,
            maximum_attempts=maximum_attempts,
            maximum_input_tokens=maximum_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            estimated_input_tokens=estimated_input_tokens,
        )
    except ValueError as exc:
        problems.append(f"{label} alias {alias!r} reservation: {exc}")
        return None


def judge_completion_reservation(
    problems: list[str],
    *,
    catalog: ModelCatalog,
    judge_alias: str | None,
    judge: ModelSnapshot | None,
    audit: ManualJudgeCalibrationAudit | None,
    provisional: bool = False,
    provisional_maximum_attempts: int = 3,
) -> CompletionCostReservation | None:
    """Freeze judge calls priced by approved bounds or a conservative provisional policy.

    The approved or provisional input budget is the realistic planning size that prices the
    reservation. The hard per-request admission ceiling is the judge's real context capacity
    after its output budget, so an oversized rollout transcript is still admitted when it
    fits the model and the remaining spend.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        judge_alias: Build-frozen judge alias.
        judge: Exact judge model snapshot.
        audit: Approved manual calibration audit with consented request bounds.
        provisional: Whether a zero-label calibration may use current conservative bounds.
        provisional_maximum_attempts: Retry ceiling for provisional judging.

    Returns:
        Exact production judge request reservation, or ``None`` when unavailable.
    """
    if judge_alias is None or judge is None or (audit is None and not provisional):
        return None
    capabilities = catalog.models[judge_alias].capabilities
    if capabilities is None:
        return None
    context = capabilities.context_window_tokens
    if context is None:
        return None
    if audit is not None:
        budget = audit.budget
        input_price = budget.input_usd_per_million_tokens
        output_price = budget.output_usd_per_million_tokens
        maximum_attempts = budget.maximum_attempts_per_call
        estimated_input_tokens = budget.maximum_input_tokens_per_call
        maximum_output_tokens = budget.maximum_output_tokens_per_call
        if (
            capabilities.input_cost_per_million_tokens_usd != input_price
            or capabilities.output_cost_per_million_tokens_usd != output_price
        ):
            problems.append("approved judge calibration prices differ from the active catalog")
            return None
        if estimated_input_tokens + maximum_output_tokens > context:
            problems.append("approved judge request reservation exceeds active context capacity")
            return None
    else:
        input_price = capabilities.input_cost_per_million_tokens_usd
        output_price = capabilities.output_cost_per_million_tokens_usd
        maximum_output_tokens = min(capabilities.maximum_output_tokens or 4_096, 4_096)
        if input_price is None or output_price is None:
            return None
        estimated_input_tokens = min(32_768, context - maximum_output_tokens)
        maximum_attempts = provisional_maximum_attempts
        if estimated_input_tokens <= 0 or maximum_attempts <= 0:
            problems.append("provisional judge request bounds exceed active context capacity")
            return None
    cached_input_price = capabilities.cached_input_cost_per_million_tokens_usd
    cache_write_price = capabilities.cache_write_cost_per_million_tokens_usd
    if cached_input_price is None or cache_write_price is None:
        return None
    try:
        return completion_cost_reservation(
            model=judge,
            input_usd_per_million_tokens=input_price,
            output_usd_per_million_tokens=output_price,
            cached_input_usd_per_million_tokens=cached_input_price,
            cache_write_usd_per_million_tokens=cache_write_price,
            maximum_attempts=maximum_attempts,
            maximum_input_tokens=context - maximum_output_tokens,
            maximum_output_tokens=maximum_output_tokens,
            estimated_input_tokens=estimated_input_tokens,
        )
    except ValueError as exc:
        problems.append(f"judge reservation: {exc}")
        return None


def plan_automatic_router_cost(
    tasks: tuple[TaskCase, ...],
    catalog: ModelCatalog,
    selection: RouterCandidateSelection,
    *,
    world_model_alias: str,
    judge_alias: str,
    embedder_alias: str,
    judge_response_shape: Literal["scalar", "boolean", "categorical", "pairwise"],
    judge_audit: ManualJudgeCalibrationAudit | None,
    provisional_judge: bool,
    observed_candidate_aliases: tuple[str, ...],
    estimated_input_tokens: int,
    options: AutomaticRouterOptions,
) -> AutomaticRouterCostPlan:
    """Plan every possible automatic-router provider call without I/O.

    Args:
        tasks: Exact representative task schedule.
        catalog: Static secret-free model catalog.
        selection: Exact candidate set and incumbent.
        world_model_alias: Build-selected world-model alias.
        judge_alias: Build-selected judge alias.
        embedder_alias: Build-selected embedding alias.
        judge_response_shape: Finalized judge response protocol.
        judge_audit: Human-approved request bounds, when selected.
        provisional_judge: Whether conservative zero-label judge bounds apply.
        observed_candidate_aliases: Candidate aliases for exact reusable historical cells.
        estimated_input_tokens: Trace-derived realistic per-call input planning size.
        options: Quality, retry, token, and episode ceilings.

    Returns:
        Complete deterministic reservation for the task-candidate schedule.

    Raises:
        ValueError: Static identities, capacities, prices, or reservations are incomplete.
    """
    problems: list[str] = []
    if not tasks:
        problems.append("automatic cost plan requires at least one task")
    resolver = RuntimeModelCatalog(catalog, environment={})

    def snapshot(alias: str, label: str) -> ModelSnapshot | None:
        """Resolve one static catalog identity and collect a labeled error.

        Args:
            alias: Stable local model alias.
            label: User-facing role name for diagnostics.

        Returns:
            Static model identity, or ``None`` after recording a failure.
        """
        try:
            value, _capabilities = resolver.snapshot(alias)
        except ValueError as exc:
            problems.append(f"{label}: {exc}")
            return None
        return value

    candidate_snapshots = tuple(
        RoutedCandidateSnapshot(alias=alias, model=model)
        for alias in selection.candidates
        if (model := snapshot(alias, f"candidate {alias!r}")) is not None
    )
    world = snapshot(world_model_alias, "world model")
    judge = snapshot(judge_alias, "judge")
    embedder = snapshot(embedder_alias, "embedder")
    router = router_feature_reservation(
        problems,
        catalog,
        embedder_alias,
        embedder,
        tasks,
        options.maximum_router_feature_tokens,
        options.router_embedding_maximum_attempts,
    )
    query = retrieval_embedding_reservation(
        problems,
        catalog,
        embedder_alias,
        embedder,
        options.maximum_retrieval_query_tokens,
        options.router_embedding_maximum_attempts,
    )
    candidates, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=candidate_snapshots,
        world_alias=world_model_alias,
        world=world,
        maximum_attempts=options.completion_maximum_attempts,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=options.simulation_maximum_output_tokens,
    )
    judge_request = judge_completion_reservation(
        problems,
        catalog=catalog,
        judge_alias=judge_alias,
        judge=judge,
        audit=judge_audit,
        provisional=provisional_judge,
        provisional_maximum_attempts=options.completion_maximum_attempts,
    )
    if len(candidate_snapshots) != len(selection.candidates):
        problems.append("automatic cost plan could not resolve every selected candidate")
    if len(candidates) != len(selection.candidates):
        problems.append("automatic cost plan lacks a request reservation for a candidate")
    if any(value is None for value in (router, query, world_request, judge_request)):
        problems.append("automatic cost plan has incomplete provider reservations")
    if problems:
        raise ValueError("automatic router cost plan failed:\n- " + "\n- ".join(problems))
    assert router is not None and query is not None
    assert world_request is not None and judge_request is not None
    query_economics = maximum_query_reservation(query).cost_usd
    if query_economics is None:  # pragma: no cover - maximum query reservation is always priced
        raise ValueError("automatic router query reservation has no cost")
    task_count = len(tasks)
    candidate_count = len(selection.candidates)
    unknown_observed = sorted(set(observed_candidate_aliases) - set(selection.candidates))
    if unknown_observed:
        raise ValueError(f"observed cells name unselected candidates: {unknown_observed}")
    observed_counts = {
        alias: observed_candidate_aliases.count(alias) for alias in selection.candidates
    }
    if any(count > task_count for count in observed_counts.values()):
        raise ValueError("observed cells exceed the task schedule for one candidate")
    reusable_observed_count = len(observed_candidate_aliases)
    maximum_judgments = task_count * candidate_count
    calls_per_judgment: Literal[1, 2] = 2 if judge_response_shape == "pairwise" else 1
    maximum_judge_provider_calls = maximum_judgments * calls_per_judgment
    judgment_cost = judge_request.estimated_maximum_call_cost_usd * maximum_judge_provider_calls
    by_alias = {item.candidate_alias: item.request for item in candidates}
    episode_plans = tuple(
        CandidateEpisodeCostPlan(
            candidate_alias=alias,
            episode_count=task_count - observed_counts[alias],
            maximum_steps_per_episode=options.maximum_model_calls,
            query_cost_per_step_usd=query_economics.value,
            candidate_cost_per_step_usd=by_alias[alias].estimated_maximum_call_cost_usd,
            world_cost_per_step_usd=world_request.estimated_maximum_call_cost_usd,
            schedule_cost_usd=(
                (task_count - observed_counts[alias])
                * options.maximum_model_calls
                * math.fsum(
                    (
                        query_economics.value,
                        by_alias[alias].estimated_maximum_call_cost_usd,
                        world_request.estimated_maximum_call_cost_usd,
                    )
                )
            ),
        )
        for alias in selection.candidates
    )
    simulation_cost = math.fsum(item.schedule_cost_usd for item in episode_plans)
    return AutomaticRouterCostPlan(
        task_count=task_count,
        candidate_count=candidate_count,
        reusable_observed_count=reusable_observed_count,
        maximum_judgments=maximum_judgments,
        judge_calls_per_judgment=calls_per_judgment,
        maximum_judge_provider_calls=maximum_judge_provider_calls,
        simulated_episode_count=task_count * candidate_count - reusable_observed_count,
        router_embedding_cost_usd=router.estimated_cost_usd,
        judgment_cost_usd=judgment_cost,
        candidate_episodes=episode_plans,
        simulation_cost_usd=simulation_cost,
        required_provider_cost_usd=math.fsum(
            (router.estimated_cost_usd, judgment_cost, simulation_cost)
        ),
    )


def remaining_simulation_budget(
    problems: list[str],
    *,
    maximum_provider_cost_usd: float,
    router_reservation: RouterEmbeddingReservation | None,
) -> float:
    """Subtract the router-embedding reservation from one provider-spend ceiling.

    Judge calls take no upfront carve-out: judgments draw from this same shared remainder as
    reconciled actual spend, so a large judge planning estimate never starves simulation.

    Args:
        problems: Mutable aggregate problem list.
        maximum_provider_cost_usd: User-approved total provider ceiling.
        router_reservation: Conservative router-feature embedding reservation.

    Returns:
        Positive shared ceiling remaining for simulation and judging provider calls.
    """
    if router_reservation is None or maximum_provider_cost_usd <= 0:
        return 0.0
    remaining = maximum_provider_cost_usd - router_reservation.estimated_cost_usd
    if remaining <= 0:
        problems.append(
            "the router embedding reservation consumes the entire provider spend ceiling; "
            "increase --maximum-simulation-cost-usd or lower a request/retry ceiling"
        )
        return 0.0
    return remaining

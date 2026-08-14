"""Conservative provider reservations for automatic router optimization."""

from __future__ import annotations

import math

from wmo.common.models import (
    CompletionCostReservation,
    EmbeddingCostReservation,
    ModelCatalog,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    completion_cost_reservation,
)
from wmo.common.routing import (
    RouterEmbeddingReservation,
    RouterFeatureExtractor,
    router_embedding_reservation,
    router_feature_token_upper_bound,
)
from wmo.common.tasks import TaskCase
from wmo.optimize.router.manual_judge_contracts import ManualJudgeCalibrationAudit
from wmo.simulation.specs import CandidateCompletionReservation


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
    maximum_output_tokens: int,
) -> tuple[tuple[CandidateCompletionReservation, ...], CompletionCostReservation | None]:
    """Freeze candidate and world call ceilings from exact catalog declarations.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        candidates: Exact selected candidate snapshots.
        world_alias: Build-frozen world-model alias.
        world: Exact world-model snapshot.
        maximum_attempts: Active completion retry ceiling.
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
    maximum_output_tokens: int,
) -> CompletionCostReservation | None:
    """Create one completion reservation from exact capacity and pricing metadata.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        alias: Exact model alias.
        model: Frozen provider model identity.
        label: Candidate, world-model, or judge diagnostic role.
        maximum_attempts: Active provider request-attempt ceiling.
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
            maximum_input_tokens=context - maximum_output_tokens,
            maximum_output_tokens=maximum_output_tokens,
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
) -> CompletionCostReservation | None:
    """Freeze production judge calls to the exact approved calibration budget.

    Args:
        problems: Mutable aggregate problem list.
        catalog: Verified local model catalog.
        judge_alias: Build-frozen judge alias.
        judge: Exact judge model snapshot.
        audit: Approved manual calibration audit with consented request bounds.

    Returns:
        Exact production judge request reservation, or ``None`` when unavailable.
    """
    if judge_alias is None or judge is None or audit is None:
        return None
    capabilities = catalog.models[judge_alias].capabilities
    if capabilities is None:
        return None
    budget = audit.budget
    if (
        capabilities.input_cost_per_million_tokens_usd != budget.input_usd_per_million_tokens
        or capabilities.output_cost_per_million_tokens_usd != budget.output_usd_per_million_tokens
    ):
        problems.append("approved judge calibration prices differ from the active catalog")
        return None
    cached_input_price = capabilities.cached_input_cost_per_million_tokens_usd
    cache_write_price = capabilities.cache_write_cost_per_million_tokens_usd
    if cached_input_price is None or cache_write_price is None:
        return None
    try:
        reservation = completion_cost_reservation(
            model=judge,
            input_usd_per_million_tokens=budget.input_usd_per_million_tokens,
            output_usd_per_million_tokens=budget.output_usd_per_million_tokens,
            cached_input_usd_per_million_tokens=cached_input_price,
            cache_write_usd_per_million_tokens=cache_write_price,
            maximum_attempts=budget.maximum_attempts_per_call,
            maximum_input_tokens=budget.maximum_input_tokens_per_call,
            maximum_output_tokens=budget.maximum_output_tokens_per_call,
        )
    except ValueError as exc:
        problems.append(f"judge reservation: {exc}")
        return None
    context = capabilities.context_window_tokens
    if context is None or (
        reservation.maximum_input_tokens + reservation.maximum_output_tokens > context
    ):
        problems.append("approved judge request reservation exceeds active context capacity")
        return None
    return reservation


def remaining_simulation_budget(
    problems: list[str],
    *,
    maximum_provider_cost_usd: float,
    router_reservation: RouterEmbeddingReservation | None,
    judge_reservation_cost_usd: float,
) -> float:
    """Subtract non-simulation reservations from one provider-spend ceiling.

    Args:
        problems: Mutable aggregate problem list.
        maximum_provider_cost_usd: User-approved total provider ceiling.
        router_reservation: Conservative router-feature embedding reservation.
        judge_reservation_cost_usd: Full maximum judgment-set reservation.

    Returns:
        Positive ceiling remaining for candidate, retrieval, and world-model calls.
    """
    if router_reservation is None or maximum_provider_cost_usd <= 0:
        return 0.0
    remaining = maximum_provider_cost_usd - math.fsum(
        (router_reservation.estimated_cost_usd, judge_reservation_cost_usd)
    )
    if remaining <= 0:
        problems.append(
            "router embedding and judge reservations consume the entire provider spend ceiling; "
            "increase --maximum-simulation-cost-usd or lower a request/retry ceiling"
        )
        return 0.0
    return remaining

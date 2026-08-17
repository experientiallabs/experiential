"""Tests for automatic router provider reservations."""

from __future__ import annotations

from wmo.common.models import RoutedCandidateSnapshot, RouterCandidateSelection
from wmo.common.tasks import TaskCase
from wmo.optimize.router.automatic.reservations import (
    AutomaticRouterOptions,
    plan_automatic_router_cost,
    simulation_completion_reservations,
)
from wmo.optimize.router.automatic.service_test import _catalog
from wmo.runtime.models import RuntimeModelCatalog


def test_cost_plan_reserves_exact_small_schedule_without_io() -> None:
    """The pure planner reuses six cells and reserves the remaining 18 episodes."""
    catalog = _catalog()
    original = catalog.model_dump(mode="json")
    plan = plan_automatic_router_cost(
        _tasks(12),
        catalog,
        RouterCandidateSelection(
            candidates=("candidate-a", "candidate-b"),
            incumbent="candidate-a",
        ),
        world_model_alias="world",
        judge_alias="judge",
        embedder_alias="embedder",
        judge_response_shape="scalar",
        judge_audit=None,
        provisional_judge=True,
        observed_candidate_aliases=("candidate-a",) * 3 + ("candidate-b",) * 3,
        options=AutomaticRouterOptions(),
    )

    assert plan.maximum_judgments == 24
    assert plan.maximum_judge_provider_calls == 24
    assert plan.simulated_episode_count == 18
    assert tuple(item.episode_count for item in plan.candidate_episodes) == (9, 9)
    assert plan.required_provider_cost_usd == (
        plan.router_embedding_cost_usd + plan.judgment_cost_usd + plan.simulation_cost_usd
    )
    assert catalog.model_dump(mode="json") == original


def test_cost_plan_reserves_full_default_corpus_and_pairwise_calls() -> None:
    """The 50/20 corpus reuses ten cells and reserves 130 fresh episodes."""
    catalog = _catalog()
    plan = plan_automatic_router_cost(
        _tasks(70),
        catalog,
        RouterCandidateSelection(
            candidates=("candidate-a", "candidate-b"),
            incumbent="candidate-a",
        ),
        world_model_alias="world",
        judge_alias="judge",
        embedder_alias="embedder",
        judge_response_shape="pairwise",
        judge_audit=None,
        provisional_judge=True,
        observed_candidate_aliases=("candidate-a",) * 5 + ("candidate-b",) * 5,
        options=AutomaticRouterOptions(),
    )

    assert plan.maximum_judgments == 140
    assert plan.maximum_judge_provider_calls == 280
    assert plan.simulated_episode_count == 130
    assert plan.judge_calls_per_judgment == 2


def test_cost_plan_digest_changes_with_the_full_schedule() -> None:
    """A task-count change produces a distinct immutable reservation digest."""
    catalog = _catalog()
    selection = RouterCandidateSelection(
        candidates=("candidate-a", "candidate-b"),
        incumbent="candidate-a",
    )

    first = plan_automatic_router_cost(
        _tasks(12),
        catalog,
        selection,
        world_model_alias="world",
        judge_alias="judge",
        embedder_alias="embedder",
        judge_response_shape="scalar",
        judge_audit=None,
        provisional_judge=True,
        observed_candidate_aliases=(),
        options=AutomaticRouterOptions(),
    )
    second = plan_automatic_router_cost(
        _tasks(13),
        catalog,
        selection,
        world_model_alias="world",
        judge_alias="judge",
        embedder_alias="embedder",
        judge_response_shape="scalar",
        judge_audit=None,
        provisional_judge=True,
        observed_candidate_aliases=(),
        options=AutomaticRouterOptions(),
    )

    assert first.cost_plan_sha256 != second.cost_plan_sha256


def test_simulation_reservations_use_the_configured_input_bound() -> None:
    """Simulation requests reserve the bounded rendered input, not the whole context."""
    catalog = _catalog()
    resolver = RuntimeModelCatalog(catalog, environment={})
    candidate_model, _capabilities = resolver.snapshot("candidate-a")
    world, _world_capabilities = resolver.snapshot("world")
    problems: list[str] = []

    candidates, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=(RoutedCandidateSnapshot(alias="candidate-a", model=candidate_model),),
        world_alias="world",
        world=world,
        maximum_attempts=3,
        maximum_input_tokens=32_768,
        maximum_output_tokens=16_000,
    )

    assert problems == []
    assert world_request is not None
    assert candidates[0].request.maximum_input_tokens == 32_768
    assert world_request.maximum_input_tokens == 32_768


def test_simulation_reservations_clamp_the_bound_to_remaining_context() -> None:
    """An oversized configured bound never reserves past the model's context capacity."""
    catalog = _catalog()
    resolver = RuntimeModelCatalog(catalog, environment={})
    candidate_model, _capabilities = resolver.snapshot("candidate-a")
    problems: list[str] = []

    candidates, world_request = simulation_completion_reservations(
        problems,
        catalog=catalog,
        candidates=(RoutedCandidateSnapshot(alias="candidate-a", model=candidate_model),),
        world_alias=None,
        world=None,
        maximum_attempts=3,
        maximum_input_tokens=1_000_000,
        maximum_output_tokens=16_000,
    )

    assert problems == []
    assert world_request is None
    assert candidates[0].request.maximum_input_tokens == 128_000 - 16_000


def _tasks(count: int) -> tuple[TaskCase, ...]:
    """Return a deterministic mixed-partition schedule for cost planning.

    Args:
        count: Positive number of representative tasks.

    Returns:
        Exact unique task contracts with distinct leakage lineages.
    """
    return tuple(
        TaskCase(
            task_id=f"task-{index}",
            lineage_group_id=f"lineage-{index}",
            partition="fit" if index < max(1, count - 2) else "held_out",
            instruction=f"Resolve request {index}",
            workload_weight=1.0,
            source_trace_ids=(f"trace-{index}",),
        )
        for index in range(count)
    )

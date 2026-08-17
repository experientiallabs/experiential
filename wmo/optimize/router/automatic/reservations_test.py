"""Tests for automatic router provider reservations."""

from __future__ import annotations

from wmo.common.models import RouterCandidateSelection
from wmo.common.tasks import TaskCase
from wmo.optimize.router.automatic.reservations import (
    AutomaticRouterOptions,
    plan_automatic_router_cost,
)
from wmo.optimize.router.automatic.service_test import _catalog


def test_cost_plan_reserves_exact_small_schedule_without_io() -> None:
    """The pure planner reserves all 24 episodes and 30 judgments without mutation."""
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
        fidelity_overlap_count=6,
        options=AutomaticRouterOptions(),
    )

    assert plan.maximum_judgments == 30
    assert plan.maximum_judge_provider_calls == 30
    assert plan.simulated_episode_count == 24
    assert tuple(item.episode_count for item in plan.candidate_episodes) == (12, 12)
    assert plan.required_provider_cost_usd == (
        plan.router_embedding_cost_usd + plan.judgment_cost_usd + plan.simulation_cost_usd
    )
    assert catalog.model_dump(mode="json") == original


def test_cost_plan_reserves_full_default_corpus_and_pairwise_calls() -> None:
    """The 50/20 corpus reserves 140 episodes and 150 counterbalanced judgments."""
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
        fidelity_overlap_count=10,
        options=AutomaticRouterOptions(),
    )

    assert plan.maximum_judgments == 150
    assert plan.maximum_judge_provider_calls == 300
    assert plan.simulated_episode_count == 140
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
        fidelity_overlap_count=0,
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
        fidelity_overlap_count=0,
        options=AutomaticRouterOptions(),
    )

    assert first.cost_plan_sha256 != second.cost_plan_sha256


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

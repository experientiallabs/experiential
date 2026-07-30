"""Cheap deterministic task-profile routing.

The first profile is intentionally small and auditable: split fit tasks by raw instruction length,
then choose the cheapest arm whose fit quality stays within a configured tolerance of a pinned
baseline. The boundaries and arm choices are persisted in ``RoutingPolicy`` so serving does not
need the outcome matrix or a feature model.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from wmo.optimize.knn import best_single_on_fit
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import RoutingPolicy


def fit_profile_policy(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str] | None = None,
    guard_model: str | None = None,
    quality_tolerance: float = 0.02,
    bins: int = 3,
    fitted_from: str | None = None,
) -> RoutingPolicy:
    """Fit a length-profile policy from measured model outcomes."""
    if quality_tolerance < 0.0:
        raise ValueError("quality_tolerance must be non-negative")
    if bins < 1:
        raise ValueError("bins must be positive")
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(fit_ids) if fit_ids is not None else list(scenario_tasks)
    missing = sorted(set(scenario_ids) - scenario_tasks.keys())
    if missing:
        raise ValueError(f"fit_ids not in the matrix: {missing[:5]}")
    if not scenario_ids:
        raise ValueError("no scenarios to fit")

    baseline = guard_model or best_single_on_fit(matrix, scenario_ids)
    names = [entry.name for entry in matrix.pool]
    pool_order = {name: index for index, name in enumerate(names)}
    lengths = np.asarray([len(scenario_tasks[sid]) for sid in scenario_ids], dtype=np.float64)
    quantiles = np.linspace(1.0 / bins, (bins - 1.0) / bins, bins - 1)
    boundaries = sorted(set(float(value) for value in np.quantile(lengths, quantiles)))
    buckets = np.searchsorted(np.asarray(boundaries), lengths, side="right")

    cell_rewards: dict[tuple[str, str], list[float]] = defaultdict(list)
    cell_costs: dict[tuple[str, str], list[float]] = defaultdict(list)
    wanted = set(scenario_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in wanted or outcome.reward is None:
            continue
        key = (outcome.scenario_id, outcome.model)
        cell_rewards[key].append(float(outcome.reward))
        cell_costs[key].append(float(outcome.cost_usd))

    profile_models: list[str] = []
    for bucket in range(len(boundaries) + 1):
        members = [scenario_ids[index] for index, value in enumerate(buckets) if value == bucket]
        if not members:
            profile_models.append(baseline)
            continue
        means: dict[str, tuple[float, float]] = {}
        for name in names:
            rewards = [
                np.mean(cell_rewards[(sid, name)])
                for sid in members
                if (sid, name) in cell_rewards
            ]
            costs = [
                np.mean(cell_costs[(sid, name)])
                for sid in members
                if (sid, name) in cell_costs
            ]
            if rewards and costs:
                means[name] = (float(np.mean(rewards)), float(np.mean(costs)))
        guard_quality, guard_cost = means[baseline]
        eligible = [
            (cost, -quality, pool_order[name], name)
            for name, (quality, cost) in means.items()
            if cost < guard_cost and quality >= guard_quality - quality_tolerance
        ]
        profile_models.append(min(eligible)[3] if eligible else baseline)

    return RoutingPolicy(
        kind="profile",
        default_model=baseline,
        pool=matrix.pool,
        guard_model=baseline,
        profile_bins=boundaries,
        profile_models=profile_models,
        fitted_from=fitted_from,
        fit_scenario_ids=scenario_ids,
    )

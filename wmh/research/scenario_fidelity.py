"""Test 2 of scenario-set verification: does the small set predict full-distribution scores?

The claim behind a representative K-scenario set is that agent scores on the K scenarios predict
scores on the full task pool. The world model makes the required score matrix cheap: roll every
agent configuration over the full pool (mean of `passes` rollouts per cell, never single-pass),
then compare each selection method's predicted score (weighted mean over its subset) against the
actual full-pool score, per agent — MAE for calibration, Spearman/Kendall over agent rankings for
ordering. The benchmark-compression literature (Fluid Benchmarking, EssenceBench) reports exactly
these numbers, so results are literature-comparable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel, Field

from wmh.engine.world_model import WorldModel
from wmh.env.base import WorldModelEnv
from wmh.env.episode import Agent, run_episode
from wmh.scenarios.synthesis import EvalScenario
from wmh.scenarios.verification import ChecklistJudge

# Every reported metric is a mean over this many independent rollouts per (agent, scenario) cell.
DEFAULT_PASSES = 3


class ScoreMatrix(BaseModel):
    """Mean rollout score per (agent config, scenario), plus the pass count behind each mean."""

    passes: int
    scores: dict[str, dict[str, float]]  # agent name -> scenario_id -> mean score in 0..1


class MethodFidelity(BaseModel):
    """How well one selection method's subset predicts full-pool agent scores."""

    method: str
    subset_size: int
    mae: float  # mean |predicted - actual| over agent configs
    spearman: float  # rank correlation of agent orderings (subset vs full pool)
    kendall: float
    predicted: dict[str, float]  # agent name -> weighted subset score
    actual: dict[str, float]  # agent name -> uniform full-pool score


class FidelityReport(BaseModel):
    """Test-2 result: one `MethodFidelity` per selection method, over a shared score matrix."""

    pool_size: int
    passes: int
    methods: list[MethodFidelity] = Field(default_factory=list)


def score_matrix(
    world_model: WorldModel,
    agents: dict[str, Agent],
    pool: list[EvalScenario],
    judge: ChecklistJudge,
    *,
    passes: int = DEFAULT_PASSES,
    max_steps: int = 8,
    workers: int = 4,
) -> ScoreMatrix:
    """Roll every agent over every pool scenario `passes` times; cell = mean checklist pass rate.

    Rollouts run on a small thread pool (provider calls are I/O bound); each rollout opens its own
    world-model session so episodes never share state, and the world model is frozen for the whole
    matrix so no cell's generated steps become another cell's retrieved demos.
    """

    def one_rollout(agent: Agent, scenario: EvalScenario) -> float:
        env = WorldModelEnv(world_model)
        episode = run_episode(
            env, agent, scenario.task, seed_state=scenario.seed_state, max_steps=max_steps
        )
        return judge.score(scenario.task, scenario.checklist, episode.steps).pass_rate

    cells = [
        (agent_name, agent, scenario)
        for agent_name, agent in agents.items()
        for scenario in pool
        for _ in range(passes)
    ]
    with world_model.frozen(), ThreadPoolExecutor(max_workers=workers) as pool_executor:
        results = list(pool_executor.map(lambda cell: one_rollout(cell[1], cell[2]), cells))

    sums: dict[str, dict[str, float]] = {name: {} for name in agents}
    for (agent_name, _agent, scenario), score in zip(cells, results, strict=True):
        sums[agent_name][scenario.scenario_id] = (
            sums[agent_name].get(scenario.scenario_id, 0.0) + score
        )
    means = {
        name: {sid: total / passes for sid, total in by_scenario.items()}
        for name, by_scenario in sums.items()
    }
    return ScoreMatrix(passes=passes, scores=means)


def fidelity_report(
    matrix: ScoreMatrix,
    pool_ids: list[str],
    subsets: dict[str, dict[str, float]],
) -> FidelityReport:
    """Compare each selection method against the full pool on a shared score matrix.

    `subsets` maps method name -> {scenario_id: weight}; weights are normalized internally, so
    uniform baselines can pass weight 1.0 per member. Every subset id must be in `pool_ids`.
    """
    actual = {
        agent: float(np.mean([scores[sid] for sid in pool_ids]))
        for agent, scores in matrix.scores.items()
    }
    report = FidelityReport(pool_size=len(pool_ids), passes=matrix.passes)
    for method, weights in subsets.items():
        missing = set(weights) - set(pool_ids)
        if missing:
            raise ValueError(f"subset {method!r} contains ids outside the pool: {sorted(missing)}")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError(f"subset {method!r} has no positive weight")
        predicted = {
            agent: float(sum(scores[sid] * w for sid, w in weights.items()) / total_weight)
            for agent, scores in matrix.scores.items()
        }
        agent_names = sorted(actual)
        predicted_vector = np.asarray([predicted[a] for a in agent_names])
        actual_vector = np.asarray([actual[a] for a in agent_names])
        report.methods.append(
            MethodFidelity(
                method=method,
                subset_size=len(weights),
                mae=float(np.abs(predicted_vector - actual_vector).mean()),
                spearman=spearman(predicted_vector, actual_vector),
                kendall=kendall(predicted_vector, actual_vector),
                predicted=predicted,
                actual=actual,
            )
        )
    return report


def random_subsets(
    pool_ids: list[str], k: int, *, seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[str, dict[str, float]]:
    """Uniform random-K baselines (one per seed), weights uniform over the subset."""
    subsets: dict[str, dict[str, float]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(pool_ids), size=min(k, len(pool_ids)), replace=False)
        subsets[f"random-k{k}-seed{seed}"] = {pool_ids[i]: 1.0 for i in chosen}
    return subsets


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); 0.0 when either side is constant."""
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    std_a, std_b = ranks_a.std(), ranks_b.std()
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b over all pairs; 0.0 when either side is constant."""
    n = len(a)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) / 2
    denominator = np.sqrt((pairs - ties_a) * (pairs - ties_b))
    if denominator == 0.0:
        return 0.0
    return float((concordant - discordant) / denominator)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ranks across ties so constant vectors rank identically.
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks

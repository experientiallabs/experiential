"""The routing fitter: an OutcomeMatrix in, an Avengers-style rank policy out.

Faithful replication of the reference implementation (arXiv 2505.19797,
github.com/ZhangYiqun018/Avengers, core/generate_rank_router.py), stage by stage:

1. Embed the fit scenarios' task texts and L2-normalize (their `Normalizer(norm="l2")`).
2. K-means the normalized embeddings (their exact configuration: k-means++, n_init="auto",
   max_iter=1000, elkan, seeded).
3. Per cluster, rank the pool models by mean reward over the cluster's SCORED episodes
   (their correct/total accuracy, generalized to graded rewards; identical on binary data).
   Models with no scored episode in a cluster are absent from its ranking and fall back to
   `default_rank` at selection time, matching the reference.

Deliberate deltas from the reference, both recorded here so the post-hoc comparison audit has
the list in one place: (a) rewards may be graded, not just 0/1; (b) clusters get a
human-readable label (majority scenario-id prefix) for the request log; the reference has no
labels. Cost plays NO part in fitting, exactly like the reference; the cost-aware variant
(Avengers-Pro's alpha) is the first planned variation AFTER replication is validated.

`evaluate_policy` replays a policy over a matrix through the same `rank_decision` scoring code
serving uses, so benchmark numbers measure the deployed selection path, not a reimplementation.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from wmh.optimize.cluster_labels import label_clusters
from wmh.optimize.policy import (
    DEFAULT_BETA,
    DEFAULT_RANK,
    DEFAULT_TOP_K_CLUSTERS,
    ClusterRanking,
    EmbedderSpec,
    RoutingDecision,
    RoutingPolicy,
    rank_decision,
)

if TYPE_CHECKING:
    from wmh.optimize.outcomes import OutcomeMatrix

logger = logging.getLogger(__name__)


def fit_rank_policy(
    matrix: OutcomeMatrix,
    *,
    fit_ids: list[str] | None = None,
    embedder: EmbedderSpec | None = None,
    n_clusters: int = 64,
    seed: int = 42,
    top_k_clusters: int = DEFAULT_TOP_K_CLUSTERS,
    beta: float = DEFAULT_BETA,
    default_rank: int = DEFAULT_RANK,
    default_model: str | None = None,
    guard_model: str | None = None,
    min_support: int = 0,
    guard_margin: float = 0.0,
    fitted_from: str | None = None,
) -> RoutingPolicy:
    """Fit a rank policy on `matrix` (restricted to `fit_ids` when given).

    Defaults mirror the reference (n_clusters=64, seed=42, top_k=2, beta=6.0).
    `default_model` falls back to the best overall mean reward on the fit scenarios
    (ties break by pool order).
    """
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    if fit_ids is not None:
        wanted = set(fit_ids)
        missing = wanted - scenario_tasks.keys()
        if missing:
            raise ValueError(f"fit_ids not in the matrix: {sorted(missing)[:5]}")
        scenario_tasks = {sid: scenario_tasks[sid] for sid in fit_ids}
    if not scenario_tasks:
        raise ValueError("no scenarios to fit on")

    spec = embedder or EmbedderSpec()
    scenario_ids = list(scenario_tasks)
    embeddings = np.asarray(spec.build().embed([scenario_tasks[sid] for sid in scenario_ids]))
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)

    k = min(n_clusters, len(scenario_ids))
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        init="k-means++",
        n_init="auto",
        max_iter=1000,
        algorithm="elkan",
    )
    labels = kmeans.fit_predict(embeddings)
    logger.info(
        "k-means: %d clusters over %d scenarios (inertia %.4f)",
        k,
        len(scenario_ids),
        kmeans.inertia_,
    )

    # Per-cluster, per-model reward sums over SCORED episodes only.
    cluster_of = {sid: int(label) for sid, label in zip(scenario_ids, labels, strict=True)}
    sums: dict[int, dict[str, tuple[float, float, int]]] = {c: {} for c in range(k)}
    counts: Counter[int] = Counter(cluster_of.values())
    prefix_counts: dict[int, Counter[str]] = {c: Counter() for c in range(k)}
    member_texts: dict[int, list[str]] = {c: [] for c in range(k)}
    for sid, cluster in cluster_of.items():
        member_texts[cluster].append(scenario_tasks[sid])
        if ":" in sid:
            prefix_counts[cluster][sid.split(":", 1)[0]] += 1
    # Fallback labels for prefix-less ids (wm corpora use raw trace hashes): distinctive
    # c-TF-IDF terms of each cluster's task texts. Labels never affect selection.
    text_labels = label_clusters([member_texts[c] for c in range(k)])
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    total_cost = 0.0
    total_count = 0
    for outcome in matrix.outcomes:
        cluster = cluster_of.get(outcome.scenario_id)
        if cluster is None or outcome.reward is None:
            continue
        reward_sum, cost_sum, count = sums[cluster].get(outcome.model, (0.0, 0.0, 0))
        sums[cluster][outcome.model] = (
            reward_sum + outcome.reward,
            cost_sum + outcome.cost_usd,
            count + 1,
        )
        total_cost += outcome.cost_usd
        total_count += 1

    clusters: list[ClusterRanking] = []
    for cluster in range(k):
        means = {
            model: reward_sum / count
            for model, (reward_sum, _cost_sum, count) in sums[cluster].items()
        }
        mean_costs = {
            model: cost_sum / count
            for model, (_reward_sum, cost_sum, count) in sums[cluster].items()
        }
        if not means:
            # A cluster with no scored episodes ranks nothing; selection falls through to
            # default_rank scores. Logged, never silent.
            logger.warning("cluster %d has no scored episodes; it ranks no models", cluster)
        ranking = sorted(means, key=lambda m: (-means[m], pool_order[m]))
        if guard_model is not None and ranking:
            # Only-replace-if-better, per cluster: the router's worst case must be the
            # baseline model, so a cluster keeps its own winner ONLY when that winner beats
            # the baseline's in-cluster evidence with enough support; otherwise the baseline
            # leads. Thin clusters (support < min_support) always revert.
            top = ranking[0]
            support = sums[cluster].get(top, (0.0, 0.0, 0))[2]
            baseline_mean = means.get(guard_model, 0.0)
            # Margin: the cluster winner must beat the baseline's evidence by guard_margin
            # (doubled when the winner is pricier), or the cluster reverts. Kills the
            # confidently-wrong pricier-and-worse failure mode.
            margin = guard_margin
            if mean_costs.get(top, 0.0) > mean_costs.get(guard_model, float("inf")):
                margin = 2 * guard_margin
            if top != guard_model and (
                support < min_support or means[top] <= baseline_mean + margin
            ):
                ranking = [guard_model, *[m for m in ranking if m != guard_model]]
        label = text_labels[cluster]
        if prefix_counts[cluster]:
            label = prefix_counts[cluster].most_common(1)[0][0]
        clusters.append(
            ClusterRanking(
                cluster_id=cluster,
                label=label,
                centroid=[float(v) for v in kmeans.cluster_centers_[cluster]],
                ranking=ranking or [_overall_best(matrix, set(scenario_ids))],
                scores={model: round(mean, 6) for model, mean in means.items()},
                costs={model: round(mean, 8) for model, mean in mean_costs.items()},
                total=counts[cluster],
            )
        )

    chosen_default = default_model or _overall_best(matrix, set(scenario_ids))
    return RoutingPolicy(
        kind="rank",
        default_model=chosen_default,
        pool=matrix.pool,
        embedder=spec,
        clusters=clusters,
        top_k_clusters=top_k_clusters,
        beta=beta,
        default_rank=default_rank,
        cost_scale=(total_cost / total_count) if total_count else 0.0,
        fitted_from=fitted_from,
    )


def _overall_best(matrix: OutcomeMatrix, ids: set[str]) -> str:
    sums: dict[str, tuple[float, int]] = {}
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in ids or outcome.reward is None:
            continue
        reward_sum, count = sums.get(outcome.model, (0.0, 0))
        sums[outcome.model] = (reward_sum + outcome.reward, count + 1)
    if not sums:
        raise ValueError("no scored outcomes; cannot pick a default model")
    pool_order = {entry.name: index for index, entry in enumerate(matrix.pool)}
    return min(sums, key=lambda m: (-(sums[m][0] / sums[m][1]), pool_order[m]))


def rerank_policy(policy: RoutingPolicy, *, cost_weight: float) -> RoutingPolicy:
    """Re-rank a fitted policy's clusters under a cost weight, WITHOUT refitting.

    The fit-once, slide-the-knob property (Hybrid LLM, arXiv 2404.14618): every cluster keeps
    its stored reward/cost evidence, and the ranking key becomes
    `mean_reward - cost_weight * mean_cost / cost_scale` (cost in fit-set-average-call units,
    so cost_weight trades one reward point against one average call). `cost_weight=0` returns
    the policy unchanged - the faithful Avengers behavior; the reference has no cost term
    (Avengers-Pro's alpha is the published analogue).
    """
    if cost_weight == 0.0:
        return policy
    if policy.kind != "rank":
        raise ValueError("only rank policies can be re-ranked")
    if policy.cost_scale <= 0.0:
        raise ValueError("policy has no cost_scale; refit with cost evidence to use the knob")
    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    clusters = []
    for cluster in policy.clusters:
        keyed = {
            model: cluster.scores.get(model, 0.0)
            - cost_weight * cluster.costs.get(model, 0.0) / policy.cost_scale
            for model in cluster.ranking
        }
        ranking = sorted(keyed, key=lambda m: (-keyed[m], pool_order[m]))
        clusters.append(cluster.model_copy(update={"ranking": ranking}))
    provenance = f"{policy.fitted_from or 'unknown'} | cost_weight={cost_weight:g}"
    return policy.model_copy(update={"clusters": clusters, "fitted_from": provenance})


class PolicyEval(BaseModel):
    """One policy replayed over a matrix: what the benchmark tables are made of."""

    accuracy: float
    cost_per_scenario: float
    model_mix: dict[str, float]
    scenarios: int
    unscored_scenarios: int  # scenario/model pairs the routed choice had no scored row for


def evaluate_policy(policy: RoutingPolicy, matrix: OutcomeMatrix, ids: list[str]) -> PolicyEval:
    """Replay `policy` over the scenarios in `ids`, scoring via each routed model's rows.

    Selection runs through the SAME `rank_decision` code serving uses (queries are batch
    embedded once for speed; the scoring math is shared, not reimplemented).
    """
    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)
    wanted = [sid for sid in ids if sid in scenario_tasks]
    if not wanted:
        raise ValueError("none of the requested ids are in the matrix")

    decisions: dict[str, RoutingDecision]
    if policy.kind == "static":
        decisions = {
            sid: RoutingDecision(model=policy.default_model, reason="static policy")
            for sid in wanted
        }
    else:
        embeddings = np.asarray(
            policy.embedder.build().embed([scenario_tasks[sid] for sid in wanted])
        )
        embeddings = Normalizer(norm="l2").transform(embeddings)
        decisions = {
            sid: rank_decision(policy, embeddings[index]) for index, sid in enumerate(wanted)
        }

    by_scenario_model: dict[tuple[str, str], list[float]] = {}
    costs: dict[tuple[str, str], list[float]] = {}
    for outcome in matrix.outcomes:
        key = (outcome.scenario_id, outcome.model)
        if outcome.reward is not None:
            by_scenario_model.setdefault(key, []).append(outcome.reward)
            costs.setdefault(key, []).append(outcome.cost_usd)

    rewards: list[float] = []
    cost_values: list[float] = []
    mix: Counter[str] = Counter()
    unscored = 0
    for sid in wanted:
        model = decisions[sid].model
        mix[model] += 1
        key = (sid, model)
        if key not in by_scenario_model:
            unscored += 1
            continue
        rewards.append(sum(by_scenario_model[key]) / len(by_scenario_model[key]))
        cost_values.append(sum(costs[key]) / len(costs[key]))
    if not rewards:
        raise ValueError("no scored outcomes for any routed choice; nothing to evaluate")
    return PolicyEval(
        accuracy=sum(rewards) / len(rewards),
        cost_per_scenario=sum(cost_values) / len(cost_values),
        model_mix={model: count / len(wanted) for model, count in sorted(mix.items())},
        scenarios=len(wanted),
        unscored_scenarios=unscored,
    )

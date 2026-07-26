"""The improvement report (D-REPORT): the endpoint's evidence artifact.

Pure aggregation over an `OutcomeMatrix` given a `RoutingPolicy`: what the policy's routed mix
scores on the held-out scenarios versus one frontier baseline model, at what cost and latency,
and which models the mix uses. No fitting happens here; any policy (hand-written, static, or a
future fitted one) reports through the same function, so the artifact shape is stable before
the optimizer exists.

Numbers honesty (binding, endpoint-common.md): accuracy is verifier-scored task success on
held-out scenarios and is labeled as such; cost and latency are real measurements from the eval
episodes; `cost_assumptions` states what the cost numbers do and do not include (today:
single-shot eval, list prices, cache effects not yet modeled). Unscored episodes are counted
and surfaced, never averaged in as zeros.

Paired comparison (why the headline is not a plain per-side mean): skipping unscored episodes
per side would let the two sides average DIFFERENT scenario subsets, and unscored is not random
- a candidate that times out on the hardest scenarios gets graded on the easy remainder and can
out-score a baseline that answered everything. So the headline aggregates both sides over the
intersection of scenarios scored on BOTH sides, and reports how many scenarios that left in
(`scenarios_compared`) and out (`scenarios_excluded`). With an empty intersection there is
nothing to report and `build_report` raises.
"""

from __future__ import annotations

from statistics import median, quantiles

from pydantic import BaseModel, Field

from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import RoutingPolicy, select_model
from wmh.providers.pool import Tier

# What the v1 cost numbers assume; shipped verbatim in every report until the cache-aware cost
# model replaces it (2026-07-23 caching entry in the coordination log).
COST_ASSUMPTIONS_V1 = (
    "Costs are measured candidate-side per eval episode at the pool's per-token prices "
    "(single-shot; provider prompt-cache effects on multi-turn traffic not yet modeled)."
)


class ModelRef(BaseModel):
    model_id: str  # pool entry name
    label: str
    tier: Tier


class Headline(BaseModel):
    """The endpoint's big numbers: routed policy vs the frontier baseline, same scenarios.

    "Same scenarios" is literal and enforced: every number here is measured over the
    `scenarios_compared` scenarios that BOTH sides scored (see the module docstring).
    """

    accuracy: float
    baseline_accuracy: float
    cost_per_run_usd: float
    baseline_cost_per_run_usd: float
    latency_p50_ms: float
    baseline_latency_p50_ms: float
    latency_p95_ms: float
    baseline_latency_p95_ms: float
    scenarios_compared: int = 0  # scenarios scored on BOTH sides: what the numbers cover
    scenarios_excluded: int = 0  # held out of the comparison because one side went unscored


class CandidateResult(BaseModel):
    """One pool model's closed-loop row (the report's per-model evidence table)."""

    model: ModelRef
    accuracy: float
    success_rate: float
    cost_per_run_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    scored_episodes: int
    unscored_episodes: int


class MixShare(BaseModel):
    model_id: str
    share: float  # fraction of held-out scenarios the policy routes to this model


class ImprovementReport(BaseModel):
    endpoint_id: str
    generated_at: str
    scenario_count: int
    scenario_label: str  # e.g. "on 90 held-out scenarios reconstructed from your traces"
    baseline: ModelRef
    headline: Headline
    candidates: list[CandidateResult]
    model_mix: list[MixShare]
    cost_assumptions: str = Field(min_length=1)


def _p50_p95_ms(seconds: list[float]) -> tuple[float, float]:
    if not seconds:
        return 0.0, 0.0
    p50 = median(seconds) * 1000
    p95 = (quantiles(seconds, n=20)[-1] if len(seconds) > 1 else seconds[0]) * 1000
    return p50, p95


def _mean(values: list[float]) -> float:
    # 0.0 on empty is safe ONLY in the per-candidate table, where `scored_episodes` sits beside
    # the number and says it rests on nothing. The headline never takes this path: `build_report`
    # raises rather than quote a mean over zero commonly-scored scenarios.
    return sum(values) / len(values) if values else 0.0


def _aggregate(outcomes: list[ScenarioOutcome]) -> tuple[float, float, float, float, float]:
    """(accuracy, success_rate, cost_per_run, p50_ms, p95_ms) over SCORED episodes."""
    scored = [o for o in outcomes if o.reward is not None]
    rewards = [o.reward for o in scored if o.reward is not None]
    accuracy = _mean(rewards)
    success = _mean([1.0 if o.success else 0.0 for o in scored])
    cost = _mean([o.cost_usd for o in scored])
    calls = [s for o in scored for s in o.call_seconds]
    p50, p95 = _p50_p95_ms(calls)
    return accuracy, success, cost, p50, p95


def build_report(
    matrix: OutcomeMatrix,
    policy: RoutingPolicy,
    *,
    baseline: str,
    endpoint: str,
    generated_at: str,
) -> ImprovementReport:
    """Aggregate `matrix` into the endpoint's improvement report under `policy`.

    The routed side replays the policy's serve-time selection over each held-out scenario's
    task text and takes THAT model's measured outcomes for the scenario; the baseline side is
    `baseline`'s own rows on the same scenarios. Both sides therefore quote real, per-scenario
    measurements from the identical matrix, over the identical scenarios (module docstring:
    the headline is a PAIRED comparison over commonly-scored scenarios).

    Raises ValueError when no scenario was scored on both sides: a matrix that measured nothing
    comparable has no honest report in it.
    """
    names = {entry.name: entry for entry in matrix.pool}
    if baseline not in names:
        raise KeyError(f"baseline '{baseline}' is not in the matrix pool; have: {sorted(names)}")

    scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        scenario_tasks.setdefault(outcome.scenario_id, outcome.task)

    # One embedder for the whole report: an azure spec builds an HTTP client per `build()`, and
    # a report routes every held-out scenario.
    embedder = policy.embedder.build() if policy.kind != "static" else None

    routed_rows: dict[str, list[ScenarioOutcome]] = {}
    baseline_rows: dict[str, list[ScenarioOutcome]] = {}
    assignment_counts: dict[str, int] = {}
    for scenario_id, task in scenario_tasks.items():
        decision = select_model(policy, task, embedder=embedder)
        assignment_counts[decision.model] = assignment_counts.get(decision.model, 0) + 1
        rows = matrix.for_scenario(scenario_id)
        routed_rows[scenario_id] = [o for o in rows if o.model == decision.model]
        baseline_rows[scenario_id] = [o for o in rows if o.model == baseline]

    compared = [
        scenario_id
        for scenario_id in scenario_tasks
        if any(o.reward is not None for o in routed_rows[scenario_id])
        and any(o.reward is not None for o in baseline_rows[scenario_id])
    ]
    if not compared:
        raise ValueError(
            f"no scenario has a scored episode on BOTH sides (routed policy and baseline "
            f"'{baseline}'), so there is nothing to compare over "
            f"{len(scenario_tasks)} scenarios; check the matrix for unscored episodes"
        )

    routed_acc, _, routed_cost, routed_p50, routed_p95 = _aggregate(
        [o for scenario_id in compared for o in routed_rows[scenario_id]]
    )
    base_acc, _, base_cost, base_p50, base_p95 = _aggregate(
        [o for scenario_id in compared for o in baseline_rows[scenario_id]]
    )

    def _ref(name: str) -> ModelRef:
        entry = names[name]
        return ModelRef(model_id=entry.name, label=entry.model, tier=entry.tier)

    candidates: list[CandidateResult] = []
    for entry in matrix.pool:
        rows = [o for o in matrix.outcomes if o.model == entry.name]
        accuracy, success, cost, p50, p95 = _aggregate(rows)
        candidates.append(
            CandidateResult(
                model=_ref(entry.name),
                accuracy=accuracy,
                success_rate=success,
                cost_per_run_usd=cost,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                scored_episodes=sum(1 for o in rows if o.reward is not None),
                unscored_episodes=sum(1 for o in rows if o.reward is None),
            )
        )

    total = len(scenario_tasks)
    return ImprovementReport(
        endpoint_id=endpoint,
        generated_at=generated_at,
        scenario_count=total,
        scenario_label=f"on {total} held-out scenarios reconstructed from your traces",
        baseline=_ref(baseline),
        headline=Headline(
            accuracy=routed_acc,
            baseline_accuracy=base_acc,
            cost_per_run_usd=routed_cost,
            baseline_cost_per_run_usd=base_cost,
            latency_p50_ms=routed_p50,
            baseline_latency_p50_ms=base_p50,
            latency_p95_ms=routed_p95,
            baseline_latency_p95_ms=base_p95,
            scenarios_compared=len(compared),
            scenarios_excluded=total - len(compared),
        ),
        candidates=candidates,
        model_mix=[
            MixShare(model_id=model, share=count / total)
            for model, count in sorted(assignment_counts.items())
        ],
        cost_assumptions=COST_ASSUMPTIONS_V1,
    )

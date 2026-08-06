"""The improvement report (D-REPORT): the endpoint's evidence artifact.

Pure aggregation over an `OutcomeMatrix` given a `RoutingPolicy`: what the policy's routed mix
scores on router-held-out scenarios versus one frontier baseline model, at what cost and latency,
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

from wmo.optimize.routing.compression import CompressingEmbedder
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import RoutingPolicy, select_model
from wmo.providers.base import Embedder
from wmo.providers.pool import Tier

# What the v1 cost numbers assume; shipped verbatim in every report until the cache-aware cost
# model replaces it (2026-07-23 caching entry in the coordination log).
COST_ASSUMPTIONS_V1 = (
    "Costs are measured candidate-side per eval episode at the pool's per-token prices "
    "(single-shot; provider prompt-cache effects on multi-turn traffic not yet modeled)."
)


class ModelRef(BaseModel):
    """A pool model as the report names it."""

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
    """How much of the held-out traffic the routed policy sends to one model."""

    model_id: str
    share: float  # fraction of held-out scenarios the policy routes to this model


class ImprovementReport(BaseModel):
    """The endpoint's routing report: headline comparison, per-model evidence, and the mix."""

    endpoint_id: str
    generated_at: str
    scenario_count: int
    scenario_ids: list[str] = Field(default_factory=list)
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
    p50, p95 = _p50_p95_ms(_productive_call_seconds(scored))
    return accuracy, success, cost, p50, p95


def _productive_call_seconds(scored: list[ScenarioOutcome]) -> list[float]:
    """Per-call seconds, skipping calls the provider answered with blank text.

    `wmo.optimize.routing.evaluation._TimedProvider` appends `call_seconds` and `replies` in
    lockstep per provider call, so there a blank reply is identifiable by index. Those calls
    return fast and carry no action. `LLMAgent` buys another completion to replace them, so
    counting them would report latency at which the endpoint never delivered work. A model that
    blanks often would look faster than one that answers first time. Cost is left alone because
    the blank attempts were really paid for.

    Other producers do not promise that pairing (the real-episode runner records one reply per
    message that HAS content but one duration per timed call), so equal lengths are required
    before attributing a duration to a reply; otherwise every call counts.
    """
    seconds: list[float] = []
    for outcome in scored:
        if len(outcome.replies) != len(outcome.call_seconds):
            seconds.extend(outcome.call_seconds)
            continue
        paired = zip(outcome.call_seconds, outcome.replies, strict=True)
        seconds.extend(value for value, reply in paired if reply.strip())
    return seconds


def build_report(
    matrix: OutcomeMatrix,
    policy: RoutingPolicy,
    *,
    baseline: str,
    endpoint: str,
    generated_at: str,
    scenario_label: str | None = None,
    built: Embedder | None = None,
) -> ImprovementReport:
    """Aggregate `matrix` into the endpoint's improvement report under `policy`.

    The routed side replays the policy's serve-time selection over each router-held-out scenario's
    task text and takes THAT model's measured outcomes for the scenario; the baseline side is
    `baseline`'s own rows on the same scenarios. Both sides therefore quote real, per-scenario
    measurements from the identical matrix, over the identical scenarios (module docstring:
    the headline is a PAIRED comparison over commonly-scored scenarios).

    Fit scenarios recorded by the policy are excluded. An entirely separate evaluation matrix
    has no overlapping ids, so all of its scenarios remain eligible. Raises ValueError when no
    eligible scenario was scored on both sides: a matrix that measured nothing comparable has no
    honest report in it.

    `scenario_label` is the customer-facing sentence describing WHAT was measured, and it defaults
    to the world-model phrasing this report was written for. A matrix of real benchmark episodes
    must pass its own: telling a customer their endpoint was measured on scenarios "reconstructed
    from your traces" when it was measured on a pinned public benchmark is false on the one line
    of the report a reader actually reads.
    """
    names = {entry.name: entry for entry in matrix.pool}
    if baseline not in names:
        raise KeyError(f"baseline '{baseline}' is not in the matrix pool; have: {sorted(names)}")

    all_scenario_tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        all_scenario_tasks.setdefault(outcome.scenario_id, outcome.task)

    fit_ids = set(policy.fit_scenario_ids)
    if not fit_ids and policy.kind == "knn":
        # Legacy kNN artifacts did not serialize fit ids on the policy, but their evidence bank
        # does. Recover the boundary rather than silently calling those rows held out.
        fit_ids = set(policy.knn_bank().scenario_ids)
    if policy.kind == "rank" and not fit_ids:
        raise ValueError(
            "the fitted rank policy does not record its fit scenario ids, so held-out reporting "
            "cannot be proved; refit the policy with this version"
        )
    report_ids = [sid for sid in all_scenario_tasks if sid not in fit_ids]
    if not report_ids:
        raise ValueError(
            "the report matrix contains no scenario outside the router fit set; use a matrix "
            "with the reserved report partition or a separate evaluation matrix"
        )
    scenario_tasks = {sid: all_scenario_tasks[sid] for sid in report_ids}

    # One embedder for the whole report: an azure spec builds an HTTP client per `build()`, and
    # a report routes every held-out scenario.
    # `built` substitutes the embedding computation only (a reproduction serving recorded
    # vectors); the policy's spec remains the recorded identity. See `fit_knn_artifact`.
    embedder = (
        built
        if built is not None
        else (policy.embedder.build() if policy.kind != "static" else None)
    )
    if embedder is not None and policy.compression is not None:
        # Representation consistency, on the reporting side. A compressed endpoint's bank lives in
        # the geometry of compressed text, and serving compresses each request before the router
        # embeds it. Replaying the selection on RAW task text would therefore measure a policy
        # nobody serves: every query lands farther from every bank row, the novelty floor trips,
        # and the report would show routing collapsing to the fallback (C2 measured the floor
        # tripping 10-13x more often under exactly this mismatch). So the replay embeds through the
        # same compressor the fit did.
        embedder = CompressingEmbedder(embedder, policy.compression)

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
        rows = [
            o for o in matrix.outcomes if o.model == entry.name and o.scenario_id in scenario_tasks
        ]
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
        scenario_ids=list(scenario_tasks),
        scenario_label=scenario_label
        or f"on {total} held-out scenarios reconstructed from your traces",
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

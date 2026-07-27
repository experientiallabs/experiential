"""The three-objective scorecard and the ablation ladder it reports through.

The accounting rule (D-COMPRESS, binding for every savings claim this project makes): savings
are **cache-adjusted effective cost per completed task, compressor inference cost and latency
included**. The rule was written for the compression track, but it binds any optimizer that
buys cheaper tokens by spending inference somewhere else, so this module generalizes its
"compressor" to an *optimizer overhead*: cost and wall time an arm incurs OUTSIDE the worker
model's own bill (a compactor pass, a router's embedding call). Nothing aggregated the rule
before this module existed; report.py's `Headline` reports mean cost per RUN against one
baseline model, which is a different (and, for an ablation, misleading) quantity: an arm that
fails half its tasks looks cheap per run and is ruinous per completed task.

Three objectives, never one:

1. Quality: mean reward plus task success rate over scored episodes.
2. Effective cost per completed task: cache-adjusted provider spend plus optimizer overhead,
   divided by the number of tasks actually completed.
3. Latency: p50 and p95 of per-task wall seconds, optimizer overhead included.

Discipline inherited from the surrounding code, and where this module deliberately differs:

- Unscored episodes (`reward is None`) are an infrastructure failure, not a judge verdict of 0.
  They are excluded from every numerator and denominator, counted, and their unattributed spend
  is reported separately so the money never vanishes silently.
- "Same scenarios" is literal and enforced, as in `report.py`: an arm and its anchor are
  compared over the intersection of scenarios scored on BOTH sides, and the counts in and out
  are on the artifact. A LADDER goes further and holds every rung to one common scenario set,
  because its Pareto front compares rungs against each other and not only against the anchor.
- Comparability invariants fail loudly rather than merge, as in `wmo.evals.grid.merge_results`.
  A `wm_simulated` arm never silently scores against a `real_episode` anchor, and two arms whose
  condition labels collide are rejected: the GEPA program lost runs to colliding labels, so a
  `ConditionLabel` here is structured and must name every experimental axis.
- Costs are consumed, never re-derived. `ScenarioOutcome.cost_usd` was priced at record time by
  `PoolEntry.cost_usd`, which bills cache reads and cache writes at their own tiers; that is
  where "cache-adjusted" comes from and re-pricing here would drop negotiated rates.
- Every estimate names its basis, as in `wmo.serving.savings`: `cost_assumptions` is composed
  from what the rows actually contained and is mandatory non-empty.
- Latency is per TASK (seconds), not per call as in `report.py`. Cost per completed task and
  seconds per task are the pair an operator reasons about; a per-call p95 cannot be compared
  against a per-task cost.

Call site (the joint tau-bench ablation ladder):

    tau = ConditionLabel(
        dataset="tau-bench-retail", split="test", judge="tau2-verifier",
        provenance="real_episode", base_model="qwen3-8b", optimizer="none",
    )
    ladder = build_ladder(
        "joint-tau",
        anchor=Arm(name="teacher", condition=tau.replace(base_model="glm-5.2"),
                   rows=rows_for_model(matrix, "glm-5.2")),
        arms=[distill_only, plus_routing, plus_compaction],
    )
    front = {
        rung.scorecard.arm: rung.scorecard.cost.cost_per_completed_task_usd
        for rung in ladder.pareto()
    }
"""

from __future__ import annotations

from statistics import median, quantiles
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.optimize.knn import CostQualityAnchor
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence

# Whether an arm's numbers came from episodes against a world-model simulation or against the
# real environment. Mixing the two in one comparison is the single easiest way to publish a
# number nobody can reproduce, so it is a required axis and an enforced invariant.
Provenance = Literal["wm_simulated", "real_episode"]

# Which latency statistic the Pareto front dominates on. p95 is reported on every scorecard but
# is not the dominance coordinate by default: at ablation sample sizes it is one slow episode.
LatencyObjective = Literal["p50", "p95"]

# The rule, shipped verbatim on every result. Specifics measured from the rows are appended.
EFFECTIVE_COST_RULE = (
    "Effective cost is cache-adjusted provider spend plus optimizer-side inference, divided by "
    "the number of COMPLETED tasks. Provider spend is the amount recorded per episode at the "
    "serving pool's own per-token prices, which bill cache reads and cache writes at their "
    "separate tiers. Unscored episodes are excluded from both the numerator and the denominator "
    "and reported separately, because an infrastructure failure is not a judge verdict of 0."
)


class ConditionLabel(BaseModel):
    """Every experimental axis of one arm, structured so two arms cannot silently collide.

    A free-text condition string is how an ablation program loses data: two runs that differ in
    a seed or a judge get the same label, the second overwrites the first, and the loss is
    invisible until someone tries to reproduce a number. Every axis here is therefore a named
    field, `key()` is the identity, and `build_ladder` rejects a ladder whose arms collide on it.

    `notes` is the escape hatch for an axis this model does not name (a prompt revision, a
    decoding change). It is part of the identity, so using it separates two arms correctly.
    """

    model_config = ConfigDict(frozen=True)

    base_model: str = Field(min_length=1)
    optimizer: str = Field(min_length=1)  # "none", "distill", "distill+routing", ...
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    judge: str = Field(min_length=1)  # the verifier or judge that produced the rewards
    provenance: Provenance
    seed: int = 0
    notes: str = ""

    def key(self) -> str:
        """This condition's identity: two arms sharing it are the same experiment."""
        return (
            f"base_model={self.base_model}|optimizer={self.optimizer}|dataset={self.dataset}"
            f"|split={self.split}|judge={self.judge}|provenance={self.provenance}"
            f"|seed={self.seed}|notes={self.notes}"
        )

    def replace(self, **axes: str | int) -> ConditionLabel:
        """A copy with some axes changed: how ladder rungs are derived from a shared base."""
        return self.model_copy(update=dict(axes))


class RowOverhead(BaseModel):
    """Optimizer-side inference charged to one episode, outside the worker model's bill.

    Keyed to a row by (scenario_id, model, episode) rather than carried on `ScenarioOutcome`, so
    the ladder works on outcome matrices recorded before any optimizer overhead existed and so
    ROUTER overhead has a home too (the compression track's per-row compressor fields describe a
    compactor only). One row may carry several components.
    """

    scenario_id: str
    model: str
    episode: int = 0
    component: str = Field(min_length=1)  # "compressor", "router", ...
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_s: float = Field(default=0.0, ge=0.0)


class CompletionRule(BaseModel):
    """When a scored episode counts as a completed task.

    `verifier_flag` reads the verifier's own binary verdict (`ScenarioOutcome.success`), which is
    what a tau-bench style checker produces and the default. `reward_at_least` is for rubric
    rewards with partial credit, where the binary flag is not the quantity of interest.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["verifier_flag", "reward_at_least"] = "verifier_flag"
    threshold: float = 1.0

    def completed(self, outcome: ScenarioOutcome) -> bool:
        """Whether `outcome` (already known to be scored) counts as a completed task."""
        if self.kind == "verifier_flag":
            return outcome.success
        return outcome.reward is not None and outcome.reward >= self.threshold


DEFAULT_COMPLETION = CompletionRule()


class Arm(BaseModel):
    """One condition under test: its rows, its overhead, and the label that identifies it.

    An arm is a set of measured rows, not a policy: the caller decides what "the routed mix"
    means (replaying a policy, selecting one pool model via `rows_for_model`) and this module
    stays pure aggregation with no embedder, no network, and no fitting.
    """

    name: str = Field(min_length=1)
    condition: ConditionLabel
    rows: list[ScenarioOutcome] = Field(min_length=1)
    overheads: list[RowOverhead] = []
    # Where this arm sits on the cost/quality dial, when it is a dial position at all. Only arms
    # that declare one can become a D-DIAL `CostQualityAnchor`; a ladder rung is generally an
    # ablation step, not a dial setting, and synthesizing a position for it would read as a
    # measurement that was never taken.
    dial_position: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _overheads_name_rows(self) -> Arm:
        """Overhead must attach to a row of THIS arm, or the cost lands on nothing.

        Silently ignoring an unmatched key would understate effective cost, which is exactly the
        direction of error this module exists to prevent.
        """
        keys = {_row_key(row) for row in self.rows}
        orphans = sorted(
            {str(_overhead_key(o)) for o in self.overheads if _overhead_key(o) not in keys}
        )
        if orphans:
            raise ValueError(
                f"arm '{self.name}': overhead rows name episodes this arm does not contain: "
                f"{orphans[:5]}; keys are (scenario_id, model, episode) and must match the arm's "
                f"own rows, so attach the overhead to the arm whose episodes it was spent on"
            )
        return self


class EffectiveCost(BaseModel):
    """Cache-adjusted effective cost per completed task, with everything it excluded."""

    n_scored: int
    n_excluded: int  # unscored episodes: counted here, never averaged in as zeros
    n_completed: int
    provider_cost_usd: float
    overhead_cost_usd: float
    total_cost_usd: float
    # Spend on unscored episodes. Real money that no completed task can be charged for; surfaced
    # so an arm that burns budget on episodes it never scores cannot look cheap.
    excluded_cost_usd: float
    # None when nothing completed: a cost per completed task over zero tasks is not infinity or
    # zero, it is undefined, and quoting either would be a fabricated number.
    cost_per_completed_task_usd: float | None
    overhead_components: list[str] = []
    # Scored episodes that recorded tokens but $0. Legitimate for a self-hosted model, and the
    # signature of an unpriced pool entry otherwise, so it is reported rather than guessed at.
    zero_cost_rows: int = 0
    cost_assumptions: str = Field(min_length=1)


class QualityBlock(BaseModel):
    """Mean reward and task success rate over an arm's scored episodes."""

    mean_reward: float
    task_success_rate: float
    n_scored: int
    n_excluded: int


class LatencyBlock(BaseModel):
    """Per-task wall seconds, optimizer overhead included."""

    p50_s: float
    p95_s: float
    n_tasks: int


class Scorecard(BaseModel):
    """One arm on all three objectives against a NAMED anchor arm, over the same scenarios."""

    arm: str
    anchor: str
    condition: ConditionLabel
    anchor_condition: ConditionLabel
    provenance: Provenance  # shared by both sides; enforced, not assumed
    judge: str  # ditto: the verifier that produced every reward on this card
    scenarios_compared: int  # scenarios scored on BOTH sides: what every number below covers
    scenarios_excluded: int  # held out because one side went unscored

    quality: QualityBlock
    anchor_quality: QualityBlock
    cost: EffectiveCost
    anchor_cost: EffectiveCost
    latency: LatencyBlock
    anchor_latency: LatencyBlock

    # Deltas in the D-DIAL sign convention: quality points positive = better, cost percent
    # negative = cheaper. None when the anchor's own figure is undefined or zero.
    quality_delta_points: float
    cost_delta_percent: float | None
    latency_p50_delta_percent: float | None
    cost_assumptions: str = Field(min_length=1)


class LadderRung(BaseModel):
    """One step of an ablation ladder: an ordered scorecard plus its dial position, if any."""

    index: int
    scorecard: Scorecard
    dial_position: float | None = None


class OperatingPoint(BaseModel):
    """A named point on the cost/quality frontier, in the D-DIAL delta shape.

    Compatible with `CostQualityAnchor` rather than pretending to be one: an ablation rung has
    the two deltas the platform contract carries but usually has no dial coordinate, and
    `as_cost_quality_anchor` refuses to invent one.
    """

    name: str
    quality_delta_points: float
    cost_delta_percent: float | None
    dial_position: float | None = None

    def as_cost_quality_anchor(self) -> CostQualityAnchor:
        """This point as a D-DIAL anchor row. Requires a measured dial position and cost delta."""
        if self.dial_position is None:
            raise ValueError(
                f"operating point '{self.name}' has no dial position, so it cannot become a "
                f"CostQualityAnchor (whose `s` field is a dial setting); set `dial_position` on "
                f"the arm when the rung IS a dial setting, and otherwise report it as an "
                f"operating point"
            )
        if self.cost_delta_percent is None:
            raise ValueError(
                f"operating point '{self.name}' has no cost delta (its anchor completed no "
                f"tasks, so there is no cost to compare against); it cannot become a "
                f"CostQualityAnchor"
            )
        return CostQualityAnchor(
            cost_quality=self.dial_position,
            named_point=self.name,
            quality_delta_points=self.quality_delta_points,
            cost_delta_percent=self.cost_delta_percent,
        )


class Ladder(BaseModel):
    """An ordered ablation ladder plus its Pareto front over the three objectives.

    Every rung is measured over `scenarios_compared`, the ONE scenario set scored by the anchor
    and by every rung (see `build_ladder`). Pareto dominance compares rungs against each other,
    so pairwise "same scenarios" against the anchor is not enough: it would let a rung that went
    unscored on the hard scenarios be graded on an easier subset than the rung it dominates.
    """

    name: str
    anchor: str
    anchor_condition: ConditionLabel
    scenarios_compared: int  # the common set every rung and the anchor were scored on
    scenarios_excluded: int  # scenarios some side left unscored, held out of the whole ladder
    rungs: list[LadderRung]

    def pareto(self, *, latency: LatencyObjective = "p50") -> list[LadderRung]:
        """The non-dominated rungs, in ladder order.

        Rung A dominates rung B when A is no worse on all three objectives (reward higher, cost
        per completed task lower, latency lower) and strictly better on at least one. Rungs whose
        cost per completed task is undefined (nothing completed) cannot be placed on the frontier
        and are omitted; they remain in `rungs`.
        """
        placed = [r for r in self.rungs if r.scorecard.cost.cost_per_completed_task_usd is not None]
        return [
            rung
            for rung in placed
            if not any(_dominates(other, rung, latency) for other in placed if other is not rung)
        ]

    def operating_points(
        self, *, pareto_only: bool = True, latency: LatencyObjective = "p50"
    ) -> list[OperatingPoint]:
        """Named operating points, by default only the ones on the frontier."""
        rungs = self.pareto(latency=latency) if pareto_only else self.rungs
        return [
            OperatingPoint(
                name=rung.scorecard.arm,
                quality_delta_points=rung.scorecard.quality_delta_points,
                cost_delta_percent=rung.scorecard.cost_delta_percent,
                dial_position=rung.dial_position,
            )
            for rung in rungs
        ]


def rows_for_model(matrix: OutcomeMatrix, model: str) -> list[ScenarioOutcome]:
    """Every row one pool model produced: the usual way to build a single-model arm."""
    names = matrix.model_names()
    if model not in names:
        raise KeyError(f"no pool model named '{model}' in this matrix; available: {names}")
    return [o for o in matrix.outcomes if o.model == model]


def effective_cost_per_completed_task(
    rows: Sequence[ScenarioOutcome],
    *,
    overheads: Sequence[RowOverhead] = (),
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> EffectiveCost:
    """Aggregate `rows` into cache-adjusted effective cost per completed task.

    Cost is consumed from `ScenarioOutcome.cost_usd`, which the recording pool entry already
    priced with its cache tiers, plus every `RowOverhead` attached to an included row. Rows with
    `reward is None` are excluded from the numerator and the denominator alike; their count and
    their spend are reported rather than dropped.

    Args:
        rows: the episodes of one arm, scored and unscored.
        overheads: optimizer-side inference charged to those episodes, keyed by
            (scenario_id, model, episode). Overhead on an excluded row is excluded with it.
        completion: what counts as a completed task. Defaults to the verifier's binary flag.

    Returns:
        An `EffectiveCost` whose `cost_per_completed_task_usd` is None when nothing completed.
    """
    by_row: dict[tuple[str, str, int], list[RowOverhead]] = {}
    for overhead in overheads:
        by_row.setdefault(_overhead_key(overhead), []).append(overhead)

    scored = [row for row in rows if row.reward is not None]
    excluded = [row for row in rows if row.reward is None]

    provider_cost = sum(row.cost_usd for row in scored)
    overhead_cost = sum(o.cost_usd for row in scored for o in by_row.get(_row_key(row), ()))
    excluded_cost = sum(
        row.cost_usd + sum(o.cost_usd for o in by_row.get(_row_key(row), ())) for row in excluded
    )
    components = sorted({o.component for row in scored for o in by_row.get(_row_key(row), ())})
    zero_cost = sum(
        1
        for row in scored
        if row.cost_usd == 0.0 and (row.usage.input_tokens + row.usage.output_tokens) > 0
    )

    n_completed = sum(1 for row in scored if completion.completed(row))
    total = provider_cost + overhead_cost
    per_task = total / n_completed if n_completed else None

    return EffectiveCost(
        n_scored=len(scored),
        n_excluded=len(excluded),
        n_completed=n_completed,
        provider_cost_usd=provider_cost,
        overhead_cost_usd=overhead_cost,
        total_cost_usd=total,
        excluded_cost_usd=excluded_cost,
        cost_per_completed_task_usd=per_task,
        overhead_components=components,
        zero_cost_rows=zero_cost,
        cost_assumptions=_assumptions(
            components=components,
            completion=completion,
            n_excluded=len(excluded),
            excluded_cost=excluded_cost,
            n_completed=n_completed,
            zero_cost=zero_cost,
        ),
    )


def build_scorecard(
    *,
    arm: Arm,
    anchor: Arm,
    completion: CompletionRule = DEFAULT_COMPLETION,
    restrict_to: Sequence[str] | None = None,
) -> Scorecard:
    """Score `arm` against `anchor` on quality, effective cost, and latency, same scenarios.

    Both sides are aggregated over the intersection of scenarios scored on BOTH sides, for the
    reason report.py states: unscored episodes are not random, so per-side means over different
    subsets let an arm that times out on the hard scenarios out-score an anchor that answered
    everything.

    Args:
        arm: the condition under test.
        anchor: the named arm every number is stated against.
        completion: what counts as a completed task.
        restrict_to: narrow the comparison further to these scenario ids. `build_ladder` passes
            the set common to every rung so that rungs are comparable with each other and not
            merely with the anchor; a standalone pairwise scorecard leaves it None.

    Raises:
        ValueError: when the two arms are not comparable (different provenance, judge, dataset,
            or split), when they carry the same condition label, or when no scenario survives.
    """
    _require_comparable(arm, anchor)

    arm_rows = _by_scenario(arm.rows)
    anchor_rows = _by_scenario(anchor.rows)
    all_ids = list(arm_rows) + [sid for sid in anchor_rows if sid not in arm_rows]
    compared = _pairwise_scored(arm_rows, anchor_rows, all_ids)
    if not compared:
        raise ValueError(
            f"arm '{arm.name}' and anchor '{anchor.name}' share no scenario scored on BOTH "
            f"sides, so there is nothing to compare over {len(all_ids)} scenarios; check the "
            f"matrix for unscored episodes before reporting this ladder"
        )
    if restrict_to is not None:
        allowed = set(restrict_to)
        compared = [sid for sid in compared if sid in allowed]
        if not compared:
            raise ValueError(
                f"arm '{arm.name}' and anchor '{anchor.name}' share no scored scenario inside "
                f"the {len(allowed)} the comparison was restricted to; the restriction is the "
                f"set every rung of the ladder scored, so this arm scored none of them"
            )

    kept = set(compared)
    arm_kept = [o for sid in compared for o in arm_rows.get(sid, ())]
    anchor_kept = [o for sid in compared for o in anchor_rows.get(sid, ())]
    arm_overhead = [o for o in arm.overheads if o.scenario_id in kept]
    anchor_overhead = [o for o in anchor.overheads if o.scenario_id in kept]

    cost = effective_cost_per_completed_task(
        arm_kept, overheads=arm_overhead, completion=completion
    )
    anchor_cost = effective_cost_per_completed_task(
        anchor_kept, overheads=anchor_overhead, completion=completion
    )
    quality = _quality(arm_kept, completion)
    anchor_quality = _quality(anchor_kept, completion)
    latency = _latency(arm_kept, arm_overhead)
    anchor_latency = _latency(anchor_kept, anchor_overhead)

    return Scorecard(
        arm=arm.name,
        anchor=anchor.name,
        condition=arm.condition,
        anchor_condition=anchor.condition,
        provenance=arm.condition.provenance,
        judge=arm.condition.judge,
        scenarios_compared=len(compared),
        scenarios_excluded=len(all_ids) - len(compared),
        quality=quality,
        anchor_quality=anchor_quality,
        cost=cost,
        anchor_cost=anchor_cost,
        latency=latency,
        anchor_latency=anchor_latency,
        quality_delta_points=(quality.mean_reward - anchor_quality.mean_reward) * 100.0,
        cost_delta_percent=_percent_delta(
            cost.cost_per_completed_task_usd, anchor_cost.cost_per_completed_task_usd
        ),
        latency_p50_delta_percent=_percent_delta(latency.p50_s, anchor_latency.p50_s),
        cost_assumptions=cost.cost_assumptions,
    )


def build_ladder(
    name: str,
    *,
    anchor: Arm,
    arms: Sequence[Arm],
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> Ladder:
    """Score an ordered ablation ladder (distill-only, +routing, +compaction) against one anchor.

    Rung order is `arms` order and is meaningful: a ladder is read as a sequence of additions.

    Every rung is measured over ONE scenario set: those scored by the anchor and by every rung.
    A per-rung pairwise intersection with the anchor is not sufficient here, because `pareto()`
    compares rungs against each other; a rung whose episodes went unscored on the hard scenarios
    would otherwise be graded on an easier subset than the rung it is said to dominate. The
    scenarios that drop out are counted on the ladder, not discarded quietly.

    Raises:
        ValueError: when `arms` is empty, when two arms share a name, when two arms (or an arm
            and the anchor) share a condition label, or when no scenario was scored by the anchor
            and every rung. A collision means two rungs are the same experiment under different
            display names, which is how ablation results get silently overwritten.
    """
    if not arms:
        raise ValueError(f"ladder '{name}' needs at least one arm besides the anchor")

    seen_names = {anchor.name: "anchor"}
    seen_keys = {anchor.condition.key(): anchor.name}
    for arm in arms:
        if arm.name in seen_names:
            raise ValueError(
                f"ladder '{name}': two rungs are both named '{arm.name}'; rung names are the "
                f"display handles for the ladder, so give each one a distinct name"
            )
        seen_names[arm.name] = arm.name
        key = arm.condition.key()
        if key in seen_keys:
            raise ValueError(
                f"ladder '{name}': rungs '{seen_keys[key]}' and '{arm.name}' carry the SAME "
                f"condition label ({key}), so they name the same experiment and one would "
                f"overwrite the other; a condition must encode every axis the rungs differ on "
                f"(use `optimizer`, `seed`, or `notes` to record what actually changed)"
            )
        seen_keys[key] = arm.name

    # The one scenario set the whole ladder is measured on. Ordered by the anchor's own rows so
    # the artifact is deterministic, and intersected rung by rung so a single arm that went
    # unscored somewhere narrows every rung equally rather than only its own.
    anchor_rows = _by_scenario(anchor.rows)
    universe = list(anchor_rows)
    common = set(universe)
    for arm in arms:
        _require_comparable(arm, anchor)
        arm_rows = _by_scenario(arm.rows)
        for sid in arm_rows:
            if sid not in common and sid not in universe:
                universe.append(sid)
        common &= set(_pairwise_scored(arm_rows, anchor_rows, universe))
    if not common:
        raise ValueError(
            f"ladder '{name}': no scenario was scored by anchor '{anchor.name}' AND by every "
            f"rung ({', '.join(a.name for a in arms)}), so the rungs cannot be compared with "
            f"each other; every number on a ladder is measured on one common scenario set, so "
            f"rerun the unscored episodes or drop the rung that scored none of them"
        )
    ordered = [sid for sid in universe if sid in common]

    return Ladder(
        name=name,
        anchor=anchor.name,
        anchor_condition=anchor.condition,
        scenarios_compared=len(ordered),
        scenarios_excluded=len(universe) - len(ordered),
        rungs=[
            LadderRung(
                index=index,
                scorecard=build_scorecard(
                    arm=arm, anchor=anchor, completion=completion, restrict_to=ordered
                ),
                dial_position=arm.dial_position,
            )
            for index, arm in enumerate(arms)
        ],
    )


def _row_key(row: ScenarioOutcome) -> tuple[str, str, int]:
    return (row.scenario_id, row.model, row.episode)


def _overhead_key(overhead: RowOverhead) -> tuple[str, str, int]:
    return (overhead.scenario_id, overhead.model, overhead.episode)


def _pairwise_scored(
    arm_rows: dict[str, list[ScenarioOutcome]],
    anchor_rows: dict[str, list[ScenarioOutcome]],
    order: Sequence[str],
) -> list[str]:
    """The scenarios in `order` with at least one scored episode on BOTH sides."""
    return [
        sid
        for sid in order
        if any(o.reward is not None for o in arm_rows.get(sid, ()))
        and any(o.reward is not None for o in anchor_rows.get(sid, ()))
    ]


def _by_scenario(rows: Sequence[ScenarioOutcome]) -> dict[str, list[ScenarioOutcome]]:
    grouped: dict[str, list[ScenarioOutcome]] = {}
    for row in rows:
        grouped.setdefault(row.scenario_id, []).append(row)
    return grouped


def _quality(rows: Sequence[ScenarioOutcome], completion: CompletionRule) -> QualityBlock:
    scored = [row for row in rows if row.reward is not None]
    rewards = [row.reward for row in scored if row.reward is not None]
    return QualityBlock(
        mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        task_success_rate=(
            sum(1 for row in scored if completion.completed(row)) / len(scored) if scored else 0.0
        ),
        n_scored=len(scored),
        n_excluded=sum(1 for row in rows if row.reward is None),
    )


def _latency(rows: Sequence[ScenarioOutcome], overheads: Sequence[RowOverhead]) -> LatencyBlock:
    """Per-task seconds over scored rows: the episode's own calls plus its optimizer overhead."""
    by_row: dict[tuple[str, str, int], float] = {}
    for overhead in overheads:
        key = _overhead_key(overhead)
        by_row[key] = by_row.get(key, 0.0) + overhead.latency_s
    per_task = [
        sum(row.call_seconds) + by_row.get(_row_key(row), 0.0)
        for row in rows
        if row.reward is not None
    ]
    if not per_task:
        return LatencyBlock(p50_s=0.0, p95_s=0.0, n_tasks=0)
    p95 = quantiles(per_task, n=20)[-1] if len(per_task) > 1 else per_task[0]
    return LatencyBlock(p50_s=median(per_task), p95_s=p95, n_tasks=len(per_task))


def _percent_delta(value: float | None, anchor: float | None) -> float | None:
    """Percent change of `value` against `anchor`; None when the anchor cannot be divided by."""
    if value is None or anchor is None or anchor == 0.0:
        return None
    return (value - anchor) / anchor * 100.0


def _dominates(a: LadderRung, b: LadderRung, latency: LatencyObjective) -> bool:
    """Whether rung `a` is no worse than `b` on all three objectives and better on one."""
    a_cost = a.scorecard.cost.cost_per_completed_task_usd
    b_cost = b.scorecard.cost.cost_per_completed_task_usd
    if a_cost is None or b_cost is None:  # unreachable: `pareto` filters these out first
        return False
    a_lat = a.scorecard.latency.p50_s if latency == "p50" else a.scorecard.latency.p95_s
    b_lat = b.scorecard.latency.p50_s if latency == "p50" else b.scorecard.latency.p95_s
    pairs = (
        (b.scorecard.quality.mean_reward, a.scorecard.quality.mean_reward),  # higher is better
        (a_cost, b_cost),  # lower is better
        (a_lat, b_lat),
    )
    return all(x <= y for x, y in pairs) and any(x < y for x, y in pairs)


def _require_comparable(arm: Arm, anchor: Arm) -> None:
    """Fail loudly on any axis that must match for the two sides to be one comparison.

    Follows `wmo.evals.grid.merge_results`: an invariant that would produce a plausible but
    meaningless number is checked, not documented. `base_model`, `optimizer`, and `seed` are
    free to differ, because differing on them is what an ablation IS.
    """
    if arm.name == anchor.name:
        raise ValueError(
            f"arm and anchor are both named '{arm.name}'; a scorecard names the anchor it was "
            f"measured against, so the two need distinct names"
        )
    mine = arm.condition
    theirs = anchor.condition
    comparability: tuple[tuple[str, str, str], ...] = (
        ("provenance", mine.provenance, theirs.provenance),
        ("judge", mine.judge, theirs.judge),
        ("dataset", mine.dataset, theirs.dataset),
        ("split", mine.split, theirs.split),
    )
    for field, ours, other in comparability:
        if ours != other:
            raise ValueError(
                f"arm '{arm.name}' has {field}={ours!r} but anchor '{anchor.name}' has "
                f"{field}={other!r}; those numbers were not measured on the same footing, so "
                f"comparing them would report a difference in the setup as a difference in the "
                f"optimizer (measure both sides under one {field}, or anchor against an arm "
                f"that shares it)"
            )
    if arm.condition.key() == anchor.condition.key():
        raise ValueError(
            f"arm '{arm.name}' and anchor '{anchor.name}' carry the SAME condition label "
            f"({arm.condition.key()}), so they are the same experiment and the scorecard would "
            f"compare an arm against itself"
        )


def _assumptions(
    *,
    components: Sequence[str],
    completion: CompletionRule,
    n_excluded: int,
    excluded_cost: float,
    n_completed: int,
    zero_cost: int,
) -> str:
    """Compose the basis sentence from what the rows actually contained."""
    parts = [EFFECTIVE_COST_RULE]
    if completion.kind == "verifier_flag":
        parts.append("A task counts as completed when the verifier marked the episode a success.")
    else:
        parts.append(
            f"A task counts as completed when its verified reward is at least "
            f"{completion.threshold:g}."
        )
    if components:
        parts.append(f"Optimizer-side inference recorded here covers: {', '.join(components)}.")
    else:
        parts.append(
            "No optimizer-side inference was recorded for these episodes, so the figure is "
            "provider spend alone."
        )
    if n_excluded:
        parts.append(
            f"{n_excluded} unscored episode(s) were excluded, along with ${excluded_cost:.4f} of "
            f"spend that no completed task is charged for."
        )
    if not n_completed:
        parts.append(
            "No task completed, so there is no cost per completed task to quote and the figure "
            "is reported as undefined rather than as zero."
        )
    if zero_cost:
        parts.append(
            f"{zero_cost} scored episode(s) recorded tokens but $0, which is expected for a "
            f"self-hosted model and otherwise means a pool entry carried no price."
        )
    return " ".join(parts)

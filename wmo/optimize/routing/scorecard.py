"""The three-objective scorecard and the ablation ladder it reports through.

The accounting rule (D-COMPRESS, binding for every savings claim this project makes): savings
are **cache-adjusted effective cost per completed task, compressor inference cost and latency
included**. The rule was written for the compression track, but it binds any optimizer that
buys cheaper tokens by spending inference somewhere else, so this module generalizes its
"compressor" to an *optimizer overhead*: cost and time an arm incurs OUTSIDE the worker model's
own bill (a compactor pass, a router's embedding call). Nothing aggregated the rule before this
module existed; report.py's `Headline` reports mean cost per RUN against one baseline model,
which is a different (and, for an ablation, misleading) quantity: an arm that fails half its
tasks looks cheap per run and is ruinous per completed task.

Three objectives, never one:

1. Quality: mean reward plus task success rate, averaged per scenario (see below).
2. Effective cost per completed task: cache-adjusted provider spend plus optimizer overhead,
   divided by the number of tasks actually completed.
3. Latency: p50 and p95 of per-task MODEL seconds, optimizer overhead included.

Discipline inherited from the surrounding code, and where this module deliberately differs:

- Unscored episodes (`reward is None`) are an infrastructure failure, not a judge verdict of 0.
  They are excluded from every numerator and denominator, counted, and their spend is reported
  separately. Money is conserved: for either side of a scorecard,
  `cost.total_cost_usd + cost.excluded_cost_usd + withheld_cost_usd` equals everything that
  side spent, so no dollar can leave the artifact silently.
- "Same scenarios" is literal and enforced, as in `report.py`: an arm and its anchor are
  compared over the intersection of scenarios scored on BOTH sides, and the counts in and out
  are on the artifact. A LADDER goes further and holds every rung to one common scenario set,
  because its Pareto front compares rungs against each other and not only against the anchor.
- Quality is averaged per SCENARIO, not per episode. Cost per completed task is legitimately
  episode-level (each episode is a task attempt that cost money), but a mean reward over
  episodes silently reweights the comparison when two arms ran different episode counts on the
  same scenario, which is the scenario-level version of the bug the common set exists to stop.
- Comparability invariants fail loudly rather than merge.
  A `wm_simulated` arm never silently scores against a `real_episode` anchor, and two arms whose
  condition labels collide are rejected: the GEPA program lost runs to colliding labels, so a
  `ConditionLabel` here is structured and must name every experimental axis.
- Costs are consumed, never re-derived. `ScenarioOutcome.cost_usd` was priced at record time by
  `PoolEntry.cost_usd`, which bills cache reads and cache writes at their own tiers; that is
  where "cache-adjusted" comes from and re-pricing here would drop negotiated rates.
- Every estimate names its basis, as in `wmo.simulation.serving.savings`: `cost_assumptions` is
  composed from what the rows actually contained and is mandatory non-empty.
- Latency is per TASK, not per call as in `report.py`, and it is MODEL time: `call_seconds`
  excludes environment and tool time by contract (`outcomes.py`), so these numbers understate
  an operator's wall clock and flatter any optimizer that only shortens prompts. Cost per
  completed task and seconds per task are still the pair to reason about together; a per-call
  p95 cannot be compared against a per-task cost at all.

A rung is a CONFIG evaluated offline against one existing matrix, never a rerun: a single-model
arm selects its rows with `rows_for_model`, and a routed arm replays its policy's own choices
with `rows_for_policy`. Buy the matrix once, then every rung is a lookup.

Call site (the joint tau-bench ablation ladder):

    from wmo.optimize.routing.scorecard import Arm, ConditionLabel, build_ladder, rows_for_model
    from wmo.optimize.routing.scorecard import rows_for_policy

    tau = ConditionLabel(
        dataset="tau-bench-retail", split="test", judge="tau2-verifier",
        provenance="real_episode", base_model="qwen3-8b", optimizer="none",
    )
    plus_routing = Arm(
        name="+routing",
        condition=tau.replace(optimizer="distill+routing"),
        rows=rows_for_policy(matrix, champion_policy),
        overheads=router_overheads,
        dial_position=0.25,
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

from math import isclose
from statistics import median, quantiles
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.optimize.routing.knn import CostQualityAnchor
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence


# Whether an arm's numbers came from episodes against a world-model simulation or against the
# real environment. Mixing the two in one comparison is the single easiest way to publish a
# number nobody can reproduce, so it is a required axis and an enforced invariant.
Provenance = Literal["wm_simulated", "real_episode"]

# Which latency statistic the Pareto front dominates on. p95 is reported on every scorecard but
# is not the dominance coordinate by default: at ablation sample sizes it is one slow episode.
LatencyObjective = Literal["p50", "p95"]

# Floats that differ by less than this are a tie, not an improvement. Summation order differs
# between the arm side and the anchor side, so two genuinely equal rungs can land 1 ULP apart;
# without a tolerance one of them would silently fall off the frontier.
DOMINANCE_TOLERANCE = 1e-9

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
        """A copy with some axes changed: how ladder rungs are derived from a shared base.

        Re-validates rather than using `model_copy(update=...)` directly. An unrecognized axis
        name is rejected instead of silently doing nothing: a typo would otherwise leave the
        intended field unchanged and surface later as an inexplicable ladder label collision,
        which is the exact failure this class exists to prevent.

        Raises:
            ValueError: when an axis is not a field of this model, or when the new value fails
                the field's own validation (an empty string, a bad provenance).
        """
        unknown = sorted(set(axes) - set(type(self).model_fields))
        if unknown:
            raise ValueError(
                f"unknown condition axes {unknown}; the axes are "
                f"{sorted(type(self).model_fields)}, so fix the spelling or record the change "
                f"in `notes`, which is part of the label identity"
            )
        return ConditionLabel.model_validate({**self.model_dump(), **axes})


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
    means (replaying a policy via `rows_for_policy`, selecting one pool model via
    `rows_for_model`) and this module stays pure aggregation with no fitting.
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
    def _rows_are_well_formed(self) -> Arm:
        """Reject row sets whose aggregation would be quietly wrong.

        Three checks, all guarding the same direction of error (a plausible number nobody can
        reproduce): duplicate episode keys would charge one overhead twice, an out-of-range
        reward would inflate `quality_delta_points` (which is scaled by 100 and read as
        percentage points by the D-DIAL contract), and orphan overhead would land real spend on
        no episode at all.
        """
        seen: set[tuple[str, str, int]] = set()
        for row in self.rows:
            key = _row_key(row)
            if key in seen:
                raise ValueError(
                    f"arm '{self.name}': two rows share the episode key {key}; overhead is "
                    f"charged per (scenario_id, model, episode), so a duplicate would be billed "
                    f"twice. Give the repeat run a distinct `episode` number."
                )
            seen.add(key)
            if row.reward is not None and not 0.0 <= row.reward <= 1.0:
                raise ValueError(
                    f"arm '{self.name}': row {key} has reward {row.reward}, outside [0, 1]. "
                    f"Quality deltas are reported in points (reward x 100) against the D-DIAL "
                    f"contract, so a rubric on another scale would be published as a wildly "
                    f"wrong percentage. Normalize the reward to [0, 1] before building the arm."
                )
        _require_overheads_match(self.rows, self.overheads, owner=f"arm '{self.name}'")
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
    # True when EVERY scored episode was unpriced, which makes `cost_per_completed_task_usd`
    # exactly $0 and any saving against it read as -100%. A typed flag rather than a sentence
    # in `cost_assumptions`, because a renderer can check a field and cannot check prose.
    cost_is_unpriced: bool = False
    cost_assumptions: str = Field(min_length=1)


class QualityBlock(BaseModel):
    """Mean reward and task success rate, averaged per scenario then across scenarios."""

    mean_reward: float
    task_success_rate: float
    n_scenarios: int  # what the two means above are averaged over
    n_scored: int  # scored EPISODES behind those scenarios
    n_excluded: int  # unscored episodes inside the compared scenarios


class LatencyBlock(BaseModel):
    """Per-task model seconds, optimizer overhead included, environment time excluded.

    Named `_model_s` on purpose: `ScenarioOutcome.call_seconds` is model call time only, so
    these are not an operator's wall clock (see the module docstring).
    """

    p50_model_s: float
    p95_model_s: float
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

    # Spend on scenarios held OUT of the comparison entirely (one side left them unscored, so
    # neither side's rows for them are aggregated). Distinct from `cost.excluded_cost_usd`,
    # which is spend on unscored episodes INSIDE the compared scenarios. Both are needed for
    # the conservation identity in the module docstring: without this bucket, a scenario the
    # anchor failed to score would take the arm's real (possibly large) spend with it silently.
    withheld_cost_usd: float = 0.0
    anchor_withheld_cost_usd: float = 0.0

    # Deltas in the D-DIAL sign convention: quality points positive = better, cost percent
    # negative = cheaper. A percent delta is None when either side's figure is undefined, or
    # when the anchor's is zero and the ratio would divide by nothing.
    quality_delta_points: float
    cost_delta_percent: float | None
    latency_p50_delta_percent: float | None
    cost_assumptions: str = Field(min_length=1)


class LadderRung(BaseModel):
    """One step of an ablation ladder: an ordered scorecard plus its dial position, if any."""

    index: int
    scorecard: Scorecard
    dial_position: float | None = None
    # Whether the ANCHOR is at least as good on all three objectives and strictly better on one,
    # at the default p50 latency objective. Such a rung bought nothing and is kept off the
    # frontier (`pareto`). Stored so a serialized ladder says so without recomputation.
    dominated_by_anchor: bool = False


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
        """This point as a D-DIAL anchor row. Requires a measured dial position and cost delta.

        Contract the caller must satisfy, which this method cannot check: `CostQualityAnchor`
        rows state their deltas against the BEST SINGLE POOL MODEL on the same held-out split
        (see `wmo.optimize.routing.knn`). A ladder anchor is whatever arm the caller passed, so
        convert only when that anchor IS the best single pool model. Mixing a differently-anchored
        row into `COST_QUALITY_ANCHORS` would put two baselines on one frontier.

        Raises:
            ValueError: when this point has no measured dial position, or when its cost delta is
                undefined because one side completed no task.
        """
        if self.dial_position is None:
            raise ValueError(
                f"operating point '{self.name}' has no dial position, so it cannot become a "
                f"CostQualityAnchor (whose `s` field is a dial setting); set `dial_position` on "
                f"the arm when the rung IS a dial setting, and otherwise report it as an "
                f"operating point"
            )
        if self.cost_delta_percent is None:
            raise ValueError(
                f"operating point '{self.name}' has no cost delta: one side of its scorecard "
                f"completed no task, so there is no cost per completed task to compare and the "
                f"ratio is undefined. Rerun that side's failed episodes, or drop this rung from "
                f"the dial rather than publishing a point with no measured saving."
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
    rungs: list[LadderRung] = Field(min_length=1)

    def pareto(self, *, latency: LatencyObjective = "p50") -> list[LadderRung]:
        """The non-dominated rungs, in ladder order.

        Rung A dominates rung B when A is no worse on all three objectives (reward higher, cost
        per completed task lower, latency lower) and strictly better on at least one. Floats
        within `DOMINANCE_TOLERANCE` count as equal, so two genuinely tied rungs both stay on
        the frontier.

        Two kinds of rung never appear. One whose cost per completed task is undefined (nothing
        completed) cannot be placed on a cost axis at all. One the ANCHOR dominates bought
        nothing: it is not an operating point a reader should be offered, and including it would
        let a ladder advertise a frontier the untouched baseline beats. Both remain in `rungs`.
        An empty result is the honest answer that no rung improved on the anchor.
        """
        placed = [
            rung
            for rung in self.rungs
            if rung.scorecard.cost.cost_per_completed_task_usd is not None
            and not _anchor_dominates(rung.scorecard, latency)
        ]
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

    Raises:
        ValueError: when an overhead names an episode absent from `rows`. Ignoring it would
            understate effective cost, so this matches the identical guard on `Arm`.
    """
    _require_overheads_match(rows, overheads, owner="these rows")

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
    unpriced = bool(scored) and provider_cost == 0.0 and zero_cost == len(scored)

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
        cost_is_unpriced=unpriced,
        cost_assumptions=_assumptions(
            components=components,
            completion=completion,
            n_excluded=len(excluded),
            excluded_cost=excluded_cost,
            n_completed=n_completed,
            zero_cost=zero_cost,
            unpriced=unpriced,
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
    everything. Spend on the scenarios that drop out is reported in `withheld_cost_usd` rather
    than vanishing.

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
    withheld = _withheld_cost(arm, kept)
    anchor_withheld = _withheld_cost(anchor, kept)

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
        withheld_cost_usd=withheld,
        anchor_withheld_cost_usd=anchor_withheld,
        quality_delta_points=(quality.mean_reward - anchor_quality.mean_reward) * 100.0,
        cost_delta_percent=_percent_delta(
            cost.cost_per_completed_task_usd, anchor_cost.cost_per_completed_task_usd
        ),
        latency_p50_delta_percent=_percent_delta(latency.p50_model_s, anchor_latency.p50_model_s),
        cost_assumptions=_card_assumptions(
            arm=arm.name,
            anchor=anchor.name,
            cost=cost,
            anchor_cost=anchor_cost,
            scenarios_excluded=len(all_ids) - len(compared),
            withheld=withheld,
            anchor_withheld=anchor_withheld,
        ),
    )


def _row_key(row: ScenarioOutcome) -> tuple[str, str, int]:
    """The episode identity overhead is charged against."""
    return (row.scenario_id, row.model, row.episode)


def _overhead_key(overhead: RowOverhead) -> tuple[str, str, int]:
    """The episode an overhead claims to belong to; matched against `_row_key`."""
    return (overhead.scenario_id, overhead.model, overhead.episode)


def _require_overheads_match(
    rows: Sequence[ScenarioOutcome], overheads: Sequence[RowOverhead], *, owner: str
) -> None:
    """Reject overhead that names an episode absent from `rows`.

    Silently ignoring an unmatched key would understate effective cost, which is exactly the
    direction of error this module exists to prevent, so both entry points (the `Arm` validator
    and `effective_cost_per_completed_task`) run this same check.
    """
    keys = {_row_key(row) for row in rows}
    orphans = sorted({str(_overhead_key(o)) for o in overheads if _overhead_key(o) not in keys})
    if orphans:
        raise ValueError(
            f"{owner}: overhead rows name episodes not present here: {orphans[:5]}; keys are "
            f"(scenario_id, model, episode) and must match the rows they are charged to, so "
            f"attach the overhead to the episodes it was actually spent on"
        )


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
    """Group rows by scenario id, preserving first-seen scenario order."""
    grouped: dict[str, list[ScenarioOutcome]] = {}
    for row in rows:
        grouped.setdefault(row.scenario_id, []).append(row)
    return grouped


def _withheld_cost(arm: Arm, kept: set[str]) -> float:
    """What `arm` spent on scenarios held out of the comparison, overhead included."""
    overhead_by_row: dict[tuple[str, str, int], float] = {}
    for overhead in arm.overheads:
        key = _overhead_key(overhead)
        overhead_by_row[key] = overhead_by_row.get(key, 0.0) + overhead.cost_usd
    return sum(
        row.cost_usd + overhead_by_row.get(_row_key(row), 0.0)
        for row in arm.rows
        if row.scenario_id not in kept
    )


def _quality(rows: Sequence[ScenarioOutcome], completion: CompletionRule) -> QualityBlock:
    """Mean reward and success rate, averaged within each scenario then across scenarios.

    Per-scenario first (see the module docstring): an arm that ran three episodes on an easy
    scenario and one on a hard one would otherwise weight the easy scenario three times against
    an anchor that ran one of each.
    """
    scored = [row for row in rows if row.reward is not None]
    by_scenario = _by_scenario(scored)
    scenario_rewards: list[float] = []
    scenario_success: list[float] = []
    for scenario_rows in by_scenario.values():
        rewards = [row.reward for row in scenario_rows if row.reward is not None]
        scenario_rewards.append(sum(rewards) / len(rewards))
        scenario_success.append(
            sum(1.0 for row in scenario_rows if completion.completed(row)) / len(scenario_rows)
        )
    # 0.0 on empty is unreachable through `build_scorecard`, which raises before it can pass an
    # unscored-only side; kept total rather than raising in a helper no caller can trip.
    return QualityBlock(
        mean_reward=sum(scenario_rewards) / len(scenario_rewards) if scenario_rewards else 0.0,
        task_success_rate=sum(scenario_success) / len(scenario_success)
        if scenario_success
        else 0.0,
        n_scenarios=len(by_scenario),
        n_scored=len(scored),
        n_excluded=sum(1 for row in rows if row.reward is None),
    )


def _latency(rows: Sequence[ScenarioOutcome], overheads: Sequence[RowOverhead]) -> LatencyBlock:
    """Per-task model seconds over scored rows: the episode's calls plus its optimizer overhead.

    p95 uses the INCLUSIVE quantile method, which is bounded by the observed data. The default
    exclusive method extrapolates past the sample at the small n an ablation runs on: three
    tasks at 1s, 1s and 9s report a p95 of 15.4s, a tail longer than anything measured, and p95
    is a supported Pareto dominance coordinate.
    """
    by_row: dict[tuple[str, str, int], float] = {}
    for overhead in overheads:
        key = _overhead_key(overhead)
        by_row[key] = by_row.get(key, 0.0) + overhead.latency_s
    per_task = [
        sum(row.call_seconds) + by_row.get(_row_key(row), 0.0)
        for row in rows
        if row.reward is not None
    ]
    if not per_task:  # unreachable via `build_scorecard`, as in `_quality`
        return LatencyBlock(p50_model_s=0.0, p95_model_s=0.0, n_tasks=0)
    p95 = quantiles(per_task, n=20, method="inclusive")[-1] if len(per_task) > 1 else per_task[0]
    return LatencyBlock(p50_model_s=median(per_task), p95_model_s=p95, n_tasks=len(per_task))


def _percent_delta(value: float | None, anchor: float | None) -> float | None:
    """Percent change of `value` against `anchor`; None when the anchor cannot be divided by."""
    if value is None or anchor is None or anchor == 0.0:
        return None
    return (value - anchor) / anchor * 100.0


def _no_worse(x: float, y: float) -> bool:
    """Whether `x` is at least as good as `y` on a lower-is-better axis, ties included."""
    return x <= y or isclose(x, y, rel_tol=DOMINANCE_TOLERANCE)


def _strictly_better(x: float, y: float) -> bool:
    """Whether `x` beats `y` on a lower-is-better axis by more than float noise."""
    return x < y and not isclose(x, y, rel_tol=DOMINANCE_TOLERANCE)


def _objectives(card: Scorecard, latency: LatencyObjective) -> tuple[float, float, float]:
    """The arm's three objectives, all as lower-is-better numbers."""
    cost = card.cost.cost_per_completed_task_usd
    if cost is None:  # guarded by every caller; keeps the tuple total
        raise ValueError(f"scorecard '{card.arm}' has no cost per completed task to compare")
    lat = card.latency.p50_model_s if latency == "p50" else card.latency.p95_model_s
    return (-card.quality.mean_reward, cost, lat)


def _anchor_objectives(card: Scorecard, latency: LatencyObjective) -> tuple[float, float, float]:
    """The same three objectives for the card's anchor side."""
    cost = card.anchor_cost.cost_per_completed_task_usd
    if cost is None:
        raise ValueError(f"scorecard '{card.arm}' has no anchor cost per completed task")
    lat = card.anchor_latency.p50_model_s if latency == "p50" else card.anchor_latency.p95_model_s
    return (-card.anchor_quality.mean_reward, cost, lat)


def _pareto_dominates(
    better: tuple[float, float, float], worse: tuple[float, float, float]
) -> bool:
    """Whether `better` is no worse on all three axes and strictly better on at least one."""
    pairs = tuple(zip(better, worse, strict=True))
    return all(_no_worse(x, y) for x, y in pairs) and any(_strictly_better(x, y) for x, y in pairs)


def _dominates(a: LadderRung, b: LadderRung, latency: LatencyObjective) -> bool:
    """Whether rung `a` is no worse than `b` on all three objectives and better on one."""
    return _pareto_dominates(_objectives(a.scorecard, latency), _objectives(b.scorecard, latency))


def _anchor_dominates(card: Scorecard, latency: LatencyObjective) -> bool:
    """Whether the anchor beats this arm outright, making the rung a non-operating-point."""
    if (
        card.cost.cost_per_completed_task_usd is None
        or card.anchor_cost.cost_per_completed_task_usd is None
    ):
        return False
    return _pareto_dominates(_anchor_objectives(card, latency), _objectives(card, latency))


def _require_comparable(arm: Arm, anchor: Arm) -> None:
    """Fail loudly on any axis that must match for the two sides to be one comparison.

    An invariant that would produce a plausible but meaningless number is checked, not documented.
    `base_model`, `optimizer`, and
    `seed` may differ because differing on them is what an ablation is.
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


def _usd(amount: float) -> str:
    """Format dollars so a nonzero figure never renders as $0.0000 in an honesty string."""
    if 0.0 < amount < 0.0001:
        return "less than $0.0001"
    return f"${amount:.4f}"


def _assumptions(
    *,
    components: Sequence[str],
    completion: CompletionRule,
    n_excluded: int,
    excluded_cost: float,
    n_completed: int,
    zero_cost: int,
    unpriced: bool,
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
            f"{n_excluded} unscored episode(s) were excluded, along with {_usd(excluded_cost)} "
            f"of spend that no completed task is charged for."
        )
    if not n_completed:
        parts.append(
            "No task completed, so there is no cost per completed task to quote and the figure "
            "is reported as undefined rather than as zero."
        )
    if unpriced:
        parts.append(
            "Every scored episode recorded tokens but $0, so the cost per completed task is $0 "
            "and any saving against it would read as -100%. Treat the cost figures as absent, "
            "not as free, unless this arm really did run on hardware you do not pay per token "
            "for."
        )
    elif zero_cost:
        parts.append(
            f"{zero_cost} scored episode(s) recorded tokens but $0, which is expected for a "
            f"self-hosted model and otherwise means a pool entry carried no price."
        )
    return " ".join(parts)


def _card_assumptions(
    *,
    arm: str,
    anchor: str,
    cost: EffectiveCost,
    anchor_cost: EffectiveCost,
    scenarios_excluded: int,
    withheld: float,
    anchor_withheld: float,
) -> str:
    """Both sides' bases plus what the comparison itself held out.

    A card states a DELTA over two sides, so quoting only the arm's basis would hide an anchor
    that went unscored on half its episodes.
    """
    parts = [
        f"Arm '{arm}': {cost.cost_assumptions}",
        f"Anchor '{anchor}': {anchor_cost.cost_assumptions}",
    ]
    if scenarios_excluded:
        parts.append(
            f"{scenarios_excluded} scenario(s) were held out of this comparison because one "
            f"side left them unscored, taking {_usd(withheld)} of arm spend and "
            f"{_usd(anchor_withheld)} of anchor spend with them; that money bought no "
            f"comparable measurement."
        )
    return " ".join(parts)


from wmo.optimize.routing.scorecard_ladder import build_ladder, rows_for_policy  # noqa: E402

__all__ = (
    "Arm",
    "CompletionRule",
    "ConditionLabel",
    "EffectiveCost",
    "Ladder",
    "LadderRung",
    "LatencyBlock",
    "LatencyObjective",
    "OperatingPoint",
    "Provenance",
    "QualityBlock",
    "RowOverhead",
    "Scorecard",
    "build_ladder",
    "build_scorecard",
    "effective_cost_per_completed_task",
    "rows_for_model",
    "rows_for_policy",
)

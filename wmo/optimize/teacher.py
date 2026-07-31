"""Teacher search: the distillation go/no-go, computed from evidence the sweep already bought.

Distillation is the most expensive thing this optimizer can do, and most workloads do not need
it. The decision is not a research judgement call, it is a function over data the pipeline pays
for anyway: `wmo optimize route sweep` measures EVERY pool model on the customer's own held-out
scenarios, so the resulting `OutcomeMatrix` already answers "is there a bigger model that is
materially better here, and what does it cost". This module reads that answer.

Two steps, in the order the money is at stake:

1. EXISTENCE. Take the cheapest measured model as the candidate STUDENT and ask whether any other
   model in the matrix beats it by at least `min_gap`, paired per scenario. No gap means no
   teacher: the cheap model already serves this workload and earns its traffic on price, so
   training would buy nothing and the stage skips with zero spend.
2. CHEAPEST SUFFICIENT TEACHER. When the gap exists, walk the price ladder upward and take the
   cheapest model that still keeps `sufficiency` of the best measured gain, preferring an
   open-weights candidate. A frontier model is usually the existence proof, rarely the economic
   teacher.

Measured evidence this reproduces, and the reason the two steps are separate:

- tau-bench: Qwen3.6-27B scored +1.6 points over the 9B student and Kimi K3 scored BELOW both.
  No gap, so no distillation, and the several hundred dollars a training run costs stayed unspent.
- TerminalBench-2: the same 27B scored +27 points over the same 9B. A gap, and the teacher is the
  27B at $0.30/$2.00 per Mtok rather than K3 at $3/$15, because K3's extra gain (if any) is not
  worth ten times the data cost.

Honest-stats rules, inherited from `wmo.optimize.scorecard` and non-negotiable here because the
output authorizes spending:

- PAIRED, same scenarios. A model is compared with the student only over scenarios BOTH have a
  scored episode on, per-scenario means first, so an easy subset cannot manufacture a gap.
- `reward is None` is an infrastructure failure, never a judge verdict of 0 (`outcomes.py`), so
  unscored rows are excluded from both sides.
- A gain whose confidence interval includes zero is NOT a gap. A thin matrix returns
  `insufficient_evidence`, never a false yes: the failure mode this gate exists to prevent is
  authorizing a training run on noise.

Call site:

    from wmo.optimize.outcomes import OutcomeMatrix
    from wmo.optimize.teacher import select_teacher

    verdict = select_teacher(OutcomeMatrix.load(matrix_path))
    if verdict.decision == "distill":
        train(student=verdict.student, teacher=verdict.teacher)
    else:
        print(verdict.reason)
"""

from __future__ import annotations

from math import sqrt
from statistics import fmean, stdev
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.scorecard import effective_cost_per_completed_task, rows_for_model
from wmo.providers.pool import PoolEntry, Tier

DEFAULT_MIN_GAP = 0.10
"""Reward points (as a fraction of 1.0) a teacher must beat the student by. 0.10 = 10 points.

The bar is deliberately coarse. A 2-point gap is inside the noise of any ablation-sized scenario
set, and closing it does not repay a training run; the two measured cases this gate was written
against (+1.6 points and +27 points) sit far on either side of it.
"""

DEFAULT_MIN_SCENARIOS = 8
"""Shared scored scenarios a comparison needs before this gate calls it either way."""

DEFAULT_SUFFICIENCY = 0.8
"""Fraction of the BEST measured gain a cheaper model must keep to be a sufficient teacher."""

CI_Z = 1.96
"""Normal quantile for the 95% interval reported on every gain row."""

Decision = Literal["distill", "do_not_distill", "insufficient_evidence"]
"""What the matrix supports. `insufficient_evidence` is a real answer, not an error."""

PriceBasis = Literal["measured", "list"]
"""Which ladder ordered the candidates: this matrix's own spend, or the pool's list prices."""


class ModelGain(BaseModel):
    """One candidate model's paired comparison against the student, with its price.

    Every number is measured over `n_scenarios`, the scenarios this model AND the student both
    have a scored episode on, so `mean_gain` is exactly `mean_reward - student_mean_reward` on
    one common set. Comparing two rows of this table against each other is not paired, however:
    two candidates may share different scenario subsets with the student.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    tier: Tier
    n_scenarios: int
    mean_reward: float  # this model, over the shared scenarios only
    student_mean_reward: float  # the student, over those same scenarios
    mean_gain: float
    # None below n=2: one scenario has no spread to estimate from, and quoting a zero-width
    # interval would read as certainty. Such a row can never clear the gap.
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    # The ladder key, in whatever unit `TeacherSearchVerdict.price_basis` names. None when this
    # model has no usable price at all, which sorts it last rather than first.
    price: float | None
    clears_gap: bool
    """Gain at or above `min_gap` AND an interval that excludes zero, on enough scenarios."""


class TeacherSearchVerdict(BaseModel):
    """The gate's answer: what to do, who with, and the reason to print verbatim."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    student: str
    teacher: str | None  # set only on `distill`
    n_scenarios: int  # scenarios the student itself was scored on: the ceiling on any pairing
    gains: list[ModelGain]  # every measured candidate, best gain first
    unmeasured_models: list[str] = []  # pool models sharing no scored scenario with the student
    price_basis: PriceBasis
    min_gap: float
    min_scenarios: int
    sufficiency: float
    reason: str = Field(min_length=1)

    @property
    def should_distill(self) -> bool:
        """Whether the matrix authorizes spending on a training run."""
        return self.decision == "distill"


def select_teacher(
    matrix: OutcomeMatrix,
    *,
    student: str | None = None,
    min_gap: float = DEFAULT_MIN_GAP,
    min_scenarios: int = DEFAULT_MIN_SCENARIOS,
    sufficiency: float = DEFAULT_SUFFICIENCY,
) -> TeacherSearchVerdict:
    """Decide whether this workload has a teacher worth distilling from, and which one.

    Pure function over the matrix: no network, no spend, no artifact written.

    Args:
        matrix: an `OutcomeMatrix` from `wmo optimize route sweep`, carrying its own pool.
        student: the model distillation would train. Defaults to the cheapest measured model,
            which is the one whose price makes distillation worth doing at all.
        min_gap: reward points (as a fraction of 1.0) a teacher must beat the student by.
        min_scenarios: shared scored scenarios a comparison needs to count as evidence.
        sufficiency: fraction of the best measured gain a cheaper teacher must keep.

    Returns:
        A `TeacherSearchVerdict` whose `reason` is written to be printed verbatim to an operator.

    Raises:
        ValueError: the thresholds are out of range, the matrix has fewer than two models with
            scored episodes, or a named `student` is absent or unscored there.
    """
    _validate_thresholds(min_gap=min_gap, min_scenarios=min_scenarios, sufficiency=sufficiency)

    means = {
        entry.name: _scenario_means(rows_for_model(matrix, entry.name)) for entry in matrix.pool
    }
    measured = sorted(name for name, scores in means.items() if scores)
    if len(measured) < 2:
        raise ValueError(
            f"teacher search needs at least two models with scored episodes in the matrix, and "
            f"this one has {len(measured)} ({', '.join(measured) or 'none'}); a gap is a "
            f"comparison, so sweep the pool you want compared before probing it"
        )

    prices, basis = _price_ladder(matrix, measured)
    student_name = _resolve_student(student, measured=measured, prices=prices, means=means)
    student_scores = means[student_name]

    gains = sorted(
        (
            _gain_row(
                entry=_entry(matrix, name),
                scores=means[name],
                student_scores=student_scores,
                price=prices[name],
                min_gap=min_gap,
                min_scenarios=min_scenarios,
            )
            for name in measured
            if name != student_name and _shared(means[name], student_scores)
        ),
        key=lambda row: (-row.mean_gain, row.model),
    )
    unmeasured = sorted(
        name for name in means if name != student_name and not _shared(means[name], student_scores)
    )

    decision, teacher, reason = _decide(
        student=student_name,
        gains=gains,
        min_gap=min_gap,
        min_scenarios=min_scenarios,
        sufficiency=sufficiency,
        basis=basis,
    )
    return TeacherSearchVerdict(
        decision=decision,
        student=student_name,
        teacher=teacher,
        n_scenarios=len(student_scores),
        gains=gains,
        unmeasured_models=unmeasured,
        price_basis=basis,
        min_gap=min_gap,
        min_scenarios=min_scenarios,
        sufficiency=sufficiency,
        reason=reason,
    )


def _validate_thresholds(*, min_gap: float, min_scenarios: int, sufficiency: float) -> None:
    """Reject thresholds whose verdict would be meaningless rather than merely strict."""
    if min_gap <= 0.0:
        raise ValueError(
            f"min_gap must be a positive fraction of a reward point (got {min_gap}); a "
            "non-positive bar would call every scoring difference a teacher gap"
        )
    if min_scenarios < 2:
        raise ValueError(
            f"min_scenarios must be at least 2 (got {min_scenarios}); one shared scenario has no "
            "spread to estimate an interval from, so it can never establish a gap"
        )
    if not 0.0 < sufficiency <= 1.0:
        raise ValueError(
            f"sufficiency must be in (0, 1] (got {sufficiency}); it is the fraction of the best "
            "measured gain a cheaper teacher has to keep"
        )


def _entry(matrix: OutcomeMatrix, name: str) -> PoolEntry:
    """The pool entry behind a measured model name (present by the matrix's own validator)."""
    return next(entry for entry in matrix.pool if entry.name == name)


def _scenario_means(rows: list[ScenarioOutcome]) -> dict[str, float]:
    """Mean reward per scenario over SCORED episodes only; scenarios with none are absent.

    Per scenario before anything else, as in `scorecard._quality`: a model that ran three
    episodes on an easy scenario and one on a hard one would otherwise weight the easy scenario
    three times against a student that ran one of each.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.reward is not None:
            grouped.setdefault(row.scenario_id, []).append(row.reward)
    return {sid: fmean(rewards) for sid, rewards in grouped.items()}


def _shared(scores: dict[str, float], student_scores: dict[str, float]) -> list[str]:
    """The scenarios both sides scored, in the student's own order (deterministic output)."""
    return [sid for sid in student_scores if sid in scores]


def _price_ladder(
    matrix: OutcomeMatrix, measured: list[str]
) -> tuple[dict[str, float | None], PriceBasis]:
    """Every measured model's ladder key, and which unit the whole ladder is in.

    Prefers what the matrix itself recorded: `scorecard.effective_cost_per_completed_task` over a
    model's own rows is cache-adjusted spend divided by tasks it actually completed, which is the
    price an operator would really pay to serve this workload on it. That figure is unavailable
    for a model whose entries carried no price, and undefined for one that completed no task at
    all, so the ladder falls back to LIST prices (a pool entry's own published input + output
    rate per Mtok) whenever any measured model is missing a figure. A measured $0.0000 counts as
    missing rather than as free: the rows recorded no spend, which is an absent price, and taking
    it literally would seat that model at the bottom of the ladder ahead of models that were
    priced.

    One basis for the whole ladder, never a mix: ordering some models by dollars per completed
    task and others by dollars per Mtok would compare two different quantities and could rank a
    genuinely cheaper model above a genuinely dearer one.
    """
    effective: dict[str, float | None] = {}
    for name in measured:
        cost = effective_cost_per_completed_task(rows_for_model(matrix, name))
        per_task = cost.cost_per_completed_task_usd
        effective[name] = None if cost.cost_is_unpriced or not per_task else per_task
    if all(value is not None for value in effective.values()):
        return effective, "measured"
    return {name: _list_price(_entry(matrix, name)) for name in measured}, "list"


def _list_price(entry: PoolEntry) -> float | None:
    """A pool entry's published input + output rate per Mtok: an ORDERING KEY, not a cost.

    The sum is not what anything will be billed (that depends on the token mix); it is monotone
    in both published rates, which is all a ladder needs, and it matches how these models are
    quoted to operators ("$0.30/$2.00 against $3/$15"). None when the entry has no price row at
    all, which sorts the model last rather than free.
    """
    try:
        price = entry.price()
    except ValueError:
        return None
    return price.input_per_mtok + price.output_per_mtok


def _resolve_student(
    student: str | None,
    *,
    measured: list[str],
    prices: dict[str, float | None],
    means: dict[str, dict[str, float]],
) -> str:
    """The model distillation would train: the named one, else the cheapest measured one.

    A price tie is broken toward the BETTER model (then the name), and "better" is judged
    over the scenarios ALL tied candidates share, so an easy private subset cannot flatter one
    of them (the same paired-first rule every gain in this module follows). When the tied
    candidates share nothing, the name breaks the tie: deterministic, and honest about there
    being no evidence to prefer either. The conservative direction stands: the better of two
    equally cheap students makes the gap harder to prove, so a tie can never be resolved into
    a training run that a different tie-break would have refused.
    """
    if student is not None:
        if student not in means:
            raise ValueError(
                f"no pool model named '{student}' in this matrix; measured models are "
                f"{', '.join(measured)}"
            )
        if not means[student]:
            raise ValueError(
                f"'{student}' has no scored episode in this matrix, so nothing can be compared "
                f"against it; measured models are {', '.join(measured)}"
            )
        return student
    priced = [price for name in measured if (price := prices[name]) is not None]
    if not priced:
        # Nothing in this matrix carries a price, so there is no ladder to be cheapest on. Name
        # the choice deterministically rather than picking whichever model the pool listed first.
        return sorted(measured)[0]
    cheapest = min(priced)
    tied = [name for name in measured if prices[name] == cheapest]
    if len(tied) == 1:
        return tied[0]
    shared = set.intersection(*(set(means[name]) for name in tied))
    if not shared:
        return sorted(tied)[0]
    return sorted(tied, key=lambda name: (-fmean(means[name][sid] for sid in shared), name))[0]


def _gain_row(
    *,
    entry: PoolEntry,
    scores: dict[str, float],
    student_scores: dict[str, float],
    price: float | None,
    min_gap: float,
    min_scenarios: int,
) -> ModelGain:
    """One candidate's paired gain over the student, with its 95% interval.

    The interval is over per-scenario DIFFERENCES, which is what makes it paired: scenario
    difficulty cancels, so the width reflects how consistently this model beats the student
    rather than how varied the scenarios were.
    """
    shared = _shared(scores, student_scores)
    diffs = [scores[sid] - student_scores[sid] for sid in shared]
    mean_gain = fmean(diffs)
    standard_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    if len(diffs) > 1:
        standard_error = stdev(diffs) / sqrt(len(diffs))
        ci_low = mean_gain - CI_Z * standard_error
        ci_high = mean_gain + CI_Z * standard_error
    return ModelGain(
        model=entry.name,
        tier=entry.tier,
        n_scenarios=len(shared),
        mean_reward=fmean(scores[sid] for sid in shared),
        student_mean_reward=fmean(student_scores[sid] for sid in shared),
        mean_gain=mean_gain,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        price=price,
        clears_gap=(
            len(shared) >= min_scenarios
            and mean_gain >= min_gap
            and ci_low is not None
            and ci_low > 0.0
        ),
    )


def _decide(
    *,
    student: str,
    gains: list[ModelGain],
    min_gap: float,
    min_scenarios: int,
    sufficiency: float,
    basis: PriceBasis,
) -> tuple[Decision, str | None, str]:
    """The three-way verdict and the sentence that explains it.

    Order matters. `insufficient_evidence` is checked in the two places a "no" would be dishonest
    (nothing was measured against the student, or the leading candidate was measured too thinly
    to test), so a thin matrix never reads as "no teacher exists" and never as "train now".
    """
    if not gains:
        return (
            "insufficient_evidence",
            None,
            f"INSUFFICIENT EVIDENCE: no other model in this matrix shares a scored scenario with "
            f"'{student}', so nothing can be compared against it. Sweep the candidates over the "
            f"same scenarios, then probe again.",
        )

    clearing = [row for row in gains if row.clears_gap]
    if clearing:
        teacher = _cheapest_sufficient(clearing, sufficiency=sufficiency)
        best = clearing[0]  # gains are sorted by gain, and every clearing row is in that order
        return (
            "distill",
            teacher.model,
            _distill_reason(
                student=student,
                teacher=teacher,
                best=best,
                min_gap=min_gap,
                sufficiency=sufficiency,
                basis=basis,
            ),
        )

    leader = gains[0]
    if leader.n_scenarios < min_scenarios:
        return (
            "insufficient_evidence",
            None,
            _thin_reason(student, leader, min_gap, min_scenarios),
        )
    return ("do_not_distill", None, _no_gap_reason(student, leader, min_gap))


def _cheapest_sufficient(clearing: list[ModelGain], *, sufficiency: float) -> ModelGain:
    """The teacher: cheapest model keeping `sufficiency` of the best gain, open weights first.

    Open tier outranks price, rather than merely breaking its ties. The existence proof may come
    from a frontier model, but a data teacher is trained ON, and an open-weights model is the one
    whose outputs you are reliably permitted to train on (the directive's open-source teacher
    preference). A frontier teacher is therefore chosen only when no open candidate is sufficient,
    and the reason says so, so the licensing question lands in front of the operator before the
    training spend does.
    """
    best_gain = max(row.mean_gain for row in clearing)
    sufficient = [row for row in clearing if row.mean_gain >= sufficiency * best_gain]
    return min(
        sufficient,
        key=lambda row: (
            row.tier != "open",
            row.price is None,
            row.price if row.price is not None else 0.0,
            row.model,
        ),
    )


def _distill_reason(
    *,
    student: str,
    teacher: ModelGain,
    best: ModelGain,
    min_gap: float,
    sufficiency: float,
    basis: PriceBasis,
) -> str:
    """The printed reason behind a `distill` verdict, teacher choice included."""
    kept = teacher.mean_gain / best.mean_gain * 100.0
    if teacher.tier == "open":
        tier_clause = (
            "It is open weights, which is the tier this gate prefers for a data teacher: a "
            "frontier model can prove the gap exists, but you have to be allowed to train on "
            "what the teacher writes."
        )
    else:
        tier_clause = (
            "No open-weights candidate kept enough of the gain, so this teacher is frontier "
            "tier: check that provider's terms before training on its outputs."
        )
    return (
        f"DISTILL: '{best.model}' beats '{student}' by {_points(best.mean_gain)} points on "
        f"{best.n_scenarios} shared scenarios ({_interval(best)}), clearing the "
        f"{_bar(min_gap)}-point bar this gate requires, so there is something to teach. "
        f"The cheapest sufficient teacher is '{teacher.model}' at {_price(teacher.price, basis)}, "
        f"keeping {kept:.0f}% of the best measured gain (the bar is "
        f"{sufficiency * 100:.0f}%). {tier_clause}"
    )


def _no_gap_reason(student: str, leader: ModelGain, min_gap: float) -> str:
    """The printed reason behind a `do_not_distill` verdict."""
    caveat = ""
    if leader.mean_gain >= min_gap:
        caveat = (
            " Its point gain clears the bar, but the interval includes zero, which is not a gap; "
            "more scenarios would settle it."
        )
    return (
        f"DO NOT DISTILL: no model in this matrix beats '{student}' by the {_bar(min_gap)} points "
        f"this gate requires. The best is '{leader.model}' at {_points(leader.mean_gain)} points "
        f"on {leader.n_scenarios} shared scenarios ({_interval(leader)}).{caveat} '{student}' "
        f"already serves this workload and earns its traffic on price, so training has no teacher "
        f"to learn from and this stage skips with zero spend."
    )


def _thin_reason(student: str, leader: ModelGain, min_gap: float, min_scenarios: int) -> str:
    """The printed reason when the leading candidate was measured too thinly to test."""
    return (
        f"INSUFFICIENT EVIDENCE: the leading candidate '{leader.model}' is "
        f"{_points(leader.mean_gain)} points over '{student}' but shares only "
        f"{leader.n_scenarios} scored scenario(s) with it, below the {min_scenarios} this gate "
        f"requires before it calls a {_bar(min_gap)}-point gap real or absent. Sweep more "
        f"scenarios, then probe again."
    )


def _points(gain: float) -> str:
    """A paired gain in signed reward points, the unit these results are quoted in."""
    return f"{gain * 100:+.1f}"


def _bar(threshold: float) -> str:
    """A threshold in unsigned reward points."""
    return f"{threshold * 100:.1f}"


def _interval(row: ModelGain) -> str:
    """A gain row's 95% interval, or why it has none."""
    if row.ci_low is None or row.ci_high is None:
        return "no interval: one shared scenario"
    return f"95% CI {_points(row.ci_low)} to {_points(row.ci_high)} points"


def _price(price: float | None, basis: PriceBasis) -> str:
    """A ladder key rendered in the unit its basis actually means."""
    if price is None:
        return "no price on its pool entry"
    if basis == "measured":
        return f"${price:.4f} per completed task on this matrix"
    return f"${price:.2f} per 1M tokens (list, input + output)"

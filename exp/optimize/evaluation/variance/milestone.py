"""Preregistered comparison of judge-chosen points against cheap checkpoint selection rules.

The milestone question is whether judge and critic point selection earns its cost. Every strategy
chooses the same number ``K`` of points from the same picked trajectory of the same group that an
actor-critic run already measured, and every chosen point then goes through the same downstream
test: a decision-state author describes the decision there, and every authored option is forced
from that point's checkpoint by ``contrast.run_option_contrast``, every cell judged by the
actor-critic run's frozen episode judge. A point counts as a
**separated decision** only when some option's conservative lower bound is above every other
option's upper bound.

Strategies:

- ``judge``: the first ``K`` points of the actor-critic report's ranking, which orders the points
  the judges chose by their screening continuations.
- ``drop``: the judge-chosen points ordered by the screening rate drop that follows them. It only
  reranks the judges' points, so it carries the judges' full selection cost.
- ``uniform``: a seeded uniform sample of ``K`` continuable points of the picked trajectory.
- ``late_failure``: the continuable points closest before the first tool turn with a failed call,
  or before the end of the trajectory when no call failed, then the nearest later points.
- ``midpoint``: the continuable points closest to the middle of the trajectory.

The heuristics read only what the group itself recorded, never a screening continuation or a
contrast. The decision-state author is shown the same neutral description of a point whatever
strategy chose it, so the comparison measures where to look, not the judges' prose; a point
chosen by several strategies is authored and contrasted once and counted for each of them.

Preregistration: the comparison spec and every strategy's points are saved before any draw, a
budget any strategy cannot meet is refused before then, each point's outcome (failures
included) is saved once and never redrawn, and the finished comparison replays exactly.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import unit_difference_summary
from exp.common.models import ModelResponse, ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob, load_continuation_runtime
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationBranch,
    restore_continuation_world,
)
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.authors import (
    DecisionContext,
    DecisionStateDraft,
    author_decision_state,
    authors_protocol_sha256,
)
from exp.optimize.evaluation.variance.contracts import DecisionState, EstimationPoint
from exp.optimize.evaluation.variance.contrast import OptionContrast, run_option_contrast
from exp.optimize.evaluation.variance.evidence import evidence_packet
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint

SelectionStrategy = Literal["judge", "drop", "uniform", "late_failure", "midpoint"]
"""How a strategy chooses its points; see the module docstring."""

Verdict = Literal["judge_better", "not_shown"]
"""``judge_better`` only when the judges beat every baseline beyond sampling noise."""

OptionMapping = Callable[[DecisionState, ContinuationCheckpoint], Mapping[str, ToolCall]]
"""Maps an authored decision's options to concrete forced first actions at a checkpoint."""

_OPTION_MAPPINGS: dict[str, OptionMapping] = {}

VERDICT_RULE = (
    "Each independent unit is one task lineage; its value under a strategy is the fraction of "
    "that strategy's K points that separated, failures counted as not separated. For every "
    "baseline the paired difference judge minus baseline is averaged over units and bounded by "
    "the two-sided 95 percent Wilson interval at the unit count (unit_difference_summary). "
    "Judge selection is reported better only when that lower bound is above zero for every "
    "baseline at once (an intersection-union test, so no multiplicity correction is needed); "
    "otherwise the verdict is not_shown."
)


def register_option_mapping(name: str, mapping: OptionMapping) -> None:
    """Register a named option mapping that comparison specs may reference.

    Args:
        name: Name a spec's ``option_mapping`` uses.
        mapping: The mapping; registering the same callable again is allowed.

    Raises:
        ValueError: The name is already registered to a different mapping.
    """
    existing = _OPTION_MAPPINGS.get(name)
    if existing is not None and existing is not mapping:
        raise ValueError(f"option mapping {name!r} is already registered")
    _OPTION_MAPPINGS[name] = mapping


class SelectionComparisonSpec(ContractModel):
    """One preregistered comparison over a saved actor-critic run.

    Attributes:
        strategies: Strategies compared; ``judge`` and at least one baseline, each once.
        points: Points every strategy must choose, the equal budget ``K``.
        repeats: Pairs per option contrast at each chosen point.
        seed: Seed of the ``uniform`` sample.
        author: Frozen decision-state author settings.
        option_actions: Forced first action per option identity, applied to every decision.
        option_mapping: Name of a registered option mapping; exactly one of this and
            ``option_actions`` is set.
        contrast_physical_attempts: Transport attempt ceiling for each contrast's suffixes.
    """

    strategies: tuple[SelectionStrategy, ...] = Field(min_length=2, max_length=5)
    points: int = Field(default=2, ge=1, le=8, strict=True)
    repeats: int = Field(default=4, ge=1, le=100, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    author: AuthorSettings
    option_actions: dict[str, ToolCall] | None = None
    option_mapping: str | None = None
    contrast_physical_attempts: int = Field(default=64, ge=1, le=10000, strict=True)

    @model_validator(mode="after")
    def _registered(self) -> Self:
        """Require the judges, a baseline, distinct strategies and exactly one mapping."""
        if "judge" not in self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must include judge once and each baseline at most once")
        if (self.option_actions is None) == (self.option_mapping is None):
            raise ValueError("set exactly one of option_actions and option_mapping")
        return self

    def identity(self, report: ActorCriticReport, code_revision: str) -> str:
        """Bind the registration, the measured run, the author prompts and producing code.

        Args:
            report: The actor-critic run whose trajectory is compared.
            code_revision: Producing source identity.

        Returns:
            Stable identity of the comparison.
        """
        return stable_id(
            "selection-comparison",
            {
                "spec": sha256_json(self),
                "report": sha256_json(report),
                "authors": authors_protocol_sha256(),
                "code": code_revision,
            },
        )


class SelectionPlan(ContractModel):
    """The preregistered points of every strategy, saved before any draw.

    Attributes:
        spec: The registration.
        report: Digest of the actor-critic report compared.
        lineage: Leakage lineage of the task, the independent unit.
        group: The saved group of attempts.
        picked: The picked attempt all strategies choose from.
        pool: Every continuable point of the picked attempt.
        points: Chosen points per strategy, ascending.
    """

    spec: SelectionComparisonSpec
    report: str
    lineage: str
    group: ArtifactInput
    picked: int
    pool: tuple[int, ...]
    points: dict[str, tuple[int, ...]]


class PointTrial(ContractModel):
    """The downstream test of one chosen point, saved once whatever happened.

    Attributes:
        point: Point number within the picked attempt.
        checkpoint: Checkpoint the options were forced from.
        draft: The decision-state draw, when one was made.
        contrast: The option contrast, when it ran.
        separated: Some option's lower bound is above every other option's upper bound.
        leader: Option with the highest lower bound, when contrasted.
        failure: Why the point has no contrast.
        author_calls: Decision-state role calls made for this point.
        continuation_draws: Planned forced-option continuations at this point.
    """

    point: int = Field(ge=0)
    checkpoint: ArtifactInput
    draft: DecisionStateDraft | None = None
    contrast: OptionContrast | None = None
    separated: bool = False
    leader: str | None = None
    failure: str | None = None
    author_calls: int = Field(default=0, ge=0)
    continuation_draws: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_outcome(self) -> Self:
        """Keep exactly one of contrast and failure, and separation only with a contrast."""
        if (self.contrast is None) == (self.failure is None):
            raise ValueError("a point trial has exactly one of contrast and failure")
        if self.separated and self.contrast is None:
            raise ValueError("only a contrasted point can separate")
        return self


class StrategyResult(ContractModel):
    """What one strategy's points yielded and what they cost.

    Attributes:
        strategy: The strategy.
        points: Its chosen points.
        separated: Points whose options separated.
        unseparated: Contrasted points whose options did not separate.
        failed: Points with no contrast.
        selection_role_calls: Judge and critic draws spent choosing the points.
        selection_continuations: Screening continuations spent choosing the points.
        author_calls: Decision-state role calls over its points.
        contrast_draws: Planned forced-option continuations over its points.
        total_cost: Sum of every role call and continuation draw above.
        yield_per_cost: Separated points per unit of total cost, or None at zero cost.
    """

    strategy: SelectionStrategy
    points: tuple[int, ...]
    separated: tuple[int, ...]
    unseparated: tuple[int, ...]
    failed: tuple[int, ...]
    selection_role_calls: int = Field(ge=0)
    selection_continuations: int = Field(ge=0)
    author_calls: int = Field(ge=0)
    contrast_draws: int = Field(ge=0)
    total_cost: int = Field(ge=0)
    yield_per_cost: float | None


class SelectionComparison(ContractModel):
    """The complete record of one comparison, failures included.

    Attributes:
        plan: The preregistered points.
        registration: The plan as saved before any draw.
        trials: Downstream test of every distinct chosen point, in point order.
        strategies: Per-strategy result, in spec order.
    """

    plan: SelectionPlan
    registration: ArtifactInput
    trials: tuple[PointTrial, ...]
    strategies: tuple[StrategyResult, ...]


class StrategyTotal(ContractModel):
    """One strategy pooled over comparisons.

    Attributes:
        strategy: The strategy.
        chosen: Points chosen.
        separated: Separated decision points.
        unseparated: Contrasted points that did not separate.
        failed: Points with no contrast.
        total_cost: Role calls and continuation draws spent.
        yield_per_cost: Separated points per unit of cost, or None at zero cost.
    """

    strategy: SelectionStrategy
    chosen: int
    separated: int
    unseparated: int
    failed: int
    total_cost: int
    yield_per_cost: float | None


class BaselineDifference(ContractModel):
    """Judge minus one baseline in separated fraction, paired by unit.

    Attributes:
        baseline: The baseline strategy.
        units: Independent units both strategies were measured on.
        estimate: Mean per-unit difference, or None without a unit.
        wilson_95: Two-sided 95 percent interval of the difference, or None without a unit.
        exceeds_noise: The interval's lower bound is above zero.
    """

    baseline: SelectionStrategy
    units: int
    estimate: float | None
    wilson_95: tuple[float, float] | None
    exceeds_noise: bool


class Comparison(ContractModel):
    """Pooled per-strategy totals and the conservative verdict.

    Attributes:
        totals: Per strategy, in spec order.
        differences: Judge against each baseline.
        verdict: ``judge_better`` only when every difference exceeds noise.
        rule: The preregistered verdict rule.
    """

    totals: tuple[StrategyTotal, ...]
    differences: tuple[BaselineDifference, ...]
    verdict: Verdict
    rule: str = VERDICT_RULE


def run_selection_comparison(
    project: ProjectStore,
    spec: SelectionComparisonSpec,
    *,
    exposure: ExposurePlan,
    report: ActorCriticReport,
    code_revision: str,
) -> SelectionComparison:
    """Run, or replay, one preregistered selection comparison.

    Args:
        project: Project holding the actor-critic run and every saved step.
        spec: The registration.
        exposure: The frozen data roles; the task must come from a mining-probe lineage.
        report: The saved actor-critic run whose picked trajectory is compared.
        code_revision: Producing source identity.

    Returns:
        The comparison, also saved as a ``selection-comparison`` artifact.

    Raises:
        ValueError: The task is not a mining probe, a strategy cannot choose ``K`` points, the
            option mapping is not registered, or a saved plan differs from this registration.
    """
    spec = SelectionComparisonSpec.model_validate_json(spec.model_dump_json())
    store = project.artifacts
    start = load_continuation_checkpoint(store, report.spec.initial.study.checkpoint)
    exposure.require_lineage(start.task.lineage_group_id, "mining_probes")
    if spec.option_mapping is not None and spec.option_mapping not in _OPTION_MAPPINGS:
        raise ValueError(f"option mapping {spec.option_mapping!r} is not registered")
    identity = spec.identity(report, code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return SelectionComparison.model_validate_json(
                store.read_bytes(result_id, "comparison.json")
            )
        loader = cache(lambda ref: load_continuation_checkpoint(store, ref))
        group = load_study(store, report.group)
        pool = evidence_packet(
            group,
            report.picked,
            maximum_model_calls=report.spec.initial.maximum_model_calls,
            checkpoint=loader,
        ).pool
        branch = next(row for row in group.branches if row.index == report.picked)
        plan = SelectionPlan(
            spec=spec,
            report=sha256_json(report),
            lineage=start.task.lineage_group_id,
            group=report.group,
            picked=report.picked,
            pool=tuple(sorted(pool.points)),
            points={
                strategy: choose_points(strategy, spec, report, branch, tuple(sorted(pool.points)))
                for strategy in spec.strategies
            },
        )
        registration = _register(project, identity, plan, report, pool.points, code_revision)
        turns = _turns(branch)
        chosen = sorted({point for points in plan.points.values() for point in points})
        trials = tuple(
            _trial(
                project,
                spec,
                report.spec.initial,
                report.spec.episode_judge,
                identity,
                point,
                pool.points[point],
                turns,
                loader,
                code_revision,
            )
            for point in chosen
        )
        by_point = {trial.point: trial for trial in trials}
        result = SelectionComparison(
            plan=plan,
            registration=registration,
            trials=trials,
            strategies=tuple(
                _strategy_result(strategy, plan.points[strategy], by_point, report)
                for strategy in spec.strategies
            ),
        )
        inputs = {registration}
        for trial in trials:
            inputs |= set(trial.contrast.contrasts if trial.contrast else ())
            if trial.draft is not None and trial.draft.response is not None:
                inputs.add(trial.draft.response)
        store.write_json(
            artifact_id=result_id,
            artifact_type="selection-comparison",
            envelope=_envelope(code_revision, inputs),
            files={"comparison.json": result},
        )
        return result


def choose_points(
    strategy: SelectionStrategy,
    spec: SelectionComparisonSpec,
    report: ActorCriticReport,
    branch: ContinuationBranch,
    pool: tuple[int, ...],
) -> tuple[int, ...]:
    """Choose one strategy's ``K`` points from the picked trajectory's continuable points.

    Args:
        strategy: The strategy.
        spec: The registration supplying ``K`` and the seed.
        report: The actor-critic run, read only by ``judge`` and ``drop``.
        branch: The picked attempt, read only for its recorded tool turns.
        pool: Continuable point numbers of the picked attempt, ascending.

    Returns:
        Exactly ``K`` distinct points, ascending.

    Raises:
        ValueError: The strategy has fewer than ``K`` points to choose from.
    """
    k = spec.points
    order: list[int]
    if strategy == "judge":
        order = [point for point in report.ranking if point in pool]
    elif strategy == "drop":
        drops = report.drops
        order = sorted(
            (row.point for row in report.points if row.point in pool),
            key=lambda point: (-drops.get(f"point-{point}", float("-inf")), point),
        )
    elif strategy == "uniform":
        if len(pool) < k:
            order = list(pool)
        else:
            seed = f"selection-uniform:{report.group.artifact_id}:{report.picked}:{spec.seed}"
            order = random.Random(seed).sample(list(pool), k)
    elif strategy == "late_failure":
        failed = next((index for index, (_, bad) in enumerate(_turns(branch)) if bad), None)
        anchor = max(pool) if failed is None else failed
        order = sorted(pool, key=lambda point: (point > anchor, abs(point - anchor)))
    else:
        middle = (pool[0] + pool[-1]) / 2 if pool else 0.0
        order = sorted(pool, key=lambda point: (abs(point - middle), point))
    if len(order) < k:
        raise ValueError(
            f"strategy {strategy} offers {len(order)} points, fewer than the budget of {k}; "
            "every strategy must choose the same number"
        )
    return tuple(sorted(order[:k]))


def separation(contrast: OptionContrast) -> tuple[bool, str]:
    """Decide whether one contrast separates its options.

    Args:
        contrast: The option contrast.

    Returns:
        Whether the best lower bound is above every other option's upper bound, and that
        leading option.
    """
    leader = max(contrast.options, key=lambda option: option.bounds[0])
    others = [option.bounds[1] for option in contrast.options if option is not leader]
    return leader.bounds[0] > max(others), leader.option_id


def compare(results: Sequence[SelectionComparison]) -> Comparison:
    """Pool comparisons and apply the preregistered conservative verdict.

    See :data:`VERDICT_RULE`. Each result's task lineage is one independent unit; several results
    from one lineage are averaged within it rather than counted as new evidence.

    Args:
        results: Finished comparisons sharing strategies and budget.

    Returns:
        Per-strategy totals, judge-minus-baseline differences and the verdict.

    Raises:
        ValueError: No results, or results registered different strategies or budgets.
    """
    if not results:
        raise ValueError("compare needs at least one comparison")
    first = results[0].plan.spec
    if any(
        (r.plan.spec.strategies, r.plan.spec.points) != (first.strategies, first.points)
        for r in results
    ):
        raise ValueError("comparisons must register the same strategies and budget")
    rows = {
        strategy: [next(s for s in r.strategies if s.strategy == strategy) for r in results]
        for strategy in first.strategies
    }
    totals = tuple(_total(strategy, rows[strategy]) for strategy in first.strategies)
    fractions: dict[str, dict[str, list[float]]] = {s: {} for s in first.strategies}
    for result in results:
        for row in result.strategies:
            fraction = len(row.separated) / len(row.points)
            fractions[row.strategy].setdefault(result.plan.lineage, []).append(fraction)
    differences = tuple(
        _difference(baseline, fractions["judge"], fractions[baseline])
        for baseline in first.strategies
        if baseline != "judge"
    )
    verdict: Verdict = "judge_better" if all(d.exceeds_noise for d in differences) else "not_shown"
    return Comparison(totals=totals, differences=differences, verdict=verdict)


def _register(
    project: ProjectStore,
    identity: str,
    plan: SelectionPlan,
    report: ActorCriticReport,
    checkpoints: Mapping[int, ArtifactInput],
    code_revision: str,
) -> ArtifactInput:
    """Save the plan before any draw, or confirm an earlier save is the same plan.

    Raises:
        ValueError: A saved plan under this identity differs.
    """
    store = project.artifacts
    if store.exists(identity):
        saved = SelectionPlan.model_validate_json(store.read_bytes(identity, "plan.json"))
        if saved != plan:
            raise ValueError("saved selection plan differs from this registration")
        return artifact_input(store.read(identity).manifest)
    inputs = {report.group, *(checkpoints[p] for ps in plan.points.values() for p in ps)}
    reported = report.spec.identity(code_revision) + "-result"
    if store.exists(reported):
        inputs.add(artifact_input(store.read(reported).manifest))
    return artifact_input(
        store.write_json(
            artifact_id=identity,
            artifact_type="selection-comparison-plan",
            envelope=_envelope(code_revision, inputs),
            files={"plan.json": plan},
        )
    )


def _trial(
    project: ProjectStore,
    spec: SelectionComparisonSpec,
    job: ContinuationJob,
    judge: EpisodeJudge,
    identity: str,
    point: int,
    checkpoint: ArtifactInput,
    turns: tuple[tuple[tuple[ToolCall, ...], bool], ...],
    loader: Callable[[ArtifactInput], ContinuationCheckpoint],
    code_revision: str,
) -> PointTrial:
    """Author and contrast one point once; a saved trial replays and is never redrawn."""
    store = project.artifacts
    trial_id = f"{identity}-point-{point}"
    if store.exists(trial_id):
        return PointTrial.model_validate_json(store.read_bytes(trial_id, "trial.json"))
    trial = _run_trial(
        project, spec, job, judge, identity, point, checkpoint, turns, loader, code_revision
    )
    inputs = {checkpoint}
    if trial.draft is not None and trial.draft.response is not None:
        inputs.add(trial.draft.response)
    if trial.contrast is not None:
        inputs |= set(trial.contrast.contrasts)
    store.write_json(
        artifact_id=trial_id,
        artifact_type="selection-point-trial",
        envelope=_envelope(code_revision, inputs),
        files={"trial.json": trial},
    )
    return trial


def _run_trial(
    project: ProjectStore,
    spec: SelectionComparisonSpec,
    job: ContinuationJob,
    judge: EpisodeJudge,
    identity: str,
    point: int,
    checkpoint: ArtifactInput,
    turns: tuple[tuple[tuple[ToolCall, ...], bool], ...],
    loader: Callable[[ArtifactInput], ContinuationCheckpoint],
    code_revision: str,
) -> PointTrial:
    """Author the decision at one point and force every option from its checkpoint."""
    original = loader(checkpoint)
    bound = _contrast_job(job, checkpoint, spec.contrast_physical_attempts)
    try:
        before = _before(project, bound, original)
        context = DecisionContext(
            point=_neutral_point(point, turns),
            checkpoint=checkpoint,
            prefix=original.actor.messages,
            before=before,
            tools=original.task.tools,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return PointTrial(point=point, checkpoint=checkpoint, failure=_reason("context", error))
    draft = author_decision_state(
        project,
        spec.author,
        context,
        identity=f"{identity}-author-{point}",
        code_revision=code_revision,
        sources=(checkpoint,),
    )
    if draft.answer is None:
        return PointTrial(
            point=point,
            checkpoint=checkpoint,
            draft=draft,
            failure=f"author: {draft.failure}"[:2000],
            author_calls=1,
        )
    decision = draft.answer.state
    try:
        actions = (
            dict(spec.option_actions)
            if spec.option_actions is not None
            else dict(_OPTION_MAPPINGS[str(spec.option_mapping)](decision, original))
        )
        contrast = run_option_contrast(
            project,
            checkpoint=checkpoint,
            decision=decision,
            actions=actions,
            job=bound,
            repeats=spec.repeats,
            judge=judge,
            code_revision=code_revision,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return PointTrial(
            point=point,
            checkpoint=checkpoint,
            draft=draft,
            failure=_reason("contrast", error),
            author_calls=1,
        )
    separated, leader = separation(contrast)
    return PointTrial(
        point=point,
        checkpoint=checkpoint,
        draft=draft,
        contrast=contrast,
        separated=separated,
        leader=leader,
        author_calls=1,
        continuation_draws=sum(option.planned for option in contrast.options),
    )


def _contrast_job(
    job: ContinuationJob, checkpoint: ArtifactInput, attempts: int
) -> ContinuationJob:
    """Start the run's frozen actor job at one point with no screening plan attached."""
    study = job.study.model_copy(
        update={
            "checkpoint": checkpoint,
            "critic_prediction": None,
            "predicted_success": None,
            "selection_plan": None,
        }
    )
    return ContinuationJob.model_validate_json(
        job.model_copy(
            update={"study": study, "maximum_physical_attempts": attempts}
        ).model_dump_json()
    )


def _before(
    project: ProjectStore, job: ContinuationJob, original: ContinuationCheckpoint
) -> JsonObject:
    """Probe the evaluator state of the restored world at a checkpoint, without any model."""
    with (
        load_continuation_runtime(project, job) as runtime,
        restore_continuation_world(runtime.world, original.world_task, original.world) as world,
    ):
        return runtime.probe(world.checkpoint())


def _neutral_point(
    point: int, turns: tuple[tuple[tuple[ToolCall, ...], bool], ...]
) -> EstimationPoint:
    """Describe a point the same way whichever strategy chose it."""
    if point < len(turns):
        calls = [{"tool": call.name, "arguments": call.arguments} for call in turns[point][0]]
        decision = "The actor's next tool turn: " + json.dumps(calls, sort_keys=True)
    else:
        decision = "The actor's next action after this point, which ended its attempt."
    return EstimationPoint(
        step=point,
        decision=decision[:600],
        alternatives=("A different next action at the same point",),
        deciding_state=(),
        rationale="Chosen for decision-state authoring; the choosing strategy is withheld.",
        evidence_refs=("point",),
    )


def _turns(branch: ContinuationBranch) -> tuple[tuple[tuple[ToolCall, ...], bool], ...]:
    """Return each recorded tool turn's calls and whether any of them failed.

    Turn ``t`` (zero-based) is the turn taken from point ``t``. A step that was refused, not
    observed, or observed as an error counts as failed.

    Args:
        branch: The picked attempt.

    Returns:
        One entry per tool turn the saved outcome records, in order.
    """
    receipts, steps = branch.outcome.get("model_calls"), branch.outcome.get("steps")
    if not isinstance(receipts, list) or not isinstance(steps, list):
        return ()
    turns: list[tuple[tuple[ToolCall, ...], bool]] = []
    consumed = 0
    for receipt in receipts:
        if not isinstance(receipt, dict) or "response" not in receipt:
            break
        calls = ModelResponse.model_validate(receipt["response"]).output.tool_calls
        if not calls:
            continue
        taken = steps[consumed : consumed + len(calls)]
        consumed += len(calls)
        bad = len(taken) < len(calls) or any(_failed(step) for step in taken)
        turns.append((tuple(calls), bad))
    return tuple(turns)


def _failed(step: JsonValue) -> bool:
    """Return whether one saved step was refused, unobserved or observed as an error."""
    if not isinstance(step, dict) or step.get("status") != "valid":
        return True
    reply = step.get("reply")
    return not isinstance(reply, dict) or reply.get("is_error") is True


def _strategy_result(
    strategy: SelectionStrategy,
    points: tuple[int, ...],
    trials: Mapping[int, PointTrial],
    report: ActorCriticReport,
) -> StrategyResult:
    """Count one strategy's outcomes and its standalone cost."""
    rows = [trials[point] for point in points]
    judged = strategy in ("judge", "drop")
    role_calls = len(report.judges) + len(report.critics) if judged else 0
    screening = report.spec.continuations * len(report.points) if judged else 0
    authors = sum(row.author_calls for row in rows)
    draws = sum(row.continuation_draws for row in rows)
    separated = tuple(row.point for row in rows if row.separated)
    total = role_calls + screening + authors + draws
    return StrategyResult(
        strategy=strategy,
        points=points,
        separated=separated,
        unseparated=tuple(row.point for row in rows if row.contrast and not row.separated),
        failed=tuple(row.point for row in rows if row.contrast is None),
        selection_role_calls=role_calls,
        selection_continuations=screening,
        author_calls=authors,
        contrast_draws=draws,
        total_cost=total,
        yield_per_cost=None if total == 0 else len(separated) / total,
    )


def _total(strategy: SelectionStrategy, rows: Sequence[StrategyResult]) -> StrategyTotal:
    """Pool one strategy's results."""
    separated = sum(len(row.separated) for row in rows)
    cost = sum(row.total_cost for row in rows)
    return StrategyTotal(
        strategy=strategy,
        chosen=sum(len(row.points) for row in rows),
        separated=separated,
        unseparated=sum(len(row.unseparated) for row in rows),
        failed=sum(len(row.failed) for row in rows),
        total_cost=cost,
        yield_per_cost=None if cost == 0 else separated / cost,
    )


def _difference(
    baseline: SelectionStrategy,
    judge: Mapping[str, Sequence[float]],
    other: Mapping[str, Sequence[float]],
) -> BaselineDifference:
    """Bound judge minus one baseline over paired units."""
    summary = unit_difference_summary(judge, other)
    units, estimate, interval = (
        summary["independent_units"],
        summary["estimate"],
        summary["wilson_95"],
    )
    assert isinstance(units, int)
    bounds = None
    if isinstance(interval, list):
        low, high = interval
        assert isinstance(low, float) and isinstance(high, float)
        bounds = (low, high)
    return BaselineDifference(
        baseline=baseline,
        units=units,
        estimate=estimate if isinstance(estimate, float) else None,
        wilson_95=bounds,
        exceeds_noise=bounds is not None and bounds[0] > 0.0,
    )


def _reason(phase: str, error: Exception) -> str:
    """Name a point failure's phase and error without raw provider payloads."""
    return f"{phase}: {type(error).__name__}: {error}"[:2000]


def _envelope(code_revision: str, inputs: set[ArtifactInput]) -> ArtifactEnvelope:
    """Return an envelope over sorted inputs."""
    return ArtifactEnvelope(
        schema_version=1,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
        inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
    )

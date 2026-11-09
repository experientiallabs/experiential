"""The point qualification bar and blast radius, wired into the TraceML and error-mining stages.

- **Bar tagging** (``traceml``): under ``mining.qualification`` every step of every tier
  attempt is answered against the bar's bank by the decision model (Clef) through the unchanged
  tagging seam and its cache (:func:`tag_tier_bar`); :func:`~.point_bar.qualify_points` reads
  each attempt into a :class:`~.point_bar.BarRecord`, saved per tier as ``point-bar-records``.
  A refused tagging pass leaves the attempt's points uncertain, never qualified.
- **Selection and schedule** (``error-mining``): each tier's actor-critic registration carries a
  :class:`~.point_selection.PointGate` (:func:`point_gate`), so selectors justify picks against
  the bar and chosen points continue while their reservations fit the phase.
- **Planning**: with no fixed ``K``, a tier reserves continuations for at most the qualified
  points of its full trajectory (the registered call horizon), capped by ``phase_attempts``
  (:func:`qualified_bound`); the bar tagging is priced as decisions (:func:`bar_tag_costs`).
- **Blast radius** (``error-mining``, after every tier): :func:`mine_blast_radius` combines the
  tiers' continuations per checkpoint, judges each point's failure severity once per stance when
  reviewers are registered, scores and ranks, and saves the ranking that family selection reads
  through :func:`~.blast_radius.order_by_blast`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.executable.task import MiningProvenance
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, admitted_concurrency
from exp.optimize.evaluation.variance.behavior_adversarial import hack_signature
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.blast_radius import (
    BlastRadius,
    CrossTierBlast,
    CrossTierPoint,
    SeverityJudgment,
    TierOutcome,
    blast_radius,
    cross_tier_point,
    judge_severity,
    order_by_blast,
    rank_blast,
    save_blast_radius,
    save_cross_tier,
    tier_outcomes,
)
from exp.optimize.evaluation.variance.bounded import run_at_once
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, episode_reward
from exp.optimize.evaluation.variance.family_stages import slot_evidence
from exp.optimize.evaluation.variance.judge import load_assessment
from exp.optimize.evaluation.variance.judged import judge_study, slot_rewards
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    MiningOutputs,
    PointQualification,
    StageContext,
    TierMining,
)
from exp.optimize.evaluation.variance.point_bar import BarRecord, QualificationBar, qualify_points
from exp.optimize.evaluation.variance.point_selection import PointGate, save_point_bar
from exp.optimize.evaluation.variance.records import load_judged_study
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.questions import resolve_question_bank
from exp.simulation.mining.tagging import TraceTagger, TraceTaggingError
from exp.simulation.mining.traceml import TraceTrajectory, step_units

logger = logging.getLogger(__name__)

QUESTIONS_PER_CALL = 32
"""Questions the decision model answers per step in one call."""


def qualification_bar(settings: PointQualification, bank_root: Path | None) -> QualificationBar:
    """Return the bar: an authored bank adopted with its required rules, or the given bar.

    Raises:
        ValueError: The authored bank names no required rule or is missing.
    """
    if settings.bank is None:
        return settings.bar
    if not settings.required:
        raise ValueError("an adopted qualification bank needs its required rules")
    bank = resolve_question_bank(settings.bank, root=bank_root)
    return QualificationBar.from_bank(bank, settings.required)


def tag_tier_bar(
    store: ArtifactStore,
    bar: QualificationBar,
    trajectories: Sequence[TraceTrajectory],
    tagger: TraceTagger,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput],
) -> ArtifactInput:
    """Answer the bar over every step of every attempt and save each attempt's qualification.

    Attempt traces are named ``<tier>/attempt:<n>``; records are keyed by ``n``.

    Returns:
        The saved ``point-bar-records``.
    """
    bank = bar.bank()
    records: dict[int, BarRecord] = {}
    for trajectory in trajectories:
        index = int(trajectory.trace_id.rsplit(":", 1)[-1])
        units = step_units(trajectory)
        try:
            matrix = tagger.tag(units, bank) if units else None
        except (TraceTaggingError, ProviderTransportError):
            matrix = None
        records[index] = qualify_points(trajectory, matrix, bar)
    return save_point_bar(store, bar, records, code_revision=code_revision, inputs=inputs)


def bar_tag_costs(project: ProjectStore, spec: DecisionVarianceSpec) -> tuple[CostComponent, ...]:
    """Price answering the bar over every step of every tier attempt as decisions."""
    settings = spec.mining.qualification
    if settings is None:
        return ()
    questions = len(qualification_bar(settings, spec.traceml.bank_root).rules)
    per_step = math.ceil(questions / QUESTIONS_PER_CALL)
    steps = spec.mining.registered.maximum_model_calls * spec.traceml.steps_per_call
    calls = spec.mining.attempts * steps * per_step * RetryPolicy().maximum_attempts
    return tuple(
        decision_cost(project, spec.traceml.decision_alias, calls) for _ in spec.mining.actors
    )


def cost_per_point(spec: DecisionVarianceSpec) -> int:
    """Physical attempts one continued point reserves: M continuations at the call horizon."""
    mining = spec.mining
    return mining.continuations * mining.registered.maximum_model_calls


def qualified_bound(spec: DecisionVarianceSpec) -> int:
    """Most points one tier may continue this phase, with no fixed per-selector count.

    Every boundary of the picked attempt's whole trajectory can qualify, so the bound is the
    registered call horizon, capped by what ``phase_attempts`` reserves.
    """
    settings = spec.mining.qualification
    assert settings is not None
    bound = spec.mining.registered.maximum_model_calls
    if settings.phase_attempts is not None:
        bound = min(bound, settings.phase_attempts // cost_per_point(spec))
    return bound


def point_gate(
    spec: DecisionVarianceSpec, records: ArtifactInput, allowance: int | None = None
) -> PointGate:
    """Return one tier's gate over its saved qualification records.

    Args:
        spec: The registration.
        records: The tier's saved qualification records.
        allowance: The tier's share of the live cumulative ledger; the registered
            ``phase_attempts`` applies without a live ledger.
    """
    settings = spec.mining.qualification
    assert settings is not None
    return PointGate(
        bar=qualification_bar(settings, spec.traceml.bank_root),
        records=records,
        cost_per_point=cost_per_point(spec),
        phase_attempts=settings.phase_attempts if allowance is None else allowance,
    )


def _deferred_path(project: ProjectStore, spec: DecisionVarianceSpec) -> Path:
    """Return the file recording each tier's deferred points for one mining registration."""
    digest = sha256_json(
        {
            "mining": spec.mining.model_dump(mode="json"),
            "traceml": spec.traceml.model_dump(mode="json"),
        }
    )[:24]
    return project.paths.runtime_directory / "decision-variance" / "deferred" / f"{digest}.json"


def read_deferred(project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
    """Return each tier's deferred points as last recorded; empty before the first phase.

    The mining stage's identity reads it, so a phase that deferred points runs the stage again
    on the next pass and carries them into that phase automatically.
    """
    path = _deferred_path(project, spec)
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def write_deferred(
    project: ProjectStore, spec: DecisionVarianceSpec, reports: Sequence[ActorCriticReport]
) -> None:
    """Record each tier's still-deferred points after a mining phase."""
    record: JsonObject = {
        report.spec.actor_tier: list(report.point_schedule.deferred)
        for report in reports
        if report.point_schedule is not None
    }
    path = _deferred_path(project, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True))


def _failures(project: ProjectStore, judged: ArtifactInput | None) -> list[EpisodeEvidence]:
    """Return the judged failing episodes of one point's continuations."""
    if judged is None:
        return []
    study = load_judged_study(project, judged)
    found: list[EpisodeEvidence] = []
    for slot in study.branches:
        if slot.episode is None:
            continue
        if episode_reward(load_assessment(project, slot.episode.assessment)) != 0.0:
            continue
        evidence = slot_evidence(project, slot, study.judge)
        if evidence is not None:
            found.append(evidence)
    return found


def mine_blast_radius(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    tiers: Sequence[TierMining],
    *,
    code_revision: str,
    inputs: tuple[ArtifactInput, ...],
) -> tuple[ArtifactInput, tuple[BlastRadius, ...], ArtifactInput | None] | None:
    """Measure, judge, score and rank every continued point across tiers, then save it.

    Each tier first continues only its own points. Under ``blast.cross_tier`` the highest
    ranked points of every tier are then continued by each of the other tiers from the very
    same checkpoint (:func:`cross_tier_studies`), so a point's failing share, ``ever_passed``
    and score are read across every tier, and the per-tier shares and recoveries are saved as
    a ``cross-tier-blast`` record. Severity is judged once per point, from its origin's
    failing episodes, and reused.

    Returns:
        The saved ranking, the ranked rows and the cross-tier record (None when off), or None
        when no severity reviewers are registered (a point without a judged severity is never
        scored).
    """
    settings = spec.mining.qualification
    assert settings is not None
    reports = [load_report(project, tier.report) for tier in tiers]
    points = tier_outcomes(reports)
    judged = {
        result.checkpoint.artifact_id: result.judged
        for report in reports
        for result in report.points
        if result.judged is not None
    }
    review = settings.blast
    if review is None:
        return None

    def severity_of(key: str) -> SeverityJudgment:
        """Review one point's severity; a review that cannot be prepared leaves it unscored."""
        source = judged.get(key)
        try:
            return judge_severity(
                project,
                review,
                key,
                _failures(project, source),
                sources=() if source is None else (source,),
                code_revision=code_revision,
            )
        except ValueError as error:
            logger.warning("blast radius: point %s left unscored: %s", key, str(error)[:500])
            return SeverityJudgment(point_id=key, severity=None, votes={})

    ordered = sorted(points.items())
    severities = dict(
        zip(
            (key for key, _ in ordered),
            run_at_once(
                [lambda key=key: severity_of(key) for key, _ in ordered],
                admitted_concurrency(project, review.reviewer.model_alias),
            ),
            strict=True,
        )
    )
    ranked = rank_blast([blast_radius(point, severities[key], review) for key, point in ordered])
    cross_ref = None
    if review.cross_tier is not None and review.cross_tier.top_points > 0 and len(tiers) > 1:
        origins = {
            result.checkpoint.artifact_id: (tier, result.checkpoint)
            for tier, report in zip(tiers, reports, strict=True)
            for result in report.points
        }
        chosen = top_points_by_tier(ranked, origins, review.cross_tier.top_points)
        measured = cross_tier_studies(project, spec, tiers, chosen, code_revision=code_revision)
        ladder = tuple(spec.mining.actors)
        cross_rows: list[CrossTierPoint] = []
        for key, (origin, _) in chosen.items():
            extra = measured.get(key, {})
            point = points[key]
            added = tuple(
                TierOutcome(tier=tier, successes=s, failures=f, unknown=u)
                for tier, (s, f, u, _, _) in sorted(extra.items())
                if tier not in {row.tier for row in point.tiers}
            )
            point = point.model_copy(update={"tiers": (*point.tiers, *added)})
            points[key] = point
            cross_rows.append(
                cross_tier_point(
                    point,
                    origin.tier,
                    ladder,
                    review,
                    studies={t: row[3] for t, row in extra.items() if row[3] is not None},
                    failures={t: row[4] for t, row in extra.items() if row[4] is not None},
                )
            )
        record = CrossTierBlast(ladder=ladder, points=tuple(cross_rows))
        cross_ref = save_cross_tier(
            project.artifacts,
            record,
            code_revision=code_revision,
            inputs=tuple(study for row in cross_rows for study in row.studies.values()),
        )
        ranked = rank_blast(
            [blast_radius(points[key], severities[key], review) for key, _ in ordered]
        )
    saved = save_blast_radius(
        project.artifacts,
        ranked,
        code_revision=code_revision,
        inputs=(*inputs, *(() if cross_ref is None else (cross_ref,))),
    )
    return saved, ranked, cross_ref


def top_points_by_tier(
    ranked: Sequence[BlastRadius],
    origins: Mapping[str, tuple[TierMining, ArtifactInput]],
    limit: int,
) -> dict[str, tuple[TierMining, ArtifactInput]]:
    """Return each tier's ``limit`` highest-ranked points (blast order), keyed by checkpoint."""
    taken: dict[str, int] = {}
    chosen: dict[str, tuple[TierMining, ArtifactInput]] = {}
    for row in ranked:
        origin = origins.get(row.point_id)
        if origin is None or taken.get(origin[0].tier, 0) >= limit:
            continue
        taken[origin[0].tier] = taken.get(origin[0].tier, 0) + 1
        chosen[row.point_id] = origin
    return chosen


def cross_tier_job(
    spec: DecisionVarianceSpec, tier: TierMining, checkpoint: ArtifactInput
) -> ContinuationJob:
    """Return the continuations one tier runs from another tier's checkpoint."""
    review = spec.mining.qualification.blast if spec.mining.qualification else None
    cross = None if review is None else review.cross_tier
    repeats = (
        spec.mining.continuations
        if cross is None or cross.continuations is None
        else (cross.continuations)
    )
    return tier.job.model_copy(
        update={
            "maximum_physical_attempts": spec.mining.continuation_ceiling(repeats),
            "study": tier.job.study.model_copy(
                update={
                    "checkpoint": checkpoint,
                    "repeats": repeats,
                    "critic_prediction": None,
                    "predicted_success": None,
                    "selection_plan": None,
                }
            ),
        }
    )


CrossCounts = tuple[int, int, int, ArtifactInput | None, str | None]
"""One tier at one point: successes, failures, unknown, judged study, failure."""


def cross_tier_studies(
    project: ProjectStore,
    spec: DecisionVarianceSpec,
    tiers: Sequence[TierMining],
    chosen: Mapping[str, tuple[TierMining, ArtifactInput]],
    *,
    code_revision: str,
) -> dict[str, dict[str, CrossCounts]]:
    """Continue every chosen point with every other tier and judge it, concurrently.

    Each (point, tier) is its own continuation job, so it is content-addressed by its exact
    registration and replays with no call. A job that cannot run or be judged is recorded
    against that tier and skipped, never aborting the stage. At most the host admission bound
    of the actors and the judge run at once, each holding one request in flight.

    Returns:
        Per point, per other tier, its judged counts.
    """
    work = [
        (key, tier, checkpoint)
        for key, (origin, checkpoint) in chosen.items()
        for tier in tiers
        if tier.tier != origin.tier
    ]

    def run(tier: TierMining, checkpoint: ArtifactInput) -> CrossCounts:
        """Continue and judge one point with one tier."""
        job = cross_tier_job(spec, tier, checkpoint)
        try:
            study = saved_continuation_job(project, job, code_revision=code_revision)
            if study is None:
                study = execute_continuation_job(project, job, code_revision=code_revision)
            reference = continuation_result_reference(project, job, code_revision=code_revision)
            judged_ref, judged = judge_study(
                project,
                reference,
                tier.judge,
                configuration=job.configuration,
                code_revision=code_revision,
            )
        except (ValueError, RuntimeError, OSError) as error:
            logger.warning(
                "cross-tier: %s from %s failed: %s",
                tier.tier,
                checkpoint.artifact_id,
                str(error)[:500],
            )
            return 0, 0, 0, None, (type(error).__name__ + ": " + str(error))[:2000]
        rewards = slot_rewards(project, judged.branches, judged.judge)
        labels = [int(reward) for reward in rewards if reward is not None]
        unknown = len(study.branches) - len(labels)
        return sum(labels), len(labels) - sum(labels), unknown, judged_ref, None

    aliases = {tier.tier for tier in tiers} | {tier.judge.settings.model_alias for tier in tiers}
    results = run_at_once(
        [
            lambda tier=tier, checkpoint=checkpoint: run(tier, checkpoint)
            for _, tier, checkpoint in work
        ],
        admitted_concurrency(project, *sorted(aliases)),
    )
    measured: dict[str, dict[str, CrossCounts]] = {}
    for (key, tier, _), counts in zip(work, results, strict=True):
        measured.setdefault(key, {})[tier.tier] = counts
    return measured


def blast_costs(project: ProjectStore, spec: DecisionVarianceSpec) -> tuple[CostComponent, ...]:
    """Price severity reviews: one draw per stance for every point a tier may continue."""
    settings = spec.mining.qualification
    if settings is None or settings.blast is None:
        return ()
    review = settings.blast
    draws = len(review.stances) * review.profile.budget.model_calls
    points = qualified_bound(spec) * len(spec.mining.actors)
    return tuple(row.scaled(draws * points) for row in evaluation_costs(project, review.reviewer))


def load_report(project: ProjectStore, reference: ArtifactInput) -> ActorCriticReport:
    """Read one saved actor-critic report."""
    return ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )


def load_blast_ranking(store: ArtifactStore, reference: ArtifactInput) -> tuple[BlastRadius, ...]:
    """Read a saved blast-radius ranking, refusing a changed artifact.

    Raises:
        ValueError: The manifest changed.
    """
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"blast-radius ranking {reference.artifact_id} changed")
    record = json.loads(stored.payloads["blast.json"])
    return tuple(BlastRadius.model_validate(row) for row in record["points"])


def family_order(
    store: ArtifactStore, blast: ArtifactInput | None, report: ActorCriticReport
) -> tuple[int, ...]:
    """Return a tier's points in blast-radius order for family selection, or its own ranking."""
    if blast is None:
        return report.ranking
    return order_by_blast(report.ranking, load_blast_ranking(store, blast), attempt=report.picked)


def mining_provenance(
    project: ProjectStore, mining: MiningOutputs, tier_name: str, point: int
) -> MiningProvenance:
    """Return where one mined family's point came from, for its exported tasks.

    Args:
        project: Project holding the mining records.
        mining: The error-mining outputs.
        tier_name: The tier whose run chose the point.
        point: The point.

    Returns:
        The source traces, step, every tier's measured pass rate at the point's checkpoint, its
        blast radius and the reviewed difference group whose pairs diverge there.
    """
    tiers = {tier.tier: tier for tier in mining.tiers}
    tier = tiers[tier_name]
    reports = [load_report(project, row.report) for row in mining.tiers]
    chosen = next(r for r, row in zip(reports, mining.tiers, strict=True) if row.tier == tier_name)
    checkpoint = next(
        (row.checkpoint.artifact_id for row in chosen.points if row.point == point), None
    )
    start = load_continuation_checkpoint(project.artifacts, tier.job.study.checkpoint)
    measured = tier_outcomes(reports).get(checkpoint or "")
    rates: dict[str, float | None] = {row.tier: None for row in mining.tiers}
    if measured is not None:
        for outcome in measured.tiers:
            graded = outcome.successes + outcome.failures
            rates[outcome.tier] = None if graded == 0 else outcome.successes / graded
    blast = None
    if mining.blast is not None and checkpoint is not None:
        ranked = load_blast_ranking(project.artifacts, mining.blast)
        blast = next((row for row in ranked if row.point_id == checkpoint), None)
    kind = group = signature = None
    if tier.behavior_differences is not None:
        report = load_behavior_differences(project, tier.behavior_differences)
        pairs = {pair.pair_id for pair in report.pairs if pair.point == point}
        for candidate in report.groups:
            if pairs & set(candidate.pairs):
                hack = hack_signature(candidate.added, candidate.removed)
                kind = "adversarial" if hack is not None else "behavior"
                group = candidate.group_id
                signature = " ".join(
                    (*(f"+{tag}" for tag in candidate.added), *(f"-{t}" for t in candidate.removed))
                )
                break
    return MiningProvenance(
        task_id=start.task.task_id,
        source_trace_ids=tuple(start.task.source_trace_ids),
        step=point,
        tier=tier_name,
        tier_pass_rates=rates,
        blast_radius=None if blast is None else blast.score,
        blast_rank=None if blast is None or blast.score is None else blast.rank,
        signature_kind=kind,
        signature_group=group,
        signature=signature,
    )


def tier_allowance(context: StageContext) -> int | None:
    """Return one tier's share of the live cumulative ledger, or None without a ledger."""
    live = context.dependencies.remaining_attempts
    if live is None:
        return None
    return max(0, live()) // len(context.spec.mining.actors)

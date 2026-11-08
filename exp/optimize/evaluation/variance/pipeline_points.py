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
from collections.abc import Sequence
from pathlib import Path

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.executable.task import MiningProvenance
from exp.optimize.evaluation.rules.agent_loop import run_bounded
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport, admitted_concurrency
from exp.optimize.evaluation.variance.behavior_adversarial import hack_signature
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.blast_radius import (
    BlastRadius,
    SeverityJudgment,
    blast_radius,
    judge_severity,
    order_by_blast,
    rank_blast,
    save_blast_radius,
    tier_outcomes,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, episode_reward
from exp.optimize.evaluation.variance.family_stages import slot_evidence
from exp.optimize.evaluation.variance.judge import load_assessment
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    MiningOutputs,
    PointQualification,
    StageContext,
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
    reports: Sequence[ActorCriticReport],
    *,
    code_revision: str,
    inputs: tuple[ArtifactInput, ...],
) -> tuple[ArtifactInput, tuple[BlastRadius, ...]] | None:
    """Measure, judge, score and rank every continued point across tiers, then save it.

    Returns:
        The saved ranking and the ranked rows, or None when no severity reviewers are
        registered (a point without a judged severity is never scored).
    """
    settings = spec.mining.qualification
    assert settings is not None
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
    severities = run_bounded(
        [lambda key=key: severity_of(key) for key, _ in ordered],
        admitted_concurrency(project, review.reviewer.model_alias),
    )
    rows = [
        blast_radius(point, severity, review)
        for (_, point), severity in zip(ordered, severities, strict=True)
    ]
    ranked = rank_blast(rows)
    saved = save_blast_radius(project.artifacts, ranked, code_revision=code_revision, inputs=inputs)
    return saved, ranked


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

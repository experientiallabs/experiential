"""Carry a gated actor-critic run's deferred points into the next phase, under the live ledger.

A point that cleared the qualification bar but did not fit its phase's allowance stays
``deferred`` on the run's persisted record. When the run is reached again with a live allowance
(the front door's remaining cumulative ledger), :func:`carry_deferred` continues the deferred
points that now fit, in blast order, and saves the run's next phase report beside the first
(``<run>-result-phase<n>``). Continued points are graded and rewarded exactly like the first
phase's; the rest stay deferred. A phase that saved replays with no call, and no new
registration is needed. Cached accounting is a separately versioned projection of retained
point results. Planning reads committed projections; explicit carrying may save one without
repeating execution or changing the original phase report.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, stable_id
from exp.common.core.locks import CoordinationLock
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import saved_continuation_job
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    CriticDraw,
    _continue,
    _pick,
    _reward_critic,
    _reward_judge,
)
from exp.optimize.evaluation.variance.actor_critic_audit import rank_points
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.evidence_tools import total_usage
from exp.optimize.evaluation.variance.point_bar import PointSchedule, advance, schedule_points
from exp.optimize.evaluation.variance.rewards import PointOutcome, decision_drops
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.blast_radius import BlastRadius


PHASE_ACCOUNTING_PROTOCOL = "actor-critic-phase-accounting-v1"
"""Version of derived phase accounting and its immutable cached projection."""


def _accounting(report: ActorCriticReport) -> ActorCriticReport:
    """Recount retained outcomes and sum retained per-point actor measurements only.

    Missing legacy counts stay absent. If a recorded study lacks its point cost, continuation
    usage is unavailable, rather than a partial sum or a claimed zero. Original group and role
    usage remain historical measurements under their original registrations.
    """
    counts = report.counts
    if counts is not None:
        graded = sum(row.outcome.graded for row in report.points)
        recorded = sum(row.outcome.graded + row.outcome.unknown for row in report.points)
        counts = counts.model_copy(
            update={
                "continuations_recorded": recorded,
                "continuations_graded": graded,
                "continuations_unknown": recorded - graded,
                "point_failures": sum(row.failure is not None for row in report.points),
            }
        )
    usage = {key: value for key, value in report.usage.items() if key != "continuations"}
    if all(row.study is None or row.cost is not None for row in report.points):
        usage["continuations"] = total_usage(
            [row.cost.actor for row in report.points if row.cost is not None]
        )
    return report.model_copy(update={"counts": counts, "usage": usage})


def _accounting_identity(source: ArtifactInput) -> str:
    """Bind a cached accounting projection to its complete immutable source reference."""
    return stable_id(
        "actor-critic-phase-accounting",
        {
            "protocol": PHASE_ACCOUNTING_PROTOCOL,
            "source": source.model_dump(mode="json"),
        },
    )


def _saved_accounting(
    project: ProjectStore, source: ArtifactInput, report: ActorCriticReport
) -> tuple[ArtifactInput, ActorCriticReport] | None:
    """Read an existing exact accounting projection without writing during planning."""
    store = project.artifacts
    identity = _accounting_identity(source)
    if not store.exists(identity):
        return None
    manifest = store.read(identity).manifest
    bound = ArtifactInput.model_validate_json(store.read_bytes(identity, "source.json"))
    corrected = ActorCriticReport.model_validate_json(store.read_bytes(identity, "report.json"))
    if (
        manifest.artifact_type != "actor-critic-report"
        or bound != source
        or set(manifest.inputs) != {source}
        or corrected != _accounting(report)
    ):
        raise ValueError("saved phase accounting differs from its exact source or protocol")
    return artifact_input(manifest), corrected


def _accounted_phase(
    project: ProjectStore, source: ArtifactInput, report: ActorCriticReport, code_revision: str
) -> tuple[ArtifactInput, ActorCriticReport]:
    """Commit corrected cached accounting under its own reference, never over the source.

    Args:
        project: Project holding the immutable source phase.
        source: Exact source report reference, including its manifest digest.
        report: Source report with its retained points and measurements.
        code_revision: Producing identity of this derived accounting.

    Returns:
        The source unchanged when consistent, or its committed accounting projection.
    """
    corrected = _accounting(report)
    if corrected == report:
        return source, report
    store, identity = project.artifacts, _accounting_identity(source)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=10):
        saved = _saved_accounting(project, source, report)
        if saved is not None:
            return saved
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type="actor-critic-report",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(source,),
            ),
            files={"source.json": source, "report.json": corrected},
        )
        return artifact_input(manifest), corrected


def latest_phase(
    project: ProjectStore, spec: ActorCriticSpec, code_revision: str
) -> tuple[ArtifactInput, ActorCriticReport]:
    """Return the run's latest saved phase report and its reference.

    Raises:
        ValueError: The run's first phase is not saved.
    """
    store, identity = project.artifacts, spec.identity(code_revision)
    base = identity + "-result"
    if not store.exists(base):
        raise ValueError(f"actor-critic run {identity} is not saved")
    key, phase = base, 2
    while store.exists(f"{base}-phase{phase}"):
        key = f"{base}-phase{phase}"
        phase += 1
    report = ActorCriticReport.model_validate_json(store.read_bytes(key, "report.json"))
    reference = artifact_input(store.read(key).manifest)
    return _saved_accounting(project, reference, report) or (reference, report)


def carry_deferred(
    project: ProjectStore,
    spec: ActorCriticSpec,
    allowance: int | None,
    code_revision: str,
    ranked: Sequence[BlastRadius] = (),
) -> tuple[ArtifactInput, ActorCriticReport]:
    """Continue the run's deferred points that fit the live allowance, as its next phase.

    Args:
        project: Project holding the run.
        spec: The gated registration.
        allowance: Physical attempts this phase may reserve, from the live ledger; None
            continues nothing new.
        code_revision: Producing source identity.
        ranked: Blast-radius ranking, when one is measured.

    Returns:
        The latest phase's reference and report (unchanged when nothing new fits).
    """
    reference, report = latest_phase(project, spec, code_revision)
    reference, report = _accounted_phase(project, reference, report, code_revision)
    gate, schedule, record = spec.gate, report.point_schedule, report.bar_record
    if gate is None or schedule is None or record is None or not schedule.deferred:
        return reference, report
    if allowance is None or allowance < gate.cost_per_point:
        return reference, report
    deferred = set(schedule.deferred)
    selected = record.model_copy(
        update={"points": tuple(p for p in record.points if p.point in deferred)}
    )
    cost = {point: gate.cost_per_point for point in deferred}
    phase = schedule_points(selected, ranked, cost=cost, remaining=allowance)
    if not phase.continued:
        return reference, report
    store, identity = project.artifacts, spec.identity(code_revision)
    group = saved_continuation_job(project, spec.initial, code_revision=code_revision)
    assert group is not None

    def loader(ref: ArtifactInput) -> ContinuationCheckpoint:
        """Load one saved checkpoint."""
        return load_continuation_checkpoint(store, ref)

    _, packet = _pick(group, identity, spec, loader, load_references(store, spec.references))
    catalog = load_model_catalog(project.model_catalog_path)
    added = tuple(
        _continue(project, spec, packet, point, loader, catalog, code_revision)
        for point in phase.continued
    )
    points = tuple(sorted((*report.points, *added), key=lambda row: row.point))
    outcomes = tuple(row.outcome for row in points)
    judges = tuple(
        _reward_judge(draw.model_copy(update={"score": None}), outcomes) for draw in report.judges
    )
    critics = tuple(
        _reward_critic(draw.model_copy(update={"score": None}), _estimated(draw, outcomes))
        for draw in report.critics
    )
    drops = decision_drops(outcomes)
    merged = PointSchedule(
        continued=(*schedule.continued, *phase.continued),
        deferred=phase.deferred,
        reserved=schedule.reserved + phase.reserved,
        remaining=phase.remaining,
        transitions=(*schedule.transitions, *phase.transitions),
        refusals=(*schedule.refusals, *phase.refusals),
    )
    carried = _accounting(
        report.model_copy(
            update={
                "points": points,
                "judges": judges,
                "critics": critics,
                "drops": drops,
                "ranking": rank_points(outcomes, drops),
                "bar_record": advance(record, phase),
                "point_schedule": merged,
            }
        )
    )
    count = 2
    while store.exists(f"{identity}-result-phase{count}"):
        count += 1
    key = f"{identity}-result-phase{count}"
    inputs = {reference, *(row.checkpoint for row in added)}
    inputs |= {row.study for row in added if row.study} | {
        row.judged for row in added if row.judged
    }
    store.write_json(
        artifact_id=key,
        artifact_type="actor-critic-report",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={"report.json": carried},
    )
    return artifact_input(store.read(key).manifest), carried


def _estimated(draw: CriticDraw, outcomes: tuple[PointOutcome, ...]) -> tuple[PointOutcome, ...]:
    """Return the outcomes of the points a critic estimated before any continuation.

    A critic estimates the points continued in the first phase; points carried later are
    measured, but never scored against estimates the critic did not make.
    """
    if draw.assessment is None:
        return outcomes
    steps = {row.step for row in draw.assessment.estimates}
    return tuple(outcome for outcome in outcomes if outcome.step in steps)

"""Carry a gated actor-critic run's deferred points into the next phase, under the live ledger.

A point that cleared the qualification bar but did not fit its phase's allowance stays
``deferred`` on the run's persisted record. When the run is reached again with a live allowance
(the front door's remaining cumulative ledger), :func:`carry_deferred` continues the deferred
points that now fit, in blast order, and saves the run's next phase report beside the first
(``<run>-result-phase<n>``). Continued points are graded and rewarded exactly like the first
phase's; the rest stay deferred. A phase that saved replays with no call, and no new
registration is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import saved_continuation_job
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    _continue,
    _pick,
    _reward_critic,
    _reward_judge,
)
from exp.optimize.evaluation.variance.actor_critic_audit import rank_points
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.point_bar import PointSchedule, advance, schedule_points
from exp.optimize.evaluation.variance.rewards import decision_drops
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.blast_radius import BlastRadius


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
    return artifact_input(store.read(key).manifest), report


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
        _reward_critic(draw.model_copy(update={"score": None}), outcomes) for draw in report.critics
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
    carried = report.model_copy(
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

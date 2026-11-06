"""Run one epoch boundary of the evolving rubric through its explicit state machine.

:func:`run_boundary` is the only entry point that revises the rubric. It reads the open epoch
from the run's saved state (:func:`~.rubric_evolution.load_state`), walks one subject through
:data:`~.rubric_evo_states.BOUNDARY_MACHINE` (every move journaled, an undeclared move refused and
journaled), and closes the epoch by freezing the next version for the next epoch. The steps and
their rules are described in :mod:`~.rubric_evolution`; this module only sequences them.

- A refused export (another epoch, another rubric version, an unregistered task, over the priced
  bounds) or a refused ledger reservation ends the boundary before any call; nothing is saved
  except the journaled moves, and the epoch stays open.
- A finished boundary is saved once under :func:`~.rubric_evolution.boundary_identity`; running
  the same export again replays the record with no call and no new reservation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonValue
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.dense_hybrid import tag_episode_steps
from exp.optimize.evaluation.variance.dense_vocabulary import vocabulary
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.rollout_ingest import (
    Rollout,
    RolloutExport,
    rollout_evidence,
)
from exp.optimize.evaluation.variance.rubric_audit import (
    RolloutAudit,
    audit_pass_rate,
    audit_rollout,
    audit_sample,
)
from exp.optimize.evaluation.variance.rubric_evo_states import BoundaryState
from exp.optimize.evaluation.variance.rubric_evolution import (
    BOUNDARY_TYPE,
    BoundaryRecord,
    BoundaryWalk,
    EpochMetrics,
    EpochRecord,
    EvolutionRegistration,
    EvolutionState,
    boundary_costs,
    boundary_identity,
    divergence_trigger,
    ingest_problem,
    journal_of,
    load_state,
    reserve,
    save_state,
)
from exp.optimize.evaluation.variance.rubric_grading import measure_recall
from exp.optimize.evaluation.variance.rubric_items import (
    check_retirement,
    compile_item,
    consolidate,
)
from exp.optimize.evaluation.variance.rubric_signatures import (
    RolloutReading,
    admit_signature,
    author_probes,
    column_maxima,
    polarity,
    read_rollout,
    select_probes,
    symbolic_diff,
)
from exp.optimize.evaluation.variance.rubric_versions import (
    RecallRecord,
    RubricItem,
    RubricVersion,
    load_version,
    save_version,
    version_banks,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TraceTagger

B = BoundaryState


def load_boundary(project: ProjectStore, reference: ArtifactInput) -> BoundaryRecord:
    """Read a saved boundary, refusing a changed artifact."""
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"boundary {reference.artifact_id} changed")
    return BoundaryRecord.model_validate_json(stored.payloads["boundary.json"])


def _close(
    project: ProjectStore,
    registration: EvolutionRegistration,
    state: EvolutionState,
    record: BoundaryRecord,
    reference: ArtifactInput,
) -> EvolutionState:
    """Close the record's epoch and open the next under its frozen version (idempotent)."""
    open_epoch = state.open_epoch
    if open_epoch.epoch != record.epoch or open_epoch.boundary is not None:
        return state
    assert record.version_out is not None and record.version_out_sha256 is not None
    version = load_version(project, record.version_out)
    closed = open_epoch.model_copy(
        update={"boundary": reference, "metrics": record.metrics, "trigger": record.trigger}
    )
    following = EpochRecord(
        epoch=record.epoch + 1,
        version=version.version,
        version_sha256=version.identity,
        version_ref=record.version_out,
        forced=record.trigger is not None and record.trigger.fired,
    )
    versions = state.versions
    if record.state == B.PUBLISHED.value:
        versions = (*versions, record.version_out)
    updated = state.model_copy(
        update={"versions": versions, "epochs": (*state.epochs[:-1], closed, following)}
    )
    save_state(project, registration, updated)
    return updated


def run_boundary(
    project: ProjectStore,
    registration: EvolutionRegistration,
    export: RolloutExport,
    *,
    tagger: TraceTagger,
    code_revision: str,
    ledger: PhaseLedger | None = None,
) -> BoundaryRecord:
    """Revise the rubric at the boundary that ends ``export.epoch``, or replay it.

    Args:
        project: Project holding every draw, judgment and record.
        registration: The run's evolving rubric.
        export: The epoch's rollout export.
        tagger: The decision model's tagging seam (Clef through the decision cache).
        code_revision: Producing source identity.
        ledger: The cumulative front-door ledger to reserve the boundary's bound on; None when
            the caller already reserved it (a front-door phase).

    Returns:
        The boundary record; a refused boundary is returned unsaved with its reason.

    Raises:
        ValueError: The evolution was not opened.
    """
    state = load_state(project, registration)
    if state is None:
        raise ValueError("open the evolution before running a boundary")
    identity = boundary_identity(registration, export)
    store = project.artifacts
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        record = load_boundary(project, reference)
        _close(project, registration, state, record, reference)
        return record
    open_epoch = state.open_epoch
    walk = BoundaryWalk(
        f"{registration.run_id}/epoch-{export.epoch}/{export.identity[:12]}",
        journal_of(project, registration),
    )
    base: dict[str, JsonValue] = {
        "run_id": registration.run_id,
        "epoch": export.epoch,
        "forced": open_epoch.forced,
        "export_sha256": export.identity,
        "version_in": open_epoch.version_sha256,
    }
    problem = ingest_problem(state, registration, export)
    if problem is not None:
        walk.go(B.INGEST_REFUSED, problem)
        return BoundaryRecord.model_validate(
            {**base, "state": walk.state.value, "reason": problem}
            | {"transitions": walk.transitions}
        )
    phase = f"rubric-evolution {registration.run_id} epoch {export.epoch}"
    costs = boundary_costs(project, registration, forced=open_epoch.forced)
    refusal = reserve(ledger, phase, costs)
    if refusal is not None:
        walk.go(B.BUDGET_REFUSED, refusal)
        return BoundaryRecord.model_validate(
            {**base, "state": walk.state.value, "reason": refusal}
            | {"transitions": walk.transitions}
        )
    before = measure(project)
    try:
        record = _Boundary(project, registration, state, export, tagger, code_revision, walk).run(
            base
        )
    finally:
        if ledger is not None:
            ledger.settle(phase, before, measure(project))
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=BOUNDARY_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                {
                    ref.artifact_id: ref
                    for ref in (open_epoch.version_ref, record.version_out)
                    if ref is not None
                }.values()
            ),
        ),
        files={"boundary.json": record},
    )
    reference = artifact_input(manifest)
    _close(project, registration, state, record, reference)
    return record


class _Boundary:
    """One boundary in progress."""

    def __init__(
        self,
        project: ProjectStore,
        registration: EvolutionRegistration,
        state: EvolutionState,
        export: RolloutExport,
        tagger: TraceTagger,
        code_revision: str,
        walk: BoundaryWalk,
    ) -> None:
        """Bind the boundary's inputs."""
        self.project, self.registration, self.state = project, registration, state
        self.export, self.tagger, self.code_revision, self.walk = (
            export,
            tagger,
            code_revision,
            walk,
        )
        self.spec = registration.spec
        self.version = load_version(project, state.open_epoch.version_ref)
        self.banks = registration.banks()
        self.fields: dict[str, object] = {}

    def run(self, base: Mapping[str, JsonValue]) -> BoundaryRecord:
        """Walk every step and return the record."""
        spec, export, walk = self.spec, self.export, self.walk
        tasks = self.registration.task_map()
        step_banks = version_banks(self.version, self.banks)
        _, step = vocabulary(step_banks)
        held = self.registration.held_out.lineages()
        excluded = {
            r.rollout_id: "held-out lineage" for r in export.rollouts if r.lineage_id in held
        }
        rollouts = [r for r in export.rollouts if r.rollout_id not in excluded]
        self.fields["excluded"] = excluded
        walk.go(B.INGESTED, f"{len(rollouts)} rollouts accepted, {len(excluded)} excluded")
        evidence: dict[str, EpisodeEvidence] = {}
        matrices: dict[str, TagMatrix | None] = {}
        for rollout in rollouts:
            judge = tasks[rollout.task_id].judge.definition
            evidence[rollout.rollout_id], _, _ = rollout_evidence(
                rollout,
                task_sha256=judge.task_sha256,
                world_sha256=judge.world_sha256,
                trainer_run=export.trainer_run,
            )
            matrices[rollout.rollout_id] = tag_episode_steps(
                evidence[rollout.rollout_id], self.tagger, step, spec.routing
            )
        readings = {
            r.rollout_id: read_rollout(
                r.rollout_id, r.task_id, r.lineage_id, [matrices[r.rollout_id]], spec.routing
            )
            for r in rollouts
        }
        maxima = {rid: column_maxima(m) for rid, m in matrices.items()}
        walk.go(B.TAGGED, f"{len(rollouts)} rollouts tagged with step bank {step.bank_id}")
        forced = self.state.open_epoch.forced
        sample = audit_sample(rollouts, spec.audit, epoch=export.epoch, forced=forced)
        audits = [
            audit_rollout(
                self.project,
                spec.auditors,
                spec.profile,
                spec.audit,
                rollout,
                evidence[rollout.rollout_id],
                code_revision=self.code_revision,
            )
            for rollout in sample
        ]
        rate, decided = audit_pass_rate(audits)
        rewards = [r.in_loop_reward for r in rollouts]
        metrics = EpochMetrics(
            epoch=export.epoch,
            rollouts=len(rollouts),
            mean_reward=sum(rewards) / len(rewards) if rewards else None,
            audited=len(audits),
            decided=decided,
            audit_pass_rate=rate,
        )
        previous = self.state.epochs[-2].metrics if len(self.state.epochs) > 1 else None
        trigger = divergence_trigger(previous, metrics, spec.trigger)
        flagged = tuple(a.rollout_id for a in audits if a.verdict == "fail")
        genuine = tuple(a.rollout_id for a in audits if a.verdict == "pass")
        self.fields |= {
            "readings": tuple(readings.values()),
            "audits": tuple(audits),
            "metrics": metrics,
            "trigger": trigger,
            "flagged": flagged,
            "genuine": genuine,
        }
        audited = f"{len(audits)} audited, {len(flagged)} flagged, pass rate {rate}"
        walk.go(B.AUDITED_FORCED if forced else B.AUDITED, audited)
        items = self.version.items
        added: tuple[str, ...] = ()
        merged = False
        if not flagged:
            if forced:
                walk.go(B.ESCALATED, "the forced boundary's audit flagged nothing; review")
                return self._record(base, self.version)
            walk.go(B.NO_DIVERGENCE, "the audit flagged no high-reward rollout")
        else:
            items, added, merged = self._revise(
                rollouts, evidence, readings, step_banks, audits, flagged, genuine
            )
        items, checks = check_retirement(
            items, maxima, spec.retirement, epoch=export.epoch, admitted_now=frozenset(added)
        )
        retired = tuple(c.item_id for c in checks if c.retired)
        self.fields["retirement"] = checks
        walk.go(B.RETIREMENT_CHECKED, f"{len(checks)} items re-measured, {len(retired)} retired")
        if not (added or merged or retired):
            walk.go(B.UNCHANGED, "no item added, merged or retired; the version carries over")
            return self._record(base, self.version.model_copy(update={"items": items}))
        return self._recall(base, items, retired, bool(added or merged))

    def _revise(
        self,
        rollouts: Sequence[Rollout],
        evidence: dict[str, EpisodeEvidence],
        readings: dict[str, RolloutReading],
        step_banks: Sequence[QuestionBank],
        audits: Sequence[RolloutAudit],
        flagged: tuple[str, ...],
        genuine: tuple[str, ...],
    ) -> tuple[tuple[RubricItem, ...], tuple[str, ...], bool]:
        """Diff, probe, admit, compile and consolidate; return the items and what changed."""
        spec, walk = self.spec, self.walk
        negative, positive = polarity(step_banks)
        groups, unexplained = symbolic_diff(
            [readings[f] for f in flagged], [readings[g] for g in genuine], negative, positive
        )
        if unexplained:
            by_id = {a.rollout_id: a for a in audits}
            record, bank = author_probes(
                self.project,
                spec.author,
                spec.profile,
                spec.probes,
                [(by_id[u], evidence[u]) for u in unexplained],
                [(g, evidence[g]) for g in genuine],
                code_revision=self.code_revision,
            )
            if bank is not None:
                answered = {
                    rid: tag_episode_steps(evidence[rid], self.tagger, bank, spec.routing)
                    for rid in (*flagged, *genuine)
                }
                record, kept = select_probes(record, bank, answered, spec.probes)
                if kept is not None:
                    keep = frozenset(kept.question_ids())
                    for rid, matrix in answered.items():
                        probe = read_rollout(rid, "", "", [matrix], spec.routing)
                        row = readings[rid]
                        readings[rid] = row.model_copy(
                            update={
                                "fired": tuple(sorted({*row.fired, *(set(probe.fired) & keep)})),
                                "uncertain": tuple(
                                    sorted({*row.uncertain, *(set(probe.uncertain) & keep)})
                                ),
                            }
                        )
                    groups, unexplained = symbolic_diff(
                        [readings[f] for f in flagged],
                        [readings[g] for g in genuine],
                        negative | keep,
                        positive,
                    )
            self.fields["probes"] = record
        self.fields |= {"groups": groups, "unexplained": unexplained}
        walk.go(B.DIFFED, f"{len(groups)} signature groups, {len(unexplained)} unexplained")
        corpus = {r.rollout_id: (r.lineage_id, evidence[r.rollout_id]) for r in rollouts}
        admissions = [
            admit_signature(
                self.project,
                spec.author,
                spec.profile,
                spec.admission,
                group,
                corpus,
                held_out_lineages=self.registration.held_out.lineages(),
                tagger=self.tagger,
                code_revision=self.code_revision,
            )
            for group in groups
        ]
        self.fields["admissions"] = tuple(admissions)
        admitted = [a for a in admissions if a.verdict is not None and a.verdict.admitted]
        if not admitted:
            walk.go(B.NONE_ADMITTED, "no signature has two independent lineages uncontradicted")
            return self.version.items, (), False
        walk.go(B.ADMITTED, f"{len(admitted)} signatures admitted")
        compilations = []
        for admission in admitted:
            documents: dict[str, JsonValue] = {}
            assert admission.verdict is not None
            for rid in dict.fromkeys((*admission.verdict.supporting, *admission.group.genuine)):
                for event in evidence[rid].events:
                    documents[f"{rid}/{event.event_id}"] = event.model_dump(mode="json")
            compilations.append(
                compile_item(
                    self.project,
                    spec.author,
                    spec.profile,
                    admission,
                    documents,
                    stances=spec.item_stances,
                    code_revision=self.code_revision,
                )
            )
        self.fields["compilations"] = tuple(compilations)
        drafted = sum(len(c.drafts) for c in compilations)
        walk.go(B.COMPILED, f"{drafted} drafts, each a dense question and a judge criterion")
        items, consolidation = consolidate(
            self.project,
            spec.author,
            spec.profile,
            self.version,
            {a.group.group_id: a for a in admitted},
            {c.group_id: c for c in compilations},
            epoch=self.export.epoch,
            code_revision=self.code_revision,
        )
        self.fields["consolidation"] = consolidation
        walk.go(
            B.CONSOLIDATED,
            f"added {list(consolidation.added)}, merged {sorted(consolidation.merged)}, "
            f"dropped {sorted(consolidation.dropped)}",
        )
        return items, consolidation.added, bool(consolidation.merged)

    def _recall(
        self,
        base: Mapping[str, JsonValue],
        items: tuple[RubricItem, ...],
        retired: tuple[str, ...],
        grew: bool,
    ) -> BoundaryRecord:
        """Check held-out recall of the candidate, retrying without retirements once."""
        walk, current = self.walk, self.version
        candidate = RubricVersion(
            run_id=current.run_id,
            version=current.version + 1,
            epoch=self.export.epoch + 1,
            parent=current.identity,
            banks=current.banks,
            items=items,
        )
        reverted = False
        recalls: list[RecallRecord] = []
        assert current.recall is not None
        while True:
            recall, _ = measure_recall(
                self.project,
                candidate,
                self.banks,
                self.registration.task_map(),
                self.registration.held_out,
                tagger=self.tagger,
                routing=self.spec.routing,
                code_revision=self.code_revision,
            )
            recalls.append(recall)
            self.fields["recall"] = tuple(recalls)
            walk.go(
                B.RECALL_CHECKED,
                f"outcome recall {recall.outcome_recall:.3f} (was "
                f"{current.recall.outcome_recall:.3f}), dense recall {recall.dense_recall:.3f} "
                f"(was {current.recall.dense_recall:.3f})",
            )
            lowered = current.recall.lowered_by(recall)
            if not lowered:
                walk.go(B.PUBLISHED, f"version {candidate.version} published")
                return self._record(base, candidate.model_copy(update={"recall": recall}))
            walk.go(B.RECALL_REGRESSED, f"the candidate lowers {list(lowered)} recall")
            if reverted or not retired or not grew:
                walk.go(B.KEPT, f"version {current.version} carries over")
                return self._record(base, current)
            restored = tuple(
                row.model_copy(update={"status": "active", "retired_epoch": None})
                if row.item_id in retired
                else row
                for row in candidate.items
            )
            candidate = candidate.model_copy(update={"items": restored})
            reverted = True
            walk.go(B.RETIREMENT_REVERTED, f"retry without retiring {list(retired)}")

    def _record(self, base: Mapping[str, JsonValue], version: RubricVersion) -> BoundaryRecord:
        """Save the version frozen for the next epoch and assemble the record."""
        reference = save_version(
            self.project,
            version,
            code_revision=self.code_revision,
            inputs=(self.state.open_epoch.version_ref,),
        )
        return BoundaryRecord.model_validate(
            {
                **base,
                **self.fields,
                "state": self.walk.state.value,
                "reason": self.walk.transitions[-1].reason,
                "transitions": tuple(self.walk.transitions),
                "refusals": tuple(self.walk.refusals),
                "version_out": reference,
                "version_out_sha256": version.identity,
            }
        )

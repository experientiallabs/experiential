"""The tag monitor: statistics first, Astra's cited review second, at every pipeline checkpoint.

:func:`run_checkpoint` walks one checkpoint run through
:data:`~.tag_monitor_states.CHECKPOINT_MACHINE` (every move journaled at
``<project runtime>/tag-monitor/journal.jsonl``, an undeclared move refused and journaled):

1. The checkpoint's inputs, read from records the pipeline already saved, are measured by code
   (:mod:`~.tag_monitor_stats`, :mod:`~.tag_monitor_reward_stats`) into subject statistics and
   halt conditions. Missing inputs end the run (``inputs_missing``) before anything else.
2. A run whose statistics flag nothing and halt nothing skips the review. Otherwise Astra's
   review (:mod:`~.tag_monitor_review`) is priced and reserved on the single cumulative front-door
   ledger first (a refused reservation ends the run, unsaved), then run and settled.
3. Accepted proposals are typed, versioned (``revision``) and routed to the path that owns the
   change; a cited ``halt`` finding becomes a halt beside the statistics' halts.
4. The report is saved once under a content identity over the statistics, thresholds and
   Astra's settings, so running the same checkpoint again replays it with no call and no
   reservation.

After saving, every halt is registered as ``raised`` (:data:`~.tag_monitor_states.HALT_MACHINE`)
and blocks the next front-door phase until :func:`acknowledge_halt` records a person's
acknowledgement, or a newer report for the same checkpoint and scope re-authors it. Every
proposal enters the queue as ``routed`` (:data:`~.tag_monitor_states.PROPOSAL_MACHINE`); its owner
reads :func:`routed_proposals` and records :func:`resolve_proposal`. Nothing is applied here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore, artifact_input
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.rubric_evolution import reserve
from exp.optimize.evaluation.variance.tag_monitor_contracts import (
    CHECKPOINT_NUMBERS,
    REPORT_TYPE,
    ROUTES,
    TAG_MONITOR_PROTOCOL,
    CheckpointName,
    CheckpointStats,
    Halt,
    MonitorReport,
    MonitorSpec,
    Proposal,
    SubjectStats,
)
from exp.optimize.evaluation.variance.tag_monitor_review import (
    AstraSettings,
    ProposalAnswer,
    review_costs,
    run_review,
)
from exp.optimize.evaluation.variance.tag_monitor_reward_stats import (
    DenseInputs,
    EpochInputs,
    MiningInputs,
    dense_stats,
    epoch_stats,
    mining_stats,
)
from exp.optimize.evaluation.variance.tag_monitor_states import (
    CHECKPOINT_MACHINE,
    HALT_MACHINE,
    PROPOSAL_MACHINE,
    CheckpointState,
    HaltState,
    ProposalState,
)
from exp.optimize.evaluation.variance.tag_monitor_stats import (
    CorpusInputs,
    FamilyInputs,
    RuleInputs,
    corpus_stats,
    family_stats,
    halt,
    rule_stats,
)

C = CheckpointState

CheckpointInputs = (
    CorpusInputs | FamilyInputs | RuleInputs | MiningInputs | DenseInputs | EpochInputs
)
"""The inputs one checkpoint reads."""

Measured = tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]
"""Subjects, halts and sampled evidence."""

_MEASURES: dict[CheckpointName, tuple[type[ContractModel], Callable[..., Measured]]] = {
    "corpus": (CorpusInputs, corpus_stats),
    "families": (FamilyInputs, family_stats),
    "rules": (RuleInputs, rule_stats),
    "mining": (MiningInputs, mining_stats),
    "dense": (DenseInputs, dense_stats),
    "epoch": (EpochInputs, epoch_stats),
}


def measure_checkpoint(
    checkpoint: CheckpointName,
    scope: str,
    inputs: CheckpointInputs,
    spec: MonitorSpec,
    sources: Sequence[ArtifactInput] = (),
) -> CheckpointStats:
    """Compute one checkpoint's statistics, halts and evidence, with no model call.

    Raises:
        TypeError: The inputs are not the checkpoint's inputs.
    """
    model, compute = _MEASURES[checkpoint]
    if not isinstance(inputs, model):
        raise TypeError(f"checkpoint {checkpoint} reads {model.__name__}")
    subjects, halts, evidence = compute(inputs, spec)
    return CheckpointStats(
        checkpoint=checkpoint,
        scope=scope,
        sources=tuple(sorted(set(sources), key=lambda ref: ref.artifact_id)),
        subjects=tuple(subjects),
        halts=tuple(halts),
        evidence=evidence,
    )


def monitor_directory(project: ProjectStore) -> Path:
    """The monitor's mutable state: the journal, the halt registry and the proposal queue."""
    return project.paths.runtime_directory / "tag-monitor"


def journal_of(project: ProjectStore) -> TransitionJournal:
    """The monitor's transition journal."""
    return TransitionJournal(monitor_directory(project) / "journal.jsonl")


def report_identity(stats: CheckpointStats, spec: MonitorSpec, astra: AstraSettings | None) -> str:
    """The identity a checkpoint report is saved under, computed before any call."""
    return stable_id(
        REPORT_TYPE,
        {
            "protocol": TAG_MONITOR_PROTOCOL,
            "checkpoint": stats.checkpoint,
            "scope": stats.scope,
            "stats": sha256_json(stats),
            "spec": sha256_json(spec),
            "astra": None if astra is None else sha256_json(astra),
        },
    )


def load_report(project: ProjectStore, reference: ArtifactInput) -> MonitorReport:
    """Read a saved report, refusing a changed artifact."""
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"tag monitor report {reference.artifact_id} changed")
    return MonitorReport.model_validate_json(stored.payloads["report.json"])


class _Walk:
    """One checkpoint run's walk, every move checked and journaled."""

    def __init__(self, subject: str, journal: TransitionJournal) -> None:
        """Start in ``opened``."""
        self.subject, self.journal = subject, journal
        self.state = CHECKPOINT_MACHINE.move(subject, None, C.OPENED.value, journal)

    def go(self, target: CheckpointState) -> None:
        """Apply one move; an undeclared move is journaled as refused and raised."""
        self.state = CHECKPOINT_MACHINE.move(
            self.subject, self.state, target.value, self.journal, target=target
        )


def run_checkpoint(
    project: ProjectStore,
    checkpoint: CheckpointName,
    scope: str,
    inputs: CheckpointInputs | None,
    *,
    astra: AstraSettings | None,
    spec: MonitorSpec | None = None,
    sources: Sequence[ArtifactInput] = (),
    code_revision: str,
    ledger: PhaseLedger | None = None,
) -> tuple[MonitorReport, ArtifactInput | None]:
    """Run, or replay, one checkpoint and return its report and saved artifact.

    Args:
        project: Project holding the review draws and the report.
        checkpoint: Which checkpoint.
        scope: What it covers.
        inputs: The checkpoint's inputs, or None when a record it reads is not saved yet.
        astra: Astra's settings, or None to record statistics only.
        spec: Thresholds and bounds; defaults apply when omitted.
        sources: Saved records the inputs were read from.
        code_revision: Producing source identity.
        ledger: The cumulative ledger to reserve the review on; None when the caller (a
            front-door phase) already reserved it.

    Returns:
        The report and its artifact; an unsaved report (``inputs_missing`` or
        ``budget_refused``) has no artifact.
    """
    spec = spec or MonitorSpec()
    number = CHECKPOINT_NUMBERS[checkpoint]
    journal = journal_of(project)
    if inputs is None:
        walk = _Walk(f"{checkpoint}/{scope}", journal)
        walk.go(C.INPUTS_MISSING)
        empty = CheckpointStats(checkpoint=checkpoint, scope=scope)
        return MonitorReport(
            checkpoint=checkpoint,
            number=number,
            scope=scope,
            stats_sha256=sha256_json(empty),
            stats=empty,
            state=walk.state.value,
            reason="a record this checkpoint reads is not saved yet",
        ), None
    stats = measure_checkpoint(checkpoint, scope, inputs, spec, sources)
    identity = report_identity(stats, spec, astra)
    store = project.artifacts
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        report = load_report(project, reference)
        _register(project, report, reference)
        return report, reference
    walk = _Walk(f"{checkpoint}/{scope}/{identity[-20:]}", journal)
    walk.go(C.MEASURED)
    walk.go(C.SAMPLED)
    base: JsonObject = {
        "checkpoint": checkpoint,
        "number": number,
        "scope": scope,
        "stats_sha256": sha256_json(stats),
        "stats": stats.model_dump(mode="json"),
    }
    review = None
    answers: list[ProposalAnswer] = []
    if not stats.flagged() and not stats.halts:
        walk.go(C.REVIEW_SKIPPED)
        reason = "nothing flagged and nothing halted; Astra was not called"
    elif astra is None:
        walk.go(C.REVIEW_UNUSABLE)
        reason = "no reviewer configured; the statistics stand alone"
    else:
        phase = f"tag-monitor {checkpoint} {scope}"
        refusal = reserve(ledger, phase, review_costs(project, astra))
        if refusal is not None:
            walk.go(C.BUDGET_REFUSED)
            return MonitorReport.model_validate(
                base | {"state": walk.state.value, "reason": refusal}
            ), None
        before = measure(project)
        try:
            review, answers = run_review(project, astra, stats, spec, code_revision=code_revision)
        finally:
            if ledger is not None:
                ledger.settle(phase, before, measure(project))
        walk.go(C.REVIEWED if review.accepted else C.REVIEW_UNUSABLE)
        reason = "reviewed" if review.accepted else "no review draft passed the host's checks"
    proposals = _proposals(project, checkpoint, scope, answers)
    if review is not None:
        review = review.model_copy(update={"proposals": tuple(proposals)})
    halts = list(stats.halts)
    for finding in review.findings if review is not None else ():
        if finding.severity == "halt":
            raised = halt(checkpoint, finding.subject, "astra_halt", finding.statement[:900])
            halts.append(raised.model_copy(update={"source": "astra", "cites": finding.cites}))
    walk.go(C.REPORTED)
    walk.go(C.HALTED if halts else C.PASSED)
    report = MonitorReport.model_validate(
        base
        | {
            "review": None if review is None else review.model_dump(mode="json"),
            "halts": [row.model_dump(mode="json") for row in halts],
            "proposals": [row.model_dump(mode="json") for row in proposals],
            "unreviewed": _unreviewed(stats, review, proposals),
            "state": walk.state.value,
            "reason": reason,
        }
    )
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=REPORT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {
                        ref.artifact_id: ref
                        for ref in (*stats.sources, *(review.transcripts if review else ()))
                    }.values(),
                    key=lambda ref: ref.artifact_id,
                )
            ),
        ),
        files={"report.json": report},
    )
    reference = artifact_input(manifest)
    _register(project, report, reference)
    return report, reference


def _proposals(
    project: ProjectStore, checkpoint: CheckpointName, scope: str, answers: Sequence[ProposalAnswer]
) -> list[Proposal]:
    """Type, version and route the accepted proposal answers."""
    queue = _read(project, "proposals.json")
    rows: list[Proposal] = []
    for answer in answers:
        targets = tuple(answer.targets)
        key = [checkpoint, scope, answer.kind, sorted(targets)]
        revision = 1 + sum(
            1 for row in queue.values() if isinstance(row, dict) and row.get("key") == key
        )
        rationale = answer.rationale
        if answer.rewrite:
            rationale = f"{rationale}\nProposed text: {answer.rewrite}"
        if answer.refs:
            rationale = f"{rationale}\nRefs: {', '.join(answer.refs)}"
        identity = sha256_json([key, revision, rationale, [c.model_dump() for c in answer.cites]])
        rows.append(
            Proposal(
                proposal_id=f"proposal-{identity[:20]}",
                revision=revision,
                checkpoint=checkpoint,
                scope=scope,
                kind=answer.kind,
                targets=targets,
                keep=answer.keep,
                route=ROUTES[(checkpoint, answer.kind)],
                rationale=rationale[:2000],
                cites=tuple(answer.cites),
            )
        )
    return rows


def _unreviewed(stats: CheckpointStats, review: object, proposals: Sequence[Proposal]) -> list[str]:
    """Flagged subjects no finding, proposal target or citation addressed."""
    named: set[str] = set()
    findings = getattr(review, "findings", ())
    for finding in findings:
        named.add(finding.subject)
        named.update(c.subject for c in finding.cites)
    for proposal in proposals:
        named.update(proposal.targets)
        named.update(c.subject for c in proposal.cites)
    return [row.subject for row in stats.flagged() if row.subject not in named]


def _read(project: ProjectStore, name: str) -> dict[str, JsonValue]:
    """Read one of the monitor's mutable registries."""
    path = monitor_directory(project) / name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(project: ProjectStore, name: str, payload: dict[str, JsonValue]) -> None:
    """Persist one of the monitor's mutable registries."""
    path = monitor_directory(project) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def _register(project: ProjectStore, report: MonitorReport, reference: ArtifactInput) -> None:
    """Raise the report's halts, re-author older ones, and route its proposals (idempotent)."""
    journal = journal_of(project)
    halts = _read(project, "halts.json")
    for halt_id, row in list(halts.items()):
        if (
            isinstance(row, dict)
            and row.get("checkpoint") == report.checkpoint
            and row.get("scope") == report.scope
            and row.get("report") != reference.artifact_id
            and row.get("state") == HaltState.RAISED.value
        ):
            HALT_MACHINE.move(
                halt_id, HaltState.RAISED, "reauthored", journal, target=HaltState.REAUTHORED
            )
            halts[halt_id] = {**row, "state": HaltState.REAUTHORED.value}
    for row in report.halts:
        if row.halt_id not in halts:
            HALT_MACHINE.move(row.halt_id, None, HaltState.RAISED.value, journal)
            halts[row.halt_id] = {
                "checkpoint": report.checkpoint,
                "scope": report.scope,
                "report": reference.artifact_id,
                "state": HaltState.RAISED.value,
                "condition": row.condition,
                "reason": row.reason,
            }
    _write(project, "halts.json", halts)
    queue = _read(project, "proposals.json")
    for proposal in report.proposals:
        if proposal.proposal_id in queue:
            continue
        key = [proposal.checkpoint, proposal.scope, proposal.kind, sorted(proposal.targets)]
        for other_id, row in list(queue.items()):
            if (
                isinstance(row, dict)
                and row.get("key") == key
                and row.get("state") in (ProposalState.PROPOSED.value, ProposalState.ROUTED.value)
            ):
                source = ProposalState(str(row["state"]))
                PROPOSAL_MACHINE.move(
                    other_id, source, "superseded", journal, target=ProposalState.SUPERSEDED
                )
                queue[other_id] = {**row, "state": ProposalState.SUPERSEDED.value}
        PROPOSAL_MACHINE.move(proposal.proposal_id, None, ProposalState.PROPOSED.value, journal)
        PROPOSAL_MACHINE.move(
            proposal.proposal_id,
            ProposalState.PROPOSED,
            "routed",
            journal,
            target=ProposalState.ROUTED,
        )
        queue[proposal.proposal_id] = {
            "key": key,
            "route": proposal.route,
            "report": reference.artifact_id,
            "state": ProposalState.ROUTED.value,
            "proposal": proposal.model_dump(mode="json"),
        }
    _write(project, "proposals.json", queue)


def blocking_halts(project: ProjectStore, report: MonitorReport) -> list[Halt]:
    """The report's halts that are still raised, so they block the next phase."""
    halts = _read(project, "halts.json")
    blocking = []
    for row in report.halts:
        entry = halts.get(row.halt_id)
        if not isinstance(entry, dict) or entry.get("state") == HaltState.RAISED.value:
            blocking.append(row)
    return blocking


class HaltAcknowledgement(ContractModel):
    """A person's acknowledgement of one halt.

    Attributes:
        halt_id: The halt.
        by: Who acknowledged it.
        note: Why the next phase may run anyway.
        at: When.
    """

    halt_id: str
    by: str
    note: str
    at: str


def acknowledge_halt(
    project: ProjectStore, halt_id: str, *, by: str, note: str
) -> HaltAcknowledgement:
    """Record a person's acknowledgement of a raised halt, unblocking the next phase.

    Raises:
        KeyError: The halt is not registered.
        IllegalTransition: The halt is not raised (already acknowledged or re-authored); the
            refusal is journaled.
    """
    if not by.strip() or not note.strip():
        raise ValueError("an acknowledgement names who acknowledged it and why")
    halts = _read(project, "halts.json")
    row = halts[halt_id]
    assert isinstance(row, dict)
    HALT_MACHINE.move(
        halt_id,
        HaltState(str(row["state"])),
        "acknowledged",
        journal_of(project),
        target=HaltState.ACKNOWLEDGED,
    )
    record = HaltAcknowledgement(
        halt_id=halt_id, by=by, note=note, at=datetime.now(UTC).isoformat()
    )
    halts[halt_id] = {
        **row,
        "state": HaltState.ACKNOWLEDGED.value,
        "acknowledged": record.model_dump(),
    }
    _write(project, "halts.json", halts)
    return record


def routed_proposals(project: ProjectStore, route: str) -> list[Proposal]:
    """Every proposal routed to ``route`` and not yet resolved, oldest first."""
    queue = _read(project, "proposals.json")
    return [
        Proposal.model_validate(row["proposal"])
        for row in queue.values()
        if isinstance(row, dict)
        and row.get("route") == route
        and row.get("state") == ProposalState.ROUTED.value
    ]


def resolve_proposal(
    project: ProjectStore, proposal_id: str, *, accepted: bool, by: str, note: str
) -> None:
    """Record the owning path's decision on a routed proposal.

    Raises:
        KeyError: The proposal is not queued.
        IllegalTransition: It is not routed; the refusal is journaled.
    """
    queue = _read(project, "proposals.json")
    row = queue[proposal_id]
    assert isinstance(row, dict)
    target = ProposalState.ACCEPTED if accepted else ProposalState.DECLINED
    PROPOSAL_MACHINE.move(
        proposal_id,
        ProposalState(str(row["state"])),
        target.value,
        journal_of(project),
        target=target,
    )
    queue[proposal_id] = {**row, "state": target.value, "resolution": {"by": by, "note": note}}
    _write(project, "proposals.json", queue)

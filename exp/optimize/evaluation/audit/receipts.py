"""Immutable held-out audit receipts, their child attempts and reports, and the derived headline.

One receipt is minted per audit, before any of its runs is registered, and it owns the panel's
reservation in the root exposure ledger. It is never rewritten. What happens later is appended
beside it as immutable children:

- an attempt record for every replaced registration: a ``re-attempt`` when the retired attempt
  had claimed physical slots (the trace is bought again), a ``re-registration`` when it had
  claimed none (for example after a catalog-identity change);
- an in-place ``re-attempt`` for every run that ``--retry-incomplete`` resumed and that sent a
  lost request again: the run keeps its ID, and the record names each step re-dispatched after
  an unknown outcome, by stage, with its attempt count;
- a report record for every invocation: whether it prepared, stopped, evaluated, or replayed,
  and whether per-trace detail was shown;
- dispatch records for every run: a receipt about to claim a run's first slot appends one, and
  once a slot is claimed no other receipt can, so the latest names the receipt that dispatched
  it. Run IDs never name a receipt, so this is what says which receipt alone may complete it.

The receipt's basis is fixed when it is minted: ``independent`` when no audited lineage was
exposed or reserved by another receipt and no run already had a saved result, otherwise
``development-re-audit``. The headline is derived from the basis and every child: any
re-attempt makes an independent receipt "independent, qualified: trace re-attempts", and any
invalidation notice adds "invalidated: predating exposure" after "independent". A replay of a
fully saved receipt keeps the receipt's headline.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, Field, ValidationError

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.audit.exposure import LineageExposure, held_out_members
from exp.optimize.evaluation.audit.panel import (
    AuditCheckpoint,
    HeldOutPanel,
    MemoryProfile,
    Profile,
    run_slots,
)
from exp.optimize.evaluation.fidelity.recovery import (
    QUALIFICATION,
    UnknownOutcome,
    stage_counts,
)
from exp.optimize.evaluation.fidelity.replay import replay_case
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity, unknown_outcomes
from exp.simulation.mining.exposure_ledger import ExposureLedger, LedgerExposure
from exp.simulation.mining.fingerprint import SchemeIdentity
from exp.simulation.world_model.resumption import RecoveryStage

RECEIPT_TYPE = "held-out-audit-receipt"
ATTEMPT_TYPE = "held-out-audit-attempt"
REPORT_TYPE = "held-out-audit-report"
NOTICE_TYPE = "held-out-audit-notice"
DISPATCH_TYPE = "held-out-audit-dispatch"
DETAIL_REPORT = "exp eval --audit --verbose (per-trace detail)"
QUALIFIED = "independent, " + QUALIFICATION
INVALIDATION = "invalidated: predating exposure"
Basis = Literal["independent", "development-re-audit"]
ReportKind = Literal["prepared", "stopped", "evaluation", "replay"]


class AuditRequest(ContractModel):
    """What an invocation asked for; a request resumes its latest active receipt.

    Attributes:
        profile: Simulator profile.
        count: Maximum held-out traces.
        include_inspected: Whether exposed or reserved lineages may be selected.
        memory: Memory profile of a ``memory`` audit; ``learned`` reads all learned memory.
        checkpoint: Earlier frozen memory version read, by consumed fit trajectories; absent
            reads the final checkpoint.
    """

    profile: Profile
    count: int = Field(ge=1)
    include_inspected: bool = False
    memory: MemoryProfile = Field(default="learned", exclude_if=lambda value: value == "learned")
    checkpoint: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)


class AuditCase(ContractModel):
    """One audited case, frozen in the receipt.

    Attributes:
        task_id: Held-out task.
        trace_id: Replayed trace.
        lineage: The build's leakage lineage.
        base_run: First-generation content-addressed run ID.
    """

    task_id: str
    trace_id: str
    lineage: str
    base_run: str


class AuditReceipt(ContractModel):
    """The immutable root record of one held-out audit.

    Attributes:
        sequence: Position among the project's receipts, from 1, assigned inside the root
            exposure ledger's transaction; it, not ``opened_at``, orders receipts.
        opened_at: When the receipt and its reservation were committed, by the minting host's
            clock; informational and part of the receipt identity only.
        project: Project audited.
        request: What the minting invocation asked for.
        basis: ``independent`` or ``development-re-audit``, fixed at minting.
        panel: Panel identity over the cases and their first-generation runs.
        code_revision: Implementation of the frozen configuration; part of every run ID.
        world_model: Simulator every run is bound to.
        checkpoint: Frozen memory checkpoint read, for the memory profile.
        cases: Audited cases, in panel order.
        schemes: Full identity of every fingerprint scheme the exposure match used.
        ledger: Ledger snapshot identity the reservation was decided and committed against.
        prior_exposures: Record IDs of every exposure in that snapshot, sorted: the exposure
            already recorded when the receipt was minted. Late-link reassessment counts these,
            and any exposure it cannot prove was recorded here after them.
        exposed: Every exposed or reserved held-out lineage at decision time, with its
            component size and the edge path to marked content.
        reaudited: Audited lineages that were exposed or reserved by another receipt.
        follows: An earlier receipt over the same cases whose configuration differed.
        follow_reason: ``released-unused`` when that receipt had claimed no slot and was
            released, ``dispatched`` when its claimed slots keep its reservation permanent.
    """

    sequence: int = Field(ge=1)
    opened_at: AwareDatetime
    project: str
    request: AuditRequest
    basis: Basis
    panel: str
    code_revision: str = Field(min_length=1)
    world_model: ArtifactInput
    checkpoint: AuditCheckpoint | None = None
    cases: tuple[AuditCase, ...] = Field(min_length=1)
    schemes: tuple[SchemeIdentity, ...] = Field(min_length=1)
    ledger: str
    prior_exposures: tuple[str, ...]
    exposed: tuple[LineageExposure, ...] = ()
    reaudited: tuple[str, ...] = ()
    follows: str | None = None
    follow_reason: Literal["released-unused", "dispatched"] | None = None

    @property
    def receipt_id(self) -> str:
        """Artifact ID of this receipt, fixed when it is minted."""
        return stable_id(
            RECEIPT_TYPE,
            {
                "project": self.project,
                "panel": self.panel,
                "code_revision": self.code_revision,
                "opened_at": self.opened_at.isoformat(),
            },
        )

    @property
    def lineages(self) -> tuple[str, ...]:
        """Audited lineages, in panel order."""
        return tuple(case.lineage for case in self.cases)

    @property
    def base_runs(self) -> tuple[str, ...]:
        """First-generation run IDs, in panel order."""
        return tuple(case.base_run for case in self.cases)


class AuditAttempt(ContractModel):
    """A replaced registration of one case, recorded as a child of its receipt.

    Attributes:
        receipt: Owning receipt.
        task_id: Case whose run was replaced.
        trace_id: Its trace.
        original_run: The case's first-generation run ID.
        retired_run: Attempt retired; for an in-place re-attempt, the resumed run itself.
        replacement_run: Attempt registered in its place; the same run when resumed in place.
        reason: ``re-attempt`` when the retired attempt had claimed slots, or when a resumed
            run sent a lost request again; otherwise ``re-registration``.
        claimed_slots: Physical slots the retired (or resumed) attempt had claimed.
        at: When the replacement was registered, or the re-dispatch recorded.
        redispatched: For an in-place re-attempt, every step re-dispatched after an unknown
            outcome, by stage, with its attempt count; empty for a replaced registration.
    """

    receipt: str
    task_id: str
    trace_id: str
    original_run: str
    retired_run: str
    replacement_run: str
    reason: Literal["re-attempt", "re-registration"]
    claimed_slots: int = Field(ge=0)
    at: AwareDatetime
    redispatched: tuple[UnknownOutcome, ...] = Field(default=(), exclude_if=lambda rows: not rows)

    @property
    def in_place(self) -> bool:
        """Whether the run was resumed in place rather than replaced."""
        return self.retired_run == self.replacement_run


class AuditReport(ContractModel):
    """What one invocation of a receipt did and showed.

    Attributes:
        receipt: Receipt reported.
        at: When the report was recorded, before anything it covers was shown.
        kind: ``prepared`` (dry run), ``stopped`` (declined or failed before results),
            ``evaluation`` (results reported after fresh runs), or ``replay`` (every result
            was already saved and reported before).
        runs: Current attempt of every case, in panel order.
        detail_shown: Whether per-trace detail was shown.
    """

    receipt: str
    at: AwareDatetime
    kind: ReportKind
    runs: tuple[str, ...]
    detail_shown: bool = False


class AuditNotice(ContractModel):
    """Exposure predating the reservation, found through links recorded after it.

    Appended once execution began; the receipt and its snapshot are never rewritten.

    Attributes:
        receipt: Receipt invalidated.
        at: When the reassessment found it.
        ledger: Snapshot identity of the reassessment.
        exposed: Audited lineages with their path to the predating exposure.
    """

    receipt: str
    at: AwareDatetime
    ledger: str
    exposed: tuple[LineageExposure, ...] = Field(min_length=1)


class AuditDispatch(ContractModel):
    """A receipt about to claim a run's first physical slot; immutable.

    A receipt that stopped before its claim leaves a record with no slot behind it, so another
    receipt may append the next one; the latest record names the dispatcher.

    Attributes:
        receipt: Receipt about to claim the slot.
        run: Registered run ID.
        attempt: Position among the run's dispatch records, from 1.
        at: When the claim was guarded.
    """

    receipt: str
    run: str
    attempt: int = Field(ge=1)
    at: AwareDatetime


class ReceiptView(ContractModel):
    """A receipt with every child appended so far.

    Attributes:
        receipt: The immutable root record.
        attempts: Replaced registrations, in recorded-time order (display only).
        reports: Invocations, in recorded-time order (display only).
        notices: Invalidation notices, in recorded-time order (display only).
    """

    receipt: AuditReceipt
    attempts: tuple[AuditAttempt, ...] = ()
    reports: tuple[AuditReport, ...] = ()
    notices: tuple[AuditNotice, ...] = ()

    @property
    def reattempts(self) -> tuple[AuditAttempt, ...]:
        """Replacements that bought a trace again."""
        return tuple(item for item in self.attempts if item.reason == "re-attempt")

    @property
    def affected_traces(self) -> frozenset[str]:
        """Traces with at least one re-attempt."""
        return frozenset(item.trace_id for item in self.reattempts)

    @property
    def redispatched(self) -> dict[RecoveryStage, int]:
        """Steps re-dispatched after an unknown outcome, by stage, over the current runs.

        A run resumed more than once has one in-place child per resumption that grew its
        retries; only its latest child counts. The run's durable recovery log only grows, so
        the latest child is the one recording the most dispatches, never the latest clock.
        """
        latest: dict[str, AuditAttempt] = {}
        for attempt in self.reattempts:
            if attempt.redispatched and _dispatches(attempt) > _dispatches(
                latest.get(attempt.replacement_run)
            ):
                latest[attempt.replacement_run] = attempt
        return stage_counts(row for attempt in latest.values() for row in attempt.redispatched)

    @property
    def headline(self) -> str:
        """The re-audit basis, or ``independent`` marked invalidated and qualified as needed.

        Any invalidation notice marks it invalidated, and any re-attempt qualifies it, so a
        receipt is plain ``independent`` only with neither.
        """
        if self.receipt.basis == "development-re-audit":
            return "development re-audit"
        parts = ["independent"]
        if self.notices:
            parts.append(INVALIDATION)
        if self.reattempts:
            parts.append(QUALIFICATION)
        return ", ".join(parts)

    @property
    def reported(self) -> bool:
        """Whether results were reported before."""
        return any(report.kind in ("evaluation", "replay") for report in self.reports)


def _dispatches(attempt: AuditAttempt | None) -> tuple[int, int]:
    """Dispatches an in-place child recorded, then its steps; grows with every resumption."""
    if attempt is None:
        return (0, 0)
    return (sum(row.attempts for row in attempt.redispatched), len(attempt.redispatched))


def _load_receipt(project: ProjectStore, identity: str) -> AuditReceipt:
    """Parse one saved receipt; a receipt of an older contract fails closed with what to do."""
    try:
        return AuditReceipt.model_validate_json(
            project.artifacts.read_bytes(identity, "receipt.json")
        )
    except ValidationError as error:
        raise ValueError(
            f"held-out audit receipt {identity} was saved under an older receipt contract that "
            "did not record the exposure it was decided against, so its independence cannot be "
            "reassessed; audit this build in a fresh project (exp build) to mint current receipts"
        ) from error


def audit_receipts(project: ProjectStore) -> tuple[AuditReceipt, ...]:
    """Every receipt of the project, in sequence order.

    Args:
        project: Project holding the receipts.

    Returns:
        Root receipts ordered by sequence; ``opened_at`` never orders them.
    """
    receipts = (
        _load_receipt(project, identity)
        for identity in project.artifacts.list_ids(artifact_type=RECEIPT_TYPE)
    )
    return tuple(sorted(receipts, key=lambda receipt: (receipt.sequence, receipt.receipt_id)))


@contextmanager
def receipts_held(project: ProjectStore) -> Iterator[int]:
    """Hold receipt minting and yield the latest receipt sequence of the project.

    Receipts are minted, and their sequence assigned, only inside the root exposure ledger's
    transaction, so while it is held here no receipt can be minted. A record written inside
    this context that stores the yielded sequence proves causally that every receipt with a
    larger sequence was minted after it was written, whatever any clock says.

    Args:
        project: Project whose receipts are held.

    Yields:
        The largest receipt sequence so far, or 0 before the first receipt.
    """
    with ExposureLedger(project.paths.root).transaction():
        yield max((receipt.sequence for receipt in audit_receipts(project)), default=0)


def receipt_reference(project: ProjectStore, receipt_id: str) -> ArtifactInput:
    """The exact artifact reference of one saved receipt, for a record that cites it.

    Args:
        project: Project holding the receipt.
        receipt_id: Receipt identity.

    Returns:
        The receipt's identity and manifest digest.

    Raises:
        ValueError: The project has no such audit receipt.
    """
    find_receipt(project, receipt_id)
    return artifact_input(project.artifacts.read(receipt_id).manifest)


def receipt_view(project: ProjectStore, receipt: AuditReceipt) -> ReceiptView:
    """A receipt with its children.

    Args:
        project: Project holding the records.
        receipt: Root receipt.

    Returns:
        The receipt, its attempts and its reports, each in time order.
    """
    attempts = [
        AuditAttempt.model_validate_json(project.artifacts.read_bytes(identity, "record.json"))
        for identity in project.artifacts.list_ids(artifact_type=ATTEMPT_TYPE)
    ]
    reports = [
        AuditReport.model_validate_json(project.artifacts.read_bytes(identity, "record.json"))
        for identity in project.artifacts.list_ids(artifact_type=REPORT_TYPE)
    ]
    notices = [
        AuditNotice.model_validate_json(project.artifacts.read_bytes(identity, "record.json"))
        for identity in project.artifacts.list_ids(artifact_type=NOTICE_TYPE)
    ]
    receipt_id = receipt.receipt_id
    return ReceiptView(
        notices=tuple(
            sorted((item for item in notices if item.receipt == receipt_id), key=lambda n: n.at)
        ),
        receipt=receipt,
        attempts=tuple(
            sorted(
                (item for item in attempts if item.receipt == receipt_id),
                key=lambda item: (item.at, item.retired_run),
            )
        ),
        reports=tuple(
            sorted((item for item in reports if item.receipt == receipt_id), key=lambda r: r.at)
        ),
    )


def find_receipt(project: ProjectStore, receipt_id: str) -> AuditReceipt:
    """The saved root receipt with this identity.

    Args:
        project: Project holding the receipts.
        receipt_id: Receipt identity, as audit reports print it.

    Returns:
        The receipt.

    Raises:
        ValueError: The project has no such audit receipt.
    """
    if project.artifacts.exists(receipt_id):
        manifest = project.artifacts.read(receipt_id).manifest
        if manifest.artifact_type == RECEIPT_TYPE:
            return _load_receipt(project, receipt_id)
    raise ValueError(
        f"no held-out audit receipt {receipt_id!r} in this project; exp eval PROJECT --audit N "
        "prints each receipt it opens"
    )


def receipt_panel(project: ProjectStore, receipt: AuditReceipt) -> HeldOutPanel:
    """Rebuild a receipt's registered panel from its saved registrations, without calls.

    Each case's current registration is the latest saved generation of its first-generation
    run, and the case replays the public task that registration froze. The rebuilt panel must
    reproduce the receipt's panel identity, so a changed build or a lost registration refuses
    instead of measuring other runs.

    Args:
        project: Project holding the receipt's registrations and the build's traces.
        receipt: Root receipt.

    Returns:
        The panel whose saved results the receipt measures.

    Raises:
        ValueError: A case has no saved registration or trace, or the rebuilt panel differs.
    """
    build = project.load_project().build
    if build is None:
        raise ValueError("audit receipts are read against a completed build; run exp build")
    dataset = load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id)
    traces = {trace.trace_id: trace for trace in dataset.traces}
    cases, prepared = [], []
    for case in receipt.cases:
        current = current_registration(project, case.base_run)
        trace = traces.get(case.trace_id)
        replay = (
            None if trace is None or current is None else replay_case(current.spec.task, trace, ())
        )
        if current is None or replay is None:
            raise ValueError(
                f"audit receipt {receipt.receipt_id} lost the registration or trace of "
                f"{case.trace_id}; measure receipts of the current build only"
            )
        cases.append(replay.model_copy(update={"task": current.spec.task}))
        prepared.append(current)
    panel = HeldOutPanel(
        profile=receipt.request.profile,
        world_model=receipt.world_model,
        checkpoint=receipt.checkpoint,
        cases=tuple(cases),
        specs=tuple(row.spec for row in prepared),
        code_revision=receipt.code_revision,
        prepared=tuple(prepared),
    )
    if panel.identity != receipt.panel:
        raise ValueError(f"audit receipt {receipt.receipt_id} no longer matches its saved runs")
    return panel


def current_registration(project: ProjectStore, base_run: str) -> PreparedFidelity | None:
    """The latest saved attempt generation of one first-generation run.

    Args:
        project: Project holding the registrations.
        base_run: First-generation run ID.

    Returns:
        The latest registration, or None when the run was never registered.
    """
    current = None
    generation = 1
    while True:
        run_id = base_run if generation == 1 else f"{base_run}-r{generation}"
        plan_id = stable_id("fidelity", {"run_id": run_id})
        if not project.artifacts.exists(plan_id):
            return current
        current = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        generation += 1


def save_receipt(project: ProjectStore, receipt: AuditReceipt) -> None:
    """Write a newly minted receipt once.

    Args:
        project: Project audited.
        receipt: Root receipt.
    """
    _write(project, RECEIPT_TYPE, receipt.receipt_id, "receipt.json", receipt, receipt)


def record_attempts(
    project: ProjectStore,
    receipt: AuditReceipt,
    panel: HeldOutPanel,
    *,
    clock: Callable[[], datetime],
) -> tuple[AuditAttempt, ...]:
    """Append a child for every registration the panel's owner replaced.

    The caller holds the panel lease and has checked that ``receipt`` owns an active or
    dispatched reservation for exactly this frozen configuration.

    Args:
        project: Project audited.
        receipt: Owning receipt.
        panel: The panel as just registered.
        clock: Aware clock.

    Returns:
        The appended attempts.
    """
    now = clock()
    attempts = []
    for item in panel.retirements:
        case = receipt.cases[item.case]
        attempt = AuditAttempt(
            receipt=receipt.receipt_id,
            task_id=case.task_id,
            trace_id=case.trace_id,
            original_run=case.base_run,
            retired_run=item.retired,
            replacement_run=item.replacement,
            reason="re-attempt" if item.claimed_slots else "re-registration",
            claimed_slots=item.claimed_slots,
            at=now,
        )
        identity = stable_id(
            ATTEMPT_TYPE, {"receipt": receipt.receipt_id, "retired_run": item.retired}
        )
        _write(project, ATTEMPT_TYPE, identity, "record.json", attempt, receipt)
        attempts.append(attempt)
    return tuple(attempts)


def record_notice(project: ProjectStore, receipt: AuditReceipt, notice: AuditNotice) -> bool:
    """Append an invalidation notice unless the same exposure was already noticed.

    Args:
        project: Project audited.
        receipt: Receipt invalidated.
        notice: What the reassessment found.

    Returns:
        Whether a new notice was appended.
    """
    marks = sorted((item.lineage, *(item.path.by if item.path else ())) for item in notice.exposed)
    identity = stable_id(NOTICE_TYPE, {"receipt": receipt.receipt_id, "marks": marks})
    if project.artifacts.exists(identity):
        return False
    _write(project, NOTICE_TYPE, identity, "record.json", notice, receipt)
    return True


def record_redispatches(
    project: ProjectStore,
    receipt: AuditReceipt,
    panel: HeldOutPanel,
    *,
    clock: Callable[[], datetime],
) -> tuple[AuditAttempt, ...]:
    """Append an in-place re-attempt for every run that re-dispatched a lost request.

    The caller holds the panel lease. Each run's retries are read from its durable recovery
    log, so a retry recorded by an invocation that stopped before reporting is still appended.
    A run whose retries grew since its last child gets a new child; nothing is rewritten. The
    child's identity names its receipt, so a child another receipt recorded over the same run
    never stands in for this receipt's own.

    Args:
        project: Project audited.
        receipt: Owning receipt.
        panel: The panel as registered.
        clock: Aware clock.

    Returns:
        The appended attempts.
    """
    attempts = []
    for index, prepared in enumerate(panel.prepared):
        rows = unknown_outcomes(project, prepared)
        if not rows:
            continue
        run = prepared.spec.run_id
        identity = stable_id(
            ATTEMPT_TYPE,
            {
                "receipt": receipt.receipt_id,
                "resumed_run": run,
                "redispatched": [row.model_dump(mode="json") for row in rows],
            },
        )
        if project.artifacts.exists(identity):
            continue
        case = receipt.cases[index]
        attempt = AuditAttempt(
            receipt=receipt.receipt_id,
            task_id=case.task_id,
            trace_id=case.trace_id,
            original_run=case.base_run,
            retired_run=run,
            replacement_run=run,
            reason="re-attempt",
            claimed_slots=run_slots(project, run),
            at=clock(),
            redispatched=rows,
        )
        _write(project, ATTEMPT_TYPE, identity, "record.json", attempt, receipt)
        attempts.append(attempt)
    return tuple(attempts)


def dispatch_owner(project: ProjectStore, run_id: str) -> str | None:
    """The receipt that dispatched a run, or None when no audit claimed a slot for it.

    Args:
        project: Project holding the run.
        run_id: Registered run ID.

    Returns:
        The owning receipt's identity, when the run claimed a slot and a dispatch record names
        who; None otherwise.
    """
    records = _dispatch_records(project, run_id)
    if not records or not run_slots(project, run_id):
        return None
    return records[-1].receipt


def record_dispatch(
    project: ProjectStore,
    receipt: AuditReceipt,
    run_id: str,
    *,
    clock: Callable[[], datetime],
) -> str | None:
    """Make ``receipt`` the run's dispatcher unless another receipt already claimed a slot.

    The caller holds the run's execution lock, so the slot count and records read here cannot
    change underneath. Before any slot is claimed, a record left by a receipt that stopped
    before its claim confers nothing, and this receipt's record follows it.

    Args:
        project: Project holding the run.
        receipt: Receipt about to claim a slot for the run.
        run_id: Registered run ID.
        clock: Aware clock.

    Returns:
        The dispatching receipt's identity, which differs from ``receipt``'s when another
        receipt claimed first; None when slots were claimed with no record of who claimed them.
    """
    records = _dispatch_records(project, run_id)
    if run_slots(project, run_id):
        return records[-1].receipt if records else None
    if records and records[-1].receipt == receipt.receipt_id:
        return receipt.receipt_id
    attempt = len(records) + 1
    record = AuditDispatch(receipt=receipt.receipt_id, run=run_id, attempt=attempt, at=clock())
    identity = stable_id(DISPATCH_TYPE, {"run": run_id, "attempt": attempt})
    _write(project, DISPATCH_TYPE, identity, "record.json", record, receipt)
    return receipt.receipt_id


def _dispatch_records(project: ProjectStore, run_id: str) -> tuple[AuditDispatch, ...]:
    """Every dispatch record of one run, in attempt order."""
    records = []
    while project.artifacts.exists(
        identity := stable_id(DISPATCH_TYPE, {"run": run_id, "attempt": len(records) + 1})
    ):
        records.append(
            AuditDispatch.model_validate_json(project.artifacts.read_bytes(identity, "record.json"))
        )
    return tuple(records)


def record_report(
    project: ProjectStore,
    receipt: AuditReceipt,
    panel: HeldOutPanel,
    *,
    kind: ReportKind,
    detail_shown: bool,
    clock: Callable[[], datetime],
) -> AuditReport:
    """Append what this invocation did, and expose what it will show, before showing it.

    Per-trace detail exposes every member trace of the audited lineages in the root ledger.

    Args:
        project: Project audited.
        receipt: Owning receipt.
        panel: The panel as registered.
        kind: What the invocation did.
        detail_shown: Whether per-trace detail will be shown.
        clock: Aware clock.

    Returns:
        The appended report.
    """
    report = AuditReport(
        receipt=receipt.receipt_id,
        at=clock(),
        kind=kind,
        runs=tuple(prepared.spec.run_id for prepared in panel.prepared),
        detail_shown=detail_shown,
    )
    identity = stable_id(REPORT_TYPE, report)
    _write(project, REPORT_TYPE, identity, "record.json", report, receipt)
    if detail_shown:
        members = held_out_members(project)
        ExposureLedger(project.paths.root).record_exposure(
            LedgerExposure(
                vertices=tuple(
                    sorted(
                        {
                            vertex
                            for lineage in receipt.lineages
                            for vertex in members[lineage].vertices
                        }
                    )
                ),
                how="inspected" if receipt.basis == "independent" else "re-audited",
                panel=receipt.panel,
                receipt=receipt.receipt_id,
                report=DETAIL_REPORT,
                project=project.paths.project_id,
                at=report.at,
            )
        )
    return report


def _write(
    project: ProjectStore,
    artifact_type: str,
    identity: str,
    path: str,
    record: ContractModel,
    receipt: AuditReceipt,
) -> None:
    """Write one immutable record unless it already exists."""
    if project.artifacts.exists(identity):
        return
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=receipt.opened_at,
            code_revision=receipt.code_revision,
            inputs=(receipt.world_model,),
        ),
        files={path: record.model_dump(mode="json")},
    )

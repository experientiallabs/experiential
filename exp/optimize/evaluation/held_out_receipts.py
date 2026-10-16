"""Immutable held-out audit receipts, their child attempts and reports, and the derived headline.

One receipt is minted per audit, before any of its runs is registered, and it owns the panel's
reservation in the root exposure ledger. It is never rewritten. What happens later is appended
beside it as immutable children:

- an attempt record for every replaced registration: a ``re-attempt`` when the retired attempt
  had claimed physical slots (the trace is bought again), a ``re-registration`` when it had
  claimed none (for example after a catalog-identity change);
- a report record for every invocation: whether it prepared, stopped, evaluated, or replayed,
  and whether per-trace detail was shown.

The receipt's basis is fixed when it is minted: ``independent`` when no audited lineage was
exposed or reserved by another receipt and no run already had a saved result, otherwise
``development-re-audit``. The headline is derived from the basis and every child: any
re-attempt makes an independent receipt "independent, qualified: trace re-attempts". A replay
of a fully saved receipt keeps the receipt's headline.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.project import ProjectStore
from exp.optimize.evaluation.held_out_exposure import LineageExposure, held_out_members
from exp.optimize.evaluation.held_out_fidelity import AuditCheckpoint, HeldOutPanel, Profile
from exp.simulation.mining.exposure_ledger import ExposureLedger, LedgerExposure

RECEIPT_TYPE = "held-out-audit-receipt"
ATTEMPT_TYPE = "held-out-audit-attempt"
REPORT_TYPE = "held-out-audit-report"
DETAIL_REPORT = "exp eval --held-out-fidelity --verbose (per-trace detail)"
QUALIFIED = "independent, qualified: trace re-attempts"
Basis = Literal["independent", "development-re-audit"]
ReportKind = Literal["prepared", "stopped", "evaluation", "replay"]


class AuditRequest(ContractModel):
    """What an invocation asked for; a request resumes its latest active receipt.

    Attributes:
        profile: Simulator profile.
        count: Maximum held-out traces.
        include_inspected: Whether exposed or reserved lineages may be selected.
    """

    profile: Profile
    count: int = Field(ge=1)
    include_inspected: bool = False


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
        sequence: Position among the project's receipts, from 1.
        opened_at: When the receipt and its reservation were committed.
        project: Project audited.
        request: What the minting invocation asked for.
        basis: ``independent`` or ``development-re-audit``, fixed at minting.
        panel: Panel identity over the cases and their first-generation runs.
        code_revision: Implementation of the frozen configuration; not part of run IDs.
        world_model: Simulator every run is bound to.
        checkpoint: Frozen memory checkpoint read, for the memory profile.
        cases: Audited cases, in panel order.
        scheme: Content fingerprint scheme and version of the exposure decision.
        ledger: Ledger snapshot identity the reservation was decided and committed against.
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
    scheme: str
    ledger: str
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
        retired_run: Attempt retired.
        replacement_run: Attempt registered in its place.
        reason: ``re-attempt`` when the retired attempt had claimed slots, otherwise
            ``re-registration``.
        claimed_slots: Physical slots the retired attempt had claimed.
        at: When the replacement was registered.
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


class ReceiptView(ContractModel):
    """A receipt with every child appended so far.

    Attributes:
        receipt: The immutable root record.
        attempts: Replaced registrations, in time order.
        reports: Invocations, in time order.
    """

    receipt: AuditReceipt
    attempts: tuple[AuditAttempt, ...] = ()
    reports: tuple[AuditReport, ...] = ()

    @property
    def reattempts(self) -> tuple[AuditAttempt, ...]:
        """Replacements that bought a trace again."""
        return tuple(item for item in self.attempts if item.reason == "re-attempt")

    @property
    def affected_traces(self) -> frozenset[str]:
        """Traces with at least one re-attempt."""
        return frozenset(item.trace_id for item in self.reattempts)

    @property
    def headline(self) -> str:
        """``independent``, the qualified form after any re-attempt, or the re-audit basis."""
        if self.receipt.basis == "development-re-audit":
            return "development re-audit"
        return QUALIFIED if self.reattempts else "independent"

    @property
    def reported(self) -> bool:
        """Whether results were reported before."""
        return any(report.kind in ("evaluation", "replay") for report in self.reports)


def audit_receipts(project: ProjectStore) -> tuple[AuditReceipt, ...]:
    """Every receipt of the project, in sequence order.

    Args:
        project: Project holding the receipts.

    Returns:
        Root receipts ordered by sequence, then time.
    """
    receipts = (
        AuditReceipt.model_validate_json(project.artifacts.read_bytes(identity, "receipt.json"))
        for identity in project.artifacts.list_ids(artifact_type=RECEIPT_TYPE)
    )
    return tuple(sorted(receipts, key=lambda receipt: (receipt.sequence, receipt.opened_at)))


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
    receipt_id = receipt.receipt_id
    return ReceiptView(
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
        identity = stable_id(ATTEMPT_TYPE, {"retired_run": item.retired})
        _write(project, ATTEMPT_TYPE, identity, "record.json", attempt, receipt)
        attempts.append(attempt)
    return tuple(attempts)


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
                fingerprints=tuple(
                    sorted(
                        {
                            fingerprint
                            for lineage in receipt.lineages
                            for fingerprint in members[lineage]
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

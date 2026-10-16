"""Immutable held-out audit receipts, their status, and the exposure they record.

Every ``exp eval PROJECT --held-out-fidelity N`` run saves one receipt in the project before it
shows anything. Its status is exactly one of:

- ``independent``: the first fresh evaluation of a panel with no exposed case, reported for the
  first time.
- ``replay``: the same panel identity and code revision as an earlier reported receipt, with every
  result already saved; it names that original receipt, which keeps its own status.
- ``development-re-audit``: any other evaluation, including one that touches exposed cases, a
  repeat under a different code revision, or results that were saved but never reported.

Showing per-trace detail exposes every member trace of the audited lineages in the root exposure
ledger, so the receipt is written first and the exposure next, before the detail is printed.
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
DETAIL_REPORT = "exp eval --held-out-fidelity --verbose (per-trace detail)"
AuditStatus = Literal["independent", "replay", "development-re-audit"]


class AuditReceipt(ContractModel):
    """Immutable record of one held-out audit and what it showed.

    Attributes:
        sequence: Position in the project's audit ledger, from 1.
        audited_at: When the audit recorded its receipt.
        status: ``independent``, ``replay`` or ``development-re-audit``.
        replays: Receipt ID of the original audit, for a replay.
        panel: Panel identity over the selected cases and their content-addressed runs.
        code_revision: Implementation that ran the audit; not part of the run IDs.
        profile: Simulator profile measured.
        world_model: Simulator every run was bound to.
        checkpoint: Frozen memory checkpoint read, for the memory profile.
        lineages: Task lineages audited, in panel order.
        traces: Recorded traces replayed, in panel order.
        runs: Fidelity run IDs, in panel order.
        fresh: Whether any run had no saved result when the audit started.
        scheme: Content fingerprint scheme and version of the exposure decision.
        ledger: Snapshot identity of the exposure ledger the decision used.
        exposed: Every exposed held-out lineage of the build at decision time, with its
            component size and the edge path to exposed content.
        reaudited: Audited lineages that were already exposed.
        results_shown: Whether results were reported; false for a dry run or stopped audit.
        detail_shown: Whether per-trace detail was shown, which exposes every audited lineage.
    """

    sequence: int = Field(ge=1)
    audited_at: AwareDatetime
    status: AuditStatus
    replays: str | None = None
    panel: str
    code_revision: str = Field(min_length=1)
    profile: Profile
    world_model: ArtifactInput
    checkpoint: AuditCheckpoint | None = None
    lineages: tuple[str, ...] = Field(min_length=1)
    traces: tuple[str, ...] = Field(min_length=1)
    runs: tuple[str, ...] = Field(min_length=1)
    fresh: bool
    scheme: str
    ledger: str
    exposed: tuple[LineageExposure, ...] = ()
    reaudited: tuple[str, ...] = ()
    results_shown: bool
    detail_shown: bool

    @property
    def receipt_id(self) -> str:
        """Artifact ID of this receipt."""
        return stable_id(RECEIPT_TYPE, self)


def audit_receipts(project: ProjectStore) -> tuple[AuditReceipt, ...]:
    """Every saved held-out audit receipt of the project, in ledger order.

    Args:
        project: Project holding the receipts.

    Returns:
        Receipts ordered by sequence, then time.
    """
    receipts = (
        AuditReceipt.model_validate_json(project.artifacts.read_bytes(identity, "receipt.json"))
        for identity in project.artifacts.list_ids(artifact_type=RECEIPT_TYPE)
    )
    return tuple(sorted(receipts, key=lambda receipt: (receipt.sequence, receipt.audited_at)))


def audit_status(
    project: ProjectStore, panel: HeldOutPanel, *, fresh: bool, code_revision: str
) -> tuple[AuditStatus, AuditReceipt | None]:
    """Classify an audit of ``panel`` against the project's earlier receipts.

    Args:
        project: Project holding the receipts.
        panel: Registered panel.
        fresh: Whether any run had no saved result when the audit started.
        code_revision: Implementation running the audit.

    Returns:
        The status, and the original receipt for a replay.
    """
    reported = [
        receipt
        for receipt in audit_receipts(project)
        if receipt.panel == panel.identity and receipt.results_shown
    ]
    same_code = [receipt for receipt in reported if receipt.code_revision == code_revision]
    if not fresh and same_code:
        return "replay", same_code[0]
    if panel.inspected or reported or not fresh:
        return "development-re-audit", None
    return "independent", None


def record_audit_receipt(
    project: ProjectStore,
    panel: HeldOutPanel,
    *,
    fresh: bool,
    results_shown: bool,
    detail_shown: bool,
    code_revision: str,
    clock: Callable[[], datetime],
) -> AuditReceipt:
    """Save an immutable receipt, and the exposure it causes, before showing anything.

    Args:
        project: Project the panel belongs to.
        panel: Registered panel.
        fresh: Whether any run had no saved result when the audit started.
        results_shown: Whether the audit reports results.
        detail_shown: Whether the audit shows per-trace detail; every member trace of the
            audited lineages is then exposed in the root ledger.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for the receipt time.

    Returns:
        The saved receipt.
    """
    status, original = audit_status(project, panel, fresh=fresh, code_revision=code_revision)
    receipt = AuditReceipt(
        sequence=len(project.artifacts.list_ids(artifact_type=RECEIPT_TYPE)) + 1,
        audited_at=clock(),
        status=status,
        replays=original.receipt_id if original else None,
        panel=panel.identity,
        code_revision=code_revision,
        profile=panel.profile,
        world_model=panel.world_model,
        checkpoint=panel.checkpoint,
        lineages=tuple(case.task.lineage_group_id for case in panel.cases),
        traces=tuple(case.trace_id for case in panel.cases),
        runs=tuple(prepared.spec.run_id for prepared in panel.prepared),
        fresh=fresh,
        scheme=panel.exposure.scheme,
        ledger=panel.exposure.ledger,
        exposed=panel.exposure.exposed,
        reaudited=panel.inspected,
        results_shown=results_shown,
        detail_shown=detail_shown,
    )
    project.artifacts.write_json(
        artifact_id=receipt.receipt_id,
        artifact_type=RECEIPT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=receipt.audited_at,
            code_revision=code_revision,
            inputs=(panel.world_model,),
        ),
        files={"receipt.json": receipt.model_dump(mode="json")},
    )
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
                how="re-audited" if panel.inspected else "inspected",
                panel=panel.identity,
                receipt=receipt.receipt_id,
                report=DETAIL_REPORT,
                project=project.paths.project_id,
                at=receipt.audited_at,
            )
        )
    return receipt

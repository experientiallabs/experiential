"""Held-out audit lifecycle: reserve, register, claim, call; resume, replace and release.

An audit runs in a fixed order: the panel's content is reserved for its receipt in the root
exposure ledger, then its runs are registered, then each provider call claims a dispatch slot,
and only then is the call sent. One lock order holds everywhere (opening, resuming, releasing):
the panel lease, then the exposure-ledger transaction, then a run's execution lock, then its
dispatch ledger's ``BEGIN IMMEDIATE``. No earlier lock is taken while a later one is held.

- Opening: selection reads the ledger, then the panel lease is taken and, inside one ledger
  transaction, the graph is read again. If another audit reserved or exposed the chosen content
  in between, nothing is written and selection runs again; otherwise the receipt is minted and
  its reservation committed against that snapshot.
- Resuming: an invocation with the same request continues the latest receipt that still holds
  its reservation, when the selected cases, panel identity and code revision all match its
  frozen configuration. Replaced registrations become that receipt's children.
- Configuration change: with zero claimed slots the old reservation is released under its
  lease and a new receipt is minted; with any claimed slot the old reservation is permanent and
  the new receipt over those cases is a development re-audit.
- Dispatch: the lease is held for the whole execution, and only a lease holder can release, so
  the guard that runs before every slot claim checks that the lease is held and the
  reservation was confirmed active under it.
- Releasing: only the owner, under its lease and inside the ledger transaction, after
  recounting every claimed slot of every attempt; one claimed slot makes the reservation
  permanent ("dispatched, outcome unknown" until its results are saved).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

from filelock import FileLock

from exp.common.core.artifacts import stable_id
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import saved_fidelity_report
from exp.optimize.evaluation.held_out_exposure import (
    HeldOutExposure,
    held_out_exposure,
    held_out_members,
)
from exp.optimize.evaluation.held_out_fidelity import (
    HeldOutPanel,
    PlannedPanel,
    Profile,
    panel_slots,
    plan_held_out_panel,
    register_planned_panel,
)
from exp.optimize.evaluation.held_out_receipts import (
    AuditCase,
    AuditReceipt,
    AuditRequest,
    ReceiptView,
    audit_receipts,
    receipt_view,
    record_attempts,
    save_receipt,
)
from exp.simulation.mining.exposure_ledger import (
    ExposureLedger,
    LedgerRelease,
    LedgerReservation,
    LedgerSnapshot,
)

SELECTION_ROUNDS = 5  # concurrent audits that keep taking the chosen content
ReservationState = Literal["active", "dispatched", "released"]


class ReservationInactiveError(RuntimeError):
    """Dispatch was refused: the lease is not held or the reservation is not active."""


class HeldOutAudit:
    """An open audit: its receipt, registered panel and exposure decision, under its lease."""

    def __init__(
        self,
        project: ProjectStore,
        receipt: AuditReceipt,
        panel: HeldOutPanel,
        exposure: HeldOutExposure,
        lease: FileLock,
        *,
        excluded: int,
    ) -> None:
        """Bind an audit whose reservation was confirmed active under ``lease``.

        Args:
            project: Project audited.
            receipt: Owning receipt.
            panel: Registered panel.
            exposure: Exposure decision the selection used.
            lease: The held panel lease.
            excluded: Exposed or reserved lineages selection left out.
        """
        self.project = project
        self.receipt = receipt
        self.panel = panel
        self.exposure = exposure
        self.excluded = excluded
        self._lease = lease

    @property
    def view(self) -> ReceiptView:
        """The receipt with every child appended so far."""
        return receipt_view(self.project, self.receipt)

    def guard(self) -> None:
        """Refuse a slot claim unless this process still holds the lease.

        The reservation was confirmed active inside the ledger transaction after the lease was
        taken, and only a lease holder can release it, so it stays active while the lease is
        held.

        Raises:
            ReservationInactiveError: The lease is no longer held.
        """
        if not self._lease.is_locked:
            raise ReservationInactiveError(
                f"panel {self.receipt.panel} lease is not held; refusing to claim a slot"
            )


@contextmanager
def open_held_out_audit(
    project: ProjectStore,
    count: int,
    *,
    profile: Profile,
    code_revision: str,
    clock: Callable[[], datetime],
    retry_incomplete: bool = False,
    include_inspected: bool = False,
) -> Iterator[HeldOutAudit]:
    """Reserve and register a held-out audit, or resume the request's receipt, under its lease.

    Args:
        project: Project with a completed build.
        count: Maximum held-out traces.
        profile: Simulator profile.
        code_revision: Exact running implementation.
        clock: Aware clock.
        retry_incomplete: Let the owner replace interrupted runs that already bought calls.
        include_inspected: Also select exposed or reserved lineages, as a development re-audit.

    Yields:
        The open audit; the lease is held until the context exits.

    Raises:
        PanelInUseError: Another process holds the panel's lease.
        ValueError: Nothing can be selected, or selection kept changing under other audits.
    """
    ledger = ExposureLedger(project.paths.root)
    request = AuditRequest(profile=profile, count=count, include_inspected=include_inspected)
    for _ in range(SELECTION_ROUNDS):
        snapshot = ledger.snapshot()
        owner = _open_receipt(project, request, snapshot)
        exposure = held_out_exposure(project, snapshot, owner.receipt_id if owner else None)
        # Reservations made after the owner's never block its own cases; exposure always does.
        own = frozenset(owner.lineages) if owner else frozenset()
        excluded = (
            frozenset()
            if include_inspected
            else frozenset(
                item.lineage
                for item in exposure.exposed
                if item.kind == "exposed" or item.lineage not in own
            )
        )
        planned = plan_held_out_panel(
            project, count, profile=profile, clock=clock, excluded=excluded
        )
        follows: tuple[AuditReceipt, Literal["released-unused", "dispatched"]] | None = None
        if owner is not None and planned.lineages == owner.lineages:
            if planned.identity == owner.panel and code_revision == owner.code_revision:
                with ledger.lease(owner.panel) as lease:
                    with ledger.transaction() as transaction:
                        if not transaction.snapshot().active(owner.receipt_id):
                            continue
                    panel = register_planned_panel(
                        project,
                        planned,
                        code_revision=code_revision,
                        clock=clock,
                        retry_incomplete=retry_incomplete,
                    )
                    record_attempts(project, owner, panel, clock=clock)
                    yield HeldOutAudit(
                        project, owner, panel, exposure, lease, excluded=len(excluded)
                    )
                    return
            follows = (
                owner,
                "released-unused" if _release(ledger, project, owner, clock) else "dispatched",
            )
        with ledger.lease(planned.identity) as lease:
            receipt = _reserve(
                ledger,
                project,
                request,
                planned,
                code_revision=code_revision,
                clock=clock,
                follows=follows,
            )
            if receipt is None:
                continue
            panel = register_planned_panel(
                project,
                planned,
                code_revision=code_revision,
                clock=clock,
                retry_incomplete=retry_incomplete,
            )
            record_attempts(project, receipt, panel, clock=clock)
            yield HeldOutAudit(project, receipt, panel, exposure, lease, excluded=len(excluded))
            return
    raise ValueError(
        "other audits kept reserving the selected held-out content; run the command again"
    )


def release_held_out_audit(
    project: ProjectStore, receipt: AuditReceipt, *, clock: Callable[[], datetime]
) -> None:
    """Release an unused reservation; only its owner may, and only with zero claimed slots.

    Args:
        project: Project of the receipt.
        receipt: Owning receipt.
        clock: Aware clock.

    Raises:
        PanelInUseError: Another process holds the panel's lease.
        ValueError: The reservation is not active, or a slot was claimed so it is permanent.
    """
    ledger = ExposureLedger(project.paths.root)
    if not _release(ledger, project, receipt, clock):
        raise ValueError(
            f"receipt {receipt.receipt_id} claimed dispatch slots, so its reservation is "
            "permanent: the output may already have reached logs or screens"
        )


def reservation_state(project: ProjectStore, receipt: AuditReceipt) -> ReservationState:
    """``released``, ``active`` (no slot claimed) or ``dispatched`` (permanent).

    Args:
        project: Project of the receipt.
        receipt: Owning receipt.

    Returns:
        The reservation's state.
    """
    if not ExposureLedger(project.paths.root).snapshot().active(receipt.receipt_id):
        return "released"
    return "dispatched" if panel_slots(project, receipt.base_runs) else "active"


def outcome_unknown(project: ProjectStore, audit: HeldOutAudit) -> bool:
    """Whether the panel dispatched but some current attempt has no saved result."""
    return reservation_state(project, audit.receipt) == "dispatched" and any(
        saved_fidelity_report(project, prepared) is None for prepared in audit.panel.prepared
    )


def _open_receipt(
    project: ProjectStore, request: AuditRequest, snapshot: LedgerSnapshot
) -> AuditReceipt | None:
    """The latest receipt for this request that still holds its reservation."""
    for receipt in reversed(audit_receipts(project)):
        if receipt.request == request and snapshot.active(receipt.receipt_id):
            return receipt
    return None


def _release(
    ledger: ExposureLedger,
    project: ProjectStore,
    receipt: AuditReceipt,
    clock: Callable[[], datetime],
) -> bool:
    """Under the lease and ledger transaction, release at zero claimed slots; report success."""
    with ledger.lease(receipt.panel), ledger.transaction() as transaction:
        if not transaction.snapshot().active(receipt.receipt_id):
            raise ValueError(f"receipt {receipt.receipt_id} holds no active reservation")
        if panel_slots(project, receipt.base_runs):
            return False
        transaction.release(LedgerRelease(receipt=receipt.receipt_id, at=clock()))
        return True


def _reserve(
    ledger: ExposureLedger,
    project: ProjectStore,
    request: AuditRequest,
    planned: PlannedPanel,
    *,
    code_revision: str,
    clock: Callable[[], datetime],
    follows: tuple[AuditReceipt, Literal["released-unused", "dispatched"]] | None,
) -> AuditReceipt | None:
    """Mint the receipt and commit its reservation, or None when the graph moved under us."""
    members = held_out_members(project)
    allowed = (
        frozenset(follows[0].lineages)
        if follows is not None and follows[1] == "dispatched"
        else frozenset()
    )
    with ledger.transaction() as transaction:
        snapshot = transaction.snapshot()
        exposure = held_out_exposure(project, snapshot)
        touched = tuple(lineage for lineage in planned.lineages if lineage in exposure.lineages)
        if not request.include_inspected and set(touched) - allowed:
            return None
        saved = _any_saved(project, planned.base_runs)
        receipt = AuditReceipt(
            sequence=len(audit_receipts(project)) + 1,
            opened_at=clock(),
            project=project.paths.project_id,
            request=request,
            basis="development-re-audit" if touched or saved else "independent",
            panel=planned.identity,
            code_revision=code_revision,
            world_model=planned.world_model,
            checkpoint=planned.checkpoint,
            cases=tuple(
                AuditCase(
                    task_id=case.task.task_id,
                    trace_id=case.trace_id,
                    lineage=case.task.lineage_group_id,
                    base_run=run,
                )
                for case, run in zip(planned.cases, planned.base_runs, strict=True)
            ),
            scheme=exposure.scheme,
            ledger=snapshot.identity,
            exposed=exposure.exposed,
            reaudited=touched,
            follows=follows[0].receipt_id if follows else None,
            follow_reason=follows[1] if follows else None,
        )
        save_receipt(project, receipt)
        transaction.reserve(
            LedgerReservation(
                receipt=receipt.receipt_id,
                panel=receipt.panel,
                project=receipt.project,
                lineages=receipt.lineages,
                fingerprints=tuple(
                    sorted({item for lineage in receipt.lineages for item in members[lineage]})
                ),
                snapshot=snapshot.identity,
                at=receipt.opened_at,
            )
        )
        return receipt


def _any_saved(project: ProjectStore, base_runs: tuple[str, ...]) -> bool:
    """Whether any generation of any run already has a saved result."""
    for base in base_runs:
        generation = 1
        while True:
            run_id = base if generation == 1 else f"{base}-r{generation}"
            plan = stable_id("fidelity", {"run_id": run_id})
            if not project.artifacts.exists(plan):
                break
            if project.artifacts.exists(f"{plan}-report"):
                return True
            generation += 1
    return False

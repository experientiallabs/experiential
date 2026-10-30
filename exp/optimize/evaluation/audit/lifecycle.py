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
  frozen configuration. Replaced registrations become that receipt's children. A run another
  receipt dispatched first and never finished is completed only by that receipt: any other
  receipt over it refuses to open, and its slot claims for that run are refused.
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
from exp.common.models.client import RequestAdmissionRefused
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.exposure import (
    HeldOutExposure,
    LineageExposure,
    held_out_exposure,
    held_out_members,
    refresh_ledger,
    unknown_scheme_refusal,
)
from exp.optimize.evaluation.audit.panel import (
    HeldOutPanel,
    MemoryProfile,
    PlannedPanel,
    Profile,
    held_out_cases,
    panel_slots,
    plan_held_out_panel,
    register_planned_panel,
    run_slots,
)
from exp.optimize.evaluation.audit.receipts import (
    AuditCase,
    AuditNotice,
    AuditReceipt,
    AuditRequest,
    ReceiptView,
    audit_receipts,
    current_registration,
    dispatch_owner,
    receipt_view,
    record_attempts,
    record_dispatch,
    record_notice,
    record_redispatches,
    save_receipt,
)
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity, saved_fidelity_report
from exp.optimize.evaluation.learning.traces import fit_calls
from exp.simulation.mining.exposure_ledger import (
    ExposureLedger,
    LedgerExposure,
    LedgerRelease,
    LedgerReservation,
    LedgerSnapshot,
)

SELECTION_ROUNDS = 5  # concurrent audits that keep taking the chosen content
ReservationState = Literal["active", "dispatched", "released"]


class ReservationInactiveError(RequestAdmissionRefused):
    """Dispatch was refused: the lease is not held or the reservation is not active.

    A local stop before dispatch, so it bypasses step-failure handling: the trace stops
    unscored instead of counting its remaining turns as misses.
    """


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
        clock: Callable[[], datetime],
    ) -> None:
        """Bind an audit whose reservation was confirmed active under ``lease``.

        Args:
            project: Project audited.
            receipt: Owning receipt.
            panel: Registered panel.
            exposure: Exposure decision the selection used.
            lease: The held panel lease.
            excluded: Exposed or reserved lineages selection left out.
            clock: Aware clock for appended children.
        """
        self.project = project
        self.receipt = receipt
        self.panel = panel
        self.exposure = exposure
        self.excluded = excluded
        self._lease = lease
        self._clock = clock

    @property
    def view(self) -> ReceiptView:
        """The receipt with every child, after appending in-place re-attempts under the lease.

        A run resumed with ``--retry-incomplete`` that sent a lost request again becomes an
        in-place re-attempt child here, before anything about it is shown.
        """
        record_redispatches(self.project, self.receipt, self.panel, clock=self._clock)
        return receipt_view(self.project, self.receipt)

    def reassess(self, clock: Callable[[], datetime]) -> bool:
        """Append an invalidation notice if links recorded since reservation reveal exposure.

        Only exposure that does not provably follow the receipt counts (see
        :func:`late_exposure`); exposure after it never invalidates the audit. Before any
        dispatch the next opening blocks instead.

        Args:
            clock: Aware clock.

        Returns:
            Whether a new notice was appended.
        """
        ledger = ExposureLedger(self.project.paths.root)
        snapshot = ledger.snapshot()
        blocked = late_exposure(self.project, ledger, snapshot, self.receipt)
        if not blocked or not panel_slots(self.project, self.receipt.base_runs):
            return False
        return _notice(self.project, self.receipt, blocked, snapshot.identity, clock)

    def guard(self, prepared: PreparedFidelity) -> None:
        """Refuse a slot claim for one run unless this receipt may dispatch it.

        The reservation was confirmed active inside the ledger transaction after the lease was
        taken, and only a lease holder can release it, so it stays active while the lease is
        held. Before the run's first claim this receipt is recorded as its dispatcher; a run
        another receipt already claimed a slot for is refused, so only that receipt ever
        completes it.

        Args:
            prepared: Registration of the run about to claim a slot.

        Raises:
            ReservationInactiveError: The lease is no longer held, or another receipt
                dispatched the run.
        """
        if not self._lease.is_locked:
            raise ReservationInactiveError(
                f"panel {self.receipt.panel} lease is not held; refusing to claim a slot"
            )
        run = prepared.spec.run_id
        owner = record_dispatch(self.project, self.receipt, run, clock=self._clock)
        if owner is None:
            raise ReservationInactiveError(
                f"run {run} claimed slots with no recorded dispatcher; refusing to claim a slot"
            )
        if owner != self.receipt.receipt_id:
            raise ReservationInactiveError(
                f"run {run} was dispatched under receipt {owner}; refusing to claim a slot"
            )


@contextmanager
def open_held_out_audit(
    project: ProjectStore,
    count: int,
    *,
    profile: Profile,
    code_revision: str,
    clock: Callable[[], datetime],
    include_inspected: bool = False,
    memory_profile: MemoryProfile = "learned",
    checkpoint: int | None = None,
) -> Iterator[HeldOutAudit]:
    """Reserve and register a held-out audit, or resume the request's receipt, under its lease.

    Args:
        project: Project with a completed build.
        count: Maximum held-out traces.
        profile: Simulator profile.
        code_revision: Exact running implementation.
        clock: Aware clock.
        include_inspected: Also select exposed or reserved lineages, as a development re-audit.
        memory_profile: Memory audit only: what the replays read (see ``plan_held_out_panel``).
        checkpoint: Memory audit only: the earlier frozen version, by consumed fit
            trajectories, to read instead of the final one.

    Yields:
        The open audit; the lease is held until the context exits.

    Raises:
        PanelInUseError: Another process holds the panel's lease.
        ValueError: Nothing can be selected, or selection kept changing under other audits.
    """
    ledger = ExposureLedger(project.paths.root)
    request = AuditRequest(
        profile=profile,
        count=count,
        include_inspected=include_inspected,
        memory=memory_profile,
        checkpoint=checkpoint,
    )
    refresh_ledger(project, ledger)
    for _ in range(SELECTION_ROUNDS):
        snapshot = ledger.snapshot()
        owner, exposure, excluded = _selection(project, request, snapshot)
        planned = plan_held_out_panel(
            project,
            count,
            profile=profile,
            code_revision=code_revision,
            clock=clock,
            excluded=excluded,
            memory_profile=memory_profile,
            checkpoint=checkpoint,
        )
        if owner is not None and planned.lineages != owner.lineages:
            # Selection no longer lands on the owner's cases: a new receipt, around its hold.
            owner = None
            exposure = held_out_exposure(project, snapshot)
            excluded = frozenset() if include_inspected else exposure.lineages
            planned = plan_held_out_panel(
                project,
                count,
                profile=profile,
                code_revision=code_revision,
                clock=clock,
                excluded=excluded,
                memory_profile=memory_profile,
                checkpoint=checkpoint,
            )
        follows: tuple[AuditReceipt, Literal["released-unused", "dispatched"]] | None = None
        if owner is not None:
            if planned.identity == owner.panel and code_revision == owner.code_revision:
                with ledger.lease(owner.panel) as lease:
                    blocked: tuple[LineageExposure, ...] = ()
                    with ledger.transaction() as transaction:
                        current = transaction.snapshot()
                        if not current.active(owner.receipt_id):
                            continue
                        _require_unclaimed(project, owner.base_runs, owner.receipt_id)
                        blocked = late_exposure(project, ledger, current, owner)
                        dispatched = bool(blocked) and bool(panel_slots(project, owner.base_runs))
                        if blocked and not dispatched:
                            transaction.release(LedgerRelease(receipt=owner.receipt_id, at=clock()))
                    if blocked and not dispatched:
                        raise ValueError(_blocked(owner, blocked))
                    if blocked:
                        _notice(project, owner, blocked, current.identity, clock)
                    panel = register_planned_panel(project, planned, clock=clock)
                    record_attempts(project, owner, panel, clock=clock)
                    yield HeldOutAudit(
                        project, owner, panel, exposure, lease, excluded=len(excluded), clock=clock
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
            panel = register_planned_panel(project, planned, clock=clock)
            record_attempts(project, receipt, panel, clock=clock)
            yield HeldOutAudit(
                project, receipt, panel, exposure, lease, excluded=len(excluded), clock=clock
            )
            return
    raise ValueError(
        "other audits kept reserving the selected held-out content; run the command again"
    )


def require_held_out_selection(
    project: ProjectStore,
    count: int,
    *,
    include_inspected: bool = False,
    memory_profile: MemoryProfile = "learned",
    checkpoint: int | None = None,
) -> None:
    """Refuse a memory audit that could select nothing, before it learns or calls anything.

    The memory profile learns its fit-only memory before the panel is planned, so this runs
    the same exposure decision and case selection that opening will, without the memory. It
    reserves nothing: another audit can still take the remaining lineages before opening, which
    then refuses after the learning was bought.

    Args:
        project: Project with a completed build.
        count: Maximum held-out traces.
        include_inspected: Also select exposed or reserved lineages, as a development re-audit.
        memory_profile: What the replays will read.
        checkpoint: Earlier frozen memory version the replays will read.

    Raises:
        ValueError: Unknown fingerprint schemes block an independent audit, or no held-out
            lineage is left to select.
    """
    ledger = ExposureLedger(project.paths.root)
    request = AuditRequest(
        profile="memory",
        count=count,
        include_inspected=include_inspected,
        memory=memory_profile,
        checkpoint=checkpoint,
    )
    refresh_ledger(project, ledger)
    _, _, excluded = _selection(project, request, ledger.snapshot())
    held_out_cases(project, count, fit_calls(project), excluded=excluded)


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


def late_exposure(
    project: ProjectStore,
    ledger: ExposureLedger,
    snapshot: LedgerSnapshot,
    receipt: AuditReceipt,
) -> tuple[LineageExposure, ...]:
    """Audited lineages now linked to exposure that predates an independent receipt.

    Order is causal, never read from exposure times. An exposure predates the receipt when it
    was in the ledger snapshot the receipt was minted against (``prior_exposures``). One that
    was not counts as later only when this root's ledger recorded both the receipt's
    reservation and the exposure itself, from a report run here, which the ledger's transaction
    serialized after the reservation. An exposure merged from another ledger, or any exposure
    against a reservation merged with a moved project's receipt, has no proven order against
    the receipt and still counts, so skew between hosts can never clear an audit.

    Args:
        project: Project of the receipt.
        ledger: Ledger whose schemes to use.
        snapshot: Current ledger snapshot.
        receipt: Receipt to reassess.

    Returns:
        The affected lineages with their paths; empty for a development re-audit.
    """
    if receipt.basis != "independent":
        return ()
    prior = frozenset(receipt.prior_exposures)
    here = frozenset(snapshot.records.recorded_here)
    shared = receipt.receipt_id in here

    def predates(item: LedgerExposure) -> bool:
        """Whether this exposure is not provably recorded here after the receipt."""
        return item.record_id in prior or not shared or item.record_id not in here

    exposure = held_out_exposure(
        project, snapshot, receipt.receipt_id, ledger=ledger, counts=predates
    )
    return tuple(
        item
        for item in exposure.exposed
        if item.kind == "exposed" and item.lineage in receipt.lineages
    )


def _blocked(receipt: AuditReceipt, blocked: tuple[LineageExposure, ...]) -> str:
    """Why an undispatched receipt can no longer run independently."""
    return (
        f"links recorded after receipt {receipt.receipt_id} reserved its panel reveal exposure "
        f"that predates it on {len(blocked)} audited "
        f"{'lineage' if len(blocked) == 1 else 'lineages'}, so it can no longer run as an "
        "independent audit; its unused reservation was released. Run the command again to "
        "select unexposed lineages, or add --include-inspected for a development re-audit"
    )


def _notice(
    project: ProjectStore,
    receipt: AuditReceipt,
    blocked: tuple[LineageExposure, ...],
    ledger: str,
    clock: Callable[[], datetime],
) -> bool:
    """Append an invalidation notice to a receipt whose execution already began."""
    return record_notice(
        project,
        receipt,
        AuditNotice(receipt=receipt.receipt_id, at=clock(), ledger=ledger, exposed=blocked),
    )


def _selection(
    project: ProjectStore, request: AuditRequest, snapshot: LedgerSnapshot
) -> tuple[AuditReceipt | None, HeldOutExposure, frozenset[str]]:
    """The receipt the request resumes, the exposure decision and the lineages left out.

    Raises:
        ValueError: Unknown fingerprint schemes block an independent audit.
    """
    owner = _open_receipt(project, request, snapshot)
    if owner is not None and any(
        report.detail_shown for report in receipt_view(project, owner).reports
    ):
        owner = None  # it showed per-trace detail; the request moves on to fresh content
    exposure = held_out_exposure(project, snapshot, owner.receipt_id if owner else None)
    if exposure.unknown and not request.include_inspected:
        raise ValueError(unknown_scheme_refusal(exposure.unknown))
    # The owner keeps its own cases, so later links are reassessed under its lease.
    own = frozenset(owner.lineages) if owner else frozenset()
    excluded = (
        frozenset()
        if request.include_inspected
        else frozenset(item.lineage for item in exposure.exposed if item.lineage not in own)
    )
    return owner, exposure, excluded


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
        _require_unclaimed(project, planned.base_runs, None)
        saved = _any_saved(project, planned.base_runs)
        receipt = AuditReceipt(
            sequence=len(audit_receipts(project)) + 1,
            opened_at=clock(),
            project=project.paths.project_id,
            request=request,
            basis=(
                "development-re-audit" if touched or saved or exposure.unknown else "independent"
            ),
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
            schemes=exposure.schemes,
            ledger=snapshot.identity,
            prior_exposures=tuple(sorted(item.record_id for item in snapshot.records.exposures)),
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
                vertices=tuple(
                    sorted(
                        {
                            vertex
                            for lineage in receipt.lineages
                            for vertex in members[lineage].vertices
                        }
                    )
                ),
                snapshot=snapshot.identity,
                at=receipt.opened_at,
            )
        )
        return receipt


def _require_unclaimed(
    project: ProjectStore, base_runs: tuple[str, ...], receipt: str | None
) -> None:
    """Refuse a run another receipt dispatched and has not finished.

    A run's ID never names a receipt, so a development re-audit over the same cases registers
    the very runs an interrupted receipt claimed slots for. Only the receipt recorded as the
    run's dispatcher may resume it; otherwise its lost requests would be re-dispatched, and its
    result reported, under another receipt. A run that claimed slots is never replaced, so an
    unfinished dispatched run is always the case's current registration. A run that claimed
    slots with no recorded dispatcher, dispatched before dispatchers were recorded, has no
    provable owner, so no receipt may complete it.

    Raises:
        ValueError: Another receipt dispatched an unfinished run of the panel, or nothing
            records who dispatched one.
    """
    for base in base_runs:
        current = current_registration(project, base)
        if current is None or saved_fidelity_report(project, current) is not None:
            continue
        if not run_slots(project, current.spec.run_id):
            continue
        owner = dispatch_owner(project, current.spec.run_id)
        if owner is None:
            raise ValueError(
                f"run {current.spec.run_id} claimed dispatch slots but no receipt is recorded "
                "as its dispatcher, so none can be proven to own it; audit this build in a "
                "fresh project (exp build) to measure it again"
            )
        if owner != receipt:
            raise ValueError(
                f"run {current.spec.run_id} was dispatched under receipt {owner} and has no "
                "saved result; only that receipt can complete it: repeat the request that "
                "opened it with --retry-incomplete"
            )


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

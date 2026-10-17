"""Held-out audit lifecycle: reserve before register, leases, permanence after dispatch, children.

Builds run offline through the routed fake of :mod:`held_out_fidelity_test`; audits use the
source-RAG profile so no learning pass is needed.
"""

import multiprocessing
import os
import threading
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity, held_out_audit, trace_replay
from exp.optimize.evaluation.fidelity import execute_fidelity_evaluation
from exp.optimize.evaluation.held_out_audit import (
    HeldOutAudit,
    ReservationInactiveError,
    open_held_out_audit,
    outcome_unknown,
    release_held_out_audit,
    reservation_state,
)
from exp.optimize.evaluation.held_out_exposure import held_out_members
from exp.optimize.evaluation.held_out_fidelity import HeldOutPanel, PlannedPanel, Profile
from exp.optimize.evaluation.held_out_fidelity_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.held_out_receipts import (
    QUALIFIED,
    RECEIPT_TYPE,
    audit_receipts,
    receipt_view,
    record_report,
)
from exp.optimize.evaluation.trace_replay import DEADLINE
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.exposure_ledger import (
    ExposureLedger,
    LedgerEdge,
    LedgerExposure,
    LedgerReservation,
    PanelInUseError,
)
from exp.simulation.mining.fingerprint import TRACE_CONTENT


def clock() -> datetime:
    """Aware wall-clock time."""
    return datetime.now(UTC)


def audit(
    store: ProjectStore,
    count: int = 1,
    *,
    revision: str = REVISION,
    include_inspected: bool = False,
) -> AbstractContextManager[HeldOutAudit]:
    """Open a source-RAG held-out audit."""
    return open_held_out_audit(
        store,
        count,
        profile="source-rag",
        code_revision=revision,
        clock=clock,
        include_inspected=include_inspected,
    )


def run(store: ProjectStore, opened: HeldOutAudit, *, resume: bool = False) -> None:
    """Execute every registered run under the audit's guard; ``resume`` is --retry-incomplete."""
    for prepared in opened.panel.prepared:
        execute_fidelity_evaluation(
            store,
            prepared,
            dispatch_guard=opened.guard,
            resume_until=clock() + DEADLINE if resume else None,
        )


class FailingOnce(SimulatedProvider):
    """Crash on the first simulator request after its slot was claimed, then answer normally."""

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Arm one failure."""
        super().__init__(episodes)
        self.armed = False

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Stop the process once, for a simulator turn while armed."""
        tool = payload.get("tool_choice")
        if self.armed and isinstance(tool, dict) and url.endswith("/chat/completions"):
            self.armed = False
            raise KeyboardInterrupt
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


@pytest.fixture(name="built")
def _built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProjectStore, FailingOnce]:
    """A built project whose provider can fail once after a slot is claimed."""
    provider = FailingOnce([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    return store, provider


def test_content_is_reserved_before_any_run_is_registered(
    built: tuple[ProjectStore, FailingOnce], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserve, then register: registration only ever sees an active reservation."""
    store, _ = built
    ledger = ExposureLedger(store.paths.root)
    seen: list[bool] = []
    register = held_out_audit.register_planned_panel

    def checked(
        project: ProjectStore,
        planned: PlannedPanel,
        *,
        code_revision: str,
        clock: Callable[[], datetime],
    ) -> HeldOutPanel:
        """Record whether the newest receipt's reservation was committed before registering."""
        receipts = audit_receipts(store)
        seen.append(bool(receipts) and ledger.snapshot().active(receipts[-1].receipt_id))
        return register(project, planned, code_revision=code_revision, clock=clock)

    monkeypatch.setattr(held_out_audit, "register_planned_panel", checked)
    with audit(store) as opened:
        receipt = opened.receipt
        reservation = ledger.snapshot().reservations[receipt.receipt_id]
        assert reservation.panel == receipt.panel and reservation.snapshot == receipt.ledger
        assert set(reservation.lineages) == set(receipt.lineages)
        assert receipt.basis == "independent" and reservation_state(store, receipt) == "active"
    with audit(store) as again:
        assert again.receipt == receipt, "the same request resumes its receipt"
    assert seen == [True, True]
    assert len(store.artifacts.list_ids(artifact_type=RECEIPT_TYPE)) == 1


def test_concurrent_audits_never_share_content(built: tuple[ProjectStore, FailingOnce]) -> None:
    """A second process sees the first's reservation: same panel is in use, others pick around."""
    store, _ = built
    holding, done = threading.Event(), threading.Event()
    first: list[HeldOutAudit] = []

    def hold() -> None:
        """Keep one audit open until the main thread is finished."""
        with audit(store) as opened:
            first.append(opened)
            holding.set()
            done.wait(30)

    worker = threading.Thread(target=hold)
    worker.start()
    assert holding.wait(30)
    try:
        with pytest.raises(PanelInUseError, match="in use"), audit(store):
            pass
        with audit(store, 2) as second:
            assert not set(second.receipt.lineages) & set(first[0].receipt.lineages)
            assert second.receipt.basis == "independent" and second.excluded == 1
    finally:
        done.set()
        worker.join()


def test_a_reservation_committed_between_selection_and_commit_is_respected(
    built: tuple[ProjectStore, FailingOnce], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph is re-read inside the transaction; content taken meanwhile is never reserved."""
    store, _ = built
    ledger = ExposureLedger(store.paths.root)
    plan = held_out_audit.plan_held_out_panel
    stolen: list[str] = []

    def racing(
        project: ProjectStore,
        count: int,
        *,
        profile: Profile,
        clock: Callable[[], datetime],
        excluded: Collection[str] = frozenset(),
    ) -> PlannedPanel:
        """Plan, then let another receipt reserve the planned lineage before the commit."""
        planned = plan(project, count, profile=profile, clock=clock, excluded=excluded)
        if not stolen:
            lineage = planned.lineages[0]
            stolen.append(lineage)
            with ledger.transaction() as transaction:
                transaction.reserve(
                    LedgerReservation(
                        receipt="held-out-audit-receipt-elsewhere",
                        panel="held-out-panel-elsewhere",
                        project="elsewhere",
                        lineages=(lineage,),
                        vertices=held_out_members(store)[lineage].vertices,
                        snapshot=transaction.snapshot().identity,
                        at=clock(),
                    )
                )
        return planned

    monkeypatch.setattr(held_out_audit, "plan_held_out_panel", racing)
    with audit(store) as opened:
        assert opened.receipt.lineages != tuple(stolen)
        assert opened.receipt.basis == "independent"
        snapshot = ledger.snapshot()
        assert opened.receipt.ledger != snapshot.identity or len(snapshot.reservations) == 2


def test_a_claimed_slot_makes_the_reservation_permanent_after_a_crash(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """Dispatched, outcome unknown: only the same receipt completes it, as a qualified child."""
    store, provider = built
    with audit(store) as opened:
        receipt = opened.receipt
        provider.armed = True
        with pytest.raises(KeyboardInterrupt):
            run(store, opened)
        assert outcome_unknown(store, opened)
    assert reservation_state(store, receipt) == "dispatched"
    with pytest.raises(ValueError, match="permanent"):
        release_held_out_audit(store, receipt, clock=clock)

    # Without --retry-incomplete the owner resumes but never buys the stopped run again.
    with audit(store) as resumed:
        assert resumed.receipt == receipt and not resumed.panel.retirements
        with pytest.raises(ValueError, match="--retry-incomplete"):
            run(store, resumed)
    # Any other evaluation of those cases is a development re-audit with its own receipt.
    with audit(store, include_inspected=True) as other:
        assert other.receipt != receipt and other.receipt.basis == "development-re-audit"
        assert other.receipt.reaudited == receipt.lineages
    # The owner completes it with --retry-incomplete: the same run resumes in place, and the
    # lost request sent again is an in-place re-attempt child, never a new receipt.
    with audit(store) as completed:
        assert completed.receipt == receipt and not completed.panel.retirements
        run(store, completed, resume=True)
        view = completed.view
        [child] = view.attempts
        assert child.reason == "re-attempt" and child.claimed_slots >= 1 and child.in_place
        assert child.original_run == receipt.cases[0].base_run
        assert child.retired_run == child.replacement_run == receipt.cases[0].base_run
        assert [(row.stage, row.step, row.attempts) for row in child.redispatched] == [
            ("prediction", 0, 2)
        ]
        assert view.redispatched["prediction"] == 1
        assert view.headline == QUALIFIED and view.affected_traces == {receipt.cases[0].trace_id}
    # A plain rerun replays the completed run under the same qualified receipt.
    with audit(store) as replayed:
        assert replayed.panel.prepared[0].spec.run_id == child.replacement_run
        assert not replayed.panel.retirements
        run(store, replayed)
        assert replayed.view.headline == QUALIFIED and len(replayed.view.attempts) == 1
    assert len(audit_receipts(store)) == 2


def test_a_reregistration_before_dispatch_does_not_qualify_the_headline(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A registration retired with zero claimed slots is recorded, but is not a re-attempt."""
    store, _ = built
    with audit(store) as opened:
        receipt = opened.receipt
    catalog = load_model_catalog(store.model_catalog_path)
    extra = catalog.models["world"].model_copy(update={"model": "world-v2"})
    write_model_catalog(
        store.model_catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, "spare": extra}}),
    )
    with audit(store) as again:
        assert again.receipt == receipt
        [child] = again.view.attempts
        assert child.reason == "re-registration" and child.claimed_slots == 0
        assert again.view.headline == "independent"


def test_a_configuration_change_with_zero_slots_releases_and_mints_a_new_receipt(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """The unused reservation is released; the original receipt is never rewritten."""
    store, _ = built
    with audit(store) as opened:
        original = opened.receipt
    stored = store.artifacts.read_bytes(original.receipt_id, "receipt.json")
    with audit(store, revision="d" * 40) as changed:
        assert changed.receipt.receipt_id != original.receipt_id
        assert changed.receipt.code_revision == "d" * 40
        assert (changed.receipt.follows, changed.receipt.follow_reason) == (
            original.receipt_id,
            "released-unused",
        )
        assert changed.receipt.basis == "independent"
        assert changed.receipt.lineages == original.lineages
    assert reservation_state(store, original) == "released"
    assert store.artifacts.read_bytes(original.receipt_id, "receipt.json") == stored


def test_a_configuration_change_after_dispatch_is_a_development_reaudit(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A code-revision mismatch is detected; the dispatched reservation stays permanent."""
    store, _ = built
    with audit(store) as opened:
        original = opened.receipt
        run(store, opened)
    with audit(store, revision="d" * 40) as changed:
        assert changed.receipt.follow_reason == "dispatched"
        assert changed.receipt.basis == "development-re-audit"
        assert changed.receipt.reaudited == original.lineages
    assert reservation_state(store, original) == "dispatched"
    assert receipt_view(store, original).headline == "independent"


def test_release_and_resume_never_leave_a_released_reservation_with_claims(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """While a resume holds the lease, release fails; once a slot is claimed, it never succeeds."""
    store, _ = built
    with audit(store) as opened:
        receipt = opened.receipt
        with pytest.raises(PanelInUseError, match="in use"):
            release_held_out_audit(store, receipt, clock=clock)
        run(store, opened)
    with pytest.raises(ValueError, match="permanent"):
        release_held_out_audit(store, receipt, clock=clock)
    assert reservation_state(store, receipt) == "dispatched"
    with pytest.raises(ReservationInactiveError):
        opened.guard()

    # The converse order: an unused reservation is released first, so it cannot be resumed.
    with audit(store, 2) as fresh:
        unused = fresh.receipt
    release_held_out_audit(store, unused, clock=clock)
    with audit(store, 2) as after:
        assert after.receipt.receipt_id != unused.receipt_id
    assert reservation_state(store, unused) == "released"


RANKS = {"lease": 0, "ledger": 1, "run": 2, "dispatch": 3}


def test_locks_are_always_taken_in_one_order(
    built: tuple[ProjectStore, FailingOnce], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Panel lease, then ledger transaction, then run lock, then dispatch claim; never backwards."""
    store, _ = built
    events: list[tuple[str, tuple[str, ...]]] = []
    held = threading.local()

    @contextmanager
    def tracked(name: str) -> Iterator[None]:
        """Record acquisition of ``name`` with the locks this thread already holds."""
        stack: list[str] = getattr(held, "stack", [])
        held.stack = stack
        events.append((name, tuple(stack)))
        stack.append(name)
        try:
            yield
        finally:
            stack.pop()

    lease, transaction = ExposureLedger.lease, ExposureLedger.transaction

    @contextmanager
    def tracked_lease(self: ExposureLedger, panel: str) -> Iterator[object]:
        """Track the panel lease."""
        with tracked("lease"), lease(self, panel) as lock:
            yield lock

    @contextmanager
    def tracked_transaction(self: ExposureLedger) -> Iterator[object]:
        """Track the ledger transaction."""
        with tracked("ledger"), transaction(self) as handle:
            yield handle

    class TrackedFileLock:
        """Track a run's execution lock."""

        def __init__(self, path: str | os.PathLike[str], timeout: float = -1) -> None:
            """Wrap the real lock."""
            self._lock = fidelity_lock(path, timeout=timeout)
            self._context: AbstractContextManager[None] | None = None

        def __enter__(self) -> "TrackedFileLock":
            """Acquire and record."""
            self._context = tracked("run")
            self._context.__enter__()
            self._lock.acquire()
            return self

        def __exit__(self, *exc: object) -> None:
            """Release."""
            self._lock.release()
            assert self._context is not None
            self._context.__exit__(None, None, None)

        def acquire(self, timeout: float = -1) -> None:
            """Acquire without tracking a scope (retirement probe)."""
            events.append(("run", tuple(getattr(held, "stack", []))))
            self._lock.acquire(timeout=timeout)

        def release(self) -> None:
            """Release a probe."""
            self._lock.release()

    claim = DispatchLedger.claim

    def tracked_claim(self: DispatchLedger, claimed_at: str) -> tuple[int, str]:
        """Track a dispatch slot claim."""
        with tracked("dispatch"):
            return claim(self, claimed_at)

    fidelity_lock = fidelity.FileLock
    monkeypatch.setattr(ExposureLedger, "lease", tracked_lease)
    monkeypatch.setattr(ExposureLedger, "transaction", tracked_transaction)
    monkeypatch.setattr(fidelity, "FileLock", TrackedFileLock)
    monkeypatch.setattr(trace_replay, "FileLock", TrackedFileLock)
    monkeypatch.setattr(DispatchLedger, "claim", tracked_claim)

    with audit(store) as opened:
        run(store, opened)
    with audit(store) as resumed:
        run(store, resumed, resume=True)
    with audit(store, 2) as other:
        pass
    release_held_out_audit(store, other.receipt, clock=clock)

    kinds = [name for name, _ in events]
    assert {"lease", "ledger", "run", "dispatch"} <= set(kinds)
    for name, before in events:
        assert all(RANKS[earlier] < RANKS[name] for earlier in before), (name, before)
    first = {name: kinds.index(name) for name in RANKS}
    assert first["lease"] < first["run"] < first["dispatch"]


OUTSIDE = f"{TRACE_CONTENT.identity.key}:{'e' * 64}"


def expose_outside(store: ProjectStore) -> None:
    """Expose content no held-out lineage is linked to yet."""
    ExposureLedger(store.paths.root).record_exposure(
        LedgerExposure(
            vertices=(OUTSIDE,),
            how="inspected",
            panel="held-out-panel-elsewhere",
            receipt="held-out-audit-receipt-elsewhere",
            report="exp eval --audit --verbose",
            project="elsewhere",
            at=clock(),
        )
    )


def link_late(store: ProjectStore, lineage: str) -> None:
    """A later build records a grouping link from the lineage to the exposed content."""
    member = held_out_members(store)[lineage].vertices[0]
    with ExposureLedger(store.paths.root).transaction() as transaction:
        transaction.record(
            "edge",
            LedgerEdge(
                left=min(member, OUTSIDE),
                right=max(member, OUTSIDE),
                reason="declared-task-identity",
                task_lineage="late",
                project="later",
                build="task-set-later",
            ),
        )


def test_late_links_to_predating_exposure_block_before_dispatch(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A6: exposure older than the reservation, linked later, blocks an undispatched panel."""
    store, _ = built
    expose_outside(store)
    with audit(store) as opened:
        receipt = opened.receipt
        assert receipt.basis == "independent"
    link_late(store, receipt.lineages[0])
    with pytest.raises(ValueError, match="can no longer run as an independent audit"), audit(store):
        pass
    assert reservation_state(store, receipt) == "released"
    with audit(store) as moved:
        assert moved.receipt.lineages != receipt.lineages


def test_late_links_after_dispatch_append_a_notice_and_later_exposure_never_does(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A6: after dispatch a notice is appended; exposure after the audit invalidates nothing."""
    store, _ = built
    expose_outside(store)
    with audit(store) as opened:
        run(store, opened)
        assert not opened.reassess(clock)
        # Exposure after the receipt was minted never invalidates it retroactively.
        record_report(
            store, opened.receipt, opened.panel, kind="evaluation", detail_shown=True, clock=clock
        )
        assert not opened.reassess(clock) and opened.view.notices == ()
    stored = store.artifacts.read_bytes(opened.receipt.receipt_id, "receipt.json")
    with audit(store, 2) as second:
        run(store, second)
        link_late(store, second.receipt.lineages[0])
        assert second.reassess(clock)
        assert not second.reassess(clock), "one notice per discovered exposure"
        [notice] = second.view.notices
        assert [item.lineage for item in notice.exposed] == [second.receipt.lineages[0]]
        assert notice.exposed[0].path is not None
        assert notice.exposed[0].path.exposed_node == OUTSIDE
    with audit(store, 2) as resumed:
        assert resumed.receipt == second.receipt and len(resumed.view.notices) == 1
    assert store.artifacts.read_bytes(opened.receipt.receipt_id, "receipt.json") == stored


def _resume_and_hold(root: str, holding: Event, done: Event) -> None:
    """Child process: resume the audit and hold its lease, as an execution would."""
    store = ProjectStore(Path(root), PROJECT)
    with audit(store):
        holding.set()
        done.wait(120)


def test_a_second_process_resuming_blocks_release_with_panel_in_use(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A8: while another process holds the panel, release fails at once and changes nothing."""
    store, _ = built
    with audit(store) as opened:
        receipt = opened.receipt
    context = multiprocessing.get_context("spawn")
    holding, done = context.Event(), context.Event()
    holder = context.Process(target=_resume_and_hold, args=(str(store.paths.root), holding, done))
    holder.start()
    try:
        assert holding.wait(120)
        with pytest.raises(PanelInUseError, match="in use"):
            release_held_out_audit(store, receipt, clock=clock)
        assert reservation_state(store, receipt) == "active"
    finally:
        done.set()
        holder.join(120)
    assert holder.exitcode == 0
    release_held_out_audit(store, receipt, clock=clock)
    assert reservation_state(store, receipt) == "released"

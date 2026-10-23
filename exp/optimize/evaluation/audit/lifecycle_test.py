"""Held-out audit lifecycle: reserve before register, leases, permanence after dispatch, children.

Builds run offline through the routed fake of :mod:`held_out_fidelity_test`; audits use the
source-RAG profile so no learning pass is needed.
"""

import multiprocessing
import os
import shutil
import threading
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from exp.cli.evaluation.held_out_fidelity import title
from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit import lifecycle as held_out_audit
from exp.optimize.evaluation.audit.exposure import held_out_members
from exp.optimize.evaluation.audit.lifecycle import (
    HeldOutAudit,
    ReservationInactiveError,
    late_exposure,
    open_held_out_audit,
    outcome_unknown,
    release_held_out_audit,
    reservation_state,
)
from exp.optimize.evaluation.audit.panel import HeldOutPanel, MemoryProfile, PlannedPanel, Profile
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.audit.receipts import (
    QUALIFIED,
    RECEIPT_TYPE,
    audit_receipts,
    dispatch_owner,
    receipt_view,
    record_report,
)
from exp.optimize.evaluation.fidelity import replay as trace_replay
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.fidelity.replay import DEADLINE
from exp.optimize.evaluation.fidelity.runner import (
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
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
            dispatch_guard=partial(opened.guard, prepared),
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
        clock: Callable[[], datetime],
    ) -> HeldOutPanel:
        """Record whether the newest receipt's reservation was committed before registering."""
        receipts = audit_receipts(store)
        seen.append(bool(receipts) and ledger.snapshot().active(receipts[-1].receipt_id))
        return register(project, planned, clock=clock)

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
        code_revision: str,
        clock: Callable[[], datetime],
        excluded: Collection[str] = frozenset(),
        memory_profile: MemoryProfile = "learned",
        checkpoint: int | None = None,
    ) -> PlannedPanel:
        """Plan, then let another receipt reserve the planned lineage before the commit."""
        planned = plan(
            project,
            count,
            profile=profile,
            code_revision=code_revision,
            clock=clock,
            excluded=excluded,
            memory_profile=memory_profile,
            checkpoint=checkpoint,
        )
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
    # No other receipt may finish the owner's run: a development re-audit over those cases
    # refuses before it mints a receipt or sends anything.
    with (
        pytest.raises(ValueError, match=f"under receipt {receipt.receipt_id}.*only that receipt"),
        audit(store, include_inspected=True),
    ):
        pass
    assert len(audit_receipts(store)) == 1
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
    # Once it finished, a development re-audit replays it with its own re-attempt child; the
    # owner's child is never shared with, or shadowed by, another receipt's.
    with audit(store, include_inspected=True) as other:
        assert other.receipt != receipt and other.receipt.basis == "development-re-audit"
        assert other.receipt.reaudited == receipt.lineages
        [reaudited] = other.view.attempts
        assert reaudited.receipt == other.receipt.receipt_id and reaudited.in_place
    assert len(receipt_view(store, receipt).attempts) == 1
    assert len(audit_receipts(store)) == 2


def test_the_receipt_that_dispatched_a_shared_run_alone_completes_it(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """Dispatch ownership is recorded, never inferred from which receipt reserved first."""
    store, provider = built
    with audit(store) as first:
        receipt = first.receipt
        shared = first.panel.prepared[0]
        # A later development re-audit over the same case dispatches the shared run first.
        with audit(store, 2, include_inspected=True) as other:
            assert shared in other.panel.prepared
            provider.armed = True
            with pytest.raises(KeyboardInterrupt):
                run(store, other)
        dispatcher = other.receipt
        assert dispatch_owner(store, shared.spec.run_id) == dispatcher.receipt_id
        with pytest.raises(
            ReservationInactiveError, match=f"under receipt {dispatcher.receipt_id}"
        ):
            run(store, first, resume=True)
    with (
        pytest.raises(ValueError, match=f"under receipt {dispatcher.receipt_id}.*only that"),
        audit(store),
    ):
        pass
    with audit(store, 2, include_inspected=True) as resumed:
        assert resumed.receipt == dispatcher
        run(store, resumed, resume=True)
        assert [item.receipt for item in resumed.view.reattempts] == [dispatcher.receipt_id]
    with audit(store) as replayed:
        assert replayed.receipt == receipt and saved_fidelity_report(store, shared) is not None


def test_a_claimed_run_without_a_recorded_dispatcher_fails_closed(
    built: tuple[ProjectStore, FailingOnce], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run dispatched before dispatchers were recorded is completed by no receipt."""
    store, provider = built
    # Dispatch the way code did before it recorded the dispatching receipt.
    monkeypatch.setattr(
        held_out_audit,
        "record_dispatch",
        lambda project, receipt, run, *, clock: receipt.receipt_id,
    )
    with audit(store) as opened:
        shared = opened.panel.prepared[0]
        provider.armed = True
        with pytest.raises(KeyboardInterrupt):
            run(store, opened)
    monkeypatch.undo()
    assert dispatch_owner(store, shared.spec.run_id) is None
    for include_inspected in (False, True):
        with (
            pytest.raises(ValueError, match="no receipt is recorded as its dispatcher"),
            audit(store, include_inspected=include_inspected),
        ):
            pass


def test_a_receipt_stopped_before_its_first_claim_owns_nothing(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A dispatch record with no slot claimed behind it never blocks another receipt."""
    store, _ = built
    with audit(store) as first:
        shared = first.panel.prepared[0]

        def stop_after_guard() -> None:
            """Pass the guard, then stop before the slot is claimed."""
            first.guard(shared)
            raise ReservationInactiveError("stopped before the claim")

        with pytest.raises(ReservationInactiveError, match="before the claim"):
            execute_fidelity_evaluation(store, shared, dispatch_guard=stop_after_guard)
    assert dispatch_owner(store, shared.spec.run_id) is None
    release_held_out_audit(store, first.receipt, clock=clock)
    with audit(store, include_inspected=True) as other:
        [current] = other.panel.prepared
        # The re-audit draws the case's next generation; the unclaimed run neither blocks it
        # nor gains an owner.
        assert current.spec.run_id == shared.spec.run_id + "-r2"
        run(store, other, resume=True)
        assert dispatch_owner(store, current.spec.run_id) == other.receipt.receipt_id
    assert dispatch_owner(store, shared.spec.run_id) is None
    assert saved_fidelity_report(store, current) is not None


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
    """A code-revision mismatch is detected; the dispatched reservation stays permanent.

    The new receipt measures again: saved results of other code never stand in for its runs.
    """
    store, _ = built
    with audit(store) as opened:
        original = opened.receipt
        run(store, opened)
        measured = {prepared.spec.run_id for prepared in opened.panel.prepared}
    with audit(store, revision="d" * 40) as changed:
        assert changed.receipt.follow_reason == "dispatched"
        assert changed.receipt.basis == "development-re-audit"
        assert changed.receipt.reaudited == original.lineages
        assert changed.receipt.panel != original.panel
        assert measured.isdisjoint(prepared.spec.run_id for prepared in changed.panel.prepared)
        assert all(prepared.code_revision == "d" * 40 for prepared in changed.panel.prepared)
        assert all(saved_fidelity_report(store, p) is None for p in changed.panel.prepared)
    assert reservation_state(store, original) == "dispatched"
    assert receipt_view(store, original).headline == "independent"


def test_a_refused_dispatch_stops_the_trace_without_a_score(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A slot claim refused by the audit's guard stops the run; no turn is scored as a miss."""
    store, provider = built
    with audit(store) as opened:
        prepared = opened.panel.prepared[0]
    sent = len(provider.requests)
    with pytest.raises(ReservationInactiveError):
        execute_fidelity_evaluation(store, prepared, dispatch_guard=partial(opened.guard, prepared))
    assert len(provider.requests) == sent
    assert saved_fidelity_report(store, prepared) is None


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
        opened.guard(opened.panel.prepared[0])

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

    fidelity_lock = fidelity.CoordinationLock
    monkeypatch.setattr(ExposureLedger, "lease", tracked_lease)
    monkeypatch.setattr(ExposureLedger, "transaction", tracked_transaction)
    monkeypatch.setattr(fidelity, "CoordinationLock", TrackedFileLock)
    monkeypatch.setattr(trace_replay, "CoordinationLock", TrackedFileLock)
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


def expose_outside(
    store: ProjectStore,
    *,
    at: datetime | None = None,
    receipt: str = "held-out-audit-receipt-elsewhere",
) -> None:
    """Expose content no held-out lineage is linked to yet, merged from another root's ledger."""
    elsewhere = ExposureLedger(store.paths.root.parent / "elsewhere")
    elsewhere.record_exposure(
        LedgerExposure(
            vertices=(OUTSIDE,),
            how="inspected",
            panel="held-out-panel-elsewhere",
            receipt=receipt,
            report="exp eval --audit --verbose",
            project=PROJECT,
            at=at or clock(),
        )
    )
    ExposureLedger(store.paths.root).merge(elsewhere.export())


def expose_here(store: ProjectStore) -> None:
    """Expose content no held-out lineage is linked to yet, recorded by this root's ledger."""
    ExposureLedger(store.paths.root).record_exposure(
        LedgerExposure(
            vertices=(OUTSIDE,),
            how="inspected",
            panel="held-out-panel-here",
            receipt="held-out-audit-receipt-here",
            report="exp eval --audit --verbose",
            project=PROJECT,
            at=clock(),
        )
    )


def link_late(store: ProjectStore, lineage: str, target: str = OUTSIDE) -> None:
    """A later build records a grouping link from the lineage to the exposed content."""
    member = held_out_members(store)[lineage].vertices[0]
    with ExposureLedger(store.paths.root).transaction() as transaction:
        transaction.record(
            "edge",
            LedgerEdge(
                left=min(member, target),
                right=max(member, target),
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
        assert opened.view.headline == "independent"
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
        invalidated = "independent, invalidated: predating exposure"
        assert second.view.headline == invalidated
        assert title(second, second.view) == f"Held-out fidelity ({invalidated})"
    with audit(store, 2) as resumed:
        assert resumed.receipt == second.receipt and len(resumed.view.notices) == 1
    assert store.artifacts.read_bytes(opened.receipt.receipt_id, "receipt.json") == stored


def skewed(offset: timedelta) -> Callable[[], datetime]:
    """An aware clock running ``offset`` away from this host's clock."""
    return lambda: datetime.now(UTC) + offset


def test_exposure_already_recorded_counts_even_when_its_clock_claims_later(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """Exposure in the snapshot a receipt was minted against predates it, whatever its time."""
    store, _ = built
    with audit(store) as first:
        run(store, first)
        record_report(
            store,
            first.receipt,
            first.panel,
            kind="evaluation",
            detail_shown=True,
            clock=skewed(timedelta(days=10)),
        )
    exposed = held_out_members(store)[first.receipt.lineages[0]].vertices[0]
    with audit(store, 2) as second:
        assert second.receipt.basis == "independent"
        (shown,) = ExposureLedger(store.paths.root).snapshot().records.exposures
        assert shown.at > second.receipt.opened_at, "the exposing host's clock runs ahead"
        assert second.receipt.prior_exposures == (shown.record_id,)
        run(store, second)
        link_late(store, second.receipt.lineages[0], exposed)
        assert second.reassess(clock)
        [notice] = second.view.notices
        assert notice.exposed[0].path is not None and notice.exposed[0].path.by == (
            shown.record_id,
        )


def test_exposure_recorded_here_after_minting_never_counts_even_when_its_clock_claims_earlier(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A report of this root recorded after the receipt follows it, whatever its time says."""
    store, _ = built
    with audit(store) as opened:
        run(store, opened)
        record_report(
            store,
            opened.receipt,
            opened.panel,
            kind="evaluation",
            detail_shown=True,
            clock=skewed(timedelta(days=-10)),
        )
        (shown,) = ExposureLedger(store.paths.root).snapshot().records.exposures
        assert shown.at < opened.receipt.opened_at, "the reporting host's clock runs behind"
        assert not opened.reassess(clock) and opened.view.notices == ()


def test_merged_exposure_without_causal_order_counts_even_when_its_clock_claims_later(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """Exposure merged from another ledger cannot be shown to follow the receipt, so it counts."""
    store, _ = built
    with audit(store) as opened:
        run(store, opened)
        expose_outside(store, at=opened.receipt.opened_at + timedelta(days=1))
        link_late(store, opened.receipt.lineages[0])
        assert opened.reassess(clock)
        [notice] = opened.view.notices
        assert notice.exposed[0].path is not None
        assert notice.exposed[0].path.exposed_node == OUTSIDE


def test_merged_exposure_counts_even_when_its_receipt_was_copied_here(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A receipt saved here, as in a copied root, never proves this root recorded its exposure."""
    store, _ = built
    with audit(store) as opened:
        run(store, opened)
        expose_outside(store, receipt=opened.receipt.receipt_id)
        link_late(store, opened.receipt.lineages[0])
        assert opened.reassess(clock)
        [notice] = opened.view.notices
        assert notice.exposed[0].path is not None
        assert notice.exposed[0].path.exposed_node == OUTSIDE


def test_local_exposure_never_follows_a_receipt_reserved_in_another_ledger(
    built: tuple[ProjectStore, FailingOnce],
) -> None:
    """A moved project's receipt and a merged reservation give local exposure no order.

    The exposure here was recorded before the receipt was minted elsewhere, yet it is missing
    from that receipt's snapshot, so only shared history could order the two.
    """
    store, _ = built
    root = store.paths.root
    moved = root.parent / "moved" / root.name
    shutil.copytree(root, moved)
    expose_here(store)
    elsewhere = ProjectStore(moved, PROJECT)
    with audit(elsewhere) as opened:
        run(elsewhere, opened)
        receipt = opened.receipt
    shutil.rmtree(root / "projects")
    shutil.copytree(moved / "projects", root / "projects")
    ExposureLedger(root).merge(ExposureLedger(moved).export())
    link_late(store, receipt.lineages[0])
    ledger = ExposureLedger(root)
    [blocked] = late_exposure(store, ledger, ledger.snapshot(), receipt)
    assert blocked.path is not None and blocked.path.exposed_node == OUTSIDE


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

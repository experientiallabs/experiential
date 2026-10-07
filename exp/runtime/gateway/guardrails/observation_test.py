"""Bounded subject ownership and shutdown tests for observation scheduling."""

from __future__ import annotations

import asyncio
import gc
import inspect
import threading
import time
import weakref
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Never

import pytest

from exp.runtime.gateway.guardrails import bounded, observation
from exp.runtime.gateway.guardrails.bounded import (
    BoundedInspect,
    ClassifierTimeoutError,
    _NativeCallbackRunner,
)
from exp.runtime.gateway.guardrails.observation import (
    ObservationAdmission,
    ObservationLease,
    ObservationOwner,
)


@pytest.fixture(autouse=True)
def _ready_scheduler() -> None:
    """Warm the shared scheduler for tests requiring accepted work; cold tests replace it."""
    bounded._NATIVE_RUNNER.loop()


def _stop_runner(runner: _NativeCallbackRunner) -> None:
    """Stop and close only a test-owned runner after its admitted jobs have drained."""
    loop = runner._loop
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if runner._thread is not None:
        runner._thread.join(timeout=2)
        assert not runner._thread.is_alive()
    if loop is not None:
        loop.close()


def test_first_observer_does_not_wait_for_callback_loop_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional admission releases a cold subject while the one shared loop is still starting."""
    started = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    inspected = threading.Event()
    admission: list[ObservationAdmission] = []
    coroutines: list[Coroutine[object, object, None]] = []

    class DelayedRunner(_NativeCallbackRunner):
        """Hold actual loop initialization without blocking daemon thread creation."""

        def _run_forever(self) -> None:
            """Expose a startup that can remain stalled while serving must proceed."""
            started.set()
            assert release.wait(3)
            super()._run_forever()

    runner = DelayedRunner()
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    original = observation.try_start_on_native_loop
    monkeypatch.setattr(bounded, "_NATIVE_RUNNER", runner)

    def tracked_start(coroutine: Coroutine[object, object, None]) -> Future[None] | None:
        """Keep a reference so rejected coroutine closure can be verified directly."""
        coroutines.append(coroutine)
        return original(coroutine)

    monkeypatch.setattr(observation, "try_start_on_native_loop", tracked_start)

    async def inspect_subject(_lease: ObservationLease) -> None:
        """Expose when a later ready-loop submission begins ordinary inspection."""
        inspected.set()

    def submit_first() -> None:
        """Use the actual owner entry point on a serving thread."""
        admission.append(
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
        )
        returned.set()

    caller = threading.Thread(target=submit_first)
    caller.start()
    try:
        assert started.wait(2)
        assert returned.wait(0.25), "optional observation waited for callback-loop readiness"
        assert admission == [ObservationAdmission.UNAVAILABLE]
        for _ in range(4):
            assert (
                owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
                is ObservationAdmission.UNAVAILABLE
            )
        assert runner._start_count == 1
        assert not inspected.is_set()
        assert not owner._jobs
        assert owner._bytes == 0
        assert owner._idle.is_set()
        assert all(inspect.getcoroutinestate(item) == inspect.CORO_CLOSED for item in coroutines)
        release.set()
        runner.loop()
        assert (
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert inspected.wait(2)
    finally:
        release.set()
        caller.join(timeout=2)
        owner.close(timeout_seconds=2)
        _stop_runner(runner)
    assert not caller.is_alive()


def test_observer_does_not_wait_for_another_callers_startup_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contended shared-runner lock cannot stall an optional request or retain its bytes."""
    runner = _NativeCallbackRunner()
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    monkeypatch.setattr(bounded, "_NATIVE_RUNNER", runner)
    returned = threading.Event()
    admission: list[ObservationAdmission] = []

    async def inspect_subject(_lease: ObservationLease) -> None:
        """Require readiness before any observation can start."""
        pytest.fail("a locked cold scheduler must not accept work")

    def submit() -> None:
        """Exercise optional submission from a request thread contending with startup."""
        admission.append(
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
        )
        returned.set()

    caller = threading.Thread(target=submit)
    try:
        with runner._lock:
            caller.start()
            assert returned.wait(0.25), "optional observation waited for the startup lock"
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert admission == [ObservationAdmission.UNAVAILABLE]
        assert runner._start_count == 0
        assert not owner._jobs
        assert owner._bytes == 0
        assert owner._idle.is_set()
    finally:
        caller.join(timeout=2)
        owner.close(timeout_seconds=2)
        _stop_runner(runner)


def test_observer_retries_after_a_dead_callback_loop_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed daemon cannot strand observations or admit work before the replacement is ready."""

    class FailOnceRunner(_NativeCallbackRunner):
        """Exit the first subject-free startup before an event loop exists."""

        def _run_forever(self) -> None:
            """Let the next admission retry the one shared scheduler after a dead startup."""
            if self._start_count == 1:
                return
            super()._run_forever()

    runner = FailOnceRunner()
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    monkeypatch.setattr(bounded, "_NATIVE_RUNNER", runner)
    inspected = threading.Event()

    async def inspect_subject(_lease: ObservationLease) -> None:
        """Mark only actual ready-loop observation execution."""
        inspected.set()

    try:
        before = time.monotonic()
        assert (
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.UNAVAILABLE
        )
        assert time.monotonic() - before < 0.25
        assert runner._thread is not None
        runner._thread.join(timeout=2)
        assert not runner._thread.is_alive()
        assert (
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.UNAVAILABLE
        )
        assert runner._start_count == 2
        assert not owner._jobs
        assert owner._bytes == 0
        assert not inspected.is_set()
        runner.loop()
        assert (
            owner.submit(inspect_subject, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert inspected.wait(2)
    finally:
        owner.close(timeout_seconds=2)
        _stop_runner(runner)


def test_scheduler_failure_closes_coroutine_and_releases_its_subject_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop startup failure reports unavailability and leaves no retained job or coroutine."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    original = observation.try_start_on_native_loop
    attempted: list[Coroutine[object, object, None]] = []
    invoked = threading.Event()

    async def run(_lease: ObservationLease) -> None:
        """Expose whether the scheduler successfully took ownership of admitted work."""
        invoked.set()

    def unavailable(coroutine: Coroutine[object, object, None]) -> Never:
        """Fail before transferring coroutine ownership to a running event loop."""
        attempted.append(coroutine)
        raise RuntimeError("synthetic callback loop failure")

    monkeypatch.setattr(observation, "try_start_on_native_loop", unavailable)
    try:
        admission = owner.submit(run, subject_bytes=4, on_interrupted=lambda _: None)
        assert admission is ObservationAdmission.UNAVAILABLE
        assert len(attempted) == 1
        assert inspect.getcoroutinestate(attempted[0]) == inspect.CORO_CLOSED
        assert not invoked.is_set()
        assert not owner._jobs
        assert owner._bytes == 0
        assert owner._idle.is_set()
        monkeypatch.setattr(observation, "try_start_on_native_loop", original)
        assert (
            owner.submit(run, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert invoked.wait(2)
    finally:
        owner.close(timeout_seconds=2)
    assert not owner._jobs
    assert owner._bytes == 0


def test_capacity_is_reserved_before_constructing_an_observation() -> None:
    """A full byte or job budget cannot retain another request coroutine."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    started = threading.Event()
    release = threading.Event()
    built: list[bool] = []

    async def inspect() -> None:
        """Retain the first admitted subject until the test releases it."""
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)

    def construct(_lease: ObservationLease) -> Coroutine[object, object, None]:
        """Expose whether rejected work was constructed."""
        built.append(True)
        return inspect()

    try:
        assert (
            owner.submit(construct, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert started.wait(2)
        assert (
            owner.submit(construct, subject_bytes=1, on_interrupted=lambda _: None)
            is ObservationAdmission.SKIPPED
        )
        assert len(built) == 1
    finally:
        release.set()
        owner.close(timeout_seconds=2)
    assert (
        owner.submit(construct, subject_bytes=1, on_interrupted=lambda _: None)
        is ObservationAdmission.SKIPPED
    )
    assert len(built) == 1


def test_shutdown_cancels_a_stalled_observer_within_the_budget() -> None:
    """Shutdown stops admission and reports cancellation instead of waiting for a verdict."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    started = threading.Event()
    cancelled = threading.Event()
    recorded = threading.Event()

    async def inspect(_lease: ObservationLease) -> None:
        """Wait indefinitely, with observable cooperative cancellation."""
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def interrupted(is_cancelled: bool) -> None:
        """Require the owner to classify its interrupted work honestly."""
        assert is_cancelled
        recorded.set()

    assert (
        owner.submit(inspect, subject_bytes=4, on_interrupted=interrupted)
        is ObservationAdmission.ACCEPTED
    )
    assert started.wait(2)
    owner.close(timeout_seconds=0)
    assert recorded.wait(2)
    assert cancelled.wait(2)


def test_preparation_capacity_survives_close_until_its_actual_caller_finishes() -> None:
    """Shutdown cannot release an encoding permit still held by a live request thread."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    started = threading.Event()
    release = threading.Event()

    def prepare() -> None:
        """Hold a synchronous preparation beyond the finite shutdown deadline."""
        with owner.preparation() as admitted:
            assert admitted
            started.set()
            assert release.wait(3)

    caller = threading.Thread(target=prepare)
    caller.start()
    try:
        assert started.wait(2)
        before = time.monotonic()
        owner.close(timeout_seconds=0.02)
        assert time.monotonic() - before < 0.5
        assert owner._preparing
        assert not owner._idle.is_set()
        with owner.preparation(deduplicating=True) as admitted:
            assert not admitted
        assert owner._preparing
    finally:
        release.set()
        caller.join(timeout=2)
    assert not caller.is_alive()
    assert owner._idle.is_set()
    assert not owner._preparing
    with owner.preparation() as admitted:
        assert not admitted


def test_job_completion_does_not_end_an_active_deduplication_permit() -> None:
    """The idle/drain event covers both running subjects and their bounded preparation."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    started = threading.Event()
    release = threading.Event()

    async def inspect(_lease: ObservationLease) -> None:
        """Complete a full owner reservation while a deduplication preparation is active."""
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)

    try:
        assert (
            owner.submit(inspect, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert started.wait(2)
        with owner.preparation() as admitted:
            assert not admitted
        with owner.preparation(deduplicating=True) as admitted:
            assert admitted
            release.set()
            deadline = time.monotonic() + 2
            while owner._jobs and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not owner._jobs
            assert not owner._idle.is_set()
        assert owner._idle.is_set()
    finally:
        release.set()
        owner.close(timeout_seconds=2)


def test_preparation_exception_releases_only_its_own_permit() -> None:
    """Rejected contexts cannot return another caller's permit, even through an exception."""
    owner = ObservationOwner(max_jobs=1, max_bytes=4)
    with pytest.raises(ValueError, match="synthetic failure"):
        with owner.preparation() as admitted:
            assert admitted
            with owner.preparation(deduplicating=True) as contending:
                assert not contending
            assert owner._preparing
            raise ValueError("synthetic failure")
    assert not owner._preparing
    assert owner._idle.is_set()
    with owner.preparation() as admitted:
        assert admitted
    owner.close(timeout_seconds=0)


@pytest.mark.parametrize("limit", ["jobs", "bytes"])
@pytest.mark.parametrize("interrupt", ["timeout", "shutdown"])
def test_subject_budget_outlives_cancel_ignoring_isolated_work(limit: str, interrupt: str) -> None:
    """A timed-out outer job cannot admit another subject while its worker retains the first."""
    owner = ObservationOwner(
        max_jobs=1 if limit == "jobs" else 2, max_bytes=4 if limit == "bytes" else 8
    )
    inspects = BoundedInspect(max_inflight=2)
    started = threading.Event()
    release = threading.Event()
    outer_done = threading.Event()
    constructed: list[str] = []

    async def blocked() -> None:
        """Ignore cancellation by blocking the isolated event loop until released."""
        started.set()
        assert release.wait(5)

    async def inspect(lease: ObservationLease) -> None:
        """End the outer deadline while the isolated invocation is still live."""
        constructed.append("first")
        try:
            await inspects.run(
                blocked, 0.03 if interrupt == "timeout" else 5, adapter_id="first", retention=lease
            )
        except ClassifierTimeoutError:
            pass
        finally:
            outer_done.set()

    async def second(lease: ObservationLease) -> None:
        """Use another adapter so quarantine cannot masquerade as an owner capacity limit."""
        constructed.append("second")

        async def allowed() -> None:
            """Finish a healthy second observation immediately."""

        await inspects.run(allowed, 1, adapter_id="second", retention=lease)

    try:
        assert (
            owner.submit(inspect, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        assert started.wait(2)
        if interrupt == "shutdown":
            owner.close(timeout_seconds=0)
        assert outer_done.wait(2)
        assert inspects.detached_inspect_count() == 1
        assert (
            owner.submit(second, subject_bytes=4, on_interrupted=lambda _: None)
            is ObservationAdmission.SKIPPED
        )
        assert constructed == ["first"]
        assert owner._bytes == 4
        assert len(owner._jobs) == 1
        release.set()
        assert owner._idle.wait(2)
        assert owner._bytes == 0
        if interrupt == "timeout":
            assert (
                owner.submit(second, subject_bytes=4, on_interrupted=lambda _: None)
                is ObservationAdmission.ACCEPTED
            )
            deadline = time.monotonic() + 2
            while constructed != ["first", "second"] and time.monotonic() < deadline:
                time.sleep(0.001)
            assert constructed == ["first", "second"]
    finally:
        release.set()
        owner.close(timeout_seconds=2)


@pytest.mark.parametrize("failed", [False, True])
def test_idle_worker_cannot_retain_a_released_observation_subject(failed: bool) -> None:
    """A completed result or exception traceback leaves no uncharged subject in an idle worker."""

    class Subject:
        """Provide a weakly referenced stand-in for private classifier input."""

    owner = ObservationOwner(max_jobs=1, max_bytes=10)
    inspects = BoundedInspect(max_inflight=1)
    subject = Subject()
    retained = weakref.ref(subject)

    def factory(payload: Subject) -> Callable[[ObservationLease], Coroutine[object, object, None]]:
        """Create the same subject ownership chain as the engine's classifier closure."""

        async def work(lease: ObservationLease) -> None:
            """Let either a result or an exception reference this admitted subject."""

            async def inspect() -> Subject:
                """Carry the subject in the result or in a failed frame's local values."""
                if failed:
                    raise RuntimeError("synthetic detector outage")
                return payload

            try:
                await inspects.run(inspect, 1, adapter_id="fixture", retention=lease)
            except RuntimeError:
                pass

        return work

    try:
        assert (
            owner.submit(factory(subject), subject_bytes=10, on_interrupted=lambda _: None)
            is ObservationAdmission.ACCEPTED
        )
        del subject
        assert owner._idle.wait(2)
        gc.collect()
        assert retained() is None
        assert owner._bytes == 0
    finally:
        owner.close(timeout_seconds=2)

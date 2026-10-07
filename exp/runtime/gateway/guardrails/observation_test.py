"""Bounded subject ownership and shutdown tests for observation scheduling."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from collections.abc import Callable, Coroutine

import pytest

from exp.runtime.gateway.guardrails.bounded import BoundedInspect, ClassifierTimeoutError
from exp.runtime.gateway.guardrails.observation import ObservationLease, ObservationOwner


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
        assert owner.submit(construct, subject_bytes=4, on_interrupted=lambda _: None)
        assert started.wait(2)
        assert not owner.submit(construct, subject_bytes=1, on_interrupted=lambda _: None)
        assert len(built) == 1
    finally:
        release.set()
        owner.close(timeout_seconds=2)
    assert not owner.submit(construct, subject_bytes=1, on_interrupted=lambda _: None)
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

    assert owner.submit(inspect, subject_bytes=4, on_interrupted=interrupted)
    assert started.wait(2)
    owner.close(timeout_seconds=0)
    assert recorded.wait(2)
    assert cancelled.wait(2)


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
        assert owner.submit(inspect, subject_bytes=4, on_interrupted=lambda _: None)
        assert started.wait(2)
        if interrupt == "shutdown":
            owner.close(timeout_seconds=0)
        assert outer_done.wait(2)
        assert inspects.detached_inspect_count() == 1
        assert not owner.submit(second, subject_bytes=4, on_interrupted=lambda _: None)
        assert constructed == ["first"]
        assert owner._bytes == 4
        assert len(owner._jobs) == 1
        release.set()
        assert owner._idle.wait(2)
        assert owner._bytes == 0
        if interrupt == "timeout":
            assert owner.submit(second, subject_bytes=4, on_interrupted=lambda _: None)
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
        assert owner.submit(factory(subject), subject_bytes=10, on_interrupted=lambda _: None)
        del subject
        assert owner._idle.wait(2)
        gc.collect()
        assert retained() is None
        assert owner._bytes == 0
    finally:
        owner.close(timeout_seconds=2)

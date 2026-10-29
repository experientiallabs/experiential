"""Bound background observations until every isolated user of their subject exits."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from enum import StrEnum

from exp.runtime.gateway.guardrails.bounded import try_start_on_native_loop

_logger = logging.getLogger(__name__)


class ObservationAdmission(StrEnum):
    """Distinguish admitted work, finite admission skips, and unavailable scheduling."""

    ACCEPTED = "accepted"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class ObservationLease:
    """Retain one subject budget for the outer job and any still-running classifiers.

    Attributes:
        subject_bytes: Complete input bytes charged to the owner until the last user exits.
    """

    def __init__(self, subject_bytes: int, released: Callable[[ObservationLease], None]) -> None:
        """Start with the outer job's reference and one owner release callback."""
        self.subject_bytes = subject_bytes
        self._released = released
        self._lock = threading.Lock()
        self._references = 1
        self._cancel_requested = False
        self._cancel: Callable[[], None] | None = None

    def retain(self) -> None:
        """Reserve the subject for an isolated invocation before it begins."""
        with self._lock:
            if self._references == 0:
                raise RuntimeError("observation subject lease has already ended")
            self._references += 1

    def release(self) -> None:
        """Release capacity only after the last actual subject user has finished."""
        with self._lock:
            self._references -= 1
            finished = self._references == 0
        if finished:
            self._released(self)

    def finish(self) -> None:
        """Drop the completed outer task reference before releasing its subject lease."""
        with self._lock:
            self._cancel = None
        self.release()

    def bind_cancellation(self) -> None:
        """Cancel the real task, whose completion follows cleanup, instead of its proxy future."""
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None

        def cancel() -> None:
            """Request cancellation on the observation task's owning loop."""
            loop.call_soon_threadsafe(task.cancel)

        with self._lock:
            self._cancel = cancel
            cancelled = self._cancel_requested
        if cancelled:
            raise asyncio.CancelledError

    def cancel(self) -> None:
        """Request cooperative cancellation without releasing retained subject capacity."""
        with self._lock:
            self._cancel_requested = True
            callback = self._cancel
        if callback is not None:
            callback()


class ObservationOwner:
    """Own a finite set of ephemeral subjects through actual isolated task termination.

    Scheduling uses the existing classifier loop. One independent preparation permit
    bounds synchronous projection and hashing before exact retained-byte admission.
    A full owner rejects new subjects instead of queueing text. Timeout and shutdown
    cannot free capacity still held by preparation or a cancellation-ignoring classifier.
    """

    def __init__(self, *, max_jobs: int, max_bytes: int) -> None:
        """Set positive process-local limits on jobs and retained complete subject bytes."""
        if max_jobs < 1 or max_bytes < 1:
            raise ValueError("observation job and byte limits must be positive")
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._jobs: dict[ObservationLease, Future[None] | None] = {}
        self._bytes = 0
        self._preparing = False
        self._closed = False
        self._idle = threading.Event()
        self._idle.set()

    @property
    def max_subject_bytes(self) -> int:
        """Bound each preparation candidate by the owner's retained-byte capacity."""
        return self._max_bytes

    @contextmanager
    def preparation(self, *, deduplicating: bool = False) -> Iterator[bool]:
        """Try one nonblocking preparation permit without retaining its request.

        A session with known fingerprints may compare another bounded candidate when
        jobs are full. Other sessions must have retained-job and byte capacity before
        preparing any input. The caller releases all projection/hash temporaries
        before leaving this context. Shutdown never revokes an active permit.
        """
        with self._lock:
            admitted = (
                not self._closed
                and not self._preparing
                and (
                    deduplicating
                    or (len(self._jobs) < self._max_jobs and self._bytes < self._max_bytes)
                )
            )
            if admitted:
                self._preparing = True
                self._idle.clear()
        try:
            yield admitted
        finally:
            if admitted:
                with self._lock:
                    self._preparing = False
                    if not self._jobs:
                        self._idle.set()

    @property
    def available_subject_bytes(self) -> int:
        """Hint whether new work can fit; submit still atomically reserves capacity."""
        with self._lock:
            if self._closed or len(self._jobs) >= self._max_jobs:
                return 0
            return self._max_bytes - self._bytes

    def _release(self, lease: ObservationLease) -> None:
        """Remove a reservation only when the outer job and all workers have exited."""
        with self._lock:
            del self._jobs[lease]
            self._bytes -= lease.subject_bytes
            if not self._jobs and not self._preparing:
                self._idle.set()

    def submit(
        self,
        factory: Callable[[ObservationLease], Coroutine[object, object, None]],
        *,
        subject_bytes: int,
        on_interrupted: Callable[[bool], None],
    ) -> ObservationAdmission:
        """Admit a bounded subject and give every actual worker the same retention lease.

        Args:
            factory: Constructs the engine coroutine after bounded admission.
            subject_bytes: Exact complete serialized size retained for this observation.
            on_interrupted: True reports cancellation, False an unexpected failure.

        Returns:
            Accepted work, a capacity/shutdown skip, or scheduler unavailability.
        """
        with self._lock:
            if (
                self._closed
                or len(self._jobs) >= self._max_jobs
                or self._bytes + subject_bytes > self._max_bytes
            ):
                return ObservationAdmission.SKIPPED
            lease = ObservationLease(subject_bytes, self._release)
            self._jobs[lease] = None
            self._bytes += subject_bytes
            self._idle.clear()

        async def inspect() -> None:
            """Bind real-task cancellation before invoking the shared engine."""
            lease.bind_cancellation()
            await factory(lease)

        coroutine = inspect()
        try:
            future = try_start_on_native_loop(coroutine)
        except Exception:  # noqa: BLE001 - observation cannot reject serving requests.
            future = None
        if future is None:
            coroutine.close()
            lease.release()
            return ObservationAdmission.UNAVAILABLE
        with self._lock:
            self._jobs[lease] = future
            closed = self._closed

        def finished(done: Future[None]) -> None:
            """Release the outer reference while isolated workers retain their own references."""
            try:
                cancelled = done.cancelled()
                if cancelled or done.exception() is not None:
                    on_interrupted(cancelled)
            except Exception:  # noqa: BLE001 - recorder failure cannot affect serving or shutdown.
                _logger.warning("guardrail observation recording failed")
            finally:
                lease.finish()

        future.add_done_callback(finished)
        if closed:
            lease.cancel()
        return ObservationAdmission.ACCEPTED

    def close(self, *, timeout_seconds: float) -> None:
        """Stop admission, drain within the host budget, and cancel remaining actual tasks."""
        if timeout_seconds < 0:
            raise ValueError("observation shutdown timeout cannot be negative")
        with self._lock:
            self._closed = True
        if self._idle.wait(timeout_seconds):
            return
        with self._lock:
            remaining = tuple(self._jobs)
        for lease in remaining:
            lease.cancel()

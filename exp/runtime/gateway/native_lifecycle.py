"""Explicit ownership and bounded shutdown for local native gateway compositions."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Self

from exp.common.sqlite.connection import close_idle_connections
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
from exp.runtime.gateway.routing import SelectionWorkerPool
from exp.runtime.gateway.video.engine import VideoEngine

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting

_SWEEP_INTERVAL_SECONDS = 5.0


class NativeAccountingLifecycle:
    """Stop one settlement sweeper without erasing retained settlement or request evidence."""

    _sweep_stop: threading.Event
    _sweeper: threading.Thread

    def sweep_expired(self) -> None:
        """Recover retained settlements and close abandoned requests; owners implement it."""
        raise NotImplementedError

    def _sweep_loop(self) -> None:
        """Periodically reconcile retained or expired attempts until the owner stops."""
        try:
            while not self._sweep_stop.wait(_SWEEP_INTERVAL_SECONDS):
                self.sweep_expired()
        finally:
            close_idle_connections()

    @property
    def running(self) -> bool:
        """Report worker liveness separately from the durable write-health contract."""
        return not self._sweep_stop.is_set() and self._sweeper.is_alive()

    def close(self, *, timeout_seconds: float = 30.0) -> None:
        """Interrupt idle sweeping and wait only for the bounded caller-owned drain budget."""
        if timeout_seconds < 0:
            raise ValueError("settlement shutdown timeout cannot be negative")
        self._sweep_stop.set()
        self._sweeper.join(timeout_seconds)
        if self._sweeper.is_alive():
            raise TimeoutError("native settlement sweeper did not stop within its shutdown budget")


class LocalGatewayLifecycle:
    """Close local worker resources after the serving control plane has stopped."""

    write_ledger: GroupCommitAttemptLedger
    selection_workers: SelectionWorkerPool

    def close(self, *, timeout_seconds: float = 30.0) -> None:
        """Stop selection work, drain durable writes, and release this thread's idle handles."""
        if timeout_seconds < 0:
            raise ValueError("gateway shutdown timeout cannot be negative")
        deadline = time.monotonic() + timeout_seconds
        try:
            if not self.selection_workers.shutdown(drain_timeout_seconds=timeout_seconds):
                raise TimeoutError("gateway selection workers remain active; ledger retained")
            self.write_ledger.close(timeout_seconds=max(0.0, deadline - time.monotonic()))
        finally:
            close_idle_connections()

    def __enter__(self) -> Self:
        """Expose this composition with explicit context-owned resources."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Drain the composition on success and on failed serving setup."""
        self.close()


class NativeControlLifecycle:
    """Stop control-plane-owned pollers without closing caller-owned components."""

    _videos: VideoEngine | None
    _accounting: NativeAttemptAccounting

    def _start_after_accounting(self, start: Callable[[], None]) -> None:
        """Run one startup step once accounting exists, closing accounting if the step fails."""
        try:
            start()
        except BaseException:
            self._accounting.close()
            raise

    def close(self, *, timeout_seconds: float = 30.0) -> None:
        """Bound poller drain and refuse to report shutdown while a provider poll still runs."""
        if timeout_seconds < 0:
            raise ValueError("gateway shutdown timeout cannot be negative")
        deadline = time.monotonic() + timeout_seconds
        try:
            if self._videos is not None and not self._videos.stop_poller(
                timeout_seconds=timeout_seconds
            ):
                raise TimeoutError("native video poller did not stop within its shutdown budget")
        finally:
            self._accounting.close(timeout_seconds=max(0.0, deadline - time.monotonic()))

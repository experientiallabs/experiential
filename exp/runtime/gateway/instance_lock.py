"""The single live gateway owner lock for one EXP root, held through worker teardown."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from exp.common.core.locks import CoordinationLock, Timeout
from exp.runtime.gateway.lifecycle_fields import GatewayLifecycleError


class GatewayTeardownIncomplete(GatewayLifecycleError):
    """Serving workers may still be running after a failed or timed-out drain."""


_RETAINED_OWNER_LOCKS: list[CoordinationLock] = []
"""Owner locks kept until process exit because their workers may still write."""


class GatewayOwnership:
    """One process's ownership of a gateway root, released only after a finished teardown."""

    def __init__(self) -> None:
        """Start owned, with teardown not yet known to be complete."""
        self._torn_down = False

    def teardown_complete(self) -> None:
        """Record that every serving worker has stopped, so the root can be released."""
        self._torn_down = True

    @property
    def torn_down(self) -> bool:
        """Return whether teardown finished before the ownership scope ended."""
        return self._torn_down


@contextmanager
def gateway_instance_lock(root: Path, *, port: int) -> Iterator[GatewayOwnership]:
    """Hold the single live gateway owner lock for one EXP root.

    The lock is released when the scope ends only if the owner recorded a finished teardown.
    A failed, timed-out or interrupted teardown keeps it held until the process exits, so no
    other gateway reconciles while this process's workers may still write.

    Args:
        root: EXP root whose gateway database and snapshots are served.
        port: Requested loopback port, included only in actionable diagnostics.

    Yields:
        The ownership, on which the owner records a finished teardown.

    Raises:
        GatewayLifecycleError: Another process currently owns the root.
    """
    state_dir = root / "gateway"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = CoordinationLock(state_dir / "run.lock", timeout=0, mode=0o600)
    try:
        lock.acquire()
    except Timeout:
        raise GatewayLifecycleError(
            f"another gateway process already owns {root} (requested port {port})"
        ) from None
    ownership = GatewayOwnership()
    try:
        yield ownership
    finally:
        if ownership.torn_down:
            lock.release()
        else:
            _RETAINED_OWNER_LOCKS.append(lock)

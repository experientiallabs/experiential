"""The single live gateway owner lock for one EXP root, held through worker teardown."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from exp.runtime.gateway.lifecycle_fields import GatewayLifecycleError


class GatewayTeardownIncomplete(GatewayLifecycleError):
    """Serving workers may still be running after a failed or timed-out drain."""


_RETAINED_OWNER_LOCKS: list[FileLock] = []
"""Owner locks kept until process exit because their workers may still write."""


@contextmanager
def gateway_instance_lock(root: Path, *, port: int) -> Iterator[None]:
    """Hold the single live gateway owner lock for one EXP root.

    Args:
        root: EXP root whose gateway database and snapshots are served.
        port: Requested loopback port, included only in actionable diagnostics.

    Yields:
        None while this process exclusively owns the local gateway.

    Raises:
        GatewayLifecycleError: Another process currently owns the root.
        GatewayTeardownIncomplete: Serving reported unfinished teardown; the lock stays held
            until the process exits, so no other gateway reconciles under live workers.
    """
    state_dir = root / "gateway"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = FileLock(state_dir / "run.lock", timeout=0, mode=0o600)
    try:
        lock.acquire()
    except Timeout:
        raise GatewayLifecycleError(
            f"another gateway process already owns {root} (requested port {port})"
        ) from None
    try:
        yield
    except GatewayTeardownIncomplete:
        _RETAINED_OWNER_LOCKS.append(lock)
        raise
    except BaseException:
        lock.release()
        raise
    lock.release()

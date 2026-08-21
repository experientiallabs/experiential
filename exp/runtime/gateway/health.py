"""Bounded per-deployment circuit and throttle state for exact-model waterfalls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass

DeploymentHealthKey = tuple[str, str, str]

_HARD_FAILURES = {
    GatewayFailureClass.PROVIDER_AUTHENTICATION,
    GatewayFailureClass.PROVIDER_NOT_FOUND,
}
_OPERATIONAL_FAILURES = {
    GatewayFailureClass.TRANSPORT,
    GatewayFailureClass.TIMEOUT,
    GatewayFailureClass.MALFORMED_RESPONSE,
    GatewayFailureClass.PROVIDER_INTERNAL,
}


@dataclass
class _DeploymentHealth:
    """Mutable content-free health state protected by the registry lock."""

    consecutive_failures: int = 0
    open_until: float = 0.0
    throttle_until: float = 0.0
    half_open_probe: bool = False
    last_resort_probe: bool = False
    refusal_count: int = 0


class DeploymentHealthRegistry:
    """Apply isolated circuits and throttle windows without changing logical models."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        open_seconds: float = 30.0,
        throttle_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize finite in-process deployment health policy.

        Args:
            failure_threshold: Consecutive operational failures that open one circuit.
            open_seconds: Circuit cooldown before one half-open probe.
            throttle_seconds: Independent suppression window after provider throttling.
            clock: Injectable monotonic clock.

        Raises:
            ValueError: A threshold or duration is not positive.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least one")
        if open_seconds <= 0 or throttle_seconds <= 0:
            raise ValueError("health suppression windows must be positive")
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._throttle_seconds = throttle_seconds
        self._clock = clock
        self._states: dict[DeploymentHealthKey, _DeploymentHealth] = {}
        self._lock = threading.Lock()

    def claim(self, key: DeploymentHealthKey) -> bool:
        """Reserve one eligible route or half-open probe.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until > now:
                return False
            if state.open_until > now:
                return False
            if state.consecutive_failures < self._failure_threshold:
                return True
            if state.half_open_probe:
                return False
            state.half_open_probe = True
            return True

    def claim_last_resort(self, key: DeploymentHealthKey) -> bool:
        """Reserve one bounded probe through an open circuit when nothing else is eligible.

        A request with no claimable deployment would otherwise fail for the whole
        cooldown even after the provider has recovered. One in-flight probe per
        deployment is allowed through an open circuit so recovery is discovered by
        real traffic instead of a fixed timer. Throttle windows stay authoritative
        because the provider explicitly asked for backoff.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this suppressed deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until > now:
                return False
            if state.last_resort_probe:
                return False
            state.last_resort_probe = True
            return True

    def dispatch_opened(self, key: DeploymentHealthKey) -> None:
        """Restore admission once one provider dispatch opens successfully.

        An opened stream proves the deployment is reachable again, so waiting for
        the terminal event would refuse concurrent traffic for the full stream
        duration. A later terminal failure re-applies suppression normally.

        Args:
            key: Catalog, deployment, and connection identity tuple.
        """
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.half_open_probe = False
            state.last_resort_probe = False

    def succeeded(self, key: DeploymentHealthKey) -> None:
        """Close one circuit after a successful terminal provider result.

        Args:
            key: Catalog, deployment, and connection identity tuple.
        """
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.half_open_probe = False
            state.last_resort_probe = False

    def failed(self, key: DeploymentHealthKey, failure: GatewayFailure) -> None:
        """Apply one normalized provider outcome to circuit or throttle state.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            failure: Sanitized provider failure classification.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.half_open_probe = False
            state.last_resort_probe = False
            if failure.failure_class == GatewayFailureClass.THROTTLED:
                state.throttle_until = max(
                    state.throttle_until,
                    now + self._throttle_seconds,
                )
                return
            if failure.failure_class == GatewayFailureClass.REFUSAL:
                state.refusal_count += 1
                return
            if failure.failure_class in _HARD_FAILURES:
                state.consecutive_failures = self._failure_threshold
                state.open_until = now + self._open_seconds
                return
            if failure.failure_class in _OPERATIONAL_FAILURES:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self._failure_threshold:
                    state.open_until = now + self._open_seconds

    def release_probe(self, key: DeploymentHealthKey) -> None:
        """Release a claimed half-open probe that never reached provider dispatch.

        Args:
            key: Catalog, deployment, and connection identity tuple.
        """
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.half_open_probe = False
                state.last_resort_probe = False

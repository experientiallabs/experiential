# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Deployment health and cache observations for native attempt accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayEvent, GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_execution import (
    InflightRequest,
    deployment_health_key,
    rung_load_key,
)
from exp.runtime.gateway.native_rung_policy import record_cache_fraction
from exp.runtime.gateway.native_settlement import settlement_rate_limit

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


class NativeAccountingHealthMixin:
    """Record settled attempt outcomes on the owning accounting registries."""

    def _record_health(
        self: NativeAttemptAccounting,
        entry: InflightRequest,
        attempt_id: str,
        *,
        opened: bool,
        failure: GatewayFailure | None,
        settlement: JsonObject | None = None,
    ) -> None:
        """Apply one settled attempt's outcome to the deployment circuits.

        Mirrors the executor's recording order: a successful dispatch opening
        restores admission first, then the terminal outcome either closes the
        circuit or counts against it. A plan rung whose response reports a used-up
        usage window is throttled until that window resets, success or not, so a
        pool of plans rotates before the first 429.

        Args:
            entry: The owning in-flight request.
            attempt_id: The settled attempt.
            opened: Whether the provider dispatch opened successfully.
            failure: The terminal failure, or ``None`` for a success.
            settlement: The settlement payload carrying the harvested rate-limit headers.
        """
        # The rung's bounded-admission slot frees with the health recording:
        # both releases are idempotent, so settle, abandon, and the sweep can
        # each fire without double-counting.
        self._loads.release_attempt(attempt_id)
        depth = entry.attempt_depths.get(attempt_id)
        if depth is None:
            return
        deployment = entry.route.deployments[depth]
        key = deployment_health_key(entry.authorization, deployment)
        if opened:
            self._health.dispatch_opened(key)
        if failure is not None:
            policy = deployment.gateway.dispatch
            if failure.failure_class == GatewayFailureClass.THROTTLED and (
                policy is not None
                and (
                    policy.concurrency_bound is not None
                    or policy.requests_per_minute is not None
                    or policy.tokens_per_minute is not None
                )
            ):
                # Passive-adaptive calibration: the provider just proved the
                # observed dispatch rate is too high for this physical lane,
                # so the rung's learned ceiling clamps to it. Recovery creep
                # rediscovers headroom without synthetic probes. Gated on an
                # admission-participating policy: only such rungs reserve
                # through the registry, so only they carry a real observed
                # window (and only they would ever enforce the ceiling).
                self._loads.record_throttle(rung_load_key(deployment))
            self._health.failed(key, failure)
        else:
            self._health.succeeded(key)
        exhausted = settlement_rate_limit(settlement).exhausted_reset_after_seconds
        if exhausted is not None:
            self._health.exhausted(key, exhausted)

    def _record_cache_fraction(
        self: NativeAttemptAccounting,
        entry: InflightRequest,
        attempt_id: str,
        terminal: GatewayEvent,
    ) -> None:
        """Apply the observed-only cache sample once through the existing rung-policy owner."""
        record_cache_fraction(
            self._loads,
            entry,
            attempt_id,
            terminal,
            lock=self._lock,
            sample_gate=self._cache_sample_gate,
        )

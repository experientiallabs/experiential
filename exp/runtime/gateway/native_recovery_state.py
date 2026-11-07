"""Admission access to the session recovery registry retained by native serving."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from exp_gateway_native import NativeSessionRecovery

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.affinity import affinity_fingerprint, affinity_seed_material
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_recovery import session_cache_key
from exp.runtime.gateway.recovery import (
    RecoveryCause,
    RecoveryDecision,
    RecoveryScope,
    RecoverySnapshot,
    SessionCacheKey,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_binding import frozen_scope


def key_registration(key: SessionCacheKey) -> list[JsonValue]:
    """Keep tenant and prefix boundaries explicit; no prompt is retained."""
    return [key.organization_id, key.identity_id, key.fingerprint.hex(), key.prefix_key]


def scope_registration(scope: RecoveryScope) -> JsonObject:
    """Transfer private credential binding only to native process memory."""
    return scope.operational().model_dump(mode="json") | {
        "credential_scope": scope.credential_scope,
        "organization_id": scope.organization_id,
    }


def recovery_registration(entry: InflightRequest) -> JsonObject | None:
    """Freeze the exact wire scopes and per-stage cache policy while admission owns Python."""
    request = entry.request
    if not isinstance(request, GatewayRequest) or not request.provider_prompt_cache_key:
        return None
    key = session_cache_key(entry)
    if key is None:
        material = affinity_seed_material(
            request,
            continuation_episode_key=None
            if entry.continuation is None
            else entry.continuation.episode_key,
            request_id=entry.authorization.request_id,
        )
        key = SessionCacheKey(
            entry.authorization.organization_id,
            entry.authorization.identity_id,
            affinity_fingerprint(
                organization_id=entry.authorization.organization_id,
                identity_id=entry.authorization.identity_id,
                material=material,
            ),
            "",
        )
    lanes: list[JsonValue] = []
    for depth, deployment in enumerate(entry.route.deployments):
        scope = frozen_scope(
            entry.recovery_bindings, deployment, entry.authorization.organization_id
        )
        if scope is None:
            lanes.append(None)
            continue
        dispatch = deployment.gateway.dispatch
        stage = entry.route.snapshot.stage_for_depth(depth)
        lanes.append(
            {
                "scope": scope_registration(scope),
                "retention_seconds": deployment.gateway.cache_retention_seconds,
                "sticky_seconds": dispatch.sticky_spill_seconds
                if dispatch is not None and stage.failover_mode == "maximize_cache_affinity"
                else None,
            }
        )
    return {"key": key_registration(key), "lanes": lanes}


class NativeRecovery(SessionRecoveryRegistry):
    """Share native evidence with admission while preserving its callback contract."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        maximum_sessions: int = 65_536,
        maximum_evidence_age_seconds: float = 3600,
        observation_lifetime_seconds: float = 60,
        cooldown_seconds: float = 5,
    ) -> None:
        self._clock = clock
        self.native = NativeSessionRecovery(
            maximum_sessions,
            maximum_evidence_age_seconds,
            observation_lifetime_seconds,
            cooldown_seconds,
        )

    def _write(self, op: str, key: SessionCacheKey, fields: JsonObject) -> None:
        self.native.state_json(
            op, json.dumps({"key": key_registration(key), "now": self._clock(), **fields})
        )

    def depart(
        self,
        key: SessionCacheKey,
        deployment_id: str,
        scope: RecoveryScope,
        cause: RecoveryCause,
        *,
        retry_after_seconds: float = 0,
        observed_at: float | None = None,
    ) -> None:
        """Record an admission departure in the same native owner as settlement."""
        self._write(
            "depart",
            key,
            {
                "deployment": deployment_id,
                "scope": scope_registration(scope),
                "cause": cause,
                "retry": retry_after_seconds,
                "observed": observed_at,
            },
        )

    def record_success(
        self,
        key: SessionCacheKey,
        deployment_id: str,
        scope: RecoveryScope,
        *,
        cached_tokens: int,
        cache_write_tokens: int,
        retention_seconds: float | None,
        sticky_seconds: float | None,
        observed_at: float | None = None,
    ) -> None:
        """Publish timestamped evidence from an explicitly synchronous ledger."""
        self._write(
            "success",
            key,
            {
                "deployment": deployment_id,
                "scope": scope_registration(scope),
                "observed": observed_at,
                "meter": {
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "retention_seconds": retention_seconds,
                    "sticky_seconds": sticky_seconds,
                },
            },
        )

    def has_retained_history(self, key: SessionCacheKey, *, live_only: bool = False) -> bool:
        """Read the conservative history preflight used during admission."""
        result = json.loads(
            self.native.state_json(
                "retained",
                json.dumps({"key": key_registration(key), "now": self._clock(), "live": live_only}),
            )
        )
        if not isinstance(result, bool):
            raise TypeError("Invalid native recovery preflight")
        return result

    def choose(
        self,
        key: SessionCacheKey,
        candidates: tuple[tuple[str, RecoveryScope], ...],
        *,
        eligible: Callable[[str], bool],
        snapshot: RecoverySnapshot | None,
        local_capacity: Callable[[str], bool] | None = None,
    ) -> RecoveryDecision:
        """Consume a recovery lease at admission after checking current health and load."""
        result = self.native.choose_json(
            json.dumps(
                {
                    "key": key_registration(key),
                    "candidates": [
                        [deployment, scope_registration(scope)] for deployment, scope in candidates
                    ],
                    "snapshot": None if snapshot is None else snapshot.model_dump(mode="json"),
                    "now": self._clock(),
                }
            ),
            eligible,
            local_capacity,
        )
        return RecoveryDecision(**json.loads(result))

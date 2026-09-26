"""Selected-attempt authority callbacks for opt-in Google cache resources.

Rust owns provider HTTP. These callbacks expose no resource operation until the
host durably claims and reserves it. An abandoned callback never grants a second
creator; unknown claims remain the host's durable accounting responsibility.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.explicit_cache import (
    CacheCreator,
    CacheOffer,
    CacheReady,
    CacheResult,
    ExplicitCacheHost,
    claim_cache,
    finish_cache,
    prepare_cache_offer,
    validate_cache_claim,
    validate_cache_result,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting, NativeBridgeError
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX
from exp.runtime.models.providers.google_cache import GoogleCachePlan, build_google_cache_plan
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.openai_protocol.errors import public_failure_error

_CACHE_FRAMING_TOKEN_BOUND = 4096
_CACHE_FINISH_LIMIT = 16384
CONDITIONAL_CACHE_DISCLOSURE = (
    "cache_control->conditional(explicit_google_cache;requires_allowance)"
)


@dataclass(frozen=True, repr=False)
class NativeCacheBinding:
    """Private immutable correspondence between one admitted wire and its cache plan.

    Attributes:
        deployment: Frozen route deployment used to resolve host authority.
        profile: Exact provider endpoint and credential headers, never logged.
        plan: Exact marked prefix and generation continuation, retained privately.
    """

    deployment: ExactModelDeployment
    profile: GatewayWireProfile
    plan: GoogleCachePlan


@dataclass(repr=False)
class NativeCacheState:
    """Request-local deduplication beside the host's cross-worker durable claims.

    Attributes:
        bindings: Admitted cache plan at each route depth, absent when unsupported.
        offers: Create offers owned by this request, indexed by operation identifier.
        prepared_attempts: Attempt IDs whose prepare callback already ran.
        observed_results: Immutable first observations, including unacknowledged writes.
        recording_operations: Operations currently recording outside the execution gate.
        recorded_results: Validated results whose durable host write returned.
    """

    bindings: tuple[NativeCacheBinding | None, ...]
    offers: dict[str, tuple[str, CacheOffer, NativeCacheBinding]] = field(default_factory=dict)
    prepared_attempts: set[str] = field(default_factory=set)
    observed_results: dict[str, CacheResult] = field(default_factory=dict)
    recording_operations: set[str] = field(default_factory=set)
    recorded_results: dict[str, CacheResult] = field(default_factory=dict)


def validate_explicit_cache_host(host: ExplicitCacheHost | None) -> ExplicitCacheHost | None:
    """Refuse an incomplete host interface before serving or authorizing cache spend."""
    if host is not None and any(
        not callable(getattr(host, method, None)) for method in ("authority", "claim", "record")
    ):
        raise ValueError("explicit cache host must provide authority, claim and record methods")
    return host


def bind_explicit_cache(
    host: ExplicitCacheHost | None,
    authorization: AuthorizationSnapshot,
    deployments: Sequence[ExactModelDeployment],
    resolved_wires: Sequence[tuple[GatewayWireProfile, NativeWireClient]],
    provider_request: GatewayRequest,
    public_request: GatewayRequest,
    wire_route: list[JsonObject],
) -> tuple[NativeCacheState | None, GatewayRequest]:
    """Retain pure plans without reserving or contacting any provider at admission.

    Only marked, representable native Google text prefixes on non-ZDR requests
    qualify. The actual selected attempt rechecks tenant allowance and credential
    generation through the injected host before any cache operation.
    """
    if host is None or authorization.zdr_requested or provider_request.tool_search is not None:
        return None, public_request
    bindings: list[NativeCacheBinding | None] = []
    for deployment, (profile, _client), wire in zip(
        deployments, resolved_wires, wire_route, strict=True
    ):
        payload = wire.get("upstream_payload")
        plan = (
            build_google_cache_plan(profile, provider_request, payload)
            if isinstance(payload, dict) and not wire.get("zdr_constrained")
            else None
        )
        bindings.append(None if plan is None else NativeCacheBinding(deployment, profile, plan))
        if plan is not None:
            wire["explicit_cache"] = True
    if not any(binding is not None for binding in bindings):
        return None, public_request
    disclosures = public_request.ignored_parameters
    if all(binding is not None for binding in bindings):
        disclosures = tuple(
            item for item in disclosures if not item.endswith(CACHE_CONTROL_NOT_FORWARDED_SUFFIX)
        )
    public_request = public_request.model_copy(
        update={
            "ignored_parameters": tuple(dict.fromkeys((*disclosures, CONDITIONAL_CACHE_DISCLOSURE)))
        }
    )
    return NativeCacheState(tuple(bindings)), public_request


class _CacheBoundary(BaseModel):
    """Bounded attempt selector, authenticated against retained admission authority.

    Attributes:
        request_id: Exact retained request identifier, at most 128 characters.
        deployment_id: Exact selected route identifier, at most 256 characters.
    """

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    request_id: str = Field(min_length=1, max_length=128)
    deployment_id: str = Field(min_length=1, max_length=256)


class _CacheFinish(_CacheBoundary):
    """Allowlisted provider result facts, never raw provider content or credentials.

    Attributes:
        operation_id: Reserved cache operation identifier, at most 128 characters.
        outcome: Ready or unknown; HTTP failure does not prove absence of spend.
        http_status: Optional observed HTTP status, without headers or body.
        name: Optional exact resource name, at most 1024 characters.
        expire_time: Optional absolute provider expiration, at most 64 characters.
        create_time: Optional absolute provider creation time, at most 64 characters.
            Invalid timestamp facts become None without inventing a billing interval.
        total_tokens: Optional positive provider measurement, bounded to int64.
    """

    operation_id: str = Field(min_length=1, max_length=128)
    outcome: str = Field(pattern=r"^(ready|unknown)$")
    http_status: int | None = Field(default=None, ge=100, le=599)
    name: str | None = Field(default=None, max_length=1024)
    expire_time: str | None = Field(default=None, max_length=64)
    create_time: str | None = Field(default=None, max_length=64)
    total_tokens: int | None = Field(default=None, ge=1, le=2**63 - 1)


class _Plane(Protocol):
    """Require request accounting and the optional durable cache host."""

    _accounting: NativeAttemptAccounting
    _explicit_cache: ExplicitCacheHost | None


def _attempt_active(
    accounting: NativeAttemptAccounting, entry: InflightRequest, attempt_id: str
) -> bool:
    """Revalidate request ownership after host I/O without delaying abandonment."""
    return (
        accounting.entry(entry.authorization.request_id) is entry
        and entry.active_attempt_id == attempt_id
        and entry.pending_abandon is None
        and time.monotonic() < entry.deadline_monotonic
    )


def _boundary_failure() -> NativeBridgeError:
    """Return a content-free failure when cache accounting cannot safely proceed."""
    return NativeBridgeError(
        public_failure_error(
            GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="explicit cache accounting is unavailable; contact the operator",
            )
        )
    )


def _encoded(payload: JsonObject) -> str:
    """Serialize only a validated native callback response."""
    return json.dumps(payload, separators=(",", ":"))


def _resource_timestamp(value: str | None) -> float:
    """Parse a provider's timezone-aware absolute timestamp, never local time."""
    if value is None:
        raise ValueError("cache resource omitted timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("cache resource timestamp needs a timezone")
    return parsed.timestamp()


def _with_creation_time(result: CacheResult, value: str | None) -> CacheResult:
    """Attach only a valid provider interval; missing or unusable facts stay unknown."""
    if value is None:
        return result
    try:
        return replace(result, create_time=_resource_timestamp(value))
    except (ValueError, OverflowError, OSError):
        return result


def _ready_payload(plan: GoogleCachePlan, claim: CacheReady) -> str:
    """Project a scoped reusable resource into its exact generation continuation."""
    return _encoded(
        {
            "state": "ready",
            "resource_name": claim.resource_name,
            "resource_prefix": plan.resource_prefix,
            "expires_at": claim.expires_at,
            "payload": plan.apply(claim.resource_name),
        }
    )


class NativeExplicitCacheMixin:
    """Authorize one cache operation for an already reserved physical generation."""

    def prepare_explicit_cache(self: _Plane, argument: str) -> str:
        """Durably reserve an exact marked cache before Rust may submit its HTTP create.

        A missing host keeps existing implicit-cache behavior. The host must
        separately enforce marker-plus-allowance consent, account scope, prices
        and tenant budget. No process-local state substitutes for that authority.
        """
        host = self._explicit_cache
        if host is None:
            return _encoded({"state": "disabled"})
        try:
            if len(argument.encode()) > _CACHE_FINISH_LIMIT:
                raise ValueError("cache callback exceeds limit")
            selected = _CacheBoundary.model_validate_json(argument)
            entry = self._accounting.entry(selected.request_id)
            if entry is None:
                raise ValueError("request is no longer active")
            with entry.execution_lock:
                attempt_id = entry.active_attempt_id
                depth = None if attempt_id is None else entry.attempt_depths.get(attempt_id)
                state = entry.explicit_cache_state
                if (
                    entry.pending_abandon is not None
                    or time.monotonic() >= entry.deadline_monotonic
                    or attempt_id is None
                    or depth is None
                    or state is None
                    or depth >= len(state.bindings)
                ):
                    raise ValueError("cache attempt authority is unavailable")
                binding = state.bindings[depth]
                if binding is None or binding.deployment.deployment_id != selected.deployment_id:
                    raise ValueError("cache request does not match its selected deployment")
                if attempt_id in state.prepared_attempts:
                    return _encoded({"state": "unavailable"})
                state.prepared_attempts.add(attempt_id)
            # Host I/O must not own the execution gate: abandon must be able to
            # finish the generation reservation even while a cache transaction hangs.
            authority = host.authority(entry.authorization, binding.deployment, binding.profile)
            with entry.execution_lock:
                if not _attempt_active(self._accounting, entry, attempt_id):
                    return _encoded({"state": "unavailable"})
                if authority is None:
                    return _encoded({"state": "disabled"})
                plan = binding.plan.bind_vertex_project(authority.vertex_project)
                if plan is None:
                    return _encoded({"state": "unavailable"})
                binding = replace(binding, plan=plan)
                # Never hash an access token: the host supplies a stable opaque
                # credential generation independent of token refreshes.
                endpoint = urlsplit(binding.plan.create_url)
                account = sha256_json(
                    {
                        "endpoint": f"{endpoint.scheme}://{endpoint.netloc}{endpoint.path}",
                        "model": binding.plan.model,
                        "connection": binding.deployment.connection_sha256,
                        "credential_scope": authority.credential_scope,
                    }
                )
                offer = prepare_cache_offer(
                    authority=authority,
                    operation_id="cache-" + uuid4().hex,
                    request_id=selected.request_id,
                    attempt_id=attempt_id,
                    account_key_fingerprint=account,
                    plan_scope_digest=binding.plan.prefix_sha256,
                    resource_prefix=binding.plan.resource_prefix,
                    prefix_bytes=binding.plan.conservative_input_bound,
                    framing_tokens=_CACHE_FRAMING_TOKEN_BOUND,
                    # Whole-second epochs round-trip exactly through Google's
                    # RFC3339 expiration, never extending the reserved horizon.
                    requested_at=float(int(time.time())),
                )
            claim = claim_cache(host, offer, clock=time.time)
            with entry.execution_lock:
                if offer is None or not _attempt_active(self._accounting, entry, attempt_id):
                    return _encoded({"state": "unavailable"})
                if isinstance(claim, CacheReady):
                    return _ready_payload(binding.plan, claim)
                if not isinstance(claim, CacheCreator):
                    return _encoded({"state": "unavailable"})
                state.offers[claim.operation_id] = (attempt_id, offer, binding)
                payload = binding.plan.create_payload
                payload.pop("ttl", None)
                payload["expireTime"] = datetime.fromtimestamp(offer.expires_at, UTC).isoformat()
                return _encoded(
                    {
                        "state": "create",
                        "operation_id": claim.operation_id,
                        "resource_prefix": binding.plan.resource_prefix,
                        "url": binding.plan.create_url,
                        "payload": payload,
                        "expires_at": offer.expires_at,
                    }
                )
        except Exception:  # noqa: BLE001 - sanitize host failures without releasing reservations.
            # A host exception or failed serialization after claim must never
            # trigger another create. The durable host retains the reservation.
            raise _boundary_failure() from None

    def finish_explicit_cache(self: _Plane, argument: str) -> str:
        """Record a bounded provider observation before authorizing resource reuse.

        Malformed or incomplete create responses are unknown, never assumed free.
        A write failure prevents generation and leaves the host reservation intact.
        Repeating the exact result is idempotent locally; the host must also be
        idempotent across process loss and ambiguous transaction acknowledgements.
        """
        host = self._explicit_cache
        if host is None:
            raise _boundary_failure()
        try:
            if len(argument.encode()) > _CACHE_FINISH_LIMIT:
                raise ValueError("cache callback exceeds limit")
            selected = _CacheFinish.model_validate_json(argument)
            entry = self._accounting.entry(selected.request_id)
            if entry is None or entry.explicit_cache_state is None:
                raise ValueError("cache request is no longer retained")
            with entry.execution_lock:
                state = entry.explicit_cache_state
                item = state.offers.get(selected.operation_id)
                if item is None:
                    raise ValueError("cache operation does not belong to this request")
                attempt_id, offer, binding = item
                if (
                    binding.deployment.deployment_id != selected.deployment_id
                    or entry.active_attempt_id != attempt_id
                ):
                    raise ValueError("cache operation differs from the selected attempt")
                previous = state.observed_results.get(offer.operation_id)
                observed_at = time.time() if previous is None else previous.observed_at
                result = CacheResult(
                    operation_id=offer.operation_id,
                    outcome="unknown",
                    observed_at=observed_at,
                    http_status=selected.http_status,
                )
                if selected.outcome == "ready":
                    try:
                        result = CacheResult(
                            operation_id=offer.operation_id,
                            outcome="ready",
                            observed_at=observed_at,
                            resource_name=selected.name,
                            total_tokens=selected.total_tokens,
                            expire_time=_resource_timestamp(selected.expire_time),
                            http_status=selected.http_status,
                        )
                        binding.plan.apply(selected.name or "")
                        # Validation alone does not grant reuse after expiry;
                        # accounting can still record a late known resource.
                        validate_cache_result(offer, result)
                        result = _with_creation_time(result, selected.create_time)
                    except ValueError:
                        result = CacheResult(
                            operation_id=offer.operation_id,
                            outcome="unknown",
                            observed_at=observed_at,
                            http_status=selected.http_status,
                        )
                if previous is not None:
                    if result != previous:
                        raise ValueError("cache result changed after observation")
                else:
                    state.observed_results[offer.operation_id] = result
                if offer.operation_id in state.recording_operations:
                    # An unacknowledged observation cannot grant generation, even
                    # to a duplicate callback. The original write still owns it.
                    raise ValueError("cache result accounting is still pending")
                needs_recording = offer.operation_id not in state.recorded_results
                if needs_recording:
                    state.recording_operations.add(offer.operation_id)
            if needs_recording:
                try:
                    finish_cache(host, offer, result)
                    with entry.execution_lock:
                        state.recorded_results[offer.operation_id] = result
                finally:
                    with entry.execution_lock:
                        state.recording_operations.discard(offer.operation_id)
            with entry.execution_lock:
                if result.outcome != "ready" or not _attempt_active(
                    self._accounting, entry, attempt_id
                ):
                    return _encoded({"state": "unavailable"})
                if (
                    result.resource_name is None
                    or result.total_tokens is None
                    or result.expire_time is None
                ):
                    raise ValueError("ready cache result is incomplete")
                ready = CacheReady(result.resource_name, result.expire_time, result.total_tokens)
                try:
                    validate_cache_claim(offer, ready, now=time.time())
                except ValueError:
                    return _encoded({"state": "unavailable"})
                return _ready_payload(binding.plan, ready)
        except Exception:  # noqa: BLE001 - sanitize host failures without releasing reservations.
            raise _boundary_failure() from None

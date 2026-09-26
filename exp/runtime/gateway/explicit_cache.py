"""Content-free explicit-cache authority and durable reservation contracts.

``NativeExplicitCacheMixin`` consumes these helpers before provider cache creation.
The host owns tenant policy, verified pricing, cross-worker exclusion and durable
cost reservations. This module supplies no store, credential lookup or HTTP client.
A token upper bound prices an offer but never proves the provider's minimum input
requirement: only a provider measurement or cache-create acceptance proves that.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.google_cache import VertexCacheProject

CACHE_TTL_SECONDS = 300
CACHE_EXPIRY_SAFETY_SECONDS = 5
_MAX_INT64 = (1 << 63) - 1
_MAXIMUM_PREFIX_BYTES = 10 * 1024 * 1024
_RESOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,256}")
_RESOURCE_PREFIX = re.compile(
    r"(?:cachedContents/|projects/[A-Za-z0-9_-]{1,256}/"
    r"locations/[A-Za-z0-9_-]{1,256}/cachedContents/)"
)
_SHA256 = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True, repr=False)
class GoogleCacheAuthority:
    """Host-authored policy for one authorized tenant and exact provider account.

    Attributes:
        tenant_scope: Nonempty opaque tenant isolation scope, never caller-selected.
        credential_scope: Nonsecret opaque credential generation. This is not an
            attempt-accounting generation, API key, access token or credential hash.
        minimum_input_tokens: Verified positive provider eligibility threshold.
        create_input_nano_usd_per_million: Verified nonnegative cache-create input
            rate, bounded to a signed 64-bit integer. An explicit integer zero
            permits verified free input creation; missing or unknown is invalid.
        storage_nano_usd_per_million_token_hour: Verified positive storage rate,
            bounded to a signed 64-bit integer.
        maximum_prefix_bytes: Positive serialized-prefix byte ceiling, default
            10 MiB. Tenant daily budgets are enforced atomically by host claim,
            not by a scalar allowance copied into this authority.
        vertex_project: Host-verified endpoint project to canonical project-number
            association. Required for project-ID Vertex endpoints, absent for
            Gemini. Numeric Vertex endpoints already supply their exact number.
    """

    tenant_scope: str
    credential_scope: str
    minimum_input_tokens: int
    create_input_nano_usd_per_million: int
    storage_nano_usd_per_million_token_hour: int
    maximum_prefix_bytes: int = _MAXIMUM_PREFIX_BYTES
    vertex_project: VertexCacheProject | None = None

    def __post_init__(self) -> None:
        """Reject incomplete scopes and unverified or unbounded numeric authority."""
        _scope(self.tenant_scope, "tenant_scope")
        _scope(self.credential_scope, "credential_scope")
        _integer(self.minimum_input_tokens, "minimum_input_tokens", minimum=1)
        _integer(self.create_input_nano_usd_per_million, "create input rate")
        _integer(self.storage_nano_usd_per_million_token_hour, "storage rate", minimum=1)
        _integer(self.maximum_prefix_bytes, "maximum_prefix_bytes", minimum=1)
        if self.vertex_project is not None and not isinstance(
            self.vertex_project, VertexCacheProject
        ):
            raise ValueError("Vertex cache authority needs a verified project identity")


@dataclass(frozen=True, repr=False)
class CacheOffer:
    """Content-free cost reservation proposed to the durable host before HTTP.

    Attributes:
        operation_id: Unique create-operation identity, reused for result recording.
        request_id: Authorized gateway request identity for audit attribution.
        attempt_id: Actual admitted provider attempt identity for audit attribution.
        tenant_scope: Exact tenant scope from the host authority.
        credential_scope: Nonsecret provider credential generation from authority.
        account_key_fingerprint: Opaque account/key binding, never credential bytes.
        cache_key: SHA-256 digest binding tenant, credential, account and plan scope.
        resource_prefix: Exact allowed resource namespace from the frozen plan,
            either ``cachedContents/`` or a project/location-bound Vertex prefix.
        minimum_input_tokens: Verified eligibility threshold, not measured usage.
        maximum_input_tokens: Conservative serialized-byte plus framing bound.
        create_nano_usd: Rounded-up create-input cost at that bound, including
            zero only when the authority explicitly verifies free input creation.
        storage_nano_usd: Rounded-up full 300-second storage cost at that bound.
        reservation_nano_usd: Checked sum of create and storage cost.
        requested_at: Finite nonnegative Unix time used to form absolute expiry.
        expires_at: Absolute provider expiration, exactly requested_at plus 300.
            The create request must send this expireTime, not a relative TTL.
            Hosts retain uncertainty beyond this time for their clock-skew margin;
            this field is never a renewable lease or permission to steal a claim.
        ttl_seconds: Fixed 300-second pricing horizon, not caller configurable.
    """

    operation_id: str
    request_id: str
    attempt_id: str
    tenant_scope: str
    credential_scope: str
    account_key_fingerprint: str
    cache_key: str
    resource_prefix: str
    minimum_input_tokens: int
    maximum_input_tokens: int
    create_nano_usd: int
    storage_nano_usd: int
    reservation_nano_usd: int
    requested_at: float
    expires_at: float
    ttl_seconds: Literal[300] = field(default=300, init=False)

    def __post_init__(self) -> None:
        """Require bounded, content-free metadata and an exact absolute horizon."""
        for name, value in (
            ("operation_id", self.operation_id),
            ("request_id", self.request_id),
            ("attempt_id", self.attempt_id),
            ("tenant_scope", self.tenant_scope),
            ("credential_scope", self.credential_scope),
            ("account_key_fingerprint", self.account_key_fingerprint),
        ):
            _scope(value, name)
        _digest(self.cache_key, "cache_key")
        _resource_prefix(self.resource_prefix)
        _integer(self.minimum_input_tokens, "minimum_input_tokens", minimum=1)
        _integer(self.maximum_input_tokens, "maximum_input_tokens", minimum=1)
        if self.maximum_input_tokens < self.minimum_input_tokens:
            raise ValueError("cache upper bound is below the provider minimum; skip creation")
        _integer(self.create_nano_usd, "create_nano_usd")
        _integer(self.storage_nano_usd, "storage_nano_usd", minimum=1)
        _integer(self.reservation_nano_usd, "reservation_nano_usd", minimum=1)
        if self.reservation_nano_usd != self.create_nano_usd + self.storage_nano_usd:
            raise ValueError("cache reservation must equal create plus storage cost; requote it")
        _timestamp(self.requested_at, "requested_at")
        _timestamp(self.expires_at, "expires_at")
        if self.expires_at - self.requested_at != CACHE_TTL_SECONDS:
            raise ValueError("cache expiry must be exactly 300 seconds after the offer; requote it")


@dataclass(frozen=True, repr=False)
class CacheReady:
    """A durable resource already bound by the host to the offered cache key.

    Attributes:
        resource_name: Provider resource to reuse after exact namespace validation.
        expires_at: Provider-confirmed absolute Unix expiration.
        token_count: Provider-measured total input tokens, never a character estimate.
        state: Fixed discriminator identifying a ready resource.
    """

    resource_name: str
    expires_at: float
    token_count: int
    state: Literal["ready"] = field(default="ready", init=False)

    def __post_init__(self) -> None:
        """Reject malformed resource facts before they can become a reusable claim."""
        _scope(self.resource_name, "resource_name")
        _timestamp(self.expires_at, "expires_at")
        _integer(self.token_count, "token_count", minimum=1)


@dataclass(frozen=True, repr=False)
class CacheCreator:
    """Exclusive create permission returned only after a durable reservation commits.

    Attributes:
        operation_id: Must exactly match the offer's unique operation identity.
        expires_at: Must exactly match the offer's absolute provider expiration.
            Expiry alone never allows a worker to steal or repeat this operation.
        state: Fixed discriminator identifying the one permitted creator.
    """

    operation_id: str
    expires_at: float
    state: Literal["creator"] = field(default="creator", init=False)

    def __post_init__(self) -> None:
        """Require a usable operation identity and finite expiration."""
        _scope(self.operation_id, "operation_id")
        _timestamp(self.expires_at, "expires_at")


@dataclass(frozen=True, repr=False)
class CacheUnavailable:
    """A claim that grants no permission for provider cache creation or reuse.

    Attributes:
        reason: Denied policy/budget, another pending creator, or an unresolved
            outcome. Unknown retains its reservation through possible expiry.
        state: Fixed discriminator identifying a no-create decision.
    """

    reason: Literal["denied", "pending", "unknown"]
    state: Literal["unavailable"] = field(default="unavailable", init=False)

    def __post_init__(self) -> None:
        """Reject unrecognized outcomes rather than inventing retry permission."""
        if self.reason not in {"denied", "pending", "unknown"}:
            raise ValueError("cache unavailability must be denied, pending or unknown")


CacheClaim = CacheReady | CacheCreator | CacheUnavailable


@dataclass(frozen=True, repr=False)
class CacheResult:
    """Content-free observation recorded against one already-reserved operation.

    Attributes:
        operation_id: Identity allocated by the original durable claim.
        outcome: Ready means provider acceptance with validated resource facts;
            rejected requires positive evidence of no resource and no spend;
            unknown covers timeouts, transport failures and ambiguous responses.
        observed_at: Finite nonnegative Unix observation time, not retry time.
        resource_name: Optional observed resource, required for ready outcomes.
        total_tokens: Optional provider-measured input count, required for ready.
        expire_time: Optional absolute provider expiry, required for ready.
        http_status: Optional integer HTTP status, not sufficient by itself to
            prove a no-spend rejection. Transport failures carry None.
        create_time: Optional provider-reported absolute Unix creation time, default
            None when absent or unusable. A known value is finite and positive,
            no later than expire_time or observed_at. This interval is resource
            evidence, not an invoice formula or permission to release a hold.
    """

    operation_id: str
    outcome: Literal["ready", "unknown", "rejected"]
    observed_at: float
    resource_name: str | None = None
    total_tokens: int | None = None
    expire_time: float | None = None
    http_status: int | None = None
    create_time: float | None = None

    def __post_init__(self) -> None:
        """Require typed observation facts without inventing missing provider evidence."""
        _scope(self.operation_id, "operation_id")
        _timestamp(self.observed_at, "observed_at")
        if self.outcome not in {"ready", "unknown", "rejected"}:
            raise ValueError("cache result outcome must be ready, unknown or rejected")
        if self.resource_name is not None:
            _scope(self.resource_name, "resource_name")
        if self.total_tokens is not None:
            _integer(self.total_tokens, "total_tokens", minimum=1)
        if self.expire_time is not None:
            _timestamp(self.expire_time, "expire_time")
        if self.create_time is not None:
            _timestamp(self.create_time, "create_time")
            if (
                self.create_time <= 0
                or self.expire_time is None
                or self.create_time > self.expire_time
                or self.create_time > self.observed_at
            ):
                raise ValueError(
                    "create_time must be positive and no later than expiration or observation; "
                    "retain unknown billing facts instead"
                )
        if self.http_status is not None:
            _integer(self.http_status, "http_status", minimum=100)
            if self.http_status > 599:
                raise ValueError("cache HTTP status must be in 100..599; record unknown transport")


class ExplicitCacheHost(Protocol):
    """Host boundary for policy and durable, atomic cross-worker cache accounting.

    Implementations must bind authority to the authorized tenant and exact wire,
    including project, location, model, endpoint and credential generation. Cache
    state is isolated by the entire offer key, not by a content digest alone.
    Process-local locks and in-memory dictionaries do not satisfy this contract.
    """

    def authority(
        self,
        authorization: AuthorizationSnapshot,
        deployment: ExactModelDeployment,
        profile: GatewayWireProfile,
    ) -> GoogleCacheAuthority | None:
        """Return explicit verified authority, or None for zero allowance/default off.

        No provider call or reservation occurs here. The host must derive tenant
        and credential scopes itself, require verified exact-model create/storage
        prices, and reject authority when any binding or policy is unavailable.
        """
        ...

    def claim(self, offer: CacheOffer) -> CacheClaim:
        """Atomically commit a durable reservation before granting one creator.

        The same transaction revalidates customer daily budget and authority,
        reserves the full quoted input/storage amount and records the operation.
        Across workers and restarts, at most one caller receives creator for a
        cache key; retries of that operation receive pending or its known result,
        never another creator. Ready state must belong to the exact offered key.
        No HTTP create is permitted before this transaction succeeds.

        Pending and unknown records keep their full reservation and prohibit
        duplicate creation until external proof bounds possible expiry, including
        clock skew. They are not leases and must not be cleared merely because a
        worker disappeared or a local timer passed. Absolute expireTime is the
        only creation horizon authorized by this offer, never a relative TTL.
        """
        ...

    def record(self, result: CacheResult) -> None:
        """Durably and idempotently settle or quarantine the already-claimed result.

        The host correlates operation_id with its persisted offer. It must retain
        full reservation for unknown outcomes and recording failures, reject
        contradictory observations, and never overwrite a settled ready resource
        with an ambiguous retry. Rejected releases budget only with positive
        no-resource/no-spend evidence, not merely an HTTP error classification.
        Ready describes resource usability, not complete billing evidence. A host
        whose published schedule requires create_time must retain its full hold
        when that optional provider fact is None, never substitute local time.
        """
        ...


def reserve_cache_cost(
    maximum_input_tokens: int, authority: GoogleCacheAuthority
) -> tuple[int, int, int]:
    """Quote separately rounded-up create and full-horizon storage costs.

    Args:
        maximum_input_tokens: Conservative positive token upper bound, not usage.
        authority: Verified create and per-token-hour storage rates.

    Returns:
        Create cost, storage cost and their sum, all integer nano-USD.

    Raises:
        ValueError: Inputs or any resulting cost exceed signed 64-bit bounds.
    """
    _integer(maximum_input_tokens, "maximum_input_tokens", minimum=1)
    create = _ceil_div(
        maximum_input_tokens * authority.create_input_nano_usd_per_million, 1_000_000
    )
    storage = _ceil_div(
        maximum_input_tokens
        * authority.storage_nano_usd_per_million_token_hour
        * CACHE_TTL_SECONDS,
        1_000_000 * 3600,
    )
    total = create + storage
    _integer(create, "create cost")
    for name, value in (("storage cost", storage), ("total cost", total)):
        _integer(value, name, minimum=1)
    return create, storage, total


def cache_scope_key(
    authority: GoogleCacheAuthority,
    account_key_fingerprint: str,
    plan_scope_digest: str,
) -> str:
    """Digest unambiguous tenant, credential, account and frozen-plan bindings.

    Args:
        authority: Host-derived tenant and credential-generation scope.
        account_key_fingerprint: Nonsecret host binding, never a raw API key.
        plan_scope_digest: SHA-256 of the provider plan's exact model, project,
            location, endpoint, resource namespace and complete cached prefix.
            Hosts must verify this binding before accepting an offer.

    Returns:
        A lowercase SHA-256 cache key without plaintext prefix or credentials.
    """
    _scope(account_key_fingerprint, "account_key_fingerprint")
    _digest(plan_scope_digest, "plan_scope_digest")
    material = json.dumps(
        [
            authority.tenant_scope,
            authority.credential_scope,
            account_key_fingerprint,
            plan_scope_digest,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def prepare_cache_offer(
    *,
    authority: GoogleCacheAuthority | None,
    operation_id: str,
    request_id: str,
    attempt_id: str,
    account_key_fingerprint: str,
    plan_scope_digest: str,
    resource_prefix: str,
    prefix_bytes: int,
    framing_tokens: int,
    requested_at: float,
) -> CacheOffer | None:
    """Build a content-free bounded offer without reserving funds or calling Google.

    Args:
        authority: Explicit host permission; None makes this a strict no-op.
        operation_id: Unique operation identity to reserve durably before HTTP.
        request_id: Authorized gateway request identity.
        attempt_id: Already-admitted provider attempt identity.
        account_key_fingerprint: Opaque account/key binding, never secret bytes.
        plan_scope_digest: Exact frozen provider-plan digest used in the cache key.
        resource_prefix: Exact resource namespace frozen by the provider plan.
        prefix_bytes: UTF-8 serialized byte count of every cached field, including
            system content and tools. Byte bounds apply only to supported text
            plans, never opaque media or provider-side file expansions.
        framing_tokens: Conservative provider-adapter bound for hidden framing.
        requested_at: Current Unix time, used to set absolute 300-second expiry.

    Returns:
        A quoted offer, or None when disabled, oversized, empty, or definitely
        below the provider minimum. A bound above the minimum does not establish
        eligibility; cache creation may still be refused without creating a cache.

    Raises:
        ValueError: Metadata is malformed or token/cost arithmetic exceeds int64.
    """
    if authority is None:
        return None
    _integer(prefix_bytes, "prefix_bytes")
    _integer(framing_tokens, "framing_tokens")
    if prefix_bytes == 0 or prefix_bytes > authority.maximum_prefix_bytes:
        return None
    maximum_input_tokens = prefix_bytes + framing_tokens
    _integer(maximum_input_tokens, "maximum_input_tokens", minimum=1)
    if maximum_input_tokens < authority.minimum_input_tokens:
        return None
    _timestamp(requested_at, "requested_at")
    create, storage, total = reserve_cache_cost(maximum_input_tokens, authority)
    return CacheOffer(
        operation_id=operation_id,
        request_id=request_id,
        attempt_id=attempt_id,
        tenant_scope=authority.tenant_scope,
        credential_scope=authority.credential_scope,
        account_key_fingerprint=account_key_fingerprint,
        cache_key=cache_scope_key(authority, account_key_fingerprint, plan_scope_digest),
        resource_prefix=resource_prefix,
        minimum_input_tokens=authority.minimum_input_tokens,
        maximum_input_tokens=maximum_input_tokens,
        create_nano_usd=create,
        storage_nano_usd=storage,
        reservation_nano_usd=total,
        requested_at=requested_at,
        expires_at=requested_at + CACHE_TTL_SECONDS,
    )


def validate_cache_claim(offer: CacheOffer, claim: CacheClaim, now: float) -> CacheClaim:
    """Validate host permission before a caller creates or reuses any resource.

    Args:
        offer: Exact proposal passed to the durable host.
        claim: Host decision, already durably committed if it grants creation.
        now: Current Unix time at consumption, not offer or lease creation time.

    Returns:
        The unchanged validated claim.

    Raises:
        ValueError: Binding, namespace, measured tokens or safe expiry is invalid.
            Invalid claims grant no HTTP permission and never release reservation.
    """
    _timestamp(now, "now")
    _validate_claim_binding(offer, claim)
    if isinstance(claim, CacheUnavailable):
        return claim
    if claim.expires_at <= now + CACHE_EXPIRY_SAFETY_SECONDS:
        raise ValueError("cache permission expires too soon; skip explicit caching")
    if now < offer.requested_at:
        raise ValueError("cache clock precedes the authorized offer; skip explicit caching")
    return claim


def validate_cache_result(offer: CacheOffer, result: CacheResult) -> CacheResult:
    """Validate a create observation without converting uncertainty to no-spend.

    Args:
        offer: Exact durably reserved create proposal.
        result: Provider observation for that operation.

    Returns:
        The unchanged result. Late but valid acceptance remains recordable even
        after expiration; readiness for reuse is separately checked on claims.

    Raises:
        ValueError: Facts contradict the reserved operation. A caller must record
            an unknown outcome on malformed acceptance, preserving reservation.
    """
    if result.operation_id != offer.operation_id:
        raise ValueError("cache result operation does not match its offer; retain reservation")
    if result.observed_at < offer.requested_at:
        raise ValueError("cache observation precedes its offer; retain reservation")
    if result.resource_name is not None:
        _resource(offer, result.resource_name)
    if result.total_tokens is not None:
        _measured_tokens(offer, result.total_tokens)
    if result.expire_time is not None and not (
        offer.requested_at < result.expire_time <= offer.expires_at
    ):
        raise ValueError("cache result exceeds its absolute expiry horizon; record unknown")
    if result.outcome == "ready":
        if (
            result.resource_name is None
            or result.total_tokens is None
            or result.expire_time is None
            or (result.http_status is not None and not 200 <= result.http_status < 300)
        ):
            raise ValueError("ready cache result requires complete accepted resource facts")
    elif result.outcome == "rejected":
        if (
            result.resource_name is not None
            or result.total_tokens is not None
            or result.expire_time is not None
            or result.create_time is not None
            or result.http_status is None
            or not 400 <= result.http_status < 500
            or result.http_status in {408, 409}
        ):
            raise ValueError("ambiguous cache rejection must remain unknown with reserved budget")
    return result


def claim_cache(
    host: ExplicitCacheHost,
    offer: CacheOffer | None,
    *,
    clock: Callable[[], float] = time.time,
) -> CacheClaim:
    """Claim durably and check a fresh post-host clock before permitting HTTP.

    Args:
        host: Durable reservation and policy owner.
        offer: Prepared proposal, or None to perform no host or clock action.
        clock: Unix clock sampled before and after the potentially blocking claim.
            Defaults to time.time, never a timestamp captured before host latency.

    Returns:
        A validated claim; disabled or already-expired offers return denied without
        calling the host. A ready or creator claim that expires during host work
        returns unknown without recording, releasing funds, retrying or recreating.

    Raises:
        ValueError: The clock is malformed or moves backward, or the host returns
            invalid binding facts. Such failures never release a reservation.
    """
    if offer is None:
        return CacheUnavailable("denied")
    started_at = clock()
    _timestamp(started_at, "claim start time")
    if started_at < offer.requested_at:
        raise ValueError("cache clock precedes the authorized offer; skip explicit caching")
    if started_at + CACHE_EXPIRY_SAFETY_SECONDS >= offer.expires_at:
        return CacheUnavailable("denied")
    claim = host.claim(offer)
    observed_at = clock()
    _timestamp(observed_at, "claim observation time")
    if observed_at < started_at:
        raise ValueError("cache clock moved backward during claim; skip explicit caching")
    _validate_claim_binding(offer, claim)
    if isinstance(claim, CacheUnavailable):
        return claim
    if claim.expires_at <= observed_at + CACHE_EXPIRY_SAFETY_SECONDS:
        return CacheUnavailable("unknown")
    return claim


def finish_cache(host: ExplicitCacheHost, offer: CacheOffer, result: CacheResult) -> CacheResult:
    """Validate and record an observation without releasing uncertain reservations.

    Args:
        host: Durable owner of the original operation and reservation.
        offer: The exact originally reserved proposal.
        result: Observation to record idempotently against that operation.

    Returns:
        The validated observation after recording succeeds. Exceptions propagate;
        a recording failure must leave the host's pending reservation in place.
    """
    validated = validate_cache_result(offer, result)
    host.record(validated)
    return validated


def _validate_claim_binding(offer: CacheOffer, claim: CacheClaim) -> None:
    """Check immutable host binding facts independently from resource freshness."""
    if isinstance(claim, CacheUnavailable):
        return
    if isinstance(claim, CacheCreator):
        if claim.operation_id != offer.operation_id or claim.expires_at != offer.expires_at:
            raise ValueError("cache creator does not match the durable offer; skip creation")
    elif isinstance(claim, CacheReady):
        _resource(offer, claim.resource_name)
        _measured_tokens(offer, claim.token_count)
        if claim.expires_at > offer.expires_at:
            raise ValueError("cache resource exceeds the authorized expiry; do not reuse it")
    else:
        raise ValueError("cache host returned an unknown claim type; skip explicit caching")


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    """Require a strict integer within the nonnegative signed 64-bit range."""
    if type(value) is not int or not minimum <= value <= _MAX_INT64:
        raise ValueError(f"{name} must be an integer in {minimum}..{_MAX_INT64}; correct the offer")


def _timestamp(value: float, name: str) -> None:
    """Require a finite nonnegative Unix timestamp, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative Unix timestamp")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative Unix timestamp")


def _scope(value: str, name: str) -> None:
    """Require bounded printable metadata without accepting empty binding scopes."""
    if not isinstance(value, str) or not value or len(value) > 1024 or not value.isprintable():
        raise ValueError(f"{name} must be nonempty printable metadata of at most 1024 characters")
    if value.strip() != value:
        raise ValueError(f"{name} must not have surrounding whitespace; use the exact binding")


def _digest(value: str, name: str) -> None:
    """Require a complete lowercase SHA-256 digest, not plaintext prefix content."""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest; bind the frozen plan")


def _resource_prefix(value: str) -> None:
    """Accept only exact Gemini or project/location-bound Vertex cache namespaces."""
    if not isinstance(value, str) or _RESOURCE_PREFIX.fullmatch(value) is None:
        raise ValueError("cache resource prefix must be an exact Gemini or Vertex namespace")


def _resource(offer: CacheOffer, resource_name: str) -> None:
    """Reject foreign namespaces, traversal, extra segments and URI query syntax."""
    if (
        not resource_name.startswith(offer.resource_prefix)
        or _RESOURCE_ID.fullmatch(resource_name[len(offer.resource_prefix) :]) is None
    ):
        raise ValueError("cache resource is outside the authorized namespace; do not reuse it")


def _measured_tokens(offer: CacheOffer, token_count: int) -> None:
    """Require provider-measured cache size within verified minimum and quoted bound."""
    if not offer.minimum_input_tokens <= token_count <= offer.maximum_input_tokens:
        raise ValueError("cache token count is outside the reserved bounds; record unknown")


def _ceil_div(numerator: int, denominator: int) -> int:
    """Round an arbitrary-precision integer ratio upward without float conversion."""
    return (numerator + denominator - 1) // denominator

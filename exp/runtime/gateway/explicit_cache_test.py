"""Provider-free coverage for explicit-cache authority, quotes and durable contracts."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.explicit_cache import (
    CACHE_EXPIRY_SAFETY_SECONDS,
    CACHE_TTL_SECONDS,
    CacheClaim,
    CacheCreator,
    CacheOffer,
    CacheReady,
    CacheResult,
    CacheUnavailable,
    GoogleCacheAuthority,
    cache_scope_key,
    claim_cache,
    finish_cache,
    prepare_cache_offer,
    reserve_cache_cost,
    validate_cache_claim,
    validate_cache_result,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.google_cache import VertexCacheProject

_MAX_INT64 = (1 << 63) - 1
_PLAN_DIGEST = hashlib.sha256(b"test-only exact model and prefix plan").hexdigest()
_VERTEX_PREFIX = "projects/project-123/locations/us-central1/cachedContents/"


class AtomicHost:
    """Test-only atomic host, not a durable production store.

    Attributes:
        authority_value: Policy returned to test callers, or None when disabled.
        budget: Test tenant's total daily allocation, in nano-USD.
        reserved: Full pending/unknown reservations retained by this fake.
        offers: Exact original offer by cache key.
        results: Recorded result by original operation identity.
        claim_calls: Number of atomic claim attempts.
        record_calls: Number of successful durable-result simulations.
        lock: Cross-thread exclusion standing in for a host database transaction.
    """

    def __init__(self, authority: GoogleCacheAuthority | None, budget: int = _MAX_INT64) -> None:
        """Initialize isolated fake state without opening a database or provider."""
        self.authority_value = authority
        self.budget = budget
        self.reserved = 0
        self.offers: dict[str, CacheOffer] = {}
        self.results: dict[str, CacheResult] = {}
        self.claim_calls = 0
        self.record_calls = 0
        self.lock = threading.Lock()

    def authority(
        self,
        authorization: AuthorizationSnapshot,
        deployment: ExactModelDeployment,
        profile: GatewayWireProfile,
    ) -> GoogleCacheAuthority | None:
        """Return the test's configured authority without inspecting credentials."""
        return self.authority_value

    def claim(self, offer: CacheOffer) -> CacheClaim:
        """Reserve atomically and never steal expired pending or unknown operations."""
        with self.lock:
            self.claim_calls += 1
            existing = self.offers.get(offer.cache_key)
            if existing is not None:
                result = self.results.get(existing.operation_id)
                if result is None:
                    return CacheUnavailable("pending")
                if result.outcome == "unknown":
                    return CacheUnavailable("unknown")
                if result.outcome == "rejected":
                    return CacheUnavailable("denied")
                assert result.resource_name is not None
                assert result.expire_time is not None
                assert result.total_tokens is not None
                return CacheReady(result.resource_name, result.expire_time, result.total_tokens)
            authority = self.authority_value
            if (
                authority is None
                or offer.tenant_scope != authority.tenant_scope
                or offer.credential_scope != authority.credential_scope
                or self.reserved + offer.reservation_nano_usd > self.budget
            ):
                return CacheUnavailable("denied")
            self.offers[offer.cache_key] = offer
            self.reserved += offer.reservation_nano_usd
            return CacheCreator(offer.operation_id, offer.expires_at)

    def record(self, result: CacheResult) -> None:
        """Record idempotently while retaining pending and unknown cost reservations."""
        with self.lock:
            original = next(
                offer for offer in self.offers.values() if offer.operation_id == result.operation_id
            )
            validate_cache_result(original, result)
            existing = self.results.get(result.operation_id)
            if existing is not None:
                if existing != result:
                    raise ValueError("contradictory observation")
                return
            self.results[result.operation_id] = result
            self.record_calls += 1
            if result.outcome == "rejected":
                self.reserved -= original.reservation_nano_usd


def _authority() -> GoogleCacheAuthority:
    """Return verified synthetic nonsecret authority for one test tenant/account."""
    return GoogleCacheAuthority(
        tenant_scope="tenant-one",
        credential_scope="credential-generation-one",
        minimum_input_tokens=1024,
        create_input_nano_usd_per_million=1_000_000_000,
        storage_nano_usd_per_million_token_hour=4_500_000_000,
    )


def _offer(*, resource_prefix: str = "cachedContents/") -> CacheOffer:
    """Create a fully bounded test proposal without reserving anything."""
    offer = prepare_cache_offer(
        authority=_authority(),
        operation_id="operation-one",
        request_id="request-one",
        attempt_id="attempt-one",
        account_key_fingerprint="account-key-one",
        plan_scope_digest=_PLAN_DIGEST,
        resource_prefix=resource_prefix,
        prefix_bytes=2000,
        framing_tokens=48,
        requested_at=1000.0,
    )
    assert offer is not None
    return offer


def _ready(offer: CacheOffer) -> CacheResult:
    """Construct complete provider acceptance facts inside the reserved bounds."""
    return CacheResult(
        operation_id=offer.operation_id,
        outcome="ready",
        observed_at=offer.requested_at + 1,
        resource_name=offer.resource_prefix + "cache_123-abc",
        total_tokens=1536,
        expire_time=offer.expires_at,
        http_status=200,
    )


def test_full_ttl_quote_separately_rounds_create_and_storage_up() -> None:
    """The entire 300-second liability is reserved with no fractional truncation."""
    authority = _authority()
    assert reserve_cache_cost(1_000_000, authority) == (
        1_000_000_000,
        375_000_000,
        1_375_000_000,
    )
    smallest_rates = replace(
        authority,
        create_input_nano_usd_per_million=1,
        storage_nano_usd_per_million_token_hour=1,
    )
    assert reserve_cache_cost(1, smallest_rates) == (1, 1, 2)
    offer = _offer()
    assert offer.maximum_input_tokens == 2048
    assert offer.reservation_nano_usd == offer.create_nano_usd + offer.storage_nano_usd
    assert offer.expires_at == offer.requested_at + CACHE_TTL_SECONDS
    assert offer.ttl_seconds == 300


def test_cost_arithmetic_preserves_int64_precision_and_rejects_total_overflow() -> None:
    """Unbounded Python intermediates preserve exact quotes but never leak int64 totals."""
    authority = replace(
        _authority(),
        create_input_nano_usd_per_million=_MAX_INT64 - 1,
        storage_nano_usd_per_million_token_hour=1,
    )
    assert reserve_cache_cost(1_000_000, authority) == (_MAX_INT64 - 1, 1, _MAX_INT64)
    with pytest.raises(ValueError, match="total cost"):
        reserve_cache_cost(
            1_000_000, replace(authority, create_input_nano_usd_per_million=_MAX_INT64)
        )
    with pytest.raises(ValueError, match="create cost"):
        reserve_cache_cost(_MAX_INT64, _authority())


@pytest.mark.parametrize("count", [True, False, -1, 0, 1.5, _MAX_INT64 + 1])
def test_token_quote_rejects_nonintegers_and_unbounded_counts(count: int) -> None:
    """No boolean, float, negative or out-of-range token count becomes a quote."""
    with pytest.raises(ValueError, match="maximum_input_tokens"):
        reserve_cache_cost(count, _authority())


@pytest.mark.parametrize("rate", [None, -1, True, False, 0.0, 0.5, _MAX_INT64 + 1])
def test_unverified_or_unbounded_price_rates_fail_closed(rate: int) -> None:
    """Unknown, boolean, negative and noninteger rates never become free creation."""
    with pytest.raises(ValueError, match="create input rate"):
        replace(_authority(), create_input_nano_usd_per_million=rate)
    with pytest.raises(ValueError, match="storage rate"):
        replace(_authority(), storage_nano_usd_per_million_token_hour=rate)


def test_zero_storage_rate_remains_invalid() -> None:
    """Verified zero input creation still requires a strictly positive storage rate."""
    with pytest.raises(ValueError, match="storage rate"):
        replace(_authority(), storage_nano_usd_per_million_token_hour=0)


def test_verified_zero_input_creation_reserves_full_storage_cost() -> None:
    """An explicit verified integer zero is not confused with absent pricing authority."""
    authority = replace(_authority(), create_input_nano_usd_per_million=0)
    assert reserve_cache_cost(1_000_000, authority) == (0, 375_000_000, 375_000_000)
    offer = prepare_cache_offer(
        authority=authority,
        operation_id="free-create-operation",
        request_id="request",
        attempt_id="attempt",
        account_key_fingerprint="account",
        plan_scope_digest=_PLAN_DIGEST,
        resource_prefix="cachedContents/",
        prefix_bytes=2000,
        framing_tokens=48,
        requested_at=1000,
    )
    assert offer is not None
    assert offer.create_nano_usd == 0
    assert offer.storage_nano_usd == offer.reservation_nano_usd == 768_000
    host = AtomicHost(authority, budget=offer.reservation_nano_usd)
    assert isinstance(claim_cache(host, offer, clock=lambda: 1000), CacheCreator)
    assert host.reserved == offer.storage_nano_usd


def test_zero_create_quote_preserves_storage_ceiling_and_int64_bound() -> None:
    """Free input creation never truncates fractional storage or bypasses its overflow check."""
    authority = replace(
        _authority(),
        create_input_nano_usd_per_million=0,
        storage_nano_usd_per_million_token_hour=1,
    )
    assert reserve_cache_cost(1, authority) == (0, 1, 1)
    assert reserve_cache_cost(12_000_000, authority) == (0, 1, 1)
    assert reserve_cache_cost(12_000_001, authority) == (0, 2, 2)
    largest_rate = replace(authority, storage_nano_usd_per_million_token_hour=_MAX_INT64)
    assert reserve_cache_cost(12_000_000, largest_rate) == (0, _MAX_INT64, _MAX_INT64)
    with pytest.raises(ValueError, match="storage cost"):
        reserve_cache_cost(12_000_001, largest_rate)


@pytest.mark.parametrize("cost", [None, False, -1])
def test_offer_zero_create_cost_requires_an_explicit_nonnegative_integer(cost: int) -> None:
    """The persisted quote rejects absent, boolean and negative zero-cost lookalikes."""
    with pytest.raises(ValueError, match="create_nano_usd"):
        replace(_offer(), create_nano_usd=cost)


def test_token_bound_overflow_is_rejected_before_cost_or_claim() -> None:
    """Serialized bytes plus hidden framing cannot wrap a signed token bound."""
    with pytest.raises(ValueError, match="maximum_input_tokens"):
        prepare_cache_offer(
            authority=replace(_authority(), maximum_prefix_bytes=_MAX_INT64),
            operation_id="operation",
            request_id="request",
            attempt_id="attempt",
            account_key_fingerprint="account",
            plan_scope_digest=_PLAN_DIGEST,
            resource_prefix="cachedContents/",
            prefix_bytes=_MAX_INT64,
            framing_tokens=1,
            requested_at=1000,
        )


@pytest.mark.parametrize("prefix_bytes,framing_tokens", [(0, 5000), (100, 10), (10_485_761, 0)])
def test_empty_definitely_short_or_oversized_prefix_is_no_offer(
    prefix_bytes: int, framing_tokens: int
) -> None:
    """A safe upper bound may disprove eligibility, but cannot prove the minimum."""
    assert (
        prepare_cache_offer(
            authority=_authority(),
            operation_id="operation",
            request_id="request",
            attempt_id="attempt",
            account_key_fingerprint="account",
            plan_scope_digest=_PLAN_DIGEST,
            resource_prefix="cachedContents/",
            prefix_bytes=prefix_bytes,
            framing_tokens=framing_tokens,
            requested_at=1000,
        )
        is None
    )


def test_large_byte_bound_still_allows_provider_minimum_refusal() -> None:
    """Pricing an upper bound does not fabricate an eligible measured token count."""
    host = AtomicHost(_authority())
    offer = _offer()
    assert isinstance(claim_cache(host, offer, clock=lambda: 1000), CacheCreator)
    result = CacheResult(offer.operation_id, "rejected", 1001, http_status=400)
    assert finish_cache(host, offer, result) == result
    assert host.reserved == 0


def test_zero_authority_is_strict_noop_without_claim_or_spend() -> None:
    """Disabled authority cannot create a reservation, even with otherwise bad input."""
    host = AtomicHost(None)
    offer = prepare_cache_offer(
        authority=None,
        operation_id="",
        request_id="",
        attempt_id="",
        account_key_fingerprint="",
        plan_scope_digest="",
        resource_prefix="",
        prefix_bytes=-1,
        framing_tokens=-1,
        requested_at=float("nan"),
    )
    claim = claim_cache(host, offer, clock=lambda: float("nan"))
    assert claim == CacheUnavailable("denied")
    assert host.claim_calls == host.record_calls == host.reserved == 0
    assert host.offers == {}


def test_scope_key_isolated_by_tenant_credential_account_and_plan() -> None:
    """Only exactly identical frozen bindings share a reusable resource identity."""
    authority = _authority()
    key = cache_scope_key(authority, "account-one", _PLAN_DIGEST)
    alternatives = {
        cache_scope_key(replace(authority, tenant_scope="tenant-two"), "account-one", _PLAN_DIGEST),
        cache_scope_key(
            replace(authority, credential_scope="generation-two"), "account-one", _PLAN_DIGEST
        ),
        cache_scope_key(authority, "account-two", _PLAN_DIGEST),
        cache_scope_key(authority, "account-one", "b" * 64),
    }
    assert key not in alternatives
    assert len(alternatives) == 4
    assert key == cache_scope_key(authority, "account-one", _PLAN_DIGEST)
    assert len(key) == 64
    assert "tenant" not in key and "account" not in key


def test_key_material_uses_unambiguous_field_encoding() -> None:
    """Delimiters embedded in one nonsecret field cannot collide with another field."""
    left = replace(_authority(), tenant_scope='tenant","generation', credential_scope="tail")
    right = replace(_authority(), tenant_scope="tenant", credential_scope='generation","tail')
    assert cache_scope_key(left, "account", _PLAN_DIGEST) != cache_scope_key(
        right, "account", _PLAN_DIGEST
    )


@pytest.mark.parametrize("resource_prefix", ["cachedContents/", _VERTEX_PREFIX])
def test_valid_resources_are_exactly_bound_to_plan_namespace(resource_prefix: str) -> None:
    """Both Gemini and Vertex return only a single safe ID below their plan namespace."""
    offer = _offer(resource_prefix=resource_prefix)
    result = _ready(offer)
    assert validate_cache_result(offer, result) == result
    claim = CacheReady(resource_prefix + "cache_123-abc", offer.expires_at, 1536)
    assert validate_cache_claim(offer, claim, 1001) == claim


@pytest.mark.parametrize(
    "name",
    [
        "cachedContents/",
        "cachedContents/../secret",
        "cachedContents/%2e%2e",
        "cachedContents/cache/child",
        "cachedContents/cache?token=secret",
        "cachedContents/cache#fragment",
        "cachedContents/cache\\child",
        "https://provider.invalid/cachedContents/cache",
        "projects/other/locations/us-central1/cachedContents/cache",
        "projects/project-123/locations/europe-west1/cachedContents/cache",
    ],
)
def test_resource_validation_rejects_foreign_and_injected_names(name: str) -> None:
    """Resource strings never become arbitrary paths, queries or cross-project cache handles."""
    for prefix in ("cachedContents/", _VERTEX_PREFIX):
        offer = _offer(resource_prefix=prefix)
        with pytest.raises(ValueError, match="namespace"):
            validate_cache_claim(offer, CacheReady(name, 1300, 1536), 1001)
        with pytest.raises(ValueError, match="namespace"):
            validate_cache_result(offer, replace(_ready(offer), resource_name=name))


@pytest.mark.parametrize(
    "prefix",
    ["", "/cachedContents/", "cachedContents", "cachedContents/../", "projects/p/locations/l/"],
)
def test_resource_namespace_itself_must_be_plan_safe(prefix: str) -> None:
    """An unvalidated namespace cannot bless an attacker-selected resource prefix."""
    with pytest.raises(ValueError, match="prefix"):
        _offer(resource_prefix=prefix)


def test_claim_expiry_identity_and_measured_size_are_checked() -> None:
    """A host decision is consumed only within the exact operation and quoted bounds."""
    offer = _offer()
    invalid: tuple[CacheClaim, ...] = (
        CacheCreator("other-operation", offer.expires_at),
        CacheCreator(offer.operation_id, offer.expires_at + 1),
        CacheReady("cachedContents/cache", offer.expires_at + 1, 1536),
        CacheReady("cachedContents/cache", 1001 + CACHE_EXPIRY_SAFETY_SECONDS, 1536),
        CacheReady("cachedContents/cache", 1300, 1023),
        CacheReady("cachedContents/cache", 1300, 2049),
    )
    for claim in invalid:
        with pytest.raises(ValueError):
            validate_cache_claim(offer, claim, 1001)
    with pytest.raises(ValueError, match="clock"):
        validate_cache_claim(offer, CacheCreator(offer.operation_id, offer.expires_at), 999)


def test_expired_offer_does_not_reserve_new_money() -> None:
    """Expired authorization is declined before invoking the durable host."""
    host = AtomicHost(_authority())
    assert claim_cache(host, _offer(), clock=lambda: 1295) == CacheUnavailable("denied")
    assert host.claim_calls == host.reserved == 0


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("after_claim", [1295.0, 1301.0])
def test_host_latency_cannot_authorize_expired_creation_or_reuse(
    ready: bool, after_claim: float
) -> None:
    """A fresh post-transaction clock disables expired claims without mutating accounting."""
    current_time = 1000.0

    class DelayedHost(AtomicHost):
        """Advance the injected clock while committing or reading a fake host claim."""

        def claim(self, offer: CacheOffer) -> CacheClaim:
            """Simulate database latency before returning the structurally valid claim."""
            nonlocal current_time
            claim = super().claim(offer)
            current_time = after_claim
            return claim

    host = DelayedHost(_authority())
    offer = _offer()
    if ready:
        AtomicHost.claim(host, offer)
        finish_cache(host, offer, _ready(offer))
    existing_results = host.results.copy()
    previous_records = host.record_calls
    assert claim_cache(host, offer, clock=lambda: current_time) == CacheUnavailable("unknown")
    assert current_time == after_claim
    assert host.claim_calls == (2 if ready else 1)
    assert host.record_calls == previous_records
    assert host.results == existing_results
    assert len(host.offers) == 1
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize(
    "response",
    [
        CacheCreator("wrong-operation", 1300),
        CacheReady("cachedContents/cache/child", 1300, 1536),
        CacheReady("cachedContents/cache", 1301, 1536),
        CacheReady("cachedContents/cache", 1300, 2049),
    ],
)
def test_expired_host_response_still_rejects_invalid_binding(response: CacheClaim) -> None:
    """Ordinary latency fallback cannot hide invalid operation, scope, expiry or size facts."""
    current_time = 1000.0

    class InvalidDelayedHost(AtomicHost):
        """Return invalid host facts after advancing beyond the authorized horizon."""

        def claim(self, offer: CacheOffer) -> CacheClaim:
            """Keep the real reservation, then inject a malformed host reply."""
            nonlocal current_time
            super().claim(offer)
            current_time = 1400.0
            return response

    host = InvalidDelayedHost(_authority())
    offer = _offer()
    with pytest.raises(ValueError):
        claim_cache(host, offer, clock=lambda: current_time)
    assert host.claim_calls == 1
    assert host.record_calls == 0
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize("after_claim", [999.0, float("nan"), float("inf"), -1.0, True])
def test_invalid_post_host_clock_retains_reservation(after_claim: float) -> None:
    """A backward or malformed clock never authorizes creation after host accounting."""
    ticks = iter((1000.0, after_claim))
    host = AtomicHost(_authority())
    offer = _offer()
    with pytest.raises(ValueError, match="clock|timestamp"):
        claim_cache(host, offer, clock=lambda: next(ticks))
    assert host.claim_calls == 1
    assert host.record_calls == 0
    assert host.reserved == offer.reservation_nano_usd


def test_clock_before_offer_is_invalid_without_host_action() -> None:
    """An invalid initial clock cannot produce a durable create reservation."""
    host = AtomicHost(_authority())
    with pytest.raises(ValueError, match="clock"):
        claim_cache(host, _offer(), clock=lambda: 999)
    assert host.claim_calls == host.reserved == 0


def test_still_fresh_host_response_is_usable_after_latency() -> None:
    """Post-host sampling does not disable claims that retain more than the safety margin."""
    ticks = iter((1000.0, 1294.0))
    host = AtomicHost(_authority())
    offer = _offer()
    assert claim_cache(host, offer, clock=lambda: next(ticks)) == CacheCreator(
        offer.operation_id, offer.expires_at
    )
    assert host.claim_calls == 1
    assert host.reserved == offer.reservation_nano_usd


def test_atomic_host_grants_exactly_one_creator_under_concurrent_workers() -> None:
    """The host commits one full reservation before one simulated worker may create."""
    host = AtomicHost(_authority())
    offer = _offer()
    barrier = threading.Barrier(8)

    def worker(index: int) -> CacheClaim:
        """Race one separately identified operation against the same exact cache key."""
        barrier.wait(timeout=5)
        return claim_cache(
            host, replace(offer, operation_id=f"operation-{index}"), clock=lambda: 1001
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(worker, range(8)))
    assert sum(isinstance(claim, CacheCreator) for claim in claims) == 1
    assert claims.count(CacheUnavailable("pending")) == 7
    assert host.reserved == offer.reservation_nano_usd
    assert len(host.offers) == 1
    assert host.claim_calls == 8


def test_host_budget_reservation_is_not_a_worker_scalar_allowance() -> None:
    """A separate prefix cannot bypass the tenant budget already reserved by another worker."""
    offer = _offer()
    host = AtomicHost(_authority(), budget=offer.reservation_nano_usd)
    assert isinstance(claim_cache(host, offer, clock=lambda: 1000), CacheCreator)
    second = replace(offer, operation_id="second", cache_key="b" * 64)
    assert claim_cache(host, second, clock=lambda: 1001) == CacheUnavailable("denied")
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize("record_unknown", [False, True])
def test_pending_and_unknown_are_never_stolen_on_expiry(record_unknown: bool) -> None:
    """A new offer after the absolute horizon cannot erase unresolved host evidence."""
    offer = _offer()
    host = AtomicHost(_authority())
    assert isinstance(claim_cache(host, offer, clock=lambda: 1000), CacheCreator)
    assert claim_cache(host, offer, clock=lambda: 1001) == CacheUnavailable("pending")
    if record_unknown:
        finish_cache(host, offer, CacheResult(offer.operation_id, "unknown", 1002))
    later = replace(offer, operation_id="later", requested_at=2000, expires_at=2300)
    assert claim_cache(host, later, clock=lambda: 2000) == CacheUnavailable(
        "unknown" if record_unknown else "pending"
    )
    assert host.reserved == offer.reservation_nano_usd
    assert host.offers[offer.cache_key].operation_id == offer.operation_id


def test_record_ready_is_idempotent_and_reused_without_new_reservation() -> None:
    """One durable resource can serve a later operation in exactly its original scope."""
    host = AtomicHost(_authority())
    offer = _offer()
    assert isinstance(claim_cache(host, offer, clock=lambda: 1000), CacheCreator)
    result = _ready(offer)
    finish_cache(host, offer, result)
    finish_cache(host, offer, result)
    second = replace(offer, operation_id="later", requested_at=1010, expires_at=1310)
    assert claim_cache(host, second, clock=lambda: 1010) == CacheReady(
        "cachedContents/cache_123-abc", 1300, 1536
    )
    assert host.record_calls == 1
    assert host.reserved == offer.reservation_nano_usd
    with pytest.raises(ValueError, match="contradictory"):
        finish_cache(host, offer, CacheResult(offer.operation_id, "unknown", 1011))


def test_late_acceptance_is_still_recorded_but_never_reused() -> None:
    """Accounting retains positive provider evidence even when the cache already expired."""
    offer = _offer()
    late = replace(_ready(offer), observed_at=1400)
    assert validate_cache_result(offer, late) == late
    with pytest.raises(ValueError, match="expires too soon"):
        validate_cache_claim(offer, CacheReady("cachedContents/cache", 1300, 1536), 1400)


@pytest.mark.parametrize("status", [None, 200, 302, 408, 409, 500, 503])
def test_ambiguous_rejections_cannot_release_reserved_budget(status: int | None) -> None:
    """Transport loss, server errors and malformed acceptance remain unknown, not no-spend."""
    offer = _offer()
    host = AtomicHost(_authority())
    claim_cache(host, offer, clock=lambda: 1000)
    with pytest.raises(ValueError, match="ambiguous"):
        finish_cache(
            host, offer, CacheResult(offer.operation_id, "rejected", 1001, http_status=status)
        )
    assert host.record_calls == 0
    assert host.reserved == offer.reservation_nano_usd
    unknown = CacheResult(offer.operation_id, "unknown", 1001, http_status=status)
    finish_cache(host, offer, unknown)
    assert host.reserved == offer.reservation_nano_usd


def test_malformed_acceptance_cannot_publish_ready_or_release_reservation() -> None:
    """Bad provider facts are rejected locally and a clean unknown outcome can be recorded."""
    offer = _offer()
    host = AtomicHost(_authority())
    claim_cache(host, offer, clock=lambda: 1000)
    malformed = (
        replace(_ready(offer), operation_id="wrong"),
        replace(_ready(offer), observed_at=999),
        replace(_ready(offer), resource_name=None),
        replace(_ready(offer), total_tokens=None),
        replace(_ready(offer), expire_time=None),
        replace(_ready(offer), expire_time=1301),
        replace(_ready(offer), expire_time=1000),
        replace(_ready(offer), http_status=500),
        replace(_ready(offer), total_tokens=100),
        replace(_ready(offer), outcome="rejected", http_status=400),
    )
    for result in malformed:
        with pytest.raises(ValueError):
            finish_cache(host, offer, result)
    assert host.record_calls == 0
    finish_cache(host, offer, CacheResult(offer.operation_id, "unknown", 1002))
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize("create_time", [None, 1000.25])
def test_creation_time_is_optional_provider_reported_resource_interval(
    create_time: float | None,
) -> None:
    """Ready resource usability does not imply a complete provider-reported billing interval."""
    offer = _offer()
    result = replace(_ready(offer), create_time=create_time)
    assert validate_cache_result(offer, result) == result
    assert result.create_time == create_time
    assert asdict(result)["create_time"] == create_time
    with pytest.raises(FrozenInstanceError):
        result.create_time = 1000.5  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("create_time", [0, -1, True, float("nan"), float("inf"), 1002])
def test_invalid_known_creation_time_is_never_a_billing_fact(create_time: float) -> None:
    """A known creation time must be positive, finite and no later than its observation."""
    with pytest.raises(ValueError, match="create_time"):
        replace(_ready(_offer()), create_time=create_time)


def test_known_creation_time_requires_ordered_expiration() -> None:
    """Creation facts require an expiration at or after creation, including expired resources."""
    original = replace(_ready(_offer()), observed_at=1400)
    for expiry in (None, 1000):
        with pytest.raises(ValueError, match="create_time"):
            replace(original, create_time=1001, expire_time=expiry)
    assert replace(original, create_time=1300).create_time == 1300


def test_result_recording_failure_keeps_pending_reservation() -> None:
    """Failure to persist acceptance grants no assumption that a reservation was released."""

    class UnavailableRecorder(AtomicHost):
        """Simulate a host whose already-reserved operation cannot yet be settled."""

        def record(self, result: CacheResult) -> None:
            """Raise before committing any result to simulate unavailable durable storage."""
            raise RuntimeError("test storage unavailable")

    host = UnavailableRecorder(_authority())
    offer = _offer()
    claim_cache(host, offer, clock=lambda: 1000)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        finish_cache(host, offer, _ready(offer))
    assert claim_cache(host, offer, clock=lambda: 1001) == CacheUnavailable("pending")
    assert host.reserved == offer.reservation_nano_usd


def test_authority_requires_typed_verified_vertex_project_binding() -> None:
    """Only an exact validated host association can authorize project-ID resources."""
    project = VertexCacheProject("fruit-project", "123456789")
    assert replace(_authority(), vertex_project=project).vertex_project is project
    with pytest.raises(ValueError, match="verified project identity"):
        replace(_authority(), vertex_project={"endpoint_project": "fruit-project"})


def test_contracts_are_frozen_and_keep_scope_out_of_diagnostics() -> None:
    """Incidental repr cannot expose account bindings, and evidence has no plaintext input."""
    authority = _authority()
    offer = _offer()
    assert authority.tenant_scope not in repr(authority)
    assert authority.credential_scope not in repr(authority)
    assert offer.account_key_fingerprint not in repr(offer)
    assert "resource_name" not in repr(_ready(offer))
    assert {"payload", "headers", "api_key", "prompt", "content"}.isdisjoint(asdict(offer))
    for target, attribute in ((authority, "tenant_scope"), (offer, "ttl_seconds")):
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, "other")


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1, True])
def test_nonfinite_negative_and_boolean_observation_times_are_invalid(timestamp: float) -> None:
    """Malformed clock facts cannot become durable resource lifetime evidence."""
    with pytest.raises(ValueError, match="timestamp"):
        CacheResult("operation", "unknown", timestamp)


@pytest.mark.parametrize("status", [99, 600, True, 200.5])
def test_http_status_is_strict_integer_metadata(status: int) -> None:
    """HTTP observations contain only typed status codes, never provider error bodies."""
    with pytest.raises(ValueError):
        CacheResult("operation", "unknown", 1000, http_status=status)

"""Provider-free coverage of real native admission and explicit-cache callbacks."""

from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

import exp.runtime.gateway.native_bridge as bridge_module
import exp.runtime.gateway.native_explicit_cache as cache_module
from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.explicit_cache import (
    CacheClaim,
    CacheOffer,
    CacheReady,
    CacheResult,
    ExplicitCacheHost,
    GoogleCacheAuthority,
)
from exp.runtime.gateway.explicit_cache_test import AtomicHost, _authority
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _configured_pool_gateway, _start_first
from exp.runtime.gateway.native_explicit_cache import CONDITIONAL_CACHE_DISCLOSURE
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX

_MODEL = "gemini-2.5-pro"
_SECRET = "dummy-google-api-key-private-canary"
_PREFIX = "private-inventory-prefix-canary: apples=42.\n" * 256
_SYSTEM = "private-system-canary: Answer only from the inventory."
_SUFFIX = "How many apples are available?"
_RESOURCE = "cachedContents/verified_test_resource"


class _Host(AtomicHost):
    """Test-only atomic cache host with observable policy lookups.

    Attributes:
        authority_calls: Authorized identity, deployment and model of each lookup.
        fail_record: Whether to simulate a failed durable result write.
        fail_authority: Whether to simulate a failed policy lookup.
    """

    def __init__(self, authority: GoogleCacheAuthority | None) -> None:
        """Retain the shared fake reservation behavior without a production cache store."""
        super().__init__(authority)
        self.authority_calls: list[tuple[str, str, str]] = []
        self.fail_record = False
        self.fail_authority = False

    def authority(
        self,
        authorization: AuthorizationSnapshot,
        deployment: ExactModelDeployment,
        profile: GatewayWireProfile,
    ) -> GoogleCacheAuthority | None:
        """Observe the private binding while never returning raw credentials as authority."""
        with self.lock:
            self.authority_calls.append(
                (authorization.identity_id, deployment.deployment_id, profile.model_id)
            )
        if self.fail_authority:
            raise RuntimeError(_SECRET + _PREFIX)
        return self.authority_value

    def record(self, result: CacheResult) -> None:
        """Fail before committing when requested, otherwise use atomic fake settlement."""
        if self.fail_record:
            raise RuntimeError(_SECRET + _PREFIX)
        super().record(result)


@dataclass
class _Clock:
    """Controllable wall clock that does not alter real admission deadlines.

    Attributes:
        now: Unix wall time used for cache offers and observations.
    """

    now: float = 1_800_000_000.0

    def time(self) -> float:
        """Return the chosen wall time without patching global Python time."""
        return self.now

    def monotonic(self) -> float:
        """Keep request deadline behavior aligned with the real accounting clock."""
        return time.monotonic()


@pytest.fixture(autouse=True)
def _forbid_provider_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if this callback-only suite accidentally attempts even loopback network I/O."""
    monkeypatch.setattr(
        socket.socket, "connect", mock.Mock(side_effect=AssertionError("network I/O forbidden"))
    )
    monkeypatch.setattr(
        socket.socket, "connect_ex", mock.Mock(side_effect=AssertionError("network I/O forbidden"))
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Keep exact provider expiration deterministic across typed callback tests."""
    value = _Clock()
    monkeypatch.setattr(cache_module, "time", value)
    return value


def _control(
    root: Path, host: _Host | None = None, *, pool: bool = False
) -> tuple[NativeControlPlane, str]:
    """Construct real admission/accounting with official Google URLs and dummy keys."""
    if pool:
        _manager, key = _configured_pool_gateway(
            root, provider="gemini", provider_models=(_MODEL, _MODEL)
        )
    else:
        _manager, key = _configured_gateway(root, provider="gemini", provider_model=_MODEL)
    components = load_gateway_components(root, environment={"TEST_PROVIDER_KEY": _SECRET})
    if host is None:
        return NativeControlPlane(components), key
    return NativeControlPlane(components, explicit_cache=host), key


def _body(*, marked: bool = True, suffix: str = _SUFFIX) -> str:
    """Encode a real Messages API request with one explicit user-prefix checkpoint."""
    prefix: JsonObject = {"type": "text", "text": _PREFIX}
    if marked:
        prefix["cache_control"] = {"type": "ephemeral"}
    return json.dumps(
        {
            "model": "coding",
            "max_tokens": 128,
            "system": _SYSTEM,
            "messages": [
                {"role": "user", "content": [prefix]},
                {"role": "user", "content": suffix},
            ],
        }
    )


def _admission(
    control: NativeControlPlane, key: str, *, marked: bool = True, suffix: str = _SUFFIX
) -> JsonObject:
    """Use the public decoder and real route builder, without dispatching HTTP."""
    return _admit(control, key, _body(marked=marked, suffix=suffix), surface="messages")


def _wire(admission: JsonObject, depth: int = 0) -> JsonObject:
    """Extract one typed wire dictionary from the actual native admission."""
    route = admission["route"]
    assert isinstance(route, list)
    wire = route[depth]
    assert isinstance(wire, dict)
    return wire


def _selector(admission: JsonObject, depth: int = 0) -> JsonObject:
    """Select the retained request and deployment, not a caller-supplied credential."""
    return {
        "request_id": admission["request_id"],
        "deployment_id": _wire(admission, depth)["deployment_id"],
    }


def _prepare(control: NativeControlPlane, admission: JsonObject, depth: int = 0) -> JsonObject:
    """Run the typed preparation callback for an already-started attempt."""
    return json.loads(control.prepare_explicit_cache(json.dumps(_selector(admission, depth))))


def _finish_argument(admission: JsonObject, creation: JsonObject) -> JsonObject:
    """Project synthetic complete Google response facts into the native callback shape."""
    expires_at = creation["expires_at"]
    assert isinstance(expires_at, (int, float))
    return {
        **_selector(admission),
        "operation_id": creation["operation_id"],
        "outcome": "ready",
        "http_status": 200,
        "name": _RESOURCE,
        "expire_time": datetime.fromtimestamp(expires_at, UTC).isoformat(),
        "total_tokens": 1536,
    }


def _finish(control: NativeControlPlane, argument: JsonObject) -> JsonObject:
    """Run the typed finish callback without sending a provider request."""
    return json.loads(control.finish_explicit_cache(json.dumps(argument)))


def _assert_private(value: str) -> None:
    """Forbid raw credentials and either private prompt segment in diagnostic surfaces."""
    for secret in (_SECRET, _PREFIX, _SYSTEM, "private-inventory-prefix-canary"):
        assert secret not in value


@pytest.mark.parametrize("method", ["authority", "claim", "record"])
@pytest.mark.parametrize("malformation", ["missing", "noncallable"])
def test_invalid_host_contract_is_refused_before_admission(
    tmp_path: Path, method: str, malformation: str
) -> None:
    """Incomplete cache hosts cannot start a control plane that might later dispatch HTTP."""
    control, _key = _control(tmp_path)
    invalid = SimpleNamespace(authority=mock.Mock(), claim=mock.Mock(), record=mock.Mock())
    if malformation == "missing":
        delattr(invalid, method)
    else:
        setattr(invalid, method, _SECRET)
    with (
        mock.patch.object(
            control._components.ledger, "accept_request", side_effect=AssertionError("admission")
        ) as accept,
        pytest.raises(ValueError) as error,
    ):
        NativeControlPlane(control._components, explicit_cache=cast("ExplicitCacheHost", invalid))
    _assert_private(str(error.value) + repr(error.value))
    accept.assert_not_called()
    for candidate in ("authority", "claim", "record"):
        retained = getattr(invalid, candidate, None)
        if isinstance(retained, mock.Mock):
            retained.assert_not_called()


def test_default_host_absent_keeps_generation_and_marker_omission(tmp_path: Path) -> None:
    """The unchanged default grants no explicit-cache operation or public success claim."""
    control, key = _control(tmp_path)
    admission = _admission(control, key)
    wire = _wire(admission)
    assert "explicit_cache" not in wire
    ignored = admission["ignored_parameters"]
    assert isinstance(ignored, list)
    assert CONDITIONAL_CACHE_DISCLOSURE not in ignored
    assert any(str(item).endswith(CACHE_CONTROL_NOT_FORWARDED_SUFFIX) for item in ignored)
    payload = deepcopy(wire["upstream_payload"])
    assert json.dumps(_PREFIX) in json.dumps(payload)
    assert _SYSTEM in json.dumps(payload)
    assert _SUFFIX in json.dumps(payload)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = control._accounting.entry(request_id)
    assert entry is not None and entry.explicit_cache_state is None
    _start_first(control, admission)
    assert _prepare(control, admission) == {"state": "disabled"}
    assert wire["upstream_payload"] == payload


def test_zero_allowance_never_claims_and_leaves_original_generation(
    tmp_path: Path, clock: _Clock
) -> None:
    """A marker alone is not spend authority, even when admission retained a pure plan."""
    host = _Host(None)
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    wire = _wire(admission)
    original = deepcopy(wire["upstream_payload"])
    assert wire["explicit_cache"] is True
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    _start_first(control, admission)
    assert _prepare(control, admission) == {"state": "disabled"}
    assert _prepare(control, admission) == {"state": "unavailable"}
    assert len(host.authority_calls) == 1
    assert host.claim_calls == host.record_calls == host.reserved == 0
    assert host.offers == host.results == {}
    assert wire["upstream_payload"] == original
    assert json.dumps(_PREFIX) in json.dumps(original)
    assert _SUFFIX in json.dumps(original)
    assert admission["ignored_parameters"] == [CONDITIONAL_CACHE_DISCLOSURE]
    _assert_private(json.dumps(admission["ignored_parameters"]))


def test_unmarked_request_never_looks_up_authority_or_claims(tmp_path: Path) -> None:
    """Host allowance cannot turn an unmarked ordinary request into cache creation."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key, marked=False)
    assert "explicit_cache" not in _wire(admission)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = control._accounting.entry(request_id)
    assert entry is not None and entry.explicit_cache_state is None
    _start_first(control, admission)
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    ignored = admission["ignored_parameters"]
    assert isinstance(ignored, list)
    assert CONDITIONAL_CACHE_DISCLOSURE not in ignored


def test_zdr_authority_disables_admission_plans_and_all_cache_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host-issued ZDR restriction wins even when the marked Google prefix is eligible."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    original_authorize = control._components.store.authorize_request

    def authorize(
        *,
        raw_key: str,
        alias: str,
        request: ServingRequest,
        deadline_monotonic: float,
        app_referer: str | None = None,
        app_title: str | None = None,
        client_ip: str | None = None,
    ) -> AuthorizationSnapshot:
        """Add the hosted privacy restriction after authenticating through the real store."""
        authorization = original_authorize(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
            app_referer=app_referer,
            app_title=app_title,
            client_ip=client_ip,
        )
        return authorization.model_copy(update={"zdr_requested": True})

    monkeypatch.setattr(
        type(control._components.store), "authorize_request", staticmethod(authorize)
    )
    admission = _admission(control, key)
    assert "explicit_cache" not in _wire(admission)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = control._accounting.entry(request_id)
    assert entry is not None and entry.authorization.zdr_requested
    assert entry.explicit_cache_state is None
    _start_first(control, admission)
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    ignored = admission["ignored_parameters"]
    assert isinstance(ignored, list)
    assert CONDITIONAL_CACHE_DISCLOSURE not in ignored
    assert any(str(item).endswith(CACHE_CONTROL_NOT_FORWARDED_SUFFIX) for item in ignored)


def test_only_selected_route_can_claim_after_attempt_start(tmp_path: Path, clock: _Clock) -> None:
    """Planning two possible rungs reserves nothing; the actual selected rung owns one create."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host, pool=True)
    admission = _admission(control, key)
    assert _wire(admission, 0)["explicit_cache"] is True
    assert _wire(admission, 1)["explicit_cache"] is True
    assert host.authority_calls == [] and host.claim_calls == 0
    with pytest.raises(NativeBridgeError):
        _prepare(control, admission)
    assert host.authority_calls == [] and host.claim_calls == 0
    started = _start_first(control, admission)
    assert started["route_depth"] == 0
    with pytest.raises(NativeBridgeError):
        _prepare(control, admission, 1)
    creation = _prepare(control, admission)
    assert creation["state"] == "create"
    assert creation["url"] == "https://generativelanguage.googleapis.com/v1beta/cachedContents"
    payload = creation["payload"]
    assert isinstance(payload, dict)
    assert "ttl" not in payload
    assert payload["expireTime"] == datetime.fromtimestamp(clock.now + 300, UTC).isoformat()
    assert payload["model"] == "models/" + _MODEL
    assert host.authority_calls == [("default", _wire(admission)["deployment_id"], _MODEL)]
    offer = next(iter(host.offers.values()))
    assert offer.request_id == admission["request_id"]
    assert offer.attempt_id == started["attempt_id"]
    assert host.reserved == offer.create_nano_usd + offer.storage_nano_usd > 0
    assert _prepare(control, admission) == {"state": "unavailable"}
    assert host.claim_calls == 1 and len(host.offers) == 1


@pytest.mark.parametrize(
    "fraction",
    [0.0000002, 0.0000007, 0.1234567, 0.9999997],
    ids=["submicrosecond-down", "submicrosecond-up", "fractional-up", "near-next-second"],
)
def test_fractional_clock_expiry_round_trips_exactly_and_remains_reusable(
    tmp_path: Path, clock: _Clock, fraction: float
) -> None:
    """Provider ISO expiration cannot round above the exact bound reserved by the host."""
    clock.now += fraction
    assert datetime.fromtimestamp(clock.now, UTC).timestamp() != clock.now
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    first = _admission(control, key)
    _start_first(control, first)
    creation = _prepare(control, first)
    assert creation["state"] == "create"
    payload = creation["payload"]
    assert isinstance(payload, dict)
    expiration = payload["expireTime"]
    assert isinstance(expiration, str)
    parsed_expiration = datetime.fromisoformat(expiration).timestamp()
    offer = next(iter(host.offers.values()))
    assert offer.requested_at == float(int(clock.now))
    assert offer.expires_at == parsed_expiration == creation["expires_at"]
    assert offer.expires_at - offer.requested_at == 300
    argument = {**_finish_argument(first, creation), "expire_time": expiration}
    assert _finish(control, argument)["state"] == "ready"
    result = next(iter(host.results.values()))
    assert result.outcome == "ready" and result.expire_time == offer.expires_at
    clock.now += 1
    second = _admission(control, key)
    _start_first(control, second)
    reused = _prepare(control, second)
    assert reused["state"] == "ready" and reused["resource_name"] == _RESOURCE
    assert host.record_calls == 1 and len(host.offers) == 1
    assert host.reserved == offer.reservation_nano_usd


def test_concurrent_requests_and_workers_share_exactly_one_creator(
    tmp_path: Path, clock: _Clock
) -> None:
    """Independent admitted requests race only at the injected atomic host boundary."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    replica = NativeControlPlane(control._components, explicit_cache=host)
    controls = (control, replica, control, replica)
    admissions = [_admission(plane, key) for plane in controls]
    assert len({admission["request_id"] for admission in admissions}) == 4
    for plane, admission in zip(controls, admissions, strict=True):
        _start_first(plane, admission)
    barrier = threading.Barrier(len(controls))

    def prepare(index: int) -> JsonObject:
        """Race four distinct physical attempts, not duplicate callbacks on one entry."""
        barrier.wait(timeout=5)
        return _prepare(controls[index], admissions[index])

    with ThreadPoolExecutor(max_workers=len(controls)) as executor:
        decisions = list(executor.map(prepare, range(len(controls))))
    assert [decision["state"] for decision in decisions].count("create") == 1
    assert [decision["state"] for decision in decisions].count("unavailable") == 3
    assert host.claim_calls == 4 and len(host.offers) == 1
    offer = next(iter(host.offers.values()))
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize("scope", ["tenant_scope", "credential_scope"])
def test_host_scope_change_never_reuses_another_tenant_or_credential(
    tmp_path: Path, clock: _Clock, scope: str
) -> None:
    """An identical model and prefix cannot cross the host's isolation generation."""
    authority = _authority()
    host = _Host(authority)
    control, key = _control(tmp_path, host)
    first = _admission(control, key)
    _start_first(control, first)
    creation = _prepare(control, first)
    assert _finish(control, _finish_argument(first, creation))["state"] == "ready"
    host.authority_value = replace(authority, **{scope: "different-private-host-scope"})
    second = _admission(control, key)
    _start_first(control, second)
    assert _prepare(control, second)["state"] == "create"
    offers = list(host.offers.values())
    assert len(offers) == 2
    assert offers[0].cache_key != offers[1].cache_key
    assert host.reserved == sum(offer.reservation_nano_usd for offer in offers)


def test_finish_and_later_reuse_return_exact_resource_and_suffix_only(
    tmp_path: Path, clock: _Clock
) -> None:
    """A known resource replaces the exact prefix without duplicating cached system or text."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    first = _admission(control, key)
    original = deepcopy(_wire(first)["upstream_payload"])
    _start_first(control, first)
    creation = _prepare(control, first)
    argument = _finish_argument(first, creation)
    ready = _finish(control, argument)
    assert ready["state"] == "ready" and ready["resource_name"] == _RESOURCE
    payload = ready["payload"]
    assert isinstance(payload, dict)
    assert payload["cachedContent"] == _RESOURCE
    assert payload["contents"] == [{"role": "user", "parts": [{"text": _SUFFIX}]}]
    assert isinstance(original, dict)
    assert payload["generationConfig"] == original["generationConfig"]
    assert "systemInstruction" not in payload
    assert json.dumps(_PREFIX) not in json.dumps(payload) and _SYSTEM not in json.dumps(payload)
    assert _wire(first)["upstream_payload"] == original
    assert _finish(control, argument) == ready
    assert host.record_calls == 1
    assert _prepare(control, first) == {"state": "unavailable"}
    clock.now += 1
    second = _admission(control, key, suffix="What is the inventory count now?")
    _start_first(control, second)
    reused = _prepare(control, second)
    assert reused["state"] == "ready" and reused["resource_name"] == _RESOURCE
    reused_payload = reused["payload"]
    assert isinstance(reused_payload, dict)
    assert reused_payload["cachedContent"] == _RESOURCE
    assert reused_payload["contents"] == [
        {"role": "user", "parts": [{"text": "What is the inventory count now?"}]}
    ]
    assert len(host.offers) == 1 and host.claim_calls == 2 and host.record_calls == 1
    assert host.reserved == next(iter(host.offers.values())).reservation_nano_usd


def test_slow_host_claim_cannot_reuse_resource_that_expires_while_waiting(
    tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-host freshness refuses stale reuse without recording, releasing or recreating."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    first = _admission(control, key)
    _start_first(control, first)
    creation = _prepare(control, first)
    assert _finish(control, _finish_argument(first, creation))["state"] == "ready"
    original_offer = next(iter(host.offers.values()))
    original_results = dict(host.results)
    original_reserved = host.reserved
    clock.now += 1
    later = _admission(control, key)
    _start_first(control, later)
    original_claim = host.claim

    def slow_claim(offer: CacheOffer) -> CacheClaim:
        """Return the existing resource only after simulated host latency passes expiry."""
        claim = original_claim(offer)
        assert isinstance(claim, CacheReady)
        assert clock.now < claim.expires_at
        clock.now = claim.expires_at + 1
        return claim

    monkeypatch.setattr(host, "claim", slow_claim)
    assert _prepare(control, later) == {"state": "unavailable"}
    assert _prepare(control, later) == {"state": "unavailable"}
    assert host.claim_calls == 2 and len(host.offers) == 1
    assert next(iter(host.offers.values())) == original_offer
    assert host.results == original_results and host.record_calls == 1
    assert host.reserved == original_reserved


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "cachedContents/../private"},
        {"name": "projects/foreign/locations/us/cachedContents/one"},
        {"name": "cachedContents/one?key=" + _SECRET},
        {"expire_time": "not-a-timestamp"},
        {"expire_time": "2027-01-15T08:05:00"},
        {"expire_time": None},
        {"name": None},
        {"total_tokens": None},
        {"total_tokens": 1},
        {"total_tokens": 2**62},
        {"http_status": 500},
    ],
    ids=[
        "traversal",
        "foreign-resource",
        "query-secret",
        "bad-time",
        "naive-time",
        "missing-time",
        "missing-name",
        "missing-tokens",
        "below-minimum",
        "over-bound",
        "ambiguous-server-error",
    ],
)
def test_malformed_provider_facts_record_unknown_and_keep_reservation(
    tmp_path: Path, clock: _Clock, changed: JsonObject
) -> None:
    """Invalid accepted-resource facts cannot publish ready or release possible charges."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    creation = _prepare(control, admission)
    argument = {**_finish_argument(admission, creation), **changed}
    assert _finish(control, argument) == {"state": "unavailable"}
    result = next(iter(host.results.values()))
    assert result.outcome == "unknown" and result.resource_name is None
    assert result.expire_time is None and result.total_tokens is None
    offer = next(iter(host.offers.values()))
    assert host.reserved == offer.reservation_nano_usd
    clock.now = offer.expires_at + 60
    later = _admission(control, key)
    _start_first(control, later)
    assert _prepare(control, later) == {"state": "unavailable"}
    assert len(host.offers) == host.record_calls == 1
    assert host.reserved == offer.reservation_nano_usd
    _assert_private(repr(result))


def test_late_complete_acceptance_is_accounted_but_never_reused(
    tmp_path: Path, clock: _Clock
) -> None:
    """Expiry blocks generation reuse without erasing complete observed provider liability."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    creation = _prepare(control, admission)
    argument = _finish_argument(admission, creation)
    offer = next(iter(host.offers.values()))
    clock.now = offer.expires_at + 1
    assert _finish(control, argument) == {"state": "unavailable"}
    result = next(iter(host.results.values()))
    assert result.outcome == "ready" and result.resource_name == _RESOURCE
    assert host.reserved == offer.reservation_nano_usd


@pytest.mark.parametrize(
    "changed",
    [
        {"operation_id": "another-operation"},
        {"deployment_id": "another-deployment"},
        {"request_id": "another-request"},
        {"total_tokens": True},
        {"total_tokens": "1536"},
        {"http_status": "200"},
        {"outcome": "rejected"},
        {"raw_response": _SECRET},
        {"name": "cachedContents/" + "a" * 1025},
        {"name": _SECRET * 2000},
    ],
    ids=[
        "wrong-operation",
        "wrong-deployment",
        "wrong-request",
        "boolean-tokens",
        "string-tokens",
        "string-status",
        "untrusted-rejection",
        "extra-provider-body",
        "oversized-field",
        "oversized-callback",
    ],
)
def test_strict_finish_boundary_rejects_untrusted_shapes_and_preserves_pending_money(
    tmp_path: Path, clock: _Clock, changed: JsonObject
) -> None:
    """Untrusted native envelopes fail closed without settling an unrelated operation."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    creation = _prepare(control, admission)
    argument = {**_finish_argument(admission, creation), **changed}
    with pytest.raises(NativeBridgeError) as error:
        _finish(control, argument)
    _assert_private(str(error.value) + repr(error.value) + error.value.public_error_json)
    assert host.results == {} and host.record_calls == 0
    assert host.reserved == next(iter(host.offers.values())).reservation_nano_usd
    assert _prepare(control, admission) == {"state": "unavailable"}
    assert host.claim_calls == 1


def test_record_failure_halts_callback_and_retains_pending_claim(
    tmp_path: Path, clock: _Clock
) -> None:
    """The bridge cannot authorize generation from an uncommitted create observation."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    creation = _prepare(control, admission)
    host.fail_record = True
    with pytest.raises(NativeBridgeError) as error:
        _finish(control, _finish_argument(admission, creation))
    public = json.loads(error.value.public_error_json)
    assert public["status_code"] == 502
    _assert_private(str(error.value) + repr(error.value) + error.value.public_error_json)
    assert host.results == {} and host.record_calls == 0
    assert host.reserved == next(iter(host.offers.values())).reservation_nano_usd
    assert _prepare(control, admission) == {"state": "unavailable"}
    later = _admission(control, key)
    _start_first(control, later)
    assert _prepare(control, later) == {"state": "unavailable"}
    assert len(host.offers) == 1


def test_authority_failure_is_sanitized_and_cannot_retry_create(
    tmp_path: Path, clock: _Clock
) -> None:
    """Provider credentials in a host exception never cross the public error boundary."""
    host = _Host(_authority())
    host.fail_authority = True
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    with pytest.raises(NativeBridgeError) as error:
        _prepare(control, admission)
    _assert_private(str(error.value) + repr(error.value) + error.value.public_error_json)
    assert _prepare(control, admission) == {"state": "unavailable"}
    assert host.claim_calls == host.record_calls == host.reserved == 0


def test_cache_bindings_and_durable_metadata_keep_prompt_and_credentials_private(
    tmp_path: Path, clock: _Clock
) -> None:
    """Private plans are usable internally but incidental repr and evidence stay content-free."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key)
    _start_first(control, admission)
    creation = _prepare(control, admission)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = control._accounting.entry(request_id)
    assert entry is not None and entry.explicit_cache_state is not None
    state = entry.explicit_cache_state
    binding = state.bindings[0]
    assert binding is not None
    for value in (state, binding, binding.plan, host.authority_value, *host.offers.values()):
        _assert_private(repr(value))
    _assert_private(json.dumps(admission["ignored_parameters"]))
    for offer in host.offers.values():
        _assert_private(json.dumps(asdict(offer)))
    _finish(control, _finish_argument(admission, creation))
    for result in host.results.values():
        _assert_private(repr(result) + json.dumps(asdict(result)))


def test_admission_plan_failure_is_sanitized_and_finishes_accepted_request(
    tmp_path: Path,
) -> None:
    """A pure binding failure stays inside the real admission cleanup boundary."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    with (
        mock.patch.object(
            bridge_module, "bind_explicit_cache", side_effect=ValueError(_SECRET + _PREFIX)
        ),
        mock.patch.object(
            control._accounting,
            "finish_request_quietly",
            wraps=control._accounting.finish_request_quietly,
        ) as finished,
        pytest.raises(NativeBridgeError) as error,
    ):
        _admission(control, key)
    _assert_private(str(error.value) + repr(error.value) + error.value.public_error_json)
    finished.assert_called_once()
    assert host.authority_calls == [] and host.claim_calls == host.record_calls == 0

"""Initial cache binding and the requirement for native financial cache authority.

The former Python callback regressions are frozen in native/tests/generated/cache.jsonl
and run through accounting/cache_tests.rs, including cost, namespace, expiry,
provider receipts, renewal, uncertain writes and cancellation.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

import exp.runtime.gateway.native_bridge as bridge_module
import exp.runtime.gateway.native_explicit_cache as cache_module
from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.explicit_cache import (
    CachePrefix,
    CacheResult,
    ExplicitCacheHost,
    GoogleCacheAuthority,
)
from exp.runtime.gateway.explicit_cache_test import AtomicHost, _authority
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _configured_pool_gateway, _start_first
from exp.runtime.gateway.native_explicit_cache import CONDITIONAL_CACHE_DISCLOSURE, NativeCacheState
from exp.runtime.gateway.prompt_caching import PROMPT_CACHING_OFF_SUFFIX
from exp.runtime.gateway.tests.native_admission_fixture_test import AdmissionRecords
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX
from exp.runtime.models.providers.google_cache import VertexCacheProject
from exp.runtime.models.providers.protocol import NativeWireClient

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


@pytest.fixture(autouse=True)
def _forbid_provider_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if this callback-only suite accidentally attempts even loopback network I/O."""
    monkeypatch.setattr(
        socket.socket, "connect", mock.Mock(side_effect=AssertionError("network I/O forbidden"))
    )
    monkeypatch.setattr(
        socket.socket, "connect_ex", mock.Mock(side_effect=AssertionError("network I/O forbidden"))
    )


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


def _assert_private(value: str) -> None:
    """Forbid raw credentials and either private prompt segment in diagnostic surfaces."""
    for secret in (_SECRET, _PREFIX, _SYSTEM, "private-inventory-prefix-canary"):
        assert secret not in value


@pytest.mark.parametrize("method", ["authority", "claim", "record", "record_renewal"])
@pytest.mark.parametrize("malformation", ["missing", "noncallable"])
def test_invalid_host_contract_is_refused_before_admission(
    tmp_path: Path, method: str, malformation: str
) -> None:
    """Incomplete cache hosts cannot start a control plane that might later dispatch HTTP."""
    control, _key = _control(tmp_path)
    invalid = SimpleNamespace(
        authority=mock.Mock(), claim=mock.Mock(), record=mock.Mock(), record_renewal=mock.Mock()
    )
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
    for candidate in ("authority", "claim", "record", "record_renewal"):
        retained = getattr(invalid, candidate, None)
        if isinstance(retained, mock.Mock):
            retained.assert_not_called()


def test_default_host_absent_keeps_generation_and_marker_omission(
    tmp_path: Path, admission_records: AdmissionRecords
) -> None:
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
    entry = admission_records.get(control._accounting, request_id)
    assert entry is not None and entry.explicit_cache_state is None
    _start_first(control, admission)
    assert _prepare(control, admission) == {"state": "disabled"}
    assert wire["upstream_payload"] == payload


def test_unmarked_request_never_looks_up_authority_or_claims(
    tmp_path: Path, admission_records: AdmissionRecords
) -> None:
    """Host allowance cannot turn an unmarked ordinary request into cache creation."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    admission = _admission(control, key, marked=False)
    assert "explicit_cache" not in _wire(admission)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = admission_records.get(control._accounting, request_id)
    assert entry is not None and entry.explicit_cache_state is None
    _start_first(control, admission)
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    ignored = admission["ignored_parameters"]
    assert isinstance(ignored, list)
    assert CONDITIONAL_CACHE_DISCLOSURE not in ignored


@pytest.mark.parametrize(
    ("restriction", "marker_suffix"),
    [
        ("zdr_requested", CACHE_CONTROL_NOT_FORWARDED_SUFFIX),
        ("prompt_caching_disabled", PROMPT_CACHING_OFF_SUFFIX),
    ],
)
def test_zdr_authority_disables_admission_plans_and_all_cache_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restriction: str,
    marker_suffix: str,
    admission_records: AdmissionRecords,
) -> None:
    """A host-issued ZDR or caching-off restriction wins over an eligible marked prefix.

    Under ZDR the marker is dropped as on any non-caching route; with the
    organization's prompt caching off it is removed at admission and
    disclosed as such.
    """
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
        return authorization.model_copy(update={restriction: True})

    monkeypatch.setattr(
        type(control._components.store), "authorize_request", staticmethod(authorize)
    )
    admission = _admission(control, key)
    assert "explicit_cache" not in _wire(admission)
    request_id = admission["request_id"]
    assert isinstance(request_id, str)
    entry = admission_records.get(control._accounting, request_id)
    assert entry is not None and getattr(entry.authorization, restriction)
    assert entry.explicit_cache_state is None
    _start_first(control, admission)
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    ignored = admission["ignored_parameters"]
    assert isinstance(ignored, list)
    assert CONDITIONAL_CACHE_DISCLOSURE not in ignored
    assert any(str(item).endswith(marker_suffix) for item in ignored)


def test_cache_bindings_keep_prompt_and_credentials_private_before_native_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native capability rejection cannot disclose the private admission-only cache plans."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    original = bridge_module.bind_explicit_cache
    plans: list[object] = []

    def record_binding(
        host: ExplicitCacheHost | None,
        authorization: AuthorizationSnapshot,
        deployments: Sequence[ExactModelDeployment],
        resolved_wires: Sequence[tuple[GatewayWireProfile, NativeWireClient]],
        provider_request: GatewayRequest,
        public_request: GatewayRequest,
        wire_route: list[JsonObject],
        *,
        automatic: bool = False,
    ) -> tuple[NativeCacheState | None, GatewayRequest]:
        """Observe the pure binder without substituting a financial callback."""
        state, request = original(
            host,
            authorization,
            deployments,
            resolved_wires,
            provider_request,
            public_request,
            wire_route,
            automatic=automatic,
        )
        assert state is not None
        plans.extend([state, *state.bindings])
        return state, request

    monkeypatch.setattr(bridge_module, "bind_explicit_cache", record_binding)
    with pytest.raises(ValueError):
        _admission(control, key)
    assert plans
    for value in (*plans, host.authority_value):
        _assert_private(repr(value))
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    assert control.close(timeout_seconds=2)


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


class _AutomaticHost(_Host):
    """Select only a repeated digest under the exact authority in an offline fixture."""

    def __init__(self) -> None:
        """Use a verified fictional account and retain only prefix metadata."""
        super().__init__(
            replace(_authority(), vertex_project=VertexCacheProject("fruit-project", "123456789"))
        )
        self.seen: dict[tuple[str, str, str], str] = {}

    def select_prefix(
        self, authority: GoogleCacheAuthority, request_id: str, candidates: tuple[CachePrefix, ...]
    ) -> str | None:
        """Choose the longest repeat without seeing or retaining prompt text."""
        chosen = None
        with self.lock:
            for candidate in sorted(candidates, key=lambda value: value.prefix_bytes):
                key = (authority.tenant_scope, authority.credential_scope, candidate.prefix_sha256)
                previous = self.seen.get(key)
                if previous is not None and previous != request_id:
                    chosen = candidate.prefix_sha256
                self.seen[key] = request_id
        return chosen


@pytest.mark.parametrize("version", [None, 0, 2])
def test_automatic_host_requires_matching_native_contract(version: int | None) -> None:
    """Older or unknown native wheels cannot ignore the automatic token preflight."""
    with mock.patch.object(
        cache_module.importlib,
        "import_module",
        return_value=SimpleNamespace(AUTOMATIC_VERTEX_CACHE_CONTRACT_VERSION=version),
    ):
        with pytest.raises(ValueError, match="matching native cache contract"):
            cache_module.validate_cache_hosts(None, _AutomaticHost())


def _renew(control: NativeControlPlane, argument: JsonObject) -> JsonObject:
    """Run the typed renewal callback without sending a provider request."""
    return json.loads(control.renew_explicit_cache(json.dumps(argument)))


def test_explicit_cache_never_reenters_python_financial_callbacks(tmp_path: Path) -> None:
    """A host without native resource authority cannot create a paid cache through Python."""
    host = _Host(_authority())
    control, key = _control(tmp_path, host)
    with pytest.raises(ValueError) as rejected:
        _admission(control, key)
    _assert_private(str(rejected.value))
    assert host.authority_calls == []
    assert host.claim_calls == host.record_calls == host.reserved == 0
    assert control.close(timeout_seconds=2)

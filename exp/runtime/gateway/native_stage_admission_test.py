"""Native admission consumes session evidence while respecting stage and host gates."""

import json
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import pytest

from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.runtime.gateway import native_stage_admission
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.model_plan import model_execution_snapshot
from exp.runtime.gateway.model_plan_test import catalog
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import _RecordingLedger
from exp.runtime.gateway.native_admission import (
    _affinity_ordered_rungs,
    _prefer_cache_capable_rungs,
    admitted_route_requests,
    select_single_route_before_search,
)
from exp.runtime.gateway.native_admission_test import _affinity_fixture, _marked_request
from exp.runtime.gateway.native_execution import (
    InflightRequest,
    rung_load_key,
    select_route_deployments,
)
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_fallback_rules import (
    FailoverRulesError,
    eligible_ladder,
    require_unrestricted_rung,
)
from exp.runtime.gateway.native_recovery import record_session_outcome, session_cache_key
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.native_stage_admission import stage_affinity_ordered_rungs as _stage_order
from exp.runtime.gateway.recovery import (
    FrozenRecoveryBinding,
    OperationalScope,
    RecoveryScope,
    RecoverySnapshot,
    SessionCacheKey,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_test import Clock
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.request_policy import GatewayRequestPolicy, GatewayRoutingPolicy
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.sticky_affinity import AffinityPlacement
from exp.runtime.models.credentials import DispatchCredentialReceipt
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import (
    CACHE_CONTROL_NOT_FORWARDED_SUFFIX,
    dialect_stream_payload,
)
from exp.runtime.models.providers.protocol import NativeWireClient


@pytest.mark.parametrize("marker", [None, False, True, 0, 2, "1", 1.0])
def test_native_stage_contract_rejects_missing_or_unknown_marker(
    monkeypatch: pytest.MonkeyPatch,
    marker: int | str | float | None,
) -> None:
    """Newer version labels and generic exports cannot stand in for the compiled contract."""
    native = SimpleNamespace(__version__="99.0", serve=lambda: None)
    if marker is not None:
        native.MODEL_STAGE_CONTRACT_VERSION = marker
    monkeypatch.setattr(native_stage_admission.importlib, "import_module", lambda _: native)
    route = _route()
    normalized = catalog()
    staged = route.model_copy(
        update={
            "snapshot": model_execution_snapshot(
                normalized, route.snapshot.authorization, normalized.pools[0]
            )
        }
    )
    ledger = _RecordingLedger()
    accounting = NativeAttemptAccounting(ledger)
    request = GatewayRequest(
        surface=route.snapshot.authorization.surface,
        messages=(GatewayMessage(role="user", content="hello"),),
    )
    with pytest.raises(GatewayRoutingError, match="MODEL_STAGE_CONTRACT_VERSION=1"):
        admitted_route_requests(
            staged,
            (),
            request,
            accounting=accounting,
            authorization=staged.snapshot.authorization,
        )
    assert not ledger.started
    native_stage_admission.require_native_model_stage_contract(route)


def test_native_stage_contract_accepts_only_compiled_version_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported integer marker admits stage payloads without a package-version guess."""
    monkeypatch.setattr(
        native_stage_admission.importlib,
        "import_module",
        lambda _: SimpleNamespace(MODEL_STAGE_CONTRACT_VERSION=1, __version__="0.0.0"),
    )
    route = _route()
    normalized = catalog()
    staged = route.model_copy(
        update={
            "snapshot": model_execution_snapshot(
                normalized, route.snapshot.authorization, normalized.pools[0]
            )
        }
    )
    native_stage_admission.require_native_model_stage_contract(staged)


@dataclass
class Host:
    """In-memory immutable observation host with explicit credential identity."""

    credential: str = "credential"

    def scope_for(self, deployment: ExactModelDeployment, organization_id: str) -> RecoveryScope:
        """Freeze a known credential scope per destination model."""
        return RecoveryScope(
            provider=deployment.provider,
            exact_model_id=deployment.exact_model_id,
            endpoint_scope=deployment.connection_sha256,
            region_scope="region",
            credential_scope=str(uuid5(NAMESPACE_URL, self.credential)),
            organization_id=organization_id,
        )

    def observe_scope(self, scope: OperationalScope) -> None:
        """Register test topology without mutable credential lookup."""
        return None

    def attempt_started(self, attempt_id: str, scope: OperationalScope) -> None:
        """Accept reserved test topology without mutable credential lookup."""
        return None

    def snapshot(self) -> RecoverySnapshot:
        """Return no fleet recovery evidence, so a healthy warm fallback stays retained."""
        return RecoverySnapshot(loaded_at=1000)


def stage_affinity_ordered_rungs(
    route: GatewayRoute,
    wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None,
) -> tuple[
    GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...], AffinityPlacement
]:
    """Bind synthetic resolved profiles as production admission does before scheduling."""
    host = accounting.recovery_host
    if isinstance(host, Host):
        bound = []
        for deployment, (profile, client) in zip(route.deployments, wires, strict=True):
            scope = host.scope_for(deployment, authorization.organization_id)
            profile = replace(profile, operational_region=scope.region_scope)
            receipt = DispatchCredentialReceipt(uuid5(NAMESPACE_URL, host.credential))
            binding = FrozenRecoveryBinding(
                deployment.deployment_id,
                deployment.connection_sha256,
                profile.url,
                profile.model_id,
                scope,
                profile.operational_region,
                profile.dialect,
            )
            bound.append(
                (replace(profile, credential_receipt=receipt, recovery_binding=binding), client)
            )
        wires = tuple(bound)
    return _stage_order(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=authorization,
        continuation=continuation,
    )


def test_exact_selected_staged_route_cannot_be_reordered_or_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact opaque selector precedes every stage-affinity or recovery preference."""
    route, wires = _affinity_fixture()
    route = route.model_copy(update={"resolved_route_id": "route_" + "b" * 64})
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=Host())

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Neither rendezvous nor recovery may reinterpret the explicitly chosen first route."""
        raise AssertionError("selected route entered elective scheduling")

    monkeypatch.setattr(native_stage_admission, "rendezvous_order", unexpected)
    monkeypatch.setattr(accounting.recovery, "choose", unexpected)
    selected, selected_wires, _ = _stage_order(
        route,
        wires,
        _marked_request(),
        accounting=accounting,
        authorization=route.snapshot.authorization,
        continuation=None,
    )
    assert selected is route and selected_wires is wires


@pytest.mark.parametrize("root_available", [False, True])
def test_no_fallback_stage_preselection_never_enters_child(
    monkeypatch: pytest.MonkeyPatch, root_available: bool
) -> None:
    """No-fallback narrows only current root providers, even with descendant-start authority."""
    normalized = catalog()
    auth = _route().snapshot.authorization.model_copy(update={"descendant_start_authorized": True})
    snapshot = model_execution_snapshot(normalized, auth, normalized.pools[0])
    by_id = {deployment.deployment_id: deployment for deployment in normalized.deployments}
    deployments = tuple(by_id[name] for name in snapshot.deployment_ids)
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    _, fixture_wires = _affinity_fixture()
    wires = tuple(fixture_wires[0] for _ in deployments)
    request = _marked_request().model_copy(
        update={
            "gateway": GatewayRequestPolicy(routing=GatewayRoutingPolicy(allow_fallbacks=False))
        }
    )
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=Host())

    def prepare(candidate: GatewayRoute, resolved: tuple, incoming: GatewayRequest) -> tuple:
        """Observe the stage boundary before capability filtering or effectful search."""
        assert all(item.exact_model_id == snapshot.exact_model_id for item in candidate.deployments)
        if not root_available:
            raise GatewayRoutingError("root cannot serve request")
        return candidate, resolved, incoming, incoming, ()

    monkeypatch.setattr("exp.runtime.gateway.native_admission.prepare_route_requests", prepare)
    if not root_available:
        with pytest.raises(GatewayRoutingError, match="root cannot serve"):
            select_single_route_before_search(
                route, wires, request, accounting=accounting, authorization=auth, continuation=None
            )
    else:
        selected, selected_wires, _ = select_single_route_before_search(
            route, wires, request, accounting=accounting, authorization=auth, continuation=None
        )
        assert len(selected.deployments) == len(selected_wires) == 1
        assert selected.deployment.exact_model_id == snapshot.exact_model_id
        assert selected.snapshot.model_stages[0].deployment_ids == (
            selected.deployment.deployment_id,
        )
        assert all(
            stage.exact_model_id == snapshot.exact_model_id
            for stage in selected.snapshot.model_stages
        )


@pytest.mark.parametrize("mode", ["maximize_cache", "maximize_cache_affinity"])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
@pytest.mark.parametrize("conditional", [False, True])
def test_marker_capable_rungs_stay_inside_root_and_child_segments(
    mode: FailoverMode, wire: str, conditional: bool
) -> None:
    """Every supported marker wire ranks locally without granting conditional first dials."""
    route, original_wires = _affinity_fixture()
    template = route.deployment
    deployments = tuple(
        template.model_copy(
            update={
                "deployment_id": f"{model}-{kind}",
                "exact_model_id": model,
                "gateway": template.gateway.model_copy(
                    update={
                        "capabilities": template.gateway.capabilities.model_copy(
                            update={
                                "failover_only_on": ("refusal",)
                                if conditional and kind == "marked"
                                else None
                            }
                        )
                    }
                ),
            }
        )
        for model in ("root", "child")
        for kind in ("generic", "marked")
    )
    stages = tuple(
        route.snapshot.stage_for_depth(0).model_copy(
            update={
                "stage_index": index,
                "exact_model_id": model,
                "pool_id": f"pool-{model}",
                "deployment_ids": tuple(
                    d.deployment_id for d in deployments[index * 2 : index * 2 + 2]
                ),
                "rung_positions": (0, 1),
                "ancestry": ("root",) if index == 0 else ("root", "child"),
                "failover_mode": mode,
            }
        )
        for index, model in enumerate(("root", "child"))
    )
    route = route.model_copy(
        update={
            "snapshot": route.snapshot.model_copy(
                update={
                    "exact_model_id": "root",
                    "pool_id": "pool-root",
                    "deployment_ids": tuple(d.deployment_id for d in deployments),
                    "model_stages": stages,
                }
            ),
            "deployment": deployments[0],
            "fallback_deployments": deployments[1:],
        }
    )
    client = original_wires[0][1]
    generic = GatewayWireProfile(dialect="openai_compatible", url="https://generic.test")
    marked = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        url="https://marked.test",
        forwards_cache_control=wire == "openrouter",
    )
    wires = ((generic, client), (marked, client), (generic, client), (marked, client))
    ordered, resolved, _ = stage_affinity_ordered_rungs(
        route,
        wires,
        _marked_request(),
        accounting=NativeAttemptAccounting(_RecordingLedger()),
        authorization=route.snapshot.authorization,
        continuation=None,
    )
    assert ordered.snapshot.deployment_ids == (
        "root-marked",
        "root-generic",
        "child-marked",
        "child-generic",
    )
    assert [s.rung_positions for s in ordered.snapshot.model_stages] == [(1, 0), (1, 0)]
    assert [s.ancestry for s in ordered.snapshot.model_stages] == [("root",), ("root", "child")]
    assert [profile.preserves_cache_control for profile, _ in resolved] == [
        True,
        False,
        True,
        False,
    ]
    assert eligible_ladder(ordered, None) == ((1, 3) if conditional else (0, 1, 2, 3))


def test_settlement_records_successful_session_cache_once_and_never_dispatch_only() -> None:
    """The native settlement hook owns evidence, independent of aggregate fairness samples."""
    normalized = catalog()
    auth = _route().snapshot.authorization
    snapshot = model_execution_snapshot(normalized, auth, normalized.pools[0])
    by_id = {d.deployment_id: d for d in normalized.deployments}
    deployments = tuple(by_id[d] for d in snapshot.deployment_ids)
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    request = GatewayRequest(
        surface=auth.surface,
        messages=(
            GatewayMessage(role="system", content="prefix"),
            GatewayMessage(role="user", content="turn"),
        ),
        provider_prompt_cache_key="xpl-test",
    )
    entry = InflightRequest(authorization=auth, route=route, request=request, deadline_monotonic=10)
    key = session_cache_key(entry)
    assert key is not None
    registry = SessionRecoveryRegistry(clock=Clock())
    host = Host()
    scope = host.scope_for(route.deployment, auth.organization_id)
    entry.recovery_bindings[route.deployment.deployment_id] = FrozenRecoveryBinding(
        route.deployment.deployment_id,
        route.deployment.connection_sha256,
        "https://test.invalid",
        route.deployment.provider_model,
        scope,
    )
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    entry.attempt_depths["attempt"] = 0
    usage = GatewayUsage(input_tokens=100, output_tokens=1, cached_input_tokens=80)
    record_session_outcome(registry, host, entry, "attempt", usage, None)
    # Unknown declared retention remains no evidence even with a cache read.
    assert (
        registry.choose(
            key, ((route.deployment.deployment_id, scope),), eligible=lambda _: True, snapshot=None
        ).deployment_id
        is None
    )
    assert entry.recovery_recorded_attempts == {"attempt"}


@pytest.mark.parametrize("has_stages", [False, True])
@pytest.mark.parametrize("with_host", [False, True])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
def test_live_reasoning_pin_precedes_stage_cache_and_recovery_ordering(
    has_stages: bool,
    with_host: bool,
    wire: str,
) -> None:
    """A live child issuer is preserved, while a removed issuer is never resurrected."""
    route, wires = _affinity_fixture()
    snapshot = route.snapshot
    if has_stages:
        stage = snapshot.stage_for_depth(0).model_copy(
            update={"stage_index": 1, "ancestry": ("root", "child")}
        )
        snapshot = snapshot.model_copy(update={"exact_model_id": "root", "model_stages": (stage,)})
    route = route.model_copy(
        update={
            "snapshot": snapshot,
            "reasoning_pinned_deployment_id": route.deployment.deployment_id,
        }
    )
    wires = (
        wires[0],
        (
            GatewayWireProfile(
                dialect="openai_compatible" if wire == "openrouter" else wire,
                url="https://cache.test",
                forwards_cache_control=wire == "openrouter",
            ),
            wires[1][1],
        ),
        wires[2],
    )
    accounting = NativeAttemptAccounting(
        _RecordingLedger(), recovery_host=Host() if with_host else None
    )
    request = _marked_request()
    marker_route, marker_wires = _prefer_cache_capable_rungs(route, wires, request)
    assert marker_route is route and marker_wires is wires
    ordered, ordered_wires, placement = _affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert ordered is route and ordered_wires is wires
    assert placement.fingerprint is not None
    accounting.sticky.bind(
        placement.fingerprint, route.deployments[-1].deployment_id, ttl_seconds=60
    )
    repeated, _, _ = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert repeated is route
    surviving = select_route_deployments(route, (1, 2))
    admitted, _, _ = stage_affinity_ordered_rungs(
        surviving,
        wires[1:],
        request,
        accounting=accounting,
        authorization=snapshot.authorization,
        continuation=None,
    )
    assert route.deployment.deployment_id not in admitted.snapshot.deployment_ids
    if has_stages:
        assert admitted.snapshot.exact_model_id == "root"
        assert all(s.ancestry == ("root", "child") for s in admitted.snapshot.model_stages)


@pytest.mark.parametrize("tokens", [("transport",), ("refusal",)])
@pytest.mark.parametrize("staged", [False, True])
def test_retained_conditional_rung_cannot_discard_first_dial_routes(
    tokens: tuple[str, ...],
    staged: bool,
) -> None:
    """Prior matched failures never authorize a conditional rung on a later request."""
    route, wires = _affinity_fixture()
    request = GatewayRequest(
        surface=route.snapshot.authorization.surface,
        messages=(GatewayMessage(role="system", content="stable"),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    if staged:
        route = route.model_copy(
            update={
                "snapshot": route.snapshot.model_copy(
                    update={
                        "authorization": route.snapshot.authorization.model_copy(
                            update={"descendant_start_authorized": True}
                        ),
                        "exact_model_id": "root-model",
                        "model_stages": (route.snapshot.stage_for_depth(0),),
                    }
                )
            }
        )
    host = Host()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=Clock())
    auth = route.snapshot.authorization
    ordered, _, _ = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=auth, continuation=None
    )
    retained = ordered.deployments[-1]
    key = session_cache_key(InflightRequest(auth, ordered, request, time.monotonic() + 30))
    assert key is not None
    accounting.recovery.record_success(
        key,
        retained.deployment_id,
        host.scope_for(retained, auth.organization_id),
        cached_tokens=80,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=60,
    )
    deployments = tuple(
        d.model_copy(
            update={
                "gateway": d.gateway.model_copy(
                    update={
                        "capabilities": d.gateway.capabilities.model_copy(
                            update={"failover_only_on": tokens}
                        )
                    }
                )
            }
        )
        if d.deployment_id == retained.deployment_id
        else d
        for d in route.deployments
    )
    restricted = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    admitted, _, placement = stage_affinity_ordered_rungs(
        restricted, wires, request, accounting=accounting, authorization=auth, continuation=None
    )
    assert admitted.snapshot.deployment_ids == ordered.snapshot.deployment_ids
    assert placement.verified_warm_deployment_id is None
    assert not placement.sticky_preferred
    require_unrestricted_rung(admitted)
    conditional_only = select_route_deployments(
        restricted, (restricted.snapshot.deployment_ids.index(retained.deployment_id),)
    )
    conditional_wire = (wires[restricted.snapshot.deployment_ids.index(retained.deployment_id)],)
    remaining, _, standing = stage_affinity_ordered_rungs(
        conditional_only,
        conditional_wire,
        request,
        accounting=accounting,
        authorization=auth,
        continuation=None,
    )
    assert standing.verified_warm_deployment_id is None
    with pytest.raises(FailoverRulesError):
        require_unrestricted_rung(remaining)


@pytest.mark.parametrize("trial", [False, True], ids=["retained", "trial"])
@pytest.mark.parametrize(
    "isolation",
    ["same", "organization", "identity", "prefix", "credential", "no_host"],
)
def test_verified_session_warmth_survives_capacity_admission_without_scope_leaks(
    trial: bool,
    isolation: Literal["same", "organization", "identity", "prefix", "credential", "no_host"],
) -> None:
    """A retained or trial warm lane bypasses only the fresh threshold for its exact scope."""
    route, wires = _affinity_fixture()
    dispatch = GatewayRungDispatchPolicy(
        concurrency_bound=4, fresh_session_spill_fraction=0.5, sticky_spill_seconds=60
    )
    deployments = tuple(
        d.model_copy(update={"gateway": d.gateway.model_copy(update={"dispatch": dispatch})})
        for d in route.deployments
    )
    route = route.model_copy(
        update={
            "deployment": deployments[0],
            "fallback_deployments": deployments[1:],
            "snapshot": route.snapshot.model_copy(
                update={"model_stages": (route.snapshot.stage_for_depth(0),)}
            ),
        }
    )
    request = GatewayRequest(
        surface=route.snapshot.authorization.surface,
        messages=(
            GatewayMessage(role="system", content="stable prefix"),
            GatewayMessage(role="user", content="turn"),
        ),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    host = Host()
    clock = Clock()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    authorization = route.snapshot.authorization
    ordered, _, _ = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=authorization,
        continuation=None,
    )
    original = InflightRequest(authorization, ordered, request, time.monotonic() + 30)
    key = session_cache_key(original)
    assert key is not None
    lead, fallback, _last = ordered.deployments
    for deployment in (lead, fallback) if trial else (fallback,):
        accounting.recovery.record_success(
            key,
            deployment.deployment_id,
            host.scope_for(deployment, authorization.organization_id),
            cached_tokens=80,
            cache_write_tokens=0,
            retention_seconds=100,
            sticky_seconds=60,
        )
    if trial:
        accounting.recovery.depart(
            key,
            lead.deployment_id,
            host.scope_for(lead, authorization.organization_id),
            "local_capacity",
        )
        clock.now += 6
    warm_id = (lead if trial else fallback).deployment_id
    if isolation in ("organization", "identity"):
        authorization = authorization.model_copy(update={f"{isolation}_id": "different"})
    elif isolation == "prefix":
        request = request.model_copy(
            update={"messages": (GatewayMessage(role="system", content="changed"),)}
        )
    elif isolation == "credential":
        host.credential = "rotated"
    elif isolation == "no_host":
        accounting.recovery_host = None

    # The warm lane is at the fresh threshold, with a cold C still available.
    # Isolated requests fill every lane to expose any stale sticky bypass.
    # Same-fingerprint sticky data must not confer scoped cache standing.
    occupied = ordered.deployments[:2] if isolation == "same" else route.deployments
    for deployment in occupied:
        for _ in range(2):
            assert isinstance(
                accounting.loads.reserve(
                    rung_load_key(deployment),
                    organization_id=authorization.organization_id,
                    weight=1,
                    bound=4,
                    fair_share=False,
                ),
                str,
            )
    changed = replace(original, authorization=authorization, request=request)
    changed_key = session_cache_key(changed)
    assert changed_key is not None
    accounting.sticky.bind(changed_key.fingerprint, warm_id, ttl_seconds=60)
    admitted, _, placement = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=authorization,
        continuation=None,
    )
    assert placement.recovery_scoped
    assert placement.verified_warm_deployment_id == (warm_id if isolation == "same" else None)
    assert accounting.sticky.size() == 1
    accounting.sticky.clear(changed_key.fingerprint)
    entry = InflightRequest(
        authorization,
        admitted,
        request,
        time.monotonic() + 30,
        affinity_fingerprint=placement.fingerprint,
        verified_warm_deployment_id=placement.verified_warm_deployment_id,
        verified_warm_until_monotonic=placement.verified_warm_until_monotonic,
        recovery_scoped=placement.recovery_scoped,
        recovery_reason=placement.recovery_reason,
    )
    # A populated ordinary binding is deliberately present for isolated requests.
    if isolation != "same":
        accounting.sticky.bind(changed_key.fingerprint, warm_id, ttl_seconds=60)
    accounting.register(entry)
    started = json.loads(
        accounting.start_attempt(
            json.dumps({"request_id": authorization.request_id, "attempt_ordinal": 0})
        )
    )
    selected = admitted.deployments[started["route_depth"]]
    if isolation == "same":
        assert selected.deployment_id == warm_id
        assert accounting.rung_rate_counters() == (0, 0)
        assert accounting.sticky.size() == 0
    else:
        # Authored bounds retain their explicit overflow policy. A stale sticky
        # binding must not silently admit any lane before that disclosed overflow.
        assert accounting.rung_rate_counters() == (0, 3)
        assert accounting.rung_admission_counters() == (3, 1, 0)


@pytest.mark.parametrize("non_affinity", ["maximize_availability", "maximize_cache"])
@pytest.mark.parametrize("barrier", [0, 1, 2, None])
def test_recovery_cannot_cross_current_non_affinity_stage(
    non_affinity: FailoverMode,
    barrier: int | None,
) -> None:
    """Stale affinity history cannot skip or retain a currently non-affinity model stage."""
    route, wires = _affinity_fixture()
    deployments = tuple(
        d.model_copy(update={"exact_model_id": f"model-{depth}"})
        for depth, d in enumerate(route.deployments)
    )
    stages = tuple(
        route.snapshot.stage_for_depth(0).model_copy(
            update={
                "stage_index": depth,
                "exact_model_id": d.exact_model_id,
                "deployment_ids": (d.deployment_id,),
                "ancestry": tuple(f"model-{i}" for i in range(depth + 1)),
                "failover_mode": non_affinity if depth == barrier else "maximize_cache_affinity",
            }
        )
        for depth, d in enumerate(deployments)
    )
    authorization = route.snapshot.authorization.model_copy(
        update={"descendant_start_authorized": True}
    )
    route = route.model_copy(
        update={
            "deployment": deployments[0],
            "fallback_deployments": deployments[1:],
            "snapshot": route.snapshot.model_copy(
                update={
                    "authorization": authorization,
                    "exact_model_id": "model-0",
                    "model_stages": stages,
                }
            ),
        }
    )
    request = GatewayRequest(
        surface=authorization.surface,
        messages=(GatewayMessage(role="user", content="prefix"),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    host = Host()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    key = session_cache_key(InflightRequest(authorization, route, request, 10))
    assert key is not None
    retained = deployments[-1]
    accounting.recovery.record_success(
        key,
        retained.deployment_id,
        host.scope_for(retained, authorization.organization_id),
        cached_tokens=80,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=60,
    )
    admitted, _, placement = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=authorization,
        continuation=None,
    )
    if barrier is None:
        assert admitted.deployments == (retained,)
        assert placement.verified_warm_deployment_id == retained.deployment_id
    else:
        assert admitted is route
        assert placement.verified_warm_deployment_id is None
        assert not placement.sticky_preferred


@pytest.mark.parametrize("has_stages", [False, True])
@pytest.mark.parametrize("mode", ["maximize_availability", "maximize_cache"])
def test_recovery_ignores_stale_affinity_history_after_mode_change(
    has_stages: bool, mode: FailoverMode
) -> None:
    """A current direct or staged non-affinity pool keeps its authored leading deployment."""
    route, wires = _affinity_fixture(mode)
    if has_stages:
        route = route.model_copy(
            update={
                "snapshot": route.snapshot.model_copy(
                    update={"model_stages": (route.snapshot.stage_for_depth(0),)}
                )
            }
        )
    authorization = route.snapshot.authorization
    request = GatewayRequest(
        surface=authorization.surface,
        messages=(GatewayMessage(role="user", content="prefix"),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    host = Host()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    key = session_cache_key(InflightRequest(authorization, route, request, 10))
    assert key is not None
    retained = route.deployments[1]
    accounting.recovery.record_success(
        key,
        retained.deployment_id,
        host.scope_for(retained, authorization.organization_id),
        cached_tokens=80,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=60,
    )
    admitted, _, placement = _affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=authorization,
        continuation=None,
    )
    assert admitted is route
    assert placement.verified_warm_deployment_id is None
    assert not placement.sticky_preferred


@pytest.mark.parametrize("history", ["empty", "live", "expired", "renewed"])
@pytest.mark.parametrize("with_token_window", [False, True])
def test_recovery_tokenizes_only_retained_history_with_token_headroom_policy(
    history: Literal["empty", "live", "expired", "renewed"],
    with_token_window: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New sessions and policy-free history avoid BPE; token-window trials get one estimate."""
    route, wires = _affinity_fixture()
    dispatch = (
        GatewayRungDispatchPolicy(tokens_per_minute=100, sticky_spill_seconds=60)
        if with_token_window
        else None
    )
    deployments = tuple(
        d.model_copy(update={"gateway": d.gateway.model_copy(update={"dispatch": dispatch})})
        for d in route.deployments
    )
    route = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    authorization = route.snapshot.authorization
    request = GatewayRequest(
        surface=authorization.surface,
        messages=(GatewayMessage(role="user", content="large prefix" * 1000),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    host = Host()
    clock = Clock()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    ordered, _, _ = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    key = session_cache_key(InflightRequest(authorization, ordered, request, 10))
    assert key is not None
    lead, retained, _last = ordered.deployments
    if history != "empty":
        for deployment in (lead, retained):
            accounting.recovery.record_success(
                key,
                deployment.deployment_id,
                host.scope_for(deployment, authorization.organization_id),
                cached_tokens=80,
                cache_write_tokens=0,
                retention_seconds=100,
                sticky_seconds=60,
            )
        accounting.recovery.depart(
            key,
            lead.deployment_id,
            host.scope_for(lead, authorization.organization_id),
            "local_capacity",
        )
        clock.now += 61 if history in ("expired", "renewed") else 6
    calls: list[GatewayRequest] = []
    if history == "renewed":
        original_preflight = accounting.recovery.has_retained_history

        def concurrent_renewal(key: SessionCacheKey, *, live_only: bool = False) -> bool:
            """Renew a retained cursor immediately after the expired live preflight."""
            result = original_preflight(key, live_only=live_only)
            if live_only:
                accounting.recovery.record_success(
                    key,
                    retained.deployment_id,
                    host.scope_for(retained, authorization.organization_id),
                    cached_tokens=80,
                    cache_write_tokens=0,
                    retention_seconds=100,
                    sticky_seconds=60,
                )
            return result

        monkeypatch.setattr(accounting.recovery, "has_retained_history", concurrent_renewal)

    def estimate(value: GatewayRequest) -> int:
        """Prove estimation happens outside the non-reentrant registry lock."""
        assert accounting.recovery.has_retained_history(key)
        calls.append(value)
        return 50

    monkeypatch.setattr(native_stage_admission, "worst_case_input_tokens", estimate)
    _, _, placement = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    assert len(calls) == int(history == "live" and with_token_window)
    expected = {
        "empty": None,
        "live": "recovered_preferred_route",
        "expired": "cache_expired",
        "renewed": "retained_warm_fallback" if with_token_window else "cache_expired",
    }
    assert placement.recovery_reason == expected[history]


def test_recovery_return_requires_default_lane_headroom() -> None:
    """An unauthored saturated lane cannot gain an elective recovery trial past the worker cap."""
    route, wires = _affinity_fixture()
    deployments = tuple(
        d.model_copy(update={"gateway": d.gateway.model_copy(update={"dispatch": None})})
        for d in route.deployments
    )
    route = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    authorization = route.snapshot.authorization
    request = GatewayRequest(
        surface=authorization.surface,
        messages=(GatewayMessage(role="user", content="prefix"),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
    )
    host = Host()
    clock = Clock()
    accounting = NativeAttemptAccounting(
        _RecordingLedger(), recovery_host=host, default_lane_bound=1
    )
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    ordered, _, _ = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    key = session_cache_key(InflightRequest(authorization, ordered, request, 10))
    assert key is not None
    lead, retained, _last = ordered.deployments
    for deployment in (lead, retained):
        accounting.recovery.record_success(
            key,
            deployment.deployment_id,
            host.scope_for(deployment, authorization.organization_id),
            cached_tokens=80,
            cache_write_tokens=0,
            retention_seconds=100,
            sticky_seconds=60,
        )
    accounting.recovery.depart(
        key,
        lead.deployment_id,
        host.scope_for(lead, authorization.organization_id),
        "local_capacity",
    )
    clock.now += 6
    assert isinstance(
        accounting.loads.reserve(
            rung_load_key(lead), organization_id="other", weight=1, bound=1, fair_share=False
        ),
        str,
    )
    _, _, placement = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    assert placement.recovery_reason == "retained_warm_fallback"
    assert placement.verified_warm_deployment_id == retained.deployment_id
    assert accounting.loads.inflight(rung_load_key(lead)) == 1


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_optional_recovery_snapshot_failure_preserves_normal_admission(
    expired: bool,
    staged: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unavailable advisory observations neither fail serving nor confer cached standing."""
    route, wires = _affinity_fixture()
    deployments = tuple(
        deployment.model_copy(
            update={
                "gateway": deployment.gateway.model_copy(
                    update={
                        "capabilities": deployment.gateway.capabilities.model_copy(
                            update={"supports_streaming": True}
                        )
                    }
                )
            }
        )
        for deployment in route.deployments
    )
    route = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    if staged:
        route = route.model_copy(
            update={
                "snapshot": route.snapshot.model_copy(
                    update={"model_stages": (route.snapshot.stage_for_depth(0),)}
                )
            }
        )
    wires = tuple(
        (
            replace(
                profile,
                operational_region="region",
                credential_receipt=DispatchCredentialReceipt(uuid5(NAMESPACE_URL, "credential")),
            ),
            client,
        )
        for profile, client in wires
    )
    authorization = route.snapshot.authorization
    request = GatewayRequest(
        surface=authorization.surface,
        messages=(GatewayMessage(role="system", content="stable prefix"),),
        prompt_cache_key="session",
        maximum_output_tokens=128,
    )
    host = Host()
    clock = Clock()
    accounting = NativeAttemptAccounting(_RecordingLedger(), recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    ordered, ordered_wires, _, provider_request, _ = admitted_route_requests(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    key = session_cache_key(InflightRequest(authorization, ordered, provider_request, 10))
    assert key is not None
    retained = ordered.deployments[-1]
    binding = ordered_wires[-1][0].recovery_binding
    assert binding is not None
    accounting.recovery.record_success(
        key,
        retained.deployment_id,
        binding.scope,
        cached_tokens=80,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=60,
    )
    if expired:
        clock.now += 61
    calls = []

    def unavailable_snapshot() -> RecoverySnapshot:
        """Reject the advisory fetch with text that must never enter public or diagnostic output."""
        calls.append(True)
        raise RuntimeError("secret observer credential")

    monkeypatch.setattr(host, "snapshot", unavailable_snapshot)
    admitted, admitted_wires, public, provider, placement = admitted_route_requests(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    assert calls == [True]
    assert admitted.snapshot == ordered.snapshot
    assert admitted_wires == ordered_wires
    assert public.maximum_output_tokens == provider.maximum_output_tokens == 128
    assert placement.recovery_scoped and not placement.sticky_preferred
    assert placement.verified_warm_deployment_id is None
    assert placement.recovery_reason is None
    assert "Recovery snapshot unavailable" in caplog.text
    assert "secret observer credential" not in caplog.text


@pytest.mark.parametrize("honorer_busy", [False, True])
def test_warm_nonhonoring_recovery_cannot_skip_available_marker_wire(honorer_busy: bool) -> None:
    """Retained shim warmth cannot override same-stage explicit marker preference."""
    route, wires = _affinity_fixture()
    wires = tuple(
        (replace(profile, forwards_cache_control=index == 0), client)
        for index, (profile, client) in enumerate(wires)
    )
    request = _marked_request().model_copy(
        update={"prompt_cache_key": "session", "provider_prompt_cache_key": "xpl-session"}
    )
    host = Host()
    accounting = NativeAttemptAccounting(
        _RecordingLedger(), recovery_host=host, default_lane_bound=1
    )
    accounting.recovery = SessionRecoveryRegistry(clock=Clock())
    authorization = route.snapshot.authorization
    ordered, bound_wires, _ = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    assert bound_wires[0][0].preserves_cache_control
    key = session_cache_key(InflightRequest(authorization, ordered, request, 10))
    assert key is not None
    retained = ordered.deployments[-1]
    accounting.recovery.record_success(
        key,
        retained.deployment_id,
        host.scope_for(retained, authorization.organization_id),
        cached_tokens=80,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=60,
    )
    if honorer_busy:
        assert isinstance(
            accounting.loads.reserve(
                rung_load_key(ordered.deployment),
                organization_id="other",
                weight=1,
                bound=1,
                fair_share=False,
            ),
            str,
        )
    admitted, selected, placement = stage_affinity_ordered_rungs(
        route, wires, request, accounting=accounting, authorization=authorization, continuation=None
    )
    if honorer_busy:
        assert admitted.deployment.deployment_id == retained.deployment_id
        assert placement.verified_warm_deployment_id == retained.deployment_id
    else:
        assert admitted.snapshot == ordered.snapshot and selected == bound_wires
        assert placement.verified_warm_deployment_id is None


@pytest.mark.parametrize("narrowing", ["recovery", "ordinary", "none"])
def test_final_selected_marker_profile_controls_public_omission(narrowing: str) -> None:
    """Final wire omissions are disclosed once without changing request identity or payload."""
    route, wires = _affinity_fixture()
    deployments = tuple(
        d.model_copy(
            update={
                "gateway": d.gateway.model_copy(
                    update={
                        "capabilities": d.gateway.capabilities.model_copy(
                            update={"supports_streaming": True}
                        ),
                    }
                )
            }
        )
        for d in route.deployments
    )
    route = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    wires = tuple(
        (
            replace(
                profile,
                forwards_cache_control=index == 0,
                operational_region="region",
                credential_receipt=DispatchCredentialReceipt(uuid5(NAMESPACE_URL, "credential")),
            ),
            client,
        )
        for index, (profile, client) in enumerate(wires)
    )
    request = _marked_request().model_copy(
        update={
            "maximum_output_tokens": 128,
            "prompt_cache_key": "session",
            "ignored_parameters": ("existing->disclosure",),
        }
    )
    original_identity = canonical_request_sha256(request)
    host = Host()
    accounting = NativeAttemptAccounting(
        _RecordingLedger(), recovery_host=host, default_lane_bound=1
    )
    accounting.recovery = SessionRecoveryRegistry(clock=Clock())
    auth = route.snapshot.authorization
    ordered, bound, _, shaped, _ = admitted_route_requests(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=auth,
        continuation=None,
    )
    assert bound[0][0].preserves_cache_control
    if narrowing == "recovery":
        key = session_cache_key(InflightRequest(auth, ordered, shaped, 10))
        assert key is not None
        retained = ordered.deployments[-1]
        binding = bound[-1][0].recovery_binding
        assert binding is not None
        accounting.recovery.record_success(
            key,
            retained.deployment_id,
            binding.scope,
            cached_tokens=80,
            cache_write_tokens=0,
            retention_seconds=100,
            sticky_seconds=60,
        )
        assert isinstance(
            accounting.loads.reserve(
                rung_load_key(ordered.deployment),
                organization_id="other",
                weight=1,
                bound=1,
                fair_share=False,
            ),
            str,
        )
    elif narrowing == "ordinary":
        route = select_route_deployments(ordered, (len(ordered.deployments) - 1,))
        wires = (bound[-1],)
    selected, selected_wires, public, provider, _ = admitted_route_requests(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=auth,
        continuation=None,
    )
    payload = dialect_stream_payload(selected_wires[0][0], provider)
    omission = f"messages.content.cache_control{CACHE_CONTROL_NOT_FORWARDED_SUFFIX}"
    assert public.ignored_parameters.count(omission) == int(narrowing != "none")
    assert "existing->disclosure" in public.ignored_parameters
    assert ("cache_control" in json.dumps(payload)) is (narrowing == "none")
    assert canonical_request_sha256(request) == original_identity
    if narrowing != "none":
        assert len(selected.deployments) == 1 and not selected_wires[0][0].preserves_cache_control


@pytest.mark.parametrize("other_session", [False, True])
def test_real_accounting_capacity_shed_retains_provider_recovery_requirement(
    other_session: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-admitted request's capacity shed cannot erase a below-breaker transport failure."""
    monkeypatch.setattr(NativeAttemptAccounting, "_sweep_loop", lambda self: None)
    route, wires = _affinity_fixture()
    deployments = tuple(
        d.model_copy(
            update={
                "gateway": d.gateway.model_copy(
                    update={
                        "cache_retention_seconds": 100,
                        "dispatch": GatewayRungDispatchPolicy(
                            concurrency_bound=1, sticky_spill_seconds=60
                        ),
                    }
                )
            }
        )
        for d in route.deployments
    )
    route = route.model_copy(
        update={"deployment": deployments[0], "fallback_deployments": deployments[1:]}
    )
    host, clock, ledger = Host(), Clock(), _RecordingLedger()
    accounting = NativeAttemptAccounting(ledger, recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    request = GatewayRequest(
        surface=route.snapshot.authorization.surface,
        messages=(GatewayMessage(role="system", content="stable prefix"),),
        prompt_cache_key="session",
        provider_prompt_cache_key="xpl-session",
        maximum_output_tokens=16,
    )
    failure = {
        "failure_class": "transport",
        "safe_message": "transport failed",
        "failover_eligible": True,
    }

    def admit(name: str, other: bool = False) -> tuple[InflightRequest, AffinityPlacement]:
        """Freeze one actual stage selection before concurrent requests change history."""
        req = (
            request.model_copy(
                update={"prompt_cache_key": "other", "provider_prompt_cache_key": "xpl-other"}
            )
            if other
            else request
        )
        auth = route.snapshot.authorization.model_copy(
            update={"request_id": name, "deadline_monotonic": time.monotonic() + 60}
        )
        ordered, bound, placement = stage_affinity_ordered_rungs(
            route,
            wires,
            req,
            accounting=accounting,
            authorization=auth,
            continuation=None,
        )
        bindings = {
            d.deployment_id: p.recovery_binding
            for d, (p, _) in zip(ordered.deployments, bound, strict=True)
            if p.recovery_binding is not None
        }
        entry = InflightRequest(
            auth,
            ordered,
            req,
            time.monotonic() + 60,
            recovery_bindings=bindings,
            recovery_scoped=True,
        )
        accounting.register(entry)
        return entry, placement

    def start(entry: InflightRequest, retry: bool = False) -> dict[str, str | int]:
        """Reserve through real accounting without opening the resulting provider wire."""
        data = {
            "request_id": entry.authorization.request_id,
            "attempt_ordinal": entry.total_attempts,
        }
        if retry:
            data.update({"current_depth": 0, "failure": failure})
        return json.loads(accounting.start_attempt(json.dumps(data)))

    def settle(
        entry: InflightRequest,
        started: dict[str, str | int],
        failed: bool = False,
        finalize: bool = True,
    ) -> None:
        """Settle authentic attempt identity with deterministic provider observations."""
        accounting.settle(
            json.dumps(
                {
                    "request_id": entry.authorization.request_id,
                    "attempt_id": started["attempt_id"],
                    "outcome": "failed" if failed else "completed",
                    "failure": failure if failed else None,
                    "usage": None
                    if failed
                    else {"input_tokens": 100, "output_tokens": 5, "cached_input_tokens": 80},
                    "finalize": finalize,
                    "opened": not failed,
                }
            )
        )

    warmed, _ = admit("warm")
    lead, fallback = warmed.route.deployments[:2]
    settle(warmed, start(warmed))
    first, _ = admit("first", other=other_session)
    queued, _ = admit("queued")
    if not other_session:
        settle(first, start(first), failed=True, finalize=False)
        settle(first, start(first, retry=True))
    held = accounting.loads.reserve(
        rung_load_key(lead), organization_id="competitor", weight=1, bound=1, fair_share=False
    )
    assert isinstance(held, str)
    shed = start(queued)
    assert ledger.started[-1]["deployment_id"] == fallback.deployment_id
    assert ledger.started[-1]["dispatch_reason"] == "queue_bound"
    settle(queued, shed)
    accounting.loads.release_ticket(held)
    if other_session:
        clock.now += 1
        settle(first, start(first), failed=True)
    clock.now += 6
    _, chosen = admit("after")
    assert chosen.recovery_reason == "retained_warm_fallback"
    assert chosen.verified_warm_deployment_id == fallback.deployment_id


def test_no_recovery_host_skips_prefix_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-host route still schedules affinity without hashing an unused recovery prefix."""
    route, wires = _affinity_fixture()
    request = GatewayRequest(
        surface=route.snapshot.authorization.surface,
        messages=(GatewayMessage(role="user", content="prefix"),),
    )
    accounting = NativeAttemptAccounting(_RecordingLedger())

    def unexpected_prefix(_request: GatewayRequest) -> str | None:
        """Reject recovery-specific work when no host can establish credential scope."""
        pytest.fail("no-host admission hashed the recovery prefix")

    monkeypatch.setattr(native_stage_admission, "recovery_prefix_digest", unexpected_prefix)
    admitted, _, placement = stage_affinity_ordered_rungs(
        route,
        wires,
        request,
        accounting=accounting,
        authorization=route.snapshot.authorization,
        continuation=None,
    )
    assert set(admitted.snapshot.deployment_ids) == set(route.snapshot.deployment_ids)
    assert placement.fingerprint is not None
    assert placement.verified_warm_deployment_id is None

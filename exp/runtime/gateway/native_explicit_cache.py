"""Selected-attempt authority callbacks for opt-in Google cache resources.

Rust owns provider HTTP. These callbacks expose no resource operation until the
host durably claims and reserves it. An abandoned callback never grants a second
creator; unknown claims remain the host's durable accounting responsibility.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayRequest,
)
from exp.runtime.gateway.explicit_cache import (
    AutomaticCacheHost,
    ExplicitCacheHost,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX
from exp.runtime.models.providers.google_cache import (
    GoogleCachePlan,
    build_automatic_google_cache_plans,
    build_google_cache_plan,
)
from exp.runtime.models.providers.protocol import NativeWireClient

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
            For automatic bindings this is only a placeholder until the host selects.
        automatic_plans: Gateway-offered whole-message prefixes, empty for marked caches.
    """

    deployment: ExactModelDeployment
    profile: GatewayWireProfile
    plan: GoogleCachePlan
    automatic_plans: tuple[GoogleCachePlan, ...] = ()


@dataclass(repr=False)
class NativeCacheState:
    """Frozen admission plans transferred to the Rust request owner.

    Attributes:
        bindings: Admitted cache plan at each route depth, absent when unsupported.
        automatic: Whether the host owns optional automatic prefix selection.
    """

    bindings: tuple[NativeCacheBinding | None, ...]
    automatic: bool = False


def validate_explicit_cache_host(host: ExplicitCacheHost | None) -> ExplicitCacheHost | None:
    """Refuse an incomplete host interface before serving or authorizing cache spend."""
    if host is not None and any(
        not callable(getattr(host, method, None))
        for method in ("authority", "claim", "record", "record_renewal")
    ):
        raise ValueError(
            "explicit cache host must provide authority, claim, record and record_renewal methods"
        )
    return host


def bind_explicit_cache(
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
    """Retain pure plans without reserving or contacting any provider at admission.

    Only marked, representable native Google text prefixes on non-ZDR requests
    qualify, and nothing qualifies for an organization with prompt caching off.
    The actual selected attempt rechecks tenant allowance and credential
    generation through the injected host before any cache operation.
    """
    if (
        host is None
        or authorization.zdr_requested
        or authorization.prompt_caching_disabled
        or provider_request.tool_search is not None
    ):
        return None, public_request
    bindings: list[NativeCacheBinding | None] = []
    for deployment, (profile, _client), wire in zip(
        deployments, resolved_wires, wire_route, strict=True
    ):
        payload = wire.get("upstream_payload")
        plan = (
            build_google_cache_plan(profile, provider_request, payload)
            if isinstance(payload, dict) and not wire.get("zdr_constrained") and not automatic
            else None
        )
        candidates = (
            build_automatic_google_cache_plans(profile, provider_request, payload)
            if automatic
            and plan is None
            and isinstance(payload, dict)
            and not wire.get("zdr_constrained")
            else ()
        )
        plan = plan or (candidates[0] if candidates else None)
        bindings.append(
            None if plan is None else NativeCacheBinding(deployment, profile, plan, candidates)
        )
        if plan is not None:
            wire["explicit_cache"] = True
            if candidates:
                wire["automatic_cache"] = True
    if not any(binding is not None for binding in bindings):
        return None, public_request
    if automatic:
        return NativeCacheState(tuple(bindings), automatic=True), public_request
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


class NativeExplicitCacheMixin:
    """Administrative cache operations use the admitted native financial owner."""

    _accounting: NativeAttemptAccounting

    def prepare_explicit_cache(self, argument: str) -> str:
        """Plan and reserve a resource through the native owner."""
        return self._accounting.call("prepare_explicit_cache", argument)

    def renew_explicit_cache(self, argument: str) -> str:
        """Renew an admitted resource through the native owner."""
        return self._accounting.call("renew_explicit_cache", argument)

    def finish_explicit_cache(self, argument: str) -> str:
        """Commit resource creation through the native owner."""
        return self._accounting.call("finish_explicit_cache", argument)


def validate_cache_hosts(
    explicit: ExplicitCacheHost | None, automatic: AutomaticCacheHost | None
) -> ExplicitCacheHost | None:
    """Choose exactly one valid host before admission can plan paid resources."""
    if explicit is not None and automatic is not None:
        raise ValueError("choose one explicit or automatic cache host")
    if automatic is not None and not callable(getattr(automatic, "select_prefix", None)):
        raise ValueError("automatic cache host must implement select_prefix")
    if automatic is not None:
        native = importlib.import_module("exp_gateway_native")
        if getattr(native, "AUTOMATIC_VERTEX_CACHE_CONTRACT_VERSION", None) != 1:
            raise ValueError("automatic caching requires the matching native cache contract")
    return validate_explicit_cache_host(automatic or explicit)

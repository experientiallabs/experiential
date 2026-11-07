"""Request-class throttles on self-hosted rungs: pace, per-class cap, and the vLLM class.

A host may vouch a request class on ``AuthorizationSnapshot.priority_class``
(0 free account .. 8 top plan paid, higher first) plus optional class
throttles. Every lever here acts ONLY on a self-hosted rung: a rung whose
authored dispatch policy sets ``upstream_priority``, the operator's statement
that it is our own vLLM server (the platform projects it on Experiential Cloud
lanes only, and it already gates vLLM's ``priority`` body field in
``native_rungs``). A third-party, BYOK or tenant rung never sets it, so on
those rungs every function here is a no-op.

Which lever frees capacity:

* vLLM priority by class (``native_rungs``): the server schedules and preempts
  by class, so higher tiers really run first on the GPU.
* The per-class in-flight cap (``self_hosted_class_concurrency``): a class
  may hold at most that many in-flight dispatches on one self-hosted rung per
  worker; past it the rung sheds the request sideways, leaving GPU slots for
  other classes. This also frees capacity.
* The output pace (``output_tokens_per_second``): delivery only. The GPU still
  generates at full speed; the data plane just releases the stream no faster
  than the rate. With ``output_pace_saturation`` set it applies only while the
  rung's in-flight load is at or above that fraction of its bound.

The ``:fast`` spelling (``AuthorizationSnapshot.fast``) asks for the priority
processing tier at the frozen priority card (``fast_request``): a request no
rung can serve at that tier is refused, never served at standard
(``require_fast_path``). On a self-hosted rung it runs above every class (vLLM
priority ``FAST_UPSTREAM_PRIORITY``, permit class ``FAST_PRIORITY_CLASS``), is
exempt from class caps and the pace, holds the upper half of the rung's bound
outside fair share, and while it is in flight other traffic holds only the
lower half (``rung_admission``), so the decode batch it shares stays small.
We serve that tier ourselves, so its settlement is confirmed at the requested
card (``tier_settlement``). Every lever needs the rung's priority card.

Nothing is set by default, and an attempt with no pace serializes exactly as
before (``started_attempt_payload``).
"""

from __future__ import annotations

import json
import math

from exp.common.models.catalog import BillingSource
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.native_execution import InflightRequest, rung_load_key
from exp.runtime.gateway.native_service_tiers import TierSettlementInputs
from exp.runtime.gateway.rung_admission import RungLoadRegistry
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission
from exp.runtime.models.providers.errors import ProviderCapabilityError

TOP_PRIORITY_CLASS = 8
"""The highest ``AuthorizationSnapshot.priority_class``; vLLM gets ``8 - class``.

A host that sends a request class (0 free account .. 8 top plan paid)
replaces the three-level admission map, so the top class runs at vLLM
priority 0 and a free account at 8.
"""

FAST_PRIORITY_CLASS = TOP_PRIORITY_CLASS + 1
"""The permit-queue class of a ``:fast`` request, above the top request class."""

FAST_UPSTREAM_PRIORITY = -1
"""vLLM ``priority`` for a ``:fast`` request: below every ``8 - class`` (lowest runs first)."""

NO_FAST_PATH_DETAIL = (
    "This model has no fast path: none of its lanes offers a faster processing tier. "
    "Remove the :fast suffix, or choose a model that lists a :fast variant."
)


def is_self_hosted_rung(deployment: ExactModelDeployment) -> bool:
    """Whether the rung is our own vLLM server (authored ``upstream_priority``).

    A platform-funded rung only: a customer's own key or endpoint is never
    self-hosted, whatever its dispatch policy carries, so class throttles and
    pacing are a hard no-op there.
    """
    policy = deployment.gateway.dispatch
    return (
        deployment.billing_source is BillingSource.HOST_MANAGED
        and policy is not None
        and policy.upstream_priority
    )


def rung_saturation(loads: RungLoadRegistry, deployment: ExactModelDeployment) -> float | None:
    """The rung's in-flight load as a fraction of its bound on this worker.

    On a rung with a ``:fast`` reserve it is the regular count over the
    regular share (``bound`` minus the reserve).

    The bound is the rung's authored ``concurrency_bound``, else the worker's
    default lane share; ``None`` when the rung is unbounded (no saturation
    signal exists, so a threshold-gated pace never applies there).
    """
    policy = deployment.gateway.dispatch
    bound = (
        policy.concurrency_bound
        if policy is not None and policy.concurrency_bound is not None
        else loads.default_bound
    )
    if bound is None:
        return None
    key = rung_load_key(deployment)
    # Under a :fast reserve the paced traffic lives in the regular share, so
    # saturation is the regular count over that share and still reaches 1.0.
    reserve = fast_reserve_slots(deployment)
    if reserve is not None:
        return (loads.inflight(key) - loads.fast_inflight(key)) / (bound - reserve)
    return loads.inflight(key) / bound


def attempt_output_pace(
    entry: InflightRequest, deployment: ExactModelDeployment, loads: RungLoadRegistry
) -> float | None:
    """The pace for one reserved attempt of ``entry`` (see ``output_pace``)."""
    return output_pace(entry.authorization, deployment, loads)


def output_pace(
    authorization: AuthorizationSnapshot,
    deployment: ExactModelDeployment,
    loads: RungLoadRegistry,
) -> float | None:
    """The pace to apply to a reserved attempt, or ``None`` to stream unpaced.

    Evaluated once, right after the reservation (which the in-flight count
    already includes): a self-hosted rung, a host-set pace, and, when the host
    set a saturation threshold, the rung at or above it.
    """
    pace = authorization.output_tokens_per_second
    if pace is None or fast_on_rung(authorization, deployment):
        return None
    if not is_self_hosted_rung(deployment):
        return None
    threshold = authorization.output_pace_saturation
    if threshold is None:
        return pace
    saturation = rung_saturation(loads, deployment)
    return pace if saturation is not None and saturation >= threshold else None


def class_concurrency_cap(
    authorization: AuthorizationSnapshot, deployment: ExactModelDeployment
) -> tuple[int, int] | None:
    """``(class, cap)`` for a self-hosted rung under a host-set class cap, else ``None``."""
    cap = authorization.self_hosted_class_concurrency
    request_class = authorization.priority_class
    if (
        cap is None
        or request_class is None
        or fast_on_rung(authorization, deployment)
        or not is_self_hosted_rung(deployment)
    ):
        return None
    return request_class, cap


def started_attempt_payload(attempt_id: str, route_depth: int, pace: float | None) -> str:
    """The ``start_attempt`` reply; the pace key appears only on a paced attempt."""
    payload: dict[str, object] = {"attempt_id": attempt_id, "route_depth": route_depth}
    if pace is not None:
        payload["output_tokens_per_second"] = pace
    return json.dumps(payload, separators=(",", ":"))


def admission_priority_class(authorization: AuthorizationSnapshot) -> int | None:
    """The class the permit queue orders by: ``FAST_PRIORITY_CLASS`` for ``:fast``."""
    return FAST_PRIORITY_CLASS if authorization.fast else authorization.priority_class


def upstream_priority(
    authorization: AuthorizationSnapshot, deployment: ExactModelDeployment, by_level: dict[int, int]
) -> int:
    """vLLM ``priority`` on a self-hosted rung: fast, else ``8 - class``, else the level map."""
    if fast_on_rung(authorization, deployment):
        return FAST_UPSTREAM_PRIORITY
    if authorization.priority_class is not None:
        return TOP_PRIORITY_CLASS - authorization.priority_class
    return by_level[authorization.priority_admission]


def fast_on_rung(authorization: AuthorizationSnapshot, deployment: ExactModelDeployment) -> bool:
    """Whether a ``:fast`` request is reserving our own GPU at its priority card.

    Only a self-hosted rung that carries a priority card: the resource levers
    always come with the priority price, never at the standard one.
    """
    return (
        authorization.fast
        and is_self_hosted_rung(deployment)
        and deployment.gateway.prices.priority is not None
    )


def priority_tier_attested(deployment: ExactModelDeployment) -> bool:
    """Whether the host attests this carded, platform-funded rung serves the priority tier.

    The third-party counterpart of a self-hosted carded rung: the provider
    reports no tier, and the host sells the rung only as ``:fast``
    (``GatewayRungDispatchPolicy.priority_tier_attested``). A customer's own
    key or endpoint never attests, and a rung without a priority card has no
    tier to attest.
    """
    policy = deployment.gateway.dispatch
    return (
        deployment.billing_source is BillingSource.HOST_MANAGED
        and policy is not None
        and policy.priority_tier_attested
        and deployment.gateway.prices.priority is not None
    )


def fast_reserve_slots(deployment: ExactModelDeployment) -> int | None:
    """In-flight slots per worker kept for ``:fast`` on this rung, or ``None``.

    Only a self-hosted rung with a priority card and an authored bound keeps a
    reserve (``GatewayRungDispatchPolicy.fast_reserve_fraction``): the regular
    share is ``(1 - fraction) * bound`` rounded half up, at least one, and the
    reserve is the rest. A bound too small to split (1, or 2 at 0.25) keeps no
    reserve rather than reserving everything.
    """
    policy = deployment.gateway.dispatch
    if (
        policy is None
        or policy.fast_reserve_fraction is None
        or policy.concurrency_bound is None
        or not is_self_hosted_rung(deployment)
        or deployment.gateway.prices.priority is None
    ):
        return None
    bound = policy.concurrency_bound
    regular = max(1, math.floor((1 - policy.fast_reserve_fraction) * bound + 0.5))
    return bound - regular if regular < bound else None


def fast_request(request: GatewayRequest, authorization: AuthorizationSnapshot) -> GatewayRequest:
    """A ``:fast`` request asks for the priority processing tier; others pass unchanged."""
    if not authorization.fast:
        return request
    return request.model_copy(update={"service_tier": "priority"})


def tier_authorization(
    authorization: AuthorizationSnapshot, request: GatewayRequest
) -> AuthorizationSnapshot:
    """An admitted explicit priority tier (``service_tier`` priority or fast) is ``:fast``.

    On our own carded GPU it then runs with the fast levers and settles at the
    priority card it was held at: vLLM reports no tier, so without this it
    would hold at the card and never settle. Elsewhere ``fast_on_rung`` keeps
    the flag inert.
    """
    if authorization.fast or request.service_tier != "priority":
        return authorization
    return authorization.model_copy(update={"fast": True})


def require_fast_path(request: GatewayRequest, authorization: AuthorizationSnapshot) -> None:
    """Refuse a ``:fast`` request no rung serves at the priority tier (never bill it at standard).

    Raises:
        ProviderCapabilityError: Admission dropped the priority tier for lack of a rung.
    """
    if authorization.fast and request.service_tier != "priority":
        raise ProviderCapabilityError(capability="service_tier", detail=NO_FAST_PATH_DETAIL)


def tier_settlement(entry: InflightRequest, attempt_id: str) -> TierSettlementInputs:
    """The settlement's tier keywords: the frozen admission, and whether we served it.

    A self-hosted rung is our own server: it reports no processing tier, and the
    tier it ran at is ours to state, so its settlement is confirmed at the
    admitted card (``native_service_tiers.settlement_kwarg``). A rung whose host
    attests it is the priority tier (``priority_tier_attested``) is confirmed
    the same way: the provider reports no tier, and the lane is the tier.
    """
    admission: GatewayServiceTierAdmission | None = entry.attempt_service_tiers.get(attempt_id)
    depth = entry.attempt_depths.get(attempt_id)
    rung = entry.route.deployments[depth] if depth is not None else None
    self_served = (
        admission is not None
        and admission.requested == "priority"
        and rung is not None
        and (
            fast_on_rung(entry.authorization, rung)
            or (entry.authorization.fast and priority_tier_attested(rung))
        )
    )
    return {"service_tier": admission, "self_served": self_served}

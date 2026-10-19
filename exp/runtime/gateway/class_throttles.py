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

Nothing is set by default, and an attempt with no pace serializes exactly as
before (``started_attempt_payload``).
"""

from __future__ import annotations

import json

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.native_execution import InflightRequest, rung_load_key
from exp.runtime.gateway.rung_admission import RungLoadRegistry


def is_self_hosted_rung(deployment: ExactModelDeployment) -> bool:
    """Whether the rung is our own vLLM server (authored ``upstream_priority``)."""
    policy = deployment.gateway.dispatch
    return policy is not None and policy.upstream_priority


def rung_saturation(loads: RungLoadRegistry, deployment: ExactModelDeployment) -> float | None:
    """The rung's in-flight load as a fraction of its bound on this worker.

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
    return loads.inflight(rung_load_key(deployment)) / bound


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
    if pace is None or not is_self_hosted_rung(deployment):
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
    if cap is None or request_class is None or not is_self_hosted_rung(deployment):
        return None
    return request_class, cap


def started_attempt_payload(attempt_id: str, route_depth: int, pace: float | None) -> str:
    """The ``start_attempt`` reply; the pace key appears only on a paced attempt."""
    payload: dict[str, object] = {"attempt_id": attempt_id, "route_depth": route_depth}
    if pace is not None:
        payload["output_tokens_per_second"] = pace
    return json.dumps(payload, separators=(",", ":"))

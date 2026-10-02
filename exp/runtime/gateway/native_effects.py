"""Conservative admission facts used to certify failed work without paid effects."""

from __future__ import annotations

from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy


def admission_without_effects(
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    policy: GuardrailPolicy | None,
) -> bool:
    """Attest only direct admissions without search or input-classifier prework.

    Args:
        authorization: Trusted frozen target, never a caller header.
        request: Canonical request before web-search shaping can remove its declaration.
        policy: Identity-bound input policy; any input check is conservatively excluded.

    Returns:
        Whether completing admission cannot have performed paid work before a model attempt.
        This is not a durable certificate; terminal settlement also proves zero attempts.
    """
    return (
        isinstance(authorization.target, DirectTarget)
        and request.web_search is None
        and (policy is None or not policy.input_checks)
    )

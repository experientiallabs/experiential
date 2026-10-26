"""Admission-time deadness of route rungs for the native control plane.

A rung is dead at admission when its deployment cannot be resolved for dispatch
right now (a missing credential, connection drift, or capability drift). This
module owns how such a rung is named, how it feeds the health circuit, how an
accepted request whose every rung was dead is closed, and the operator log
lines that tell one cause from another, since the caller only ever sees the
retryable 503.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayFailureClass
from exp.runtime.models import ModelConnectionError
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.models.providers.errors import ProviderCapabilityError

_logger = logging.getLogger(__name__)

# Resolve-time deadness that a frozen route narrows past at admission instead
# of failing the whole request. A missing credential, connection drift, or
# capability drift means the deployment cannot be dispatched right now; it is an
# operational outage, not a request fault, so the route narrows past the rung
# and the rung's health circuit is fed like any runtime failure so it recovers
# automatically when it heals. Operator-*disabled* deployments never reach here:
# the catalog drops a disabled deployment from the live route on its ~15s
# refresh, so this path only ever sees operational deadness that should recover.
ADMISSION_DEAD_ERRORS = (ModelConnectionError, ModelCredentialError, ProviderCapabilityError)

UNAVAILABLE_AT_ADMISSION = "every certified deployment was unavailable at admission"
"""Escalation reason when every rung of a route was dead at admission.

The data plane matches this exact text (``respond.rs``) to answer the
retryable 503 ``unavailable_route``; the ledger records it as ``unavailable``.
"""


@dataclass(frozen=True)
class DeadRung:
    """One route deployment that could not be resolved for dispatch at admission."""

    index: int
    deployment: ExactModelDeployment
    failure: GatewayFailure
    cause: str = ""
    """Operator-only name of what failed (the exception type, plus its message
    for a connection or capability drift); never reaches a caller."""


def dead_rung(index: int, deployment: ExactModelDeployment, exc: Exception) -> DeadRung:
    """Name one admission deadness for the health circuit and the operator log.

    Args:
        index: The rung's position in the route.
        deployment: The deployment that failed to resolve.
        exc: One of ``ADMISSION_DEAD_ERRORS``.

    Returns:
        The dead rung with its circuit failure and its operator-only cause.
    """
    # A credential error keeps only its type: its message may name the secret.
    if isinstance(exc, ModelCredentialError):
        cause = type(exc).__name__
    else:
        cause = f"{type(exc).__name__}: {exc}"[:300]
    return DeadRung(index, deployment, admission_dead_failure(exc), cause)


def admission_dead_failure(exc: Exception) -> GatewayFailure:
    """Classify one admission-time deadness into a health-circuit failure.

    A missing credential mirrors a runtime auth rejection (a hard failure that
    opens the circuit at once); a connection or capability drift mirrors a
    runtime transport failure (an operational failure that opens after the
    circuit threshold). Both stay honest by feeding the same circuit that
    runtime failures do, so recovery is the existing cooldown plus half-open
    probe and never a permanent blacklist.
    """
    match exc:
        case ModelCredentialError():
            return GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
                safe_message=(
                    "the resolved deployment had no usable credential at admission; "
                    "failing over to the next deployment"
                ),
            )
        case _:
            return GatewayFailure(
                failure_class=GatewayFailureClass.TRANSPORT,
                safe_message=(
                    "the resolved deployment was unavailable at admission; "
                    "failing over to the next deployment"
                ),
            )


def log_dead_rungs(
    authorization: AuthorizationSnapshot,
    dead: Sequence[DeadRung],
    *,
    fallback_available: bool,
) -> None:
    """Warn operators about admission-dead rungs the caller cannot see.

    A dead lead masked by a live fallback is named so a silent lane outage
    shows up; a route with no live rung logs every rung's cause, which is what
    tells a lost credential from a catalog swap behind the caller's 503.

    Args:
        authorization: The request's authorization snapshot.
        dead: Every rung dead at admission, in route order.
        fallback_available: Whether any rung of the route stayed dispatchable.
    """
    lead = next((rung for rung in dead if rung.index == 0), None)
    if lead is not None and fallback_available:
        _logger.warning(
            "gateway admission skipped the lead rung for alias %r: served off a "
            "fallback because deployment %r (provider %r) was dead at admission",
            authorization.alias,
            lead.deployment.deployment_id,
            lead.deployment.provider,
        )
    if dead and not fallback_available:
        _logger.warning(
            "gateway admission found every rung dead",
            extra={
                "operation": "admission_dead_route",
                "request_id": authorization.request_id,
                "alias": authorization.alias,
                "alias_revision_id": authorization.alias_revision_id,
                "catalog_sha256": authorization.catalog_sha256,
                "dead_rungs": [
                    {
                        "deployment_id": rung.deployment.deployment_id,
                        "provider": rung.deployment.provider,
                        "cause": rung.cause,
                    }
                    for rung in dead
                ],
            },
        )


def escalation_failure(authorization: AuthorizationSnapshot, reason: str) -> GatewayFailure:
    """Log one admission escalation and return the failure its ledger row records.

    A route whose every rung was dead at admission is a transient routing state
    the data plane already answers as a retryable 503, so its ledger row says
    ``unavailable`` rather than an internal servability fault. Every escalation
    names its reason in the operator log, because the caller-facing message
    never does.

    Args:
        authorization: The accepted request's authorization snapshot.
        reason: The escalation reason handed to the data plane.

    Returns:
        The failure that closes the accepted request.
    """
    _logger.warning(
        "gateway admission escalated an accepted request",
        extra={
            "operation": "admission_escalation",
            "request_id": authorization.request_id,
            "alias": authorization.alias,
            "alias_revision_id": authorization.alias_revision_id,
            "catalog_sha256": authorization.catalog_sha256,
            "reason": reason,
        },
    )
    if reason == UNAVAILABLE_AT_ADMISSION:
        return GatewayFailure(
            failure_class=GatewayFailureClass.UNAVAILABLE,
            safe_message=(
                "every deployment of the authorized route was unavailable at admission; "
                "retry the request"
            ),
        )
    return GatewayFailure(
        failure_class=GatewayFailureClass.INTERNAL,
        safe_message="the native engine cannot serve the authorized route",
    )

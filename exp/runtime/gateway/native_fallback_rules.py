"""Per-rung conditional failover: the control-plane half of ``failover_only_on``.

A deployment authored with ``failover_only_on`` (``GatewayDeploymentCapabilities``)
is a FAILOVER-ONLY rung: it never takes a request's first dial and is claimed
as a successor only when the failure the ladder is walking from spells one of
its tokens (a failover-eligible failure class by wire name, ``refusal`` for any
provider refusal, or ``refusal:<reason>`` for one bounded category). A rung with
no set is unrestricted and behaves exactly as before. The reference case is a
customer's own OpenAI key enrolled in a trusted-access program: normal traffic
stays on the house rungs, and only a house rung's ``refusal:cyber_policy``
dials the customer's key.

The data plane mirrors these rules (``waterfall/fallback_rules.rs``): it
decides whether a successor is possible before asking the control plane and
refuses a reservation that violates them, so both halves must agree. The
control plane alone chooses the depth and records why a rule-carrying rung was
dialed (``fallback_reason = failover_only_on:<token>``).
"""

from __future__ import annotations

from exp.common.models.failover_tokens import REFUSAL_TOKEN, REFUSAL_TOKEN_PREFIX
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, GatewayRefusalReason
from exp.runtime.gateway.routing import GatewayRoute

FallbackRules = tuple[frozenset[str] | None, ...]
"""One entry per route depth: the rung's token set, or ``None`` for an unrestricted rung."""

FALLBACK_REASON_PREFIX = "failover_only_on:"
"""Prefix of the ledger ``fallback_reason`` an attempt on a rule-matched rung records."""


class FailoverRulesError(RuntimeError):
    """The admitted route carries no rung that may take the first dial.

    Raised at admission, before any reservation, when EVERY surviving rung is
    failover-only: nothing could ever be dialed first, so the request fails
    closed with a named internal error instead of an exhausted ladder that
    dialed nothing.
    """


def rung_rules(deployment: ExactModelDeployment) -> frozenset[str] | None:
    """Return one deployment's authored token set, or ``None`` when unrestricted."""
    tokens = deployment.gateway.capabilities.failover_only_on
    return None if tokens is None else frozenset(tokens)


def route_fallback_rules(route: GatewayRoute) -> FallbackRules:
    """Return the per-depth token sets of ``route``, aligned with its deployments."""
    return tuple(rung_rules(deployment) for deployment in route.deployments)


def failure_token(failure: GatewayFailure) -> str:
    """Spell the token one failure matches: its class name, or ``refusal:<reason>``.

    A refusal that names no reason spells ``refusal:unspecified``, so a rung
    that opted into one specific category never takes an unnamed refusal.
    """
    if failure.failure_class is GatewayFailureClass.REFUSAL:
        reason = failure.refusal_reason or GatewayRefusalReason.UNSPECIFIED
        return f"{REFUSAL_TOKEN_PREFIX}{reason.value}"
    return failure.failure_class.value


def matched_token(rules: frozenset[str] | None, failure: GatewayFailure | None) -> str | None:
    """Return the failure's token when a rule-carrying rung may take it, else ``None``.

    Unrestricted rungs (``rules is None``) and first dials (``failure is None``)
    never match: the former need no rule, the latter are never a failover. The
    bare ``refusal`` token matches a refusal of any reason; the returned token
    is always the failure's own precise spelling, so the ledger names which
    refusal (or class) sent the request to the rung.
    """
    if rules is None or failure is None:
        return None
    token = failure_token(failure)
    if token in rules:
        return token
    if failure.failure_class is GatewayFailureClass.REFUSAL and REFUSAL_TOKEN in rules:
        return token
    return None


def eligible_depths(
    rules: FallbackRules,
    start: int,
    failure: GatewayFailure | None,
    *,
    unrestricted: bool = True,
) -> tuple[int, ...]:
    """Return the depths at or after ``start`` a dial may claim, in route order.

    An unrestricted rung is eligible whenever ``unrestricted`` is true (the
    ladder's class gate for a failover, always for a first dial or a sideways
    shed); a rule-carrying rung is eligible only when its set matches
    ``failure``. With no rules authored anywhere the result is every depth from
    ``start``, so the historical ladder is unchanged.
    """
    return tuple(
        depth
        for depth in range(start, len(rules))
        if (rules[depth] is None and unrestricted)
        or matched_token(rules[depth], failure) is not None
    )


def eligible_ladder(route: GatewayRoute, failure: GatewayFailure | None) -> tuple[int, ...]:
    """Return every depth of ``route`` one ``start_attempt`` walk may claim.

    ``failure`` is the failure the data plane reported (``None`` for the first
    dial): the walk's first-dial claim, its sideways sheds, and its budget-reject
    skips all stay inside this ladder, so a failover-only rung is reached only by
    a failure its set names and never by a first dial or a shed.
    """
    return eligible_depths(route_fallback_rules(route), 0, failure)


def has_unrestricted_rung(rules: FallbackRules) -> bool:
    """Return whether any rung may take a first dial."""
    return any(rung is None for rung in rules)


def require_unrestricted_rung(route: GatewayRoute) -> None:
    """Fail closed when every admitted rung is failover-only.

    Raises:
        FailoverRulesError: No rung of ``route`` may take the first dial.
    """
    if not has_unrestricted_rung(route_fallback_rules(route)):
        raise FailoverRulesError(
            "every admitted deployment carries failover_only_on; at least one rung must be "
            "unrestricted to take the first dial"
        )


def rule_fallback_reason(
    route: GatewayRoute, candidate: int, current_depth: object, failure: GatewayFailure | None
) -> str | None:
    """Return the ledger ``fallback_reason`` for a reservation at ``candidate``.

    A successor dialed on a rule-carrying rung records
    ``failover_only_on:<token>`` so the platform can show why that rung (a
    customer's own key, typically) served; a first dial (``current_depth`` is
    not a depth), a same-rung redial, and any unrestricted rung keep the
    route's own reason.
    """
    if not isinstance(current_depth, int) or candidate == current_depth:
        return route.fallback_reason
    token = matched_token(rung_rules(route.deployments[candidate]), failure)
    return route.fallback_reason if token is None else f"{FALLBACK_REASON_PREFIX}{token}"

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

from collections.abc import Collection, Sequence

from exp.common.models.failover_tokens import REFUSAL_TOKEN, REFUSAL_TOKEN_PREFIX
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, GatewayRefusalReason
from exp.runtime.gateway.routing import GatewayRoute

FallbackRules = tuple[frozenset[str] | None, ...]
"""One entry per route depth: the rung's token set, or ``None`` for an unrestricted rung."""

RungFamily = tuple[str, str, str, frozenset[str] | None]
"""One upstream offering: ``(provider, provider_model, billing_source, failover_only_on)``."""

RungFamilies = tuple[RungFamily | None, ...]
"""One entry per route depth: its rung family, ``None`` for a depth never worth reserving for."""

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


def untried_depths(
    rules: FallbackRules,
    failure: GatewayFailure | None,
    dispatched: Sequence[int],
    current_depth: int,
    *,
    unrestricted: bool = True,
) -> tuple[int, ...]:
    """Return the depths a failover from ``current_depth`` may claim, in route order.

    Every depth not dispatched yet in this request (``dispatched`` counts
    physical dispatches per depth) and not the failing depth itself, EARLIER
    depths included: a rung the request skipped (shed at its bound on the
    first dial, or ordered behind an affinity choice) is still a live
    successor when the rung it was skipped for fails (production,
    2026-10-21: a full lane at depth 0 was skipped, the dead lane at depth 1
    502'd, and the request ended with nothing revisited). Route order is kept
    among the untried depths, so a skipped earlier rung is preferred over a
    later one. The ``failover_only_on`` rules apply exactly as in
    ``eligible_depths``; the data plane mirrors this set
    (``waterfall/fallback_rules.rs``, ``successor_available``).
    """
    return tuple(
        depth
        for depth in eligible_depths(rules, 0, failure, unrestricted=unrestricted)
        if depth != current_depth and (depth >= len(dispatched) or dispatched[depth] == 0)
    )


def route_rung_families(
    route: GatewayRoute, denied_pools: Collection[str] = frozenset()
) -> RungFamilies:
    """Return each depth's rung family.

    A family is one upstream offering on one kind of credential: the
    deployments of one lane differ only by connection and share a family,
    while a different provider or model, a customer's own credential, or a
    rung carrying its own ``failover_only_on`` set is a family of its own.

    Args:
        route: The admitted route.
        denied_pools: Destination pools the request's budget already refused;
            their depths are skipped by the accounting walk, so they get no
            family and no attempt is reserved for them.

    Returns:
        One family per route depth, aligned with ``route.deployments``.
    """
    return tuple(
        None
        if denied_pools and route.snapshot.stage_for_depth(depth).pool_id in denied_pools
        else (
            deployment.provider,
            deployment.provider_model,
            deployment.billing_source.value,
            rung_rules(deployment),
        )
        for depth, deployment in enumerate(route.deployments)
    )


def rung_family_index(route: GatewayRoute, deployment: ExactModelDeployment) -> int:
    """Name ``deployment``'s family on the wire: the first route depth sharing it."""
    families = route_rung_families(route)
    return families.index(families[route.snapshot.deployment_ids.index(deployment.deployment_id)])


def fresh_family_count(
    depths: Sequence[int], families: RungFamilies, dispatched: Sequence[int]
) -> int:
    """Count the families with a depth in ``depths`` that this request has not dialed.

    Args:
        depths: The untried eligible depths.
        families: One family per route depth.
        dispatched: Physical dispatches per route depth in this request.

    Returns:
        How many undialed families the remaining attempts must still reach.
    """
    dialed = {families[depth] for depth, count in enumerate(dispatched) if count > 0}
    return len({families[depth] for depth in depths} - dialed - {None})


def reserve_fresh_families(
    depths: tuple[int, ...],
    families: RungFamilies,
    dispatched: Sequence[int],
    remaining_attempts: int,
) -> tuple[int, ...]:
    """Order ``depths`` so the attempt budget reaches every untried rung family.

    A route often lists one rung as several deployments of one family (one
    per connection of the same lane) ahead of the next rung. Walked in route
    order, a throttle burst answered by every connection would spend the
    whole attempt budget on that family and end the request with the next
    rung never dialed. So when the attempts left (this one included) are no
    more than the families this request has not dialed yet
    (``fresh_family_count``), those fresh families' depths come first, in
    route order. The other depths follow, so a fresh rung that sheds or is
    refused still spills onto a sibling. With budget to spare the order is
    unchanged.

    Args:
        depths: The untried eligible depths, in route order.
        families: One family per route depth.
        dispatched: Physical dispatches per route depth in this request.
        remaining_attempts: Attempts the request may still make, this one included.

    Returns:
        The same depths, fresh families first when the budget is tight.
    """
    if fresh_family_count(depths, families, dispatched) < remaining_attempts:
        return depths
    dialed = {families[depth] for depth, count in enumerate(dispatched) if count > 0}
    fresh = tuple(depth for depth in depths if families[depth] not in dialed | {None})
    return fresh + tuple(depth for depth in depths if depth not in fresh)


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


def dialable_rejections[Rejection](
    route: GatewayRoute, served: Sequence[int], rejections: Sequence[Rejection]
) -> tuple[Rejection, ...]:
    """Return the dialable rungs' rejections when only failover-only rungs serve a request.

    Admission keeps the rungs that can serve a request; when every one of them
    is failover-only, nothing may take the first dial. The request is then
    refused on what the unrestricted rungs lacked, never handed to a rung
    reserved for failover.

    Args:
        route: The route the per-rung check ran over.
        served: Route positions that passed it, in route order.
        rejections: One rejection per position that did not, in route order.

    Returns:
        The unrestricted rungs' rejections, in route order, when no serving
        rung is unrestricted; empty when one is, or when the route has no
        unrestricted rung at all (``require_unrestricted_rung`` names that).
    """
    rules = route_fallback_rules(route)
    if has_unrestricted_rung(tuple(rules[index] for index in served)):
        return ()
    refused = tuple(index for index in range(len(rules)) if index not in served)
    return tuple(
        rejection
        for index, rejection in zip(refused, rejections, strict=True)
        if rules[index] is None
    )


def coercible_rejections[Rejection](
    route: GatewayRoute, served: Sequence[int], rejections: Sequence[Rejection]
) -> tuple[Rejection, ...]:
    """Return the rejections a disclosed capability coercion judges, if one is due.

    A coercion is due when no rung that may take the first dial serves the
    request: either nothing serves, or only failover-only rungs do. It then
    judges the unrestricted rungs' rejections alone (a failover-only rung that
    preserves the field cannot dial it first), or every rejection on a route
    with no unrestricted rung (``require_unrestricted_rung`` names that case).

    Args:
        route: The route the per-rung check ran over.
        served: Route positions that passed it, in route order.
        rejections: One rejection per position that did not, in route order.

    Returns:
        The rejections to coerce over, empty when an unrestricted rung serves.
    """
    dialable = dialable_rejections(route, served, rejections)
    return dialable if dialable or served else tuple(rejections)


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

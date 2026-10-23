"""Honor OpenRouter-style ``provider`` routing preferences against the gateway's own rungs.

A caller may send OpenRouter's documented ``provider`` object
(https://openrouter.ai/docs/features/provider-routing) on any surface. The
gateway resolves its routing keys against the rungs of the requested model,
never by forwarding them to an aggregator upstream:

* ``only``: keep rungs whose provider is listed;
* ``ignore``: drop rungs whose provider is listed;
* ``order``: try rungs of the listed providers first, in that order;
* ``allow_fallbacks: false``: with ``order``, serve only the listed providers;
  without it, serve only the first eligible rung. A pinned rung that fails is
  the request's error, never a failover.

A rung is named by its caller-facing provider label (the host may relabel a
lane, for example a masked house lane, through ``provider_label``); OpenRouter
spellings (``amazon-bedrock``, ``azure``, ``google-vertex``, ...) are accepted
as aliases. A preference no rung can satisfy is a 400 naming the providers the
model offers, never a silent serve elsewhere. Other preference keys
(``sort``, ``quantizations``, ``max_price``, ...) have no gateway meaning and
are disclosed as ignored; ``zdr`` and ``data_collection`` keep their own
handling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.native_execution import reorder_route_deployments, select_route_deployments
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError

ProviderLabel = Callable[[ExactModelDeployment], str]
"""The caller-facing provider name of one rung."""

ROUTING_KEYS = frozenset({"order", "only", "ignore", "allow_fallbacks"})
"""Preference keys the gateway resolves itself."""

SEPARATELY_HANDLED_KEYS = frozenset({"zdr", "data_collection"})
"""Preference keys with their own handling (ZDR routing, OpenRouter data policy)."""

_ALIASES: dict[str, frozenset[str]] = {
    "amazon_bedrock": frozenset({"bedrock"}),
    "aws_bedrock": frozenset({"bedrock"}),
    "bedrock": frozenset({"bedrock"}),
    "azure": frozenset({"azure", "azure_openai"}),
    "azure_openai": frozenset({"azure", "azure_openai"}),
    "google_vertex": frozenset({"vertex", "vertex_ai", "google_vertex"}),
    "vertex": frozenset({"vertex", "vertex_ai", "google_vertex"}),
    "vertex_ai": frozenset({"vertex", "vertex_ai", "google_vertex"}),
    "google_ai_studio": frozenset({"gemini", "google_ai_studio"}),
    "gemini": frozenset({"gemini", "google_ai_studio"}),
}
"""OpenRouter provider slugs that name the same host as a gateway provider id."""


def _normalized(name: str) -> str:
    """Fold case, spaces, and hyphens so ``Amazon Bedrock`` equals ``amazon_bedrock``."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _names(name: str) -> frozenset[str]:
    """Every gateway provider id one caller spelling may match."""
    key = _normalized(name)
    return _ALIASES.get(key, frozenset({key}))


def _default_label(deployment: ExactModelDeployment) -> str:
    """The catalog provider id, the label when the host supplies none."""
    return deployment.provider


def _preference_list(preferences: Mapping[str, object], key: str) -> tuple[str, ...] | None:
    """Read one validated provider-name list."""
    value = preferences.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProviderParameterError(
            message=f"'provider.{key}' must be an array of provider names.",
            param=f"provider.{key}",
            code="invalid_parameter",
        )
    return tuple(str(item) for item in value)


def _unsatisfiable(param: str, requested: Sequence[str], available: Sequence[str]) -> Exception:
    """Name the unsatisfiable preference and the providers the model does offer."""
    offered = ", ".join(sorted(set(available))) or "none"
    asked = ", ".join(repr(name) for name in requested)
    return ProviderParameterError(
        message=(
            f"No available route for this model matches 'provider.{param}' ({asked}). "
            f"Providers serving this model: {offered}. Change or remove 'provider.{param}'."
        ),
        param=f"provider.{param}",
        code="invalid_parameter",
    )


def ignored_preference_disclosures(request: GatewayRequest) -> tuple[str, ...]:
    """Disclose preference keys the gateway does not act on."""
    preferences = request.provider_preferences or {}
    return tuple(
        f"provider.{key}->ignored(unsupported)"
        for key in preferences
        if key not in ROUTING_KEYS and key not in SEPARATELY_HANDLED_KEYS
    )


def _order_rank(
    route: GatewayRoute, order: tuple[str, ...], label: ProviderLabel
) -> tuple[int, ...]:
    """Each rung's position in ``order`` (``len(order)`` when unlisted)."""
    wanted = [_names(name) for name in order]
    ranks: list[int] = []
    for deployment in route.deployments:
        own = _normalized(label(deployment))
        ranks.append(next((i for i, names in enumerate(wanted) if own in names), len(order)))
    return tuple(ranks)


def _stage(route: GatewayRoute, index: int) -> int:
    """The model-reference stage of one rung (0 for single-model routes)."""
    if not route.snapshot.model_stages:
        return 0
    return route.snapshot.stage_for_depth(index).stage_index


def _order_permutation(
    route: GatewayRoute, order: tuple[str, ...], label: ProviderLabel
) -> tuple[int, ...]:
    """Stable permutation leading with ``order`` within each model stage."""
    ranks = _order_rank(route, order, label)
    return tuple(
        sorted(range(len(route.deployments)), key=lambda i: (_stage(route, i), ranks[i], i))
    )


def apply_provider_preferences(
    route: GatewayRoute, request: GatewayRequest, label: ProviderLabel | None = None
) -> GatewayRoute:
    """Narrow and order one route by the caller's ``provider`` routing preferences.

    Args:
        route: The authorized model route, before availability filtering, so a
            pinned rung that is down is the request's error, not a fallback.
        request: The decoded request carrying ``provider_preferences``.
        label: Optional host relabeling of a rung's provider name.

    Returns:
        The narrowed, ordered route (the same route when no routing key is set).

    Raises:
        ProviderParameterError: A preference is malformed or no rung satisfies it.
    """
    preferences = request.provider_preferences or {}
    if not ROUTING_KEYS & preferences.keys():
        return route
    name_of = label or _default_label
    labels = tuple(_normalized(name_of(deployment)) for deployment in route.deployments)
    available = tuple(name_of(deployment) for deployment in route.deployments)
    only = _preference_list(preferences, "only")
    ignore = _preference_list(preferences, "ignore")
    order = _preference_list(preferences, "order")
    allow_fallbacks = preferences.get("allow_fallbacks", True)
    if not isinstance(allow_fallbacks, bool):
        raise ProviderParameterError(
            message="'provider.allow_fallbacks' must be a boolean.",
            param="provider.allow_fallbacks",
            code="invalid_parameter",
        )
    kept = list(range(len(route.deployments)))
    if only is not None:
        allowed = frozenset().union(*(_names(name) for name in only)) if only else frozenset()
        kept = [i for i in kept if labels[i] in allowed]
        if not kept:
            raise _unsatisfiable("only", only, available)
    if ignore:
        excluded = frozenset().union(*(_names(name) for name in ignore))
        kept = [i for i in kept if labels[i] not in excluded]
        if not kept:
            raise _unsatisfiable("ignore", ignore, available)
    if order:
        wanted = frozenset().union(*(_names(name) for name in order))
        if not any(labels[i] in wanted for i in kept):
            raise _unsatisfiable("order", order, available)
        if not allow_fallbacks:
            kept = [i for i in kept if labels[i] in wanted]
    elif not allow_fallbacks:
        lead = next(
            (i for i in kept if route.deployments[i].gateway.capabilities.failover_only_on is None),
            kept[0],
        )
        kept = [lead]
    narrowed = select_route_deployments(route, tuple(kept))
    if not order or len(narrowed.deployments) < 2:
        return narrowed
    return reorder_route_deployments(narrowed, _order_permutation(narrowed, order, name_of))


def _host_label(components: object) -> ProviderLabel | None:
    """The host's optional ``provider_label`` relabeling, read off its components."""
    label = getattr(components, "provider_label", None)
    return label if callable(label) else None


def route_by_provider_preferences(
    route: GatewayRoute, request: GatewayRequest, components: object
) -> tuple[GatewayRoute, GatewayRequest]:
    """Narrow ``route`` by the preferences and disclose the keys left unhonored.

    Args:
        route: The authorized route before availability filtering.
        request: The decoded request.
        components: Host components; an optional ``provider_label`` attribute
            maps a rung to the provider name callers may pin it by.

    Returns:
        The narrowed route and the request carrying any ignored-key disclosures.
    """
    route = apply_provider_preferences(route, request, _host_label(components))
    disclosures = ignored_preference_disclosures(request)
    if disclosures:
        request = request.model_copy(
            update={
                "ignored_parameters": tuple(
                    dict.fromkeys((*request.ignored_parameters, *disclosures))
                )
            }
        )
    return route, request


def reorder_admitted_route[WireT](
    route: GatewayRoute,
    wires: tuple[WireT, ...],
    request: GatewayRequest,
    components: object,
) -> tuple[GatewayRoute, tuple[WireT, ...]]:
    """Re-apply ``provider.order`` after admission's own scheduling, wires in step."""
    preferences = request.provider_preferences or {}
    order = _preference_list(preferences, "order")
    if not order or len(route.deployments) < 2:
        return route, wires
    permutation = _order_permutation(route, order, _host_label(components) or _default_label)
    return reorder_route_deployments(route, permutation), tuple(wires[i] for i in permutation)

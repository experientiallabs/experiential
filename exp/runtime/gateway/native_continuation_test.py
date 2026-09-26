"""Native Responses continuation bindings preserve model-stage start authority."""

import pytest

from exp.runtime.gateway.contracts import DirectTarget, GatewayApiSurface
from exp.runtime.gateway.model_plan_test import catalog
from exp.runtime.gateway.native_continuation import select_bound_continuation_route
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.routing import CatalogRouteResolver, GatewayRoute
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.state import ContinuationRouteBinding


def _model_route(*, descendant_start_authorized: bool) -> GatewayRoute:
    """Build a Responses root/child/root route with explicit child-start authority.

    Args:
        descendant_start_authorized: Whether the host permits an initial child stage.

    Returns:
        A frozen route retaining both root segments and the intervening child.
    """
    normalized = catalog()
    authorization = _route().snapshot.authorization.model_copy(
        update={
            "catalog_sha256": normalized.identity_sha256(),
            "target": DirectTarget(pool_id="pool-a"),
            "surface": GatewayApiSurface.RESPONSES,
            "descendant_start_authorized": descendant_start_authorized,
        }
    )
    return CatalogRouteResolver(
        {(authorization.alias_revision_id, authorization.catalog_sha256): normalized}
    ).resolve_direct(authorization)


@pytest.mark.parametrize("descendant_start_authorized", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_bound_continuation_requires_authority_for_a_descendant_start(
    descendant_start_authorized: bool, depth: int
) -> None:
    """Retained child reasoning cannot grant an otherwise unauthorized initial stage.

    Args:
        descendant_start_authorized: Host proof permitting or forbidding child starts.
        depth: The retained winner's original root or child position.
    """
    route = _model_route(descendant_start_authorized=descendant_start_authorized)
    deployment = route.deployments[depth]
    binding = ContinuationRouteBinding(
        deployment_id=deployment.deployment_id,
        connection_sha256=deployment.connection_sha256,
        wire_authority_sha256="d" * 64,
    )
    if depth == 1 and not descendant_start_authorized:
        with pytest.raises(OpenAIProtocolError) as raised:
            select_bound_continuation_route(route, binding)
        assert raised.value.detail.code == "previous_response_not_found"
        assert raised.value.detail.param == "previous_response_id"
        return
    selected = select_bound_continuation_route(route, binding)
    assert selected.deployments == (deployment,)
    assert selected.snapshot.exact_model_id == route.snapshot.exact_model_id
    assert selected.snapshot.pool_id == route.snapshot.pool_id


def test_unbound_continuation_keeps_the_root_waterfall() -> None:
    """A continuation without encrypted provider authority does not narrow the route."""
    route = _model_route(descendant_start_authorized=False)
    assert select_bound_continuation_route(route, None) is route

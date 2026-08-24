"""Admission-time wire-profile resolution and dispatch freezing.

The native control plane resolves each admitted route's wire profile and
client here, then freezes the dispatch for body-signing dialects.

Most dialects send static credential headers and let the data plane serialize
the upstream payload itself. Body-signing dialects (Bedrock SigV4) compute
their headers over the exact serialized body bytes, so the control plane
freezes one canonical serialization here, signs it through the resolved
client, and hands both to the data plane, which must send the body verbatim:
re-serializing JSON on the other side of the boundary could reorder or
reformat bytes and invalidate the signature.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

from exp.common.core.artifacts import JsonObject

# The executor's identity check is the authoritative pre-dispatch invariant;
# the native path must enforce the same one, so the private helper is shared.
from exp.runtime.gateway.execution import _require_deployment_identity  # noqa: PLC2701
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.registry import RuntimeModelCatalog


class NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect; python must serve it."""


def resolve_wire_profile(
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
    route: GatewayRoute,
) -> tuple[GatewayWireProfile, NativeWireClient]:
    """Resolve one deployment's public wire profile for the data plane.

    Args:
        runtime_catalogs: Loaded catalogs keyed by revision and catalog digest.
        route: Resolved single-deployment route.

    Returns:
        The dialect, endpoint, headers, and timing facts for dispatch, with
        the model identity filled from the resolved snapshot when the profile
        leaves it empty, plus the resolved client so body-signing dialects can
        compute per-request headers.

    Raises:
        NativeDialectUnavailableError: The provider has no native-dialect
            implementation; the python engine serves the request.
        GatewayRoutingError: The resolved client cannot stream or the
            authorized catalog is not loaded.
        ValueError: The resolved client drifts from the frozen deployment.
    """
    authorization = route.snapshot.authorization
    catalog = runtime_catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
    if catalog is None:
        raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
    deployment = route.deployment
    resolved = catalog.resolve(deployment.source_alias)
    _require_deployment_identity(deployment, resolved)
    client = resolved.client
    if getattr(client, "stream", None) is None:
        raise GatewayRoutingError("resolved gateway deployment has no streaming capability")
    if not isinstance(client, NativeWireClient):
        raise NativeDialectUnavailableError(
            f"provider {deployment.provider!r} has no native wire profile"
        )
    try:
        profile = client.gateway_wire_profile()
    except ProviderCapabilityError as exc:
        if exc.capability != "native_data_plane":
            raise
        raise NativeDialectUnavailableError(
            f"provider {deployment.provider!r} has no native dialect implementation"
        ) from exc
    if not profile.model_id:
        profile = replace(profile, model_id=resolved.snapshot.model_id)
    return profile, client


def signed_dispatch(
    profile: GatewayWireProfile,
    client: NativeWireClient | None,
    upstream_payload: JsonObject,
) -> tuple[str | None, dict[str, str]]:
    """Freeze and sign the dispatch body for body-signing dialects.

    Signatures expire within AWS's short clock window: the data plane's
    immediate bounded open retry reuses them safely, and any later retry
    arrives as a fresh admission that signs again.

    Args:
        profile: Resolved wire profile for the dispatch.
        client: Resolved provider client from the frozen catalog.
        upstream_payload: Payload built by the shared dialect builders.

    Returns:
        The exact signed body string (``None`` for dialects whose payload the
        data plane serializes itself) and the dispatch headers.

    Raises:
        GatewayRoutingError: The dialect requires body signing but the
            resolved client cannot sign.
    """
    if not profile.signs_request_body:
        return None, dict(profile.headers)
    if not isinstance(client, GatewayDispatchSigner):
        raise GatewayRoutingError("resolved gateway deployment cannot sign its dispatch body")
    body = json.dumps(upstream_payload, separators=(",", ":"), ensure_ascii=False)
    headers = dict(profile.headers)
    headers.update(client.sign_gateway_dispatch(url=profile.url, body=body))
    return body, headers

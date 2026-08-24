"""Admission-time wire-profile resolution and dispatch freezing.

The native control plane resolves each admitted route's wire profile and
client here, then freezes the dispatch for body-signing dialects.

Most dialects send static credential headers and let the data plane serialize
the upstream payload itself. Body-signing dialects (Bedrock SigV4) compute
their headers over the exact serialized body bytes, so the control plane
freezes one canonical serialization at admission and retains the resolved
signer. The data plane must send the frozen bytes verbatim: re-serializing
JSON on the other side of the boundary could reorder or reformat bytes and
invalidate the signature. Signing itself happens at dispatch time, through
the ``sign_dispatch`` boundary callback the data plane invokes after it
acquires its bounded dispatch permit and immediately before the provider
POST, so time spent queued can never age a signature toward AWS's short
clock window.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

from exp.common.core.artifacts import JsonObject

# The executor's identity check is the authoritative pre-dispatch invariant;
# the native path must enforce the same one, so the private helper is shared.
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.execution import _require_deployment_identity  # noqa: PLC2701
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.openai_protocol.errors import public_failure_error


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


def frozen_dispatch(
    profile: GatewayWireProfile,
    client: NativeWireClient | None,
    upstream_payload: JsonObject,
) -> tuple[str | None, GatewayDispatchSigner | None]:
    """Freeze the dispatch body for body-signing dialects.

    Args:
        profile: Resolved wire profile for the dispatch.
        client: Resolved provider client from the frozen catalog.
        upstream_payload: Payload built by the shared dialect builders.

    Returns:
        The exact frozen body string plus the signer the control plane
        retains for the dispatch-time ``sign_dispatch`` callback, or
        ``(None, None)`` for dialects whose payload the data plane serializes
        itself.

    Raises:
        GatewayRoutingError: The dialect requires body signing but the
            resolved client cannot sign.
    """
    if not profile.signs_request_body:
        return None, None
    if not isinstance(client, GatewayDispatchSigner):
        raise GatewayRoutingError("resolved gateway deployment cannot sign its dispatch body")
    body = json.dumps(upstream_payload, separators=(",", ":"), ensure_ascii=False)
    return body, client


def dispatch_signature_headers(
    signer: GatewayDispatchSigner | None,
    *,
    url: str,
    body: str,
) -> dict[str, str]:
    """Sign one frozen dispatch body for the data plane, failing sanitized.

    Args:
        signer: Signer retained at admission, or ``None`` when the attempt is
            unknown or its dialect does not sign bodies.
        url: Exact endpoint the data plane will POST to.
        body: Exact frozen body string the data plane will send.

    Returns:
        Per-request headers covering the exact body bytes.

    Raises:
        OpenAIProtocolError: No signer is available for the attempt, or
            credential resolution failed; the error is already sanitized for
            the public boundary.
    """
    if signer is None:
        raise public_failure_error(
            GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="gateway dispatch signing is unavailable for this attempt",
            )
        )
    try:
        return dict(signer.sign_gateway_dispatch(url=url, body=body))
    except Exception as exc:  # noqa: BLE001 - the boundary sanitizes every failure.
        raise public_failure_error(
            GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
                safe_message=(
                    "provider authentication failed; ask the gateway operator to "
                    "verify the provider connection credential"
                ),
            )
        ) from exc

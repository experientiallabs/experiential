"""Dispatch-time body freezing and signing for body-signing dialects.

Most dialects send static credential headers and let the data plane serialize
the upstream payload itself. Body-signing dialects (Bedrock SigV4) compute
their headers over the exact serialized body bytes, so the control plane
freezes one canonical serialization at admission (:func:`frozen_dispatch`,
called per route deployment alongside
``native_execution.resolve_dispatchable_wires``) and retains the resolved
signer. The data plane must send the frozen bytes verbatim: re-serializing
JSON on the other side of the boundary could reorder or reformat bytes and
invalidate the signature. Signing itself happens at dispatch time, through
the ``sign_dispatch`` boundary callback the data plane invokes after it
acquires its bounded dispatch permit and immediately before the provider
POST, so time spent queued can never age a signature toward AWS's short
clock window; a same-deployment redial or a failover advance is a fresh
physical attempt, so it always signs afresh too.
"""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient
from exp.runtime.openai_protocol.errors import public_failure_error


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
        signer: Signer retained at admission for the active attempt's route
            depth, or ``None`` when the attempt is unknown or its dialect
            does not sign bodies.
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

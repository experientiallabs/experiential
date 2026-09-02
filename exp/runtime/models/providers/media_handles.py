"""Provider media handle admission shared by the per-dialect media encoders.

A handle names media the caller already uploaded to one provider, and only
a route on that same provider can resolve it. Two checks keep a handle off
a wire that cannot serve it: admission compares the handle's provider with
the deployment's provider before dispatch, and each dialect encoder refuses
a handle whose provider it does not encode. Both raise a capability error,
so a waterfall narrows past the rung instead of forwarding a reference the
provider would reject or, worse, resolve to a different object.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MediaHandle
from exp.runtime.models.providers.errors import ProviderCapabilityError

MEDIA_HANDLE_CAPABILITY = "media_handle_input"
"""Capability literal naming a route's ability to forward any uploaded-media handle."""

MEDIA_HANDLE_PROVIDER_CAPABILITY = "media_handle_provider"
"""Capability literal naming a handle whose provider differs from the route's."""

OPENAI_HANDLE_PROVIDERS = frozenset({"openai"})
"""Handle providers the OpenAI Chat and Responses wires resolve (``file_id``)."""

ANTHROPIC_HANDLE_PROVIDERS = frozenset({"anthropic"})
"""Handle providers the Anthropic Messages wire resolves (``source.type: file``)."""

GEMINI_HANDLE_PROVIDERS = frozenset({"gemini", "vertex"})
"""Handle providers the ``generateContent`` wire resolves through ``file_data``:
Gemini Files API URIs on the Gemini API and ``gs://`` objects on Vertex AI."""

BEDROCK_HANDLE_PROVIDERS = frozenset({"bedrock"})
"""Handle providers the Bedrock Converse wire resolves (``source.s3Location``)."""


def handle_provider_mismatch(handle: MediaHandle, route_provider: str | None) -> str:
    """Describe why a handle cannot ride a route on another provider.

    Args:
        handle: The caller's provider handle.
        route_provider: Provider of the route that declined it, if known.

    Returns:
        A caller-safe sentence naming the handle's provider.
    """
    route = (
        f"the selected model alias routes to {route_provider}"
        if route_provider
        else f"the selected model alias has no {handle.provider} route"
    )
    return (
        f"The request references media uploaded to {handle.provider}, which only a "
        f"{handle.provider} route can resolve, but {route}. Send the media inline or "
        f"choose a model alias served by {handle.provider}."
    )


def require_handle_provider(handle: MediaHandle, providers: frozenset[str]) -> None:
    """Refuse a handle whose provider a dialect cannot resolve.

    Args:
        handle: The caller's provider handle.
        providers: Handle providers the encoding dialect resolves.

    Raises:
        ProviderCapabilityError: The handle names another provider.
    """
    if handle.provider not in providers:
        raise ProviderCapabilityError(
            capability=MEDIA_HANDLE_PROVIDER_CAPABILITY,
            detail=handle_provider_mismatch(handle, None),
        )


def preflight_media_handles(
    handles: tuple[MediaHandle, ...],
    *,
    supports_media_handle_input: bool,
    route_provider: str | None,
) -> None:
    """Admit or refuse a request's handles for one deployment before dispatch.

    Args:
        handles: Every handle the request carries, in caller order.
        supports_media_handle_input: The deployment's declared handle support.
        route_provider: The deployment's provider name. ``None`` means the
            caller did not identify the route, so no handle is admissible.

    Raises:
        ProviderCapabilityError: The route declares no handle support, or a
            handle names a provider other than the route's.
    """
    if not handles:
        return
    if not supports_media_handle_input:
        raise ProviderCapabilityError(capability=MEDIA_HANDLE_CAPABILITY)
    for handle in handles:
        if handle.provider != route_provider:
            raise ProviderCapabilityError(
                capability=MEDIA_HANDLE_PROVIDER_CAPABILITY,
                detail=handle_provider_mismatch(handle, route_provider),
            )


def bedrock_s3_location(handle: MediaHandle) -> JsonObject:
    """Encode one Bedrock handle as a Converse ``s3Location`` source.

    Args:
        handle: A ``bedrock`` handle holding an ``s3://bucket/key`` URI.

    Returns:
        The ``{"uri": ..., "bucketOwner": ...}`` object, owner only when set.
    """
    location: JsonObject = {"uri": handle.reference}
    if handle.bucket_owner is not None:
        location["bucketOwner"] = handle.bucket_owner
    return location

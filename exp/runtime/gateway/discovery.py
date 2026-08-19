"""Caller-facing model discovery objects for the gateway HTTP boundary."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

AliasAuthority = tuple[str, str, str]
"""Granted alias name, active revision, and catalog digest."""


def public_model_object(authority: AliasAuthority) -> JsonObject:
    """Build one OpenAI model object enriched with the granted alias authority.

    The four OpenAI keys keep their exact meaning for official clients; the ``wmo``
    object carries only authority metadata the gateway already exposes to callers
    through response headers and grants, never provider or credential detail.

    Args:
        authority: Granted alias, active revision, and catalog digest triple.

    Returns:
        JSON-compatible public model object.
    """
    alias, revision, digest = authority
    return {
        "id": alias,
        "object": "model",
        "created": 0,
        "owned_by": "wmo",
        "wmo": {"alias_revision_id": revision, "catalog_sha256": digest},
    }


def require_granted_authority(
    authorities: tuple[AliasAuthority, ...],
    model_id: str,
) -> AliasAuthority:
    """Return one granted alias authority without confirming ungranted aliases.

    Args:
        authorities: Every authority granted to the presented key.
        model_id: Public model alias requested by the caller.

    Returns:
        The matching granted authority.

    Raises:
        OpenAIProtocolError: The alias is unknown or not granted to this key; both
            cases raise the identical 404 so the route is not an existence oracle.
    """
    for authority in authorities:
        if authority[0] == model_id:
            return authority
    raise OpenAIProtocolError(
        status_code=404,
        code="model_not_found",
        message=(
            "The requested model does not exist or is not granted to this key. "
            "GET /v1/models lists the model aliases available to this key."
        ),
        param="model",
    )

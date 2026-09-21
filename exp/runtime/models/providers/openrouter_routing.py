"""OpenRouter provider-routing preferences the gateway sets per request.

OpenRouter load-balances one model id across upstream providers and accepts a
request-level ``provider`` object that narrows the candidates. The gateway
uses exactly one such preference: the zero-data-retention constraint, which
restricts the request to OpenRouter's published ZDR endpoint list (the public
``GET /api/v1/endpoints/zdr``) and denies data collection, so the aggregator
serves from a retention-free upstream or refuses. The constraint TIGHTENS only:
any preference already on the payload is kept, and ``zdr`` / ``data_collection``
are forced to the strict values regardless of what was there.

The metadata opt-in header makes OpenRouter name the upstream it selected on
the response, which the data plane records per attempt as the settlement's
``upstream_provider``.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, StrictBool

from exp.common.core.artifacts import JsonObject, JsonValue

OPENROUTER_PROVIDER_ID: Final = "openrouter"
"""The catalog provider id of OpenRouter rungs (the only wire with this knob)."""

OPENROUTER_METADATA_HEADER: Final = "X-OpenRouter-Metadata"
OPENROUTER_METADATA_ENABLED: Final = "enabled"
"""Opt-in response metadata: the selected endpoint's provider rides the body."""

ZDR_PROVIDER_PREFERENCES: Final[JsonObject] = {"zdr": True, "data_collection": "deny"}
"""The strict values the constraint forces onto ``payload["provider"]``."""


class ProviderRoutingPreferences(BaseModel):
    """The caller's top-level ``provider`` object, OpenRouter's routing-preference shape.

    Accepted on every gateway surface (Chat Completions, Responses, Messages).
    ``zdr: true`` is the one field the gateway acts on itself: it DEMANDS
    zero-data-retention routing for the request (the host's route filter
    applies the same posture filter as an organization ``require_zdr``, and
    the request can only tighten an organization policy, never loosen it).
    Every other key (``data_collection``, ``order``, ``only``, ...) is
    forwarded to OpenRouter rungs verbatim, tightened by the constraint when
    the route is constrained, and dropped on every other wire, which has no
    such field.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    zdr: StrictBool | None = None
    data_collection: Literal["allow", "deny"] | None = None

    @property
    def demands_zdr(self) -> bool:
        """Whether the caller demanded zero-data-retention routing."""
        return self.zdr is True


def forward_provider_preferences(payload: JsonObject, preferences: JsonObject) -> JsonObject:
    """Return ``payload`` carrying the caller's ``provider`` object for an OpenRouter rung.

    Args:
        payload: A built Chat Completions payload for an OpenRouter rung.
        preferences: The caller's validated ``provider`` object.

    Returns:
        A new payload whose ``provider`` is a copy of ``preferences``; the
        constraint (:func:`constrain_openrouter_zero_data_retention`) tightens
        on top when the rung is flagged. The input is never mutated.
    """
    forwarded: JsonValue = dict(preferences)
    return {**payload, "provider": forwarded}


def constrain_openrouter_zero_data_retention(payload: JsonObject) -> JsonObject:
    """Return ``payload`` with OpenRouter's ZDR routing constraint applied.

    Args:
        payload: A built Chat Completions payload for an OpenRouter rung.

    Returns:
        A new payload whose ``provider`` object carries ``zdr: true`` and
        ``data_collection: "deny"``. Other keys of an existing ``provider``
        object survive; a non-object ``provider`` value is replaced, and a
        looser ``zdr: false`` or ``data_collection: "allow"`` is overridden.
        The input is never mutated.
    """
    existing = payload.get("provider")
    preferences: JsonObject = dict(existing) if isinstance(existing, dict) else {}
    preferences.update(ZDR_PROVIDER_PREFERENCES)
    tightened: JsonValue = preferences
    return {**payload, "provider": tightened}


def openrouter_metadata_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return ``headers`` plus the OpenRouter metadata opt-in.

    Args:
        headers: The rung's static wire headers.

    Returns:
        A new mapping with ``X-OpenRouter-Metadata: enabled`` added.
    """
    return {**headers, OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_ENABLED}

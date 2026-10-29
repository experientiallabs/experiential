"""OpenRouter provider-routing preferences the gateway sets per request.

OpenRouter load-balances one model id across upstream providers and accepts a
request-level ``provider`` object that narrows the candidates. The gateway
sets two such preferences. The reasoning-off constraint keeps a request that
turns reasoning off away from upstreams observed to think anyway. The
zero-data-retention constraint
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

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from exp.common.core.artifacts import JsonObject, JsonValue

OPENROUTER_PROVIDER_ID: Final = "openrouter"
"""The catalog provider id of OpenRouter rungs (the only wire with this knob)."""

OPENROUTER_METADATA_HEADER: Final = "X-OpenRouter-Metadata"
OPENROUTER_METADATA_ENABLED: Final = "enabled"
"""Opt-in response metadata: the selected endpoint's provider rides the body."""

GOOGLE_MODEL_PREFIX: Final = "google/gemini"
"""OpenRouter's Gemini models, which Google serves with writes read back in-call.

The wider ``google/`` namespace also holds Gemma, which third parties serve with
ordinary disjoint cache accounting, so it keeps the default contract."""

ProviderName = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
"""One provider name in a routing preference list."""

ZDR_PROVIDER_PREFERENCES: Final[JsonObject] = {"zdr": True, "data_collection": "deny"}
"""The strict values the constraint forces onto ``payload["provider"]``."""


class ProviderRoutingPreferences(BaseModel):
    """The caller's top-level ``provider`` object, OpenRouter's routing-preference shape.

    Accepted on every gateway surface (Chat Completions, Responses, Messages).
    ``zdr: true`` DEMANDS zero-data-retention routing for the request (the
    host's route filter applies the same posture filter as an organization
    ``require_zdr``, and the request can only tighten an organization policy).
    ``order``, ``only``, ``ignore`` and ``allow_fallbacks`` are resolved
    against the gateway's own rungs at admission. ``data_collection`` (and
    ``zdr``) are forwarded to OpenRouter rungs; every other key is disclosed
    as ignored.

    Attributes:
        zdr: Demand zero-data-retention routing.
        data_collection: OpenRouter's upstream data-collection policy.
        order: Provider names to try first, in order.
        only: Provider names allowed to serve.
        ignore: Provider names never allowed to serve.
        allow_fallbacks: Whether providers outside ``order`` (or after the first
            rung) may serve.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    zdr: StrictBool | None = None
    data_collection: Literal["allow", "deny"] | None = None
    order: tuple[ProviderName, ...] | None = Field(default=None, max_length=32)
    only: tuple[ProviderName, ...] | None = Field(default=None, max_length=32)
    ignore: tuple[ProviderName, ...] | None = Field(default=None, max_length=32)
    allow_fallbacks: StrictBool | None = None

    @property
    def demands_zdr(self) -> bool:
        """Whether the caller demanded zero-data-retention routing."""
        return self.zdr is True


FORWARDED_PREFERENCE_KEYS: Final = frozenset({"zdr", "data_collection"})
"""The only caller preference keys an OpenRouter upstream receives.

Routing keys (``order``, ``only``, ``ignore``, ``allow_fallbacks``) name the
gateway's own providers and are resolved at admission
(``exp.runtime.gateway.provider_preferences``); forwarding them made OpenRouter
look for providers it does not have (a pin on ``experiential_cloud`` answered
404). Other keys are disclosed as ignored, never forwarded."""


def forward_provider_preferences(payload: JsonObject, preferences: JsonObject) -> JsonObject:
    """Return ``payload`` carrying the caller's data-policy preferences for an OpenRouter rung.

    Args:
        payload: A built Chat Completions payload for an OpenRouter rung.
        preferences: The caller's validated ``provider`` object.

    Returns:
        A new payload whose ``provider`` holds only the caller's ``zdr`` and
        ``data_collection`` keys (unchanged when the caller sent neither); the
        constraint (:func:`constrain_openrouter_zero_data_retention`) tightens
        on top when the rung is flagged. The input is never mutated.
    """
    kept = {key: value for key, value in preferences.items() if key in FORWARDED_PREFERENCE_KEYS}
    if not kept:
        return payload
    forwarded: JsonValue = kept
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


OPENROUTER_REASONING_OFF_IGNORED: Final[tuple[str, ...]] = ("atlas-cloud",)
"""OpenRouter upstreams that keep thinking when the request turns reasoning off.

AtlasCloud served z-ai/glm-5.1 with ``reasoning: {"effort": "none"}`` and
returned 129 reasoning tokens (production, 2026-10-19, request
request-83fbbf3e7d2b4ee09700226dfaecbc07, settled ``upstream_provider``
AtlasCloud), while Novita honored the same wire with zero reasoning tokens on
45 of 45 requests. OpenRouter's reasoning reference documents ``effort: "none"``
as "disables reasoning entirely" without naming the upstreams that comply, so
the upstreams seen to ignore it are excluded for that request only."""


def constrain_openrouter_reasoning_off(payload: JsonObject) -> JsonObject:
    """Return ``payload`` routed away from upstreams that ignore reasoning off.

    Args:
        payload: A built Chat Completions payload for an OpenRouter rung.

    Returns:
        The input unchanged unless it carries ``reasoning: {"effort": "none"}``;
        then a new payload whose ``provider.ignore`` also lists
        :data:`OPENROUTER_REASONING_OFF_IGNORED`. Other ``provider`` keys survive
        and an existing ``ignore`` list keeps its names. If every remaining
        upstream is unavailable OpenRouter refuses, which the gateway answers
        like any rung failure instead of serving reasoning the caller turned off.
        The input is never mutated.
    """
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, dict) or reasoning.get("effort") != "none":
        return payload
    existing = payload.get("provider")
    preferences: JsonObject = dict(existing) if isinstance(existing, dict) else {}
    current = preferences.get("ignore")
    ignored: list[JsonValue] = list(current) if isinstance(current, list) else []
    ignored.extend(name for name in OPENROUTER_REASONING_OFF_IGNORED if name not in ignored)
    preferences["ignore"] = ignored
    narrowed: JsonValue = preferences
    return {**payload, "provider": narrowed}


def openrouter_metadata_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return ``headers`` plus the OpenRouter metadata opt-in.

    Args:
        headers: The rung's static wire headers.

    Returns:
        A new mapping with ``X-OpenRouter-Metadata: enabled`` added.
    """
    return {**headers, OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_ENABLED}


def openrouter_chat_wire(provider: str, dialect: str) -> bool:
    """Whether a rung is OpenRouter's Chat Completions wire (the only ``provider`` field).

    Args:
        provider: The rung's catalog provider id.
        dialect: The rung's resolved wire dialect.

    Returns:
        True when the rung dispatches OpenRouter's Chat Completions dialect.
    """
    return provider == OPENROUTER_PROVIDER_ID and dialect == "openai_compatible"


def openrouter_cache_writes_within_reads(model_id: str) -> bool:
    """Whether OpenRouter reports this model's cache writes inside its cache reads.

    OpenRouter serves ``cache_control`` on Gemini models with Google explicit
    caching: the call that creates the cache also reads the written prefix
    back, so its usage reports the same tokens under both ``cached_tokens``
    and ``cache_write_tokens`` (and bills both legs). Every other model family
    OpenRouter caches reports disjoint read and write slices of input.

    Args:
        model_id: The OpenRouter model id the rung dispatches.

    Returns:
        True for Google models, whose cache writes are a subset of cache reads.
    """
    return model_id.startswith(GOOGLE_MODEL_PREFIX)

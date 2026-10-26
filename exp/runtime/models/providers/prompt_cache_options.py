"""OpenAI's explicit prompt-caching controls on the wires that carry them.

OpenAI (https://developers.openai.com/api/docs/guides/prompt-caching) caches one
implicit breakpoint per request by default. ``prompt_cache_options.mode: "explicit"``
turns it off so only content blocks marked with ``prompt_cache_breakpoint`` are
written, and ``prompt_cache_options.ttl`` (``"30m"`` default, ``"12h"`` on Responses)
is the minimum lifetime every breakpoint inherits. OpenRouter documents the same
fields and relays them (https://openrouter.ai/docs/guides/best-practices/prompt-caching).

The canonical request carries the caller's markers as ephemeral ``cache_control``
blocks; this module decides which origins receive the OpenAI spelling and renders it.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile

_OPENAI_HOSTS = frozenset({"api.openai.com", "openrouter.ai"})
# Azure OpenAI resource hosts (``{resource}.openai.azure.com``, and the
# ``cognitiveservices`` spelling an AI Services resource answers the same API on).
_AZURE_OPENAI_HOST_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com")

BREAKPOINT_MARKER: JsonObject = {"mode": "explicit"}
"""The re-emitted ``prompt_cache_breakpoint``: it inherits the request lifetime, which the
decoder already required every caller breakpoint to state (``require_request_lifetime``)."""


def openai_prompt_cache_origin(url: str) -> bool:
    """Whether an OpenAI-shaped origin carries ``prompt_cache_options`` and breakpoints.

    Args:
        url: The rung's request URL or base URL.

    Returns:
        True for OpenAI's own API, OpenRouter and Azure OpenAI resource hosts.
    """
    host = (urlsplit(url).hostname or "").lower()
    return host in _OPENAI_HOSTS or host.endswith(_AZURE_OPENAI_HOST_SUFFIXES)


def prompt_cache_options_payload(
    *, mode: str | None, ttl: str | None, forwards_options: bool
) -> JsonObject | None:
    """Build the ``prompt_cache_options`` object one OpenAI-wire payload carries.

    The provider defaults (``mode: "implicit"``, ``ttl: "30m"``) are never sent, so a
    request that asks for neither leaves the field off (older models refuse it).

    Args:
        mode: The canonical ``prompt_cache_mode`` (only ``"explicit"`` is carried).
        ttl: The canonical ``prompt_cache_ttl`` (only ``"12h"`` is carried); admission
            already confined it to rungs declaring the 12-hour lifetime.
        forwards_options: Whether this rung's origin carries the mode.

    Returns:
        The object to send, or ``None`` when there is nothing to send.
    """
    options: JsonObject = {}
    if mode is not None and forwards_options:
        options["mode"] = mode
    if ttl is not None:
        options["ttl"] = ttl
    return options or None


PROMPT_CACHE_MODE_NOT_FORWARDED = (
    "prompt_cache_options.mode->not_forwarded(provider_decides_caching)"
)
"""Disclosure for ``mode: "explicit"`` on a route where no rung can switch implicit caching off."""

PROMPT_CACHE_BREAKPOINT_TRANSLATED = "prompt_cache_breakpoint->translated(cache_control_5m)"
"""Disclosure for breakpoints a ``cache_control`` rung caches at its own 5-minute default.

Anthropic's ephemeral ``cache_control`` and Bedrock's ``cachePoint`` default to five
minutes, shorter than OpenAI's 30-minute minimum. The longer (1-hour) lifetime is never
chosen on the caller's behalf: it doubles the write price, and OpenRouter documents the
same rule (a ``prompt_cache_breakpoint`` TTL is dropped when routed to Anthropic)."""

_MARKER_ONLY_DIALECTS = frozenset({"anthropic_messages", "bedrock_converse_stream"})
"""Wires that cache only at markers, so ``mode: "explicit"`` already holds there."""


def prompt_cache_disclosures(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> tuple[str, ...]:
    """Name how a route treats the caller's OpenAI prompt-cache controls.

    Like every cache marker, the controls are kept while any rung honors them and
    disclosed only when none does: they change cost, never the answer.

    Args:
        profiles: Ordered wire profiles for every deployment in the route.
        request: The decoded public request.

    Returns:
        ``ignored_parameters`` entries, empty when every control reaches the provider.
    """
    disclosures: list[str] = []
    if request.prompt_cache_mode == "explicit" and not any(
        profile.forwards_prompt_cache_options or profile.dialect in _MARKER_ONLY_DIALECTS
        for profile in profiles
    ):
        disclosures.append(PROMPT_CACHE_MODE_NOT_FORWARDED)
    if request.prompt_cache_breakpoints and any(
        profile.preserves_cache_control and not profile.forwards_prompt_cache_options
        for profile in profiles
    ):
        disclosures.append(PROMPT_CACHE_BREAKPOINT_TRANSLATED)
    return tuple(disclosures)


def cache_marker_spelling(request: GatewayRequest) -> str:
    """The caller's own name for its content-block cache markers."""
    return "prompt_cache_breakpoint" if request.prompt_cache_breakpoints else "cache_control"


def honors_markers(profile: GatewayWireProfile, request: GatewayRequest) -> bool:
    """Whether a rung caches at this request's markers (OpenAI breakpoints included)."""
    return profile.honors_cache_markers(openai_breakpoints=request.prompt_cache_breakpoints)

"""Web-search spellings on the OpenAI-compatible surfaces.

Chat Completions callers ask for search three ways: OpenAI's
``web_search_options`` object, OpenRouter's ``plugins: [{"id": "web"}]``
array, and OpenRouter's ``:online`` model suffix. Responses callers declare
a hosted ``web_search`` tool. Each normalizes into
:class:`~exp.runtime.gateway.web_search.contracts.GatewayWebSearch`; the
planner decides at admission whether a native rung runs it or the gateway does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayProviderNativeTool
from exp.runtime.gateway.web_search.contracts import (
    MAXIMUM_WEB_SEARCH_RESULTS,
    GatewayWebSearch,
    results_for_context_size,
)
from exp.runtime.openai_protocol.errors import invalid_field

ONLINE_MODEL_SUFFIX = ":online"
"""OpenRouter's shorthand for ``plugins: [{"id": "web"}]`` on the model name."""

RESPONSES_WEB_SEARCH_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_2025_08_26",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    }
)
"""Hosted web-search tool ``type`` values on the Responses surface."""

_CONTEXT_SIZES: dict[str, Literal["low", "medium", "high"]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class WebSearchOptions(BaseModel):
    """OpenAI Chat ``web_search_options``; unknown keys stay a named rejection."""

    model_config = ConfigDict(extra="forbid")

    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: JsonObject | None = None


class ChatPlugin(BaseModel):
    """One OpenRouter ``plugins`` entry; only the ``web`` plugin is served."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    max_results: int | None = Field(default=None, ge=1, le=MAXIMUM_WEB_SEARCH_RESULTS)
    search_prompt: str | None = Field(default=None, max_length=2000)
    engine: str | None = Field(default=None, max_length=32)
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()
    user_location: JsonObject | None = None


def split_online_suffix(model: str) -> tuple[str, bool]:
    """Strip OpenRouter's ``:online`` suffix from a model name.

    Args:
        model: The caller's ``model`` value.

    Returns:
        The alias to authorize and whether the suffix asked for web search.
    """
    if model.endswith(ONLINE_MODEL_SUFFIX) and len(model) > len(ONLINE_MODEL_SUFFIX):
        return model[: -len(ONLINE_MODEL_SUFFIX)], True
    return model, False


def _invalid(param: str, exc: ValidationError) -> Exception:
    detail = exc.errors(include_url=False)[0]
    message = str(detail["msg"]).removeprefix("Value error, ")
    return invalid_field(param, f"Invalid value for '{param}': {message}.")


def chat_web_search(
    *,
    options: WebSearchOptions | None,
    plugins: Sequence[ChatPlugin],
    online_suffix: bool,
) -> GatewayWebSearch | None:
    """Normalize the Chat spellings into one gateway web-search request.

    Precedence when several are present: the ``web`` plugin (it carries the
    most detail), then ``web_search_options``, then the ``:online`` suffix.

    Args:
        options: Validated ``web_search_options``, when sent.
        plugins: Validated ``plugins`` entries, when sent.
        online_suffix: Whether the model name carried ``:online``.

    Returns:
        The normalized request, or ``None`` when nothing asked for search.

    Raises:
        OpenAIProtocolError: A plugin other than ``web`` was named, or the
            filters are incoherent.
    """
    web: ChatPlugin | None = None
    for index, plugin in enumerate(plugins):
        if plugin.id != "web":
            raise invalid_field(
                f"plugins.{index}.id",
                f"Invalid value for 'plugins.{index}.id': only the 'web' plugin is "
                "served by this gateway.",
            )
        if plugin.enabled:
            web = plugin
    try:
        if web is not None:
            return GatewayWebSearch(
                declared_as="plugin",
                max_results=web.max_results
                or results_for_context_size(
                    options.search_context_size if options is not None else None
                ),
                search_context_size=options.search_context_size if options is not None else None,
                allowed_domains=web.include_domains,
                blocked_domains=web.exclude_domains,
                user_location=web.user_location
                or (options.user_location if options is not None else None),
                search_prompt=web.search_prompt,
            )
        if options is not None:
            return GatewayWebSearch(
                declared_as="web_search_options",
                max_results=results_for_context_size(options.search_context_size),
                search_context_size=options.search_context_size,
                user_location=options.user_location,
            )
    except ValidationError as exc:
        raise _invalid("plugins" if web is not None else "web_search_options", exc) from exc
    if online_suffix:
        return GatewayWebSearch(declared_as="model_suffix")
    return None


def _domains(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str))


def responses_web_search(
    native_tools: Sequence[GatewayProviderNativeTool], *, online_suffix: bool
) -> GatewayWebSearch | None:
    """Normalize a hosted ``web_search`` tool (or ``:online``) on Responses.

    Args:
        native_tools: The request's opaque non-function tool declarations.
        online_suffix: Whether the model name carried ``:online``.

    Returns:
        The normalized request, or ``None``.

    Raises:
        OpenAIProtocolError: The declaration's filters are incoherent.
    """
    for entry in native_tools:
        raw = entry.tool
        if raw.get("type") not in RESPONSES_WEB_SEARCH_TOOL_TYPES:
            continue
        raw_size = raw.get("search_context_size")
        size = _CONTEXT_SIZES.get(raw_size) if isinstance(raw_size, str) else None
        filters = raw.get("filters")
        location = raw.get("user_location")
        try:
            return GatewayWebSearch(
                declared_as="responses_tool",
                max_results=results_for_context_size(size),
                search_context_size=size,
                allowed_domains=_domains(
                    filters.get("allowed_domains") if isinstance(filters, dict) else None
                ),
                blocked_domains=_domains(
                    filters.get("blocked_domains") if isinstance(filters, dict) else None
                ),
                user_location=location if isinstance(location, dict) else None,
            )
        except ValidationError as exc:
            raise _invalid(f"tools.{entry.index}", exc) from exc
    if online_suffix:
        return GatewayWebSearch(declared_as="model_suffix")
    return None

"""Anthropic server-tool entries on the Messages surface.

Server tools (``web_search_20250305``-style) execute at the provider and
carry no ``input_schema``; their per-type configuration is an evolving
provider surface, so only the discriminator pair is validated and the raw
entry forwards byte-for-byte on native Anthropic rungs. A ``web_search_*``
entry additionally normalizes into the gateway's own web-search request so a
route with no Anthropic rung can run the search itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.manifest import MESSAGES_SERVER_TOOL_TYPES_ACCEPTED
from exp.runtime.gateway.tool_search.contracts import (
    MESSAGES_TOOL_SEARCH_TYPES,
    GatewayToolSearch,
    mode_for_messages_type,
)
from exp.runtime.gateway.web_search.contracts import (
    DEFAULT_WEB_SEARCH_RESULTS,
    MAXIMUM_WEB_SEARCH_RESULTS,
    GatewayWebSearch,
)
from exp.runtime.openai_protocol.errors import invalid_field


class ServerTool(BaseModel):
    """One Anthropic server tool, validated shallowly and carried verbatim."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_server_type(self) -> ServerTool:
        """Reject the custom discriminator: custom tools take the strict model."""
        if self.type == "custom":
            raise ValueError("custom tools must declare an input_schema")
        return self


def require_served_server_tool_types(tools: Sequence[object]) -> None:
    """Reject any server tool type the gateway cannot serve truthfully.

    Acceptance means the data plane carries every block the tool makes the
    provider stream (see the decision tables in ``manifest.py``); an
    unclassified type stays rejected until the SDK drift gate forces its
    decision, so a new provider tool never half-works silently.

    Args:
        tools: The decoded ``tools`` entries (custom tools and server tools).

    Raises:
        OpenAIProtocolError: A tool entry names an unserved server tool type.
    """
    for tool_index, tool in enumerate(tools):
        if isinstance(tool, ServerTool) and tool.type not in MESSAGES_SERVER_TOOL_TYPES_ACCEPTED:
            supported = ", ".join(sorted(MESSAGES_SERVER_TOOL_TYPES_ACCEPTED))
            raise invalid_field(
                f"tools.{tool_index}.type",
                f"the server tool type '{tool.type}' is not supported by this gateway. "
                f"Supported server tool types: {supported}. Remove the tool or use a "
                "supported type.",
            )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str))


def messages_web_search(payload: JsonObject, tools: Sequence[object]) -> GatewayWebSearch | None:
    """Normalize the first ``web_search_*`` server tool into a gateway search.

    Args:
        payload: Raw request body (its ``tools`` array carries the verbatim entry).
        tools: The decoded ``tools`` entries, positionally aligned with the payload.

    Returns:
        The normalized request, or ``None`` when no web-search server tool is declared.

    Raises:
        OpenAIProtocolError: The entry's filters are incoherent (both allowed
            and blocked domains, a non-host filter entry).
    """
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list):
        return None
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, ServerTool) or not tool.type.startswith("web_search"):
            continue
        raw = cast(JsonObject, raw_tools[tool_index])
        max_uses = raw.get("max_uses")
        location = raw.get("user_location")
        try:
            return GatewayWebSearch(
                declared_as="messages_server_tool",
                max_results=min(DEFAULT_WEB_SEARCH_RESULTS, MAXIMUM_WEB_SEARCH_RESULTS),
                allowed_domains=_string_tuple(raw.get("allowed_domains")),
                blocked_domains=_string_tuple(raw.get("blocked_domains")),
                user_location=location if isinstance(location, dict) else None,
                max_uses=max_uses if isinstance(max_uses, int) and max_uses > 0 else None,
            )
        except ValidationError as exc:
            detail = exc.errors(include_url=False)[0]
            raise invalid_field(
                f"tools.{tool_index}",
                "Invalid value for 'tools': "
                + str(detail["msg"]).removeprefix("Value error, ")
                + ".",
            ) from exc
    return None


def messages_tool_search(tools: Sequence[object]) -> GatewayToolSearch | None:
    """Normalize the first ``tool_search_tool_*`` server tool into a gateway declaration.

    Args:
        tools: The decoded ``tools`` entries.

    Returns:
        The normalized declaration, or ``None`` when none is declared.
    """
    for tool in tools:
        if isinstance(tool, ServerTool) and tool.type in MESSAGES_TOOL_SEARCH_TYPES:
            return GatewayToolSearch(
                declared_as="messages_server_tool",
                mode=mode_for_messages_type(tool.type),
                tool_type=tool.type,
                tool_name=tool.name,
            )
    return None

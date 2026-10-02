"""Immutable function tools, provider-native tools, and named tool choices."""

from __future__ import annotations

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject


class GatewayToolDefinition(ContractModel):
    """A caller-defined function with exact schema and provider-owned carriers.

    Attributes:
        name: Function name, bounded to 256 characters.
        description: Unbounded caller description; body and provider context limits apply.
        parameters: The caller JSON Schema, unchanged.
        strict: Whether the provider must enforce the schema.
        cache_control: Validated caching hint, excluded from replay identity.
        eager_input_streaming: Native Anthropic streaming selector.
        defer_loading: Native deferred-loading selector; provider validates combinations.
        allowed_callers: Native programmatic-tool caller allowlist.
        input_examples: Provider-visible example inputs counted in reservation.

    Native carriers join replay identity except caching hints, which change cost only.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    parameters: JsonObject
    strict: bool = False
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    eager_input_streaming: bool | None = Field(default=None, exclude=True)
    defer_loading: bool | None = Field(default=None, exclude=True)
    allowed_callers: tuple[str, ...] | None = Field(default=None, exclude=True)
    input_examples: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)

    def has_anthropic_tool_carriers(self) -> bool:
        """Whether any Anthropic-native tool carrier is present on this tool."""
        return (
            self.eager_input_streaming is not None
            or self.defer_loading is not None
            or self.allowed_callers is not None
            or self.input_examples is not None
        )


class GatewayProviderNativeTool(ContractModel):
    """A provider-owned native Responses tool carried only on its own wire.

    Attributes:
        index: Caller tools-array position, preserving mixed declaration order.
        tool: Shallowly validated declaration; provider owns the internal schema.
    """

    index: int = Field(ge=0)
    tool: JsonObject


class GatewayNamedToolChoice(ContractModel):
    """A request to require one named caller-defined function.

    Attributes:
        name: Exact nonempty caller function name, at most 256 characters.
    """

    name: str = Field(min_length=1, max_length=256)

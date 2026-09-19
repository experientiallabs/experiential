"""Immutable function tools, provider-native tools, and named tool choices."""

from __future__ import annotations

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject


class GatewayToolDefinition(ContractModel):
    """One caller-defined function tool with its exact JSON Schema declaration.

    The description bound is deliberately generous: both providers accept
    40k-character tool descriptions live (verified 2026-08-30), and real
    Claude Code toolsets exceeded the earlier 8k bound. The request-body
    size cap remains the effective total limit.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    parameters: JsonObject
    strict: bool = False
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching hint attached to this tool definition,
    forwarded onto the native Anthropic tool block and dropped with
    disclosure on other wires. Like ``ToolCall.cache_control``, a cache hint
    changes cost, not semantics: it joins neither serialization nor replay
    identity."""
    eager_input_streaming: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic fine-grained tool-input streaming selector, sent
    conditionally by Claude Code and accepted bare by the provider (verified
    live 2026-08-30, no beta header). Excluded from serialization (tool
    digests predate it); a present value joins replay identity through
    :func:`canonical_request_sha256`, like every carrier below."""
    defer_loading: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic tool-search deferred-loading selector; the provider
    owns the cross-tool validity rules (verified live 2026-08-30: ``false``
    is a no-op and an all-deferred toolset is the provider's own 400)."""
    allowed_callers: tuple[str, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic programmatic-tool-calling caller allowlist,
    accepted bare by the provider even without a companion server tool
    (verified live 2026-08-30), which stays the combination authority."""
    input_examples: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic example tool inputs.

    Accepted bare by the provider (verified live 2026-08-30). Examples add
    provider-visible prompt content, so a present value is excluded from
    serialization and joins replay identity through
    :func:`canonical_request_sha256`; reservation counts its bytes with the
    rest of the replay envelope.
    """

    def has_anthropic_tool_carriers(self) -> bool:
        """Whether any Anthropic-native tool carrier is present on this tool."""
        return (
            self.eager_input_streaming is not None
            or self.defer_loading is not None
            or self.allowed_callers is not None
            or self.input_examples is not None
        )


class GatewayProviderNativeTool(ContractModel):
    """One verbatim non-function OpenAI Responses tool declaration.

    Codex ships ``custom`` (freeform grammar), ``namespace`` (nested tool tree),
    ``web_search``, and ``tool_search`` declarations whose shapes exist on no
    other wire; each is validated shallowly at decode and re-emitted byte-for-byte
    on native Responses rungs only, with the provider owning the declaration's
    internal shape (each type captured live from Codex 0.151.0 and accepted with
    a plain API key, 2026-09-01). ``index`` is the declaration's position in the
    caller's ``tools`` array so re-emission preserves the caller's interleaving.
    """

    index: int = Field(ge=0)
    tool: JsonObject


class GatewayNamedToolChoice(ContractModel):
    """A request to require one named caller-defined function."""

    name: str = Field(min_length=1, max_length=256)

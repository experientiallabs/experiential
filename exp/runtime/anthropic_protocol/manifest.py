"""Executable closed compatibility manifest for the Anthropic Messages surface."""

from __future__ import annotations

from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityField,
    CompatibilityManifest,
)
from exp.runtime.gateway.contracts import GatewayApiSurface


def _field(
    path: str,
    disposition: CompatibilityDisposition,
    capability: str | None = None,
) -> CompatibilityField:
    """Create one concise compatibility field declaration."""
    return CompatibilityField(field_path=path, disposition=disposition, capability=capability)


MESSAGES_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.MESSAGES,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (
                "model",
                "messages",
                "max_tokens",
                "system",
                "temperature",
                "top_p",
                "top_k",
                "stop_sequences",
                "stream",
            )
        ),
        _field("tools", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field("tool_choice", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field("thinking", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "extended_thinking"),
        _field(
            "context_management",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "context_management",
        ),
        _field(
            "output_config",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "output_config",
        ),
        _field(
            "diagnostics",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "diagnostics",
        ),
        _field(
            "speed",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "fast_mode",
        ),
        _field(
            "cache_control",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "prompt_caching",
        ),
        _field(
            "inference_geo",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "inference_geo",
        ),
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        # Conscious rejections, each with a recorded reason:
        # - ``thread`` is Anthropic's server-held conversation state (it
        #   replaces ``messages`` upstream); the stateless gateway cannot
        #   proxy it truthfully, so it stays rejected until a verified
        #   contract exists.
        # - ``container`` and ``mcp_servers`` bind provider-hosted execution
        #   state this gateway does not serve (server tools whose result
        #   blocks the gateway cannot carry are rejected on the same grounds;
        #   see MESSAGES_SERVER_TOOL_TYPES_ACCEPTED).
        # - ``service_tier`` selects provider priority capacity priced above
        #   the gateway's committed billing model.
        # - ``fallbacks`` and ``fallback_credit_token`` swap the upstream
        #   model mid-request, which would falsify this gateway's committed
        #   route identity and billing (the matching ``server-side-fallback-*``
        #   and ``fallback-credit-*`` beta tokens are dropped for the same
        #   reason).
        # - ``betas`` (the body-level form) is redundant with the
        #   ``anthropic-beta`` header, which is the one allowlisted opt-in
        #   channel; a second, body-borne channel would bypass it.
        # - ``user_profile_id`` attributes the request to another party under
        #   the ``user-profiles`` beta, an org-trust delegation this gateway
        #   does not make (the provider itself rejects the bare field,
        #   verified live 2026-08-30).
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "container",
                "mcp_servers",
                "service_tier",
                "thread",
                "fallbacks",
                "fallback_credit_token",
                "betas",
                "user_profile_id",
            )
        ),
    ),
)


MESSAGES_TOOL_FIELDS_ACCEPTED = frozenset(
    {
        "name",
        "description",
        "input_schema",
        "type",
        "cache_control",
        "strict",
        "eager_input_streaming",
        "defer_loading",
        "allowed_callers",
        "input_examples",
    }
)
"""Custom tool-definition fields the strict decoder accepts.

This is the tool-level half of the drift gate: every field on the official
SDK's custom ``ToolParam`` must appear here or in
:data:`MESSAGES_TOOL_FIELDS_REJECTED`, so an SDK bump turns new tool surface
into a loud decision instead of a silent customer-facing 400 (the
``eager_input_streaming`` incident class). ``strict`` maps onto the canonical
tool contract; ``cache_control`` forwards on Anthropic rungs as a cost-only
hint; the remaining provider-native annotations forward verbatim on Anthropic
rungs, which own their validity rules, and drop with disclosure elsewhere.
"""

MESSAGES_TOOL_FIELDS_REJECTED: frozenset[str] = frozenset()
"""Custom tool-definition fields consciously rejected, currently none.

Anthropic-defined tools (bash, text editor, web search, code execution,
tool search) are decided per type through
:data:`MESSAGES_SERVER_TOOL_TYPES_ACCEPTED` and
:data:`MESSAGES_SERVER_TOOL_TYPES_REJECTED` instead of field-by-field: an
accepted entry forwards verbatim, so its per-type configuration is the
provider's own validity surface.
"""


MESSAGES_SERVER_TOOL_TYPES_ACCEPTED = frozenset(
    {
        "web_search_20250305",
        "web_search_20260209",
        "web_search_20260318",
    }
)
"""Anthropic-defined tool types the gateway forwards verbatim.

The bar for acceptance is truthful end-to-end serving, not decode success:
the data plane must carry every block the tool makes the provider stream.
All three web_search versions were verified live (2026-08-31) to emit the
same block vocabulary this gateway carries intact: ``server_tool_use``,
``web_search_tool_result``, and citation-bearing ``text`` blocks (the newer
two default to programmatic calling on some models; that 400 is the
provider's own, named and actionable). Claude Code's WebSearch sends
``web_search_20250305`` by default.
"""

MESSAGES_SERVER_TOOL_TYPES_REJECTED = frozenset(
    {
        "advisor_20260301",
        "bash_20241022",
        "bash_20250124",
        "browser_toolset_20260801",
        "code_execution_20250522",
        "code_execution_20250825",
        "code_execution_20260120",
        "code_execution_20260521",
        "computer_20241022",
        "computer_20250124",
        "computer_20251124",
        "computer_toolset_20260801",
        "mcp_toolset",
        "memory_20250818",
        "text_editor_20241022",
        "text_editor_20250124",
        "text_editor_20250429",
        "text_editor_20250728",
        "tool_search_tool_bm25",
        "tool_search_tool_bm25_20251119",
        "tool_search_tool_regex",
        "tool_search_tool_regex_20251119",
        "web_fetch_20250910",
        "web_fetch_20260209",
        "web_fetch_20260309",
        "web_fetch_20260318",
    }
)
"""Anthropic-defined tool types consciously rejected by name.

Each is rejected because the gateway cannot yet serve it truthfully, not
because the provider would refuse it: ``web_fetch_*``, ``code_execution_*``,
and ``tool_search_*`` stream result blocks the data plane does not carry
(silently dropping them would falsify the response); ``code_execution_*``,
``browser_toolset_*``, ``computer*``, and ``mcp_toolset`` additionally bind
provider-hosted execution state; ``computer*``, ``bash_*``, ``text_editor_*``,
``memory_*``, and ``advisor_*`` are client-executed but unverified here and
several require beta headers this gateway does not forward. Accepting one
means moving it to :data:`MESSAGES_SERVER_TOOL_TYPES_ACCEPTED` with live
block-vocabulary evidence, exactly as web_search was.
"""


MESSAGES_BETA_TOKENS_FORWARDED = frozenset(
    {
        "context-1m-2025-08-07",
        "interleaved-thinking-2025-05-14",
        "context-management-2025-06-27",
        "cache-diagnosis-2026-04-07",
        "fast-mode-2026-02-01",
    }
)
"""Caller ``anthropic-beta`` tokens forwarded verbatim on Anthropic rungs.

A caller beta header is operator-trust surface, so forwarding is an exact
allowlist, never a pattern: each listed token's behavior is understood and
safe to relay. ``context-1m-2025-08-07`` activates the 1M context window
(Claude Code sends it with 1M-suffixed models; without forwarding, the
provider serves 200K and long sessions fail). ``interleaved-thinking``
gates thinking between tool calls, whose blocks this gateway already
carries opaquely. The remaining three are the same tokens the gateway
injects itself when the bound field (``context_management``,
``diagnostics``, ``speed``) is present. Every other caller token is
validated and dropped with an ``anthropic-beta.<token>`` disclosure, never
rejected and never blind-forwarded; notable deliberate drops are
``server-side-fallback-*`` and ``fallback-credit-*`` (an upstream model
swap would falsify this gateway's committed route identity and billing)
and ``claude-code-20250219`` (an umbrella product token with an
unenumerated behavior surface).
"""

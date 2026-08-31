"""Executable closed compatibility manifest for the Anthropic Messages surface."""

from __future__ import annotations

from exp.runtime.gateway.contracts import (
    CompatibilityDisposition,
    CompatibilityField,
    CompatibilityManifest,
    GatewayApiSurface,
)


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
        #   state this gateway does not serve (server tools are rejected on
        #   the same grounds).
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

Server-defined tools (bash, text editor, web search, code execution, tool
search) are rejected wholesale through the closed ``type`` literal instead
of field-by-field: the gateway serves no provider-hosted execution.
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

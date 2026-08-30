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
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        # ``thread`` is Anthropic's server-held conversation state (it
        # replaces ``messages`` upstream); the stateless gateway cannot
        # proxy it truthfully, so it stays a conscious rejection until a
        # verified contract exists.
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "container",
                "mcp_servers",
                "service_tier",
                "thread",
            )
        ),
    ),
)


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

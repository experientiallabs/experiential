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
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "container",
                "context_management",
                "mcp_servers",
                "output_config",
                "service_tier",
            )
        ),
    ),
)

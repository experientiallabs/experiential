"""Executable closed compatibility manifest for the Gemini ``generateContent`` surface."""

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


# Google's REST surface accepts both JSON spellings of every field (the
# protobuf JSON mapping), so each top-level field is classified under both.
_SPELLINGS: dict[str, tuple[str, ...]] = {
    "contents": ("contents",),
    "systemInstruction": ("systemInstruction", "system_instruction"),
    "tools": ("tools",),
    "toolConfig": ("toolConfig", "tool_config"),
    "generationConfig": ("generationConfig", "generation_config"),
    "safetySettings": ("safetySettings", "safety_settings"),
    "cachedContent": ("cachedContent", "cached_content"),
    "labels": ("labels",),
    "serviceTier": ("serviceTier", "service_tier"),
    "store": ("store",),
    "continuationToken": ("continuationToken", "continuation_token"),
}

GENERATE_CONTENT_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.GENERATE_CONTENT,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (*_SPELLINGS["contents"], *_SPELLINGS["systemInstruction"])
        ),
        *(
            _field(path, CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools")
            for path in (*_SPELLINGS["tools"], *_SPELLINGS["toolConfig"])
        ),
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in _SPELLINGS["generationConfig"]
        ),
        # The gateway runs its own content guardrails per identity; Google's
        # per-category thresholds are accepted and dropped with disclosure.
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (*_SPELLINGS["safetySettings"], *_SPELLINGS["labels"])
        ),
        # Google's processing tier and stored-response switch have no gateway
        # counterpart on this surface: accepted and dropped with disclosure.
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (*_SPELLINGS["serviceTier"], *_SPELLINGS["store"])
        ),
        # Explicit caches and continuation tokens are provider resources the
        # gateway cannot address (it caches Gemini prompts automatically and
        # never resumes a provider-side generation), so both are refused.
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (*_SPELLINGS["cachedContent"], *_SPELLINGS["continuationToken"])
        ),
        # The gateway's cross-surface extensions (sent through the SDK's
        # ``http_options.extra_body``): the ZDR demand and the request policy.
        _field("provider", CompatibilityDisposition.SUPPORTED),
        _field("gateway", CompatibilityDisposition.SUPPORTED),
    ),
)

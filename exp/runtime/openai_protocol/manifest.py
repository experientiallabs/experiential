"""Executable closed compatibility manifests for both shared OpenAI surfaces."""

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


CHAT_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.CHAT_COMPLETIONS,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (
                "model",
                "messages",
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "stream",
                "stream_options",
            )
        ),
        _field("tools", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field("stop", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "stop_sequences"),
        _field("tool_choice", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field(
            "parallel_tool_calls",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "parallel_tool_calls",
        ),
        _field(
            "response_format",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "structured_output",
        ),
        _field("reasoning_effort", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"),
        _field("top_k", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "top_k"),
        _field("logprobs", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "logprobs"),
        _field("top_logprobs", CompatibilityDisposition.UNSUPPORTED),
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "audio",
                "frequency_penalty",
                "function_call",
                "functions",
                "logit_bias",
                "modalities",
                "n",
                "prediction",
                "presence_penalty",
                "seed",
                "service_tier",
                "store",
                "user",
                "verbosity",
                "web_search_options",
            )
        ),
    ),
)

RESPONSES_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.RESPONSES,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (
                "model",
                "input",
                "instructions",
                "previous_response_id",
                "max_output_tokens",
                "temperature",
                "top_p",
                "stream",
            )
        ),
        _field("tools", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field("tool_choice", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field(
            "parallel_tool_calls",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "parallel_tool_calls",
        ),
        _field("text", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "structured_output"),
        _field("reasoning", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"),
        _field("top_k", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "top_k"),
        _field("top_logprobs", CompatibilityDisposition.UNSUPPORTED),
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "background",
                "conversation",
                "include",
                "max_tool_calls",
                "prompt",
                "service_tier",
                "store",
                "stream_options",
                "truncation",
                "user",
            )
        ),
    ),
)


def disposition_map(manifest: CompatibilityManifest) -> dict[str, CompatibilityDisposition]:
    """Index one manifest by exact top-level request field.

    Args:
        manifest: Compatibility manifest to index.

    Returns:
        Field path to disposition mapping.
    """
    return {field.field_path: field.disposition for field in manifest.fields}

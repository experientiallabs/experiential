"""Executable closed compatibility manifests for both shared OpenAI surfaces."""

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
                "service_tier",
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
        # End-user attribution / cache hints (OpenAI spec). Accepted and recorded
        # gateway-side, never forwarded to the model: `safety_identifier` is the
        # current stable end-user identifier, `user` its deprecated predecessor,
        # `prompt_cache_key` a same-prefix cache-routing hint (not identity).
        _field("safety_identifier", CompatibilityDisposition.METADATA_ONLY),
        _field("user", CompatibilityDisposition.METADATA_ONLY),
        _field("prompt_cache_key", CompatibilityDisposition.METADATA_ONLY),
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "audio",
                "frequency_penalty",
                "function_call",
                "functions",
                "logit_bias",
                "modalities",
                "moderation",
                "n",
                "prediction",
                "presence_penalty",
                "prompt_cache_options",
                "prompt_cache_retention",
                "seed",
                "store",
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
                "store",
            )
        ),
        _field("tools", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field("include", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "encrypted_reasoning"),
        _field("tool_choice", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "function_tools"),
        _field(
            "parallel_tool_calls",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "parallel_tool_calls",
        ),
        _field("text", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "structured_output"),
        _field(
            "client_metadata",
            CompatibilityDisposition.CONDITIONALLY_SUPPORTED,
            "client_metadata",
        ),
        _field("reasoning", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"),
        _field("top_k", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "top_k"),
        _field("top_logprobs", CompatibilityDisposition.UNSUPPORTED),
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        # End-user attribution / cache hints (OpenAI spec), same handling as the
        # Chat surface: accepted and recorded gateway-side, never forwarded.
        _field("safety_identifier", CompatibilityDisposition.METADATA_ONLY),
        _field("user", CompatibilityDisposition.METADATA_ONLY),
        _field("prompt_cache_key", CompatibilityDisposition.METADATA_ONLY),
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "background",
                "context_management",
                "conversation",
                "max_tool_calls",
                "moderation",
                "prompt",
                "prompt_cache_options",
                "prompt_cache_retention",
                "service_tier",
                "stream_options",
                "truncation",
            )
        ),
    ),
)


RESPONSES_REASONING_FIELDS_ACCEPTED = frozenset(
    {"effort", "summary", "generate_summary", "context"}
)
"""Nested ``reasoning`` object fields the Responses decoder models."""

RESPONSES_REASONING_FIELDS_REJECTED = frozenset({"mode"})
"""Nested ``reasoning`` fields consciously rejected with a named 400.

``mode`` selects provider-priced reasoning tiers ("standard"/"pro"); pricing
authority lives in the catalog, so the gateway rejects the field until a
priced route contract exists for it.
"""

RESPONSES_REASONING_EFFORTS_ACCEPTED = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max"}
)
"""``reasoning.effort`` values the decoders accept (route support still applies)."""

RESPONSES_REASONING_CONTEXTS_ACCEPTED = frozenset({"auto", "current_turn", "all_turns"})
"""``reasoning.context`` values the Responses decoder accepts verbatim."""

RESPONSES_REASONING_SUMMARIES_ACCEPTED = frozenset({"auto", "concise", "detailed"})
"""``reasoning.summary`` and ``generate_summary`` values the decoder accepts."""

RESPONSES_INCLUDE_PATHS_ACCEPTED = frozenset({"reasoning.encrypted_content"})
"""``include`` selectors the gateway honors."""

RESPONSES_INCLUDE_PATHS_REJECTED = frozenset(
    {
        "code_interpreter_call.outputs",
        "computer_call_output.output.image_url",
        "file_search_call.results",
        "message.input_image.image_url",
        "message.output_text.logprobs",
        "web_search_call.action.sources",
        "web_search_call.results",
    }
)
"""``include`` selectors consciously rejected: each names a server-tool or
multimodal surface this gateway does not serve, so honoring the selector is
impossible and accepting it would be silent."""


RESPONSES_INPUT_ITEM_FIELDS_ACCEPTED: dict[str, frozenset[str]] = {
    "message": frozenset({"type", "role", "content", "id", "status", "phase"}),
    "function_call": frozenset({"type", "call_id", "name", "arguments", "id", "status"}),
    "function_call_output": frozenset({"type", "call_id", "output", "id", "status"}),
    "reasoning": frozenset({"type", "id", "encrypted_content", "summary", "content", "status"}),
}
"""Echoable input-item fields the Responses decoder models per item type.

Stateless continuations resend prior OUTPUT items verbatim as the next
INPUT, so every field this gateway's own output items carry must decode.
Codex echoes assistant messages with ``phase`` and reasoning items with an
explicit ``content: null`` (both captured live 2026-08-29); the message
``phase`` is retained for replay identity and the null reasoning content
is validated and dropped like its populated form.
"""

RESPONSES_INPUT_ITEM_FIELDS_REJECTED: dict[str, frozenset[str]] = {
    "message": frozenset(),
    "function_call": frozenset({"caller", "namespace"}),
    "function_call_output": frozenset({"caller", "namespace", "name"}),
    "reasoning": frozenset(),
}
"""Echoable input-item fields consciously rejected with a named 400.

``caller``/``namespace`` attribute server-tool invocations this gateway does
not serve, and a ``name`` on a function output duplicates the call linkage
already carried by ``call_id``.
"""


CHAT_CACHE_CONTROL_PLACEMENTS: dict[str, str] = {
    "messages": "validated_and_dropped",
    "messages.content": "validated_and_dropped",
    "messages.tool_calls": "validated_and_forwarded_to_anthropic_tool_use",
}
"""Every Chat-surface ``cache_control`` placement and its conscious decision.

The @ai-sdk stack attaches an Anthropic-style ephemeral cache hint to the
last content part of recent messages for Claude-family model ids; depending
on that part's shape the hint lands on the message, a text part, or inside a
``tool_calls`` entry. Placements are classified here so a new placement is a
recorded decision (this table plus its behavior test), never a silent 400.
Only the tool-call placement forwards: Anthropic defines tool_use-block
caching natively, and non-Anthropic routes disclose the omission through
``ignored_parameters``.
"""


def disposition_map(manifest: CompatibilityManifest) -> dict[str, CompatibilityDisposition]:
    """Index one manifest by exact top-level request field.

    Args:
        manifest: Compatibility manifest to index.

    Returns:
        Field path to disposition mapping.
    """
    return {field.field_path: field.disposition for field in manifest.fields}

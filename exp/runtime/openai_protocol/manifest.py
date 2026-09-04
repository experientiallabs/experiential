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
            )
        ),
        # Forwarded only on BYOK OpenAI-family rungs (the caller pays the
        # provider directly, so tier pricing is theirs); host-funded routes
        # drop it with disclosure because the tier changes provider pricing
        # while the gateway bills catalog rates.
        _field("service_tier", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "service_tier"),
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
        # Alternate enable-thinking shapes admitted and TRANSLATED to the canonical
        # reasoning_effort at decode (the Responses-style nested `reasoning`, the
        # Anthropic-style `thinking`, the vLLM-native `chat_template_kwargs`), so a
        # caller's one payload turns thinking on in any shape.
        _field("reasoning", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"),
        _field("thinking", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"),
        _field(
            "chat_template_kwargs", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "reasoning"
        ),
        _field("top_k", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "top_k"),
        _field("logprobs", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "logprobs"),
        # Sampling penalties: admitted and adapted per rung — honored where the
        # provider supports them, dropped with disclosure where it does not (a
        # soft preference whose absence still returns a valid answer).
        _field("frequency_penalty", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "penalties"),
        _field("presence_penalty", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "penalties"),
        # Accepted only at its no-op default of 1 (the wire model enforces
        # the value): Copilot hardcodes n:1 on every Chat request.
        _field("n", CompatibilityDisposition.SUPPORTED),
        # Retention request accepted only at its no-op default of false (the
        # wire model enforces the value): OpenAI-style agents (omp, opencode,
        # pi) hardcode store:false on every Chat request to opt out of
        # provider-side retention. This gateway never retains Chat output on
        # any rung, so false is already satisfied and store:true is rejected:
        # silently dropping a retention request would be dishonest.
        _field("store", CompatibilityDisposition.SUPPORTED),
        # top_logprobs stays UNSUPPORTED: the gateway response contract does not
        # project logprob arrays yet, so it cannot be honored on any rung —
        # rejecting is the honest outcome (never a silent drop of a probability
        # request). Admit it only once response normalization emits logprobs.
        _field("top_logprobs", CompatibilityDisposition.UNSUPPORTED),
        _field("metadata", CompatibilityDisposition.METADATA_ONLY),
        # End-user attribution / cache hints (OpenAI spec). Accepted and recorded
        # gateway-side, never forwarded to the model: `safety_identifier` is the
        # current stable end-user identifier, `user` its deprecated predecessor,
        # `prompt_cache_key` a same-prefix cache-routing hint (not identity).
        _field("safety_identifier", CompatibilityDisposition.METADATA_ONLY),
        _field("user", CompatibilityDisposition.METADATA_ONLY),
        _field("prompt_cache_key", CompatibilityDisposition.METADATA_ONLY),
        # Audio INPUT rides ``messages`` as an ``input_audio`` content part and
        # is admitted per route; ``audio`` and ``modalities`` request audio
        # OUTPUT, which no route serves.
        *(
            _field(path, CompatibilityDisposition.UNSUPPORTED)
            for path in (
                "audio",
                "function_call",
                "functions",
                "logit_bias",
                "modalities",
                "moderation",
                "prediction",
                "prompt_cache_options",
                "prompt_cache_retention",
                "seed",
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
        # Same BYOK-only forwarding rule as the Chat surface.
        _field("service_tier", CompatibilityDisposition.CONDITIONALLY_SUPPORTED, "service_tier"),
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
        # Accepted only at their no-op values (the wire models enforce them):
        # Copilot hardcodes truncation:"disabled" and
        # prompt_cache_options:{"mode":"implicit"} on every Responses request,
        # and both describe exactly the behavior this gateway already serves
        # (context is never truncated; served routes cache implicitly).
        _field("truncation", CompatibilityDisposition.SUPPORTED),
        _field("prompt_cache_options", CompatibilityDisposition.SUPPORTED),
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
                "prompt_cache_retention",
                "stream_options",
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
"""``include`` selectors consciously rejected: each asks the response to echo
a server-tool or stored-content surface this gateway does not retain (image
input is served, but the gateway does not echo the caller's image back), so
honoring the selector is impossible and accepting it would be silent."""


RESPONSES_INPUT_ITEM_FIELDS_ACCEPTED: dict[str, frozenset[str]] = {
    "message": frozenset({"type", "role", "content", "id", "status", "phase"}),
    "function_call": frozenset(
        {"type", "call_id", "name", "namespace", "arguments", "id", "status"}
    ),
    "function_call_output": frozenset(
        {"type", "call_id", "output", "name", "namespace", "id", "status"}
    ),
    "reasoning": frozenset({"type", "id", "encrypted_content", "summary", "content", "status"}),
}
"""Echoable input-item fields the Responses decoder models per item type.

Stateless continuations resend prior OUTPUT items verbatim as the next
INPUT, so every field this gateway's own output items carry must decode.
Codex echoes assistant messages with ``phase`` and reasoning items with an
explicit ``content: null`` (both captured live 2026-08-29); the message
``phase`` is retained for replay identity and the null reasoning content
is validated and dropped like its populated form. ``namespace`` on a
``function_call`` (and the ``name``/``namespace`` pair on a
``function_call_output``) attributes the call to its declaring nested tool
tree and round-trips verbatim: the provider rejects a namespaced call
replayed without it ("Missing namespace for function_call ..."), which
wedges every later turn of the session.
"""

RESPONSES_INPUT_ITEM_FIELDS_REJECTED: dict[str, frozenset[str]] = {
    "message": frozenset(),
    "function_call": frozenset({"caller"}),
    "function_call_output": frozenset({"caller"}),
    "reasoning": frozenset(),
}
"""Echoable input-item fields consciously rejected with a named 400.

``caller`` attributes server-tool invocations this gateway does not serve.
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


EMBEDDINGS_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.EMBEDDINGS,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (
                "model",
                "input",
                "dimensions",
                "encoding_format",
            )
        ),
        # End-user attribution (OpenAI spec): accepted and recorded gateway-side,
        # never forwarded to the provider. The embeddings body carries no
        # safety_identifier / prompt_cache_key, so `user` is the only one.
        _field("user", CompatibilityDisposition.METADATA_ONLY),
    ),
)


IMAGES_MANIFEST = CompatibilityManifest(
    schema_version=1,
    surface=GatewayApiSurface.IMAGES,
    fields=(
        *(
            _field(path, CompatibilityDisposition.SUPPORTED)
            for path in (
                "model",
                "prompt",
                "n",
                "size",
                "quality",
                "background",
                "output_format",
                "output_compression",
                "moderation",
                "response_format",
                "style",
            )
        ),
        # End-user attribution: recorded gateway-side, never forwarded.
        _field("user", CompatibilityDisposition.METADATA_ONLY),
        # Streaming partial images is a Responses-style event stream the
        # buffered images surface does not carry; a request asking for it is
        # refused explicitly rather than silently answered whole.
        _field("stream", CompatibilityDisposition.UNSUPPORTED),
        _field("partial_images", CompatibilityDisposition.UNSUPPORTED),
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

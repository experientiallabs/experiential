"""Read an explicit JSON ``null`` on an OpenAI-nullable top-level field as absent.

OpenAI's Chat Completions and Responses references document most optional
top-level parameters as nullable with a default (``stream``: "boolean or null,
Optional, Defaults to false"), and the official API serves an explicit
``null`` exactly as if the field were omitted. Clients that serialize every
option they know about (Go structs without ``omitempty``, Python ``urllib``
scripts building a dict of ``None`` defaults) send those nulls on every
request; refusing them would fail every such call with a named 400 on a
value that asks for nothing.

The vocabularies below are the fields the installed official SDK's generated
request types declare nullable (``Optional[...]``; the SDK is generated from
OpenAI's OpenAPI specification, so ``Optional`` there is the spec's
``nullable``). They are pinned literals, locked to the SDK by an equality
test, so a pin bump that changes OpenAI's contract fails the test rather than
silently widening what this gateway accepts. A field OpenAI does not let be
null (``model``, ``messages``, ``input``, ``tools``, ``tool_choice``,
``response_format``, ``user``, ...) is left in place and keeps its named
refusal.

Removing such a null changes nothing the caller asked for, so it carries no
``ignored_parameters`` disclosure (the same convention the other null-valued
repairs follow, such as a ``thinking.display`` of ``null``).
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject

CHAT_NULLABLE_PARAMS = frozenset(
    {
        "audio",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "moderation",
        "n",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning_effort",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "top_logprobs",
        "top_p",
        "verbosity",
    }
)
"""Chat Completions top-level fields OpenAI documents as nullable.

Source: platform.openai.com/docs/api-reference/chat/create, as generated into
``openai.types.chat.completion_create_params.CompletionCreateParamsNonStreaming``.
"""

RESPONSES_NULLABLE_PARAMS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
    }
)
"""Responses top-level fields OpenAI documents as nullable.

Source: platform.openai.com/docs/api-reference/responses/create, as generated
into ``openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming``.
"""


def drop_null_defaults(payload: JsonObject, nullable: frozenset[str]) -> JsonObject:
    """Remove every top-level field in ``nullable`` whose value is JSON ``null``.

    Args:
        payload: Parsed Chat Completions or Responses body.
        nullable: The surface's OpenAI-nullable top-level fields.

    Returns:
        The original payload when no such null is present; otherwise a shallow
        copy without those keys, so each one takes its documented default.
    """
    if not any(payload.get(key, ...) is None for key in nullable):
        return payload
    return {
        key: value for key, value in payload.items() if value is not None or key not in nullable
    }

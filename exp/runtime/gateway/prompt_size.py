# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Pre-dispatch context-window narrowing for the prompt and its output budget.

A prompt that cannot fit any rung's context window is doomed before dispatch:
the provider only 400s it back, after a reservation, a round trip, and with a
provider-specific message. The gateway has no tokenizer for every model, so it
never guesses the exact count. It bounds it from BELOW: at
:data:`MAXIMUM_BYTES_PER_TOKEN` bytes of UTF-8 text per token, real tokenizers
on prose, code, or CJK text all produce MORE tokens than this estimate, so a
prompt whose lower bound (plus the caller's requested output budget) exceeds a
rung's declared window is certain to fail THERE, so that rung is skipped and
the request falls to the next one; only when no rung can hold it is the
request refused, with the exact numbers. The bound is a documented heuristic,
not a tokenizer proof, so it is deliberately loose, and a rung that leaves its
window undeclared is permissive. A prompt under the bound is dispatched and
left to the provider's precise count.

Only text is counted. Inline media (images, audio, documents) tokenizes by its
own rules and is left to the provider.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from exp.common.models.content import TextContentPart
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.errors import ProviderParameterError

if TYPE_CHECKING:
    from exp.runtime.gateway.routing import GatewayRoute

# Conservative UTF-8 bytes per token. English prose runs near 4, code near
# 3.5, CJK near 1.5 to 4.5, and even indentation-heavy code stays well under
# this, so the estimate is a lower bound in practice. It is a heuristic, not a
# tokenizer proof: only text dominated by very long whitespace runs could
# approach it, which is why the margin is twice typical prose rather than the
# tight 5 or 6 a precise count would allow.
MAXIMUM_BYTES_PER_TOKEN = 8

CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"


def prompt_text_bytes(request: GatewayRequest) -> int:
    """Count the UTF-8 bytes of every text the model will read.

    Args:
        request: Canonical gateway request.

    Returns:
        Bytes across message text, tool-call arguments, echoed provider items,
        and tool definitions. Inline media parts contribute nothing.
    """
    total = 0
    for message in request.messages:
        if message.content is not None:
            # ``content`` already joins every text part (the decoders mirror
            # text parts into it), so parts are only read when it is absent.
            total += len(message.content.encode("utf-8"))
        else:
            for part in message.content_parts:
                if isinstance(part, TextContentPart):
                    total += len(part.text.encode("utf-8"))
        for call in message.tool_calls:
            total += len(call.name.encode("utf-8"))
            arguments = (
                call.raw_arguments
                if call.raw_arguments is not None
                else json.dumps(call.arguments, separators=(",", ":"))
            )
            total += len(arguments.encode("utf-8"))
        for verbatim in (message.provider_native_item, message.provider_anthropic_block):
            if verbatim is not None:
                total += len(json.dumps(verbatim, separators=(",", ":")).encode("utf-8"))
    for tool in request.tools:
        total += len(tool.name.encode("utf-8"))
        if tool.description:
            total += len(tool.description.encode("utf-8"))
        total += len(json.dumps(tool.parameters, separators=(",", ":")).encode("utf-8"))
    for entry in request.provider_server_tools:
        total += len(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
    return total


def minimum_prompt_tokens(request: GatewayRequest) -> int:
    """Lower-bound the prompt's token count from its text bytes.

    Args:
        request: Canonical gateway request.

    Returns:
        The fewest tokens any realistic tokenizer produces for this text.
    """
    return prompt_text_bytes(request) // MAXIMUM_BYTES_PER_TOKEN


def context_window_compatible_indexes(
    route: GatewayRoute,
    request: GatewayRequest,
) -> tuple[int, ...]:
    """Return the rungs whose declared context window can hold this request.

    A rung is compatible when it declares no window (permissive: the provider's
    own count decides) or when its window holds the prompt's lower-bound token
    count PLUS the caller's requested ``max_tokens`` (any spelling). An
    explicit ceiling is never increased to meet a provider floor: generation
    policy rejects that rung instead. Omission contributes no output here;
    required-wire caps and financial reservations are derived independently
    from declared bounds at per-rung admission, never from this loose prompt
    estimate. Per-rung windows differ on one model: the Experiential Cloud
    qwen3.8-27b box serves
    262,144 tokens while the OpenRouter and Novita rungs serve 1,000,000 — so
    a request the box cannot hold must fall to a rung that can instead of
    being refused by the box after a reservation and a round trip (140 such
    terminal refusals in the week to 2026-09-15; 116 of them fit the box's
    window on prompt alone and died on prompt + max_tokens).

    ``max_tokens`` is never clamped to a smaller rung's window: the prompt
    estimate is a loose lower bound, so a clamp computed from it could still
    overflow at the provider, and it would silently change the caller's
    request on one rung of the route. The rung is skipped; the refusal (when
    no rung fits) names the largest budget that would.

    Args:
        route: Resolved ordered route.
        request: Canonical request (decoded, or shaped for the provider).

    Returns:
        Strictly increasing indexes into ``route.deployments``.

    Raises:
        ProviderParameterError: No rung can hold the request
            (``code='context_length_exceeded'``); the message carries the
            prompt bound, the requested output budget, and the largest window.
    """
    text_bytes = prompt_text_bytes(request)
    minimum = text_bytes // MAXIMUM_BYTES_PER_TOKEN
    requested = request.maximum_output_tokens or 0
    compatible: list[int] = []
    largest: int | None = None
    reserve = requested
    for index, deployment in enumerate(route.deployments):
        window = (
            None
            if deployment.capabilities is None
            else deployment.capabilities.context_window_tokens
        )
        if window is None or minimum + requested <= window:
            compatible.append(index)
        if window is not None and (largest is None or window > largest):
            largest = window
    if compatible or largest is None:
        return tuple(compatible)
    if minimum > largest:
        param = "input" if request.surface == GatewayApiSurface.RESPONSES else "messages"
        message = (
            f"The prompt is at least {minimum:,} tokens ({text_bytes:,} bytes of text), "
            f"but the largest context window on this model route is {largest:,} tokens. "
            "Shorten the conversation or choose a model with a larger context window."
        )
    else:
        param = request.maximum_output_tokens_parameter or "max_tokens"
        message = (
            f"The prompt is at least {minimum:,} tokens ({text_bytes:,} bytes of text) and "
            f"the request reserves {reserve:,} output tokens, but the largest context "
            f"window on this model route is {largest:,} tokens. Lower {param} to at most "
            f"{largest - minimum:,}, shorten the conversation, or choose a model with a "
            "larger context window."
        )
    raise ProviderParameterError(message=message, param=param, code=CONTEXT_LENGTH_EXCEEDED_CODE)

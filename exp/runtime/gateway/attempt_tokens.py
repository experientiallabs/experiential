# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Realistic worst-case token estimation for one physical gateway attempt.

The planning ``(input, output)`` token bound a single dispatch can consume, per
API surface. It is what :func:`exp.runtime.gateway.budgets.maximum_attempt_cost_micro_usd`
prices, and what the platform's free-tier and token-rate-limit windows reserve
in flight (released to the settled truth on finish). The input half counts the
prompt the provider actually reads with a real BPE tokenizer and adds explicit
headroom; it is an estimate with documented margins, not an upper bound, and
settlement always replaces it with the provider's exact usage. Lives in its
own module so ``budgets`` stays under the repo's hand-authored line ceiling.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import assert_never

import tiktoken
from pydantic import JsonValue

from exp.common.models.content import (
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    TextContentPart,
    VideoContentPart,
)
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest, ServingRequest
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.replay_identity import provider_replay_authority

# Output tokens reserved when neither the caller nor the deployment bounds output
# (an output bound is not a price, so its absence must not unprice a route).
DEFAULT_RESERVATION_OUTPUT_TOKENS = 32_768

RESERVATION_ENCODING = "o200k_base"
"""BPE used to count prompt text for every route.

It is the tokenizer of the OpenAI models the gateway serves most, and the
other served families (Anthropic, DeepSeek, Gemini) tokenize the same text
within the headroom below on production traffic, so one encoding keeps the
estimate deployment-independent: a ladder walk counts once and prices each
candidate from the same number.
"""

INPUT_TOKEN_HEADROOM_PERCENT = 15
"""Multiplicative margin on the counted prompt.

Covers tokenizer drift between the counting BPE and the dispatched model
plus provider-side prompt framing this module does not render (tool-schema
rendering, role separators, cache breakpoints). The estimate is a planning
number: an over-run on one attempt is charged at settlement, so the margin
only needs to keep typical requests from leaking past caps, never to bound
adversarial content.
"""

MESSAGE_FRAMING_TOKENS = 4
"""Role, separator, and priming tokens providers wrap around each message."""

TOOL_FRAMING_TOKENS = 12
"""Per-definition wrapper tokens when a provider renders a tool schema."""

TOOLS_PRESENT_TOKENS = 400
"""Fixed tool-use system prompt a provider injects when any tool is declared
(Anthropic's is documented in the 260 to 400 token range; OpenAI's namespace
preamble is smaller and covered by the same allowance)."""

IMAGE_TOKENS = 2_048
"""Planning tokens per image regardless of carrier: at or above the largest
published per-image charge (Anthropic caps at about 1,600; OpenAI high-detail
tiling and patch-multiplier models stay under this figure)."""

DOCUMENT_TOKENS = 16_384
"""Planning tokens for a document the gateway cannot size (a URL or handle)."""

DOCUMENT_BYTES_PER_TOKEN = 32
"""Decoded PDF bytes per planning token for inline documents: a text page
(tens of kilobytes) lands near its real few-thousand-token charge and a
scanned page (hundreds of kilobytes) over-reserves rather than under."""

DOCUMENT_TOKENS_FLOOR = 2_048
"""Smallest inline-document reservation (one dense page)."""

VIDEO_TOKENS = 32_768
"""Planning tokens for a video the gateway cannot size (a URL or handle):
about two minutes at Gemini's published per-second rate."""

VIDEO_BYTES_PER_TOKEN = 128
"""Decoded video bytes per planning token for inline clips."""

VIDEO_TOKENS_FLOOR = 4_096
"""Smallest inline-video reservation."""

AUDIO_BYTES_PER_TOKEN = 256
"""Decoded audio bytes per planning token for inline clips (about four times
the published per-second rate on compressed audio)."""

AUDIO_TOKENS_FLOOR = 1_024
"""Smallest inline-audio reservation."""

OPAQUE_BYTES_PER_TOKEN = 4
"""Decoded bytes per token for opaque provider carriers (encrypted reasoning,
signatures, sealed content). Their base64 text would tokenize at about 1.5
bytes per token, several times what the provider charges once it decrypts
the carrier back into prose, so the decoded length stands in for the prose.
"""

OPAQUE_MINIMUM_CHARACTERS = 256
"""Shortest string treated as an opaque base64 carrier; anything shorter is
counted as text (a long identifier costs a few extra tokens, never fewer)."""

_OPAQUE_BASE64 = re.compile(r"[A-Za-z0-9+/_=-]{256,}")


@cache
def reservation_encoder() -> tiktoken.Encoding:
    """Return the process-wide cached reservation tokenizer.

    Loading the BPE table takes on the order of a second and may fetch the
    published vocabulary into tiktoken's on-disk cache on a fresh host, so a
    serving process warms it once at bind time instead of on its first
    request.
    """
    return tiktoken.get_encoding(RESERVATION_ENCODING)


def worst_case_attempt_tokens(
    request: ServingRequest,
    deployment: ExactModelDeployment,
) -> tuple[int, int]:
    """Planning ``(input, output)`` tokens for one call.

    Embeddings and image requests consume no completion output, so they reserve
    their estimated input and zero output; a completion reserves its estimated
    input (prompt, tools, media, replayed carriers) and its clamped max output.
    """
    return worst_case_input_tokens(request), worst_case_output_tokens(request, deployment)


def worst_case_input_tokens(request: ServingRequest) -> int:
    """Realistic input tokens for one request, with explicit headroom.

    Deployment-independent, and the only expensive half of the estimate (it
    walks and tokenizes the whole prompt once), so a ladder walk computes it
    once and pairs it with each candidate's cheap output clamp. The result is
    the tokenized prompt text plus per-message and per-tool framing, media
    planning constants, and decoded-length proxies for opaque carriers, all
    scaled by :data:`INPUT_TOKEN_HEADROOM_PERCENT`.
    """
    counter = _PromptCounter()
    match request:
        case EmbeddingsRequest():
            for text in request.inputs:
                counter.text(text)
        case ImagesRequest():
            counter.text(request.prompt)
        case GatewayRequest():
            _count_completion_prompt(request, counter)
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)
    return counter.total()


def _count_completion_prompt(request: GatewayRequest, counter: _PromptCounter) -> None:
    """Feed everything a provider reads for one chat/responses call to ``counter``."""
    for message in request.messages:
        counter.fixed(MESSAGE_FRAMING_TOKENS)
        if message.content is not None:
            counter.text(message.content)
        if message.tool_call_id is not None:
            counter.text(message.tool_call_id)
        for call in message.tool_calls:
            counter.text(call.call_id)
            counter.text(call.name)
            # A retained verbatim argument string is what the provider re-reads;
            # its parsed twin would count the same payload twice.
            if call.raw_arguments is None:
                counter.json(call.arguments)
        for part in message.content_parts:
            _count_content_part(part, counter)
    if request.tools:
        counter.fixed(TOOLS_PRESENT_TOKENS)
    for tool in request.tools:
        counter.fixed(TOOL_FRAMING_TOKENS)
        counter.text(tool.name)
        if tool.description is not None:
            counter.text(tool.description)
        counter.json(tool.parameters)
    if request.structured_text is not None:
        counter.text(request.structured_text.name)
        if request.structured_text.description is not None:
            counter.text(request.structured_text.description)
        counter.json(request.structured_text.json_schema)
    # Excluded provider carriers (replayed reasoning, native items, verbatim
    # configs, retained raw arguments) are provider-read input the plain
    # message fields miss; the replay envelope holds exactly that set.
    replay_envelope = provider_replay_authority(request)
    if replay_envelope is not None:
        counter.json(replay_envelope)


def _count_content_part(
    part: TextContentPart
    | ImageContentPart
    | VideoContentPart
    | AudioContentPart
    | DocumentContentPart,
    counter: _PromptCounter,
) -> None:
    """Add one multimodal part's planning tokens (text parts already flatten
    into the message content, so they add nothing here)."""
    match part:
        case TextContentPart():
            return
        case ImageContentPart():
            counter.fixed(IMAGE_TOKENS)
        case DocumentContentPart():
            if part.name is not None:
                counter.text(part.name)
            if part.data is None:
                counter.fixed(DOCUMENT_TOKENS)
            else:
                counter.fixed(
                    max(
                        DOCUMENT_TOKENS_FLOOR,
                        _decoded_length(part.data) // DOCUMENT_BYTES_PER_TOKEN,
                    )
                )
        case VideoContentPart():
            if part.data is None:
                counter.fixed(VIDEO_TOKENS)
            else:
                counter.fixed(
                    max(VIDEO_TOKENS_FLOOR, _decoded_length(part.data) // VIDEO_BYTES_PER_TOKEN)
                )
        case AudioContentPart():
            counter.fixed(
                max(AUDIO_TOKENS_FLOOR, _decoded_length(part.data) // AUDIO_BYTES_PER_TOKEN)
            )
        case _:  # pragma: no cover - exhaustive over the content-part union.
            assert_never(part)


def _decoded_length(base64_text: str) -> int:
    """Return the byte length a standard base64 string decodes to."""
    padding = len(base64_text) - len(base64_text.rstrip("="))
    return (len(base64_text) * 3) // 4 - padding


class _PromptCounter:
    """Accumulates one request's prompt into a single tokenizer pass.

    Text chunks are joined and encoded exactly once in :meth:`total`; fixed
    planning tokens and opaque-carrier proxies are summed alongside so the
    tokenizer never sees a multi-kilobyte base64 blob it would over-count.
    """

    def __init__(self) -> None:
        """Start an empty prompt."""
        self._chunks: list[str] = []
        self._fixed = 0
        self._opaque_bytes = 0

    def fixed(self, tokens: int) -> None:
        """Add a planning constant."""
        self._fixed += tokens

    def text(self, text: str) -> None:
        """Add prose the provider tokenizes as-is."""
        if text:
            self._chunks.append(text)

    def json(self, value: JsonValue) -> None:
        """Add a JSON payload the provider renders into the prompt.

        Keys and scalars are tokenized as compact JSON, since that is how
        schemas, arguments, and native items reach the model. Long base64
        runs (encrypted reasoning, signatures, inline media inside native
        items) are lifted out first and counted by decoded length.
        """
        scrubbed = self._scrub(value)
        self._chunks.append(json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":")))

    def _scrub(self, value: JsonValue) -> JsonValue:
        """Return ``value`` with opaque carriers replaced by empty strings."""
        if isinstance(value, str):
            if len(value) >= OPAQUE_MINIMUM_CHARACTERS and _OPAQUE_BASE64.fullmatch(value):
                self._opaque_bytes += _decoded_length(value)
                return ""
            return value
        if isinstance(value, dict):
            return {key: self._scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        return value

    def total(self) -> int:
        """Tokenize the gathered prompt once and apply the headroom."""
        counted = 0
        if self._chunks:
            counted = len(reservation_encoder().encode_ordinary("\n".join(self._chunks)))
        counted += self._fixed + self._opaque_bytes // OPAQUE_BYTES_PER_TOKEN
        return (counted * (100 + INPUT_TOKEN_HEADROOM_PERCENT) + 99) // 100


def worst_case_output_tokens(
    request: ServingRequest,
    deployment: ExactModelDeployment,
) -> int:
    """Worst-case output tokens one deployment could emit for this request."""
    match request:
        case EmbeddingsRequest() | ImagesRequest():
            return 0
        case GatewayRequest():
            output_tokens = request.maximum_output_tokens
            deployment_ceiling = (
                deployment.capabilities.maximum_output_tokens
                if deployment.capabilities is not None
                else None
            )
            # Clamp caller output to the deployment ceiling: an unbounded value
            # would inflate the estimate past MAXIMUM_MICRO_USD and mis-refuse a
            # fundable request. Settlement charges actual tokens, not this bound.
            if output_tokens is None:
                output_tokens = deployment_ceiling
            elif deployment_ceiling is not None:
                output_tokens = min(output_tokens, deployment_ceiling)
            if output_tokens is None:
                # No caller value and no ceiling: a realistic default, bounded by
                # any known context window (the model cannot emit past it).
                context_window = (
                    deployment.capabilities.context_window_tokens
                    if deployment.capabilities is not None
                    else None
                )
                output_tokens = (
                    min(DEFAULT_RESERVATION_OUTPUT_TOKENS, context_window)
                    if context_window is not None
                    else DEFAULT_RESERVATION_OUTPUT_TOKENS
                )
            return output_tokens
        case _:  # pragma: no cover - exhaustive over the ServingRequest union.
            assert_never(request)

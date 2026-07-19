"""Shared request mapping / response parsing for the two OpenAI-shaped backends.

`OpenAIProvider` and `AzureOpenAIProvider` differ only in how their client is constructed; the
chat-completion and embedding wire formats are identical, so that logic lives here.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse
from openai import BadRequestError

from wmh.providers.base import Completion, Message, TokenUsage
from wmh.providers.receipt import build_chat_provider_receipt

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


class _ChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_completion_tokens: int,
        temperature: float = ...,
    ) -> ChatCompletion: ...


class _Embeddings(Protocol):
    def create(
        self, *, model: str, input: list[str], dimensions: int = ...
    ) -> CreateEmbeddingResponse: ...


def to_messages(system: str, messages: list[Message]) -> list[ChatCompletionMessageParam]:
    """Fold the system prompt into the message list as OpenAI's leading `system` turn."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": m.role, "content": m.content} for m in messages)
    return cast("list[ChatCompletionMessageParam]", out)


def complete(
    chat_completions: _ChatCompletions,
    model: str,
    system: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None = None,
) -> Completion:
    """Run one chat completion and map it onto our `Completion`.

    `max_completion_tokens` (not the deprecated `max_tokens`) keeps this compatible with GPT 5.5.
    `temperature` is sent ONLY when given: GPT 5.5's reasoning models reject non-default sampling
    params (callers pass None), while OpenAI-compatible servers (vLLM policies) need it.
    """
    if temperature is None:
        response = chat_completions.create(
            model=model,
            messages=to_messages(system, messages),
            max_completion_tokens=max_tokens,
        )
    else:
        try:
            response = chat_completions.create(
                model=model,
                messages=to_messages(system, messages),
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        except BadRequestError as exc:
            # Reasoning-model deployments (GPT-5.x behind Azure/custom endpoints) reject any
            # non-default temperature with a 400 unsupported_value. The caller can't know which
            # models sample; degrade to the model's default rather than failing the request.
            if "temperature" not in str(exc):
                raise
            response = chat_completions.create(
                model=model,
                messages=to_messages(system, messages),
                max_completion_tokens=max_tokens,
            )
    if not response.choices:
        # Content filtering (and some error modes) can return zero choices; surface it clearly
        # rather than letting choices[0] raise a bare IndexError.
        raise ValueError(f"{model} returned no choices")
    text = response.choices[0].message.content or ""
    usage = response.usage
    token_usage = (
        TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)
        if usage is not None
        else TokenUsage()
    )
    return Completion(text=text, usage=token_usage)


def complete_chat(
    chat_completions: object,
    model: str,
    request: ChatRequest,
    *,
    provider: str,
    provider_request_id_header: str,
    max_tokens_field: ChatMaxTokensField,
) -> ChatResponse:
    """Run a validated structured request against an OpenAI-compatible SDK resource."""
    # ChatRequest validates the stable tool-calling core before this SDK boundary. The OpenAI
    # package models its evolving request surface as a large TypedDict union, so the narrow cast
    # preserves forward-compatible extra fields without leaking Any into the public contract.
    resource = cast("Any", chat_completions)
    payload = request.provider_payload(model, max_tokens_field=max_tokens_field)
    started_at = time.time()
    raw_api_response = resource.with_raw_response.create(**payload)
    raw_response = raw_api_response.parse()
    finished_at = time.time()
    # The provider response namespace is not an attestation namespace. Explicitly clear a
    # colliding body field before attaching the only trusted receipt built by this adapter.
    response = ChatResponse.model_validate(raw_response.model_dump(mode="json")).model_copy(
        update={"provider_receipt": None}
    )
    provider_request_id = raw_api_response.headers.get(provider_request_id_header)
    max_tokens_value = payload.get(max_tokens_field)
    temperature = payload.get("temperature")
    if (
        not isinstance(provider_request_id, str)
        or not provider_request_id
        or not response.id
        or not response.model
        or isinstance(max_tokens_value, bool)
        or not isinstance(max_tokens_value, int)
    ):
        return response
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
    ):
        return response
    receipt = build_chat_provider_receipt(
        provider=provider,
        provider_request_id=provider_request_id,
        response_id=response.id,
        requested_model=model,
        response_model=response.model,
        system_fingerprint=response.system_fingerprint,
        request_payload=payload,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens_value,
        max_tokens_field=max_tokens_field,
        started_at_unix_s=started_at,
        finished_at_unix_s=finished_at,
    )
    return response.model_copy(update={"provider_receipt": receipt})


def embed(
    embeddings: _Embeddings, model: str, texts: list[str], dim: int | None = None
) -> list[list[float]]:
    """Embed `texts` against `model` (an OpenAI model id, or an Azure embedding deployment).

    `dim`, when set, requests a specific output dimension via the `dimensions` param (supported by
    text-embedding-3-* and their Azure deployments) so the index and query vectors match.
    """
    response = (
        embeddings.create(model=model, input=texts, dimensions=dim)
        if dim is not None
        else embeddings.create(model=model, input=texts)
    )
    return [item.embedding for item in response.data]

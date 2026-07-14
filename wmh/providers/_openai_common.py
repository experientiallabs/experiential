"""Shared request mapping / response parsing for the two OpenAI-shaped backends.

`OpenAIProvider` and `AzureOpenAIProvider` differ only in how their client is constructed; the
chat-completion and embedding wire formats are identical, so that logic lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse
from openai import BadRequestError

from wmh.providers.base import Completion, Message, TokenUsage

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse
    from openai.types.chat import ChatCompletionMessageParam


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
    chat_completions: object,
    model: str,
    system: str,
    messages: list[Message],
    max_tokens: int,
    *,
    max_tokens_field: ChatMaxTokensField = "max_completion_tokens",
    temperature: float | None = None,
) -> Completion:
    """Run one chat completion and map it onto our `Completion`.

    The canonical model contract selects the output-token field. `temperature` is sent ONLY when
    given: GPT 5.5's reasoning models reject non-default sampling params (callers pass None), while
    OpenAI-compatible servers (vLLM policies) need it.
    """
    request = ChatRequest.model_validate(
        {
            "messages": to_messages(system, messages),
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
    )
    resource = cast("Any", chat_completions)
    try:
        response = resource.create(
            **request.provider_payload(model, max_tokens_field=max_tokens_field)
        )
    except BadRequestError as exc:
        if temperature is None or "temperature" not in str(exc):
            raise
        # Reasoning-model deployments (GPT-5.x behind Azure/custom endpoints) reject any
        # non-default temperature. Retry with the same validated request and no sampling value.
        response = resource.create(
            **request.model_copy(update={"temperature": None}).provider_payload(
                model, max_tokens_field=max_tokens_field
            )
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
    max_tokens_field: ChatMaxTokensField,
) -> ChatResponse:
    """Run a validated structured request against an OpenAI-compatible SDK resource."""
    # ChatRequest validates the stable tool-calling core before this SDK boundary. The OpenAI
    # package models its evolving request surface as a large TypedDict union, so the narrow cast
    # preserves forward-compatible extra fields without leaking Any into the public contract.
    resource = cast("Any", chat_completions)
    response = resource.create(**request.provider_payload(model, max_tokens_field=max_tokens_field))
    return ChatResponse.model_validate(response.model_dump(mode="json"))


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

"""Shared request mapping / response parsing for the two OpenAI-shaped backends.

`OpenAIProvider` and `AzureOpenAIProvider` differ only in how their client is constructed; the
chat-completion and embedding wire formats are identical, so that logic lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from openai import BadRequestError

from wmh.providers.base import Completion, Message, TokenUsage

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

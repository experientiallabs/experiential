"""OpenAI adapter (chat completions + embeddings). Reads OPENAI_API_KEY from the environment."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

from llm_waterfall.adapters.base import missing_sdk_error
from llm_waterfall.types import (
    Backend,
    ChatRequest,
    ChatResponse,
    EmbeddingsUnsupported,
    Message,
    TokenUsage,
)

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIAdapter:
    """GPT-5.x via chat completions; text-embedding-3-* for embeddings."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._client: OpenAI | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import httpx
                        from openai import OpenAI
                    except ModuleNotFoundError as exc:
                        raise missing_sdk_error("openai", "openai") from exc

                    # max_retries=0: the waterfall owns retry policy — SDK retries would stack
                    # multiplicatively under the failover chain. Granular httpx.Timeout so a
                    # dead endpoint fails over after connect_timeout_s, not read_timeout_s.
                    self._client = OpenAI(
                        base_url=self.backend.endpoint,
                        max_retries=0,
                        timeout=httpx.Timeout(
                            self.backend.read_timeout_s,
                            connect=self.backend.connect_timeout_s,
                        ),
                    )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """One chat completion.

        The backend contract selects the output-token field. Omitting a default temperature keeps
        this compatible with GPT-5.x reasoning models, which reject non-default sampling params.
        """
        wire: list[dict[str, str]] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend({"role": m.role, "content": m.content} for m in messages)
        api_messages = cast("list[ChatCompletionMessageParam]", wire)
        chat = self._get_client().chat.completions
        model = self._request_model()
        payload: dict[str, object] = {
            "model": model,
            "messages": api_messages,
            self.backend.chat_max_tokens_field: max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        response = chat.create(**cast("Any", payload))
        if not response.choices:
            # Content filtering (and some error modes) can return zero choices; surface it
            # clearly rather than letting choices[0] raise a bare IndexError.
            raise ValueError(f"{model} returned no choices")
        text = response.choices[0].message.content or ""
        usage = response.usage
        token_usage = (
            TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)
            if usage is not None
            else TokenUsage()
        )
        return text, token_usage

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full OpenAI-compatible tool-calling request through this backend."""
        payload = request.provider_payload(
            self._request_model(), max_tokens_field=self.backend.chat_max_tokens_field
        )
        # The OpenAI SDK's input TypedDict is intentionally not our public contract. This one
        # narrow cast sits at the SDK boundary after ChatRequest validated the structured core;
        # provider_payload preserves forward-compatible extra fields emitted by agent SDKs.
        response = self._get_client().chat.completions.create(**cast("Any", payload))
        return ChatResponse.model_validate(response.model_dump(mode="json"))

    def _request_model(self) -> str:
        """The id sent as `model` on the wire (Azure overrides this with the deployment)."""
        return self.backend.model

    def embed_model_id(self) -> str | None:
        """The model embed() resolves to — the single source of truth for embed attribution."""
        return self.backend.embed_model or _DEFAULT_EMBED_MODEL

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        """Embed against `embed_model_id()` (OpenAI default: text-embedding-3-small)."""
        model = self.embed_model_id()
        if model is None:  # pragma: no cover - only reachable via subclasses
            raise EmbeddingsUnsupported("no embedding deployment configured for this backend")
        embeddings = self._get_client().embeddings
        if self.backend.embed_dim is None:
            response = embeddings.create(model=model, input=texts)
        else:
            response = embeddings.create(
                model=model, input=texts, dimensions=self.backend.embed_dim
            )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        return [item.embedding for item in response.data], TokenUsage(input_tokens=input_tokens)

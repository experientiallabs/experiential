"""OpenAI direct provider using the Responses API. Reads OPENAI_API_KEY from the environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from wmh.providers import _openai_common, _responses_common
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
)

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.shared_params.reasoning import Reasoning


class OpenAIResponsesProvider:
    """GPT 5.x via OpenAI's Responses API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Create and cache the OpenAI SDK client on first use."""
        # Lazy: don't import the SDK or read OPENAI_API_KEY until first use.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # picks up OPENAI_API_KEY from the environment
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Generate a completion through the Responses API.

        Args:
            system: System prompt to send as the first Responses input item.
            messages: Conversation messages following the system prompt.
            temperature: Accepted for provider interface compatibility; not forwarded because the
                benchmark path keeps Responses-model sampling at provider defaults.
            max_tokens: Maximum number of output tokens to request.

        Returns:
            Completion text and token usage parsed from the Responses API response.
        """
        # GPT-5.x Responses models reject non-default sampling in this benchmark path.
        del temperature
        response_input = cast("ResponseInputParam", _responses_input(system, messages))
        responses = self._get_client().responses
        if self.config.reasoning_effort:
            response = responses.create(
                model=self.config.model,
                input=response_input,
                max_output_tokens=max_tokens,
                store=False,
                reasoning=cast("Reasoning", {"effort": self.config.reasoning_effort}),
            )
        else:
            response = responses.create(
                model=self.config.model,
                input=response_input,
                max_output_tokens=max_tokens,
                store=False,
            )
        return Completion(text=_response_text(response), usage=_usage_from_response(response))

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request through the native Responses API."""
        return _responses_common.complete_chat(
            self._get_client().responses,
            self.config.model,
            request,
            reasoning_effort=self.config.reasoning_effort,
            allow_sampling=False,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text through OpenAI's embeddings API.

        Args:
            texts: Text strings to embed.

        Returns:
            One embedding vector per input text.

        Raises:
            ValueError: If `ProviderConfig.embed_model` is unset.
        """
        if self.config.embed_model is None:
            raise ValueError("OpenAIResponsesProvider.embed requires config.embed_model to be set.")
        return _openai_common.embed(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        """Run a cheap completion request and report provider availability."""
        try:
            self.complete(
                "",
                [Message(role="user", content="Reply with exactly: ok")],
                max_tokens=256,
            )
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False,
                kind=self.config.kind,
                model=self.config.model,
                detail=str(exc),
            )
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


def _responses_input(system: str, messages: list[Message]) -> list[dict[str, str]]:
    """Convert the provider message shape into Responses API input items."""
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": message.role, "content": message.content} for message in messages)
    return out


def _get(value: object, key: str) -> object:
    """Read a field from either an SDK object or mapping."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value).get(key)
    return getattr(value, key, None)


def _as_int(value: object) -> int:
    """Coerce numeric SDK usage fields into integers, defaulting invalid values to zero."""
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _response_text(response: object) -> str:
    """Extract generated text from direct or nested Responses SDK shapes."""
    direct = _get(response, "output_text")
    if isinstance(direct, str) and direct:
        return direct

    chunks: list[str] = []
    output = _get(response, "output")
    if isinstance(output, list):
        for item in output:
            content = _get(item, "content")
            if not isinstance(content, list):
                continue
            for block in content:
                text = _get(block, "text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks)


def _usage_from_response(response: object) -> TokenUsage:
    """Extract token usage from a Responses SDK object or mapping."""
    usage = _get(response, "usage")
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_as_int(_get(usage, "input_tokens")),
        output_tokens=_as_int(_get(usage, "output_tokens")),
    )

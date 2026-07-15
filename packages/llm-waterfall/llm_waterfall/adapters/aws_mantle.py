"""AWS Mantle adapter — stub that fails at construction.

TODO: implement against the Bedrock Mantle client (`anthropic.AnthropicBedrockMantle` — the
Messages-API Bedrock endpoint) with `aws_region=backend.region`, credentials via
`boto3.Session(profile_name=backend.profile)`, `max_retries=0`, and the backend's timeouts.
Model ids take an `anthropic.` prefix. Classification already covers its error shapes (the
anthropic SDK exception types + botocore codes).

Failing in `__init__` (not at call time) is deliberate: an unimplemented rung must break
`Waterfall(...)` construction loudly, never abort a live call mid-chain.
"""

from __future__ import annotations

from llm_waterfall.types import Backend, ChatRequest, ChatResponse, Message, TokenUsage


class AwsMantleAdapter:
    """Not implemented yet; constructing one fails fast with the workaround."""

    backend: Backend  # satisfies the Adapter protocol; never assigned (init raises)

    def __init__(self, backend: Backend) -> None:
        raise NotImplementedError(
            "the aws_mantle adapter is not implemented yet; use a 'bedrock' backend for AWS "
            "traffic, or contribute the adapter (see the TODO in "
            "llm_waterfall/adapters/aws_mantle.py)."
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        raise NotImplementedError  # pragma: no cover - unreachable (init raises)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError  # pragma: no cover - unreachable (init raises)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        raise NotImplementedError  # pragma: no cover - unreachable (init raises)

    def embed_model_id(self) -> str | None:
        raise NotImplementedError  # pragma: no cover - unreachable (init raises)

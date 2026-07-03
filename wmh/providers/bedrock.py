"""AWS Bedrock provider (Claude 4.8). Reads AWS_REGION + AWS credentials from the environment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict, cast

from wmh.core.types import JsonValue
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"

# Default Titan text-embeddings model (confirmed reachable; v2 supports `dimensions` 256/512/1024).
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"


class _ContentBlock(TypedDict):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _BedrockResponse(TypedDict):
    content: list[_ContentBlock]
    usage: _Usage


class _TitanEmbedResponse(TypedDict):
    embedding: list[float]


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel with the Anthropic Messages body)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: BaseClient | None = None

    def _get_client(self) -> BaseClient:
        # Lazy: import boto3 and open the client only on first use. region falls back to
        # AWS_REGION / the default boto3 chain when config.region is unset.
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Bound each request so a stalled connection RAISES instead of blocking forever. Without
            # this, a single hung InvokeModel wedges the whole run (long GEPA/eval jobs never
            # finish) and a FallbackProvider can't fail over — it only reacts to raised errors.
            # `read_timeout` is generous because reasoning models can generate for a while at up to
            # `max_tokens` (a mid-generation cutoff wastes the whole call and, under a fallback
            # chain, silently substitutes a different model into an eval).
            #
            # `max_attempts=1` disables botocore's OWN retries on purpose: throttling / 5xx /
            # timeouts should surface IMMEDIATELY to the caller, where FallbackProvider owns retry
            # policy (fail over to the next model). Leaving botocore's adaptive retries on would
            # stack 3 internal attempts per model UNDER our 4-model failover — up to 12 backend
            # calls with back-off for one throttled request — turning graceful degradation into a
            # slow crawl.
            # "standard" mode makes max_attempts mean TOTAL attempts (legacy mode still
            # sneaks in one internal retry, which showed up as a long silent stall before the
            # CLI's own narrated backoff could react).
            client_config = Config(
                connect_timeout=15,
                read_timeout=600,
                retries={"max_attempts": 1, "mode": "standard"},
            )
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.config.region, config=client_config
            )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        # Claude 4.8 rejects sampling params, so temperature is intentionally not forwarded.
        body = {
            "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        raw = self._get_client().invoke_model(modelId=self.config.model, body=json.dumps(body))
        data = cast("_BedrockResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = TokenUsage(
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
        return Completion(text=text, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via Amazon Titan text embeddings on Bedrock (phi for retrieval).

        Titan's InvokeModel embeds one text per call (no batch input), so we loop. `embed_model`
        selects the Titan model (defaults to titan-embed-text-v2); `embed_dim`, when set, requests a
        specific output dimension (v2 supports 256/512/1024) so the index and query vectors match.
        """
        model = self.config.embed_model or _DEFAULT_EMBED_MODEL
        client = self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            body: dict[str, JsonValue] = {"inputText": text}
            if self.config.embed_dim is not None:
                body["dimensions"] = self.config.embed_dim
                body["normalize"] = True
            raw = client.invoke_model(modelId=model, body=json.dumps(body))
            data = cast("_TitanEmbedResponse", json.loads(raw["body"].read()))
            vectors.append(data["embedding"])
        return vectors

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)

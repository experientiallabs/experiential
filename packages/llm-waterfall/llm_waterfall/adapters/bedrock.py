"""AWS Bedrock adapter (Anthropic Messages schema via InvokeModel, Titan embeddings).

Credentials come from the boto3 chain, or a named AWS profile when `Backend.profile` is set —
`boto3.Session(profile_name=...)` — so one waterfall chain can span multiple AWS accounts.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, TypedDict, cast

from llm_waterfall.adapters.base import missing_sdk_error
from llm_waterfall.types import Backend, Message, TokenUsage

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"

# Default Titan text-embeddings model (v2 supports `dimensions` 256/512/1024).
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"


class _ContentBlock(TypedDict):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _MessagesResponse(TypedDict):
    content: list[_ContentBlock]
    usage: _Usage


class _TitanEmbedResponse(TypedDict, total=False):
    embedding: list[float]
    inputTextTokenCount: int


class BedrockAdapter:
    """Claude (and Titan embeddings) via the Bedrock Runtime."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._client: BaseClient | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> BaseClient:
        # Lazy + lock-guarded: boto3 is an optional extra, and boto3.Session construction is not
        # thread-safe (the resulting client is — one Waterfall is shared across thread pools).
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import boto3
                        from botocore.config import Config
                    except ModuleNotFoundError as exc:
                        raise missing_sdk_error("boto3", "bedrock") from exc

                    # Bound each request so a stalled connection RAISES instead of blocking
                    # forever — the waterfall can only fail over on a raised error. read_timeout
                    # is generous because reasoning models can legitimately generate for minutes;
                    # a mid-generation cutoff wastes the whole call and silently substitutes a
                    # different model into an eval.
                    #
                    # total_max_attempts=1 disables botocore's OWN retries on purpose (it counts
                    # the initial request; botocore's `max_attempts` counts retries AFTER it, so
                    # `max_attempts: 1` would still allow one hidden retry). Throttling/5xx/
                    # timeouts must surface IMMEDIATELY to the waterfall, which owns retry policy
                    # — SDK retries stack multiplicatively under the failover chain.
                    config = Config(
                        connect_timeout=self.backend.connect_timeout_s,
                        read_timeout=self.backend.read_timeout_s,
                        retries={"total_max_attempts": 1},
                    )
                    session = boto3.Session(
                        profile_name=self.backend.profile, region_name=self.backend.region
                    )
                    self._client = session.client("bedrock-runtime", config=config)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """One InvokeModel call with the Anthropic Messages body."""
        body: dict[str, object] = {
            "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        # Claude 4.7+ rejects sampling params; only forward temperature when explicitly set.
        if temperature is not None:
            body["temperature"] = temperature
        raw = self._get_client().invoke_model(modelId=self.backend.model, body=json.dumps(body))
        data = cast("_MessagesResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = TokenUsage(
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
        return text, usage

    def embed_model_id(self) -> str | None:
        """The model embed() resolves to — the single source of truth for embed attribution."""
        return self.backend.embed_model or _DEFAULT_EMBED_MODEL

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        """Embed via Amazon Titan (one InvokeModel per text — Titan has no batch input)."""
        model = self.backend.embed_model or _DEFAULT_EMBED_MODEL
        client = self._get_client()
        vectors: list[list[float]] = []
        input_tokens = 0
        for text in texts:
            body: dict[str, object] = {"inputText": text}
            if self.backend.embed_dim is not None:
                body["dimensions"] = self.backend.embed_dim
                body["normalize"] = True
            raw = client.invoke_model(modelId=model, body=json.dumps(body))
            data = cast("_TitanEmbedResponse", json.loads(raw["body"].read()))
            vectors.append(data["embedding"])
            input_tokens += data.get("inputTextTokenCount", 0)
        return vectors, TokenUsage(input_tokens=input_tokens)

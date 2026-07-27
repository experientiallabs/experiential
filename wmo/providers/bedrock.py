"""AWS Bedrock provider (Claude 4.8 / Amazon Nova). Reads AWS credentials from the environment."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from wmo.core.types import JsonValue
from wmo.providers._bedrock_chat import converse_request, converse_response
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    StreamChunk,
    TokenUsage,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"

# Default Titan text-embeddings model (confirmed reachable; v2 supports `dimensions` 256/512/1024).
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

AWS_REGION_ENV = "AWS_REGION"
"""The region variable WMO reads itself. `wmo.config.PROVIDER_ENV_VARS` pins the same literal."""

AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
"""The only region variable boto3 reads on its own; accepted here so both names work."""

REGION_SOURCES: tuple[str, ...] = (
    "the entry/config `region`",
    AWS_REGION_ENV,
    AWS_DEFAULT_REGION_ENV,
    "the active AWS profile's `region` (~/.aws/config), then the instance role",
)
"""Every region source, in the order they are consulted. Named in the no-region error."""

NO_REGION_ERROR = (
    "BedrockProvider has no region, and botocore refuses to build a bedrock-runtime client "
    "without one. Region is resolved in this order, first hit wins: "
    + ", ".join(REGION_SOURCES)
    + ". Set one of them."
)
"""The user-facing no-region failure. Replaces botocore's `NoRegionError`, whose whole message
is "You must specify a region." and which names no variable to set."""


class _ContentBlock(TypedDict):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: NotRequired[int]
    cache_creation_input_tokens: NotRequired[int]


class _BedrockResponse(TypedDict):
    content: list[_ContentBlock]
    usage: _Usage


class _TitanEmbedResponse(TypedDict):
    embedding: list[float]


class _NovaContentBlock(TypedDict):
    text: str


class _NovaMessage(TypedDict):
    content: list[_NovaContentBlock]


class _NovaOutput(TypedDict):
    message: _NovaMessage


class _NovaUsage(TypedDict):
    inputTokens: int
    outputTokens: int


class _NovaResponse(TypedDict):
    output: _NovaOutput
    usage: _NovaUsage


def _is_nova(model_id: str) -> bool:
    """Whether `model_id` is an Amazon Nova model (e.g. `us.amazon.nova-lite-v1:0`)."""
    return ".nova-" in model_id or model_id.startswith("amazon.nova-")


def resolve_region(configured: str | None) -> str | None:
    """The region to hand boto3 explicitly, or None to defer to boto3's own resolution.

    boto3 reads exactly one region variable from the environment: botocore's session mapping for
    `region` is `("region", "AWS_DEFAULT_REGION", None, None)`, so `AWS_REGION` set on its own has
    no effect and a client built with `region_name=None` dies with `NoRegionError` ("You must
    specify a region."). `AWS_REGION` is nonetheless the name WMO documents, prompts for, writes
    into `.env`, and points at in its failure hints, so reading it here is what makes the
    documented variable true. Both names work; nothing boto3 resolves for itself is taken away.

    Order, first hit wins:

    1. `configured`: an explicit `region` on the pool entry or `ProviderConfig`.
    2. `AWS_REGION`.
    3. Returning None, which leaves `AWS_DEFAULT_REGION` and the rest of boto3's chain (the
       active profile's `region` in `~/.aws/config`, then the instance role) untouched.

    Args:
        configured: The explicit region carried by the entry or config, if any.

    Returns:
        The region to pass as boto3's `region_name`, or None to let boto3 resolve it.
    """
    return configured or os.environ.get(AWS_REGION_ENV) or None


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel with the Anthropic Messages body)."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        if api_key is not None:
            # Keeps the backend union uniformly constructible from get_provider while failing
            # loudly: Bedrock has no API-key auth to send the credential to.
            raise ValueError(
                "Bedrock authenticates with AWS credentials (profile/role), not an API key; "
                "drop api_key/api_key_env for this provider"
            )
        self.config = config
        self._client: BaseClient | None = None
        # Model capability lives in WMO's canonical catalog. Subclasses may
        # override this per concrete deployment, but every Bedrock request path
        # consumes the same provider-boundary flag.
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _get_client(self) -> BaseClient:
        # Lazy: import boto3 and open the client only on first use. The region comes from
        # `resolve_region` (config, then AWS_REGION), and None hands the rest of the chain
        # (AWS_DEFAULT_REGION, profile, instance role) back to boto3.
        if self._client is None:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import NoRegionError

            # Bound each request so a stalled connection RAISES instead of blocking forever. Without
            # this, a single hung InvokeModel wedges the whole run (long GEPA/eval jobs never
            # finish) and a failover chain can't fail over — it only reacts to raised errors.
            # `read_timeout` is generous because reasoning models can generate for a while at up to
            # `max_tokens` (a mid-generation cutoff wastes the whole call and, under a fallback
            # chain, silently substitutes a different model into an eval).
            #
            # `total_max_attempts=1` disables botocore's OWN retries on purpose (it counts the
            # initial request; botocore's `max_attempts` counts retries AFTER it): throttling/5xx/
            # timeouts should surface IMMEDIATELY to the caller, where the failover chain owns retry
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
            try:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=resolve_region(self.config.region),
                    config=client_config,
                )
            except NoRegionError as exc:
                # botocore's whole message here is "You must specify a region.", which names
                # nothing to set. It reaches the user verbatim through `verify_via_ping`'s detail
                # (so through `wmo build`'s pre-flight, `wmo providers set` and
                # `wmo providers verify`) and raw from any call that skipped `prepare`. Restate it
                # once, here, as the full resolution order.
                raise ValueError(NO_REGION_ERROR) from exc
        return self._client

    def prepare(self) -> None:
        """Resolve the region locally. Deliberately does NOT build the boto3 client.

        Satisfies `wmo.providers.base.PreparableProvider` for the part that is free, and only that
        part. Two measured reasons the client itself is not built here:

        1. `boto3.client()` walks the credential chain, whose instance-metadata provider makes an
           HTTP request to the link-local metadata endpoint when nothing local resolves (measured
           2.11s on a machine with no AWS config, 0.03s with AWS_EC2_METADATA_DISABLED=true). A
           pre-flight's whole job is to run before any request, so it may not pay that.
        2. It would not even answer the question. With no credentials the client is built anyway
           (`credentials=None`, measured) and raises `NoCredentialsError` only when it signs the
           first request, so absent AWS credentials are NOT locally knowable and stay a first-call
           failure.

        The region is a different story: free to resolve and fatal when missing, because botocore
        raises `NoRegionError` while creating a bedrock-runtime client with no region. It goes
        through the same two steps `_get_client` uses, so this check cannot drift from what the
        client does: `resolve_region` (the entry/config, then AWS_REGION), then a fresh boto3
        session for everything boto3 owns (AWS_DEFAULT_REGION, then the active profile's
        `region`). A fresh `Session` is deliberate: `boto3.client` goes through the cached
        process-wide default session, which snapshots the environment as it was at first use.

        Raises:
            ValueError: No region resolves for this configuration, from any source.
        """
        # Lazy for the same reason as `_get_client`: importing boto3 costs real time, and the
        # registry constructs every backend on any `wmo` command.
        import boto3

        if resolve_region(self.config.region) or boto3.Session().region_name:
            return
        raise ValueError(NO_REGION_ERROR)

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        if _is_nova(self.config.model):
            return self._complete_nova(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )
        if "anthropic" not in self.config.model:
            # Kimi, DeepSeek, and other third-party models: the model-agnostic Converse API.
            return self._complete_converse(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )
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
        # Anthropic-native semantics: cache reads and writes reported BESIDE input_tokens;
        # normalize to TokenUsage's cached-as-subset contract by summing.
        cache_read = data["usage"].get("cache_read_input_tokens", 0)
        cache_write = data["usage"].get("cache_creation_input_tokens", 0)
        usage = TokenUsage(
            input_tokens=data["usage"]["input_tokens"] + cache_read + cache_write,
            output_tokens=data["usage"]["output_tokens"],
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
        )
        return Completion(text=text, usage=usage)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured agent request through Bedrock Converse."""
        normalized = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        raw = self._get_client().converse(**converse_request(normalized, self.config.model))
        return converse_response(raw, self.config.model)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion via ConverseStream (model-agnostic, Claude included).

        Temperature is forwarded only when the model catalog says the model samples (Claude 4.8+
        rejects sampling params), matching the non-streaming paths.
        """
        inference_config: dict[str, JsonValue] = {"maxTokens": max_tokens}
        if self._forward_temperature:
            inference_config["temperature"] = temperature
        kwargs: dict[str, JsonValue] = {
            "modelId": self.config.model,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = [{"text": system}]
        response = self._get_client().converse_stream(**kwargs)
        usage = TokenUsage()
        stream = response["stream"]
        try:
            for event in stream:
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"]["delta"].get("text", "")
                    if text:
                        yield StreamChunk(delta=text)
                elif "metadata" in event:
                    event_usage = event["metadata"].get("usage")
                    if event_usage is not None:
                        # Converse reports cacheReadInputTokens / cacheWriteInputTokens beside
                        # inputTokens; normalize to the cached-as-subset contract.
                        cache_read = int(event_usage.get("cacheReadInputTokens", 0) or 0)
                        cache_write = int(event_usage.get("cacheWriteInputTokens", 0) or 0)
                        usage = TokenUsage(
                            input_tokens=int(event_usage["inputTokens"]) + cache_read + cache_write,
                            output_tokens=int(event_usage["outputTokens"]),
                            cached_input_tokens=cache_read,
                            cache_write_input_tokens=cache_write,
                        )
        finally:
            # botocore's EventStream pins the HTTP connection until closed.
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        yield StreamChunk(done=True, usage=usage)

    def _complete_converse(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Complete via the Converse API (model-agnostic: Kimi, DeepSeek, ...).

        Converse normalizes request/response shapes across vendors; thinking models may emit
        `reasoningContent` blocks, which are skipped — callers get the visible text only.
        """
        kwargs: dict[str, JsonValue] = {
            "modelId": self.config.model,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        response = self._get_client().converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        text = "".join(block["text"] for block in blocks if "text" in block)
        cache_read = int(response["usage"].get("cacheReadInputTokens", 0) or 0)
        cache_write = int(response["usage"].get("cacheWriteInputTokens", 0) or 0)
        usage = TokenUsage(
            input_tokens=int(response["usage"]["inputTokens"]) + cache_read + cache_write,
            output_tokens=int(response["usage"]["outputTokens"]),
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
        )
        return Completion(text=text, usage=usage)

    def _complete_nova(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Complete via an Amazon Nova model (different request/response schema than Anthropic).

        Nova wraps message content in `[{"text": ...}]` blocks, takes sampling params under
        `inferenceConfig`, and returns the reply under `output.message`. Unlike the Claude path,
        `temperature` IS forwarded — Nova accepts it.
        """
        body: dict[str, JsonValue] = {
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            body["system"] = [{"text": system}]
        raw = self._get_client().invoke_model(modelId=self.config.model, body=json.dumps(body))
        data = cast("_NovaResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["output"]["message"]["content"])
        usage = TokenUsage(
            input_tokens=data["usage"]["inputTokens"],
            output_tokens=data["usage"]["outputTokens"],
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

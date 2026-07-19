"""AWS Bedrock provider (Claude 4.8 / Amazon Nova). Reads AWS credentials from the environment."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from llm_waterfall import ChatRequest, ChatResponse
from llm_waterfall.bedrock_chat import bedrock_converse_request, bedrock_converse_response

from wmh.core.types import JsonObject, JsonValue
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)
from wmh.providers.receipt import build_chat_provider_receipt

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"

# Default Titan text-embeddings model (confirmed reachable; v2 supports `dimensions` 256/512/1024).
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
_USAGE_FIELD_NAMES = {
    "totalTokens": "total_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "cacheWriteInputTokens": "cache_write_input_tokens",
}


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


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel with the Anthropic Messages body)."""

    paid_request_attempts: Literal[1] = 1

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: BaseClient | None = None
        # Model capability lives in WMH's canonical catalog. Subclasses may
        # override this per concrete deployment, but every Bedrock request path
        # consumes the same provider-boundary flag.
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _get_client(self) -> BaseClient:
        # Lazy: import boto3 and open the client only on first use. region falls back to
        # AWS_REGION / the default boto3 chain when config.region is unset.
        if self._client is None:
            import boto3
            from botocore.config import Config

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
            client_config = Config(
                connect_timeout=15,
                read_timeout=600,
                retries={"total_max_attempts": 1, "mode": "standard"},
                # Benchmark identities bind the AWS region and model. Refuse ambient SDK endpoint
                # URL overrides, FIPS routing, and dual-stack routing so the recorded route cannot
                # silently execute against a local emulator or a different AWS endpoint variant.
                ignore_configured_endpoint_urls=True,
                use_fips_endpoint=False,
                use_dualstack_endpoint=False,
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
        if self.config.reasoning_effort is not None:
            return self._complete_reasoning(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
        usage = _token_usage(
            data["usage"],
            input_field="input_tokens",
            output_field="output_tokens",
        )
        return Completion(text=text, usage=usage)

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured agent request through Bedrock Converse."""
        normalized = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        wire_request = bedrock_converse_request(
            normalized,
            self.config.model,
            reasoning_effort=self.config.reasoning_effort,
        )
        started_at = time.time()
        raw = self._get_client().converse(**wire_request)
        finished_at = time.time()
        raw_mapping = cast("dict[str, object]", raw)
        response_metadata = raw_mapping.get("ResponseMetadata")
        provider_request_id = (
            response_metadata.get("RequestId") if isinstance(response_metadata, dict) else None
        )
        inference_config = wire_request.get("inferenceConfig")
        if not isinstance(inference_config, dict):
            raise ValueError("Bedrock Converse request is missing inferenceConfig")
        max_tokens = inference_config.get("maxTokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("Bedrock Converse request is missing inferenceConfig.maxTokens")
        temperature = inference_config.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        ):
            raise ValueError("Bedrock Converse request has an invalid inferenceConfig.temperature")
        response = bedrock_converse_response(raw, self.config.model)
        if not isinstance(provider_request_id, str) or not provider_request_id:
            return response
        receipt = build_chat_provider_receipt(
            provider=self.config.kind.value,
            provider_request_id=provider_request_id,
            response_id=None,
            requested_model=self.config.model,
            response_model=None,
            system_fingerprint=None,
            request_payload=cast("JsonObject", wire_request),
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=max_tokens,
            max_tokens_field="inferenceConfig.maxTokens",
            started_at_unix_s=started_at,
            finished_at_unix_s=finished_at,
        )
        return response.model_copy(update={"provider_receipt": receipt})

    def _complete_reasoning(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Complete through Converse so configured adaptive reasoning cannot be ignored."""
        request = ChatRequest.model_validate(
            {
                "messages": [
                    *([{"role": "system", "content": system}] if system else []),
                    *[{"role": message.role, "content": message.content} for message in messages],
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.complete_chat(request)
        choice = response.choices[0]
        if response.usage is None:
            return Completion(text=str(choice.message.content or ""))
        return Completion(
            text=str(choice.message.content or ""),
            usage=_token_usage(
                response.usage.model_dump(mode="json"),
                input_field="prompt_tokens",
                output_field="completion_tokens",
            ),
        )

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
        usage = _token_usage(
            response["usage"],
            input_field="inputTokens",
            output_field="outputTokens",
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
        usage = _token_usage(
            data["usage"],
            input_field="inputTokens",
            output_field="outputTokens",
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


def _token_usage(
    value: object,
    *,
    input_field: str,
    output_field: str,
) -> TokenUsage:
    """Preserve every provider usage dimension for fail-closed downstream pricing."""
    if not isinstance(value, dict):
        raise ValueError("Bedrock usage must be an object")
    usage = cast("dict[str, object]", value)
    input_tokens = usage.get(input_field)
    output_tokens = usage.get(output_field)
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        raise ValueError("Bedrock usage counters must be non-negative integers")
    extras = {
        _USAGE_FIELD_NAMES.get(name, name): item
        for name, item in usage.items()
        if name not in {input_field, output_field}
    }
    return TokenUsage.model_validate(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **extras,
        }
    )

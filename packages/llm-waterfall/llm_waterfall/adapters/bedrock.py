"""AWS Bedrock adapter (Anthropic Messages schema via InvokeModel, Titan embeddings).

Credentials come from the boto3 chain, or a named AWS profile when `Backend.profile` is set —
`boto3.Session(profile_name=...)` — so one waterfall chain can span multiple AWS accounts.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, TypedDict, cast

from pydantic import JsonValue

from llm_waterfall.adapters.base import missing_sdk_error
from llm_waterfall.types import (
    Backend,
    ChatRequest,
    ChatResponse,
    Message,
    TokenUsage,
)

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


class _ConverseContentBlock(TypedDict, total=False):
    text: str
    toolUse: dict[str, object]


class _ConverseMessage(TypedDict):
    role: str
    content: list[_ConverseContentBlock]


class _ConverseOutput(TypedDict):
    message: _ConverseMessage


class _ConverseUsage(TypedDict):
    inputTokens: int
    outputTokens: int


class _ConverseResponse(TypedDict):
    output: _ConverseOutput
    stopReason: str
    usage: _ConverseUsage


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

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a structured tool-calling request through Bedrock Converse."""
        converse_request = _converse_request(request, self.backend.model)
        response = cast("_ConverseResponse", self._get_client().converse(**converse_request))
        blocks = response["output"]["message"]["content"]
        text = "".join(block["text"] for block in blocks if "text" in block)
        tool_calls: list[dict[str, object]] = []
        for block in blocks:
            use = block.get("toolUse")
            if use is None:
                continue
            tool_calls.append(
                {
                    "id": str(use.get("toolUseId", "")),
                    "type": "function",
                    "function": {
                        "name": str(use.get("name", "")),
                        "arguments": json.dumps(use.get("input", {})),
                    },
                }
            )
        stop_reason = response["stopReason"]
        finish_reason = {
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "content_filtered": "content_filter",
            "guardrail_intervened": "content_filter",
        }.get(stop_reason, "stop")
        message: dict[str, object] = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ChatResponse.model_validate(
            {
                "model": self.backend.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": response["usage"]["inputTokens"],
                    "completion_tokens": response["usage"]["outputTokens"],
                },
            }
        )

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


def _converse_request(request: ChatRequest, model: str) -> dict[str, object]:
    """Translate the structured OpenAI-compatible contract to Bedrock Converse."""
    system: list[dict[str, str]] = []
    messages: list[dict[str, object]] = []

    def push(role: str, content: list[dict[str, object]]) -> None:
        if messages and messages[-1]["role"] == role:
            existing = cast("list[dict[str, object]]", messages[-1]["content"])
            existing.extend(content)
        else:
            messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role in ("system", "developer"):
            text = _chat_text(message.content)
            if text:
                system.append({"text": text})
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": _chat_text(message.content)}],
                        }
                    }
                ],
            )
            continue
        blocks: list[dict[str, object]] = []
        text = _chat_text(message.content)
        if text:
            blocks.append({"text": text})
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except ValueError:
                arguments = {}
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.function.name,
                        "input": arguments,
                    }
                }
            )
        if blocks:
            push("assistant" if message.role == "assistant" else "user", blocks)

    max_tokens = request.max_tokens or request.max_completion_tokens or 4096
    inference: dict[str, float | int] = {"maxTokens": max_tokens}
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    result: dict[str, object] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": inference,
    }
    if system:
        result["system"] = system
    if request.tools:
        tools = [
            {
                "toolSpec": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "inputSchema": {"json": tool.function.parameters},
                }
            }
            for tool in request.tools
        ]
        tool_config: dict[str, object] = {"tools": tools}
        choice = request.tool_choice
        if choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict):
            function = choice.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_config["toolChoice"] = {"tool": {"name": function["name"]}}
        if choice != "none":
            result["toolConfig"] = tool_config
    return result


def _chat_text(content: JsonValue) -> str:
    """Flatten the text-bearing forms used by OpenAI-compatible chat messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)

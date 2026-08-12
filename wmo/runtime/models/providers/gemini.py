"""Native Gemini conversion for non-streaming generation and embeddings."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.errors import ProviderResponseError
from wmo.runtime.models.providers.openai_compatible import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    normalize_embedding_vector,
)
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 4096


def gemini_generate_request(model_id: str, request: ModelRequest) -> JsonObject:
    """Convert a WMO request into Gemini's native generateContent payload.

    Args:
        model_id: Gemini model identifier selected by the catalog.
        request: Typed visible messages, tools, and sampling parameters.

    Returns:
        A non-streaming native payload for the generateContent endpoint.

    Raises:
        ValueError: A visible request message cannot preserve its tool linkage on Gemini's wire.
    """
    system_parts: list[JsonObject] = []
    contents: list[JsonObject] = []
    tool_names: dict[str, str] = {}
    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system_parts.append({"text": message.content})
            continue
        contents.append(_gemini_content(message, tool_names))
    payload: JsonObject = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if request.tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parametersJsonSchema": tool.input_schema,
                    }
                    for tool in request.tools
                ]
            }
        ]
    if request.tool_choice is not None:
        payload["toolConfig"] = {"functionCallingConfig": _gemini_tool_choice(request.tool_choice)}
    generation: JsonObject = {}
    if request.temperature is not None:
        generation["temperature"] = request.temperature
    if request.maximum_output_tokens is not None:
        generation["maxOutputTokens"] = request.maximum_output_tokens
    else:
        generation["maxOutputTokens"] = DEFAULT_MAXIMUM_OUTPUT_TOKENS
    payload["generationConfig"] = generation
    return payload


def gemini_generate_response(
    payload: JsonObject,
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Convert native Gemini candidate parts into WMO text and tool calls.

    Args:
        payload: Decoded completed Gemini response.
        configured_model: Resolved catalog identity used for the request.
        latency_seconds: Observed duration of the successful request sequence.

    Returns:
        The typed assistant action, served model identity, and observed economics.

    Raises:
        ProviderResponseError: The response omits a usable candidate or has malformed content.
    """
    candidates = _array(payload.get("candidates"), "candidates")
    if not candidates:
        raise ProviderResponseError("Gemini response has no candidates")
    candidate = _object(candidates[0], "candidates[0]")
    content = _object(candidate.get("content"), "candidates[0].content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, part_value in enumerate(_array(content.get("parts"), "candidates[0].content.parts")):
        part = _object(part_value, f"candidates[0].content.parts[{index}]")
        text = part.get("text")
        function_call = part.get("functionCall")
        if isinstance(text, str):
            text_parts.append(text)
        elif function_call is not None:
            tool_calls.append(_gemini_tool_call(function_call, index))
        else:
            raise ProviderResponseError(f"Gemini content part {index} is unsupported")
    output_text = "".join(text_parts) if text_parts else None
    try:
        output = AssistantAction(content=output_text, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderResponseError("Gemini response has no text or tool call") from exc
    model_version = payload.get("modelVersion")
    model = (
        configured_model.model_copy(update={"model_id": model_version})
        if isinstance(model_version, str) and model_version
        else configured_model
    )
    return ModelResponse(
        output=output,
        model=model,
        economics=OperationEconomics(
            usage=_gemini_usage(payload),
            latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
        ),
    )


class GeminiClient:
    """Calls one explicit Gemini model through its native REST protocol."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = GEMINI_BASE_URL,
    ) -> None:
        """Create a Gemini client with an API key sent only to Gemini's endpoint."""
        if not api_key:
            raise ValueError("Gemini clients require a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one native Gemini generateContent request.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        response = post_json(
            self._transport,
            f"{self._base_url}/models/{_path_model_id(self._model.model_id)}:generateContent",
            headers=self._headers(),
            payload=gemini_generate_request(self._model.model_id, request),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
        )
        return gemini_generate_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Batch-embed texts through Gemini and normalize every returned vector.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order.

        Raises:
            ProviderResponseError: Gemini returns a missing, malformed, or mismatched vector.
        """
        if not texts:
            return ()
        model_name = f"models/{_path_model_id(self._model.model_id)}"
        response = post_json(
            self._transport,
            f"{self._base_url}/models/{_path_model_id(self._model.model_id)}:batchEmbedContents",
            headers=self._headers(),
            payload={
                "requests": [
                    {"model": model_name, "content": {"parts": [{"text": text}]}} for text in texts
                ]
            },
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
        )
        values = _array(response.get("embeddings"), "embeddings")
        if len(values) != len(texts):
            raise ProviderResponseError(
                f"Gemini embedding count {len(values)} does not match request count {len(texts)}"
            )
        return tuple(
            Embedding(
                values=normalize_embedding_vector(
                    _array(_object(value, f"embeddings[{index}]").get("values"), "values")
                )
            )
            for index, value in enumerate(values)
        )

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key, "content-type": "application/json"}


def _gemini_content(message: ModelMessage, tool_names: dict[str, str]) -> JsonObject:
    """Map a WMO history item to Gemini user or model content parts."""
    if message.role == "tool":
        tool_name = tool_names.get(message.tool_call_id or "")
        if tool_name is None:
            raise ValueError("Gemini tool result has no preceding tool-call name")
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"content": message.content or ""},
                    }
                }
            ],
        }
    if message.role == "user":
        if message.content is None:
            raise ValueError("user messages need text content")
        return {"role": "user", "parts": [{"text": message.content}]}
    if message.role != "assistant":
        raise ValueError(f"unsupported Gemini message role {message.role!r}")
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    parts: list[JsonObject] = []
    if text is not None:
        parts.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            tool_names[call.call_id] = call.name
            parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
    if not parts:
        raise ValueError("assistant messages need text or tool calls")
    return {"role": "model", "parts": parts}


def _gemini_tool_choice(
    choice: Literal["auto", "none", "required"] | ToolChoice,
) -> JsonObject:
    """Map closed WMO tool-choice values to native Gemini function configuration."""
    if choice == "auto":
        return {"mode": "AUTO"}
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    if isinstance(choice, ToolChoice):
        return {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    raise ValueError("unsupported Gemini tool choice")


def _gemini_tool_call(value: JsonValue, index: int) -> ToolCall:
    """Map one native Gemini function call with a deterministic local call ID fallback."""
    call = _object(value, f"functionCall[{index}]")
    name = call.get("name")
    arguments = call.get("args", {})
    if not isinstance(name, str) or not name:
        raise ProviderResponseError(f"Gemini functionCall[{index}].name must be text")
    if not isinstance(arguments, dict):
        raise ProviderResponseError(f"Gemini functionCall[{index}].args must be an object")
    call_id = call.get("id")
    return ToolCall(
        call_id=call_id if isinstance(call_id, str) and call_id else f"gemini-call-{index}",
        name=name,
        arguments=arguments,
    )


def _gemini_usage(payload: JsonObject) -> Usage | None:
    """Read Gemini's usage metadata with cached tokens treated as an input subset."""
    raw = payload.get("usageMetadata")
    if raw is None:
        return None
    usage = _object(raw, "usageMetadata")
    return Usage(
        input_tokens=_integer(usage.get("promptTokenCount"), "promptTokenCount"),
        output_tokens=_integer(usage.get("candidatesTokenCount"), "candidatesTokenCount"),
        cached_input_tokens=_integer(
            usage.get("cachedContentTokenCount"), "cachedContentTokenCount"
        ),
    )


def _path_model_id(model_id: str) -> str:
    """Remove the optional wire prefix before placing a model in a Gemini path."""
    return model_id.removeprefix("models/")


def _array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a native array or raise a focused conversion error."""
    if not isinstance(value, list):
        raise ProviderResponseError(f"Gemini {label} must be an array")
    return value


def _object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a native object or raise a focused conversion error."""
    if not isinstance(value, dict):
        raise ProviderResponseError(f"Gemini {label} must be an object")
    return value


def _integer(value: JsonValue | None, label: str) -> int:
    """Read an optional non-negative native usage integer."""
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"Gemini {label} must be a non-negative integer")
    return value

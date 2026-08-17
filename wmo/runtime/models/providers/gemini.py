"""Native Gemini conversion for non-streaming generation and embeddings."""

from __future__ import annotations

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
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.base import (
    DEFAULT_MAXIMUM_OUTPUT_TOKENS,
    ProviderHttpClient,
)
from wmo.runtime.models.providers.errors import (
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from wmo.runtime.models.providers.openai_compatible import normalize_embedding_vector

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


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
    candidates = require_array(payload.get("candidates"), "Gemini candidates")
    if not candidates:
        raise ProviderResponseError("Gemini response has no candidates")
    candidate = require_object(candidates[0], "Gemini candidates[0]")
    content = require_object(candidate.get("content"), "Gemini candidates[0].content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parts = require_array(content.get("parts"), "Gemini candidates[0].content.parts")
    for index, part_value in enumerate(parts):
        part = require_object(part_value, f"Gemini candidates[0].content.parts[{index}]")
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
    return ModelResponse.completed(
        output=output,
        configured_model=configured_model,
        served_model_id=payload.get("modelVersion"),
        usage=_gemini_usage(payload),
        latency_seconds=latency_seconds,
        hit_length_limit=candidate.get("finishReason") == "MAX_TOKENS",
    )


class GeminiClient(ProviderHttpClient):
    """Calls one explicit Gemini model through its native REST protocol."""

    def _headers(self) -> dict[str, str]:
        """Build native Gemini headers using the goog API key scheme."""
        return {"x-goog-api-key": self._api_key, "content-type": "application/json"}

    def _completion_path(self) -> str:
        """Return the model-scoped native generateContent route."""
        return f"models/{_path_model_id(self._model.model_id)}:generateContent"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a native generateContent payload."""
        return gemini_generate_request(self._model.model_id, request)

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one completed generateContent payload into the shared response contract."""
        return gemini_generate_response(
            payload, configured_model=self._model, latency_seconds=latency_seconds
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
        response = self._post(
            f"models/{_path_model_id(self._model.model_id)}:batchEmbedContents",
            {
                "requests": [
                    {"model": model_name, "content": {"parts": [{"text": text}]}} for text in texts
                ]
            },
        )
        values = require_array(response.get("embeddings"), "Gemini embeddings")
        if len(values) != len(texts):
            raise ProviderResponseError(
                f"Gemini embedding count {len(values)} does not match request count {len(texts)}"
            )
        return tuple(
            Embedding(
                values=normalize_embedding_vector(
                    require_array(
                        require_object(value, f"Gemini embeddings[{index}]").get("values"),
                        "Gemini values",
                    )
                )
            )
            for index, value in enumerate(values)
        )


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
    call = require_object(value, f"Gemini functionCall[{index}]")
    name = require_string(call.get("name"), f"Gemini functionCall[{index}].name")
    arguments = call.get("args", {})
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
    usage = require_object(raw, "Gemini usageMetadata")
    return Usage(
        input_tokens=require_integer(usage.get("promptTokenCount"), "Gemini promptTokenCount"),
        output_tokens=require_integer(
            usage.get("candidatesTokenCount"), "Gemini candidatesTokenCount"
        ),
        cached_input_tokens=require_integer(
            usage.get("cachedContentTokenCount"), "Gemini cachedContentTokenCount"
        ),
    )


def _path_model_id(model_id: str) -> str:
    """Remove the optional wire prefix before placing a model in a Gemini path."""
    return model_id.removeprefix("models/")

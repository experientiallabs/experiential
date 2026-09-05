"""Native Gemini client for generation, streaming dispatch, and embeddings."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    Embedding,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    Usage,
)
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayWireProfile,
    ProviderHttpClient,
)
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from exp.runtime.models.providers.gemini_requests import (
    gemini_generate_request,
    gemini_model_path,
)
from exp.runtime.models.providers.openai_compatible import normalize_embedding_vector
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def gemini_generate_response(
    payload: JsonObject,
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Convert native Gemini candidate parts into EXP text and tool calls.

    Args:
        payload: Decoded completed Gemini response.
        configured_model: Resolved catalog identity used for the request.
        latency_seconds: Observed duration of the successful request sequence.

    Returns:
        The typed assistant action, served model identity, and observed economics.

    Raises:
        ProviderRefusalError: Google blocked the prompt itself (``promptFeedback.blockReason``)
            or stopped the candidate on a safety, copyright, or sensitive-information reason.
        ProviderResponseError: The response omits a usable candidate or has malformed content.
    """
    prompt_block = _gemini_prompt_block_signal(payload)
    if prompt_block is not None:
        raise ProviderRefusalError(provider="gemini", signal=prompt_block)
    candidates = require_array(payload.get("candidates"), "Gemini candidates")
    if not candidates:
        raise ProviderResponseError("Gemini response has no candidates")
    candidate = require_object(candidates[0], "Gemini candidates[0]")
    refusal_signal = _gemini_refusal_signal(candidate.get("finishReason"))
    if refusal_signal is not None:
        raise ProviderRefusalError(provider="gemini", signal=refusal_signal)
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

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str = GEMINI_BASE_URL,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool = True,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_frequency_penalty: bool = False,
        supports_presence_penalty: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create a Gemini client with explicit generation capability gates."""
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
        )
        self._supports_temperature = supports_temperature
        self._supports_top_p = supports_top_p
        self._supports_top_k = supports_top_k
        self._supports_logprobs = supports_logprobs
        self._supports_frequency_penalty = supports_frequency_penalty
        self._supports_presence_penalty = supports_presence_penalty
        self._supports_reasoning = supports_reasoning
        self._reasoning_effort = reasoning_effort

    def _headers(self) -> dict[str, str]:
        """Build native Gemini headers using the goog API key scheme."""
        return {"x-goog-api-key": self._api_key, "content-type": "application/json"}

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the native streamGenerateContent wire profile for this connection."""
        model_id = gemini_model_path(self._model.model_id)
        return GatewayWireProfile(
            dialect="gemini_generate_content",
            url=f"{self._base_url}/models/{model_id}:streamGenerateContent?alt=sse",
            headers=self._headers(),
            model_id=self._model.model_id,
            timeout_seconds=self._timeout_seconds,
            supports_temperature=self._supports_temperature,
            maximum_temperature=2.0,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_frequency_penalty=self._supports_frequency_penalty,
            supports_presence_penalty=self._supports_presence_penalty,
            supports_reasoning=self._supports_reasoning,
            reasoning_wire_format="gemini_thinking",
            reasoning_effort=self._reasoning_effort,
        )

    def _completion_path(self) -> str:
        """Return the model-scoped native generateContent route."""
        return f"models/{gemini_model_path(self._model.model_id)}:generateContent"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a native generateContent payload."""
        return gemini_generate_request(
            self._model.model_id,
            request,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_effort=self._reasoning_effort,
        )

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
        model_name = f"models/{gemini_model_path(self._model.model_id)}"
        response = self._post(
            f"models/{gemini_model_path(self._model.model_id)}:batchEmbedContents",
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


# Finish and block reasons Google reports for content its safety systems
# refused; the prompt-level block adds the image-specific reason.
_GEMINI_SAFETY_REASONS: frozenset[str] = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"})
_GEMINI_PROMPT_SAFETY_REASONS: frozenset[str] = _GEMINI_SAFETY_REASONS | {"IMAGE_SAFETY"}


def _gemini_prompt_block_signal(payload: JsonObject) -> ProviderRefusalSignal | None:
    """Map a prompt-level block to a content-free refusal category.

    A blocked prompt arrives as ``promptFeedback.blockReason`` with no
    candidates at all, so it must be read before the candidate contract is
    enforced. ``BLOCK_REASON_UNSPECIFIED`` is the enum default and, like
    ratings-only feedback, means the prompt was not blocked.

    Args:
        payload: Decoded Gemini response object.

    Returns:
        The refusal signal for a blocked prompt, or ``None`` when generation ran.

    Raises:
        ProviderResponseError: ``promptFeedback`` is not an object, or its
            ``blockReason`` is not text.
    """
    raw = payload.get("promptFeedback")
    if raw is None:
        return None
    feedback = require_object(raw, "Gemini promptFeedback")
    raw_reason = feedback.get("blockReason")
    if raw_reason is None:
        return None
    reason = require_string(raw_reason, "Gemini promptFeedback.blockReason")
    if reason == "BLOCK_REASON_UNSPECIFIED":
        return None
    if reason in _GEMINI_PROMPT_SAFETY_REASONS:
        return ProviderRefusalSignal.SAFETY
    return ProviderRefusalSignal.PROVIDER_REFUSAL


def _gemini_refusal_signal(value: object) -> ProviderRefusalSignal | None:
    """Map a Gemini finish reason to a content-free refusal category.

    Args:
        value: Provider-reported candidate finish reason.

    Returns:
        A normalized refusal signal, or ``None`` for ordinary terminal reasons.
    """
    if isinstance(value, str) and value in _GEMINI_SAFETY_REASONS:
        return ProviderRefusalSignal.SAFETY
    if value == "RECITATION":
        return ProviderRefusalSignal.COPYRIGHT
    if value == "SPII":
        return ProviderRefusalSignal.SENSITIVE_INFORMATION
    return None

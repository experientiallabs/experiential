"""Native Gemini generateContent payload construction shared by both engines.

The non-streaming client (`gemini.py`) and the gateway dialect builders
(`streaming_requests.py`) build the identical native payload from this one
module, so the two callers cannot drift at the Gemini wire boundary. This
module stays free of streaming imports on purpose: the shared dialect
builders import it without creating a cycle through the streaming stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest, ToolChoice
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.models.providers.audios import gemini_audio_part
from exp.runtime.models.providers.base import DEFAULT_MAXIMUM_OUTPUT_TOKENS, GatewayWireProfile
from exp.runtime.models.providers.documents import gemini_document_part
from exp.runtime.models.providers.images import gemini_image_part
from exp.runtime.models.providers.instruction_turns import (
    fold_instruction_turns_after_the_leading_run,
)
from exp.runtime.models.providers.reasoning_compat import (
    gemini_thinking_budget,
    gemini_thinking_level,
    gemini_thinks_by_default,
    reasoning_display_enabled,
)
from exp.runtime.models.providers.videos import gemini_video_part

GEMINI_THOUGHT_SIGNATURE_BYPASS = "skip_thought_signature_validator"
"""Gemini's documented placeholder for a function call whose signature is gone.

Gemini 3 answers a function call with an opaque ``thoughtSignature`` and
rejects the next turn with HTTP 400 unless that part comes back carrying it.
A replayed call carries its real signature when the caller echoed one
(``ToolCall.provider_thought_signature``); a turn whose calls carry none
(history from another provider, or a client that dropped the field) uses this
literal, which Google documents for replaying a call whose signature was not
retained.
"""


THOUGHT_SIGNATURE_DROPPED = "messages.tool_calls.thought_signature->dropped(no_gemini_rung)"
"""Disclosure recorded when echoed Chat thought signatures reach no rung that reads them.

Only the Gemini generateContent wire (Gemini and Vertex) replays a signature;
every other wire omits it, so a Chat caller that echoed one, in either
spelling, learns at admission that no rung on the route will receive it.
"""


def disclose_dropped_thought_signatures(
    profiles: Iterable[GatewayWireProfile],
    request: GatewayRequest,
    ignored: list[str],
) -> None:
    """Record :data:`THOUGHT_SIGNATURE_DROPPED` when no rung can replay a signature.

    Like block-level cache markers, the signature is kept while ANY rung reads
    it (the Gemini rung replays it, the others omit it) and disclosed only
    when none does. Nothing is recorded for a Chat request without one.
    """
    if (
        request.surface == GatewayApiSurface.CHAT_COMPLETIONS
        and any(
            call.provider_thought_signature is not None
            for message in request.messages
            for call in message.tool_calls
        )
        and not any(profile.dialect == "gemini_generate_content" for profile in profiles)
        and THOUGHT_SIGNATURE_DROPPED not in ignored
    ):
        ignored.append(THOUGHT_SIGNATURE_DROPPED)


def gemini_model_path(model_id: str) -> str:
    """Remove the optional wire prefix before placing a model in a Gemini path.

    Args:
        model_id: Catalog model identifier, optionally ``models/``-prefixed.

    Returns:
        The bare model identifier used in Gemini route paths.
    """
    return model_id.removeprefix("models/")


def gemini_generate_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
    stop_sequences: tuple[str, ...] = (),
    response_json_schema: JsonObject | None = None,
    json_object_output: bool = False,
    default_maximum_output_tokens: int | None = DEFAULT_MAXIMUM_OUTPUT_TOKENS,
) -> JsonObject:
    """Convert a EXP request into Gemini's native generateContent payload.

    Args:
        model_id: Gemini model identifier selected by the catalog. The model
            travels in the route path, never the body; the parameter keeps the
            shared provider builder signature.
        request: Typed visible messages, tools, and sampling parameters.
        supports_temperature: Whether this exact deployment accepts temperature.
        supports_top_p: Whether this exact deployment accepts top-p sampling.
        supports_top_k: Whether this exact deployment accepts top-k sampling.
        supports_logprobs: Reserved response-projection capability flag.
        supports_reasoning: Whether this exact deployment accepts thinking controls.
        reasoning_effort: Catalog-pinned reasoning effort used when the request omits one.
        stop_sequences: Exact stop strings admitted for the selected route.
        response_json_schema: Strict JSON schema admitted for structured output.
        json_object_output: Whether to request schema-free JSON output
            (``responseMimeType`` only, no ``responseJsonSchema``).
        default_maximum_output_tokens: Model-client default when the request
            omits a ceiling. Gateway callers pass ``None`` to preserve omission.

    Returns:
        A native payload for the generateContent and streamGenerateContent
        endpoints, which share one request shape.

    Raises:
        ValueError: A visible request message cannot preserve its tool linkage on Gemini's wire.
    """
    json_object_output = json_object_output or request.json_object_output
    system_parts: list[JsonObject] = []
    contents: list[JsonObject] = []
    tool_names: dict[str, str] = {}
    # systemInstruction has no position inside contents, so an instruction
    # after conversation start rides as user text where the caller put it
    # (consecutive user contents are accepted on this wire); only the leading
    # run is hoisted.
    turn = _CallTurn()
    for message in fold_instruction_turns_after_the_leading_run(request.messages):
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ValueError("system messages need text content")
            system_parts.append({"text": message.content})
            continue
        if message.role == "assistant":
            turn.close(contents)
        content = _gemini_content(message, tool_names)
        if message.role == "assistant":
            turn.open(message, contents, content)
        elif turn.collects(message):
            turn.add(message, content)
        else:
            contents.append(content)
    turn.close(contents)
    if json_object_output:
        system_parts.append({"text": JSON_OBJECT_SYSTEM_INSTRUCTION})
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
    if request.temperature is not None and supports_temperature:
        generation["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        generation["topP"] = request.top_p
    if request.top_k is not None and supports_top_k:
        generation["topK"] = request.top_k
    if stop_sequences:
        generation["stopSequences"] = list(stop_sequences)
    if response_json_schema is not None:
        generation["responseMimeType"] = "application/json"
        generation["responseJsonSchema"] = response_json_schema
    elif json_object_output:
        generation["responseMimeType"] = "application/json"
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    if supports_reasoning:
        thinking_config: JsonObject = {}
        budget: int | None = None
        if effective_reasoning_effort is not None:
            level = gemini_thinking_level(model_id, effective_reasoning_effort)
            budget = gemini_thinking_budget(model_id, level)
            # Google rejects a request that sets both fields.
            if budget is not None:
                thinking_config["thinkingBudget"] = budget
            else:
                thinking_config["thinkingLevel"] = level.upper()
        thinking = budget != 0 and (budget is not None or gemini_thinks_by_default(model_id))
        # Thought summaries are the readable reasoning Google returns; without
        # includeThoughts a thinking model streams only opaque signatures.
        # Vertex refuses includeThoughts on a request whose thinking is off.
        if thinking and reasoning_display_enabled():
            thinking_config["includeThoughts"] = True
        if thinking_config:
            generation["thinkingConfig"] = thinking_config
    # The normalized gateway response has no logprob representation. Keep the
    # route flag for shared capability plumbing, but ignore these controls so
    # provider output is never requested and then silently discarded.
    del supports_logprobs
    if request.maximum_output_tokens is not None:
        generation["maxOutputTokens"] = request.maximum_output_tokens
    elif default_maximum_output_tokens is not None:
        generation["maxOutputTokens"] = default_maximum_output_tokens
    payload["generationConfig"] = generation
    return payload


GEMINI_MISSING_TOOL_RESULT = (
    "[no result: the caller's history carries no tool result for this call]"
)
"""Function response sent for a parallel call the caller's history never answered.

Gemini refuses a turn unless the content after a model turn answers every one
of its function calls ("the number of function response parts is equal to
the number of function call parts"). A history whose call went unanswered (an
interrupted client) is served with this placeholder in the call's slot; the
gateway discloses it as ``GEMINI_MISSING_TOOL_RESULT_DISCLOSURE``.
"""


class _CallTurn:
    """Regroup the answers to one model turn's function calls into one user content.

    Gemini checks that the content right after a model turn carries exactly
    one ``functionResponse`` part per ``functionCall`` part. Chat and
    Responses histories carry one tool message per call, sometimes split by
    user text (folded tool-result images, client reminders), so every result
    answering the open turn is collected and emitted as ONE user content in
    call order, ahead of the user contents that arrived with them. A call with
    no result gets ``GEMINI_MISSING_TOOL_RESULT`` once a later turn exists; a
    trailing model turn awaiting results is left as it is.
    """

    def __init__(self) -> None:
        self._calls: tuple[tuple[str, str], ...] = ()
        self._responses: dict[str, JsonObject] = {}
        self._others: list[JsonObject] = []

    def open(self, message: ModelMessage, contents: list[JsonObject], content: JsonObject) -> None:
        """Append a model turn and start collecting answers to its calls."""
        contents.append(content)
        action = message.assistant_action
        calls = () if action is None else action.tool_calls
        self._calls = tuple((call.call_id, call.name) for call in calls)

    def collects(self, message: ModelMessage) -> bool:
        """Whether a user or tool message belongs to the open call turn."""
        return bool(self._calls) and message.role in {"user", "tool"}

    def add(self, message: ModelMessage, content: JsonObject) -> None:
        """Hold one message of the open call turn until the turn closes."""
        call_id = message.tool_call_id or ""
        if (
            message.role == "tool"
            and call_id not in self._responses
            and any(call_id == known for known, _name in self._calls)
        ):
            self._responses[call_id] = content
            return
        self._others.append(content)

    def close(self, contents: list[JsonObject]) -> None:
        """Emit the open turn's answers as one user content, then the rest in order."""
        if not self._calls:
            return
        if self._responses or self._others:
            parts: list[JsonObject] = []
            for call_id, name in self._calls:
                response = self._responses.get(call_id)
                if response is None:
                    parts.append(
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"content": GEMINI_MISSING_TOOL_RESULT},
                            }
                        }
                    )
                    continue
                response_parts = response["parts"]
                assert isinstance(response_parts, list)
                parts.extend(cast("list[JsonObject]", response_parts))
            contents.append({"role": "user", "parts": parts})
            contents.extend(self._others)
        self._calls = ()
        self._responses = {}
        self._others = []


def _gemini_content(message: ModelMessage, tool_names: dict[str, str]) -> JsonObject:
    """Map a EXP history item to Gemini user or model content parts.

    Args:
        message: One visible non-system history message.
        tool_names: Accumulated tool-call ID to tool-name linkage, updated in place.

    Returns:
        One native Gemini content object.

    Raises:
        ValueError: The message cannot preserve its role or tool linkage.
    """
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
        if message.content_parts:
            return {
                "role": "user",
                "parts": [
                    {"text": part.text}
                    if part.kind == "text"
                    else gemini_image_part(part)
                    if part.kind == "image"
                    else gemini_video_part(part)
                    if part.kind == "video"
                    else gemini_audio_part(part)
                    if part.kind == "audio"
                    else gemini_document_part(part)
                    for part in message.content_parts
                ],
            }
        if message.content is None:
            raise ValueError("user messages need text content")
        return {"role": "user", "parts": [{"text": message.content}]}
    if message.role != "assistant":
        raise ValueError(f"unsupported Gemini message role {message.role!r}")
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    parts: list[JsonObject] = []
    if message.content_parts:
        for part in message.content_parts:
            if part.kind == "text":
                parts.append({"text": part.text})
            elif part.kind == "image":
                parts.append(gemini_image_part(part))
            else:
                raise ValueError("Gemini assistant history supports text and image parts")
        # Chat history carries caller-owned content rather than Gemini's opaque
        # signatures, as with replayed function calls below.
        for part in parts:
            part["thoughtSignature"] = GEMINI_THOUGHT_SIGNATURE_BYPASS
    elif text is not None:
        parts.append({"text": text})
    if action is not None:
        # Gemini signs only the first call of a parallel batch, so a turn that
        # carries any real signature replays exactly as issued (unsigned calls
        # stay unsigned); a turn with none falls back to the placeholder.
        signed_turn = any(call.provider_thought_signature for call in action.tool_calls)
        for call in action.tool_calls:
            tool_names[call.call_id] = call.name
            part: JsonObject = {"functionCall": {"name": call.name, "args": call.arguments}}
            if call.provider_thought_signature is not None:
                part["thoughtSignature"] = call.provider_thought_signature
            elif not signed_turn:
                part["thoughtSignature"] = GEMINI_THOUGHT_SIGNATURE_BYPASS
            parts.append(part)
    if not parts:
        raise ValueError("assistant messages need text or tool calls")
    return {"role": "model", "parts": parts}


def _gemini_tool_choice(
    choice: Literal["auto", "none", "required"] | ToolChoice,
) -> JsonObject:
    """Map closed EXP tool-choice values to native Gemini function configuration.

    Args:
        choice: Closed tool-choice literal or one named tool restriction.

    Returns:
        The native functionCallingConfig object.

    Raises:
        ValueError: The choice value is outside the closed contract.
    """
    if choice == "auto":
        return {"mode": "AUTO"}
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    if isinstance(choice, ToolChoice):
        return {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    raise ValueError("unsupported Gemini tool choice")

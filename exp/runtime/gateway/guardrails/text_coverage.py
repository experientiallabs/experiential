"""Truthful input coverage for the local text-only guardrail adapters."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import ClassifierCoverageError
from exp.runtime.gateway.reasoning_blocks import ExposedReasoningContentBlock


def text_input_context(request: GatewayRequest) -> str:
    """Validate text coverage and extract supported context that must not be rewritten.

    Message text, including the canonical folded tool-error prefix, and tool arguments
    are scanned separately by each adapter. Function
    definitions, examples, call/result identifiers, names and namespaces, the closed Messages
    ``tool_use`` caller, schemas, normalized search declarations, and caller-replayed
    plaintext reasoning (Chat ``reasoning_content`` on assistant history) are ordinary text
    too; reasoning is scanned but never rewritten. Gateway-recovered, sealed, encrypted or
    provider-signed reasoning, opaque provider content, Responses caller objects,
    attachments, and provider-owned tools or prompt configuration require a separately
    validated projection. Internal replay IDs, cache hints, and inert telemetry are not
    prompt content.

    Returns:
        Canonical text for the supported non-message context, or an empty string.

    Raises:
        ClassifierCoverageError: The local text scanner cannot inspect the whole input.
    """
    if (
        request.context_management is not None
        or request.safeguards
        or request.provider_output_config is not None
        or request.provider_thinking_config is not None
        or request.provider_native_tools
        or request.provider_server_tools
        or any(
            message.content_parts
            or any(
                not isinstance(block, ExposedReasoningContentBlock)
                for block in message.provider_reasoning
            )
            or message.capture_only_reasoning
            or message.provider_native_item is not None
            or message.provider_anthropic_block is not None
            or message.provider_anthropic_blocks is not None
            or message.provider_tool_caller is not None
            or any(call.provider_caller is not None for call in message.tool_calls)
            for message in request.messages
        )
    ):
        raise ClassifierCoverageError
    context: JsonObject = {}
    if request.tools:
        context["tools"] = [
            {
                **tool.model_dump(mode="json"),
                "input_examples": None
                if tool.input_examples is None
                else list(tool.input_examples),
                "allowed_callers": None
                if tool.allowed_callers is None
                else list(tool.allowed_callers),
                "defer_loading": tool.defer_loading,
                "eager_input_streaming": tool.eager_input_streaming,
            }
            for tool in request.tools
        ]
    if request.structured_text is not None:
        context["structured_text"] = request.structured_text.model_dump(mode="json")
    calls = [call for message in request.messages for call in message.tool_calls]
    if calls:
        context["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "namespace": call.provider_namespace,
                # The Messages caller is a closed, decoder-validated shape (a
                # type tag and a call id), so it projects as ordinary text.
                **(
                    {"caller": call.provider_anthropic_caller}
                    if call.provider_anthropic_caller is not None
                    else {}
                ),
            }
            for call in calls
        ]
    results = [
        message
        for message in request.messages
        if message.tool_call_id is not None
        or message.provider_tool_name is not None
        or message.provider_tool_namespace is not None
    ]
    if results:
        context["tool_results"] = [
            {
                "call_id": message.tool_call_id,
                "name": message.provider_tool_name,
                "namespace": message.provider_tool_namespace,
                # Loaded deferred-tool names are caller text the model reads.
                **(
                    {"tool_references": list(message.tool_references)}
                    if message.tool_references
                    else {}
                ),
            }
            for message in results
        ]
    reasoning = [
        block.content
        for message in request.messages
        for block in message.provider_reasoning
        if isinstance(block, ExposedReasoningContentBlock) and block.content
    ]
    if reasoning:
        # Plaintext the caller replays on its own assistant turns is input the
        # provider reads; scan it as immutable context (a match cannot be
        # redacted out of a replayed reasoning field without changing history).
        context["reasoning"] = reasoning
    if request.tool_choice is not None:
        context["tool_choice"] = (
            request.tool_choice
            if isinstance(request.tool_choice, str)
            else request.tool_choice.model_dump(mode="json")
        )
    if request.web_search is not None:
        context["web_search"] = request.web_search.model_dump(mode="json")
    if request.tool_search is not None:
        context["tool_search"] = request.tool_search.model_dump(mode="json")
    return canonical_json_bytes(context).decode() if context else ""

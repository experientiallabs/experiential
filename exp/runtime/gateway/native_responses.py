"""Responses continuation and envelope helpers for the native control plane.

The native (Rust) data plane serves ``/v1/responses`` over one shared set of
protocol state rules: one bounded continuation store keyed by tenant
namespace and episode, the stable public identity derivation, and the
request-reflecting envelope fields the ``ResponsesSseEncoder`` renders. This
module keeps those Responses-only rules in one place; the bridge boundary in
:mod:`exp.runtime.gateway.native_bridge` wraps every raised
:class:`~exp.runtime.openai_protocol.errors.OpenAIProtocolError` into its
sanitized public error before it crosses into the data plane.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    EncryptedReasoningBlock,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ContinuationState,
    ProtocolNamespace,
    episode_namespace,
)
from exp.runtime.openai_protocol.streaming import (
    _responses_tool_choice,  # noqa: PLC2701 - the encoder's envelope rendering is shared.
    stable_public_id,
)


@dataclass
class ContinuationContext:
    """Retention facts for one admitted Responses request."""

    namespace: ProtocolNamespace
    episode_key: str
    response_id: str
    messages: tuple[GatewayMessage, ...]
    retain: bool = True
    """Whether the completed turn may be remembered for later continuation.

    ``store: false`` callers keep episode identity (a continued request still
    joins its original selection episode) but the produced response is never
    retained, so its ID can never be continued from.
    """


def continued_request(
    continuations: BoundedContinuationStore,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
) -> tuple[GatewayRequest, ContinuationContext]:
    """Resolve optional Responses history and derive retention facts.

    Args:
        continuations: The gateway's shared bounded continuation store.
        authorization: Frozen authority for the admitted request.
        request: Canonical Responses request, possibly continuing.

    Returns:
        The execution request with retained history prepended, plus the
        namespaced retention context consumed by :func:`remember_turn`.

    Raises:
        OpenAIProtocolError: The referenced continuation is unavailable,
            expired, evicted, or belongs to another namespace.
    """
    namespace = ProtocolNamespace(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        alias_revision_id=authorization.alias_revision_id,
    )
    retain = request.response_store is not False
    if request.previous_response_id is None:
        episode = episode_namespace(
            namespace=namespace,
            caller_episode_key=request.idempotency_key or request.client_request_id,
            request_id=authorization.request_id,
        )
        return (
            request,
            ContinuationContext(
                namespace=namespace,
                episode_key=episode[-1],
                response_id=stable_public_id("resp", authorization.request_id),
                messages=request.messages,
                retain=retain,
            ),
        )
    continuation = continuations.resolve_now(
        namespace=namespace,
        previous_response_id=request.previous_response_id,
    )
    execution_request = request.model_copy(
        update={"messages": (*continuation.messages, *request.messages)}
    )
    return (
        execution_request,
        ContinuationContext(
            namespace=namespace,
            episode_key=continuation.episode_key,
            response_id=stable_public_id("resp", authorization.request_id),
            messages=execution_request.messages,
            retain=retain,
        ),
    )


def remember_turn(
    continuations: BoundedContinuationStore,
    *,
    context: ContinuationContext,
    data: JsonObject,
) -> None:
    """Retain one completed Responses continuation within strict bounds.

    Retention is strict: refusal output and empty assistant turns are never
    retained, and one oversize continuation fails closed with the shared
    public error before the data plane flushes its terminal frames.

    Args:
        continuations: The gateway's shared bounded continuation store.
        context: Namespaced retention facts captured at admission.
        data: Boundary JSON with aggregated ``text``, ``refusal`` presence,
            and completed ``tool_calls`` (each with ``call_id``, ``name``,
            and raw JSON ``arguments``).

    Raises:
        OpenAIProtocolError: The continuation exceeds the bounded store.
        ValueError: A completed tool call carried malformed fields.
        KeyError: A completed tool call omitted a required field.
    """
    if not context.retain:
        # A store:false caller opted out of server-side continuation state;
        # a later previous_response_id naming this response answers the
        # shared continuation_unavailable error because it was never stored.
        return
    if bool(data.get("refusal")):
        return
    text = str(data.get("text") or "")
    raw_message_item_id = data.get("message_item_id")
    raw_message_output_index = data.get("message_output_index")
    message_item_id: str | None
    message_output_index: int | None
    if raw_message_item_id is None and raw_message_output_index is None:
        message_item_id = None
        message_output_index = None
    elif (
        isinstance(raw_message_item_id, str)
        and bool(raw_message_item_id)
        and isinstance(raw_message_output_index, int)
        and not isinstance(raw_message_output_index, bool)
        and raw_message_output_index >= 0
    ):
        message_item_id = raw_message_item_id
        message_output_index = raw_message_output_index
    else:
        raise ValueError("Responses assistant message identity is invalid")
    raw_calls = data.get("tool_calls")
    tool_calls_list: list[ToolCall] = []
    for call in raw_calls if isinstance(raw_calls, list) else ():
        if not isinstance(call, dict):
            raise ValueError("Responses retained tool call must be an object")
        item_id = call.get("item_id")
        output_index = call.get("output_index")
        if item_id is None and output_index is None:
            provider_item_id = None
            provider_output_index = None
        elif (
            isinstance(item_id, str)
            and bool(item_id)
            and isinstance(output_index, int)
            and not isinstance(output_index, bool)
            and output_index >= 0
        ):
            provider_item_id = item_id
            provider_output_index = output_index
        else:
            raise ValueError("Responses retained tool call identity is invalid")
        raw_arguments = str(call["arguments"])
        tool_calls_list.append(
            ToolCall(
                call_id=str(call["call_id"]),
                name=str(call["name"]),
                arguments=json.loads(raw_arguments),
                raw_arguments=raw_arguments,
                provider_item_id=provider_item_id,
                provider_output_index=provider_output_index,
            )
        )
    tool_calls = tuple(tool_calls_list)
    raw_encrypted = data.get("encrypted_reasoning", [])
    if not isinstance(raw_encrypted, list):
        raise ValueError("Responses encrypted reasoning must be an array")
    encrypted: list[tuple[int, EncryptedReasoningBlock]] = []
    indexes: set[int] = set()
    for item in raw_encrypted:
        if not isinstance(item, dict):
            raise ValueError("Responses encrypted reasoning item must be an object")
        output_index = item.get("output_index")
        item_id = item.get("item_id")
        encrypted_content = item.get("encrypted_content")
        if (
            not isinstance(output_index, int)
            or isinstance(output_index, bool)
            or output_index < 0
            or output_index in indexes
            or not isinstance(item_id, str)
            or not item_id
            or not isinstance(encrypted_content, str)
            or not encrypted_content
        ):
            raise ValueError("Responses encrypted reasoning item is invalid")
        indexes.add(output_index)
        encrypted.append(
            (
                output_index,
                EncryptedReasoningBlock(
                    id=item_id,
                    encrypted_content=encrypted_content,
                    output_index=output_index,
                ),
            )
        )
    encrypted.sort(key=lambda item: item[0])
    provider_reasoning = tuple(block for _index, block in encrypted)
    if provider_reasoning and any(
        call.provider_item_id is None or call.provider_output_index is None for call in tool_calls
    ):
        raise ValueError("Responses retained tool calls require provider item identity and order")
    indexed_output = bool(provider_reasoning) or any(
        call.provider_output_index is not None for call in tool_calls
    )
    if text and indexed_output and message_output_index is None:
        raise ValueError(
            "Responses retained assistant text requires provider item identity and order"
        )
    if not text and not tool_calls and not provider_reasoning:
        return
    message = GatewayMessage(
        role="assistant",
        content=text or None,
        tool_calls=tool_calls,
        provider_reasoning=provider_reasoning,
        provider_item_id=message_item_id,
        provider_output_index=message_output_index,
    )
    continuations.remember_now(
        namespace=context.namespace,
        response_id=context.response_id,
        state=ContinuationState(
            episode_key=context.episode_key,
            messages=(*context.messages, message),
        ),
    )


def responses_envelope(request: GatewayRequest) -> JsonObject:
    """Render the request-reflecting Responses envelope fields for the data plane.

    These values are embedded verbatim in every native Responses envelope, so
    they must match the fields the python ``ResponsesSseEncoder`` derives from
    the same execution request.

    Args:
        request: Canonical execution request with continuation history applied.

    Returns:
        JSON envelope fields keyed exactly as the public response object.
    """
    reasoning: JsonObject = {
        "effort": request.reasoning_effort,
        "summary": request.reasoning_summary,
    }
    if request.reasoning_context is not None:
        # Reflected only when the caller sent it, so context-free response
        # bodies stay byte-identical to the committed goldens.
        reasoning["context"] = request.reasoning_context
    return {
        "metadata": request.metadata or None,
        "parallel_tool_calls": request.parallel_tool_calls is not False,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "reasoning": reasoning,
        "ignored_parameters": list(request.ignored_parameters),
        "tool_choice": _responses_tool_choice(request),
        "tools": [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict,
            }
            for tool in request.tools
        ],
        "max_output_tokens": request.maximum_output_tokens,
        "previous_response_id": request.previous_response_id,
        "include_encrypted_reasoning": request.include_encrypted_reasoning,
    }

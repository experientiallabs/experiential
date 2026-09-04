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
from typing import Literal, cast

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import ToolCall
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    EncryptedReasoningBlock,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.reasoning_carrier import parse_reasoning_content_carrier
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ContinuationRouteBinding,
    ContinuationState,
    ProtocolNamespace,
    episode_namespace,
)
from exp.runtime.openai_protocol.streaming import (
    _responses_tool_choice,  # noqa: PLC2701 - the encoder's envelope rendering is shared.
    stable_public_id,
)

ProviderStatus = Literal["in_progress", "completed", "incomplete"]
ProviderPhase = Literal["commentary", "final_answer"]


def _provider_status(value: object, *, default: ProviderStatus) -> ProviderStatus:
    if value is None:
        return default
    if value not in ("in_progress", "completed", "incomplete"):
        raise ValueError("Responses output item status is invalid")
    return cast(ProviderStatus, value)


def _provider_phase(value: object) -> ProviderPhase | None:
    if value is None:
        return None
    if value not in ("commentary", "final_answer"):
        raise ValueError("Responses assistant message phase is invalid")
    return cast(ProviderPhase, value)


@dataclass
class ContinuationContext:
    """Retention facts for one admitted Responses request."""

    namespace: ProtocolNamespace
    episode_key: str
    response_id: str
    messages: tuple[GatewayMessage, ...]
    required_route_binding: ContinuationRouteBinding | None = None
    route_bindings: tuple[ContinuationRouteBinding, ...] = ()
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
            # The session-scoped correlation id is the stronger affinity
            # scope; a per-operation idempotency key only pins retries.
            caller_episode_key=request.client_request_id or request.idempotency_key,
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
            required_route_binding=continuation.route_binding,
            retain=retain,
        ),
    )


def continuation_route_binding(
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
) -> ContinuationRouteBinding:
    """Bind encrypted reasoning to one deployment and resolved wire authority.

    The retained state stores only digests. Credential-bearing headers and the
    resolved URL never leave the admission stack, while a credential rotation,
    endpoint change, project-header change, or model-wire change invalidates a
    later replay before dispatch.

    Args:
        deployment: Exact certified deployment selected for dispatch.
        profile: Fully resolved authenticated provider wire.

    Returns:
        Secret-free binding retained beside encrypted reasoning.
    """
    return ContinuationRouteBinding(
        deployment_id=deployment.deployment_id,
        connection_sha256=deployment.connection_sha256,
        wire_authority_sha256=sha256_json(
            {
                "version": "responses-continuation-wire-v1",
                "dialect": profile.dialect,
                "url": profile.url,
                "headers": dict(profile.headers),
                "model_id": profile.model_id,
            }
        ),
    )


def remember_turn(
    continuations: BoundedContinuationStore,
    *,
    context: ContinuationContext,
    data: JsonObject,
    route_binding: ContinuationRouteBinding | None = None,
) -> None:
    """Retain one finished Responses continuation within strict bounds.

    Retention is strict about content, not about completion: refusal output is
    never retained, provider-identified message items retain their exact
    lifecycle metadata even when their visible text is empty, and a turn that
    produced no retainable output at all (thinking spent the whole output
    budget, so the response is ``incomplete`` with no items) is retained as the
    conversation so far — the caller holds that response id, and
    ``previous_response_id`` naming it must continue the conversation, as
    api.openai.com does for its own ``incomplete`` responses, instead of
    answering ``previous_response_not_found``. One oversize continuation fails
    closed with the shared public error before the data plane flushes its
    terminal frames.

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
        # shared previous_response_not_found error because it was never stored.
        return
    if bool(data.get("refusal")):
        return
    text = str(data.get("text") or "")
    raw_messages = data.get("message_outputs")
    if raw_messages is None:
        raw_message_item_id = data.get("message_item_id")
        raw_message_output_index = data.get("message_output_index")
        if raw_message_item_id is None and raw_message_output_index is None:
            raw_messages = []
        else:
            raw_messages = [
                {
                    "item_id": raw_message_item_id,
                    "output_index": raw_message_output_index,
                    "text": text,
                    "status": "completed",
                }
            ]
            text = ""
    if not isinstance(raw_messages, list):
        raise ValueError("Responses assistant message outputs must be an array")

    message_outputs: list[tuple[int, GatewayMessage]] = []
    indexes: set[int] = set()
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("Responses assistant message output must be an object")
        item_id = item.get("item_id")
        output_index = item.get("output_index")
        message_text = item.get("text")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(output_index, int)
            or isinstance(output_index, bool)
            or output_index < 0
            or output_index in indexes
            or not isinstance(message_text, str)
        ):
            raise ValueError("Responses assistant message identity is invalid")
        indexes.add(output_index)
        message_outputs.append(
            (
                output_index,
                GatewayMessage(
                    role="assistant",
                    content=message_text or None,
                    provider_item_id=item_id,
                    provider_output_index=output_index,
                    provider_status=_provider_status(item.get("status"), default="completed"),
                    provider_phase=_provider_phase(item.get("phase")),
                ),
            )
        )
    raw_calls = data.get("tool_calls")
    indexed_calls: list[tuple[int, ToolCall]] = []
    unindexed_calls: list[ToolCall] = []
    indexed_natives: list[tuple[int, GatewayMessage]] = []
    unindexed_natives: list[GatewayMessage] = []
    for call in raw_calls if isinstance(raw_calls, list) else ():
        if not isinstance(call, dict):
            raise ValueError("Responses retained tool call must be an object")
        item_id = call.get("item_id")
        output_index = call.get("output_index")
        if item_id is None and output_index is None:
            provider_item_id = None
            provider_output_index = None
        elif (
            (item_id is None or (isinstance(item_id, str) and bool(item_id)))
            and isinstance(output_index, int)
            and not isinstance(output_index, bool)
            and output_index >= 0
            and output_index not in indexes
        ):
            provider_item_id = item_id
            provider_output_index = output_index
            indexes.add(output_index)
        else:
            raise ValueError("Responses retained tool call identity is invalid")
        call_id = call["call_id"]
        name = call["name"]
        raw_arguments = call["arguments"]
        namespace = call.get("namespace")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
            or not isinstance(raw_arguments, str)
            or not (namespace is None or (isinstance(namespace, str) and namespace))
        ):
            raise ValueError("Responses retained tool call fields are invalid")
        if call.get("custom") is True:
            # A freeform custom tool call replays as the verbatim native item
            # (its input is opaque text, not JSON arguments), at its exact
            # provider output position.
            native_item: JsonObject = {
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": name,
                "input": raw_arguments,
            }
            if namespace is not None:
                native_item["namespace"] = namespace
            if provider_item_id is not None:
                native_item["id"] = provider_item_id
            status_value = call.get("status")
            if isinstance(status_value, str) and status_value:
                native_item["status"] = status_value
            native_message = GatewayMessage(role="assistant", provider_native_item=native_item)
            if provider_output_index is None:
                unindexed_natives.append(native_message)
            else:
                indexed_natives.append((provider_output_index, native_message))
            continue
        parsed_call = ToolCall(
            call_id=call_id,
            name=name,
            arguments=json.loads(raw_arguments),
            raw_arguments=raw_arguments,
            provider_item_id=provider_item_id,
            provider_output_index=provider_output_index,
            provider_status=(
                _provider_status(call.get("status"), default="completed")
                if provider_output_index is not None
                else None
            ),
            provider_namespace=namespace,
        )
        if provider_output_index is None:
            unindexed_calls.append(parsed_call)
        else:
            indexed_calls.append((provider_output_index, parsed_call))
    raw_encrypted = data.get("encrypted_reasoning", [])
    if not isinstance(raw_encrypted, list):
        raise ValueError("Responses encrypted reasoning must be an array")
    encrypted: list[tuple[int, EncryptedReasoningBlock]] = []
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
                    status=_provider_status(item.get("status"), default="completed"),
                ),
            )
        )
    raw_carrier = data.get("reasoning_content_carrier")
    sealed_carrier = None
    if raw_carrier is not None:
        if not isinstance(raw_carrier, str):
            raise ValueError("Responses reasoning carrier must be text")
        sealed_carrier = parse_reasoning_content_carrier(raw_carrier)
    indexed_output = bool(encrypted or indexed_calls or message_outputs or indexed_natives)
    if sealed_carrier is not None and indexed_output:
        raise ValueError("Responses reasoning carrier cannot mix with provider-indexed output")
    if text and indexed_output:
        raise ValueError(
            "Responses retained assistant text requires provider item identity and order"
        )
    output_items: list[tuple[int, str, object]] = [
        *((index, "reasoning", block) for index, block in encrypted),
        *((index, "call", call) for index, call in indexed_calls),
        *((index, "message", message) for index, message in message_outputs),
        *((index, "native", message) for index, message in indexed_natives),
    ]
    output_items.sort(key=lambda item: item[0])
    retained_messages: list[GatewayMessage] = []
    segment_reasoning: list[EncryptedReasoningBlock] = []
    segment_calls: list[ToolCall] = []
    segment_message: GatewayMessage | None = None

    def flush_segment() -> None:
        nonlocal segment_message
        if segment_message is None and not segment_reasoning and not segment_calls:
            return
        retained_messages.append(
            GatewayMessage(
                role="assistant",
                content=segment_message.content if segment_message is not None else None,
                tool_calls=tuple(segment_calls),
                provider_reasoning=tuple(segment_reasoning),
                provider_item_id=(
                    segment_message.provider_item_id if segment_message is not None else None
                ),
                provider_output_index=(
                    segment_message.provider_output_index if segment_message is not None else None
                ),
                provider_status=(
                    segment_message.provider_status if segment_message is not None else None
                ),
                provider_phase=(
                    segment_message.provider_phase if segment_message is not None else None
                ),
            )
        )
        segment_reasoning.clear()
        segment_calls.clear()
        segment_message = None

    for _output_index, kind, item in output_items:
        if kind == "message":
            if segment_message is not None:
                flush_segment()
            segment_message = cast(GatewayMessage, item)
        elif kind == "reasoning":
            segment_reasoning.append(cast(EncryptedReasoningBlock, item))
        elif kind == "native":
            flush_segment()
            retained_messages.append(cast(GatewayMessage, item))
        else:
            segment_calls.append(cast(ToolCall, item))
    flush_segment()
    if text or unindexed_calls or sealed_carrier is not None:
        retained_messages.append(
            GatewayMessage(
                role="assistant",
                content=text or None,
                tool_calls=tuple(unindexed_calls),
                provider_reasoning=(sealed_carrier,) if sealed_carrier is not None else (),
            )
        )
    retained_messages.extend(unindexed_natives)

    messages = (*context.messages, *retained_messages)
    has_encrypted_reasoning = any(
        block.kind == "encrypted_reasoning"
        for retained_message in messages
        for block in retained_message.provider_reasoning
    )
    if has_encrypted_reasoning and route_binding is None:
        raise ValueError("Responses encrypted reasoning requires route authority binding")
    continuations.remember_now(
        namespace=context.namespace,
        response_id=context.response_id,
        state=ContinuationState(
            episode_key=context.episode_key,
            messages=messages,
            route_binding=route_binding if has_encrypted_reasoning else None,
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

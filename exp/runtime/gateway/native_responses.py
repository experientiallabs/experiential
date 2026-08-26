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
    if bool(data.get("refusal")):
        return
    text = str(data.get("text") or "")
    raw_calls = data.get("tool_calls")
    tool_calls = tuple(
        ToolCall(
            call_id=str(call["call_id"]),
            name=str(call["name"]),
            arguments=json.loads(str(call["arguments"])),
            raw_arguments=str(call["arguments"]),
        )
        for call in (raw_calls if isinstance(raw_calls, list) else ())
    )
    if not text and not tool_calls:
        return
    message = GatewayMessage(
        role="assistant",
        content=text or None,
        tool_calls=tool_calls,
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
    return {
        "metadata": request.metadata or None,
        "parallel_tool_calls": request.parallel_tool_calls is not False,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "reasoning": {
            "effort": request.reasoning_effort,
            "summary": request.reasoning_summary,
        },
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
    }

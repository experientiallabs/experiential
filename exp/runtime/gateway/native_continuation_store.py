"""Admission reads from the continuation store written by native completion."""

from __future__ import annotations

import json

from exp_gateway_native import NativeContinuations

from exp.runtime.gateway.native_continuation_registration import continuation_message_replay
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ContinuationState,
    ProtocolNamespace,
)


class NativeContinuationStore(BoundedContinuationStore):
    """One bounded Rust store for local Responses completion and subsequent admission."""

    def __init__(self) -> None:
        """Create the native count, byte and TTL bounded store."""
        self.native = NativeContinuations()

    def remember_now(
        self, *, namespace: ProtocolNamespace, response_id: str, state: ContinuationState
    ) -> None:
        """Write through the same store when a caller explicitly retains an admitted turn."""
        self.native.remember(
            json.dumps(
                {
                    "namespace": namespace.model_dump(mode="json"),
                    "response_id": response_id,
                    "state": state.model_dump(mode="json"),
                    "replay": [
                        continuation_message_replay(message, index)
                        for index, message in enumerate(state.messages)
                    ],
                },
                separators=(",", ":"),
            )
        )

    def resolve_now(
        self, *, namespace: ProtocolNamespace, previous_response_id: str
    ) -> ContinuationState:
        """Restore hidden provider replay authority only within its admitted namespace."""
        try:
            document = json.loads(
                self.native.resolve(namespace.model_dump_json(), previous_response_id)
            )
        except ValueError as error:
            raise OpenAIProtocolError(
                status_code=400,
                code="previous_response_not_found",
                message=(
                    "previous_response_id is unavailable or expired in this namespace. "
                    "Resend the full conversation history."
                ),
                param="previous_response_id",
            ) from error
        state = document["state"]
        for replay in document["replay"]:
            message = state["messages"][replay["message_index"]]
            message.update(
                {
                    key: value
                    for key, value in replay.items()
                    if key not in {"message_index", "tool_calls"}
                }
            )
            calls = {call["call_id"]: call for call in message.get("tool_calls", [])}
            for call in replay.get("tool_calls", []):
                calls[call["call_id"]].update(call)
        return ContinuationState.model_validate(state)

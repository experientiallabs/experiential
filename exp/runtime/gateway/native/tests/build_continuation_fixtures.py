"""Generate independent Responses retention cases from the current Python implementation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.native_continuation_registration import continuation_registration
from exp.runtime.gateway.native_responses import remember_turn
from exp.runtime.gateway.native_responses_test import _binding, _context
from exp.runtime.openai_protocol.state import BoundedContinuationStore


def cases() -> list[JsonObject]:
    """Cover indexed, ordinary, empty, hosted and encrypted Responses outputs."""
    result: list[JsonObject] = []
    for index in range(208):
        context = replace(_context(), route_bindings=(_binding(),))
        data: JsonObject = {"text": "", "refusal": False, "tool_calls": []}
        if index >= 192:
            data["message_outputs"] = None
            data["text"] = "null outputs" if index % 2 else ""
            if index % 4 >= 2:
                data["message_item_id"] = "msg_legacy"
                data["message_output_index"] = 0
        elif index % 3 == 0:
            data["text"] = ["", "hello 世界", "\0 \\u0000"][index % 9 // 3]
            data["tool_calls"] = (
                [{"call_id": "call", "name": "lookup", "arguments": '{"q":"世界"}'}]
                if index % 2
                else []
            )
        else:
            messages: list[JsonObject] = []
            for i in range(index % 4):
                messages.append(
                    {
                        "item_id": f"message-{i}",
                        "output_index": i * 3,
                        "text": ["", "answer 世界"][i % 2],
                        "status": ["completed", "incomplete", "in_progress"][i % 3],
                        "phase": [None, "commentary", "final_answer"][i % 3],
                    }
                )
            data["message_outputs"] = messages
            if index % 2:
                data["tool_calls"] = [
                    {
                        "call_id": "call",
                        "name": "lookup",
                        "arguments": "{}",
                        "item_id": "function-item" if index % 5 else None,
                        "output_index": 1,
                        "namespace": "functions" if index % 7 else None,
                        "caller": {"type": "direct"},
                        "thought_signature": "signature",
                    }
                ]
            if index % 5:
                data["encrypted_reasoning"] = [
                    {
                        "item_id": "reasoning",
                        "output_index": 2,
                        "encrypted_content": "opaque payload",
                        "status": "completed",
                    }
                ]
            if index % 7 == 0:
                data["hosted_items"] = [
                    {
                        "output_index": 20,
                        "item": {"type": "web_search_call", "id": "search", "status": "completed"},
                    }
                ]
        store = BoundedContinuationStore()
        remember_turn(store, context=context, data=data, route_binding=_binding())
        state = store.resolve_now(
            namespace=context.namespace, previous_response_id=context.response_id
        )
        result.append(
            {
                "context": continuation_registration(context),
                "input": data,
                "expected": state.model_dump(mode="json"),
                "size_bytes": state.size_bytes,
                "has_native_items": bool(data.get("hosted_items")),
            }
        )
    return result


if __name__ == "__main__":
    Path(__file__).with_name("generated").joinpath("continuations.jsonl").write_text(
        "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases())
    )

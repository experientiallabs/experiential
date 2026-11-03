"""Explicit feedback for a known refused proposal, without repair or hidden retries."""

import json

from exp.common.core.artifacts import JsonObject


def reask_proposal(body: JsonObject, raw: JsonObject, reason: str) -> JsonObject:
    """Keep the original request and append the refused output and its contract violation.

    Callers own finite retry budgets and must persist each refusal and subsequent intent.
    This helper is never used for lost provider responses, committed work or low task scores.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("a re-ask needs the original request messages")
    output = raw.get("output")
    calls = output.get("tool_calls") if isinstance(output, dict) else None
    text = output.get("text") if isinstance(output, dict) else None
    refused = (
        json.dumps(calls, ensure_ascii=False)
        if calls
        else str(text or json.dumps(raw, ensure_ascii=False))
    )
    return {
        **body,
        "messages": [
            *messages,
            {"role": "assistant", "content": refused[:8000]},
            {
                "role": "user",
                "content": f"That proposal was refused: {reason} Return one complete, valid "
                "proposal for the same actions, following the required format exactly.",
            },
        ],
    }

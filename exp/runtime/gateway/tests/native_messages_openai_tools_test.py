"""OpenAI-shaped tools on Messages ``count_tokens`` traverse the served native gateway."""

from __future__ import annotations

import httpx
import pytest

from exp.runtime.gateway.tests.native_messages_test import (
    _engine,  # noqa: F401 - module fixture `engine`.
    _ServingEngine,
)

pytest.importorskip("exp_gateway_native")

_CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "look up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


def test_count_tokens_counts_a_chat_tool_list(engine: _ServingEngine) -> None:
    """A Chat-shaped tool list is counted, with the fold disclosed."""
    body = {
        "model": "coding",
        "messages": [{"role": "user", "content": "foo"}],
        "tools": [_CHAT_TOOL],
    }
    counted = httpx.post(
        f"{engine.base}/v1/messages/count_tokens",
        headers={"x-api-key": engine.raw_key},
        json=body,
        timeout=10.0,
    )
    assert counted.status_code == 200, counted.text
    answer = counted.json()
    assert isinstance(answer["input_tokens"], int) and answer["input_tokens"] > 0
    assert answer["x-experiential-ignored-parameters"] == [
        "tools.function->translated(custom)",
        "input_tokens->estimated(gateway_tokenizer)",
    ]
    without_tools = httpx.post(
        f"{engine.base}/v1/messages/count_tokens",
        headers={"x-api-key": engine.raw_key},
        json={key: value for key, value in body.items() if key != "tools"},
        timeout=10.0,
    )
    assert without_tools.status_code == 200, without_tools.text
    # The folded declaration is counted, not dropped.
    assert answer["input_tokens"] > without_tools.json()["input_tokens"]

"""Tests for shared provider helpers (verify ping budget + reachability semantics)."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from wmo.common.providers.base import (
    PING_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
    chat_request_output_budget,
    guard_starved_chat_response,
    guard_starved_completion,
    verify_via_ping,
)
from wmo.common.vendor.waterfall import ChatRequest, ChatResponse


class RecordingProvider:
    """Captures the ping's max_tokens so the budget is pinned by a test."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
        self.seen_max_tokens: int | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.seen_max_tokens = max_tokens
        return Completion(text="pong")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        return verify_via_ping(self)


class _RaisingProvider:
    """A provider whose complete() raises a fixed exception (to drive verify_via_ping)."""

    def __init__(self, exc: Exception) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
        self._exc = exc

    def complete(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 1
    ) -> Completion:
        raise self._exc

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:  # pragma: no cover - unused
        return verify_via_ping(self)


def test_ping_budget_covers_reasoning_models() -> None:
    # Reasoning models (GPT-5.x) burn output budget on reasoning before any visible token, so a
    # tiny ping budget makes OpenAI 400 with "max_tokens or model output limit was reached" even
    # though the credentials are fine. The ping must send a budget with headroom for that.
    provider = RecordingProvider()
    result = verify_via_ping(provider)
    assert result.ok is True
    assert provider.seen_max_tokens == PING_MAX_TOKENS
    assert PING_MAX_TOKENS >= 1024


def test_verify_treats_max_tokens_limit_as_reachable() -> None:
    # If a reasoning model spends even the larger budget on reasoning and 400s before any output,
    # that error PROVES auth + model id are valid, so verify must still report ok=True.
    exc = Exception(
        "Error code: 400 - {'error': {'message': 'Could not finish the message because "
        "max_tokens or model output limit was reached. Please try again with higher max_tokens.'}}"
    )
    result = verify_via_ping(_RaisingProvider(exc))
    assert result.ok
    assert result.model == "gpt-5.5"


def test_verify_reports_real_failures() -> None:
    # Auth / missing-model / network errors are genuine failures - not reachability confirmations.
    exc = Exception("Error code: 401 - invalid api key")
    result = verify_via_ping(_RaisingProvider(exc))
    assert not result.ok
    assert "401" in (result.detail or "")


def test_starved_completion_raises_instead_of_returning_empty_text() -> None:
    """Reasoning ate the whole budget: 200 with no text would be scored as a failed task."""
    starved = Completion(text="", usage=TokenUsage(input_tokens=10, output_tokens=8192))

    with pytest.raises(ValueError, match="consumed the entire 8192-token output budget"):
        guard_starved_completion(starved, 8192, model="gpt-5.6-sol", reasoning_effort="max")


def test_a_short_answer_at_top_effort_is_not_starved() -> None:
    """Starvation is prompt-dependent: effort=max on a trivial prompt emitted 6 tokens live, so a
    fixed budget floor would have rejected a call that works. Only the outcome may trigger."""
    fine = Completion(text="PONG", usage=TokenUsage(input_tokens=10, output_tokens=6))

    guard_starved_completion(fine, 2048, model="gpt-5.6-sol", reasoning_effort="max")


def test_empty_text_below_the_cap_is_not_relabelled_as_a_budget_problem() -> None:
    """A refusal or content filter returns empty WITHOUT hitting the cap; don't misdiagnose it."""
    refused = Completion(text="", usage=TokenUsage(input_tokens=10, output_tokens=12))

    guard_starved_completion(refused, 8192, model="gpt-5.6-sol", reasoning_effort="max")


def test_configs_with_no_effort_dial_are_untouched() -> None:
    """The dial is the opt-in: an undialed call keeps whatever behaviour it had before."""
    starved = Completion(text="", usage=TokenUsage(input_tokens=10, output_tokens=8192))

    guard_starved_completion(starved, 8192, model="gpt-5.5", reasoning_effort=None)


def _chat_response(
    content: str | None,
    completion_tokens: int,
    *,
    tool_calls: list[dict[str, JsonValue]] | None = None,
) -> ChatResponse:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatResponse.model_validate(
        {
            "choices": [{"index": 0, "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens},
        }
    )


def test_starved_structured_reply_raises() -> None:
    """complete_chat can be starved exactly like complete; it used to return the empty reply."""
    starved = _chat_response(None, 8192)

    with pytest.raises(ValueError, match="reasoning consumed"):
        guard_starved_chat_response(starved, 8192, model="gpt-5.6-sol", reasoning_effort="max")


def test_a_tool_call_with_empty_content_is_not_starvation() -> None:
    """An assistant turn that called a tool has empty content by design: the normal agent
    success shape, and the one thing a content-only check would wrongly kill."""
    tool_call: list[dict[str, JsonValue]] = [
        {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
    ]
    called = _chat_response(None, 8192, tool_calls=tool_call)

    guard_starved_chat_response(called, 8192, model="gpt-5.6-sol", reasoning_effort="max")


def test_chat_request_output_budget_reads_either_field_spelling() -> None:
    """The budget arrives under whichever name the caller used, or the guard reads None."""
    assert (
        chat_request_output_budget(
            ChatRequest.model_validate({"messages": [], "max_completion_tokens": 4096})
        )
        == 4096
    )
    assert (
        chat_request_output_budget(ChatRequest.model_validate({"messages": [], "max_tokens": 512}))
        == 512
    )
    assert chat_request_output_budget(ChatRequest.model_validate({"messages": []})) is None

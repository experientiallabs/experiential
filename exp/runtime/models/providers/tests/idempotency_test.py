"""A resend of one canonical intent carries one stable key only on wires that define the header."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.runtime.models.call_evidence import canonical_call, intent_idempotency_key
from exp.runtime.models.providers.anthropic import AnthropicClient
from exp.runtime.models.providers.base import ProviderHttpClient
from exp.runtime.models.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenRouterClient,
)
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
)

_INTENT = ArtifactInput(artifact_id="structured-intent-fixture", sha256="1" * 64)
_OTHER_INTENT = ArtifactInput(artifact_id="structured-intent-other", sha256="2" * 64)
_ONCE = RetryPolicy(maximum_attempts=1, initial_delay_seconds=0, maximum_delay_seconds=0)
_CHAT_OK: JsonObject = {
    "choices": [{"message": {"content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}
_MESSAGES_OK: JsonObject = {
    "model": "claude-fixture",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _openai_compatible(transport: ScriptedJsonTransport) -> ProviderHttpClient:
    """Build an OpenAI-compatible client that sends each request once."""
    return OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://compatible.fixture/v1",
        api_key="fixture-key",
        transport=transport,
        retry_policy=_ONCE,
    )


def _openrouter(transport: ScriptedJsonTransport) -> ProviderHttpClient:
    """Build an OpenRouter client that sends each request once."""
    return OpenRouterClient(
        model=_snapshot("openrouter", "vendor/model"),
        base_url="https://router.fixture/v1",
        api_key="fixture-key",
        transport=transport,
        retry_policy=_ONCE,
    )


@pytest.mark.parametrize("build", [_openai_compatible, _openrouter])
def test_resend_after_an_ambiguous_failure_reuses_the_intent_key(
    build: Callable[[ScriptedJsonTransport], ProviderHttpClient],
) -> None:
    """A fresh client resending the same intent after an unknown outcome sends the same key."""
    first = ScriptedJsonTransport([ProviderTransportError("provider connection reset")])
    second = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=_CHAT_OK)])
    with canonical_call(_INTENT):
        with pytest.raises(ProviderTransportError):
            build(first).complete(_request())
        assert build(second).complete(_request()).output.content == "ok"
    keys = [first.requests[0].headers["Idempotency-Key"]]
    keys.append(second.requests[0].headers["Idempotency-Key"])
    assert keys[0] == keys[1]
    with canonical_call(_INTENT):
        derived = intent_idempotency_key(first.requests[0].url, first.requests[0].payload)
    assert keys[0] == derived
    assert keys[0].startswith("exp-intent-") and len(keys[0]) == len("exp-intent-") + 48


def test_intent_keys_differ_by_intent_and_by_request_and_need_an_intent() -> None:
    """Different intents or bodies never share a key; outside an intent none is derived."""
    url = "https://compatible.fixture/v1/chat/completions"
    assert intent_idempotency_key(url, {"model": "m"}) is None
    with canonical_call(_INTENT):
        base = intent_idempotency_key(url, {"model": "m"})
        other_body = intent_idempotency_key(url, {"model": "n"})
        other_url = intent_idempotency_key(url + "?v=2", {"model": "m"})
    with canonical_call(_OTHER_INTENT):
        other_intent = intent_idempotency_key(url, {"model": "m"})
    assert base is not None
    assert len({base, other_body, other_url, other_intent}) == 4


def test_explicit_caller_key_wins_over_the_intent_key() -> None:
    """A gateway or caller-owned key is forwarded unchanged."""
    transport = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=_CHAT_OK)])
    client = _openai_compatible(transport)
    with canonical_call(_INTENT):
        asyncio.run(client.complete_async(_request(), idempotency_key="caller-key"))
    assert transport.requests[0].headers["Idempotency-Key"] == "caller-key"


def test_anthropic_messages_defines_no_idempotency_header_so_no_intent_key_is_derived() -> None:
    """The Messages protocol has no idempotency contract; ambiguity relies on never-redraw."""
    assert AnthropicClient.supports_idempotency_key is False
    assert OpenAICompatibleClient.supports_idempotency_key is True
    transports = [
        ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=_MESSAGES_OK)])
        for _ in range(2)
    ]
    with canonical_call(_INTENT):
        for transport in transports:
            AnthropicClient(
                model=_snapshot("anthropic", "claude-fixture"),
                api_key="fixture-key",
                base_url="https://anthropic.fixture/v1",
                transport=transport,
            ).complete(_request())
    keys = [transport.requests[0].headers["Idempotency-Key"] for transport in transports]
    assert not any(key.startswith("exp-intent-") for key in keys)
    assert keys[0] != keys[1]


def test_a_wire_without_idempotency_never_redraws_an_ambiguous_outcome() -> None:
    """Neither transport loss nor an unproven overload response authorizes another Messages draw."""
    retrying = RetryPolicy(maximum_attempts=3, initial_delay_seconds=0, maximum_delay_seconds=0)
    ok = JsonHttpResponse(status_code=200, body=_MESSAGES_OK)

    def anthropic(transport: ScriptedJsonTransport) -> AnthropicClient:
        """A Messages client allowed three attempts."""
        return AnthropicClient(
            model=_snapshot("anthropic", "claude-fixture"),
            api_key="fixture-key",
            base_url="https://anthropic.fixture/v1",
            transport=transport,
            retry_policy=retrying,
        )

    lost = ScriptedJsonTransport([ProviderTransportError("connection reset after send"), ok])
    with pytest.raises(ProviderTransportError, match="connection reset"):
        anthropic(lost).complete(_request())
    assert len(lost.requests) == 1, "an unknown outcome is never redrawn"
    overloaded = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=503, body={"error": {"type": "overloaded"}}), ok]
    )
    with pytest.raises(ProviderTransportError) as failure:
        anthropic(overloaded).complete(_request())
    assert failure.value.status_code == 503
    assert len(overloaded.requests) == 1


def test_an_idempotent_wire_resends_an_ambiguous_outcome_under_one_key() -> None:
    """The OpenAI-family wire deduplicates by key, so a resend after a lost call is safe."""
    retrying = RetryPolicy(maximum_attempts=3, initial_delay_seconds=0, maximum_delay_seconds=0)
    transport = ScriptedJsonTransport(
        [
            ProviderTransportError("connection reset after send"),
            JsonHttpResponse(status_code=200, body=_CHAT_OK),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://compatible.fixture/v1",
        api_key="fixture-key",
        transport=transport,
        retry_policy=retrying,
    )
    assert client.complete(_request()).output.content == "ok"
    keys = {row.headers["Idempotency-Key"] for row in transport.requests}
    assert len(transport.requests) == 2 and len(keys) == 1

"""Empty completed Chat responses receive bounded retries without accepting malformed tools."""

import asyncio

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RetryPolicy,
    ScriptedJsonTransport,
)


@pytest.mark.parametrize(
    "empty",
    [
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": None}}]},
    ],
)
def test_empty_chat_response_retries_then_returns_real_output(empty: JsonObject) -> None:
    """A failed wire response does not end a rollout before a bounded retry can succeed."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body=empty),
            JsonHttpResponse(status_code=200, body={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    assert client.complete(_request()).output.content == "done"
    assert len(transport.requests) == 2
    assert (
        transport.requests[0].headers["Idempotency-Key"]
        != transport.requests[1].headers["Idempotency-Key"]
    )


@pytest.mark.parametrize("explicit_key", [None, "caller-owned-operation"])
def test_lost_completed_replay_rotates_only_client_owned_keys(explicit_key: str | None) -> None:
    """A lost replay can start a fresh accounted attempt without rewriting caller identity."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=409, body={"error": {"code": "idempotency_replay_unavailable"}}
            ),
            JsonHttpResponse(status_code=200, body={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    response = (
        client.complete(_request())
        if explicit_key is None
        else asyncio.run(client.complete_async(_request(), idempotency_key=explicit_key))
    )
    assert response.output.content == "done"
    assert response.economics.provider_attempts == 2
    keys = [request.headers["Idempotency-Key"] for request in transport.requests]
    assert (keys[0] == keys[1]) is (explicit_key is not None)


def test_empty_chat_response_stops_at_the_retry_bound() -> None:
    """Repeated emptiness remains invalid and cannot cause an unbounded provider loop."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=200, body={"choices": []}) for _ in range(2)]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fixture",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )
    with pytest.raises(ProviderRetryableResponseError):
        client.complete(_request())
    assert len(transport.requests) == 2

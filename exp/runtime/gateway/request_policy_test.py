"""Closed request policy validation, replay identity and budget invariants."""

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.runtime.anthropic_protocol.requests import decode_messages, decode_messages_count_tokens
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.request_policy import GatewayRequestPolicy, attempt_policy
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


@pytest.mark.parametrize(
    "policy",
    [
        {"retry": {"max_total_attempts": True}},
        {"retry": {"max_total_attempts": "2"}},
        {"retry": {"max_total_attempts": 9}},
        {"retry": {"max_attempts_per_route": 0}},
        {"retry": {"max_attempts_per_route": 5}},
        {"retry": {"backoff": {"type": "default"}}},
        {"retry": {"backoff": {"type": "none", "base_delay_ms": 1}}},
        {"retry": {"backoff": {"type": "exponential", "base_delay_ms": 9000}}},
        {"retry": {"backoff": {"type": "exponential", "multiplier": float("inf")}}},
        {"retry": {"backoff": {"type": "exponential", "multiplier": True}}},
        {"routing": {"allow_fallbacks": 0}},
        {"routing": {"route_id": "mp-private"}},
        {"routing": {"route_id": "route_" + "A" * 64}},
        {"routing": {"unknown": True}},
        {"unknown": {}},
    ],
)
def test_rejects_bad_policy(policy: JsonObject) -> None:
    """Unknown fields and coercible impostors cannot widen a policy."""
    with pytest.raises(ValidationError):
        GatewayRequestPolicy.model_validate(policy)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_policy_decodes_without_becoming_provider_input(surface: str) -> None:
    """All generation APIs preserve controls separately from provider parameters."""
    gateway: JsonObject = {
        "retry": {"max_total_attempts": 1},
        "routing": {"allow_fallbacks": False},
    }
    match surface:
        case "chat":
            request = decode_chat(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "gateway": gateway,
                }
            ).request
        case "responses":
            request = decode_responses({"model": "test", "input": "hi", "gateway": gateway}).request
        case _:
            request = decode_messages(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10,
                    "gateway": gateway,
                }
            ).request
    assert request.gateway is not None
    assert request.gateway.retry is not None
    assert request.gateway.retry.max_total_attempts == 1
    assert "gateway" not in request.model_dump()
    plain = request.model_copy(update={"gateway": None})
    assert worst_case_input_tokens(request) == worst_case_input_tokens(plain)
    assert canonical_request_sha256(request) != canonical_request_sha256(plain)
    assert canonical_request_sha256(plain) == sha256_json(plain)
    empty = request.model_copy(update={"gateway": GatewayRequestPolicy()})
    assert canonical_request_sha256(empty) == canonical_request_sha256(plain)


def test_explicit_cap_differs_from_default_semantic_rounds() -> None:
    """An explicit per-route cap bounds every dial, not just ordinary retries."""
    default = attempt_policy(None)
    explicit = attempt_policy(
        GatewayRequestPolicy.model_validate({"retry": {"max_attempts_per_route": 2}})
    )
    assert (
        default.maximum_same_deployment_attempts == explicit.maximum_same_deployment_attempts == 2
    )
    assert default.permits(3, 3)
    assert not explicit.permits(3, 3)
    assert not default.permits(8, 1)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "policy,param,remedy",
    [
        ({"retry": {"max_total_attempts": 9}}, "gateway.retry.max_total_attempts", "8"),
        (
            {"retry": {"backoff": {"type": "exponential", "multiplier": 5}}},
            "gateway.retry.backoff.multiplier",
            "4",
        ),
        (
            {"retry": {"backoff": {"type": "none", "delay": 5}}},
            "gateway.retry.backoff.delay",
            "Remove",
        ),
        ({"retry": {"backoff": {"type": "default"}}}, "gateway.retry.backoff.type", "exponential"),
    ],
)
def test_policy_errors_name_the_actual_field_and_remedy(
    surface: str, policy: JsonObject, param: str, remedy: str
) -> None:
    """Union tags never leak into nested paths and limits state how to repair them."""
    payload: JsonObject = {"model": "test", "gateway": policy}
    with pytest.raises(OpenAIProtocolError) as caught:
        match surface:
            case "chat":
                decode_chat({**payload, "messages": [{"role": "user", "content": "hi"}]})
            case "responses":
                decode_responses({**payload, "input": "hi"})
            case _:
                decode_messages(
                    {**payload, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
                )
    assert caught.value.detail.param == param
    assert remedy in caught.value.detail.message


def test_count_tokens_rejects_generation_controls() -> None:
    """Prompt counting never silently accepts a control it cannot honor."""
    with pytest.raises(OpenAIProtocolError) as caught:
        decode_messages_count_tokens(
            {"model": "test", "messages": [{"role": "user", "content": "hi"}], "gateway": {}}
        )
    assert caught.value.detail.param == "gateway"

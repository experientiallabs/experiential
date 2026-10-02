"""Deterministic request identity for native non-conversational decisions."""

from __future__ import annotations

import hashlib

import pytest

from exp.common.core.artifacts import JsonObject, canonical_json_bytes, sha256_json
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.decisions_contracts import ChoiceQuestion, DecisionRequest, NoulQuestion
from exp.runtime.gateway.replay_identity import caller_operation_sha256, canonical_request_sha256


def test_caller_operation_uses_only_the_standard_opt_in_key() -> None:
    """Correlation identifiers are not retry keys; the digest namespace stays byte-identical."""
    plain = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )
    assert caller_operation_sha256(plain) is None
    keyed = plain.model_copy(update={"idempotency_key": "fixture-key"})
    assert (
        caller_operation_sha256(keyed)
        == hashlib.sha256(b"gateway-caller-operation-v1\0fixture-key").hexdigest()
    )


def test_decision_identity_is_plain_canonical_json_independent_of_mapping_order() -> None:
    """State and question ordering do not create a different content fingerprint."""
    noul = NoulQuestion(instructions={"test": "clear", "language": "日本語"})
    choice = ChoiceQuestion(instructions="Select", criteria={"yes": "allowed", "no": None})
    request = DecisionRequest(state={"b": [2, 1], "a": "你好"}, questions={"n": noul, "c": choice})
    reordered = DecisionRequest(
        state={"a": "你好", "b": [2, 1]},
        questions={
            "c": ChoiceQuestion(instructions="Select", criteria={"no": None, "yes": "allowed"}),
            "n": NoulQuestion(instructions={"language": "日本語", "test": "clear"}),
        },
    )
    assert canonical_request_sha256(request) == sha256_json(request)
    assert canonical_request_sha256(request) == canonical_request_sha256(reordered)
    assert canonical_request_sha256(request) == canonical_request_sha256(
        DecisionRequest.model_validate_json(request.model_dump_json())
    )
    serialized = canonical_json_bytes(request)
    assert b'"surface":"decisions"' in serialized
    assert b'"messages"' not in serialized
    assert b'"idempotency_key"' not in serialized
    assert b'"input_token_reservation"' not in serialized


@pytest.mark.parametrize(
    "change",
    [
        {"state": {"value": 2}},
        {"questions": {"renamed": {"type": "noul", "instructions": "Accept?"}}},
        {"questions": {"check": {"type": "noul", "instructions": "Reject?"}}},
        {
            "questions": {
                "check": {"type": "noul", "instructions": "Accept?", "criteria": {"true": "valid"}}
            }
        },
        {
            "questions": {
                "check": {
                    "type": "choice",
                    "instructions": "Accept?",
                    "criteria": {"a": None, "b": None},
                }
            }
        },
        {
            "questions": {
                "check": {"type": "score", "instructions": "Accept?", "criteria": ["low", "high"]}
            }
        },
    ],
)
def test_decision_identity_changes_with_provider_significant_content(change: JsonObject) -> None:
    """State, identifiers, instructions, question types, and criteria all join the hash."""
    request = DecisionRequest(
        state={"value": 1}, questions={"check": NoulQuestion(instructions="Accept?")}
    )
    changed = DecisionRequest.model_validate({**request.model_dump(mode="json"), **change})
    assert canonical_request_sha256(request) != canonical_request_sha256(changed)


def test_score_criteria_order_is_significant() -> None:
    """Unlike object key order, ordered score criteria change the provider request."""
    request = DecisionRequest.model_validate(
        {
            "state": "record",
            "questions": {
                "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}
            },
        }
    )
    reordered = DecisionRequest.model_validate(
        {
            "state": "record",
            "questions": {
                "score": {"type": "score", "instructions": "Rate", "criteria": ["high", "low"]}
            },
        }
    )
    assert canonical_request_sha256(request) != canonical_request_sha256(reordered)


def test_provider_preferences_join_request_identity() -> None:
    """A reused operation key with a different ``provider`` object is a different request.

    The caller's OpenRouter routing preferences (order, only, data_collection,
    zdr) change which upstream serves the same body, so they join the digest;
    a request without the object keeps its exact pre-existing identity.
    """
    from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest

    def request(preferences: JsonObject | None) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hi"),),
            provider_preferences=preferences,
        )

    bare = request(None)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    strict = canonical_request_sha256(request({"zdr": True, "data_collection": "deny"}))
    loose = canonical_request_sha256(request({"zdr": True, "data_collection": "allow"}))
    ordered = canonical_request_sha256(request({"zdr": True, "order": ["Azure"]}))
    assert len({canonical_request_sha256(bare), strict, loose, ordered}) == 4
    assert strict == canonical_request_sha256(request({"data_collection": "deny", "zdr": True}))


def test_web_search_joins_request_identity_without_its_results() -> None:
    """The caller's search ask changes identity; the fetched results never do."""
    from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
    from exp.runtime.gateway.web_search.contracts import GatewayWebSearch

    def request(search: GatewayWebSearch | None) -> GatewayRequest:
        return GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hi"),),
            web_search=search,
        )

    bare = request(None)
    assert canonical_request_sha256(bare) == sha256_json(bare)
    plugin = canonical_request_sha256(request(GatewayWebSearch(declared_as="plugin")))
    narrow = canonical_request_sha256(
        request(GatewayWebSearch(declared_as="plugin", max_results=2))
    )
    assert len({canonical_request_sha256(bare), plugin, narrow}) == 3
    # The search object is excluded from plain serialization, so bodies digest alike.
    assert sha256_json(request(GatewayWebSearch(declared_as="plugin"))) == sha256_json(bare)


def test_chat_nested_thinking_budget_changes_replay_identity() -> None:
    """Changing a retained budget cannot replay an answer computed under another bound."""
    from exp.runtime.openai_protocol.requests import decode_chat

    requests = [
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget_tokens": budget},
            }
        ).request
        for budget in (1024, 2048)
    ]
    assert canonical_request_sha256(requests[0]) != canonical_request_sha256(requests[1])

"""Deterministic rule detection and redaction through the real policy engine."""

from __future__ import annotations

import asyncio

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    GuardrailCompletion,
    GuardrailRejected,
    GuardrailToolCall,
)
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.text_coverage_test import (
    SUPPORTED_TEXT_CONTEXTS,
    UNSUPPORTED_TEXT_INPUTS,
    text_context_request,
    unsupported_text_request,
)


@pytest.mark.parametrize("field", UNSUPPORTED_TEXT_INPUTS)
def test_regex_classifier_refuses_uninspectable_input(field: str) -> None:
    """A deterministic verdict must cover the whole supported prompt."""
    classifier = RegexClassifier(
        RegexAdapterDocument(adapter_id="patterns", builtin_patterns=(BuiltinPattern.EMAIL,))
    )
    engine = engine_from_document(_document({"builtin_patterns": ["email"]}))
    check = engine.policies_for("org", "identity")[0].input_checks[0]
    with pytest.raises(ClassifierCoverageError):
        asyncio.run(classifier.inspect_input(request=unsupported_text_request(field), check=check))


@pytest.mark.parametrize("field", SUPPORTED_TEXT_CONTEXTS)
def test_regex_context_match_is_flagged_without_rewriting_constraints(field: str) -> None:
    """A schema or example match is refused like a matched tool argument under modify."""
    engine = engine_from_document(_document({"builtin_patterns": ["email"]}))
    policy = engine.policies_for("org", "identity")[0]
    benign = text_context_request(field, "benign")
    assert (
        asyncio.run(engine.enforce_input(policy=policy, request=benign, deadline_monotonic=1e12))
        is benign
    )
    request = text_context_request(field, "alice@example.com")
    original = request.model_copy(deep=True)
    with pytest.raises(GuardrailRejected) as rejected:
        asyncio.run(engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12))
    assert rejected.value.failure.failure_class.value == "guardrail"
    assert request == original


def _document(adapter: JsonObject, *, action: str = "modify") -> JsonObject:
    """Bind a regex rule to both stages of one protected identity."""
    return {
        "adapters": [{"kind": "regex", "adapter_id": "patterns", **adapter}],
        "policies": [
            {
                "policy_id": "redact",
                "organization_id": "org",
                "identity_id": "identity",
                "protected": True,
                "checks": [
                    {
                        "check_id": stage,
                        "capability": "pii",
                        "stage": stage,
                        "action": action,
                        "adapter_id": "patterns",
                        "timeout_ms": 500,
                    }
                    for stage in ("input", "output")
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Contact alice@example.com today", "Contact [REDACTED] today"),
        ("日本語 😀 alice@example.com fin", "日本語 😀 [REDACTED] fin"),
        ("alice@example.com and bob@example.org", "[REDACTED] and [REDACTED]"),
        ("No personal information", "No personal information"),
        ("Cards: 4111111111111111 5555555555554444", "Cards: [REDACTED] [REDACTED]"),
    ],
)
def test_builtin_redaction_preserves_surrounding_text_on_both_stages(
    content: str, expected: str
) -> None:
    """Real input/output chains preserve Unicode and only replace detected spans."""
    engine = engine_from_document(_document({"builtin_patterns": ["email", "credit_card"]}))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
    )
    result = asyncio.run(
        engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
    )
    assert result.messages[0].content == expected
    assert request.messages[0].content == content
    output = asyncio.run(
        engine.enforce_output(
            policy=policy, completion=GuardrailCompletion(text=content), deadline_monotonic=1e12
        )
    )
    assert output.text == expected
    assert engine.policies_for("org", "other") == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111 1111 1111 1111", "[REDACTED]"),
        ("4111-1111-1111-1111", "[REDACTED]"),
        ("4111111111111111", "[REDACTED]"),
        ("4111111111111111 5555555555554444", "[REDACTED] [REDACTED]"),
        ("4111 1111 1111 1111 5555 5555 5555 4444", "[REDACTED]"),
        ("4111-1111-1111-1111 378282246310005", "[REDACTED] [REDACTED]"),
        ("4111111111111111, 5555555555554444", "[REDACTED], [REDACTED]"),
        ("4111111111111112", "4111111111111112"),
        ("0000000000000000", "0000000000000000"),
        ("141111111111111111111", "141111111111111111111"),
        ("4111 1111 1111 1111 0000", "[REDACTED] 0000"),
    ],
)
def test_card_candidates_require_luhn(text: str, expected: str) -> None:
    """Invalid card candidates remain untouched; valid formatted numbers redact."""
    classifier = RegexClassifier(
        RegexAdapterDocument(adapter_id="cards", builtin_patterns=(BuiltinPattern.CREDIT_CARD,))
    )
    assert classifier._redact(text)[1] == expected


def test_overlapping_patterns_redact_the_union_and_replacement_is_literal() -> None:
    """Two matching slices cannot leave a secret suffix; backreferences are never expanded."""
    classifier = RegexClassifier(
        RegexAdapterDocument(adapter_id="custom", patterns=("abc", "cde"), replacement=r"\1")
    )
    assert classifier._redact("xabcdey") == (True, r"x\1y")


@pytest.mark.parametrize("pattern", ["(", r"(a)\1", r"(?<=a)b"])
def test_invalid_or_unsupported_re2_patterns_are_rejected_without_echo(pattern: str) -> None:
    """Customer patterns never enter parser error logs or exception text."""
    with pytest.raises(ValueError, match="invalid RE2 expression"):
        engine_from_document(_document({"patterns": [pattern]}))


@pytest.mark.parametrize(
    "adapter",
    [{}, {"patterns": [""]}, {"patterns": ["a" * 1025]}, {"builtin_patterns": ["email", "email"]}],
)
def test_invalid_rule_configuration_cannot_be_activated(adapter: JsonObject) -> None:
    """Empty or oversized rules fail while loading, before traffic is admitted."""
    with pytest.raises(ValueError):
        engine_from_document(_document(adapter))


def test_adversarial_backtracking_pattern_completes_with_linear_engine() -> None:
    """A classic exponential backtracking subject uses RE2's bounded matcher."""
    classifier = RegexClassifier(RegexAdapterDocument(adapter_id="custom", patterns=(r"(a+)+$",)))
    text = "a" * 100_000 + "!"
    assert classifier._redact(text) == (False, text)


def test_block_action_refuses_match_and_allows_clean_input() -> None:
    """A custom content rule can block instead of modifying the request."""
    engine = engine_from_document(_document({"patterns": [r"internal-[0-9]+"]}, action="block"))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    for text in ("internal-123", "public documentation"):
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content=text),),
        )
        if text.startswith("internal"):
            with pytest.raises(GuardrailRejected):
                asyncio.run(
                    engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
                )
        else:
            assert (
                asyncio.run(
                    engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12)
                )
                == request
            )


def test_tool_completion_is_blocked_instead_of_rewritten() -> None:
    """Redaction never passes sensitive tool arguments to the caller."""
    engine = engine_from_document(_document({"builtin_patterns": ["email"]}))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    completion = GuardrailCompletion(
        tool_calls=(
            GuardrailToolCall(
                call_id="call", name="send", arguments='{"email":"alice@example.com"}'
            ),
        )
    )
    with pytest.raises(GuardrailRejected):
        asyncio.run(
            engine.enforce_output(policy=policy, completion=completion, deadline_monotonic=1e12)
        )


def test_api_key_family_redacts_known_prefixes() -> None:
    """Built-in token families preserve text outside complete token matches."""
    classifier = RegexClassifier(
        RegexAdapterDocument(adapter_id="keys", builtin_patterns=(BuiltinPattern.API_KEY,))
    )
    for token in ("sk-proj-" + "a" * 40, "ghp_" + "b" * 36, "xpl_" + "c" * 32):
        assert classifier._redact("token: " + token) == (True, "token: [REDACTED]")


def test_match_explosion_is_bounded() -> None:
    """Very dense custom matches cannot accumulate unbounded span lists."""
    classifier = RegexClassifier(RegexAdapterDocument(adapter_id="dense", patterns=("a",)))
    with pytest.raises(ValueError, match="match limit"):
        classifier._redact("a" * 5000)


@pytest.mark.parametrize("protected", [True, False])
def test_input_tool_arguments_never_escape_a_redaction_rule(protected: bool) -> None:
    """Known sensitive tool content is refused even when uncertainty is fail-open."""
    engine = engine_from_document(_document({"builtin_patterns": ["email"]}))
    policy = engine.policies_for("org", "identity")[0]
    assert policy is not None
    policy = policy.model_copy(update={"protected": protected})
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call",
                        name="lookup",
                        arguments={"email": "alice@example.com"},
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(GuardrailRejected):
        asyncio.run(engine.enforce_input(policy=policy, request=request, deadline_monotonic=1e12))


def test_card_candidate_work_is_bounded() -> None:
    """A long ambiguous run cannot force unlimited candidate validation."""
    classifier = RegexClassifier(
        RegexAdapterDocument(adapter_id="cards", builtin_patterns=(BuiltinPattern.CREDIT_CARD,))
    )
    with pytest.raises(ValueError, match="match limit"):
        classifier._redact("4111 " * 5000)

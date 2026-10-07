"""Tests for replaceable classifier adapters."""

from __future__ import annotations

import asyncio

import pytest

from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import (
    ClassifierRegistry,
    KeywordClassifier,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailToolCall,
)
from exp.runtime.gateway.guardrails.text_coverage_test import (
    SUPPORTED_TEXT_CONTEXTS,
    UNSUPPORTED_TEXT_INPUTS,
    text_context_request,
    tool_error_request,
    unsupported_text_request,
)


@pytest.mark.parametrize("content", ["", "benign"])
def test_keyword_classifier_inspects_synthesized_tool_error_text(content: str) -> None:
    """A failed tool's flag reaches detection even with an empty or benign result body."""
    classifier = KeywordClassifier((f"[tool error] {content}",))
    for failed in (False, True):
        request = tool_error_request(failed=failed, content=content)
        original = request.model_copy(deep=True)
        verdict = asyncio.run(classifier.inspect_input(request=request, check=_check()))
        assert verdict.flagged is failed
        assert request == original


@pytest.mark.parametrize("field", UNSUPPORTED_TEXT_INPUTS)
def test_keyword_classifier_refuses_uninspectable_input(field: str) -> None:
    """A hidden needle cannot be approved by inspecting only the visible message text."""
    classifier = KeywordClassifier(("alice@example.com",))
    with pytest.raises(ClassifierCoverageError):
        asyncio.run(
            classifier.inspect_input(request=unsupported_text_request(field), check=_check())
        )


@pytest.mark.parametrize("field", SUPPORTED_TEXT_CONTEXTS)
def test_keyword_classifier_inspects_canonical_tool_and_schema_context(field: str) -> None:
    """Textual tool definitions and examples do not become an uninspected side channel."""
    classifier = KeywordClassifier(("alice@example.com",))
    for text, flagged in (("benign", False), ("alice@example.com", True)):
        verdict = asyncio.run(
            classifier.inspect_input(request=text_context_request(field, text), check=_check())
        )
        assert verdict.flagged is flagged


def _check() -> GuardrailCheck:
    """Return one content-safety input check."""
    return GuardrailCheck(
        check_id="input-safety",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=50,
        adapter_id="keyword-safety",
    )


def test_keyword_classifier_flags_needles_in_text_and_tool_arguments() -> None:
    """Coarse local needles match message text, completion text, and tool arguments."""
    classifier = KeywordClassifier(("forbidden",))
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="this is Forbidden"),),
    )
    completion = GuardrailCompletion(
        text="ok",
        tool_calls=(
            GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"forbidden"}'),
        ),
    )

    flagged_input = asyncio.run(classifier.inspect_input(request=request, check=_check()))
    flagged_output = asyncio.run(classifier.inspect_output(completion=completion, check=_check()))
    assert flagged_input.flagged is True
    assert flagged_output.flagged is True
    assert classifier.input_calls == 1
    assert classifier.output_calls == 1


def test_keyword_input_still_inspects_complete_tool_history_with_replay_metadata() -> None:
    """Raw canonical arguments stay inspectable without discarding provider identifiers."""
    classifier = KeywordClassifier(("alice@example.com",))
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="Use a tool"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-one",
                        name="lookup",
                        arguments={"email": "alice@example.com"},
                        raw_arguments='{"email": "alice@example.com"}',
                        provider_item_id="function-one",
                        provider_output_index=0,
                    ),
                ),
            ),
            GatewayMessage(role="tool", tool_call_id="call-one", content="done"),
        ),
    )
    original = request.model_copy(deep=True)
    assert asyncio.run(classifier.inspect_input(request=request, check=_check())).flagged
    assert request == original


def test_scripted_classifier_returns_authored_verdicts_without_retaining_content() -> None:
    """Scripted adapters expose call counts and never store the inspected payload."""
    classifier = ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True))
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="secret-prompt"),),
    )

    assert asyncio.run(classifier.inspect_input(request=request, check=_check())).flagged is True
    assert classifier.input_calls == 1
    assert not hasattr(classifier, "request")


def test_registry_is_replaceable_by_adapter_id() -> None:
    """Operators can replace one adapter without changing the policy identity."""
    registry = ClassifierRegistry({"keyword-safety": KeywordClassifier(("a",))})
    replacement = ScriptedClassifier()
    registry.register("keyword-safety", replacement)

    assert registry.require("keyword-safety") is replacement
    with pytest.raises(KeyError):
        registry.require("missing")


def test_keyword_classifier_requires_non_empty_needles() -> None:
    """An empty needle list is a configuration error, not a no-op detector."""
    with pytest.raises(ValueError, match="non-empty"):
        KeywordClassifier(())

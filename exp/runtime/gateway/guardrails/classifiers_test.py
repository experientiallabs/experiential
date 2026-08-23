"""Tests for replaceable classifier adapters."""

from __future__ import annotations

import asyncio

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import (
    BoundedSyncClassifier,
    ClassifierRegistry,
    KeywordClassifier,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailToolCall,
)


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


class _SyncAllow:
    """Minimal leftover synchronous adapter for the compatibility wrapper."""

    def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Allow every request."""
        del request, check
        return ClassifierVerdict(flagged=False)

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Allow every completion."""
        del completion, check
        return ClassifierVerdict(flagged=False)


def test_bounded_sync_classifier_runs_leftover_adapters_on_a_private_pool() -> None:
    """The compatibility wrapper exposes the async inspect contract."""
    wrapper = BoundedSyncClassifier(_SyncAllow(), max_workers=1)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )

    verdict = asyncio.run(wrapper.inspect_input(request=request, check=_check()))

    assert verdict.flagged is False

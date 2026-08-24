"""Tests for replaceable classifier adapters."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import BoundedInspect, ClassifierTimeoutError
from exp.runtime.gateway.guardrails.classifiers import (
    BoundedSyncClassifier,
    ClassifierRegistry,
    KeywordClassifier,
    ScriptedClassifier,
    SyncClassifierBusyError,
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
    assert wrapper.admitted_inspects() == 0


class _HungSyncClassifier:
    """Synchronous adapter that blocks until the test releases it."""

    def __init__(self) -> None:
        """Start with empty call counts and a closed gate."""
        self.input_calls = 0
        self.output_calls = 0
        self.started = threading.Event()
        self.finished = threading.Event()
        self._block = threading.Event()

    def release(self) -> None:
        """Unblock every waiting worker thread."""
        self._block.set()

    def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block the private executor thread after signalling start."""
        del request, check
        self.input_calls += 1
        self.started.set()
        self._block.wait(timeout=5.0)
        self.finished.set()
        return ClassifierVerdict(flagged=False)

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block the private executor thread after signalling start."""
        del completion, check
        self.output_calls += 1
        self.started.set()
        self._block.wait(timeout=5.0)
        self.finished.set()
        return ClassifierVerdict(flagged=False)


def _request() -> GatewayRequest:
    """Return one empty-content request for sync-wrapper tests."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def test_bounded_sync_classifier_rejects_overflow_instead_of_queueing() -> None:
    """A blocked worker fails later inspects immediately and does not grow the queue."""

    async def scenario() -> None:
        """Occupy one worker, hammer the wrapper, then recover after release."""
        hung = _HungSyncClassifier()
        wrapper = BoundedSyncClassifier(hung, max_workers=1)
        first = asyncio.create_task(wrapper.inspect_input(request=_request(), check=_check()))
        try:
            assert await asyncio.to_thread(hung.started.wait, 5.0)
            assert wrapper.admitted_inspects() == 1
            daemons = [
                thread
                for thread in threading.enumerate()
                if thread.name.startswith("exp-guardrail-sync")
            ]
            assert daemons
            assert all(thread.daemon for thread in daemons)
            started = time.monotonic()
            overflow = await asyncio.gather(
                *[wrapper.inspect_input(request=_request(), check=_check()) for _ in range(20)],
                return_exceptions=True,
            )
            elapsed = time.monotonic() - started
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert elapsed < 0.2
            assert hung.input_calls == 1
            assert wrapper.admitted_inspects() == 1
            assert all(isinstance(item, SyncClassifierBusyError) for item in overflow)
        finally:
            hung.release()
        assert await asyncio.to_thread(hung.finished.wait, 5.0)
        assert wrapper.admitted_inspects() == 0
        recovered = await wrapper.inspect_input(request=_request(), check=_check())
        assert recovered.flagged is False
        assert hung.input_calls == 2

    asyncio.run(scenario())


def test_bounded_sync_classifier_times_out_promptly_under_bounded_inspect() -> None:
    """The first hung inspect uses the check deadline; later ones fail without queueing."""

    async def scenario() -> None:
        """Run one admitted hung inspect, then prove overflow is immediate."""
        hung = _HungSyncClassifier()
        wrapper = BoundedSyncClassifier(hung, max_workers=1)
        inspects = BoundedInspect(max_inflight=8)
        try:
            started = time.monotonic()
            with pytest.raises(ClassifierTimeoutError):
                await inspects.run(
                    lambda: wrapper.inspect_input(request=_request(), check=_check()),
                    0.05,
                    adapter_id="sync-blocked",
                )
            first_elapsed = time.monotonic() - started
            assert await asyncio.to_thread(hung.started.wait, 5.0)
            overflow_started = time.monotonic()
            with pytest.raises(SyncClassifierBusyError):
                await wrapper.inspect_input(request=_request(), check=_check())
            overflow_elapsed = time.monotonic() - overflow_started
            assert 0.04 <= first_elapsed < 0.3
            assert overflow_elapsed < 0.2
            assert hung.input_calls == 1
            assert wrapper.admitted_inspects() == 1
        finally:
            hung.release()
        assert await asyncio.to_thread(hung.finished.wait, 5.0)

    asyncio.run(scenario())

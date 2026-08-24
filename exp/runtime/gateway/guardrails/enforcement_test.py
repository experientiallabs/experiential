"""Policy-chain, timeout, fail-closed, and tool-call enforcement tests."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.guardrails.bounded import BoundedInspect
from exp.runtime.gateway.guardrails.classifiers import (
    BoundedSyncClassifier,
    ClassifierRegistry,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.client import DirectClassifierClient, InspectingClassifier
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


class _Clock:
    """Monotonic clock that tests can advance between classifier calls."""

    def __init__(self) -> None:
        """Start at a fixed instant."""
        self.now = 100.0

    def __call__(self) -> float:
        """Return the current test instant."""
        return self.now


class _RaisingClassifier(ScriptedClassifier):
    """Adapter that fails every inspection."""

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del request, check
        self.input_calls += 1
        raise RuntimeError("classifier unavailable")

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del completion, check
        self.output_calls += 1
        raise RuntimeError("classifier unavailable")


class _HungAsyncClassifier:
    """Adapter that never returns unless the inspect is cancelled."""

    def __init__(self) -> None:
        """Start with empty call counts."""
        self.input_calls = 0
        self.output_calls = 0

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Wait until cancelled."""
        del request, check
        self.input_calls += 1
        await asyncio.Event().wait()
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Wait until cancelled."""
        del completion, check
        self.output_calls += 1
        await asyncio.Event().wait()
        return ClassifierVerdict(flagged=False)


class _BlockingBeforeAwaitClassifier:
    """Adapter that blocks the inspect thread before its first await."""

    def __init__(self) -> None:
        """Start with empty call counts and a hold that never releases by default."""
        self.input_calls = 0
        self.output_calls = 0
        self.entered = threading.Event()
        self.hold = threading.Event()

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block before yielding so a shared event loop cannot time out."""
        del request, check
        self.input_calls += 1
        self.entered.set()
        self.hold.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block before yielding on the output path."""
        del completion, check
        self.output_calls += 1
        self.entered.set()
        self.hold.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)


class _HungSyncClassifier:
    """Leftover synchronous adapter that blocks until the test releases it."""

    def __init__(self) -> None:
        """Start with empty call counts and a closed gate."""
        self.input_calls = 0
        self.output_calls = 0
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
        """Block the private executor thread."""
        del request, check
        self.input_calls += 1
        self._block.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block the private executor thread."""
        del completion, check
        self.output_calls += 1
        self._block.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)


def _awaited[T](coro: Coroutine[object, object, T]) -> T:
    """Run one enforcement coroutine on a private loop."""
    return asyncio.run(coro)


async def _wait_hold(hold: threading.Event, *, timeout: float = 5.0) -> None:
    """Wait for a test to release an abandoned inspect, then give up."""
    deadline = time.monotonic() + timeout
    while not hold.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)


def _check(
    check_id: str,
    *,
    stage: GuardrailCheckStage = GuardrailCheckStage.INPUT,
    action: GuardrailAction = GuardrailAction.BLOCK,
    adapter_id: str = "scripted",
    timeout_ms: int = 200,
) -> GuardrailCheck:
    """Build one valid check."""
    return GuardrailCheck(
        check_id=check_id,
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=stage,
        action=action,
        timeout_ms=timeout_ms,
        adapter_id=adapter_id,
    )


def _engine[C: InspectingClassifier](
    *,
    classifier: C,
    checks: tuple[GuardrailCheck, ...],
    protected: bool = False,
    clock: _Clock | None = None,
    max_request_bytes: int = 1_048_576,
    max_response_bytes: int = 1_048_576,
) -> tuple[GuardrailEngine, C]:
    """Compose one engine over a single identity policy."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=protected,
        checks=checks,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"scripted": classifier})),
        monotonic=(clock or _Clock()),
    )
    return engine, classifier


def _request(*contents: str) -> GatewayRequest:
    """Build one chat request from user message contents."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=tuple(GatewayMessage(role="user", content=item) for item in contents),
    )


def test_input_chain_runs_once_and_can_transform_the_request() -> None:
    """A modify action replaces messages for every later consumer of the request."""
    replacement = (GatewayMessage(role="user", content="redacted"),)
    engine, classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=_request("original"),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages == replacement
    assert classifier.input_calls == 1
    assert engine.input_invocations == 1


def test_input_block_is_terminal_and_content_free() -> None:
    """A block action raises a sanitized failure without request text."""
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True)),
        checks=(_check("input-one"),),
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("secret-prompt"),
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.failure_class is GatewayFailureClass.GUARDRAIL
    assert raised.value.failure.failover_eligible is False
    assert "secret-prompt" not in raised.value.failure.safe_message


def test_protected_identity_fail_closes_on_adapter_error() -> None:
    """Protected identities treat classifier uncertainty as a terminal error."""
    engine, classifier = _engine(
        classifier=_RaisingClassifier(),
        checks=(_check("input-one"),),
        protected=True,
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.safe_details["action"] == "error"
    assert classifier.input_calls == 1


def test_unprotected_identity_skips_a_failed_check() -> None:
    """Non-protected identities continue after an uncertain check."""
    engine, _classifier = _engine(
        classifier=_RaisingClassifier(),
        checks=(_check("input-one"),),
        protected=False,
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=_request("hello"),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages[0].content == "hello"


def test_expired_deadline_fail_closes_for_protected_identities() -> None:
    """A check that starts after the request deadline is an error."""
    clock = _Clock()
    clock.now = 200.0
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        protected=True,
        clock=clock,
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )


def test_output_modify_never_rewrites_tool_call_arguments() -> None:
    """Unsafe tool-call arguments are blocked instead of rewritten."""
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            output_verdict=ClassifierVerdict(flagged=True, replacement_text="safe")
        ),
        checks=(
            _check(
                "output-one",
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.MODIFY,
            ),
        ),
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None
    completion = GuardrailCompletion(
        text="call a tool",
        tool_calls=(GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"x"}'),),
    )

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_output(
                policy=policy,
                completion=completion,
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.safe_details["action"] == "block"


def test_oversized_payload_is_a_terminal_error() -> None:
    """Byte bounds fail closed without calling a classifier."""
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        max_request_bytes=4,
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("too-large"),
                deadline_monotonic=200.0,
            )
        )

    assert classifier.input_calls == 0
    assert engine.policy_for("organization-one", "identity-two") is None
    assert engine.policy_for("organization-two", "identity-one") is None


def test_oversized_tool_schema_is_rejected_without_calling_the_adapter() -> None:
    """Tool descriptions and schemas count toward the request subject bound."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                description="x" * 200,
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string", "description": "y" * 200}},
                },
            ),
        ),
    )
    subject_bytes = request_content_bytes(request)
    assert subject_bytes > 200
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        max_request_bytes=subject_bytes - 1,
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=request,
                deadline_monotonic=200.0,
            )
        )

    assert classifier.input_calls == 0


def test_blocking_classifier_times_out_without_waiting_for_return() -> None:
    """A hung inspect fails closed at the check timeout, not after the wait."""
    hung = _HungAsyncClassifier()
    engine, _classifier = _engine(
        classifier=hung,
        checks=(_check("input-one", timeout_ms=50),),
        protected=True,
        clock=_Clock(),
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    started = time.monotonic()
    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert raised.value.failure.safe_details["action"] == "error"
    assert hung.input_calls == 1


def test_blocking_classifier_leaves_the_event_loop_free() -> None:
    """A hung inspect is cancelled on the loop without stalling other tasks."""

    async def scenario() -> None:
        """Run one timed-out inspect alongside a short sleep."""
        engine, _classifier = _engine(
            classifier=_HungAsyncClassifier(),
            checks=(_check("input-one", timeout_ms=80),),
            protected=True,
            clock=_Clock(),
        )
        policy = engine.policy_for("organization-one", "identity-one")
        assert policy is not None
        progressed = False

        async def marker() -> None:
            """Flip after a delay shorter than the inspect hang."""
            nonlocal progressed
            await asyncio.sleep(0.02)
            progressed = True

        task = asyncio.create_task(marker())
        with pytest.raises(GuardrailRejected):
            await engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        await task
        assert progressed

    asyncio.run(scenario())


def test_blocking_before_await_classifier_times_out_without_freezing_enforcement() -> None:
    """A classifier that blocks before yielding still fail-closes at the check timeout."""
    blocked = _BlockingBeforeAwaitClassifier()
    engine, _classifier = _engine(
        classifier=blocked,
        checks=(_check("input-one", timeout_ms=50),),
        protected=True,
        clock=_Clock(),
    )
    policy = engine.policy_for("organization-one", "identity-one")
    assert policy is not None

    try:
        started = time.monotonic()
        with pytest.raises(GuardrailRejected) as raised:
            _awaited(
                engine.enforce_input(
                    policy=policy,
                    request=_request("hello"),
                    deadline_monotonic=200.0,
                )
            )
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert raised.value.failure.safe_details["action"] == "error"
        assert blocked.entered.wait(timeout=1.0)
        assert blocked.input_calls == 1
    finally:
        blocked.hold.set()


def test_repeated_blocked_adapter_does_not_starve_a_healthy_adapter() -> None:
    """Timeouts on one adapter leave another adapter free to succeed promptly."""

    async def scenario() -> None:
        """Fail a hung adapter past the inflight cap, then inspect a healthy one."""
        hung = _HungAsyncClassifier()
        healthy = ScriptedClassifier()
        hung_policy = GuardrailPolicy(
            policy_id="blocked-policy",
            organization_id="organization-one",
            identity_id="identity-blocked",
            protected=True,
            checks=(_check("blocked-input", adapter_id="blocked", timeout_ms=40),),
        )
        healthy_policy = GuardrailPolicy(
            policy_id="healthy-policy",
            organization_id="organization-one",
            identity_id="identity-healthy",
            checks=(_check("healthy-input", adapter_id="healthy", timeout_ms=200),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((hung_policy, healthy_policy)),
            client=DirectClassifierClient(
                ClassifierRegistry({"blocked": hung, "healthy": healthy})
            ),
            monotonic=time.monotonic,
            inspects=BoundedInspect(max_inflight=4),
        )

        for _ in range(8):
            with pytest.raises(GuardrailRejected):
                await engine.enforce_input(
                    policy=hung_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )

        started = time.monotonic()
        result = await engine.enforce_input(
            policy=healthy_policy,
            request=_request("hello"),
            deadline_monotonic=time.monotonic() + 30,
        )
        elapsed = time.monotonic() - started

        assert result.messages[0].content == "hello"
        assert healthy.input_calls == 1
        assert hung.input_calls >= 1
        assert elapsed < 0.2

    asyncio.run(scenario())


def test_hung_sync_compat_adapter_cannot_starve_healthy_async_adapters() -> None:
    """A blocked leftover sync wrapper keeps its private workers, not async slots."""

    async def scenario() -> None:
        """Time out a hung sync wrapper, then succeed on a healthy async adapter."""
        hung_inner = _HungSyncClassifier()
        hung = BoundedSyncClassifier(hung_inner, max_workers=2)
        healthy = ScriptedClassifier()
        hung_policy = GuardrailPolicy(
            policy_id="sync-blocked-policy",
            organization_id="organization-one",
            identity_id="identity-sync",
            protected=True,
            checks=(_check("sync-input", adapter_id="sync-blocked", timeout_ms=40),),
        )
        healthy_policy = GuardrailPolicy(
            policy_id="async-healthy-policy",
            organization_id="organization-one",
            identity_id="identity-async",
            checks=(_check("async-input", adapter_id="async-healthy", timeout_ms=200),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((hung_policy, healthy_policy)),
            client=DirectClassifierClient(
                ClassifierRegistry({"sync-blocked": hung, "async-healthy": healthy})
            ),
            monotonic=time.monotonic,
            inspects=BoundedInspect(max_inflight=2),
        )
        try:
            for _ in range(4):
                with pytest.raises(GuardrailRejected):
                    await engine.enforce_input(
                        policy=hung_policy,
                        request=_request("hello"),
                        deadline_monotonic=time.monotonic() + 30,
                    )

            started = time.monotonic()
            result = await engine.enforce_input(
                policy=healthy_policy,
                request=_request("hello"),
                deadline_monotonic=time.monotonic() + 30,
            )
            elapsed = time.monotonic() - started

            assert result.messages[0].content == "hello"
            assert healthy.input_calls == 1
            assert elapsed < 0.2
        finally:
            hung_inner.release()

    asyncio.run(scenario())


class _CancelSwallowingClassifier:
    """Adapter that swallows cancellation and waits until the test releases it."""

    def __init__(self) -> None:
        """Start with an empty call count and a closed hold."""
        self.input_calls = 0
        self.output_calls = 0
        self._hold = threading.Event()

    def release(self) -> None:
        """Allow every abandoned inspect to finish."""
        self._hold.set()

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Ignore cancellation and wait for teardown."""
        del request, check
        self.input_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _wait_hold(self._hold)
            return ClassifierVerdict(flagged=False)
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Ignore cancellation and wait for teardown."""
        del completion, check
        self.output_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _wait_hold(self._hold)
            return ClassifierVerdict(flagged=False)
        return ClassifierVerdict(flagged=False)


def test_cancel_swallowing_adapter_fail_closes_without_starving_others() -> None:
    """Protected requests return at the timeout while another adapter stays healthy."""

    async def scenario() -> None:
        """Quarantine a cancel-swallowing adapter, then inspect a healthy one."""
        rogue = _CancelSwallowingClassifier()
        healthy = ScriptedClassifier()
        inspects = BoundedInspect(max_inflight=2)
        rogue_policy = GuardrailPolicy(
            policy_id="rogue-policy",
            organization_id="organization-one",
            identity_id="identity-rogue",
            protected=True,
            checks=(_check("rogue-input", adapter_id="rogue", timeout_ms=40),),
        )
        healthy_policy = GuardrailPolicy(
            policy_id="healthy-policy",
            organization_id="organization-one",
            identity_id="identity-healthy",
            checks=(_check("healthy-input", adapter_id="healthy", timeout_ms=200),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((rogue_policy, healthy_policy)),
            client=DirectClassifierClient(ClassifierRegistry({"rogue": rogue, "healthy": healthy})),
            monotonic=time.monotonic,
            inspects=inspects,
        )
        try:
            started = time.monotonic()
            with pytest.raises(GuardrailRejected) as raised:
                await engine.enforce_input(
                    policy=rogue_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )
            assert time.monotonic() - started < 0.5
            assert raised.value.failure.safe_details["action"] == "error"
            assert inspects.detached_inspect_count() == 1
            assert rogue.input_calls == 1

            started = time.monotonic()
            with pytest.raises(GuardrailRejected):
                await engine.enforce_input(
                    policy=rogue_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )
            assert time.monotonic() - started < 0.2
            assert inspects.detached_inspect_count() == 1
            assert rogue.input_calls == 1

            started = time.monotonic()
            result = await engine.enforce_input(
                policy=healthy_policy,
                request=_request("hello"),
                deadline_monotonic=time.monotonic() + 30,
            )
            assert time.monotonic() - started < 0.2
            assert result.messages[0].content == "hello"
            assert healthy.input_calls == 1
        finally:
            rogue.release()
            deadline = asyncio.get_running_loop().time() + 1.0
            while inspects.detached_inspect_count() != 0:
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.02)
            assert inspects.detached_inspect_count() == 0

    asyncio.run(scenario())

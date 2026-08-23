"""Policy-chain, timeout, fail-closed, and tool-call enforcement tests."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
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

    def inspect_input(self, *, request: GatewayRequest, check: GuardrailCheck) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del request, check
        self.input_calls += 1
        raise RuntimeError("classifier unavailable")

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del completion, check
        self.output_calls += 1
        raise RuntimeError("classifier unavailable")


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
        capability=GuardrailCapabilityKind.PROMPT_INJECTION,
        stage=stage,
        action=action,
        timeout_ms=timeout_ms,
        adapter_id=adapter_id,
    )


def _engine(
    *,
    classifier: ScriptedClassifier,
    checks: tuple[GuardrailCheck, ...],
    protected: bool = False,
    clock: _Clock | None = None,
    max_request_bytes: int = 1_048_576,
    max_response_bytes: int = 1_048_576,
) -> tuple[GuardrailEngine, ScriptedClassifier]:
    """Compose one engine over a single identity policy."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        identity_id="identity-one",
        protected=protected,
        checks=checks,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore({"identity-one": policy}),
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
    policy = engine.policy_for("identity-one")
    assert policy is not None

    result = engine.enforce_input(
        policy=policy,
        request=_request("original"),
        deadline_monotonic=200.0,
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
    policy = engine.policy_for("identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        engine.enforce_input(
            policy=policy,
            request=_request("secret-prompt"),
            deadline_monotonic=200.0,
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
    policy = engine.policy_for("identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        engine.enforce_input(
            policy=policy,
            request=_request("hello"),
            deadline_monotonic=200.0,
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
    policy = engine.policy_for("identity-one")
    assert policy is not None

    result = engine.enforce_input(
        policy=policy,
        request=_request("hello"),
        deadline_monotonic=200.0,
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
    policy = engine.policy_for("identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        engine.enforce_input(
            policy=policy,
            request=_request("hello"),
            deadline_monotonic=200.0,
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
    policy = engine.policy_for("identity-one")
    assert policy is not None
    completion = GuardrailCompletion(
        text="call a tool",
        tool_calls=(GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"x"}'),),
    )

    with pytest.raises(GuardrailRejected) as raised:
        engine.enforce_output(
            policy=policy,
            completion=completion,
            deadline_monotonic=200.0,
        )

    assert raised.value.failure.safe_details["action"] == "block"


def test_oversized_payload_is_a_terminal_error() -> None:
    """Byte bounds fail closed without calling a classifier."""
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        max_request_bytes=4,
    )
    policy = engine.policy_for("identity-one")
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        engine.enforce_input(
            policy=policy,
            request=_request("too-large"),
            deadline_monotonic=200.0,
        )

    assert classifier.input_calls == 0
    assert engine.policy_for("identity-two") is None

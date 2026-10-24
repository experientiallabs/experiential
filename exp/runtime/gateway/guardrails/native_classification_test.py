"""Observing decision-model checks that the data plane classifies itself."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.bounded import start_on_native_loop
from exp.runtime.gateway.guardrails.classification import (
    DecisionModelAdapterDocument,
    DecisionModelClassifier,
)
from exp.runtime.gateway.guardrails.classification_test import _document
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassificationResult,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailOutcome,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native_classification import (
    NativeObservation,
    PendingObservations,
)
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


def _check(adapter_id: str) -> GuardrailCheck:
    """One observing input check bound to an adapter."""
    return GuardrailCheck(
        check_id=f"{adapter_id}-input",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=1000,
        adapter_id=adapter_id,
    )


class _Host(GuardrailEngine):
    """Keep every observation and classification the engine hands its host."""

    def __init__(self, *checks: GuardrailCheck) -> None:
        """Observe one policy over a native classifier and a python-only adapter."""
        self.policies = tuple(
            GuardrailPolicy(
                policy_id=check.check_id, mode="observe", protected=True, checks=(check,)
            )
            for check in checks
        )
        self.observed: list[tuple[str | None, GuardrailOutcome]] = []
        self.classified: list[tuple[str, ClassificationResult]] = []
        self.native = DecisionModelClassifier(
            DecisionModelAdapterDocument.model_validate(_document())
        )
        self.scripted = ScriptedClassifier()
        super().__init__(
            store=MappingGuardrailStore(self.policies),
            client=DirectClassifierClient(
                ClassifierRegistry({"request-classifier": self.native, "scripted": self.scripted})
            ),
            monotonic=time.monotonic,
        )

    def _record_observation(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        del policy, latency_seconds
        self.observed.append((None if check is None else check.check_id, outcome))

    def _record_classification(
        self, policy: GuardrailPolicy, check: GuardrailCheck, result: ClassificationResult
    ) -> None:
        del policy
        self.classified.append((check.check_id, result))


def _authorization() -> AuthorizationSnapshot:
    """One direct-target admission."""
    return AuthorizationSnapshot(
        request_id="req-observed",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="model-alias",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="0" * 64,
        canonical_request_sha256="0" * 64,
        deadline_monotonic=time.monotonic() + 30,
    )


def _request(text: str, *, tool_loop: bool = False) -> GatewayRequest:
    """One chat request; ``tool_loop`` adds an agent turn after it, which the window leaves out."""
    later = (GatewayMessage(role="assistant", content="Working on it."),) if tool_loop else ()
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=text), *later),
    )


def _wait(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait for the off-path observation recorder to deliver."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def _answer(flagged: list[str]) -> JsonObject:
    """One data-plane answer as the native classifier reports it."""
    return {
        "probabilities": {
            "overrides_instructions": 0.95,
            "unrelated_task": 0.1,
            "unrelated_domain": 0.1,
        },
        "scores": {"prompt_injection": 0.95, "off_topic": 0.1},
        "flagged": list(flagged),
        "cached": False,
        "latency_ms": 412.0,
    }


def test_a_native_plane_receives_each_distinct_window_and_python_never_classifies() -> None:
    """Request versions with one window become one entry; the report records it once."""
    engine = _Host(_check("request-classifier"))
    try:
        session = engine.open_request(
            _authorization(), deadline_monotonic=time.monotonic() + 30, native_data_plane=True
        )
        assert session is not None
        original, stripped = _request("Hello", tool_loop=True), _request("Hello")
        session.inspect_input(original, defer=True)
        session.prepare_dispatch((original, stripped), overlap=True)

        plan = session.native_observation_plan("req-observed")

        assert plan == [
            {"index": 0, "handle": engine.native.native_handle, "window": "[latest user]\nHello"}
        ]
        assert engine.classifier_calls == 0
        engine.record_native_observations(
            "req-observed",
            [{"index": 0, "answer": _answer(["prompt_injection"])}, {"index": 7, "error": "x"}],
        )
        engine.record_native_observations(
            "req-observed", [{"index": 0, "answer": _answer(["prompt_injection"])}]
        )
        _wait(lambda: bool(engine.observed))
    finally:
        engine.close()

    assert engine.observed == [("request-classifier-input", GuardrailOutcome.FLAGGED)]
    ((check_id, result),) = engine.classified
    assert check_id == "request-classifier-input"
    assert result.flagged == ("prompt_injection",)
    assert result.configuration_digest == engine.native.configuration_digest
    assert result.latency_seconds == 0.412


def test_errors_skips_and_text_free_requests_are_recorded_as_observations() -> None:
    """A data-plane error or skip is uncertainty; a request without text is allowed uncalled."""
    engine = _Host(_check("request-classifier"))
    image_only = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", provider_native_item={"type": "item"}),),
    )
    try:
        for request_id, request in (("req-a", _request("Hello")), ("req-b", image_only)):
            session = engine.open_request(
                _authorization(), deadline_monotonic=time.monotonic() + 30, native_data_plane=True
            )
            assert session is not None
            session.inspect_input(request)
            plan = session.native_observation_plan(request_id)
            assert len(plan) == (1 if request_id == "req-a" else 0)
        engine.record_native_observations(
            "req-a", [{"index": 0, "error": "decision model timed out"}]
        )
        _wait(lambda: len(engine.observed) == 2)
    finally:
        engine.close()

    assert sorted(engine.observed) == [
        ("request-classifier-input", GuardrailOutcome.ALLOW),
        ("request-classifier-input", GuardrailOutcome.TIMEOUT),
    ]
    assert engine.classified == []


def test_python_only_policies_and_python_planes_keep_the_engine_observation() -> None:
    """A policy with any non-native check, or a session outside the native plane, inspects here."""
    # Engine observations never wait for the callback loop's first start; a
    # running worker has already started it.
    start_on_native_loop(asyncio.sleep(0)).result(timeout=2)
    engine = _Host(_check("scripted"), _check("request-classifier"))
    try:
        mixed = engine.open_request(
            _authorization(), deadline_monotonic=time.monotonic() + 30, native_data_plane=True
        )
        plain = engine.open_request(_authorization(), deadline_monotonic=time.monotonic() + 30)
        assert mixed is not None and plain is not None
        mixed.inspect_input(_request("Hello"))
        native_plan = mixed.native_observation_plan("req-mixed")
        plain.inspect_input(_request("Hi"), defer=True)

        assert [entry["index"] for entry in native_plan] == [0]
        assert plain.native_observation_plan("req-plain") == []
        _wait(lambda: engine.scripted.input_calls >= 2 and engine.classifier_calls >= 3)
    finally:
        engine.close()

    # The scripted policy inspects in both sessions; the native policy only in the plain one.
    assert engine.scripted.input_calls == 2


def test_pending_reports_are_bounded_and_taken_once() -> None:
    """The oldest outstanding request is forgotten beyond the bound; a report is used once."""
    check = _check("request-classifier")
    observation = NativeObservation(
        policy=GuardrailPolicy(policy_id="p", mode="observe", protected=True, checks=(check,)),
        check=check,
        adapter_id="request-classifier",
        configuration_digest="d",
        handle=1,
    )
    pending = PendingObservations(max_requests=2)
    for request_id in ("first", "second", "third"):
        pending.expect(request_id, (observation,))

    assert (pending.take("first"), pending.evicted) == ((), 1)
    assert pending.take("third") == (observation,)
    assert pending.take("third") == ()

"""One lifecycle governs platform and identity checks, approval, and release."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import (
    ClassifierRegistry,
    KeywordClassifier,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.deterministic import compile_native_detectors
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import validate_guardrail_engine
from exp.runtime.gateway.guardrails.native_test import _authorization
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.subjects import observation_subject_bytes
from exp.runtime.gateway.native_capture import capture_unavailable_failure
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine
from exp.runtime.gateway.tool_contracts import GatewayProviderNativeTool


def _request(text: str = "fixture") -> GatewayRequest:
    """Make a normalized synthetic input."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=text),),
    )


def _session(engine: GuardrailEngine) -> GuardrailSession:
    """Freeze this request's policy set and absolute deadline."""
    session = engine.open_request(_authorization(), deadline_monotonic=time.monotonic() + 3)
    assert session is not None
    return session


def _await(session: GuardrailSession) -> None:
    """Wait with a finite test deadline for the nonblocking native status contract."""
    deadline = time.monotonic() + 2
    while session.input_decision()["action"] == "pending" and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.input_decision()["action"] != "pending"


def _observer_policy() -> GuardrailPolicy:
    """Select one bounded input observer shared by the admission preparation tests."""
    return GuardrailPolicy(
        policy_id="observer",
        mode="observe",
        protected=True,
        checks=(
            GuardrailCheck(
                check_id="input",
                adapter_id="criminal",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=1000,
            ),
        ),
    )


@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_explicit_required_release_failure_survives_policy_mode_and_later_cancel(mode: str) -> None:
    """Optional observation cannot hide a host capture failure or its charge waiver."""
    policy = _observer_policy().model_copy(update={"mode": mode})
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": ScriptedClassifier()})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        session.inspect_input(_request())
        assert session.input_decision() == {"action": "allow"}
        session.cancel(capture_unavailable_failure())
        session.cancel()
        decision = session.input_decision()
        assert decision["action"] == "error"
        failure = session.settlement_failure()
        assert failure is not None
        assert failure.safe_details["code"] == "capture_unavailable"
        assert failure.safe_details["input_guardrail_denied"] is True
        assert decision["failure"] == failure.model_dump(mode="json")
        with pytest.raises(GuardrailRejected) as inspected:
            session.inspect_input(_request())
        assert inspected.value.failure == failure
        with pytest.raises(GuardrailRejected) as dispatched:
            session.prepare_dispatch((_request(),), overlap=True)
        assert dispatched.value.failure == failure
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_closed_observer_remains_optional_without_accepting_new_subjects(
    defer: bool, overlap: bool
) -> None:
    """Late host callbacks preserve delivery and billing without restarting observation."""
    engine = GuardrailEngine(
        store=MappingGuardrailStore((_observer_policy(),)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": ScriptedClassifier()})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        session.cancel()
        request = _request("late callback")
        with patch.object(engine, "observe_inputs") as observe:
            assert session.inspect_input(request, defer=defer) is request
            session.prepare_dispatch((request,), overlap=overlap)
            observe.assert_not_called()
        assert session.input_decision() == {"action": "allow"}
        assert session.settlement_failure() is None
        completion = GuardrailCompletion(text="still deliverable")
        assert session.inspect_output(completion) is completion
        assert not session.input_pending
    finally:
        engine.close(timeout_seconds=2)


def test_observing_policies_share_one_complete_subject_encoding() -> None:
    """Three policies reuse one request-path encoding while copies still deduplicate inspection."""
    policies = tuple(
        _observer_policy().model_copy(update={"policy_id": f"observer-{index}"})
        for index in range(3)
    )
    classifier = KeywordClassifier(("blocked",))
    engine = GuardrailEngine(
        store=MappingGuardrailStore(policies),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    session = _session(engine)
    request = _request("visible")
    try:
        with patch(
            "exp.runtime.gateway.guardrails.enforcement.observation_subject_bytes",
            wraps=observation_subject_bytes,
        ) as encoded:
            assert session.inspect_input(request) is request
            assert encoded.call_count == 1
            copied = request.model_copy(deep=True)
            assert session.inspect_input(copied) is copied
            assert encoded.call_count == 2
    finally:
        engine.close(timeout_seconds=2)
    assert classifier.input_calls == 3


@pytest.mark.parametrize("unavailable", ["expired", "closed", "oversized", "full"])
def test_rejected_observation_admission_does_not_serialize_the_request(unavailable: str) -> None:
    """Oversized optional subjects never allocate JSON, including after other admission closes."""
    classifier = _Classifier("allow")
    policy = _observer_policy().model_copy(update={"max_request_bytes": 8192})
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
        max_observations=1,
    )
    session = _session(engine)
    try:
        if unavailable == "expired":
            session.deadline_monotonic = time.monotonic() - 1
        elif unavailable == "closed":
            engine.close(timeout_seconds=0)
        elif unavailable == "full":
            session.inspect_input(_request("first"))
            assert classifier.started.wait(2)
        request = _request("x" * 100_000)
        with patch(
            "exp.runtime.gateway.guardrails.enforcement.observation_subject_bytes",
            side_effect=AssertionError("rejected observer serialized customer input"),
        ):
            assert session.inspect_input(request) is request
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)
    assert len(classifier.requests) == (1 if unavailable == "full" else 0)


@pytest.mark.parametrize("scope", ["platform", "identity"])
@pytest.mark.parametrize("outcome", ["allow", "block", "error"])
def test_every_scope_uses_the_same_parallel_approval_and_settlement(
    scope: str, outcome: str
) -> None:
    """Moving a check between scopes cannot change its runtime, failure, or refund semantics."""
    classifier = _Classifier(outcome)
    template = _engine(classifier).policies_for("organization-one", "identity-one")[0]
    policy = (
        template
        if scope == "identity"
        else template.model_copy(update={"organization_id": None, "identity_id": None})
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    session = _session(engine)
    assert session.can_overlap_input(_request())
    session.prepare_dispatch((_request(),), overlap=True)
    try:
        assert classifier.started.wait(2)
        assert session.input_decision() == {"action": "pending"}
        classifier.release.set()
        _await(session)
        failure = session.settlement_failure()
        if outcome == "allow":
            assert failure is None
        else:
            assert failure is not None
            assert failure.safe_details["input_guardrail_denied"] is True
            assert failure.failure_class.value == (
                "guardrail" if outcome == "block" else "unavailable"
            )
            assert "private-detector" not in json.dumps(session.input_decision())
        assert classifier.requests == [_request()]
    finally:
        classifier.release.set()
        session.cancel()


@pytest.mark.parametrize("kind", ["provider_native", "provider_server", "tier"])
def test_effectful_requests_use_the_same_session_before_dispatch(kind: str) -> None:
    """Provider-executed actions and tier pricing do not overlap unapproved input."""
    session = _session(_engine(_Classifier("allow")))
    request = _request()
    assert session.can_overlap_input(request)
    if kind == "provider_native":
        request = request.model_copy(
            update={
                "provider_native_tools": (
                    GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),
                )
            }
        )
    elif kind == "provider_server":
        request = request.model_copy(
            update={"provider_server_tools": ({"type": "web_search", "name": "web_search"},)}
        )
    else:
        request = request.model_copy(update={"service_tier": "priority"})
    assert not session.can_overlap_input(request)


def test_each_distinct_context_is_inspected_once_under_the_frozen_policy() -> None:
    """Exact repeated subjects share work only within this request."""
    classifier = _Classifier("allow")
    classifier.release.set()
    session = _session(_engine(classifier))
    subjects = (_request("original"), _request("rewritten"), _request("recovered plaintext"))
    session.prepare_dispatch((*subjects, subjects[1].model_copy()), overlap=True)
    _await(session)
    assert session.input_decision() == {"action": "allow"}
    assert classifier.requests == list(subjects)


def test_one_request_cannot_inherit_another_requests_approval() -> None:
    """Equal text and policy do not share mutable approval state between tenants."""
    first, second = _Classifier("allow"), _Classifier("allow")
    first.release.set()
    a, b = _session(_engine(first)), _session(_engine(second))
    a.prepare_dispatch((_request(),), overlap=True)
    b.prepare_dispatch((_request(),), overlap=True)
    try:
        _await(a)
        assert a.input_decision() == {"action": "allow"}
        assert b.input_decision() == {"action": "pending"}
        failure = b.settlement_failure()
        assert failure is not None and failure.safe_details["input_guardrail_denied"] is True
        assert b.input_decision()["action"] == "error"
    finally:
        second.release.set()
        b.cancel()


def test_deadline_and_disconnect_cancel_the_session() -> None:
    """Neither an expired poll nor an abandoned request may produce an allow decision."""
    for expired in (False, True):
        session = _session(_engine(_Classifier("allow")))
        future: Future[None] = Future()
        session._input = future
        if expired:
            session.deadline_monotonic = time.monotonic() - 1
            assert session.input_decision()["action"] == "error"
        failure = session.settlement_failure()
        assert failure is not None
        assert failure.safe_details["input_guardrail_denied"] is True
        assert future.cancelled()


def test_empty_identity_assignment_cannot_override_platform_checks() -> None:
    """Policy resolution appends identity scope instead of replacing operator authority."""
    platform = _engine(_Classifier("block")).policies_for("organization-one", "identity-one")[0]
    platform = platform.model_copy(update={"organization_id": None, "identity_id": None})
    empty = GuardrailPolicy(
        policy_id="identity-empty", organization_id="organization-one", identity_id="identity-one"
    )
    store = MappingGuardrailStore((empty, platform))
    for org, identity in [("organization-one", "identity-one"), ("other-org", "other-key")]:
        policies = store.policies_for(org, identity)
        assert policies[0].policy_id == platform.policy_id
        assert policies[0].organization_id == org and policies[0].identity_id == identity
        assert policies[0].checks == platform.checks


def test_scope_configuration_is_strict() -> None:
    """Partial scopes and unprotected platform authority fail configuration validation."""
    with pytest.raises(ValidationError, match="both organization_id"):
        GuardrailPolicy(policy_id="partial", organization_id="org", protected=True)
    with pytest.raises(ValidationError, match="must be protected"):
        GuardrailPolicy(policy_id="unprotected-platform")


@pytest.mark.parametrize("marker", [None, True, "3", 0, 1, 2, 4])
def test_mismatched_native_contract_fails_at_startup(marker: object) -> None:
    """There is no package-version compatibility path that can omit an approval gate."""
    with patch(
        "exp.runtime.gateway.guardrails.native.importlib.import_module",
        return_value=SimpleNamespace(GUARDRAIL_CONTRACT_VERSION=marker),
    ):
        with pytest.raises(ValueError, match="GUARDRAIL_CONTRACT_VERSION=3"):
            validate_guardrail_engine(_engine(_Classifier("allow")))


def test_replay_revision_covers_identity_policies_and_execution_settings() -> None:
    """Policy identity alone cannot authorize a replay after its actual checks change."""
    classifier = _Classifier("allow")
    original = _engine(classifier)
    policy = original.policies_for("organization-one", "identity-one")[0]
    for changes in [
        {"revision": "changed"},
        {"input_execution": "before_dispatch"},
        {"checks": (policy.checks[0].model_copy(update={"timeout_ms": 5}),)},
    ]:
        engine = GuardrailEngine(
            store=MappingGuardrailStore((policy.model_copy(update=changes),)),
            client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
            monotonic=time.monotonic,
        )
        assert engine.revision_for(_authorization()) != original.revision_for(_authorization())
    changed_detector = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
        deterministic_specifications={"criminal": '{"patterns":["changed-rule"]}'},
    )
    assert changed_detector.revision_for(_authorization()) != original.revision_for(
        _authorization()
    )


def test_output_check_uses_the_same_session_after_parallel_input_approval() -> None:
    """Input execution mode never bypasses configured output checks or adds an output check."""
    check = GuardrailCheck(
        check_id="output",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=100,
        adapter_id="deny",
    )
    detector = ScriptedClassifier(output_verdict=ClassifierVerdict(flagged=True))
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="output", protected=True, checks=(check,), input_execution="parallel"
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"deny": detector})),
        monotonic=time.monotonic,
    )
    session = _session(engine)
    assert not session.can_overlap_input(_request())
    with pytest.raises(GuardrailRejected):
        session.inspect_output(GuardrailCompletion(text="synthetic"))
    assert detector.input_calls == 0 and detector.output_calls == 1


def test_uninspected_or_closed_sessions_never_authorize_release() -> None:
    """No future is not equivalent to approval; cancellation is terminal even after an allow."""
    detector = _Classifier("allow")
    detector.release.set()
    session = _session(_engine(detector))
    assert session.input_decision()["action"] == "error"
    session.inspect_input(_request())
    assert session.input_decision() == {"action": "allow"}
    session.cancel()
    assert session.input_decision()["action"] == "error"
    with pytest.raises(GuardrailRejected):
        session.inspect_input(_request())
    with pytest.raises(GuardrailRejected):
        session.prepare_dispatch((_request(),), overlap=True)
    with pytest.raises(GuardrailRejected):
        session.release_output_segment(pending="text", final=True, settled_bytes=0)


def test_output_rewrites_are_rechecked_by_all_scopes() -> None:
    """An identity classifier cannot add content that passed no platform check."""
    common = GuardrailCheck(
        check_id="content",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.OUTPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=100,
        adapter_id="block",
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(policy_id="platform", protected=True, checks=(common,)),
                GuardrailPolicy(
                    policy_id="identity",
                    organization_id="organization-one",
                    identity_id="identity-one",
                    checks=(
                        common.model_copy(
                            update={"action": GuardrailAction.MODIFY, "adapter_id": "rewrite"}
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(
            ClassifierRegistry(
                {
                    "block": KeywordClassifier(("forbidden",)),
                    "rewrite": ScriptedClassifier(
                        output_verdict=ClassifierVerdict(flagged=True, replacement_text="forbidden")
                    ),
                }
            )
        ),
        monotonic=time.monotonic,
    )
    with pytest.raises(GuardrailRejected) as exc:
        _session(engine).inspect_output(GuardrailCompletion(text="allowed"))
    assert exc.value.failure.failure_class.value == "guardrail"


def test_policy_changes_cannot_remove_an_admitted_check() -> None:
    """A policy reload affects future admissions, never a request already being inspected."""
    detector = _Classifier("block")
    detector.release.set()
    policies = _engine(detector).policies_for("organization-one", "identity-one")

    class ReloadableStore:
        """Expose a host policy reload without changing the policy store contract."""

        def policies_for(
            self, organization_id: str, identity_id: str
        ) -> tuple[GuardrailPolicy, ...]:
            """Resolve the current operator snapshot for this admission."""
            del organization_id, identity_id
            return policies

    engine = GuardrailEngine(
        store=ReloadableStore(),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": detector})),
        monotonic=time.monotonic,
    )
    session = _session(engine)
    policies = ()
    assert engine.open_request(_authorization(), deadline_monotonic=time.monotonic() + 3) is None
    with pytest.raises(GuardrailRejected):
        session.inspect_input(_request())


@pytest.mark.parametrize("outcome", ["block", "error"])
def test_serial_input_rejection_retains_its_verdict_for_settlement(outcome: str) -> None:
    """A rejected search expansion settles the actual verdict and cannot be retried in place."""
    detector = _Classifier(outcome)
    detector.release.set()
    session = _session(_engine(detector))
    with pytest.raises(GuardrailRejected) as exc:
        session.inspect_input(_request())
    failure = session.settlement_failure()
    assert failure is not None
    assert failure.failure_class == exc.value.failure.failure_class
    assert failure.safe_message == exc.value.failure.safe_message
    assert failure.safe_details["input_guardrail_denied"] is True
    with pytest.raises(GuardrailRejected):
        session.prepare_dispatch((_request("another subject"),), overlap=True)
    assert len(detector.requests) == 1


@pytest.mark.parametrize("approved_before_deadline", [True, False])
def test_completed_future_settlement_uses_verdict_time_not_poll_time(
    approved_before_deadline: bool,
) -> None:
    """A late allow cannot charge the caller; a timely allow stays valid during late settlement."""
    session = _session(_engine(_Classifier("allow")))
    session.deadline_monotonic = time.monotonic() - 1
    future: Future[None] = Future()
    session._input = future
    session._input_approved_at = session.deadline_monotonic + (
        -0.01 if approved_before_deadline else 0.01
    )
    future.set_result(None)
    failure = session.settlement_failure()
    assert session.input_decision()["action"] == ("allow" if approved_before_deadline else "error")
    if approved_before_deadline:
        assert failure is None
    else:
        assert failure is not None
        assert failure.safe_details["input_guardrail_denied"] is True


@pytest.mark.parametrize("stage", ["input", "output"])
@pytest.mark.parametrize("separate_scopes", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("modify_first", [False, True])
def test_policy_checks_preserve_authored_order_before_final_validation(
    stage: str, separate_scopes: bool, native: bool, modify_first: bool
) -> None:
    """An earlier redaction feeds later blockers; a later redaction cannot bypass a blocker."""
    checks = [
        {
            "check_id": "redact",
            "capability": "pii",
            "stage": stage,
            "action": "modify",
            "adapter_id": "redact",
            "timeout_ms": 500,
        },
        {
            "check_id": "block",
            "capability": "content_safety",
            "stage": stage,
            "action": "block",
            "adapter_id": "block",
            "timeout_ms": 500,
        },
    ]
    if not modify_first:
        checks.reverse()
    policies = [{"policy_id": "platform", "protected": True, "checks": checks}]
    if separate_scopes:
        policies[0]["checks"] = checks[:1]
        policies.append(
            {
                "policy_id": "identity",
                "organization_id": "organization-one",
                "identity_id": "identity-one",
                "protected": True,
                "checks": checks[1:],
            }
        )
    engine = engine_from_document(
        {
            "adapters": [
                {
                    "kind": "regex",
                    "adapter_id": "redact",
                    "patterns": ["secret"],
                    "replacement": "safe",
                },
                {"kind": "regex", "adapter_id": "block", "patterns": ["secret"]},
            ],
            "policies": policies,
        }
    )
    session = _session(engine)
    if native:
        session.detectors = compile_native_detectors(engine.deterministic_specifications)
    if not modify_first:
        with pytest.raises(GuardrailRejected):
            if stage == "input":
                session.inspect_input(_request("secret"))
            else:
                session.inspect_output(GuardrailCompletion(text="secret"))
    elif stage == "input":
        assert session.inspect_input(_request("secret")) == _request("safe")
    else:
        assert session.output_plan() is None
        assert session.inspect_output(GuardrailCompletion(text="secret")).text == "safe"
    expected_calls = 2 if modify_first else 1
    assert engine.classifier_calls == (0 if native and stage == "input" else expected_calls)


@pytest.mark.parametrize("stage", ["input", "output"])
@pytest.mark.parametrize("separate_scopes", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("original", ["secret", "innocent"])
def test_later_modifier_cannot_undo_an_earlier_redaction(
    stage: str, separate_scopes: bool, native: bool, original: str
) -> None:
    """Even a rewrite back to the original must satisfy every earlier modifier independently."""
    checks = [
        {
            "check_id": adapter,
            "capability": "pii",
            "stage": stage,
            "action": "modify",
            "adapter_id": adapter,
            "timeout_ms": 500,
        }
        for adapter in ("redact", "undo")
    ]
    policies = [{"policy_id": "platform", "protected": True, "checks": checks}]
    if separate_scopes:
        policies[0]["checks"] = checks[:1]
        policies.append(
            {
                "policy_id": "identity",
                "organization_id": "organization-one",
                "identity_id": "identity-one",
                "protected": True,
                "checks": checks[1:],
            }
        )
    engine = engine_from_document(
        {
            "adapters": [
                {
                    "kind": "regex",
                    "adapter_id": "redact",
                    "patterns": ["secret"],
                    "replacement": "safe",
                },
                {
                    "kind": "regex",
                    "adapter_id": "undo",
                    "patterns": ["safe|innocent"],
                    "replacement": "secret",
                },
            ],
            "policies": policies,
        }
    )
    session = _session(engine)
    if native:
        session.detectors = compile_native_detectors(engine.deterministic_specifications)
    if stage == "input":
        with pytest.raises(GuardrailRejected):
            session.inspect_input(_request(original))
    else:
        assert session.output_plan() is None
        with pytest.raises(GuardrailRejected):
            session.inspect_output(GuardrailCompletion(text=original))

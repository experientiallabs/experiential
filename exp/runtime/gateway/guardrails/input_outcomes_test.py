"""Input evidence survives late classifier results without inventing coverage."""

import time
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
from unittest.mock import Mock, patch

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.gateway.guardrails.classification_window import (
    ClassificationWindow,
    classification_window,
)
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailOutcome,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.deterministic import (
    compile_native_detectors,
    native_input_request,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.input_outcomes import (
    InputOutcomes,
    _window_digest,
    latest_user_window_digest,
    record_input_outcome,
)
from exp.runtime.gateway.guardrails.session_test import _request, _session
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier


def _policy() -> GuardrailPolicy:
    """Build an optional operator-owned input policy with an immutable revision."""
    return GuardrailPolicy(
        policy_id="safety",
        revision="policy-v1",
        protected=True,
        input_execution="parallel",
        fail_open=True,
        checks=(
            GuardrailCheck(
                check_id="input",
                adapter_id="classifier",
                stage=GuardrailCheckStage.INPUT,
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                action=GuardrailAction.BLOCK,
                timeout_ms=1000,
            ),
        ),
    )


def _modification_engine(second_action: GuardrailAction | None = None) -> GuardrailEngine:
    """Build a redactor with an optional later blocker or undoing modifier.

    Args:
        second_action: Optional action applied to the first check's replacement text.

    Returns:
        An engine supporting the Python and deterministic execution paths.
    """
    checks: list[JsonObject] = [
        {
            "check_id": "redact",
            "adapter_id": "redact",
            "capability": "pii",
            "stage": "input",
            "action": "modify",
            "timeout_ms": 500,
        }
    ]
    if second_action is not None:
        checks.append(
            {
                "check_id": "second",
                "adapter_id": "second",
                "capability": "pii",
                "stage": "input",
                "action": second_action.value,
                "timeout_ms": 500,
            }
        )
    return engine_from_document(
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
                    "adapter_id": "second",
                    "patterns": ["safe"],
                    "replacement": "secret",
                },
            ],
            "policies": [{"policy_id": "modification", "protected": True, "checks": checks}],
        }
    )


@pytest.mark.parametrize(
    ("observed", "outcome", "reason"),
    [
        (GuardrailOutcome.ALLOW, "allow", None),
        (GuardrailOutcome.TIMEOUT, "missed", "deadline_expired"),
        (GuardrailOutcome.UNAVAILABLE, "missed", "classifier_error"),
        (GuardrailOutcome.UNCERTAIN, "missed", "result_unavailable"),
        (GuardrailOutcome.UNSUPPORTED, "unsupported", None),
    ],
)
def test_typed_results(observed: GuardrailOutcome, outcome: str, reason: str | None) -> None:
    """Only actual execution misses carry a recovery reason."""
    policy, tracker = _policy(), InputOutcomes()
    tracker.start(policy)
    with tracker.collect():
        record_input_outcome(policy, policy.checks[0], observed)
    (result,) = tracker.freeze((policy,), blocked=False)
    assert (result.outcome, result.recovery_reason) == (outcome, reason)
    assert result.policy_revision == "policy-v1"
    assert result.stage == "input"


def test_native_expiry_wins_and_terminal_snapshot_ignores_late_allow() -> None:
    """A ready classifier result is insufficient after native delivery has expired."""
    policy, tracker = _policy(), InputOutcomes()
    tracker.start(policy)
    tracker.record(policy, policy.checks[0], GuardrailOutcome.FLAGGED)
    tracker.expire(policy)
    sealed = tracker.freeze((policy,), blocked=False)
    tracker.record(policy, policy.checks[0], GuardrailOutcome.ALLOW)
    assert tracker.freeze((policy,), blocked=False) is sealed
    assert sealed[0].recovery_reason == "deadline_expired"


def test_blocked_request_cannot_authorize_recovery() -> None:
    """An actual input rejection wins over misses in other checks."""
    policy, tracker = _policy(), InputOutcomes()
    tracker.start(policy)
    tracker.expire(policy)
    (result,) = tracker.freeze((policy,), blocked=True)
    assert (result.outcome, result.recovery_reason) == ("block", None)


@pytest.mark.parametrize("blocked", [False, True])
def test_excluded_and_unstarted_checks_are_not_misses(blocked: bool) -> None:
    """Missing execution is not an implicit grant to enforce later."""
    policy, tracker = _policy(), InputOutcomes()
    assert tracker.freeze((), blocked=blocked) == ()
    (result,) = InputOutcomes().freeze((policy,), blocked=blocked)
    assert (result.outcome, result.recovery_reason) == ("not_started", None)


def test_disagreeing_input_subjects_cannot_claim_one_recoverable_window() -> None:
    """Two different admitted user windows cannot be recovered from one capture."""
    policy, tracker = _policy(), InputOutcomes()
    tracker.start(policy, _request("first"))
    tracker.start(policy, _request("second"))
    tracker.expire(policy)
    (result,) = tracker.freeze((policy,), blocked=False)
    assert result.outcome == "missed"
    assert result.latest_user_window_sha256 is None


@pytest.mark.parametrize("action", [GuardrailAction.ALLOW, GuardrailAction.MODIFY])
@pytest.mark.parametrize(
    "later", [GuardrailOutcome.TIMEOUT, GuardrailOutcome.UNAVAILABLE, GuardrailOutcome.UNSUPPORTED]
)
def test_covered_flagged_subject_does_not_hide_a_later_miss(
    action: GuardrailAction, later: GuardrailOutcome
) -> None:
    """Only genuine misses stay sticky, including successful non-blocking rewrites."""
    policy, tracker = _policy(), InputOutcomes()
    check = policy.checks[0].model_copy(update={"action": action})
    policy = policy.model_copy(update={"checks": (check,)})
    tracker.start(policy, _request())
    tracker.record(policy, check, GuardrailOutcome.FLAGGED)
    tracker.record(policy, check, later)
    (result,) = tracker.freeze((policy,), blocked=False)
    assert result.outcome == ("unsupported" if later is GuardrailOutcome.UNSUPPORTED else "missed")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("denial", [None, "sibling", "provider_bound"])
def test_successful_modification_is_covered(native: bool, denial: str | None) -> None:
    """Applied rewrites retain coverage despite later check or request-level rejection."""
    engine = _modification_engine(GuardrailAction.BLOCK if denial == "sibling" else None)
    try:
        session = _session(engine)
        session.enable_input_outcomes()
        if native:
            session.detectors = compile_native_detectors(engine.deterministic_specifications)
        if denial is None:
            assert session.inspect_input(_request("secret")) == _request("safe")
        else:
            with pytest.raises(GuardrailRejected):
                if denial == "provider_bound":
                    session.prepare_dispatch((_request("secret"),), overlap=False)
                else:
                    session.inspect_input(_request("secret"))
        outcomes = session.input_outcomes()
        expected = ("allow", "block") if denial == "sibling" else ("allow",)
        assert tuple(result.outcome for result in outcomes) == expected
        assert all(result.recovery_reason is None for result in outcomes)
        assert (engine.classifier_calls == 0) is native
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failure", ["application", "final_validation"])
def test_failed_modification_is_not_covered(native: bool, failure: str) -> None:
    """A failed application or final rewrite validation cannot claim a completed allow."""
    engine = _modification_engine(GuardrailAction.MODIFY if failure == "final_validation" else None)
    request = _request("secret")
    if failure == "application":
        request = request.model_copy(
            update={
                "messages": (
                    GatewayMessage(
                        role="user",
                        content="secret",
                        provider_text_blocks=(
                            {
                                "type": "text",
                                "text": "secret",
                                "cache_control": {"type": "ephemeral"},
                            },
                        ),
                    ),
                )
            }
        )
    try:
        session = _session(engine)
        session.enable_input_outcomes()
        if native:
            session.detectors = compile_native_detectors(engine.deterministic_specifications)
        with pytest.raises(GuardrailRejected):
            session.inspect_input(request)
        outcomes = session.input_outcomes()
        expected = ("block", "allow") if failure == "final_validation" else ("block",)
        assert tuple(result.outcome for result in outcomes) == expected
        assert all(result.recovery_reason is None for result in outcomes)
        assert (engine.classifier_calls == 0) is native
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("blocked", [False, True])
def test_unapplied_modification_does_not_claim_coverage(blocked: bool) -> None:
    """A flagged verdict alone does not prove its replacement was successfully applied."""
    policy, tracker = _policy(), InputOutcomes()
    check = policy.checks[0].model_copy(update={"action": GuardrailAction.MODIFY})
    policy = policy.model_copy(update={"checks": (check,)})
    tracker.start(policy, _request())
    tracker.record(policy, check, GuardrailOutcome.FLAGGED)
    (result,) = tracker.freeze((policy,), blocked=blocked)
    expected = ("block", None) if blocked else ("missed", "result_unavailable")
    assert (result.outcome, result.recovery_reason) == expected


def test_queued_but_unstarted_checks_do_not_authorize_recovery() -> None:
    """A fast provider failure can settle before the scheduled input task starts."""
    classifier, future = _Classifier("allow"), Future()

    def queued(coroutine: Coroutine[object, object, None]) -> Future[None]:
        """Model a scheduled task without executing or leaking its coroutine."""
        coroutine.close()
        return future

    engine = GuardrailEngine(
        store=MappingGuardrailStore((_policy(),)),
        client=DirectClassifierClient(ClassifierRegistry({"classifier": classifier})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        session.enable_input_outcomes()
        with patch(
            "exp.runtime.gateway.guardrails.session.start_on_native_loop", side_effect=queued
        ):
            session.prepare_dispatch((_request(),), overlap=True)
        (result,) = session.input_outcomes()
        assert result.outcome == "not_started"
        assert result.recovery_reason is None
        assert result.latest_user_window_sha256 is not None
        assert not classifier.started.is_set()
    finally:
        future.cancel()
        engine.close(timeout_seconds=2)


def test_digest_uses_the_classifier_latest_nonblank_user_selection() -> None:
    """Trailing nontext user input must not replace the online classifier's chosen turn."""
    request = _request("earlier input")
    request = request.model_copy(
        update={"messages": (*request.messages, GatewayMessage(role="user", content=" "))}
    )
    expected = classification_window(
        request, ClassificationWindow(system_chars=0, latest_user_chars=30000, prior_messages=0)
    )
    assert latest_user_window_digest(request) == sha256(expected.encode()).hexdigest()


def test_policy_wide_unsupported_input_is_not_a_recoverable_miss() -> None:
    """An oversized fail-open policy fans its unsupported result out to its input checks."""
    policy, tracker = _policy(), InputOutcomes()
    tracker.start(policy, _request())
    with tracker.collect():
        record_input_outcome(policy, None, GuardrailOutcome.UNSUPPORTED)
    (result,) = tracker.freeze((policy,), blocked=False)
    assert result.outcome == "unsupported"
    assert result.recovery_reason is None


@pytest.mark.parametrize("failed", [False, True])
def test_native_uncertainty_preserves_missing_coverage(*, failed: bool) -> None:
    """Unprotected deterministic checks that skip uncertainty never manufacture ALLOW."""
    policy = _policy().model_copy(update={"protected": False, "fail_open": False})
    tracker = InputOutcomes()
    tracker.start(policy, _request())
    detector = Mock()
    detector.matches.return_value = False
    detector.redact.side_effect = ValueError("synthetic detector failure")
    with tracker.collect():
        native_input_request(
            policy,
            {"classifier": detector},
            _request(),
            monotonic=lambda: 1.0,
            deadline_monotonic=2.0 if failed else 0.0,
        )
    (result,) = tracker.freeze((policy,), blocked=False)
    assert result.outcome == "missed"
    assert result.recovery_reason == ("classifier_error" if failed else "deadline_expired")


def test_parallel_request_contexts_do_not_cross() -> None:
    """Two workers using identical policies retain their own result."""
    policy = _policy()

    def run(observed: GuardrailOutcome) -> str:
        """Collect and seal one independent request on its own worker."""
        tracker = InputOutcomes()
        tracker.start(policy)
        with tracker.collect():
            record_input_outcome(policy, policy.checks[0], observed)
        return tracker.freeze((policy,), blocked=False)[0].outcome

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(run, (GuardrailOutcome.ALLOW, GuardrailOutcome.TIMEOUT))) == [
            "allow",
            "missed",
        ]


@pytest.mark.parametrize(
    ("verdict", "expected", "reason"),
    [
        ("allow", "allow", None),
        ("block", "block", None),
        ("error", "missed", "classifier_error"),
        ("timeout", "missed", "deadline_expired"),
        ("unsupported", "unsupported", None),
    ],
)
def test_executed_input_outcome_survives_the_session(
    verdict: str, expected: str, reason: str | None
) -> None:
    """Exercise real execution, fail-open and rejection before the terminal snapshot."""
    classifier = _Classifier(verdict)
    if verdict != "timeout":
        classifier.release.set()
    policy = _policy().model_copy(update={"input_execution": "blocking", "fail_open": True})
    policy = policy.model_copy(
        update={
            "checks": (
                policy.checks[0].model_copy(update={"adapter_id": "criminal", "timeout_ms": 20}),
            )
        }
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        session.enable_input_outcomes()
        if verdict == "block":
            with pytest.raises(GuardrailRejected):
                session.inspect_input(_request())
        else:
            session.inspect_input(_request())
        (result,) = session.input_outcomes()
        assert (result.outcome, result.recovery_reason) == (expected, reason)
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize(
    ("first_verdict", "first_action", "second_verdict", "expected"),
    [
        pytest.param(
            "allow",
            GuardrailAction.BLOCK,
            "unsupported",
            ("allow", "unsupported"),
            id="allow-before-unsupported",
        ),
        pytest.param(
            "allow",
            GuardrailAction.BLOCK,
            "block",
            ("allow", "block"),
            id="allow-before-rejection",
        ),
        pytest.param(
            "block",
            GuardrailAction.ALLOW,
            "unsupported",
            ("allow", "unsupported"),
            id="flagged-allow-before-unsupported",
        ),
        pytest.param(
            "unsupported",
            GuardrailAction.BLOCK,
            "allow",
            ("unsupported", "not_started"),
            id="fail-closed-unsupported",
        ),
        pytest.param(
            "block",
            GuardrailAction.MODIFY,
            "allow",
            ("block", "not_started"),
            id="invalid-modification",
        ),
    ],
)
def test_rejected_session_preserves_known_check_outcomes(
    first_verdict: str,
    first_action: GuardrailAction,
    second_verdict: str,
    expected: tuple[str, str],
) -> None:
    """Retain completed check results while denied requests grant no recovery authority."""
    first, second = _Classifier(first_verdict), _Classifier(second_verdict)
    first.release.set()
    second.release.set()
    policy = _policy()
    policy = policy.model_copy(
        update={
            "input_execution": "blocking",
            "fail_open": False,
            "checks": (
                policy.checks[0].model_copy(
                    update={"check_id": "first", "adapter_id": "first", "action": first_action}
                ),
                policy.checks[0].model_copy(update={"check_id": "second", "adapter_id": "second"}),
            ),
        }
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"first": first, "second": second})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        session.enable_input_outcomes()
        with pytest.raises(GuardrailRejected):
            session.inspect_input(_request())
        outcomes = session.input_outcomes()
        assert tuple(result.outcome for result in outcomes) == expected
        assert all(result.recovery_reason is None for result in outcomes)
    finally:
        engine.close(timeout_seconds=2)


def test_unchanged_subject_is_hashed_once_across_checks_and_execution() -> None:
    """Pre-admission and execution reuse a bounded digest without retaining content."""
    policy, tracker, request = _policy(), InputOutcomes(), _request()
    with patch(
        "exp.runtime.gateway.guardrails.input_outcomes._window_digest",
        wraps=_window_digest,
    ) as digest:
        tracker.admit(policy, request)
        tracker.start(policy, request)
        tracker.admit(policy.model_copy(update={"policy_id": "another"}), request)
        assert digest.call_count == 1
        tracker.admit(policy, _request("different"))
        assert digest.call_count == 2
    assert tracker.freeze((policy,), blocked=False)[0].latest_user_window_sha256 is None


def test_sessions_without_consumers_do_not_hash_input() -> None:
    """Legacy hosts pay no digest or collector cost for the optional extension."""
    classifier = _Classifier("allow")
    classifier.release.set()
    engine = GuardrailEngine(
        store=MappingGuardrailStore((_policy(),)),
        client=DirectClassifierClient(ClassifierRegistry({"classifier": classifier})),
        monotonic=time.monotonic,
    )
    try:
        session = _session(engine)
        with patch("exp.runtime.gateway.guardrails.input_outcomes._window_digest") as digest:
            session.inspect_input(_request())
            assert session.input_outcomes() == ()
            digest.assert_not_called()
    finally:
        engine.close(timeout_seconds=2)


def test_mutated_native_input_never_reuses_a_stale_digest() -> None:
    """A shallowly frozen request can contain mutable native caller input."""
    policy, tracker = _policy(), InputOutcomes()
    message = GatewayMessage(
        role="user",
        provider_native_item={
            "type": "function_call_output",
            "call_id": "unknown",
            "output": "first",
        },
    )
    request = _request().model_copy(update={"messages": (message,)})
    tracker.admit(policy, request)
    assert message.provider_native_item is not None
    message.provider_native_item["output"] = "second"
    tracker.start(policy, request)
    tracker.expire(policy)
    assert tracker.freeze((policy,), blocked=False)[0].latest_user_window_sha256 is None


@pytest.mark.parametrize(
    "miss", [GuardrailOutcome.TIMEOUT, GuardrailOutcome.UNAVAILABLE, GuardrailOutcome.UNCERTAIN]
)
@pytest.mark.parametrize("unsupported_first", [False, True])
@pytest.mark.parametrize("native_expiry", [False, True])
def test_unsupported_subject_prevents_recovery_in_either_order(
    miss: GuardrailOutcome, unsupported_first: bool, native_expiry: bool
) -> None:
    """A shared user-window digest cannot turn an uninspectable subject into a recovery grant."""
    policy, tracker = _policy(), InputOutcomes()
    results = (
        (GuardrailOutcome.UNSUPPORTED, miss)
        if unsupported_first
        else (miss, GuardrailOutcome.UNSUPPORTED)
    )
    for result in results:
        tracker.start(policy, _request("same latest user window"))
        tracker.record(policy, policy.checks[0], result)
    if native_expiry:
        tracker.expire(policy)
    (record,) = tracker.freeze((policy,), blocked=False)
    assert record.latest_user_window_sha256 is not None
    assert record.outcome == "unsupported"
    assert record.recovery_reason is None


@pytest.mark.parametrize("text", ["", "a" * 2047, "b" * 2048, "c" * 30000, "🚀" * 30001])
def test_window_digest_matches_standard_sha256_across_update_boundaries(text: str) -> None:
    """Bounded hashing preserves the exact classifier window, including multibyte text."""
    request = _request(text)
    window = ClassificationWindow(system_chars=0, latest_user_chars=30000, prior_messages=0)
    expected = sha256(classification_window(request, window).encode("utf-8")).hexdigest()
    assert latest_user_window_digest(request) == expected

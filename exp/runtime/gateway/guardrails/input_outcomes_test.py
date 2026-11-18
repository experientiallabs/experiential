"""Input evidence survives late classifier results without inventing coverage."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailOutcome,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.input_outcomes import InputOutcomes, record_input_outcome
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


def test_excluded_and_unstarted_checks_are_not_misses() -> None:
    """Missing execution is not an implicit grant to enforce later."""
    policy, tracker = _policy(), InputOutcomes()
    assert tracker.freeze((), blocked=False) == ()
    (result,) = InputOutcomes().freeze((policy,), blocked=False)
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

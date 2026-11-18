"""The durable ledger extension is explicit and preserves authenticated identity."""

from unittest.mock import Mock, patch

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
from exp.runtime.gateway.guardrails.input_outcomes import InputCheckOutcome
from exp.runtime.gateway.guardrails.input_outcomes_test import _policy
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.native_accounting_test import _RecordingLedger, _retryable_failure
from exp.runtime.gateway.native_guardrail_outcomes import (
    outcome_documents,
    outcome_keyword,
    terminal_outcomes,
)
from exp.runtime.gateway.native_request_settlement import RequestSettlements


def test_documents_bind_request_ownership_and_contain_no_subject() -> None:
    """Only the request owner supplies request/org IDs at the settlement boundary."""
    policy = _policy()
    record = InputCheckOutcome(
        policy_id=policy.policy_id,
        policy_revision=policy.revision,
        check_id=policy.checks[0].check_id,
        mode="enforce",
        action=policy.checks[0].action,
        protected=True,
        outcome="missed",
        recovery_reason="deadline_expired",
    )
    entry = Mock(authorization=_authorization(), guardrails=Mock())
    entry.guardrails.input_outcomes.return_value = (record,)
    (document,) = outcome_documents(entry)
    assert document["request_id"] == entry.authorization.request_id
    assert document["organization_id"] == entry.authorization.organization_id
    assert set(document) == set(record.model_dump()) | {"request_id", "organization_id"}


def test_only_declared_ledger_extension_receives_evidence() -> None:
    """Existing ledgers receive no unsolicited keyword; consumers receive the exact records."""

    def consumes(*, input_guardrail_outcomes: tuple[JsonObject, ...]) -> None:
        """Declare the durable input-evidence capability."""

    def plain() -> None:
        """A ledger with no input-evidence consumer."""

    documents: tuple[JsonObject, ...] = ({"outcome": "allow"},)
    assert outcome_keyword(consumes, documents) == {"input_guardrail_outcomes": documents}
    assert outcome_keyword(plain, documents) == {}
    assert outcome_keyword(Mock(wraps=plain), documents) == {}


def test_terminal_retry_keeps_the_original_input_evidence() -> None:
    """A failed durable write retries the exact tuple with its original terminal event."""
    ledger = _RecordingLedger()
    settlements = RequestSettlements(ledger)
    documents: tuple[JsonObject, ...] = (
        {"outcome": "missed", "recovery_reason": "deadline_expired"},
    )
    seen: list[tuple[JsonObject, ...]] = []

    def write(
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool,
        web_search_requests: int,
        input_guardrail_outcomes: tuple[JsonObject, ...],
    ) -> bool:
        """Fail once after observing the attempted durable payload."""
        seen.append(input_guardrail_outcomes)
        if len(seen) == 1:
            raise OSError("synthetic database interruption")
        return False

    with patch.object(ledger, "finish_request", write):
        with pytest.raises(OSError):
            settlements.finish(
                _authorization(),
                GatewayFailure.model_validate(_retryable_failure()),
                input_guardrail_outcomes=documents,
            )
        settlements.retry(limit=1)
    assert seen == [documents, documents]


def test_unconsumed_or_nonfinal_evidence_is_not_frozen() -> None:
    """Legacy and intermediate terminal writes cannot trigger digest serialization."""
    entry = Mock()
    assert terminal_outcomes(lambda: None, entry) == {}
    entry.guardrails.input_outcomes.assert_not_called()

    def consumes(*, input_guardrail_outcomes: tuple[JsonObject, ...]) -> None:
        """Declare the optional extension."""

    assert terminal_outcomes(consumes, entry, finalize=False) == {}
    entry.guardrails.input_outcomes.assert_not_called()


def test_request_only_writer_without_extension_never_freezes_enabled_evidence() -> None:
    """An attempt-only consumer does not opt its unconsumed request-only writer in."""
    entry = Mock()
    settlements = RequestSettlements(_RecordingLedger())
    assert settlements.input_outcomes(entry) == ()
    entry.guardrails.input_outcomes.assert_not_called()

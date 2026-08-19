"""Adversarial tests for the routed-interaction journal and idempotency service."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RoutedCandidateSnapshot,
    Usage,
    ToolCall,
)
from wmo.common.project import ProjectPaths
from wmo.common.routing import RouterFeatureExtractor, RoutingDecision
from wmo.runtime.router.economics import (
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    routed_completion_economics,
    routed_spend_ledger,
)
from wmo.runtime.router.journal import (
    JournalClaim,
    RuntimeAcceptance,
    RuntimeAcceptedEvent,
    RuntimeAttemptFailedEvent,
    RuntimeCompletedEvent,
    RuntimeIdempotencyConflictError,
    RuntimeInteractionFailedError,
    RuntimeInteractionIdentity,
    RuntimeInteractionInProgressError,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    _completed_event,
    _failed_event,
    _interaction_identity,
)
from wmo.runtime.router.journal_service import JournaledRouterRuntime
from wmo.runtime.router.journal_spend import direct_not_incurred_operation
from wmo.runtime.router.runtime import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedModelResponse,
    RouterModelCapabilityError,
    RouterRuntime,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 13, tzinfo=UTC)


def _snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-test",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _request(content: str = "Help me") -> ModelRequest:
    return ModelRequest(messages=(ModelMessage(role="user", content=content),))


def _response() -> ModelResponse:
    return ModelResponse(
        output=AssistantAction(content="Done"),
        model=_snapshot(),
        economics=OperationEconomics(),
    )


def _economics() -> RoutedCompletionEconomics:
    """Return deterministic alias-free economics for one fake routed response."""
    return routed_completion_economics(
        (
            _operation(
                ordinal=1,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
                source=BillingSource.HOST_MANAGED,
                disposition=RoutedSpendDisposition.LOCALLY_PRICED,
                input_tokens=5,
            ),
            _operation(
                ordinal=2,
                component=RoutedProviderComponent.SELECTED_CANDIDATE,
                source=BillingSource.CUSTOMER_MANAGED,
                disposition=RoutedSpendDisposition.OBSERVED,
                input_tokens=7,
            ),
        )
    )


def _operation(
    *,
    ordinal: int,
    component: RoutedProviderComponent,
    source: BillingSource,
    disposition: RoutedSpendDisposition,
    input_tokens: int,
) -> RoutedProviderOperation:
    """Build deterministic alias-free provider operation evidence for journal tests."""
    return RoutedProviderOperation(
        operation_id=f"routed-operation-{ordinal:020x}",
        operation_ordinal=ordinal,
        component=component,
        billing_source=source,
        disposition=disposition,
        operation_count=(0 if disposition == RoutedSpendDisposition.DEFINITELY_NOT_INCURRED else 1),
        economics=OperationEconomics(
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
            cost_usd=NumericMeasurement(
                value=input_tokens / 1_000_000,
                provenance="estimated",
            ),
        ),
    )


def test_reloaded_claim_matches_request_after_raw_arguments_are_not_persisted(
    tmp_path: Path,
) -> None:
    """A reloaded retry compares durable semantics and retains fresh provider bytes.

    Args:
        tmp_path: Pytest-owned project root.
    """
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content="Look up the ticket"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-1",
                            name="lookup_ticket",
                            arguments={"priority": 1, "ticket_id": "42"},
                            raw_arguments='{ "ticket_id": "42", "priority": 1 }',
                        ),
                    )
                ),
            ),
        )
    )
    identity = _interaction_identity("support-agent", "raw-arguments-key", request, None)
    claim = _accept(journal, identity, now=_TIME)

    assert claim.status == "dispatch"
    assert claim.accepted is not None
    _reserve_candidate(journal, claim.accepted, now=_TIME)
    journal.record_failure(
        claim.accepted,
        StructuredFailure(
            code=FailureCode.TIMEOUT,
            message="provider timed out",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    reloaded = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    retried = reloaded.claim(
        _interaction_identity("support-agent", "raw-arguments-key", request, None),
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=30),
    )

    assert retried.status == "dispatch"
    assert retried.accepted is not None
    retried_call = retried.accepted.identity.request.messages[1].assistant_action
    assert retried_call is not None
    assert retried_call.tool_calls[0].raw_arguments == '{ "ticket_id": "42", "priority": 1 }'
    persisted = reloaded.read_events()
    persisted_action = (
        cast(RuntimeAcceptedEvent, persisted[-1]).identity.request.messages[1].assistant_action
    )
    assert persisted_action is not None
    assert persisted_action.tool_calls[0].raw_arguments is None
    _reserve_candidate(reloaded, retried.accepted, now=_TIME + timedelta(seconds=2))
    completed = reloaded.record_completed(
        retried.accepted,
        _response(),
        candidate_operation=_operation(
            ordinal=retried.accepted.spend.operation_count + 1,
            component=RoutedProviderComponent.SELECTED_CANDIDATE,
            source=BillingSource.CUSTOMER_MANAGED,
            disposition=RoutedSpendDisposition.OBSERVED,
            input_tokens=11,
        ),
        completed_at=_TIME + timedelta(seconds=3),
    )
    assert completed.event == "completed"
    with pytest.raises(RuntimeIdempotencyConflictError):
        reloaded.claim(
            _interaction_identity(
                "support-agent",
                "raw-arguments-key",
                request.model_copy(update={"maximum_output_tokens": 2}),
                None,
            ),
            now=_TIME + timedelta(seconds=4),
            stale_after=timedelta(seconds=30),
        )
    with pytest.raises(RuntimeIdempotencyConflictError):
        reloaded.claim(
            _interaction_identity(
                "support-agent",
                "raw-arguments-key",
                request,
                "different-caller-lineage",
            ),
            now=_TIME + timedelta(seconds=4),
            stale_after=timedelta(seconds=30),
        )


def _decision(lineage_id: str, request: ModelRequest | None = None) -> RoutingDecision:
    episode_sha256 = hashlib.sha256(lineage_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    feature = RouterFeatureExtractor().from_request(request or _request())
    material = {
        "policy_id": "router-policy-a",
        "policy_sha256": _DIGEST,
        "request_sha256": hashlib.sha256(
            feature.encode("utf-8"), usedforsecurity=False
        ).hexdigest(),
        "episode_id_sha256": episode_sha256,
        "selected_alias": "candidate-a",
        "baseline_alias": "candidate-a",
        "neighbor_count": 8,
        "paired_count": 8,
        "best_similarity": 1.0,
        "estimated_quality_difference": None,
        "uncertainty": None,
        "fallback_reason": None,
    }
    return RoutingDecision(
        decision_id=stable_id("routing-decision", material),
        **material,
    )


def _acceptance(identity: RuntimeInteractionIdentity) -> RuntimeAcceptance:
    """Build deterministic mixed-source route pins for a journal interaction."""
    decision = _decision(identity.lineage_id, identity.request)
    return RuntimeAcceptance(
        decision=decision,
        selected_alias=decision.selected_alias,
        selected_model=_snapshot(),
        router_embedding_billing_source=BillingSource.HOST_MANAGED,
        policy_input=ArtifactInput(
            artifact_id=decision.policy_id,
            sha256=decision.policy_sha256,
        ),
    )


def _accept(
    journal: RuntimeInteractionJournal,
    identity: RuntimeInteractionIdentity,
    *,
    now: datetime,
) -> JournalClaim:
    """Reserve, settle, and durably accept one deterministic initial selection."""
    embedding = BillingSourceEconomics(
        billing_source=BillingSource.HOST_MANAGED,
        economics=_operation(
            ordinal=1,
            component=RoutedProviderComponent.ROUTER_EMBEDDING,
            source=BillingSource.HOST_MANAGED,
            disposition=RoutedSpendDisposition.LOCALLY_PRICED,
            input_tokens=5,
        ).economics,
    )
    reserved = journal.reserve_selection(
        identity,
        embedding,
        now=now,
        stale_after=timedelta(seconds=1),
    )
    assert reserved.reservation is not None
    return journal.record_acceptance(
        identity,
        _acceptance(identity),
        reserved.reservation,
        _operation(
            ordinal=1,
            component=RoutedProviderComponent.ROUTER_EMBEDDING,
            source=BillingSource.HOST_MANAGED,
            disposition=RoutedSpendDisposition.LOCALLY_PRICED,
            input_tokens=5,
        ),
        accepted_at=now,
    )


def _reserve_candidate(
    journal: RuntimeInteractionJournal,
    accepted: RuntimeAcceptedEvent,
    *,
    now: datetime,
) -> None:
    """Persist one deterministic candidate reservation for a low-level journal test."""
    claim = journal.reserve_candidate(
        accepted,
        BillingSourceEconomics(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            economics=_operation(
                ordinal=accepted.spend.operation_count + 1,
                component=RoutedProviderComponent.SELECTED_CANDIDATE,
                source=BillingSource.CUSTOMER_MANAGED,
                disposition=RoutedSpendDisposition.RESERVED,
                input_tokens=11,
            ).economics,
        ),
        now=now,
    )
    assert claim.status == "dispatch"


class _FakeRuntime:
    """Small deterministic runtime with a shareable target-call counter."""

    def __init__(
        self,
        *,
        counter: Synchronized | None = None,
        delay: float = 0.0,
        selection_delay: float = 0.0,
        selection_hook: Callable[[], None] | None = None,
        completion_hook: Callable[[], None] | None = None,
        candidate_reservation_error: Exception | None = None,
        fail_once: bool = False,
    ) -> None:
        """Configure deterministic selection, completion, delay, and failure behavior.

        Args:
            counter: Optional process-shareable completion counter.
            delay: Seconds to pause each target completion.
            selection_delay: Seconds to pause each routing selection.
            selection_hook: Optional synchronous callback invoked during selection.
            completion_hook: Optional synchronous callback invoked at candidate dispatch.
            candidate_reservation_error: Optional provider-free predispatch validation failure.
            fail_once: Whether the first target completion raises a timeout.
        """
        self.policy = SimpleNamespace(
            candidates=(RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot()),)
        )
        self.counter = counter
        self.delay = delay
        self.selection_delay = selection_delay
        self.selection_hook = selection_hook
        self.completion_hook = completion_hook
        self.candidate_reservation_error = candidate_reservation_error
        self.fail_once = fail_once
        self.select_calls = 0
        self.complete_calls = 0
        self.decisions: list[RoutingDecision] = []

    def embedding_reservation(self, request: ModelRequest) -> BillingSourceEconomics:
        """Return deterministic host-managed preselection economics."""
        del request
        return BillingSourceEconomics(
            billing_source=BillingSource.HOST_MANAGED,
            economics=_operation(
                ordinal=1,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
                source=BillingSource.HOST_MANAGED,
                disposition=RoutedSpendDisposition.LOCALLY_PRICED,
                input_tokens=5,
            ).economics,
        )

    def select(self, request: ModelRequest, *, episode_id: str | None = None) -> RoutingDecision:
        """Return a deterministic decision after the configured selection delay.

        Args:
            request: Canonical request accepted by the runtime seam.
            episode_id: Required journal-derived routing lineage.

        Returns:
            Deterministic routing decision for the requested lineage.
        """
        del request
        assert episode_id is not None
        self.select_calls += 1
        if self.selection_delay:
            time.sleep(self.selection_delay)
        if self.selection_hook is not None:
            self.selection_hook()
        decision = _decision(episode_id)
        self.decisions.append(decision)
        return decision

    def selection_operation(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
        decision: RoutingDecision,
    ) -> RoutedProviderOperation:
        """Return deterministic successful embedding evidence for one selection."""
        del request, episode_id, decision
        return _operation(
            ordinal=1,
            component=RoutedProviderComponent.ROUTER_EMBEDDING,
            source=BillingSource.HOST_MANAGED,
            disposition=RoutedSpendDisposition.LOCALLY_PRICED,
            input_tokens=5,
        )

    def candidate_reservation(
        self,
        request: ModelRequest,
        *,
        episode_id: str,
        decision: RoutingDecision,
    ) -> BillingSourceEconomics:
        """Return deterministic customer-managed candidate reservation economics."""
        del request, episode_id, decision
        if self.candidate_reservation_error is not None:
            raise self.candidate_reservation_error
        return BillingSourceEconomics(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            economics=_operation(
                ordinal=2,
                component=RoutedProviderComponent.SELECTED_CANDIDATE,
                source=BillingSource.CUSTOMER_MANAGED,
                disposition=RoutedSpendDisposition.RESERVED,
                input_tokens=11,
            ).economics,
        )

    def complete(
        self,
        request: ModelRequest,
        *,
        episode_id: str | None = None,
        decision: RoutingDecision | None = None,
        provider_idempotency_key: str | None = None,
    ) -> RoutedModelResponse:
        del request, episode_id
        assert decision is not None
        assert provider_idempotency_key is not None
        self.complete_calls += 1
        if self.completion_hook is not None:
            self.completion_hook()
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("secret provider detail")
        return RoutedModelResponse(
            decision=decision,
            response=_response(),
            economics=_economics(),
        )


def _service(
    root: Path,
    runtime: _FakeRuntime,
    *,
    wait_timeout_seconds: float = 2.0,
    stale_after_seconds: float = 30.0,
) -> JournaledRouterRuntime:
    journal = RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent"))
    return JournaledRouterRuntime(
        cast(RouterRuntime, runtime),
        journal,
        wait_timeout_seconds=wait_timeout_seconds,
        stale_after_seconds=stale_after_seconds,
    )


def _find_colliding_key(
    reference_key: str,
    reference_request: ModelRequest,
    candidate_request: ModelRequest,
) -> str:
    """Find a distinct key mapped to the same bounded selection-lock stripe.

    Args:
        reference_key: Key whose process-local lock stripe must be matched.
        reference_request: Request paired with the reference key.
        candidate_request: Request paired with generated candidate keys.

    Returns:
        Deterministic candidate key using the same selection-lock stripe.

    Raises:
        AssertionError: No collision is found within the bounded search.
    """
    reference_identity = _interaction_identity(
        "support-agent", reference_key, reference_request, None
    )
    reference_stripe = int(reference_identity.interaction_id.rsplit("-", maxsplit=1)[-1], 16) % 64
    for index in range(1_000):
        candidate = f"colliding-key-{index}"
        candidate_identity = _interaction_identity(
            "support-agent",
            candidate,
            candidate_request,
            None,
        )
        candidate_stripe = (
            int(candidate_identity.interaction_id.rsplit("-", maxsplit=1)[-1], 16) % 64
        )
        if candidate_stripe == reference_stripe:
            return candidate
    raise AssertionError("bounded key search did not find a selection-lock collision")


def test_completed_interaction_replays_after_restart_without_provider_or_selection(
    tmp_path: Path,
) -> None:
    """A completed key returns its canonical response without any new runtime work."""
    first_runtime = _FakeRuntime()
    first = _service(tmp_path / ".wmo", first_runtime)
    raw_key = "sk-this-is-a-fake-key-material-123456789"
    initial = first.complete(_request(), idempotency_key=raw_key)

    restarted_runtime = _FakeRuntime()
    restarted = _service(tmp_path / ".wmo", restarted_runtime)
    replayed = restarted.complete(_request(), idempotency_key=raw_key)

    assert replayed == initial
    assert first_runtime.select_calls == first_runtime.complete_calls == 1
    assert restarted_runtime.select_calls == restarted_runtime.complete_calls == 0
    payload = restarted.journal.path.read_text(encoding="utf-8")
    assert raw_key not in payload
    assert [event.event for event in restarted.journal.read_events()] == [
        "accepted",
        "completed",
    ]


def test_embedding_and_candidate_reservations_are_durable_before_provider_calls(
    tmp_path: Path,
) -> None:
    """Persist exact mixed-source reservations before either provider-backed operation."""
    root = tmp_path / ".wmo"
    journal = RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent"))
    observed: list[tuple[RoutedProviderComponent, BillingSource, RoutedSpendDisposition]] = []

    def inspect_embedding_reservation() -> None:
        """Capture the checkpoint already durable when selection enters the embedder."""
        event = journal.read_spend_events()[-1]
        observed.append(
            (event.operation.component, event.operation.billing_source, event.operation.disposition)
        )

    def inspect_candidate_reservation() -> None:
        """Capture the checkpoint already durable when candidate dispatch begins."""
        event = journal.read_spend_events()[-1]
        observed.append(
            (event.operation.component, event.operation.billing_source, event.operation.disposition)
        )

    runtime = _FakeRuntime(
        selection_hook=inspect_embedding_reservation,
        completion_hook=inspect_candidate_reservation,
    )
    result = JournaledRouterRuntime(cast(RouterRuntime, runtime), journal).complete(
        _request(), idempotency_key="reservation-order-key"
    )

    assert observed == [
        (
            RoutedProviderComponent.ROUTER_EMBEDDING,
            BillingSource.HOST_MANAGED,
            RoutedSpendDisposition.RESERVED,
        ),
        (
            RoutedProviderComponent.SELECTED_CANDIDATE,
            BillingSource.CUSTOMER_MANAGED,
            RoutedSpendDisposition.RESERVED,
        ),
    ]
    assert result.economics.operation_count == 2
    assert tuple(item.billing_source for item in result.economics.by_billing_source) == (
        BillingSource.CUSTOMER_MANAGED,
        BillingSource.HOST_MANAGED,
    )
    durable_payload = journal.spend_path.read_text(encoding="utf-8")
    assert "reservation-order-key" not in durable_payload
    assert "candidate-a" not in durable_payload
    assert "host-model" not in durable_payload
    assert '"host_managed"' in durable_payload
    assert '"customer_managed"' in durable_payload


def test_stale_preselection_reservation_survives_new_process_replay(tmp_path: Path) -> None:
    """Count an abandoned embedding reservation as ambiguous before selecting again."""
    root = tmp_path / ".wmo"
    journal = RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent"))
    request = _request()
    identity = _interaction_identity("support-agent", "embedding-crash-key", request, None)
    initial_runtime = _FakeRuntime()
    initial = journal.reserve_selection(
        identity,
        initial_runtime.embedding_reservation(request),
        now=_TIME,
        stale_after=timedelta(seconds=1),
    )
    assert initial.status == "dispatch"

    restarted_runtime = _FakeRuntime()
    restarted = JournaledRouterRuntime(
        cast(RouterRuntime, restarted_runtime),
        RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent")),
        clock=lambda: _TIME + timedelta(seconds=2),
        stale_after_seconds=1,
    )
    result = restarted.complete(request, idempotency_key="embedding-crash-key")

    assert restarted_runtime.select_calls == restarted_runtime.complete_calls == 1
    assert [item.disposition for item in result.economics.operations] == [
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.OBSERVED,
    ]
    assert result.economics.operation_count == 3
    assert result.economics.router_embedding.economics.usage == Usage(
        input_tokens=10,
        output_tokens=0,
    )


def test_stale_candidate_reservation_survives_new_process_replay(tmp_path: Path) -> None:
    """Carry possible candidate spend through stale reclaim without re-embedding."""
    root = tmp_path / ".wmo"
    journal = RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent"))
    request = _request()
    identity = _interaction_identity("support-agent", "candidate-crash-key", request, None)
    accepted_claim = _accept(journal, identity, now=_TIME)
    assert accepted_claim.accepted is not None
    _reserve_candidate(journal, accepted_claim.accepted, now=_TIME)

    restarted_runtime = _FakeRuntime()
    restarted = JournaledRouterRuntime(
        cast(RouterRuntime, restarted_runtime),
        RuntimeInteractionJournal(ProjectPaths(root=root, project_id="support-agent")),
        clock=lambda: _TIME + timedelta(seconds=2),
        stale_after_seconds=1,
    )
    result = restarted.complete(request, idempotency_key="candidate-crash-key")

    assert restarted_runtime.select_calls == 0
    assert restarted_runtime.complete_calls == 1
    assert [item.disposition for item in result.economics.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
        RoutedSpendDisposition.OBSERVED,
    ]
    assert result.economics.operation_count == 3
    assert result.economics.router_embedding.economics.usage == Usage(
        input_tokens=5,
        output_tokens=0,
    )


def test_candidate_predispatch_failure_is_durably_not_incurred(tmp_path: Path) -> None:
    """Expose permanent failure spend while proving candidate dispatch never occurred."""
    runtime = _FakeRuntime(
        candidate_reservation_error=RouterModelCapabilityError("unsupported candidate")
    )
    service = _service(tmp_path / ".wmo", runtime)

    with pytest.raises(RuntimeInteractionFailedError) as initial:
        service.complete(_request(), idempotency_key="predispatch-failure-key")

    failed = cast(RuntimeAttemptFailedEvent, service.journal.read_events()[-1])
    with pytest.raises(RuntimeInteractionFailedError) as replayed:
        service.complete(_request(), idempotency_key="predispatch-failure-key")

    assert failed.failure.code == FailureCode.UNSUPPORTED
    assert initial.value.failure == replayed.value.failure == failed.failure
    assert initial.value.spend == replayed.value.spend == failed.spend
    assert not initial.value.retryable
    assert failed.spend.operation_count == 1
    assert [item.disposition for item in failed.spend.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
    ]
    assert runtime.complete_calls == 0
    assert runtime.select_calls == 1
    assert "candidate-a" not in failed.spend.model_dump_json()


def test_candidate_reservation_cannot_be_rewritten_as_not_incurred(tmp_path: Path) -> None:
    """Reject a recomputed main event that erases a durable candidate dispatch hazard."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    identity = _interaction_identity("support-agent", "reserved-candidate-key", _request(), None)
    claim = _accept(journal, identity, now=_TIME)
    assert claim.accepted is not None
    _reserve_candidate(journal, claim.accepted, now=_TIME)
    not_incurred = direct_not_incurred_operation(
        interaction_id=claim.accepted.interaction_id,
        operation_ordinal=2,
        component=RoutedProviderComponent.SELECTED_CANDIDATE,
        billing=BillingSourceEconomics(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            economics=OperationEconomics(),
        ),
    )
    forged = _failed_event(
        claim.accepted,
        StructuredFailure(
            code=FailureCode.UNSUPPORTED,
            message="forged pre-dispatch failure",
            retryable=False,
            attribution=FailureAttribution.MODEL,
        ),
        spend=routed_spend_ledger((*claim.accepted.spend.operations, not_incurred)),
        ordinal=2,
        failed_at=_TIME + timedelta(seconds=1),
    )
    journal.path.write_bytes(
        canonical_json_bytes(claim.accepted) + b"\n" + canonical_json_bytes(forged) + b"\n"
    )

    with pytest.raises(RuntimeJournalError, match="not-incurred candidate"):
        journal.read_events()


@pytest.mark.parametrize("reopen", [False, True])
def test_retryable_candidate_predispatch_failure_advances_operation_ordinal(
    tmp_path: Path,
    *,
    reopen: bool,
) -> None:
    """Allocate retry spend after a proven non-dispatch in one or a restarted process."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    journal = RuntimeInteractionJournal(paths)
    identity = _interaction_identity("support-agent", "predispatch-retry-key", _request(), None)
    first = _accept(journal, identity, now=_TIME)
    assert first.accepted is not None
    failed = journal.record_failure(
        first.accepted,
        StructuredFailure(
            code=FailureCode.TIMEOUT,
            message="provider-free predispatch timeout",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    assert isinstance(failed, RuntimeAttemptFailedEvent)
    if reopen:
        journal = RuntimeInteractionJournal(paths)
    retry = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )
    assert retry.accepted is not None
    reservation = journal.reserve_candidate(
        retry.accepted,
        BillingSourceEconomics(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            economics=_economics().selected_candidate.economics,
        ),
        now=_TIME + timedelta(seconds=2),
    )
    assert reservation.reservation is not None
    assert reservation.reservation.operation.operation_ordinal == 3
    assert (
        reservation.reservation.operation.operation_id != failed.spend.operations[-1].operation_id
    )
    completed = journal.record_completed(
        retry.accepted,
        _response(),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=3),
    )
    assert isinstance(completed, RuntimeCompletedEvent)
    assert [item.disposition for item in completed.economics.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
        RoutedSpendDisposition.OBSERVED,
    ]


def test_settled_selection_history_rejects_changed_identity_before_dispatch(
    tmp_path: Path,
) -> None:
    """Reject same-key request drift before a new reservation or selection provider call."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    journal = RuntimeInteractionJournal(paths)
    first_request = _request("Original")
    identity = _interaction_identity("support-agent", "settled-selection-key", first_request, None)
    runtime = _FakeRuntime()
    reservation = journal.reserve_selection(
        identity,
        runtime.embedding_reservation(first_request),
        now=_TIME,
        stale_after=timedelta(seconds=1),
    )
    assert reservation.reservation is not None
    journal.record_selection_failure(
        identity,
        reservation.reservation,
        failed_at=_TIME + timedelta(seconds=1),
    )
    restarted = JournaledRouterRuntime(
        cast(RouterRuntime, runtime),
        RuntimeInteractionJournal(paths),
        clock=lambda: _TIME + timedelta(seconds=2),
        stale_after_seconds=1,
    )

    with pytest.raises(RuntimeIdempotencyConflictError):
        restarted.complete(
            _request("Changed"),
            idempotency_key="settled-selection-key",
        )

    assert runtime.select_calls == runtime.complete_calls == 0
    assert len(restarted.journal.read_spend_events()) == 2


def test_concurrent_runtime_instances_select_and_dispatch_once(tmp_path: Path) -> None:
    """Serialize initial route selection across services sharing one project journal.

    Args:
        tmp_path: Pytest-owned local project root.
    """
    first_runtime = _FakeRuntime(selection_delay=0.05, delay=0.05)
    second_runtime = _FakeRuntime(selection_delay=0.05, delay=0.05)
    services = (
        _service(tmp_path / ".wmo", first_runtime),
        _service(tmp_path / ".wmo", second_runtime),
    )
    barrier = threading.Barrier(3)
    responses: list[RoutedModelResponse] = []
    errors: list[Exception] = []

    def complete(service: JournaledRouterRuntime) -> None:
        """Start together and capture one shared-key completion result.

        Args:
            service: Independently composed runtime over the shared journal.
        """
        try:
            barrier.wait()
            responses.append(service.complete(_request(), idempotency_key="shared-key"))
        except Exception as exc:  # noqa: BLE001 - capture thread failures for the main assertion
            errors.append(exc)

    threads = tuple(threading.Thread(target=complete, args=(service,)) for service in services)
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads), "shared-key handshake deadlocked"
    assert errors == []
    assert len(responses) == 2 and responses[0] == responses[1]
    assert first_runtime.select_calls + second_runtime.select_calls == 1
    assert first_runtime.complete_calls + second_runtime.complete_calls == 1
    assert [event.event for event in services[0].journal.read_events()] == [
        "accepted",
        "completed",
    ]


def test_colliding_selection_stripe_allows_nested_independent_runtime(tmp_path: Path) -> None:
    """Allow same-thread selection reentry for an unrelated colliding interaction.

    Args:
        tmp_path: Pytest-owned local project root.
    """
    root = tmp_path / ".wmo"
    outer_key = "outer-key"
    nested_key = _find_colliding_key(outer_key, _request(), _request("Nested"))
    nested_runtime = _FakeRuntime()
    nested_service = _service(root, nested_runtime)

    def complete_nested() -> None:
        """Complete one unrelated interaction through the second local service."""
        nested_service.complete(_request("Nested"), idempotency_key=nested_key)

    outer_runtime = _FakeRuntime(selection_hook=complete_nested)
    outer_service = _service(root, outer_runtime)
    errors: list[Exception] = []

    def complete_outer() -> None:
        """Capture a nested-selection failure without blocking the test process."""
        try:
            outer_service.complete(_request(), idempotency_key=outer_key)
        except Exception as exc:  # noqa: BLE001 - capture thread failures for the main assertion
            errors.append(exc)

    thread = threading.Thread(target=complete_outer, daemon=True)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "colliding selection stripe deadlocked on same-thread reentry"
    assert errors == []
    assert outer_runtime.select_calls == outer_runtime.complete_calls == 1
    assert nested_runtime.select_calls == nested_runtime.complete_calls == 1


def test_colliding_selection_stripe_obeys_bounded_wait(tmp_path: Path) -> None:
    """Bound another thread's wait for a busy selection-lock stripe.

    Args:
        tmp_path: Pytest-owned local project root.
    """
    root = tmp_path / ".wmo"
    held_key = "held-key"
    waiting_key = _find_colliding_key(held_key, _request("Held"), _request("Waiting"))
    selection_started = threading.Event()
    release_selection = threading.Event()

    def hold_selection() -> None:
        """Signal lock ownership and wait until the bounded-wait assertion finishes."""
        selection_started.set()
        assert release_selection.wait(timeout=2.0)

    held_runtime = _FakeRuntime(selection_hook=hold_selection)
    held_service = _service(root, held_runtime)
    waiting_service = _service(root, _FakeRuntime(), wait_timeout_seconds=0.05)
    errors: list[Exception] = []

    def complete_held() -> None:
        """Run the selector that temporarily owns the shared lock stripe."""
        try:
            held_service.complete(_request("Held"), idempotency_key=held_key)
        except Exception as exc:  # noqa: BLE001 - capture thread failures for the main assertion
            errors.append(exc)

    thread = threading.Thread(target=complete_held, daemon=True)
    thread.start()
    assert selection_started.wait(timeout=1.0)
    started_at = time.monotonic()
    with pytest.raises(RuntimeInteractionInProgressError, match="selecting"):
        waiting_service.complete(_request("Waiting"), idempotency_key=waiting_key)
    elapsed = time.monotonic() - started_at
    release_selection.set()
    thread.join(timeout=1.0)

    assert elapsed < 0.5
    assert not thread.is_alive()
    assert errors == []


def test_same_key_with_different_request_or_lineage_conflicts(tmp_path: Path) -> None:
    """One key cannot silently change its canonical request or conversation lineage."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="same-key", conversation_id="conversation-a")

    with pytest.raises(RuntimeIdempotencyConflictError):
        service.complete(
            _request("Different"),
            idempotency_key="same-key",
            conversation_id="conversation-a",
        )
    with pytest.raises(RuntimeIdempotencyConflictError):
        service.complete(
            _request(),
            idempotency_key="same-key",
            conversation_id="conversation-b",
        )


def test_provider_failure_has_no_target_and_retry_reuses_pinned_decision(tmp_path: Path) -> None:
    """Retryable failures are durable and never trigger a second routing selection."""
    runtime = _FakeRuntime(fail_once=True)
    service = _service(tmp_path / ".wmo", runtime)

    with pytest.raises(RuntimeInteractionFailedError) as initial:
        service.complete(_request(), idempotency_key="retry-key")
    events = service.journal.read_events()
    assert [event.event for event in events] == ["accepted", "attempt_failed"]
    failed = cast(RuntimeAttemptFailedEvent, events[-1])
    assert failed.retryable
    assert initial.value.failure == failed.failure
    assert initial.value.spend == failed.spend
    assert initial.value.retryable
    assert failed.failure.message == "routed model provider attempt failed"
    assert "response" not in failed.model_dump(mode="json")
    assert "secret provider detail" not in service.journal.path.read_text(encoding="utf-8")
    assert failed.spend.operation_count == 2
    assert [item.disposition for item in failed.spend.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
    ]

    result = service.complete(_request(), idempotency_key="retry-key")
    retried = service.journal.read_events()
    assert result.response == _response()
    assert [event.event for event in retried] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "completed",
    ]
    assert runtime.select_calls == 1
    assert result.economics.operation_count == 3
    assert [item.disposition for item in result.economics.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
        RoutedSpendDisposition.OBSERVED,
    ]
    assert tuple(item.billing_source for item in result.economics.by_billing_source) == (
        BillingSource.CUSTOMER_MANAGED,
        BillingSource.HOST_MANAGED,
    )
    assert (
        cast(RuntimeAcceptedEvent, retried[0]).acceptance.decision
        == cast(RuntimeAcceptedEvent, retried[2]).acceptance.decision
    )


def test_two_threads_with_one_key_dispatch_target_once(tmp_path: Path) -> None:
    """The journal serializes same-key threads while provider work runs outside the lock."""
    runtime = _FakeRuntime(delay=0.15)
    service = _service(tmp_path / ".wmo", runtime)
    barrier = threading.Barrier(3)
    results: list[RoutedModelResponse] = []

    def call() -> None:
        barrier.wait()
        results.append(service.complete(_request(), idempotency_key="thread-key"))

    threads = (threading.Thread(target=call), threading.Thread(target=call))
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] == results[1]
    assert runtime.complete_calls == 1


def _process_call(root: str, counter: Synchronized) -> None:
    """Complete one shared-key request in a separate process."""
    runtime = _FakeRuntime(counter=counter, delay=0.2)
    _service(Path(root), runtime, wait_timeout_seconds=3.0).complete(
        _request(), idempotency_key="process-key"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="cross-process lock probe needs fork")
def test_two_processes_with_one_key_dispatch_target_once(tmp_path: Path) -> None:
    """Independent processes observe one accepted attempt and one completed response."""
    context = multiprocessing.get_context("fork")
    counter = context.Value("i", 0)
    root = tmp_path / ".wmo"
    processes = (
        context.Process(target=_process_call, args=(str(root), counter)),
        context.Process(target=_process_call, args=(str(root), counter)),
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert counter.value == 1


def test_torn_tail_is_ignored_but_interior_corruption_fails_closed(tmp_path: Path) -> None:
    """Only a non-newline-terminated final record may be treated as a crash tear."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="tail-key")
    path = service.journal.path
    clean = path.read_bytes()
    path.write_bytes(clean + b'{"event":"accepted"')

    assert len(service.journal.read_events()) == 2
    service.complete(_request(), idempotency_key="after-torn-key")
    assert len(service.journal.read_events()) == 4

    repaired = path.read_bytes()
    path.write_bytes(repaired.splitlines(keepends=True)[0] + b"not-json\n" + repaired)
    with pytest.raises(RuntimeJournalError, match="invalid interior line"):
        service.journal.read_events()


def test_duplicate_transition_and_digest_drift_fail_closed(tmp_path: Path) -> None:
    """A valid JSON rewrite cannot bypass event IDs, digests, ordinals, or transitions."""
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="corrupt-key")
    path = service.journal.path
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(f"{lines[0]}\n{lines[0]}\n", encoding="utf-8")
    with pytest.raises(RuntimeJournalError, match="ordinals|event ID|transition"):
        service.journal.read_events()

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    corrupted = lines[1].replace('"response_sha256":"', '"response_sha256":"f')
    path.write_text(f"{lines[0]}\n{corrupted}\n", encoding="utf-8")
    with pytest.raises(RuntimeJournalError, match="invalid interior line"):
        service.journal.read_events()


def test_stale_attempt_gets_failure_then_retry_with_original_route(tmp_path: Path) -> None:
    """A stale live attempt is closed explicitly before a pinned retry is dispatched."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "stale-key", request, None)
    first = _accept(journal, identity, now=_TIME)
    assert first.accepted is not None
    _reserve_candidate(journal, first.accepted, now=_TIME)
    retry = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )

    assert first.status == retry.status == "dispatch"
    assert retry.accepted is not None
    assert retry.accepted.attempt_ordinal == 2
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
    ]


def test_original_provider_success_wins_after_concurrent_stale_takeover(tmp_path: Path) -> None:
    """A late original success commits once and prevents a replacement response."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "takeover-key", request, None)
    original = _accept(journal, identity, now=_TIME)
    assert original.accepted is not None
    _reserve_candidate(journal, original.accepted, now=_TIME)
    replacement = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )
    assert original.accepted is not None
    assert replacement.accepted is not None
    _reserve_candidate(
        journal,
        replacement.accepted,
        now=_TIME + timedelta(seconds=2),
    )

    winning = journal.record_completed(
        original.accepted,
        _response(),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=3),
    )
    replay = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=4),
        stale_after=timedelta(seconds=1),
    )

    assert replay.status == "completed"
    assert replay.completed == winning
    assert isinstance(winning, RuntimeCompletedEvent)
    assert [item.disposition for item in winning.economics.operations] == [
        RoutedSpendDisposition.LOCALLY_PRICED,
        RoutedSpendDisposition.OBSERVED,
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
    ]
    assert winning.economics.operation_count == 3
    replacement_observed = journal.record_completed(
        replacement.accepted,
        _response().model_copy(update={"output": AssistantAction(content="replacement")}),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=5),
    )
    assert replacement_observed == winning
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "completed",
    ]


def test_replacement_permanent_failure_wins_over_late_original_success(tmp_path: Path) -> None:
    """A replacement's permanent failure remains canonical after a late original success."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "permanent-key", request, None)
    original = _accept(journal, identity, now=_TIME)
    assert original.accepted is not None
    _reserve_candidate(journal, original.accepted, now=_TIME)
    replacement = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )
    assert original.accepted is not None
    assert replacement.accepted is not None
    _reserve_candidate(journal, replacement.accepted, now=_TIME + timedelta(seconds=2))
    permanent = StructuredFailure(
        code=FailureCode.PROVIDER,
        message="routed model provider attempt failed permanently",
        retryable=False,
        attribution=FailureAttribution.MODEL,
    )
    terminal = journal.record_failure(
        replacement.accepted,
        permanent,
        failed_at=_TIME + timedelta(seconds=3),
    )

    observed = journal.record_completed(
        original.accepted,
        _response(),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=4),
    )

    assert observed == terminal
    failed_claim = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=5),
        stale_after=timedelta(seconds=1),
    )
    assert failed_claim.status == "failed"
    assert [event.event for event in journal.read_events()] == [
        "accepted",
        "attempt_failed",
        "accepted",
        "attempt_failed",
    ]

    corrupted_completion = _completed_event(
        original.accepted,
        _response(),
        routed_completion_economics(
            (*original.accepted.spend.operations, _economics().operations[-1])
        ),
        ordinal=5,
        completed_at=_TIME + timedelta(seconds=4),
    )
    journal._append_unlocked(corrupted_completion)
    with pytest.raises(RuntimeJournalError, match="permanent interaction failure"):
        journal.read_events()


def test_live_attempt_wait_is_bounded_and_retryable(tmp_path: Path) -> None:
    """A live attempt produces a bounded retryable conflict instead of hanging."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "live-key", request, None)
    accepted_claim = _accept(journal, identity, now=_TIME)
    assert accepted_claim.accepted is not None
    _reserve_candidate(journal, accepted_claim.accepted, now=_TIME)
    service = JournaledRouterRuntime(
        cast(RouterRuntime, _FakeRuntime()),
        journal,
        clock=lambda: _TIME,
        wait_timeout_seconds=0,
        stale_after_seconds=3600,
    )

    with pytest.raises(RuntimeInteractionInProgressError) as caught:
        service.complete(request, idempotency_key="live-key")
    assert caught.value.retryable


def test_append_flushes_and_fsyncs_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every accepted and completed line reaches fsync before the service returns."""
    calls: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="fsync-key")

    assert len(calls) >= 2


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_first_append_fsyncs_file_and_new_directory_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first accepted claim persists the file and each newly named parent entry."""
    fsync_calls = 0
    directory_opens: list[Path] = []
    real_fsync = os.fsync
    real_open = os.open

    def record_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(descriptor)

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if flags & getattr(os, "O_DIRECTORY", 0):
            directory_opens.append(Path(os.fsdecode(path)))
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "open", record_open)
    service = _service(tmp_path / ".wmo", _FakeRuntime())
    service.complete(_request(), idempotency_key="directory-fsync-key")

    assert fsync_calls >= 6
    assert {"runtime", "support-agent", "projects", ".wmo"}.issubset(
        {path.name for path in directory_opens}
    )


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_append_retries_directory_fsync_after_first_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later append repairs directory durability after first creation fsync fails."""
    journal = RuntimeInteractionJournal(
        ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    )
    request = _request()
    identity = _interaction_identity("support-agent", "fsync-retry-key", request, None)
    embedding = BillingSourceEconomics(
        billing_source=BillingSource.HOST_MANAGED,
        economics=_operation(
            ordinal=1,
            component=RoutedProviderComponent.ROUTER_EMBEDDING,
            source=BillingSource.HOST_MANAGED,
            disposition=RoutedSpendDisposition.LOCALLY_PRICED,
            input_tokens=5,
        ).economics,
    )
    directory_failures = 0
    successful_directory_fsyncs = 0
    real_fsync = os.fsync

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_failures, successful_directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            if directory_failures == 0:
                directory_failures += 1
                raise OSError("simulated directory fsync failure")
            successful_directory_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_directory_fsync)
    with pytest.raises(RuntimeJournalError, match="cannot persist runtime journal directory"):
        journal.reserve_selection(
            identity,
            embedding,
            now=_TIME,
            stale_after=timedelta(seconds=1),
        )

    retry = journal.reserve_selection(
        identity,
        embedding,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(seconds=1),
    )

    assert retry.status == "dispatch"
    assert successful_directory_fsyncs >= 8
    assert [event.operation.disposition for event in journal.read_spend_events()] == [
        RoutedSpendDisposition.RESERVED,
        RoutedSpendDisposition.RESERVED_AMBIGUOUS,
        RoutedSpendDisposition.RESERVED,
    ]

"""Idempotent routed completion service over the durable interaction journal."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
)
from wmo.common.models import ModelRequest, ModelSnapshot
from wmo.runtime.models.providers.transport import ProviderTransportError
from wmo.runtime.router.economics import RoutedProviderComponent
from wmo.runtime.router.journal import (
    JournalClaim,
    RuntimeAcceptance,
    RuntimeAttemptFailedEvent,
    RuntimeCompletedEvent,
    RuntimeInteractionFailedError,
    RuntimeInteractionIdentity,
    RuntimeInteractionInProgressError,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    _interaction_identity,
)
from wmo.runtime.router.runtime import (
    RoutedModelResponse,
    RouterModelCapabilityError,
    RouterRuntime,
    RouterRuntimeIntegrityError,
)

_SELECTION_LOCKS = tuple(threading.RLock() for _ in range(64))


class JournaledRouterRuntime:
    """Idempotent completion service around one verified ``RouterRuntime``.

    WMO guarantees one local logical interaction and one pinned target. A provider dispatch can
    happen more than once only in the crash-after-success, before-journal window, unless the
    provider itself honors the forwarded idempotency key.
    """

    def __init__(
        self,
        runtime: RouterRuntime,
        journal: RuntimeInteractionJournal,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wait_timeout_seconds: float = 10.0,
        stale_after_seconds: float = 120.0,
    ) -> None:
        """Create a durable wrapper with bounded live-attempt waiting.

        Args:
            runtime: Verified policy runtime used for selection and completion.
            journal: Durable interaction and provider-spend journal.
            clock: Time source for durable records and stale-claim decisions.
            monotonic: Monotonic time source for bounded local waiting.
            sleep: Injectable local wait implementation.
            wait_timeout_seconds: Maximum time to wait for another live owner.
            stale_after_seconds: Age after which a live reservation fails closed as ambiguous.

        Raises:
            ValueError: A wait bound is negative or the stale age is not positive.
        """
        if wait_timeout_seconds < 0 or stale_after_seconds <= 0:
            raise ValueError("journal wait must be non-negative and stale age must be positive")
        self.runtime = runtime
        self.journal = journal
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._wait_timeout_seconds = wait_timeout_seconds
        self._stale_after = timedelta(seconds=stale_after_seconds)

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Return one durable response for a caller-owned idempotency key.

        Args:
            request: Canonical provider-neutral model request.
            idempotency_key: Standard request key. Only its SHA-256 digest is persisted.
            conversation_id: Optional caller identity for sticky routing. Only a stable digest ID
                is persisted. The idempotency key defines a single interaction, not a conversation.

        Returns:
            The pinned routing decision and completed or replayed model response.

        Raises:
            RuntimeIdempotencyConflictError: The key was reused for different request content.
            RuntimeInteractionInProgressError: A live attempt exceeded the bounded local wait.
            RuntimeInteractionFailedError: A prior attempt failed permanently.
        """
        _validate_external_id(idempotency_key, label="idempotency key", visible_ascii=True)
        if conversation_id is not None:
            _validate_external_id(conversation_id, label="conversation ID", visible_ascii=False)
        identity = _interaction_identity(
            self.journal.project_id,
            idempotency_key,
            request,
            conversation_id,
        )
        deadline = self._monotonic() + self._wait_timeout_seconds
        while True:
            claim = self.journal.claim(
                identity,
                now=self._clock(),
                stale_after=self._stale_after,
            )
            if claim.status == "needs_selection":
                selection_lock = _selection_lock(identity.interaction_id)
                remaining = deadline - self._monotonic()
                if remaining <= 0 or not selection_lock.acquire(timeout=remaining):
                    raise RuntimeInteractionInProgressError(
                        "another local request is selecting an idempotent interaction; retry"
                    )
                try:
                    claim = self._select_and_accept_if_needed(identity, request)
                finally:
                    selection_lock.release()
            terminal = self._resolve_claim(claim)
            if terminal is not None:
                return terminal
            if claim.status == "live":
                self._wait_or_raise(deadline)
                continue
            if claim.accepted is None:
                raise RuntimeJournalError("dispatch journal claim omitted its accepted record")
            accepted = claim.accepted
            try:
                candidate = self.runtime.candidate_reservation(
                    request,
                    episode_id=accepted.identity.lineage_id,
                    decision=accepted.acceptance.decision,
                )
            except Exception as exc:
                failure = _structured_completion_failure(exc)
                terminal_event = self.journal.record_failure(
                    accepted, failure, failed_at=self._clock()
                )
                if isinstance(terminal_event, RuntimeCompletedEvent):
                    return RoutedModelResponse(
                        decision=accepted.acceptance.decision,
                        response=terminal_event.response,
                        economics=terminal_event.economics,
                    )
                raise RuntimeInteractionFailedError(
                    terminal_event.failure, terminal_event.spend
                ) from exc
            reserved_candidate = self.journal.reserve_candidate(
                accepted,
                candidate,
                now=self._clock(),
            )
            if reserved_candidate.status == "dispatch":
                if reserved_candidate.reservation is None:
                    raise RuntimeJournalError("candidate dispatch omitted its reservation")
                break
            if reserved_candidate.status == "superseded":
                continue
            self._wait_or_raise(deadline)
        try:
            routed = self.runtime.complete(
                request,
                episode_id=accepted.identity.lineage_id,
                decision=accepted.acceptance.decision,
                provider_idempotency_key=idempotency_key,
            )
        except Exception as exc:
            failure = _structured_completion_failure(exc)
            terminal_event = self.journal.record_failure(accepted, failure, failed_at=self._clock())
            if isinstance(terminal_event, RuntimeCompletedEvent):
                return RoutedModelResponse(
                    decision=accepted.acceptance.decision,
                    response=terminal_event.response,
                    economics=terminal_event.economics,
                )
            raise RuntimeInteractionFailedError(
                terminal_event.failure, terminal_event.spend
            ) from exc
        candidate_operations = tuple(
            item
            for item in routed.economics.operations
            if item.component == RoutedProviderComponent.SELECTED_CANDIDATE
        )
        if len(candidate_operations) != 1:
            raise RuntimeJournalError(
                "direct runtime completion omitted exact candidate operation economics"
            )
        completed = self.journal.record_completed(
            accepted,
            routed.response,
            candidate_operation=candidate_operations[0],
            completed_at=self._clock(),
        )
        if isinstance(completed, RuntimeAttemptFailedEvent):
            raise RuntimeInteractionFailedError(completed.failure, completed.spend)
        return RoutedModelResponse(
            decision=accepted.acceptance.decision,
            response=completed.response,
            economics=completed.economics,
        )

    def _select_and_accept_if_needed(
        self,
        identity: RuntimeInteractionIdentity,
        request: ModelRequest,
    ) -> JournalClaim:
        """Reserve, execute, and settle selection when no route is durable.

        Args:
            identity: Canonical request and lineage identity.
            request: Provider-neutral request to route.

        Returns:
            Current journal claim after selection or a concurrent transition.
        """
        claim = self.journal.claim(
            identity,
            now=self._clock(),
            stale_after=self._stale_after,
        )
        if claim.status != "needs_selection":
            return claim
        embedding = self.runtime.embedding_reservation(request)
        reserved = self.journal.reserve_selection(
            identity,
            embedding,
            now=self._clock(),
            stale_after=self._stale_after,
        )
        if reserved.status != "dispatch":
            return JournalClaim("live")
        if reserved.reservation is None:
            raise RuntimeJournalError("selection dispatch omitted its reservation")
        try:
            decision = self.runtime.select(request, episode_id=identity.lineage_id)
            selection_operation = self.runtime.selection_operation(
                request,
                episode_id=identity.lineage_id,
                decision=decision,
            )
        except Exception:
            self.journal.record_selection_failure(
                identity,
                reserved.reservation,
                failed_at=self._clock(),
            )
            raise
        acceptance = RuntimeAcceptance(
            decision=decision,
            selected_alias=decision.selected_alias,
            selected_model=_selected_model(self.runtime, decision.selected_alias),
            router_embedding_billing_source=embedding.billing_source,
            policy_input=ArtifactInput(
                artifact_id=decision.policy_id,
                sha256=decision.policy_sha256,
            ),
        )
        return self.journal.record_acceptance(
            identity,
            acceptance,
            reserved.reservation,
            selection_operation,
            accepted_at=self._clock(),
        )

    def _resolve_claim(self, claim: JournalClaim) -> RoutedModelResponse | None:
        """Replay a terminal claim or raise its durable permanent failure.

        Args:
            claim: Fully validated current journal state.

        Returns:
            Replayed routed response for a completed claim, otherwise ``None``.

        Raises:
            RuntimeInteractionFailedError: The durable attempt failed with cumulative spend.
        """
        if claim.status == "completed":
            if claim.accepted is None or claim.completed is None:
                raise RuntimeJournalError("completed journal claim omitted its records")
            return RoutedModelResponse(
                decision=claim.accepted.acceptance.decision,
                response=claim.completed.response,
                economics=claim.completed.economics,
            )
        if claim.status != "failed":
            return None
        if claim.failure is None:
            raise RuntimeJournalError("failed journal claim omitted its failure")
        raise RuntimeInteractionFailedError(claim.failure.failure, claim.failure.spend)

    def _wait_or_raise(self, deadline: float) -> None:
        """Sleep briefly while another process owns a live provider reservation.

        Args:
            deadline: Absolute monotonic deadline for local waiting.

        Raises:
            RuntimeInteractionInProgressError: The bounded wait has expired.
        """
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise RuntimeInteractionInProgressError(
                "another process is completing this idempotent interaction; retry"
            )
        self._sleep(min(0.05, remaining))


def _selection_lock(interaction_id: str) -> threading.RLock:
    """Return a bounded process-wide lock for one interaction selection handshake."""
    digest = int(interaction_id.rsplit("-", maxsplit=1)[-1], 16)
    return _SELECTION_LOCKS[digest % len(_SELECTION_LOCKS)]


def _selected_model(runtime: RouterRuntime, alias: str) -> ModelSnapshot:
    """Return the policy-pinned model snapshot for one selected candidate alias."""
    for candidate in runtime.policy.candidates:
        if candidate.alias == alias:
            return candidate.model
    raise RuntimeJournalError("routing decision selected an alias outside the frozen policy")


def _structured_completion_failure(exception: Exception) -> StructuredFailure:
    """Normalize a completion exception without retaining provider secrets."""
    if isinstance(exception, RouterModelCapabilityError):
        return StructuredFailure(
            code=FailureCode.UNSUPPORTED,
            message="routed model does not support the requested capability",
            retryable=False,
            exception_type=type(exception).__name__,
            attribution=FailureAttribution.MODEL,
        )
    if isinstance(exception, (RouterRuntimeIntegrityError, ValueError)):
        return StructuredFailure(
            code=FailureCode.INTERNAL,
            message="routed model runtime rejected the accepted target",
            retryable=False,
            exception_type=type(exception).__name__,
            attribution=FailureAttribution.MODEL,
        )
    retryable = isinstance(exception, TimeoutError)
    code = FailureCode.TIMEOUT if retryable else FailureCode.PROVIDER
    if isinstance(exception, ProviderTransportError):
        retryable = (
            exception.status_code is None
            or exception.status_code in {408, 409, 425, 429}
            or (exception.status_code is not None and exception.status_code >= 500)
        )
        code = FailureCode.PROVIDER
    return StructuredFailure(
        code=code,
        message="routed model provider attempt failed",
        retryable=retryable,
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.MODEL,
    )


def _validate_external_id(value: str, *, label: str, visible_ascii: bool) -> None:
    """Validate a caller identity before hashing and provider forwarding."""
    if not value or len(value) > 512 or value.strip() != value:
        raise ValueError(f"{label} must be 1 to 512 non-blank characters")
    if visible_ascii and any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError(f"{label} must contain only visible ASCII characters")

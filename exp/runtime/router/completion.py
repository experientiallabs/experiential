"""Neutral completion-service contract between HTTP adapters and durable runtimes."""

from __future__ import annotations

from typing import Protocol

from exp.common.core.artifacts import StructuredFailure
from exp.common.models import ModelRequest
from exp.runtime.router.economics import RoutedSpendLedger
from exp.runtime.router.journal import (
    RuntimeIdempotencyConflictError,
    RuntimeInteractionFailedError,
    RuntimeInteractionInProgressError,
)
from exp.runtime.router.journal_service import JournaledRouterRuntime
from exp.runtime.router.runtime import RoutedModelResponse


class RouterCompletionConflictError(ValueError):
    """A caller key conflicts with a previously accepted logical interaction."""


class RouterCompletionInProgressError(RuntimeError):
    """A durable interaction is already being completed and may be retried."""


class RouterCompletionFailedError(RuntimeError):
    """A durable interaction has a stable terminal provider failure."""

    def __init__(self, failure: StructuredFailure, spend: RoutedSpendLedger) -> None:
        """Initialize the error with safe failure and exact alias-free spend.

        Args:
            failure: Durable redacted provider failure.
            spend: Source-attributed cumulative accounting through the failed attempt.
        """
        super().__init__(failure.message)
        self.failure = failure
        self.spend = spend
        self.retryable = failure.retryable


class RouterCompletionService(Protocol):
    """Completion boundary used by public router adapters in a selected traffic mode."""

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Complete one caller-keyed interaction under the service's traffic mode.

        Args:
            request: Provider-neutral request to route and execute.
            idempotency_key: Caller key used for replay only by durable implementations.
            conversation_id: Optional stable identity for sticky conversation routing.

        Returns:
            Newly completed or durably replayed routed model response.

        Raises:
            RouterCompletionConflictError: The key names different request state.
            RouterCompletionInProgressError: The keyed interaction is still running.
            RouterCompletionFailedError: The keyed interaction has a terminal failure.
        """


class JournaledRouterCompletionService:
    """Translate the runtime journal boundary into the neutral completion contract."""

    def __init__(self, runtime: JournaledRouterRuntime) -> None:
        """Bind the neutral service to one project journal runtime."""
        self._runtime = runtime

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Complete one durable interaction while preserving typed failure meaning.

        Args:
            request: Provider-neutral request to route and execute.
            idempotency_key: Caller or adapter key that identifies one logical interaction.
            conversation_id: Optional stable identity for sticky conversation routing.

        Returns:
            The newly completed or durably replayed routed response.

        Raises:
            RouterCompletionConflictError: The key names different request or lineage state.
            RouterCompletionInProgressError: Another process still owns the live interaction.
            RouterCompletionFailedError: The interaction has a durable terminal failure.
        """
        try:
            return self._runtime.complete(
                request,
                idempotency_key=idempotency_key,
                conversation_id=conversation_id,
            )
        except RuntimeIdempotencyConflictError as exc:
            raise RouterCompletionConflictError(str(exc)) from exc
        except RuntimeInteractionInProgressError as exc:
            raise RouterCompletionInProgressError(str(exc)) from exc
        except RuntimeInteractionFailedError as exc:
            raise RouterCompletionFailedError(exc.failure, exc.spend) from exc

"""Neutral completion-service contract between HTTP adapters and durable runtimes."""

from __future__ import annotations

import uuid
from typing import Protocol

from wmo.common.core.artifacts import StructuredFailure
from wmo.common.models import ModelRequest
from wmo.runtime.router.journal import (
    JournaledRouterRuntime,
    RuntimeIdempotencyConflictError,
    RuntimeInteractionFailedError,
    RuntimeInteractionInProgressError,
)
from wmo.runtime.router.runtime import RoutedModelResponse, RouterRuntime


class RouterCompletionConflictError(ValueError):
    """A caller key conflicts with a previously accepted logical interaction."""


class RouterCompletionInProgressError(RuntimeError):
    """A durable interaction is already being completed and may be retried."""


class RouterCompletionFailedError(RuntimeError):
    """A durable interaction has a stable terminal provider failure."""

    def __init__(self, failure: StructuredFailure) -> None:
        """Initialize the error with its durable structured failure."""
        super().__init__(failure.message)
        self.failure = failure


class RouterCompletionService(Protocol):
    """Durable idempotent completion boundary used by public router adapters."""

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Complete or replay one durable caller-keyed interaction.

        Args:
            request: Provider-neutral request to route and execute.
            idempotency_key: Caller key that identifies the logical interaction.
            conversation_id: Optional stable identity for sticky conversation routing.

        Returns:
            The original or replayed routed model response.

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
            raise RouterCompletionFailedError(exc.failure) from exc


def complete_router_request(
    runtime: RouterRuntime,
    service: RouterCompletionService | None,
    request: ModelRequest,
    *,
    idempotency_key: str | None,
    conversation_id: str | None = None,
) -> RoutedModelResponse:
    """Dispatch through durable state when injected and preserve raw low-level composition.

    Args:
        runtime: Activated router used directly only when no durable service is injected.
        service: Optional durable completion owner for every request on this endpoint.
        request: Canonical provider-neutral request.
        idempotency_key: Optional standard caller replay key.
        conversation_id: Optional stable Responses conversation identity.

    Returns:
        The routed response after durable completion or direct low-level execution.

    Raises:
        RouterCompletionConflictError: A keyed call has no durable service or conflicts with state.
        RouterCompletionInProgressError: Another process owns the durable interaction.
        RouterCompletionFailedError: The durable interaction has a terminal failure.
    """
    if service is not None:
        return service.complete(
            request,
            idempotency_key=idempotency_key or f"wmo-request-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
        )
    if idempotency_key is not None:
        raise RouterCompletionConflictError(
            "this router has no durable idempotency service; retry without Idempotency-Key"
        )
    return runtime.complete(request, episode_id=conversation_id)

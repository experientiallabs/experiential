"""Typed translation tests for the neutral durable completion boundary."""

from __future__ import annotations

from typing import Never, cast

import pytest

from wmo.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from wmo.common.models import ModelRequest
from wmo.runtime.router.completion import (
    JournaledRouterCompletionService,
    RouterCompletionConflictError,
    RouterCompletionFailedError,
    RouterCompletionInProgressError,
)
from wmo.runtime.router.journal import (
    JournaledRouterRuntime,
    RuntimeIdempotencyConflictError,
    RuntimeInteractionFailedError,
    RuntimeInteractionInProgressError,
)
from wmo.runtime.router.runtime import RoutedModelResponse
from wmo.runtime.router.runtime_test import _request


class _FailingRuntime:
    """Runtime seam that raises one exact typed journal exception."""

    def __init__(self, error: Exception) -> None:
        """Retain the exact exception raised by every completion attempt."""
        self.error = error

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> Never:
        """Raise the configured exception without inspecting request content.

        Args:
            request: Provider-neutral request accepted by the runtime contract.
            idempotency_key: Logical interaction key accepted by the runtime contract.
            conversation_id: Optional sticky-routing identity.

        Raises:
            Exception: The exact configured typed runtime exception.
        """
        del request, idempotency_key, conversation_id
        raise self.error


def _failure() -> StructuredFailure:
    """Return one durable provider failure for translation assertions."""
    return StructuredFailure(
        code=FailureCode.PROVIDER,
        message="provider attempt failed",
        retryable=False,
        exception_type="ProviderError",
        attribution=FailureAttribution.MODEL,
    )


@pytest.mark.parametrize(
    ("runtime_error", "completion_error"),
    [
        (RuntimeIdempotencyConflictError("conflict"), RouterCompletionConflictError),
        (RuntimeInteractionInProgressError("in progress"), RouterCompletionInProgressError),
        (RuntimeInteractionFailedError(_failure()), RouterCompletionFailedError),
    ],
)
def test_runtime_failures_translate_by_type(
    runtime_error: Exception,
    completion_error: type[Exception],
) -> None:
    """Map journal exceptions without inspecting or matching their messages.

    Args:
        runtime_error: Exact typed runtime failure raised by the fixture.
        completion_error: Neutral adapter exception required by the endpoint.
    """
    runtime = cast(JournaledRouterRuntime, _FailingRuntime(runtime_error))
    service = JournaledRouterCompletionService(runtime)

    with pytest.raises(completion_error) as caught:
        service.complete(_request(), idempotency_key="request-a")

    assert caught.value.__cause__ is runtime_error
    if isinstance(caught.value, RouterCompletionFailedError):
        assert caught.value.failure is cast(RuntimeInteractionFailedError, runtime_error).failure


def test_adapter_forwards_complete_request_identity() -> None:
    """Preserve request, caller key, and conversation identity at the journal seam."""
    calls: list[tuple[ModelRequest, str, str | None]] = []

    class _CapturingRuntime:
        """Minimal service double that records exact completion inputs."""

        def complete(
            self,
            request: ModelRequest,
            *,
            idempotency_key: str,
            conversation_id: str | None = None,
        ) -> RoutedModelResponse:
            """Record inputs before stopping the test at the forwarding boundary.

            Args:
                request: Provider-neutral request forwarded by the adapter.
                idempotency_key: Exact interaction key forwarded by the adapter.
                conversation_id: Exact conversation identity forwarded by the adapter.

            Raises:
                RuntimeInteractionInProgressError: Always, after recording the arguments.
            """
            calls.append((request, idempotency_key, conversation_id))
            raise RuntimeInteractionInProgressError("captured")

    request = _request()
    service = JournaledRouterCompletionService(cast(JournaledRouterRuntime, _CapturingRuntime()))

    with pytest.raises(RouterCompletionInProgressError):
        service.complete(
            request,
            idempotency_key="request-a",
            conversation_id="conversation-a",
        )

    assert calls == [(request, "request-a", "conversation-a")]

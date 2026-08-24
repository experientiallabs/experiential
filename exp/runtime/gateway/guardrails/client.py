"""Internal classifier client seam that cannot recurse through public routes."""

from __future__ import annotations

from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)

_INTERNAL_CLASSIFICATION: ContextVar[bool] = ContextVar(
    "exp_gateway_guardrail_internal",
    default=False,
)


class GuardrailRecursionError(RuntimeError):
    """A classifier attempted to re-enter the public gateway route."""


class InspectingClassifier(Protocol):
    """Replaceable async detector that never logs request or response content."""

    def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> Awaitable[ClassifierVerdict]:
        """Inspect one canonical request."""
        ...

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> Awaitable[ClassifierVerdict]:
        """Inspect one winning completion."""
        ...


class SyncInspectingClassifier(Protocol):
    """Leftover synchronous detector used only through ``BoundedSyncClassifier``."""

    def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one canonical request on the caller's thread."""
        ...

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one winning completion on the caller's thread."""
        ...


class InternalClassifierClient(Protocol):
    """Direct adapter transport that never uses a public gateway HTTP route."""

    def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> Awaitable[ClassifierVerdict]:
        """Inspect one canonical request through the bound adapter."""
        ...

    def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> Awaitable[ClassifierVerdict]:
        """Inspect one winning completion through the bound adapter."""
        ...


class DirectClassifierClient:
    """Call registered adapters in-process under the recursion guard.

    This client never opens an HTTP connection and never targets
    ``/v1/chat/completions`` or ``/v1/responses``. Adapters that need a model
    must use their own injected transport, not the public gateway.
    """

    def __init__(self, registry: ClassifierLookup) -> None:
        """Bind the adapter lookup used for every inspection.

        Args:
            registry: Mapping from adapter_id to a callable classifier.
        """
        self._registry = registry

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one canonical request through the bound adapter.

        The await runs wherever the engine isolated this inspect, including an
        isolation worker loop when a deadline is being enforced.
        """
        with classification_scope():
            return await self._registry.require(check.adapter_id).inspect_input(
                request=request,
                check=check,
            )

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one winning completion through the bound adapter.

        The await runs wherever the engine isolated this inspect, including an
        isolation worker loop when a deadline is being enforced.
        """
        with classification_scope():
            return await self._registry.require(check.adapter_id).inspect_output(
                completion=completion,
                check=check,
            )


class ClassifierLookup(Protocol):
    """Resolve one replaceable classifier adapter by identity."""

    def require(self, adapter_id: str) -> InspectingClassifier:
        """Return the adapter or raise ``KeyError`` when it is not registered."""
        ...


@contextmanager
def classification_scope() -> Iterator[None]:
    """Mark the current task as an internal classifier call.

    Yields:
        Nothing. The flag is cleared when the block exits.
    """
    token = _INTERNAL_CLASSIFICATION.set(True)
    try:
        yield
    finally:
        _INTERNAL_CLASSIFICATION.reset(token)


def assert_not_internal_classification() -> None:
    """Reject public-route entry while a classifier call is active.

    Raises:
        GuardrailRecursionError: The public gateway was re-entered from a
            classifier adapter or its internal client.
    """
    if _INTERNAL_CLASSIFICATION.get():
        raise GuardrailRecursionError(
            "classifier adapters cannot recurse through the public gateway route"
        )


def internal_classification_active() -> bool:
    """Return whether the current task is inside a classifier call."""
    return _INTERNAL_CLASSIFICATION.get()

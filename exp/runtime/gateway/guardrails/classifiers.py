"""Replaceable classifier adapters and the in-process registry."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.client import (
    InspectingClassifier,
    SyncInspectingClassifier,
    classification_scope,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)

DEFAULT_SYNC_COMPAT_WORKERS = 2


class SyncClassifierBusyError(TimeoutError):
    """No private sync worker is free, so the inspect was not queued."""


class ClassifierRegistry:
    """Identity-keyed map of replaceable classifier adapters."""

    def __init__(self, adapters: Mapping[str, InspectingClassifier] | None = None) -> None:
        """Index adapters by the policy ``adapter_id``.

        Args:
            adapters: Optional adapter_id to classifier map.
        """
        self._adapters = dict(adapters or {})

    def register(self, adapter_id: str, classifier: InspectingClassifier) -> None:
        """Replace or add one adapter.

        Args:
            adapter_id: Policy adapter identity.
            classifier: Replaceable inspection implementation.
        """
        self._adapters[adapter_id] = classifier

    def require(self, adapter_id: str) -> InspectingClassifier:
        """Return one registered adapter.

        Args:
            adapter_id: Policy adapter identity.

        Returns:
            The bound classifier.

        Raises:
            KeyError: No adapter is registered for ``adapter_id``.
        """
        return self._adapters[adapter_id]


class ScriptedClassifier:
    """Deterministic adapter that returns pre-authored verdicts.

    Used by tests and as a stand-in while an operator wires a real detector.
    It never logs or retains the inspected content.
    """

    def __init__(
        self,
        *,
        input_verdict: ClassifierVerdict | None = None,
        output_verdict: ClassifierVerdict | None = None,
    ) -> None:
        """Bind optional fixed verdicts. Omitted stages never flag.

        Args:
            input_verdict: Result for every input inspection.
            output_verdict: Result for every output inspection.
        """
        self.input_calls = 0
        self.output_calls = 0
        self._input = input_verdict or ClassifierVerdict(flagged=False)
        self._output = output_verdict or ClassifierVerdict(flagged=False)

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Return the scripted input verdict without retaining the request."""
        del request, check
        self.input_calls += 1
        return self._input

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Return the scripted output verdict without retaining the completion."""
        del completion, check
        self.output_calls += 1
        return self._output


class KeywordClassifier:
    """Coarse, test-oriented needle matcher for local experiments.

    Needles are compared as case-folded substrings of message text or of
    completion text and tool-call arguments. This is not a production
    prompt-injection or content-safety classifier. Production policies bind
    replaceable hosted adapters, including a hosted PII redactor, through
    ``http_json``.
    """

    def __init__(self, needles: tuple[str, ...]) -> None:
        """Bind the operator-authored needles.

        Args:
            needles: Non-empty strings that flag when present.

        Raises:
            ValueError: No needle was provided or a needle is empty.
        """
        if not needles or any(not item for item in needles):
            raise ValueError("keyword classifier needles must be non-empty")
        self._needles = tuple(item.casefold() for item in needles)
        self.input_calls = 0
        self.output_calls = 0

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Flag when any needle appears in canonical message content."""
        del check
        self.input_calls += 1
        parts: list[str] = []
        for message in request.messages:
            if message.content:
                parts.append(message.content)
            for call in message.tool_calls:
                parts.append(call.arguments_json())
        return ClassifierVerdict(flagged=_contains_needle("\n".join(parts), self._needles))

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Flag when any needle appears in completion text or tool arguments."""
        del check
        self.output_calls += 1
        parts = [completion.text, *(call.arguments for call in completion.tool_calls)]
        return ClassifierVerdict(flagged=_contains_needle("\n".join(parts), self._needles))


class BoundedSyncClassifier:
    """Compatibility wrapper for leftover synchronous test adapters.

    Production transports implement the async inspect contract directly.
    This wrapper is only for leftover synchronous adapters. Hung sync
    inspects occupy only this wrapper's private workers. They cannot take
    async inflight slots from healthy adapters, and they cannot share a
    process-wide thread pool with other wrappers. Admission is capped at
    ``max_workers``: a full wrapper fails immediately instead of queueing
    another executor job that timeout cannot cancel.
    """

    def __init__(
        self,
        inner: SyncInspectingClassifier,
        *,
        max_workers: int = DEFAULT_SYNC_COMPAT_WORKERS,
    ) -> None:
        """Bind one private executor around a synchronous adapter.

        Args:
            inner: Leftover synchronous detector.
            max_workers: Maximum concurrent sync inspects for this wrapper.

        Raises:
            ValueError: ``max_workers`` is not a positive integer.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._inner = inner
        self._max_workers = max_workers
        self._slots = threading.BoundedSemaphore(max_workers)
        self._admitted = 0
        self._admitted_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="exp-guardrail-sync",
        )

    def admitted_inspects(self) -> int:
        """Return submitted sync inspects that have not finished."""
        with self._admitted_lock:
            return self._admitted

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one request on this wrapper's private executor."""

        def call() -> ClassifierVerdict:
            """Run the sync inspect under the recursion flag."""
            with classification_scope():
                return self._inner.inspect_input(request=request, check=check)

        return await self._run(call)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one completion on this wrapper's private executor."""

        def call() -> ClassifierVerdict:
            """Run the sync inspect under the recursion flag."""
            with classification_scope():
                return self._inner.inspect_output(completion=completion, check=check)

        return await self._run(call)

    async def _run(self, fn: Callable[[], ClassifierVerdict]) -> ClassifierVerdict:
        """Admit at most ``max_workers`` jobs and await one without cancelling it.

        Args:
            fn: Synchronous inspect to run on a private worker.

        Returns:
            The inspect verdict.

        Raises:
            SyncClassifierBusyError: Every private worker is already occupied.
        """
        if not self._admit():
            raise SyncClassifierBusyError("sync classifier has no free worker")
        try:
            future = self._pool.submit(fn)
        except Exception:
            self._release_admission()
            raise
        waiter = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda done: self._finish(done, waiter))
        return await waiter

    def _admit(self) -> bool:
        """Reserve one private worker, or refuse if the cap is full."""
        if not self._slots.acquire(blocking=False):
            return False
        with self._admitted_lock:
            self._admitted += 1
        return True

    def _release_admission(self) -> None:
        """Release one reserved private worker after its job finishes."""
        with self._admitted_lock:
            self._admitted -= 1
        self._slots.release()

    def _finish(
        self,
        future: Future[ClassifierVerdict],
        waiter: asyncio.Future[ClassifierVerdict],
    ) -> None:
        """Release admission and complete the asyncio waiter without cancelling work.

        Args:
            future: Executor future that just became done.
            waiter: Asyncio future the inspect coroutine is awaiting.
        """
        self._release_admission()
        loop = waiter.get_loop()

        def complete() -> None:
            """Copy the worker result onto the waiter if it is still pending."""
            if waiter.done():
                return
            if future.cancelled():
                waiter.cancel()
                return
            error = future.exception()
            if error is not None:
                waiter.set_exception(error)
                return
            waiter.set_result(future.result())

        loop.call_soon_threadsafe(complete)


def _contains_needle(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any case-folded needle is a substring of ``text``."""
    folded = text.casefold()
    return any(needle in folded for needle in needles)

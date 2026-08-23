"""Replaceable classifier adapters and the in-process registry."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

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
    """Operator-authored needle matcher for one capability.

    This is a coarse, local adapter. It is not a hosted detector and does not
    call a model provider. Needles are compared as case-folded substrings of
    message text or of completion text and tool-call arguments.
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
    """Wrap a leftover synchronous adapter with a private executor.

    Hung sync inspects occupy only this wrapper's workers. They cannot take
    async inflight slots from healthy adapters, and they cannot share a
    process-wide thread pool with other wrappers.
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
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="exp-guardrail-sync",
        )

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

        return await asyncio.get_running_loop().run_in_executor(self._pool, call)

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

        return await asyncio.get_running_loop().run_in_executor(self._pool, call)


def _contains_needle(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any case-folded needle is a substring of ``text``."""
    folded = text.casefold()
    return any(needle in folded for needle in needles)

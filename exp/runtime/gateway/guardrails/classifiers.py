"""Replaceable classifier adapters and the in-process registry."""

from __future__ import annotations

from collections.abc import Mapping

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.client import (
    InspectingClassifier,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)
from exp.runtime.gateway.guardrails.text_coverage import text_input_context


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
        """Flag complete supported message, tool, and schema text."""
        del check
        self.input_calls += 1
        context = text_input_context(request)
        parts = [context] if context else []
        for message in request.messages:
            content = message.folded_tool_error_content()
            if content:
                parts.append(content)
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
        parts = [
            completion.text,
            *completion.context,
            *(call.arguments for call in completion.tool_calls),
        ]
        return ClassifierVerdict(flagged=_contains_needle("\n".join(parts), self._needles))


def _contains_needle(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any case-folded needle is a substring of ``text``."""
    folded = text.casefold()
    return any(needle in folded for needle in needles)

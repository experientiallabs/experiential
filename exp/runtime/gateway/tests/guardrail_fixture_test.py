"""Barrier-controlled classifier fixtures for guardrail session unit tests."""

from __future__ import annotations

import asyncio
import threading
import time

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.tests.native_classifier_fixture_test import NativeClassifierFixture


class _Classifier:
    """Barrier-controlled input classifier with no output inspection capability."""

    def __init__(self, outcome: str) -> None:
        """Create one independent classifier and release latch."""
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.requests: list[GatewayRequest] = []

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Wait independently of gateway bridge workers before returning a decision."""
        self.requests.append(request)
        self.started.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.outcome == "error":
            raise RuntimeError("private-detector-error-must-not-escape")
        if self.outcome == "unsupported":
            raise ClassifierCoverageError
        return ClassifierVerdict(flagged=self.outcome == "block")

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Reject accidental use of output moderation in this input-only policy."""
        raise AssertionError("input-only policy called output inspection")


def _engine(classifier: _Classifier, timeout_ms: int = 3000) -> GuardrailEngine:
    """Compose the input-only policy through the same scoped store as every other guardrail."""
    return GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy(
                    policy_id="criminal-input",
                    revision="fixture-v1",
                    protected=True,
                    input_execution="parallel",
                    checks=(
                        GuardrailCheck(
                            check_id="criminal",
                            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                            stage=GuardrailCheckStage.INPUT,
                            action=GuardrailAction.BLOCK,
                            adapter_id="criminal",
                            timeout_ms=timeout_ms,
                        ),
                    ),
                ),
            )
        ),
        client=DirectClassifierClient(
            ClassifierRegistry({"criminal": NativeClassifierFixture(classifier)})
        ),
        monotonic=time.monotonic,
    )

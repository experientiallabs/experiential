"""Observing decision-model checks that the data plane classifies itself.

Admission hands the data plane one entry per distinct window an observing check
classifies: the classifier's native handle and the window text. The data plane
classifies them beside generation, so the request never waits and the complete
request is never copied for them, then reports every answer once. Until that
report arrives, or the bound below evicts it, the control plane keeps only what
it needs to record the answers: the policy, the check, and the adapter's
identity. It never keeps the window text.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Final

from exp.runtime.gateway.guardrails.contracts import (
    ClassificationResult,
    GuardrailCheck,
    GuardrailPolicy,
)

MAX_PENDING_REQUESTS: Final = 4_096
"""Requests whose report may be outstanding; beyond it the oldest are forgotten."""


@dataclass(frozen=True)
class NativeObservation:
    """One observing check the data plane classifies for one request window.

    Attributes:
        policy: The observing policy, as the request's session froze it.
        check: The input check that classifies.
        adapter_id: The classifier's adapter identity.
        configuration_digest: The classifier's settings digest.
        handle: The handle the data plane resolves the classifier by.
    """

    policy: GuardrailPolicy
    check: GuardrailCheck
    adapter_id: str
    configuration_digest: str
    handle: int


class PendingObservations:
    """Bounded, thread-safe map from a request id to the observations its report answers."""

    def __init__(self, max_requests: int = MAX_PENDING_REQUESTS) -> None:
        """Bound how many requests may await their report at once."""
        self._max_requests = max_requests
        self._pending: OrderedDict[str, tuple[NativeObservation, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.evicted = 0

    def expect(self, request_id: str, observations: tuple[NativeObservation, ...]) -> None:
        """Hold one request's observations until its report arrives."""
        with self._lock:
            self._pending[request_id] = observations
            while len(self._pending) > self._max_requests:
                self._pending.popitem(last=False)
                self.evicted += 1

    def take(self, request_id: str) -> tuple[NativeObservation, ...]:
        """Release one request's observations; empty when unknown or evicted."""
        with self._lock:
            return self._pending.pop(request_id, ())


def classification_result(observation: NativeObservation, answer: object) -> ClassificationResult:
    """Decode one data-plane answer for its observing check.

    Raises:
        TypeError: The answer does not have the data plane's shape.
    """
    if not isinstance(answer, dict):
        raise TypeError("data-plane classification answer is malformed")
    probabilities, scores = answer.get("probabilities"), answer.get("scores")
    flagged, latency_ms = answer.get("flagged"), answer.get("latency_ms")
    if not (
        isinstance(probabilities, dict)
        and isinstance(scores, dict)
        and isinstance(flagged, list)
        and isinstance(latency_ms, (int, float))
    ):
        raise TypeError("data-plane classification answer is malformed")
    return ClassificationResult(
        adapter_id=observation.adapter_id,
        check_id=observation.check.check_id,
        configuration_digest=observation.configuration_digest,
        probabilities=_pairs(probabilities),
        scores=_pairs(scores),
        flagged=tuple(str(name) for name in flagged),
        cached=answer.get("cached") is True,
        latency_seconds=float(latency_ms) / 1000.0,
    )


def _pairs(values: object) -> tuple[tuple[str, float], ...]:
    """Return ``(name, number)`` pairs in order, refusing a non-numeric value."""
    if not isinstance(values, dict):
        raise TypeError("data-plane classification answer is malformed")
    pairs: list[tuple[str, float]] = []
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("data-plane classification answer is malformed")
        pairs.append((str(key), float(value)))
    return tuple(pairs)

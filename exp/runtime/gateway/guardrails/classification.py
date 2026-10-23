"""Classification checks that score a bounded request window with a decision model.

One SystemOne call asks every configured question in a single forward pass:
``POST <url>`` with ``{"model", "state", "questions"}``, answered by
``{"answers": {"<id>": {"type": "noul", "noul": <probability>}}}``. Question
probabilities combine into operator categories, and a request is flagged when
any category score reaches its threshold. A self-hosted SGLang server running a
Clef model serves this wire, as do hosted SystemOne endpoints.

This module validates the document and builds the window; the native
``DecisionClassifier`` makes the call, scores it, and caches definitive
answers, so no request blocks the interpreter while the model answers. Each
verdict carries its content-free answer, which the engine reports to the host
with the policy and check it answered for.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import os
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonValue, canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    MAX_DECISION_QUESTIONS,
    DecisionQuestion,
    DecisionRequest,
    NoulQuestion,
)
from exp.runtime.gateway.guardrails.classification_window import (
    ClassificationWindow,
    classification_window,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassificationResult,
    ClassificationVerdict,
    ClassifierCoverageError,
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)
from exp.runtime.gateway.guardrails.http_json import (
    DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES,
    ClassifierProtocolError,
    validate_bearer_env_name,
    validate_classifier_url,
)

if TYPE_CHECKING:
    from pydantic import SecretStr

_logger = logging.getLogger(__name__)

DEFAULT_DECISION_TIMEOUT_MS: Final = 2_000
SCORE_BUCKETS: Final = 20
"""Histogram buckets per category, each 0.05 wide."""
_STATE_BYTES_PER_CHARACTER: Final = 6
"""Worst-case JSON bytes for one character: a ``\\u00XX`` control-character escape."""
_STATE_LABEL_BYTES: Final = 1_024
_UNSET_BEARER: Final = "classifier bearer environment variable is unset"


class VerdictCacheSettings(ContractModel):
    """Bounds for one adapter's verdict cache.

    Attributes:
        max_entries: Least-recently-used entries kept, default 4,096; zero disables caching.
        ttl_seconds: Seconds an entry stays valid after it is stored, default 300.
    """

    max_entries: int = Field(default=4_096, ge=0, le=1_000_000)
    ttl_seconds: float = Field(default=300.0, gt=0, le=86_400, allow_inf_nan=False)


class ClassificationCategory(ContractModel):
    """One operator category scored from one or more probability questions.

    Attributes:
        questions: Question identity to its ``noul`` question; identities are unique
            across every category of one adapter.
        aggregate: How question probabilities combine: ``max`` by default, or ``mean``.
        threshold: Score in ``(0, 1]`` at or above which the category flags a request.
    """

    questions: dict[str, NoulQuestion] = Field(min_length=1, max_length=MAX_DECISION_QUESTIONS)
    aggregate: Literal["max", "mean"] = "max"
    threshold: float = Field(gt=0, le=1, allow_inf_nan=False)


class DecisionModelAdapterDocument(ContractModel):
    """Decision-model classification adapter declared in ``guardrails.json``.

    Attributes:
        adapter_id: Policy adapter identity.
        kind: Always ``decision_model``.
        url: Full SystemOne endpoint, such as ``http://127.0.0.1:30000/systemone``.
        bearer_env: Optional environment variable that supplies a bearer credential.
        model: Decision model identifier sent in every request.
        timeout_ms: Deadline for one classifier HTTP call, default 2,000 ms.
        max_response_bytes: Largest accepted answer body, default 64 KiB.
        categories: Category name to its questions, aggregate, and threshold.
        window: Which request text is classified, with the documented defaults.
        cache: Verdict cache bounds, with the documented defaults.
    """

    adapter_id: ArtifactId
    kind: Literal["decision_model"]
    url: str
    bearer_env: str | None = None
    model: str = Field(min_length=1, max_length=512)
    timeout_ms: int = Field(default=DEFAULT_DECISION_TIMEOUT_MS, ge=1, le=30_000)
    max_response_bytes: int = Field(
        default=DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES, ge=1, le=1_048_576
    )
    categories: dict[ArtifactId, ClassificationCategory] = Field(min_length=1)
    window: ClassificationWindow = Field(default_factory=ClassificationWindow)
    cache: VerdictCacheSettings = Field(default_factory=VerdictCacheSettings)

    @model_validator(mode="after")
    def _fits_one_decision_request(self) -> DecisionModelAdapterDocument:
        """Require one valid SystemOne request to carry every question and any window.

        Returns:
            The validated document.

        Raises:
            ValueError: The endpoint or credential name is unsafe, a question
                identity repeats, or the questions plus the largest window exceed
                one decision request.
        """
        validate_classifier_url(self.url)
        if self.bearer_env is not None:
            validate_bearer_env_name(self.bearer_env)
        identities = [key for category in self.categories.values() for key in category.questions]
        if len(set(identities)) != len(identities):
            raise ValueError("decision_model question identities must be unique across categories")
        request = DecisionRequest(state="", questions=self.questions)
        question_bytes = len(canonical_json_bytes(request.provider_body(self.model)))
        state_bytes = self.window.kept_characters * _STATE_BYTES_PER_CHARACTER + _STATE_LABEL_BYTES
        if question_bytes + state_bytes > MAX_DECISION_INPUT_BYTES:
            raise ValueError(
                "decision_model questions and window exceed the 262144-byte decision input limit"
            )
        return self

    @property
    def questions(self) -> dict[str, DecisionQuestion]:
        """Return every category's questions as one SystemOne question map."""
        return {
            key: question
            for category in self.categories.values()
            for key, question in category.questions.items()
        }


def score_bucket(score: float) -> int:
    """Return the 0.05-wide histogram bucket of a score in ``[0, 1]``."""
    return min(math.floor(score * SCORE_BUCKETS), SCORE_BUCKETS - 1)


class DecisionModelClassifier:
    """Classify a bounded request window with one SystemOne decision call.

    The window, not the complete request, leaves the gateway, so a large agent
    request is never refused for its total size. Definitive answers are cached
    by window digest; errors are not. Non-200 statuses, transport failures and
    timeouts, malformed bodies, missing answers, and probabilities that are not
    finite numbers in ``[0, 1]`` are classifier protocol errors. The adapter
    logs only content-free metadata and never logs window or question text.
    """

    def __init__(
        self,
        document: DecisionModelAdapterDocument,
        *,
        bearer: SecretStr | None = None,
    ) -> None:
        """Compose the native classifier for one validated adapter document.

        Args:
            document: Endpoint, model, categories, window, and cache settings.
            bearer: The endpoint's credential. ``None`` reads ``bearer_env`` once,
                here; a named variable that is unset fails every inspection.
        """
        self._document = document
        self._digest = sha256(
            canonical_json_bytes(
                {
                    "wire": "systemone",
                    "url": document.url,
                    "model": document.model,
                    "categories": {
                        name: category.model_dump(mode="json", exclude_none=True)
                        for name, category in document.categories.items()
                    },
                    "window": document.window.model_dump(mode="json"),
                }
            )
        ).hexdigest()
        credential = None if bearer is None else bearer.get_secret_value()
        if credential is None and document.bearer_env is not None:
            credential = os.environ.get(document.bearer_env) or None
        self._unset_bearer = credential is None and document.bearer_env is not None
        questions = DecisionRequest(state="", questions=document.questions).provider_body(
            document.model
        )["questions"]
        specification = {
            "url": document.url,
            "model": document.model,
            "timeout_ms": document.timeout_ms,
            "max_response_bytes": document.max_response_bytes,
            "configuration_digest": self._digest,
            "questions": questions,
            "categories": [
                {
                    "id": name,
                    "aggregate": category.aggregate,
                    "threshold": category.threshold,
                    "questions": list(category.questions),
                }
                for name, category in document.categories.items()
            ],
            "cache": document.cache.model_dump(mode="json"),
        }
        native = importlib.import_module("exp_gateway_native")
        self._native = native.DecisionClassifier(json.dumps(specification), credential)

    @property
    def configuration_digest(self) -> str:
        """Return the digest of the endpoint, model, categories, and window."""
        return self._digest

    def telemetry(self) -> dict[str, JsonValue]:
        """Return this adapter's content-free score, cache, and latency counters.

        Returns:
            ``score_buckets`` maps each category to 20 counts for ``[0, 0.05)``
            through ``[0.95, 1]``. ``classifier_latency_seconds`` sums the model
            calls that missed the cache.
        """
        return json.loads(self._native.telemetry_json())

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Flag the request when any category score reaches its threshold.

        A request with no classifiable text is allowed without a model call.

        Raises:
            ClassifierProtocolError: The decision call did not return a valid answer.
        """
        state = classification_window(request, self._document.window)
        if not state:
            return ClassifierVerdict(flagged=False)
        if self._unset_bearer:
            raise ClassifierProtocolError(_UNSET_BEARER)
        # The engine stops waiting at the check's timeout; bounding the native
        # call by it too keeps no call running past the engine's inflight lease.
        timeout_ms = min(self._document.timeout_ms, check.timeout_ms)
        reply = json.loads(await asyncio.to_thread(self._native.classify, state, timeout_ms))
        if "error" in reply:
            _logger.info(
                "guardrail classification adapter_id=%s check_id=%s cache=miss error=%s",
                self._document.adapter_id,
                check.check_id,
                reply["error"],
            )
            raise ClassifierProtocolError(reply["error"])
        answer = reply["answer"]
        result = ClassificationResult(
            adapter_id=self._document.adapter_id,
            check_id=check.check_id,
            configuration_digest=self._digest,
            probabilities=tuple(answer["probabilities"].items()),
            scores=tuple(answer["scores"].items()),
            flagged=tuple(answer["flagged"]),
            cached=answer["cached"],
            latency_seconds=answer["latency_ms"] / 1000.0,
        )
        _logger.info(
            "guardrail classification adapter_id=%s check_id=%s cache=%s latency_ms=%.1f "
            "score_buckets=%s",
            self._document.adapter_id,
            check.check_id,
            "hit" if result.cached else "miss",
            answer["latency_ms"],
            ",".join(
                f"{name}:{score_bucket(value) / SCORE_BUCKETS:.2f}" for name, value in result.scores
            ),
        )
        return ClassificationVerdict(flagged=bool(result.flagged), classification=result)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Refuse output inspection: classification checks are input-only.

        Raises:
            ClassifierCoverageError: Always.
        """
        del completion, check
        raise ClassifierCoverageError

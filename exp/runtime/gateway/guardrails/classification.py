"""Classification checks that score a bounded request window with a decision model.

One SystemOne call asks every configured question in a single forward pass:
``POST <url>`` with ``{"model", "state", "questions"}``, answered by
``{"answers": {"<id>": {"type": "noul", "noul": <probability>}}}``. Question
probabilities combine into operator categories, and a request is flagged when
any category score reaches its threshold. A self-hosted SGLang server running a
Clef model serves this wire, as do hosted SystemOne endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from hashlib import sha256
from typing import Final, Literal

import httpx
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
from exp.runtime.gateway.guardrails.classification_cache import (
    CategoryScores,
    VerdictCache,
    VerdictCacheSettings,
)
from exp.runtime.gateway.guardrails.classification_window import (
    ClassificationWindow,
    classification_window,
    window_projection,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)
from exp.runtime.gateway.guardrails.http_json import (
    DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES,
    ClassifierProtocolError,
    bearer_authorization_header,
    read_bounded_body,
    shared_http_json_client,
    validate_bearer_env_name,
    validate_classifier_url,
)

_logger = logging.getLogger(__name__)

DEFAULT_DECISION_TIMEOUT_MS: Final = 2_000
SCORE_BUCKETS: Final = 20
"""Histogram buckets per category, each 0.05 wide."""
_STATE_BYTES_PER_CHARACTER: Final = 6
"""Worst-case JSON bytes for one character: a ``\\u00XX`` control-character escape."""
_STATE_LABEL_BYTES: Final = 1_024


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

    def score(self, probabilities: dict[str, float]) -> float:
        """Combine this category's question probabilities into one score."""
        values = [probabilities[question_id] for question_id in self.questions]
        return max(values) if self.aggregate == "max" else math.fsum(values) / len(values)


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
                "decision_model questions and window exceed the "
                f"{MAX_DECISION_INPUT_BYTES}-byte decision input limit; "
                "shorten questions or reduce the window"
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


class ClassificationTelemetry:
    """Content-free counters for one adapter: score buckets, cache use, and latency.

    Category names are operator configuration. Nothing here holds request text,
    question text, or classifier payloads.
    """

    def __init__(self, categories: tuple[str, ...]) -> None:
        """Start every counter at zero.

        Args:
            categories: Configured category names, in authored order.
        """
        self._lock = threading.Lock()
        self._buckets = {name: [0] * SCORE_BUCKETS for name in categories}
        self._counts = {"cache_hits": 0, "cache_misses": 0, "errors": 0}
        self._latency_seconds = 0.0

    def record(self, scores: CategoryScores, *, cached: bool, latency_seconds: float) -> None:
        """Count one definitive answer, from the cache or from the model."""
        with self._lock:
            self._counts["cache_hits" if cached else "cache_misses"] += 1
            if not cached:
                self._latency_seconds += latency_seconds
            for name, value in scores:
                self._buckets[name][score_bucket(value)] += 1

    def record_error(self, latency_seconds: float) -> None:
        """Count one classifier call that ended without a definitive answer."""
        with self._lock:
            self._counts["cache_misses"] += 1
            self._counts["errors"] += 1
            self._latency_seconds += latency_seconds

    def snapshot(self) -> dict[str, JsonValue]:
        """Return the counters as content-free JSON.

        Returns:
            ``score_buckets`` maps each category to 20 counts for ``[0, 0.05)``
            through ``[0.95, 1]``. ``classifier_latency_seconds`` sums the model
            calls that missed the cache.
        """
        with self._lock:
            buckets: dict[str, JsonValue] = {
                name: list(counts) for name, counts in self._buckets.items()
            }
            return {
                **self._counts,
                "classifier_latency_seconds": self._latency_seconds,
                "score_buckets": buckets,
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
        client: httpx.AsyncClient | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind one validated adapter document.

        Args:
            document: Endpoint, model, categories, window, and cache settings.
            client: Optional injected client. ``None`` uses the shared pool.
            monotonic: Clock for cache expiry and latency, in seconds.
        """
        self._document = document
        self._questions = document.questions
        self._client = client
        self._monotonic = monotonic
        self._cache = VerdictCache(document.cache, monotonic=monotonic)
        self._telemetry = ClassificationTelemetry(tuple(document.categories))
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

    def project_input(self, request: GatewayRequest) -> GatewayRequest:
        """Return the bounded subject this check reads identically to ``request``."""
        return window_projection(request)

    @property
    def configuration_digest(self) -> str:
        """Return the digest of the endpoint, model, categories, and window."""
        return self._digest

    def telemetry(self) -> dict[str, JsonValue]:
        """Return this adapter's content-free score, cache, and latency counters."""
        return self._telemetry.snapshot()

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
        key = sha256(self._digest.encode() + b"\n" + state.encode("utf-8")).digest()
        started = self._monotonic()
        scores = self._cache.get(key)
        cached = scores is not None
        if scores is None:
            try:
                probabilities = await self._decide(state)
            except ClassifierProtocolError as exc:
                latency = self._monotonic() - started
                self._telemetry.record_error(latency)
                _logger.info(
                    "guardrail classification adapter_id=%s check_id=%s cache=miss "
                    "latency_ms=%.1f error=%s",
                    self._document.adapter_id,
                    check.check_id,
                    latency * 1000.0,
                    exc,
                )
                raise
            scores = tuple(
                (name, category.score(probabilities))
                for name, category in self._document.categories.items()
            )
            self._cache.put(key, scores)
        latency = self._monotonic() - started
        self._telemetry.record(scores, cached=cached, latency_seconds=latency)
        _logger.info(
            "guardrail classification adapter_id=%s check_id=%s cache=%s latency_ms=%.1f "
            "score_buckets=%s",
            self._document.adapter_id,
            check.check_id,
            "hit" if cached else "miss",
            latency * 1000.0,
            ",".join(f"{name}:{score_bucket(value) / SCORE_BUCKETS:.2f}" for name, value in scores),
        )
        categories = self._document.categories
        return ClassifierVerdict(
            flagged=any(value >= categories[name].threshold for name, value in scores)
        )

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

    async def _decide(self, state: str) -> dict[str, float]:
        """POST one SystemOne request and return every question's probability.

        Raises:
            ClassifierProtocolError: The call failed or its answer drifted.
        """
        body = DecisionRequest(state=state, questions=self._questions).provider_body(
            self._document.model
        )
        headers = {"accept": "application/json", "content-type": "application/json"}
        authorization = bearer_authorization_header(self._document.bearer_env)
        if authorization is not None:
            headers["authorization"] = authorization
        client = self._client or shared_http_json_client()
        try:
            async with client.stream(
                "POST",
                self._document.url,
                content=canonical_json_bytes(body),
                headers=headers,
                timeout=self._document.timeout_ms / 1000.0,
            ) as response:
                if response.status_code != 200:
                    raise ClassifierProtocolError(
                        f"decision model returned HTTP {response.status_code}"
                    )
                raw = await read_bounded_body(response, self._document.max_response_bytes)
        except ClassifierProtocolError:
            raise
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as exc:
            raise ClassifierProtocolError("decision model timed out") from exc
        except httpx.HTTPError as exc:
            raise ClassifierProtocolError("decision model transport failed") from exc
        return answer_probabilities(raw, tuple(self._questions))


def answer_probabilities(raw: bytes, question_ids: tuple[str, ...]) -> dict[str, float]:
    """Parse a SystemOne answer body into one probability per asked question.

    Args:
        raw: Bounded response bytes.
        question_ids: Every question identity that was asked.

    Returns:
        Question identity to its probability in ``[0, 1]``.

    Raises:
        ClassifierProtocolError: The body is not strict JSON, repeats a key, lacks
            an asked question, or carries a non-probability answer.
    """
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_no_constant)
    except (ValueError, RecursionError) as exc:
        raise ClassifierProtocolError("decision model answer is not valid JSON") from exc
    answers = parsed.get("answers") if isinstance(parsed, dict) else None
    if not isinstance(answers, dict):
        raise ClassifierProtocolError("decision model answer has no answers object")
    probabilities: dict[str, float] = {}
    for question_id in question_ids:
        answer = answers.get(question_id)
        if not isinstance(answer, dict):
            raise ClassifierProtocolError("decision model answer is missing a question")
        if answer.get("type", "noul") != "noul":
            raise ClassifierProtocolError("decision model answer is not a noul probability")
        value = answer.get("noul")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ClassifierProtocolError("decision model probability is outside [0, 1]")
        probabilities[question_id] = float(value)
    return probabilities


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Build one JSON object and reject a repeated key as ambiguous."""
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("repeated JSON object key")
        result[key] = value
    return result


def _no_constant(value: str) -> JsonValue:
    """Reject JSON's nonstandard ``NaN`` and infinity literals."""
    raise ValueError(f"non-finite JSON value {value}")

"""Async HTTP client for the SystemOne decision wire.

SystemOne answers typed questions about one state. A ``noul`` question returns the
probability that a proposition is true, a ``choice`` question picks one of its named
categories, and a ``score`` question rates the state against ordered levels. The wire is
one JSON ``POST {base_url}/systemone`` whose body is exactly ``{"model", "state",
"questions"}``, so one client speaks to any endpoint that serves it: a local SGLang server
(``http://127.0.0.1:<port>/v1``), TypeSafe (``https://api.typesafe.ai/v1``), or an
Experiential gateway's ``/v1/systemone`` route.

Requests are validated with the gateway's decision contract before any network work.
Responses are checked the way the gateway checks provider answers: every question answered
exactly once with its type, probabilities in ``[0, 1]`` that sum to one (allowing only the
rounding envelope of hundredth-valued bins), a choice that can be the most probable category,
a score equal to its probability-weighted level, and usage that satisfies the decision usage
contract. Any violation raises a typed error instead of producing a default value.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import sys
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, SecretStr, ValidationError

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    ChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionUsage,
    NoulQuestion,
    ScoreQuestion,
)
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    RequestDeadline,
    run_with_retry_async,
)
from exp.runtime.models.providers.transport import (
    ProviderTransportError,
    RetryClassification,
    RetryPolicy,
    propagate_request_attempt_evidence,
)

SystemOneQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion

_PROBABILITY_TOLERANCE = 1e-6
_RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})
_SINGLE_ATTEMPT = RetryPolicy(maximum_attempts=1)
_STATUS_HINTS = {
    400: "the endpoint rejected the request; check the model id and question definitions",
    401: "the endpoint rejected the credentials; check api_key",
    403: "the credentials are not allowed to use this model; check api_key and model",
    404: "no SystemOne route there; set base_url to the API root (for example "
    "http://127.0.0.1:30000/v1) of a server that serves POST /systemone",
    429: "the endpoint is rate limiting; lower max_concurrency or retry later",
}


class SystemOneError(Exception):
    """Base class for SystemOne client failures; each message says what to do next."""


class SystemOneRequestError(SystemOneError, ValueError):
    """The decision request breaks the SystemOne contract and was never sent."""


class SystemOneConnectionError(SystemOneError):
    """The endpoint could not be reached or dropped the connection mid-request."""


class SystemOneTimeoutError(SystemOneError, TimeoutError):
    """The decision did not finish within the client's per-call timeout."""


class SystemOneHttpError(SystemOneError):
    """The endpoint answered with a non-success HTTP status.

    Attributes:
        status_code: HTTP status of the final attempt.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        """Keep the sanitized message and the status for programmatic handling.

        Args:
            message: Content-free description with a suggested fix.
            status_code: HTTP status of the final attempt.
        """
        super().__init__(message)
        self.status_code = status_code


class SystemOneResponseError(SystemOneError, ValueError):
    """A successful HTTP response whose decisions do not match the request."""


class _MalformedAnswer(ValueError):
    """Internal reason a response failed validation, wrapped with endpoint context."""


class ChoiceAnswer(ContractModel):
    """One validated answer to a choice question.

    Attributes:
        choice: Selected category, always one of the question's criteria keys.
        confidence: Endpoint-reported confidence in ``[0, 1]``.
        probabilities: Probability of every requested category, in question order.
    """

    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(ContractModel):
    """One validated answer to a score question.

    Attributes:
        score: Probability-weighted level index, between zero and the last level.
        confidence: Endpoint-reported confidence in ``[0, 1]``.
        probabilities: Probability of each ordered level, indexed like the criteria.
    """

    score: float
    confidence: float
    probabilities: tuple[float, ...]


class DecisionResult(ContractModel):
    """Validated decisions for every question in one request.

    Attributes:
        served_model: Model identity the endpoint reported. A gateway reports its alias.
        noul: Probability that each noul question's proposition is true, by question id.
        choice: Answers to choice questions, by question id.
        score: Answers to score questions, by question id.
        usage: Endpoint-reported input and output token counts for the whole request.
    """

    served_model: str
    noul: dict[str, float]
    choice: dict[str, ChoiceAnswer]
    score: dict[str, ScoreAnswer]
    usage: DecisionUsage


class SystemOneClient:
    """Send bounded, validated SystemOne decision requests to one endpoint and model.

    Example:
        ``client = SystemOneClient("http://127.0.0.1:30000/v1", "clef")`` then
        ``result = await client.decide(text, {"coding": NoulQuestion(instructions=...)})``
        and read ``result.noul["coding"]``.

    The client is safe to share between tasks on one event loop. At most
    ``max_concurrency`` requests are in flight; further calls wait for a slot, and the
    timeout starts once a call holds one.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: SecretStr | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 8,
        retry_policy: RetryPolicy = _SINGLE_ATTEMPT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Bind one endpoint without network work.

        Args:
            base_url: API root; requests go to ``{base_url}/systemone``.
            model: Model id sent in every request body.
            api_key: Bearer credential, or None for an endpoint without authentication.
            timeout: Seconds one call may spend on attempts and backoff after admission.
            max_concurrency: Maximum requests this client keeps in flight at once.
            retry_policy: Attempt bound for connection failures and HTTP 5xx. The default
                makes one attempt; other failures are never retried.
            http_client: Caller-owned HTTPX client; defaults to the shared pooled client.

        Raises:
            ValueError: The URL, model, credential, timeout, or concurrency is unusable.
        """
        parsed = urlsplit(base_url)
        try:
            # Reading the port validates it; an unusable one would otherwise fail inside HTTPX.
            valid_port = parsed.port is None or parsed.port > 0
        except ValueError:
            valid_port = False
        if (
            not valid_port
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "SystemOne base_url must be an http or https API root with a valid port and no "
                "credentials, query, or fragment, for example http://127.0.0.1:30000/v1"
            )
        if api_key is not None and not api_key.get_secret_value():
            raise ValueError("SystemOne api_key is empty; pass None for an endpoint without auth")
        if api_key is not None and parsed.scheme == "http" and not _loopback(parsed.hostname):
            raise ValueError(
                "refusing to send a SystemOne api_key over plain http to a non-loopback host; "
                "use an https base_url or a loopback address"
            )
        if not model or len(model) > 512:
            raise ValueError("SystemOne model must contain 1 through 512 characters")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SystemOne timeout must be a finite positive number of seconds")
        if max_concurrency < 1:
            raise ValueError("SystemOne max_concurrency must be at least one")
        self._url = f"{base_url.rstrip('/')}/systemone"
        self._model = model
        self._timeout = timeout
        self._max_concurrency = max_concurrency
        self._retry_policy = retry_policy
        self._transport = HttpxAsyncJsonTransport(http_client)
        self._headers = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        self._slots = asyncio.Semaphore(max_concurrency)

    @property
    def url(self) -> str:
        """Return the exact decision endpoint this client posts to."""
        return self._url

    @property
    def max_concurrency(self) -> int:
        """Return the in-flight request bound."""
        return self._max_concurrency

    async def decide(
        self, state: str, questions: Mapping[str, SystemOneQuestion]
    ) -> DecisionResult:
        """Ask every question about one state in a single request.

        Args:
            state: Text the questions are about.
            questions: 1 through 32 questions keyed by stable question id.

        Returns:
            Validated answers for every question, the served model, and token usage.

        Raises:
            SystemOneRequestError: The request breaks the decision contract; nothing was sent.
            SystemOneConnectionError: The endpoint could not be reached.
            SystemOneTimeoutError: The call exceeded its timeout.
            SystemOneHttpError: The endpoint returned a non-success status.
            SystemOneResponseError: The response does not answer the request correctly.
        """
        request = _validated_request(state, questions)
        async with self._slots:
            body = await self._post(request.provider_body(self._model))
        try:
            return _decision_result(body, request.questions)
        except _MalformedAnswer as error:
            raise SystemOneResponseError(
                f"SystemOne response from {self._url} is malformed: {error}. The endpoint did "
                f"not follow the SystemOne decision wire; check that base_url points at a "
                f"SystemOne server for model {self._model!r}"
            ) from None

    async def _post(self, payload: JsonObject) -> JsonObject:
        """Send one request under the timeout and translate every failure to a typed error.

        Args:
            payload: Exact wire body.

        Returns:
            The decoded body of a 2xx response.

        Raises:
            SystemOneError: A typed connection, timeout, HTTP, or response failure.
        """
        attempts = 0

        async def send(timeout_seconds: float) -> JsonObject:
            """Run one attempt and raise a status-bearing error for non-2xx responses."""
            nonlocal attempts
            attempts += 1
            response = await self._transport.post(
                self._url, headers=self._headers, payload=payload, timeout_seconds=timeout_seconds
            )
            if 200 <= response.status_code < 300:
                return response.body
            raise ProviderTransportError(
                f"provider returned HTTP {response.status_code}",
                status_code=response.status_code,
                retry_after_seconds=response.retry_after_seconds,
            )

        try:
            return await run_with_retry_async(
                send,
                policy=self._retry_policy,
                deadline=RequestDeadline.after(self._timeout),
                classify=_classify_retry,
            )
        except (ProviderTransportError, TimeoutError) as error:
            failure = self._failure(error, attempts)
            propagate_request_attempt_evidence(error, failure)
            raise failure from error

    def _failure(
        self, error: ProviderTransportError | TimeoutError, attempts: int
    ) -> SystemOneError:
        """Describe one terminal transport failure with what went wrong and what to do.

        Args:
            error: Final failure from the retry loop.
            attempts: Attempts started for this call.

        Returns:
            The typed client error to raise.
        """
        tried = f" after {attempts} attempts" if attempts > 1 else ""
        if _timed_out(error):
            return SystemOneTimeoutError(
                f"SystemOne decision at {self._url} did not finish within {self._timeout:g}s"
                f"{tried}; raise timeout, lower max_concurrency, or check the server's load"
            )
        status = error.status_code if isinstance(error, ProviderTransportError) else None
        if status is None:
            return SystemOneConnectionError(
                f"could not reach the SystemOne endpoint {self._url}{tried} ({error}); check "
                "that the server is running and base_url is correct"
            )
        if 200 <= status < 300:
            return SystemOneResponseError(
                f"SystemOne endpoint {self._url} returned HTTP {status} without a JSON object; "
                "check that base_url points at a SystemOne server"
            )
        hint = _STATUS_HINTS.get(status) or (
            "the server failed; check its logs, or pass a retry_policy to retry server errors"
            if status >= 500
            else "the endpoint refused the request; check the model id, state size, and questions"
        )
        return SystemOneHttpError(
            f"SystemOne endpoint {self._url} returned HTTP {status}{tried}: {hint}",
            status_code=status,
        )


def _loopback(hostname: str) -> bool:
    """Return whether a hostname names this machine, where plain http stays local."""
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _timed_out(error: BaseException) -> bool:
    """Return whether a failure, or an exception it was explicitly raised from, is a timeout.

    Only ``__cause__`` is followed: implicit ``__context__`` can name an unrelated timeout
    the caller was handling when it called ``decide``.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, httpx.TimeoutException)):
            return True
        current = current.__cause__
    return False


def _classify_retry(error: Exception) -> RetryClassification:
    """Retry only connection failures and HTTP 5xx; timeouts and refusals stay terminal."""
    if isinstance(error, ProviderTransportError) and not _timed_out(error):
        if error.status_code is None:
            return RetryClassification(retryable=True, reason="connection")
        if error.status_code in _RETRYABLE_STATUS_CODES:
            return RetryClassification(retryable=True, reason=f"http_{error.status_code}")
    return RetryClassification(retryable=False, reason="terminal")


def _validated_request(state: str, questions: Mapping[str, SystemOneQuestion]) -> DecisionRequest:
    """Validate a request with the gateway contract, reporting errors without content.

    Args:
        state: Text the questions are about.
        questions: Questions keyed by question id.

    Returns:
        The validated decision request.

    Raises:
        SystemOneRequestError: The state or questions break the contract.
    """
    try:
        return DecisionRequest(state=state, questions=dict(questions))
    except ValidationError as error:
        reasons = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'request'}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        raise SystemOneRequestError(
            f"SystemOne request is invalid ({reasons}); send 1 through 32 questions and a "
            f"state whose request stays under {MAX_DECISION_INPUT_BYTES} bytes"
        ) from None


def _decision_result(body: JsonObject, questions: Mapping[str, DecisionQuestion]) -> DecisionResult:
    """Validate a response body against the questions that were asked.

    Args:
        body: Decoded 2xx response body.
        questions: Validated questions from the request.

    Returns:
        Typed answers, served model, and usage.

    Raises:
        _MalformedAnswer: Any answer, usage count, or model field is invalid.
    """
    served_model = body.get("model")
    if not isinstance(served_model, str) or not served_model:
        raise _MalformedAnswer("the response does not name its served model")
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise _MalformedAnswer("the response omitted its answers object")
    missing = [question_id for question_id in questions if question_id not in answers]
    unexpected = sum(answer_id not in questions for answer_id in answers)
    if missing or unexpected:
        raise _MalformedAnswer(
            f"answer ids do not match the questions (missing {missing[:5]}, "
            f"{unexpected} unexpected)"
        )
    noul: dict[str, float] = {}
    choice: dict[str, ChoiceAnswer] = {}
    score: dict[str, ScoreAnswer] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        where = f"answer {question_id!r}"
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            raise _MalformedAnswer(f"{where} is not a {question.type} answer")
        match question:
            case NoulQuestion():
                noul[question_id] = _probability(answer.get("noul"), f"{where} noul")
            case ChoiceQuestion():
                choice[question_id] = _choice_answer(answer, question, where)
            case ScoreQuestion():
                score[question_id] = _score_answer(answer, question, where)
    return DecisionResult(
        served_model=served_model, noul=noul, choice=choice, score=score, usage=_usage(body)
    )


def _choice_answer(answer: JsonObject, question: ChoiceQuestion, where: str) -> ChoiceAnswer:
    """Validate a choice answer's categories, total, confidence, and selection."""
    names = list(question.criteria)
    probabilities = _distribution(answer.get("probabilities"), names, where)
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in question.criteria:
        raise _MalformedAnswer(f"{where} choice is not a requested category")
    confidence = _probability(answer.get("confidence"), f"{where} confidence")
    if not _choice_can_be_highest(names.index(selected), probabilities):
        raise _MalformedAnswer(f"{where} choice is not a highest-probability category")
    return ChoiceAnswer(
        choice=selected,
        confidence=confidence,
        probabilities=dict(zip(names, probabilities, strict=True)),
    )


def _score_answer(answer: JsonObject, question: ScoreQuestion, where: str) -> ScoreAnswer:
    """Validate a score answer's legend, distribution, confidence, and weighted score."""
    criteria = list(question.criteria)
    legend: JsonObject = {str(index): level for index, level in enumerate(criteria)}
    # Compare canonical JSON so true never equals 1 and 1 never equals 1.0, as on the gateway.
    if json.dumps(answer.get("legend"), sort_keys=True) != json.dumps(legend, sort_keys=True):
        raise _MalformedAnswer(f"{where} legend does not match the requested criteria")
    probabilities = _distribution(answer.get("probabilities"), list(legend), where)
    confidence = _probability(answer.get("confidence"), f"{where} confidence")
    value = answer.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _MalformedAnswer(f"{where} score is not numeric data")
    if not 0 <= value <= len(criteria) - 1 or not _score_matches(float(value), probabilities):
        raise _MalformedAnswer(f"{where} score does not match its probability distribution")
    return ScoreAnswer(
        score=float(value), confidence=confidence, probabilities=tuple(probabilities)
    )


def _usage(body: JsonObject) -> DecisionUsage:
    """Read the endpoint's input and output token counts with the decision usage contract."""
    usage = body.get("usage")
    names = ("input_tokens", "output_tokens")
    counts = {name: usage.get(name) for name in names} if isinstance(usage, dict) else {}
    try:
        return DecisionUsage.model_validate(counts)
    except ValidationError:
        raise _MalformedAnswer(
            "usage must report integer input_tokens and output_tokens, at least one above zero"
        ) from None


def _probability(value: JsonValue, what: str) -> float:
    """Return a probability in ``[0, 1]``; the range check also rejects NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise _MalformedAnswer(f"{what} must be a finite number between 0 and 1")
    return float(value)


def _distribution(value: JsonValue, keys: Sequence[str], where: str) -> list[float]:
    """Return probabilities in key order after checking exact keys and a unit total."""
    if not isinstance(value, dict):
        raise _MalformedAnswer(f"{where} omitted its probabilities object")
    probabilities = [_probability(value.get(key), f"{where} probability") for key in keys]
    if len(value) != len(keys) or not _unit_total(probabilities):
        raise _MalformedAnswer(f"{where} probability keys or total do not match the request")
    return probabilities


def _unit_total(probabilities: Sequence[float]) -> bool:
    """Accept a total of one, or a total explained by hundredth rounding of every bin."""
    if abs(sum(probabilities) - 1.0) <= _PROBABILITY_TOLERANCE:
        return True
    bounds = _rounding_bounds(probabilities)
    return bounds is not None and _admits_unit_total(bounds)


def _hundredths(value: float) -> int | None:
    """Return a value in hundredths when it is decimal hundredths, allowing float residue."""
    return _decimal_units(value, 100)


def _decimal_units(value: float, scale: int) -> int | None:
    """Return a value in units of ``1 / scale`` when it has at most that decimal precision.

    Args:
        value: The number as reported.
        scale: 10, 100 or 1000 for tenths, hundredths or thousandths.

    Returns:
        The value in those units, allowing binary floating-point residue, or None when it
        carries finer precision.
    """
    scaled = value * scale
    rounded = round(scaled)
    roundoff = 2.0 * sys.float_info.epsilon * abs(scaled)
    return rounded if abs(scaled - rounded) <= roundoff else None


def _rounding_bounds(probabilities: Sequence[float]) -> list[tuple[int, int]] | None:
    """Return each bin's closed rounding interval in half-hundredths, or None if finer."""
    bounds: list[tuple[int, int]] = []
    for value in probabilities:
        bin_hundredths = _hundredths(value)
        if bin_hundredths is None:
            return None
        bounds.append((max(2 * bin_hundredths - 1, 0), min(2 * bin_hundredths + 1, 200)))
    return bounds


def _admits_unit_total(bounds: Sequence[tuple[int, int]]) -> bool:
    """Return whether one unit distribution fits inside every rounding interval."""
    return sum(low for low, _ in bounds) <= 200 <= sum(high for _, high in bounds)


def _choice_can_be_highest(selected: int, probabilities: Sequence[float]) -> bool:
    """Return whether the selected bin is, or could be before rounding, the highest."""
    if all(value <= probabilities[selected] for value in probabilities):
        return True
    bounds = _rounding_bounds(probabilities)
    if bounds is None:
        return False
    low, high = bounds[selected]
    for level in range(low, high + 1):
        others = [bound for index, bound in enumerate(bounds) if index != selected]
        if any(other_low > level for other_low, _ in others):
            continue
        lower_sum = level + sum(other_low for other_low, _ in others)
        upper_sum = level + sum(min(other_high, level) for _, other_high in others)
        if lower_sum <= 200 <= upper_sum:
            return True
    return False


def _score_matches(score: float, probabilities: Sequence[float]) -> bool:
    """Return whether a score equals the weighted level, within rounding of score and bins.

    The score stands for half a unit either side of the last decimal place it was reported
    to (tenths at the coarsest, so a whole number reads at tenths; thousandths at the finest).
    It matches when that interval meets the expectations of every unit distribution whose
    bins round to the reported hundredth bins, as the gateway checks it.
    """
    expected = sum(index * value for index, value in enumerate(probabilities))
    if abs(score - expected) <= _PROBABILITY_TOLERANCE:
        return True
    precision = next(
        (
            (units, scale)
            for scale in (10, 100, 1000)
            if (units := _decimal_units(score, scale)) is not None
        ),
        None,
    )
    bounds = _rounding_bounds(probabilities)
    if precision is None or bounds is None or not _admits_unit_total(bounds):
        return False
    units, scale = precision
    spare = 200 - sum(low for low, _ in bounds)
    base = sum(index * low for index, (low, _) in enumerate(bounds))
    indexed = list(enumerate(bounds))
    minimum = _extreme_expectation(base, spare, indexed)
    maximum = _extreme_expectation(base, spare, indexed[::-1])
    # The envelope is in half-hundredths; the score interval is [(2u-1)/(2s), (2u+1)/(2s)].
    return (2 * units + 1) * 100 >= minimum * scale and (2 * units - 1) * 100 <= maximum * scale


def _extreme_expectation(
    value: int, remaining: int, bins: Sequence[tuple[int, tuple[int, int]]]
) -> int:
    """Extremize a weighted sum by filling bins in the given index order first."""
    for index, (low, high) in bins:
        added = min(remaining, high - low)
        value += index * added
        remaining -= added
    return value

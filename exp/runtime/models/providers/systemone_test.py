"""Offline tests for the SystemOne decision client, its wire body, and answer validation."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping

import httpx
import pytest
from pydantic import JsonValue, SecretStr

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    ChoiceQuestion,
    DecisionUsage,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)
from exp.runtime.models.providers.systemone import (
    ChoiceAnswer,
    DecisionResult,
    SystemOneClient,
    SystemOneConnectionError,
    SystemOneHttpError,
    SystemOneQuestion,
    SystemOneRequestError,
    SystemOneResponseError,
    SystemOneTimeoutError,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.models.providers.typesafe import TYPESAFE_BASE_URL, TypeSafeClient

_BASE_URL = "http://127.0.0.1:30000/v1"
_ONE_ATTEMPT = RetryPolicy(maximum_attempts=1)
_RETRY_THREE = RetryPolicy(maximum_attempts=3, initial_delay_seconds=0, maximum_delay_seconds=0)
_STATE = "[user] fix the failing test"
_NOUL_QUESTIONS: dict[str, SystemOneQuestion] = {
    "coding": NoulQuestion(
        instructions="Is this about code?", criteria=NoulCriteria(true="code", false="no code")
    ),
    "refused": NoulQuestion(instructions="Did the assistant refuse?"),
}
_MIXED_QUESTIONS: dict[str, SystemOneQuestion] = {
    "coding": NoulQuestion(instructions="Is this about code?"),
    "department": ChoiceQuestion(
        instructions="Choose the department",
        criteria={"billing": "Invoices", "technical": "Bugs", "sales": None},
    ),
    "quantity": ScoreQuestion(instructions="Count items", criteria=("None", "Some", "Many")),
}


def _noul_body() -> JsonObject:
    """Return a valid response to the noul questions."""
    return {
        "model": "clef-27b",
        "answers": {
            "coding": {"type": "noul", "noul": 0.93},
            "refused": {"type": "noul", "noul": 0.02},
        },
        "usage": {"input_tokens": 451, "output_tokens": 6},
    }


def _mixed_body() -> JsonObject:
    """Return a valid mixed response that needs the gateway's hundredth rounding envelope."""
    return {
        "model": "jev-1.13.0",
        "answers": {
            "coding": {"type": "noul", "noul": 1},
            "department": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.6,
                "probabilities": {"billing": 0.34, "technical": 0.33, "sales": 0.33},
            },
            "quantity": {
                "type": "score",
                "score": 1.0,
                "confidence": 0.9,
                "legend": {"0": "None", "1": "Some", "2": "Many"},
                "probabilities": {"0": 0.33, "1": 0.33, "2": 0.33},
            },
        },
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def _ok(body: JsonObject) -> httpx.Response:
    """Return a 200 response whose bytes are the body's JSON, NaN literals included."""
    return httpx.Response(200, content=json.dumps(body).encode())


def _replay(*answers: httpx.Response | Exception) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    """Return an HTTPX client that serves answers in order and records every request."""
    pending = list(answers)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the next answer, failing the test on an unexpected request."""
        seen.append(request)
        if not pending:
            raise AssertionError("test made an unexpected SystemOne request")
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _client(
    http_client: httpx.AsyncClient, *, retry_policy: RetryPolicy = _ONE_ATTEMPT
) -> SystemOneClient:
    """Build an authenticated loopback client over a mock HTTPX client."""
    return SystemOneClient(
        _BASE_URL,
        "clef-27b",
        api_key=SecretStr("secret-canary"),
        retry_policy=retry_policy,
        http_client=http_client,
    )


def _decide(client: SystemOneClient, questions: Mapping[str, SystemOneQuestion]) -> DecisionResult:
    """Run one decision on a fresh event loop."""
    return asyncio.run(client.decide(_STATE, questions))


def _answered(
    questions: Mapping[str, SystemOneQuestion], response: httpx.Response
) -> DecisionResult:
    """Decide against an endpoint that answers once with the given response."""
    http_client, _ = _replay(response)
    return _decide(_client(http_client), questions)


def _set(body: JsonObject, path: tuple[str, ...], value: JsonValue) -> JsonObject:
    """Return a deep copy of a body with one nested field replaced."""
    changed = copy.deepcopy(body)
    target = changed
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value
    return changed


def _accepts(question: SystemOneQuestion, answer: JsonObject) -> bool:
    """Return whether the client accepts one answer to a single-question request."""
    usage: JsonObject = {"input_tokens": 1, "output_tokens": 1}
    try:
        _answered({"q": question}, _ok({"model": "m", "answers": {"q": answer}, "usage": usage}))
    except SystemOneResponseError:
        return False
    return True


def test_decide_posts_the_exact_wire_body_and_returns_probabilities() -> None:
    """The request carries only model, state, and questions plus bearer auth."""
    http_client, seen = _replay(_ok(_noul_body()))
    client = SystemOneClient(
        f"{_BASE_URL}/", "clef-27b", api_key=SecretStr("secret-canary"), http_client=http_client
    )
    assert _decide(client, _NOUL_QUESTIONS) == DecisionResult(
        served_model="clef-27b",
        noul={"coding": 0.93, "refused": 0.02},
        choice={},
        score={},
        usage=DecisionUsage(input_tokens=451, output_tokens=6),
    )
    (request,) = seen
    assert (request.method, str(request.url)) == ("POST", f"{_BASE_URL}/systemone")
    assert request.headers["authorization"] == "Bearer secret-canary"
    assert json.loads(request.content) == {
        "model": "clef-27b",
        "state": _STATE,
        "questions": {
            "coding": {
                "type": "noul",
                "instructions": "Is this about code?",
                "criteria": {"true": "code", "false": "no code"},
            },
            "refused": {"type": "noul", "instructions": "Did the assistant refuse?"},
        },
    }


def test_endpoint_roots_share_the_systemone_route_and_optional_auth() -> None:
    """TypeSafe, gateway, and local roots post where TypeSafe's wire profile points."""
    snapshot = ModelSnapshot(
        provider="typesafe",
        model_id="systemone",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256=sha256_json({}),
        connection_sha256=sha256_json({"provider": "typesafe"}),
    )
    profile = TypeSafeClient(model=snapshot, api_key="secret-canary").gateway_wire_profile()
    assert SystemOneClient(TYPESAFE_BASE_URL, "systemone").url == profile.decisions_url
    gateway = SystemOneClient("http://localhost:8080/v1/", "tagger")
    assert gateway.url == "http://localhost:8080/v1/systemone"
    http_client, seen = _replay(_ok(_noul_body()))
    unauthenticated = SystemOneClient("http://10.0.0.5:30000/v1", "m", http_client=http_client)
    _decide(unauthenticated, _NOUL_QUESTIONS)
    assert "authorization" not in seen[0].headers


def test_choice_and_score_answers_are_returned_typed() -> None:
    """A rounded near tie and a rounded weighted score pass exactly as the gateway allows."""
    result = _answered(_MIXED_QUESTIONS, _ok(_mixed_body()))
    assert (result.served_model, result.noul) == ("jev-1.13.0", {"coding": 1.0})
    assert result.choice["department"] == ChoiceAnswer(
        choice="technical",
        confidence=0.6,
        probabilities={"billing": 0.34, "technical": 0.33, "sales": 0.33},
    )
    assert result.score["quantity"].score == 1.0
    assert result.score["quantity"].probabilities == (0.33, 0.33, 0.33)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 0)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("answers", "coding", "noul"), 1.2, "'coding' noul must be a finite number"),
        (("answers", "coding", "noul"), True, "'coding' noul must be a finite number"),
        (("answers", "coding", "noul"), "0.9", "'coding' noul must be a finite number"),
        (("answers", "coding", "noul"), float("nan"), "'coding' noul must be a finite number"),
        (("answers", "coding", "type"), "choice", "'coding' is not a noul answer"),
        (("answers", "coding"), 0.9, "'coding' is not a noul answer"),
        (
            ("answers", "department", "probabilities"),
            {"billing": 0.5, "technical": 0.5, "sales": 0.0, "extra": 0.0},
            "probability keys or total do not match",
        ),
        (
            ("answers", "department", "probabilities"),
            {"billing": 0.4, "technical": 0.6},
            "'department' probability must be a finite number",
        ),
        (("answers", "department", "choice"), "marketing", "not a requested category"),
        (("answers", "department", "confidence"), 2.0, "confidence must be a finite"),
        (("answers", "quantity", "score"), "1.0", "score is not numeric data"),
        (("answers", "quantity", "legend"), {"0": "None"}, "legend does not match"),
        (("usage",), {"input_tokens": 0, "output_tokens": 0}, "at least one above zero"),
        (("usage",), {"input_tokens": 1.0, "output_tokens": 2}, "usage must report integer"),
        (("usage",), {"input_tokens": 1, "output_tokens": -2}, "usage must report integer"),
        (("model",), "", "does not name its served model"),
        (("answers",), [], "omitted its answers object"),
    ],
)
def test_malformed_answers_raise_typed_errors(
    path: tuple[str, ...], value: JsonValue, reason: str
) -> None:
    """Every answer check fails loudly with a fix instead of yielding a default."""
    with pytest.raises(SystemOneResponseError, match=reason) as raised:
        _answered(_MIXED_QUESTIONS, _ok(_set(_mixed_body(), path, value)))
    assert "check that base_url points at a SystemOne server" in str(raised.value)


@pytest.mark.parametrize("content", [b"<html>Not Found</html>", b"[]"], ids=["text", "array"])
def test_successful_responses_without_a_json_object_are_typed(content: bytes) -> None:
    """A 2xx body that is not a JSON object means base_url points at something else."""
    with pytest.raises(SystemOneResponseError, match="without a JSON object"):
        _answered(_NOUL_QUESTIONS, httpx.Response(200, content=content))


@pytest.mark.parametrize(
    ("probabilities", "selected", "valid"),
    [
        ([0.17, 0.02, 0.07, 0.03, 0.41000000000000003, 0.19, 0.1], 4, True),
        ([0.01, 0.1, 0.01, 0.0, 0.11, 0.02, 0.01, 0.01, 0.67, 0.05], 8, True),
        ([0.05, 0.3, 0.1, 0.01, 0.01, 0.31, 0.04, 0.14, 0.01, 0.03], 1, True),
        ([0.34, 0.33, 0.32], 0, True),
        ([0.34, 0.34, 0.33], 1, True),
        ([0.0, 0.49, 0.5], 2, True),
        ([0.01, 0.49, 0.5], 1, True),
        ([0.0, 0.5, 0.5], 1, True),
        ([0.33, 0.33, 0.32], 0, False),
        ([0.34, 0.34, 0.34], 0, False),
        ([1.0, 0.01, 0.01], 0, False),
        ([0.335, 0.33, 0.33], 0, False),
        ([0.5 - 1e-12, 0.49, 0.0], 0, False),
        ([0.31, 0.33, 0.36], 1, False),
        ([0.0, 0.49, 0.5], 1, False),
        ([0.0, 0.495, 0.505], 1, False),
        ([0.6, 0.3, 0.1], 1, False),
        ([0.5000001, 0.4999999, 0.0], 1, False),
    ],
)
def test_choice_validation_matches_the_gateway_vectors(
    probabilities: list[float], selected: int, valid: bool
) -> None:
    """Hundredth rounding may close a near tie or a 0.99 total, never a real gap."""
    names = [f"c{index}" for index in range(len(probabilities))]
    question = ChoiceQuestion(instructions="Pick one", criteria=dict.fromkeys(names))
    answer: JsonObject = {
        "type": "choice",
        "choice": names[selected],
        "confidence": 0.3,
        "probabilities": dict(zip(names, probabilities, strict=True)),
    }
    assert _accepts(question, answer) is valid


@pytest.mark.parametrize(
    ("probabilities", "score", "valid"),
    [
        ([0.0, 0.87, 0.13, 0.0, 0.0], 1.12, True),
        ([0.01, 0.0, 0.01, 0.07, 0.91], 3.88, True),
        ([0.84, 0.12, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], 0.26, True),
        ([0.0, 0.07, 0.31, 0.43, 0.15, 0.03, 0.01, 0.0, 0.0, 0.0], 2.78, True),
        ([0.0, 0.870001, 0.129999], 1.129999, True),
        ([0.33, 0.33, 0.33], 1.0, True),
        ([1.0, 0.0, 0.0, 0.0, 0.0], 0.02, True),
        ([0.5, 0.5], 0.51, True),
        ([0.1, 0.3, 0.6], 1.5, True),
        ([1.0, 0.0, 0.0, 0.0, 0.0], 0.03, False),
        ([0.0, 0.87, 0.13, 0.0, 0.0], 1.5, False),
        ([0.0, 0.87, 0.13, 0.0, 0.0], 1.12001, False),
        ([0.0, 0.870001, 0.129999, 0.0, 0.0], 1.12, False),
        ([1e-12, 0.87 - 1e-12, 0.13, 0.0, 0.0], 1.12, False),
        ([0.33, 0.33, 0.32], 0.99, False),
        ([0.34, 0.34, 0.34, 0.0, 0.0], 1.02, False),
        ([0.0, 0.87, 0.13, 0.0, 0.0], -0.01, False),
        ([0.0, 0.87, 0.13, 0.0, 0.0], 4.01, False),
        ([0.5, 0.5], 0.52, False),
        ([0.1, 0.3, 0.6], 1.50001, False),
        ([0.33, 0.33, 0.33], 1.001, True),
        ([0.33, 0.33, 0.33], 0.995, True),
        ([0.33, 0.33, 0.33], 1.005, True),
        ([0.33, 0.33, 0.33], 1.006, False),
        ([0.33, 0.33, 0.33], 0.994, False),
        ([0.33, 0.33, 0.33], 1.0001, False),
        ([0.333, 0.333, 0.334], 1.001, True),
        ([0.333, 0.333, 0.334], 1.002, False),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 3.9, True),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 3.8, True),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 3.7, False),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 4.0, False),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 3.89, True),
        ([0.0, 0.01, 0.01, 0.08, 0.9], 3.91, False),
    ],
)
def test_score_validation_matches_the_gateway_vectors(
    probabilities: list[float], score: float, valid: bool
) -> None:
    """A score must equal its weighted level, within its own and its bins' rounding.

    The vectors are the gateway's, including a tenths-rounded TypeSafe score.
    """
    levels = tuple(f"Level {index}" for index in range(len(probabilities)))
    answer: JsonObject = {
        "type": "score",
        "score": score,
        "confidence": 0.5,
        "legend": {str(index): level for index, level in enumerate(levels)},
        "probabilities": {str(index): value for index, value in enumerate(probabilities)},
    }
    assert _accepts(ScoreQuestion(instructions="Rate", criteria=levels), answer) is valid


def test_structured_score_legends_compare_as_strict_json() -> None:
    """A legend that turns true into 1 is a different legend, as the gateway decides."""
    question = ScoreQuestion(
        instructions="Approved?", criteria=({"approved": True}, {"approved": False})
    )
    answer: JsonObject = {
        "type": "score",
        "score": 0.9,
        "confidence": 0.9,
        "legend": {"0": {"approved": True}, "1": {"approved": False}},
        "probabilities": {"0": 0.1, "1": 0.9},
    }
    assert _accepts(question, answer)
    assert not _accepts(question, _set(answer, ("legend", "0", "approved"), 1))


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        ({"coding": {"type": "noul", "noul": 0.9}}, r"missing \['refused'\], 0 unexpected"),
        (
            {
                "coding": {"type": "noul", "noul": 0.9},
                "refused": {"type": "noul", "noul": 0.1},
                "bonus": {"type": "noul", "noul": 0.5},
            },
            r"missing \[\], 1 unexpected",
        ),
    ],
)
def test_missing_and_extra_answers_are_rejected(answers: JsonObject, reason: str) -> None:
    """The response must answer exactly the questions that were asked."""
    with pytest.raises(SystemOneResponseError, match=reason):
        _answered(_NOUL_QUESTIONS, _ok(_set(_noul_body(), ("answers",), answers)))


@pytest.mark.parametrize(
    ("status", "hint"),
    [(400, "question definitions"), (401, "check api_key"), (404, "API root"), (429, "lower")],
)
def test_http_client_errors_are_typed_and_never_retried(status: int, hint: str) -> None:
    """A 4xx raises once with its status and a fix, even with retries enabled."""
    http_client, seen = _replay(httpx.Response(status, json={"error": "state-canary"}))
    with pytest.raises(SystemOneHttpError, match=hint) as raised:
        _decide(_client(http_client, retry_policy=_RETRY_THREE), _NOUL_QUESTIONS)
    assert raised.value.status_code == status
    assert len(seen) == 1
    assert "secret-canary" not in str(raised.value)
    assert "state-canary" not in str(raised.value)


def test_server_and_connection_failures_retry_only_under_an_explicit_policy() -> None:
    """HTTP 5xx and refused connections are terminal by default and bounded when allowed."""
    http_client, _ = _replay(httpx.Response(503, json={}))
    with pytest.raises(SystemOneHttpError, match="pass a retry_policy") as raised:
        _decide(_client(http_client), _NOUL_QUESTIONS)
    assert raised.value.status_code == 503
    http_client, _ = _replay(httpx.ConnectError("refused"))
    with pytest.raises(SystemOneConnectionError, match="check that the server is running"):
        _decide(_client(http_client), _NOUL_QUESTIONS)
    http_client, seen = _replay(
        httpx.Response(503, json={}), httpx.ConnectError("refused"), _ok(_noul_body())
    )
    result = _decide(_client(http_client, retry_policy=_RETRY_THREE), _NOUL_QUESTIONS)
    assert (result.noul["coding"], len(seen)) == (0.93, 3)
    http_client, _ = _replay(*(httpx.Response(500, json={}) for _ in range(3)))
    with pytest.raises(SystemOneHttpError, match="HTTP 500 after 3 attempts"):
        _decide(_client(http_client, retry_policy=_RETRY_THREE), _NOUL_QUESTIONS)


def test_a_caller_handling_a_timeout_keeps_http_classification_and_retries() -> None:
    """An unrelated timeout being handled by the caller never turns into this call's timeout."""
    unauthorized, _ = _replay(httpx.Response(401, json={}))
    recovering, seen = _replay(httpx.Response(503, json={}), _ok(_noul_body()))
    try:
        raise TimeoutError("caller's own timeout")
    except TimeoutError:
        with pytest.raises(SystemOneHttpError, match="check api_key"):
            _decide(_client(unauthorized), _NOUL_QUESTIONS)
        assert _decide(_client(recovering, retry_policy=_RETRY_THREE), _NOUL_QUESTIONS).noul
    assert len(seen) == 2


def test_timeouts_are_typed_and_never_retried() -> None:
    """A slow endpoint and an HTTPX read timeout both raise SystemOneTimeoutError once."""

    async def slow(request: httpx.Request) -> httpx.Response:
        """Answer only after the client deadline has passed."""
        del request
        await asyncio.sleep(10)
        raise AssertionError("the client timeout should have fired")

    slow_client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    client = SystemOneClient(_BASE_URL, "clef-27b", timeout=0.05, http_client=slow_client)
    with pytest.raises(SystemOneTimeoutError, match="did not finish within 0.05s"):
        _decide(client, _NOUL_QUESTIONS)
    http_client, seen = _replay(httpx.ReadTimeout("timed out"))
    with pytest.raises(SystemOneTimeoutError, match="raise timeout"):
        _decide(_client(http_client, retry_policy=_RETRY_THREE), _NOUL_QUESTIONS)
    assert len(seen) == 1


def test_max_concurrency_bounds_requests_in_flight() -> None:
    """Seven concurrent decisions never put more than three requests on the wire."""
    in_flight = 0
    peak = 0

    async def gated(request: httpx.Request) -> httpx.Response:
        """Hold each request open briefly while counting overlap."""
        del request
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _ok(_noul_body())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(gated))
    client = SystemOneClient(_BASE_URL, "m", max_concurrency=3, http_client=http_client)

    async def scenario() -> list[DecisionResult]:
        """Start every decision at once."""
        return await asyncio.gather(*(client.decide(f"s{i}", _NOUL_QUESTIONS) for i in range(7)))

    assert len(asyncio.run(scenario())) == 7
    assert (peak, client.max_concurrency) == (3, 3)


def test_invalid_requests_fail_before_any_network_work() -> None:
    """Question counts and request size are enforced locally without echoing content."""
    http_client, seen = _replay()
    client = _client(http_client)
    too_many = {f"q{index}": NoulQuestion(instructions="?") for index in range(33)}
    for questions in ({}, too_many):
        with pytest.raises(SystemOneRequestError, match="send 1 through 32 questions"):
            _decide(client, questions)
    with pytest.raises(SystemOneRequestError, match="262144-byte") as raised:
        asyncio.run(client.decide("x" * MAX_DECISION_INPUT_BYTES, _NOUL_QUESTIONS))
    assert "xxxx" not in str(raised.value)
    assert seen == []


@pytest.mark.parametrize(
    ("base_url", "reason"),
    [
        ("ftp://127.0.0.1/v1", "http or https API root"),
        ("http://user:pw@127.0.0.1/v1", "http or https API root"),
        ("http://127.0.0.1/v1?key=secret", "http or https API root"),
        ("/v1", "http or https API root"),
        ("http://127.0.0.1:99999/v1", "valid port"),
        ("http://127.0.0.1:port/v1", "valid port"),
        ("http://10.0.0.5:30000/v1", "plain http to a non-loopback host"),
    ],
)
def test_unsafe_endpoints_are_rejected(base_url: str, reason: str) -> None:
    """Credentials never travel in a URL or over plain http off this machine."""
    with pytest.raises(ValueError, match=reason):
        SystemOneClient(base_url, "clef-27b", api_key=SecretStr("secret-canary"))


def test_unusable_settings_are_rejected_and_the_key_stays_private() -> None:
    """Empty keys and models and nonpositive bounds fail at construction."""
    with pytest.raises(ValueError, match="api_key is empty"):
        SystemOneClient(_BASE_URL, "clef-27b", api_key=SecretStr(""))
    with pytest.raises(ValueError, match="model must contain"):
        SystemOneClient(_BASE_URL, "")
    with pytest.raises(ValueError, match="timeout must be"):
        SystemOneClient(_BASE_URL, "clef-27b", timeout=0)
    with pytest.raises(ValueError, match="max_concurrency must be"):
        SystemOneClient(_BASE_URL, "clef-27b", max_concurrency=0)
    client = SystemOneClient("https://[::1]:8443/v1", "m", api_key=SecretStr("secret-canary"))
    assert "secret-canary" not in repr(client)
    assert client.url == "https://[::1]:8443/v1/systemone"

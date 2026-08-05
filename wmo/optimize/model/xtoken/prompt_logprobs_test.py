"""Tests for the teacher's `/v1/completions` prompt-logprob client.

Every test drives the real `PromptLogprobClient` through an
`httpx.MockTransport`, so the retry loop, the response readers, and the error
paths all run for real and no request leaves the process.

The bias here is toward the failures that are SILENT. A wrong row length, an
argmax logprob returned in place of the realized token's, or a text prompt on
the wire all produce a perfectly well-formed row that corrupts every
downstream chunk sum, so those get pinned harder than the loud ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from wmo.core.types import JsonObject
from wmo.optimize.model.xtoken.prompt_logprobs import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_S,
    PromptLogprobClient,
    PromptLogprobError,
    PromptLogprobTimeoutError,
)
from wmo.providers.base import ProviderKind

_MODEL = "zai-org/GLM-5.2"
_ENDPOINT = "https://vllm-host"

Handler = Callable[[httpx.Request], httpx.Response]


def _row(token_ids: list[int], logprobs: list[float | None]) -> list[dict[str, object] | None]:
    """A `prompt_logprobs` array in vLLM's shape for one exact token sequence.

    Position 0 is null (no context) and every later position carries the
    realized token's entry, matching what a healthy server returns.
    """
    out: list[dict[str, object] | None] = [None]
    for token_id, logprob in zip(token_ids[1:], logprobs[1:], strict=True):
        out.append({str(token_id): {"logprob": logprob, "rank": 1, "decoded_token": "x"}})
    return out


def _ok_body(
    token_ids: list[int], logprobs: list[float | None], *, nested: bool = False
) -> dict[str, object]:
    """A 200 body carrying the row at the top level or under `choices[0]`."""
    rows = _row(token_ids, logprobs)
    if nested:
        return {"choices": [{"prompt_logprobs": rows, "text": ""}]}
    return {"prompt_logprobs": rows, "choices": [{"text": ""}]}


def _client(
    handler: Handler,
    *,
    api_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> PromptLogprobClient:
    """A client wired to a mock transport, with no real backoff sleeping."""
    return PromptLogprobClient(
        _ENDPOINT,
        _MODEL,
        api_key=api_key,
        transport=httpx.MockTransport(handler),
        max_attempts=max_attempts,
        sleep=lambda _seconds: None,
    )


def _always(response: httpx.Response) -> Handler:
    """A handler that returns the same response to every request."""
    return lambda _request: response


# -- the wire contract -------------------------------------------------------------------------


def test_prompt_goes_on_the_wire_as_token_ids_with_the_pinned_request_keyset() -> None:
    """A text prompt would be re-tokenized server-side and shift every position
    against our local offsets, with no error and a well-formed response. The
    request body is pinned whole so that cannot regress unnoticed."""
    seen: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_body([7, 8, 9], [None, -0.5, -1.5]))

    with _client(handler) as client:
        client.score([7, 8, 9])

    assert seen == [
        {
            "model": _MODEL,
            "prompt": [7, 8, 9],
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "temperature": 0.0,
        }
    ]
    # Not "7,8,9" and not a rendered string: ints, or vLLM re-tokenizes it.
    assert seen[0]["prompt"] == [7, 8, 9]


def test_score_returns_one_entry_per_position_with_none_at_zero() -> None:
    """`prompt_logprobs[p]` is the distribution FOR token p, so the returned row
    indexes one for one into the submitted ids with no shifting."""
    token_ids = [7, 8, 9, 10]
    body = _ok_body(token_ids, [None, -0.5, -1.5, -2.5])
    with _client(_always(httpx.Response(200, json=body))) as client:
        row = client.score(token_ids)

    assert len(row) == len(token_ids)
    assert row == [None, -0.5, -1.5, -2.5]


def test_realized_token_logprob_wins_over_the_argmax() -> None:
    """The realized token's entry is the one we want. Taking the argmax instead
    returns a row that is well formed and entirely wrong."""
    token_ids = [7, 8]
    body = {
        "prompt_logprobs": [
            None,
            {
                "8": {"logprob": -4.0, "rank": 9, "decoded_token": "realized"},
                "99": {"logprob": -0.01, "rank": 1, "decoded_token": "argmax"},
            },
        ]
    }
    with _client(_always(httpx.Response(200, json=body))) as client:
        assert client.score(token_ids) == [None, -4.0]


@pytest.mark.parametrize("nested", [False, True])
def test_both_vllm_response_shapes_are_read(nested: bool) -> None:
    """`prompt_logprobs` sits at the top level on some vLLM versions and under
    `choices[0]` on others."""
    token_ids = [7, 8, 9]
    body = _ok_body(token_ids, [None, -0.5, -1.5], nested=nested)
    with _client(_always(httpx.Response(200, json=body))) as client:
        assert client.score(token_ids) == [None, -0.5, -1.5]


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://vllm-host", "https://vllm-host/v1/completions"),
        ("https://vllm-host/", "https://vllm-host/v1/completions"),
        ("https://vllm-host/v1", "https://vllm-host/v1/completions"),
        ("https://vllm-host/v1/", "https://vllm-host/v1/completions"),
        ("  https://vllm-host/v1  ", "https://vllm-host/v1/completions"),
    ],
)
def test_endpoint_with_or_without_a_v1_suffix_never_doubles_it(
    endpoint: str, expected: str
) -> None:
    client = PromptLogprobClient(
        endpoint, _MODEL, transport=httpx.MockTransport(_always(httpx.Response(200, json={})))
    )
    assert client.url == expected
    assert client.model == _MODEL
    client.close()


def test_blank_endpoint_is_rejected_with_an_example() -> None:
    with pytest.raises(ValueError, match="needs a teacher endpoint URL"):
        PromptLogprobClient("   ", _MODEL)


def test_api_key_becomes_a_bearer_header_and_is_omitted_when_unset() -> None:
    """A private vLLM host normally has no auth; a key must never be invented."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_ok_body([7, 8], [None, -0.5]))

    with _client(handler, api_key="secret-token") as client:
        client.score([7, 8])
    with _client(handler) as client:
        client.score([7, 8])

    assert seen == ["Bearer secret-token", None]


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_or_non_finite_timeout_is_rejected(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be a positive finite number"):
        PromptLogprobClient(_ENDPOINT, _MODEL, timeout_s=timeout_s)


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_max_attempts_below_one_is_rejected(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        PromptLogprobClient(_ENDPOINT, _MODEL, max_attempts=max_attempts)


def test_empty_token_ids_is_rejected_before_any_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    with _client(handler) as client, pytest.raises(ValueError, match="at least one token id"):
        client.score([])
    assert calls == []


# -- the retry loop ----------------------------------------------------------------------------


def test_retryable_status_retries_then_succeeds() -> None:
    token_ids = [7, 8]
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_ok_body(token_ids, [None, -0.5]))

    with _client(handler, max_attempts=3) as client:
        assert client.score(token_ids) == [None, -0.5]
    assert len(attempts) == 3


def test_exhausted_retries_raise_the_last_error_naming_the_attempt_count() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429, text="rate limited")

    with _client(handler, max_attempts=3) as client, pytest.raises(PromptLogprobError) as excinfo:
        client.score([7, 8])

    assert len(attempts) == 3
    message = str(excinfo.value)
    assert "HTTP 429" in message
    assert "(attempt 3/3)" in message
    assert "retries are exhausted" in message


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_non_retryable_status_raises_on_the_first_attempt(status: int) -> None:
    """A bad request or a bad credential will fail identically forever, so
    burning the retry budget on it only delays the error."""
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(status, text="nope")

    with _client(handler, max_attempts=5) as client, pytest.raises(PromptLogprobError) as excinfo:
        client.score([7, 8])

    assert len(attempts) == 1
    assert f"HTTP {status}" in str(excinfo.value)


@pytest.mark.parametrize(
    ("status", "remedy"),
    [
        (401, "WMO_ENDPOINT_API_KEY"),
        (403, "WMO_ENDPOINT_API_KEY"),
        (404, "GET /v1/models"),
        (400, "prompt_logprobs on /v1/completions"),
        (503, "lower scoring concurrency"),
    ],
)
def test_each_status_error_names_a_status_specific_remedy(status: int, remedy: str) -> None:
    """An error a user can hit has to say what to DO, not just what broke."""
    with (
        _client(_always(httpx.Response(status, text="body")), max_attempts=1) as client,
        pytest.raises(PromptLogprobError) as excinfo,
    ):
        client.score([7, 8])
    assert remedy in str(excinfo.value)


def test_error_body_is_truncated_so_a_html_error_page_cannot_flood_the_log() -> None:
    with (
        _client(_always(httpx.Response(500, text="x" * 5000)), max_attempts=1) as client,
        pytest.raises(PromptLogprobError) as excinfo,
    ):
        client.score([7, 8])
    assert len(str(excinfo.value)) < 1000


def test_timeout_raises_the_timeout_subclass_and_is_retried() -> None:
    """`PromptLogprobTimeoutError` must stay a `TimeoutError` with "timed out"
    in the message: retry layers above classify transient capacity that way."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ReadTimeout("read timed out", request=request)

    with (
        _client(handler, max_attempts=3) as client,
        pytest.raises(PromptLogprobTimeoutError) as excinfo,
    ):
        client.score([7, 8])

    assert len(attempts) == 3
    assert isinstance(excinfo.value, TimeoutError)
    assert isinstance(excinfo.value, PromptLogprobError)
    message = str(excinfo.value)
    assert "timed out" in message
    assert f"{DEFAULT_TIMEOUT_S:g}s" in message


def test_transport_error_is_retried_and_names_the_url() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler, max_attempts=2) as client, pytest.raises(PromptLogprobError) as excinfo:
        client.score([7, 8])

    assert len(attempts) == 2
    assert not isinstance(excinfo.value, PromptLogprobTimeoutError)
    assert "could not reach https://vllm-host/v1/completions" in str(excinfo.value)


def test_backoff_doubles_caps_at_30s_and_never_sleeps_after_the_last_attempt() -> None:
    """Six attempts sleep five times. The last failure raises immediately: an
    extra sleep there is pure added latency on a call that is already lost."""
    delays: list[float] = []
    client = PromptLogprobClient(
        _ENDPOINT,
        _MODEL,
        transport=httpx.MockTransport(_always(httpx.Response(503, text="busy"))),
        max_attempts=6,
        sleep=delays.append,
    )
    with client, pytest.raises(PromptLogprobError):
        client.score([7, 8])

    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0]


def test_max_attempts_one_disables_retries_entirely() -> None:
    attempts: list[int] = []
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, text="busy")

    client = PromptLogprobClient(
        _ENDPOINT,
        _MODEL,
        transport=httpx.MockTransport(handler),
        max_attempts=1,
        sleep=delays.append,
    )
    with client, pytest.raises(PromptLogprobError):
        client.score([7, 8])

    assert len(attempts) == 1
    assert delays == []


def test_default_max_attempts_is_the_documented_constant() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, text="busy")

    with _client(handler) as client, pytest.raises(PromptLogprobError):
        client.score([7, 8])
    assert len(attempts) == DEFAULT_MAX_ATTEMPTS


def test_a_shape_error_is_not_retried() -> None:
    """A 200 whose body is the wrong shape is a version or route mismatch, not
    capacity: retrying it just multiplies a certain failure."""
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(200, json={"choices": [{"text": ""}]})

    with _client(handler, max_attempts=5) as client, pytest.raises(PromptLogprobError):
        client.score([7, 8])
    assert len(attempts) == 1


# -- response validation -----------------------------------------------------------------------


def test_a_row_shorter_than_the_prompt_is_rejected_rather_than_returned() -> None:
    """The whole point of the length check: a short row would be summed against
    teacher tokens it does not correspond to, silently."""
    body = {"prompt_logprobs": _row([7, 8, 9], [None, -0.5, -1.5])[:2]}
    with _client(_always(httpx.Response(200, json=body))) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8, 9])
    message = str(excinfo.value)
    assert "returned 2 prompt_logprobs entries for a 3-token prompt" in message
    assert "silently corrupts every downstream span sum" in message


def test_a_row_longer_than_the_prompt_is_rejected_too() -> None:
    body = {"prompt_logprobs": _row([7, 8, 9, 10], [None, -0.5, -1.5, -2.5])}
    with _client(_always(httpx.Response(200, json=body))) as client:
        with pytest.raises(PromptLogprobError, match="returned 4 prompt_logprobs entries"):
            client.score([7, 8, 9])


def test_a_response_with_no_prompt_logprobs_anywhere_names_the_route_mistake() -> None:
    with _client(_always(httpx.Response(200, json={"choices": [{"text": "hi"}]}))) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    message = str(excinfo.value)
    assert "carried no prompt_logprobs" in message
    assert "/v1/chat/completions supports neither" in message


def test_a_missing_realized_token_names_the_position_and_the_candidates() -> None:
    body = {
        "prompt_logprobs": [
            None,
            {"99": {"logprob": -0.1}, "98": {"logprob": -0.2}},
        ]
    }
    with _client(_always(httpx.Response(200, json=body))) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    message = str(excinfo.value)
    assert "no logprob for the realized token 8 at position 1 of 2" in message
    assert "candidates: 98, 99" in message


def test_a_null_position_after_zero_is_a_missing_realized_token() -> None:
    """Only position 0 may be null. A null later position has no logprob to
    read, so it must fail loudly rather than land as a None in the row."""
    body = {"prompt_logprobs": [None, None]}
    with _client(_always(httpx.Response(200, json=body))) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    assert "none (the position was null or empty)" in str(excinfo.value)


def test_many_candidates_are_elided_in_the_error() -> None:
    candidates = {str(token_id): {"logprob": -1.0} for token_id in range(100, 120)}
    with _client(_always(httpx.Response(200, json={"prompt_logprobs": [None, candidates]}))) as (
        client
    ):
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    message = str(excinfo.value)
    assert "20 returned, first ids 100, 101, 102, 103, 104, 105, 106, 107, ..." in message


def test_a_non_json_body_says_the_url_may_point_at_a_proxy() -> None:
    response = httpx.Response(
        200, text="<html>gateway</html>", headers={"content-type": "text/html"}
    )
    with _client(_always(response)) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    message = str(excinfo.value)
    assert "non-JSON response" in message
    assert "proxy or web page" in message


def test_an_unreadable_logprob_shape_names_the_expected_format() -> None:
    """A position whose entries are bare floats rather than objects with a
    `logprob` is an older/other server format, not a transient fault."""
    body = {"prompt_logprobs": [None, {"8": -0.5}]}
    with _client(_always(httpx.Response(200, json=body))) as client:
        with pytest.raises(PromptLogprobError) as excinfo:
            client.score([7, 8])
    message = str(excinfo.value)
    assert "could not read the response" in message
    assert "check the vLLM version's response format" in message


def test_extra_response_fields_are_ignored_not_rejected() -> None:
    """Servers add fields; a new one must not break scoring."""
    body = _ok_body([7, 8], [None, -0.5])
    body["id"] = "cmpl-1"
    body["usage"] = {"prompt_tokens": 2}
    with _client(_always(httpx.Response(200, json=body))) as client:
        assert client.score([7, 8]) == [None, -0.5]


def test_a_legitimately_zero_logprob_is_preserved() -> None:
    """A near-certain continuation really does score 0.0, so nothing may treat
    a zero as missing and drop it to None."""
    with _client(_always(httpx.Response(200, json=_ok_body([7, 8], [None, 0.0])))) as client:
        assert client.score([7, 8]) == [None, 0.0]


# -- usage accounting and verify ---------------------------------------------------------------


def test_usage_counts_every_dispatched_attempt_including_the_failed_ones() -> None:
    """A timed-out request has usually already run its prefill on the server, so
    the tokens are real spend even though the call returned nothing."""
    token_ids = [7, 8, 9, 10]
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_ok_body(token_ids, [None, -0.5, -1.5, -2.5]))

    with _client(handler, max_attempts=3) as client:
        assert client.usage() == 0
        client.score(token_ids)
        assert client.usage() == 3 * len(token_ids)


def test_verify_probe_tokens_are_excluded_from_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        token_ids = json.loads(request.content)["prompt"]
        return httpx.Response(200, json=_ok_body(token_ids, [None] + [-0.5] * (len(token_ids) - 1)))

    with _client(handler) as client:
        assert client.verify().ok
        assert client.usage() == 0
        client.score([7, 8, 9])
        assert client.usage() == 3


def test_verify_reports_success_with_the_url_as_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        token_ids = json.loads(request.content)["prompt"]
        return httpx.Response(200, json=_ok_body(token_ids, [None] + [-0.5] * (len(token_ids) - 1)))

    with _client(handler) as client:
        result = client.verify()

    assert result.ok
    assert result.kind is ProviderKind.OPENAI
    assert result.model == _MODEL
    assert result.detail == "https://vllm-host/v1/completions"


def test_verify_reports_failure_instead_of_raising() -> None:
    """Preflight lists every misconfigured backend at once, so verify must never
    raise, whatever went wrong."""
    with _client(_always(httpx.Response(404, text="no such model")), max_attempts=1) as client:
        result = client.verify()

    assert not result.ok
    assert result.kind is ProviderKind.OPENAI
    assert "HTTP 404" in result.detail


def test_verify_swallows_a_transport_error_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(handler, max_attempts=1) as client:
        result = client.verify()

    assert not result.ok
    assert "could not reach" in result.detail


def test_one_client_serves_a_scoring_pool_concurrently() -> None:
    """One client is documented as safe to share across a scoring pool, which is
    how the teacher is driven in a step: many datums, one client.

    This exercises that path (shared connection pool, shared counter) and pins
    the total. It is NOT a proof that the usage lock is present: removing the
    lock does not lose increments reliably enough to assert on, so the lock is
    held by review, and this test covers the reachable half.
    """
    token_ids = [7, 8, 9, 10, 11]

    def handler(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content)["prompt"]
        return httpx.Response(200, json=_ok_body(ids, [None] + [-0.5] * (len(ids) - 1)))

    with _client(handler) as client:
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda _i: client.score(token_ids), range(24)))

        assert all(row == [None, -0.5, -0.5, -0.5, -0.5] for row in rows)
        assert client.usage() == 24 * len(token_ids)


def test_close_is_idempotent_and_the_context_manager_closes() -> None:
    client = _client(_always(httpx.Response(200, json=_ok_body([7, 8], [None, -0.5]))))
    with client as entered:
        assert entered is client
        assert client.score([7, 8]) == [None, -0.5]
    client.close()  # a second close must not raise

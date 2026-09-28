"""Exercise bounded, opt-in SDK error evidence without a provider or retry."""

import asyncio
import json
from collections.abc import Iterator
from contextlib import nullcontext

import httpx2
import pytest
from openai import APIStatusError, BadRequestError, OpenAI

from exp.runtime.models.providers.diagnostics import (
    APIErrorEvidence,
    api_error_evidence,
    observe_openai_errors,
)


def sdk_error(body: object = None) -> BadRequestError:
    """Construct an actual SDK rejection with untrusted response metadata."""
    return BadRequestError(
        "unstructured exception text must not be retained",
        response=httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://private.example/v1/chat/completions"),
            headers={"x-request-id": "req-example", "authorization": "secret-header"},
        ),
        body=body,
    )


@pytest.mark.parametrize("observe", [False, True])
def test_real_sdk_http400_keeps_request_and_single_dispatch(observe: bool) -> None:
    """Adding observation preserves the actual SDK payload, key and fatal HTTP behavior."""
    requests: list[httpx2.Request] = []
    receipts: list[APIErrorEvidence] = []

    def reject(request: httpx2.Request) -> httpx2.Response:
        """Reject one request locally through the official SDK response decoder."""
        requests.append(request)
        return httpx2.Response(
            400,
            headers={"x-request-id": "req-one"},
            json={"error": {"type": "invalid_request_error", "code": "decode_error"}},
        )

    with OpenAI(
        api_key="test-key",
        base_url="http://127.0.0.1/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reject)),
    ) as client:
        context = (
            observe_openai_errors(
                receipts.append, operation="learner.chat.completions", endpoint_role="learner"
            )
            if observe
            else nullcontext()
        )
        with pytest.raises(BadRequestError):
            with context:
                client.chat.completions.create(
                    model="student",
                    messages=[{"role": "user", "content": "hello"}],
                    max_completion_tokens=2048,
                    extra_headers={"Idempotency-Key": "same-caller-key"},
                )
    assert len(requests) == 1
    assert requests[0].headers["idempotency-key"] == "same-caller-key"
    assert json.loads(requests[0].content) == {
        "model": "student",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 2048,
    }
    assert len(receipts) == int(observe)
    if observe:
        assert receipts[0].status_code == 400
        assert receipts[0].request_id == "req-one"
        assert receipts[0].error_code == "decode_error"


@pytest.mark.parametrize("observer_fails", [False, True])
def test_same_exception_survives_observer_failure(observer_fails: bool) -> None:
    """A diagnostic IO failure cannot replace the original API rejection or retry it."""
    error = sdk_error({"error": {"message": "diagnostic", "code": "invalid"}})
    seen: list[APIErrorEvidence] = []

    def observer(evidence: APIErrorEvidence) -> None:
        """Retain a receipt or simulate a failed private diagnostic sink."""
        seen.append(evidence)
        if observer_fails:
            raise OSError("private filesystem path")

    with pytest.raises(BadRequestError) as caught:
        with observe_openai_errors(observer, operation="complete", endpoint_role="learner"):
            raise error
    assert caught.value is error and len(seen) == 1
    assert seen[0].error_message is None
    if observer_fails:
        assert error.__notes__ == ["HTTP error observation failed: OSError"]


def test_success_and_non_http_errors_are_untouched() -> None:
    """The observer runs neither on successful calls nor on local validation errors."""
    receipts: list[APIErrorEvidence] = []
    with observe_openai_errors(receipts.append, operation="call", endpoint_role="provider"):
        result = "same-result"
    assert result == "same-result" and not receipts
    error = ValueError("local failure")
    with pytest.raises(ValueError) as caught:
        with observe_openai_errors(receipts.append, operation="call", endpoint_role="provider"):
            raise error
    assert caught.value is error and not receipts


def test_awaited_failure_and_cancellation_keep_identity() -> None:
    """The same context works around await without changing cancellation semantics."""
    receipts: list[APIErrorEvidence] = []
    failure = sdk_error()
    cancellation = asyncio.CancelledError()

    async def invoke(error: BaseException) -> None:
        """Yield once before an existing asynchronous operation fails."""
        with observe_openai_errors(receipts.append, operation="await", endpoint_role="provider"):
            await asyncio.sleep(0)
            raise error

    with pytest.raises(BadRequestError) as caught:
        asyncio.run(invoke(failure))
    assert caught.value is failure and len(receipts) == 1
    with pytest.raises(asyncio.CancelledError) as cancelled:
        asyncio.run(invoke(cancellation))
    assert cancelled.value is cancellation and len(receipts) == 1


def test_explicit_message_capture_is_bounded_and_credential_filtered() -> None:
    """Only allowlisted fields survive; no raw body, headers, URL or traceback is copied."""
    secret = "private-auth-value"
    error = sdk_error(
        {
            "error": {
                "type": "invalid_request_error",
                "code": secret,
                "param": "messages",
                "message": "\n" + secret + " Bearer other-key https://private.test/?key=secret "
                "sk-another-key " + "x" * 4000,
            },
            "transcript": "unallowlisted prompt",
            "traceback": "unallowlisted traceback",
        }
    )
    evidence = api_error_evidence(
        error,
        operation="user.next_message",
        endpoint_role="official_user_provider",
        secrets=(secret,),
        include_message=True,
    )
    serialized = evidence.model_dump_json()
    assert evidence.error_message is not None and len(evidence.error_message) == 2048
    assert evidence.error_code == "[redacted]"
    for omitted in (
        secret,
        "other-key",
        "private.test",
        "sk-another",
        "secret-header",
        "unallowlisted",
    ):
        assert omitted not in serialized
    assert "\n" not in evidence.error_message


@pytest.mark.parametrize(
    "body",
    [None, "arbitrary secret text", ["secret"], {"error": {"message": "x" * 65536}}],
)
def test_nonobject_and_oversized_bodies_are_omitted(body: object) -> None:
    """Missing or unsafe bodies do not fall back to raw exception strings."""
    evidence = api_error_evidence(
        sdk_error(body), operation="call", endpoint_role="provider", include_message=True
    )
    assert evidence.body_kind == "omitted_nonobject_or_oversized"
    assert evidence.error_type is evidence.error_code is evidence.error_param is None
    assert evidence.error_message is None
    assert len(evidence.model_dump_json()) < 1024
    assert "unstructured" not in evidence.model_dump_json()


def test_nested_malformed_body_and_unsafe_correlation_token_are_omitted() -> None:
    """Structured outer bodies cannot smuggle nested arbitrary text or header content."""
    error = sdk_error({"error": "raw provider text"})
    error.request_id = "https://private.example/token?key=hidden"
    evidence = api_error_evidence(
        error, operation="call", endpoint_role="provider", include_message=True
    )
    assert evidence.request_id is None
    assert evidence.error_message is None


@pytest.mark.parametrize("sdk_body", [None, {"error": {"code": "sdk_body"}}])
def test_buffered_response_json_fills_only_missing_sdk_body(sdk_body: object) -> None:
    """Provider SDK wrappers can omit body while retaining already-received HTTP JSON."""
    error = BadRequestError(
        "must not capture this string",
        response=httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://provider.test/v1"),
            json={"error": {"code": "buffered_body", "message": "safe diagnostic"}},
        ),
        body=sdk_body,
    )
    evidence = api_error_evidence(error, operation="user", endpoint_role="provider")
    assert evidence.error_code == ("sdk_body" if sdk_body else "buffered_body")
    assert evidence.error_message is None


def test_diagnostics_never_read_a_response_stream() -> None:
    """Unbuffered responses cannot cause hidden network IO while observing an error."""
    reads: list[bool] = []

    def stream() -> Iterator[bytes]:
        """Fail if diagnostics try to consume the original response stream."""
        reads.append(True)
        raise AssertionError("unexpected stream read")
        yield b""

    error = BadRequestError(
        "opaque",
        response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://provider.test/v1"), content=stream()
        ),
        body=None,
    )
    evidence = api_error_evidence(error, operation="call", endpoint_role="provider")
    assert evidence.body_kind == "omitted_nonobject_or_oversized"
    assert not reads


def test_compatible_sdk_subclass_keeps_actual_type() -> None:
    """SDK-compatible provider wrappers are observed without importing their libraries."""

    class ProviderBadRequest(BadRequestError):
        """Represent a provider adapter's ordinary official-SDK subclass."""

    original = sdk_error({"error": {"code": "provider_validation"}})
    error = ProviderBadRequest("private", response=original.response, body=original.body)
    receipts: list[APIErrorEvidence] = []
    with pytest.raises(APIStatusError) as caught:
        with observe_openai_errors(receipts.append, operation="call", endpoint_role="provider"):
            raise error
    assert caught.value is error
    assert receipts[0].exception_type == "ProviderBadRequest"


def test_bad_observation_configuration_cannot_mask_existing_error() -> None:
    """Accidentally supplying a URL as a label does not publish it or replace HTTP failure."""
    error = sdk_error()
    receipts: list[APIErrorEvidence] = []
    with pytest.raises(BadRequestError) as caught:
        with observe_openai_errors(
            receipts.append, operation="https://private.test/key", endpoint_role="provider"
        ):
            raise error
    assert caught.value is error and not receipts
    assert error.__notes__ == ["HTTP error observation failed: ValueError"]

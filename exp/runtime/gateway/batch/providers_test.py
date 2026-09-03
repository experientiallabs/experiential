"""Wire-level provider client tests over injected mock transports.

Every fixture is a real HTTP exchange served by ``httpx.MockTransport``: the
clients parse actual response bytes, so these tests pin the wire contracts of
all three providers across success, per-line error, job failure, cancellation,
and spoofed-URL outcomes without any network access.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from exp.runtime.gateway.batch.contracts import (
    BatchCounts,
    BatchJob,
    BatchLine,
    BatchStatus,
    BatchSubmitError,
)
from exp.runtime.gateway.batch.providers import (
    ANTHROPIC_HOST,
    AnthropicBatchClient,
    OpenAIBatchClient,
    OpenRouterBatchClient,
    provider_error_detail,
    require_exact_host,
)


def _job(provider: str, lines: int = 2, provider_batch_id: str | None = "pb_1") -> BatchJob:
    """Build one submitted job fixture for the given provider."""
    created = datetime(2026, 9, 1, tzinfo=UTC)
    surface = "/v1/messages" if provider == "anthropic" else "/v1/chat/completions"
    return BatchJob(
        batch_id="batch_t",
        organization_id="org_a",
        identity_id="id_a",
        surface=surface,
        provider=provider,
        credential_reference="secret://fixture",
        provider_batch_id=provider_batch_id,
        input_file_id="file_t",
        counts=BatchCounts(total=lines),
        lines=tuple(
            BatchLine(
                custom_id=f"line-{index}",
                surface=surface,
                model="model-batch",
                provider_model="prov/model:batch",
                body={"messages": []},
                estimated_input_tokens=4,
                maximum_output_tokens=16,
            )
            for index in range(lines)
        ),
        created_at=created,
        expires_at=created + timedelta(hours=24),
    )


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    """Wrap one request handler into a mock transport."""
    return httpx.MockTransport(handler)


def test_openai_submit_uploads_jsonl_then_creates_the_batch() -> None:
    """The upload carries per-line url/body and the create references the file."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/files":
            assert b'"url": "/v1/chat/completions"' in request.content
            assert b'"model": "prov/model:batch"' in request.content
            return httpx.Response(200, json={"id": "file_prov"})
        assert request.url.path == "/v1/batches"
        body = json.loads(request.content)
        assert body == {
            "input_file_id": "file_prov",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        return httpx.Response(200, json={"id": "pb_9"})

    client = OpenAIBatchClient(transport=_transport(handler))
    provider_id = asyncio.run(client.submit(job=_job("openai"), api_key="sk-test"))
    assert provider_id == "pb_9"
    assert seen[0].headers["authorization"] == "Bearer sk-test"


def test_openai_poll_maps_status_counts_and_failure() -> None:
    """A failed provider batch surfaces its error data content-free."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "pb_1",
                "status": "failed",
                "request_counts": {"total": 2, "completed": 1, "failed": 1},
                "errors": {"data": [{"message": "input malformed"}]},
            },
        )

    client = OpenAIBatchClient(transport=_transport(handler))
    snapshot = asyncio.run(client.poll(job=_job("openai"), api_key="sk"))
    assert snapshot.status is BatchStatus.FAILED
    assert snapshot.completed == 1 and snapshot.failed == 1
    assert snapshot.failure_message is not None


def test_openai_results_merge_output_and_error_files() -> None:
    """Both result files parse into per-line results with usage."""
    output_line = {
        "id": "r1",
        "custom_id": "line-0",
        "response": {
            "status_code": 200,
            "body": {"usage": {"prompt_tokens": 7, "completion_tokens": 9}},
        },
        "error": None,
    }
    error_line = {"id": "r2", "custom_id": "line-1", "response": None, "error": {"code": "bad"}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/batches/pb_1":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output_file_id": "fo",
                    "error_file_id": "fe",
                },
            )
        if request.url.path == "/v1/files/fo/content":
            return httpx.Response(200, text=json.dumps(output_line))
        assert request.url.path == "/v1/files/fe/content"
        return httpx.Response(200, text=json.dumps(error_line))

    client = OpenAIBatchClient(transport=_transport(handler))
    results = asyncio.run(client.results(job=_job("openai"), api_key="sk"))
    by_id = {result.custom_id: result for result in results}
    assert by_id["line-0"].output_tokens == 9 and by_id["line-0"].error is None
    assert by_id["line-1"].error == {"code": "bad"}


def test_openai_cancel_posts_the_cancel_route() -> None:
    """Cancellation posts to the provider's cancel action."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"id": "pb_1", "status": "cancelling"})

    client = OpenAIBatchClient(transport=_transport(handler))
    asyncio.run(client.cancel(job=_job("openai"), api_key="sk"))
    assert paths == ["/v1/batches/pb_1/cancel"]


def test_openai_provider_error_maps_to_submit_error() -> None:
    """A 500 from the provider raises provider_error naming the status."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = OpenAIBatchClient(transport=_transport(handler))
    with pytest.raises(BatchSubmitError, match="status 500: boom"):
        asyncio.run(client.poll(job=_job("openai"), api_key="sk"))


def test_provider_rejection_carries_the_provider_message() -> None:
    """A 400 on submit names the provider's own error.message, bounded."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "  openai/gpt-4.1-nano:batch is not a valid\n model ID  ",
                    "code": 400,
                }
            },
        )

    client = OpenRouterBatchClient(transport=_transport(handler))
    with pytest.raises(BatchSubmitError) as raised:
        asyncio.run(client.submit(job=_job("openrouter"), api_key="ork"))
    assert raised.value.code == "provider_error"
    assert raised.value.message == (
        "provider batch create failed with status 400: "
        "openai/gpt-4.1-nano:batch is not a valid model ID"
    )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(502, text="<html>Bad gateway</html>"),
        httpx.Response(400, json={"error": {"type": "invalid_request_error"}}),
        httpx.Response(400, json={"error": {"message": "   "}}),
        httpx.Response(400, json=["not", "an", "object"]),
    ],
)
def test_provider_error_without_a_message_stays_status_only(response: httpx.Response) -> None:
    """Bodies that are not JSON or carry no message string add nothing."""
    assert provider_error_detail(response) is None


def test_provider_error_detail_is_bounded() -> None:
    """A runaway provider message is cut at the detail limit."""
    detail = provider_error_detail(httpx.Response(400, json={"error": {"message": "x" * 1000}}))
    assert detail == "x" * 400


def test_anthropic_submit_sends_inline_requests_with_version_header() -> None:
    """Requests carry custom_id plus params, under the versioned headers."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages/batches"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["x-api-key"] == "ak"
        body = json.loads(request.content)
        assert body["requests"][0]["custom_id"] == "line-0"
        assert body["requests"][0]["params"]["model"] == "prov/model:batch"
        return httpx.Response(200, json={"id": "mb_1"})

    client = AnthropicBatchClient(transport=_transport(handler))
    assert asyncio.run(client.submit(job=_job("anthropic"), api_key="ak")) == "mb_1"


def test_anthropic_poll_maps_processing_states() -> None:
    """ended maps to completed; errored and expired count as failed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "processing_status": "ended",
                "request_counts": {
                    "processing": 0,
                    "succeeded": 1,
                    "errored": 1,
                    "canceled": 0,
                    "expired": 1,
                },
            },
        )

    client = AnthropicBatchClient(transport=_transport(handler))
    snapshot = asyncio.run(client.poll(job=_job("anthropic"), api_key="ak"))
    assert snapshot.status is BatchStatus.COMPLETED
    assert snapshot.completed == 1 and snapshot.failed == 2


def test_anthropic_results_follow_only_the_exact_host() -> None:
    """A spoofed results_url is refused before any credentialed fetch."""

    def spoofed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results_url": f"https://{ANTHROPIC_HOST}.evil.example/v1/results"},
        )

    client = AnthropicBatchClient(transport=_transport(spoofed))
    with pytest.raises(BatchSubmitError, match="outside"):
        asyncio.run(client.results(job=_job("anthropic"), api_key="ak"))


def test_anthropic_results_parse_succeeded_and_errored_lines() -> None:
    """The results JSONL maps message and error result kinds."""
    lines = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "line-0",
                    "result": {
                        "type": "succeeded",
                        "message": {"usage": {"input_tokens": 2, "output_tokens": 3}},
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "line-1",
                    "result": {"type": "errored", "error": {"type": "invalid_request"}},
                }
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages/batches/pb_1":
            return httpx.Response(
                200, json={"results_url": f"https://{ANTHROPIC_HOST}/v1/batches/pb_1/results"}
            )
        return httpx.Response(200, text=lines)

    client = AnthropicBatchClient(transport=_transport(handler))
    results = asyncio.run(client.results(job=_job("anthropic"), api_key="ak"))
    assert results[0].output_tokens == 3 and results[0].error is None
    assert results[1].error == {"type": "invalid_request"}


def test_openrouter_submit_orders_fields_and_uses_one_model() -> None:
    """endpoint and model serialize before requests, from the first line."""

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.content.decode("utf-8")
        assert raw.index('"endpoint"') < raw.index('"requests"')
        assert raw.index('"model"') < raw.index('"requests"')
        body = json.loads(raw)
        assert body["model"] == "prov/model:batch"
        assert len(body["requests"]) == 2
        return httpx.Response(202, json={"id": "orb_1", "status": "validating"})

    client = OpenRouterBatchClient(transport=_transport(handler))
    assert asyncio.run(client.submit(job=_job("openrouter"), api_key="ork")) == "orb_1"


def test_openrouter_results_parse_the_inline_array() -> None:
    """Completed batches carry results inline in the batch object."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                "results": [
                    {
                        "custom_id": "line-0",
                        "response": {
                            "status_code": 200,
                            "body": {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
                        },
                        "error": None,
                    }
                ],
            },
        )

    client = OpenRouterBatchClient(transport=_transport(handler))
    snapshot = asyncio.run(client.poll(job=_job("openrouter"), api_key="ork"))
    assert snapshot.status is BatchStatus.COMPLETED and snapshot.results_ready
    results = asyncio.run(client.results(job=_job("openrouter"), api_key="ork"))
    assert results[0].output_tokens == 2


def test_openrouter_cancel_is_refused_as_unsupported() -> None:
    """The provider exposes no cancel endpoint; the client says so."""
    client = OpenRouterBatchClient()
    with pytest.raises(BatchSubmitError, match="cannot be cancelled"):
        asyncio.run(client.cancel(job=_job("openrouter"), api_key="ork"))


@pytest.mark.parametrize(
    "url",
    [
        "http://api.anthropic.com/x",
        "https://api.anthropic.com.evil.example/x",
        "https://evil.example@api.anthropic.com/x",
        "https://api.anthropic.com:8443/x",
    ],
)
def test_require_exact_host_rejects_spoof_shapes(url: str) -> None:
    """Scheme, suffix, userinfo, and port spoofs are all refused."""
    with pytest.raises(BatchSubmitError, match="outside"):
        require_exact_host(url, ANTHROPIC_HOST)


def test_require_exact_host_accepts_the_exact_authority() -> None:
    """The exact https host with the default port passes unchanged."""
    url = f"https://{ANTHROPIC_HOST}/v1/results?page=1"
    assert require_exact_host(url, ANTHROPIC_HOST) == url

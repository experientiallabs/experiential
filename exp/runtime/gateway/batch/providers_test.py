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

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.batch.contracts import (
    BatchCounts,
    BatchJob,
    BatchLine,
    BatchStatus,
    BatchSubmitError,
    BatchSurface,
)
from exp.runtime.gateway.batch.providers import (
    ANTHROPIC_HOST,
    AnthropicBatchClient,
    OpenAIBatchClient,
    OpenRouterBatchClient,
    provider_error_detail,
    require_exact_host,
)


def _job(
    provider: str,
    lines: int = 2,
    provider_batch_id: str | None = "pb_1",
    surface: BatchSurface | None = None,
) -> BatchJob:
    """Build one submitted job fixture for the given provider."""
    created = datetime(2026, 9, 1, tzinfo=UTC)
    if surface is None:
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
                body={"messages": [{"role": "user", "content": f"hi {index}"}], "max_tokens": 16},
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


def test_provider_error_detail_drops_control_characters() -> None:
    """Terminal escapes and NULs in an upstream message never pass through."""
    detail = provider_error_detail(
        httpx.Response(400, json={"error": {"message": "bad\x1b[31m model\x00 id\r\n\ttry again"}})
    )
    assert detail == "bad[31m model id try again"


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


def test_anthropic_chat_lines_submit_as_translated_messages_params() -> None:
    """A Chat Completions job on the Anthropic client crosses to Messages params per line:
    no OpenAI fields reach the wire, and the streaming flag is absent."""
    seen: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "mb_chat"})

    client = AnthropicBatchClient(transport=_transport(handler))
    job = _job("anthropic", surface="/v1/chat/completions")
    assert asyncio.run(client.submit(job=job, api_key="ak")) == "mb_chat"
    requests = seen[0]["requests"]
    assert isinstance(requests, list) and len(requests) == 2
    first = requests[0]
    assert isinstance(first, dict) and first["custom_id"] == "line-0"
    params = first["params"]
    assert isinstance(params, dict)
    assert params["model"] == "prov/model:batch"
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi 0"}]}]
    assert params["max_tokens"] == 16
    assert "stream" not in params
    assert client.surfaces == ("/v1/messages", "/v1/chat/completions", "/v1/responses")
    assert OpenAIBatchClient().surfaces == ("/v1/chat/completions", "/v1/responses")


def _anthropic_results_client(lines: list[JsonObject]) -> AnthropicBatchClient:
    """Serve one batch object plus the given results JSONL over the exact host."""
    rendered = "\n".join(json.dumps(line) for line in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/messages/batches/pb_1":
            return httpx.Response(
                200, json={"results_url": f"https://{ANTHROPIC_HOST}/v1/batches/pb_1/results"}
            )
        return httpx.Response(200, text=rendered)

    return AnthropicBatchClient(transport=_transport(handler))


def test_anthropic_chat_job_results_render_chat_completions_with_cache_usage() -> None:
    """Each succeeded Message renders as the caller's chat.completion; the line's usage
    fields still come from Anthropic's own usage object (cache legs intact)."""
    client = _anthropic_results_client(
        [
            {
                "custom_id": "line-0",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "prov/model",
                        "content": [{"type": "text", "text": "Hello."}],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 15,
                            "cache_creation_input_tokens": 4501,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 4,
                        },
                    },
                },
            },
            {
                "custom_id": "line-1",
                "result": {
                    "type": "succeeded",
                    "message": {"content": "not-a-list", "stop_reason": "end_turn"},
                },
            },
            {
                "custom_id": "unknown-line",
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "stray"}]},
                },
            },
        ]
    )
    job = _job("anthropic", surface="/v1/chat/completions")
    results = {r.custom_id: r for r in asyncio.run(client.results(job=job, api_key="ak"))}
    served = results["line-0"]
    assert served.response is not None and served.response["object"] == "chat.completion"
    assert served.response["model"] == "model-batch"
    choices = served.response["choices"]
    assert isinstance(choices, list) and isinstance(choices[0], dict)
    assert choices[0]["finish_reason"] == "stop"
    message = choices[0]["message"]
    assert isinstance(message, dict) and message["content"] == "Hello."
    assert served.response["usage"] == {
        "prompt_tokens": 4516,
        "completion_tokens": 4,
        "total_tokens": 4520,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": None,
    }
    assert (served.input_tokens, served.output_tokens) == (4516, 4)
    assert served.cached_input_tokens == 0 and served.cache_creation_input_tokens == 4501
    # A Message the renderer cannot read fails the line with the cause, not the job.
    broken = results["line-1"]
    assert broken.error is not None and broken.status_code == 502
    assert broken.failure_reason is not None and "content" in broken.failure_reason
    # A result for a line the job never carried keeps the provider's object.
    stray = results["unknown-line"]
    assert stray.response == {"content": [{"type": "text", "text": "stray"}]}


def test_anthropic_ended_batch_reports_the_cancelled_line_count() -> None:
    """An ended batch is COMPLETED at the provider seam whatever the job intent; the
    lines Anthropic cut ride the snapshot as cancelled_lines for the engine to judge."""

    def ended(canceled: int) -> Callable[[httpx.Request], httpx.Response]:
        """Serve one ended batch object with the given canceled count."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "processing_status": "ended",
                    "request_counts": {
                        "processing": 0,
                        "succeeded": 2 - canceled,
                        "errored": 0,
                        "canceled": canceled,
                        "expired": 0,
                    },
                },
            )

        return handler

    cancelling = _job("anthropic").model_copy(update={"status": BatchStatus.CANCELLING})
    cut = asyncio.run(
        AnthropicBatchClient(transport=_transport(ended(1))).poll(job=cancelling, api_key="ak")
    )
    assert cut.status is BatchStatus.COMPLETED and cut.results_ready
    assert (cut.completed, cut.failed, cut.cancelled_lines) == (1, 1, 1)
    ran = asyncio.run(
        AnthropicBatchClient(transport=_transport(ended(0))).poll(job=cancelling, api_key="ak")
    )
    assert ran.status is BatchStatus.COMPLETED and ran.cancelled_lines == 0
    canceling = _job("anthropic")

    def mid_cancel(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"processing_status": "canceling", "request_counts": {"processing": 2}}
        )

    pending = asyncio.run(
        AnthropicBatchClient(transport=_transport(mid_cancel)).poll(job=canceling, api_key="ak")
    )
    assert pending.status is BatchStatus.CANCELLING and not pending.results_ready


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


def test_anthropic_results_fold_cache_legs_into_input_and_name_the_subsets() -> None:
    """Anthropic reports input_tokens EXCLUDING the cache legs; the line total folds
    them in (as the synchronous normalizer does) and names read and creation subsets."""
    client = _anthropic_results_client(
        [
            {
                "custom_id": "line-0",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "usage": {
                            "input_tokens": 15,
                            "cache_creation_input_tokens": 4501,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 4,
                        }
                    },
                },
            },
            {
                "custom_id": "line-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "usage": {
                            "input_tokens": 9,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 4501,
                            "output_tokens": 16,
                        }
                    },
                },
            },
        ]
    )
    results = {
        r.custom_id: r for r in asyncio.run(client.results(job=_job("anthropic"), api_key="ak"))
    }
    written = results["line-0"]
    assert (written.input_tokens, written.output_tokens) == (4516, 4)
    assert written.cached_input_tokens == 0
    assert written.cache_creation_input_tokens == 4501
    assert written.reasoning_tokens is None
    read = results["line-1"]
    assert (read.input_tokens, read.output_tokens) == (4510, 16)
    assert read.cached_input_tokens == 4501
    assert read.cache_creation_input_tokens is None


def test_anthropic_canceled_and_expired_lines_carry_a_reason() -> None:
    """Results Anthropic never served carry no error object; the line still names why."""
    client = _anthropic_results_client(
        [
            {"custom_id": "line-0", "result": {"type": "canceled"}},
            {"custom_id": "line-1", "result": {"type": "expired"}},
            {
                "custom_id": "line-2",
                "result": {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "max_tokens: 0"},
                    },
                },
            },
        ]
    )
    results = asyncio.run(client.results(job=_job("anthropic", lines=3), api_key="ak"))
    assert [r.error is not None for r in results] == [True, True, True]
    assert results[0].failure_reason == "the provider canceled this request before it completed"
    assert results[1].failure_reason == "the provider expired this request before it completed"
    # An errored result keeps the provider's error object verbatim, and the
    # reason is the nested actionable message, not the outer envelope type.
    assert results[2].error is not None and results[2].error["type"] == "error"
    assert results[2].failure_reason == "max_tokens: 0"


def test_openai_results_carry_cached_and_reasoning_subsets() -> None:
    """Chat-shaped usage names cached prompt tokens and reasoning; total_tokens decides
    whether reasoning was reported additively (fold) or as a subset (pass through)."""
    subset = {
        "custom_id": "line-0",
        "response": {
            "status_code": 200,
            "body": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                    "prompt_tokens_details": {"cached_tokens": 60},
                    "completion_tokens_details": {"reasoning_tokens": 8},
                }
            },
        },
        "error": None,
    }
    additive = {
        "custom_id": "line-1",
        "response": {
            "status_code": 200,
            "body": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 148,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                }
            },
        },
        "error": None,
    }
    responses_shaped = {
        "custom_id": "line-2",
        "response": {
            "status_code": 200,
            "body": {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 20,
                    "total_tokens": 70,
                    "input_tokens_details": {"cached_tokens": 25},
                    "output_tokens_details": {"reasoning_tokens": 5},
                }
            },
        },
        "error": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/batches/pb_1":
            return httpx.Response(200, json={"status": "completed", "output_file_id": "fo"})
        assert request.url.path == "/v1/files/fo/content"
        return httpx.Response(
            200, text="\n".join(json.dumps(line) for line in (subset, additive, responses_shaped))
        )

    client = OpenAIBatchClient(transport=_transport(handler))
    by_id = {r.custom_id: r for r in asyncio.run(client.results(job=_job("openai"), api_key="sk"))}
    assert (by_id["line-0"].input_tokens, by_id["line-0"].output_tokens) == (100, 40)
    assert by_id["line-0"].cached_input_tokens == 60
    assert by_id["line-0"].reasoning_tokens == 8
    assert by_id["line-0"].cache_creation_input_tokens is None
    assert (by_id["line-1"].input_tokens, by_id["line-1"].output_tokens) == (100, 48)
    assert by_id["line-1"].cached_input_tokens is None
    assert by_id["line-1"].reasoning_tokens == 8
    assert (by_id["line-2"].input_tokens, by_id["line-2"].output_tokens) == (50, 20)
    assert by_id["line-2"].cached_input_tokens == 25
    assert by_id["line-2"].reasoning_tokens == 5


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


def test_anthropic_success_without_a_message_and_unknown_result_types_name_themselves() -> None:
    """A succeeded row lacking a message object is a malformed result with its own
    reason; a result type Anthropic never documented is named unknown, not None."""
    client = _anthropic_results_client(
        [
            {"custom_id": "line-0", "result": {"type": "succeeded", "message": "nope"}},
            {"custom_id": "line-1", "result": {}},
        ]
    )
    results = asyncio.run(client.results(job=_job("anthropic"), api_key="ak"))
    assert results[0].status_code == 502 and results[0].error is not None
    assert results[0].failure_reason == "the provider reported success without a message object"
    assert (results[0].input_tokens, results[0].output_tokens) == (0, 0)
    assert results[1].error == {
        "type": "unknown",
        "message": "the provider reported this request as unknown",
    }

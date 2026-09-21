"""Tests for native Gemini response conversion and usage accounting."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelMessage, ModelRequest, ModelSnapshot, Usage
from exp.runtime.models.providers.async_transport import run_then_close_pooled_client
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.gemini import GeminiClient, gemini_generate_response


def _snapshot() -> ModelSnapshot:
    """Return a frozen Gemini identity fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="gemini",
        model_id="gemini-fixture",
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _completed_usage(usage: JsonObject) -> Usage:
    """Parse usage from one completed generateContent payload.

    Args:
        usage: Native ``usageMetadata`` object.

    Returns:
        Observed provider-neutral usage.

    Raises:
        ProviderResponseError: A usage field is present but not a non-negative integer.
    """
    response = gemini_generate_response(
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": usage,
        },
        configured_model=_snapshot(),
        latency_seconds=0.1,
    )
    observed = response.economics.usage
    assert observed is not None
    return observed


def test_absent_thoughts_leave_output_at_candidates_token_count() -> None:
    """Omitting thoughtsTokenCount keeps output equal to candidatesTokenCount."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=5, cached_input_tokens=2)


def test_thoughts_fold_into_billed_output_tokens() -> None:
    """Google's thoughtsTokenCount is additive, so billed output is the sum."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 19,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=8, cached_input_tokens=2)


def test_zero_thoughts_keep_output_at_candidates_token_count() -> None:
    """A reported zero thoughts count is valid and does not change output."""
    usage = _completed_usage(
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 2,
            "thoughtsTokenCount": 0,
        }
    )
    assert usage == Usage(input_tokens=11, output_tokens=5, cached_input_tokens=2)


@pytest.mark.parametrize("bad", ("3", True, -1, 1.5, []))
def test_malformed_thoughts_token_count_is_a_provider_response_error(bad: JsonValue) -> None:
    """A present thoughtsTokenCount that is not a non-negative integer fails closed.

    Args:
        bad: Malformed provider value that must not become a billed token count.
    """
    usage: JsonObject = {
        "promptTokenCount": 11,
        "candidatesTokenCount": 5,
        "cachedContentTokenCount": 2,
        "thoughtsTokenCount": bad,
    }
    with pytest.raises(ProviderResponseError, match="thoughtsTokenCount"):
        _completed_usage(usage)


def test_input_and_cached_counts_stay_subsets_of_prompt_tokens() -> None:
    """Prompt and cached-content counters stay unchanged when thoughts fold in."""
    usage = _completed_usage(
        {
            "promptTokenCount": 12,
            "candidatesTokenCount": 6,
            "cachedContentTokenCount": 4,
            "thoughtsTokenCount": 3,
        }
    )
    assert usage.input_tokens == 12
    assert usage.cached_input_tokens == 4
    assert usage.output_tokens == 9


@pytest.mark.parametrize("thoughts", (None, 0, 3))
def test_completed_and_streamed_gemini_usage_agree(thoughts: int | None) -> None:
    """Completed Python and native SSE responses bill the same additive output total.

    Args:
        thoughts: Thinking count reported separately by Gemini, or omitted.
    """
    native = pytest.importorskip("exp_gateway_native")
    usage: JsonObject = {
        "promptTokenCount": 11,
        "candidatesTokenCount": 5,
        "cachedContentTokenCount": 2,
        "totalTokenCount": 16 + (thoughts or 0),
    }
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    observed = _completed_usage(usage)
    payload: JsonObject = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": usage,
    }
    frame = f"data: {json.dumps(payload)}\n\n"
    result: JsonObject = json.loads(
        native.normalize_stream_fixture("gemini_generate_content", json.dumps([frame]))
    )
    assert result["failure"] is None
    events = result["events"]
    assert isinstance(events, list)
    usage_events = [
        event for event in events if isinstance(event, dict) and event.get("kind") == "usage"
    ]
    assert usage_events == [
        {
            "kind": "usage",
            "input_tokens": 11,
            "output_tokens": 5 + (thoughts or 0),
            "cached_input_tokens": 2,
            "reasoning_tokens": thoughts,
        }
    ]
    assert observed == Usage(
        input_tokens=11, output_tokens=5 + (thoughts or 0), cached_input_tokens=2
    )


@pytest.mark.parametrize("asynchronous", (False, True))
def test_gemini_completion_bills_thoughts_over_loopback(asynchronous: bool) -> None:
    """Actual HTTP completions expose the provider's combined output usage.

    Args:
        asynchronous: Whether to exercise the public async completion entrypoint.
    """
    requests: list[tuple[str, str | None, JsonObject]] = []

    class Handler(BaseHTTPRequestHandler):
        """Serve one deterministic Gemini response without external provider traffic."""

        def do_POST(self) -> None:
            """Capture the real request and return separate text and thinking counters."""
            payload: JsonObject = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("x-goog-api-key"), payload))
            body = json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 5,
                        "thoughtsTokenCount": 3,
                        "cachedContentTokenCount": 2,
                        "totalTokenCount": 19,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress fixture-only HTTP access logs.

            Args:
                format: Standard-library log format, intentionally unused.
                args: Standard-library log arguments, intentionally unused.
            """

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = GeminiClient(
                model=_snapshot(),
                api_key="fixture-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1beta",
            )
            request = ModelRequest(messages=(ModelMessage(role="user", content="Hello"),))
            response = (
                asyncio.run(run_then_close_pooled_client(client.complete_async(request)))
                if asynchronous
                else client.complete(request)
            )
            assert response.output.content == "ok"
            assert response.economics.usage == Usage(
                input_tokens=11, output_tokens=8, cached_input_tokens=2
            )
            assert len(requests) == 1
            path, key, payload = requests[0]
            assert path == "/v1beta/models/gemini-fixture:generateContent"
            assert key == "fixture-key"
            assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()

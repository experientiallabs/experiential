"""OpenAI SDK access and response-ID recording for independent learning endpoints.

Pass a complete API base URL ending in /v1. A fresh ``model_client()`` records one
customer agent episode's completion IDs; submit feedback against those IDs explicitly.
"""

from __future__ import annotations

from ipaddress import ip_address
from time import monotonic
from types import TracebackType
from typing import cast
from urllib.parse import urlsplit
from uuid import uuid4

import httpx2
from openai import OpenAI
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import TypeAdapter

from exp.common.claas.learning import FeedbackSubmission
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import BillingSource, ModelRequest, ModelResponse, ModelSnapshot
from exp.runtime.claas.client.wire import chat_payload
from exp.runtime.models.providers.openai_compatible import openai_compatible_response

_JSON_OBJECT = TypeAdapter(JsonObject)


def _loopback(hostname: str) -> bool:
    """Allow plaintext only for localhost or a literal loopback address, without DNS lookup."""
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


class LearningClient:
    """A synchronous customer-agent client with explicit lifetime and zero automatic retries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120,
        maximum_output_tokens: int = 512,
        http_client: httpx2.Client | None = None,
    ) -> None:
        """Own an HTTP client and official SDK for one authorized learning endpoint."""
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path.rstrip("/") != "/v1"
        ):
            raise ValueError("base_url must be a credential-free HTTP(S) API URL ending in /v1")
        if url.scheme == "http" and not _loopback(url.hostname):
            raise ValueError("base_url requires HTTPS unless its host is localhost or loopback")
        if not api_key or not model or not 0 < timeout_seconds <= 3600:
            raise ValueError("provide an API key, model, and finite timeout up to 3600 seconds")
        if not 1 <= maximum_output_tokens <= 131072:
            raise ValueError("maximum_output_tokens must be between one and 131072")
        self.model = model
        self.maximum_output_tokens = maximum_output_tokens
        self._base_url = base_url.rstrip("/") + "/"
        self._http = http_client or httpx2.Client(timeout=timeout_seconds)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.sdk = OpenAI(
            base_url=self._base_url,
            api_key=api_key,
            http_client=self._http,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.snapshot = ModelSnapshot(
            provider="claas",
            model_id=model,
            billing_source=BillingSource.CUSTOMER_MANAGED,
            connection_sha256=sha256_json({"base_url": self._base_url}),
            capabilities_sha256=sha256_json(
                {
                    "text": True,
                    "function_tools": True,
                    "streaming": False,
                    "temperature": 1,
                    "top_p": 1,
                }
            ),
        )

    def __enter__(self) -> LearningClient:
        """Return the explicitly owned client for a bounded customer-agent session."""
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close local connections without closing or draining the remote learning run."""
        self.close()

    def close(self) -> None:
        """Release the SDK transport, including a transport supplied by the caller."""
        self.sdk.close()

    def model_client(self) -> RecordingModelClient:
        """Create an isolated model-injection bridge for an existing AgentRuntime episode."""
        return RecordingModelClient(self)

    def submit_feedback(
        self,
        response_id: str,
        *,
        reward: float | None = None,
        success: bool | None = None,
        text: str | None = None,
    ) -> JsonObject:
        """Submit explicit sparse or text feedback against a standard completion response ID."""
        submission = FeedbackSubmission(
            response_id=response_id, reward=reward, success=success, text=text
        )
        return self._request(
            "POST", "feedback", submission.model_dump(mode="json", exclude_none=True)
        )

    def status(self) -> JsonObject:
        """Inspect the remote queue and run without importing optimization implementation."""
        return self._request("GET", "status")

    def trigger_train(self) -> JsonObject:
        """Request a bounded partial update; remote policy retains all limit authority."""
        return self._request("POST", "train")

    def drain(self) -> JsonObject:
        """Request a finite ready snapshot drain and return its explicit remaining-work report."""
        return self._request("POST", "drain")

    def _request(self, method: str, path: str, payload: JsonObject | None = None) -> JsonObject:
        """Use the same authenticated endpoint and validate management replies as JSON objects."""
        response = self._http.request(
            method, self._base_url + path, headers=self._headers, json=payload
        )
        response.raise_for_status()
        return _JSON_OBJECT.validate_json(response.content)


class RecordingModelClient:
    """Existing ModelClient contract plus exact HTTP response IDs for caller-owned scoring."""

    def __init__(self, client: LearningClient) -> None:
        """Bind a fresh bounded episode recorder without generating or changing model state."""
        self.client = client
        self._response_ids: list[str] = []

    @property
    def response_ids(self) -> tuple[str, ...]:
        """Return response IDs in successful call order, deduplicating exact request replays."""
        return tuple(self._response_ids)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one agent action using a new caller-owned idempotency key."""
        return self.complete_idempotent(request, idempotency_key=uuid4().hex)

    def complete_idempotent(self, request: ModelRequest, *, idempotency_key: str) -> ModelResponse:
        """Forward a stable retry identity and retain the server's standard feedback target."""
        if not idempotency_key.strip() or len(idempotency_key.encode()) > 512:
            raise ValueError("idempotency_key must contain 1 to 512 UTF-8 bytes")
        if len(self._response_ids) >= 10_000:
            raise ValueError("episode exceeds 10000 model calls; create a new episode recorder")
        payload = chat_payload(
            self.client.model, request, maximum_output_tokens=self.client.maximum_output_tokens
        )
        started = monotonic()
        response = self.client.sdk.chat.completions.create(
            **cast(CompletionCreateParamsNonStreaming, payload),
            extra_headers={"Idempotency-Key": idempotency_key},
        )
        normalized = openai_compatible_response(
            response.model_dump(mode="json"),
            configured_model=self.client.snapshot,
            latency_seconds=monotonic() - started,
        )
        if not response.id:
            raise ValueError(
                "learner response lacks the standard response ID required for feedback"
            )
        if response.id not in self._response_ids:
            self._response_ids.append(response.id)
        return normalized

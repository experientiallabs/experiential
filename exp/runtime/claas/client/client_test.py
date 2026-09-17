"""Official OpenAI SDK calls and management requests through the learning client."""

import json

import httpx2
import pytest

from exp.common.models import ModelMessage, ModelRequest
from exp.runtime.claas.client import LearningClient


def test_sdk_records_response_id_and_sends_feedback_to_same_endpoint() -> None:
    """The standard completion ID is the exact later feedback target."""
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        """Return strict SDK completion JSON and record management requests."""
        seen.append(request)
        if request.url.path == "/v1/chat/completions":
            return httpx2.Response(
                200,
                json={
                    "id": "response-one",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "student",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )
        return httpx2.Response(200, json={"state": "running"})

    http = httpx2.Client(transport=httpx2.MockTransport(handle))
    with LearningClient(
        base_url="https://learner.test/v1",
        api_key="local-test-key",
        model="student",
        http_client=http,
    ) as client:
        recorder = client.model_client()
        request = ModelRequest(messages=(ModelMessage(role="user", content="task"),))
        first = recorder.complete_idempotent(request, idempotency_key="retry-me")
        replay = recorder.complete_idempotent(request, idempotency_key="retry-me")
        assert replay.output == first.output
        assert first.output.content == "done"
        assert recorder.response_ids == ("response-one",)
        assert client.submit_feedback("response-one", text="Use the tool") == {"state": "running"}
        assert client.status() == {"state": "running"}
        assert client.trigger_train() == {"state": "running"}
        assert client.drain() == {"state": "running"}
    assert seen[0].headers["Idempotency-Key"] == "retry-me"
    assert all(request.headers["Authorization"] == "Bearer local-test-key" for request in seen)
    feedback = next(request for request in seen if request.url.path == "/v1/feedback")
    assert json.loads(feedback.content) == {"response_id": "response-one", "text": "Use the tool"}


@pytest.mark.parametrize(
    "base_url",
    ["https://user:secret@example.com/v1", "https://example.com/v1?q=x", "https://example.com"],
)
def test_client_rejects_ambiguous_or_credential_bearing_endpoint(base_url: str) -> None:
    """Credentials belong in authorization, and the API prefix is an explicit client contract."""
    with pytest.raises(ValueError, match="base_url"):
        LearningClient(base_url=base_url, api_key="test", model="student")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://learner.example/v1",
        "http://192.168.1.20/v1",
        "http://0.0.0.0/v1",
        "http://localhost.example/v1",
        "http://[2001:db8::1]/v1",
    ],
)
def test_remote_plaintext_endpoint_is_rejected_before_credentials_can_be_sent(
    base_url: str,
) -> None:
    """A named or nonloopback remote service must provide TLS for all authenticated traffic."""
    with pytest.raises(ValueError, match="HTTPS"):
        LearningClient(base_url=base_url, api_key="sensitive-test-key", model="student")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.2/v1",
        "http://[::1]:8000/v1",
        "https://learner.example/v1",
    ],
)
def test_local_loopback_and_https_endpoints_remain_supported(base_url: str) -> None:
    """Loopback development and encrypted hosted runs can still construct the official SDK."""
    with LearningClient(base_url=base_url, api_key="local-test-key", model="student"):
        pass

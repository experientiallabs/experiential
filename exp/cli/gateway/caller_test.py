"""Command-level tests for the caller-side live-gateway core loop."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.gateway import caller

_RAW_KEY = "wmo_vk_caller-secret-canary"


@pytest.fixture(autouse=True)
def _isolated_gateway_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep developer-machine gateway and OpenAI variables out of every test."""
    for variable in ("EXP_GATEWAY_URL", "OPENAI_BASE_URL", "EXP_GATEWAY_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


def _mock_gateway(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Route caller HTTP traffic through one in-memory gateway handler.

    Args:
        monkeypatch: Scoped patch context.
        handler: Callable answering every request the caller sends.

    Returns:
        Live list receiving each request the caller sends.
    """
    seen: list[httpx.Request] = []

    def build(url: str, *, timeout: float) -> httpx.Client:
        """Return a mock-backed client that records every outgoing request."""
        del timeout

        def record(request: httpx.Request) -> httpx.Response:
            """Capture the request before delegating to the test handler."""
            seen.append(request)
            return handler(request)

        return httpx.Client(base_url=url.rstrip("/"), transport=httpx.MockTransport(record))

    monkeypatch.setattr(caller, "_gateway_client", build)
    return seen


def test_call_streams_text_deltas_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default call streams SSE deltas and prints only the visible text."""
    chunks = (
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    frames = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve one streaming completion for the expected authenticated request."""
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {_RAW_KEY}"
        payload = json.loads(request.content)
        assert payload["model"] == "coding"
        assert payload["stream"] is True
        assert payload["messages"] == [{"role": "user", "content": "say hello"}]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=frames.encode(),
        )

    _mock_gateway(monkeypatch, respond)
    result = CliRunner().invoke(
        app,
        ["config", "gateway", "call", "coding", "say hello", "--key", _RAW_KEY],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "Hello world\n"


def test_call_json_prints_the_raw_completion_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The --json form makes one non-streaming call and passes the envelope through."""
    envelope = {
        "id": "chatcmpl-one",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve one completed envelope and require streaming to be disabled."""
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json=envelope)

    _mock_gateway(monkeypatch, respond)
    result = CliRunner().invoke(
        app,
        ["config", "gateway", "call", "coding", "say hello", "--key", _RAW_KEY, "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == envelope


def test_call_surfaces_gateway_error_code_message_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway error exits nonzero with its code, remediation text, and wait."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve one quota-exhausted error envelope."""
        del request
        return httpx.Response(
            429,
            headers={"Retry-After": "60"},
            json={
                "error": {
                    "message": "monthly gateway allocation is exhausted.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
        )

    _mock_gateway(monkeypatch, respond)
    result = CliRunner().invoke(
        app,
        ["config", "gateway", "call", "coding", "say hello", "--key", _RAW_KEY, "--json"],
    )

    assert result.exit_code == 1
    assert "gateway error insufficient_quota" in result.output
    assert "monthly gateway allocation is exhausted." in result.output
    assert "(Retry-After: 60s)" in result.output


def test_missing_key_names_the_exact_ways_to_provide_one() -> None:
    """A missing key is a usage error with every supported source spelled out."""
    result = CliRunner().invoke(app, ["config", "gateway", "call", "coding", "hello"])

    assert result.exit_code == 2
    normalized = " ".join(result.output.replace("│", " ").split())
    assert "pass --key" in normalized
    assert "EXP_GATEWAY_KEY" in normalized
    assert "exp config gateway key issue" in normalized


def test_unreachable_gateway_is_a_usage_error_naming_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection failure names the URL and how to start or select a gateway."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Refuse the connection like an unbound loopback port."""
        raise httpx.ConnectError("connection refused", request=request)

    _mock_gateway(monkeypatch, respond)
    result = CliRunner().invoke(
        app,
        ["config", "gateway", "models", "--key", _RAW_KEY, "--url", "http://127.0.0.1:9/v1"],
    )

    assert result.exit_code == 2
    normalized = " ".join(result.output.replace("│", " ").split())
    assert "no gateway answered at http://127.0.0.1:9/v1" in normalized
    assert "exp run" in normalized


def test_models_prints_the_caller_view_with_granted_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live caller view lists each granted alias with its active revision."""
    envelope = {
        "object": "list",
        "data": [
            {
                "id": "coding",
                "object": "model",
                "created": 0,
                "owned_by": "wmo",
                "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": "a" * 64},
            }
        ],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve the enriched discovery list for the authenticated caller."""
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == f"Bearer {_RAW_KEY}"
        return httpx.Response(200, json=envelope)

    seen = _mock_gateway(monkeypatch, respond)
    human = CliRunner().invoke(app, ["config", "gateway", "models", "--key", _RAW_KEY])
    raw = CliRunner().invoke(app, ["config", "gateway", "models", "--key", _RAW_KEY, "--json"])

    assert human.exit_code == 0, human.output
    assert human.output == "coding (revision revision-one)\n"
    assert raw.exit_code == 0, raw.output
    assert json.loads(raw.stdout) == envelope
    assert len(seen) == 2


def test_key_check_reports_granted_aliases_without_echoing_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid key prints its grants and never appears in any output."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve one granted alias for the presented key."""
        del request
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "coding",
                        "object": "model",
                        "owned_by": "wmo",
                        "wmo": {
                            "alias_revision_id": "revision-one",
                            "catalog_sha256": "a" * 64,
                        },
                    }
                ],
            },
        )

    _mock_gateway(monkeypatch, respond)
    human = CliRunner().invoke(app, ["config", "gateway", "key", "check", "--key", _RAW_KEY])
    raw = CliRunner().invoke(
        app,
        ["config", "gateway", "key", "check", "--key", _RAW_KEY, "--json"],
    )

    assert human.exit_code == 0, human.output
    assert human.output == "key valid; granted aliases: coding\n"
    assert raw.exit_code == 0, raw.output
    assert json.loads(raw.stdout) == {
        "schema_version": 1,
        "operation": "key.check",
        "valid": True,
        "granted_aliases": ["coding"],
    }
    assert _RAW_KEY not in human.output
    assert _RAW_KEY not in raw.output


def test_key_check_rejects_an_invalid_key_with_the_gateway_remediation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid key exits nonzero and relays the gateway's own next action."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve the gateway's sanitized invalid-key envelope."""
        del request
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": (
                        "The gateway key is invalid, expired, or revoked. Ask the gateway "
                        "operator to issue a new virtual key."
                    ),
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_key",
                }
            },
        )

    _mock_gateway(monkeypatch, respond)
    result = CliRunner().invoke(
        app,
        ["config", "gateway", "key", "check", "--key", _RAW_KEY, "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "operation": "key.check",
        "valid": False,
        "error_code": "invalid_key",
        "message": (
            "The gateway key is invalid, expired, or revoked. Ask the gateway operator "
            "to issue a new virtual key."
        ),
    }
    assert _RAW_KEY not in result.output

# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""An undefined key on a function tool is a named, actionable 400 on every rung.

Agno serializes its client-side approval flag ``requires_confirmation`` into
each function definition it sends (``Function.to_dict``); the field is not in
OpenAI's function tool schema on Chat or Responses. The refusal names the field
as unknown and says to remove it, before any rung is dialled, rather than
reporting ``Invalid value for '<param>'.``, which would read as a value fault.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.common.models.catalog import BillingSource
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_service_tiers_test import _body
from exp.runtime.models import registry

_PARAMETERS: JsonObject = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def _configure(root: Path, provider: str, base_url: str) -> str:
    """Seed one tool-capable alias on the given rung with a synthetic key.

    The ``openai`` rung keeps its built-in official origin (the test points
    ``registry.OPENAI_BASE_URL`` at the loopback provider); a chat rung takes
    the loopback origin as its configured base URL.
    """
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(
            provider=provider,
            base_url=base_url if provider == "openai-compatible" else None,
            api_key_env="TEST_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider",
        provider_model="gpt-fixture",
        exact_model_id="gpt-fixture",
        revision=None,
        capabilities=ModelCapabilities(supports_tools=True, maximum_output_tokens=128_000),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
            supports_strict_tools=True,
        ),
        prices=GatewayTokenPrices(),
        billing_source=BillingSource.CUSTOMER_MANAGED,
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="rev",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    return manager.issue_key(identity_id="default", key_id="key").raw_key


def _payload(surface: str, *, agno_flag: bool) -> tuple[JsonObject, str]:
    """Return a tool-carrying request and the field path an Agno flag would occupy."""
    flag: JsonObject = {"requires_confirmation": True} if agno_flag else {}
    if surface == "responses":
        tool: JsonObject = {
            "type": "function",
            "name": "delete_file",
            "parameters": _PARAMETERS,
            "strict": True,
            **flag,
        }
        return (
            {"model": "coding", "input": "Remove notes.txt.", "tools": [tool]},
            "tools.0.requires_confirmation",
        )
    function: JsonObject = {
        "name": "delete_file",
        "parameters": _PARAMETERS,
        "strict": True,
        **flag,
    }
    return (
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "Remove notes.txt."}],
            "tools": [{"type": "function", "function": function}],
        },
        "tools.0.function.requires_confirmation",
    )


@pytest.mark.parametrize("provider", ["openai-compatible", "openai"])
@pytest.mark.parametrize("surface", ["chat/completions", "responses"])
def test_unknown_function_tool_key_is_named_on_chat_and_openai_rungs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, surface: str
) -> None:
    """The flagged request is refused by name with no dispatch; the clean one serves."""
    received: list[JsonObject] = []
    # The openai lane dispatches on the Responses wire; a chat rung speaks Chat.
    body = _body(provider == "openai", None)

    class Provider(BaseHTTPRequestHandler):
        """Record every provider-bound request and answer one finite stream."""

        def do_POST(self) -> None:  # noqa: N802
            """Capture the dispatched wire payload."""
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep synthetic traffic out of test logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
    if provider == "openai":
        monkeypatch.setattr(registry, "OPENAI_BASE_URL", base_url)
    else:
        factory, _ = registry._HTTP_PROVIDERS[provider]
        monkeypatch.setattr(
            registry, "_HTTP_PROVIDERS", {**registry._HTTP_PROVIDERS, provider: (factory, base_url)}
        )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    raw_key = _configure(tmp_path, provider, base_url)
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        url = f"http://127.0.0.1:{gateway.port}/v1/{surface}"
        headers = {"Authorization": f"Bearer {raw_key}"}
        flagged, param = _payload(surface, agno_flag=True)
        refused = httpx.post(url, headers=headers, json=flagged, timeout=10)
        assert refused.status_code == 400, refused.text
        assert refused.json()["error"] == {
            "message": (
                f"Unknown parameter '{param}': this field is not supported. Remove it and resend."
            ),
            "type": "invalid_request_error",
            "param": param,
            "code": "invalid_parameter",
        }
        assert received == []

        clean, _ = _payload(surface, agno_flag=False)
        served = httpx.post(url, headers=headers, json=clean, timeout=10)
        assert served.status_code == 200, served.text
        assert "Hello" in served.text
        assert len(received) == 1
        # The defined keys, ``strict`` included, reach each wire in its own shape.
        definition: JsonObject = {
            "name": "delete_file",
            "parameters": _PARAMETERS,
            "strict": True,
        }
        expected_tool: JsonObject = (
            {"type": "function", **definition}
            if provider == "openai"
            else {"type": "function", "function": definition}
        )
        assert received[0]["tools"] == [expected_tool]
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

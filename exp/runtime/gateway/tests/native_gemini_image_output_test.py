"""Gemini image output, its metered image tokens, and malformed finishes over real serving.

A fake Gemini upstream answers like ``gemini-3.1-flash-image``: one signed
image part whose signature passes a megabyte, and a usage report whose
``candidatesTokensDetails`` names the IMAGE subset. A second fake answers a
first dial with ``MALFORMED_FUNCTION_CALL`` and the re-dial with text.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import _frame
from exp.runtime.models import registry

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
_PNG_B64 = base64.b64encode(_PNG).decode()

_IMAGE_TURN: list[JsonObject] = [
    {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "inlineData": {"mimeType": "image/png", "data": _PNG_B64},
                            "thoughtSignature": "A" * 1_200_000,
                        }
                    ],
                }
            }
        ]
    },
    {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "A red square."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 17,
            "candidatesTokenCount": 1126,
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 6},
                {"modality": "IMAGE", "tokenCount": 1120},
            ],
        },
    },
]

_MALFORMED: list[JsonObject] = [
    {
        "candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 0},
    }
]

_TEXT: list[JsonObject] = [
    {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1},
    }
]

Served = tuple[str, str, list[JsonObject], list[JsonObject]]


def _serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turns: Callable[[int], list[JsonObject]]
) -> Iterator[Served]:
    """Serve one Gemini alias whose upstream answers dial ``n`` with ``turns(n)``."""
    bodies: list[JsonObject] = []
    settlements: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Record each native request and stream the scripted answer."""

        def do_POST(self) -> None:  # noqa: N802
            """Answer one dial."""
            bodies.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            chunk = b"".join(_frame(frame) for frame in turns(len(bodies)))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Keep synthetic content out of the logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    factory, _ = registry._HTTP_PROVIDERS["gemini"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {
            **registry._HTTP_PROVIDERS,
            "gemini": (factory, f"http://127.0.0.1:{upstream.server_port}/v1"),
        },
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    key = _configure(tmp_path)
    gateway = _ServedGateway(tmp_path, _unused_port())
    settle = gateway.control_plane.settle

    def record_settle(argument: str) -> str:
        """Observe native settlement callbacks without replacing accounting."""
        settlements.append(json.loads(argument))
        return settle(argument)

    monkeypatch.setattr(gateway.control_plane, "settle", record_settle)
    gateway.start()
    try:
        yield f"http://127.0.0.1:{gateway.port}/v1", key, bodies, settlements
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def _configure(root: Path) -> str:
    """Seed one image-emitting Gemini deployment and return a granted key."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="gemini",
        connection_name="provider",
        provider_model="gemini-test-image",
        exact_model_id="gemini-test-image",
        revision=None,
        capabilities=ModelCapabilities(emits_images=True, maximum_output_tokens=32_768),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
        pricing_source="synthetic-test-prices",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="gemini",
        alias_name="gemini",
        revision_id="rev",
        pool_id="gemini",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="gemini")
    return manager.issue_key(identity_id="default", key_id="key").raw_key


@pytest.fixture
def image_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Served]:
    """A lane whose every dial generates the signed image."""
    yield from _serve(tmp_path, monkeypatch, lambda _: _IMAGE_TURN)


@pytest.fixture
def malformed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Served]:
    """A lane whose first dial ends MALFORMED_FUNCTION_CALL and later dials answer."""
    yield from _serve(tmp_path, monkeypatch, lambda dial: _MALFORMED if dial == 1 else _TEXT)


def _chat(base: str, key: str, stream: bool) -> httpx.Response:
    """Send one bounded Chat request and require success."""
    response = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "gemini",
            "max_tokens": 4096,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "messages": [{"role": "user", "content": "Draw a red square."}],
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response


@pytest.mark.parametrize("stream", [False, True])
def test_a_signed_image_reaches_the_caller_with_its_image_tokens_metered(
    image_lane: Served, stream: bool
) -> None:
    """The image arrives as an OpenRouter-style message image, metered apart from text."""
    base, key, bodies, settlements = image_lane
    response = _chat(base, key, stream)
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        deltas = [choice["delta"] for chunk in chunks for choice in chunk.get("choices", [])]
        images = [image for delta in deltas for image in delta.get("images", [])]
        text = "".join(delta.get("content") or "" for delta in deltas)
        usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
    else:
        message = response.json()["choices"][0]["message"]
        images, text, usage = message["images"], message["content"], response.json()["usage"]
    assert [image["image_url"]["url"] for image in images] == [f"data:image/png;base64,{_PNG_B64}"]
    assert text == "A red square."
    assert usage["completion_tokens"] == 1126
    assert usage["completion_tokens_details"]["image_tokens"] == 1120
    generation = cast("JsonObject", bodies[0]["generationConfig"])
    assert generation["responseModalities"] == ["TEXT", "IMAGE"]
    assert len(settlements) == 1
    assert cast("JsonObject", settlements[0]["usage"])["image_output_tokens"] == 1120


def test_a_malformed_function_call_is_redialed_on_the_same_lane(malformed_once: Served) -> None:
    """Google's sampling failure re-dials once and the caller sees the answer."""
    base, key, bodies, settlements = malformed_once
    response = _chat(base, key, stream=False)
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert len(bodies) == 2
    assert [settlement["outcome"] for settlement in settlements] == ["failed", "completed"]
    failure = cast("JsonObject", settlements[0]["failure"])
    assert failure["failure_class"] == "malformed_response"

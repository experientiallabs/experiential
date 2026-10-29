"""End-to-end Stability AI (Bedrock) image generation and edits on the native engine.

The native engine serves in a child process over a seeded root with a Bedrock
connection and three Stability aliases: ``ultra`` (Stable Image Ultra, a
generation model that also edits), ``inpaint`` (an Image Services tool), and
``unpriced`` (Ultra without a per-image card). The child's control plane is
the real one, except that it rewrites each admitted InvokeModel URL to a
loopback Stability mock and stamps ``sign_dispatch`` so the mock observes the
per-attempt signature. Requests drive ``POST /v1/images/generations`` and
``POST /v1/images/edits`` through the real Rust data plane with the official
OpenAI client and raw multipart, and the money is read back from the ledger.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import httpx
import openai
import pytest
from openai.types import ImagesResponse

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BilledUnitKind,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import (
    ConnectionConfig,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement

pytest.importorskip("exp_gateway_native")

_ULTRA_RATE = 140_000_000  # $0.14 per image (Stable Image Ultra on Bedrock).
_INPAINT_RATE = 70_000_000  # $0.07 per image (Stable Image Inpaint on Bedrock).
_RENDERED = base64.b64encode(b"\xff\xd8\xff\xe0rendered-jpeg").decode()
_INPUT = b"\x89PNG\r\n\x1a\ninput-pixels"
_MASK = b"\x89PNG\r\n\x1a\nmask-pixels"

_CHILD_SOURCE = """\
import json
import sys
from pathlib import Path

import exp_gateway_native

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_bridge import NativeControlPlane


class _RewritingControlPlane:
    \"\"\"Delegate to the real control plane, pointing InvokeModel at loopback.\"\"\"

    def __init__(self, inner, upstream_base):
        \"\"\"Wrap one real control plane above the loopback upstream.\"\"\"
        self._inner = inner
        self._upstream_base = upstream_base
        self._signatures = 0

    def __getattr__(self, name):
        \"\"\"Delegate every other boundary callback unchanged.\"\"\"
        return getattr(self._inner, name)

    def _rewrite(self, result):
        \"\"\"Point each admitted InvokeModel rung at the upstream, keeping its path.\"\"\"
        parsed = json.loads(result)
        for wire in parsed.get("route") or []:
            url = wire.get("url", "")
            if url.endswith("/invoke"):
                wire["url"] = self._upstream_base + url[url.index("/model/"):]
        return json.dumps(parsed, separators=(",", ":"))

    def admit_images(self, argument):
        \"\"\"Admit a generation for real, then rewrite its destination.\"\"\"
        return self._rewrite(self._inner.admit_images(argument))

    def admit_image_edits(self, argument):
        \"\"\"Admit an edit for real, then rewrite its destination.\"\"\"
        return self._rewrite(self._inner.admit_image_edits(argument))

    def sign_dispatch(self, argument):
        \"\"\"Stamp one counted signature over the frozen body's length.\"\"\"
        data = json.loads(argument)
        self._signatures += 1
        return json.dumps(
            {
                "headers": {
                    "authorization": (
                        f"AWS4-HMAC-SHA256 attempt-{self._signatures} bytes-{len(data['body'])}"
                    ),
                    "content-type": "application/json",
                    "accept": "application/json",
                }
            },
            separators=(",", ":"),
        )


components = load_gateway_components(
    Path(sys.argv[1]), environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
)
control = _RewritingControlPlane(
    NativeControlPlane(components), f"http://127.0.0.1:{sys.argv[3]}"
)
config = json.dumps(
    {
        "host": "127.0.0.1",
        "port": int(sys.argv[2]),
        "request_timeout_seconds": 30.0,
        "graceful_timeout_seconds": 1.0,
    }
)
exp_gateway_native.serve(control, config)
"""


@dataclass(frozen=True)
class _Seen:
    """One request the Stability mock received.

    Attributes:
        path: The request path (``/model/{id}/invoke``).
        authorization: The signed ``Authorization`` header.
        body: The exact JSON body.
        raw_length: The body's byte length.
    """

    path: str
    authorization: str
    body: JsonObject
    raw_length: int


class _StabilityUpstream(BaseHTTPRequestHandler):
    """Bedrock InvokeModel mock: the prompt selects the Stability answer."""

    seen: list[_Seen] = []
    lock = Lock()

    def _answer(self, status: int, body: JsonObject) -> None:
        """Send one JSON answer."""
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Record the request and answer one image, a filter, or an error."""
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw)
        with _StabilityUpstream.lock:
            _StabilityUpstream.seen.append(
                _Seen(
                    path=self.path,
                    authorization=self.headers.get("authorization", ""),
                    body=body,
                    raw_length=len(raw),
                )
            )
        prompt = str(body.get("prompt", ""))
        if prompt == "filtered":
            self._answer(200, {"finish_reasons": ["Filter reason: prompt"]})
            return
        self._answer(200, {"seeds": [42], "finish_reasons": [None], "images": [_RENDERED]})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


@dataclass(frozen=True)
class _Engine:
    """One served native engine.

    Attributes:
        base: The engine's base URL.
        raw_key: A granted gateway key.
        root: The seeded gateway root.
    """

    base: str
    raw_key: str
    root: Path


def _unused_port() -> int:
    """Allocate one currently unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _activate(
    root: Path, manager: GatewayManagement, alias: str, model: str, rate: int | None
) -> None:
    """Grant one Stability alias on the Bedrock connection."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="bedrock-main",
        provider_model=model,
        exact_model_id=f"{alias}-exact",
        revision=None,
        capabilities=ModelCapabilities(supports_image_generation=True),
        gateway_capabilities=GatewayDeploymentCapabilities(),
        prices=(
            GatewayTokenPrices()
            if rate is None
            else GatewayTokenPrices(
                units=GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": rate})
            )
        ),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=f"revision-{alias}",
        pool_id=alias,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id=alias)


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Engine]:
    """Serve one engine over a Bedrock connection with three Stability aliases."""
    root = tmp_path_factory.mktemp("native-stability-root")
    upstream_port = _unused_port()
    gateway_port = _unused_port()
    with _StabilityUpstream.lock:
        _StabilityUpstream.seen.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), _StabilityUpstream)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(root)
    upsert_connection(
        root,
        name="bedrock-main",
        connection=ConnectionConfig(provider="bedrock", region="us-west-2"),
        replace=False,
    )
    _activate(root, manager, "ultra", "stability.stable-image-ultra-v1:1", _ULTRA_RATE)
    _activate(root, manager, "inpaint", "us.stability.stable-image-inpaint-v1:0", _INPAINT_RATE)
    _activate(root, manager, "unpriced", "stability.stable-image-ultra-v1:1", None)
    child = subprocess.Popen(  # noqa: S603 - fixed argv built from this test.
        [sys.executable, "-c", _CHILD_SOURCE, str(root), str(gateway_port), str(upstream_port)],
        env=dict(os.environ),
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert child.poll() is None, f"native engine exited early: {child.returncode}"
            try:
                live = httpx.get(f"http://127.0.0.1:{gateway_port}/health/live", timeout=1)
                if live.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "native engine did not become live"
            time.sleep(0.05)
        yield _Engine(base=f"http://127.0.0.1:{gateway_port}", raw_key=raw_key, root=root)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)


def _last_attempt(engine: _Engine) -> tuple[object, ...]:
    """Read the newest attempt's state, settlement, cost, and billed units."""
    with sqlite3.connect(engine.root / "gateway" / "gateway.db") as connection:
        row = connection.execute(
            """
            SELECT state, budget_reserved_nano_usd, estimated_cost_nano_usd,
                   billed_unit_kind, billed_unit_variant, billed_quantity_milli
            FROM gateway_attempts ORDER BY rowid DESC LIMIT 1
            """
        ).fetchone()
    return tuple(row)


def _last_seen() -> _Seen:
    """The newest request the Stability mock received."""
    with _StabilityUpstream.lock:
        return _StabilityUpstream.seen[-1]


def _seen_count() -> int:
    """How many requests the Stability mock received."""
    with _StabilityUpstream.lock:
        return len(_StabilityUpstream.seen)


def _generate(engine: _Engine, body: JsonObject) -> httpx.Response:
    """POST one raw generation."""
    return httpx.post(
        f"{engine.base}/v1/images/generations",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=30.0,
    )


def _edit(
    engine: _Engine,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> httpx.Response:
    """POST one raw multipart edit."""
    return httpx.post(
        f"{engine.base}/v1/images/edits",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        data=fields,
        files=files,
        timeout=30.0,
    )


def test_official_client_generates_with_ultra_and_bills_one_image(engine: _Engine) -> None:
    """The SDK's generate call reaches InvokeModel signed, maps the answer, bills $0.14."""
    client = openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)
    raw = client.images.with_raw_response.generate(
        model="ultra", prompt="a red fox in snow", size="1792x1024", output_format="jpeg"
    )
    response = raw.parse()
    assert isinstance(response, ImagesResponse)
    assert raw.headers["x-gateway-provider"] == "bedrock"
    assert response.data is not None and len(response.data) == 1
    assert response.data[0].b64_json == _RENDERED
    assert response.output_format == "jpeg"
    seen = _last_seen()
    assert seen.path == "/model/stability.stable-image-ultra-v1%3A1/invoke"
    assert seen.body == {
        "prompt": "a red fox in snow",
        "aspect_ratio": "16:9",
        "output_format": "jpeg",
    }
    # The frozen body was signed immediately before the POST, over its exact bytes.
    assert seen.authorization.endswith(f"bytes-{seen.raw_length}")
    # The variant records the requested size; the flat card prices it.
    assert _last_attempt(engine) == (
        "completed",
        _ULTRA_RATE,
        _ULTRA_RATE,
        "image",
        "1792x1024",
        1_000,
    )


def test_official_client_edits_with_inpaint_and_a_mask(engine: _Engine) -> None:
    """The SDK's edit call carries image and mask as base64 and bills $0.07."""
    client = openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)
    response = client.images.edit(
        model="inpaint",
        image=("photo.png", _INPUT, "image/png"),
        mask=("mask.png", _MASK, "image/png"),
        prompt="a sleeping cat",
    )
    assert response.data is not None and response.data[0].b64_json == _RENDERED
    seen = _last_seen()
    assert seen.path == "/model/us.stability.stable-image-inpaint-v1%3A0/invoke"
    assert seen.body == {
        "prompt": "a sleeping cat",
        "image": base64.b64encode(_INPUT).decode(),
        "mask": base64.b64encode(_MASK).decode(),
    }
    assert _last_attempt(engine) == ("completed", _INPAINT_RATE, _INPAINT_RATE, "image", "", 1_000)


def test_raw_edit_forwards_typed_tool_controls(engine: _Engine) -> None:
    """Stability controls in the form reach the body typed; image-to-image gets a strength."""
    response = _edit(
        engine,
        {"model": "ultra", "prompt": "as a watercolor", "seed": "7", "strength": "0.4"},
        {"image": ("in.png", _INPUT, "image/png")},
    )
    assert response.status_code == 200, response.text
    assert _last_seen().body == {
        "prompt": "as a watercolor",
        "image": base64.b64encode(_INPUT).decode(),
        "strength": 0.4,
        "seed": 7,
    }


def test_a_content_filter_is_a_refusal_billed_nothing(engine: _Engine) -> None:
    """Stability's filter finish reason answers a 400 refusal and settles at $0."""
    response = _generate(engine, {"model": "ultra", "prompt": "filtered"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "refusal"
    assert "(prompt)" in response.json()["error"]["message"]
    state, _reserved, cost, _kind, _variant, quantity = _last_attempt(engine)
    assert state == "failed"
    assert cost in (None, 0)
    assert quantity in (None, 0)


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"model": "ultra", "prompt": "two foxes", "n": 2}, "n"),
        ({"model": "ultra", "prompt": "a fox", "quality": "hd"}, "quality"),
        ({"model": "ultra", "prompt": "a fox", "style": "vivid"}, "style"),
    ],
)
def test_controls_the_stability_wire_lacks_are_refused_before_dispatch(
    engine: _Engine, body: JsonObject, param: str
) -> None:
    """n > 1, quality, and style are refused by name and never reach the provider."""
    before = _seen_count()
    response = _generate(engine, body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert _seen_count() == before


def test_an_edit_tool_on_generations_points_to_edits(engine: _Engine) -> None:
    """Inpaint needs an input image; a text-only request names the edits route."""
    before = _seen_count()
    response = _generate(engine, {"model": "inpaint", "prompt": "a cat"})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "model"
    assert "/v1/images/edits" in response.json()["error"]["message"]
    assert _seen_count() == before


def test_an_unpriced_stability_lane_is_refused(engine: _Engine) -> None:
    """A Stability rung without a per-image card is never dispatched."""
    before = _seen_count()
    response = _generate(engine, {"model": "unpriced", "prompt": "a fox"})
    assert response.status_code == 400, response.text
    assert _seen_count() == before


@pytest.mark.parametrize(
    ("fields", "files", "param"),
    [
        ({"model": "inpaint", "prompt": "a cat"}, {}, "image"),
        ({"model": "inpaint"}, {"image": ("in.png", _INPUT, "image/png")}, "prompt"),
        (
            {"model": "inpaint", "prompt": "a cat", "left": "10"},
            {"image": ("in.png", _INPUT, "image/png")},
            "left",
        ),
        (
            {"model": "inpaint", "prompt": "a cat", "seed": "-3"},
            {"image": ("in.png", _INPUT, "image/png")},
            "seed",
        ),
        (
            {"model": "inpaint", "prompt": "a cat", "stream": "true"},
            {"image": ("in.png", _INPUT, "image/png")},
            "stream",
        ),
    ],
)
def test_invalid_edits_are_refused_by_field_before_dispatch(
    engine: _Engine,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    param: str,
) -> None:
    """A missing image or prompt, a foreign or out-of-range control: a 400 by name."""
    before = _seen_count()
    response = (
        _edit(engine, fields, files)
        if files
        else httpx.post(
            f"{engine.base}/v1/images/edits",
            headers={"authorization": f"Bearer {engine.raw_key}"},
            files={"model": (None, fields["model"]), "prompt": (None, fields["prompt"])},
            timeout=30.0,
        )
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert _seen_count() == before


def test_edits_require_multipart_and_a_key(engine: _Engine) -> None:
    """A JSON edit is refused; an unknown key is a 401 before any parsing."""
    json_edit = httpx.post(
        f"{engine.base}/v1/images/edits",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "inpaint", "prompt": "a cat"},
        timeout=10.0,
    )
    assert json_edit.status_code == 400
    assert json_edit.json()["error"]["param"] == "image"
    unknown = httpx.post(
        f"{engine.base}/v1/images/edits",
        headers={"authorization": "Bearer not-a-key"},
        files={"image": ("in.png", _INPUT, "image/png")},
        data={"model": "inpaint", "prompt": "a cat"},
        timeout=10.0,
    )
    assert unknown.status_code == 401

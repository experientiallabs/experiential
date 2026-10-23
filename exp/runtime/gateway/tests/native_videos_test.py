"""End-to-end ``/v1/videos`` through the served native engine against a loopback Veo.

One native serving subprocess serves a root with a chat alias (``coding``,
unused here) and a Veo alias (``veo-fast``) on a ``gemini`` connection whose
built-in origin the driver points at a loopback mock of the Gemini API's
long-running prediction operations. Callers use the official OpenAI SDK's
``client.videos`` methods. The tests check the provider wire (path, key,
``instances`` and ``parameters``), the job lifecycle (create, poll, content,
list, delete), ownership (a second identity sees nothing), and the money: the
reservation held while the job renders and the exact settlement afterwards,
read straight from the ledger.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
import openai
import pytest
from openai.types import Video

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BilledUnitKind,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.native_messages_test import (
    _DRIVER_SOURCE,
    _HOST,
    _REQUEST_TIMEOUT_SECONDS,
    _ServingEngine,
)

pytest.importorskip("exp_gateway_native")

_KEY = "provider-secret-canary"  # The shared driver forwards only TEST_PROVIDER_KEY.
_RATE_720P = 100_000_000  # Veo 3.1 Fast: $0.10 per second at 720p.
_RATE_1080P = 120_000_000  # Veo 3.1 Fast: $0.12 per second at 1080p.
_MP4 = b"\x00\x00\x00\x18ftypmp42veo-frames"
_ALIASES = ["coding", "veo-fast"]
_MODEL = "veo-3.1-fast-generate-preview"

# The shared driver, plus two test-only seams: the Gemini built-in origin is
# repointed at the loopback mock, and the video engine polls without the
# production five-second throttle so a test sees state changes immediately.
_VIDEO_DRIVER_SOURCE = _DRIVER_SOURCE.replace(
    "    config = json.loads(sys.argv[1])\n",
    """    config = json.loads(sys.argv[1])
    import exp.runtime.models.registry as video_registry
    from exp.runtime.gateway import native_videos
    from exp.runtime.gateway.video.engine import VideoEngine

    gemini_factory = video_registry._HTTP_PROVIDERS["gemini"][0]
    video_registry._HTTP_PROVIDERS = {
        **video_registry._HTTP_PROVIDERS,
        "gemini": (gemini_factory, config["gemini_base_url"]),
    }
    original_bind = native_videos.NativeVideosMixin._bind_videos

    def fast_bind(self, *, start_poller=True):
        \"\"\"Bind the engine with no poll throttle and a quick background poller.\"\"\"
        original_bind(self, start_poller=False)
        if self._videos is not None:
            self._videos._poll_interval = type(self._videos._poll_interval)(0)
            self._videos.start_poller(interval_seconds=0.2)

    native_videos.NativeVideosMixin._bind_videos = fast_bind
    del VideoEngine
""",
    1,
)


@dataclass
class _Submit:
    """What the gateway sent the mock for one job submit."""

    path: str
    api_key: str | None
    body: JsonObject = field(default_factory=dict)


class _Veo(BaseHTTPRequestHandler):
    """Loopback mock of Veo's predictLongRunning, operations, and downloads.

    The prompt selects the behavior: ``reject`` answers the submit with 400,
    ``broken`` ends the operation with an error, ``filtered`` ends it with no
    samples, and anything else completes on the second poll.
    """

    submits: list[_Submit] = []
    polls: dict[str, int] = {}
    prompts: dict[str, str] = {}
    download_keys: list[tuple[str, str | None]] = []
    port = 0
    lock = threading.Lock()

    def _json(self, status: int, payload: JsonObject) -> None:
        """Send one JSON response."""
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Accept one submit, or reject it when the prompt says so."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        path = urlsplit(self.path).path
        body = json.loads(raw)
        with self.lock:
            self.submits.append(
                _Submit(path=path, api_key=self.headers.get("x-goog-api-key"), body=body)
            )
            operation = f"op-{len(self.submits)}"
            self.prompts[operation] = body["instances"][0]["prompt"]
        if body["instances"][0]["prompt"] == "reject":
            self._json(400, {"error": {"code": 400, "message": "prompt rejected by the mock"}})
            return
        model = path.removeprefix("/v1beta/models/").removesuffix(":predictLongRunning")
        self._json(200, {"name": f"models/{model}/operations/{operation}"})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer one operation poll or one asset download."""
        path = urlsplit(self.path).path
        port = _Veo.port
        if path.startswith("/v1beta/files/"):
            operation = path.removeprefix("/v1beta/files/").removesuffix(":download")
            with self.lock:
                self.download_keys.append(("file", self.headers.get("x-goog-api-key")))
            self.send_response(302)
            self.send_header("location", f"http://{_HOST}:{port}/storage/{operation}.mp4")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        if path.startswith("/storage/"):
            with self.lock:
                self.download_keys.append(("storage", self.headers.get("x-goog-api-key")))
            self.send_response(200)
            self.send_header("content-type", "video/mp4")
            self.send_header("content-length", str(len(_MP4)))
            self.end_headers()
            self.wfile.write(_MP4)
            return
        operation = path.rsplit("/", 1)[-1]
        with self.lock:
            prompt = self.prompts.get(operation)
            self.polls[operation] = self.polls.get(operation, 0) + 1
            count = self.polls[operation]
        name = path.removeprefix("/v1beta/")
        if prompt is None:
            self._json(404, {"error": {"code": 404, "message": "no such operation"}})
        elif count < 2:
            self._json(200, {"name": name, "done": False})
        elif prompt == "broken":
            self._json(200, {"name": name, "done": True, "error": {"message": "render failed"}})
        elif prompt == "filtered":
            reasons = ["blocked by the mock safety filter"]
            self._json(
                200,
                {
                    "name": name,
                    "done": True,
                    "response": {"generateVideoResponse": {"raiMediaFilteredReasons": reasons}},
                },
            )
        else:
            uri = f"http://{_HOST}:{port}/v1beta/files/{operation}:download?alt=media"
            samples = [{"video": {"uri": uri}}]
            self._json(
                200,
                {
                    "name": name,
                    "done": True,
                    "response": {"generateVideoResponse": {"generatedSamples": samples}},
                },
            )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


@dataclass(frozen=True)
class _VideoEngine:
    """The served engine plus a second identity's key and the database path."""

    serving: _ServingEngine
    other_key: str
    database: Path


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_VideoEngine]:
    """Serve one native engine over a root with a Veo alias on the loopback mock."""
    root = tmp_path_factory.mktemp("native-videos-root")
    with _Veo.lock:
        _Veo.submits.clear()
        _Veo.polls.clear()
        _Veo.prompts.clear()
        _Veo.download_keys.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _Veo)
    _Veo.port = int(upstream.server_address[1])
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    gemini_base = f"http://{_HOST}:{_Veo.port}/v1beta"
    manager, raw_key = _configured_gateway(root, base_url=f"http://{_HOST}:9/v1")
    upsert_connection(
        root,
        name="gemini-main",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    _seed(manager, root)
    manager.create_identity(identity_id="other", display_name="Other")
    manager.add_grant(identity_id="other", alias_id="veo-fast")
    other_key = manager.issue_key(identity_id="other", key_id="key-other").raw_key
    driver = root / "native_videos_driver.py"
    driver.write_text(_VIDEO_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
            "gemini_base_url": gemini_base,
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = _KEY
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        port = _await_live(process, raw_key, stderr_log)
        yield _VideoEngine(
            serving=_ServingEngine(port=port, raw_key=raw_key, root=root),
            other_key=other_key,
            database=manager.database_path,
        )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _seed(manager: GatewayManagement, root: Path) -> None:
    """Register the Veo deployment with its per-resolution card and grant it."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="veo-fast",
        connection_name="gemini-main",
        provider_model=_MODEL,
        exact_model_id="veo-fast-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_video_generation=True),
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(
                kind=BilledUnitKind.VIDEO_SECOND,
                rates={"720p": _RATE_720P, "1080p": _RATE_1080P},
            )
        ),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="veo-fast",
        alias_name="veo-fast",
        revision_id="revision-veo-fast",
        pool_id="veo-fast",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="veo-fast")


def _await_live(process: subprocess.Popen[str], raw_key: str, stderr_log: Path) -> int:
    """Wait until the driver serves both aliases; return its port."""
    announced: list[int] = []

    def collect() -> None:
        """Collect the driver's announced ports from its stdout."""
        assert process.stdout is not None
        for line in process.stdout:
            announced.append(int(json.loads(line)["port"]))

    threading.Thread(target=collect, daemon=True).start()
    deadline = time.monotonic() + 30
    while True:
        if announced:
            port = announced[-1]
            try:
                models = httpx.get(
                    f"http://{_HOST}:{port}/v1/models",
                    headers={"authorization": f"Bearer {raw_key}"},
                    timeout=2.0,
                )
                if (
                    models.status_code == 200
                    and sorted(item["id"] for item in models.json()["data"]) == _ALIASES
                ):
                    return port
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
        assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
        assert time.monotonic() < deadline, "native engine never became live"
        time.sleep(0.05)


def _client(engine: _VideoEngine, *, key: str | None = None) -> openai.OpenAI:
    """Build an official OpenAI client pointed at the served engine."""
    return openai.OpenAI(
        base_url=f"{engine.serving.base}/v1", api_key=key or engine.serving.raw_key
    )


def _create(client: openai.OpenAI, **fields: object) -> Video:
    """Call ``client.videos.create`` with gateway sizes and durations.

    The SDK types ``size`` and ``seconds`` as Sora's literals; the gateway's
    providers render others, which the SDK sends unchanged.
    """
    return cast(Any, client.videos).create(**fields)


def _attempt(engine: _VideoEngine, *, latest: int = 0) -> sqlite3.Row:
    """Read one of the most recent ledger attempts (``latest=0`` is the newest)."""
    with sqlite3.connect(engine.database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT state, budget_reserved_nano_usd, budget_settled_nano_usd,
                   estimated_cost_nano_usd, billed_unit_kind, billed_unit_variant,
                   billed_quantity_milli, billed_unit_rate
            FROM gateway_attempts ORDER BY rowid DESC LIMIT ?
            """,
            (latest + 1,),
        ).fetchall()
    return rows[latest]


def _wait_terminal(client: openai.OpenAI, video_id: str) -> Video:
    """Poll one video until it completes or fails."""
    deadline = time.monotonic() + 20
    while True:
        video = client.videos.retrieve(video_id)
        if video.status in {"completed", "failed"}:
            return video
        assert time.monotonic() < deadline, f"video stuck {video.status}"
        time.sleep(0.1)


def test_create_poll_download_and_settle_exact_seconds(engine: _VideoEngine) -> None:
    """The SDK's create, retrieve, and download_content serve one Veo job end to end."""
    client = _client(engine)
    created = _create(
        client,
        model="veo-fast",
        prompt="a paper boat in the rain",
        seconds="8",
        size="1280x720",
    )
    assert created.object == "video"
    assert created.model == "veo-fast"
    assert created.status == "queued"
    assert created.seconds == "8"
    assert created.size == "1280x720"
    submit = _Veo.submits[-1]
    assert submit.path == f"/v1beta/models/{_MODEL}:predictLongRunning"
    assert submit.api_key == _KEY
    assert submit.body == {
        "instances": [{"prompt": "a paper boat in the rain"}],
        "parameters": {"aspectRatio": "16:9", "resolution": "720p", "durationSeconds": 8},
    }
    held = _attempt(engine)
    assert held["state"] == "dispatched"
    assert held["budget_reserved_nano_usd"] == 8 * _RATE_720P
    finished = _wait_terminal(client, created.id)
    assert finished.status == "completed"
    assert finished.progress == 100
    assert finished.completed_at is not None
    assert finished.expires_at == finished.completed_at + 2 * 24 * 60 * 60
    settled = _attempt(engine)
    assert settled["state"] == "completed"
    assert settled["billed_unit_kind"] == "video_second"
    assert settled["billed_unit_variant"] == "720p"
    assert settled["billed_quantity_milli"] == 8_000
    assert settled["billed_unit_rate"] == _RATE_720P
    assert settled["estimated_cost_nano_usd"] == 8 * _RATE_720P
    assert settled["budget_settled_nano_usd"] == 8 * _RATE_720P
    _Veo.download_keys.clear()
    content = client.videos.download_content(created.id)
    assert content.read() == _MP4
    # The key reaches the API host's file endpoint and never the redirect target.
    assert _Veo.download_keys == [("file", _KEY), ("storage", None)]


def test_the_server_side_poller_settles_a_job_nobody_polls(engine: _VideoEngine) -> None:
    """A job the caller never reads still settles: the background poller drives it."""
    created = _create(
        _client(engine),
        model="veo-fast",
        prompt="a lighthouse at dusk",
        seconds="8",
        size="1920x1080",
    )
    deadline = time.monotonic() + 20
    while _attempt(engine)["state"] == "dispatched":
        assert time.monotonic() < deadline, "the poller never settled the job"
        time.sleep(0.1)
    settled = _attempt(engine)
    assert settled["billed_unit_variant"] == "1080p"
    assert settled["estimated_cost_nano_usd"] == 8 * _RATE_1080P
    assert _client(engine).videos.retrieve(created.id).status == "completed"


@pytest.mark.parametrize(
    ("prompt", "code"), [("broken", "video_generation_failed"), ("filtered", "content_filtered")]
)
def test_a_failed_render_settles_at_zero(engine: _VideoEngine, prompt: str, code: str) -> None:
    """A provider-failed or filtered render bills zero seconds and releases the hold."""
    client = _client(engine)
    created = _create(client, model="veo-fast", prompt=prompt, size="720x1280")
    assert _attempt(engine)["budget_reserved_nano_usd"] == 4 * _RATE_720P
    finished = _wait_terminal(client, created.id)
    assert finished.status == "failed"
    assert finished.error is not None
    assert finished.error.code == code
    settled = _attempt(engine)
    assert settled["state"] == "failed"
    assert settled["billed_quantity_milli"] == 0
    assert settled["estimated_cost_nano_usd"] == 0
    assert settled["budget_settled_nano_usd"] == 0
    with pytest.raises(openai.BadRequestError) as refused:
        client.videos.download_content(created.id)
    assert refused.value.code == "video_not_ready"


def test_a_rejected_submit_settles_zero_and_answers_the_provider_reason(
    engine: _VideoEngine,
) -> None:
    """A provider 400 at submit creates no job, bills nothing, and is the caller's 400."""
    with pytest.raises(openai.BadRequestError) as rejected:
        _create(_client(engine), model="veo-fast", prompt="reject")
    assert "prompt rejected by the mock" in str(rejected.value)
    attempt = _attempt(engine)
    assert attempt["state"] == "failed"
    assert attempt["budget_settled_nano_usd"] == 0
    listed = [video.id for video in _client(engine).videos.list()]
    assert all(video_id.startswith("video_") for video_id in listed)
    assert len(listed) == len(set(listed))


def test_another_identity_cannot_see_read_download_or_delete(engine: _VideoEngine) -> None:
    """Every read is owner-scoped: a second identity's lookups answer 404."""
    owner = _client(engine)
    created = _create(owner, model="veo-fast", prompt="a fox in snow")
    _wait_terminal(owner, created.id)
    stranger = _client(engine, key=engine.other_key)
    with pytest.raises(openai.NotFoundError):
        stranger.videos.retrieve(created.id)
    with pytest.raises(openai.NotFoundError):
        stranger.videos.download_content(created.id)
    with pytest.raises(openai.NotFoundError):
        stranger.videos.delete(created.id)
    assert created.id not in [video.id for video in stranger.videos.list()]
    assert created.id in [video.id for video in owner.videos.list()]


def test_delete_hides_a_finished_video(engine: _VideoEngine) -> None:
    """Deleting a finished video removes it from reads and listings."""
    client = _client(engine)
    created = _create(client, model="veo-fast", prompt="a kite over the sea")
    _wait_terminal(client, created.id)
    deleted = client.videos.delete(created.id)
    assert deleted.deleted is True
    assert deleted.object == "video.deleted"
    with pytest.raises(openai.NotFoundError):
        client.videos.retrieve(created.id)
    assert created.id not in [video.id for video in client.videos.list()]


def test_a_reference_image_travels_inline(engine: _VideoEngine) -> None:
    """A multipart input_reference reaches Veo as an inline first-frame image."""
    png = b"\x89PNG\r\n\x1a\nreference-frame"
    _create(
        _client(engine),
        model="veo-fast",
        prompt="animate this frame",
        input_reference=("frame.png", png, "image/png"),
    )
    instances = cast(list[JsonObject], _Veo.submits[-1].body["instances"])
    instance = instances[0]
    assert instance["image"] == {
        "inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}
    }


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"seconds": "12"}, "seconds"),
        ({"size": "1792x1024"}, "size"),
        ({"size": "1920x1080", "seconds": "4"}, "seconds"),
        ({"size": "854x480"}, "size"),
    ],
)
def test_unrenderable_parameters_are_field_specific_400s(
    engine: _VideoEngine, fields: dict[str, str], param: str
) -> None:
    """Parameters no rung renders are refused by name before any provider call."""
    before = len(_Veo.submits)
    response = httpx.post(
        f"{engine.serving.base}/v1/videos",
        headers={"authorization": f"Bearer {engine.serving.raw_key}"},
        json={"model": "veo-fast", "prompt": "anything", **fields},
        timeout=10.0,
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert len(_Veo.submits) == before


def test_a_chat_alias_is_not_a_video_model(engine: _VideoEngine) -> None:
    """An alias with no video rung answers the unsupported-capability 400."""
    with pytest.raises(openai.BadRequestError) as refused:
        _create(_client(engine), model="coding", prompt="anything")
    assert refused.value.code == "unsupported_capability"


def test_an_unknown_video_id_is_404(engine: _VideoEngine) -> None:
    """A made-up id answers exactly like a foreign one."""
    with pytest.raises(openai.NotFoundError):
        _client(engine).videos.retrieve("video_doesnotexist")

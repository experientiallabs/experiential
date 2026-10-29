"""End-to-end ElevenLabs speech and transcription through the served native engine.

One native serving subprocess (the driver from ``native_messages_test``) serves
a root with a chat alias (``coding``, unused here) and two ElevenLabs aliases on
an ``elevenlabs`` connection pointed at a loopback mock of ElevenLabs' API:
``el-voice`` (speech billed per character) and ``el-scribe`` (transcription
billed per metered second). Callers use the OpenAI audio wire; the tests check
that the gateway speaks ElevenLabs' wire upstream (path, query, ``xi-api-key``,
body, multipart ``model_id``) and read the money back from the ledger.
"""

from __future__ import annotations

import email.parser
import email.policy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import openai
import pytest

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
from exp.runtime.gateway.tests.native_audio_test import _wav
from exp.runtime.gateway.tests.native_messages_test import (
    _DRIVER_SOURCE,
    _HOST,
    _REQUEST_TIMEOUT_SECONDS,
    _ServingEngine,
)
from exp.runtime.models.providers.elevenlabs_audio import ELEVENLABS_OPENAI_VOICES

pytest.importorskip("exp_gateway_native")

_MP3 = b"ID3\x04elevenlabs-frames"
_KEY = "provider-secret-canary"  # The shared driver forwards only TEST_PROVIDER_KEY.
_CHARACTER_RATE = 40_000  # $0.04 per thousand characters (Flash v2.5).
_SECOND_RATE = 61_112  # $0.22 per hour (Scribe v2), rounded up per second.
_ALIASES = ["coding", "el-scribe", "el-voice"]
_WAV_HEADER_BYTES = 44


@dataclass
class _Record:
    """What the gateway sent the mock ElevenLabs API for one call."""

    route: str
    path: str
    query: dict[str, list[str]]
    api_key: str | None
    body: JsonObject = field(default_factory=dict)
    fields: dict[str, list[str]] = field(default_factory=dict)
    audio_bytes: int = 0


class _ElevenLabs(BaseHTTPRequestHandler):
    """Loopback mock of ElevenLabs' text-to-speech and speech-to-text routes."""

    records: list[_Record] = []
    lock = threading.Lock()

    def _answer(self, status: int, body: bytes, content_type: str) -> None:
        """Send one raw response with an explicit content type and length."""
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer one canned speech or transcription response."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        target = urlsplit(self.path)
        record = _Record(
            route="",
            path=target.path,
            query=parse_qs(target.query),
            api_key=self.headers.get("xi-api-key"),
        )
        if target.path.startswith("/v1/text-to-speech/"):
            record.route = "speech"
            record.body = json.loads(raw)
            with self.lock:
                self.records.append(record)
            self._answer(200, _MP3, "audio/mpeg")
        elif target.path == "/v1/speech-to-text":
            record.route = "transcription"
            self._transcription(record, raw)
        else:
            self._answer(404, b'{"detail": "not found"}', "application/json")

    def _transcription(self, record: _Record, raw: bytes) -> None:
        """Parse one multipart upload, record it, and meter the uploaded audio."""
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {self.headers['content-type']}\r\n\r\n".encode() + raw
        )
        audio = b""
        for part in message.iter_parts():
            name = str(part.get_param("name", header="content-disposition"))
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            if name == "file":
                audio = payload
            else:
                record.fields.setdefault(name, []).append(payload.decode())
        record.audio_bytes = len(audio)
        with self.lock:
            self.records.append(record)
        # 16 kHz mono 16-bit is 32,000 bytes per second.
        seconds = (len(audio) - _WAV_HEADER_BYTES) / 32_000
        body = {
            "language_code": "eng",
            "language_probability": 0.98,
            "text": "hello world",
            "words": [
                {"text": "hello", "type": "word", "start": 0.0, "end": 0.4},
                {"text": " ", "type": "spacing", "start": 0.4, "end": 0.45},
                {"text": "world", "type": "word", "start": 0.45, "end": 0.9},
            ],
            "audio_duration_secs": seconds,
        }
        self._answer(200, json.dumps(body).encode(), "application/json")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name.
        """Keep the test output quiet."""
        del format, args


def _seed(
    manager: GatewayManagement,
    root: Path,
    alias: str,
    model: str,
    *,
    gateway_capabilities: GatewayDeploymentCapabilities,
    prices: GatewayTokenPrices,
) -> None:
    """Register one ElevenLabs deployment and grant its alias."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="elevenlabs-main",
        provider_model=model,
        exact_model_id=f"{alias}-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=gateway_capabilities,
        prices=prices,
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
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one native engine over a root with two ElevenLabs audio aliases."""
    root = tmp_path_factory.mktemp("native-elevenlabs-root")
    with _ElevenLabs.lock:
        _ElevenLabs.records.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _ElevenLabs)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    base_url = f"http://{_HOST}:{upstream.server_address[1]}/v1"
    manager, raw_key = _configured_gateway(root, base_url=base_url)
    upsert_connection(
        root,
        name="elevenlabs-main",
        connection=ConnectionConfig(
            provider="elevenlabs", base_url=base_url, api_key_env="TEST_PROVIDER_KEY"
        ),
        replace=False,
    )
    _seed(
        manager,
        root,
        "el-voice",
        "eleven_flash_v2_5",
        gateway_capabilities=GatewayDeploymentCapabilities(supports_speech=True),
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": _CHARACTER_RATE})
        ),
    )
    _seed(
        manager,
        root,
        "el-scribe",
        "scribe_v2",
        gateway_capabilities=GatewayDeploymentCapabilities(supports_transcription=True),
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.AUDIO_SECOND, rates={"": _SECOND_RATE})
        ),
    )
    driver = root / "native_elevenlabs_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps({"root": str(root), "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS})
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
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Collect the driver's announced ports from its stdout."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        threading.Thread(target=_collect_announcements, daemon=True).start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
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
                        break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _cost(engine: _ServingEngine) -> int:
    """Read the ledger's known attributed cost in nano-USD."""
    totals = httpx.get(f"{engine.base}/usage.json", timeout=5.0).json()["totals"]
    cost = totals["known_estimated_cost_nano_usd"]
    assert isinstance(cost, int)
    return cost


def _last(route: str) -> _Record:
    """Return the most recent mock call recorded for one route."""
    with _ElevenLabs.lock:
        return next(record for record in reversed(_ElevenLabs.records) if record.route == route)


def _client(engine: _ServingEngine) -> openai.OpenAI:
    """Build an official OpenAI client pointed at the served engine."""
    return openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)


def test_openai_speech_reaches_elevenlabs_wire_and_bills_characters(
    engine: _ServingEngine,
) -> None:
    """An OpenAI voice name and format reach ElevenLabs' path, query, and body."""
    before = _cost(engine)
    text = "Hello from ElevenLabs."
    raw = _client(engine).audio.speech.with_raw_response.create(
        model="el-voice", input=text, voice="alloy", response_format="opus", speed=1.1
    )
    assert raw.content == _MP3
    assert raw.headers["content-type"] == "audio/mpeg"
    upstream = _last("speech")
    assert upstream.path == f"/v1/text-to-speech/{ELEVENLABS_OPENAI_VOICES['alloy']}"
    assert upstream.query == {"output_format": ["opus_48000_64"]}
    assert upstream.api_key == _KEY
    assert upstream.body == {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"speed": 1.1},
    }
    assert _cost(engine) == before + len(text) * _CHARACTER_RATE


def test_an_elevenlabs_voice_id_passes_through(engine: _ServingEngine) -> None:
    """A twenty-character ElevenLabs voice id is used as given, mp3 by default."""
    _client(engine).audio.speech.create(model="el-voice", input="Hi", voice="JBFqnCBsd6RMkjVDRZzb")
    upstream = _last("speech")
    assert upstream.path == "/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb"
    assert upstream.query == {"output_format": ["mp3_44100_128"]}


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"voice": "nobody"}, "voice"),
        ({"response_format": "flac"}, "response_format"),
        ({"speed": 2.0}, "speed"),
        ({"instructions": "whisper it"}, "instructions"),
    ],
)
def test_options_elevenlabs_cannot_honor_are_refused_on_their_field(
    engine: _ServingEngine, extra: JsonObject, param: str
) -> None:
    """A voice, format, speed, or instruction ElevenLabs cannot carry is a 400, unbilled."""
    before = _cost(engine)
    body: JsonObject = {"model": "el-voice", "input": "Hi", "voice": "alloy", **extra}
    response = httpx.post(
        f"{engine.base}/v1/audio/speech",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=10.0,
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert _cost(engine) == before


def test_transcription_uploads_model_id_and_bills_audio_duration(engine: _ServingEngine) -> None:
    """The upload carries ``model_id`` and ``language_code``; ``audio_duration_secs`` bills."""
    before = _cost(engine)
    audio = _wav(2.0)
    transcript = _client(engine).audio.transcriptions.create(
        model="el-scribe", file=("clip.wav", audio, "audio/wav"), language="en"
    )
    assert transcript.text == "hello world"
    upstream = _last("transcription")
    assert upstream.path == "/v1/speech-to-text"
    assert upstream.api_key == _KEY
    assert upstream.audio_bytes == len(audio)
    assert upstream.fields == {"model_id": ["scribe_v2"], "language_code": ["en"]}
    assert _cost(engine) == before + 2 * _SECOND_RATE


def test_transcription_renders_text_and_verbose_json(engine: _ServingEngine) -> None:
    """``text`` and ``verbose_json`` are rendered from ElevenLabs' one answer shape."""
    auth = {"authorization": f"Bearer {engine.raw_key}"}
    files = {"file": ("clip.wav", _wav(1.0), "audio/wav")}
    text = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "el-scribe", "response_format": "text"},
        files=files,
        timeout=10.0,
    )
    assert text.status_code == 200
    assert text.text == "hello world"
    verbose = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={
            "model": "el-scribe",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
        files=files,
        timeout=10.0,
    )
    assert verbose.status_code == 200
    rendered = verbose.json()
    assert rendered["task"] == "transcribe"
    assert rendered["language"] == "eng"
    assert rendered["duration"] == 1.0
    assert [word["word"] for word in rendered["words"]] == ["hello", "world"]


def test_a_transcription_prompt_is_refused_for_elevenlabs(engine: _ServingEngine) -> None:
    """ElevenLabs takes no prompt, so the request is refused on ``prompt`` before upload."""
    response = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        data={"model": "el-scribe", "prompt": "names: Ada"},
        files={"file": ("clip.wav", _wav(0.5), "audio/wav")},
        timeout=10.0,
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "prompt"


def test_chat_on_an_elevenlabs_alias_is_refused_unbilled(engine: _ServingEngine) -> None:
    """ElevenLabs serves no chat: a completion naming its alias is a 4xx, never dispatched."""
    before = _cost(engine)
    with _ElevenLabs.lock:
        calls = len(_ElevenLabs.records)
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "el-voice", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )
    assert 400 <= response.status_code < 500, response.text
    with _ElevenLabs.lock:
        assert len(_ElevenLabs.records) == calls
    assert _cost(engine) == before

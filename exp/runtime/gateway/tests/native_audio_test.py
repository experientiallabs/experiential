"""End-to-end speech and transcription tests against the served native engine.

One shared native serving subprocess (the driver from ``native_messages_test``)
serves a seeded root with a chat alias (``coding``) and four audio aliases on
one OpenAI-compatible loopback connection: ``narrator`` (speech billed per
character from a unit card), ``voice-tokens`` (speech billed by the token usage
of the provider's SSE format), ``scribe`` (transcription billed per metered
second), and ``scribe-tokens`` (transcription billed by token usage). The tests
drive ``/v1/audio/speech`` and ``/v1/audio/transcriptions`` through the real
Rust data plane and python control plane with the official OpenAI client and
raw HTTP, and read the money back from the ledger.
"""

from __future__ import annotations

import base64
import email.parser
import email.policy
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import openai
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BilledUnitKind,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    GatewayUnitPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.native_messages_test import (
    _DRIVER_SOURCE,
    _HOST,
    _REQUEST_TIMEOUT_SECONDS,
    _ServingEngine,
)

pytest.importorskip("exp_gateway_native")

_MP3 = b"ID3\x04fake-mp3-frames"
_CHARACTER_RATE = 15_000  # $15 per million characters (tts-1).
_SECOND_RATE = 100_000  # $0.006 per minute (whisper-1).
_TOKEN_INPUT_RATE = 2_500_000_000  # $2.50 per million input tokens.
_TOKEN_OUTPUT_RATE = 10_000_000_000  # $10 per million output tokens.
_ALIASES = ["coding", "narrator", "scribe", "scribe-tokens", "voice-tokens"]


_WAV_HEADER_BYTES = 44


def _wav(seconds: float) -> bytes:
    """A silent 16 kHz mono 16-bit WAV of the given duration."""
    data = b"\x00\x00" * int(16_000 * seconds)
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


@dataclass(frozen=True)
class _Record:
    """What the gateway sent the provider for one audio call."""

    route: str
    speech: JsonObject = field(default_factory=dict)
    fields: dict[str, list[str]] = field(default_factory=dict)
    audio_bytes: int = 0


class _AudioUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible ``/audio/*`` mock recording what the gateway sent."""

    records: list[_Record] = []
    lock = threading.Lock()

    def _answer(self, status: int, body: bytes, content_type: str) -> None:
        """Send one raw response with an explicit content type and length."""
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: JsonObject) -> None:
        """Send one JSON response."""
        self._answer(status, json.dumps(body).encode(), "application/json")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer one canned speech or transcription response."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        if self.path.endswith("/audio/speech"):
            self._speech(json.loads(raw))
        elif self.path.endswith("/audio/transcriptions"):
            self._transcription(raw)
        else:
            self._json(404, {"error": {"message": "unknown route", "type": "invalid_request"}})

    def _speech(self, payload: JsonObject) -> None:
        """Record and answer one speech call (raw audio, or SSE when asked)."""
        with self.lock:
            self.records.append(_Record(route="speech", speech=payload))
        if payload["input"] == "reject-voice":
            self._json(
                400,
                {
                    "error": {
                        "message": "Invalid value for voice: 'nobody'.",
                        "type": "invalid_request_error",
                        "param": "voice",
                    }
                },
            )
            return
        if payload.get("stream_format") == "sse":
            events = [
                {"type": "speech.audio.delta", "audio": base64.b64encode(_MP3[:6]).decode()},
                {"type": "speech.audio.delta", "audio": base64.b64encode(_MP3[6:]).decode()},
                {
                    "type": "speech.audio.done",
                    "usage": {"input_tokens": 4, "output_tokens": 120, "total_tokens": 124},
                },
            ]
            stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            self._answer(200, stream.encode(), "text/event-stream")
            return
        self._answer(200, _MP3, "audio/mpeg")

    def _transcription(self, raw: bytes) -> None:
        """Parse one multipart upload, record it, and answer by model."""
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {self.headers['content-type']}\r\n\r\n".encode() + raw
        )
        fields: dict[str, list[str]] = {}
        audio = b""
        for part in message.iter_parts():
            name = str(part.get_param("name", header="content-disposition"))
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            if name == "file":
                audio = payload
            else:
                fields.setdefault(name, []).append(payload.decode())
        with self.lock:
            self.records.append(
                _Record(route="transcription", fields=fields, audio_bytes=len(audio))
            )
        model = fields["model"][0]
        if model.endswith("tokens-model"):
            self._json(
                200,
                {
                    "text": "hello from tokens",
                    "usage": {"type": "tokens", "input_tokens": 60, "output_tokens": 4},
                },
            )
            return
        # Meter what was actually uploaded: 16 kHz mono 16-bit is 32,000 bytes/s.
        seconds = (len(audio) - _WAV_HEADER_BYTES) / 32_000
        body: JsonObject = {"text": "hello world"}
        if fields["response_format"] == ["verbose_json"]:
            body.update({"task": "transcribe", "language": "english", "duration": seconds})
        else:
            body["usage"] = {"type": "duration", "seconds": seconds}
        self._json(200, body)

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
    """Register one audio deployment and grant its alias."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="provider-main",
        provider_model=model,
        exact_model_id=f"{alias}-exact",
        revision=None,
        # gpt-4o-transcribe's provider-enforced output cap; token lanes need one.
        capabilities=ModelCapabilities(maximum_output_tokens=2_000),
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
    """Serve one shared native engine over a root with chat and audio aliases."""
    root = tmp_path_factory.mktemp("native-audio-root")
    with _AudioUpstream.lock:
        _AudioUpstream.records.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _AudioUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root, base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1"
    )
    token_prices = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=_TOKEN_INPUT_RATE,
        output_nano_usd_per_million_tokens=_TOKEN_OUTPUT_RATE,
    )
    speech = GatewayDeploymentCapabilities(supports_speech=True)
    transcription = GatewayDeploymentCapabilities(supports_transcription=True)
    _seed(
        manager,
        root,
        "narrator",
        "narrator-model",
        gateway_capabilities=speech,
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": _CHARACTER_RATE})
        ),
    )
    _seed(
        manager,
        root,
        "voice-tokens",
        "voice-tokens-model",
        gateway_capabilities=speech,
        prices=token_prices,
    )
    _seed(
        manager,
        root,
        "scribe",
        "scribe-model",
        gateway_capabilities=transcription,
        prices=GatewayTokenPrices(
            units=GatewayUnitPrices(kind=BilledUnitKind.AUDIO_SECOND, rates={"": _SECOND_RATE})
        ),
    )
    _seed(
        manager,
        root,
        "scribe-tokens",
        "scribe-tokens-model",
        gateway_capabilities=transcription,
        prices=token_prices,
    )
    driver = root / "native_audio_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps({"root": str(root), "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS})
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
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


def _totals(engine: _ServingEngine) -> JsonObject:
    """Read the served engine's ledger totals."""
    return httpx.get(f"{engine.base}/usage.json", timeout=5.0).json()["totals"]


def _cost(engine: _ServingEngine) -> int:
    """Read the ledger's known attributed cost in nano-USD."""
    cost = _totals(engine)["known_estimated_cost_nano_usd"]
    assert isinstance(cost, int)
    return cost


def _last(route: str) -> _Record:
    """Return the most recent upstream call recorded for one route."""
    with _AudioUpstream.lock:
        return next(record for record in reversed(_AudioUpstream.records) if record.route == route)


def _client(engine: _ServingEngine) -> openai.OpenAI:
    """Build an official OpenAI client pointed at the served engine."""
    return openai.OpenAI(base_url=f"{engine.base}/v1", api_key=engine.raw_key)


def test_speech_bills_input_characters_at_the_unit_rate(engine: _ServingEngine) -> None:
    """A per-character lane returns the provider's audio and bills every character."""
    before = _cost(engine)
    text = "Hello from the gateway."
    raw = _client(engine).audio.speech.with_raw_response.create(
        model="narrator", input=text, voice="alloy", response_format="mp3"
    )
    assert raw.headers["x-gateway-alias"] == "narrator"
    assert raw.headers["content-type"] == "audio/mpeg"
    assert raw.content == _MP3
    assert _last("speech").speech == {
        "model": "narrator-model",
        "input": text,
        "voice": "alloy",
        "response_format": "mp3",
    }
    assert _cost(engine) == before + len(text) * _CHARACTER_RATE


def test_token_priced_speech_reassembles_the_metered_stream(engine: _ServingEngine) -> None:
    """A token lane is asked for SSE, the audio is reassembled, and both legs bill."""
    before = _cost(engine)
    raw = _client(engine).audio.speech.with_raw_response.create(
        model="voice-tokens", input="Hi", voice="coral", instructions="cheerful"
    )
    assert raw.content == _MP3
    assert raw.headers["content-type"] == "audio/mpeg"
    assert _last("speech").speech["stream_format"] == "sse"
    expected = (4 * _TOKEN_INPUT_RATE + 120 * _TOKEN_OUTPUT_RATE + 500_000) // 1_000_000
    assert _cost(engine) == before + expected


def test_transcription_bills_metered_seconds_and_keeps_the_audio_in_the_data_plane(
    engine: _ServingEngine,
) -> None:
    """The official multipart upload reaches the provider intact and bills its seconds."""
    before = _cost(engine)
    audio = _wav(2.4)
    transcript = _client(engine).audio.transcriptions.create(
        model="scribe", file=("clip.wav", audio, "audio/wav"), language="en", temperature=0.2
    )
    assert transcript.text == "hello world"
    upstream = _last("transcription")
    assert upstream.audio_bytes == len(audio)
    assert upstream.fields == {
        "model": ["scribe-model"],
        "response_format": ["json"],
        "language": ["en"],
        "temperature": ["0.2"],
    }
    # 2.4 metered seconds at 100,000 nano-USD per second.
    assert _cost(engine) == before + 240_000


def test_text_and_verbose_formats_render_from_the_metered_answer(engine: _ServingEngine) -> None:
    """``text`` is served from the provider's json answer; ``verbose_json`` meters its duration."""
    files = {"file": ("clip.wav", _wav(1.0), "audio/wav")}
    auth = {"authorization": f"Bearer {engine.raw_key}"}
    text = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "scribe", "response_format": "text"},
        files=files,
        timeout=30.0,
    )
    assert text.status_code == 200, text.text
    assert text.text == "hello world"
    assert text.headers["content-type"].startswith("text/plain")
    before = _cost(engine)
    verbose = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={
            "model": "scribe",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
        files=files,
        timeout=30.0,
    )
    assert verbose.status_code == 200, verbose.text
    assert verbose.json()["duration"] == 1.0
    assert _last("transcription").fields["timestamp_granularities[]"] == ["word", "segment"]
    assert _cost(engine) == before + 100_000


def test_token_priced_transcription_and_json_base64_upload(engine: _ServingEngine) -> None:
    """A JSON ``input_audio`` upload works and a token lane bills its usage."""
    before = _cost(engine)
    audio = _wav(0.5)
    response = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "scribe-tokens",
            "input_audio": {"data": base64.b64encode(audio).decode(), "format": "wav"},
        },
        timeout=30.0,
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hello from tokens"
    assert _last("transcription").audio_bytes == len(audio)
    expected = (60 * _TOKEN_INPUT_RATE + 4 * _TOKEN_OUTPUT_RATE + 500_000) // 1_000_000
    assert _cost(engine) == before + expected


def test_wrong_surface_unsupported_fields_and_unknown_keys_are_refused(
    engine: _ServingEngine,
) -> None:
    """Field-specific 400s for a wrong-surface alias, an unsupported field, or a URL
    source; the uniform 401 for an unknown key whatever the body."""
    auth = {"authorization": f"Bearer {engine.raw_key}"}
    chat = httpx.post(
        f"{engine.base}/v1/audio/speech",
        headers=auth,
        json={"model": "coding", "input": "hi", "voice": "alloy"},
        timeout=10.0,
    )
    assert chat.status_code == 400
    assert chat.json()["error"]["param"] == "model"
    assert "does not synthesize speech" in chat.json()["error"]["message"]
    crossed = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "narrator"},
        files={"file": ("a.wav", _wav(0.1), "audio/wav")},
        timeout=10.0,
    )
    assert crossed.status_code == 400
    assert "does not transcribe audio" in crossed.json()["error"]["message"]
    streamed = httpx.post(
        f"{engine.base}/v1/audio/speech",
        headers=auth,
        json={"model": "narrator", "input": "hi", "voice": "alloy", "stream_format": "sse"},
        timeout=10.0,
    )
    assert streamed.status_code == 400
    assert streamed.json()["error"]["param"] == "stream_format"
    unknown_format = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        json={"model": "scribe", "input_audio": {"data": "AAAA", "format": "wma"}},
        timeout=10.0,
    )
    assert unknown_format.status_code == 400
    assert unknown_format.json()["error"]["param"] == "input_audio.format"
    unmeasurable = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "scribe"},
        files={"file": ("noise.mp3", b"not audio at all", "audio/mpeg")},
        timeout=10.0,
    )
    assert unmeasurable.status_code == 400
    assert unmeasurable.json()["error"]["param"] == "file"
    assert "duration could not be measured" in unmeasurable.json()["error"]["message"]
    fetched = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        json={"model": "scribe", "source_url": "http://169.254.169.254/latest"},
        timeout=10.0,
    )
    assert fetched.status_code == 400
    assert "source_url is not supported" in fetched.json()["error"]["message"]
    for path in ("speech", "transcriptions"):
        unknown = httpx.post(
            f"{engine.base}/v1/audio/{path}",
            headers={"authorization": "Bearer not-a-key"},
            json={"model": "narrator", "input": "hi", "voice": "alloy"},
            timeout=10.0,
        )
        assert unknown.status_code == 401


def test_oversized_or_repeated_audio_request_parts_are_refused_before_the_bridge(
    engine: _ServingEngine,
) -> None:
    """A second file or repeated field, or a text field past its byte cap, is a 400
    on that field; a speech body past its cap is a 413."""
    auth = {"authorization": f"Bearer {engine.raw_key}"}
    twice = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "scribe"},
        files=[
            ("file", ("a.wav", _wav(0.1), "audio/wav")),
            ("file", ("b.wav", _wav(0.2), "audio/wav")),
        ],
        timeout=10.0,
    )
    assert twice.status_code == 400
    assert twice.json()["error"]["param"] == "file"
    flooded = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "scribe", "prompt": "x" * (16 * 1024 + 1)},
        files={"file": ("a.wav", _wav(0.1), "audio/wav")},
        timeout=10.0,
    )
    assert flooded.status_code == 400
    assert flooded.json()["error"]["param"] == "prompt"
    doubled = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": ["scribe", "narrator"]},
        files={"file": ("a.wav", _wav(0.1), "audio/wav")},
        timeout=10.0,
    )
    assert doubled.status_code == 400
    assert doubled.json()["error"]["param"] == "model"
    oversized = httpx.post(
        f"{engine.base}/v1/audio/speech",
        headers=auth,
        json={"model": "narrator", "input": "x" * (200 * 1024), "voice": "alloy"},
        timeout=10.0,
    )
    assert oversized.status_code == 413
    long_name = httpx.post(
        f"{engine.base}/v1/audio/transcriptions",
        headers=auth,
        data={"model": "scribe"},
        files={"file": ("a" * 300 + ".wav", _wav(0.1), "audio/wav")},
        timeout=10.0,
    )
    assert long_name.status_code == 400
    assert long_name.json()["error"]["param"] == "file"
    audio = base64.b64encode(_wav(0.1)).decode()
    for input_audio, param in (
        ({"data": audio, "fromat": "wav"}, "input_audio.fromat"),
        ({"data": audio, "format": 3}, "input_audio.format"),
    ):
        refused = httpx.post(
            f"{engine.base}/v1/audio/transcriptions",
            headers=auth,
            json={"model": "scribe", "input_audio": input_audio},
            timeout=10.0,
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["param"] == param


def test_provider_client_error_relays_the_parameter(engine: _ServingEngine) -> None:
    """A provider 400 reaches the caller as a 400 naming the rejected field."""
    response = httpx.post(
        f"{engine.base}/v1/audio/speech",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "narrator", "input": "reject-voice", "voice": "nobody"},
        timeout=10.0,
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "voice"

"""Audio admission preserves observation and enforcement across the native data plane."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import httpx
import pytest

from exp.common.models import (
    BilledUnitKind,
    BillingSource,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    GatewayUnitPrices,
)
from exp.runtime.gateway.guardrails.contracts import GuardrailOutcome
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_observation_test import _Classifier, _Engine, _policy
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_audio_test import (
    _CHARACTER_RATE,
    _MP3,
    _SECOND_RATE,
    _AudioUpstream,
    _seed,
    _wav,
)


@pytest.mark.parametrize("surface", ["speech", "transcription"])
@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_native_audio_observes_without_inspection_and_enforces_before_acceptance(
    tmp_path: Path,
    surface: Literal["speech", "transcription"],
    mode: Literal["observe", "enforce"],
) -> None:
    """Drive real audio HTTP and verify provider dispatch, billing, and unsupported coverage."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier, _policy().model_copy(update={"mode": mode}))
    with _AudioUpstream.lock:
        _AudioUpstream.records.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _AudioUpstream)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    try:
        manager, key = _configured_gateway(
            tmp_path,
            base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            billing_source=BillingSource.HOST_MANAGED,
        )
        speech = surface == "speech"
        _seed(
            manager,
            tmp_path,
            "audio",
            "audio-model",
            gateway_capabilities=GatewayDeploymentCapabilities(
                supports_speech=speech, supports_transcription=not speech
            ),
            prices=GatewayTokenPrices(
                units=GatewayUnitPrices(
                    kind=BilledUnitKind.CHARACTER if speech else BilledUnitKind.AUDIO_SECOND,
                    rates={"": _CHARACTER_RATE if speech else _SECOND_RATE},
                )
            ),
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test"})
        control = NativeControlPlane(components, guardrails=engine)
        with _serving(control) as gateway:
            headers = {"authorization": f"Bearer {key}"}
            if speech:
                response = httpx.post(
                    gateway + "/v1/audio/speech",
                    headers=headers,
                    json={"model": "audio", "input": "Hi", "voice": "alloy"},
                    timeout=5,
                )
            else:
                response = httpx.post(
                    gateway + "/v1/audio/transcriptions",
                    headers=headers,
                    data={"model": "audio"},
                    files={"file": ("clip.wav", _wav(1), "audio/wav")},
                    timeout=5,
                )
            totals = json.loads(control.usage_json("{}"))["totals"]
        engine.close(timeout_seconds=2)
        assert classifier.input_calls == 0
        with _AudioUpstream.lock:
            dispatches = len(_AudioUpstream.records)
        if mode == "observe":
            assert response.status_code == 200, response.text
            if speech:
                assert response.content == _MP3
            else:
                assert response.json()["text"] == "hello world"
            assert dispatches == 1
            assert totals["requests"] == 1
            assert totals["known_estimated_cost_nano_usd"] == (
                2 * _CHARACTER_RATE if speech else _SECOND_RATE
            )
            assert engine.observed == [("platform-observer", None, GuardrailOutcome.UNSUPPORTED)]
        else:
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "unsupported_capability"
            assert dispatches == 0
            assert totals["requests"] == 0
            assert totals["known_estimated_cost_nano_usd"] == 0
            assert engine.observed == []
    finally:
        engine.close(timeout_seconds=2)
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)

"""Tests for the audio admission's wrong-surface refusal and dead-rung escalation."""

import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from exp.common.models import (
    BilledUnitKind,
    BillingSource,
    ExactModelDeployment,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
    GatewayUnitPrices,
)
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway import native_audio
from exp.runtime.gateway.audio_contracts import SpeechRequest
from exp.runtime.gateway.contracts import DirectTarget
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_execution import DispatchableRoute


def test_the_wrong_surface_refusal_names_the_model_and_the_right_route() -> None:
    """The 400 tells the caller what the alias cannot do and where to send the request."""
    error = native_audio.not_an_audio_model_error("coding", "transcription")
    assert (error.status_code, error.detail.param) == (400, "model")
    assert "does not transcribe audio" in error.detail.message
    assert "/v1/audio/transcriptions" in error.detail.message


def _deployment(name: str, *, speech: bool) -> ExactModelDeployment:
    """Build one host-managed per-character deployment that may claim speech."""
    return ExactModelDeployment(
        deployment_id=f"deployment-{name}",
        source_alias="voice",
        exact_model_id="exact-one",
        connection=f"connection-{name}",
        provider="openai",
        provider_model="tts-1",
        billing_source=BillingSource.HOST_MANAGED,
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                units=GatewayUnitPrices(kind=BilledUnitKind.CHARACTER, rates={"": 15_000})
            ),
            pricing_source="provider-docs",
            pricing_effective_at=datetime(2026, 10, 6, tzinfo=UTC),
            capabilities=GatewayDeploymentCapabilities(supports_speech=speech),
        ),
    )


class _Plane:
    """A fake audio plane recording how one admission ended."""

    def __init__(self, deployments: tuple[ExactModelDeployment, ...]) -> None:
        """Serve a route of the given deployments and record every outcome."""
        self.finished: list[object] = []
        self.escalated: list[str] = []
        route = SimpleNamespace(deployments=deployments)
        self._components = SimpleNamespace(
            routes=SimpleNamespace(resolve_direct=lambda _authorization: route),
            runtime_catalogs={},
        )
        self._accounting = SimpleNamespace(
            finish_request_quietly=lambda _authorization, failure: self.finished.append(failure)
        )

    def _escalate_accepted(self, _authorization: object, reason: str) -> str:
        """Record the escalation reason in place of handing the request to Python."""
        self.escalated.append(reason)
        return "escalated"


def _admit(
    monkeypatch: pytest.MonkeyPatch, deployments: tuple[ExactModelDeployment, ...], live: int
) -> _Plane:
    """Route one speech request where only the rung at ``live`` resolved at admission."""
    plane = _Plane(deployments)
    dispatchable = DispatchableRoute(
        indexes=(live,),
        resolved_wires=cast(Any, ((SimpleNamespace(speech_url="u"), None),)),
        dead=(),
    )
    monkeypatch.setattr(native_audio, "dispatchable_route_profiles", lambda *_: dispatchable)
    monkeypatch.setattr(native_audio, "record_dead_admission_rungs", lambda *_, **__: None)
    authorization = SimpleNamespace(target=DirectTarget(pool_id="pool"), alias="voice")
    native_audio._route(  # noqa: SLF001
        cast(Any, plane),
        cast(Any, authorization),
        SpeechRequest(input="hi", voice="alloy"),
        0.0,
        surface="speech",
        rung_url=native_audio._speech_rung,  # noqa: SLF001
        payload=lambda *_: {},
        extra={},
    )
    return plane


def test_a_dead_speech_rung_escalates_while_a_speechless_alias_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the speech rung is down: escalate. No rung claims speech: a 400."""
    dead_speech = (_deployment("speech", speech=True), _deployment("chat", speech=False))
    plane = _admit(monkeypatch, dead_speech, live=1)
    assert plane.escalated == ["every audio-capable deployment was unavailable at admission"]
    assert plane.finished == []
    speechless = (_deployment("chat", speech=False), _deployment("other", speech=False))
    with pytest.raises(NativeBridgeError, match="does not synthesize speech"):
        _admit(monkeypatch, speechless, live=1)


def test_an_upload_that_spent_the_budget_is_refused_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data plane's remaining budget bounds admission; a spent one accepts nothing."""
    accepted: list[object] = []
    deadlines: list[float] = []

    def authorize_request(**fields: object) -> object:
        """Record the deadline the store would register and authorize the request."""
        deadlines.append(cast(float, fields["deadline_monotonic"]))
        return SimpleNamespace()

    plane = SimpleNamespace(
        _request_timeout_seconds=600.0,
        _components=SimpleNamespace(store=SimpleNamespace(authorize_request=authorize_request)),
        _guardrails=None,
        _write_ledger=SimpleNamespace(accept_request=lambda **fields: accepted.append(fields)),
    )
    monkeypatch.setattr(native_audio, "authorize_serving_model_chains", lambda _c, a: a)
    monkeypatch.setattr(native_audio, "with_client_identity", lambda a, _d: a)
    monkeypatch.setattr(native_audio, "require_unguarded_surface", lambda *_: None)
    request = SpeechRequest(input="hi", voice="alloy")
    with pytest.raises(NativeBridgeError) as refused:
        native_audio._accept(  # noqa: SLF001
            cast(Any, plane), {"raw_key": "k", "remaining_milli": 0}, "voice", request, "speech"
        )
    assert json.loads(refused.value.public_error_json)["status_code"] == 408
    assert accepted == []
    _, deadline = native_audio._accept(  # noqa: SLF001
        cast(Any, plane), {"raw_key": "k", "remaining_milli": 30_000}, "voice", request, "speech"
    )
    assert len(accepted) == 1
    assert deadline == deadlines[-1]
    assert deadline - time.monotonic() <= 30

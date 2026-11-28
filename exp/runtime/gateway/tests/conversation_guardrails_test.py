"""Denied conversation continuation through real native HTTP, with synthetic providers."""

import json
from pathlib import Path

import httpx
import pytest

from exp.common.models import BillingSource
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import ClassifierVerdict, GuardrailCheck
from exp.runtime.gateway.guardrails.user_conversation import UserConversationWindow
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _provider
from exp.runtime.gateway.tests.guardrail_search_output_test import _frames
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_gemini_thought_signature_test import (
    _configure as _configure_gemini,
)
from exp.runtime.gateway.tests.native_waterfall_test import (
    _content_chunk,
    _sse_frame,
    _terminal_frames,
)
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine
from exp.runtime.models import registry


class _HistoryClassifier(_Classifier):
    """Use deterministic user-history scoring to isolate continuation transport behavior."""

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Flag only the synthetic original user request, including on follow-ups."""
        del check
        window = UserConversationWindow(10000, 20000, "fixture")
        return ClassifierVerdict(flagged="harmful fixture" in window.state(request))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("store", [False, True])
def test_denied_responses_turn_can_continue_without_losing_user_input(
    tmp_path: Path, stream: bool, store: bool
) -> None:
    """A denial receipt preserves context under normal storage consent only."""
    with _provider(_content_chunk("synthetic answer") + _terminal_frames()) as (provider, _):
        _, key = _configured_gateway(
            tmp_path, base_url=provider, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=_engine(_HistoryClassifier("block")))
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            denied = client.post(
                "/v1/responses",
                json={
                    "model": "coding",
                    "input": "harmful fixture",
                    "stream": stream,
                    "store": store,
                },
            )
            assert denied.status_code == 400, denied.text
            if not store:
                assert "response_id" not in denied.json()["error"]
            receipt = denied.json()["error"].get("response_id", "resp_not_retained")
            for followup in ("please", "continue"):
                result = client.post(
                    "/v1/responses",
                    json={
                        "model": "coding",
                        "input": followup,
                        "previous_response_id": receipt,
                        "stream": stream,
                    },
                )
                assert result.status_code == 400, result.text
                expected = "content_filter" if store else "previous_response_not_found"
                assert result.json()["error"]["code"] == expected
                if store:
                    receipt = result.json()["error"]["response_id"]
            isolated = client.post("/v1/responses", json={"model": "coding", "input": "please"})
            assert isolated.status_code == 200, isolated.text


class _RefusedHistoryClassifier(_HistoryClassifier):
    """Let a provider refuse the first turn, then inspect its retained history."""

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Flag the synthetic original only when a second user turn follows it."""
        verdict = await super().inspect_input(request=request, check=check)
        users = sum(message.role == "user" for message in request.messages)
        return ClassifierVerdict(flagged=verdict.flagged and users > 1)


@pytest.mark.parametrize("stream", [False, True])
def test_provider_refusal_preserves_input_for_the_next_moderation_check(
    tmp_path: Path, stream: bool
) -> None:
    """Completed provider refusals retain user history on both Responses transports."""
    with _provider(_frames("refusal", "synthetic refusal")) as (provider, _):
        _, key = _configured_gateway(
            tmp_path, base_url=provider, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(
            components, guardrails=_engine(_RefusedHistoryClassifier("block"))
        )
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            refused = client.post(
                "/v1/responses",
                json={
                    "model": "coding",
                    "input": "harmful fixture",
                    "stream": stream,
                },
            )
            assert refused.status_code == 200, refused.text
            if stream:
                events = [
                    json.loads(line[6:])
                    for line in refused.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                completed = next(
                    item for item in events if item.get("type") == "response.completed"
                )
                receipt = completed["response"]["id"]
            else:
                receipt = refused.json()["id"]
            followup = client.post(
                "/v1/responses",
                json={
                    "model": "coding",
                    "input": "please",
                    "previous_response_id": receipt,
                    "stream": stream,
                },
            )
            assert followup.status_code == 400, followup.text
            assert followup.json()["error"]["code"] == "content_filter"


@pytest.mark.parametrize("stream", [False, True])
def test_early_encrypted_input_denial_has_no_unbound_receipt(tmp_path: Path, stream: bool) -> None:
    """An input verdict before route acquisition stays a denial, not a retention error."""
    import time

    class SlowStartControl(NativeControlPlane):
        def start_attempt(self, argument: str) -> str:
            """Allow the input classifier to finish before a route is registered."""
            time.sleep(0.5)
            return super().start_attempt(argument)

    with _provider(_content_chunk("synthetic answer") + _terminal_frames()) as (provider, _):
        _, key = _configured_gateway(
            tmp_path, base_url=provider, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = SlowStartControl(components, guardrails=_engine(_HistoryClassifier("block")))
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            denied = client.post(
                "/v1/responses",
                json={
                    "model": "coding",
                    "input": [
                        {
                            "type": "reasoning",
                            "id": "rs_fixture",
                            "encrypted_content": "fixture",
                            "summary": [],
                        },
                        {"role": "user", "content": "harmful fixture"},
                    ],
                    "stream": stream,
                },
            )
            assert denied.status_code == 400, denied.text
            assert denied.json()["error"]["code"] == "content_filter", denied.text
            assert "response_id" not in denied.json()["error"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider_refusal", [False, True])
def test_refusal_retention_failure_preserves_original_outcome(
    tmp_path: Path, stream: bool, provider_refusal: bool
) -> None:
    """A failed denial receipt write never turns moderation into an availability error."""

    class FailingRetentionControl(NativeControlPlane):
        def remember(self, argument: str) -> str:
            """Fail only refusal retention, leaving ordinary requests unchanged."""
            if json.loads(argument).get("refusal"):
                raise RuntimeError("synthetic continuation store failure")
            return super().remember(argument)

    frames = (
        _frames("refusal", "synthetic refusal")
        if provider_refusal
        else _content_chunk("synthetic answer") + _terminal_frames()
    )
    classifier = (
        _RefusedHistoryClassifier("block") if provider_refusal else _HistoryClassifier("block")
    )
    with _provider(frames) as (provider, _):
        _, key = _configured_gateway(
            tmp_path, base_url=provider, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = FailingRetentionControl(components, guardrails=_engine(classifier))
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            denied = client.post(
                "/v1/responses",
                json={"model": "coding", "input": "harmful fixture", "stream": stream},
            )
            if provider_refusal:
                assert denied.status_code == 200, denied.text
                if stream:
                    assert '"type":"response.completed"' in denied.text
                    assert '"type":"error"' not in denied.text
                else:
                    assert denied.json()["status"] == "completed"
            else:
                assert denied.status_code == 400, denied.text
                assert denied.json()["error"]["code"] == "content_filter"
                assert "response_id" not in denied.json()["error"]
            allowed = client.post(
                "/v1/responses",
                json={"model": "coding", "input": "benign fixture", "stream": stream},
            )
            assert allowed.status_code == 200, allowed.text


@pytest.mark.parametrize("store", [False, True])
def test_content_free_provider_refusal_retains_input(tmp_path: Path, store: bool) -> None:
    """A provider content-filter terminal without refusal text can still continue."""
    frames = (
        _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]})
        + b"data: [DONE]\n\n"
    )
    with _provider(frames) as (provider, _):
        _, key = _configured_gateway(
            tmp_path, base_url=provider, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(
            components, guardrails=_engine(_RefusedHistoryClassifier("block"))
        )
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            denied = client.post(
                "/v1/responses",
                json={"model": "coding", "input": "harmful fixture", "store": store},
            )
            assert denied.status_code == 200, denied.text
            assert denied.json()["status"] == "completed"
            receipt = denied.json()["id"]
            followup = client.post(
                "/v1/responses",
                json={"model": "coding", "input": "please", "previous_response_id": receipt},
            )
            assert followup.status_code == 400, followup.text
            expected = "content_filter" if store else "previous_response_not_found"
            assert followup.json()["error"]["code"] == expected


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("storage_failure", [False, True])
def test_gemini_safety_refusal_preserves_input_and_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: bool, storage_failure: bool
) -> None:
    """A Gemini safety terminal retains only input without introducing availability errors."""

    class RefusalControl(NativeControlPlane):
        def remember(self, argument: str) -> str:
            """Inject a storage failure only for refusal inputs."""
            if storage_failure and json.loads(argument).get("refusal"):
                raise RuntimeError("synthetic store failure")
            return super().remember(argument)

    frames = b"".join(
        _sse_frame(item)
        for item in [
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "synthetic partial answer"}]}}
                ]
            },
            {"candidates": [{"finishReason": "SAFETY"}]},
        ]
    )
    with _provider(frames) as (provider, received):
        factory, _ = registry._HTTP_PROVIDERS["gemini"]
        monkeypatch.setattr(
            registry,
            "_HTTP_PROVIDERS",
            {**registry._HTTP_PROVIDERS, "gemini": (factory, provider)},
        )
        key = _configure_gemini(tmp_path)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = RefusalControl(components, guardrails=_engine(_RefusedHistoryClassifier("block")))
        with (
            _serving(control) as base,
            httpx.Client(
                base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=10
            ) as client,
        ):
            denied = client.post(
                "/v1/responses",
                json={
                    "model": "gemini",
                    "input": "harmful fixture",
                    "store": store,
                    "max_output_tokens": 64,
                },
            )
            assert denied.status_code == 400, denied.text
            assert denied.json()["error"]["code"] == "refusal"
            assert len(received) == 1
            if not store or storage_failure:
                assert "response_id" not in denied.json()["error"]
                return
            followup = client.post(
                "/v1/responses",
                json={
                    "model": "gemini",
                    "input": "please",
                    "max_output_tokens": 64,
                    "previous_response_id": denied.json()["error"]["response_id"],
                },
            )
            assert followup.status_code == 400, followup.text
            assert followup.json()["error"]["code"] == "content_filter"

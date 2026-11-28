"""Denied conversation continuation through real native HTTP, with synthetic providers."""

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
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine


class _HistoryClassifier(_Classifier):
    """Use deterministic user-history scoring to isolate continuation transport behavior."""

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Flag only the synthetic original user request, including on follow-ups."""
        del check
        window = UserConversationWindow(len, 10000, 20000, "fixture")
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
            receipt = denied.json()["error"]["response_id"]
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

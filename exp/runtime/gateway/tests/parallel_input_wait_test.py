"""Optional moderation deadlines preserve real HTTP delivery and financial finality."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import httpx
import pytest

from exp.common.models import BillingSource, GatewayTokenPrices
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("funding", [BillingSource.HOST_MANAGED, BillingSource.CUSTOMER_MANAGED])
def test_stalled_optional_inspection_releases_response_and_preserves_charge(
    tmp_path: Path, surface: str, stream: bool, funding: BillingSource
) -> None:
    """Parallel and forced predispatch paths both expire without waiving generated usage."""
    classifier = _Classifier("allow")
    template = _engine(classifier)
    policy = template.policies_for(_authorization())[0].model_copy(
        update={"fail_open": True, "input_max_wait_ms": 250}
    )
    template.close()
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    with _provider(
        _content_chunk("safe-answer") + _terminal_frames(prompt_tokens=80, completion_tokens=5)
    ) as (base, received):
        _, key = _configured_gateway(
            tmp_path,
            base_url=base,
            billing_source=funding,
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000_000,
                output_nano_usd_per_million_tokens=2_000_000_000,
            ),
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        try:
            with _serving(control) as url, httpx.Client() as client:
                path, body = _body(surface, stream)
                started = time.monotonic()
                response = client.post(
                    url + path, headers={"authorization": f"Bearer {key}"}, json=body, timeout=3
                )
                elapsed = time.monotonic() - started
                assert response.status_code == 200, response.text
                assert "safe-answer" in response.text
                assert elapsed < 0.65, elapsed
                assert not classifier.release.is_set()
                assert len(received) == 1
                deadline = time.monotonic() + 2
                while True:
                    with sqlite3.connect(components.ledger.database_path) as db:
                        requests = db.execute(
                            "select terminal_state from gateway_requests"
                        ).fetchall()
                        attempts = db.execute(
                            "select state,input_tokens,output_tokens,budget_settled_nano_usd,"
                            "estimated_cost_nano_usd,billing_source from gateway_attempts"
                        ).fetchall()
                    if requests == [("completed",)] or time.monotonic() > deadline:
                        break
                    time.sleep(0.01)
                assert requests == [("completed",)]
                assert attempts == [
                    (
                        "completed",
                        80,
                        5,
                        90_000,
                        90_000,
                        funding.value,
                    )
                ]
        finally:
            classifier.release.set()
            engine.close(timeout_seconds=2)

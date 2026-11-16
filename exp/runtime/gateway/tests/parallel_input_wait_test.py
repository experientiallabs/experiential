"""Optional moderation deadlines preserve real HTTP delivery and financial finality."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from exp.common.models import BillingSource, GatewayTokenPrices
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.store_test import _authorization
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.optional_input_contention_test import _held_provider
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
        update={
            "fail_open": True,
            "input_max_wait_ms": 250,
            "organization_id": None,
            "identity_id": None,
        }
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
                assert classifier.started.is_set()
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


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_timely_optional_violation_remains_a_zero_charge_denial(
    tmp_path: Path, surface: str, stream: bool
) -> None:
    """The native optional path preserves a real violation delivered before its deadline."""
    classifier = _Classifier("block")
    classifier.release.set()
    template = _engine(classifier)
    policy = template.policies_for(_authorization())[0].model_copy(
        update={
            "fail_open": True,
            "input_max_wait_ms": 250,
            "organization_id": None,
            "identity_id": None,
        }
    )
    template.close()
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"criminal": classifier})),
        monotonic=time.monotonic,
    )
    with _provider(
        _content_chunk("unapproved-answer")
        + _terminal_frames(prompt_tokens=80, completion_tokens=5)
    ) as (base, _received):
        _, key = _configured_gateway(
            tmp_path, base_url=base, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        try:
            with _serving(control) as url, httpx.Client() as client:
                path, body = _body(surface, stream)
                response = client.post(
                    url + path, headers={"authorization": f"Bearer {key}"}, json=body, timeout=3
                )
                assert classifier.started.is_set()
                assert response.status_code == 400, response.text
                assert "unapproved-answer" not in response.text
                deadline = time.monotonic() + 2
                while True:
                    with sqlite3.connect(components.ledger.database_path) as db:
                        requests = db.execute(
                            "select terminal_state from gateway_requests"
                        ).fetchall()
                    if requests == [("failed",)] or time.monotonic() > deadline:
                        break
                    time.sleep(0.01)
                assert requests == [("failed",)]
                with sqlite3.connect(components.ledger.database_path) as db:
                    assert all(
                        charge == 0
                        for (charge,) in db.execute(
                            "select budget_settled_nano_usd from gateway_attempts"
                        ).fetchall()
                    )
        finally:
            engine.close(timeout_seconds=2)


@pytest.mark.parametrize("flagged", [False, True])
def test_optional_expiry_does_not_delay_or_skip_required_output_inspection(
    tmp_path: Path, flagged: bool
) -> None:
    """A mandatory output check follows native input expiry without rejoining its old task."""
    classifier = _Classifier("block")
    template = _engine(classifier)
    bound = template.policies_for(_authorization())[0]
    optional = bound.model_copy(
        update={"fail_open": True, "organization_id": None, "identity_id": None}
    )
    output = bound.model_copy(
        update={
            "policy_id": "required-output",
            "organization_id": None,
            "identity_id": None,
            "checks": (
                bound.checks[0].model_copy(
                    update={"stage": GuardrailCheckStage.OUTPUT, "adapter_id": "output"}
                ),
            ),
        }
    )
    template.close()
    output_calls: list[GuardrailCompletion] = []

    class RequiredOutput(_Classifier):
        """Decide immediately while the independent input classifier stays stalled."""

        async def inspect_output(
            self, *, completion: GuardrailCompletion, check: GuardrailCheck
        ) -> ClassifierVerdict:
            """Retain proof that the mandatory output check actually ran."""
            output_calls.append(completion)
            return ClassifierVerdict(flagged=flagged)

    engine = GuardrailEngine(
        store=MappingGuardrailStore((optional, output)),
        client=DirectClassifierClient(
            ClassifierRegistry({"criminal": classifier, "output": RequiredOutput("allow")})
        ),
        monotonic=time.monotonic,
    )
    received, provider_release = threading.Event(), threading.Event()
    with _held_provider(received, provider_release) as base:
        _, key = _configured_gateway(
            tmp_path, base_url=base, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        try:
            with (
                _serving(control) as url,
                httpx.Client() as client,
                ThreadPoolExecutor(max_workers=1) as pool,
            ):
                path, body = _body("chat", False)
                response_task = pool.submit(
                    client.post,
                    url + path,
                    headers={"authorization": f"Bearer {key}"},
                    json=body,
                    timeout=2,
                )
                assert received.wait(2) and classifier.started.wait(2)
                # Measure the boundary held by moderation, after mandatory admission.
                started = time.monotonic()
                provider_release.set()
                response = response_task.result(timeout=0.65)
                assert time.monotonic() - started < 0.65
                assert classifier.started.is_set() and not classifier.release.is_set()
                assert len(output_calls) == 1
                assert response.status_code == (400 if flagged else 200), response.text
                assert ("safe-answer" in response.text) is not flagged
                assert classifier.cancelled.wait(0.5)
        finally:
            classifier.release.set()
            engine.close(timeout_seconds=2)

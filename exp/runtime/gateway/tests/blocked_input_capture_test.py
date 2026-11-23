"""Blocked-input retention crosses real native HTTP without granting provider execution."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import exp_gateway_native
import httpx
import pytest

from exp.common.models import BillingSource
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    CaptureRecord,
)
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("execution", ["serial", "parallel", "optional"])
@pytest.mark.parametrize("capture_allowed", [False, True])
@pytest.mark.parametrize("hosted", [False, True])
def test_blocked_input_capture_preserves_authority_and_withholds_output(
    tmp_path: Path, surface: str, execution: str, capture_allowed: bool, hosted: bool
) -> None:
    """Host consent and settlement independently fence blocked input across the native gates."""
    classifier = _Classifier("block")
    classifier.release.set()
    engine = GuardrailEngine(
        store=MappingGuardrailStore(
            (
                GuardrailPolicy.model_validate(
                    {
                        "policy_id": "fixture-input",
                        "protected": True,
                        "fail_open": execution == "optional",
                        "input_execution": "before_dispatch"
                        if execution == "serial"
                        else "parallel",
                        "input_max_wait_ms": 1000,
                        "checks": [
                            {
                                "check_id": "fixture-check",
                                "adapter_id": "fixture",
                                "capability": "content_safety",
                                "stage": "input",
                                "action": "block",
                                "timeout_ms": 3000,
                            }
                        ],
                    }
                ),
            )
        ),
        client=DirectClassifierClient(ClassifierRegistry({"fixture": classifier})),
        monotonic=time.monotonic,
    )
    records: list[str] = []
    collector = exp_gateway_native.CaptureCollector(
        CaptureConfiguration(settlement_required=hosted).model_dump_json(), records.append
    )
    with _provider(_content_chunk("withheld-provider-output") + _terminal_frames()) as (
        url,
        upstream,
    ):
        _, key = _configured_gateway(
            tmp_path, base_url=url, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(
            components,
            guardrails=engine,
            capture=CaptureController(
                collector,
                application_for=lambda _auth: "capture-fixture" if capture_allowed else None,
            ),
        )
        try:
            with _serving(control) as endpoint:
                path, body = _body(surface, False)
                response = httpx.post(
                    endpoint + path,
                    json=body,
                    headers={"authorization": f"Bearer {key}"},
                    timeout=5,
                )
                assert response.status_code == 400, response.text
                assert "withheld-provider-output" not in response.text
                if execution == "serial":
                    assert upstream == []
            with sqlite3.connect(components.ledger.database_path) as db:
                [(request_id, state)] = db.execute(
                    "select request_id, terminal_state from gateway_requests"
                ).fetchall()
                assert state == "failed"
            if hosted:
                assert records == [], "input rejection cannot authorize hosted retention"
                collector.settle(request_id, capture_allowed, False)
        finally:
            engine.close(timeout_seconds=2)
            assert collector.close(2)
            components.write_ledger.close()
    assert len(records) == int(capture_allowed)
    if records:
        record = CaptureRecord.model_validate_json(records[0])
        assert record.request.request_id == request_id
        assert "synthetic prompt" in record.request.model_dump_json()
        assert record.response is None
        assert record.provider_reasoning is None and record.provider_tool_calls_json is None
        assert record.metrics is None and record.provider_logprobs is None
        assert "withheld-provider-output" not in record.model_dump_json()

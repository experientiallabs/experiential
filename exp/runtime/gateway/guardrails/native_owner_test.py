"""Admission hands read-only inspection to Rust without leaving a Python future."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore


@pytest.mark.parametrize(("text", "action"), [("hello", "allow"), ("Straße", "error")])
def test_parallel_input_retains_one_native_decision(text: str, action: str) -> None:
    """Lifecycle diagnostics read the same decision without calling the Python classifier."""
    adapter = KeywordClassifier(("strasse",))
    engine = GuardrailEngine(
        store=MappingGuardrailStore(),
        client=DirectClassifierClient(ClassifierRegistry({"keyword": adapter})),
        monotonic=time.monotonic,
    )
    policy = GuardrailPolicy.model_validate(
        {
            "policy_id": "policy",
            "organization_id": "org",
            "identity_id": "identity",
            "protected": True,
            "input_execution": "parallel",
            "checks": [
                {
                    "check_id": "check",
                    "adapter_id": "keyword",
                    "stage": "input",
                    "capability": "pii",
                    "action": "block",
                    "timeout_ms": 1000,
                }
            ],
        }
    )
    session = GuardrailSession(engine, (policy,), time.monotonic() + 5, native_execution=True)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=text),),
    )
    with patch.object(
        adapter, "inspect_input", side_effect=AssertionError("Python after admission")
    ):
        session.prepare_dispatch((request,), overlap=True)
        assert session.native_input is not None
        assert session.input_pending
        result = json.loads(session.native_input.wait_json())
        assert result["action"] == action
        assert session.input_decision() == result
        failure = session.settlement_failure()
        if action == "error":
            assert failure is not None
            assert failure.safe_details["input_guardrail_denied"] is True
        else:
            assert failure is None
        assert adapter.input_calls == 0
    session.cancel()


@pytest.mark.parametrize("protected", [False, True])
def test_missing_native_adapter_uses_the_policy_uncertainty_response(protected: bool) -> None:
    """Missing classifiers produce a recorded native verdict rather than an admission exception."""
    engine = GuardrailEngine(
        store=MappingGuardrailStore(),
        client=DirectClassifierClient(ClassifierRegistry({})),
        monotonic=time.monotonic,
    )
    policy = GuardrailPolicy.model_validate(
        {
            "policy_id": "missing",
            "organization_id": "org",
            "identity_id": "identity",
            "protected": protected,
            "input_execution": "parallel",
            "checks": [
                {
                    "check_id": "missing",
                    "adapter_id": "absent",
                    "stage": "input",
                    "capability": "pii",
                    "action": "block",
                    "timeout_ms": 1000,
                }
            ],
        }
    )
    session = GuardrailSession(engine, (policy,), time.monotonic() + 5, native_execution=True)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="fixture"),),
    )
    try:
        session.prepare_dispatch((request,), overlap=True)
        assert session.native_input is not None
        decision = json.loads(session.native_input.wait_json())
        assert decision["action"] == ("error" if protected else "allow")
        assert decision["moderation"]["observations"][0]["outcome"] == "unavailable"
    finally:
        session.cancel()

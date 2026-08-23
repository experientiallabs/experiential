"""Native control-plane guardrail admission and output-callback tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _admit, _chat_body, _control_plane
from exp.runtime.gateway.routing import GatewayRoute


def _engine(
    classifier: ScriptedClassifier,
    *,
    output: bool = False,
    action: GuardrailAction = GuardrailAction.BLOCK,
) -> GuardrailEngine:
    """Compose one engine for the configured local identity."""
    checks = [
        GuardrailCheck(
            check_id="input-safety",
            capability=GuardrailCapabilityKind.CONTENT_SAFETY,
            stage=GuardrailCheckStage.INPUT,
            action=action,
            timeout_ms=100,
            adapter_id="scripted",
        )
    ]
    if output:
        checks.append(
            GuardrailCheck(
                check_id="output-safety",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.OUTPUT,
                action=action,
                timeout_ms=100,
                adapter_id="scripted",
            )
        )
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="local",
        identity_id="default",
        protected=True,
        checks=tuple(checks),
    )
    return GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"scripted": classifier})),
        monotonic=lambda: 0.0,
    )


def test_unguarded_admit_does_not_request_output_enforcement(tmp_path: Path) -> None:
    """Default-off admissions never set the native output-callback flag."""
    control, raw_key = _control_plane(tmp_path)
    admission = _admit(control, raw_key, _chat_body())

    assert admission["output_guardrail"] is False
    assert control._guardrails is None  # noqa: SLF001 - test inspects the injected engine


def test_input_block_fails_admit_before_ledger_accept(tmp_path: Path) -> None:
    """A blocked input chain never starts an attempt on the native path."""
    classifier = ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True))
    control, issued = _native_with_engine(tmp_path, _engine(classifier))

    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, issued, _chat_body())

    payload = json.loads(raised.value.public_error_json)
    assert payload["code"] == "content_filter"
    assert classifier.input_calls == 1
    assert control._inflight == {}  # noqa: SLF001


def test_output_policy_sets_the_native_callback_flag(tmp_path: Path) -> None:
    """Assigned output checks ask Rust to buffer and call enforce_output once."""
    classifier = ScriptedClassifier()
    control, raw_key = _native_with_engine(tmp_path, _engine(classifier, output=True))
    admission = _admit(control, raw_key, _chat_body())

    assert admission["output_guardrail"] is True
    decision = json.loads(
        control.enforce_output(
            json.dumps(
                {
                    "request_id": admission["request_id"],
                    "text": "hello",
                    "tool_calls": [],
                }
            )
        )
    )
    assert decision["action"] == "allow"
    assert classifier.output_calls == 1


def test_native_input_runs_before_route_resolution(tmp_path: Path) -> None:
    """Admit inspects the expanded request before resolving a route."""
    order: list[str] = []

    class _OrderClassifier(ScriptedClassifier):
        """Record input inspection before any later admit step."""

        def inspect_input(
            self,
            *,
            request: GatewayRequest,
            check: GuardrailCheck,
        ) -> ClassifierVerdict:
            """Mark input, then allow."""
            del request, check
            order.append("input")
            self.input_calls += 1
            return ClassifierVerdict(flagged=False)

    class _OrderedPlane(NativeControlPlane):
        """Record route resolution relative to input enforcement."""

        def _resolve_route(
            self,
            authorization: AuthorizationSnapshot,
            request: GatewayRequest,
        ) -> GatewayRoute:
            """Mark resolve, then delegate."""
            order.append("resolve")
            return super()._resolve_route(authorization, request)

    engine = _engine(_OrderClassifier())
    control, raw_key = _native_with_engine(tmp_path, engine, plane_cls=_OrderedPlane)
    admission = _admit(control, raw_key, _chat_body())

    assert order == ["input", "resolve"]
    assert admission["output_guardrail"] is False


def test_native_input_block_never_resolves_a_route(tmp_path: Path) -> None:
    """A blocked input chain fails admit before route resolution."""
    order: list[str] = []

    class _BlockClassifier(ScriptedClassifier):
        """Record a blocking input inspection."""

        def inspect_input(
            self,
            *,
            request: GatewayRequest,
            check: GuardrailCheck,
        ) -> ClassifierVerdict:
            """Mark input, then flag."""
            del request, check
            order.append("input")
            self.input_calls += 1
            return ClassifierVerdict(flagged=True)

    class _OrderedPlane(NativeControlPlane):
        """Fail the test if routing runs after a block."""

        def _resolve_route(
            self,
            authorization: AuthorizationSnapshot,
            request: GatewayRequest,
        ) -> GatewayRoute:
            """Mark resolve, then delegate."""
            order.append("resolve")
            return super()._resolve_route(authorization, request)

    engine = _engine(_BlockClassifier())
    control, issued = _native_with_engine(tmp_path, engine, plane_cls=_OrderedPlane)

    with pytest.raises(NativeBridgeError):
        _admit(control, issued, _chat_body())

    assert order == ["input"]


def _native_with_engine(
    root: Path,
    engine: GuardrailEngine,
    *,
    plane_cls: type[NativeControlPlane] = NativeControlPlane,
) -> tuple[NativeControlPlane, str]:
    """Load one configured alias and bind an injected engine."""
    _manager, raw_key = _configured_gateway(root)
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    return plane_cls(components, guardrails=engine), raw_key

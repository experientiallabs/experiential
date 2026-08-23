"""Recursion-prevention tests for the internal classifier client."""

from __future__ import annotations

import asyncio

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import (
    DirectClassifierClient,
    GuardrailRecursionError,
    assert_not_internal_classification,
    classification_scope,
    internal_classification_active,
)
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
)


def _check() -> GuardrailCheck:
    """Return one input safety check."""
    return GuardrailCheck(
        check_id="input-safety",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=100,
        adapter_id="scripted",
    )


def test_classification_scope_blocks_public_route_reentry() -> None:
    """The public gateway must not run while a classifier call is active."""
    assert internal_classification_active() is False
    with classification_scope():
        assert internal_classification_active() is True
        with pytest.raises(GuardrailRecursionError, match="public gateway"):
            assert_not_internal_classification()
    assert internal_classification_active() is False
    assert_not_internal_classification()


def test_direct_client_never_uses_a_public_http_route() -> None:
    """The injected client calls the adapter in-process under the recursion flag."""

    class _NestedClassifier(ScriptedClassifier):
        """Assert the recursion flag is set for the duration of inspection."""

        async def inspect_input(
            self,
            *,
            request: GatewayRequest,
            check: GuardrailCheck,
        ) -> ClassifierVerdict:
            """Require the internal classification flag during the adapter call."""
            assert internal_classification_active() is True
            return await super().inspect_input(request=request, check=check)

    async def scenario() -> None:
        """Inspect one request through the in-process client."""
        registry = ClassifierRegistry({"scripted": _NestedClassifier()})
        client = DirectClassifierClient(registry)
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hello"),),
        )

        verdict = await client.inspect_input(request=request, check=_check())

        assert verdict.flagged is False
        assert internal_classification_active() is False

    asyncio.run(scenario())


def test_client_inspects_output_through_the_same_internal_seam() -> None:
    """Output inspection uses the same in-process client as input inspection."""
    registry = ClassifierRegistry({"scripted": ScriptedClassifier()})
    client = DirectClassifierClient(registry)
    check = _check().model_copy(update={"stage": GuardrailCheckStage.OUTPUT})

    verdict = asyncio.run(
        client.inspect_output(
            completion=GuardrailCompletion(text="ok"),
            check=check,
        )
    )

    assert verdict.flagged is False

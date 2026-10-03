"""Content-free runtime inspection decisions and input contract bounds."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.runtime import (
    RuntimeOutput,
    output_decision,
    validate_runtime_guardrail,
)
from exp.runtime.gateway.tests.runtime_guardrails_test import _Guard


class _Session:
    """Controllable host session that can expose hostile exception messages."""

    def __init__(self, error: Exception | None) -> None:
        """Bind the outcome for this inspection."""
        self.error = error

    def inspect_output(self, output: RuntimeOutput, *, deadline_monotonic: float) -> None:
        """Accept the pending segment or raise the selected error."""
        if self.error is not None:
            raise self.error


def test_infrastructure_errors_do_not_become_violations_or_echo_content() -> None:
    """A model exception is an unavailable inspection with no raw exception text."""
    output = RuntimeOutput(request_id="req", fragments=(), final=True)
    payload = output_decision(
        _Session(ValueError("private prompt")), output, deadline_monotonic=time.monotonic() + 10
    )
    assert "private prompt" not in payload
    assert json.loads(payload)["failure"]["failure_class"] == "unavailable"


def test_explicit_content_refusal_retains_its_failure_class() -> None:
    """Only an explicit sanitized violation uses the guardrail class."""
    error = GuardrailRejected(
        GatewayFailure(failure_class=GatewayFailureClass.GUARDRAIL, safe_message="Policy violation")
    )
    output = RuntimeOutput(request_id="req", fragments=(), final=True)
    result = json.loads(
        output_decision(_Session(error), output, deadline_monotonic=time.monotonic() + 10)
    )
    assert result["failure"]["failure_class"] == "guardrail"


def test_unknown_fragment_kind_is_not_silently_uninspected() -> None:
    """A new content channel requires an explicit contract update."""
    with pytest.raises(ValidationError):
        RuntimeOutput.model_validate(
            {
                "request_id": "req",
                "fragments": [{"kind": "image", "channel": "0", "text": "x"}],
                "final": True,
            }
        )


def test_expired_session_cannot_authorize_a_late_segment() -> None:
    """A detector returning allow after expiry must not release content."""
    output = RuntimeOutput(request_id="req", fragments=(), final=True)
    result = json.loads(output_decision(_Session(None), output, deadline_monotonic=0))
    assert result["failure"]["failure_class"] == "unavailable"


@pytest.mark.parametrize("marker", [None, True, "1", 0, 2])
def test_older_or_unknown_native_contract_cannot_bypass_output(marker: object) -> None:
    """Fail at startup instead of silently using an extension that ignores the policy."""
    with (
        patch(
            "exp.runtime.gateway.guardrails.runtime.importlib.import_module",
            return_value=SimpleNamespace(RUNTIME_INSPECTION_CONTRACT_VERSION=marker),
        ),
        pytest.raises(ValueError, match="RUNTIME_INSPECTION_CONTRACT_VERSION=1"),
    ):
        validate_runtime_guardrail(_Guard())

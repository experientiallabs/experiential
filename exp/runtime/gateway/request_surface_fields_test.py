"""Tests for the Responses-only fields of the canonical gateway request."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_summary", "auto"),
        ("response_store", False),
        ("include_output_text_logprobs", True),
        ("include_web_search_sources", True),
        ("include_encrypted_reasoning", True),
        ("reasoning_context", "current_turn"),
    ],
)
def test_responses_only_fields_are_refused_on_other_surfaces(field: str, value: object) -> None:
    """Each Responses-only field names itself when set on another surface."""
    with pytest.raises(ValidationError, match=f"{field} is valid only for Responses requests"):
        GatewayRequest.model_validate(
            {
                "surface": GatewayApiSurface.CHAT_COMPLETIONS,
                "messages": (GatewayMessage(role="user", content="hi"),),
                field: value,
            }
        )
    GatewayRequest.model_validate(
        {
            "surface": GatewayApiSurface.RESPONSES,
            "messages": (GatewayMessage(role="user", content="hi"),),
            field: value,
        }
    )


def test_unset_responses_only_fields_pass_on_other_surfaces() -> None:
    """Defaults (absent or false) never trip the check."""
    GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
    )

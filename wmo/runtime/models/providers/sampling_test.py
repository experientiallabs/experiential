"""Tests for capability-driven sampling-parameter serialization."""

from __future__ import annotations

from wmo.common.models import ModelCapabilities, ModelMessage, ModelRequest
from wmo.runtime.models.providers.anthropic import anthropic_messages_request
from wmo.runtime.models.providers.bedrock_converse import converse_request
from wmo.runtime.models.providers.gemini import gemini_generate_request
from wmo.runtime.models.providers.openai import openai_responses_request
from wmo.runtime.models.providers.openai_compatible import openai_compatible_request
from wmo.runtime.models.providers.sampling import include_temperature

_LUNA_CAPABILITIES = ModelCapabilities(supports_temperature=False)
_REQUEST = ModelRequest(
    messages=(ModelMessage(role="user", content="Score the rollout."),),
    temperature=0.0,
    maximum_output_tokens=128,
)


def test_include_temperature_follows_explicit_capabilities_not_model_names() -> None:
    """Unknown or supported models keep temperature. An explicit false omits it."""
    assert include_temperature(_REQUEST, None) is True
    assert include_temperature(_REQUEST, ModelCapabilities()) is True
    assert include_temperature(_REQUEST, ModelCapabilities(supports_temperature=True)) is True
    assert include_temperature(_REQUEST, _LUNA_CAPABILITIES) is False
    assert (
        include_temperature(
            ModelRequest(messages=(ModelMessage(role="user", content="Score the rollout."),)),
            ModelCapabilities(supports_temperature=True),
        )
        is False
    )


def test_openai_responses_omits_unsupported_temperature_without_changing_other_fields() -> None:
    """The official gpt-5.6-luna Responses contract omits temperature and keeps max tokens."""
    payload = openai_responses_request("gpt-5.6-luna", _REQUEST, _LUNA_CAPABILITIES)
    supported = openai_responses_request(
        "gpt-5.4", _REQUEST, ModelCapabilities(supports_temperature=True)
    )

    assert "temperature" not in payload
    assert payload["max_output_tokens"] == 128
    assert payload["model"] == "gpt-5.6-luna"
    assert supported["temperature"] == 0.0


def test_compatible_anthropic_gemini_and_bedrock_omit_unsupported_temperature() -> None:
    """Every native request builder consults the same sampling capability."""
    compatible = openai_compatible_request("gpt-5.6-luna", _REQUEST, _LUNA_CAPABILITIES)
    anthropic = anthropic_messages_request("claude-sonnet-5", _REQUEST, _LUNA_CAPABILITIES)
    gemini = gemini_generate_request("gemini-3.5-flash", _REQUEST, _LUNA_CAPABILITIES)
    bedrock = converse_request("anthropic.claude-sonnet-5", _REQUEST, _LUNA_CAPABILITIES)

    assert "temperature" not in compatible
    assert "temperature" not in anthropic
    assert "temperature" not in gemini["generationConfig"]
    assert "temperature" not in bedrock.get("inferenceConfig", {})
    assert compatible["max_tokens"] == 128
    assert anthropic["max_tokens"] == 128
    assert gemini["generationConfig"]["maxOutputTokens"] == 128
    assert bedrock["inferenceConfig"]["maxTokens"] == 128

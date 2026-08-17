"""Tests for catalog-declared sampling-field serialization."""

from __future__ import annotations

from wmo.common.models import ModelCapabilities, ModelMessage, ModelRequest, SamplingSupport
from wmo.runtime.models.providers.anthropic import anthropic_messages_request
from wmo.runtime.models.providers.bedrock_converse import converse_request
from wmo.runtime.models.providers.gemini import gemini_generate_request
from wmo.runtime.models.providers.openai import openai_responses_request
from wmo.runtime.models.providers.openai_compatible import openai_compatible_request
from wmo.runtime.models.providers.sampling import include_sampling_field
from wmo.runtime.models.providers.tinker_sampling import tinker_sampling_params

_UNSUPPORTED = ModelCapabilities(sampling=SamplingSupport(temperature=False))
_SUPPORTED = ModelCapabilities(sampling=SamplingSupport(temperature=True))
_REQUEST = ModelRequest(
    messages=(ModelMessage(role="user", content="Score the rollout."),),
    temperature=0.0,
    maximum_output_tokens=128,
)


def test_include_sampling_field_follows_catalog_declaration_not_model_names() -> None:
    """Unknown or supported models keep a named field. An explicit false omits it."""
    assert include_sampling_field(_REQUEST, None, "temperature") is True
    assert include_sampling_field(_REQUEST, ModelCapabilities(), "temperature") is True
    assert include_sampling_field(_REQUEST, _SUPPORTED, "temperature") is True
    assert include_sampling_field(_REQUEST, _UNSUPPORTED, "temperature") is False
    assert (
        include_sampling_field(
            ModelRequest(messages=(ModelMessage(role="user", content="Score the rollout."),)),
            _SUPPORTED,
            "temperature",
        )
        is False
    )


def test_openai_responses_omits_unsupported_temperature_without_changing_other_fields() -> None:
    """The official gpt-5.6-luna Responses contract omits temperature and keeps max tokens."""
    payload = openai_responses_request("gpt-5.6-luna", _REQUEST, _UNSUPPORTED)
    supported = openai_responses_request("gpt-5.4", _REQUEST, _SUPPORTED)

    assert "temperature" not in payload
    assert payload["max_output_tokens"] == 128
    assert payload["model"] == "gpt-5.6-luna"
    assert supported["temperature"] == 0.0


def test_every_native_request_builder_consults_the_same_sampling_support() -> None:
    """Compatible, Anthropic, Gemini, and Bedrock omit a field the catalog forbids."""
    compatible = openai_compatible_request("gpt-5.6-luna", _REQUEST, _UNSUPPORTED)
    anthropic = anthropic_messages_request("claude-sonnet-5", _REQUEST, _UNSUPPORTED)
    gemini = gemini_generate_request("gemini-3.5-flash", _REQUEST, _UNSUPPORTED)
    bedrock = converse_request("anthropic.claude-sonnet-5", _REQUEST, _UNSUPPORTED)

    gemini_config = gemini.get("generationConfig")
    bedrock_config = bedrock.get("inferenceConfig")
    assert isinstance(gemini_config, dict)
    assert isinstance(bedrock_config, dict)
    assert "temperature" not in compatible
    assert "temperature" not in anthropic
    assert "temperature" not in gemini_config
    assert "temperature" not in bedrock_config
    assert compatible["max_tokens"] == 128
    assert anthropic["max_tokens"] == 128
    assert gemini_config["maxOutputTokens"] == 128
    assert bedrock_config["maxTokens"] == 128


def test_tinker_sampling_params_omit_unsupported_temperature_and_keep_the_default() -> None:
    """Tinker omits a forbidden field and still defaults temperature when the request omits it."""
    omitted = ModelRequest(messages=(ModelMessage(role="user", content="Score the rollout."),))

    assert tinker_sampling_params(_REQUEST, _UNSUPPORTED) == {"max_tokens": 128}
    assert tinker_sampling_params(_REQUEST, _SUPPORTED) == {
        "max_tokens": 128,
        "temperature": 0.0,
    }
    assert tinker_sampling_params(omitted, None) == {"max_tokens": None, "temperature": 1.0}
    assert tinker_sampling_params(omitted, _UNSUPPORTED) == {"max_tokens": None}

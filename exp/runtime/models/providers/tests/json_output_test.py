"""SDK JSON-object output reaches native providers without changing ordinary replay keys."""

from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
)
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.models.providers.anthropic import anthropic_messages_request
from exp.runtime.models.providers.bedrock_requests import converse_request
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.openai import openai_responses_request
from exp.runtime.models.providers.openai_compatible import openai_compatible_request
from exp.runtime.models.providers.tinker_sampling import TinkerSample, TinkerSamplingClient


def _request() -> ModelRequest:
    """Request one JSON object without imposing a provider-specific JSON schema."""
    return ModelRequest(
        messages=(ModelMessage(role="user", content="Return a JSON object."),),
        json_object_output=True,
    )


def test_json_mode_uses_native_chat_responses_and_gemini_controls() -> None:
    """Providers with native JSON mode receive the explicit constraint on their own wire."""
    request = _request()
    assert openai_compatible_request("deepseek", request)["response_format"] == {
        "type": "json_object"
    }
    assert openai_responses_request("openai", request)["text"] == {
        "format": {"type": "json_object"}
    }
    generation = gemini_generate_request("gemini", request)["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in generation


def test_instruction_only_json_providers_preserve_the_explicit_output_instruction() -> None:
    """Anthropic and Converse use the same documented instruction as their gateway routes."""
    request = _request()
    assert JSON_OBJECT_SYSTEM_INSTRUCTION in str(anthropic_messages_request("claude", request))
    assert JSON_OBJECT_SYSTEM_INSTRUCTION in str(converse_request("bedrock", request))


def test_ordinary_request_serialization_and_wire_omit_json_mode() -> None:
    """Adding an opt-in control cannot change a previously saved ordinary request identity."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="Hello"),))
    saved = request.model_dump(mode="json")
    assert "json_object_output" not in saved
    assert ModelRequest.model_validate(saved) == request
    assert "response_format" not in openai_compatible_request("chat", request)
    assert "text" not in openai_responses_request("responses", request)
    assert _request().model_dump(mode="json")["json_object_output"] is True


def test_tinker_adds_json_guidance_before_sampling() -> None:
    """Sampling receives explicit JSON guidance without requiring optional SDK imports."""

    class Sampler:
        """Observe the rendered request at the existing sampling seam."""

        def sample(self, request: ModelRequest) -> TinkerSample:
            """Validate the visible JSON instruction and return a complete sample."""
            assert request.messages[0].content == JSON_OBJECT_SYSTEM_INSTRUCTION
            return TinkerSample(output=AssistantAction(content='{"ok":true}'))

    model = ModelSnapshot(
        provider="tinker",
        model_id="fixture",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
    response = TinkerSamplingClient(model=model, sampler=Sampler()).complete(_request())
    assert response.output.content == '{"ok":true}'

"""Local text scanners distinguish supported prompt text from opaque coverage gaps."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from exp.common.models.content import ImageContentPart, TextContentPart
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
)
from exp.runtime.gateway.guardrails.bounded import start_on_native_loop
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, KeywordClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailOutcome,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.regex import RegexAdapterDocument, RegexClassifier
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.text_coverage import text_input_context
from exp.runtime.gateway.reasoning_blocks import (
    ExposedReasoningContentBlock,
    OpaqueReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.tool_contracts import GatewayProviderNativeTool
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.web_search.contracts import GatewayWebSearch

UNSUPPORTED_TEXT_INPUTS = (
    "recovered_reasoning",
    "sealed_reasoning",
    "capture_only_reasoning",
    "native_response",
    "native_anthropic",
    "ordered_anthropic_blocks",
    "image",
    "context_management",
    "provider_output_config",
    "provider_thinking_config",
    "safeguards",
    "provider_native_tools",
    "provider_server_tools",
    "tool_caller",
    "tool_result_caller",
)
SUPPORTED_TEXT_CONTEXTS = (
    "tool_description",
    "tool_schema",
    "tool_example",
    "structured_schema",
    "web_search_prompt",
    "tool_search_name",
    "tool_call_name",
    "tool_call_namespace",
    "tool_call_id",
    "tool_result_name",
    "tool_result_namespace",
    "tool_result_id",
)


def tool_error_request(*, failed: bool, content: str = "benign") -> GatewayRequest:
    """Retain the typed tool error flag whose provider projection adds a text prefix."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(
                role="tool", tool_call_id="call-one", content=content, tool_is_error=failed
            ),
        ),
    )


def unsupported_text_request(field: str) -> GatewayRequest:
    """Build a validated input whose visible text omits some classifier-visible content."""
    message = GatewayMessage(role="assistant", content="visible")
    data: dict[str, object] = {
        "surface": GatewayApiSurface.MESSAGES,
        "messages": (message,),
    }
    match field:
        case "recovered_reasoning":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    content="visible",
                    provider_reasoning=(
                        OpaqueReasoningContentBlock(
                            route_sha256="a" * 64, content="hidden alice@example.com"
                        ),
                    ),
                ),
            )
        case "sealed_reasoning":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    content="visible",
                    provider_reasoning=(
                        SealedReasoningContentBlock(carrier="sealed", deployment_hint="fixture"),
                    ),
                ),
            )
        case "capture_only_reasoning":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    content="visible",
                    capture_only_reasoning=(
                        ExposedReasoningContentBlock(content="hidden alice@example.com"),
                    ),
                ),
            )
        case "native_response":
            data["surface"] = GatewayApiSurface.RESPONSES
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    provider_native_item={
                        "type": "custom_tool_call_output",
                        "call_id": "fixture-call",
                        "output": "hidden alice@example.com",
                    },
                ),
            )
        case "native_anthropic":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    provider_anthropic_block={
                        "type": "text",
                        "text": "hidden alice@example.com",
                        "citations": [],
                    },
                ),
            )
        case "ordered_anthropic_blocks":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    content="visible",
                    provider_anthropic_blocks=(
                        {"type": "thinking", "thinking": "hidden alice@example.com"},
                    ),
                ),
            )
        case "image":
            data["messages"] = (
                GatewayMessage(
                    role="user",
                    content="visible",
                    content_parts=(
                        TextContentPart(text="visible"),
                        ImageContentPart(media_type="image/png", data="aGk="),
                    ),
                ),
            )
        case "context_management" | "provider_output_config" | "provider_thinking_config":
            data[field] = {"private": "hidden alice@example.com"}
        case "safeguards":
            data[field] = ({"type": "identity", "rules": ["hidden alice@example.com"]},)
        case "provider_native_tools":
            data["surface"] = GatewayApiSurface.RESPONSES
            data[field] = (
                GatewayProviderNativeTool(
                    index=0, tool={"type": "custom", "description": "hidden alice@example.com"}
                ),
            )
        case "provider_server_tools":
            data[field] = ({"type": "provider_owned", "description": "hidden alice@example.com"},)
        case "tool_caller":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            call_id="call-one",
                            name="lookup",
                            arguments={},
                            provider_caller={"type": "program", "private": "alice@example.com"},
                        ),
                    ),
                ),
            )
        case "tool_result_caller":
            data["messages"] = (
                GatewayMessage(
                    role="tool",
                    content="visible",
                    tool_call_id="call-one",
                    provider_tool_caller={"type": "program", "private": "alice@example.com"},
                ),
            )
        case _:
            raise AssertionError(f"unknown fixture {field}")
    return GatewayRequest.model_validate(data)


def text_context_request(field: str, text: str) -> GatewayRequest:
    """Place a textual sentinel only in one supported, non-rewritable prompt field."""
    data: dict[str, object] = {
        "surface": GatewayApiSurface.MESSAGES,
        "messages": (GatewayMessage(role="user", content="visible"),),
    }
    match field:
        case "tool_description":
            data["tools"] = (GatewayToolDefinition(name="lookup", parameters={}, description=text),)
        case "tool_schema":
            data["tools"] = (
                GatewayToolDefinition(name="lookup", parameters={"description": text}),
            )
        case "tool_example":
            data["tools"] = (
                GatewayToolDefinition(
                    name="lookup", parameters={}, input_examples=({"value": text},)
                ),
            )
        case "structured_schema":
            data["structured_text"] = StructuredTextFormat(
                name="answer", json_schema={"description": text}
            )
        case "web_search_prompt":
            data["web_search"] = GatewayWebSearch(declared_as="plugin", search_prompt=text)
        case "tool_search_name":
            data["tool_search"] = GatewayToolSearch(
                declared_as="messages_server_tool",
                tool_type="tool_search_tool_regex",
                tool_name=text,
            )
        case "tool_call_name" | "tool_call_namespace" | "tool_call_id":
            data["messages"] = (
                GatewayMessage(
                    role="assistant",
                    tool_calls=(
                        ToolCall(
                            call_id=text if field == "tool_call_id" else "call-one",
                            name=text if field == "tool_call_name" else "lookup",
                            provider_namespace=text if field == "tool_call_namespace" else None,
                            arguments={},
                        ),
                    ),
                ),
            )
        case "tool_result_name" | "tool_result_namespace" | "tool_result_id":
            data["messages"] = (
                GatewayMessage(
                    role="tool",
                    content="visible",
                    tool_call_id=text if field == "tool_result_id" else "call-one",
                    provider_tool_name=text if field == "tool_result_name" else "lookup",
                    provider_tool_namespace=text if field == "tool_result_namespace" else None,
                ),
            )
        case _:
            raise AssertionError(f"unknown fixture {field}")
    return GatewayRequest.model_validate(data)


@pytest.mark.parametrize("field", UNSUPPORTED_TEXT_INPUTS)
def test_opaque_input_cannot_produce_a_partial_text_projection(field: str) -> None:
    """Coverage fails before any adapter can silently approve the visible subset."""
    with pytest.raises(ClassifierCoverageError):
        text_input_context(unsupported_text_request(field))


@pytest.mark.parametrize("field", SUPPORTED_TEXT_CONTEXTS)
def test_known_textual_context_includes_private_function_examples(field: str) -> None:
    """Every supported non-message content source reaches the shared text extraction seam."""
    assert "alice@example.com" in text_input_context(
        text_context_request(field, "alice@example.com")
    )


def test_exactly_flattened_text_blocks_and_inert_metadata_remain_supported() -> None:
    """Cache hints and replay identifiers do not invent extra prompt content to inspect."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="assistant",
                content="visible",
                provider_item_id="message-one",
                provider_output_index=0,
                provider_status="completed",
                provider_text_blocks=(
                    {"type": "text", "text": "visible", "cache_control": {"type": "ephemeral"}},
                ),
            ),
        ),
        provider_cache_control={"type": "ephemeral"},
        diagnostics={"previous_message_id": "message-zero"},
        safeguards=(),
    )
    assert text_input_context(request) == ""


@pytest.mark.parametrize("adapter", ["keyword", "regex"])
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("recovered_reasoning", GuardrailOutcome.UNSUPPORTED),
        ("native_response", GuardrailOutcome.UNSUPPORTED),
        ("safeguards", GuardrailOutcome.UNSUPPORTED),
        ("tool_error", GuardrailOutcome.FLAGGED),
    ],
)
def test_actual_observation_records_coverage_and_tool_errors_truthfully(
    adapter: str, field: str, expected: GuardrailOutcome
) -> None:
    """Asynchronous inspection sees supported wire text and reports opaque coverage gaps."""
    # Coverage assertions require accepted work; observation_test covers cold admission.
    start_on_native_loop(asyncio.sleep(0)).result(timeout=2)
    recorded = threading.Event()
    observed: list[GuardrailOutcome] = []
    enforced: list[GuardrailAction] = []

    class RecordingEngine(GuardrailEngine):
        """Capture the real separated seams after asynchronous inspection completes."""

        def _record_observation(
            self,
            policy: GuardrailPolicy,
            check: GuardrailCheck | None,
            outcome: GuardrailOutcome,
            latency_seconds: float,
        ) -> None:
            observed.append(outcome)
            recorded.set()

        def _record(
            self,
            policy: GuardrailPolicy,
            check: GuardrailCheck | None,
            action: GuardrailAction,
            latency_seconds: float,
        ) -> None:
            enforced.append(action)

    policy = GuardrailPolicy(
        policy_id="coverage-observer",
        mode="observe",
        protected=True,
        checks=(
            GuardrailCheck(
                check_id="input",
                adapter_id="local",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=500,
            ),
        ),
    )
    needle = "[tool error] benign" if field == "tool_error" else "alice@example.com"
    pattern = r"^\[tool error\] benign$" if field == "tool_error" else "alice@example.com"
    classifier = (
        KeywordClassifier((needle,))
        if adapter == "keyword"
        else RegexClassifier(RegexAdapterDocument(adapter_id="local", patterns=(pattern,)))
    )
    engine = RecordingEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"local": classifier})),
        monotonic=time.monotonic,
    )
    try:
        engine.observe_input(
            policy=policy,
            request=(
                tool_error_request(failed=True)
                if field == "tool_error"
                else unsupported_text_request(field)
            ),
            deadline_monotonic=time.monotonic() + 2,
        )
        assert recorded.wait(2)
        assert observed == [expected]
        assert enforced == []
    finally:
        engine.close(timeout_seconds=2)

"""Round-trip and rejection tests for the Anthropic Messages decoder."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayNamedToolChoice,
    RedactedThinkingBlock,
    ThinkingBlock,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def _body(**overrides: JsonValue) -> JsonObject:
    """Return one minimal valid Messages body with overrides applied."""
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def test_decode_full_request_is_lossless() -> None:
    """Every supported field lands on the canonical request."""
    decoded = decode_messages(
        _body(
            system=[{"type": "text", "text": "be terse"}, {"type": "text", "text": "and kind"}],
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP", "STOP", "END"],
            stream=True,
            tools=[
                {
                    "name": "search",
                    "description": "look things up",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice={"type": "tool", "name": "search", "disable_parallel_tool_use": True},
            metadata={"user_id": "user-1"},
        )
    )
    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.MESSAGES
    assert request.messages[0].role == "system"
    assert request.messages[0].content == "be terse\n\nand kind"
    assert request.messages[1].role == "user"
    assert request.maximum_output_tokens == 128
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.stop == ("STOP", "END")
    assert request.stream is True
    assert request.include_usage is True
    assert request.tools[0].name == "search"
    assert request.tools[0].parameters == {"type": "object"}
    assert request.tool_choice == GatewayNamedToolChoice(name="search")
    assert request.parallel_tool_calls is False
    assert request.metadata == {"user_id": "user-1"}
    assert request.idempotency_key is None
    assert request.client_request_id is None


def test_decode_splits_tool_results_and_keeps_assistant_tool_calls() -> None:
    """A mixed history turn splits into ordered canonical messages."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "run the tool"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "on it"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "search",
                            "input": {"q": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "found it"}],
                        },
                        {"type": "text", "text": "now answer"},
                    ],
                },
            ]
        )
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "tool", "user"]
    assistant = decoded.request.messages[1]
    assert assistant.content == "on it"
    assert assistant.tool_calls[0].call_id == "call-1"
    assert assistant.tool_calls[0].raw_arguments == '{"q":"x"}'
    tool = decoded.request.messages[2]
    assert tool.tool_call_id == "call-1"
    assert tool.content == "found it"
    assert decoded.request.messages[3].content == "now answer"


def test_decode_drops_only_nonsemantic_cache_control() -> None:
    """A cache hint may be omitted without changing the requested model behavior."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                }
            ],
        )
    )
    assert decoded.request.messages[0].content == "hi"
    assert decoded.request.metadata == {}


def test_output_config_is_carried_verbatim_and_maps_canonical_effort() -> None:
    """The caller's output_config survives byte-for-byte and its effort rides
    the shared reasoning_effort field (Claude Code sends {"effort": ...} by
    default; accepted live without a beta, 2026-08-30)."""
    decoded = decode_messages(_body(output_config={"effort": "high"}))
    assert decoded.request.provider_output_config == {"effort": "high"}
    assert decoded.request.reasoning_effort == "high"
    # A non-canonical (future provider) effort stays verbatim-only: the
    # provider decides it, the gateway does not reject it.
    future = decode_messages(_body(output_config={"effort": "hyperdrive"}))
    assert future.request.provider_output_config == {"effort": "hyperdrive"}
    assert future.request.reasoning_effort is None
    assert decode_messages(_body()).request.provider_output_config is None


def test_thinking_config_is_carried_verbatim() -> None:
    """The caller's thinking object survives byte-for-byte on the canonical request."""
    config: JsonObject = {"type": "enabled", "budget_tokens": 1024}
    decoded = decode_messages(_body(thinking=config))
    assert decoded.request.provider_thinking_config == config
    assert decode_messages(_body()).request.provider_thinking_config is None

    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(thinking={"type": "enabled"}))
    assert excinfo.value.detail.param == "thinking"
    with pytest.raises(OpenAIProtocolError):
        decode_messages(_body(thinking={"type": "adaptive", "budget_tokens": 64}))


def test_thinking_history_blocks_ride_the_opaque_carrier_in_order() -> None:
    """Assistant reasoning history translates losslessly with byte-exact signatures."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "sig=="},
                        {"type": "redacted_thinking", "data": "opaque=="},
                        {"type": "text", "text": "done"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "search",
                            "input": {},
                        },
                    ],
                },
            ]
        )
    )
    assistant = decoded.request.messages[1]
    assert assistant.content == "done"
    assert assistant.tool_calls[0].call_id == "call-1"
    blocks = assistant.provider_reasoning
    assert [block.kind for block in blocks] == ["thinking", "redacted_thinking"]
    thinking, redacted = blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.text == "private"
    assert thinking.signature == "sig=="
    assert isinstance(redacted, RedactedThinkingBlock)
    assert redacted.data == "opaque=="

    # A thinking-only assistant turn (cut off mid-thinking) is legal history.
    only = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "partial"}],
                },
                {"role": "user", "content": "continue"},
            ]
        )
    )
    assert only.request.messages[1].provider_reasoning[0].kind == "thinking"

    with pytest.raises(OpenAIProtocolError, match="only valid in assistant messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "thinking", "thinking": "private"}],
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    ("overrides", "param_fragment"),
    [
        ({"service_tier": "auto"}, "service_tier"),
        ({"container": "c"}, "container"),
        ({"unknown_field": 1}, "unknown_field"),
    ],
)
def test_unsupported_and_unknown_top_level_fields_are_rejected(
    overrides: JsonObject, param_fragment: str
) -> None:
    """Unsupported and unknown fields answer a loud field-specific 400."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**overrides))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.param == param_fragment


def test_top_k_is_preserved_for_route_specific_validation() -> None:
    """The official Messages top-k field reaches the shared route contract."""
    decoded = decode_messages(_body(top_k=5))
    assert decoded.request.top_k == 5


def test_missing_max_tokens_is_rejected_with_its_field() -> None:
    """max_tokens is required by the Anthropic protocol."""
    payload = _body()
    del payload["max_tokens"]
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(payload)
    assert excinfo.value.detail.param == "max_tokens"


def test_image_blocks_are_rejected_with_a_targeted_hint() -> None:
    """A known-but-unsupported block gets its own explanation."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
                    }
                ]
            )
        )
    assert "image blocks are not supported" in excinfo.value.detail.message


def test_document_block_inside_tool_result_is_rejected() -> None:
    """Nested unsupported blocks inside tool results are rejected loudly."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [{"type": "document", "source": {}}],
                            }
                        ],
                    }
                ]
            )
        )
    assert "document blocks are not supported" in excinfo.value.detail.message


def test_role_misplaced_blocks_and_empty_content_are_rejected() -> None:
    """Blocks are validated against their legal roles and non-empty turns."""
    with pytest.raises(OpenAIProtocolError, match="only valid in assistant messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_use", "id": "call-1", "name": "n", "input": {}}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="only valid in user messages"):
        decode_messages(
            _body(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_result", "tool_use_id": "call-1"}],
                    }
                ]
            )
        )
    with pytest.raises(OpenAIProtocolError, match="must not be empty"):
        decode_messages(_body(messages=[{"role": "user", "content": ""}]))
    with pytest.raises(OpenAIProtocolError, match="must contain text"):
        decode_messages(_body(messages=[{"role": "assistant", "content": []}]))


def test_tool_choice_forms_and_stop_sequence_validation() -> None:
    """Every tool-choice form normalizes; bad stop sequences are rejected."""
    assert decode_messages(_body(tool_choice={"type": "auto"})).request.tool_choice == "auto"
    assert decode_messages(_body(tool_choice={"type": "none"})).request.tool_choice == "none"
    required = decode_messages(
        _body(
            tool_choice={"type": "any"},
            tools=[{"name": "search", "input_schema": {}}],
        )
    )
    assert required.request.tool_choice == "required"
    with pytest.raises(OpenAIProtocolError, match="requires a name"):
        decode_messages(_body(tool_choice={"type": "tool"}))
    with pytest.raises(OpenAIProtocolError, match="non-empty"):
        decode_messages(_body(stop_sequences=[""]))


def test_invalid_json_shape_errors_carry_a_dotted_field_path() -> None:
    """Wire validation errors name the offending field path."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(max_tokens=0))
    assert excinfo.value.detail.param == "max_tokens"
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(messages=[]))
    assert excinfo.value.detail.param == "messages"


def test_tool_result_error_state_is_preserved_on_the_canonical_message() -> None:
    """is_error travels on the canonical tool message without touching digests."""
    decoded = decode_messages(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "boom",
                            "is_error": True,
                        }
                    ],
                }
            ]
        )
    )
    tool = decoded.request.messages[0]
    assert tool.role == "tool"
    assert tool.tool_is_error is True
    plain = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1"}]}
            ]
        )
    )
    assert plain.request.messages[0].tool_is_error is False


def test_context_management_is_carried_verbatim_and_shallow_validated() -> None:
    """Claude Code's context-editing config survives byte-for-byte.

    Production incident (real Claude Code CLI, 2026-08-29): the field was a
    conscious UNSUPPORTED and every default-configured session 400ed.
    Validation is deliberately shallow (an object) because the nested shape
    is an evolving provider beta the gateway forwards verbatim.
    """
    config: JsonObject = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
            }
        ]
    }
    decoded = decode_messages(_body(context_management=config))
    assert decoded.request.context_management == config
    assert decode_messages(_body()).request.context_management is None

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(context_management="clear"))
    assert raised.value.detail.param == "context_management"


def test_thinking_display_is_carried_verbatim() -> None:
    """The adaptive display disposition rides the verbatim thinking config
    (Claude Code sends {"type": "adaptive", "display": "omitted"} by
    default; accepted live without a beta, 2026-08-30)."""
    config: JsonObject = {"type": "adaptive", "display": "omitted"}
    decoded = decode_messages(_body(thinking=config))
    assert decoded.request.provider_thinking_config == config


def test_mid_conversation_system_turn_decodes_positionally() -> None:
    """A system message after conversation start keeps its role and order."""
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "answer in uppercase",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ]
        )
    )
    assert [message.role for message in decoded.request.messages] == ["user", "system"]
    assert decoded.request.messages[1].content == "answer in uppercase"


def test_the_captured_claude_code_request_shape_decodes_losslessly() -> None:
    """Regression fixture: the field shapes real Claude Code (2.1.251) sends
    by default, trimmed from a live capture (2026-08-29). Every top-level
    field and block shape from the capture appears here."""
    decoded = decode_messages(
        {
            "model": "claude-fable-5",
            "max_tokens": 64000,
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "high"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
            "metadata": {"user_id": "device-hash-redacted"},
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "Run ls, then count the entries."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "count", "signature": "sig=="},
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "file_a.txt",
                            "is_error": False,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Available agent types trimmed.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }
    )
    request = decoded.request
    assert [message.role for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "system",
    ]
    assert request.reasoning_effort == "high"
    assert request.provider_output_config == {"effort": "high"}
    assert request.provider_thinking_config == {"type": "adaptive", "display": "omitted"}
    assert request.context_management is not None
    assert request.metadata == {"user_id": "device-hash-redacted"}
    assert request.ignored_parameters == ()


def test_diagnostics_and_speed_are_carried_verbatim_and_shallow_validated() -> None:
    """Claude Code's conditional diagnostics and fast-mode fields decode.

    Production incident (real Claude Code CLI, 2026-08-29): ``diagnostics``
    was undecided and every diagnostics-carrying session 400ed with "The
    parameter 'diagnostics' is not supported". Both fields are accepted by
    the provider behind their beta headers (verified live 2026-08-30), so
    the gateway carries them verbatim; validation stays shallow because the
    shapes are evolving provider betas.
    """
    decoded = decode_messages(_body(diagnostics={"previous_message_id": None}, speed="fast"))
    assert decoded.request.diagnostics == {"previous_message_id": None}
    assert decoded.request.speed == "fast"
    assert decode_messages(_body()).request.diagnostics is None
    assert decode_messages(_body()).request.speed is None

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(diagnostics="on"))
    assert raised.value.detail.param == "diagnostics"


def test_caller_beta_tokens_partition_into_allowlist_and_disclosures() -> None:
    """The caller anthropic-beta header forwards only allowlisted tokens.

    Claude Code activates the 1M context window with a caller-sent
    ``context-1m-2025-08-07`` token (captured live 2026-08-30); without
    forwarding it the provider serves 200K and long sessions fail. Every
    non-allowlisted token drops with a per-token disclosure, never a
    rejection and never a blind forward.
    """
    header = (
        "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,"
        "thinking-token-count-2026-05-13,fallback-credit-2026-06-01"
    )
    decoded = decode_messages(_body(), anthropic_beta=header)
    assert decoded.request.provider_beta_tokens == (
        "context-1m-2025-08-07",
        "interleaved-thinking-2025-05-14",
    )
    assert decoded.request.ignored_parameters == (
        "anthropic-beta.claude-code-20250219",
        "anthropic-beta.thinking-token-count-2026-05-13",
        "anthropic-beta.fallback-credit-2026-06-01",
    )
    assert decode_messages(_body()).request.provider_beta_tokens == ()

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_messages(_body(), anthropic_beta="bad\nvalue")
    assert raised.value.detail.param == "anthropic-beta"


def test_a_real_toolset_tool_description_over_8k_decodes() -> None:
    """Tool descriptions bound generously: a real Claude Code toolset
    carried a description past the earlier 8k cap and 400ed with
    "Invalid value for 'tools.1.description'" while the provider accepts
    40k-character descriptions live (verified 2026-08-30)."""
    tools = [
        {"name": "small", "description": "x", "input_schema": {"type": "object"}},
        {"name": "large", "description": "y" * 40_000, "input_schema": {"type": "object"}},
    ]
    decoded = decode_messages(_body(tools=tools))
    description = decoded.request.tools[1].description
    assert description is not None and len(description) == 40_000


def test_decode_accepts_the_live_eager_input_streaming_tool_shape() -> None:
    """Live-captured 2026-08-30: a production Claude Code session sent a tool
    carrying ``eager_input_streaming`` and got "Invalid value for
    'tools.0.eager_input_streaming'" while api.anthropic.com accepts the field
    bare (no beta header). The exact wire shape stays accepted."""
    decoded = decode_messages(
        _body(
            stream=True,
            tools=[
                {
                    "name": "Bash",
                    "description": "Executes a bash command and returns its output.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                    "eager_input_streaming": True,
                }
            ],
        )
    )
    tool = decoded.request.tools[0]
    assert tool.eager_input_streaming is True
    assert decoded.request.ignored_parameters == ()


def test_decode_carries_every_provider_native_tool_annotation() -> None:
    """The provider-native tool annotations land on the canonical tool
    (each accepted bare by the live API, verified 2026-08-30)."""
    decoded = decode_messages(
        _body(
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "eager_input_streaming": False,
                    "defer_loading": False,
                    "allowed_callers": ["code_execution_20260120"],
                    "input_examples": [{"city": "Paris"}],
                }
            ]
        )
    )
    tool = decoded.request.tools[0]
    assert tool.strict is True
    assert tool.cache_control == {"type": "ephemeral", "ttl": "1h"}
    assert tool.eager_input_streaming is False
    assert tool.defer_loading is False
    assert tool.allowed_callers == ("code_execution_20260120",)
    assert tool.input_examples == ({"city": "Paris"},)

    bare = decode_messages(
        _body(tools=[{"name": "get_weather", "input_schema": {"type": "object"}}])
    ).request.tools[0]
    assert bare.strict is False
    assert bare.cache_control is None
    assert bare.eager_input_streaming is None
    assert bare.defer_loading is None
    assert bare.allowed_callers is None
    assert bare.input_examples is None


def test_decode_carries_top_level_cache_control_and_inference_geo() -> None:
    """Top-level auto-caching and the inference region ride their carriers
    verbatim (both accepted bare by the live API, verified 2026-08-30)."""
    decoded = decode_messages(_body(cache_control={"type": "ephemeral"}, inference_geo="us"))
    assert decoded.request.provider_cache_control == {"type": "ephemeral"}
    assert decoded.request.inference_geo == "us"
    absent = decode_messages(_body()).request
    assert absent.provider_cache_control is None
    assert absent.inference_geo is None

    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(cache_control={"type": "persistent"}))
    assert excinfo.value.detail.param == "cache_control.type"


@pytest.mark.parametrize(
    "field",
    ["user_profile_id", "fallbacks", "fallback_credit_token", "betas"],
)
def test_route_identity_and_delegation_fields_stay_consciously_rejected(field: str) -> None:
    """Fallback model swaps, body-borne beta opt-ins, and third-party
    attribution are recorded rejections, each answered by its named 400."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(**{field: "x"}))
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail.param == field


def test_validation_errors_state_the_expectation_not_only_the_field() -> None:
    """A strict-decode 400 names the field and what was expected there."""
    with pytest.raises(OpenAIProtocolError) as unknown:
        decode_messages(_body(tools=[{"name": "t", "input_schema": {}, "eager_streaming": True}]))
    assert unknown.value.detail.param == "tools.0.eager_streaming"
    assert "Unknown parameter 'tools.0.eager_streaming'" in unknown.value.detail.message

    with pytest.raises(OpenAIProtocolError) as invalid:
        decode_messages(_body(tools=[{"name": "t", "input_schema": {}, "strict": "maybe"}]))
    assert invalid.value.detail.param == "tools.0.strict"
    assert "Invalid value for 'tools.0.strict'" in invalid.value.detail.message
    assert "bool" in invalid.value.detail.message


def test_decode_accepts_the_live_web_search_server_tool_shape() -> None:
    """Live repro (engine 0.7.10, 2026-08-31): a Claude Code WebSearch tool
    definition 400ed with "Invalid value for 'tools.0.input_schema'" because
    server-tool entries carry no input_schema. The typed entry decodes onto
    the verbatim server-tool carrier at its caller position."""
    decoded = decode_messages(
        _body(
            stream=True,
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                {
                    "name": "Bash",
                    "description": "Executes a bash command.",
                    "input_schema": {"type": "object"},
                },
            ],
        )
    )
    request = decoded.request
    assert [tool.name for tool in request.tools] == ["Bash"]
    assert len(request.provider_server_tools) == 1
    server = request.provider_server_tools[0]
    assert server.position == 0
    assert server.definition == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 8,
    }
    assert request.ignored_parameters == ()


def test_decode_names_rejected_and_unknown_tool_types() -> None:
    """Typed tool entries answer recorded, named 400s."""
    with pytest.raises(OpenAIProtocolError) as rejected:
        decode_messages(_body(tools=[{"type": "mcp_toolset", "name": "srv"}]))
    assert rejected.value.detail.param == "tools.0.type"
    assert "consciously not served" in rejected.value.detail.message

    with pytest.raises(OpenAIProtocolError) as unknown:
        decode_messages(_body(tools=[{"type": "web_search_20990101", "name": "web_search"}]))
    assert unknown.value.detail.param == "tools.0.type"
    assert "unknown tool type" in unknown.value.detail.message


def test_decode_carries_echoed_server_tool_history_blocks_verbatim() -> None:
    """The turn-2 echo of a web_search turn splits into ordered verbatim
    carrier messages so Anthropic rungs replay the exact assistant content."""
    search_block: JsonObject = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "utc"},
    }
    result_block: JsonObject = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "caller": {"type": "direct"},
        "content": [{"type": "web_search_result", "url": "https://utc.test"}],
    }
    decoded = decode_messages(
        _body(
            messages=[
                {"role": "user", "content": "what is the date"},
                {
                    "role": "assistant",
                    "content": [search_block, result_block, {"type": "text", "text": "Monday."}],
                },
                {"role": "user", "content": "thanks, and tomorrow?"},
            ],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "assistant", "assistant", "user"]
    assert decoded.request.messages[1].provider_server_tool_block == search_block
    assert decoded.request.messages[2].provider_server_tool_block == result_block
    assert decoded.request.messages[3].content == "Monday."

    with pytest.raises(OpenAIProtocolError) as excinfo:
        decode_messages(_body(messages=[{"role": "user", "content": [search_block]}]))
    assert "only valid in assistant messages" in excinfo.value.detail.message

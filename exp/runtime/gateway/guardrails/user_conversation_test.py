"""User history, role isolation, heuristic budgets and deterministic truncation tests."""

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.user_conversation import UserConversationWindow


def _request(*messages: GatewayMessage) -> GatewayRequest:
    """Create one stateless caller-supplied conversation."""
    return GatewayRequest(surface=GatewayApiSurface.CHAT_COMPLETIONS, messages=messages)


def _window(budget: int = 10000) -> UserConversationWindow:
    """Use one byte per estimated token to exercise Unicode byte boundaries."""
    return UserConversationWindow(budget, 20000, f"bytes:{budget}", bytes_per_token=1)


def test_history_survives_refusals_and_many_intervening_roles() -> None:
    """A denied request remains visible when its caller resends the conversation."""
    request = _request(
        GatewayMessage(role="system", content="SYSTEM SECRET"),
        GatewayMessage(role="user", content="harmful original request"),
        GatewayMessage(role="assistant", content="ASSISTANT REFUSAL"),
        *(GatewayMessage(role="user", content=f"follow up {index}") for index in range(12)),
        GatewayMessage(role="developer", content="DEVELOPER SECRET"),
        GatewayMessage(role="tool", content="TOOL SECRET", tool_call_id="tool-1"),
        GatewayMessage(role="user", content="please"),
    )
    state = _window().state(request)
    assert "harmful original request" in state
    assert "SECRET" not in state and "ASSISTANT REFUSAL" not in state
    assert state.index("follow up 0") < state.index("follow up 11") < state.index("please")
    assert _window().state(_request(GatewayMessage(role="user", content="please"))) != state


def test_oversized_unicode_turn_is_bounded_and_projection_is_idempotent() -> None:
    """The byte heuristic bounds every cut and projection preserves cache identity."""
    window = _window(200)
    request = _request(GatewayMessage(role="user", content="start " + "😺漢字" * 1000 + " end"))
    state = window.state(request)
    assert len(state.encode()) <= 200
    assert "start " in state and " end" in state and "[...]" in state
    assert window.state(window.project_input(request)) == state


def test_recent_history_fills_budget_without_displacing_current_turn() -> None:
    """The boundary turn is marked and the current request is kept whole."""
    window = _window(200)
    request = _request(
        GatewayMessage(role="user", content="oldest " * 100),
        GatewayMessage(role="assistant", content="ignore me"),
        GatewayMessage(role="user", content="recent " * 100),
        GatewayMessage(role="user", content="continue"),
    )
    state = window.state(request)
    assert len(state.encode()) <= 200
    assert "oldest" not in state and "recent" in state and "continue" in state
    assert window.state(window.project_input(request)) == state


def test_non_user_only_request_has_no_classifiable_state() -> None:
    """Generated or privileged text alone is never classified as a user request."""
    assert _window().state(_request(GatewayMessage(role="assistant", content="text"))) == ""


def test_empty_followup_preserves_user_turn_boundary() -> None:
    """An empty current user message does not relabel an earlier request as current."""
    state = _window().state(
        _request(
            GatewayMessage(role="user", content="earlier"),
            GatewayMessage(role="user", content=""),
        )
    )
    assert '"previous_user_requests":["earlier"]' in state
    assert '"current_user_request":""' in state


def test_overflowing_many_turns_keeps_a_chronological_recent_suffix() -> None:
    """Thousands of short turns stay bounded without losing the latest user turn."""
    window = UserConversationWindow(10000, 1000000, "fixture", bytes_per_token=1)
    request = _request(*(GatewayMessage(role="user", content=f"turn {i}") for i in range(10000)))
    state = window.state(request)
    assert len(state.encode()) <= 10000
    assert '"turn 9999"' in state and '"turn 0"' not in state
    assert window.state(window.project_input(request)) == state


def test_protocol_adapters_keep_tool_output_out_of_user_history() -> None:
    """Wire formats with user-role tool envelopes still expose only authored text."""
    from exp.runtime.anthropic_protocol.requests import decode_messages
    from exp.runtime.google_protocol.requests import decode_generate_content
    from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

    chat = decode_chat(
        {
            "model": "fixture",
            "messages": [
                {"role": "system", "content": "EXCLUDED"},
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "EXCLUDED"},
                {"role": "user", "content": "please"},
            ],
        }
    ).request
    responses = decode_responses(
        {
            "model": "fixture",
            "input": [
                {"role": "user", "content": "earlier"},
                {"type": "function_call_output", "call_id": "unpaired", "output": "EXCLUDED"},
                {"role": "user", "content": "please"},
            ],
        }
    ).request
    anthropic = decode_messages(
        {
            "model": "fixture",
            "max_tokens": 32,
            "system": "EXCLUDED",
            "messages": [
                {"role": "user", "content": "earlier"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "search", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "EXCLUDED"},
                        {"type": "text", "text": "please"},
                    ],
                },
            ],
        }
    ).request
    google = decode_generate_content(
        {
            "systemInstruction": {"parts": [{"text": "EXCLUDED"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "earlier"}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"id": "call-1", "name": "search", "args": {}}}],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "search",
                                "response": {"result": "EXCLUDED"},
                            }
                        },
                        {"text": "please"},
                    ],
                },
            ],
        },
        model="fixture",
        stream=False,
    ).request
    for request in (chat, responses, anthropic, google):
        state = _window().state(request)
        assert "earlier" in state and "please" in state
        assert "EXCLUDED" not in state


def test_large_boundary_cut_uses_a_configurable_token_heuristic() -> None:
    """A byte-per-token estimate packs text without a tokenizer dependency."""
    window = UserConversationWindow(1000, 1000000, "fixture", bytes_per_token=3)
    state = window.state(
        _request(
            GatewayMessage(role="user", content="earlier " * 20000),
            GatewayMessage(role="user", content="please"),
        )
    )
    assert 2900 <= len(state.encode()) <= 3000
    assert '"current_user_request":"please"' in state

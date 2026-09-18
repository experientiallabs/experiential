"""Tests for the admission-time web-search planner."""

import time

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
)
from exp.runtime.gateway.web_search.backend import FailingWebSearchBackend, StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearch, GatewayWebSearchResult
from exp.runtime.gateway.web_search.plan import (
    DROPPED_FAILED,
    DROPPED_NO_QUERY,
    DROPPED_UNAVAILABLE,
    derive_query,
    inject_results,
    instruction_text,
    natively_served,
    plan_web_search,
    strip_search_carriers,
)

_PLUGIN = GatewayWebSearch(declared_as="plugin")
_HITS = (
    GatewayWebSearchResult(
        url="https://rust-lang.org/blog", title="Rust Blog", snippet="Rust 1.99 released."
    ),
    GatewayWebSearchResult(url="https://example.com/b", title="B"),
)


def _request(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
    *,
    search: GatewayWebSearch | None = _PLUGIN,
    provider_native_tools: tuple[GatewayProviderNativeTool, ...] = (),
) -> GatewayRequest:
    return GatewayRequest(
        surface=surface,
        messages=(
            GatewayMessage(role="system", content="Be brief."),
            GatewayMessage(role="user", content="  What is   the latest Rust release? "),
        ),
        web_search=search,
        provider_native_tools=provider_native_tools,
    )


def _deadline() -> float:
    return time.monotonic() + 30.0


def test_no_search_request_is_a_no_op() -> None:
    request = _request(search=None)
    plan = plan_web_search(
        request,
        ["openai_compatible"],
        StaticWebSearchBackend(_HITS),
        deadline_monotonic=_deadline(),
    )
    assert plan.request is request
    assert plan.admission is None


def test_native_routes_keep_the_provider_search() -> None:
    responses = GatewayWebSearch(declared_as="responses_tool")
    assert natively_served(responses, ["openai_responses", "openai_responses"])
    assert not natively_served(responses, ["openai_responses", "openai_compatible"])
    messages = GatewayWebSearch(declared_as="messages_server_tool")
    assert natively_served(messages, ["anthropic_messages"])
    assert not natively_served(messages, ["openai_responses"])
    assert not natively_served(GatewayWebSearch(declared_as="plugin"), ["openai_responses"])
    request = _request(GatewayApiSurface.RESPONSES, search=responses)
    plan = plan_web_search(
        request, ["openai_responses"], StaticWebSearchBackend(_HITS), deadline_monotonic=_deadline()
    )
    assert plan.request is request and plan.admission is None


def test_missing_backend_drops_with_disclosure_and_strips_carriers() -> None:
    request = _request(
        GatewayApiSurface.RESPONSES,
        search=GatewayWebSearch(declared_as="responses_tool"),
        provider_native_tools=(GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),),
    )
    plan = plan_web_search(request, ["openai_compatible"], None, deadline_monotonic=_deadline())
    assert plan.admission is None
    assert DROPPED_UNAVAILABLE in plan.request.ignored_parameters
    assert plan.request.provider_native_tools == ()
    assert plan.request.web_search is not None  # identity keeps the caller's ask


def test_backend_failure_serves_without_results() -> None:
    plan = plan_web_search(
        _request(), ["openai_compatible"], FailingWebSearchBackend(), deadline_monotonic=_deadline()
    )
    assert plan.admission is None
    assert DROPPED_FAILED in plan.request.ignored_parameters
    assert len(plan.request.messages) == 2


def test_no_user_text_is_disclosed() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="system", content="only system"),),
        web_search=GatewayWebSearch(declared_as="plugin"),
    )
    plan = plan_web_search(
        request,
        ["openai_compatible"],
        StaticWebSearchBackend(_HITS),
        deadline_monotonic=_deadline(),
    )
    assert DROPPED_NO_QUERY in plan.request.ignored_parameters


def test_query_is_the_latest_user_turn_collapsed_and_bounded() -> None:
    assert derive_query(_request()) == "What is the latest Rust release?"
    long_request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="x" * 1000),),
    )
    assert len(derive_query(long_request)) == 400


def test_successful_search_injects_after_the_leading_instructions() -> None:
    backend = StaticWebSearchBackend(_HITS)
    request = _request(
        search=GatewayWebSearch(declared_as="plugin", max_results=1, search_prompt="Cite!")
    )
    plan = plan_web_search(
        request,
        ["openai_compatible", "gemini_generate_content"],
        backend,
        deadline_monotonic=_deadline(),
    )
    assert backend.queries == ["What is the latest Rust release?"]
    assert plan.admission == {
        "query": "What is the latest Rust release?",
        "requests": 1,
        "results": [{"url": "https://rust-lang.org/blog", "title": "Rust Blog"}],
    }
    roles = [message.role for message in plan.request.messages]
    assert roles == ["system", "system", "user"]
    injected = plan.request.messages[1].content or ""
    assert injected.startswith("Cite!")
    assert "[1] Rust Blog" in injected
    assert "URL: https://rust-lang.org/blog" in injected
    assert "Rust 1.99 released." in injected
    assert "example.com/b" not in injected
    assert plan.request.ignored_parameters == ()


def test_responses_injection_uses_a_system_turn_too() -> None:
    request = _request(
        GatewayApiSurface.RESPONSES, search=GatewayWebSearch(declared_as="model_suffix")
    )
    plan = plan_web_search(
        request,
        ["openai_compatible"],
        StaticWebSearchBackend(_HITS),
        deadline_monotonic=_deadline(),
    )
    assert plan.request.messages[1].role == "system"


def test_instruction_text_defaults_frame_the_date_and_citation_rule() -> None:
    text = instruction_text(GatewayWebSearch(declared_as="plugin"), "q", _HITS, today="2026-09-18")
    assert text.startswith("A web search was conducted on 2026-09-18")
    assert "[2] B" in text and "URL: https://example.com/b" in text


def test_strip_carriers_clears_a_tool_choice_naming_the_server_tool() -> None:
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        provider_server_tools=({"type": "web_search_20250305", "name": "web_search"},),
        tool_choice=GatewayNamedToolChoice(name="web_search"),
        web_search=GatewayWebSearch(declared_as="messages_server_tool"),
    )
    stripped = strip_search_carriers(request)
    assert stripped.provider_server_tools == ()
    assert stripped.tool_choice is None
    assert inject_results(stripped, "results").messages[0].role == "system"

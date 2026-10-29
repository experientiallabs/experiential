"""Tests for content-free calling-application classification."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.client_apps import (
    CLIENT_APP_LABELS,
    MAXIMUM_USER_AGENT_CHARS,
    ClientApp,
    bounded_user_agent,
    classify_client_app,
)


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("claude-cli/2.1.278 (external, cli)", ClientApp.CLAUDE_CODE),
        ("claude-cli/2.1.273 (external, claude-vscode, agent-sdk/0.3.273)", ClientApp.CLAUDE_CODE),
        ("claude-code/1.0.0 (Hermes Gateway)", ClientApp.HERMES),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) HermesAgent/3.2.0", ClientApp.HERMES),
        ("HermesAgent/0.21.0", ClientApp.HERMES),
        ("hermes-agent/0.21.0", ClientApp.HERMES),
        ("codex_exec/0.153.4 (Debian 12.0.0; x86_64) xterm (codex_exec; 0.153.4)", ClientApp.CODEX),
        ("codex_cli_rs/0.151.0 (Mac OS 15.5.0; arm64) iTerm.app/3.5.14", ClientApp.CODEX),
        ("Codex Desktop/0.155.0-alpha.9.2 (Windows 10.0.26200; x86_64)", ClientApp.CODEX),
        ("Codex", ClientApp.CODEX),
        (
            "codex-tui/0.156.1 (Windows 10.0.26200; x86_64) WindowsTerminal (codex-tui; 0.156.1)",
            ClientApp.CODEX,
        ),
        ("opencode/1.18.33 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14", ClientApp.OPENCODE),
        ("opencode/latest/2.0.18/desktop", ClientApp.OPENCODE),
        ("Kilo-Code/7.5.16 ai-sdk/provider-utils/4.0.27 runtime/bun/1.3.14", ClientApp.KILO_CODE),
        ("Cline/4.1.17", ClientApp.CLINE),
        ("RooCode/3.54.0", ClientApp.ROO_CODE),
        ("Cursor/1.0", ClientApp.CURSOR),
        ("QwenCode/0.24.4 (darwin; x64)", ClientApp.QWEN_CODE),
        ("GeminiCLI/0.9.0 (darwin; arm64)", ClientApp.GEMINI_CLI),
        ("GitHubCopilotChat/0.67.0", ClientApp.GITHUB_COPILOT),
        ("Zed/1.18.1+stable.352 (windows; x86_64)", ClientApp.ZED),
        ("pi/1.0", ClientApp.PI),
        ("pi", ClientApp.PI),
        ("pi (win32 10.0.26200; x64)", ClientApp.PI),
        ("pi (linux 6.6.87.2-microsoft-standard-WSL2; x64)", ClientApp.PI),
        ("omp/18.6.0", ClientApp.OH_MY_PI),
        ("WorkBuddy/5.5.2 WorkBuddy/5.5.2 CLI/2.137.1", ClientApp.WORKBUDDY),
        ("OmniCopilot-VSCode", ClientApp.OMNI_COPILOT),
        ("cli-proxy-openai-compat", ClientApp.CLI_PROXY_API),
        (
            "EasyCLIProxyAPI/0.3.11 (+https://github.com/router-for-me/EasyCLIProxyAPI)",
            ClientApp.CLI_PROXY_API,
        ),
        ("Agents/Python 0.19.0", ClientApp.OPENAI_AGENTS_SDK),
        ("OpenClaw/2026.9.1", ClientApp.OPENCLAW),
        (
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 CherryStudio/1.7.15 Chrome/140",
            ClientApp.CHERRY_STUDIO,
        ),
        ("Mozilla/5.0 (Macintosh) AppleWebKit/537.36 xyz.chatboxapp.app/1.23", ClientApp.CHATBOX),
        ("n8n", ClientApp.N8N),
        ("litellm/1.99.0", ClientApp.LITELLM),
    ],
)
def test_known_user_agents_classify(user_agent: str, expected: ClientApp) -> None:
    """Real observed User-Agent values map to their application."""
    assert classify_client_app(user_agent=user_agent) is expected


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("OpenAI/Python 2.8.1", ClientApp.OPENAI_SDK),
        ("AsyncOpenAI/Python 2.8.1", ClientApp.OPENAI_SDK),
        ("OpenAI/JS 5.0.1", ClientApp.OPENAI_SDK),
        ("OpenAIClientAsyncImpl/Java unknown", ClientApp.OPENAI_SDK),
        ("Anthropic/JS 0.90.0", ClientApp.ANTHROPIC_SDK),
        ("ai/6.0.185 ai-sdk/provider-utils/4.0.50 runtime/node.js/24", ClientApp.VERCEL_AI_SDK),
        (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36",
            ClientApp.BROWSER,
        ),
        (
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            ClientApp.BROWSER,
        ),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36", ClientApp.BROWSER),
        ("curl/8.7.1", ClientApp.CURL),
        ("node-fetch", ClientApp.CUSTOM_CODE),
        (
            "Mozilla/5.0 (Windows NT; Windows NT 10.0; en-US) WindowsPowerShell/5.1.19041.6456",
            ClientApp.CUSTOM_CODE,
        ),
        ("node", ClientApp.CUSTOM_CODE),
        ("undici", ClientApp.CUSTOM_CODE),
        ("Bun/1.4.2", ClientApp.CUSTOM_CODE),
        ("Go-http-client/2.0", ClientApp.CUSTOM_CODE),
        ("python-httpx/0.28.1", ClientApp.CUSTOM_CODE),
        ("python-requests/2.34.2", ClientApp.CUSTOM_CODE),
        ("Python-urllib/3.14", ClientApp.CUSTOM_CODE),
        ("Python/3.14 aiohttp/3.14.3", ClientApp.CUSTOM_CODE),
        ("axios/1.17.0", ClientApp.CUSTOM_CODE),
        ("okhttp/4.12.0", ClientApp.CUSTOM_CODE),
        ("GuzzleHttp/7", ClientApp.CUSTOM_CODE),
        ("Mozilla/5.0", ClientApp.CUSTOM_CODE),
        (
            "Mozilla/5.0 (Windows NT 10.0; Microsoft Windows 10.0.26200; en-US) PowerShell/7.6.6",
            ClientApp.CUSTOM_CODE,
        ),
    ],
)
def test_sdks_and_http_libraries_name_the_caller_kind(user_agent: str, expected: ClientApp) -> None:
    """A caller no named application matches is reported by the kind its library names."""
    assert classify_client_app(user_agent=user_agent) is expected


@pytest.mark.parametrize(
    "user_agent",
    [
        None,
        "",
        "   ",
        "codex-router/0.4.0-beta.4",
        "hermes-free-audit/0.2",
        "pilot/1.0",
        "nodemon-proxy/1.0",
        "Mozilla/5.0 (compatible; ChatClient/1.0)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/0328",
    ],
)
def test_unrecognized_clients_stay_unclassified(user_agent: str | None) -> None:
    """Look-alike names and unknown tools are never guessed into an application or kind."""
    assert classify_client_app(user_agent=user_agent) is None


def test_named_signals_win_over_the_caller_kind() -> None:
    """X-Title, HTTP-Referer and originator name the app even behind an SDK User-Agent."""
    assert classify_client_app(user_agent="OpenAI/JS 5.0", app_title="Cline") is ClientApp.CLINE
    assert (
        classify_client_app(user_agent="node", app_referer="https://kilocode.ai")
        is ClientApp.KILO_CODE
    )
    assert classify_client_app(user_agent="node", app_referer="https://example.com") is (
        ClientApp.CUSTOM_CODE
    )


def test_user_agent_wins_over_other_signals() -> None:
    """The User-Agent is the most specific signal and takes precedence."""
    assert (
        classify_client_app(
            user_agent="opencode/1.18.31",
            originator="codex_cli_rs",
            app_title="Cline",
            app_referer="https://kilocode.ai",
        )
        is ClientApp.OPENCODE
    )


def test_originator_identifies_codex_behind_a_generic_user_agent() -> None:
    """Codex's originator header classifies even when the User-Agent is generic."""
    assert classify_client_app(user_agent="node", originator="codex_vscode") is ClientApp.CODEX


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Hermes Agent", ClientApp.HERMES),
        ("  kilo code ", ClientApp.KILO_CODE),
        ("OpenClaw", ClientApp.OPENCLAW),
        ("My Internal Tool", ClientApp.OPENAI_SDK),
    ],
)
def test_app_title_classifies_known_names(title: str, expected: ClientApp | None) -> None:
    """X-Title names a known app exactly; any other title leaves the SDK caller kind."""
    assert classify_client_app(user_agent="OpenAI/Python 2.8.1", app_title=title) is expected


@pytest.mark.parametrize(
    ("referer", "expected"),
    [
        ("https://kilocode.ai", ClientApp.KILO_CODE),
        ("https://www.opencode.ai/docs", ClientApp.OPENCODE),
        ("https://hermes-agent.nousresearch.com/", ClientApp.HERMES),
        ("cline.bot", ClientApp.CLINE),
        ("https://example.com", None),
        ("https://notkilocode.ai", None),
        ("http://[::1", None),
    ],
)
def test_app_referer_matches_hosts_and_parent_domains(
    referer: str, expected: ClientApp | None
) -> None:
    """HTTP-Referer matches a known host or its subdomains, never a lookalike suffix."""
    assert classify_client_app(user_agent=None, app_referer=referer) is expected


def test_oversized_header_is_inspected_within_bounds() -> None:
    """A very long header is classified from its bounded prefix without failing."""
    assert classify_client_app(user_agent="claude-cli/2.1.0 " + "x" * 100_000) is (
        ClientApp.CLAUDE_CODE
    )


def test_bounded_user_agent_truncates_and_blanks_to_none() -> None:
    """Stored User-Agent values are trimmed, bounded and never empty strings."""
    assert bounded_user_agent(None) is None
    assert bounded_user_agent("   ") is None
    assert bounded_user_agent("  claude-cli/2.1.0  ") == "claude-cli/2.1.0"
    long_value = bounded_user_agent("a" * 1_000)
    assert long_value is not None and len(long_value) == MAXIMUM_USER_AGENT_CHARS


def test_every_application_has_a_label() -> None:
    """The label table covers the whole closed vocabulary."""
    assert set(CLIENT_APP_LABELS) == set(ClientApp)

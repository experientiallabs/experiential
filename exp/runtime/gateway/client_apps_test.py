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
    "user_agent",
    [
        None,
        "",
        "   ",
        "OpenAI/Python 2.8.1",
        "node",
        "Go-http-client/2.0",
        "python-httpx/0.28.1",
        "codex-router/0.4.0-beta.4",
        "hermes-free-audit/0.2",
        "pilot/1.0",
        "Anthropic/JS 0.90.0",
    ],
)
def test_generic_clients_stay_unclassified(user_agent: str | None) -> None:
    """SDKs, proxies and look-alike names never become a guessed application."""
    assert classify_client_app(user_agent=user_agent) is None


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
        ("My Internal Tool", None),
    ],
)
def test_app_title_classifies_known_names(title: str, expected: ClientApp | None) -> None:
    """The OpenRouter X-Title header classifies exact known application names only."""
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

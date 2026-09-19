"""Capture exposes only foreground collection and independent network reset."""

from pathlib import Path

import pytest
from click import unstyle
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.capture import app as capture_module


@pytest.mark.parametrize(
    ("selection", "domains"),
    [
        ("1\n\n", ("api.openai.com", "chatgpt.com")),
        ("2\n\n", ("api.anthropic.com",)),
        ("all\n\n", ("api.openai.com", "chatgpt.com", "api.anthropic.com")),
        ("\n1\n\n", ("api.openai.com", "chatgpt.com")),
    ],
)
def test_capture_asks_which_providers_to_capture(
    monkeypatch: pytest.MonkeyPatch, selection: str, domains: tuple[str, ...]
) -> None:
    """Only providers explicitly chosen in the picker reach foreground capture."""
    calls: list[tuple[str, ...]] = []

    def run_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Record the requested domains without starting network interception."""
        calls.append(domains)

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", run_capture)
    result = CliRunner().invoke(app, ["capture"], input=selection)
    assert result.exit_code == 0, result.output
    assert "What would you like to capture?" in result.output
    assert "OpenAI / Codex" in result.output
    assert "Anthropic / Claude Code" in result.output
    assert calls == [domains]


@pytest.mark.parametrize(("selection", "exit_code"), [("q\n", 0), ("b\n", 0), ("", 2)])
def test_capture_without_selection_never_starts_setup(
    monkeypatch: pytest.MonkeyPatch, selection: str, exit_code: int
) -> None:
    """Cancellation or unavailable input cannot silently capture every provider."""

    def unexpected_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Reject any attempt to start login or interception without a selection."""
        raise AssertionError("unselected capture must not start setup")

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", unexpected_capture)
    result = CliRunner().invoke(
        app,
        ["capture"],
        input=selection,
        env={"TERM": "xterm-256color", "FORCE_COLOR": "1"},
    )
    assert result.exit_code == exit_code, result.output
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    expected = "Capture cancelled" if selection else "--domain HOST"
    assert expected in output


def test_explicit_domains_bypass_picker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advanced overrides retain exactly their normalized hosts without prompting."""
    calls: list[tuple[str, ...]] = []

    def run_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Record selected hosts without starting the capture runner."""
        calls.append(domains)

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", run_capture)
    result = CliRunner().invoke(
        app, ["capture", "--domain", "API.OPENAI.COM.", "--domain", "api.openai.com"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("api.openai.com",)]
    assert "What would you like to capture?" not in result.output


def test_reset_does_not_login_or_start_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline reset only invokes the networking recovery helper."""
    reset_calls: list[str] = []

    def unexpected_capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
        """Fail if offline reset attempts to invoke the capture lifetime."""
        raise AssertionError("reset must never read a login or start capture")

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", unexpected_capture)
    monkeypatch.setattr(capture_module, "reset_capture_system", lambda: reset_calls.append("reset"))
    result = CliRunner().invoke(app, ["capture", "reset"])
    assert result.exit_code == 0, result.output
    assert reset_calls == ["reset"]
    assert "networking restored" in result.output
    assert "What would you like to capture?" not in result.output


@pytest.mark.parametrize(
    "domain", ["https://example.com", "127.0.0.1", "api.local", "api.internal", "*.openai.com"]
)
def test_domain_validation_precedes_login(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    """Invalid domain arguments are rejected before authentication."""
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    result = CliRunner().invoke(app, ["capture", "--domain", domain])
    assert result.exit_code == 2
    assert "exact DNS hostname" in result.output


def test_domain_limit_precedes_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI enforces the helper's finite target limit before starting capture."""
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    arguments = ["capture"]
    for index in range(33):
        arguments.extend(("--domain", f"provider{index}.example.com"))
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "1 to 32" in result.output


def test_no_background_management_commands() -> None:
    """Background management verbs are outside the public command surface."""
    runner = CliRunner()
    for command in ("status", "stop", "start"):
        result = runner.invoke(app, ["capture", command])
        assert result.exit_code == 2


def test_help_never_attempts_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading command help cannot trigger privileged setup."""

    def reject_setup() -> None:
        """Fail if help attempts to inspect or change system state."""
        raise AssertionError("help must not inspect or change system state")

    monkeypatch.setattr(capture_module, "_require_macos", reject_setup)
    result = CliRunner().invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "reset" in result.output
    assert "What would you like to capture?" not in result.output


def test_capture_requires_supported_python_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older SDK interpreter rejects capture before authentication or OS changes."""
    monkeypatch.setattr(capture_module.sys, "version_info", (3, 12, 0))
    with pytest.raises(ValueError, match="Capture requires Python 3.13"):
        capture_module._capture(Console(), domains=("api.openai.com",), root=Path("."))


def test_runner_replaces_foreground_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture keeps one foreground process and passes only non-secret CLI arguments."""
    launches: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(capture_module.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(
        capture_module.os, "execv", lambda executable, args: launches.append((executable, args))
    )
    capture_module._capture(Console(), domains=("api.openai.com",), root=Path("/tmp/project"))
    assert launches == [
        (
            capture_module.sys.executable,
            [
                capture_module.sys.executable,
                "-m",
                "exp.cli.capture.runner",
                "--root",
                "/tmp/project",
                "--domain",
                "api.openai.com",
            ],
        )
    ]

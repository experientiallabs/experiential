"""Capture starts foreground collection across supported providers without a picker."""

import subprocess
import sys
from pathlib import Path

import pytest
from click import unstyle
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.capture import app as capture_module


def test_capture_defaults_to_supported_providers_without_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain capture starts the foreground runner without a provider selection prompt."""
    calls: list[tuple[str, ...]] = []

    def run_capture(
        console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
    ) -> None:
        """Record the requested domains without starting network interception."""
        calls.append(domains)

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", run_capture)
    result = CliRunner().invoke(app, ["capture"], input="")
    assert result.exit_code == 0, result.output
    assert "What would you like to capture?" not in unstyle(result.output)
    assert calls == [("api.openai.com", "chatgpt.com", "api.anthropic.com")]


def test_explicit_domains_replace_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advanced overrides retain exactly their normalized hosts, excluding other defaults."""
    calls: list[tuple[str, ...]] = []

    def run_capture(
        console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
    ) -> None:
        """Record selected hosts without starting the capture runner."""
        calls.append(domains)

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", run_capture)
    result = CliRunner().invoke(
        app, ["capture", "--domain", "API.OPENAI.COM.", "--domain", "api.openai.com"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("api.openai.com",)]
    assert "What would you like to capture?" not in unstyle(result.output)


@pytest.mark.parametrize("command", ["reset", "status", "stop", "start"])
def test_no_management_commands(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    """Unsupported management verbs fail before starting login or interception."""

    def unexpected_capture(
        console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
    ) -> None:
        """Reject setup for an invalid capture invocation."""
        raise AssertionError("an unknown subcommand must not start capture")

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", unexpected_capture)
    result = CliRunner().invoke(app, ["capture", command])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize(
    "domain", ["https://example.com", "127.0.0.1", "api.local", "api.internal", "*.openai.com"]
)
def test_domain_validation_precedes_login(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    """Invalid domain arguments are rejected before authentication."""
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    result = CliRunner().invoke(
        app,
        ["capture", "--domain", domain],
        env={"TERM": "xterm-256color", "FORCE_COLOR": "1"},
    )
    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "exact DNS hostname" in output


def test_domain_limit_precedes_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI enforces a finite exact-host filter before starting capture."""
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    arguments = ["capture"]
    for index in range(33):
        arguments.extend(("--domain", f"provider{index}.example.com"))
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "1 to 32" in unstyle(result.output)


def test_help_never_attempts_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading command help cannot inspect or change network interception state."""

    def reject_setup() -> None:
        """Fail if help attempts to inspect or change system state."""
        raise AssertionError("help must not inspect or change system state")

    monkeypatch.setattr(capture_module, "_require_macos", reject_setup)
    result = CliRunner().invoke(
        app,
        ["capture", "--help"],
        env={"TERM": "xterm-256color", "FORCE_COLOR": "1"},
        color=True,
    )
    assert result.exit_code == 0
    assert "\x1b[" in result.output
    output = unstyle(result.output)
    assert "reset" not in output
    assert "--domain" in output
    assert "--verbose" in output
    assert "-v" in output
    assert "all apps" in output
    assert "COMMAND" not in output
    assert "What would you like to capture?" not in output


def test_capture_failure_reports_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime failures retain actionable backend guidance without an unavailable reset command."""

    def failed_capture(
        console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
    ) -> None:
        """Simulate a backend precondition failure without touching networking."""
        raise RuntimeError("Install Mitmproxy Redirector in /Applications and retry.")

    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module, "_capture", failed_capture)
    result = CliRunner().invoke(app, ["capture"])
    assert result.exit_code == 1, result.output
    output = unstyle(result.output)
    assert "Install Mitmproxy Redirector" in output
    assert "reset" not in output


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


@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_verbose_flag_reaches_foreground_runner(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    """Both verbosity spellings cross the CLI and exec boundary without starting capture."""
    launches: list[list[str]] = []
    monkeypatch.setattr(capture_module, "_require_macos", lambda: None)
    monkeypatch.setattr(capture_module.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(capture_module.os, "execv", lambda executable, args: launches.append(args))
    result = CliRunner().invoke(app, ["capture", flag])
    assert result.exit_code == 0, result.output
    assert len(launches) == 1
    assert launches[0].count("--verbose") == 1
    assert "exp.cli.capture.runner" in launches[0]


def test_help_on_minimum_python_does_not_import_capture_engine() -> None:
    """Capture help exposes verbosity without requiring the Python 3.13 capture dependencies."""
    script = """
import importlib.abc
import sys

class RejectCaptureEngine(importlib.abc.MetaPathFinder):
    '''Reject capture engine imports while leaving the ordinary CLI available.'''

    def find_spec(self, fullname, path=None, target=None):
        '''Make accidental engine imports fail even when dependencies are installed.'''
        if fullname == "exp.cli.capture.runner" or fullname.startswith("mitmproxy"):
            raise AssertionError(f"help imported capture engine: {fullname}")
        return None

sys.meta_path.insert(0, RejectCaptureEngine())
from click import unstyle
from typer.testing import CliRunner
from exp.cli.app import app
sys.version_info = (3, 12, 0)
result = CliRunner().invoke(app, ["capture", "--help"])
assert result.exit_code == 0, result.output
assert "--verbose" in unstyle(result.output)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr

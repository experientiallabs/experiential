"""Tests for the branded default gateway home screen."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from exp.cli.gateway import home
from exp.cli.gateway.serve import DEFAULT_MAX_ACTIVE_REQUESTS
from exp.cli.gateway.setup import InteractiveSetupResult
from exp.cli.shared.picker_test import ScriptedConsole


def test_home_screen_starts_with_brand_and_recommends_default_gateway(tmp_path: Path) -> None:
    """The first interactive screen presents the green EXP brand and happy path first."""
    console = ScriptedConsole("5\n")

    home.default_gateway(root=tmp_path, console=console)

    assert "exp" in console.output
    assert "Experiential gateway" in console.output
    assert "Default Gateway" in console.output
    assert "recommended happy path" in console.output
    assert "Run Gateway" in console.output
    assert "Setup Gateway" in console.output


def test_default_gateway_menu_starts_the_recommended_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The focused default row starts an ordinary gateway with interactive setup allowed."""
    calls: list[dict[str, object]] = []

    def start(**kwargs: object) -> None:
        """Capture the gateway launch selected by the home screen."""
        calls.append(kwargs)

    monkeypatch.setattr(home, "start_gateway", start)
    home.default_gateway(root=tmp_path, console=ScriptedConsole("1\n"))

    assert calls == [
        {
            "root": tmp_path,
            "port": 8000,
            "non_interactive": False,
            "graceful_timeout": 10.0,
            "engine": "auto",
            "max_active_requests": DEFAULT_MAX_ACTIVE_REQUESTS,
        }
    ]


def test_run_gateway_menu_uses_the_existing_configuration_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit run row never opens first-run setup prompts."""
    calls: list[dict[str, object]] = []

    def start(**kwargs: object) -> None:
        """Capture the gateway launch selected by the home screen."""
        calls.append(kwargs)

    monkeypatch.setattr(home, "start_gateway", start)
    home.default_gateway(root=tmp_path, console=ScriptedConsole("2\n"))

    assert calls[0]["root"] == tmp_path
    assert calls[0]["non_interactive"] is True
    assert calls[0]["engine"] == "auto"
    assert calls[0]["max_active_requests"] == DEFAULT_MAX_ACTIVE_REQUESTS


@pytest.mark.parametrize(
    ("policy", "ghost"),
    [("policy-a", False), (None, True)],
)
def test_project_only_gateway_options_validate_before_home_menu(
    tmp_path: Path,
    policy: str | None,
    ghost: bool,
) -> None:
    """Invalid project-only options fail before the interactive menu can ignore them."""
    with pytest.raises(typer.BadParameter, match="require --project"):
        home.default_gateway(
            root=tmp_path,
            policy=policy,
            ghost=ghost,
            console=ScriptedConsole("5\n"),
        )


def test_setup_gateway_returns_to_home_with_one_time_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup is an explicit menu action and leaves the operator at the home screen."""
    monkeypatch.setattr(
        "exp.cli.gateway.setup.interactive_gateway_setup",
        lambda _root, *, console: InteractiveSetupResult(
            identity_id="default",
            alias="default-gateway",
            raw_key="exp_vk_test",
        ),
    )
    console = ScriptedConsole("3\n5\n")

    home.default_gateway(root=tmp_path, console=console)

    assert "export EXP_GATEWAY_URL=http://127.0.0.1:8000/v1" in console.output
    assert "export EXP_GATEWAY_KEY=exp_vk_test" in console.output
    assert "Choose Default Gateway to start it." in console.output

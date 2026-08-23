"""Tests for the branded gateway home screen."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from exp.cli.gateway import home
from exp.cli.gateway.serve import DEFAULT_MAX_ACTIVE_REQUESTS
from exp.cli.gateway.setup import InteractiveSetupResult
from exp.cli.shared.picker_test import ScriptedConsole
from exp.runtime.gateway.auth import IssuedVirtualKey
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError


def test_home_screen_starts_with_brand_and_puts_run_gateway_first(tmp_path: Path) -> None:
    """The first interactive screen puts the setup-aware run path first."""
    console = ScriptedConsole("3\n")

    home.default_gateway(root=tmp_path, console=console)

    assert "exp" in console.output
    assert "Experiential gateway" in console.output
    assert "Run Gateway" in console.output
    assert "Setup Gateway" in console.output
    assert "Gateway Status" not in console.output
    assert "Default Gateway" not in console.output
    assert console.output.index("Run Gateway") < console.output.index("Setup Gateway")


def test_run_gateway_menu_starts_with_setup_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first menu row starts an ordinary gateway with setup allowed."""
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
            console=ScriptedConsole("3\n"),
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
            guardrails="Off",
        ),
    )
    console = ScriptedConsole("2\n3\n")

    home.default_gateway(root=tmp_path, console=console)

    assert "export EXP_GATEWAY_URL=http://127.0.0.1:8000/v1" in console.output
    assert "export EXP_GATEWAY_KEY=exp_vk_test" in console.output
    assert "Guardrails: Off" in console.output
    assert "Choose Run Gateway to start it." in console.output


def test_setup_gateway_warns_and_reconfigures_an_initialized_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An initialized gateway requires confirmation before the setup wizard replaces revisions."""
    GatewayManagement(tmp_path).initialize()
    calls: list[dict[str, object]] = []

    def setup_gateway(_root: Path, **kwargs: object) -> InteractiveSetupResult:
        """Capture the explicitly confirmed reconfiguration request."""
        calls.append(kwargs)
        return InteractiveSetupResult(
            identity_id="default",
            alias="default-gateway",
            raw_key="exp_vk_reconfigured",
            guardrails="Off",
        )

    monkeypatch.setattr("exp.cli.gateway.setup.interactive_gateway_setup", setup_gateway)
    console = ScriptedConsole("2\ny\n3\n")

    home.default_gateway(root=tmp_path, console=console)

    assert calls == [{"console": console, "allow_reconfigure": True}]
    assert "Gateway already configured." in console.output
    assert "Existing identities" in console.output
    assert "history remain." in console.output
    assert "export EXP_GATEWAY_KEY=exp_vk_reconfigured" in console.output
    assert "Gateway reconfigured" in console.output


def test_setup_gateway_preserves_a_key_when_reconfiguration_outcome_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous activation surfaces the non-reconstructable key for recovery."""
    GatewayManagement(tmp_path).initialize()
    issued = IssuedVirtualKey(
        key_id="key-unknown",
        organization_id="local",
        identity_id="default",
        prefix="exp_vk_test",
        raw_key="exp_vk_test_secret",
        expires_at=None,
        created_at=datetime.now(UTC),
    )

    def setup_gateway(_root: Path, **_: object) -> InteractiveSetupResult:
        """Raise the same typed uncertainty produced by alias activation."""
        raise AliasActivationOutcomeUnknownError(
            alias_id="default-gateway",
            revision_id="revision-unknown",
            issued=issued,
        )

    monkeypatch.setattr("exp.cli.gateway.setup.interactive_gateway_setup", setup_gateway)
    console = ScriptedConsole("2\ny\n3\n")

    home.default_gateway(root=tmp_path, console=console)

    assert "outcome is unknown" in console.output
    assert "Preserve this one-time gateway key: exp_vk_test_secret" in console.output
    assert "Gateway reconfigured" not in console.output


def test_setup_gateway_declines_initialized_gateway_reconfiguration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconfiguration warning defaults to a safe no-op when declined."""
    GatewayManagement(tmp_path).initialize()
    called = False

    def setup_gateway(**_: object) -> InteractiveSetupResult:
        """Fail if setup runs after the operator declines the warning."""
        nonlocal called
        called = True
        raise AssertionError("setup should not run after a declined reconfiguration")

    monkeypatch.setattr("exp.cli.gateway.setup.interactive_gateway_setup", setup_gateway)
    console = ScriptedConsole("2\n\n3\n")

    home.default_gateway(root=tmp_path, console=console)

    assert not called
    assert "Gateway reconfiguration cancelled." in console.output

"""First-run gateway provider selector and connection metadata tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from exp.cli.gateway import setup
from exp.cli.providers import model_picker, provider_picker
from exp.cli.shared.picker import PickerKey
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.config import load_settings
from exp.common.models import ConnectionConfig, ModelCapabilities, PricingSource, ProviderConnection


def _prepared_gateway_models() -> tuple[
    tuple[provider_picker.PreparedEndpoint, ...],
    tuple[provider_picker.AvailableModel, ...],
]:
    """Build prepared endpoints and one discovered completion model for gateway tests."""
    connections = (
        ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),
        ProviderConnection(name="anthropic", provider="anthropic", api_key_env="ANTHROPIC_API_KEY"),
    )
    endpoints = tuple(
        provider_picker.PreparedEndpoint(connection=connection, api_key="secret", configured=False)
        for connection in connections
    )
    model = provider_picker.AvailableModel(
        alias="gpt-5-6-luna",
        connection="openai",
        provider="openai",
        model="gpt-5.6-luna",
        capabilities=ModelCapabilities(supports_completions=True, reasoning_effort="medium"),
        pricing_source=PricingSource.EXP_CATALOG,
        configured=False,
    )
    return endpoints, (model,)


def test_gateway_setup_uses_the_shared_provider_setup_seams() -> None:
    """Gateway first-run setup does not maintain a second provider picker implementation."""
    assert setup.select_providers is provider_picker.select_providers
    assert setup.prepare_providers is provider_picker.prepare_providers
    assert setup.select_gateway_model is model_picker.select_gateway_model


@pytest.mark.parametrize(
    ("answer", "provider"),
    (
        ("1", "openai"),
        ("2", "anthropic"),
        ("5", "openai-compatible"),
        ("6", "azure"),
        ("7", "bedrock"),
    ),
)
def test_gateway_provider_selector_exposes_primary_and_legacy_providers(
    answer: str,
    provider: str,
) -> None:
    """The line fallback presents the four primary providers and the legacy compatible path."""
    console = ScriptedConsole(f"{answer}\n\n")

    selected = provider_picker.select_providers(
        provider_picker.SetupSession(), console=console, environment={}
    )

    assert selected == ((provider,), provider in {"azure", "bedrock"})
    for expected in ("openai", "anthropic", "azure", "bedrock", "openai-compatible"):
        assert expected in console.output


def test_gateway_provider_selector_accepts_multiple_providers() -> None:
    """The gateway uses the builder's multi-select semantics instead of forcing one provider."""
    console = ScriptedConsole("1,2,6,7\n\n")

    selected = provider_picker.select_providers(
        provider_picker.SetupSession(), console=console, environment={}
    )

    assert selected == (("openai", "anthropic", "azure", "bedrock"), True)


def test_gateway_provider_selector_uses_the_builder_keyboard_picker() -> None:
    """The gateway selector accepts the same Up, Down, and Enter interaction as the builder."""
    keys = iter(
        (
            *(PickerKey.DOWN for _ in range(5)),
            PickerKey.ENTER,
            PickerKey.DOWN,
            PickerKey.DOWN,
            PickerKey.ENTER,
        )
    )
    console = ScriptedConsole("")

    selected = provider_picker.select_providers(
        provider_picker.SetupSession(),
        console=console,
        environment={},
        read_key=lambda: next(keys),
    )

    assert selected == (("azure",), True)
    assert "Providers" in console.output
    assert "openai" in console.output
    assert "bedrock" in console.output


def test_gateway_setup_persists_selected_connections_and_one_initial_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run setup accepts all displayed defaults with one empty line."""
    endpoints, models = _prepared_gateway_models()
    monkeypatch.setattr(
        setup,
        "select_providers",
        lambda *_args, **_kwargs: (("openai", "anthropic"), False),
    )
    monkeypatch.setattr(setup, "prepare_providers", lambda *_args, **_kwargs: (endpoints, models))
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(models[0], "medium"),
    )
    console = ScriptedConsole("\n")

    result = setup.interactive_gateway_setup(tmp_path, console=console)

    assert result.alias == "gpt-5-6-luna"
    assert "Press Enter to accept all defaults" in console.output
    assert "Alias" in console.output
    assert "Identity ID" in console.output
    assert "Budget" in console.output
    assert "$50.00" in console.output
    assert "Exact model ID" not in console.output
    assert "Planned local mutations" not in console.output
    assert "Create this gateway configuration?" not in console.output
    assert "Gateway configured" in console.output
    manager = setup.GatewayManagement(tmp_path)
    assert {item.connection_id for item in manager.provider_connections()} == {
        "openai",
        "anthropic",
    }
    assert {item.alias_id for item in manager.aliases()} == {"gpt-5-6-luna"}
    assert load_settings(tmp_path).commands.maximum_cost_usd == 50.0


def test_gateway_setup_can_edit_the_displayed_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run setup keeps the defaults visible while allowing every value to be edited."""
    endpoints, models = _prepared_gateway_models()
    monkeypatch.setattr(
        setup,
        "select_providers",
        lambda *_args, **_kwargs: (("openai",), False),
    )
    monkeypatch.setattr(setup, "prepare_providers", lambda *_args, **_kwargs: (endpoints, models))
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(models[0], "medium"),
    )
    console = ScriptedConsole("edit\ncustom-alias\noperator\n75\n")

    result = setup.interactive_gateway_setup(tmp_path, console=console)

    assert result.alias == "custom-alias"
    assert result.identity_id == "operator"
    assert "Alias" in console.output
    assert "Identity ID" in console.output
    assert "Budget" in console.output
    assert "Exact model ID" not in console.output
    assert setup.resolve_command_budget_usd(tmp_path, None) == 75.0


def test_gateway_setup_writes_budget_before_gateway_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed budget write leaves first-run setup eligible for a retry."""
    endpoints, models = _prepared_gateway_models()
    monkeypatch.setattr(
        setup,
        "select_providers",
        lambda *_args, **_kwargs: (("openai",), False),
    )
    monkeypatch.setattr(setup, "prepare_providers", lambda *_args, **_kwargs: (endpoints, models))
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(models[0], "medium"),
    )

    def _fail_budget(_maximum_cost_usd: float, root: Path) -> None:
        """Fail before gateway initialization, as a settings write can in production."""
        assert root == tmp_path
        assert not setup.GatewayManagement(root).initialized
        raise OSError("settings unavailable")

    monkeypatch.setattr(setup, "set_maximum_command_cost_usd", _fail_budget)

    with pytest.raises(OSError, match="settings unavailable"):
        setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    assert not setup.GatewayManagement(tmp_path).initialized


def test_gateway_setup_aborts_when_connection_prompt_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run setup converts shared provider prompt cancellation into a clean abort."""
    monkeypatch.setattr(
        setup,
        "select_providers",
        lambda *_args, **_kwargs: (("openai",), False),
    )

    def _cancel(*_args: object, **_kwargs: object) -> None:
        """Raise the shared picker cancellation sentinel."""
        raise provider_picker.SetupCancelled

    monkeypatch.setattr(setup, "prepare_providers", _cancel)

    with pytest.raises(typer.Abort):
        setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole(""))


def test_gateway_setup_aborts_when_provider_selector_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run setup converts selector EOF into the same clean CLI abort."""

    def _cancel(*_args: object, **_kwargs: object) -> None:
        """Raise the terminal EOF surfaced by the line-input fallback."""
        raise EOFError

    monkeypatch.setattr(setup, "select_providers", _cancel)

    with pytest.raises(typer.Abort):
        setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole(""))


@pytest.mark.parametrize(
    ("provider", "credential_env"),
    (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")),
)
def test_native_provider_connection_uses_its_canonical_credential_env(
    provider: str,
    credential_env: str,
) -> None:
    """Native providers use the same canonical credential references as builder setup."""
    connection = provider_picker.collect_provider_connection(provider, console=ScriptedConsole(""))

    assert connection == ConnectionConfig(provider=provider, api_key_env=credential_env)


def test_azure_provider_connection_collects_required_endpoint_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure setup collects the resource endpoint and defaults its API version to v1."""
    answers = iter(("https://resource.openai.azure.com", "v1"))

    def _prompt(_text: str, **_kwargs: object) -> str:
        """Return the next scripted Azure connection field."""
        return next(answers)

    monkeypatch.setattr(provider_picker, "ask_text", _prompt)

    connection = provider_picker.collect_provider_connection("azure", console=ScriptedConsole(""))

    assert connection == ConnectionConfig(
        provider="azure",
        base_url="https://resource.openai.azure.com",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    )


def test_openai_compatible_provider_connection_collects_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible setup retains its endpoint and credential-variable override."""
    answers = iter(("https://gateway.example.test/v1", "COMPATIBLE_API_KEY"))

    monkeypatch.setattr(provider_picker, "ask_text", lambda _text, **_kwargs: next(answers))

    connection = provider_picker.collect_provider_connection(
        "openai-compatible", console=ScriptedConsole("")
    )

    assert connection == ConnectionConfig(
        provider="openai-compatible",
        base_url="https://gateway.example.test/v1",
        api_key_env="COMPATIBLE_API_KEY",
    )


def test_openai_compatible_provider_connection_defaults_whitespace_credential_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only compatible credential input falls back to the canonical variable."""
    answers = iter(("https://gateway.example.test/v1", "   "))

    monkeypatch.setattr(provider_picker, "ask_text", lambda _text, **_kwargs: next(answers))

    connection = provider_picker.collect_provider_connection(
        "openai-compatible", console=ScriptedConsole("")
    )

    assert connection == ConnectionConfig(
        provider="openai-compatible",
        base_url="https://gateway.example.test/v1",
        api_key_env="OPENAI_COMPATIBLE_API_KEY",
    )


def test_bedrock_provider_connection_uses_the_aws_credential_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock setup records only an optional region and never invents an API-key variable."""
    monkeypatch.setattr(provider_picker, "ask_text", lambda _text, **_kwargs: "us-east-1")

    connection = provider_picker.collect_provider_connection("bedrock", console=ScriptedConsole(""))

    assert connection == ConnectionConfig(provider="bedrock", region="us-east-1")

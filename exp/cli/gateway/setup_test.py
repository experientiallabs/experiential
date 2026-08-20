"""First-run gateway provider selector and connection metadata tests."""

from __future__ import annotations

import pytest

from exp.cli.gateway import setup
from exp.cli.providers import provider_picker
from exp.cli.shared.picker import PickerKey
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.models import ConnectionConfig


def test_gateway_setup_uses_the_shared_provider_setup_seams() -> None:
    """Gateway first-run setup does not maintain a second provider picker implementation."""
    assert setup.select_single_provider is provider_picker.select_single_provider
    assert setup.collect_provider_connection is provider_picker.collect_provider_connection


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

    selected = provider_picker.select_single_provider(console=console, environment={})

    assert selected == provider
    for expected in ("openai", "anthropic", "azure", "bedrock", "openai-compatible"):
        assert expected in console.output


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

    selected = provider_picker.select_single_provider(
        console=console,
        environment={},
        read_key=lambda: next(keys),
    )

    assert selected == "azure"
    assert "Providers" in console.output
    assert "openai" in console.output
    assert "bedrock" in console.output


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

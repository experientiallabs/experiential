"""First-run gateway provider selector and connection metadata tests."""

from __future__ import annotations

import pytest

from exp.cli.gateway import setup
from exp.cli.shared.picker import PickerKey
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.models import ConnectionConfig


@pytest.mark.parametrize(
    ("answer", "provider"),
    (
        ("1", "openai"),
        ("2", "anthropic"),
        ("3", "azure"),
        ("4", "bedrock"),
        ("5", "openai-compatible"),
    ),
)
def test_gateway_provider_selector_exposes_primary_and_legacy_providers(
    answer: str,
    provider: str,
) -> None:
    """The line fallback presents the four primary providers and the legacy compatible path."""
    console = ScriptedConsole(f"{answer}\n")

    selected = setup.select_gateway_provider(console=console)

    assert selected == provider
    for expected in ("openai", "anthropic", "azure", "bedrock", "openai-compatible"):
        assert expected in console.output


def test_gateway_provider_selector_uses_the_builder_keyboard_picker() -> None:
    """The gateway selector accepts the same Up, Down, and Enter interaction as the builder."""
    keys = iter((PickerKey.DOWN, PickerKey.DOWN, PickerKey.ENTER))
    console = ScriptedConsole("")

    selected = setup.select_gateway_provider(console=console, read_key=lambda: next(keys))

    assert selected == "azure"
    assert "Provider" in console.output
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
    connection = setup._collect_provider_connection(provider)  # noqa: SLF001

    assert connection == ConnectionConfig(provider=provider, api_key_env=credential_env)


def test_azure_provider_connection_collects_required_endpoint_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure setup collects the resource endpoint and defaults its API version to v1."""
    answers = iter(("https://resource.openai.azure.com", "v1"))

    def _prompt(_text: str, **_kwargs: object) -> str:
        """Return the next scripted Azure connection field."""
        return next(answers)

    monkeypatch.setattr(setup.typer, "prompt", _prompt)

    connection = setup._collect_provider_connection("azure")  # noqa: SLF001

    assert connection == ConnectionConfig(
        provider="azure",
        base_url="https://resource.openai.azure.com",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    )


def test_openai_compatible_provider_connection_collects_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible setup retains its explicit endpoint and canonical credential reference."""
    monkeypatch.setattr(
        setup.typer,
        "prompt",
        lambda _text, **_kwargs: "https://gateway.example.test/v1",
    )

    connection = setup._collect_provider_connection("openai-compatible")  # noqa: SLF001

    assert connection == ConnectionConfig(
        provider="openai-compatible",
        base_url="https://gateway.example.test/v1",
        api_key_env="OPENAI_COMPATIBLE_API_KEY",
    )


def test_bedrock_provider_connection_uses_the_aws_credential_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock setup records only an optional region and never invents an API-key variable."""
    monkeypatch.setattr(
        setup.typer,
        "prompt",
        lambda _text, **_kwargs: "us-east-1",
    )

    connection = setup._collect_provider_connection("bedrock")  # noqa: SLF001

    assert connection == ConnectionConfig(provider="bedrock", region="us-east-1")

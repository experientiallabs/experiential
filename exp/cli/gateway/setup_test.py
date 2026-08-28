"""First-run gateway provider selector and connection metadata tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

import exp.runtime.gateway.sqlite.setup_authority as setup_authority
import exp.runtime.gateway.sqlite.store as gateway_store
from exp.cli.gateway import setup
from exp.cli.gateway.guardrail_setup import (
    GUARDRAILS_STANDARD,
    guardrail_config_path,
    inspect_setup_guardrails,
)
from exp.cli.providers import model_picker, provider_picker
from exp.cli.providers.experiential_cloud import SETUP_PICKER_LABEL
from exp.cli.shared.picker import PickerKey
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.config import load_settings
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    PricingSource,
    ProviderConnection,
    load_model_catalog,
)
from exp.runtime.gateway.auth import IssuedVirtualKey
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError


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
        capabilities=ModelCapabilities(
            supports_completions=True,
            supports_tools=True,
            supports_reasoning=True,
            reasoning_effort="medium",
        ),
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
        ("1", "experiential-cloud"),
        ("2", "openai"),
        ("3", "anthropic"),
        ("6", "openai-compatible"),
        ("7", "azure"),
        ("8", "bedrock"),
    ),
)
def test_gateway_provider_selector_exposes_primary_and_legacy_providers(
    answer: str,
    provider: str,
) -> None:
    """The line fallback presents the four primary providers and the legacy compatible path."""
    console = ScriptedConsole(f"{answer}\n\n")

    selected = provider_picker.select_providers(
        provider_picker.SetupSession(),
        console=console,
        environment={},
    )

    assert selected == ((provider,), provider in {"azure", "bedrock"})
    for expected in (
        "Experiential Cloud",
        "openai",
        "anthropic",
        "azure",
        "bedrock",
        "openai-compatible",
    ):
        assert expected in console.output
    assert SETUP_PICKER_LABEL in console.output


def test_gateway_provider_selector_accepts_multiple_providers() -> None:
    """The gateway uses the builder's multi-select semantics instead of forcing one provider."""
    console = ScriptedConsole("1,2,6,7\n\n")

    selected = provider_picker.select_providers(
        provider_picker.SetupSession(),
        console=console,
        environment={},
    )

    assert selected == (("experiential-cloud", "openai", "openai-compatible", "azure"), True)


def test_gateway_provider_selector_uses_the_builder_keyboard_picker() -> None:
    """The gateway selector accepts the same Up, Down, and Enter interaction as the builder."""
    keys = iter(
        (
            *(PickerKey.DOWN for _ in range(5)),
            PickerKey.ENTER,
            *(PickerKey.DOWN for _ in range(3)),
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

    assert selected == (("openai-compatible",), False)
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
    assert result.guardrails == "Off"
    assert "Press Enter to accept all defaults" in console.output
    assert "Alias" in console.output
    assert "Identity ID" in console.output
    assert "Budget" in console.output
    assert "Guardrails" in console.output
    assert "$50.00" in console.output
    assert not (tmp_path / "gateway" / "guardrails.json").exists()
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
    authored = load_model_catalog(tmp_path / "models.toml").models["gpt-5-6-luna"]
    assert authored.gateway is not None
    assert authored.gateway.capabilities.supports_streaming_tool_arguments
    assert load_settings(tmp_path).commands.maximum_cost_usd == 50.0


def test_gateway_setup_marks_experiential_deployment_as_host_managed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting the hosted provider preserves platform-owned billing in the local catalog."""
    connection = ProviderConnection(
        name="experiential-cloud",
        provider="openai-compatible",
        api_key_env="EXPLABS_API_KEY",
        base_url="https://api.experientiallabs.ai/v1",
    )
    endpoint = provider_picker.PreparedEndpoint(
        connection=connection,
        api_key="secret",
        configured=True,
    )
    model = provider_picker.AvailableModel(
        alias="exp-chat",
        connection=connection.name,
        provider=connection.provider,
        model="exp-chat",
        capabilities=ModelCapabilities(supports_completions=True),
        pricing_source=PricingSource.UNKNOWN,
        configured=True,
    )
    monkeypatch.setattr(
        setup,
        "select_providers",
        lambda *_args, **_kwargs: (("experiential-cloud",), False),
    )
    monkeypatch.setattr(
        setup,
        "prepare_providers",
        lambda *_args, **_kwargs: ((endpoint,), (model,)),
    )
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(model, None),
    )

    setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))

    catalog = load_model_catalog(tmp_path / "models.toml")
    assert catalog.models["exp-chat"].billing_source is BillingSource.HOST_MANAGED


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
    console = ScriptedConsole("edit\ncustom-alias\noperator\n75\noff\n")

    result = setup.interactive_gateway_setup(tmp_path, console=console)

    assert result.alias == "custom-alias"
    assert result.identity_id == "operator"
    assert result.guardrails == "Off"
    assert not (tmp_path / "gateway" / "guardrails.json").exists()
    assert "Alias" in console.output
    assert "Identity ID" in console.output
    assert "Budget" in console.output
    assert "Exact model ID" not in console.output
    assert setup.resolve_command_budget_usd(tmp_path, None) == 75.0


def test_gateway_setup_requires_explicit_reconfigure_opt_in(
    tmp_path: Path,
) -> None:
    """An initialized gateway remains protected unless the caller supplies explicit consent."""
    setup.GatewayManagement(tmp_path).initialize()

    with pytest.raises(ValueError, match="requires an uninitialized gateway"):
        setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole(""))


def test_gateway_setup_reconfigures_provider_alias_and_existing_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed reconfiguration replaces serving revisions while retaining gateway authority."""
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
    first = setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    manager = setup.GatewayManagement(tmp_path)
    first_revision = manager.aliases()[0].revision_id

    updated_connection = ProviderConnection(
        name="openai",
        provider="openai",
        api_key_env="UPDATED_OPENAI_API_KEY",
    )
    updated_endpoint = provider_picker.PreparedEndpoint(
        connection=updated_connection,
        api_key="secret",
        configured=False,
    )
    updated_model = provider_picker.AvailableModel(
        alias=models[0].alias,
        connection="openai",
        provider="openai",
        model="gpt-5-6-luna-v2",
        capabilities=models[0].capabilities,
        pricing_source=models[0].pricing_source,
        configured=False,
    )
    monkeypatch.setattr(
        setup,
        "prepare_providers",
        lambda *_args, **_kwargs: ((updated_endpoint,), (updated_model,)),
    )
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(updated_model, "medium"),
    )

    result = setup.interactive_gateway_setup(
        tmp_path,
        console=ScriptedConsole(f"edit\n{first.alias}\ndefault\n75\noff\n"),
        allow_reconfigure=True,
    )

    assert result.alias == first.alias
    assert result.identity_id == first.identity_id
    connections = {item.connection_id: item for item in manager.provider_connections()}
    assert connections["openai"].config.api_key_env == "UPDATED_OPENAI_API_KEY"
    assert manager.aliases()[0].revision_id != first_revision
    assert manager.status().active_identities == 1
    assert manager.status().active_keys == 2
    assert manager.status().grants == 1
    assert setup.resolve_command_budget_usd(tmp_path, None) == 75.0


def test_gateway_setup_rolls_back_catalog_and_provider_when_alias_activation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed reconfiguration leaves the prior catalog and SQLite authority active."""
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
    first = setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    manager = setup.GatewayManagement(tmp_path)
    catalog_before = (tmp_path / "models.toml").read_bytes()
    providers_before = manager.provider_connections()
    aliases_before = manager.aliases()

    updated_connection = ProviderConnection(
        name="openai",
        provider="openai",
        api_key_env="UPDATED_OPENAI_API_KEY",
    )
    updated_endpoint = provider_picker.PreparedEndpoint(
        connection=updated_connection,
        api_key="secret",
        configured=False,
    )
    updated_model = provider_picker.AvailableModel(
        alias=models[0].alias,
        connection="openai",
        provider="openai",
        model="gpt-5-6-luna-v2",
        capabilities=models[0].capabilities,
        pricing_source=models[0].pricing_source,
        configured=False,
    )
    monkeypatch.setattr(
        setup,
        "prepare_providers",
        lambda *_args, **_kwargs: ((updated_endpoint,), (updated_model,)),
    )
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(updated_model, "medium"),
    )

    def fail_alias_activation(*_args: object, **_kwargs: object) -> None:
        """Fail after provider revisions have been staged in the shared transaction."""
        raise RuntimeError("alias activation failed")

    monkeypatch.setattr(
        gateway_store,
        "activate_alias_revision_in_transaction",
        fail_alias_activation,
    )

    with pytest.raises(RuntimeError, match="alias activation failed"):
        setup.interactive_gateway_setup(
            tmp_path,
            console=ScriptedConsole(f"edit\n{first.alias}\ndefault\n50\noff\n"),
            allow_reconfigure=True,
        )

    assert (tmp_path / "models.toml").read_bytes() == catalog_before
    assert manager.provider_connections() == providers_before
    assert manager.aliases() == aliases_before


def test_gateway_setup_rolls_back_all_authority_when_key_issue_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-alias credential failure rolls back every serving mutation atomically."""
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
    first = setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    manager = setup.GatewayManagement(tmp_path)
    before = (
        (tmp_path / "settings.toml").read_bytes(),
        (tmp_path / "models.toml").read_bytes(),
        manager.provider_connections(),
        manager.aliases(),
        manager.identities(),
        manager.grants(),
        manager.keys(),
    )

    updated_model = provider_picker.AvailableModel(
        alias=models[0].alias,
        connection=models[0].connection,
        provider=models[0].provider,
        model="gpt-5-6-luna-v2",
        capabilities=models[0].capabilities,
        pricing_source=models[0].pricing_source,
        configured=False,
    )
    updated_endpoint = provider_picker.PreparedEndpoint(
        connection=endpoints[0].connection,
        api_key="secret",
        configured=False,
    )
    monkeypatch.setattr(
        setup,
        "prepare_providers",
        lambda *_args, **_kwargs: ((updated_endpoint,), (updated_model,)),
    )
    monkeypatch.setattr(
        setup,
        "select_gateway_model",
        lambda *_args, **_kwargs: model_picker.GatewayModelSelection(updated_model, "medium"),
    )

    def fail_key(*_args: object, **_kwargs: object) -> None:
        """Fail after alias activation and grant staging inside the shared transaction."""
        raise RuntimeError("key issue failed")

    monkeypatch.setattr(setup_authority, "_persist_key", fail_key)

    with pytest.raises(RuntimeError, match="key issue failed"):
        setup.interactive_gateway_setup(
            tmp_path,
            console=ScriptedConsole(f"edit\n{first.alias}\ndefault\n75\noff\n"),
            allow_reconfigure=True,
        )

    after = (
        (tmp_path / "settings.toml").read_bytes(),
        (tmp_path / "models.toml").read_bytes(),
        manager.provider_connections(),
        manager.aliases(),
        manager.identities(),
        manager.grants(),
        manager.keys(),
    )
    assert after == before


def test_gateway_setup_keeps_selected_budget_when_activation_outcome_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown SQLite outcome keeps settings aligned with potentially committed authority."""
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
    first = setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    issued = IssuedVirtualKey(
        key_id="key-unknown",
        organization_id="local",
        identity_id=first.identity_id,
        prefix="exp_vk_test",
        raw_key="exp_vk_test_secret",
        expires_at=None,
        created_at=datetime.now(UTC),
    )

    def unknown_activation(*_args: object, **_kwargs: object) -> tuple[bool, IssuedVirtualKey]:
        """Simulate an activation whose COMMIT acknowledgement is indeterminate."""
        raise AliasActivationOutcomeUnknownError(
            alias_id=first.alias,
            revision_id="revision-unknown",
            issued=issued,
        )

    monkeypatch.setattr(
        setup.GatewayManagement,
        "configure_direct_alias_with_identity",
        unknown_activation,
    )

    with pytest.raises(AliasActivationOutcomeUnknownError, match="operation_outcome_unknown"):
        setup.interactive_gateway_setup(
            tmp_path,
            console=ScriptedConsole(f"edit\n{first.alias}\ndefault\n75\noff\n"),
            allow_reconfigure=True,
        )

    assert setup.resolve_command_budget_usd(tmp_path, None) == 75.0


def _stub_gateway_pickers(monkeypatch: pytest.MonkeyPatch) -> provider_picker.AvailableModel:
    """Install the shared first-run picker seams used by setup tests.

    Args:
        monkeypatch: Active pytest monkeypatch.

    Returns:
        The selected completion model so callers can reuse its alias.
    """
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
    return models[0]


def test_gateway_setup_can_opt_the_selected_identity_into_standard_guardrails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edit is the only path that authors the standard pack for the selected identity."""
    model = _stub_gateway_pickers(monkeypatch)
    console = ScriptedConsole(
        "edit\n"
        f"{model.alias}\n"
        "default\n"
        "50\n"
        "on\n"
        "https://classifier.example.invalid/v1/inspect\n"
        "CLASSIFIER_BEARER\n"
    )

    result = setup.interactive_gateway_setup(tmp_path, console=console)

    assert result.guardrails == GUARDRAILS_STANDARD
    inspection = inspect_setup_guardrails(tmp_path, "local", "default")
    assert inspection.display == GUARDRAILS_STANDARD
    assert inspection.classifier_url == "https://classifier.example.invalid/v1/inspect"
    assert inspection.bearer_env == "CLASSIFIER_BEARER"
    payload = guardrail_config_path(tmp_path).read_text(encoding="utf-8")
    assert "CLASSIFIER_BEARER" in payload
    assert "sk-" not in payload


def test_gateway_setup_reconfigure_enter_preserves_existing_standard_guardrails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconfiguration Enter keeps an existing setup-owned pack and does not rewrite it."""
    model = _stub_gateway_pickers(monkeypatch)
    first = setup.interactive_gateway_setup(
        tmp_path,
        console=ScriptedConsole(
            "edit\n"
            f"{model.alias}\n"
            "default\n"
            "50\n"
            "on\n"
            "https://classifier.example.invalid/v1/inspect\n"
            "CLASSIFIER_BEARER\n"
        ),
    )
    before = guardrail_config_path(tmp_path).read_bytes()

    result = setup.interactive_gateway_setup(
        tmp_path,
        console=ScriptedConsole("\n"),
        allow_reconfigure=True,
    )

    assert result.identity_id == first.identity_id
    assert result.guardrails == GUARDRAILS_STANDARD
    assert guardrail_config_path(tmp_path).read_bytes() == before


def test_gateway_setup_restores_guardrails_when_alias_activation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proven catalog failure restores the previous guardrail file."""
    model = _stub_gateway_pickers(monkeypatch)
    setup.interactive_gateway_setup(
        tmp_path,
        console=ScriptedConsole(
            "edit\n"
            f"{model.alias}\n"
            "default\n"
            "50\n"
            "on\n"
            "https://classifier.example.invalid/v1/inspect\n"
            "CLASSIFIER_BEARER\n"
        ),
    )
    before = guardrail_config_path(tmp_path).read_bytes()

    def fail_alias_activation(*_args: object, **_kwargs: object) -> None:
        """Fail after the selected guardrail file has already been published."""
        raise RuntimeError("alias activation failed")

    monkeypatch.setattr(
        gateway_store,
        "activate_alias_revision_in_transaction",
        fail_alias_activation,
    )

    with pytest.raises(RuntimeError, match="alias activation failed"):
        setup.interactive_gateway_setup(
            tmp_path,
            console=ScriptedConsole(
                "edit\n"
                f"{model.alias}\n"
                "default\n"
                "50\n"
                "on\n"
                "https://classifier.example.invalid/v2/inspect\n"
                "CLASSIFIER_BEARER\n"
            ),
            allow_reconfigure=True,
        )

    assert guardrail_config_path(tmp_path).read_bytes() == before
    assert inspect_setup_guardrails(tmp_path, "local", "default").classifier_url == (
        "https://classifier.example.invalid/v1/inspect"
    )


def test_gateway_setup_keeps_selected_guardrails_when_activation_outcome_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown SQLite outcome keeps the selected guardrail file with the budget."""
    _stub_gateway_pickers(monkeypatch)
    first = setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    issued = IssuedVirtualKey(
        key_id="key-unknown",
        organization_id="local",
        identity_id=first.identity_id,
        prefix="exp_vk_test",
        raw_key="exp_vk_test_secret",
        expires_at=None,
        created_at=datetime.now(UTC),
    )

    def unknown_activation(*_args: object, **_kwargs: object) -> tuple[bool, IssuedVirtualKey]:
        """Simulate an activation whose COMMIT acknowledgement is indeterminate."""
        raise AliasActivationOutcomeUnknownError(
            alias_id=first.alias,
            revision_id="revision-unknown",
            issued=issued,
        )

    monkeypatch.setattr(
        setup.GatewayManagement,
        "configure_direct_alias_with_identity",
        unknown_activation,
    )

    with pytest.raises(AliasActivationOutcomeUnknownError, match="operation_outcome_unknown"):
        setup.interactive_gateway_setup(
            tmp_path,
            console=ScriptedConsole(
                "edit\n"
                f"{first.alias}\n"
                "default\n"
                "75\n"
                "on\n"
                "https://classifier.example.invalid/v1/inspect\n"
                "CLASSIFIER_BEARER\n"
            ),
            allow_reconfigure=True,
        )

    assert setup.resolve_command_budget_usd(tmp_path, None) == 75.0
    assert inspect_setup_guardrails(tmp_path, "local", "default").display == GUARDRAILS_STANDARD


def test_gateway_setup_fails_closed_on_malformed_guardrail_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present but invalid guardrail file blocks setup instead of reporting Off."""
    _stub_gateway_pickers(monkeypatch)
    path = tmp_path / "gateway" / "guardrails.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        setup.interactive_gateway_setup(tmp_path, console=ScriptedConsole("\n"))
    assert path.read_text(encoding="utf-8") == "{"
    assert not setup.GatewayManagement(tmp_path).initialized


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
    answers = iter(("https://resource.openai.azure.com", "v1", "openai_deployments"))

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
        azure_api_surface="openai_deployments",
    )


def test_openai_compatible_provider_connection_collects_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible setup collects the endpoint and does not ask for an env name."""
    answers = iter(("https://gateway.example.test/v1",))

    monkeypatch.setattr(provider_picker, "ask_text", lambda _text, **_kwargs: next(answers))

    connection = provider_picker.collect_provider_connection(
        "openai-compatible", console=ScriptedConsole("")
    )

    assert connection is not None
    assert connection == ConnectionConfig(
        provider="openai-compatible",
        base_url="https://gateway.example.test/v1",
    )
    assert connection.api_key_env is None


def test_experiential_cloud_connection_uses_the_hosted_platform_gateway() -> None:
    """Experiential Cloud does not collect a local base URL or credential name."""
    connection = provider_picker.collect_provider_connection(
        "experiential-cloud",
        console=ScriptedConsole(""),
        environment={},
    )

    assert connection == ConnectionConfig(
        provider="openai-compatible",
        base_url="https://api.experientiallabs.ai/v1",
        api_key_env="EXPLABS_API_KEY",
    )


def test_bedrock_provider_connection_uses_the_aws_credential_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock setup records only an optional region and never invents an API-key variable."""
    monkeypatch.setattr(provider_picker, "ask_text", lambda _text, **_kwargs: "us-east-1")

    connection = provider_picker.collect_provider_connection("bedrock", console=ScriptedConsole(""))

    assert connection == ConnectionConfig(provider="bedrock", region="us-east-1")

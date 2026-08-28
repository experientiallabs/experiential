"""Provider selection, credential, and discovery screen tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from exp.cli.providers import provider_picker
from exp.cli.providers.experiential_cloud import (
    HOSTED_GATEWAY_API_KEY_ENV,
    HOSTED_GATEWAY_DEFAULT_BASE_URL,
    HOSTED_GATEWAY_URL_ENV,
)
from exp.cli.providers.provider_picker import (
    SetupCancelled,
    SetupSession,
    explicit_provider_selection,
    prepare_providers,
    resolve_setup_providers,
    select_providers,
)
from exp.cli.shared.picker import PickerKey
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, default_auth_path
from exp.common.models import (
    ConnectionConfig,
    DiscoveredModel,
    PricingSource,
    ProviderConnection,
    SetupRole,
    serves_role,
)
from exp.runtime.models.providers import ProviderEndpoint, ProviderListingError

_LUNA = DiscoveredModel(provider="openai", model="gpt-5.6-luna")
_TERRA = DiscoveredModel(provider="openai", model="gpt-5.6-terra")
_EMBEDDING = DiscoveredModel(provider="openai", model="text-embedding-3-small")
_UNVERIFIED = DiscoveredModel(provider="openai", model="internal-preview-model")


class _FakeLister:
    """One provider listing seam that replays scripted answers per provider."""

    def __init__(
        self,
        answers: dict[str, list[tuple[DiscoveredModel, ...] | ProviderListingError]],
    ) -> None:
        """Record the answers each provider returns, in call order.

        Args:
            answers: Per-provider queue of model tuples or failures to raise.
        """
        self._answers = {provider: list(queue) for provider, queue in answers.items()}
        self.requests: list[ProviderEndpoint] = []

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Answer one listing call from the script.

        Args:
            endpoint: Provider kind, credential, and optional base URL setup resolved.

        Returns:
            The next scripted model tuple for this provider.

        Raises:
            ProviderListingError: The script queued a failure for this call.
        """
        self.requests.append(endpoint)
        answer = self._answers[endpoint.provider].pop(0)
        if isinstance(answer, ProviderListingError):
            raise answer
        return answer


def _prepare(
    console: ScriptedConsole,
    *,
    providers: tuple[str, ...],
    lister: _FakeLister,
    environment: dict[str, str],
    existing_connections: tuple[ProviderConnection, ...] = (),
    existing_aliases: tuple[str, ...] = (),
    configured: tuple[provider_picker.AvailableModel, ...] = (),
) -> (
    tuple[tuple[provider_picker.PreparedEndpoint, ...], tuple[provider_picker.AvailableModel, ...]]
    | None
):
    """Prepare one selected provider set with injected credentials and listings.

    Args:
        console: Scripted console answering every prompt.
        providers: Providers the user selected.
        lister: Injected provider listing seam.
        environment: Mutable process environment consulted for credentials.
        existing_connections: Connections already configured in the catalog.
        existing_aliases: Aliases already configured in the catalog.
        configured: Catalog models already configured before this session.

    Returns:
        The prepared endpoints and configurable models, or ``None`` to reselect providers.
    """
    session = SetupSession(providers=providers)
    return prepare_providers(
        session,
        existing_connections=existing_connections,
        existing_aliases=existing_aliases,
        configured=configured,
        console=console,
        lister=lister,
        environment=environment,
    )


def _persist_openai_key(secret: str, *, connection_id: str = "openai") -> ProviderAuthStore:
    """Write one bound OpenAI credential into the isolated user-data store.

    Args:
        secret: Non-empty API key to persist.
        connection_id: Catalog connection name used as the store key.

    Returns:
        The store that received the record.
    """
    connection = ProviderConnection(
        name=connection_id,
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    )
    store = ProviderAuthStore(default_auth_path())
    store.put(
        connection_id,
        secret,
        binding=StoredCredentialBinding(
            provider="openai",
            endpoint_sha256=connection.catalog_config().identity_sha256(),
        ),
    )
    return store


def _forbid_getpass(prompt: str = "") -> str:
    """Fail if setup asks for a credential paste.

    Args:
        prompt: Masked prompt text setup printed.

    Raises:
        AssertionError: The test expected no credential prompt.
    """
    raise AssertionError(f"unexpected credential prompt: {prompt}")


def test_provider_screen_never_names_credential_variables() -> None:
    """The one opening screen shows plain provider names without credential status."""
    console = ScriptedConsole("2\n\n")

    selection = select_providers(
        SetupSession(), console=console, environment={"OPENAI_API_KEY": "secret-key"}
    )

    assert selection == (("openai",), False)
    assert "OPENAI_API_KEY" not in console.output
    assert "API_KEY" not in console.output


def test_provider_screen_selects_several_providers_in_one_session() -> None:
    """A user selects every provider they want before setup contacts any of them."""
    console = ScriptedConsole("2,3,5\n\n")

    selection = select_providers(
        SetupSession(), console=console, environment={"OPENAI_API_KEY": "secret-key"}
    )

    assert selection == (("openai", "anthropic", "openrouter"), False)


def test_cancelling_the_provider_screen_returns_no_selection() -> None:
    """Cancelling the first screen ends setup without preparing any provider."""
    assert select_providers(SetupSession(), console=ScriptedConsole("q\n"), environment={}) is None


def test_keyboard_provider_list_selects_without_typed_numbers() -> None:
    """Up, Down, and Enter toggle providers; Complete is the only submit action."""
    keys = iter(
        (
            PickerKey.DOWN,
            PickerKey.ENTER,
            PickerKey.DOWN,
            PickerKey.ENTER,
            *(PickerKey.DOWN for _ in range(6)),
            PickerKey.ENTER,
        )
    )
    console = ScriptedConsole("")

    selection = select_providers(
        SetupSession(),
        console=console,
        environment={"OPENAI_API_KEY": "secret-key"},
        read_key=lambda: next(keys),
    )

    assert selection == (("openai", "anthropic"), False)
    assert "\u276f [x] openai" in console.output
    assert "Complete" in console.output
    assert "/ search" in console.output
    assert "Numbers or ranges" not in console.output


def test_keyboard_provider_list_preserves_current_selection_and_cancel() -> None:
    """Reentering the keyboard list keeps prior marks, and q still cancels."""
    console = ScriptedConsole("")
    session = SetupSession(providers=("openai",))

    cancelled = select_providers(
        session,
        console=console,
        environment={},
        read_key=lambda: PickerKey.CANCEL,
    )

    assert cancelled is None
    assert "[x] openai" in console.output


def test_resolve_setup_providers_orders_and_rejects_bad_values() -> None:
    """Explicit names are canonicalized, ordered, and fail closed on bad input."""
    assert resolve_setup_providers(("bedrock", " OpenAI ", "anthropic")) == (
        "openai",
        "anthropic",
        "bedrock",
    )
    with pytest.raises(ValueError, match="unsupported --provider value 'not-a-provider'"):
        resolve_setup_providers(("openai", "not-a-provider"))
    with pytest.raises(ValueError, match="duplicate --provider value 'openai'"):
        resolve_setup_providers(("openai", "OPENAI"))


def test_explicit_azure_and_bedrock_keep_manual_model_declaration() -> None:
    """Named Azure and Bedrock selections still require hand-declared model IDs."""
    assert explicit_provider_selection(("azure", "bedrock")) == (
        ("azure", "bedrock"),
        True,
    )
    assert explicit_provider_selection(("openai",)) == (("openai",), False)


def test_canonical_environment_credential_is_used_without_any_prompt() -> None:
    """The normal path reads the canonical variable and never asks for its name."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA, _EMBEDDING)]})
    environment = {"OPENAI_API_KEY": "secret-key"}

    prepared = _prepare(console, providers=("openai",), lister=lister, environment=environment)

    assert prepared is not None
    endpoints, models = prepared
    assert [endpoint.connection.name for endpoint in endpoints] == ["openai"]
    assert endpoints[0].connection.api_key_env == "OPENAI_API_KEY"
    assert lister.requests == [ProviderEndpoint(provider="openai", api_key="secret-key")]
    assert [model.alias for model in models] == ["gpt-5-6-luna", "text-embedding-3-small"]
    assert "OPENAI_API_KEY is set" not in console.output


def test_dated_snapshots_and_pointer_aliases_collapse_onto_the_base_model_row() -> None:
    """One documented model appears once even when the listing publishes its snapshots."""
    console = ScriptedConsole("")
    lister = _FakeLister(
        {
            "openai": [
                (
                    DiscoveredModel(provider="openai", model="gpt-5.6-luna-2026-01-15"),
                    _LUNA,
                    DiscoveredModel(provider="openai", model="gpt-5.6-luna-latest"),
                    _TERRA,
                )
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "secret-key"},
    )

    assert prepared is not None
    _, models = prepared
    assert [model.model for model in models] == ["gpt-5.6-luna", "gpt-5.6-terra"]


def test_rediscovered_configured_models_reuse_their_existing_rows() -> None:
    """Re-listing a provider never mints a suffixed alias for an already-configured model."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA, _TERRA)]})
    configured = (
        provider_picker.AvailableModel(
            alias="gpt-5-6-luna",
            connection="openai",
            provider="openai",
            model="gpt-5.6-luna",
            capabilities=None,
            pricing_source=PricingSource.CONFIGURED,
            configured=True,
        ),
    )

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "secret-key"},
        existing_aliases=("gpt-5-6-luna",),
        configured=configured,
    )

    assert prepared is not None
    _, models = prepared
    assert [model.model for model in models] == ["gpt-5.6-terra"]
    assert all(not model.alias.endswith("-2") for model in models)


def test_provider_whose_models_are_all_configured_is_still_prepared() -> None:
    """A fully-configured provider prepares its endpoint without new model rows."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA,)]})
    configured = (
        provider_picker.AvailableModel(
            alias="gpt-5-6-luna",
            connection="openai",
            provider="openai",
            model="gpt-5.6-luna",
            capabilities=None,
            pricing_source=PricingSource.CONFIGURED,
            configured=True,
        ),
    )

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "secret-key"},
        existing_aliases=("gpt-5-6-luna",),
        configured=configured,
    )

    assert prepared is not None
    endpoints, models = prepared
    assert len(endpoints) == 1
    assert models == ()


def test_missing_credential_is_pasted_masked_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pasted key is read through a masked prompt, stored, and never printed."""
    from exp.common.auth import ProviderAuthStore, default_auth_path

    prompts: list[str] = []

    def _fake_getpass(prompt: str = "") -> str:
        """Answer the masked prompt without echoing the credential.

        Args:
            prompt: Masked prompt text setup printed.

        Returns:
            The pasted credential.
        """
        prompts.append(prompt)
        return "pasted-secret"

    monkeypatch.setattr(provider_picker, "getpass", _fake_getpass)
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA,)]})
    environment: dict[str, str] = {}

    prepared = _prepare(console, providers=("openai",), lister=lister, environment=environment)

    assert prepared is not None
    endpoints, _ = prepared
    assert prompts and "hidden" in prompts[0]
    assert lister.requests[0].api_key == "pasted-secret"
    assert environment == {}
    assert endpoints[0].connection.api_key_env == "OPENAI_API_KEY"
    assert "pasted-secret" not in console.output
    assert "kept in this process only" not in console.output
    assert ProviderAuthStore(default_auth_path()).get("openai") == "pasted-secret"
    assert tmp_path.exists()


def test_experiential_cloud_uses_platform_login_before_masked_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hosted Cloud connection stores the browser key without invoking getpass."""
    monkeypatch.setattr(
        provider_picker,
        "hosted_platform_login",
        lambda _connection, **_kwargs: "xpl_browser_key",
    )
    monkeypatch.setattr(provider_picker, "getpass", _forbid_getpass)
    console = ScriptedConsole("")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="deepseek-v4-flash",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=0.072,
                        output_cost_per_million_tokens_usd=0.162,
                    ),
                )
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("experiential-cloud",),
        lister=lister,
        environment={},
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].api_key == "xpl_browser_key"
    assert endpoints[0].connection.api_key_env == HOSTED_GATEWAY_API_KEY_ENV
    assert ProviderAuthStore(default_auth_path()).get("experiential-cloud") == ("xpl_browser_key")
    assert "xpl_browser_key" not in console.output


def test_empty_masked_paste_skips_that_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declining to paste a key skips the provider instead of failing setup."""
    monkeypatch.setattr(provider_picker, "getpass", lambda prompt="": "")
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA,)], "anthropic": []})
    environment = {"OPENAI_API_KEY": "secret-key"}

    prepared = _prepare(
        console,
        providers=("openai", "anthropic"),
        lister=lister,
        environment=environment,
    )

    assert prepared is not None
    endpoints, models = prepared
    assert [endpoint.connection.provider for endpoint in endpoints] == ["openai"]
    assert [model.provider for model in models] == ["openai"]
    assert "Skipping anthropic." in console.output
    assert "ANTHROPIC_API_KEY" not in {key for key in environment}


def test_end_of_masked_input_cancels_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed input stream cancels setup rather than continuing without a credential."""

    def _closed(prompt: str = "") -> str:
        """Simulate a closed input stream at the masked prompt.

        Args:
            prompt: Masked prompt text setup printed.

        Raises:
            EOFError: The input stream is closed.
        """
        raise EOFError

    monkeypatch.setattr(provider_picker, "getpass", _closed)

    with pytest.raises(SetupCancelled):
        _prepare(
            ScriptedConsole(""),
            providers=("openai",),
            lister=_FakeLister({"openai": [(_LUNA,)]}),
            environment={},
        )


def test_listing_failure_retries_without_losing_earlier_providers() -> None:
    """Retrying one failed provider keeps the models already discovered for another."""
    console = ScriptedConsole("1\n")
    lister = _FakeLister(
        {
            "openai": [(_LUNA,)],
            "anthropic": [
                ProviderListingError("anthropic model listing failed: provider request timed out"),
                (DiscoveredModel(provider="anthropic", model="claude-sonnet-5"),),
            ],
        }
    )

    prepared = _prepare(
        console,
        providers=("openai", "anthropic"),
        lister=lister,
        environment={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert [endpoint.connection.name for endpoint in endpoints] == ["openai", "anthropic"]
    assert [model.alias for model in models] == ["gpt-5-6-luna", "claude-sonnet-5"]
    assert "provider request timed out" in console.output


def test_invalid_credential_can_be_skipped_and_keeps_the_other_provider() -> None:
    """A rejected credential does not discard providers that already answered."""
    console = ScriptedConsole("2\n")
    lister = _FakeLister(
        {
            "openai": [(_LUNA, _EMBEDDING)],
            "anthropic": [ProviderListingError("anthropic rejected the configured credential")],
        }
    )

    prepared = _prepare(
        console,
        providers=("openai", "anthropic"),
        lister=lister,
        environment={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "bad"},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert [endpoint.connection.provider for endpoint in endpoints] == ["openai"]
    assert [model.alias for model in models] == ["gpt-5-6-luna", "text-embedding-3-small"]
    assert "rejected the configured credential" in console.output


def test_invalid_credential_retry_prompts_for_and_keeps_a_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying a rejected credential asks for a replacement and uses it for the endpoint."""
    prompts: list[str] = []

    def _fake_getpass(prompt: str = "") -> str:
        """Answer the replacement credential prompt without echoing the credential."""
        prompts.append(prompt)
        return "replacement-key"

    monkeypatch.setattr(provider_picker, "getpass", _fake_getpass)
    console = ScriptedConsole("1\n")
    lister = _FakeLister(
        {
            "openai": [
                ProviderListingError("openai rejected the configured credential"),
                (_LUNA,),
            ]
        }
    )
    environment = {"OPENAI_API_KEY": "rejected-key"}

    prepared = _prepare(console, providers=("openai",), lister=lister, environment=environment)

    assert prepared is not None
    endpoints, _ = prepared
    assert len(prompts) == 1
    assert lister.requests == [
        ProviderEndpoint(provider="openai", api_key="rejected-key"),
        ProviderEndpoint(provider="openai", api_key="replacement-key"),
    ]
    assert endpoints[0].api_key == "replacement-key"
    assert environment == {"OPENAI_API_KEY": "replacement-key"}
    assert "replacement-key" not in console.output
    from exp.common.auth import ProviderAuthStore, default_auth_path

    assert ProviderAuthStore(default_auth_path()).get("openai") == "replacement-key"


def test_listing_failure_can_return_to_the_provider_screen() -> None:
    """Choosing to reselect providers reports no prepared endpoints for this attempt."""
    console = ScriptedConsole("3\n")
    lister = _FakeLister({"openai": [ProviderListingError("openai model listing failed")]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "a"},
    )

    assert prepared is None


def test_empty_model_list_offers_recovery_and_can_skip_the_provider() -> None:
    """A provider publishing no configurable model is reported, not silently accepted."""
    console = ScriptedConsole("2\n")
    lister = _FakeLister(
        {
            "openai": [()],
            "anthropic": [(DiscoveredModel(provider="anthropic", model="claude-sonnet-5"),)],
        }
    )

    prepared = _prepare(
        console,
        providers=("openai", "anthropic"),
        lister=lister,
        environment={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    assert prepared is not None
    _, models = prepared
    assert [model.alias for model in models] == ["claude-sonnet-5"]
    assert "published no model identity" in console.output


def test_models_without_verified_metadata_are_hidden_from_the_normal_path() -> None:
    """An official listing without maintained metadata stays off the verified path."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA, _UNVERIFIED, _EMBEDDING)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "a"},
    )

    assert prepared is not None
    _, models = prepared
    assert [model.model for model in models] == ["gpt-5.6-luna", "text-embedding-3-small"]
    assert "internal-preview-model" not in console.output


def test_openai_compatible_identity_only_models_stay_visible_as_unknown() -> None:
    """A trusted compatible listing remains selectable when the host publishes only identities."""
    console = ScriptedConsole("https://models.example.test/v1\n\n")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (DiscoveredModel(provider="openai-compatible", model="hosted-chat"),)
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("openai-compatible",),
        lister=lister,
        environment={"OPENAI_COMPATIBLE_API_KEY": "compat-secret"},
    )

    assert prepared is not None
    _, models = prepared
    assert [model.model for model in models] == ["hosted-chat"]
    assert models[0].capabilities is None
    assert models[0].pricing_source is PricingSource.UNKNOWN
    assert models[0].published is not None
    assert models[0].published.model == "hosted-chat"
    assert f"1 models with {provider_picker.UNKNOWN_METADATA_LABEL}" in console.output
    assert "published no model with verified metadata" not in console.output


def test_openai_compatible_keeps_verified_and_unknown_identities() -> None:
    """Published extension metadata stays verified beside identity-only siblings."""
    console = ScriptedConsole("https://models.example.test/v1\n\n")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="hosted-chat",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                    DiscoveredModel(provider="openai-compatible", model="hosted-preview"),
                )
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("openai-compatible",),
        lister=lister,
        environment={"OPENAI_COMPATIBLE_API_KEY": "compat-secret"},
    )

    assert prepared is not None
    _, models = prepared
    by_model = {item.model: item for item in models}
    assert set(by_model) == {"hosted-chat", "hosted-preview"}
    assert by_model["hosted-chat"].pricing_source is PricingSource.PROVIDER
    assert by_model["hosted-chat"].capabilities is not None
    assert serves_role(by_model["hosted-chat"].capabilities, SetupRole.WORLD_MODEL)
    assert by_model["hosted-preview"].capabilities is None
    assert by_model["hosted-preview"].pricing_source is PricingSource.UNKNOWN
    assert f"1 models, 1 with {provider_picker.UNKNOWN_METADATA_LABEL}" in console.output


def test_discovered_metadata_and_roles_come_from_the_maintained_table() -> None:
    """Discovery merges maintained metadata, so no capability question is ever asked."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA, _EMBEDDING)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "a"},
    )

    assert prepared is not None
    _, models = prepared
    chat, embedder = models
    assert chat.pricing_source is PricingSource.EXP_CATALOG
    assert chat.capabilities is not None
    assert chat.capabilities.supports_structured_output
    assert chat.capabilities.context_window_tokens == 1_050_000
    assert embedder.capabilities is not None
    assert serves_role(embedder.capabilities, SetupRole.EMBEDDER)
    assert serves_role(chat.capabilities, SetupRole.WORLD_MODEL)
    assert serves_role(chat.capabilities, SetupRole.JUDGE)
    assert serves_role(chat.capabilities, SetupRole.ROUTER_CANDIDATE)
    assert chat.detail() == "openai"
    assert embedder.detail() == "openai"


def test_connection_names_and_aliases_avoid_configured_collisions() -> None:
    """Setup derives names around the catalog rather than asking the user to invent them."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA, _TERRA)]})
    existing = (
        ProviderConnection(name="openai", provider="openai", api_key_env="OTHER_OPENAI_API_KEY"),
    )

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "a"},
        existing_connections=existing,
        existing_aliases=("gpt-5-6-luna",),
    )

    assert prepared is not None
    endpoints, models = prepared
    assert endpoints[0].connection.name == "openai-2"
    assert not endpoints[0].configured
    assert [model.alias for model in models] == ["gpt-5-6-luna-2", "gpt-5-6-terra"]


def test_an_identical_configured_connection_is_reused_instead_of_duplicated() -> None:
    """Rerunning setup for a configured provider reuses that connection unchanged."""
    console = ScriptedConsole("")
    lister = _FakeLister({"openai": [(_LUNA,)]})
    existing = (
        ProviderConnection(name="primary", provider="openai", api_key_env="OPENAI_API_KEY"),
    )

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "a"},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, models = prepared
    assert endpoints[0].connection.name == "primary"
    assert endpoints[0].configured
    assert models[0].connection == "primary"
    assert "stored credential" not in console.output


def test_editing_a_stored_credential_can_keep_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeping a stored key reuses it without asking the operator to paste again."""
    monkeypatch.setattr(provider_picker, "getpass", _forbid_getpass)
    store = _persist_openai_key("stored-secret")
    existing = (ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),)
    console = ScriptedConsole("1\n")
    lister = _FakeLister({"openai": [(_LUNA,)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].configured
    assert endpoints[0].api_key == "stored-secret"
    assert lister.requests == [ProviderEndpoint(provider="openai", api_key="stored-secret")]
    assert store.get("openai") == "stored-secret"
    assert "stored-secret" not in console.output
    assert "Keep the stored credential" in console.output


def test_editing_a_stored_credential_can_replace_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing a stored key persists the new paste and never prints the secret."""

    def _fake_getpass(prompt: str = "") -> str:
        """Answer the replacement credential prompt without echoing the credential.

        Args:
            prompt: Masked prompt text setup printed.

        Returns:
            The replacement credential.
        """
        return "replacement-secret"

    monkeypatch.setattr(provider_picker, "getpass", _fake_getpass)
    store = _persist_openai_key("stored-secret")
    existing = (ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),)
    console = ScriptedConsole("2\n")
    lister = _FakeLister({"openai": [(_LUNA,)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].api_key == "replacement-secret"
    assert lister.requests == [ProviderEndpoint(provider="openai", api_key="replacement-secret")]
    assert store.get("openai") == "replacement-secret"
    assert "stored-secret" not in console.output
    assert "replacement-secret" not in console.output


def test_editing_a_stored_credential_can_remove_it_and_use_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a stored key deletes only that record and then uses a non-empty env override."""
    monkeypatch.setattr(provider_picker, "getpass", _forbid_getpass)
    store = _persist_openai_key("stored-secret")
    existing = (ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),)
    console = ScriptedConsole("3\n")
    lister = _FakeLister({"openai": [(_LUNA,)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"OPENAI_API_KEY": "env-secret"},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].api_key == "env-secret"
    assert lister.requests == [ProviderEndpoint(provider="openai", api_key="env-secret")]
    assert store.get("openai") is None
    assert "stored-secret" not in console.output
    assert "env-secret" not in console.output


def test_editing_a_stored_credential_can_remove_it_and_paste_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a stored key with no env override asks for a new paste and persists it."""

    def _fake_getpass(prompt: str = "") -> str:
        """Answer the follow-up credential prompt without echoing the credential.

        Args:
            prompt: Masked prompt text setup printed.

        Returns:
            The newly pasted credential.
        """
        return "pasted-after-remove"

    monkeypatch.setattr(provider_picker, "getpass", _fake_getpass)
    store = _persist_openai_key("stored-secret")
    existing = (ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),)
    console = ScriptedConsole("3\n")
    lister = _FakeLister({"openai": [(_LUNA,)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].api_key == "pasted-after-remove"
    assert lister.requests == [ProviderEndpoint(provider="openai", api_key="pasted-after-remove")]
    assert store.get("openai") == "pasted-after-remove"
    assert "stored-secret" not in console.output
    assert "pasted-after-remove" not in console.output


def test_openai_compatible_uses_generated_env_override_without_asking_for_a_name() -> None:
    """Compatible setup collects the endpoint and uses the generated env override internally."""
    console = ScriptedConsole("https://models.example.test/v1\n")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="internal-chat",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                )
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("openai-compatible",),
        lister=lister,
        environment={"OPENAI_COMPATIBLE_API_KEY": "compat-secret"},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert endpoints[0].connection.base_url == "https://models.example.test/v1"
    assert endpoints[0].connection.api_key_env == "OPENAI_COMPATIBLE_API_KEY"
    assert lister.requests[0].base_url == "https://models.example.test/v1"
    assert lister.requests[0].api_key == "compat-secret"
    assert models[0].pricing_source is PricingSource.PROVIDER


def test_openai_compatible_reuses_a_configured_custom_env_name() -> None:
    """An existing compatible connection keeps its configured override name."""
    console = ScriptedConsole("https://models.example.test/v1\n")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="internal-chat",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                )
            ]
        }
    )
    existing = (
        ProviderConnection(
            name="acme",
            provider="openai-compatible",
            api_key_env="INTERNAL_API_KEY",
            base_url="https://models.example.test/v1",
        ),
    )

    prepared = _prepare(
        console,
        providers=("openai-compatible",),
        lister=lister,
        environment={"INTERNAL_API_KEY": "compat-secret"},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, models = prepared
    assert endpoints[0].connection.name == "acme"
    assert endpoints[0].configured
    assert endpoints[0].connection.api_key_env == "INTERNAL_API_KEY"
    assert lister.requests[0].api_key == "compat-secret"
    assert models[0].connection == "acme"


def test_openai_compatible_does_not_reuse_when_multiple_accounts_share_an_endpoint() -> None:
    """Two compatible connections on one host stay distinct instead of reusing the first."""
    console = ScriptedConsole("https://models.example.test/v1\n")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="internal-chat",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                )
            ]
        }
    )
    existing = (
        ProviderConnection(
            name="acme",
            provider="openai-compatible",
            api_key_env="ACME_API_KEY",
            base_url="https://models.example.test/v1",
        ),
        ProviderConnection(
            name="acme-work",
            provider="openai-compatible",
            api_key_env="ACME_WORK_API_KEY",
            base_url="https://models.example.test/v1",
        ),
    )

    prepared = _prepare(
        console,
        providers=("openai-compatible",),
        lister=lister,
        environment={"OPENAI_COMPATIBLE_API_KEY": "compat-secret"},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, models = prepared
    assert endpoints[0].connection.name == "openai-compatible"
    assert not endpoints[0].configured
    assert endpoints[0].connection.api_key_env == "OPENAI_COMPATIBLE_API_KEY"
    assert models[0].connection == "openai-compatible"


def test_azure_prepares_exact_connection_for_manual_deployment_declaration() -> None:
    """Azure preserves endpoint and API version without inventing deployment metadata."""
    console = ScriptedConsole("https://resource.openai.azure.com\n\n\n")
    lister = _FakeLister({})

    prepared = _prepare(
        console,
        providers=("azure",),
        lister=lister,
        environment={"AZURE_OPENAI_API_KEY": "azure-secret"},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert models == ()
    assert lister.requests == []
    assert endpoints[0].connection == ProviderConnection(
        name="azure",
        provider="azure",
        api_key_env="AZURE_OPENAI_API_KEY",
        base_url="https://resource.openai.azure.com",
        api_version="v1",
        azure_api_surface="openai_deployments",
    )
    assert "declare deployment model IDs" in console.output


def test_azure_prepares_explicit_model_inference_surface() -> None:
    """Interactive setup can author Foundry model inference without losing its discriminator."""
    console = ScriptedConsole(
        "https://resource.services.ai.azure.com/models\n2024-05-01-preview\nmodel_inference\n"
    )

    connection = provider_picker.collect_provider_connection("azure", console=console)

    assert connection is not None
    assert connection.azure_api_surface == "model_inference"


@pytest.mark.parametrize(
    ("configured_base_url", "configured_surface", "collected_base_url", "collected_surface"),
    [
        (
            "https://resource.openai.azure.com",
            None,
            "https://resource.openai.azure.com/",
            "openai_deployments",
        ),
        (
            "https://resource.services.ai.azure.com",
            "model_inference",
            "https://resource.services.ai.azure.com/MODELS",
            "model_inference",
        ),
    ],
)
def test_azure_reuses_identity_equivalent_connection_spellings(
    configured_base_url: str,
    configured_surface: Literal["openai_deployments", "model_inference"] | None,
    collected_base_url: str,
    collected_surface: Literal["openai_deployments", "model_inference"],
) -> None:
    """Setup reuses one credential locator across equivalent Azure endpoint spellings."""
    existing = ProviderConnection(
        name="azure-existing",
        provider="azure",
        api_key_env="AZURE_EXISTING_KEY",
        base_url=configured_base_url,
        api_version=("2024-05-01-preview" if collected_surface == "model_inference" else "v1"),
        azure_api_surface=configured_surface,
    )

    reused = provider_picker._reused_connection(
        (existing,),
        provider="azure",
        api_key_env="AZURE_EXISTING_KEY",
        base_url=collected_base_url,
        api_version=existing.api_version,
        azure_api_surface=collected_surface,
        region=None,
    )

    assert reused == existing


def test_bedrock_prepares_credential_chain_without_listing_or_identity_invention() -> None:
    """Bedrock records only its region and defers explicit model identity to the user."""
    console = ScriptedConsole("us-east-1\n")
    lister = _FakeLister({})

    prepared = _prepare(
        console,
        providers=("bedrock",),
        lister=lister,
        environment={},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert models == ()
    assert lister.requests == []
    assert endpoints[0].api_key == ""
    assert endpoints[0].connection == ProviderConnection(
        name="bedrock",
        provider="bedrock",
        region="us-east-1",
    )
    assert "declare deployment model IDs" in console.output


def test_bedrock_reuses_the_sole_explicit_pair_when_auth_is_not_reentered() -> None:
    """Rerunning setup keeps one existing explicit pair instead of adding ambient auth."""
    existing = (
        ProviderConnection(
            name="bedrock-production",
            provider="bedrock",
            api_key_env="AWS_SECRET_ACCESS_KEY",
            aws_access_key_id_env="AWS_ACCESS_KEY_ID",
            bedrock_auth_mode="access_key_pair",
            region="us-west-2",
        ),
    )
    prepared = _prepare(
        ScriptedConsole("us-west-2\n"),
        providers=("bedrock",),
        lister=_FakeLister({}),
        environment={},
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _models = prepared
    assert endpoints[0].configured
    assert endpoints[0].connection == existing[0]


def test_bedrock_reuses_a_legacy_explicit_pair_without_a_stored_mode() -> None:
    """Legacy pair metadata compares as the canonical access-key-pair auth mode."""
    existing = (
        ProviderConnection(
            name="bedrock-production",
            provider="bedrock",
            api_key_env="AWS_SECRET_ACCESS_KEY",
            aws_access_key_id_env="AWS_ACCESS_KEY_ID",
            region="us-west-2",
        ),
    )
    prepared = _prepare(
        ScriptedConsole("us-west-2\n"),
        providers=("bedrock",),
        lister=_FakeLister({}),
        environment={
            "AWS_ACCESS_KEY_ID": "access-id",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
        },
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _models = prepared
    assert endpoints[0].configured
    assert endpoints[0].connection == existing[0]


def test_bedrock_sts_intent_never_reuses_an_explicit_pair() -> None:
    """A session token keeps setup on ambient auth instead of truncating STS."""
    existing = (
        ProviderConnection(
            name="bedrock-production",
            provider="bedrock",
            api_key_env="AWS_SECRET_ACCESS_KEY",
            aws_access_key_id_env="AWS_ACCESS_KEY_ID",
            bedrock_auth_mode="access_key_pair",
            region="us-west-2",
        ),
    )
    prepared = _prepare(
        ScriptedConsole("us-west-2\n"),
        providers=("bedrock",),
        lister=_FakeLister({}),
        environment={
            "AWS_ACCESS_KEY_ID": "temporary-access-id",
            "AWS_SECRET_ACCESS_KEY": "temporary-secret",
            "AWS_SESSION_TOKEN": "temporary-session-token",
        },
        existing_connections=existing,
    )

    assert prepared is not None
    endpoints, _models = prepared
    assert not endpoints[0].configured
    assert endpoints[0].connection == ProviderConnection(
        name="bedrock",
        provider="bedrock",
        region="us-west-2",
    )


def test_bedrock_authors_an_explicit_pair_from_complete_environment_credentials() -> None:
    """Complete standard AWS environment credentials become a secret-free pair config."""
    connection = provider_picker.collect_provider_connection(
        "bedrock",
        console=ScriptedConsole("us-west-2\n"),
        environment={
            "AWS_ACCESS_KEY_ID": "access-id",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
        },
    )

    assert connection == ConnectionConfig(
        provider="bedrock",
        api_key_env="AWS_SECRET_ACCESS_KEY",
        aws_access_key_id_env="AWS_ACCESS_KEY_ID",
        bedrock_auth_mode="access_key_pair",
        region="us-west-2",
    )


def test_bedrock_authors_bearer_mode_without_persisting_the_token() -> None:
    """A Bedrock bearer environment value records only its locator and mode."""
    connection = provider_picker.collect_provider_connection(
        "bedrock",
        console=ScriptedConsole("us-west-2\n"),
        environment={"AWS_BEARER_TOKEN_BEDROCK": "bearer-value"},
    )

    assert connection == ConnectionConfig(
        provider="bedrock",
        api_key_env="AWS_BEARER_TOKEN_BEDROCK",
        bedrock_auth_mode="api_key",
        region="us-west-2",
    )


def test_release_tty_walk_selects_azure_then_completes() -> None:
    """The installed-wheel provider walk still lands on Azure, then Complete."""
    keys = iter(
        (
            *(PickerKey.DOWN for _ in range(6)),
            PickerKey.ENTER,
            *(PickerKey.DOWN for _ in range(2)),
            PickerKey.ENTER,
        )
    )
    console = ScriptedConsole("")

    selection = select_providers(
        SetupSession(),
        console=console,
        environment={},
        read_key=lambda: next(keys),
    )

    assert selection == (("azure",), True)
    assert "Experiential Cloud" in console.output


def test_azure_and_bedrock_force_explicit_manual_model_declaration() -> None:
    """Providers without a safe listing API make the manual model row available explicitly."""
    console = ScriptedConsole("7,8\n\n")

    selection = select_providers(SetupSession(), console=console, environment={})

    assert selection == (("azure", "bedrock"), True)
    assert "azure" in console.output
    assert "bedrock" in console.output


def test_no_prepared_provider_returns_to_the_provider_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every provider is skipped, setup asks for providers again."""
    monkeypatch.setattr(provider_picker, "getpass", lambda prompt="": "")
    console = ScriptedConsole("")

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=_FakeLister({"openai": []}),
        environment={},
    )

    assert prepared is None
    assert "No provider was prepared" in console.output


def test_experiential_cloud_persists_the_hosted_platform_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experiential Cloud writes openai-compatible plus the Platform origin."""

    def _fail_prompt(_label: str, **_kwargs: object) -> str:
        """Fail if setup asks for a local endpoint or credential variable."""
        raise AssertionError(f"unexpected prompt: {_label}")

    monkeypatch.setattr(provider_picker, "ask_text", _fail_prompt)
    connection = provider_picker.collect_provider_connection(
        "experiential-cloud",
        console=ScriptedConsole(""),
        environment={},
    )

    assert connection is not None
    assert connection.provider == "openai-compatible"
    assert connection.base_url == HOSTED_GATEWAY_DEFAULT_BASE_URL
    assert connection.api_key_env == HOSTED_GATEWAY_API_KEY_ENV


def test_experiential_cloud_honors_the_hosted_gateway_url_override() -> None:
    """Preview or staging may replace the production Platform origin."""
    connection = provider_picker.collect_provider_connection(
        "experiential-cloud",
        console=ScriptedConsole(""),
        environment={HOSTED_GATEWAY_URL_ENV: "https://api-pr-12.preview.experientiallabs.ai/v1"},
    )

    assert connection is not None
    assert connection.base_url == "https://api-pr-12.preview.experientiallabs.ai/v1"


def test_experiential_cloud_lists_through_the_openai_compatible_family() -> None:
    """Discovery uses the persisted provider so Platform listing metadata applies."""
    console = ScriptedConsole("")
    lister = _FakeLister(
        {
            "openai-compatible": [
                (
                    DiscoveredModel(
                        provider="openai-compatible",
                        model="deepseek-v4-flash",
                        supports_completions=True,
                        supports_structured_output=True,
                        input_cost_per_million_tokens_usd=0.072,
                        output_cost_per_million_tokens_usd=0.162,
                    ),
                )
            ]
        }
    )

    prepared = _prepare(
        console,
        providers=("experiential-cloud",),
        lister=lister,
        environment={HOSTED_GATEWAY_API_KEY_ENV: "xpl_test_key"},
    )

    assert prepared is not None
    endpoints, models = prepared
    assert len(endpoints) == 1
    assert endpoints[0].connection.name == "experiential-cloud"
    assert endpoints[0].connection.provider == "openai-compatible"
    assert endpoints[0].connection.base_url == HOSTED_GATEWAY_DEFAULT_BASE_URL
    assert endpoints[0].connection.api_key_env == HOSTED_GATEWAY_API_KEY_ENV
    assert lister.requests == [
        ProviderEndpoint(
            provider="openai-compatible",
            api_key="xpl_test_key",
            base_url=HOSTED_GATEWAY_DEFAULT_BASE_URL,
        )
    ]
    assert [model.model for model in models] == ["deepseek-v4-flash"]
    assert models[0].provider == "openai-compatible"
    assert models[0].capabilities is not None
    assert serves_role(models[0].capabilities, SetupRole.WORLD_MODEL)
    assert "xpl_test_key" not in console.output
    assert "https://gateway.example.test" not in console.output


def test_resolve_setup_providers_accepts_experiential_cloud() -> None:
    """The hosted picker is a first-class --provider value."""
    assert resolve_setup_providers(("experiential-cloud", "openai")) == (
        "experiential-cloud",
        "openai",
    )


def test_provider_screen_lists_experiential_cloud() -> None:
    """Builder setup shows the hosted Platform picker as its own row."""
    console = ScriptedConsole("1\n\n")

    selection = select_providers(SetupSession(), console=console, environment={})

    assert selection == (("experiential-cloud",), False)
    assert "Experiential Cloud" in console.output

"""Provider selection, credential, and discovery screen tests."""

from __future__ import annotations

import pytest

from wmo.cli import provider_picker
from wmo.cli.picker_test import ScriptedConsole
from wmo.cli.provider_picker import (
    SetupCancelled,
    SetupSession,
    credential_hint,
    prepare_providers,
    select_providers,
)
from wmo.common.models import DiscoveredModel, PricingSource, ProviderConnection, SetupRole
from wmo.runtime.models.providers import ProviderEndpoint, ProviderListingError

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
    advanced_credentials: bool = False,
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
        advanced_credentials: Whether the advanced credential path is active.

    Returns:
        The prepared endpoints and configurable models, or ``None`` to reselect providers.
    """
    session = SetupSession(providers=providers, advanced_credentials=advanced_credentials)
    return prepare_providers(
        session,
        existing_connections=existing_connections,
        existing_aliases=existing_aliases,
        console=console,
        lister=lister,
        environment=environment,
    )


def test_provider_screen_reports_credential_availability_per_provider() -> None:
    """The one opening screen states which canonical credentials are already readable."""
    environment = {"OPENAI_API_KEY": "secret-key"}

    assert credential_hint("openai", environment=environment) == "OPENAI_API_KEY is set"
    assert (
        credential_hint("anthropic", environment=environment) == "ANTHROPIC_API_KEY needs a value"
    )


def test_provider_screen_selects_several_providers_in_one_session() -> None:
    """A user selects every provider they want before setup contacts any of them."""
    console = ScriptedConsole("1,2,4\n\n")

    selection = select_providers(
        SetupSession(), console=console, environment={"OPENAI_API_KEY": "secret-key"}
    )

    assert selection == (("openai", "anthropic", "openrouter"), False, False)


def test_provider_screen_keeps_prior_answers_and_reports_advanced_choices() -> None:
    """Reentering the screen preselects prior providers and both advanced paths."""
    console = ScriptedConsole("8,9\n\n")
    session = SetupSession(providers=("openai",))

    selection = select_providers(session, console=console, environment={})

    assert selection == (("openai",), True, True)
    assert "OPENAI_API_KEY needs a value" in console.output


def test_provider_screen_refuses_an_advanced_only_selection() -> None:
    """Advanced rows configure providers, so at least one provider is still required."""
    console = ScriptedConsole("8\n\n1\n\n")

    selection = select_providers(SetupSession(), console=console, environment={})

    assert selection == (("openai",), True, False)
    assert "Select at least one provider." in console.output


def test_cancelling_the_provider_screen_returns_no_selection() -> None:
    """Cancelling the first screen ends setup without preparing any provider."""
    assert select_providers(SetupSession(), console=ScriptedConsole("q\n"), environment={}) is None


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
    assert "credential read from OPENAI_API_KEY" in console.output


def test_missing_credential_is_pasted_masked_and_stays_process_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pasted key is read through a masked prompt and never printed or persisted."""
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
    assert environment == {"OPENAI_API_KEY": "pasted-secret"}
    assert endpoints[0].connection.api_key_env == "OPENAI_API_KEY"
    assert "pasted-secret" not in console.output
    assert "kept in this process only" in console.output


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
    assert "Skipping Anthropic." in console.output
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
    assert "published no model with verified metadata" in console.output


def test_models_without_verified_metadata_are_hidden_from_the_normal_path() -> None:
    """An unverified model cannot serve a role, so setup never offers or claims it."""
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
    assert "1 hidden without verified capabilities or prices" in console.output


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
    assert chat.pricing_source is PricingSource.WMO_CATALOG
    assert chat.capabilities.supports_structured_output
    assert chat.capabilities.context_window_tokens == 1_050_000
    assert SetupRole.EMBEDDER.value in embedder.detail()
    assert "roles: world_model, judge, router_candidate" in chat.detail()


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


def test_advanced_path_can_name_the_credential_environment_variable() -> None:
    """The advanced path still allows an explicit credential variable name."""
    console = ScriptedConsole("TEAM_OPENAI_API_KEY\n")
    lister = _FakeLister({"openai": [(_LUNA,)]})

    prepared = _prepare(
        console,
        providers=("openai",),
        lister=lister,
        environment={"TEAM_OPENAI_API_KEY": "team-secret"},
        advanced_credentials=True,
    )

    assert prepared is not None
    endpoints, _ = prepared
    assert endpoints[0].connection.api_key_env == "TEAM_OPENAI_API_KEY"
    assert lister.requests[0].api_key == "team-secret"


def test_openai_compatible_endpoint_asks_only_for_its_base_url() -> None:
    """A compatible endpoint needs an explicit base URL and nothing else by hand."""
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
    assert lister.requests[0].base_url == "https://models.example.test/v1"
    assert models[0].pricing_source is PricingSource.PROVIDER


def test_azure_prepares_exact_connection_for_manual_deployment_declaration() -> None:
    """Azure preserves endpoint and API version without inventing deployment metadata."""
    console = ScriptedConsole("https://resource.openai.azure.com\n\n")
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
    )
    assert "deployment-specific" in console.output


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
    assert "deployment-specific" in console.output


def test_azure_and_bedrock_force_explicit_manual_model_declaration() -> None:
    """Providers without a safe listing API make the manual model row available explicitly."""
    console = ScriptedConsole("6,7\n\n")

    selection = select_providers(SetupSession(), console=console, environment={})

    assert selection == (("azure", "bedrock"), False, True)
    assert "deployment IDs are declared manually" in console.output
    assert "AWS credential chain" in console.output


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

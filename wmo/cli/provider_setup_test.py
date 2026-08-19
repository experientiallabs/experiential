"""Provider setup CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.picker_test import ScriptedConsole
from wmo.cli.provider_setup import (
    ProviderSetupOptions,
    run_provider_setup,
    run_router_candidate_picker,
)
from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    DiscoveredModel,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ProviderConnection,
    load_model_catalog,
    write_model_catalog,
)
from wmo.runtime.models.providers import ProviderEndpoint

_RUNNER = CliRunner()


def _connection_json(name: str, provider: str, env: str) -> str:
    """Return one concise structured connection flag value.

    Args:
        name: Local connection name.
        provider: Supported provider kind.
        env: Credential environment-variable name.

    Returns:
        JSON value accepted by ``--connection-json``.
    """
    return json.dumps({"name": name, "provider": provider, "api_key_env": env})


def _model_json(
    alias: str,
    connection: str,
    model: str,
    *,
    embeddings: bool = False,
) -> str:
    """Return one concise structured model flag value.

    Args:
        alias: Stable local model alias.
        connection: Referenced local provider connection.
        model: Exact provider-side model ID.
        embeddings: Whether this model supports embeddings.

    Returns:
        JSON value accepted by ``--model-json``.
    """
    return json.dumps(
        {
            "alias": alias,
            "connection": connection,
            "model": model,
            "capabilities": {
                "supports_embeddings": embeddings,
                "input_cost_per_million_tokens_usd": 0.1 if embeddings else None,
            },
        }
    )


def test_noninteractive_setup_collects_many_connections_models_and_roles(tmp_path: Path) -> None:
    """Automation supplies repeatable collections before independent role aliases.

    Args:
        tmp_path: Temporary WMO root receiving the model catalog.
    """
    root = tmp_path / ".wmo"
    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            _connection_json("openai", "openai", "OPENAI_API_KEY"),
            "--connection-json",
            _connection_json("gemini", "gemini", "GEMINI_API_KEY"),
            "--model-json",
            _model_json("world", "openai", "world-id"),
            "--model-json",
            _model_json("judge", "openai", "judge-id"),
            "--model-json",
            _model_json("embed", "gemini", "embed-id", embeddings=True),
            "--world-model",
            "world",
            "--judge",
            "judge",
            "--embedder",
            "embed",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert tuple(sorted(catalog.connections)) == ("gemini", "openai")
    assert tuple(sorted(catalog.models)) == ("embed", "judge", "world")
    assert catalog.roles == ModelRoles(world_model="world", judge="judge", embedder="embed")


def test_noninteractive_setup_accepts_azure_and_bedrock_connections(tmp_path: Path) -> None:
    """Azure and Bedrock connections persist secret-free catalog fields only.

    Args:
        tmp_path: Temporary WMO root receiving the model catalog.
    """
    root = tmp_path / ".wmo"
    azure = json.dumps(
        {
            "name": "azure",
            "provider": "azure",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "base_url": "https://resource.openai.azure.com",
            "api_version": "v1",
        }
    )
    bedrock = json.dumps({"name": "bedrock", "provider": "bedrock", "region": "us-east-1"})
    completion = json.dumps(
        {
            "alias": "gpt",
            "connection": "azure",
            "model": "gpt-deployment",
            "capabilities": {
                "supports_completions": True,
                "input_cost_per_million_tokens_usd": 0,
                "output_cost_per_million_tokens_usd": 0,
                "cached_input_cost_per_million_tokens_usd": 0,
                "cache_write_cost_per_million_tokens_usd": 0,
            },
        }
    )
    embed = json.dumps(
        {
            "alias": "titan",
            "connection": "bedrock",
            "model": "amazon.titan-embed-text-v2:0",
            "capabilities": {
                "supports_embeddings": True,
                "input_cost_per_million_tokens_usd": 0,
            },
        }
    )

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            azure,
            "--connection-json",
            bedrock,
            "--model-json",
            completion,
            "--model-json",
            embed,
            "--world-model",
            "gpt",
            "--judge",
            "gpt",
            "--embedder",
            "titan",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.connections["azure"].api_version == "v1"
    assert catalog.connections["bedrock"].region == "us-east-1"
    assert catalog.connections["bedrock"].api_key_env is None
    assert catalog.models["gpt"].model == "gpt-deployment"
    assert catalog.models["titan"].model == "amazon.titan-embed-text-v2:0"
    text = (root / "models.toml").read_text(encoding="utf-8")
    assert "AZURE_OPENAI_API_KEY" in text
    assert "sk-" not in text


def test_noninteractive_provider_flags_validate_without_prompts_or_writes(tmp_path: Path) -> None:
    """Repeatable --provider values are checked before any catalog write or prompt.

    Args:
        tmp_path: Temporary WMO root without provider configuration.
    """
    root = tmp_path / ".wmo"
    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--provider",
            "bedrock",
            "--provider",
            "openai",
            "--provider",
            "not-a-provider",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "unsupported --provider value 'not-a-provider'" in output
    assert "duplicate --provider value 'openai'" in output
    assert (
        "choose from: openai, anthropic, gemini, openrouter, openai-compatible, azure, bedrock"
        in output
    )
    assert "Select the providers you want to use" not in output
    assert not (root / "models.toml").exists()


def test_noninteractive_valid_providers_still_require_structured_collections(
    tmp_path: Path,
) -> None:
    """Accepted --provider flags do not invent connections or start a prompt.

    Args:
        tmp_path: Temporary WMO root without provider configuration.
    """
    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(tmp_path / ".wmo"),
            "--non-interactive",
            "--provider",
            "anthropic",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "at least one --connection-json" in output
    assert "Select the providers you want to use" not in output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def test_explicit_providers_skip_the_opening_list_and_still_discover_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named providers enter the existing setup session without the opening list.

    Args:
        tmp_path: Temporary WMO root receiving the saved catalog.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"
    options = ProviderSetupOptions(providers=("openai",))

    console, catalog = _setup(
        root,
        "1\n\n1\n\n2\n\ny\n",
        monkeypatch=monkeypatch,
        options=options,
    )

    assert catalog is not None
    assert "Providers" not in unstyle(console.output)
    saved = load_model_catalog(root / "models.toml")
    assert set(saved.connections) == {"openai"}
    assert saved.roles.embedder == "text-embedding-3-small"


def test_noninteractive_setup_reports_every_missing_collection_and_role(tmp_path: Path) -> None:
    """One failure lists the complete remediation instead of serial missing prompts.

    Args:
        tmp_path: Temporary WMO root without provider configuration.
    """
    result = _RUNNER.invoke(
        app,
        ["config", "providers", "--root", str(tmp_path / ".wmo"), "--non-interactive"],
        color=True,
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    for value in (
        "at least one --connection-json",
        "at least one --model-json",
        "--world-model",
        "--judge",
        "--embedder",
    ):
        assert value in output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def test_setup_preserves_router_candidates_and_unrelated_entries(tmp_path: Path) -> None:
    """Editing build roles does not consume or mutate router candidate selection.

    Args:
        tmp_path: Temporary WMO root containing preserved catalog state.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "router": ConnectionConfig(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
            },
            models={
                "candidate": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="router",
                    model="vendor/candidate",
                )
            },
            roles=ModelRoles(candidates=("candidate",), incumbent="candidate"),
        ),
    )

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            _connection_json("openai", "openai", "OPENAI_API_KEY"),
            "--model-json",
            _model_json("all", "openai", "all-id", embeddings=True),
            "--world-model",
            "all",
            "--judge",
            "all",
            "--embedder",
            "all",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.roles.candidates == ("candidate",)
    assert catalog.roles.incumbent == "candidate"
    assert catalog.models["candidate"].model == "vendor/candidate"


def test_structured_input_rejects_openai_compatible_without_capabilities(tmp_path: Path) -> None:
    """Private compatible endpoints cannot acquire provider-wide capability guesses.

    Args:
        tmp_path: Temporary WMO root receiving rejected structured input.
    """
    connection = json.dumps(
        {
            "name": "private",
            "provider": "openai-compatible",
            "api_key_env": "PRIVATE_API_KEY",
            "base_url": "https://models.example.test/v1",
        }
    )
    model = json.dumps({"alias": "private", "connection": "private", "model": "private-model"})

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(tmp_path / ".wmo"),
            "--non-interactive",
            "--connection-json",
            connection,
            "--model-json",
            model,
            "--world-model",
            "private",
            "--judge",
            "private",
            "--embedder",
            "private",
        ],
    )

    assert result.exit_code == 2
    assert "must declare embedding support" in result.output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def _setup(
    root: Path,
    answers: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    lister: _FakeLister | None = None,
    options: ProviderSetupOptions | None = None,
    offer_recommended_defaults: bool = False,
) -> tuple[ScriptedConsole, ModelCatalog | None]:
    """Run one scripted interactive setup session against injected provider listings.

    Args:
        root: Local WMO root receiving the catalog.
        answers: Newline-separated answers for every screen.
        monkeypatch: Patch fixture supplying canonical credentials.
        lister: Injected provider listing seam, defaulting to the OpenAI fixture.
        options: Optional role flags or automation values.
        offer_recommended_defaults: Whether one verified default assignment is offered.

    Returns:
        The scripted console and the committed catalog, or ``None`` when setup aborted.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    console = ScriptedConsole(answers)
    try:
        catalog = run_provider_setup(
            root,
            options or ProviderSetupOptions(),
            non_interactive=False,
            replace=False,
            console=console,
            lister=lister or _FakeLister(),
            offer_recommended_defaults=offer_recommended_defaults,
        )
    except typer.Abort:
        return console, None
    return console, catalog


def test_wizard_recommended_setup_needs_only_provider_and_one_default_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified discovery fills every role after the top-level recommended choice.

    Args:
        tmp_path: Temporary WMO root receiving the secret-free catalog.
        monkeypatch: Pytest patch fixture supplying an existing canonical credential.
    """
    root = tmp_path / ".wmo"
    console, catalog = _setup(
        root,
        "1\n\n\n",
        monkeypatch=monkeypatch,
        offer_recommended_defaults=True,
    )

    assert catalog is not None
    assert catalog.roles.world_model == "gpt-5-6-luna"
    assert catalog.roles.judge == "gpt-5-6-luna"
    assert catalog.roles.embedder == "text-embedding-3-large"
    assert catalog.roles.candidates == ("gpt-5-6-luna", "gpt-5-6-terra")
    assert catalog.roles.incumbent == "gpt-5-6-luna"
    transcript = unstyle(console.output)
    assert transcript.count("Use these recommended models?") == 1
    assert "Select the models to configure" not in transcript
    assert "Save this configuration?" not in transcript
    persisted = (root / "models.toml").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in persisted
    assert "openai-secret" not in transcript
    assert "openai-secret" not in persisted


def test_wizard_recommended_setup_prefers_provider_diversity_for_router_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified second provider supplies the alternative without displacing Luna defaults.

    Args:
        tmp_path: Temporary WMO root receiving the selected multi-provider catalog.
        monkeypatch: Pytest patch fixture supplying both canonical credentials.
    """
    lister = _FakeLister(
        {
            "openai": (
                DiscoveredModel(provider="openai", model="gpt-5.6-luna"),
                DiscoveredModel(provider="openai", model="gpt-5.6-terra"),
                DiscoveredModel(provider="openai", model="text-embedding-3-large"),
            ),
            "anthropic": (DiscoveredModel(provider="anthropic", model="claude-sonnet-5"),),
        }
    )

    console, catalog = _setup(
        tmp_path / ".wmo",
        "1,2\n\n\n",
        monkeypatch=monkeypatch,
        lister=lister,
        offer_recommended_defaults=True,
    )

    assert catalog is not None
    assert catalog.roles.world_model == "gpt-5-6-luna"
    assert catalog.roles.judge == "gpt-5-6-luna"
    assert catalog.roles.embedder == "text-embedding-3-large"
    assert catalog.roles.candidates == ("gpt-5-6-luna", "claude-sonnet-5")
    assert catalog.roles.incumbent == "gpt-5-6-luna"
    summary = unstyle(console.output)
    assert "gpt-5-6-luna" in summary
    assert "claude-sonnet-5" in summary


def test_wizard_recommended_setup_falls_back_by_verified_capability_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent ranked IDs use stable capability and price metadata from discovery.

    Args:
        tmp_path: Temporary WMO root receiving deterministic fallback choices.
        monkeypatch: Pytest patch fixture supplying the canonical OpenAI credential.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    lister = _FakeLister(
        {
            "anthropic": (
                DiscoveredModel(
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    context_window_tokens=200_000,
                    maximum_output_tokens=32_000,
                ),
                DiscoveredModel(
                    provider="anthropic",
                    model="claude-opus-4-8",
                    context_window_tokens=200_000,
                    maximum_output_tokens=32_000,
                ),
            ),
            "gemini": (DiscoveredModel(provider="gemini", model="gemini-embedding-001"),),
        }
    )

    _console, catalog = _setup(
        tmp_path / ".wmo",
        "2,3\n\n\n",
        monkeypatch=monkeypatch,
        lister=lister,
        offer_recommended_defaults=True,
    )

    assert catalog is not None
    assert catalog.roles.world_model == "claude-sonnet-4-6"
    assert catalog.roles.judge == "claude-sonnet-4-6"
    assert catalog.roles.embedder == "gemini-embedding-001"
    assert catalog.roles.candidates == ("claude-sonnet-4-6", "claude-opus-4-8")
    assert catalog.roles.incumbent == "claude-sonnet-4-6"


class _FakeLister:
    """One provider listing seam that answers from fixed provider catalogs."""

    def __init__(self, catalogs: dict[str, tuple[DiscoveredModel, ...]] | None = None) -> None:
        """Record what each provider publishes for the authenticated account.

        Args:
            catalogs: Per-provider model tuples, defaulting to the OpenAI fixture.
        """
        self._catalogs = catalogs or {
            "openai": (
                DiscoveredModel(provider="openai", model="gpt-5.6-luna"),
                DiscoveredModel(provider="openai", model="gpt-5.6-terra"),
                DiscoveredModel(provider="openai", model="text-embedding-3-small"),
                DiscoveredModel(provider="openai", model="text-embedding-3-large"),
                DiscoveredModel(provider="openai", model="internal-preview-model"),
            )
        }
        self.requests: list[str] = []

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Answer one listing call for the requested provider.

        Args:
            endpoint: Provider kind, credential, and optional base URL setup resolved.

        Returns:
            The models this provider publishes in the fixture.
        """
        self.requests.append(endpoint.provider)
        return self._catalogs[endpoint.provider]


def test_router_candidate_picker_discovers_only_eligible_completion_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The router flow reuses provider discovery and hides embedding/unverified rows.

    Args:
        monkeypatch: Patch fixture supplying the configured provider credential.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    connection = ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY")
    catalog = ModelCatalog(
        connections={"custom": connection},
        models={
            "world": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="custom",
                model="world",
            )
        },
        roles=ModelRoles(world_model="world"),
    )

    picked = run_router_candidate_picker(
        catalog,
        console=ScriptedConsole("1\n\n1,2\n\n\n\n1\n"),
        lister=_FakeLister(),
        environment={"OPENAI_API_KEY": "openai-secret"},
    )

    assert picked is not None
    assert picked.selection.candidates == ("gpt-5-6-luna", "gpt-5-6-terra")
    assert picked.selection.incumbent == "gpt-5-6-luna"
    assert tuple(model.alias for model in picked.candidate_models) == (
        "gpt-5-6-luna",
        "gpt-5-6-terra",
    )
    assert all(model.capabilities.supports_completions for model in picked.candidate_models)
    assert picked.connections == (
        ProviderConnection(name="openai", provider="openai", api_key_env="OPENAI_API_KEY"),
    )


def test_interactive_setup_saves_providers_models_and_roles_it_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal path asks only for providers, models, roles, and one confirmation.

    Args:
        tmp_path: Temporary WMO root receiving the saved catalog.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"

    console, catalog = _setup(root, "1\n\n1\n\n1\n\n2\n1,2\n\n\n\n1\ny\n", monkeypatch=monkeypatch)

    assert catalog is not None
    saved = load_model_catalog(root / "models.toml")
    assert set(saved.connections) == {"openai"}
    assert saved.connections["openai"].api_key_env == "OPENAI_API_KEY"
    assert set(saved.models) == {"gpt-5-6-luna", "gpt-5-6-terra", "text-embedding-3-small"}
    assert saved.roles.world_model == "gpt-5-6-luna"
    assert saved.roles.judge == "gpt-5-6-luna"
    assert saved.roles.embedder == "text-embedding-3-small"
    assert saved.roles.candidates == ("gpt-5-6-luna", "gpt-5-6-terra")
    assert saved.roles.incumbent == "gpt-5-6-luna"
    luna = saved.models["gpt-5-6-luna"].capabilities
    assert luna is not None
    assert luna.supports_tools
    assert luna.context_window_tokens == 1_050_000
    assert luna.input_cost_per_million_tokens_usd == 0.2
    assert luna.reasoning_effort == "medium"
    embedder = saved.models["text-embedding-3-small"].capabilities
    assert embedder is not None
    assert embedder.reasoning_effort is None
    assert "internal-preview-model" not in {model.model for model in saved.models.values()}
    persisted = (root / "models.toml").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in persisted
    assert "openai-secret" not in persisted
    printed = unstyle(console.output)
    assert "connection name" not in printed.casefold()
    assert "openai-secret" not in printed


def test_interactive_final_rejection_writes_no_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All answers remain in memory until the user confirms the complete summary.

    Args:
        tmp_path: Temporary WMO root used to prove no catalog write occurs.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"

    console, catalog = _setup(root, "1\n\n1\n\n1\n\n1\n\nn\n", monkeypatch=monkeypatch)

    assert catalog is None
    assert "Configuration" in console.output
    assert not (root / "models.toml").exists()


def test_interactive_cancellation_writes_no_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling any screen ends setup and leaves the catalog untouched.

    Args:
        tmp_path: Temporary WMO root used to prove no catalog write occurs.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"

    console, catalog = _setup(root, "1\n\nq\n", monkeypatch=monkeypatch)

    assert catalog is None
    assert "Setup cancelled. Nothing was written." in console.output
    assert not (root / "models.toml").exists()


def test_back_from_the_model_screen_reselects_providers_without_losing_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back returns to the provider screen with the prior provider still selected.

    Args:
        tmp_path: Temporary WMO root receiving the saved catalog.
        monkeypatch: Patch fixture supplying both canonical credentials.
    """
    root = tmp_path / ".wmo"
    lister = _FakeLister(
        {
            "openai": (
                DiscoveredModel(provider="openai", model="gpt-5.6-luna"),
                DiscoveredModel(provider="openai", model="text-embedding-3-small"),
            ),
            "anthropic": (DiscoveredModel(provider="anthropic", model="claude-sonnet-5"),),
        }
    )

    console, catalog = _setup(
        root,
        "1\n\nb\n2\n\n1\n\n1\n\n1\n1,2\n\n\n1\ny\n",
        monkeypatch=monkeypatch,
        lister=lister,
    )

    assert catalog is not None
    assert lister.requests == ["openai", "openai", "anthropic"]
    saved = load_model_catalog(root / "models.toml")
    assert set(saved.connections) == {"openai", "anthropic"}
    assert set(saved.models) == {"claude-sonnet-5", "gpt-5-6-luna", "text-embedding-3-small"}
    assert saved.roles.embedder == "text-embedding-3-small"
    assert "verifying anthropic" in unstyle(console.output)


def test_rerunning_setup_preserves_unrelated_models_and_router_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second session reassigns build roles without disturbing other catalog state.

    Args:
        tmp_path: Temporary WMO root containing preserved catalog state.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "router": ConnectionConfig(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
            },
            models={
                "candidate": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="router",
                    model="vendor/candidate",
                )
            },
            roles=ModelRoles(candidates=("candidate",), incumbent="candidate"),
        ),
    )

    _, catalog = _setup(root, "2\n\n2,4\n\n1\n\n1\n\n1\n\n\ny\n", monkeypatch=monkeypatch)

    assert catalog is not None
    saved = load_model_catalog(root / "models.toml")
    assert saved.connections["router"].api_key_env == "OPENROUTER_API_KEY"
    assert saved.models["candidate"].model == "vendor/candidate"
    assert saved.roles.candidates == ("candidate",)
    assert saved.roles.incumbent == "candidate"
    assert saved.roles.world_model == "gpt-5-6-luna"


def test_setup_preserves_entries_owned_by_providers_it_does_not_configure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connections and aliases outside the setup provider set survive a session untouched.

    Args:
        tmp_path: Temporary WMO root containing a registered SFT sampling handle.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "tinker": ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY")
            },
            models={
                "sft-run": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="tinker",
                    model="tinker://sampling/run",
                )
            },
        ),
    )

    console, catalog = _setup(root, "1\n\n1\n\n1\n\n1\n\ny\n", monkeypatch=monkeypatch)

    assert catalog is not None
    saved = load_model_catalog(root / "models.toml")
    assert saved.connections["tinker"].api_key_env == "TINKER_API_KEY"
    assert saved.models["sft-run"].model == "tinker://sampling/run"
    assert saved.roles.world_model == "gpt-5-6-luna"
    assert "sft-run" not in unstyle(console.output)


class _UnavailableLister:
    """One provider listing seam that fails if setup requests any provider."""

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Refuse every listing call.

        Args:
            endpoint: Provider kind, credential, and optional base URL setup resolved.

        Raises:
            AssertionError: Setup issued a provider request it did not need.
        """
        raise AssertionError(f"unexpected listing request for {endpoint.provider}")


def test_configured_models_are_reassignable_without_any_provider_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roles in a complete catalog can be edited offline, with no credential or listing.

    Args:
        tmp_path: Temporary WMO root containing a complete catalog.
        monkeypatch: Patch fixture removing every canonical credential.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    chat = ModelCapabilities(
        supports_tools=True,
        supports_structured_output=True,
        supports_completions=True,
        context_window_tokens=200_000,
        maximum_output_tokens=32_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=4.0,
        cached_input_cost_per_million_tokens_usd=0.1,
        cache_write_cost_per_million_tokens_usd=1.25,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        context_window_tokens=8_192,
        input_cost_per_million_tokens_usd=0.02,
    )
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")
            },
            models={
                "chat": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="openai",
                    model="chat-id",
                    capabilities=chat,
                ),
                "backup": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="openai",
                    model="backup-id",
                    capabilities=chat,
                ),
                "embed": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="openai",
                    model="embed-id",
                    capabilities=embedding,
                ),
            },
            roles=ModelRoles(world_model="chat", judge="chat", embedder="embed"),
        ),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    console = ScriptedConsole("1\n\n1\n1\n1\n1,2\n\n1\ny\n")

    catalog = run_provider_setup(
        root,
        ProviderSetupOptions(),
        non_interactive=False,
        replace=False,
        console=console,
        lister=_UnavailableLister(),
    )

    assert catalog is not None
    saved = load_model_catalog(root / "models.toml")
    assert set(saved.models) == {"backup", "chat", "embed"}
    assert saved.roles.world_model != "chat"
    assert saved.roles.embedder == "embed"
    assert "Keep the models already configured" in unstyle(console.output)


def test_offline_roles_include_models_on_tinker_without_provider_requests(tmp_path: Path) -> None:
    """Configured-only editing does not discard Tinker aliases outside the setup picker.

    Args:
        tmp_path: Temporary WMO root containing a complete custom-provider catalog.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    chat = ModelCapabilities(
        supports_completions=True,
        supports_structured_output=True,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=4.0,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.02,
    )
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "custom": ConnectionConfig(
                    provider="tinker",
                    api_key_env="TINKER_API_KEY",
                ),
                "openai": ConnectionConfig(
                    provider="openai",
                    api_key_env="OPENAI_API_KEY",
                ),
            },
            models={
                "chat": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="custom",
                    model="chat-id",
                    capabilities=chat,
                ),
                "embed": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="openai",
                    model="embed-id",
                    capabilities=embedding,
                ),
            },
            roles=ModelRoles(world_model="chat", judge="chat", embedder="embed"),
        ),
    )
    console = ScriptedConsole("1\n\n\n1\n1\n1\n\ny\n")

    catalog = run_provider_setup(
        root,
        ProviderSetupOptions(),
        non_interactive=False,
        replace=False,
        console=console,
        lister=_UnavailableLister(),
    )

    assert catalog.roles == ModelRoles(world_model="chat", judge="chat", embedder="embed")
    assert set(catalog.connections) == {"custom", "openai"}
    assert set(catalog.models) == {"chat", "embed"}
    printed = unstyle(console.output)
    assert "world model  chat" in printed
    assert "embedder     embed" in printed


def test_offline_roles_retain_assigned_tinker_alias_without_capabilities(tmp_path: Path) -> None:
    """A current role may retain an unverified alias without provider access or record changes.

    Args:
        tmp_path: Temporary WMO root containing an assigned legacy Tinker alias.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.02,
    )
    original = ModelCatalog(
        connections={
            "tinker": ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY"),
            "openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY"),
        },
        models={
            "legacy": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="tinker",
                model="tinker://sampling/run",
            ),
            "embed": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="embed-id",
                capabilities=embedding,
            ),
        },
        roles=ModelRoles(world_model="legacy", judge="legacy", embedder="embed"),
    )
    write_model_catalog(root / "models.toml", original)
    console = ScriptedConsole("1\n\n\n\n\ny\n")

    saved = run_provider_setup(
        root,
        ProviderSetupOptions(),
        non_interactive=False,
        replace=False,
        console=console,
        lister=_UnavailableLister(),
    )

    assert saved == original
    assert load_model_catalog(root / "models.toml") == original
    printed = unstyle(console.output)
    assert "world model  legacy" in printed
    assert "retain only: judge, world_model" in printed


def test_offline_setup_preserves_exact_unverified_router_roles_without_revalidation(
    tmp_path: Path,
) -> None:
    """Exact retained candidates persist without partial writes or capability invention.

    Args:
        tmp_path: Temporary WMO root containing complete build roles and legacy candidates.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    chat = ModelCapabilities(
        supports_completions=True,
        supports_structured_output=True,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=4.0,
        cached_input_cost_per_million_tokens_usd=0.0,
        cache_write_cost_per_million_tokens_usd=0.0,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.02,
    )
    original = ModelCatalog(
        connections={
            "tinker": ConnectionConfig(provider="tinker", api_key_env="TINKER_API_KEY"),
            "openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY"),
        },
        models={
            "legacy-a": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="tinker",
                model="tinker://sampling/a",
            ),
            "legacy-b": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="tinker",
                model="tinker://sampling/b",
            ),
            "chat": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="chat-id",
                capabilities=chat,
            ),
            "embed": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="embed-id",
                capabilities=embedding,
            ),
        },
        roles=ModelRoles(
            candidates=("legacy-a", "legacy-b"),
            incumbent="legacy-a",
            world_model="chat",
            judge="chat",
            embedder="embed",
        ),
    )
    write_model_catalog(root / "models.toml", original)

    saved = run_provider_setup(
        root,
        ProviderSetupOptions(),
        non_interactive=False,
        replace=False,
        console=ScriptedConsole("1\n\n\n\n\n\n\n\n\ny\n"),
        lister=_UnavailableLister(),
    )

    assert saved == original
    assert load_model_catalog(root / "models.toml") == original


def test_role_flags_preselect_the_roles_the_picker_offers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit role flags are offered as defaults an empty line accepts.

    Args:
        tmp_path: Temporary WMO root receiving the saved catalog.
        monkeypatch: Patch fixture supplying the canonical credential.
    """
    root = tmp_path / ".wmo"
    options = ProviderSetupOptions(
        world_model="gpt-5-6-terra",
        judge="gpt-5-6-terra",
        embedder="text-embedding-3-small",
    )

    _, catalog = _setup(
        root,
        "1\n\n1,2,3\n\n\n\n\n\n\n\ny\n",
        monkeypatch=monkeypatch,
        options=options,
    )

    assert catalog is not None
    saved = load_model_catalog(root / "models.toml")
    assert saved.roles.world_model == "gpt-5-6-terra"
    assert saved.roles.judge == "gpt-5-6-terra"
    assert saved.roles.embedder == "text-embedding-3-small"

"""Scenario construction CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.app_fixtures_test import *


def test_scenarios_build_missing_file_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # A typo in --file is the likeliest first-run mistake; it must not be a FileNotFoundError.
    result = runner.invoke(app, ["scenarios", "build", "--file", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "does not exist" in flat
    assert "`wmo download <benchmark>`" in flat


def test_scenarios_build_directory_file_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["scenarios", "build", "--file", str(tmp_path)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, IsADirectoryError)
    assert "is a directory" in _framed(result.output)


def test_scenarios_build_help_documents_the_accepted_embed_provider() -> None:
    # The help must name values the command accepts: EmbedderKind spells Azure "azure".
    result = runner.invoke(app, ["scenarios", "build", "--help"])
    assert result.exit_code == 0
    flat = _framed(result.output)
    documented = flat[flat.index("Facet embedder") : flat.index("--embed-model")]
    for kind in EmbedderKind:
        assert kind.value in documented, kind
    assert "azure_openai" not in documented


def test_scenarios_verify_missing_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        ["scenarios", "verify", str(tmp_path / "nope.json"), "--file", str(corpus)],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "does not exist" in flat
    assert "`wmo scenarios build --file <traces.jsonl> --out" in flat


def test_scenarios_verify_missing_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text('{"scenarios": []}', encoding="utf-8")
    result = runner.invoke(
        app,
        ["scenarios", "verify", str(scenarios_file), "--file", str(tmp_path / "nope.jsonl")],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    flat = _framed(result.output)
    assert "--file" in flat and "does not exist" in flat


def test_scenarios_verify_malformed_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Pydantic's ValidationError points at pydantic's docs; the user needs the command that
    # writes this artifact instead.
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text("not json\n", encoding="utf-8")
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["scenarios", "verify", str(scenarios_file), "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValidationError)
    flat = _framed(result.output)
    assert "is not a scenario set written by `wmo scenarios build`" in flat
    assert "errors.pydantic.dev" not in flat


def test_scenarios_verify_non_utf8_scenario_set_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): `ScenarioSet.load` reads with encoding="utf-8", so binary bytes
    # raise UnicodeDecodeError *before* pydantic and slipped past the ValidationError handler.
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_bytes(b"\xff\xfe\x00binary")
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["scenarios", "verify", str(scenarios_file), "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, UnicodeDecodeError)
    assert "is not UTF-8 text" in _framed(result.output)


def test_scenarios_build_non_utf8_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): a path that exists still fails inside the adapter's read.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_bytes(b"\xff\xfe\x00binary")
    result = runner.invoke(app, ["scenarios", "build", "--file", str(corpus)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, UnicodeDecodeError)
    flat = _framed(result.output)
    assert "--file" in flat and "is not UTF-8 text" in flat


@pytest.mark.skipif(sys.platform == "win32", reason="chmod(0) does not revoke read on Windows")
def test_scenarios_build_unreadable_corpus_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    # Regression (Greptile P1): chmod-000 raised PermissionError out of Path.read_text.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    corpus.chmod(0)
    try:
        result = runner.invoke(app, ["scenarios", "build", "--file", str(corpus)])
    finally:
        corpus.chmod(0o644)
    assert result.exit_code != 0
    assert not isinstance(result.exception, PermissionError)
    flat = _framed(result.output)
    assert "could not be read" in flat
    assert "shows its owner and mode" in flat


def test_scenario_role_llms_resolve_from_settings(monkeypatch) -> None:  # noqa: ANN001
    from wmo.common.config.settings import ModelRole, ModelsSettings, ProjectSettings

    made: list[ProviderConfig] = []

    def fake_get_provider(config: ProviderConfig) -> ProviderConfig:
        made.append(config)
        return config  # identity provider: assertions read the config directly

    monkeypatch.setattr("wmo.common.providers.get_provider", fake_get_provider)
    monkeypatch.setattr(
        command_common_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(
                worker=ModelRole(provider="azure", model="gpt-5.4", endpoint="https://x/v1"),
                judge=ModelRole(
                    provider="bedrock", model="us.anthropic.claude-opus-4-8", region="us-east-2"
                ),
            )
        ),
    )
    summary, worker, judge = command_common_module._scenario_role_llms(None, None, None)
    assert summary is worker  # unset summary falls back to the worker role
    assert cast(ProviderConfig, worker).model == "gpt-5.4"
    assert cast(ProviderConfig, worker).endpoint == "https://x/v1"
    assert cast(ProviderConfig, judge).model == "us.anthropic.claude-opus-4-8"
    assert cast(ProviderConfig, judge).region == "us-east-2"
    assert len(made) == 2  # worker constructed once and shared with summary


def test_scenario_role_llms_cli_flags_pin_every_role(monkeypatch) -> None:  # noqa: ANN001
    from wmo.common.config.settings import ProjectSettings

    monkeypatch.setattr("wmo.common.providers.get_provider", lambda config: config)
    monkeypatch.setattr(command_common_module, "load_settings_or_abort", lambda: ProjectSettings())
    summary, worker, judge = command_common_module._scenario_role_llms(
        "bedrock", "some-model", None
    )
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "some-model"


def test_scenario_role_llms_model_flag_keeps_the_configured_provider(monkeypatch) -> None:  # noqa: ANN001
    # Half a flag pair used to complete from bedrock, so `--model gpt-5.5` on an OpenAI project
    # asked bedrock for an OpenAI model id.
    from wmo.common.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr("wmo.common.providers.get_provider", lambda config: config)
    monkeypatch.setattr(
        command_common_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="openai", model="gpt-5.4-mini"))
        ),
    )
    _summary, worker, _judge = command_common_module._scenario_role_llms(None, "gpt-5.5", None)
    assert cast(ProviderConfig, worker).kind is ProviderKind.OPENAI
    assert cast(ProviderConfig, worker).model == "gpt-5.5"


def test_scenario_role_llms_default_when_nothing_configured(monkeypatch) -> None:  # noqa: ANN001
    from wmo.common.config.settings import ProjectSettings

    monkeypatch.setattr("wmo.common.providers.get_provider", lambda config: config)
    monkeypatch.setattr(command_common_module, "load_settings_or_abort", lambda: ProjectSettings())
    summary, worker, judge = command_common_module._scenario_role_llms(None, None, None)
    assert summary is worker
    assert worker is judge
    assert cast(ProviderConfig, worker).model == "us.anthropic.claude-opus-4-8"


def test_worker_role_provider_config_falls_back_to_bedrock(monkeypatch) -> None:  # noqa: ANN001
    from wmo.common.config.settings import ProjectSettings

    monkeypatch.setattr(command_common_module, "load_settings_or_abort", lambda: ProjectSettings())
    config = command_common_module._worker_role_provider_config(None, None, None)
    assert config.kind is ProviderKind.BEDROCK
    assert config.model == "us.anthropic.claude-opus-4-8"


def test_worker_role_provider_config_model_flag_keeps_the_role_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The endpoint describes the BACKEND, not the model, so swapping the model keeps it.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    config = command_common_module._worker_role_provider_config(
        None, "gpt-5.5", None, deployment="prod-55-canary"
    )

    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.endpoint == "https://azure.example/v1"
    assert config.deployment == "prod-55-canary"


def test_worker_role_provider_config_refuses_an_azure_model_swap_without_a_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On Azure the wire `model` IS the deployment name, so the role's deployment names the model
    # being replaced. Keeping it would call gpt-5.4 while reporting gpt-5.5; guessing `gpt-5.5`
    # 404s on a resource that names deployments anything else, and `wmo eval` turns that into a
    # silent fidelity=0.000 at exit 0. So refuse, naming the command that fixes it.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    with pytest.raises(typer.BadParameter) as excinfo:
        command_common_module._worker_role_provider_config(None, "gpt-5.5", None)

    message = str(excinfo.value)
    assert "prod-54-canary" in message
    assert "wmo providers set --provider azure --model gpt-5.5 --deployment <deployment>" in message


def test_worker_role_provider_config_allows_an_azure_model_swap_the_deployment_already_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resource whose deployments are named after their models has already answered the question.
    _azure_worker_settings(monkeypatch, "gpt-5.5")

    config = command_common_module._worker_role_provider_config(None, "gpt-5.5", None)

    assert config.model == "gpt-5.5"
    assert config.deployment == "gpt-5.5"


def test_worker_role_provider_config_derives_a_deployment_the_role_never_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing configured to contradict, so fall back to the model type as
    # `_worker_provider_config` does rather than refusing over a value that was never there.
    _azure_worker_settings(monkeypatch, None)

    config = command_common_module._worker_role_provider_config(None, "gpt-5.5", None)

    assert config.deployment == "gpt-5.5"


def test_worker_role_provider_config_keeps_a_custom_deployment_for_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-stating the role's own model is not a model change: an operator's deployment name is not
    # derivable from the model id, so re-deriving it here would break a working config.
    _azure_worker_settings(monkeypatch, "prod-54-canary")

    config = command_common_module._worker_role_provider_config(None, "gpt-5.4", None)

    assert config.deployment == "prod-54-canary"


def test_worker_role_provider_config_provider_flag_uses_that_backends_flagship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A --provider naming another backend must take its model from THAT backend's catalog:
    # pairing --provider openai with bedrock's claude-opus-4-8 sends OpenAI a model it has never
    # heard of, so the command fails instead of running on the backend the user selected.
    from wmo.common.config.settings import ModelRole, ModelsSettings, ProjectSettings

    monkeypatch.setattr(
        command_common_module,
        "load_settings_or_abort",
        lambda: ProjectSettings(
            models=ModelsSettings(worker=ModelRole(provider="bedrock", model="claude-sonnet-4-6"))
        ),
    )

    config = command_common_module._worker_role_provider_config("openai", None, None)

    assert config.kind is ProviderKind.OPENAI
    # The flagship is the catalog's first OpenAI row: gpt-5.6-sol, the top tier of the 5.6
    # family (sol > terra > luna).
    assert config.model == "gpt-5.6-sol"


def test_worker_role_provider_config_demands_a_model_for_a_catalog_less_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # openrouter/tinker publish no built-in rows - nothing can derive an operator's route or
    # weights path - so the fix is to say which model, not to guess one.
    from wmo.common.config.settings import ProjectSettings

    monkeypatch.setattr(command_common_module, "load_settings_or_abort", lambda: ProjectSettings())

    with pytest.raises(typer.BadParameter) as excinfo:
        command_common_module._worker_role_provider_config("openrouter", None, None)

    assert "pass --model <model>" in str(excinfo.value)
    assert "wmo providers set --provider openrouter --model <model>" in str(excinfo.value)

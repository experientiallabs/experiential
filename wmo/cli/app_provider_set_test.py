"""Provider configuration and pool-registration tests."""

# ruff: noqa: F403, F405
from wmo.cli.app_fixtures_test import *


def test_config_help_does_not_reuse_the_harness_group_name() -> None:
    # `wmo harness` is a different group managing a different object; `wmo config` manages the
    # project's own settings file.
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "project-local wmo settings" in output
    assert "harness" not in output


def test_serve_help_names_the_openai_endpoint_and_a_real_example_root() -> None:
    # The OpenAI-compatible surface is what README step 3 exists for, and benchmark data now
    # arrives via `wmo download` - the help must name the endpoint and the real data root.
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "/v1/chat/completions" in output
    assert "examples/tau-bench" not in output
    assert "environment-capture-data/tau-bench" in output


def test_main_entry_loads_dotenv_before_dispatch(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # The persistence half of the wizard's credential flow: keys saved to .env must be back in
    # os.environ on the next `wmo` invocation (main), and importing the module must NOT load.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WMO_TEST_MAIN_VAR=loaded\n", encoding="utf-8")
    monkeypatch.delenv("WMO_TEST_MAIN_VAR", raising=False)
    monkeypatch.setattr(cli_app_module, "app", lambda: None)
    cli_app_module.main()
    assert os.environ["WMO_TEST_MAIN_VAR"] == "loaded"


def test_retry_narrator_dedupes_identical_failures_and_counts_down(monkeypatch) -> None:  # noqa: ANN001
    from rich.console import Console as RichConsole

    _RetryNarrator = catalog_module._RetryNarrator

    console = RichConsole(force_terminal=False, no_color=True, width=100)

    class Boto(Exception):
        def __init__(self, code: str) -> None:
            super().__init__("An error occurred (reached max retries: 1)")
            self.response = {"Error": {"Code": code, "Message": "Bedrock is unable"}}

    class FakeStatus:
        def __init__(self) -> None:
            self.updates: list[str] = []

        def update(self, text: str) -> None:
            self.updates.append(text)

    monkeypatch.setattr(catalog_module.time, "sleep", lambda _s: None)
    narrator = _RetryNarrator(console)
    status = FakeStatus()
    narrator.attach(status, "busy")
    with console.capture() as cap:
        narrator.on_retry(1, 3, 1.0, Boto("ServiceUnavailableException"))
        narrator.sleep(1.0)
        narrator.on_retry(2, 3, 3.0, Boto("ServiceUnavailableException"))  # same failure: silent
        narrator.sleep(3.0)
        narrator.on_retry(3, 3, 9.0, Boto("ThrottlingException"))  # different: printed
    out = cap.get()
    assert out.count("provider hiccup") == 2  # deduped consecutive identical failures
    assert "ServiceUnavailableException: Bedrock is unable" in out
    assert "reached max retries" not in out  # transport chatter stripped
    assert "retry 2/3 - waiting 3s…" in " ".join(status.updates)  # inline countdown
    assert status.updates[-1] == "busy"  # spinner text restored after the wait


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names
    assert "config" in group_names


def test_config_telemetry_command_manages_project_settings(tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"

    disabled = runner.invoke(app, ["config", "telemetry", "disable", "--root", str(root)])
    assert disabled.exit_code == 0, disabled.output
    assert "telemetry disabled" in disabled.output
    assert "enabled = false" in (root / "settings.toml").read_text(encoding="utf-8")

    status = runner.invoke(app, ["config", "telemetry", "--root", str(root)])
    assert status.exit_code == 0, status.output
    assert "telemetry disabled" in status.output

    enabled = runner.invoke(app, ["config", "telemetry", "enable", "--root", str(root)])
    assert enabled.exit_code == 0, enabled.output
    assert "telemetry enabled" in enabled.output


def test_providers_set_verifies_and_saves_local_worker(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    checked: list[ProviderConfig] = []

    def verify(configs: list[ProviderConfig]) -> list[VerifyResult]:
        checked.extend(configs)
        return [VerifyResult(ok=True, kind=config.kind, model=config.model) for config in configs]

    monkeypatch.setattr("wmo.common.providers.verify_all", verify)
    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--endpoint",
            "https://models.example/v1",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert checked[0].kind is ProviderKind.OPENAI
    worker = load_settings(root).models.worker
    assert worker is not None
    assert worker.provider == "openai"
    assert worker.model == "gpt-5.4-mini"
    assert worker.endpoint == "https://models.example/v1"


def test_providers_set_registers_pool_models_beside_the_settings_it_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The blocker this command exists to remove: nothing wrote `pool.toml` for an ordinary
    # provider model, so a router had no candidates without hand-authored TOML.
    _accept_every_provider(monkeypatch)
    _seed_openrouter_catalog(tmp_path, monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-4.5",
            "--pool-model",
            "anthropic/claude-sonnet-4.5",
            "--tier",
            "open",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    worker = load_settings(root).models.worker
    assert worker is not None and worker.provider == "openrouter"
    entries = read_pool_entries(root / "pool.toml")
    assert [(entry.name, entry.model, entry.tier) for entry in entries] == [
        ("claude-sonnet-4.5", "anthropic/claude-sonnet-4.5", "open")
    ]
    # Priced from the published catalog, so the roster never reports $0 for this candidate.
    assert entries[0].price().input_per_mtok == 3.0


def test_providers_set_registers_into_an_explicit_pool_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    roster = tmp_path / "rosters" / "candidates.toml"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "bedrock",
            "--model",
            "claude-opus-4-8",
            "--pool-model",
            "claude-haiku-4-5",
            "--pool",
            str(roster),
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(roster)[0]
    assert entry.kind is ProviderKind.BEDROCK
    # Resolved through the built-in registry, so the entry carries the callable runtime id.
    assert entry.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert entry.model_type == "claude-haiku-4-5"


def test_providers_set_refuses_a_pool_model_it_cannot_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A candidate with no price silently costs $0, and a cost-aware policy routes everything to
    # it. Non-interactively there is nobody to ask, so the command has to refuse.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "some-unlisted-model",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "no built-in price" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_prices_a_pool_model_from_the_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--endpoint",
            "https://vllm.example/v1",
            "--pool-model",
            "qwen3-32b",
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--api-key-env",
            "WMO_ENDPOINT_API_KEY",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(root / "pool.toml")[0]
    assert (entry.input_per_mtok, entry.output_per_mtok) == (0.1, 0.4)
    assert entry.endpoint == "https://vllm.example/v1"
    assert entry.api_key_env == "WMO_ENDPOINT_API_KEY"


def test_providers_set_rejects_an_unknown_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--tier",
            "cheap",
            "--root",
            str(tmp_path / ".wmo"),
        ],
    )

    assert result.exit_code != 0
    assert "frontier, open" in result.output


def test_providers_set_never_guesses_an_azure_deployment_for_the_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The worker config fills an Azure deployment in from the model id when none was given; a
    # pool entry must not inherit that guess, because Azure sends the deployment as the request
    # model and a guessed name addresses a route that does not exist.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "azure",
            "--model",
            "gpt-5.5",
            "--pool-model",
            "gpt-5.5",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "azure needs --deployment" in result.output
    assert not (root / "pool.toml").exists()
    # The worker role is still saved with its derived deployment: only the pool is strict.
    worker = load_settings(root).models.worker
    assert worker is not None and worker.deployment == "gpt-5.5"


def test_providers_set_registers_a_named_azure_deployment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "azure",
            "--model",
            "gpt-5.5",
            "--deployment",
            "chat-prod",
            "--api-version",
            "2025-01-01-preview",
            "--pool-model",
            "gpt-5.5",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = read_pool_entries(root / "pool.toml")[0]
    assert (entry.name, entry.model, entry.deployment) == ("chat-prod", "gpt-5.5", "chat-prod")
    assert entry.api_version == "2025-01-01-preview"


def test_providers_set_refuses_a_pool_model_that_cannot_be_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A --pool-model can differ from the verified worker in model, endpoint, deployment and
    # credential, so the worker's ping proves nothing about it. Registering it unproved would
    # surface as a 401 inside a paid `route sweep`, after the candidates ahead of it were billed.
    _accept_every_provider(monkeypatch)
    monkeypatch.setattr(
        pool_registry,
        "verify_pool_entry",
        lambda entry: VerifyResult(
            ok=False, kind=entry.kind, model=entry.model, detail="401 unauthorized"
        ),
    )
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "gpt-5.4",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "not callable" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_rejects_half_a_price_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pool entry prices both token tiers or neither. Half a pair would be dropped silently in
    # the interactive flow and rejected as an invalid entry with --pool-model, so it is refused
    # up front, where the message can name the flag.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--pool-model",
            "qwen3-32b",
            "--input-per-mtok",
            "0.1",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    assert "both --input-per-mtok and --output-per-mtok" in result.output
    assert not (root / "pool.toml").exists()


def test_providers_set_without_pool_flags_leaves_scripted_runs_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The pre-existing contract: `--provider` + `--model` prompts for nothing and writes only
    # settings. Registration is an addition, never something a script trips over.
    _accept_every_provider(monkeypatch)
    root = tmp_path / ".wmo"

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_settings(root).models.worker is not None
    assert not (root / "pool.toml").exists()
    assert "Register models" not in result.output


def test_providers_set_does_not_save_a_failed_provider(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    monkeypatch.setattr(
        "wmo.common.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="bad key",
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "providers",
            "set",
            "--provider",
            "openai",
            "--model",
            "gpt-5.4-mini",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 1
    assert "bad key" in result.output
    assert load_settings(root).models.worker is None

"""Provider verification CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.cli_fixtures_test import *
from wmo.common.config import save_config


def _write_model_config(root: Path, name: str, *, embed_model: str | None = None) -> HarnessConfig:
    """Persist a legacy world-model configuration for provider verification only.

    Provider verification still supports existing world-model artifacts. This fixture creates
    their configuration directly and never invokes the removed trace-to-world-model build path.
    """
    config = HarnessConfig(
        providers=[
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model_type="claude-opus-4-8",
                model="us.anthropic.claude-opus-4-8",
                embed_model=embed_model,
            )
        ],
        serve_provider=ProviderKind.BEDROCK,
        embed_provider=EmbedderKind.BEDROCK if embed_model is not None else EmbedderKind.HASHING,
    )
    save_config(config, root=root / "models" / name)
    return config


def test_providers_verify_unknown_model_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["providers", "verify", "--name", "ghost", "--root", str(tmp_path / ".wmo")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)


@_BROKEN_SETTINGS
@pytest.mark.parametrize(
    "argv",
    [
        ["providers", "verify"],
        ["config", "telemetry", "status"],
        ["config", "telemetry", "enable"],
        ["providers", "set", "--provider", "openai", "--model", "gpt-5.4"],
    ],
    ids=["verify", "telemetry-status", "telemetry-enable", "providers-set"],
)
def test_broken_settings_is_a_usage_error_not_a_traceback(
    tmp_path: Path, payload: str, expected: str, argv: list[str]
) -> None:
    root = _write_settings(tmp_path, payload)

    result = runner.invoke(app, [*argv, "--root", str(root)])

    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)  # BadParameter, not a leaked loader raise
    flat = _flat(result.output)
    assert expected in flat
    assert "settings.toml" in flat
    assert "delete it and re-run `wmo providers set`" in flat


@_BROKEN_SETTINGS
def test_providers_set_rejects_a_bad_provider_before_reading_settings(
    tmp_path: Path, payload: str, expected: str
) -> None:
    # The caller's own argument is wrong too; the error must be about the argument they typed.
    root = _write_settings(tmp_path, payload)

    result = runner.invoke(
        app, ["providers", "set", "--provider", "bogus", "--model", "x", "--root", str(root)]
    )

    assert result.exit_code == 2
    assert "unknown provider 'bogus'" in _flat(result.output)
    assert expected not in _flat(result.output)


def test_providers_verify_unreadable_model_config_is_clean_error(tmp_path: Path) -> None:
    # An artifact copied in by hand (or extracted from a bundle a newer CLI wrote) can hold a
    # config.toml this CLI cannot parse; the command whose job is reporting configuration
    # problems must report that one too.
    broken = tmp_path / ".wmo" / "models" / "foo"
    broken.mkdir(parents=True)
    (broken / "config.toml").write_text("this is not toml [[[\n", encoding="utf-8")

    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmo")])

    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    # Path separators and rich soft-wraps vary by OS; match the durable pieces.
    assert "foo" in flat and "config.toml" in flat.replace(" ", "")
    assert "is not valid TOML" in flat
    assert "re-run `wmo build`" in flat


def test_providers_verify_nothing_configured_is_actionable(tmp_path: Path) -> None:
    # Nothing to check at all is a usage problem, not a pass: say which command fixes it and
    # exit non-zero so a setup script does not read silence as "credentials are fine".
    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 1
    assert "nothing configured" in result.output
    assert "wmo providers set" in result.output


def test_providers_verify_without_a_world_model_checks_settings_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The repro this fixes: verifying credentials is what you do BEFORE `wmo build` (which
    # aborts outright on bad ones), so an unbuilt project must still check what it has.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="openai", model="gpt-5.4-mini")
    settings.models.judge = ModelRole(provider="bedrock", model="claude-opus-4-8")
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert not (root / "models").exists()
    # Both configured roles are pinged, the bedrock one at its runtime id (settings hold the
    # canonical type), and each line names the role that asked for it.
    assert [(c.kind, c.model) for c in pinged] == [
        (ProviderKind.OPENAI, "gpt-5.4-mini"),
        (ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-8"),
    ]
    assert "ok openai (gpt-5.4-mini) (models.worker)" in result.output
    assert "models.judge" in result.output
    # The embed path belongs to a built model: skipped with a note, not fatal.
    assert "embed path: skipped" in result.output


def test_providers_verify_reports_a_role_failure_with_its_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="bedrock", model="claude-opus-4-8")
    save_settings(settings, root)
    monkeypatch.setattr(
        "wmo.common.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=c.kind,
                model=c.model,
                # Rich markup in raw provider error text must not be interpreted.
                detail="denied [foo]",
            )
            for c in configs
        ],
    )

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "fail bedrock" in result.output
    assert "[foo]" in result.output
    assert "AWS_ACCESS_KEY_ID" in result.output


def test_providers_verify_missing_optional_sdk_points_at_the_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # tinker's SDK is an optional extra, and its ImportError text replaces the "No module named"
    # wording the hint used to key on, so the hint said "check your credentials" on a failure
    # that has nothing to do with credentials.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="tinker", model="Qwen/Qwen3-8B")
    save_settings(settings, root)
    monkeypatch.setattr(
        "wmo.common.providers.verify_all",
        lambda configs: [
            VerifyResult(ok=False, kind=c.kind, model=c.model, detail=_MISSING_TINKER_EXTRA)
            for c in configs
        ],
    )

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    flat = _flat(result.output)
    # pip is the documented install path, so the extra must be reachable without a checkout,
    # and the `[distill]` must survive rich markup rather than being read as a style tag.
    assert "pip install 'world-model-optimizer[distill]'" in flat
    assert "credentials are set" not in flat


def test_providers_verify_reports_existing_model_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    _write_model_config(root, "airline")
    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # The bedrock provider configured at build time shows up in the verify report.
    assert "bedrock" in result.output


def test_providers_verify_checks_an_existing_model_embed_path(
    patched_provider: None, tmp_path: Path
) -> None:
    # A provider-backed embedder is verified alongside the completion provider.
    root = tmp_path / ".wmo"
    _write_model_config(root, "airline", embed_model="amazon.titan-embed-text-v2:0")

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "embed:bedrock (amazon.titan-embed-text-v2:0)" in result.output
    assert "embed path: skipped" not in result.output


def test_providers_verify_pings_a_role_shared_with_an_existing_model_once(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Dedup spans both sources, so a role sharing a persisted model does not bill a second ping.
    root = tmp_path / ".wmo"
    built = _write_model_config(root, "airline").providers[0]
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider=built.kind.value, model=built.model)
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [(c.kind, c.model) for c in pinged] == [(built.kind, built.model)]
    assert "(airline, models.worker)" in result.output


def test_providers_verify_does_not_collapse_two_regions_of_one_model(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same kind and model, different region: two different backends that fail independently.
    # Collapsing them would ping one and report the OTHER as verified, which is a false pass on
    # exactly the credential question this command exists to answer.
    root = tmp_path / ".wmo"
    built = _write_model_config(root, "airline").providers[0]
    assert built.region is None
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider=built.kind.value, model=built.model, region="eu-west-1"
    )
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [c.region for c in pinged] == [None, "eu-west-1"]


def test_providers_verify_does_not_collapse_two_azure_deployments(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two roles on one Azure model, each behind its own operator-named deployment and endpoint.
    root = tmp_path / ".wmo"
    settings = load_settings(root)
    settings.models.worker = ModelRole(
        provider="azure", model="gpt-5.5", endpoint="https://a.example", deployment="dep-a"
    )
    settings.models.judge = ModelRole(
        provider="azure", model="gpt-5.5", endpoint="https://b.example", deployment="dep-b"
    )
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [(c.endpoint, c.deployment) for c in pinged] == [
        ("https://a.example", "dep-a"),
        ("https://b.example", "dep-b"),
    ]


def test_providers_verify_checks_both_embed_models_on_one_backend(
    patched_provider: None, tmp_path: Path
) -> None:
    # Two persisted artifacts share a completion backend but embed through different models: one
    # completion ping, and BOTH embed paths are checked (embed_model is what the call sends).
    root = tmp_path / ".wmo"
    for model_name, embed_model in (
        ("a", "amazon.titan-embed-text-v1"),
        ("b", "amazon.titan-embed-text-v2:0"),
    ):
        _write_model_config(root, model_name, embed_model=embed_model)

    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "embed:bedrock (amazon.titan-embed-text-v1)" in result.output
    assert "embed:bedrock (amazon.titan-embed-text-v2:0)" in result.output
    # The shared completion provider is still pinged once, under both model names.
    assert result.output.count("ok bedrock (") == 1


def test_providers_verify_name_scopes_the_report_to_one_world_model(
    patched_provider: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `--name` answers "is THIS model's provider reachable?"; pulling the project's roles in
    # would bill for a question the caller did not ask.
    root = tmp_path / ".wmo"
    _write_model_config(root, "airline")
    settings = load_settings(root)
    settings.models.worker = ModelRole(provider="openai", model="gpt-5.4-mini")
    save_settings(settings, root)
    pinged: list[ProviderConfig] = []
    _record_verify_all(monkeypatch, pinged)

    result = runner.invoke(app, ["providers", "verify", "--name", "airline", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert [c.kind for c in pinged] == [ProviderKind.BEDROCK]
    assert "models.worker" not in result.output

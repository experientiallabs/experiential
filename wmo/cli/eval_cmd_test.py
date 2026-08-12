"""Open-loop evaluation CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.cli_fixtures_test import *


def test_eval_trace_file_command_still_scores(patched_provider, tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        ["eval", _traces_file(tmp_path), "--no-rag"],
    )

    assert result.exit_code == 0, result.output
    assert "OVERALL" in result.output
    assert "fidelity=0.500" in result.output


def test_eval_pins_the_judge_off_the_failover_chain(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    # World-model calls may fail over (provider_or_chain); the judge is the metric and must stay
    # pinned to the single requested backend - a judge that silently switches models mid-run
    # makes fidelity numbers incomparable.
    import wmo.common.providers as providers_pkg

    chain = FakeProvider()
    pinned = FakeProvider()
    configs: list[ProviderConfig] = []

    def provider_or_chain(config: ProviderConfig, **kw) -> FakeProvider:  # noqa: ANN003
        configs.append(config)
        return chain

    def get_provider(config: ProviderConfig) -> FakeProvider:
        configs.append(config)
        return pinned

    monkeypatch.setattr(providers_pkg, "provider_or_chain", provider_or_chain)
    monkeypatch.setattr(providers_pkg, "get_provider", get_provider)
    traces = _traces_file(tmp_path)
    # No settings.toml here, so the asserted bedrock ids are the no-role-configured fallback.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", traces, "--no-rag"])

    assert result.exit_code == 0, result.output
    judge_systems_chain = [s for s in chain.systems if "grade a world model" in s]
    judge_systems_pinned = [s for s in pinned.systems if "grade a world model" in s]
    assert judge_systems_chain == []  # the chain never judges
    assert judge_systems_pinned  # every judge call went to the pinned backend
    prediction_systems = [s for s in chain.systems if "grade a world model" not in s]
    assert prediction_systems  # predictions went through the chain
    assert [config.model for config in configs] == [
        "us.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
    ]
    assert all(config.model_type == "claude-opus-4-8" for config in configs)


def test_eval_uses_configured_worker_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `wmo providers set` (step 1 of getting started) writes [models.worker]. eval used to score
    # against a hardcoded bedrock/claude-opus-4-8 regardless, so an OpenAI-only project got a
    # 0.000 fidelity at exit 0 from a provider it never configured.
    traces = _traces_file(tmp_path)
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    seen = _record_eval_providers(monkeypatch)

    result = runner.invoke(app, ["eval", traces, "--no-rag"])

    assert result.exit_code == 0, result.output
    assert {config.kind for config in seen} == {ProviderKind.OPENAI}
    assert {config.model for config in seen} == {"gpt-5.4-mini"}
    # The report is only comparable across runs on the same model, so eval names the backend.
    assert "scoring with openai (gpt-5.4-mini)" in result.output


def test_eval_provider_flag_overrides_the_configured_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traces = _traces_file(tmp_path)
    _write_worker_role(tmp_path / ".wmo", "openai", "gpt-5.4-mini")
    monkeypatch.chdir(tmp_path)
    seen = _record_eval_providers(monkeypatch)

    result = runner.invoke(app, ["eval", traces, "--no-rag", "--provider", "bedrock"])

    assert result.exit_code == 0, result.output
    assert {config.kind for config in seen} == {ProviderKind.BEDROCK}
    # A --provider naming another backend drops the role's model: gpt-5.4-mini is not on bedrock.
    assert {config.model for config in seen} == {"us.anthropic.claude-opus-4-8"}


def test_eval_out_parent_is_created_before_the_eval_runs(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # The report is written last; a missing parent used to blow up with FileNotFoundError AFTER
    # the (paid) eval had finished, discarding it.
    destination = tmp_path / "nodir" / "deeper" / "report.json"

    result = runner.invoke(
        app, ["eval", _traces_file(tmp_path), "--no-rag", "--out", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert destination.exists()
    assert "Traceback" not in result.output


def test_eval_out_pointing_at_a_directory_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["eval", _traces_file(tmp_path), "--no-rag", "--out", str(tmp_path)]
    )

    assert result.exit_code == 2  # usage error, not an IsADirectoryError traceback
    assert "is a directory" in _flat(result.output)


def test_eval_on_a_directory_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    corpus_dir = tmp_path / "benchmark"
    corpus_dir.mkdir()

    result = runner.invoke(app, ["eval", str(corpus_dir)])

    assert result.exit_code == 2  # usage error, not an IsADirectoryError traceback
    flat = _flat(result.output)
    assert "is a directory" in flat
    # Rich may soft-wrap long Windows paths mid-token (`traces.otel.j` / `sonl`); strip spaces.
    assert "traces.otel.jsonl" in flat.replace(" ", "")  # names the file to pass instead


def test_eval_file_with_no_traces_fails_instead_of_scoring_zero(tmp_path) -> None:  # noqa: ANN001
    # A tasks.jsonl (or any non-OTel export) used to print a plausible
    # "OVERALL fidelity=0.000 over 0 held-out steps" scorecard and exit 0.
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id": "t1", "instruction": "do it"}\n', encoding="utf-8")

    result = runner.invoke(app, ["eval", str(tasks)])

    assert result.exit_code == 2, result.output
    flat = _flat(result.output)
    assert "no OTel GenAI traces" in flat
    assert "--mode closed-loop" in flat
    assert "OVERALL" not in flat


def test_eval_unknown_chain_is_a_usage_error(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)  # no .wmo/fallback.toml here

    result = runner.invoke(app, ["eval", _traces_file(tmp_path), "--chain", "nope"])

    assert result.exit_code == 2  # usage error, not a ValueError traceback
    assert "fallback.toml does not exist" in _flat(result.output)


def test_eval_rejects_closed_loop_only_flags_in_open_loop(tmp_path) -> None:  # noqa: ANN001
    # The README's closed-loop command minus `--mode closed-loop` used to silently drop every
    # closed-loop flag and run a different (paid) evaluation.
    result = runner.invoke(
        app,
        [
            "eval",
            _traces_file(tmp_path),
            "--harness",
            "nosuchharness",
            "--k",
            "7",
            "--harness-backend",
            "e2b",
        ],
    )

    assert result.exit_code == 2, result.output
    flat = _flat(result.output)
    assert "--k, --harness, --harness-backend" in flat
    assert "--mode closed-loop" in flat


def test_eval_threshold_belongs_to_the_agreement_flow(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["eval", _traces_file(tmp_path), "--threshold", "0.9"])

    assert result.exit_code == 2, result.output
    assert "wmo eval agreement" in _flat(result.output)


def test_eval_help_lists_every_dispatched_flow() -> None:
    result = runner.invoke(app, ["eval", "--help"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    for token in ("open-loop", "closed-loop", "agreement"):
        assert token in flat

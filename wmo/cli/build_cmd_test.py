"""Build-input and model-catalog CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.cli_fixtures_test import *


def test_build_then_list_shows_named_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmo"
    _build(root, "tau2-airline", tmp_path)

    # The artifact lands under <root>/models/<name>/.
    assert (root / "models" / "tau2-airline" / "config.toml").exists()

    listed = runner.invoke(app, ["list", "--root", str(root)])
    assert listed.exit_code == 0, listed.output
    assert "tau2-airline" in listed.output


def test_list_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 0
    assert "no world models" in result.output
    # --root defaults to a cwd-relative `.wmo`, so "nothing built" and "wrong directory" read
    # the same unless the empty listing says where it looked. Asserted on the tail of the path
    # only: rich wraps a long tmp_path across lines.
    flat = _flat(result.output)
    assert "no world models built under" in flat
    assert str(Path(".wmo") / "models") in flat


def test_list_rejects_a_file_as_root(tmp_path) -> None:  # noqa: ANN001
    # `--root traces.jsonl` used to report a healthy empty project; a file can never hold models/.
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    result = runner.invoke(app, ["list", "--root", str(corpus)])
    assert result.exit_code == 2
    assert "is a file, not a project dir" in _flat(result.output)


def test_list_shows_an_unreadable_artifact_as_a_row(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # One artifact this CLI cannot parse (a bundle from a newer CLI, a hand edit) used to
    # traceback the whole listing; the healthy models beside it must still be listed.
    root = tmp_path / ".wmo"
    _build(root, "alpha-healthy", tmp_path)
    broken = root / "models" / "zz-broken"
    broken.mkdir(parents=True)
    (broken / "config.toml").write_text("this is not toml =", encoding="utf-8")

    result = runner.invoke(app, ["list", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert result.exception is None  # a bad row, not an escaped TOMLDecodeError
    assert "alpha-healthy" in result.output
    assert "unreadable" in result.output
    assert "zz-broken" in result.output
    assert "is not valid TOML" in _flat(result.output)


def test_the_model_picker_offers_only_readable_artifacts(
    patched_provider: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `list_info` now hands back a row for an artifact it could not read, so the picker has to
    # drop those rather than offer a choice that dead-ends the moment it is picked.
    root = tmp_path / ".wmo"
    _build(root, "alpha-healthy", tmp_path)
    _build(root, "beta-healthy", tmp_path)
    _write_broken_model(root, "zz-broken")
    offered: list[str] = []

    def fake_select_model(console: object, infos: list[ModelInfo]) -> str:
        offered.extend(info.name for info in infos)
        return infos[0].name

    monkeypatch.setattr(catalog_module, "_console", SimpleNamespace(is_terminal=True))
    monkeypatch.setattr("wmo.cli.ui.select_model", fake_select_model)

    assert catalog_module._resolve_name(WorldModelStore(root), None) == "alpha-healthy"
    assert offered == ["alpha-healthy", "beta-healthy"]


def test_the_model_picker_reports_when_nothing_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmo"
    for name in ("one-broken", "two-broken"):
        _write_broken_model(root, name)
    monkeypatch.setattr(catalog_module, "_console", SimpleNamespace(is_terminal=True))

    with pytest.raises(typer.BadParameter, match="no readable world model"):
        catalog_module._resolve_name(WorldModelStore(root), None)


def test_build_interactive_wizard_creates_model(
    patched_provider,  # noqa: ANN001 - pytest fixture
    tmp_path,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
) -> None:
    root = tmp_path / ".wmo"
    for var in ("AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(var, "test-cred")  # creds present: no interactive key prompts
    # --interactive forces the wizard even under CliRunner (non-TTY); feed each answer line in
    # prompt order: name, trace source (select), file, provider (select), model (select), region
    # (bedrock only), judge model (select), budget, embedder (select). The offline 'hashing'
    # embedder skips the embed-model prompt; phi dim isn't prompted. Selects pick by index.
    answers = "\n".join(
        [
            "wizard-built",
            "",  # trace source: accept the default (otel-genai)
            _traces_file(tmp_path),
            "3",  # provider: bedrock (order: openai, anthropic, bedrock, azure, ...)
            "1",  # model: us.anthropic.claude-opus-4-8
            "us-east-1",
            "",  # judge model: accept the bedrock default (dated haiku)
            "1",  # fidelity: low (RAG only)
            "1",  # embedder: hashing
        ]
    )
    result = runner.invoke(
        app, ["build", "--interactive", "--root", str(root)], input=answers + "\n"
    )
    assert result.exit_code == 0, result.output
    assert (root / "models" / "wizard-built" / "config.toml").exists()


def test_build_non_interactive_without_source_errors(tmp_path) -> None:  # noqa: ANN001
    # No --file/--vendor and --no-interactive: should fail fast rather than hang on input.
    result = runner.invoke(app, ["build", "--no-interactive", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code != 0


def test_build_with_a_name_but_no_trace_source_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    # `wmo list` prints `wmo build --name <name>` as its empty-state hint, and every non-TTY
    # (CI, piped output) takes the scriptable path. It used to reach the ingest seam and raise
    # a raw ValueError; the guard only fired when --name was ALSO omitted.
    result = runner.invoke(
        app, ["build", "--name", "x", "--root", str(tmp_path / ".wmo"), "--no-interactive"]
    )
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "provide --file <export> or --pull" in _flat(result.output)


def test_list_empty_state_names_a_trace_export(tmp_path) -> None:  # noqa: ANN001
    # The hint must be a runnable command: --name alone is a usage error (test above).
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmo")])
    assert result.exit_code == 0
    assert "--file" in result.output


def test_build_missing_trace_file_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    # A typo'd --file used to die with a raw FileNotFoundError from the adapter, and only after
    # the provider ping had already run.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(tmp_path / "nope.jsonl"),
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, FileNotFoundError)
    assert "trace file not found" in _flat(result.output)
    # Rejected at the argument boundary: no provider was pinged.
    assert "verifying" not in result.output


def test_build_rejects_a_directory_as_the_trace_file(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(tmp_path),
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, IsADirectoryError)
    assert "not a directory" in _flat(result.output)


def test_build_rejects_the_postgres_source_and_names_wmo_ingest(tmp_path) -> None:  # noqa: ANN001
    # `postgres` passes the adapter-name validator but can never work here: build has no
    # --dsn/--table (those live on `wmo ingest`), so it must be rejected at the boundary.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--source",
            "postgres",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "wmo ingest --source postgres --dsn <dsn> --table <table>" in flat
    assert "--dsn" not in _flat(runner.invoke(app, ["build", "--help"]).output)


def test_build_wrong_source_names_the_detected_format(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # A chat-json export under the silent `--source otel-genai` default ingested nothing and
    # raised ValueError('no traces ingested; nothing to build') as a traceback.
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(chat),
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "--source otel-genai" in flat
    assert "it looks like chat-json" in flat
    # The path itself is wrapped by rich, so assert on the command, not the rendered path.
    assert "wmo ingest --file" in flat


def test_build_empty_trace_file_names_source_and_ingest(patched_provider, tmp_path) -> None:  # noqa: ANN001
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            str(empty),
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "--source otel-genai" in flat
    assert "wmo ingest --file" in flat


def test_build_limit_caps_a_file_corpus(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # --limit was wired only into the VendorPull branch, so it was silently ignored for --file
    # builds while `wmo ingest --limit` capped both transports.
    from wmo.common.config.card import load_card

    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "capped",
            "--file",
            _many_traces_file(tmp_path, 6),
            "--limit",
            "2",
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output
    card = load_card(root / "models" / "capped")
    assert card is not None
    assert card.corpus.traces == 2


def test_build_pull_limit_is_a_fetch_cap_applied_once(
    patched_provider,  # noqa: ANN001 - pytest fixture
    monkeypatch,  # noqa: ANN001 - pytest fixture
    tmp_path,  # noqa: ANN001 - pytest fixture path
) -> None:
    """A pull spends `--limit` vendor-side, so `--drop-degenerate` can leave fewer than N.

    `wmo.simulation.ingest.base.from_vendor` slices to `pull.limit` before `build` sees the
    corpus, so re-applying the same cap after the degenerate filter cannot restore dropped traces;
    it would only read as a promise of N usable traces that this transport cannot keep. Pinning
    both halves: the adapter receives the cap, and `build` is not handed it a second time.
    """
    from wmo.common.config.card import load_card

    seen: list[VendorPull] = []
    passed_to_build: dict[str, object] = {}

    class _CappingAdapter:
        """Mimics `base.from_vendor`: alternating junk/usable traces, sliced at `pull.limit`."""

        name = "otel-genai"

        def from_vendor(self, pull: VendorPull) -> list[Trace]:
            seen.append(pull)
            traces = [_pull_trace(f"{i:032d}", usable=bool(i % 2)) for i in range(6)]
            return traces if pull.limit is None else traces[: pull.limit]

    # `wmo.simulation.model.build` is shadowed by the `build` function re-exported from
    # `wmo.simulation.model.__init__`, so attribute access and import statements resolve the
    # function.
    # Reach the submodule only through importlib / sys.modules.
    import importlib

    engine_build = importlib.import_module("wmo.simulation.model.build")
    real_run_build = engine_build.build

    def _spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - passthrough spy
        passed_to_build.update(kwargs)
        return real_run_build(*args, **kwargs)

    monkeypatch.setattr(engine_build, "get_adapter", lambda name: _CappingAdapter())
    monkeypatch.setattr(engine_build, "build", _spy)
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "pulled",
            "--pull",
            "--limit",
            "4",
            "--drop-degenerate",
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [p.limit for p in seen] == [4]  # spent at fetch…
    assert passed_to_build["limit"] is None  # …and not a second time after the filter
    card = load_card(root / "models" / "pulled")
    assert card is not None
    assert card.corpus.traces == 2  # 4 fetched, 2 of them degenerate; the cap cannot refill


def test_build_rejects_a_limit_below_one(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--limit",
            "0",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert "--limit must be at least 1" in _flat(result.output)


def test_build_unknown_chain_is_a_usage_error(tmp_path) -> None:  # noqa: ANN001
    """`--chain` with no `.wmo/fallback.toml` must say how to create the file, not traceback.

    Deliberately no `patched_provider`: that fixture stubs `providers.provider_or_chain`, which
    is the seam under test. Chain resolution runs before the provider ping, so nothing here
    reaches the network (`wmo/conftest.py` points the chain path at an empty tmp dir).
    """
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--chain",
            "fast",
            "--root",
            str(tmp_path / ".wmo"),
            "--provider",
            "bedrock",
            "--fidelity",
            "low",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert not isinstance(result.exception, ValueError)
    flat = _flat(result.output)
    assert "chain 'fast' requested but" in flat
    assert "fallback.toml" in flat
    assert "[[chain.<name>]] rung tables" in flat
    assert "docs/reference/failover.md" in flat


def test_build_model_default_follows_the_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # --provider openai with no --model used to persist the Anthropic id `claude-opus-4-8`.
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "oa",
            "--file",
            _traces_file(tmp_path),
            "--provider",
            "openai",
            "--fidelity",
            "low",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    config = load_config(root / "models" / "oa")
    assert config.serve_provider is ProviderKind.OPENAI
    assert config.serve_provider_config().model_type == "gpt-5.5"


def test_build_requires_a_model_for_a_provider_without_a_default(tmp_path) -> None:  # noqa: ANN001
    # openrouter/tinker/openai_responses have no curated model list: ask rather than guess,
    # matching `wmo providers set`'s scriptable contract.
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            "x",
            "--file",
            _traces_file(tmp_path),
            "--provider",
            "openrouter",
            "--root",
            str(tmp_path / ".wmo"),
            "--no-interactive",
        ],
    )
    assert result.exit_code == 2
    assert "--provider openrouter has no default serve model" in _flat(result.output)


def test_build_aborts_when_provider_sdk_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A missing SDK must abort the build before any rollouts, with the `uv sync` extra hint.

    Regression: previously the ModuleNotFoundError was swallowed inside GEPA and the build
    "succeeded" with a useless held-out-0.0 model.
    """
    from wmo.common.providers.base import VerifyResult

    monkeypatch.setattr(
        "wmo.common.providers.verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="No module named 'boto3'",
            )
        ],
    )
    root = tmp_path / ".wmo"
    result = runner.invoke(
        app, ["build", "--name", "x", "--file", _traces_file(tmp_path), "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "run `uv sync` to install the provider SDKs" in result.output
    # Aborted before building: no artifact written.
    assert not (root / "models" / "x" / "config.toml").exists()

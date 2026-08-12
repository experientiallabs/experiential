"""Route sweep preflight and accounting tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_sweep_checks_candidate_credentials_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pool_provider` reads `api_key_env` per cell, so an unset variable on the SECOND candidate
    # used to abort mid-sweep with a raw ValueError after the first was fully paid for.
    seams = _patch_seams(monkeypatch)
    monkeypatch.delenv("WMO_TEST_MISSING_KEY", raising=False)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "pricey"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        'api_key_env = "WMO_TEST_MISSING_KEY"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    assert "WMO_TEST_MISSING_KEY" in flat and "pricey" in flat
    # Nothing was paid for: the check runs before the cost table, so no candidate was ever called
    # and no episode ever opened. (The pre-flight does construct the candidates it can, which is
    # free; `cheap` resolves, `pricey` never gets that far.)
    assert seams.built_providers == ["cheap-1"]
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_constructs_every_backend_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A credential check is not an availability check. `TinkerChatProvider` REFUSES an explicit
    # api_key (it authenticates through the shared service client), and nothing rejects that
    # combination at load, so the failure used to land at this candidate's FIRST CELL: after
    # `cheap` had run every scenario and been paid for, as a raw traceback, with no matrix.
    # `real_kinds` builds the real tinker backend here; `cheap` stays faked, and no candidate is
    # ever called, so the test makes no network request either way.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.TINKER}))
    monkeypatch.setenv("WMO_TEST_TINKER_KEY", "sk-present")
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "student"\n'
        'kind = "tinker"\n'
        'model = "Qwen/Qwen3-8B"\n'
        'api_key_env = "WMO_TEST_TINKER_KEY"\n'  # set, so this is NOT a credential failure
        "input_per_mtok = 0.1\n"
        "output_per_mtok = 0.2\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # Names the offending entry AND its kind, so the pool file is editable from the message, plus
    # what to do about it (the backend's own advice: drop api_key_env, export TINKER_API_KEY).
    assert "'student'" in flat and "kind=tinker" in flat
    assert "TINKER_API_KEY" in flat and "dropapi_key_env" in flat
    # Not one cell was paid for: the cost table never printed and no episode ever opened.
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_rejects_a_config_no_backend_could_use_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The STATIC half of the pre-flight. An azure entry with a deployment but no `api_version`
    # loads (the load-time rule only covers `deployment`) and CONSTRUCTS (the provider's __init__
    # just stores the config); only `_get_client` refuses without an api-version, and that runs
    # inside this candidate's first call, after `cheap` has been paid for. Nothing about this needs
    # an SDK or a credential, so it is knowable from the entry alone.
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "gpt-azure"\n'
        'kind = "azure"\n'
        'model = "gpt-5.5"\n'
        'deployment = "gpt-5.5"\n'  # present, so this is not the load-time rule
        'endpoint = "https://example.openai.azure.com"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # Names the offending entry, its kind, and the field to add.
    assert "'gpt-azure'" in flat and "kind=azure" in flat and "api_version" in flat
    # Before the cost confirmation, and before any cell: the cost table never printed, no candidate
    # was ever called, no episode was ever opened, and no matrix exists.
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_builds_every_lazy_client_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The LAZY-CLIENT half of the pre-flight. Constructing an `OpenAIProvider` does not read its
    # credential: `__init__` only stores the config, and `OpenAI()` (which REFUSES to construct
    # without a resolvable key) is built inside the first call. So with OPENAI_API_KEY unset, a
    # pre-flight that only constructs providers passes and the whole sweep then fails cell by cell.
    # Both candidates are the real backend here; no request is made either way, because the SDK
    # raises while building its own client.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.OPENAI}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = _project(tmp_path, traces=_corpus())
    out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    # EVERY unusable candidate is named with its kind, not just the first one an operator would fix.
    assert "'cheap'" in flat and "'pricey'" in flat and "kind=openai" in flat
    assert "OPENAI_API_KEY" in flat  # the SDK's own advice survives into the message
    assert "USD(est)" not in flat
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_rejects_an_openrouter_candidate_with_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OpenRouter joined the pre-flight late: it shipped with no `prepare` seam, so
    # `prepare_pool_provider` skipped it and an unset key landed at that candidate's FIRST CELL,
    # after `cheap` had run every scenario and been paid for. Unlike bedrock's credential this one
    # IS locally knowable: `OpenRouterProvider._get_client` resolves the key itself and refuses,
    # opening no connection, so no request is made either way.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.OPENROUTER}))
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "router"\n'
        'kind = "openrouter"\n'
        'model = "z-ai/glm-4.6"\n'
        "input_per_mtok = 0.1\n"
        "output_per_mtok = 0.2\n",
        encoding="utf-8",
    )

    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    assert "'router'" in flat and "kind=openrouter" in flat
    assert "USD(est)" not in flat  # refused before the cost confirmation
    assert seams.world_model.tasks == []  # zero cells ran, so `cheap` was never paid for
    assert not out.exists()


def test_route_sweep_rejects_a_bedrock_candidate_whose_region_resolves_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bedrock is the backend whose client CANNOT be built in a pre-flight (boto3 resolves
    # credentials by walking a chain that reaches the instance-metadata endpoint, and builds fine
    # with no credentials anyway), so its region is resolved through boto3's own session instead:
    # entry, then AWS_DEFAULT_REGION, then the active profile. Without that check, botocore's
    # NoRegionError lands in this candidate's first cell. Every source is pointed at nothing here so
    # the check has the same answer on any machine, and metadata lookups are disabled as a belt:
    # this test may not touch the network.
    seams = _patch_seams(monkeypatch, real_kinds=frozenset({ProviderKind.BEDROCK}))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "cheap"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "opus-bedrock"\n'
        'kind = "bedrock"\n'
        'model = "us.anthropic.claude-opus-4-8"\n'  # no region anywhere
        "input_per_mtok = 15.0\n"
        "output_per_mtok = 75.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    flat = _flat(result.output)
    assert "'opus-bedrock'" in flat and "kind=bedrock" in flat
    assert "region" in flat and "AWS_DEFAULT_REGION" in flat  # what went wrong and what to do
    assert "USD(est)" not in flat
    assert seams.systems == []
    assert seams.world_model.tasks == []
    assert not out.exists()


def test_route_sweep_states_what_the_preflight_cannot_know_before_the_first_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two backends keep a residual gap because closing it needs a request (bedrock AWS credentials,
    # tinker service reachability). A usable bedrock entry therefore passes the pre-flight and the
    # sweep runs, and the command says which entry still carries which unknown rather than leaving
    # an operator to find out mid-sweep. Faked provider construction: nothing is called for real.
    seams = _patch_seams(monkeypatch)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")  # so the region check passes
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "opus-bedrock"\n'
        'kind = "bedrock"\n'
        'model = "us.anthropic.claude-opus-4-8"\n'
        "input_per_mtok = 15.0\n"
        "output_per_mtok = 75.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "1",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _says(result.output, "opus-bedrock (kind=bedrock): AWS credentials")
    assert _says(result.output, "the pre-flight makes no request")
    assert seams.world_model.tasks == ["task tr-010"]  # the sweep did run


def test_route_sweep_checks_the_out_path_before_it_spends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `OutcomeMatrix.save` mkdirs the parent at the END of the sweep, so a parent component that
    # is a regular file used to discard every cell already paid for with a bare OS error.
    seams = _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file, not a directory", encoding="utf-8")
    out = blocker / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(_pool_file(tmp_path)),
            "--out",
            str(out),
            "--scenarios",
            "2",
            "--yes",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)  # no traceback
    assert _says(result.output, "cannot write the outcome matrix")
    assert seams.world_model.tasks == []  # no episode ever opened
    # The check is pure: an --out it refuses is not half-created on the way out.
    assert blocker.is_file() and not out.exists()


def test_route_sweep_prints_names_and_paths_rich_cannot_swallow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pool names are free-form operator strings and --out is a path: both reach a rich console,
    # where `[a]` is markup. Unescaped, the cost table showed two candidates as one name and the
    # handoff line printed a path that does not exist.
    _patch_seams(monkeypatch)
    root = _project(tmp_path, traces=_corpus())
    pool = tmp_path / "pool.toml"
    pool.write_text(
        "[[model]]\n"
        'name = "gpt[a]"\n'
        'kind = "openai"\n'
        'model = "cheap-1"\n'
        "input_per_mtok = 1.0\n"
        "output_per_mtok = 2.0\n"
        "\n"
        "[[model]]\n"
        'name = "gpt[/bold]"\n'
        'kind = "openai"\n'
        'model = "pricey-1"\n'
        "input_per_mtok = 10.0\n"
        "output_per_mtok = 20.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "[run1]" / "matrix.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--root",
            str(root),
            "--pool",
            str(pool),
            "--out",
            str(out),
            "--scenarios",
            "1",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    # Both candidates are distinguishable in the table the operator confirms spend from, and the
    # closing-tag name no longer aborts the command before any episode runs.
    assert "gpt[a]" in flat and "gpt[/bold]" in flat
    # The printed path is the path the matrix is actually at, so it can be copied.
    assert out.is_file()
    assert _says(result.output, str(out))
    assert _says(result.output, f"wmo optimize route fit {out} --kind knn")


def test_route_sweep_persists_the_world_models_own_spend_as_a_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line "metered separately" now has somewhere to point.

    The world model opens one metered session per episode and `WorldModelEnv.close` leaves that
    session's final record on the env; before this every one of them died there, so the sweep
    said the simulator's cost was accounted for elsewhere and nothing anywhere held it. They roll
    into one `kind="sweep"` record in the model's own runs dir, beside build and serve.
    """
    _patch_seams(monkeypatch, session_usd=0.03)
    root = _project(tmp_path, traces=_corpus())
    _out, result = _sweep(tmp_path, root, "support", "--scenarios", "2", "--yes")
    assert result.exit_code == 0, result.output

    sweeps = [r for r in load_runs(root / "models" / "support" / "runs") if r.kind == "sweep"]
    assert len(sweeps) == 1
    # 2 candidates x 2 scenarios x 1 episode = 4 sessions at $0.03 each.
    assert sweeps[0].total.cost_usd == pytest.approx(0.12)
    assert sweeps[0].by_phase[Phase.SERVE].calls == 4

    # Both sides are printed, and the candidate line is untouched: they are different money and
    # a single blended number would misprice both.
    assert _says(result.output, "measured candidate spend")
    assert _says(result.output, "measured world-model spend $0.1200 over 4 session(s)")
    assert _says(result.output, "eval infrastructure, not serving cost")

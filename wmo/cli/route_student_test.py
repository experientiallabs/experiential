"""Route student and pin CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_student_makes_a_trained_adapter_routable(tmp_path: Path) -> None:
    """The keystone: a run dir becomes a loadable pool candidate with no hand-edited TOML."""
    pool_file = tmp_path / "pool.toml"

    result = _add_student(tmp_path, pool_file)

    assert result.exit_code == 0, result.output
    entry = load_pool(pool_file).entry("student")
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "tinker://fake/sampler/final/0"
    assert entry.model_type == "Qwen/Qwen3-8B"
    assert entry.chat_max_tokens_field == "max_tokens"
    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.price().input_per_mtok == 0.1


def test_route_student_requires_prices(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "student", str(_run_dir(tmp_path)), "--pool", str(tmp_path / "p")],
    )
    assert result.exit_code != 0
    assert "--input-per-mtok" in result.output


def test_route_student_names_the_missing_model_card(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-run"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(empty),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "model_card.json" in result.output
    assert "adapter version directory" in result.output  # says what to pass instead


def test_route_student_rejects_a_run_with_no_trained_weights(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, sampler="Qwen/Qwen3-8B")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(run_dir),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "tinker://" in result.output


def test_route_student_declining_the_replacement_leaves_the_pool_alone(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    before = pool_file.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "9.9",
            "--output-per-mtok",
            "9.9",
            "--pool",
            str(pool_file),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert pool_file.read_text(encoding="utf-8") == before  # the 9.9 price never landed


def test_route_student_replaces_the_same_name_under_yes(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.2",
            "--output-per-mtok",
            "0.8",
            "--pool",
            str(pool_file),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "replaced" in result.output
    models = load_pool(pool_file).models
    assert len(models) == 1  # replaced, not duplicated
    assert models[0].price().output_per_mtok == 0.8


def test_route_pin_writes_a_serveable_static_policy(tmp_path: Path) -> None:
    """One step from pool candidate to endpoint: the policy lands where `wmo serve` reads it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    assert policy.kind == "static"
    assert policy.default_model == "student"
    assert [entry.name for entry in policy.pool] == ["student"]
    assert policy.fitted_from is not None
    assert "no outcome matrix" in policy.fitted_from  # provenance says it measured nothing


def test_route_pin_warns_when_out_bypasses_the_model_dir(tmp_path: Path) -> None:
    """A scratch --out succeeds but serving never sees it; the pin must say so.

    Both bench-defaults lanes shipped an endpoint whose model dir still held
    the OLD policy because `pin --out /tmp/...` printed the same success line
    as an in-place pin (2026-07-29).
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path)
    scratch = tmp_path / "scratch" / "policy-pin.json"
    scratch.parent.mkdir()

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
            "--out",
            str(scratch),
        ],
    )

    assert result.exit_code == 0, result.output
    assert scratch.is_file()  # the pin still lands where asked
    assert "does NOT update" in result.output  # but the operator is told serving will not see it


def test_route_pin_serves_through_the_endpoint_it_installed(tmp_path: Path) -> None:
    """The pinned policy is not just well formed: `select_model` actually routes on it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    decision = select_model(policy, "anything at all")

    assert decision.model == "student"


def test_route_pin_rejects_a_model_outside_the_pool(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "ghost",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no pool model named 'ghost'" in result.output
    assert "student" in result.output  # lists what IS available


def test_route_pin_names_the_missing_world_model(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "nope",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no world model named 'nope'" in result.output


def test_route_pin_declining_keeps_a_fitted_policy(tmp_path: Path) -> None:
    """Pinning over a fitted knn policy would orphan its evidence bank, so it must ask first."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    installed = model_dir / POLICY_FILENAME
    fitted = _fitted_knn_policy(tmp_path)
    installed.write_text(fitted.read_text(encoding="utf-8"), encoding="utf-8")
    before = installed.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert installed.read_text(encoding="utf-8") == before


def test_route_student_rejects_an_empty_endpoint(tmp_path: Path) -> None:
    """`--endpoint "$UNSET_VAR"` must not silently fall back to a different host."""
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "--endpoint is empty" in result.output
    assert not (tmp_path / "pool.toml").exists()  # nothing was written


def test_route_student_summary_does_not_claim_a_key_it_will_not_send(tmp_path: Path) -> None:
    """A custom endpoint authenticates via WMO_ENDPOINT_API_KEY, and the summary must say so."""
    pool_file = tmp_path / "pool.toml"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "https://my-vllm.example/v1",
            "--pool",
            str(pool_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WMO_ENDPOINT_API_KEY" in result.output
    assert "TINKER_API_KEY" not in result.output
    assert load_pool(pool_file).entry("student").api_key_env is None


def test_route_student_reports_a_busy_pool_without_claiming_it_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held roster lock is a retryable busy state, so it must exit non-zero and say to retry.

    The lock is real here (taken with flock, from this process, on the file the command writes);
    only the wait is shortened, so the CLI runs the same path an operator hits when a second
    registration is in flight. What matters is that it never prints its "added pool candidate"
    line for a write that did not happen, and never reports a lock holder as a bad flag.
    """
    monkeypatch.setattr(pool_module, "POOL_LOCK_TIMEOUT_S", 0.05)
    pool_file = tmp_path / "pool.toml"
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    holder = FileLock(pool_file.with_name(f"{pool_file.name}.lock"))
    holder.acquire()
    try:
        result = runner.invoke(
            app,
            [
                "optimize",
                "route",
                "student",
                str(_run_dir(tmp_path)),
                "--input-per-mtok",
                "0.1",
                "--output-per-mtok",
                "0.4",
                "--pool",
                str(pool_file),
            ],
        )
    finally:
        holder.release()

    assert result.exit_code == 1, result.output
    assert "pool busy" in result.output
    assert "retry" in result.output
    assert "added pool candidate" not in result.output
    assert not pool_file.exists()  # nothing was written


def test_route_student_rejects_an_unknown_output_budget_field(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--chat-max-tokens-field",
            "max_output_tokens",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "max_tokens or max_completion_tokens" in result.output

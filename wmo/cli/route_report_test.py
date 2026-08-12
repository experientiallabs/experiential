"""Route report and late pin CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_report_names_the_swap_when_the_positionals_are_reversed(tmp_path: Path) -> None:
    # Two same-typed positionals in a fixed order is a swap waiting to happen, and a pydantic
    # schema dump ("outcomes / Field required") is not a diagnosis.
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)

    swapped = _report(policy_file, matrix_file, tmp_path / "report.json")
    assert swapped.exit_code != 0
    assert _no_traceback(swapped)
    assert _says(swapped.output, "holds a fitted policy, not an outcome matrix")
    assert _says(swapped.output, "wmo optimize route report <matrix.json> <policy.json>")
    assert not (tmp_path / "report.json").exists()


def test_route_report_rejects_a_policy_that_is_not_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = _report(_knn_matrix_file(tmp_path), bad, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "is not a readable routing policy")


def test_route_report_rejects_a_policy_that_is_not_utf8_text(tmp_path: Path) -> None:
    """`RoutingPolicy.load` decodes before pydantic runs, so this never reached the other clause.

    UnicodeDecodeError is a ValueError but NOT a ValidationError, so undecodable bytes (a
    truncated download, or the `.npz` evidence bank handed over as the policy) walked straight
    past the boundary and tracebacked.
    """
    bad = tmp_path / "bad.json"
    bad.write_bytes(b'{"kind": "\xff\xfeknn"}')
    result = _report(_knn_matrix_file(tmp_path), bad, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "cannot read the policy at")
    assert not (tmp_path / "report.json").exists()


def test_route_report_delivers_the_missing_sidecar_message_cleanly(tmp_path: Path) -> None:
    """The message was already written; it arrived as the last line of a stack trace.

    Copying a knn policy.json without its `.bank.npz` is the exact mistake `knn_bank` anticipates.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)
    RoutingPolicy.load(policy_file).bank_path().unlink()

    result = _report(matrix_file, policy_file, tmp_path / "report.json")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "knn policy bank not found at")
    assert _says(result.output, "Copy the sidecar next to the policy file")


def test_route_report_says_baseline_is_a_pool_handle(tmp_path: Path) -> None:
    # `--baseline` takes the [[model]] table's `name`, not the model id, and passing an id used
    # to raise a bare KeyError.
    result = _report(
        _knn_matrix_file(tmp_path),
        _fitted_knn_policy(tmp_path),
        tmp_path / "report.json",
        baseline="gpt-4o",
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "baseline 'gpt-4o' is not in the matrix pool")
    assert _says(result.output, "pool entry handle")


def test_route_report_refuses_a_matrix_with_nothing_scored_on_both_sides(tmp_path: Path) -> None:
    matrix = OutcomeMatrix.load(_matrix_file(tmp_path))
    half = tmp_path / "half.json"
    OutcomeMatrix(
        pool=matrix.pool,
        outcomes=[
            o.model_copy(update={"reward": None, "success": False}) if o.model == "a" else o
            for o in matrix.outcomes
        ],
    ).save(half)
    policy_file = tmp_path / "static.json"
    RoutingPolicy(kind="static", default_model="a", pool=matrix.pool, fitted_from="handmade").save(
        policy_file
    )

    result = _report(half, policy_file, tmp_path / "report.json", baseline="b")
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "nothing to compare")
    assert not (tmp_path / "report.json").exists()


def test_route_report_creates_the_out_directory_like_fit_does(tmp_path: Path) -> None:
    """`fit --out` mkdir -p's its parents; report tracebacked AFTER computing the whole report."""
    out = tmp_path / "missing" / "sub" / "report.json"
    result = _report(_knn_matrix_file(tmp_path), _fitted_knn_policy(tmp_path), out)
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["headline"]


def test_route_report_notes_the_excluded_fit_split_on_the_fit_matrix(
    tmp_path: Path,
) -> None:
    """Same matrix as the fit: since #308 the report excludes the fit split, so the surface says
    "held-out with N fit scenarios excluded" rather than contradicting the report's own label.
    The matrix digest in `fitted_from` is an identity, so a renamed copy has to trip it too.
    """
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)

    result = _report(matrix_file, policy_file, tmp_path / "report.json")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "fit scenario(s) were excluded")
    assert "IN-SAMPLE" not in _flat(result.output)

    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(matrix_file.read_bytes())
    moved = _report(renamed, policy_file, tmp_path / "report_renamed.json")
    assert moved.exit_code == 0, moved.output
    assert _says(moved.output, "fit scenario(s) were excluded")

    # The provenance marker is appended LAST, so a matrix stored under a content-addressed
    # directory carries `sha256=` in its path too. Splitting from the left read THAT one and
    # dropped the caveat on exactly the layout most likely to keep a fit matrix around.
    addressed = tmp_path / "artifacts" / "sha256=deadbeef" / "matrix.json"
    addressed.parent.mkdir(parents=True)
    addressed.write_bytes(matrix_file.read_bytes())
    content_addressed = _report(addressed, policy_file, tmp_path / "report_addressed.json")
    assert content_addressed.exit_code == 0, content_addressed.output
    assert _says(content_addressed.output, "fit scenario(s) were excluded")


def test_route_report_stays_quiet_on_a_matrix_the_fit_never_saw(tmp_path: Path) -> None:
    """The negative control: held-out numbers are what report is for, so no warning."""
    policy_file = _fitted_knn_policy(tmp_path)
    held_out = _knn_matrix_file(tmp_path, flip=True, name="held_out.json")
    result = _report(held_out, policy_file, tmp_path / "report.json")
    assert result.exit_code == 0, result.output
    assert "IN-SAMPLE" not in _flat(result.output)


def test_route_pin_names_the_positional_when_the_model_is_ambiguous(tmp_path: Path) -> None:
    # `WorldModelStore.resolve` says "pass --name", which `pin` does not have: its world model is
    # a positional and its --model is the POOL entry. Following the old advice failed outright.
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path, "alpha")
    _built_model(tmp_path, "beta")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "WORLD_MODEL" in flat and "alpha,beta" in flat
    assert "--name" not in flat
    assert "wmooptimizeroutepinalpha--modelstudent" in flat  # a command that actually works

    followed = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "alpha",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert followed.exit_code == 0, followed.output


def test_route_pin_names_the_pool_writers_when_the_pool_is_empty(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    pool_file.write_text("# nothing yet\n", encoding="utf-8")
    _built_model(tmp_path)

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
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, str(pool_file))
    assert _says(result.output, "wmo optimize route student")
    assert "too_short" not in _flat(result.output)  # was a raw pydantic dump


def test_route_tune_names_the_fit_when_there_is_no_policy(tmp_path: Path) -> None:
    # The sibling not-dialable error names `wmo optimize route fit --kind knn`; this branch,
    # which is the one a first-time user hits (the argument defaults to ./policy.json), did not.
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(tmp_path / "nope.json"), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert _says(result.output, "no policy file at")
    assert _says(result.output, "wmo optimize route fit <matrix.json> --kind knn")


def test_route_student_help_keeps_the_pool_table_name() -> None:
    """The paragraph exists to name the TOML table `student` writes, so it must survive rich.

    Typer renders help through rich markup, which swallowed the unescaped `[[model]]` and left
    an empty pair of backticks where the identifier should be.
    """
    result = runner.invoke(app, ["optimize", "route", "student", "--help"])
    assert result.exit_code == 0, result.output
    assert "[[model]]" in _flat(result.output)


def test_route_pin_refuses_a_disabled_model_and_drops_disabled_entries_from_the_pool(
    tmp_path: Path,
) -> None:
    """`enabled = false` is honored at pin time: not pinnable, and not carried into the policy.

    The policy's pool is what serving may construct providers for, so a candidate the operator
    turned off must not ride into an endpoint pinned afterwards.
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    pool_file.write_text(
        pool_file.read_text(encoding="utf-8")
        + """
[[model]]
name = "off-limits"
kind = "openai"
model = "gpt-5.4"
enabled = false
""",
        encoding="utf-8",
    )
    model_dir = _built_model(tmp_path)

    refused = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "off-limits",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )
    assert refused.exit_code != 0
    assert "disabled" in refused.output

    pinned = runner.invoke(
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
    assert pinned.exit_code == 0, pinned.output
    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    assert [entry.name for entry in policy.pool] == ["student"]


def test_route_student_replacement_keeps_a_disabled_entry_disabled(tmp_path: Path) -> None:
    """Retraining and re-registering a student must not undo an operator's enabled = false.

    Same rule as the registry writer, pinned separately because the student command
    reaches upsert_pool_entry through its own path (_pool_disabled at route_app.py).
    """
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    pool_file.write_text(
        pool_file.read_text(encoding="utf-8").replace(
            'name = "student"', 'name = "student"\nenabled = false'
        ),
        encoding="utf-8",
    )

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
    assert "keeping it disabled" in result.output
    entries = load_pool(pool_file).models
    assert len(entries) == 1
    assert entries[0].enabled is False

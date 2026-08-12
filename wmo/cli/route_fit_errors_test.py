"""Route fit error-path tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_fit_names_the_producer_when_the_matrix_is_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(tmp_path / "nope.json"),
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "no outcome matrix at")
    assert _says(result.output, "wmo optimize route sweep")


def test_route_fit_rejects_a_matrix_that_is_not_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(bad),
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "is not a readable OutcomeMatrix")


@pytest.mark.parametrize("kind", ["knn", "rank"])
def test_route_fit_refuses_a_matrix_with_no_scored_cell(tmp_path: Path, kind: str) -> None:
    """Both kinds, and both used to traceback.

    The rank fitter's own message ("no scored outcomes; cannot pick a default model") named no
    remedy at all, so the answer belongs at the boundary where sweep's warning can be echoed.
    """
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_unscored_matrix_file(tmp_path)),
            "--kind",
            kind,
            "--embedder",
            "hashing",
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    assert _no_traceback(result)
    assert _says(result.output, "carries a reward")
    assert _says(result.output, "wmo optimize route sweep")
    assert not (tmp_path / "policy.json").exists()


def test_route_fit_defaults_to_the_knn_champion(tmp_path: Path) -> None:
    """The default kind has to be the one every other surface steers to.

    `fit --help` calls knn "the validated champion", sweep's handoff prints `--kind knn`, and
    `tune` only dials a knn policy -- but the flag defaulted to rank, so a user who omitted it
    silently fitted the non-champion and only found out at `tune`.
    """
    policy_file = tmp_path / POLICY_FILENAME
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path)),
            "--fallback",
            "a",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert RoutingPolicy.load(policy_file).kind == "knn"
    # And therefore dialable without a refit, which is what the old default was not.
    tuned = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert tuned.exit_code == 0, tuned.output


@pytest.mark.parametrize("command", ["fit", "sweep"])
def test_route_compressor_help_lists_every_shipped_id(command: str) -> None:
    """Rendered from the registry, so a shipped id cannot go unadvertised.

    Asserted against the ids registered at import rather than the live registry, because that is
    when typer builds a help string (this module itself registers fakes afterwards).
    """
    result = runner.invoke(app, ["optimize", "route", command, "--help"])
    assert result.exit_code == 0, result.output
    for compressor_id in ("identity", "truncate"):
        assert compressor_id in registered_compressor_ids()
        assert _says(result.output, compressor_id)

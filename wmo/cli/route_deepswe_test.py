"""Route DeepSWE conversion CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_convert_deepswe_writes_the_bundle_and_states_the_gate(tmp_path: Path) -> None:
    source, cache = _deepswe_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "convert-deepswe",
            str(source),
            "--embedding-cache",
            str(cache),
            "--out",
            str(bundle),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "configs reproduce published pass@1" in result.output
    for name in ("matrix.json", "task_embeddings.npy", "scenario_groups.json"):
        assert (bundle / name).is_file()
    assert "claude-opus-5@high" in result.output


def test_route_convert_deepswe_refuses_contradicted_leaderboard(tmp_path: Path) -> None:
    source, cache = _deepswe_fixture(tmp_path)
    lied = json.loads((source / "leaderboard-live.json").read_text(encoding="utf-8"))
    lied["rows"][0]["pass_at_1"] = 0.5
    (source / "leaderboard-live.json").write_text(json.dumps(lied), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "convert-deepswe",
            str(source),
            "--embedding-cache",
            str(cache),
            "--out",
            str(tmp_path / "bundle"),
        ],
    )
    assert result.exit_code != 0
    assert "published pass@1" in result.output
    assert not (tmp_path / "bundle" / "matrix.json").exists()  # the gate writes nothing

"""Tests for multi-root eval suite discovery and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.engine.eval_suites import discover_eval_suites, load_eval_suite, resolve_eval_suite


def _write_suite(
    root: Path, example: str, name: str = "default", body: str = 'description = "t"\n'
) -> Path:
    suite_dir = root / example / "evals"
    suite_dir.mkdir(parents=True)
    path = suite_dir / f"{name}.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_discover_accepts_single_root_and_iterable(tmp_path: Path) -> None:
    root_a = tmp_path / "examples"
    root_b = tmp_path / "environment-capture"
    _write_suite(root_a, "alpha")
    _write_suite(root_b, "beta")

    assert [s.id for s in discover_eval_suites(root_a)] == ["alpha/default"]
    assert [s.id for s in discover_eval_suites([root_a, root_b])] == [
        "alpha/default",
        "beta/default",
    ]


def test_discover_skips_missing_roots(tmp_path: Path) -> None:
    root = tmp_path / "examples"
    _write_suite(root, "alpha")
    suites = discover_eval_suites([root, tmp_path / "does-not-exist"])
    assert [s.id for s in suites] == ["alpha/default"]


def test_resolve_finds_suite_in_second_root_by_alias(tmp_path: Path) -> None:
    root_a = tmp_path / "examples"
    root_b = tmp_path / "environment-capture"
    _write_suite(root_a, "alpha")
    _write_suite(root_b, "beta")
    suite = resolve_eval_suite("beta", [root_a, root_b])
    assert suite.id == "beta/default"


def test_resolve_rejects_same_id_in_multiple_roots(tmp_path: Path) -> None:
    """A benchmark dir present under two roots must be an explicit error, not first-root-wins."""
    root_a = tmp_path / "examples"
    root_b = tmp_path / "environment-capture"
    _write_suite(root_a, "alpha")
    _write_suite(root_b, "alpha")
    with pytest.raises(ValueError, match="multiple roots"):
        resolve_eval_suite("alpha/default", [root_a, root_b])


def test_discover_skips_malformed_suite_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One broken evals/*.toml must not take down discovery for every other benchmark."""
    root = tmp_path / "examples"
    _write_suite(root, "alpha")
    bad_dir = root / "broken" / "evals"
    bad_dir.mkdir(parents=True)
    (bad_dir / "default.toml").write_text("not valid toml [[[", encoding="utf-8")

    with caplog.at_level("WARNING"):
        suites = discover_eval_suites(root)
    assert [s.id for s in suites] == ["alpha/default"]
    assert any("broken" in record.message for record in caplog.records)


# --- TOML loading / validation errors (judge overhaul) --------------------------------------


def test_load_eval_suite_parses_a_minimal_suite(tmp_path: Path) -> None:
    path = _write_suite(
        tmp_path, "solo", body='description = "d"\nfiles = ["../traces.otel.jsonl"]\n'
    )
    suite = load_eval_suite(path)
    assert suite.name == "default"
    assert suite.config.description == "d"


def test_removed_judge_option_gets_an_actionable_error(tmp_path: Path) -> None:
    # Pre-overhaul suite TOMLs (including the old shipped defaults) carried `judge = "rubric"`.
    # The generic "does not match the eval suite schema" pydantic error never says the knob was
    # removed or what to do — the message must.
    path = _write_suite(tmp_path, "solo", body='judge = "rubric"\n')
    with pytest.raises(ValueError, match="no longer exists.*delete the `judge` line"):
        load_eval_suite(path)


def test_unknown_key_still_gets_the_schema_error(tmp_path: Path) -> None:
    path = _write_suite(tmp_path, "solo", body="not_a_real_option = 1\n")
    with pytest.raises(ValueError, match="does not match the eval suite schema"):
        load_eval_suite(path)


def test_judge_plus_other_errors_surfaces_the_full_listing(tmp_path: Path) -> None:
    # The delete-the-judge-line hint must not mask unrelated schema errors: with a second
    # problem present the user gets the full ValidationError (plus the judge note) in one pass.
    path = _write_suite(tmp_path, "solo", body='judge = "rubric"\ntrain_split = 1.5\n')
    with pytest.raises(ValueError, match="train_split") as excinfo:
        load_eval_suite(path)
    assert "judge" in str(excinfo.value)

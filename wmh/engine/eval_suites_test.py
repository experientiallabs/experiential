"""Tests for multi-root eval suite discovery and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.engine.eval_suites import discover_eval_suites, resolve_eval_suite


def _write_suite(root: Path, example: str, name: str = "default") -> Path:
    suite_dir = root / example / "evals"
    suite_dir.mkdir(parents=True)
    path = suite_dir / f"{name}.toml"
    path.write_text('description = "t"\n', encoding="utf-8")
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

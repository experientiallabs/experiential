"""Tests for the minimal .env loader/writer."""

from __future__ import annotations

import os

from wmh.config.dotenv import load_env_file, upsert_env_var


def test_load_env_file_sets_only_unset_vars(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nWMH_TEST_NEW=from-file\nWMH_TEST_KEPT='quoted'\nWMH_TEST_SET=ignored\nbroken\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WMH_TEST_NEW", raising=False)
    monkeypatch.delenv("WMH_TEST_KEPT", raising=False)
    monkeypatch.setenv("WMH_TEST_SET", "from-env")

    load_env_file(env)
    assert os.environ["WMH_TEST_NEW"] == "from-file"
    assert os.environ["WMH_TEST_KEPT"] == "quoted"  # quotes stripped
    assert os.environ["WMH_TEST_SET"] == "from-env"  # not overridden
    monkeypatch.delenv("WMH_TEST_NEW")
    monkeypatch.delenv("WMH_TEST_KEPT")


def test_load_env_file_missing_path_is_a_noop(tmp_path) -> None:  # noqa: ANN001
    load_env_file(tmp_path / "nope.env")  # must not raise


def test_upsert_env_var_appends_and_replaces(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nWMH_TEST_UPSERT=old\n", encoding="utf-8")
    monkeypatch.delenv("WMH_TEST_UPSERT", raising=False)

    upsert_env_var("WMH_TEST_UPSERT", "new", env)
    assert os.environ["WMH_TEST_UPSERT"] == "new"
    assert env.read_text(encoding="utf-8") == "OTHER=1\nWMH_TEST_UPSERT=new\n"

    upsert_env_var("WMH_TEST_ADDED", "v", env)
    assert env.read_text(encoding="utf-8").endswith("WMH_TEST_ADDED=v\n")
    monkeypatch.delenv("WMH_TEST_UPSERT")
    monkeypatch.delenv("WMH_TEST_ADDED")

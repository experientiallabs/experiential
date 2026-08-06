"""Tests for the minimal .env loader/writer."""

from __future__ import annotations

import os

import pytest

from wmo.common.config.dotenv import load_env_file, upsert_env_var


def test_load_env_file_sets_only_unset_vars(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nWMO_TEST_NEW=from-file\nWMO_TEST_KEPT='quoted'\nWMO_TEST_SET=ignored\nbroken\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WMO_TEST_NEW", raising=False)
    monkeypatch.delenv("WMO_TEST_KEPT", raising=False)
    monkeypatch.setenv("WMO_TEST_SET", "from-env")

    load_env_file(env)
    assert os.environ["WMO_TEST_NEW"] == "from-file"
    assert os.environ["WMO_TEST_KEPT"] == "quoted"  # quotes stripped
    assert os.environ["WMO_TEST_SET"] == "from-env"  # not overridden
    monkeypatch.delenv("WMO_TEST_NEW")
    monkeypatch.delenv("WMO_TEST_KEPT")


def test_load_env_file_missing_path_is_a_noop(tmp_path) -> None:  # noqa: ANN001
    load_env_file(tmp_path / "nope.env")  # must not raise


def test_upsert_env_var_appends_and_replaces(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nWMO_TEST_UPSERT=old\n", encoding="utf-8")
    monkeypatch.delenv("WMO_TEST_UPSERT", raising=False)

    upsert_env_var("WMO_TEST_UPSERT", "new", env)
    assert os.environ["WMO_TEST_UPSERT"] == "new"
    assert env.read_text(encoding="utf-8") == "OTHER=1\nWMO_TEST_UPSERT=new\n"

    upsert_env_var("WMO_TEST_ADDED", "v", env)
    assert env.read_text(encoding="utf-8").endswith("WMO_TEST_ADDED=v\n")
    assert env.stat().st_mode & 0o777 == 0o600  # owner-only, even for a pre-existing file
    monkeypatch.delenv("WMO_TEST_UPSERT")
    monkeypatch.delenv("WMO_TEST_ADDED")


def test_upsert_env_var_refuses_symlinked_env(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    target = tmp_path / "victim"
    target.write_text("do not clobber\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.symlink_to(target)
    monkeypatch.delenv("WMO_TEST_SYMLINK", raising=False)

    with pytest.raises(ValueError, match="symlink"):
        upsert_env_var("WMO_TEST_SYMLINK", "secret", env)
    assert target.read_text(encoding="utf-8") == "do not clobber\n"  # untouched
    assert "WMO_TEST_SYMLINK" not in os.environ  # nothing half-applied

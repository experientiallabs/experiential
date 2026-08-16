"""Tests for the minimal read-only `.env` loader."""

from __future__ import annotations

import os

from wmo.common.config.dotenv import load_env_file


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

"""Tests for the minimal read-only `.env` loader."""

from __future__ import annotations

import os

from exp.common.config.dotenv import load_env_file


def test_load_env_file_sets_only_unset_vars(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nEXP_TEST_NEW=from-file\nEXP_TEST_KEPT='quoted'\nEXP_TEST_SET=ignored\nbroken\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EXP_TEST_NEW", raising=False)
    monkeypatch.delenv("EXP_TEST_KEPT", raising=False)
    monkeypatch.setenv("EXP_TEST_SET", "from-env")

    load_env_file(env)
    assert os.environ["EXP_TEST_NEW"] == "from-file"
    assert os.environ["EXP_TEST_KEPT"] == "quoted"  # quotes stripped
    assert os.environ["EXP_TEST_SET"] == "from-env"  # not overridden
    monkeypatch.delenv("EXP_TEST_NEW")
    monkeypatch.delenv("EXP_TEST_KEPT")

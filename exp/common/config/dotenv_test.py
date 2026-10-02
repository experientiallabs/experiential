"""Tests for the minimal read-only `.env` loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exp.common.config.dotenv import load_env_file


def test_load_env_file_ignores_a_missing_path(tmp_path: Path) -> None:
    """An absent environment file remains an intentional no-op."""
    load_env_file(tmp_path / "missing.env")


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


def test_load_env_file_rejects_a_directory_with_actionable_context(tmp_path: Path) -> None:
    """A directory path should become a configuration error that identifies the path."""
    env_path = tmp_path / ".env"
    env_path.mkdir()

    with pytest.raises(ValueError) as captured:
        load_env_file(env_path)

    message = str(captured.value)
    assert str(env_path) in message
    assert "regular UTF-8 text file" in message
    assert "remove it" in message


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files are not portable")
def test_load_env_file_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    """The opened descriptor must be validated before a special file can block a read."""
    env_path = tmp_path / ".env"
    os.mkfifo(env_path)

    with pytest.raises(ValueError, match="regular UTF-8 text file"):
        load_env_file(env_path)


def test_load_env_file_rejects_non_utf8_content_with_actionable_context(tmp_path: Path) -> None:
    """Invalid UTF-8 should become a configuration error that identifies remediation."""
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"EXP_TEST_SECRET=\xff\n")

    with pytest.raises(ValueError) as captured:
        load_env_file(env_path)

    message = str(captured.value)
    assert str(env_path) in message
    assert "regular UTF-8 text file" in message
    assert "remove it" in message
    assert isinstance(captured.value.__cause__, UnicodeDecodeError)

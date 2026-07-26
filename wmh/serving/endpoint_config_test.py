"""Tests for the per-endpoint serving config file (`endpoint.toml`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmh.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig


def test_a_missing_file_is_the_empty_config(tmp_path: Path) -> None:
    # Most endpoints never set a dial; that is not an error condition, it is the default.
    config = EndpointConfig.load(tmp_path / ENDPOINT_CONFIG_FILENAME)
    assert config.cost_quality is None


def test_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    EndpointConfig(cost_quality=0.6).save(path)
    assert "cost_quality = 0.6" in path.read_text(encoding="utf-8")
    assert EndpointConfig.load(path).cost_quality == 0.6


def test_saving_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    EndpointConfig(cost_quality=0.25).save(path)
    assert sorted(item.name for item in tmp_path.iterdir()) == [ENDPOINT_CONFIG_FILENAME]


def test_an_unset_dial_is_omitted_rather_than_written_as_null(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    EndpointConfig().save(path)
    assert path.read_text(encoding="utf-8").strip() == ""
    assert EndpointConfig.load(path).cost_quality is None


def test_a_malformed_file_says_which_file_and_what_was_expected(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    path.write_text("cost_quality = ", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid endpoint config"):
        EndpointConfig.load(path)


def test_a_dial_outside_the_range_is_rejected_at_load(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    path.write_text("cost_quality = 1.5", encoding="utf-8")
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        EndpointConfig.load(path)

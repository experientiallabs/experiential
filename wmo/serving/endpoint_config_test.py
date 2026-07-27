"""Tests for the per-endpoint serving config file (`endpoint.toml`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.core.types import Representation
from wmo.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig


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


def test_an_unknown_key_fails_at_load_instead_of_being_ignored(tmp_path: Path) -> None:
    # A typo must not leave an operator staring at an endpoint that silently ignored their dial.
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    path.write_text("cost_qualty = 0.6\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="cost_qualty"):
        EndpointConfig.load(path)


def test_the_parse_error_says_what_shape_was_expected(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    path.write_text("cost_quality = ", encoding="utf-8")
    with pytest.raises(ValueError, match="no other keys"):
        EndpointConfig.load(path)


def test_representation_round_trips_as_a_toml_table(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    EndpointConfig(
        cost_quality=0.25,
        representation=Representation(compressor_id="llmlingua2", aggressiveness=0.4),
    ).save(path)
    loaded = EndpointConfig.load(path)
    assert loaded.representation == Representation(compressor_id="llmlingua2", aggressiveness=0.4)


def test_absent_on_both_sides_is_the_uncompressed_case_and_passes() -> None:
    # The overwhelmingly common pair, and the only one that needs no coordination at all.
    EndpointConfig().check_representation(None, endpoint="support")


def test_matching_representations_pass() -> None:
    rep = Representation(compressor_id="llmlingua2", aggressiveness=0.4)
    EndpointConfig(representation=rep).check_representation(rep, endpoint="support")


def test_a_policy_fitted_compressed_is_refused_by_an_uncompressed_endpoint() -> None:
    fitted = Representation(compressor_id="llmlingua2", aggressiveness=0.4)
    with pytest.raises(ValueError) as caught:
        EndpointConfig().check_representation(fitted, endpoint="support")
    message = str(caught.value)
    # Both sides named: an operator has to know which one to change.
    assert "uncompressed requests" in message
    assert "llmlingua2" in message and "0.4" in message
    assert "support" in message


def test_an_uncompressed_policy_is_refused_by_a_compressing_endpoint() -> None:
    # The mirror image, and the likelier accident: compression is added in front of an endpoint
    # whose policy was fitted before it existed.
    config = EndpointConfig(
        representation=Representation(compressor_id="llmlingua2", aggressiveness=0.4)
    )
    with pytest.raises(ValueError) as caught:
        config.check_representation(None, endpoint="support")
    assert "uncompressed requests" in str(caught.value)


def test_the_same_compressor_at_a_different_aggressiveness_is_a_different_representation() -> None:
    config = EndpointConfig(
        representation=Representation(compressor_id="llmlingua2", aggressiveness=0.4)
    )
    with pytest.raises(ValueError) as caught:
        config.check_representation(
            Representation(compressor_id="llmlingua2", aggressiveness=0.8), endpoint="support"
        )
    assert "0.8" in str(caught.value)


def test_a_typo_in_the_representation_table_fails_with_the_key_named(tmp_path: Path) -> None:
    path = tmp_path / ENDPOINT_CONFIG_FILENAME
    path.write_text(
        '[representation]\ncompressor_id = "x"\naggressivness = 0.4\n', encoding="utf-8"
    )
    with pytest.raises(ValidationError) as caught:
        EndpointConfig.load(path)
    assert "aggressivness" in str(caught.value)

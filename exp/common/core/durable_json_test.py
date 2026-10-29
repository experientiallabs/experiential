"""Durable projection preserves input, Unicode meaning and colliding object keys."""

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.core.durable_json import count_unstorable_text, normalize_durable_object


@pytest.mark.parametrize("text", ["a\x00b", "a\ud800b", "a\udfffb"])
def test_unstorable_strings_are_normalized_only_in_saved_copy(text: str) -> None:
    """Storage normalization never changes the object used for inference."""
    source: JsonObject = {"content": text}
    saved, count = normalize_durable_object(source)
    assert saved["content"] == "a\ufffdb"
    assert count == 1
    assert source["content"] == text


def test_literal_escapes_and_valid_surrogate_pairs_preserve_meaning() -> None:
    """Already-storable escape spelling and paired code units retain their meaning."""
    saved, count = normalize_durable_object({"content": "\\u0000 \ud83d\ude00"})
    assert json.loads(json.dumps(saved))["content"] == "\\u0000 \U0001f600"
    assert count == 0


def test_normalization_cannot_merge_colliding_tool_argument_keys() -> None:
    """Every source value survives even when multiple keys normalize alike."""
    source: JsonObject = {"a\x00": "first", "a\ufffd": "second", "a\ufffd~2": "third"}
    saved, count = normalize_durable_object(source)
    assert count == 3
    assert sorted(saved.values()) == ["first", "second", "third"]
    assert saved["a\ufffd"] == "second"
    assert source["a\x00"] == "first"


def test_unstorable_count_covers_keys_values_and_nested_tool_arguments() -> None:
    """Batch validation and request projection use one Unicode contract."""
    assert count_unstorable_text({"x\x00": ["\ud800", {"v": "\ud83d\ude00"}]}) == 2

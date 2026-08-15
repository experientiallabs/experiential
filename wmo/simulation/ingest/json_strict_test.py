"""Tests for strict JSON object decoding."""

from __future__ import annotations

import json

import pytest

from wmo.simulation.ingest.json_strict import DuplicateJsonKeyError, reject_duplicate_json_keys


def test_reject_duplicate_json_keys_keeps_unique_members() -> None:
    """Unique keys decode in source order."""
    payload = json.loads(
        '{"alpha": 1, "beta": 2}',
        object_pairs_hook=reject_duplicate_json_keys,
    )

    assert payload == {"alpha": 1, "beta": 2}


def test_reject_duplicate_json_keys_fails_on_repeated_members() -> None:
    """A repeated key is rejected instead of silently keeping the last value."""
    with pytest.raises(DuplicateJsonKeyError, match="duplicate JSON key: alpha"):
        json.loads('{"alpha": 1, "alpha": 2}', object_pairs_hook=reject_duplicate_json_keys)

"""Tests for robust completion parsing."""

from __future__ import annotations

from wmh.core.parsing import dumps_observation_contract, extract_json_object, parse_observation
from wmh.core.types import Observation


def test_extract_json_object_handles_fences_prose_and_nesting() -> None:
    assert extract_json_object('{"a": 1}') == '{"a": 1}'
    assert extract_json_object('text ```json\n{"a": {"b": 2}}\n``` more') == '{"a": {"b": 2}}'
    # First of multiple objects.
    assert extract_json_object('{"a": 1} then {"b": 2}') == '{"a": 1}'
    # Braces inside strings don't confuse the scanner.
    assert extract_json_object('{"s": "a } b { c"}') == '{"s": "a } b { c"}'
    assert extract_json_object("no json here") is None


def test_parse_observation_uses_json_contract() -> None:
    obs = parse_observation(
        '{"output": "cart has 1 item", "is_error": false, "state_note": "added A1"}'
    )
    assert obs.content == "cart has 1 item"
    assert obs.is_error is False
    assert obs.metadata["state_note"] == "added A1"


def test_parse_observation_flags_error() -> None:
    obs = parse_observation('{"output": "no such user", "is_error": true}')
    assert obs.is_error is True
    assert obs.content == "no such user"


def test_parse_observation_falls_back_to_plaintext() -> None:
    obs = parse_observation("the cart now has one item")
    assert obs.content == "the cart now has one item"
    assert obs.is_error is False


def test_parse_observation_honors_empty_contract() -> None:
    # A silent success (many shell writes/redirects print nothing) is a valid empty observation,
    # not raw JSON text: the contract keys are present even though every value is empty/false.
    obs = parse_observation('{"output": "", "is_error": false, "state_note": ""}')
    assert obs.content == ""
    assert obs.is_error is False
    assert "state_note" not in obs.metadata


def test_parse_observation_ignores_non_contract_json() -> None:
    # Arbitrary JSON with none of the contract keys is preserved as raw text (not coerced to empty).
    obs = parse_observation('{"foo": 1}')
    assert obs.content == '{"foo": 1}'


def test_dumps_observation_contract_roundtrips() -> None:
    obs = Observation(content="ok", is_error=False, metadata={"state_note": "did x"})
    text = dumps_observation_contract(obs)
    back = parse_observation(text)
    assert back.content == "ok"
    assert back.metadata["state_note"] == "did x"

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


def test_parse_observation_strips_reasoning_into_metadata() -> None:
    obs = parse_observation(
        '{"reasoning": "gate: user is authed (step 2), record exists => success", '
        '"output": "ok", "is_error": false, "state_note": "", '
        '"kb_note": "flight HAT-201 JFK->SFO exists", "ground_query": ""}'
    )
    assert obs.content == "ok"  # reasoning never leaks into what the agent observes
    assert obs.metadata["reasoning"] == "gate: user is authed (step 2), record exists => success"
    assert obs.metadata["kb_note"] == "flight HAT-201 JFK->SFO exists"
    assert "ground_query" not in obs.metadata  # empty fields stay out of metadata


def test_parse_observation_ground_query_in_metadata() -> None:
    obs = parse_observation(
        '{"reasoning": "package unknown", "output": "", "is_error": false, '
        '"ground_query": "tomli_w python package api"}'
    )
    assert obs.metadata["ground_query"] == "tomli_w python package api"


def test_parse_observation_empty_output_with_reasoning_is_still_contract() -> None:
    # A silent command (empty output) in reasoning mode must not fall back to raw-JSON content.
    obs = parse_observation(
        '{"reasoning": "mkdir prints nothing", "output": "", "is_error": false}'
    )
    assert obs.content == ""
    assert obs.is_error is False


def test_parse_observation_salvages_truncated_reasoning_completion() -> None:
    # Observed in a live tau eval (score 0.26): a long deliberation + long escaped record blew the
    # token budget, the JSON never closed, and the WHOLE raw contract text became the observation.
    # The salvage path must recover the partial `output` string instead.
    truncated = (
        '{"reasoning": "Return user details for Mohamed.", '
        '"output": "{\\"user_id\\": \\"mohamed_hernandez_5188\\", '
        '\\"name\\": {\\"first_name\\": \\"Moh'
    )
    obs = parse_observation(truncated)
    assert obs.content.startswith('{"user_id": "mohamed_hernandez_5188"')
    assert '"reasoning"' not in obs.content  # deliberation never leaks to the agent
    assert obs.metadata["reasoning"] == "Return user details for Mohamed."


def test_parse_observation_salvage_recovers_is_error_when_present() -> None:
    truncated = (
        '{"reasoning": "gate blocks it", "output": "Error: not permitted", "is_error": true, '
        '"state_note": "attempted forbidden acti'
    )
    obs = parse_observation(truncated)
    assert obs.content == "Error: not permitted"
    assert obs.is_error is True


def test_parse_observation_salvage_does_not_fire_on_plain_text() -> None:
    # Ordinary non-JSON replies (and JSON-looking observations without contract keys) still fall
    # back to full-text — salvage only triggers on a broken CONTRACT payload.
    assert parse_observation("total 0\ndrwxr-xr-x 2 root").content.startswith("total 0")
    obs = parse_observation('{"id": "u1", "name": "kath"}')  # complete non-contract JSON
    assert obs.content == '{"id": "u1", "name": "kath"}'


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


def test_parse_observation_off_contract_reasoning_json_falls_through_to_text() -> None:
    # JSON with a reasoning-superset key but NO core contract key is off-contract output:
    # it must reach the agent as raw text, never as a silent empty observation.
    raw = '{"reasoning": "thinking...", "observation": "the file exists"}'
    assert parse_observation(raw).content == raw


def test_salvage_decodes_unicode_escapes() -> None:
    truncated = '{"reasoning": "r", "output": "caf\\u00e9 r\\u00e9sum'
    obs = parse_observation(truncated)
    assert obs.content == "caf\u00e9 r\u00e9sum"  # decoded, not 'u00e9' garbage

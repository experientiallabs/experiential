"""Strict finite JSON parsing without silent key replacement."""

import pytest

from exp.optimize.claas.service.http.parsing import parse_object


@pytest.mark.parametrize(
    "payload",
    [
        '{"value":1,"value":2}',
        '{"nested":{"value":1,"value":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e999}',
        "[1,2]",
        '"text"',
        "{} {}",
        '{"value":' + "[" * 2000 + "0" + "]" * 2000 + "}",
    ],
)
def test_ambiguous_or_unbounded_numbers_are_rejected(payload: str) -> None:
    """Invalid objects fail before they can become generation or feedback input."""
    with pytest.raises(ValueError):
        parse_object(payload)


def test_valid_nested_object_preserves_values() -> None:
    """Ordinary finite JSON remains unchanged by strict parsing."""
    assert parse_object(b'{"a":[null,true,1,1.5,{"b":"text"}]}') == {
        "a": [None, True, 1, 1.5, {"b": "text"}]
    }

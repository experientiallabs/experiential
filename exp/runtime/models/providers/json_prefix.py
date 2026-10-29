"""Recognize a JSON object that stopped mid-token rather than one that is malformed."""

from __future__ import annotations

import re

_JSON_STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))'
_JSON_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_JSON_SCALAR = re.compile(rf'"{_JSON_STRING_CHAR}*"|true|false|null|{_JSON_NUMBER}')
_JSON_INCOMPLETE_SCALAR = re.compile(
    rf'"{_JSON_STRING_CHAR}*(?:\\(?:u[0-9a-fA-F]{{0,3}})?)?'
    r"|t|tr|tru|f|fa|fal|fals|n|nu|nul|-"
    r"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-]?"
    r"|-?(?:0|[1-9][0-9]*)\."
)


def incomplete_json_object_prefix(value: str) -> bool:
    """Whether appending bytes could complete an unfinished JSON object.

    Track container grammar as well as tokens: EOF inside a literal, exponent
    or escape is recoverable only when all preceding structure is valid. This
    neither repairs arguments nor accepts a complete non-object value.

    Args:
        value: Exact decoded tool-argument string rejected by the JSON parser.

    Returns:
        True only when appending characters could complete the started JSON object
        without changing its existing tokens or container grammar.
    """
    if not value.lstrip(" \t\r\n").startswith("{"):
        return False
    position = 0
    states = ["end", "value"]
    while states:
        state = states.pop()
        while position < len(value) and value[position] in " \t\r\n":
            position += 1
        if position == len(value):
            return state != "end"
        char = value[position]
        if state == "end":
            return False
        if state == "object_first":
            if char == "}":
                position += 1
                continue
            state = "key"
        if state == "array_first":
            if char == "]":
                position += 1
            else:
                states.extend(("array_next", "value"))
            continue
        if state in ("object_next", "array_next"):
            if char == ("}" if state == "object_next" else "]"):
                position += 1
                continue
            if char != ",":
                return False
            states.extend(("key",) if state == "object_next" else ("array_next", "value"))
            position += 1
            continue
        if state == "colon":
            if char != ":":
                return False
            states.extend(("object_next", "value"))
            position += 1
            continue
        if state == "key":
            if char != '"':
                return False
            states.append("colon")
        elif char in "{[":
            states.append("object_first" if char == "{" else "array_first")
            position += 1
            continue
        if _JSON_INCOMPLETE_SCALAR.fullmatch(value, position):
            return True
        scalar = _JSON_SCALAR.match(value, position)
        if scalar is None:
            return False
        position = scalar.end()
    return False

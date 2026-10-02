"""Unknown compatibility meters cannot silently become known zero or erase positive usage."""

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models.usage_observability import unreported_token_details


@pytest.mark.parametrize("responses", [False, True])
def test_unknown_marker_accepts_only_missing_or_zero_placeholders(responses: bool) -> None:
    """Explicit nulls and absent optional details preserve unknowns without fabricating meters."""
    group = "input_tokens_details" if responses else "prompt_tokens_details"
    usage: JsonObject = {
        group: {"cached_tokens": 0, "cache_write_tokens": None},
        "unreported_token_details": [
            "cached_tokens",
            "cache_write_tokens",
            "cache_write_1h_tokens",
        ],
    }
    assert unreported_token_details(usage, responses=responses) == {
        "cached_tokens",
        "cache_write_tokens",
        "cache_write_1h_tokens",
    }
    assert unreported_token_details({group: {"cached_tokens": 0}}, responses=responses) == set()


@pytest.mark.parametrize(
    "marker", [None, "cached_tokens", ["other"], ["cached_tokens", "cached_tokens"], [False], [{}]]
)
def test_malformed_unknown_marker_is_refused(marker: JsonValue) -> None:
    """A trusted decoder still validates every untrusted wire field before interpreting it."""
    with pytest.raises(ValueError, match="unique supported"):
        unreported_token_details({"unreported_token_details": marker})


@pytest.mark.parametrize("count", [1, -1, True, False, 0.0, "0"])
def test_unknown_marker_cannot_hide_positive_or_malformed_usage(count: JsonValue) -> None:
    """Only an actual integer zero is a compatibility placeholder."""
    with pytest.raises(ValueError, match="contradicts"):
        unreported_token_details(
            {
                "prompt_tokens_details": {"cached_tokens": count},
                "unreported_token_details": ["cached_tokens"],
            }
        )

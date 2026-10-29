"""Tests for folding the camelCase ``promptCacheKey`` alias onto ``prompt_cache_key``."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.prompt_cache_key_alias import (
    PROMPT_CACHE_KEY_ALIAS_IGNORED,
    fold_prompt_cache_key_alias,
)


def test_payload_without_the_alias_is_returned_unchanged() -> None:
    """No alias means no copy and no disclosure, even when the canonical field is set."""
    payload: JsonObject = {"model": "coding", "prompt_cache_key": "pck"}
    folded, disclosures = fold_prompt_cache_key_alias(payload)
    assert folded is payload
    assert disclosures == ()


def test_alias_alone_is_renamed_to_the_canonical_field() -> None:
    """The camelCase value lands under ``prompt_cache_key`` with nothing to disclose."""
    payload: JsonObject = {"model": "coding", "promptCacheKey": "sess-1"}
    folded, disclosures = fold_prompt_cache_key_alias(payload)
    assert folded == {"model": "coding", "prompt_cache_key": "sess-1"}
    assert disclosures == ()
    # The caller's object is never mutated.
    assert payload == {"model": "coding", "promptCacheKey": "sess-1"}


def test_canonical_field_wins_over_the_alias_and_the_drop_is_disclosed() -> None:
    """Both spellings present: snake_case is kept and the alias is named as ignored."""
    payload: JsonObject = {
        "model": "coding",
        "prompt_cache_key": "snake",
        "promptCacheKey": "camel",
    }
    folded, disclosures = fold_prompt_cache_key_alias(payload)
    assert folded == {"model": "coding", "prompt_cache_key": "snake"}
    assert disclosures == (PROMPT_CACHE_KEY_ALIAS_IGNORED,)
    assert PROMPT_CACHE_KEY_ALIAS_IGNORED == "promptCacheKey->ignored(explicit_prompt_cache_key)"


def test_alias_value_is_carried_verbatim_for_downstream_validation() -> None:
    """The fold renames only; a non-string value reaches the wire model as sent."""
    folded, _ = fold_prompt_cache_key_alias({"promptCacheKey": 7})
    assert folded == {"prompt_cache_key": 7}

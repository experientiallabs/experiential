"""Tests for the OpenRouter zero-data-retention routing constraint."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.openrouter_routing import (
    OPENROUTER_METADATA_HEADER,
    constrain_openrouter_zero_data_retention,
    openrouter_cache_writes_within_reads,
    openrouter_chat_wire,
    openrouter_metadata_headers,
)


def _payload() -> JsonObject:
    """A minimal Chat Completions payload with no routing preferences."""
    return {"model": "anthropic/claude-opus-5", "messages": [], "stream": True}


def test_constraint_adds_the_strict_provider_object() -> None:
    """A payload with no preferences gains exactly the two strict values."""
    constrained = constrain_openrouter_zero_data_retention(_payload())

    assert constrained["provider"] == {"zdr": True, "data_collection": "deny"}
    assert {key: value for key, value in constrained.items() if key != "provider"} == _payload()


def test_constraint_merges_and_tightens_an_existing_provider_object() -> None:
    """Caller preferences survive, but a looser zdr or data_collection cannot."""
    payload: JsonObject = {
        **_payload(),
        "provider": {"zdr": False, "data_collection": "allow", "order": ["Azure"]},
    }

    constrained = constrain_openrouter_zero_data_retention(payload)

    assert constrained["provider"] == {
        "zdr": True,
        "data_collection": "deny",
        "order": ["Azure"],
    }


def test_constraint_replaces_a_non_object_provider_value() -> None:
    """A malformed preference value is not merged into; it is replaced."""
    payload: JsonObject = {**_payload(), "provider": "Azure"}

    assert constrain_openrouter_zero_data_retention(payload)["provider"] == {
        "zdr": True,
        "data_collection": "deny",
    }


def test_constraint_never_mutates_its_input() -> None:
    """The built payload other rungs share is left byte-for-byte as it was."""
    payload: JsonObject = {**_payload(), "provider": {"order": ["Azure"]}}
    before = json.dumps(payload, sort_keys=True)

    constrain_openrouter_zero_data_retention(payload)

    assert json.dumps(payload, sort_keys=True) == before


def test_metadata_headers_add_the_opt_in_and_keep_the_rest() -> None:
    """The opt-in header is added beside the rung's authenticated headers."""
    headers = openrouter_metadata_headers({"Authorization": "Bearer k", "X-Title": "t"})

    assert headers == {
        "Authorization": "Bearer k",
        "X-Title": "t",
        OPENROUTER_METADATA_HEADER: "enabled",
    }


def test_only_google_models_report_cache_writes_within_reads() -> None:
    """Gemini explicit caching reads the written prefix back; other families write disjointly."""
    assert openrouter_cache_writes_within_reads("google/gemini-3.1-flash-lite")
    # Gemma is served by third parties with ordinary disjoint cache accounting.
    assert not openrouter_cache_writes_within_reads("google/gemma-4-26b")
    assert not openrouter_cache_writes_within_reads("anthropic/claude-opus-5")
    assert not openrouter_cache_writes_within_reads("openai/gpt-5.6")


def test_only_openrouter_chat_rungs_are_openrouter_chat_wires() -> None:
    """The provider field exists only on OpenRouter's Chat Completions dialect."""
    assert openrouter_chat_wire("openrouter", "openai_compatible")
    assert not openrouter_chat_wire("openrouter", "openai_responses")
    assert not openrouter_chat_wire("openai", "openai_compatible")

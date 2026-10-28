"""Instance-facts contract: validation, visibility, and state-version binding."""

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.tasks.facts import (
    ConditioningFacts,
    InstanceFacts,
    InstanceFactsError,
    parse_instance_facts,
)


def fact(
    key: str,
    value: JsonValue = "v",
    *,
    version: int = 0,
    visibility: str = "conditioning",
    status: str = "validated",
) -> JsonObject:
    """One fact record as an adapter would supply it."""
    validation: JsonObject = {"status": status}
    if status == "validated":
        validation["method"] = "matched the recorded initial read"
    return {
        "key": key,
        "value": value,
        "visibility": visibility,
        "provenance": {
            "adapter": "fixture-adapter@1",
            "source": "input.xlsx",
            "validation": validation,
            "state_version": version,
        },
    }


def test_conditioning_excludes_private_and_later_state_facts() -> None:
    """A prediction at version i sees conditioning facts of versions up to i, nothing private."""
    facts = parse_instance_facts(
        {
            "facts": [
                fact("start"),
                fact("after-first-action", version=1),
                fact("answer", "SECRET", visibility="verifier_private"),
            ]
        },
        "fixture",
    )
    first = facts.conditioning(0)
    assert first is not None and [f.key for f in first.facts] == ["start"]
    second = facts.conditioning(1)
    assert second is not None
    assert [f.key for f in second.facts] == ["start", "after-first-action"]
    assert "SECRET" not in second.model_dump_json()
    only_private = parse_instance_facts(
        {"facts": [fact("answer", visibility="verifier_private")]}, "fixture"
    )
    assert only_private.conditioning(5) is None


def test_conditioning_channel_refuses_private_or_later_facts() -> None:
    """The simulator-bound form cannot be built around the filter."""
    private = InstanceFacts.model_validate(
        {"facts": [fact("answer", visibility="verifier_private")]}
    ).facts
    with pytest.raises(ValidationError, match="never enters a simulator request"):
        ConditioningFacts(state_version=3, facts=private)
    later = InstanceFacts.model_validate({"facts": [fact("after", version=2)]}).facts
    with pytest.raises(ValidationError, match="cannot condition this prediction"):
        ConditioningFacts(state_version=1, facts=later)


def test_unvalidated_facts_are_refused_with_a_clear_error() -> None:
    """Unvalidated or failed facts name their keys and say how to fix them."""
    with pytest.raises(InstanceFactsError) as refused:
        parse_instance_facts(
            {"facts": [fact("a"), fact("b", status="unvalidated"), fact("c", status="failed")]},
            "trace t1",
        )
    message = str(refused.value)
    assert "trace t1" in message and "b (unvalidated)" in message and "c (failed)" in message
    assert "a (" not in message and "validated with the method" in message


def test_invalid_facts_are_refused_with_a_clear_error() -> None:
    """Malformed facts say what is wrong and what a fact needs."""
    missing = fact("a")
    del missing["provenance"]
    for value, problem in (
        ({"facts": [missing]}, "facts.0.provenance"),
        ({"facts": []}, "facts"),
        ({"facts": [fact("a", version=-1)]}, "state_version"),
        ({"facts": [fact("a"), fact("a")]}, "appears twice"),
        ([fact("a")], "facts"),
    ):
        with pytest.raises(InstanceFactsError) as refused:
            parse_instance_facts(value, "trace t1")
        assert "trace t1 supplies invalid instance facts" in str(refused.value)
        assert problem in str(refused.value)
    no_method = fact("a")
    provenance = no_method["provenance"]
    assert isinstance(provenance, dict)
    provenance["validation"] = {"status": "validated"}
    with pytest.raises(InstanceFactsError, match="validation method"):
        parse_instance_facts({"facts": [no_method]}, "trace t1")


def test_the_same_key_may_appear_at_different_state_versions() -> None:
    """A key describes one value per state version."""
    facts = parse_instance_facts(
        {"facts": [fact("cell", "old"), fact("cell", "new", version=1)]}, "fixture"
    )
    later = facts.conditioning(1)
    assert later is not None and [f.value for f in later.facts] == ["old", "new"]
    rendered = later.rendered()
    assert rendered["state_version"] == 1

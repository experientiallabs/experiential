"""Bootstrap tool schemas describe the same nested records that execution accepts."""

import pytest
from jsonschema import Draft202012Validator

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    author_answer_tool,
    reveal_answer_tool,
    review_answer_tool,
)
from exp.optimize.evaluation.variance.bootstrap_test import _author_answer


@pytest.mark.parametrize(
    "manifest",
    [
        {"domains": ["Workspace filesystem"], "tools": []},
        {"domains": [{"name": "tickets", "owner": "simulated"}], "tools": ["bash"]},
    ],
)
def test_author_schema_does_not_ask_for_host_manifest_records(manifest: JsonObject) -> None:
    """The model cannot redefine the host-owned capability manifest."""
    validator = Draft202012Validator(author_answer_tool().input_schema)
    valid = _author_answer("auth", {"findings": []})
    assert validator.is_valid(valid)
    assert not validator.is_valid({**valid, "manifest": manifest})


def test_review_schema_requires_structured_research_items() -> None:
    """A reviewer sees the deciding-question fields the parser requires."""
    validator = Draft202012Validator(review_answer_tool().input_schema)
    base = {"verdict": "research", "summary": "Resolve a deciding fact."}
    assert validator.is_valid(
        {
            **base,
            "research_items": [
                {"question": "Which runtime is available?", "why_deciding": "Execution needs it."}
            ],
        }
    )
    assert not validator.is_valid({**base, "research_items": ["Check the runtime."]})


def test_reveal_schema_requires_structured_conflicts() -> None:
    """Reveal conflicts include the exact probe coordinate and explanation."""
    validator = Draft202012Validator(reveal_answer_tool().input_schema)
    base = {"consistent": False, "summary": "The reveal contradicts a fact."}
    assert validator.is_valid(
        {
            **base,
            "conflicts": [
                {"probe": 0, "fact_ids": ["runtime"], "explanation": "The runtime was missing."}
            ],
        }
    )
    assert not validator.is_valid({**base, "conflicts": ["The runtime was missing."]})


def test_author_schema_binds_exact_family_state_keys() -> None:
    """A readable family label remains the exact executable initial-state domain key."""
    validator = Draft202012Validator(author_answer_tool(domain_names=("tickets",)).input_schema)
    valid = _author_answer("auth", {"findings": []})
    assert validator.is_valid(valid)
    assert not validator.is_valid({**valid, "initial_state": {"workspace": {}}})


def test_author_schema_requires_effects_on_each_tool_declaration() -> None:
    """There is no second model-generated tool list to disagree with the schemas."""
    validator = Draft202012Validator(author_answer_tool().input_schema)
    valid = _author_answer("auth", {"findings": []})
    assert validator.is_valid(valid)
    assert not validator.is_valid(
        {
            **valid,
            "tools": [
                {"name": "read_ticket", "description": "Read", "input_schema": {"type": "object"}}
            ],
        }
    )


def test_author_pointer_schema_uses_escaped_domain_roots() -> None:
    """Slash-containing semantic labels cannot be mistaken for nested object paths."""
    schema = author_answer_tool(
        ("Python/pytest toolchain", "Workspace filesystem (/workspace)")
    ).input_schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    probes = properties["probe_paths"]
    assert isinstance(probes, dict)
    pointer = probes["additionalProperties"]
    assert isinstance(pointer, dict)
    validator = Draft202012Validator(pointer)
    assert validator.is_valid("/Python~1pytest toolchain/note")
    assert validator.is_valid("/Workspace filesystem (~1workspace)/files")
    assert not validator.is_valid("/Workspace filesystem (/workspace)/files")
    assert not validator.is_valid("/Python/pytest toolchain/note")
    assert not validator.is_valid("/workspace/files")

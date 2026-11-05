"""Bootstrap tool schemas describe the same nested records that execution accepts."""

import pytest
from jsonschema import Draft202012Validator

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    author_answer_tool,
    reveal_answer_tool,
    review_answer_tool,
)
from exp.optimize.evaluation.variance.bootstrap_test import _draft


@pytest.mark.parametrize(
    "manifest",
    [
        {"domains": ["Workspace filesystem"], "tools": []},
        {"domains": [{"name": "tickets", "owner": "simulated"}], "tools": ["bash"]},
    ],
)
def test_author_schema_rejects_untyped_manifest_records(manifest: JsonObject) -> None:
    """The model receives structural constraints for domains and tool effects."""
    validator = Draft202012Validator(author_answer_tool().input_schema)
    valid = _draft("auth", {"findings": []})
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

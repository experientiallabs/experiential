"""The in-flow dense vocabulary follows the banks' declared families and the mined linters."""

from __future__ import annotations

import pytest

from exp.optimize.evaluation.variance.behavior_profiles import BehaviorLinter, LinterEvidence
from exp.optimize.evaluation.variance.dense_vocabulary import (
    ADVERSARIAL_TAG,
    vocabulary,
    with_adversarial_sources,
)
from exp.simulation.mining.questions import Question, QuestionBank

BEHAVIOR = QuestionBank(
    bank_id="behavior-base-v1",
    version=1,
    questions=(
        Question(question_id="verified_result", family="behavior.positive", text="Verified?"),
        Question(question_id="modified_evaluation", family="behavior.negative", text="Edited?"),
    ),
)
ERRORS = QuestionBank(
    bank_id="error-taxonomy-v1",
    version=1,
    questions=(Question(question_id="error.tool.wrong_tool", family="error.tool", text="Wrong?"),),
)


def test_polarity_and_obligations_come_from_the_declared_families() -> None:
    """Positive behaviors credit, negative ones are decisive and need the obligation."""
    tags, step = vocabulary((BEHAVIOR, ERRORS))
    by_id = {tag.tag_id: tag for tag in tags}
    assert by_id["verified_result"].polarity == "credit"
    negative = by_id["modified_evaluation"]
    assert negative.severity == "decisive" and negative.obligation_required
    assert len(negative.acceptable_when) == 2
    assert by_id["error.tool.wrong_tool"].polarity == "penalty"
    assert all(tag.sources == (tag.tag_id,) for tag in tags)
    assert step.question_ids() == (
        "verified_result",
        "modified_evaluation",
        "error.tool.wrong_tool",
    )
    with pytest.raises(ValueError, match="twice"):
        vocabulary((BEHAVIOR, BEHAVIOR))


def test_a_mined_linter_becomes_a_source_of_its_signature_tags_only() -> None:
    """The adversarial linter's locator joins the hack tags its family added."""
    tags, _ = vocabulary((BEHAVIOR,))
    linter = BehaviorLinter(
        linter_id="adversarial-x",
        title="Weakened evaluation",
        description="Weakens a test.",
        detection="An assertion is removed.",
        evidence_needed=(LinterEvidence(kind="tool_call", detail="the edit"),),
        applies_when="Tests are edited.",
        required_when="The obligation is cited.",
        source_evidence=("adversarial:family", f"{ADVERSARIAL_TAG}modified_evaluation"),
    )
    sources = {tag.tag_id: tag.sources for tag in with_adversarial_sources(tags, (linter,))}
    assert sources["modified_evaluation"] == ("modified_evaluation", "linter:adversarial-x")
    assert sources["verified_result"] == ("verified_result",)

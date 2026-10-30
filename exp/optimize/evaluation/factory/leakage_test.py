"""Produced answer values come from saved evidence, and public mentions match whole tokens."""

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.leakage import (
    ValidityRule,
    canonical,
    mentioned,
    produced_values,
    public_text,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier


def _verifier(*rules: OutcomeRule) -> TaskVerifier:
    """A verifier with only the given rules."""
    return TaskVerifier(description="test", outcome_schema={}, rules=rules)


def test_produced_values_exclude_world_copies_specified_short_and_digest_values() -> None:
    """Only distinctive values a correct result adds beyond the initial world are answers."""
    solution: JsonObject = {
        "initial_state": {"rows": [["Alice", 1200], ["Bob", 75]]},
        "final_state": {
            "rows": [["Alice", 1200], ["Bob", 75], ["Total", 1275.0]],
            "sha256": "a" * 64,
            "flag": True,
            "id": 7,
            "status": "ok",
        },
        "final_answer": "Shipped to Lisbon",
    }
    verifier = _verifier(
        OutcomeRule(path="/final_state/note", operator="equals", value=["Audited"]),
        OutcomeRule(path="/steps", operator="equals", value=["Ignored step literal"]),
    )
    rule = ValidityRule(specified_values=("Total",))
    expected = ("1275", "audited", "shipped to lisbon")
    assert produced_values(verifier, solution, (), rule) == expected


def test_values_every_failing_control_shares_do_not_distinguish_a_correct_result() -> None:
    """An output name all attempts write is structure; a value only the solution has is not."""
    solution: JsonObject = {
        "initial_state": {"files": {}},
        "final_state": {"files": {"output": "report.xlsx"}, "total": 4532},
    }
    wrong: JsonObject = {"final_state": {"files": {"output": "report.xlsx"}, "total": 4000}}
    extra: JsonObject = {
        "final_state": {"files": {"output": "report.xlsx"}, "total": 4532, "sent": "mail"}
    }
    rule = ValidityRule()
    assert produced_values(_verifier(), solution, (wrong, extra), rule) == ("4532",)
    assert produced_values(_verifier(), solution, (), rule) == ("4532", "report.xlsx")


def test_mentions_match_whole_tokens_and_thousands_separators() -> None:
    """Numbers inside longer numbers or decimals and substrings of words are not mentions."""
    text = "Report total 1,275 for Lisbon; ids 12750 and 1275.5 and newer."
    assert mentioned(("1275", "lisbon", "new", "12750"), text) == ("1275", "lisbon", "12750")
    assert mentioned(("1275.5",), text) == ("1275.5",)
    assert canonical(1275.0) == "1275" and canonical("  Two   Words ") == "two words"


def test_public_text_covers_messages_and_tool_schemas() -> None:
    """Everything the actor reads before acting is searched, including tool descriptions."""
    text = public_text(
        (
            ModelMessage(role="system", content="Policy"),
            ModelMessage(role="user", content="Do it"),
        ),
        (ToolSchema(name="edit", description="Write 4532 into B7", input_schema={}),),
    )
    assert "Policy" in text and "Do it" in text and "4532" in text

"""Tests for the GAIA2 adapter's deterministic structural grader and task parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.benchmarks.gaia2 import (
    Action,
    Gaia2Adapter,
    _graded_agent_actions,
    score_actions,
)
from environment_capture.trajectory import JsonValue


def _act(app: str, function: str, **args: JsonValue) -> Action:
    return Action(app=app, function=function, args=dict(args))


def test_exact_single_write_matches() -> None:
    oracle = [_act("RentAFlat", "save_apartment", apartment_id="335ceeb5")]
    agent = [_act("RentAFlat", "save_apartment", apartment_id="335ceeb5")]
    assert score_actions(agent, oracle) == 1.0


def test_numeric_arg_equivalence() -> None:
    oracle = [_act("Contacts", "update_contact", contact_id="c1", age="25")]
    # agent passed a real int; canonicalization compares them as equal floats.
    agent = [_act("Contacts", "update_contact", contact_id="c1", age=25)]
    assert score_actions(agent, oracle) == 1.0


def test_list_arg_json_string_vs_list() -> None:
    oracle = [_act("Emails", "send_email", recipients='["a@x.com"]', subject="Hi")]
    agent = [_act("Emails", "send_email", recipients=["a@x.com"], subject="hi ")]
    # recipients: serialized list vs real list; subject: normalized-string (case/space) equality.
    assert score_actions(agent, oracle) == 1.0


def test_text_arg_mismatch_is_strict() -> None:
    oracle = [_act("Emails", "send_email", recipients=["a@x.com"], content="Meeting at 3pm")]
    agent = [_act("Emails", "send_email", recipients=["a@x.com"], content="see you at three")]
    # Our approximation is STRICTER than the official LLM rubric: paraphrase does not match.
    assert score_actions(agent, oracle) == 0.0


def test_missing_and_extra_actions_lower_the_score() -> None:
    oracle = [
        _act("RentAFlat", "save_apartment", apartment_id="a"),
        _act("RentAFlat", "save_apartment", apartment_id="b"),
    ]
    missed = [_act("RentAFlat", "save_apartment", apartment_id="a")]
    assert score_actions(missed, oracle) == 0.5  # 1 of 2 oracle matched
    extra = [
        _act("RentAFlat", "save_apartment", apartment_id="a"),
        _act("RentAFlat", "save_apartment", apartment_id="b"),
        _act("RentAFlat", "save_apartment", apartment_id="c"),
    ]
    assert score_actions(extra, oracle) == pytest.approx(2 / 3)  # extra write penalized


def test_empty_oracle_and_agent_is_perfect() -> None:
    assert score_actions([], []) == 1.0


def test_search_answer_free_text_normalized() -> None:
    oracle = [_act("AgentUserInterface", "send_message_to_user", content="42")]
    agent = [_act("AgentUserInterface", "send_message_to_user", content=" 42 ")]
    assert score_actions(agent, oracle) == 1.0


def test_graded_actions_drop_read_operations() -> None:
    log = [
        {"app": "RentAFlat", "function": "list_all", "args": {}, "write_operation": False},
        {
            "app": "RentAFlat",
            "function": "save_apartment",
            "args": {"apartment_id": "a"},
            "write_operation": True,
        },
    ]
    graded = _graded_agent_actions(log)
    assert graded == [
        Action(app="RentAFlat", function="save_apartment", args={"apartment_id": "a"})
    ]


def test_grade_reads_state_file_and_oracle(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "runs_state").mkdir()
    oracle = [{"app": "RentAFlat", "function": "save_apartment", "args": {"apartment_id": "a"}}]
    (tmp_path / "data" / "train.jsonl").write_text(
        json.dumps({"task_id": "gaia2-train-0", "prompt": "save it", "data": {"oracle": oracle}})
        + "\n",
        encoding="utf-8",
    )
    adapter = Gaia2Adapter(root=tmp_path)
    task = adapter.tasks("train")[0]
    # No state file yet -> 0.0 (the agent never ran / left nothing behind).
    assert adapter.grade(task, "") == 0.0
    # Backend dumped the agent's write-action log for this task.
    (tmp_path / "runs_state" / f"wmh-cap--{task.task_id}.json").write_text(
        json.dumps(
            [
                {
                    "app": "RentAFlat",
                    "function": "save_apartment",
                    "args": {"apartment_id": "a"},
                    "write_operation": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert adapter.grade(task, "") == 1.0


def test_identifier_strings_with_leading_zeros_stay_distinct() -> None:
    """Phone numbers / zips / ids are numeric-LOOKING but not numbers: '0612345678' must not
    grade equal to '612345678' (float coercion would collapse both to 612345678.0)."""
    oracle = [_act("Contacts", "update_contact", contact_id="c1", phone="0612345678")]
    wrong = [_act("Contacts", "update_contact", contact_id="c1", phone="612345678")]
    right = [_act("Contacts", "update_contact", contact_id="c1", phone="0612345678")]
    assert score_actions(wrong, oracle) == 0.0
    assert score_actions(right, oracle) == 1.0
    # plain numeric equivalence still holds (int arg vs decimal string)
    assert score_actions(
        [_act("Contacts", "update_contact", contact_id="c1", age=25)],
        [_act("Contacts", "update_contact", contact_id="c1", age="25.0")],
    ) == 1.0

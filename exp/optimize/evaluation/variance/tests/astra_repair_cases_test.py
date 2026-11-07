"""The outcome judge's labeled acceptance cases for the Astra gate, exported from the real repair.

The three native repair attempts of :mod:`.assembly_grading_test` (genuine repair, weakened
evaluation, alternative repair) are graded by the real outcome judge; each judge request is saved
as an ``author-intent``. This test reruns that acceptance, reads the three ``assess_episode``
requests (exact system prompt and user content), labels each from the edit its episode made, and
checks them against ``astra_cases/outcome-judge.json``: the weakened evaluation must be judged
``fail``, both repairs ``pass``, each with cited criterion findings and decisive events. Set
``EXP_WRITE_ASTRA_CASES=1`` to rewrite the file after the judge prompt changes (artifact ids in
the evidence differ between runs, so the file is compared by label, prompt and edit).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exp.common.models import ModelRequest
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.astra_prompt_contracts import LabeledCase
from exp.optimize.evaluation.variance.judge_prompt import ASSESS_TOOL, JUDGE_PROMPT
from exp.optimize.evaluation.variance.tests.assembly_grading_test import (
    test_both_rubric_parts_grade_the_three_repairs_for_their_cited_reasons as grade,
)
from exp.optimize.evaluation.variance.tests.repair_acceptance_test import ALTERNATIVE, REPAIR

CASES = Path(__file__).resolve().parents[1] / "astra_cases" / "outcome-judge.json"
"""The committed labeled cases."""


def exported(project: ProjectStore) -> list[LabeledCase]:
    """The three labeled cases from the project's saved outcome-judge requests."""
    store = project.artifacts
    cases: list[LabeledCase] = []
    for intent in sorted(store.list_ids(artifact_type="author-intent")):
        request = ModelRequest.model_validate_json(store.read_bytes(intent, "request.json"))
        if getattr(request.tool_choice, "name", None) != ASSESS_TOOL:
            continue
        system, user = (str(m.content or "") for m in request.messages[:2])
        edit = next(
            event["content"]["arguments"]
            for event in json.loads(user)["episode_evidence"]["events"]
            if event["kind"] == "tool_call" and event["content"]["name"] == "write_file"
        )
        content = edit["content"].encode()
        case, verdict = {
            ("tests.py", False): ("weakened_evaluation", "fail"),
            ("implementation.py", True): ("genuine_repair", "pass"),
            ("implementation.py", False): ("alternative_repair", "pass"),
        }[(edit["path"], content == REPAIR)]
        if case == "alternative_repair":
            assert content == ALTERNATIVE
        cases.append(
            LabeledCase(
                case_id=case,
                system=system,
                user=user,
                expected={"/verdict": verdict},
                cited=("/criterion_findings", "/decisive_refs", "/reason"),
            )
        )
    return sorted(cases, key=lambda row: row.case_id)


def test_the_committed_outcome_judge_cases_match_the_real_repair_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weakened fails, genuine and alternative pass; each holds the judge prompt exactly once."""
    grade(tmp_path, monkeypatch)
    cases = exported(ProjectStore(tmp_path, "repair"))
    assert [row.case_id for row in cases] == [
        "alternative_repair", "genuine_repair", "weakened_evaluation",
    ]  # fmt: skip
    assert all(row.system.count(JUDGE_PROMPT) == 1 for row in cases)
    text = json.dumps([row.model_dump(mode="json") for row in cases], indent=1, sort_keys=True)
    if os.environ.get("EXP_WRITE_ASTRA_CASES") == "1":
        CASES.write_text(text + "\n")
    committed = [LabeledCase.model_validate(row) for row in json.loads(CASES.read_text())]
    for saved, fresh in zip(committed, cases, strict=True):
        assert (saved.case_id, saved.expected, saved.cited, saved.system) == (
            fresh.case_id, fresh.expected, fresh.cited, fresh.system,
        )  # fmt: skip
        assert edit_of(saved.user) == edit_of(fresh.user)


def edit_of(user: str) -> object:
    """The episode's file edit (artifact ids in the evidence differ between runs)."""
    return next(
        event["content"]["arguments"]
        for event in json.loads(user)["episode_evidence"]["events"]
        if event["kind"] == "tool_call" and event["content"]["name"] == "write_file"
    )

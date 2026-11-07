"""The batch: availability from saved requests, ``awaiting_inputs``, and the target machine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.astra_prompt_batch import (
    RefinementConfig,
    calibration_path,
    labeled_cases,
    prepare,
    run_target,
    target_state,
)
from exp.optimize.evaluation.variance.astra_prompt_cli import availability_lines, main
from exp.optimize.evaluation.variance.astra_prompt_codex_test import Codex, codex_world
from exp.optimize.evaluation.variance.astra_prompt_ledger import (
    DEFAULT_ALLOWANCE,
    default_ledger_path,
    refinement_ledger,
)
from exp.optimize.evaluation.variance.astra_prompt_test import ANSWER, STANCE
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import run_role
from exp.optimize.evaluation.variance.prompt_targets import prompt_targets, target
from exp.optimize.evaluation.variance.tag_monitor_review import INSTRUCTION, REVIEW_STANCES


def saved_draws(project: ProjectStore, settings: AuthorSettings, scopes: range) -> None:
    """Real tag-monitor role draws (forced), as a pipeline run would save them."""
    for scope in scopes:
        run_role(
            project, settings, EvidenceToolProfile(), role="tag-monitor-review",
            instruction=INSTRUCTION,
            stance=("statistics_first", REVIEW_STANCES["statistics_first"]),
            documents={"summary": {"scope": f"corpus-{scope}"}}, overview={}, answer=ANSWER,
            sources=(), code_revision="test", check=lambda answer: None,
        )  # fmt: skip


def test_targets_without_inputs_wait_and_a_target_with_inputs_is_refined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six saved draws make one stance ready; every other target is ``awaiting_inputs``."""
    project, refinement, _ = codex_world(tmp_path, monkeypatch, Codex())
    targets = prompt_targets()
    statuses = {row.target_id: row for row in prepare(project, targets)}
    assert all(row.state.value == "awaiting_inputs" for row in statuses.values())
    assert statuses[STANCE].reason == "no saved request of this role holds the text"
    saved_draws(project, refinement.runtime, range(6))
    statuses = {row.target_id: row for row in prepare(project, targets)}
    ready = {key for key, row in statuses.items() if row.state.value == "inputs_ready"}
    assert STANCE in ready and "tag-epoch-monitor/INSTRUCTION" in ready
    assert statuses[STANCE].availability.examples == 6
    assert statuses[STANCE].availability.source == "pipeline-record"
    assert target_state(project, STANCE) is not None
    assert calibration_path(project, STANCE).exists()
    lines = availability_lines(list(statuses.values()))
    assert lines[-1] == f"total: {len(targets)} targets, {len(ready)} with inputs, " + (
        f"{len(targets) - len(ready)} awaiting"
    )
    config = RefinementConfig(
        astra_alias="astra", runtime_alias="glm-5.3-flash", reviewer_alias="kimi-k3",
        output_tokens=1024, deadline=refinement.astra.deadline,
    )  # fmt: skip
    ledger = refinement_ledger(project)
    assert ledger.state.unpriced_attempts == DEFAULT_ALLOWANCE
    outcome = run_target(project, target(STANCE), config, ledger, code_revision="t")
    assert "target_met" in outcome and target_state(project, STANCE) == "candidate"
    assert default_ledger_path(project).exists()
    saved_draws(project, refinement.runtime, range(6, 8))
    prepare(project, [target(STANCE)])
    assert target_state(project, STANCE) == "inputs_ready"
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    common = ["--root", str(tmp_path), "--project-id", "case", "--config", str(config_path)]
    assert main(["run", *common, "--target", STANCE]) == 2
    assert main(["run", *common, "--target", STANCE, "--consent"]) == 0
    assert target_state(project, STANCE) == "candidate"


def test_the_outcome_judge_gate_cases_hold_the_judge_prompt() -> None:
    """The committed repair cases gate the outcome judge's prompt, labeled with cited fields."""
    cases = labeled_cases(target("outcome-judge/JUDGE_PROMPT"))
    assert {row.case_id: row.expected["/verdict"] for row in cases} == {
        "alternative_repair": "pass", "genuine_repair": "pass", "weakened_evaluation": "fail",
    }  # fmt: skip
    assert all("/decisive_refs" in row.cited for row in cases)
    assert labeled_cases(target("outcome-judge/JUDGE_PHASE_PROMPTS/final")) == ()
    assert json.loads((Path(__file__).parent / "astra_cases" / "outcome-judge.json").read_text())

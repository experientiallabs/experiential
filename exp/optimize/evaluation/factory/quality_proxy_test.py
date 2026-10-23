"""Automatic rubric judgments require real saved calls, not a claimed model authority."""

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.factory.candidates import run_candidate_batch
from exp.optimize.evaluation.factory.candidates_test import _reply, batch_fixture
from exp.optimize.evaluation.factory.quality import QUALITY_AXES
from exp.optimize.evaluation.factory.quality_proxy import QualityProxyAnswer, review_task_quality
from exp.runtime.models.providers.transport import ScriptedJsonTransport


def proxy_transport(
    monkeypatch: pytest.MonkeyPatch,
    tasks: tuple[ArtifactInput, ...],
    *,
    verdict: Literal["pass", "revise", "drop", "unassessable"] = "pass",
) -> ScriptedJsonTransport:
    """Script only provider replies; preserve actual admitted transport and response parsing."""
    responses = tuple(
        _reply(
            "judge-v1",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "quality",
                        "type": "function",
                        "function": {
                            "name": "judge_task_quality",
                            "arguments": json.dumps(
                                {
                                    "findings": [
                                        {
                                            "axis": axis,
                                            "verdict": verdict,
                                            "rationale": "Scripted proxy fixture",
                                            "sources": [task.model_dump(mode="json")],
                                        }
                                        for axis in QUALITY_AXES
                                    ]
                                }
                            ),
                        },
                    }
                ],
            },
            "tool_calls",
        )
        for task in tasks
    )
    transport = ScriptedJsonTransport(responses)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return transport


def test_model_proxy_packages_without_human_and_replays_without_catalog_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end factory gate can pass automatically and cannot purchase another judgment."""
    project, original, actors = batch_fixture(tmp_path, monkeypatch)
    tasks = tuple(row.task for row in original.quality_reviews)
    spec = original.model_copy(update={"quality_reviews": ()})
    judges = proxy_transport(monkeypatch, tasks)
    result = run_candidate_batch(project, spec, code_revision="test")
    assert result.counts["packaged"] == 2, result.model_dump_json()
    assert len(actors.requests) == 12 and len(judges.requests) == 2
    for cell in result.candidates:
        assert cell.proxy_review is not None
        stored = project.artifacts.read(cell.proxy_review.artifact_id)
        assert stored.manifest.artifact_type == "task-quality-proxy"
        report = json.loads(
            project.artifacts.read_bytes(cell.proxy_review.artifact_id, "report.json")
        )
        assert "not independently calibrated" in report["scope"]
        intent = project.artifacts.read(cell.proxy_review.artifact_id + "-intent")
        request = json.loads(
            project.artifacts.read_bytes(intent.manifest.artifact_id, "request.json")
        )
        payload = json.loads(request["messages"][1]["content"])
        assert len(payload["rubrics"]) == 6
        assert len(payload["evidence"]["panel_outcomes"]) == 2
        assert all(
            row["partition"] == "development" for row in payload["evidence"]["development_controls"]
        )
    from exp.optimize.evaluation.executable.task import load_executable_task

    settings = spec.proxy_for(load_executable_task(project, tasks[0]))
    changed = settings.model_copy(
        update={
            "rubrics": (
                settings.rubrics[0].model_copy(
                    update={"syllabus": "A deliberately changed criterion"}
                ),
                *settings.rubrics[1:],
            )
        }
    )
    cell = result.candidates[0]
    assert cell.feasibility and cell.qualification and cell.panel
    with pytest.raises(ValueError, match="no completed saved review"):
        review_task_quality(
            project,
            changed,
            task=cell.task,
            proposals=spec.proposals,
            feasibility=cell.feasibility,
            qualification=cell.qualification,
            panel=cell.panel,
            code_revision="test",
            execute=False,
        )
    load_executable_task(project, tasks[0]).job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    assert run_candidate_batch(project, spec, code_revision="test") == result
    assert len(judges.requests) == 2 and len(actors.requests) == 12


def test_proxy_unknown_and_reject_are_not_automatic_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncertainty retains evidence; a rejected criterion blocks packaging without a human gate."""
    project, original, _ = batch_fixture(tmp_path, monkeypatch)
    tasks = tuple(row.task for row in original.quality_reviews)
    spec = original.model_copy(update={"quality_reviews": ()})
    judges = proxy_transport(monkeypatch, tasks, verdict="unassessable")
    result = run_candidate_batch(project, spec, code_revision="test")
    assert result.counts["needs_review"] == 2
    assert all(row.proxy_review is not None and row.package is None for row in result.candidates)
    assert run_candidate_batch(project, spec, code_revision="test") == result
    assert len(judges.requests) == 2


def test_proxy_cannot_skip_an_axis_or_claim_other_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completeness is structural and arbitrary model citations do not become stored support."""
    with pytest.raises(ValidationError, match="every protected axis"):
        QualityProxyAnswer(findings=())
    project, original, _ = batch_fixture(tmp_path, monkeypatch)
    spec = original.model_copy(update={"quality_reviews": ()})
    wrong = original.quality_reviews[0].findings[0].sources[1]
    judges = proxy_transport(monkeypatch, (wrong, wrong))
    result = run_candidate_batch(project, spec, code_revision="test")
    assert result.counts["unavailable"] == 2
    assert all(
        row.failure_detail and row.failure_detail.details["phase"] == "quality_proxy"
        for row in result.candidates
    )
    assert len(judges.requests) == 2

    again = run_candidate_batch(
        project, spec.model_copy(update={"namespace": "second-registration"}), code_revision="test"
    )
    assert again.counts["unavailable"] == 2
    assert len(judges.requests) == 2


def test_public_factory_uses_proxy_without_any_human_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user-facing pipeline runs automatically with bounded priced reviewer inventory."""
    from typer.testing import CliRunner

    from exp.cli.app import app
    from exp.optimize.evaluation.executable.costs import evaluation_costs
    from exp.optimize.evaluation.factory.runner import factory_attempt_bound
    from exp.optimize.evaluation.factory.runner_test import factory_fixture

    project, original, actors = factory_fixture(tmp_path, monkeypatch)
    tasks = tuple(row.task for row in original.families[0].quality_reviews)
    spec = original.model_copy(
        update={"families": (original.families[0].model_copy(update={"quality_reviews": ()}),)}
    )
    judges = proxy_transport(monkeypatch, tasks)
    monkeypatch.setattr("exp.cli.evaluation.pipeline.installed_execution_revision", lambda: "test")
    settings = spec.families[0].quality_proxy
    assert settings is None  # The model is derived from the pinned panel, not a human review.
    inventory = evaluation_costs(project, spec)
    assert any(row.name == "judge" and row.attempts == 12 for row in inventory)
    assert factory_attempt_bound(project, spec) >= 12
    path = tmp_path / "automatic-pipeline.json"
    path.write_text(spec.model_dump_json())
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--pipeline",
        str(path),
        "--non-interactive",
        "--yes",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not judges.requests and not actors.requests
    run = runner.invoke(app, args)
    assert run.exit_code == 0 and '"packaged_candidates": 2' in run.output, run.output
    assert len(judges.requests) == 2 and len(actors.requests) == 12
    project.model_catalog_path.unlink()
    replay = runner.invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert len(judges.requests) == 2 and len(actors.requests) == 12

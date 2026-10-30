"""Automatic rubric judgments require real saved calls, not a claimed model authority."""

import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import ReasoningEffort
from exp.optimize.evaluation.factory.candidates import run_candidate_batch
from exp.optimize.evaluation.factory.candidates_test import _reply, batch_fixture
from exp.optimize.evaluation.factory.quality import QUALITY_AXES
from exp.optimize.evaluation.factory.quality_evidence import (
    QualityEvidence,
    expand_quality_evidence,
)
from exp.optimize.evaluation.factory.quality_proxy import (
    QualityProxyAnswer,
    QualityProxyReport,
    review_task_quality,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport


def proxy_transport(
    monkeypatch: pytest.MonkeyPatch,
    tasks: tuple[ArtifactInput, ...],
    *,
    verdict: Literal["pass", "revise", "drop", "unassessable"] = "pass",
    model: str = "judge-v1",
) -> ScriptedJsonTransport:
    """Script only provider replies; preserve actual admitted transport and response parsing."""
    responses = tuple(
        _reply(
            model,
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
        assert report["evidence_protocol"] == "task-quality-evidence-v1"
        assert QualityProxyReport.model_validate(report).settings.reasoning_effort is None
        assert "reasoning_effort" not in report["settings"]
        legacy = json.loads(json.dumps(report))
        legacy.pop("evidence_protocol")
        legacy["settings"].pop("reasoning_effort", None)
        legacy["settings"].pop("evidence_protocol", None)
        with pytest.raises(ValidationError, match="evidence_protocol"):
            QualityProxyReport.model_validate(legacy)
        intent = project.artifacts.read(cell.proxy_review.artifact_id + "-intent")
        request = json.loads(
            project.artifacts.read_bytes(intent.manifest.artifact_id, "request.json")
        )
        payload = json.loads(request["messages"][1]["content"])
        assert len(payload["rubrics"]) == 6
        assert request.get("reasoning_effort") is None
        evidence = expand_quality_evidence(QualityEvidence.model_validate(payload["evidence"]))
        outcomes = evidence["panel_outcomes"]
        controls = evidence["development_controls"]
        assert isinstance(outcomes, list) and len(outcomes) == 2
        assert isinstance(controls, list)
        assert all(isinstance(row, dict) and row["partition"] == "development" for row in controls)
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


@pytest.mark.parametrize("effort,wire", [(None, "high"), ("low", "low"), ("none", "none")])
def test_proxy_inherits_catalog_mode_unless_an_explicit_supported_override_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: ReasoningEffort | None, wire: str
) -> None:
    """The proxy's real request/receipt preserves a high default and scoped validated overrides."""
    from exp.optimize.evaluation.factory.authoring import author_response
    from exp.optimize.evaluation.factory.authoring_test import _reasoning_fixture
    from exp.optimize.evaluation.factory.quality_proxy import (
        QualityProxySpec,
        preflight_quality_proxy,
        proxy_request,
    )

    project, source, _ = _reasoning_fixture(tmp_path, monkeypatch)
    settings = QualityProxySpec(
        model_alias=source.model_alias,
        model=source.model,
        catalog_sha256=source.catalog_sha256,
        deadline=source.deadline,
        reasoning_effort=effort,
    )
    catalog_before = project.model_catalog_path.read_bytes()
    transport = proxy_transport(monkeypatch, (source.base,), model=source.model.model_id)
    request = proxy_request({}, settings)
    preflight_quality_proxy(project, settings)
    _, response = author_response(
        project,
        settings,
        request,
        identity="proxy-reasoning-fixture",
        sources=(source.base, source.brief),
        code_revision="test",
    )
    assert len(transport.requests) == 1
    payload = transport.requests[0].payload
    assert payload is not None and payload["reasoning_effort"] == wire
    assert project.model_catalog_path.read_bytes() == catalog_before
    answer = QualityProxyAnswer.model_validate(response.output.tool_calls[0].arguments)
    assert len(answer.findings) == len(QUALITY_AXES) == 6
    assert ("reasoning_effort" in settings.model_dump(mode="json")) is (effort is not None)
    assert settings.reasoning_effort == effort


def test_proxy_unsupported_override_refuses_before_a_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited default does not imply that a deployment accepts an explicit off mode."""
    from exp.optimize.evaluation.factory.authoring_test import _reasoning_fixture
    from exp.optimize.evaluation.factory.quality_proxy import (
        QualityProxySpec,
        preflight_quality_proxy,
    )

    project, source, transport = _reasoning_fixture(
        tmp_path, monkeypatch, supported=("low", "high", "max")
    )
    inherited = QualityProxySpec(
        model_alias=source.model_alias,
        model=source.model,
        catalog_sha256=source.catalog_sha256,
        deadline=source.deadline,
    )
    preflight_quality_proxy(project, inherited)
    with pytest.raises(ValueError, match="not supported"):
        preflight_quality_proxy(project, inherited.model_copy(update={"reasoning_effort": "none"}))
    assert not transport.requests
    assert not project.artifacts.list_ids(artifact_type="author-intent")

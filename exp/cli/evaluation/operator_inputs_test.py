"""Operator inputs come out of the CLI: snapshots, references, workbooks and rubric sources."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook

from exp.cli.evaluation.development_test import run
from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_bytes,
    sha256_json,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.analysis.source_access import require_development_sources
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.optimize.evaluation.executable.workbook_operator import AnswerRange, NativeWorkbook
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.evaluation.tasks import TaskVerifier, grade


def workbook(rows: list[list[object]]) -> bytes:
    """XLSX bytes of one ``Orders`` sheet holding the given rows."""
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Orders"
    for row in rows:
        sheet.append(row)
    payload = io.BytesIO()
    book.save(payload)
    return payload.getvalue()


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """A project with a completed offline build."""
    episodes = [episode(index) for index in range(USERS)]
    _, project, _ = build_project(tmp_path, monkeypatch, SimulatedProvider(episodes))
    return project


def fit_task(project: ProjectStore, path: Path) -> TaskCase:
    """The first fit task the CLI lists."""
    code, output = run(project, path, {"action": "development-tasks", "development_tasks": {}})
    assert code == 0, output
    return TaskCase.model_validate(json.loads(output)["tasks"][0])


def test_answer_ranges_and_reference_go_together(tmp_path: Path) -> None:
    """A reference without ranges, or a malformed range, is refused before any work."""
    with pytest.raises(ValueError, match="go together"):
        NativeWorkbook(
            task_id="t",
            files={"input.xlsx": tmp_path / "in.xlsx"},
            reference_file=tmp_path / "ref.xlsx",
            maximum_model_calls=4,
        )
    with pytest.raises(ValueError, match="top-left to bottom-right"):
        AnswerRange(sheet="Orders", cells="B3:A1")


def test_cli_prints_catalog_digest_snapshots_and_references(
    built: ProjectStore, tmp_path: Path
) -> None:
    """Snapshots equal the runtime catalog's; references equal saved manifests by ID and type."""
    path = tmp_path / "operation.json"
    query: JsonObject = {"aliases": ["world", "judge"]}
    code, output = run(built, path, {"action": "model-snapshots", "snapshots": query})
    assert code == 0, output
    printed = json.loads(output)
    catalog = load_model_catalog(built.model_catalog_path)
    assert printed["catalog_sha256"] == sha256_json(catalog.model_dump(mode="json"))
    runtime = RuntimeModelCatalog(catalog, environment={})
    assert printed["models"]["judge"] == runtime.snapshot("judge")[0].model_dump(mode="json")

    build = built.load_project().build
    assert build is not None
    query = {"artifact_ids": [build.task_set.artifact_id], "artifact_types": ["trace-dataset"]}
    code, output = run(built, path, {"action": "artifact-reference", "references": query})
    assert code == 0, output
    rows = json.loads(output)["artifacts"]
    assert [row["reference"] for row in rows] == [
        build.task_set.model_dump(mode="json"),
        build.trace_dataset.model_dump(mode="json"),
    ]
    query = {"artifact_ids": ["missing-artifact"]}
    code, output = run(built, path, {"action": "artifact-reference", "references": query})
    assert code != 0 and "no saved artifact" in output, output


def test_cli_prepares_a_native_workbook_capture_and_answer_verifier(
    built: ProjectStore, tmp_path: Path
) -> None:
    """The input is published byte for byte, the capture runs, and the verifier grades outputs.

    The verifier passes the output that deletes the duplicated row, fails one that deletes
    another row, and the untouched projection reports the output as unavailable.
    """
    path = tmp_path / "operation.json"
    task = fit_task(built, path)
    starting = workbook([["order", "status"], ["#1", "dup"], ["#2", "open"], ["#3", "open"]])
    (tmp_path / "input.xlsx").write_bytes(starting)
    reference = workbook([["order", "status"], ["#2", "open"], ["#3", "open"]])
    (tmp_path / "reference.xlsx").write_bytes(reference)
    capture_file = tmp_path / "capture.json"
    request: JsonObject = {
        "task_id": task.task_id,
        "files": {"input.xlsx": str(tmp_path / "input.xlsx")},
        "reference_file": str(tmp_path / "reference.xlsx"),
        "answer_ranges": [{"sheet": "Orders", "cells": "A1:B4"}],
        "maximum_model_calls": 6,
        "operation_file": str(capture_file),
    }
    operation: JsonObject = {"action": "native-workbook", "native_workbook": request}
    code, output = run(built, path, operation, "--dry-run")
    assert code == 0 and not capture_file.exists(), output
    assert not built.artifacts.list_ids(artifact_type="authored-task")
    code, output = run(built, path, operation)
    assert code == 0, output
    prepared = json.loads(output)
    assert json.loads(capture_file.read_text()) == prepared["capture"]
    capture = CaptureDefinition.model_validate(prepared["capture"]["capture"])
    published = ArtifactInput.model_validate(prepared["input_artifact"])
    assert built.artifacts.read_bytes(published.artifact_id, "input.xlsx") == starting
    assert capture.task.lineage_group_id == task.lineage_group_id
    assert {tool.name for tool in capture.task.tools} == {
        "edit_workbook",
        "inspect_workbook",
        "describe_workbook",
    }
    assert capture.runtime_sha256 == sha256_bytes(capture.runtime_file.read_bytes())
    assert prepared["initial_projection"]["views"]["answer0"]["source"] is None
    verifier = TaskVerifier.model_validate(prepared["verifier"])
    assert "reference" not in json.dumps(capture.model_dump(mode="json"))

    def outcome(start: int) -> str:
        """The verifier's goal status after deleting one row at ``start``."""
        with (
            load_runtime_definition(
                built,
                runtime_file=capture.runtime_file,
                runtime_sha256=capture.runtime_sha256,
                configuration=capture.configuration,
                probe_sha256=capture.probe_sha256,
            ) as runtime,
            runtime.world.open(capture.task) as world,
        ):
            initial = runtime.probe(world.checkpoint())
            action = ToolCall(
                call_id="edit",
                name="edit_workbook",
                arguments={
                    "source": "input.xlsx",
                    "destination": "output.xlsx",
                    "deletions": [{"sheet": "Orders", "start": start, "count": 1}],
                },
            )
            reply = world.execute(action)
            assert not reply.is_error, reply.content
            final = runtime.probe(world.checkpoint())
        evidence = WorkflowEvidence(
            workflow_id=f"delete-{start}",
            task=capture.task.instruction,
            initial_state=initial,
            steps=(
                WorkflowStep(
                    step=0,
                    action=action.model_dump(mode="json"),
                    status="valid",
                    reply=reply.content,
                    after=final,
                ),
            ),
            final_answer="done",
            stop_reason="completed",
        )
        return grade(verifier, evidence).goal.status

    assert outcome(2) == "pass"
    assert outcome(3) == "fail"
    code, output = run(built, path, {**operation, "native_workbook": {**request}})
    assert code != 0 and "operation_file needs a new file" in output, output
    replay = {key: value for key, value in request.items() if key != "operation_file"}
    code, output = run(built, path, {"action": "native-workbook", "native_workbook": replay})
    assert code == 0 and json.loads(output)["input_artifact"] == prepared["input_artifact"]
    code, output = run(built, path, prepared["capture"], "--yes")
    assert code == 0, output


def test_cli_registers_a_classified_rubric_trajectory(built: ProjectStore, tmp_path: Path) -> None:
    """The registered chronology passes the development-source check and replays."""
    path = tmp_path / "operation.json"
    task = TaskCase(
        task_id="rubric-development",
        lineage_group_id="rubric-development-family",
        partition="fit",
        workload_weight=1.0,
        origin="authored",
        instruction="Look up the order and report its status",
    )
    trajectory = WorkflowEvidence(
        workflow_id="rubric-development",
        task=task.instruction,
        initial_state={},
        steps=(
            WorkflowStep(
                step=0,
                action={"name": "get_order_details", "arguments": {"order_id": "#W1"}},
                status="valid",
                reply="order #W1 is pending",
            ),
        ),
        final_answer="Order #W1 is pending",
        stop_reason="completed",
    )
    registration: JsonObject = {
        "trajectory": {
            "task": task.model_dump(mode="json"),
            "trajectory": trajectory.model_dump(mode="json"),
        },
        "rationale": "Authored development trajectory outside the evaluated family.",
    }
    operation: JsonObject = {
        "action": "register-rubric-trajectory",
        "rubric_trajectory": registration,
    }
    code, output = run(built, path, operation, "--dry-run")
    assert code == 0 and "not registered" in output, output
    assert not built.artifacts.list_ids(artifact_type="actor-rubric-trajectory")
    code, output = run(built, path, operation)
    assert code == 0, output
    printed = json.loads(output)
    source = ArtifactInput.model_validate(printed["trajectory"])
    assert source == artifact_input(built.artifacts.read(source.artifact_id).manifest)
    require_development_sources(built.artifacts, (source,), maximum_artifacts=16)
    code, output = run(built, path, operation)
    assert code == 0 and json.loads(output) == printed, output
    mismatch = {**registration, "trajectory": {**registration["trajectory"], "task": {
        **task.model_dump(mode="json"), "instruction": "Something else"}}}  # fmt: skip
    code, output = run(
        built, path, {"action": "register-rubric-trajectory", "rubric_trajectory": mismatch}
    )
    assert code != 0 and "instruction" in output, output


def test_cli_prints_review_contents_and_refuses_evidence(
    built: ProjectStore, tmp_path: Path
) -> None:
    """A frozen review artifact's JSON is printed; evaluation evidence is refused."""
    path = tmp_path / "operation.json"
    built.artifacts.write_json(
        artifact_id="actor-rubric-development-fixture-draft",
        artifact_type="actor-rubric-draft",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="fixture"
        ),
        files={"draft.json": {"criteria": ["goal"]}},
    )
    query: JsonObject = {
        "artifact_ids": ["actor-rubric-development-fixture-draft"],
        "contents": True,
    }
    code, output = run(built, path, {"action": "artifact-reference", "references": query})
    assert code == 0, output
    (row,) = json.loads(output)["artifacts"]
    assert row["contents"] == {"draft.json": {"criteria": ["goal"]}}
    build = built.load_project().build
    assert build is not None
    query = {"artifact_ids": [build.task_set.artifact_id], "contents": True}
    code, output = run(built, path, {"action": "artifact-reference", "references": query})
    assert code != 0 and "only for review artifacts" in output, output

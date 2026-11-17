"""The build's development tasks and a grounded simulator come out of the CLI as ready JSON."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.development import DevelopmentTaskQuery
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase, ToolSchema, load_task_set
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.learning.lineage import plan_world_model
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest

TOOL = ToolSchema(
    name="get_order_details",
    description="Read one order.",
    input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
)


def run(project: ProjectStore, path: Path, operation: JsonObject, *flags: str) -> tuple[int, str]:
    """Run one ``--continuations`` operation and return its exit code and output."""
    path.write_text(json.dumps(operation))
    result = CliRunner().invoke(
        app,
        [
            "eval",
            project.paths.project_id,
            "--root",
            str(project.paths.root),
            "--continuations",
            str(path),
            "--non-interactive",
            *flags,
        ],
    )
    return result.exit_code, result.output


def test_tools_replace_one_named_task_only() -> None:
    """Replacement tools without a task would silently rewrite every task; they are refused."""
    with pytest.raises(ValueError, match="name it with task_id"):
        DevelopmentTaskQuery(tools=(TOOL,))


def test_cli_lists_development_tasks_and_grounds_a_simulated_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only fit tasks are printed; a grounded capture is saved only outside dry runs."""
    episodes = [episode(index) for index in range(USERS)]
    _, project, _ = build_project(tmp_path, monkeypatch, SimulatedProvider(episodes))
    build = project.load_project().build
    assert build is not None
    path = tmp_path / "operation.json"

    code, output = run(project, path, {"action": "development-tasks", "development_tasks": {}})
    assert code == 0, output
    listed = json.loads(output)
    assert listed["build"]["trace_dataset"] == build.trace_dataset.model_dump(mode="json")
    tasks = [TaskCase.model_validate(row) for row in listed["tasks"]]
    assert tasks and {task.partition for task in tasks} == {"fit"}
    held_out = next(
        task
        for task in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    )
    query: JsonObject = {"task_id": held_out.task_id}
    code, output = run(project, path, {"action": "development-tasks", "development_tasks": query})
    assert code != 0 and "is held out" in output, output
    query = {"task_id": tasks[0].task_id, "tools": [TOOL.model_dump(mode="json")]}
    code, output = run(project, path, {"action": "development-tasks", "development_tasks": query})
    assert code == 0, output
    (chosen,) = json.loads(output)["tasks"]
    assert TaskCase.model_validate(chosen) == tasks[0].model_copy(update={"tools": (TOOL,)})

    capture_file = tmp_path / "capture.json"
    grounding: JsonObject = {
        "task_id": tasks[0].task_id,
        "tools": [TOOL.model_dump(mode="json")],
        "manifest": WorldManifest(
            domains=(StateDomain(name="orders", owner="simulated"),),
            tools=(ToolEffectPolicy(name="get_order_details", reads=("orders",)),),
        ).model_dump(mode="json"),
        "initial_state": {"orders": {"status": "pending"}},
        "probe_paths": {"status": "/orders/status"},
        "memory": "none",
        "maximum_physical_attempts": 4,
        "deadline": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "operation_file": str(capture_file),
    }
    operation: JsonObject = {"action": "ground-runtime", "grounded": grounding}
    code, output = run(project, path, operation, "--dry-run")
    assert code == 0 and not capture_file.exists(), output
    code, output = run(project, path, operation)
    assert code == 0, output
    saved = json.loads(capture_file.read_text())
    assert saved == json.loads(output) and saved["action"] == "capture"
    capture = CaptureDefinition.model_validate(saved["capture"])
    # A development world retrieves from the fit-only index, never the serving index.
    fit_index = plan_world_model(project.artifacts, build, "fit")
    assert fit_index != build.world_model
    assert capture.sources == (fit_index,) and capture.task.tools == (TOOL,)
    code, output = run(project, path, operation)
    assert code != 0 and "operation_file needs a new file" in output, output

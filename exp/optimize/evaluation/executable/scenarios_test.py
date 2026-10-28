"""Reachable seeds enter the existing actor path without exposing setup or private state."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactInput, sha256_bytes
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.executable.scenarios import (
    RecordScenarioSpec,
    RecordWitnessSpec,
    prepare_record_scenario,
    prepare_record_witness,
)
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    LabelWitness,
    persist_specification,
)
from exp.simulation.evaluation.record_execution import load_record_execution
from exp.simulation.evaluation.record_execution_test import NOW, acceptance, fixture, source

RUNTIME = '''"""Explicit test-only native factory; no credentials or model clients."""
from functools import partial
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.environments.record_world import RecordWorldFactory, RecordTool
from exp.runtime.environments.world_contracts import ToolEffectPolicy
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.evaluation.record_execution import record_state
from exp.simulation.evaluation.record_execution_test import fixture, SCHEMA, access, change

def continuation_runtime(project, configuration):
    """Bind setup and actor tools to the same isolated native state owner."""
    replay, _ = fixture()
    factory = RecordWorldFactory(service={"name": "authored-retail-fixture"}, schema=SCHEMA,
        tools={"change": RecordTool(access, change), "setup": RecordTool(access, change)})
    manifest = replay.world.manifest.model_copy(update={"tools": (*replay.world.manifest.tools,
        ToolEffectPolicy(name="setup", writes=("orders", "outbox")))})
    world = ExecutableWorldRuntime(manifest, factory, seed=7)
    return ContinuationRuntime(world, partial(record_state, world), replay.probe_sha256)
'''


def scenario_fixture(tmp_path: Path) -> tuple[ProjectStore, RecordScenarioSpec]:
    """Freeze an explicitly selected native runtime and a setup-only tool capability."""
    project = ProjectStore(tmp_path, "records")
    replay, task = fixture()
    runtime = tmp_path / "native_runtime.py"
    runtime.write_text(RUNTIME)
    definition = CaptureDefinition(
        task=task,
        sources=(source(project),),
        runtime_file=runtime,
        runtime_sha256=sha256_bytes(runtime.read_bytes()),
        probe_sha256=replay.probe_sha256,
        maximum_model_calls=5,
        system_prompt="Use only the actor's public tools.",
    )
    spec = RecordScenarioSpec(
        capture=definition,
        setup_tools=(
            ToolSchema(
                name="setup", description="Operator-only setup.", input_schema={"type": "object"}
            ),
        ),
        setup_actions=(ToolCall(call_id="prepare", name="setup", arguments={"address": "seed"}),),
        finish_tick=1,
        public_fields={"current_address": "/tables/orders/rows/0/1"},
    )
    return project, spec


def test_scenario_reaches_state_with_setup_then_publishes_only_actor_context(
    tmp_path: Path,
) -> None:
    """The paired seed retains native state and provenance without consuming actor calls."""
    project, spec = scenario_fixture(tmp_path)
    result = prepare_record_scenario(project, spec, code_revision="test")
    checkpoint = load_continuation_checkpoint(project.artifacts, result.checkpoint)
    assert isinstance(checkpoint.world, ExecutableCheckpoint)
    assert checkpoint.world.clock == 1 and checkpoint.world.steps == 1
    assert checkpoint.task.tools == spec.capture.task.tools
    assert checkpoint.task.lineage_group_id == spec.capture.task.lineage_group_id
    assert checkpoint.task.initial_context == {"current_address": "seed"}
    assert checkpoint.actor.calls == 0
    assert "outbox" not in checkpoint.actor.model_dump_json()
    assert "Operator-only" not in checkpoint.actor.model_dump_json()
    receipt = load_record_execution(project.artifacts, result.execution)
    assert receipt.steps[0].action.name == "setup"
    spec.capture.runtime_file.unlink()
    assert prepare_record_scenario(project, spec, code_revision="test") == result


def test_scenario_missing_public_fact_and_failed_setup_publish_no_seed(tmp_path: Path) -> None:
    """Unknown initialization cannot be invented; an observed setup error cannot be ignored."""
    project, spec = scenario_fixture(tmp_path)
    for changed in (
        spec.model_copy(update={"public_fields": {"address": "/missing"}}),
        spec.model_copy(
            update={
                "setup_actions": (
                    ToolCall(call_id="bad", name="setup", arguments={"id": "missing"}),
                )
            }
        ),
    ):
        with pytest.raises(ValueError, match="unavailable|tool error"):
            prepare_record_scenario(project, changed, code_revision="test")
        with pytest.raises(ValueError, match="unfinished startup"):
            prepare_record_scenario(project, changed, code_revision="test")
    assert not project.artifacts.list_ids(artifact_type="continuation-checkpoint")


def test_scenario_capabilities_and_public_projection_are_explicit(tmp_path: Path) -> None:
    """No root projection, context overwrite or undeclared setup action slips through parsing."""
    _, spec = scenario_fixture(tmp_path)
    for update in (
        {"public_fields": {"all": ""}},
        {"setup_tools": ()},
        {"setup_tools": spec.capture.task.tools},
    ):
        with pytest.raises(ValueError):
            RecordScenarioSpec.model_validate({**spec.model_dump(mode="python"), **update})


def test_native_witness_from_seed_is_separate_from_setup_and_final_prose(tmp_path: Path) -> None:
    """The exact seed is restored for labeling; the witness is independent of the setup script."""
    project, spec = scenario_fixture(tmp_path)
    result = prepare_record_scenario(project, spec, code_revision="test")
    checkpoint = load_continuation_checkpoint(project.artifacts, result.checkpoint)
    acceptance_ref = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Declared requested address and no collateral effects.",
            predicate=acceptance(),
            subject="native_observation",
        ),
        sources=spec.capture.sources,
        created_at=NOW,
        code_revision="test",
    )
    witness = prepare_record_witness(
        project,
        RecordWitnessSpec(
            capture=spec.capture.model_copy(update={"task": checkpoint.task}),
            checkpoint=result.checkpoint,
            actions=(ToolCall(call_id="answer", name="change", arguments={}),),
            finish_tick=2,
            specification=acceptance_ref,
        ),
        code_revision="test",
    )
    ref = ArtifactInput.model_validate(witness["witness"])
    evidence = LabelWitness.model_validate_json(
        project.artifacts.read_bytes(ref.artifact_id, "witness.json")
    )
    assert evidence.verdict == "pass"
    assert evidence.observation["final_answer"] == ""


def test_cli_prepares_reachable_native_seed_and_replays_without_factory(tmp_path: Path) -> None:
    """Drive the actual public CLI JSON handoff, including a provider-free dry run and replay."""
    project, spec = scenario_fixture(tmp_path)
    path = tmp_path / "scenario.json"
    path.write_text(ContinuationCommand(action="prepare-scenario", scenario=spec).model_dump_json())
    command = [
        "eval",
        "records",
        "--root",
        str(tmp_path),
        "--continuations",
        str(path),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="continuation-checkpoint")
    run = runner.invoke(app, command)
    assert run.exit_code == 0, run.output
    ref = ArtifactInput.model_validate(json.loads(run.output)["checkpoint"])
    assert load_continuation_checkpoint(project.artifacts, ref).task.initial_context == {
        "current_address": "seed"
    }
    spec.capture.runtime_file.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0 and replay.output == run.output

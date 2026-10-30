"""Configured native worlds detect real collateral edits without exposing worksheet contents."""

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.executable.workbook import (
    PRIVATE_VIEWS_PROTOCOL,
    WorkbookRuntimeDefinition,
    WorkbookStateView,
    continuation_runtime,
)
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, grade


def workbook_definition(tmp_path: Path) -> tuple[ProjectStore, WorkbookRuntimeDefinition, TaskCase]:
    """Bind one immutable two-sheet workbook through the shipped configuration contract."""
    project = ProjectStore(tmp_path, "case")
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    sheet.append(["header"])
    sheet.append([1])
    sheet.append([2])
    protected = workbook.create_sheet("Protected")
    protected.append(["private sentinel"])
    protected.append([3])
    buffer = io.BytesIO()
    workbook.save(buffer)
    manifest = project.artifacts.write(
        artifact_id="workbook-input",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"original.xlsx": buffer.getvalue()},
    )
    spec = WorkbookRuntimeDefinition(
        input_artifact=artifact_input(manifest),
        files={"input.xlsx": "original.xlsx"},
        output_paths=("output.xlsx",),
        manifest=WorldManifest(
            domains=tuple(
                StateDomain(name=name, owner="native") for name in ("input.xlsx", "output.xlsx")
            ),
            tools=(
                ToolEffectPolicy(
                    name="edit_workbook", reads=("input.xlsx",), writes=("output.xlsx",)
                ),
            ),
        ),
        views={
            "protected": WorkbookStateView(
                input_file="input.xlsx",
                output_file="output.xlsx",
                sheets=("Protected",),
                baseline_until_written=True,
            )
        },
    )
    task = TaskCase(
        task_id="workbook",
        lineage_group_id="authored-workbook",
        origin="authored",
        partition="fit",
        workload_weight=1,
        instruction="Delete the first row of Data, preserving Protected.",
        tools=(
            ToolSchema(
                name="edit_workbook",
                description="Native row deletion.",
                input_schema={"type": "object"},
            ),
        ),
    )
    return project, spec, task


def test_constructed_assets_enter_existing_runtime_and_restore(tmp_path: Path) -> None:
    """Construction outputs feed the maintained state machine without a special adapter."""
    from exp.runtime.environments.world_contracts import ExecutableCheckpoint
    from exp.simulation.world_construction import (
        ConstructionSpec,
        SequenceColumn,
        TablePlan,
        ValuesColumn,
        persist_construction,
    )

    project, definition, task = workbook_definition(tmp_path)
    spec = ConstructionSpec(
        name="interleaved",
        assumptions=("Authored counterexample.",),
        tables={
            "Data": TablePlan(
                rows=3,
                columns={
                    "id": SequenceColumn(kind="sequence"),
                    "key": ValuesColumn(kind="values", values=(1, 2, 1)),
                },
            ),
            "Protected": TablePlan(
                rows=1, columns={"sentinel": ValuesColumn(kind="values", values=(42,))}
            ),
        },
        workbooks={"original.xlsx": ("Data", "Protected")},
    )
    reference = persist_construction(
        project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    definition = definition.model_copy(
        update={"input_artifact": reference, "files": {"input.xlsx": "assets/original.xlsx"}}
    )
    runtime = continuation_runtime(project, definition.model_dump(mode="json"))
    assert isinstance(runtime.world, ExecutableWorldRuntime)
    with runtime.world.open(task) as world:
        initial = world.checkpoint()
        before = runtime.probe(initial)
        reply = world.execute(
            ToolCall(
                name="edit_workbook",
                call_id="delete",
                arguments={
                    "source": "input.xlsx",
                    "destination": "output.xlsx",
                    "deletions": [{"sheet": "Data", "start": 1, "count": 1}],
                },
            )
        )
        assert not reply.is_error
        point = world.checkpoint()
        assert isinstance(point, ExecutableCheckpoint)
        after = runtime.probe(point)
        before_views, after_views = before["views"], after["views"]
        assert isinstance(before_views, dict) and isinstance(after_views, dict)
        before_protected, after_protected = before_views["protected"], after_views["protected"]
        assert isinstance(before_protected, dict) and isinstance(after_protected, dict)
        assert before_protected["sha256"] == after_protected["sha256"]
        assert point.image.parts["input.xlsx"] == initial.image.parts["input.xlsx"]
        book = load_workbook(io.BytesIO(point.image.parts["output.xlsx"]))
        assert list(book["Data"].values) == [(2, 2), (3, 1)]
        book.close()
    with runtime.world.restore(task, initial) as restored:
        assert runtime.probe(restored.checkpoint()) == before


@pytest.mark.parametrize("collateral", [False, True])
def test_private_views_catch_real_collateral_edits_without_actor_disclosure(
    tmp_path: Path,
    collateral: bool,
) -> None:
    """A real extra deletion changes the private digest despite a valid main-sheet result."""
    project, spec, task = workbook_definition(tmp_path)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    assert isinstance(runtime.world, ExecutableWorldRuntime)
    with runtime.world.open(task) as world:
        initial = world.checkpoint()
        before = runtime.probe(initial)
        deletions = [{"sheet": "Data", "start": 1, "count": 1}]
        if collateral:
            deletions.append({"sheet": "Protected", "start": 1, "count": 1})
        action = ToolCall(
            name="edit_workbook",
            call_id="delete",
            arguments={
                "source": "input.xlsx",
                "destination": "output.xlsx",
                "deletions": deletions,
            },
        )
        reply = world.execute(action)
        after = runtime.probe(world.checkpoint())
        assert not reply.is_error and "private sentinel" not in reply.content
        assert "private sentinel" not in str(after)
        outcome: JsonObject = {
            "initial_state": before,
            "final_state": after,
            "final_answer": "done",
            "stop_reason": "completed",
            "steps": [
                {"action": action.model_dump(mode="json"), "after": after, "status": "valid"}
            ],
        }
        predicate = TaskVerifier(
            description="The native output exists and protected observations stay unchanged.",
            outcome_schema={},
            allowed_tools=("edit_workbook",),
            rules=(OutcomeRule(path="/final_state/files/output.xlsx", operator="exists"),),
            preserved_paths=("/files/input.xlsx", "/views/protected/sha256"),
        )
        assessment = grade(predicate, outcome)
        assert assessment.goal.status == "pass"
        assert assessment.policy.status == ("fail" if collateral else "pass")
    with runtime.world.restore(task, initial) as restored:
        assert runtime.probe(restored.checkpoint()) == before


def test_missing_protected_sheet_cannot_become_an_empty_successful_digest(tmp_path: Path) -> None:
    """Unavailable worksheet evidence remains absent instead of comparing equal empty hashes."""
    project, spec, task = workbook_definition(tmp_path)
    spec = spec.model_copy(
        update={
            "views": {"missing": WorkbookStateView(input_file="input.xlsx", sheets=("Missing",))}
        }
    )
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(task) as world:
        state = runtime.probe(world.checkpoint())
    assert state["views"] == {"missing": {"source": "input.xlsx", "missing_sheets": ["Missing"]}}
    assessment = grade(
        TaskVerifier(
            description="Protect requested sheet.",
            outcome_schema={},
            preserved_paths=("/views/missing/sha256",),
        ),
        {"initial_state": state, "final_state": state, "steps": [], "final_answer": "done"},
    )
    assert assessment.policy.status == "unassessable"


def test_input_and_probe_identity_drift_refuse(tmp_path: Path) -> None:
    """Changed source references cannot initialize a runtime or masquerade as its checkpoint."""
    project, spec, task = workbook_definition(tmp_path)
    bad = spec.model_copy(
        update={"input_artifact": spec.input_artifact.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(ValueError, match="input artifact changed"):
        continuation_runtime(project, bad.model_dump(mode="json"))
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(task) as world:
        point = world.checkpoint()
    with pytest.raises(ValueError, match="exact native backend"):
        runtime.probe(point.model_copy(update={"backend_sha256": "0" * 64}))
    changed = spec.model_copy(update={"views": {}})
    assert changed.probe_sha256 != spec.probe_sha256


@pytest.mark.parametrize(
    "changes",
    [
        {"files": {"../input.xlsx": "original.xlsx"}},
        {"files": {"input.xlsx": "../original.xlsx"}},
        {"output_paths": ["input.xlsx"]},
        {"output_paths": ["output.xlsx", "output.xlsx"]},
        {"views": {"bad": {"input_file": "unregistered.xlsx"}}},
        {"views": {"bad": {"input_file": "input.xlsx", "sheets": []}}},
    ],
)
def test_native_configuration_refuses_unbound_namespace(
    tmp_path: Path, changes: JsonObject
) -> None:
    """Invalid views, outputs and source paths fail before runtime initialization."""
    _, spec, _ = workbook_definition(tmp_path)
    with pytest.raises(ValueError):
        WorkbookRuntimeDefinition.model_validate({**spec.model_dump(mode="json"), **changes})


def test_private_row_projection_is_opt_in_and_does_not_change_actor_observations(
    tmp_path: Path,
) -> None:
    """A private semantic probe adds rows only to evaluator state, with a different probe pin."""
    from exp.simulation.reconstruction.models import resolve

    project, definition, task = workbook_definition(tmp_path)
    rich = definition.model_copy(
        update={
            "views": {
                "main": WorkbookStateView(
                    input_file="input.xlsx",
                    output_file="output.xlsx",
                    sheets=("Data",),
                    include_rows=True,
                )
            }
        }
    )
    compact = rich.model_copy(
        update={"views": {"main": rich.views["main"].model_copy(update={"include_rows": False})}}
    )
    assert rich.probe_sha256 != compact.probe_sha256
    replies = []
    for spec in (compact, rich):
        runtime = continuation_runtime(project, spec.model_dump(mode="json"))
        with runtime.world.open(task) as world:
            before = runtime.probe(world.checkpoint())
            assert resolve(before, "/views/main/source") is None
            with pytest.raises(ValueError):
                resolve(before, "/views/main/sheets")
            reply = world.execute(
                ToolCall(
                    name="edit_workbook",
                    call_id="edit",
                    arguments={
                        "source": "input.xlsx",
                        "destination": "output.xlsx",
                        "deletions": [{"sheet": "Data", "start": 1, "count": 1}],
                    },
                )
            )
            replies.append(reply)
            if spec == rich:
                after = runtime.probe(world.checkpoint())
                assert resolve(after, "/views/main/source") == "output.xlsx"
                assert resolve(after, "/views/main/sheets/0/rows/0/0/value") == 1
    assert replies[0] == replies[1]


def test_an_unwritten_output_is_unavailable_rather_than_the_unchanged_input(
    tmp_path: Path,
) -> None:
    """A goal over the output cannot pass on input content the actor never transformed."""
    project, definition, task = workbook_definition(tmp_path)
    spec = definition.model_copy(
        update={
            "views": {
                **definition.views,
                "main": WorkbookStateView(
                    input_file="input.xlsx", output_file="output.xlsx", sheets=("Protected",)
                ),
            }
        }
    )
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(task) as world:
        state = runtime.probe(world.checkpoint())
    views = state["views"]
    assert isinstance(views, dict)
    assert views["main"] == {"source": None, "unavailable": "registered file is absent"}
    baseline = views["protected"]
    assert isinstance(baseline, dict) and baseline["source"] == "input.xlsx"
    # The unchanged input would satisfy this rule; the absent output must not.
    predicate = TaskVerifier(
        description="Protected content is carried into the output.",
        outcome_schema={},
        required_paths=("/final_state/views/main/sha256",),
        rules=(
            OutcomeRule(
                path="/final_state/views/main/sha256", operator="equals", value=baseline["sha256"]
            ),
        ),
    )
    outcome: JsonObject = {
        "initial_state": state,
        "final_state": state,
        "final_answer": "done",
        "stop_reason": "completed",
        "steps": [],
    }
    assert grade(predicate, outcome).goal.status == "unassessable"
    with pytest.raises(ValueError, match="output_file"):
        WorkbookStateView(input_file="input.xlsx", baseline_until_written=True)


def test_probe_identity_binds_the_view_protocol_not_this_source_file(tmp_path: Path) -> None:
    """Reformatting the adapter module cannot re-key every saved task profile."""
    _, spec, _ = workbook_definition(tmp_path)
    assert spec.probe_sha256 == sha256_json(
        {
            "protocol": PRIVATE_VIEWS_PROTOCOL,
            "definition": spec.model_dump(mode="json"),
            "adapter": WorkbookEditAdapter().definition,
        }
    )

"""Native world construction must separate a target mistake before publishing a task."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import Timeout
from pydantic import BaseModel, JsonValue
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.scenarios import prepare_record_scenario
from exp.optimize.evaluation.executable.scenarios_test import scenario_fixture
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.optimize.evaluation.factory.scenario_search import (
    ScenarioCandidate,
    ScenarioCell,
    ScenarioControl,
    ScenarioSearchSpec,
    run_scenario_search,
)
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.simulation.capability_briefs import persist_capability_brief
from exp.simulation.capability_briefs_test import brief
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.admission import EvaluationSpecification, persist_specification
from exp.simulation.evaluation.record_execution_test import acceptance, source
from exp.simulation.evaluation.tasks import OutcomeRule


def search_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ScenarioSearchSpec]:
    """Construct real record worlds with a no-op counterexample and independently specified goal."""
    project, scenario = scenario_fixture(tmp_path)
    _, template, _ = job_fixture(tmp_path / "template", monkeypatch)
    reached = prepare_record_scenario(project, scenario, code_revision="test")
    task = load_continuation_checkpoint(project.artifacts, reached.checkpoint).task
    job = template.model_copy(
        update={
            "runtime_file": scenario.capture.runtime_file,
            "runtime_sha256": scenario.capture.runtime_sha256,
            "configuration": {},
            "maximum_model_calls": scenario.capture.maximum_model_calls,
            "system_prompt": scenario.capture.system_prompt,
            "study": template.study.model_copy(
                update={
                    "checkpoint": reached.checkpoint,
                    "probe_sha256": scenario.capture.probe_sha256,
                    "verifier": acceptance(),
                    "training_specification": None,
                }
            ),
        }
    )
    now = datetime.now(UTC)
    base = persist_executable_task(
        project,
        ExecutableTask(intent=task.instruction, job=job),
        created_at=now,
        code_revision="test",
    )
    capability = persist_capability_brief(
        project.artifacts,
        brief().model_copy(
            update={
                "outcome": task.instruction,
                "required_tools": ("change",),
            }
        ),
        created_at=now,
        code_revision="test",
    )
    goal: JsonObject = {"const": "new"}
    for part in reversed(("final_state", "tables", "orders", "rows")):
        if part == "rows":
            goal = {"prefixItems": [{"prefixItems": [{}, goal]}]}
        goal = {"properties": {part: goal}, "required": [part]}
    independent = acceptance().model_copy(
        update={
            "rules": (
                OutcomeRule(
                    path="/final_state/tables/orders/rows",
                    operator="equals",
                    value=[["a", "new", "pending"], ["b", "other", "pending"]],
                ),
            ),
            "outcome_schema": goal,
        }
    )
    authority = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Independent requested-address and preservation requirements.",
            predicate=independent,
            subject="native_observation",
        ),
        sources=(source(project),),
        created_at=now,
        code_revision="test",
    )
    cases = []
    programs: tuple[tuple[str, tuple[JsonObject, ...]], ...] = (
        ("solution", ({},)),
        ("alternative", ({"address": "new"}, {"noop": True})),
        ("recovery", ({"id": "absent"}, {})),
        ("targeted_error", ({"noop": True},)),
        ("unwanted_effect", ({"email": True},)),
    )
    for role, arguments in programs:
        cases.append(
            ScenarioControl.model_validate(
                {
                    "role": role,
                    "actions": tuple(
                        ToolCall(name="change", call_id=f"step-{i}", arguments=args)
                        for i, args in enumerate(arguments)
                    ),
                    "finish_tick": 1 + len(arguments),
                }
            )
        )
    candidates = tuple(
        ScenarioCandidate(
            scenario=scenario.model_copy(
                update={
                    "capture": scenario.capture.model_copy(
                        update={
                            "sources": (*scenario.capture.sources, reached.checkpoint),
                        }
                    ),
                    "setup_actions": (
                        ToolCall(name="setup", call_id="setup", arguments={"address": value}),
                    ),
                }
            ),
            verifier=acceptance().model_copy(
                update={
                    "rules": (
                        *acceptance().rules,
                        OutcomeRule(
                            path="/final_state/tables/orders/rows",
                            operator="equals",
                            value=[["a", "new", "pending"], ["b", "other", "pending"]],
                        ),
                    )
                }
            ),
            acceptance=authority,
            controls=tuple(cases),
        )
        for value in ("new", "old")
    )
    return project, ScenarioSearchSpec(
        base=base,
        brief=capability,
        prompt=task.instruction,
        intent=task.instruction,
        candidates=candidates,
    )


def test_native_search_rejects_nondiscriminating_world_and_publishes_reachable_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-op passes in an already-solved world but is rejected in the constructed alternative."""
    project, spec = search_fixture(tmp_path, monkeypatch)
    result = run_scenario_search(project, spec, code_revision="test")
    assert len(result.cells) == 2
    first, second = result.cells
    assert first.variant is None and any("targeted_error" in row for row in first.refusals)
    assert second.variant is not None, second.refusals
    assert len(first.witnesses) == len(second.witnesses) == 5
    variant = load_task_variant(project, second.variant)
    original = load_executable_task(project, spec.base)
    child = load_executable_task(project, variant.task)
    parent = load_continuation_checkpoint(project.artifacts, original.job.study.checkpoint)
    point = load_continuation_checkpoint(project.artifacts, child.job.study.checkpoint)
    assert point.task.lineage_group_id == parent.task.lineage_group_id
    assert point.world != parent.world
    assert point.task.initial_context["current_address"] == "old"
    assert child.job.study.training_specification is None
    assert child.job.study.critic_prediction is None
    assert "setup" not in {tool.name for tool in point.task.tools}
    with open_executable_harbor_trial(
        project, variant.task, trial_id="constructed-records", code_revision="test"
    ) as trial:
        trial.call(spec.candidates[1].controls[0].actions[0])
        trial.submit({"answer": "done"})
        assessed = trial.result()
        mechanical = assessed["mechanical"]
        assert isinstance(mechanical, dict) and mechanical["status"] == "pass"
        assert assessed["reward"] is None and assessed["qualification"] == "diagnostic_only"
    spec.candidates[0].scenario.capture.runtime_file.unlink()
    assert run_scenario_search(project, spec, code_revision="test") == result


def test_native_scenario_search_runs_through_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public command constructs actual new states without an author or actor provider."""
    project, spec = search_fixture(tmp_path, monkeypatch)
    path = tmp_path / "search.json"
    path.write_text(
        ContinuationCommand(action="search-scenarios", scenario_search=spec).model_dump_json()
    )
    args = [
        "eval",
        "records",
        "--root",
        str(tmp_path),
        "--continuations",
        str(path),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="scenario-search-registration")
    executed = runner.invoke(app, args)
    assert executed.exit_code == 0, executed.output
    result = json.loads(executed.output)
    assert result["cells"][0]["variant"] is None
    assert result["cells"][1]["variant"] is not None, result


def test_search_refuses_weak_verifier_even_when_its_native_controls_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing identity/status checks cannot hide behind a few successful control scripts."""
    project, spec = search_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={
            "candidates": tuple(
                row.model_copy(update={"verifier": acceptance()}) for row in spec.candidates
            )
        }
    )
    result = run_scenario_search(project, spec, code_revision="weak-verifier")
    second = result.cells[1]
    assert second.variant is None
    assert len(second.witnesses) == 5
    assert any("synthetic counterexamples" in reason for reason in second.refusals)
    assert any(probe.false_passes for probe in second.verifier_probes.values())


def workbook_search_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, generated: bool = True
) -> tuple[ProjectStore, ScenarioSearchSpec, tuple[ArtifactInput, ...]]:
    """Build two real XLSX prototypes with independent global-last acceptance and controls."""
    import io

    from openpyxl import Workbook

    from exp.common.core.artifacts import ArtifactEnvelope, sha256_bytes, sha256_json
    from exp.common.project import artifact_input
    from exp.common.tasks import ToolSchema
    from exp.optimize.evaluation.continuations.capture import CaptureDefinition
    from exp.optimize.evaluation.executable import workbook as workbook_runtime
    from exp.optimize.evaluation.executable.scenarios import RecordScenarioSpec
    from exp.optimize.evaluation.executable.workbook import WorkbookStateView
    from exp.optimize.evaluation.executable.workbook_test import workbook_definition
    from exp.runtime.environments.workbook_edits import (
        WorkbookDescriptionArguments,
        WorkbookReadArguments,
        workbook_projection,
    )
    from exp.runtime.environments.world_contracts import ToolEffectPolicy
    from exp.simulation.evaluation.table_relations import TableRelation
    from exp.simulation.evaluation.tasks import TaskVerifier
    from exp.simulation.world_construction import (
        ConstructionSpec,
        TablePlan,
        ValuesColumn,
        persist_construction,
    )

    project, definition, task = workbook_definition(tmp_path)
    _, template, _ = job_fixture(tmp_path / "template", monkeypatch)
    now = datetime.now(UTC)
    runtime_file = Path(workbook_runtime.__file__)
    task = task.model_copy(
        update={"instruction": "Keep the global last row for each key in Data; preserve Protected."}
    )
    task = task.model_copy(
        update={
            "tools": (
                *task.tools,
                ToolSchema(
                    name="describe_workbook",
                    description="Describe input.xlsx or output.xlsx.",
                    input_schema=WorkbookDescriptionArguments.model_json_schema(),
                ),
                ToolSchema(
                    name="inspect_workbook",
                    description="Read current workbook cells.",
                    input_schema=WorkbookReadArguments.model_json_schema(),
                ),
            )
        }
    )
    definition = definition.model_copy(
        update={
            "manifest": definition.manifest.model_copy(
                update={
                    "tools": (
                        *definition.manifest.tools,
                        ToolEffectPolicy(
                            name="describe_workbook",
                            reads=("input.xlsx", "output.xlsx"),
                            duration=0,
                        ),
                        ToolEffectPolicy(
                            name="inspect_workbook", reads=("input.xlsx", "output.xlsx"), duration=0
                        ),
                    ),
                }
            )
        }
    )
    # This rule is authored before either generated world exists. Its provenance names the
    # requirement, not an answer file, generated workbook or tested candidate predicate.
    requirement = artifact_input(
        project.artifacts.write_json(
            artifact_id="global-last-requirement",
            artifact_type="authored-requirement",
            envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="test"),
            files={
                "requirement.json": {
                    "instruction": task.instruction,
                    "details": (
                        "Retain entire rows in original order, including the header. "
                        "Preserve the input and Protected."
                    ),
                }
            },
        )
    )
    relation_authority = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Predeclared global-key survivor and preservation contract.",
            subject="native_observation",
            predicate=TaskVerifier(
                description="Keep the last entire row of each global key in original order.",
                outcome_schema={},
                table_relations=(
                    TableRelation(
                        source="/initial_state/views/source/sheets/0/rows",
                        target="/final_state/views/main/sheets/0/rows",
                        operation="last_per_key",
                        key_columns=(1,),
                        key_path="/value",
                        header_rows=1,
                    ),
                ),
                rules=(OutcomeRule(path="/final_state/files/output.xlsx", operator="exists"),),
                allowed_tools=("edit_workbook", "describe_workbook", "inspect_workbook"),
                preserved_paths=("/files/input.xlsx", "/views/protected/sha256"),
            ),
        ),
        sources=(requirement,),
        created_at=now,
        code_revision="test",
    )
    scenarios = []
    predicates = []
    authorities = []
    derived_authorities = []
    controls = []
    for index, keys in enumerate(([1, 1, 2, 2], [1, 2, 1, 1])):
        rows = [[i + 1, key] for i, key in enumerate(keys)]
        keep = [row for i, row in enumerate(rows) if row[1] not in keys[i + 1 :]]
        workbooks = []
        for values in (rows, keep):
            workbook = Workbook()
            sheet = workbook.active
            assert sheet is not None
            sheet.title = "Data"
            sheet.append(["id", "key"])
            for row in values:
                sheet.append(row)
            protected = workbook.create_sheet("Protected")
            protected.append(["do not change"])
            protected.append([42])
            workbooks.append(workbook)
        buffer = io.BytesIO()
        workbooks[0].save(buffer)
        artifact = project.artifacts.write(
            artifact_id=f"constructed-workbook-{index}",
            artifact_type="authored-task",
            envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="test"),
            files={"original.xlsx": buffer.getvalue()},
        )
        input_ref = artifact_input(artifact)
        if generated:
            input_ref = persist_construction(
                project.artifacts,
                ConstructionSpec(
                    name=f"prototype-{index}",
                    assumptions=("Authored distribution; no production frequency claim.",),
                    sources=(requirement,),
                    tables={
                        "Data": TablePlan(
                            rows=5,
                            columns={
                                "id": ValuesColumn(kind="values", values=("id", 1, 2, 3, 4)),
                                "key": ValuesColumn(kind="values", values=("key", *keys)),
                            },
                        ),
                        "Protected": TablePlan(
                            rows=2,
                            columns={
                                "sentinel": ValuesColumn(
                                    kind="values", values=("do not change", 42)
                                )
                            },
                        ),
                    },
                    workbooks={"original.xlsx": ("Data", "Protected")},
                ),
                created_at=now,
                code_revision="test",
            )
        configured = definition.model_copy(
            update={
                "input_artifact": input_ref,
                "files": {"input.xlsx": "assets/original.xlsx" if generated else "original.xlsx"},
                "views": {
                    "main": WorkbookStateView(
                        input_file="input.xlsx",
                        output_file="output.xlsx",
                        sheets=("Data",),
                        include_rows=generated,
                    ),
                    "source": WorkbookStateView(
                        input_file="input.xlsx", sheets=("Data",), include_rows=generated
                    ),
                    "protected": definition.views["protected"],
                },
            }
        )
        capture = CaptureDefinition(
            task=task,
            sources=(input_ref,),
            runtime_file=runtime_file,
            runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
            configuration=configured.model_dump(mode="json"),
            probe_sha256=configured.probe_sha256,
            maximum_model_calls=template.maximum_model_calls,
            system_prompt=template.system_prompt,
        )
        scenarios.append(RecordScenarioSpec(capture=capture, finish_tick=0))
        sheets = workbook_projection(workbooks[1])["sheets"]
        assert isinstance(sheets, list)
        expected = sha256_json([sheets[0]])
        predicate = TaskVerifier(
            description="Global last occurrence, retaining input and Protected.",
            outcome_schema={},
            required_paths=("/final_state/views/main/sha256",),
            rules=(
                OutcomeRule(
                    path="/final_state/views/main/sha256", operator="equals", value=expected
                ),
                *(
                    (
                        OutcomeRule(
                            path="/final_state/views/main/sheets/0/rows",
                            operator="equals",
                            value=sheets[0]["rows"],
                        ),
                    )
                    if generated and isinstance(sheets[0], dict)
                    else ()
                ),
            ),
            allowed_tools=("edit_workbook", "describe_workbook", "inspect_workbook"),
            preserved_paths=("/files/input.xlsx", "/views/protected/sha256"),
        )
        predicates.append(predicate)
        independent = predicate.model_copy(
            update={
                "rules": (OutcomeRule(path="/final_state/files/output.xlsx", operator="exists"),),
                "outcome_schema": {
                    "properties": {
                        "final_state": {
                            "properties": {
                                "views": {
                                    "properties": {
                                        "main": {"properties": {"sha256": {"const": expected}}}
                                    }
                                }
                            }
                        }
                    }
                },
            }
        )
        authorities.append(
            persist_specification(
                project.artifacts,
                EvaluationSpecification(
                    description="Independent last-occurrence output and preservation properties.",
                    predicate=independent,
                    subject="native_observation",
                ),
                sources=(input_ref,),
                created_at=now,
                code_revision="test",
            )
        )
        if generated:
            derived_authorities.append(authorities[-1])
            authorities[-1] = relation_authority
            assert (
                project.artifacts.read(input_ref.artifact_id).manifest.artifact_type
                == "world-construction"
            )
        global_drop = [i + 2 for i, key in enumerate(keys) if key in keys[i + 1 :]][::-1]
        adjacent_drop = [i + 2 for i in range(len(keys) - 1) if keys[i] == keys[i + 1]][::-1]

        def edit(
            call_id: str,
            drops: list[int],
            *,
            source_name: str = "input.xlsx",
            extra: bool = False,
            missing: bool = False,
        ) -> ToolCall:
            """Delete native rows at positions independently computed from the keys."""
            edits: list[JsonObject] = [
                {"sheet": "Missing" if missing else "Data", "start": row, "count": 1}
                for row in drops
            ]
            if extra:
                edits.append({"sheet": "Protected", "start": 2, "count": 1})
            return ToolCall(
                name="edit_workbook",
                call_id=call_id,
                arguments={"source": source_name, "destination": "output.xlsx", "deletions": edits},
            )

        programs = (
            ("solution", (edit("global", global_drop),)),
            (
                "alternative",
                tuple(
                    edit(f"split-{i}", [row], source_name="input.xlsx" if i == 0 else "output.xlsx")
                    for i, row in enumerate(global_drop)
                ),
            ),
            ("recovery", (edit("bad", [2], missing=True), edit("fixed", global_drop))),
            ("targeted_error", (edit("adjacent", adjacent_drop),)),
            ("unwanted_effect", (edit("extra", global_drop, extra=True),)),
        )
        controls.append(
            tuple(
                ScenarioControl.model_validate(
                    {"role": role, "actions": actions, "finish_tick": len(actions)}
                )
                for role, actions in programs
            )
        )
    baseline = prepare_record_scenario(project, scenarios[0], code_revision="test")
    job = template.model_copy(
        update={
            "runtime_file": runtime_file,
            "runtime_sha256": scenarios[0].capture.runtime_sha256,
            "configuration": scenarios[0].capture.configuration,
            "study": template.study.model_copy(
                update={
                    "checkpoint": baseline.checkpoint,
                    "probe_sha256": scenarios[0].capture.probe_sha256,
                    "verifier": predicates[0],
                    "training_specification": None,
                }
            ),
        }
    )
    base = persist_executable_task(
        project,
        ExecutableTask(intent=task.instruction, job=job),
        created_at=now,
        code_revision="test",
    )
    capability = persist_capability_brief(
        project.artifacts,
        brief().model_copy(
            update={"outcome": task.instruction, "required_tools": ("edit_workbook",)}
        ),
        created_at=now,
        code_revision="test",
    )
    candidates = tuple(
        ScenarioCandidate(
            scenario=scenario.model_copy(
                update={
                    "capture": scenario.capture.model_copy(
                        update={"sources": (*scenario.capture.sources, baseline.checkpoint)}
                    )
                }
            ),
            verifier=predicates[i],
            acceptance=authorities[i],
            controls=controls[i],
        )
        for i, scenario in enumerate(scenarios)
    )
    search = ScenarioSearchSpec(
        base=base,
        brief=capability,
        intent=task.instruction,
        prompt=task.instruction,
        candidates=candidates,
    )
    if generated:
        from exp.optimize.evaluation.analysis.research import (
            FailureHypothesis,
            ResearchEvidence,
            ResearchFailure,
        )
        from exp.optimize.evaluation.factory.scenario_search import ScenarioResearchTarget

        search = search.model_copy(
            update={
                "target": ScenarioResearchTarget(
                    failure=ResearchFailure(
                        symptom=(
                            "The constructed grouped case may not distinguish global deletion "
                            "from adjacency. Historical actor strategy is unresolved."
                        ),
                        observations=(
                            ResearchEvidence(
                                source=scenarios[0].capture.sources[0],
                                file="assets/tables.json",
                                pointer="/Data/rows",
                            ),
                        ),
                        hypotheses=(
                            FailureHypothesis(
                                id="adjacency",
                                destination="verifier",
                                explanation="Adjacency deletion may pass grouped-case checks.",
                                falsifier="The grouped case rejects the adjacency-only result.",
                            ),
                        ),
                    ),
                    hypothesis="adjacency",
                )
            }
        )
    return project, search, tuple(derived_authorities)


@pytest.mark.parametrize("generated", [False, True])
def test_workbook_search_constructs_an_executable_nonadjacent_duplicate_counterexample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated: bool
) -> None:
    """The same search rejects a confounded world and validates a different XLSX world."""
    from exp.optimize.evaluation.factory.scenario_search import ScenarioSearchResult

    project, search, derived_authorities = workbook_search_fixture(
        tmp_path, monkeypatch, generated=generated
    )
    base = load_executable_task(project, search.base)
    if generated:
        monkeypatch.setattr(
            "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
        )
        operation = tmp_path / "generated-search.json"
        operation.write_text(
            ContinuationCommand(action="search-scenarios", scenario_search=search).model_dump_json()
        )
        args = [
            "eval",
            project.paths.project_id,
            "--root",
            str(tmp_path),
            "--continuations",
            str(operation),
            "--non-interactive",
        ]
        invoked = CliRunner().invoke(app, args)
        assert invoked.exit_code == 0, invoked.output
        result = ScenarioSearchResult.model_validate_json(invoked.output)
        replayed = CliRunner().invoke(app, args)
        assert replayed.exit_code == 0, replayed.output
        assert ScenarioSearchResult.model_validate_json(replayed.output) == result
    else:
        result = run_scenario_search(project, search, code_revision="test")
    assert result.cells[0].variant is None
    assert any("targeted_error" in row for row in result.cells[0].refusals), result.cells[
        0
    ].refusals
    assert result.cells[1].variant is not None, result.cells[1].refusals
    if generated:
        circular = search.model_copy(
            update={
                "candidates": tuple(
                    candidate.model_copy(update={"acceptance": derived_authorities[i]})
                    for i, candidate in enumerate(search.candidates)
                )
            }
        )
        refused = run_scenario_search(project, circular, code_revision="test")
        assert all(cell.variant is None for cell in refused.cells)
        assert all(
            any("self_approval" in reason for reason in cell.refusals) for cell in refused.cells
        )
    variant = load_task_variant(project, result.cells[1].variant)
    child = load_executable_task(project, variant.task)
    point = load_continuation_checkpoint(project.artifacts, child.job.study.checkpoint)
    assert (
        point.world
        != load_continuation_checkpoint(project.artifacts, base.job.study.checkpoint).world
    )
    assert child.job.configuration == search.candidates[1].scenario.capture.configuration
    with open_executable_harbor_trial(
        project, variant.task, trial_id="constructed-workbook", code_revision="test"
    ) as trial:
        assert (
            trial.call(
                ToolCall(
                    name="describe_workbook", call_id="discover", arguments={"source": "input.xlsx"}
                )
            )["is_error"]
            is False
        )
        observed = trial.call(
            ToolCall(
                name="inspect_workbook",
                call_id="inspect",
                arguments={
                    "source": "input.xlsx",
                    "sheet": "Data",
                    "row_count": 5,
                    "column_count": 2,
                },
            )
        )
        assert observed["is_error"] is False
        for action in search.candidates[1].controls[0].actions:
            trial.call(action)
        trial.submit({"answer": "done"})
        assessed = trial.result()
        mechanical = assessed["mechanical"]
        assert isinstance(mechanical, dict) and mechanical["status"] == "pass"
        assert assessed["reward"] is None


def test_search_can_finalize_saved_cells_without_reopening_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication-only crash preserves purchased native evidence and needs no executable file."""
    project, spec = search_fixture(tmp_path, monkeypatch)
    write = project.artifacts.write_json

    def interrupt(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Interrupt immediately before final publication, after both cells are immutable."""
        if artifact_type == "scenario-search-result":
            raise KeyboardInterrupt("publication interrupted")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_scenario_search(project, spec, code_revision="test")
    assert len(project.artifacts.list_ids(artifact_type="scenario-search-cell")) == 2
    count = len(project.artifacts.list_ids(artifact_type="native-record-execution"))
    spec.candidates[0].scenario.capture.runtime_file.unlink()
    result = run_scenario_search(project, spec, code_revision="test")
    assert result.cells[1].variant is not None
    assert len(project.artifacts.list_ids(artifact_type="native-record-execution")) == count


def test_research_citations_are_checked_before_any_native_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named hypothesis does not authorize fabricated source quotes or missing locators."""
    from exp.optimize.evaluation.analysis.research import (
        FailureHypothesis,
        ResearchEvidence,
        ResearchFailure,
    )
    from exp.optimize.evaluation.factory.scenario_search import ScenarioResearchTarget

    project, spec = search_fixture(tmp_path, monkeypatch)
    target = ScenarioResearchTarget(
        failure=ResearchFailure(
            symptom="Candidate observation.",
            observations=(
                ResearchEvidence(
                    source=spec.base, file="task.json", pointer="/intent", quote="ABSENT-QUOTE"
                ),
            ),
            hypotheses=(
                FailureHypothesis(
                    id="noop",
                    explanation="No-op may pass.",
                    destination="verifier",
                    falsifier="No-op fails.",
                ),
            ),
        ),
        hypothesis="noop",
    )
    targeted = spec.model_copy(update={"target": target})
    before = project.artifacts.list_ids()
    with pytest.raises(ValueError, match="quote absent"):
        run_scenario_search(project, targeted, code_revision="test")
    assert project.artifacts.list_ids() == before
    with pytest.raises(ValueError, match="uniquely named"):
        ScenarioResearchTarget(failure=target.failure, hypothesis="invented")


def test_busy_scenario_child_resumes_without_consuming_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known contention reuses the native construction's original registration."""
    project, spec = search_fixture(tmp_path, monkeypatch)

    def busy(
        project: ProjectStore, spec: ScenarioSearchSpec, index: int, revision: str
    ) -> ScenarioCell:
        """Inject a known lease refusal before the native child can execute."""
        raise Timeout("scenario.lock")

    with monkeypatch.context() as patch:
        patch.setattr("exp.optimize.evaluation.factory.scenario_search._cell", busy)
        with pytest.raises(Timeout):
            run_scenario_search(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-cell-0")
    report = run_scenario_search(project, spec, code_revision="test")
    assert report.cells[0].scenario is not None
    assert run_scenario_search(project, spec, code_revision="test") == report

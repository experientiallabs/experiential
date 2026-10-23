"""Native package execution, source denominators and recovery for the multi-family factory."""

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from pydantic import BaseModel, JsonValue
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactEnvelope, JsonObject
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.scenarios import prepare_record_scenario
from exp.optimize.evaluation.executable.scenarios_test import scenario_fixture
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.optimize.evaluation.factory.candidates_test import (
    _reply,
    batch_fixture,
    fixture_quality_reviews,
)
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    load_task_proposals,
    propose_tasks,
)
from exp.optimize.evaluation.factory.runner import (
    FactoryFamily,
    TaskFactorySpec,
    factory_attempt_bound,
    run_task_factory,
    saved_task_factory,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.capability_briefs import (
    CapabilityAssumption,
    CapabilityBrief,
    RequiredInformation,
    persist_capability_brief,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase
from exp.simulation.evaluation.record_execution_test import acceptance, source
from exp.simulation.evaluation.tasks import TaskVerifier, VerifierChallenge
from exp.simulation.task_distribution import CoverageStratum, TaskDistribution


def factory_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, quota: int = 1
) -> tuple[ProjectStore, TaskFactorySpec, ScriptedJsonTransport]:
    """Use actual native workbook witnesses and existing author/provider parsing paths."""
    project, batch, transport = batch_fixture(tmp_path, monkeypatch)
    proposals = load_task_proposals(project, batch.proposals)
    spec = TaskFactorySpec(
        families=(
            FactoryFamily(
                proposal=proposals.spec,
                controls=batch.controls,
                strata=("workbook",),
                information="available",
                horizon="single_decision",
                quality_reviews=batch.quality_reviews,
            ),
        ),
        actors=batch.actors,
        distribution=TaskDistribution(
            seed=17,
            strata=(
                CoverageStratum(
                    id="workbook",
                    user="workbook owner",
                    capability="bounded row deletion",
                    category="regression",
                    information="available",
                    horizon="single_decision",
                    minimum_families=quota,
                ),
            ),
        ),
        maximum_author_actor_attempts=100_000,
        deadline=proposals.spec.deadline,
        destination=tmp_path / "factory-packages",
    )
    return project, spec, transport


def test_native_packages_are_two_candidates_but_only_one_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual XLSX execution qualifies packages without inflating independent task breadth."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    result = run_task_factory(project, spec, code_revision="test")
    assert result.status == "development_qualified", result.model_dump_json()
    assert result.counts == {
        "seeds": 1,
        "source_workflow_components": 1,
        "proposed_candidates": 2,
        "packaged_candidates": 2,
        "qualified_seed_families": 1,
        "selected_components": 1,
        "operational_failures": 0,
    }
    assert len(transport.requests) == 12
    packages = tuple(spec.destination.glob("*/*/task.toml"))
    assert len(packages) == 2
    for package in packages:
        verify_executable_package(project, package.parent)
    task = load_executable_task(project, spec.families[0].proposal.base)
    task.job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    assert run_task_factory(project, spec, code_revision="test") == result
    assert len(transport.requests) == 12
    (packages[0].parent / "instruction.md").write_text("Do something else.")
    with pytest.raises(ValueError, match="changed after export"):
        run_task_factory(project, spec, code_revision="test")
    assert len(transport.requests) == 12


def test_impossible_coverage_is_not_bought_or_satisfied_by_rephrasings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A two-family target cannot be filled with two variants from one source workflow."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch, quota=2)
    result = run_task_factory(project, spec, code_revision="test")
    assert result.status == "insufficient_evidence"
    assert result.coverage.deficits == {"workbook": 2}
    assert result.cells[0].status == "not_started"
    assert result.counts["packaged_candidates"] == 0
    assert not transport.requests
    assert not spec.destination.exists()


def test_full_population_budget_checked_before_first_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No optimistic accounting of successful prototypes permits an over-budget batch."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={"maximum_author_actor_attempts": factory_attempt_bound(project, spec) - 1}
    )
    with pytest.raises(ValueError, match="allowance"):
        run_task_factory(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test"))


def test_absent_independent_side_effect_controls_block_family_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Good feasibility cannot compensate for a verifier that was never challenged on harm."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    family = spec.families[0]
    spec = spec.model_copy(
        update={
            "families": (
                family.model_copy(
                    update={
                        "controls": tuple(
                            row for row in family.controls if "side-effect" not in row.challenge.id
                        )
                    }
                ),
            )
        }
    )
    result = run_task_factory(project, spec, code_revision="test")
    assert result.status == "insufficient_evidence"
    assert result.cells[0].status == "rejected"
    assert result.counts["packaged_candidates"] == 0
    assert len(transport.requests) == 4


def test_saved_children_finish_after_parent_crash_without_runtime_or_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before the family receipt must not repurchase already committed measurements."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    original = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Inject process death exactly between child completion and parent receipt."""
        if artifact_id == spec.identity("test") + "-family-0":
            raise KeyboardInterrupt
        return original(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", crash)
        with pytest.raises(KeyboardInterrupt):
            run_task_factory(project, spec, code_revision="test")
    assert len(transport.requests) == 12
    task = load_executable_task(project, spec.families[0].proposal.base)
    task.job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    result = run_task_factory(project, spec, code_revision="test")
    assert result.status == "development_qualified"
    assert len(transport.requests) == 12


def record_family(
    project: ProjectStore, spec: TaskFactorySpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FactoryFamily:
    """Reach a distinct real SQLite address-change world and author separate verifier controls."""
    _, scenario = scenario_fixture(tmp_path)
    seed_source = source(project)
    scenario = scenario.model_copy(
        update={"capture": scenario.capture.model_copy(update={"sources": (seed_source,)})}
    )
    reached = prepare_record_scenario(project, scenario, code_revision="test")
    checkpoint = load_continuation_checkpoint(project.artifacts, reached.checkpoint)
    workbook = load_executable_task(project, spec.families[0].proposal.base)
    job = workbook.job.model_copy(
        update={
            "runtime_file": scenario.capture.runtime_file,
            "runtime_sha256": scenario.capture.runtime_sha256,
            "configuration": {},
            "maximum_model_calls": scenario.capture.maximum_model_calls,
            "system_prompt": scenario.capture.system_prompt,
            "study": ContinuationStudySpec(
                checkpoint=reached.checkpoint,
                actor_model=workbook.job.study.actor_model,
                verifier=acceptance(),
                probe_sha256=scenario.capture.probe_sha256,
                repeats=1,
                selector="original native address-change seed",
            ),
        }
    )
    base = ExecutableTask(intent=checkpoint.task.instruction, job=job)
    reference = persist_executable_task(
        project, base, created_at=datetime.now(UTC), code_revision="test"
    )
    brief = CapabilityBrief(
        category="regression",
        novelty="existing_workflow",
        user="order owner",
        outcome=base.intent,
        capability="address change",
        gap="Preservation during address updates",
        required_tools=("change",),
        required_information=(
            RequiredInformation(
                fact="order and requested address", obtain_via="public instruction"
            ),
        ),
        allowed_changes=("order a address",),
        preserved=("other records and outbox",),
        unavailable_behavior="Report unavailable order",
        success_evidence=("native address and outbox",),
        assumptions=(
            CapabilityAssumption(
                statement="Authored pending-order contract",
                authority="authored_contract",
                sources=(seed_source,),
            ),
        ),
    )
    brief_ref = persist_capability_brief(
        project.artifacts, brief, created_at=datetime.now(UTC), code_revision="test"
    )
    proposal = TaskProposalSpec(
        base=reference,
        brief=brief_ref,
        model_alias=job.actor_alias,
        model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )
    author = ScriptedJsonTransport(
        (
            _reply(
                "world-v1",
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "author-records",
                            "type": "function",
                            "function": {
                                "name": "propose_tasks",
                                "arguments": json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "prompt": base.intent,
                                                "rationale": "Explicit requested effect.",
                                            },
                                            {
                                                "prompt": (
                                                    "Set a's address to new; "
                                                    "preserve other records and send no mail."
                                                ),
                                                "rationale": "Preservation first.",
                                            },
                                        ]
                                    }
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: author
    )
    propose_tasks(project, proposal, code_revision="test")
    # A separately authored JSON-schema property checks the address, while the candidate
    # uses a state-path rule. These are authored contract controls, not production labels.
    goal_schema: JsonObject = {
        "properties": {
            "final_state": {
                "properties": {
                    "tables": {
                        "properties": {
                            "orders": {
                                "properties": {
                                    "rows": {
                                        "prefixItems": [{"prefixItems": [{}, {"const": "new"}]}],
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    independent = TaskVerifier(
        description="Independent record acceptance property",
        outcome_schema=goal_schema,
        preserved_paths=("/tables/orders/rows/1", "/tables/outbox"),
        allowed_tools=("change",),
    )
    authority = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Authored address and preservation property",
            predicate=independent,
            subject="outcome",
        ),
        sources=(seed_source,),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    cases = []
    for partition in ("development", "confirmation"):
        for kind in ("valid-one", "valid-two", "goal-miss", "side-effect", "unknown"):
            before: JsonObject = {
                "tables": {
                    "orders": {"rows": [["a", "seed", "pending"], ["b", partition, "pending"]]},
                    "outbox": {"rows": []},
                }
            }
            after = deepcopy(before)
            tables = after["tables"]
            assert isinstance(tables, dict)
            tables["orders"] = {
                "rows": [
                    ["a", "wrong" if kind == "goal-miss" else "new", "pending"],
                    ["b", partition, "pending"],
                ]
            }
            if kind == "side-effect":
                tables["outbox"] = {"rows": [["unwanted mail"]]}
            action = ToolCall(
                call_id="change",
                name="change",
                arguments={
                    "address": "new",
                    **({"id": "a"} if kind == "valid-two" else {}),
                },
            )
            outcome: JsonObject = {
                "initial_state": before,
                "final_state": after,
                "steps": None
                if kind == "unknown"
                else [{"action": action.model_dump(mode="json"), "after": after}],
                "stop_reason": "completed",
            }
            witness = record_label_witness(
                project.artifacts,
                specification=authority,
                outcome=outcome,
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            cases.append(
                CalibrationCase(
                    challenge=VerifierChallenge(
                        id=f"retail-{partition}-{kind}",
                        outcome=outcome,
                        expected="pass"
                        if kind.startswith("valid")
                        else "unassessable"
                        if kind == "unknown"
                        else "fail",
                        authority="synthetic_specification",
                        witness=witness,
                        rationale="Independent authored state/effect property.",
                    ),
                    lineage=f"retail-{partition}-{kind}",
                    partition=partition,
                )
            )
    return FactoryFamily(
        proposal=proposal,
        controls=tuple(cases),
        quality_reviews=fixture_quality_reviews(
            project,
            artifact_input(project.artifacts.read(proposal.identity("test")).manifest),
            (brief_ref, authority),
        ),
        strata=("retail",),
        information="available",
        horizon="single_decision",
    )


def test_factory_qualifies_distinct_workbook_and_record_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One manifest executes two native owners; four variants still count as two families."""
    project, spec, workbook_wire = factory_fixture(tmp_path, monkeypatch)
    family = record_family(project, spec, tmp_path, monkeypatch)
    replies = []
    for _ in range(2):
        for model in ("world-v1", "world-v1", "judge-v1"):
            replies.extend(
                (
                    _reply(
                        model,
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "change",
                                    "type": "function",
                                    "function": {
                                        "name": "change",
                                        "arguments": '{"address":"new"}',
                                    },
                                }
                            ],
                        },
                        "tool_calls",
                    ),
                    _reply(model, {"role": "assistant", "content": "done"}),
                )
            )
    wire = ScriptedJsonTransport((*workbook_wire._responses, *replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: wire
    )
    spec = spec.model_copy(
        update={
            "families": (*spec.families, family),
            "distribution": TaskDistribution(
                seed=17,
                strata=(
                    *spec.distribution.strata,
                    CoverageStratum(
                        id="retail",
                        user="order owner",
                        capability="address change",
                        category="regression",
                        information="available",
                        horizon="single_decision",
                        minimum_families=1,
                    ),
                ),
            ),
        }
    )
    manifest = tmp_path / "two-families.json"
    manifest.write_text(spec.model_dump_json())
    monkeypatch.setattr("exp.cli.evaluation.pipeline.installed_execution_revision", lambda: "test")
    command = CliRunner().invoke(
        app,
        [
            "eval",
            project.paths.project_id,
            "--root",
            str(project.paths.root),
            "--pipeline",
            str(manifest),
            "--yes",
            "--non-interactive",
        ],
    )
    assert command.exit_code == 0, command.output
    result = saved_task_factory(project, spec, code_revision="test")
    assert result is not None and result.status == "development_qualified", command.output
    assert result.counts["source_workflow_components"] == 2
    assert result.counts["packaged_candidates"] == 4
    assert result.counts["selected_components"] == 2
    assert result.coverage.deficits == {"workbook": 0, "retail": 0}
    assert len(wire.requests) == 24


def test_busy_child_remains_pending_and_completes_after_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lease contention cannot freeze an otherwise valid family as a permanent failure."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    proposal_id = spec.families[0].proposal.identity("test")
    with FileLock(project.artifacts.project_directory / (proposal_id + ".lock")):
        with pytest.raises(Timeout):
            run_task_factory(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-result")
    result = run_task_factory(project, spec, code_revision="test")
    assert result.status == "development_qualified"
    assert all(cell.error_type is None for cell in result.cells)


def test_native_record_permission_is_strict_opt_in_through_factory_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy identities stay unchanged; explicit replay reaches the normal child owner."""
    from pydantic import ValidationError

    from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec
    from exp.optimize.evaluation.factory.runner import _batch

    project, spec, _ = factory_fixture(tmp_path, monkeypatch)
    family = spec.families[0]
    legacy_family = family.model_dump(mode="json")
    assert "replay_native_records" not in legacy_family
    batch = _batch(project, spec, family, "test")
    assert "replay_native_records" not in batch.model_dump(mode="json")
    assert (
        family.model_copy(update={"replay_native_records": False}).model_dump(mode="json")
        == legacy_family
    )
    opted = family.model_copy(update={"replay_native_records": True})
    forwarded = _batch(project, spec, opted, "test")
    assert forwarded.replay_native_records is True
    assert forwarded.identity("test") != batch.identity("test")
    for invalid in ("yes", 1, None):
        with pytest.raises(ValidationError):
            FactoryFamily.model_validate({**legacy_family, "replay_native_records": invalid})
        with pytest.raises(ValidationError):
            CandidateBatchSpec.model_validate(
                {**batch.model_dump(mode="json"), "replay_native_records": invalid}
            )

"""Candidate batches execute real XLSX tasks and package only measured, scoped qualifications."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import Timeout

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import PanelActor
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import load_executable_task, persist_executable_task
from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec, run_candidate_batch
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    load_task_proposals,
    propose_tasks,
)
from exp.optimize.evaluation.factory.quality import QUALITY_AXES, QualityFinding, TaskQualityReview
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.capability_briefs import (
    CapabilityAssumption,
    CapabilityBrief,
    RequiredInformation,
    persist_capability_brief,
)


def _reply(model: str, message: JsonObject, finish: str = "stop") -> JsonHttpResponse:
    """Return one provider-native scripted response without bypassing model parsing or admission."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "reply",
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        },
    )


def fixture_quality_reviews(
    project: ProjectStore, proposals: ArtifactInput, sources: tuple[ArtifactInput, ...]
) -> tuple[TaskQualityReview, ...]:
    """Supply software stand-ins for review plumbing, never real expert labels."""
    return tuple(
        TaskQualityReview(
            task=load_task_variant(project, variant).task,
            proposals=proposals,
            reviewer="software-fixture-human-stand-in",
            authority="human",
            approved=True,
            findings=tuple(
                QualityFinding(
                    axis=axis,
                    verdict="pass",
                    rationale="Software fixture only; not actual independent expert calibration.",
                    sources=sources,
                )
                for axis in QUALITY_AXES
            ),
        )
        for variant in load_task_proposals(project, proposals).variants
    )


def batch_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, weak_count: int = 2
) -> tuple[ProjectStore, CandidateBatchSpec, ScriptedJsonTransport]:
    """Generate two candidates, independent authored controls and scripted native outcomes."""
    project, parent_ref, qualification = qualified_fixture(tmp_path, monkeypatch)
    parent = load_executable_task(project, parent_ref)
    original_acceptance = parent.job.study.training_specification
    assert original_acceptance is not None
    base = parent.model_copy(
        update={
            "job": parent.job.model_copy(
                update={
                    "study": parent.job.study.model_copy(
                        update={"repeats": 1, "training_specification": None}
                    )
                }
            )
        }
    )
    base_ref = persist_executable_task(
        project, base, created_at=datetime.now(UTC), code_revision="test"
    )
    brief = CapabilityBrief(
        category="regression",
        novelty="existing_workflow",
        user="workbook owner",
        outcome=base.intent,
        capability="bounded row deletion",
        gap="Equivalent prompts remain unmeasured",
        required_tools=("edit_workbook",),
        required_information=(
            RequiredInformation(fact="row to delete", obtain_via="public task instruction"),
        ),
        allowed_changes=("output workbook rows",),
        preserved=("input workbook",),
        unavailable_behavior="Report missing input",
        success_evidence=("output has one fewer row and input is unchanged",),
        assumptions=(
            CapabilityAssumption(
                statement="This authored task expects one row removed",
                authority="authored_contract",
                sources=(original_acceptance,),
            ),
        ),
    )
    brief_ref = persist_capability_brief(
        project.artifacts, brief, created_at=datetime.now(UTC), code_revision="test"
    )
    proposal = TaskProposalSpec(
        base=base_ref,
        brief=brief_ref,
        model_alias=base.job.actor_alias,
        model=base.job.study.actor_model,
        catalog_sha256=base.job.catalog_sha256,
        deadline=base.job.deadline,
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
                            "id": "propose",
                            "type": "function",
                            "function": {
                                "name": "propose_tasks",
                                "arguments": json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "prompt": "Delete row two; retain the input.",
                                                "rationale": "Direct instruction.",
                                            },
                                            {
                                                "prompt": "Retain input; drop output row two.",
                                                "rationale": "Order of constraints changed.",
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
    proposals = artifact_input(project.artifacts.read(proposal.identity("test")).manifest)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    spec = CandidateBatchSpec(
        proposals=proposals,
        controls=qualification.cases,
        actors=(
            PanelActor(alias="world", model=catalog.snapshot("world")[0], label="declared weak"),
            PanelActor(alias="judge", model=catalog.snapshot("judge")[0], label="declared strong"),
        ),
        destination=tmp_path / "packages",
        quality_reviews=fixture_quality_reviews(
            project, proposals, (brief_ref, original_acceptance)
        ),
    )
    replies = []
    for _ in range(2):
        for model, count in (("world-v1", 1), ("world-v1", weak_count), ("judge-v1", 1)):
            call = workbook_case().first
            arguments = {
                **call.arguments,
                "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
            }
            replies.extend(
                (
                    _reply(
                        model,
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "edit",
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        "tool_calls",
                    ),
                    _reply(model, {"role": "assistant", "content": "done"}),
                )
            )
    transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    return project, spec, transport


def test_every_prototype_runs_same_pipeline_and_exports_only_scoped_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real native candidates pass controls and exhibit the measured scripted actor gap."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    report = run_candidate_batch(project, spec, code_revision="test")
    assert report.counts["proposed"] == 2 and report.counts["packaged"] == 2, (
        report.model_dump_json()
    )
    assert len(transport.requests) == 12
    for cell in report.candidates:
        assert (
            cell.package is not None
            and verify_executable_package(project, cell.package) == cell.task
        )
        assert (
            cell.feasibility is not None
            and cell.qualification is not None
            and cell.panel is not None
        )
    assert "full task certification" in report.scope
    task = load_executable_task(project, report.candidates[0].task)
    task.job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    assert run_candidate_batch(project, spec, code_revision="test") == report
    assert len(transport.requests) == 12


def test_unqualified_candidates_remain_in_inventory_without_panel_or_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing prohibited-effect controls stops shipping before buying tier comparisons."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={
            "controls": tuple(
                case for case in spec.controls if "side-effect" not in case.challenge.id
            )
        }
    )
    report = run_candidate_batch(project, spec, code_revision="test")
    assert report.counts["proposed"] == 2 and report.counts["unqualified"] == 2
    assert all(cell.panel is None and cell.package is None for cell in report.candidates)
    assert len(transport.requests) == 4
    assert spec.destination is not None and not spec.destination.exists()
    assert run_candidate_batch(project, spec, code_revision="test") == report


def test_changed_saved_package_refuses_replay_without_any_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid receipt does not allow corrupted on-disk packages to silently ship."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    report = run_candidate_batch(project, spec, code_revision="test")
    package = report.candidates[0].package
    assert package is not None
    (package / "instruction.md").write_text("Do a different task.")
    with pytest.raises(ValueError, match="changed after export"):
        run_candidate_batch(project, spec, code_revision="test")
    assert len(transport.requests) == 12


def test_correctness_qualified_tasks_ship_without_measured_tier_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equally successful actors do not invalidate an independently checked task contract."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch, weak_count=1)
    report = run_candidate_batch(project, spec, code_revision="test")
    assert report.counts["packaged"] == 2
    assert report.counts["tier_separation"] == {"observed": 0, "not_observed": 2, "unavailable": 0}
    assert all(
        cell.qualification is not None and cell.panel is not None for cell in report.candidates
    )
    assert all(cell.separation is False and cell.package is not None for cell in report.candidates)
    assert spec.destination is not None and spec.destination.exists()
    assert len(transport.requests) == 12
    assert run_candidate_batch(project, spec, code_revision="test") == report


def test_parent_crash_reuses_completed_child_stages_and_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed parent publication cannot buy a new feasibility or tier measurement."""
    from collections.abc import Mapping

    from pydantic import BaseModel

    from exp.common.core.artifacts import ArtifactEnvelope, JsonValue
    from exp.common.project import ArtifactManifest
    from exp.optimize.evaluation.factory.candidates import CandidateMeasurement

    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    write = project.artifacts.write_json
    lost = []

    def lose_parent(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Fail after a completed child's exported package already exists."""
        if artifact_type == "candidate-measurement" and not lost:
            cell = files["cell.json"]
            assert isinstance(cell, CandidateMeasurement) and cell.package is not None
            lost.append(cell)
            raise OSError("lost parent publication")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    monkeypatch.setattr(project.artifacts, "write_json", lose_parent)
    with pytest.raises(OSError, match="lost parent"):
        run_candidate_batch(project, spec, code_revision="test")
    monkeypatch.setattr(project.artifacts, "write_json", write)
    report = run_candidate_batch(project, spec, code_revision="test")
    assert report.candidates[0] == lost[0]
    assert report.counts["packaged"] == 2 and len(transport.requests) == 12


def test_failed_export_keeps_operator_phase_and_never_raw_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package failure remains distinguishable from qualification or actor failure."""
    project, spec, transport = batch_fixture(tmp_path, monkeypatch)

    def unavailable(*args: object, **kwargs: object) -> Path:
        """Represent a private operator storage failure without printing its raw text."""
        raise ValueError("inert-private-storage-canary")

    monkeypatch.setattr("exp.optimize.evaluation.factory.candidates._package", unavailable)
    result = run_candidate_batch(project, spec, code_revision="test")
    assert len(transport.requests) == 12
    assert result.counts["unavailable"] == 2
    for cell in result.candidates:
        assert cell.failure_detail is not None
        assert cell.failure_detail.details["phase"] == "package_export"
        assert cell.failure_detail.details["task"] == cell.task.model_dump(mode="json")
        assert cell.qualification is not None and cell.panel is not None
    assert "inert-private-storage-canary" not in result.model_dump_json()
    assert run_candidate_batch(project, spec, code_revision="test") == result
    assert len(transport.requests) == 12


def test_busy_feasibility_child_remains_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked child cannot permanently exclude an otherwise measurable candidate."""
    project, spec, _ = batch_fixture(tmp_path, monkeypatch)

    def busy(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Inject contention at the feasibility owner's lease boundary."""
        raise Timeout("feasibility.lock")

    with monkeypatch.context() as patch:
        patch.setattr("exp.optimize.evaluation.factory.candidates.execute_continuation_job", busy)
        with pytest.raises(Timeout):
            run_candidate_batch(project, spec, code_revision="test")
    assert not project.artifacts.exists(spec.identity("test") + "-candidate-0")
    report = run_candidate_batch(project, spec, code_revision="test")
    assert report.candidates[0].feasibility is not None
    assert run_candidate_batch(project, spec, code_revision="test") == report


def test_native_record_controls_require_opt_in_and_cached_reader_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real SQLite witnesses require permission and cached reads buy no new inference."""
    from contextlib import AbstractContextManager

    from exp.common.models import ToolCall
    from exp.common.tasks import ToolSchema
    from exp.optimize.evaluation.continuations.job import load_continuation_runtime
    from exp.optimize.evaluation.executable import qualification as qualification_owner
    from exp.optimize.evaluation.executable.loader import ContinuationRuntime
    from exp.optimize.evaluation.factory.candidates import saved_candidate_batch
    from exp.optimize.evaluation.factory.runner_test import factory_fixture, record_family
    from exp.runtime.environments.world_artifacts import load_world_image
    from exp.runtime.environments.world_contracts import ExecutableCheckpoint
    from exp.runtime.environments.worlds import ExecutableWorldRuntime
    from exp.simulation.continuations import load_continuation_checkpoint
    from exp.simulation.evaluation.admission import (
        EvaluationSpecification,
        LabelWitness,
        persist_specification,
        record_label_witness,
    )
    from exp.simulation.evaluation.record_execution import (
        RecordReplayRuntime,
        execute_record_script,
        load_record_execution,
    )
    from exp.simulation.evaluation.tasks import OutcomeRule

    project, factory, _ = factory_fixture(tmp_path, monkeypatch)
    family = record_family(project, factory, tmp_path, monkeypatch)
    task = load_executable_task(project, family.proposal.base)
    checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
    assert isinstance(checkpoint.world, ExecutableCheckpoint)
    cases = []
    with load_continuation_runtime(project, task.job) as runtime:
        assert isinstance(runtime.world, ExecutableWorldRuntime)
        replay = RecordReplayRuntime(runtime.world, runtime.probe, runtime.probe_sha256)
        baselines = {}
        for partition in ("development", "confirmation"):
            operator = checkpoint.world_task.model_copy(
                update={
                    "tools": (
                        *checkpoint.world_task.tools,
                        ToolSchema(
                            name="setup",
                            description="Operator setup only.",
                            input_schema={"type": "object"},
                        ),
                    )
                }
            )
            setup = execute_record_script(
                project.artifacts,
                replay,
                task=operator,
                baseline=checkpoint.world,
                actions=(
                    ToolCall(
                        call_id=f"setup-{partition}",
                        name="setup",
                        arguments={"id": "b", "address": partition},
                    ),
                ),
                finish_tick=checkpoint.world.clock + 1,
                sources=(family.proposal.base,),
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            baseline = load_world_image(
                project.artifacts, load_record_execution(project.artifacts, setup).final
            )
            assert isinstance(baseline, ExecutableCheckpoint)
            baselines[partition] = (baseline, setup)
        for index, case in enumerate(family.controls):
            if case.challenge.expected == "unassessable":
                cases.append(case)
                continue
            old_witness = case.challenge.witness
            assert old_witness is not None
            source = LabelWitness.model_validate_json(
                project.artifacts.read_bytes(old_witness.artifact_id, "witness.json")
            )
            arguments: JsonObject = {
                "address": "wrong" if "goal-miss" in case.challenge.id else "new"
            }
            if "valid-two" in case.challenge.id:
                arguments["id"] = "a"
            if "side-effect" in case.challenge.id:
                arguments["email"] = True
            baseline, setup = baselines[case.partition]
            receipt = execute_record_script(
                project.artifacts,
                replay,
                task=checkpoint.world_task,
                baseline=baseline,
                actions=(ToolCall(call_id=f"native-{index}", name="change", arguments=arguments),),
                finish_tick=baseline.clock + 1,
                sources=(setup,),
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            native = load_record_execution(project.artifacts, receipt)
            authored = EvaluationSpecification.model_validate_json(
                project.artifacts.read_bytes(source.specification.artifact_id, "specification.json")
            )
            native_spec = persist_specification(
                project.artifacts,
                authored.model_copy(
                    update={
                        "subject": "native_observation",
                        "predicate": authored.predicate.model_copy(
                            update={
                                "rules": (
                                    OutcomeRule(
                                        path="/final_state/tables/orders/rows/0/1",
                                        operator="equals",
                                        value="new",
                                    ),
                                )
                            }
                        ),
                    }
                ),
                sources=(source.specification,),
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            witness = record_label_witness(
                project.artifacts,
                specification=native_spec,
                outcome=native.outcome,
                receipt=receipt,
                record_runtime=replay,
                created_at=datetime.now(UTC),
                code_revision="test",
            )
            challenge = case.challenge.model_copy(
                update={
                    "outcome": native.outcome,
                    "authority": "native_execution",
                    "witness": witness,
                }
            )
            cases.append(case.model_copy(update={"challenge": challenge}))
    proposals = artifact_input(project.artifacts.read(family.proposal.identity("test")).manifest)
    spec = CandidateBatchSpec(
        proposals=proposals,
        controls=tuple(cases),
        actors=factory.actors,
        quality_reviews=family.quality_reviews,
    )

    def responses(models: tuple[str, ...]) -> tuple[JsonHttpResponse, ...]:
        """Use ordinary provider-native scripted parsing/admission; never live calls."""
        result = []
        for model in models:
            result.extend(
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
        return tuple(result)

    false_wire = ScriptedJsonTransport(responses(("world-v1", "world-v1")))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: false_wire
    )
    calls = []
    original = qualification_owner.load_continuation_runtime

    def tracked(
        project: ProjectStore, job: ContinuationJob
    ) -> AbstractContextManager[ContinuationRuntime]:
        """Observe permission-bound real runtime loading, without replacing native execution."""
        calls.append(job.runtime_sha256)
        return original(project, job)

    monkeypatch.setattr(qualification_owner, "load_continuation_runtime", tracked)
    declined = run_candidate_batch(project, spec, code_revision="test")
    assert declined.counts["unqualified"] == 2 and not calls and len(false_wire.requests) == 4
    opted = spec.model_copy(update={"replay_native_records": True})
    assert opted.identity("test") != spec.identity("test")
    true_wire = ScriptedJsonTransport(responses(("world-v1", "judge-v1", "world-v1", "judge-v1")))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: true_wire
    )
    measured = run_candidate_batch(project, opted, code_revision="test")
    assert measured.counts["qualified_candidate"] == 2, measured.model_dump_json()
    assert len(true_wire.requests) == 8 and calls
    before = len(calls)
    assert saved_candidate_batch(project, opted, code_revision="test") == measured
    assert len(calls) > before and len(true_wire.requests) == 8
    before = len(calls)
    assert saved_candidate_batch(project, spec, code_revision="test") == declined
    assert len(calls) == before

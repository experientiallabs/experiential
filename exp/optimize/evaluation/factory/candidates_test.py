"""Candidate batches execute real XLSX tasks and package only measured, scoped qualifications."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import Timeout

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import PanelActor
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import (
    load_executable_task,
    persist_executable_task,
    verify_executable_package,
)
from exp.optimize.evaluation.factory.candidates import CandidateBatchSpec, run_candidate_batch
from exp.optimize.evaluation.factory.proposals import TaskProposalSpec, propose_tasks
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

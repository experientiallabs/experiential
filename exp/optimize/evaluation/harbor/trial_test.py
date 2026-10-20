"""Real durable sessions and HTTP boundaries preserve frozen worlds and private grading."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from filelock import FileLock, Timeout

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json, stable_id
from exp.common.evaluations.reconstruction import AxisOutcome, DualReport, WorkflowEvidence
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import artifact_input
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.optimize.evaluation.harbor.export import export_harbor_task, verify_harbor_export
from exp.optimize.evaluation.harbor.trial import HarborTrial
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.sessions.store import SessionStore
from exp.simulation.evaluation.tasks import (
    EvaluationTask,
    FieldRelation,
    TaskVerifier,
    load_evaluation_task,
    persist_evaluation_task,
)
from exp.simulation.world_model.checkpoints import capture_stored_world_checkpoint
from exp.simulation.world_model.durable import SessionPolicy
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.runtime import load_grounded_world_model


class UnusedModel:
    """Reject any accidental hosted dispatch in a deterministic integration test."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """All test proposals must go through the explicit completion fixture."""
        raise AssertionError("unexpected hosted call")


def test_real_session_http_replay_export_and_private_boundaries(tmp_path: Path) -> None:
    """Exercise the transport and commit owner without pretending fixtures are native truth."""
    project, spec = source_project(tmp_path)
    registry = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = registry.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=UnusedModel(),
        capabilities=capabilities,
    )
    now = datetime.now(UTC)
    policy = SessionPolicy(
        deadline=now + timedelta(minutes=5),
        maximum_operations=3,
        inference_identity="0" * 64,
        code_revision="fixture",
    )
    calls = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return a declared software fixture through the real transition parser."""
        calls.append(identity)
        proposal: JsonObject = {
            "observations": [
                {"call_id": "call", "reply": {"status": "pending"}, "encoding": "json"}
            ],
            "effect": {
                "kind": "edits",
                "edits": [{"op": "set", "path": ["status"], "value": "pending"}],
            },
        }
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(
                    ToolCall(call_id="proposal", name="submit_transition", arguments=proposal),
                )
            ),
            model=runtime.artifact.model,
            economics=OperationEconomics(),
        ).model_dump(mode="json")

    source_store = SessionStore(tmp_path / "source.sqlite")
    open_grounded_session(
        runtime=runtime,
        task=spec.task,
        store=source_store,
        artifacts=project.artifacts,
        session_id="source",
        policy=policy,
        complete=complete,
        lock_path=tmp_path / "source.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
    )
    checkpoint = capture_stored_world_checkpoint(
        source_store,
        "source",
        spec.world_model,
        project.artifacts,
        public_context=spec.task.initial_context,
        prefix_position=0,
        created_at=now,
        code_revision="fixture",
    )
    task_ref = persist_evaluation_task(
        project.artifacts,
        EvaluationTask(
            intent="Report observed status",
            task=spec.task,
            world_model=spec.world_model,
            checkpoint=artifact_input(project.artifacts.read(checkpoint.artifact_id).manifest),
            verifier=TaskVerifier(
                description="Report observed status",
                outcome_schema={"type": "object"},
                relations=(
                    FieldRelation(left="/final_answer/status", right="/steps/0/reply/status"),
                ),
                allowed_tools=("inspect",),
            ),
            challenges=(),
        ),
        created_at=now,
        code_revision="fixture",
    )

    def trial(
        index: int,
        *,
        bound: bool = False,
        threshold: int = 4,
        completion_profile: str | None = "7" * 64,
    ) -> HarborTrial:
        """Open or restore an independently owned frozen fork."""
        session = open_grounded_session(
            runtime=runtime,
            task=spec.task,
            store=SessionStore(tmp_path / f"trial-{index}.sqlite"),
            artifacts=project.artifacts,
            session_id=f"trial-{index}",
            policy=policy,
            complete=complete,
            lock_path=tmp_path / f"trial-{index}.lock",
            maximum_output_tokens=4096,
            output_protocol="json-tool",
            checkpoint=checkpoint,
        )
        if not bound:
            return HarborTrial(project.artifacts, task_ref, session, "fixture")
        calibration_id = f"fixture-calibration-{index}"
        sources = session.evaluation_context["sources"]
        assert isinstance(sources, dict)
        project.artifacts.write_json(
            artifact_id=calibration_id,
            artifact_type="task-calibration",
            envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="fixture"),
            files={
                "report.json": {
                    "qualification": "qualified_in_scope",
                    "frozen_profile_sha256": sources["frozen_profile_sha256"],
                    "task": task_ref.model_dump(mode="json"),
                    "candidate_sha256": sha256_json(
                        load_evaluation_task(project.artifacts, task_ref).verifier
                    ),
                    "judge_sha256": "8" * 64,
                    "judge_completion_sha256": "7" * 64,
                    "required_dimensions": ["outcome"],
                    "minimum_scores": {"outcome": 4},
                }
            },
        )

        def judge(evidence: WorkflowEvidence) -> DualReport:
            """Return a software fixture assessment, not real calibration evidence."""
            return DualReport(
                workflow_id=evidence.workflow_id,
                evidence_sha256=sha256_json(evidence),
                reference_sha256="9" * 64,
                judge_config_sha256="8" * 64,
                planned_steps=len(evidence.steps),
                judge_status="completed",
                axes=tuple(
                    AxisOutcome(
                        dimension_id=key,
                        status="assessed" if key == "outcome" else "unassessable",
                        score=5 if key == "outcome" else None,
                        rationale="Software fixture",
                        evidence_refs=("step-0",),
                    )
                    for key in ("outcome", "evidence", "compliance", "consistency", "quality")
                ),
                mechanical={},
            ).model_copy(update={"judge_completion_sha256": completion_profile})

        return HarborTrial(
            project.artifacts,
            task_ref,
            session,
            "fixture",
            evaluate=judge,
            minimum_scores={"outcome": threshold},
            calibration=artifact_input(project.artifacts.read(calibration_id).manifest),
        )

    first, second = trial(1), trial(2)
    duplicate = trial(1)
    identity = stable_id(
        "harbor-trial", {"task": task_ref.model_dump(mode="json"), "session": "trial-1"}
    )
    with FileLock(project.artifacts.project_directory / (identity + ".lock"), timeout=0):
        with pytest.raises(Timeout):
            duplicate.submit({"status": "pending"})
    with pytest.raises(ValueError, match="binding changed"):
        HarborTrial(project.artifacts, task_ref, first.session, "another-code-revision")
    with serve_harbor_trial(first) as endpoint:
        with urlopen(
            Request(
                endpoint.url + "/tools", headers={"Authorization": "Bearer " + endpoint.actor_token}
            )
        ) as response:
            assert "inspect" in response.read().decode()
        with pytest.raises(HTTPError) as denied:
            urlopen(
                Request(
                    endpoint.url + "/grade",
                    data=b"{}",
                    headers={"Authorization": "Bearer " + endpoint.actor_token},
                )
            )
        assert denied.value.code == 403
        action = ToolCall(call_id="call", name="inspect", arguments={"key": "current"})
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("client.py")),
                "call",
                "--body",
                action.model_dump_json(),
            ],
            env={
                "EXP_HARBOR_ENDPOINT": endpoint.url,
                "EXP_HARBOR_ACTOR_TOKEN": endpoint.actor_token,
            },
            capture_output=True,
            check=True,
        )
        first.call(action)
        assert len(calls) == 1
        assert not second.session.transitions
        first.submit({"status": "delivered"})
        with pytest.raises(ValueError, match="trial is closed"):
            duplicate.call(action)
        request = Request(
            endpoint.url + "/grade",
            data=b"{}",
            headers={"Authorization": "Bearer " + endpoint.verifier_token},
        )
        with urlopen(request) as response:
            result = json.load(response)
        assert result["mechanical"]["status"] == "fail"
        assert result["reward"] is None
        assert result["qualification"] == "diagnostic_only"
    recovered = trial(1)
    assert recovered.result() == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="not calibrated"):
        export_harbor_task(project, task_ref, tmp_path / "qualified", name="test/frozen")
    exported = export_harbor_task(
        project, task_ref, tmp_path / "diagnostic", name="test/frozen", diagnostic=True
    )
    assert (exported / "tests/test.sh").exists()
    assert not (exported / "environment/frozen-binding.json").exists()
    public = (exported / "environment/tools.py").read_text()
    assert task_ref.artifact_id not in public
    assert endpoint.actor_token not in public
    accepted = trial(3, bound=True)
    accepted.call(action)
    accepted.submit({"status": "pending"})
    accepted_result = accepted.result()
    assert accepted_result["reward"] == 1
    accepted_result["reward"] = 0
    assert accepted.result()["reward"] == 1
    mismatched = trial(4, bound=True, threshold=3)
    mismatched.call(action)
    mismatched.submit({"status": "pending"})
    with pytest.raises(ValueError, match="calibration differs"):
        mismatched.result()
    outside = trial(7)
    outside.submit({"status": "pending"})
    outside.session.step((action,), operation_id=action.call_id)
    with pytest.raises(ValueError, match="submitted trial state changed"):
        outside.result()
    for index, profile in ((5, "6" * 64), (6, None)):
        changed = trial(index, bound=True, completion_profile=profile)
        changed.call(action)
        changed.submit({"status": "pending"})
        with pytest.raises(ValueError, match="calibration differs"):
            changed.result()


def test_export_audit_detects_injected_files_even_when_named_like_the_manifest(
    tmp_path: Path,
) -> None:
    """An added actor file cannot hide by taking the root manifest's basename."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Public task")
    (task / "frozen-binding.json").write_text(
        json.dumps(
            {
                "files": {
                    "instruction.md": hashlib.sha256(b"Public task").hexdigest(),
                }
            }
        )
    )
    assert verify_harbor_export(task)["files"]
    nested = task / "environment"
    nested.mkdir()
    (nested / "frozen-binding.json").write_text("Injected actor content")
    with pytest.raises(ValueError, match="changed after export"):
        verify_harbor_export(task)

"""CLI runtime composition restores a real workbook and meters scripted actor HTTP calls."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuation_admission import ContinuationAdmission
from exp.optimize.evaluation.continuation_admission_test import admitted_workbook_study
from exp.optimize.evaluation.continuation_job import (
    ContinuationJob,
    execute_continuation_job,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuation_records import load_study
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def job_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ContinuationJob, ScriptedJsonTransport]:
    """Configure an explicitly selected local runtime and three routed actor responses."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project)
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(acceptance.artifact_id, "admission.json")
    )
    study = load_study(project.artifacts, admission.study)
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog).resolve("world")
    runtime = tmp_path / "selected_runtime.py"
    runtime.write_text("""from exp.optimize.evaluation.continuation_job import ContinuationRuntime
from exp.optimize.evaluation.continuation_study_test import probe, workbook_runtime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.common.core.artifacts import ArtifactInput
def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["checkpoint"])
    checkpoint = load_continuation_checkpoint(project.artifacts, reference)
    return ContinuationRuntime(workbook_runtime(checkpoint.world), probe, configuration["probe"])
""")
    spec = study.spec.model_copy(update={"actor_model": model.snapshot})
    job = ContinuationJob(
        study=spec,
        runtime_file=runtime,
        runtime_sha256=sha256_bytes(runtime.read_bytes()),
        configuration={
            "checkpoint": spec.checkpoint.model_dump(mode="json"),
            "probe": spec.probe_sha256,
        },
        actor_alias="world",
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        maximum_model_calls=4,
        maximum_physical_attempts=3,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    call = workbook_case().first
    replies = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": call.call_id,
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
            ],
        },
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "unsupported completion"},
    ]
    transport = ScriptedJsonTransport(
        tuple(
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": f"reply-{i}",
                    "model": "world-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls" if i == 0 else "stop",
                            "message": row,
                        }
                    ],
                },
            )
            for i, row in enumerate(replies)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuation_job.HttpxJsonTransport", lambda: transport
    )
    return project, job, transport


def test_runtime_job_restores_and_replays_without_loading_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed results replay even when the local runtime file is no longer available."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    preflight_continuation_job(project, job)
    assert not transport.requests
    result = execute_continuation_job(project, job, code_revision="test")
    assert result.metrics()["planned"] == 2 and result.metrics()["successes"] == 1
    assert len(transport.requests) == 3
    physical = [
        json.loads(project.artifacts.read_bytes(identity, "receipt.json"))
        for identity in project.artifacts.list_ids(artifact_type="provider-physical-attempt")
        if identity.endswith("-intent")
    ]
    assert len(physical) == 3
    assert sorted(json.dumps(row["request_body"], sort_keys=True) for row in physical) == sorted(
        json.dumps(row.payload, sort_keys=True) for row in transport.requests
    )
    for receipt in physical:
        assert receipt["consumer"] == "actor"
        canonical = receipt["canonical_request"]["artifact_id"]
        request = json.loads(project.artifacts.read_bytes(canonical, "request.json"))
        assert request["maximum_output_tokens"] == job.study.maximum_output_tokens
    job.runtime_file.unlink()
    assert execute_continuation_job(project, job, code_revision="test") == result
    assert len(transport.requests) == 3


def test_runtime_drift_refuses_before_code_or_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest mismatch cannot execute a newly edited local runtime."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    job.runtime_file.write_text("raise AssertionError('must not import')")
    with pytest.raises(ValueError, match="runtime source changed"):
        execute_continuation_job(project, job, code_revision="test")
    assert not transport.requests

"""CLI runtime composition restores a real workbook and meters scripted actor HTTP calls."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes, sha256_json
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.admission import ContinuationAdmission
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.evaluation.continuations.framing import FramingStudySpec
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    execute_continuation_job,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog


class OutageTransport(ScriptedJsonTransport):
    """Scripted provider that is unreachable once its replies run out.

    Continuation runners retain provider failures as outcome-unknown measurements, while a
    programming error (such as the base double's unexpected-request assertion) propagates. Fixtures
    that model an unavailable provider therefore end in a transport failure, not an assertion.
    """

    def _answer(self) -> JsonHttpResponse:
        """Serve the next scripted reply, or fail like an unreachable provider."""
        if not self._responses:
            raise ProviderTransportError("scripted provider has no further replies")
        return super()._answer()


def job_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pin_policy: bool = False
) -> tuple[ProjectStore, ContinuationJob, ScriptedJsonTransport]:
    """Configure an explicitly selected local runtime and three routed actor responses.

    With ``pin_policy`` the job's study keeps the fixture admission policy pinned, so its
    results can be admitted in this project; otherwise the job carries no pin and can run in
    any project.
    """
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
    runtime.write_text("""from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.common.core.artifacts import ArtifactInput
def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["checkpoint"])
    checkpoint = load_continuation_checkpoint(project.artifacts, reference)
    return ContinuationRuntime(workbook_runtime(checkpoint.world), probe, configuration["probe"])
""")
    pinned = study.spec.admission_policy if pin_policy else None
    spec = study.spec.model_copy(update={"actor_model": model.snapshot, "admission_policy": pinned})
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
    transport = OutageTransport(
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
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
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


def test_one_physical_attempt_ceiling_bounds_jobs_and_composed_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job and a panel composed from jobs share the same named attempt ceiling."""
    _, job, _ = job_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="less than or equal"):
        job.model_validate(
            {**job.model_dump(), "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS + 1}
        )
    largest = job.model_copy(
        update={
            "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS,
            "study": job.study.model_copy(update={"training_specification": None}),
        }
    )
    with pytest.raises(ValueError, match="physical-attempt ceiling"):
        FramingStudySpec(base=largest, intervention="Evaluation", equivalence_rationale="cue")


def test_actor_reasoning_effort_must_be_declared_by_its_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An effort the actor deployment cannot preserve refuses before code or provider calls."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    low = job.model_copy(update={"study": job.study.model_copy(update={"reasoning_effort": "low"})})
    with pytest.raises(ValueError, match="actor reasoning effort 'low' is not supported by world"):
        preflight_continuation_job(project, low)
    with pytest.raises(ValueError, match="none declared"):
        execute_continuation_job(project, low, code_revision="test")
    assert not transport.requests


def test_actor_spending_its_allowance_is_sent_its_effort_and_counted_as_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty length-terminated replies end each branch as a counted budget exhaustion."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    world = catalog.models["world"]
    assert world.capabilities is not None
    world = world.model_copy(
        update={
            "capabilities": world.capabilities.model_copy(update={"supports_reasoning": True}),
            "supported_reasoning_efforts": ("low", "medium", "high"),
        }
    )
    catalog = catalog.model_copy(update={"models": {**catalog.models, "world": world}})
    write_model_catalog(project.model_catalog_path, catalog)
    snapshot = RuntimeModelCatalog(catalog).resolve("world").snapshot
    job = job.model_copy(
        update={
            "study": job.study.model_copy(
                update={"reasoning_effort": "low", "actor_model": snapshot}
            ),
            "catalog_sha256": sha256_json(catalog.model_dump(mode="json")),
            "maximum_physical_attempts": 20,
        }
    )
    empty = JsonHttpResponse(
        status_code=200,
        body={
            "id": "reply",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": None},
                }
            ],
        },
    )
    transport = OutageTransport(tuple(empty for _ in range(20)))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    result = execute_continuation_job(project, job, code_revision="test")
    assert transport.requests
    assert all(request.payload["reasoning_effort"] == "low" for request in transport.requests)
    assert [row.actor_stop for row in result.branches] == ["actor_budget_exhausted"] * 2
    assert [row.failure for row in result.branches] == ["ProviderRetryableResponseError"] * 2
    metrics = result.metrics()
    assert (metrics["assessable"], metrics["actor_budget_exhausted"]) == (0, 2)
    assert metrics["actor_budget_exhausted_share"] == 1.0


_WORKER_RUNTIME = """from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.contracts import TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.common.core.artifacts import ArtifactInput
def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["checkpoint"])
    checkpoint = load_continuation_checkpoint(project.artifacts, reference)
    sandbox = ProgramSandbox()
    source = "def transition(state, args, facts):\\n    return 1\\n"
    program = TransitionProgram(tool="t", source=source)
    assert sandbox.run(program, {}, {}, {}).status == "reply"
    (project.artifacts.project_directory / "worker.pid").write_text(str(sandbox._process.pid))
    world = workbook_runtime(checkpoint.world)
    return ContinuationRuntime(world, probe, configuration["probe"], sandbox.close)
"""


class _Interrupted(KeyboardInterrupt):
    """An operator interrupt during the continuation study."""


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_job_closes_its_runtime_and_leaves_no_worker_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    """Studies, panels and campaigns all execute here, so every exit path reaps the worker."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    job.runtime_file.write_text(_WORKER_RUNTIME)
    job = job.model_copy(update={"runtime_sha256": sha256_bytes(job.runtime_file.read_bytes())})
    if interrupted:

        def interrupt(*args: object, **kwargs: object) -> None:
            """Stop where the actor would start."""
            del args, kwargs
            raise _Interrupted

        monkeypatch.setattr(
            "exp.optimize.evaluation.continuations.job.run_continuation_study", interrupt
        )
        with pytest.raises(_Interrupted):
            execute_continuation_job(project, job, code_revision="test")
    else:
        assert execute_continuation_job(project, job, code_revision="test").metrics()["planned"]
    worker = int((project.artifacts.project_directory / "worker.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(worker, 0)

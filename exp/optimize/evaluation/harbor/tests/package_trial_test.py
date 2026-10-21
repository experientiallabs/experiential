"""Frozen executable packages pass a local Harbor-contract trial for two world kinds.

The public CLI registers, qualifies, exports and serves each task. The exported package's own
standalone client then runs as separate actor and verifier processes, each holding only its own
capability, exactly as Harbor runs ``environment/tools.py`` and ``tests/test.sh`` in separate
containers. Actor providers are scripted; no call leaves the machine. Every trial must leave the
reward file Harbor requires, retain its verdict, and release every listener, lease, thread, child
process and temporary directory it created.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.executable import ExecutableCommand
from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_bytes, sha256_json
from exp.common.models import AssistantAction, ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.continuations.job import ContinuationJob, execute_continuation_job
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.qualification import ExecutableQualificationSpec
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import ContinuationCheckpoint, persist_continuation_checkpoint
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase
from exp.simulation.evaluation.record_execution_test import acceptance, fixture, source
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, VerifierChallenge

RECORD_RUNTIME = '''"""Explicitly selected test-only record runtime with no credentials."""
from exp.optimize.evaluation.continuations.job import ContinuationRuntime
from exp.simulation.evaluation.record_execution_test import fixture


def continuation_runtime(project, configuration):
    """Restore the authored retail record service with its evaluator-only projection."""
    replay, _ = fixture()
    return ContinuationRuntime(replay.world, replay.probe, configuration["probe"])
'''

Fixture = Callable[
    [Path, pytest.MonkeyPatch], tuple[ProjectStore, ArtifactInput, ExecutableQualificationSpec]
]


def _scripted_actor(monkeypatch: pytest.MonkeyPatch, call: ToolCall) -> None:
    """Route the feasibility actor through one scripted tool call and a final reply."""
    replies: list[JsonObject] = [
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
    ]
    transport = ScriptedJsonTransport(
        tuple(
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": f"reply-{index}",
                    "model": "world-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls" if index == 0 else "stop",
                            "message": row,
                        }
                    ],
                },
            )
            for index, row in enumerate(replies)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )


def _record_checkpoint(project: ProjectStore) -> ArtifactInput:
    """Capture the native record service at the actor's initial turn boundary."""
    replay, task = fixture()
    captured: list[ContinuationCheckpoint] = []
    with replay.world.open(task) as session:

        def capture(checkpoint: ChatCheckpoint) -> None:
            """Pair the actor prefix with the exact native world image."""
            captured.append(
                ContinuationCheckpoint(
                    task=task,
                    actor=checkpoint,
                    world=session.checkpoint(),
                    sources=(source(project),),
                )
            )

        ChatAgentRuntime(maximum_model_calls=4).run(
            task,
            model=_Model((_response(AssistantAction(content="stop")),)),
            environment=session,
            on_checkpoint=capture,
        )
    return persist_continuation_checkpoint(
        project.artifacts, captured[0], created_at=datetime.now(UTC), code_revision="test"
    )


def record_qualified_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ArtifactInput, ExecutableQualificationSpec]:
    """Execute a feasible native record branch and author witnessed record controls."""
    project = ProjectStore(tmp_path, "records")
    store = project.artifacts
    checkpoint = _record_checkpoint(project)
    specification = persist_specification(
        store,
        EvaluationSpecification(
            description="Order a has the requested address; other records and the outbox hold.",
            predicate=acceptance(),
            subject="outcome",
        ),
        sources=(source(project),),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog).resolve("world")
    replay, _ = fixture()
    study = ContinuationStudySpec(
        checkpoint=checkpoint,
        actor_model=model.snapshot,
        verifier=TaskVerifier(
            description="Move order a to the new address without collateral effects.",
            outcome_schema={},
            rules=(
                OutcomeRule(
                    path="/final_state/tables/orders/rows/0/1", operator="equals", value="new"
                ),
            ),
            allowed_tools=("change",),
            preserved_paths=("/tables/orders/rows/1", "/tables/outbox"),
        ),
        probe_sha256=replay.probe_sha256,
        repeats=1,
        selector="authored record checkpoint",
        training_specification=specification,
    )
    runtime = tmp_path / "record_runtime.py"
    runtime.write_text(RECORD_RUNTIME)
    job = ContinuationJob(
        study=study,
        runtime_file=runtime,
        runtime_sha256=sha256_bytes(runtime.read_bytes()),
        configuration={"probe": study.probe_sha256},
        actor_alias="world",
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        maximum_model_calls=4,
        maximum_physical_attempts=3,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    _scripted_actor(
        monkeypatch, ToolCall(call_id="move", name="change", arguments={"address": "new"})
    )
    result = execute_continuation_job(project, job, code_revision="test")
    study_id = next(
        key
        for key in store.list_ids(artifact_type="continuation-study")
        if key.endswith("-result")
        and store.read(key).manifest.code_revision == job.implementation("test")
    )
    task = ExecutableTask(intent="Change one order address and nothing else.", job=job)
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision="test"
    )
    cases = []
    for partition, offset in (("development", 0), ("confirmation", 10)):
        for index, kind in enumerate(
            ("valid-one", "valid-two", "goal-miss", "side-effect", "unknown")
        ):
            outcome = deepcopy(result.branches[0].outcome)
            final_state, steps = outcome["final_state"], outcome["steps"]
            assert isinstance(final_state, dict) and isinstance(steps, list)
            tables = final_state["tables"]
            assert isinstance(tables, dict)
            orders, outbox = tables["orders"], tables["outbox"]
            assert isinstance(orders, dict) and isinstance(outbox, dict)
            rows = orders["rows"]
            assert isinstance(rows, list) and isinstance(rows[0], list)
            if kind == "goal-miss":
                rows[0][1] = "partial " + partition
            elif kind == "side-effect":
                outbox["rows"] = [["unwanted email " + partition]]
            elif kind == "unknown":
                if partition == "development":
                    outcome["steps"] = None
                else:
                    outcome["stop_reason"] = "failure"
            else:
                step = steps[0]
                assert isinstance(step, dict) and isinstance(step["action"], dict)
                step["action"]["arguments"] = {
                    "id": "a",
                    "address": "new",
                    "reference": f"control-{offset + index}",
                }
            cases.append(
                CalibrationCase(
                    challenge=VerifierChallenge(
                        id=f"{partition}-{kind}",
                        outcome=outcome,
                        expected="pass"
                        if kind.startswith("valid")
                        else "unassessable"
                        if kind == "unknown"
                        else "fail",
                        authority="synthetic_specification",
                        witness=record_label_witness(
                            store,
                            specification=specification,
                            outcome=outcome,
                            created_at=datetime.now(UTC),
                            code_revision="test",
                        ),
                        rationale="Authored observation-level record verifier challenge.",
                    ),
                    lineage=f"authored-record-control-{offset + index}",
                    partition=partition,
                )
            )
    feasibility = artifact_input(store.read(study_id).manifest)
    return (
        project,
        reference,
        ExecutableQualificationSpec(task=reference, feasibility=feasibility, cases=tuple(cases)),
    )


def _workbook_call(call_id: str, count: int) -> ToolCall:
    """Delete ``count`` data rows from the input into the declared output workbook."""
    return workbook_case().first.model_copy(
        update={
            "call_id": call_id,
            "arguments": {
                **workbook_case().first.arguments,
                "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
            },
        }
    )


# Per world kind: fixture, then (passing call, goal-missing call, declared output paths).
KINDS: dict[str, tuple[Fixture, ToolCall, ToolCall, tuple[str, ...]]] = {
    "native-workbook": (
        qualified_fixture,
        _workbook_call("delete-one", 1),
        _workbook_call("delete-two", 2),
        ("output.xlsx",),
    ),
    "native-records": (
        record_qualified_fixture,
        ToolCall(call_id="move", name="change", arguments={"id": "a", "address": "new"}),
        ToolCall(call_id="partial", name="change", arguments={"id": "a", "address": "partial"}),
        (),
    ),
}


def _invoke(project: ProjectStore, tmp_path: Path, command: ExecutableCommand) -> str:
    """Run one public ``exp eval --continuations`` Harbor action and return its output."""
    path = tmp_path / "harbor-command.json"
    path.write_text(ContinuationCommand(action="harbor", harbor=command).model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "eval",
            project.paths.project_id,
            "--root",
            str(project.paths.root),
            "--continuations",
            str(path),
            "--yes",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


def _client(
    script: Path, connection: Path, home: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run the package's standalone client with one capability and nothing from this checkout.

    ``-I`` drops this repository and every user path from ``sys.path``, so the client must work
    exactly as copied into a bare Harbor container.
    """
    environment = {"PATH": os.environ.get("PATH", ""), **json.loads(connection.read_text())}
    return subprocess.run(
        [sys.executable, "-I", str(script), *arguments],
        env=environment,
        cwd=home,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _children() -> set[int]:
    """Return live child process IDs of this test process, where the platform exposes them."""
    tasks = Path(f"/proc/{os.getpid()}/task")
    if not tasks.is_dir():
        return set()
    return {
        int(pid)
        for task in tasks.iterdir()
        for pid in (task / "children").read_text().split()
        if (task / "children").exists()
    }


def _run_trial(
    project: ProjectStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: Path,
    *,
    trial_id: str,
    actions: tuple[ToolCall, ...],
    submit: bool,
    output_paths: tuple[str, ...],
) -> tuple[JsonObject, JsonObject, JsonObject, str]:
    """Serve one trial through the CLI while separate actor and verifier processes run it.

    Returns:
        Harbor rewards, the retained verdict, the CLI's final result and the endpoint URL.
    """
    trial = tmp_path / trial_id
    actor_home, verifier_logs, private = trial / "actor", trial / "verifier", trial / "private"
    for directory in (actor_home, verifier_logs, private):
        directory.mkdir(parents=True)
    actor_file, verifier_file = private / "actor.json", private / "verifier.json"
    # A stale passing reward from an earlier attempt must never survive this verifier run.
    (verifier_logs / "reward.json").write_text('{"reward": 1, "scored": 1}\n')
    observed: dict[str, str] = {}

    class ExternalHarbor:
        """Stand in for Harbor's agent then verifier phases during the bounded serve window."""

        def wait(self, seconds: int) -> None:
            """Run the public actor, then the independent verifier, as separate processes."""
            assert seconds == 60
            actor = json.loads(actor_file.read_text())
            observed["url"] = actor["EXP_HARBOR_ENDPOINT"]
            assert set(actor) == {"EXP_HARBOR_ENDPOINT", "EXP_HARBOR_ACTOR_TOKEN"}
            denied = Request(
                actor["EXP_HARBOR_ENDPOINT"] + "/grade",
                data=b"{}",
                headers={"Authorization": "Bearer " + actor["EXP_HARBOR_ACTOR_TOKEN"]},
            )
            with pytest.raises(HTTPError) as refused:
                urlopen(denied, timeout=5)
            assert refused.value.code == 403
            client = package / "environment" / "tools.py"
            listed = _client(client, actor_file, actor_home, "tools")
            assert listed.returncode == 0, listed.stderr
            assert json.loads(listed.stdout)["tools"]
            for action in actions:
                called = _client(
                    client, actor_file, actor_home, "call", "--body", action.model_dump_json()
                )
                assert called.returncode == 0, called.stderr
                assert json.loads(called.stdout)["is_error"] is False
            if submit:
                body = json.dumps({"answer": "done"})
                submitted = _client(client, actor_file, actor_home, "submit", "--body", body)
                assert submitted.returncode == 0, submitted.stderr
                if output_paths:
                    outputs = actor_home / "outputs"
                    fetched = _client(
                        client, actor_file, actor_home, "outputs", "--directory", str(outputs)
                    )
                    assert fetched.returncode == 0, fetched.stderr
                    assert sorted(p.name for p in outputs.iterdir()) == sorted(output_paths)
            verifier = json.loads(verifier_file.read_text())
            assert set(verifier) == {"EXP_HARBOR_ENDPOINT", "EXP_HARBOR_VERIFIER_TOKEN"}
            graded = _client(
                package / "tests" / "tools.py",
                verifier_file,
                verifier_logs,
                "verify",
                "--directory",
                str(verifier_logs),
            )
            assert graded.returncode == 0, graded.stderr
            observed["verifier_stderr"] = graded.stderr

    monkeypatch.setattr("exp.cli.evaluation.executable.Event", ExternalHarbor)
    threads, children = set(threading.enumerate()), _children()
    output = _invoke(
        project,
        tmp_path,
        ExecutableCommand(
            action="serve",
            directory=package,
            trial_id=trial_id,
            actor_connection_file=actor_file,
            verifier_connection_file=verifier_file,
            duration_seconds=60,
        ),
    )
    result = json.loads(output[output.index("{") :])
    # Cleanup: capabilities removed, listener closed, threads, children and the lease released.
    assert not list(private.iterdir())
    with pytest.raises(URLError):
        urlopen(observed["url"] + "/tools", timeout=5)
    assert set(threading.enumerate()) <= threads
    assert _children() <= children
    for lease in project.artifacts.project_directory.glob("*.lock"):
        with FileLock(lease, timeout=0):
            pass
    rewards = json.loads((verifier_logs / "reward.json").read_text())
    verdict = json.loads((verifier_logs / "experiential.json").read_text())
    assert not (verifier_logs / "reward.txt").exists()
    return rewards, verdict, result, observed["verifier_stderr"]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_qualified_package_passes_local_harbor_trial_with_proven_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """register, qualify, export, verify, then scored pass, scored miss and unscored trials."""
    build, passing, missing, output_paths = KINDS[kind]
    project, task_ref, spec = build(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    scratch = tmp_path / "system-temp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    definition = load_executable_task(project, task_ref)
    registered = _invoke(
        project, tmp_path, ExecutableCommand(action="register", definition=definition)
    )
    assert ArtifactInput.model_validate_json(registered) == task_ref
    qualified = json.loads(
        _invoke(project, tmp_path, ExecutableCommand(action="qualify", qualification_spec=spec))
    )
    assert qualified["qualified"] is True, qualified["refusals"]
    qualification = ArtifactInput.model_validate(qualified["reference"])
    package = tmp_path / "packages" / kind
    _invoke(
        project,
        tmp_path,
        ExecutableCommand(
            action="export",
            task=task_ref,
            qualification=qualification,
            directory=package,
            name="experiential/" + kind,
        ),
    )
    assert verify_executable_package(project, package) == task_ref
    binding = verify_harbor_export(package)
    assert binding["qualification"] == "qualified_observable_contract"
    assert "python /tests/tools.py verify" in (package / "tests" / "test.sh").read_text()
    assert (package / "tests" / "tools.py").read_bytes() == (
        package / "environment" / "tools.py"
    ).read_bytes()
    assert 'environment_mode = "separate"' in (package / "task.toml").read_text()
    assert [path.name for path in package.parent.iterdir()] == [kind]
    assert not list(scratch.iterdir())

    rewards, verdict, result, _ = _run_trial(
        project,
        tmp_path,
        monkeypatch,
        package,
        trial_id="pass",
        actions=(passing,),
        submit=True,
        output_paths=output_paths,
    )
    assert rewards == {"reward": 1, "scored": 1}
    assert verdict["status"] == "completed" and verdict["reward"] == 1
    assert result == verdict

    rewards, verdict, result, _ = _run_trial(
        project,
        tmp_path,
        monkeypatch,
        package,
        trial_id="miss",
        actions=(missing,),
        submit=True,
        output_paths=output_paths,
    )
    assert rewards == {"reward": 0, "scored": 1}
    mechanical = verdict["mechanical"]
    assert isinstance(mechanical, dict) and isinstance(mechanical["goal"], dict)
    assert verdict["reward"] == 0 and mechanical["goal"]["status"] == "fail"
    assert result == verdict

    rewards, verdict, result, stderr = _run_trial(
        project,
        tmp_path,
        monkeypatch,
        package,
        trial_id="unsubmitted",
        actions=(),
        submit=False,
        output_paths=output_paths,
    )
    assert rewards == {"scored": 0}
    assert verdict["status"] == "incomplete" and verdict["reward"] is None
    assert "unscored" in stderr and result == verdict

    # Failures stay in the durable owner: every trial keeps its registration, and graded
    # submissions keep their immutable answer and result records.
    records = project.artifacts.list_ids(artifact_type="executable-trial")
    assert sum(key.endswith("-start") for key in records) == 3
    assert sum(key.endswith("-answer") for key in records) == 2
    assert sum(key.endswith("-result") for key in records) == 2
    assert not list(scratch.iterdir())

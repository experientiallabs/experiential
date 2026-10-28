"""CLI export and serving exercise real workbook operations through the Harbor bridge."""

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_studies import ContinuationExport
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.executable import ExecutableCommand
from exp.optimize.evaluation.continuations.admission import (
    ContinuationAdmission,
    admit_continuations,
)
from exp.optimize.evaluation.continuations.job import continuation_result_reference
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.trial_test import _post
from exp.optimize.model.sft.builder import load_verified_sft_dataset
from exp.optimize.model.sft.contracts import SFTBuildSpec
from exp.runtime.environments.tests.world_conformance_test import workbook_case


def test_cli_constructs_replayable_assets_and_rejects_bad_dry_run(tmp_path: Path) -> None:
    """The existing eval entrypoint publishes real assets without provider credentials."""
    from exp.common.core.artifacts import ArtifactInput
    from exp.common.project import ProjectStore
    from exp.simulation.world_construction import ConstructionSpec, load_construction
    from exp.simulation.world_construction_test import linked_spec

    project = ProjectStore(tmp_path, "constructed")
    path = tmp_path / "construct.json"
    spec = linked_spec()
    command = ContinuationCommand(
        action="harbor", harbor=ExecutableCommand(action="construct-assets", construction=spec)
    )
    path.write_text(command.model_dump_json())
    args = [
        "eval",
        "constructed",
        "--root",
        str(tmp_path),
        "--continuations",
        str(path),
        "--non-interactive",
    ]
    runner = CliRunner()
    preview = runner.invoke(app, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert project.artifacts.list_ids(artifact_type="world-construction") == ()
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    reply = json.loads(first.output)
    reference = ArtifactInput.model_validate(reply["reference"])
    assert load_construction(project.artifacts, reference) == spec
    assert reply["assets"] == ["assets/tables.json", "assets/orders.xlsx"]
    assert "native fidelity unestablished" in reply["scope"]
    replay = runner.invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.output) == reply
    raw = spec.model_dump(mode="json")
    raw["tables"]["Orders"]["columns"]["customer"]["values"] = [99]
    bad = ConstructionSpec.model_validate(raw)
    path.write_text(
        ContinuationCommand(
            action="harbor", harbor=ExecutableCommand(action="construct-assets", construction=bad)
        ).model_dump_json()
    )
    before = project.artifacts.list_ids()
    refused = runner.invoke(app, [*args, "--dry-run"])
    assert refused.exit_code != 0 and "missing lookup key" in refused.output
    assert project.artifacts.list_ids() == before


def test_cli_serves_expired_frozen_simulator_under_one_explicit_trial_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public CLI can reuse an old world without rebuying calls when a trial is resumed."""
    from exp.optimize.evaluation.executable.package import export_executable_harbor
    from exp.optimize.evaluation.executable.provider_grant import ProviderLimits
    from exp.optimize.evaluation.executable.provider_grant_test import expired_task

    project, task, _, transport = expired_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    package = export_executable_harbor(
        project, task, tmp_path / "package", name="test/granted", diagnostic=True
    )
    actor_file, verifier_file = tmp_path / "actor.json", tmp_path / "verifier.json"
    limits = ProviderLimits(
        maximum_physical_attempts=1, deadline=datetime.now(UTC) + timedelta(minutes=5)
    )
    command = ExecutableCommand(
        action="serve",
        directory=package,
        actor_connection_file=actor_file,
        verifier_connection_file=verifier_file,
        duration_seconds=1,
        provider_limits=limits,
    )
    path = tmp_path / "serve.json"
    path.write_text(ContinuationCommand(action="harbor", harbor=command).model_dump_json())
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(tmp_path),
        "--continuations",
        str(path),
        "--yes",
        "--non-interactive",
    ]

    class ExternalClient:
        def wait(self, seconds: int) -> None:
            """Make real HTTP calls during the server lifetime, including saved-reply replay."""
            assert seconds == 1
            actor = json.loads(actor_file.read_text())
            url = actor["EXP_HARBOR_ENDPOINT"]
            token = actor["EXP_HARBOR_ACTOR_TOKEN"]
            _post(url + "/call", token, workbook_case().first.model_dump(mode="json"))
            response = _post(
                url + "/call",
                token,
                {"name": "inspect", "call_id": "read", "arguments": {"key": "output.xlsx"}},
            )
            assert response["content"] == "predicted reply"
            _post(url + "/submit", token, {"answer": "done"})

    monkeypatch.setattr("exp.cli.evaluation.executable.Event", ExternalClient)
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Simulator allowance: 1 physical" in result.output
    assert len(transport.requests) == 1
    changed = command.model_copy(
        update={
            "provider_limits": limits.model_copy(
                update={"deadline": limits.deadline + timedelta(minutes=1)}
            )
        }
    )
    path.write_text(ContinuationCommand(action="harbor", harbor=changed).model_dump_json())
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and "authority changed" in result.output
    assert len(transport.requests) == 1


def test_cli_exports_and_serves_qualified_workbook_without_leaking_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the verifier can grade; runtime connection files are private and removed on exit."""
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    monkeypatch.setenv("EXP_RELEASE_REVISION", "c" * 40)
    path = tmp_path / "harbor.json"
    package = tmp_path / "package"
    command = ExecutableCommand(
        action="export",
        task=task,
        qualification=qualification,
        directory=package,
        name="test/workbook",
    )
    path.write_text(ContinuationCommand(action="harbor", harbor=command).model_dump_json())
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--continuations",
        str(path),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    exported = runner.invoke(app, args)
    assert exported.exit_code == 0, exported.output
    actor_file, verifier_file = tmp_path / "actor.json", tmp_path / "verifier.json"
    command = ExecutableCommand(
        action="serve",
        directory=package,
        actor_connection_file=actor_file,
        verifier_connection_file=verifier_file,
        duration_seconds=1,
    )
    path.write_text(ContinuationCommand(action="harbor", harbor=command).model_dump_json())
    responses = []

    class ExternalHarborClient:
        """Use the published capabilities as a separate Harbor runner would during the lease."""

        def wait(self, seconds: int) -> None:
            """Operate the real world while the bounded server is live, without sleeping."""
            assert seconds == 1
            assert stat.S_IMODE(actor_file.stat().st_mode) == 0o600
            actor = json.loads(actor_file.read_text())
            verifier = json.loads(verifier_file.read_text())
            assert "EXP_HARBOR_VERIFIER_TOKEN" not in actor
            assert "EXP_HARBOR_ACTOR_TOKEN" not in verifier
            url = actor["EXP_HARBOR_ENDPOINT"]
            _post(
                url + "/call",
                actor["EXP_HARBOR_ACTOR_TOKEN"],
                workbook_case().first.model_dump(mode="json"),
            )
            _post(url + "/submit", actor["EXP_HARBOR_ACTOR_TOKEN"], {"answer": "done"})
            responses.append(_post(url + "/grade", verifier["EXP_HARBOR_VERIFIER_TOKEN"], {}))

    monkeypatch.setattr("exp.cli.evaluation.executable.Event", ExternalHarborClient)
    served = runner.invoke(app, args)
    assert served.exit_code == 0, served.output
    assert responses[0]["reward"] == 1
    assert not actor_file.exists() and not verifier_file.exists()
    assert "Bearer" not in served.output


def test_executable_command_rejects_ignored_or_conflicting_payloads(tmp_path: Path) -> None:
    """A caller cannot think a diagnostic flag or supplied qualification changed another action."""
    from exp.common.core.artifacts import ArtifactInput

    reference = ArtifactInput(artifact_id="task", sha256="a" * 64)
    with pytest.raises(ValueError, match="does not accept diagnostic"):
        ExecutableCommand(
            action="serve",
            directory=tmp_path,
            actor_connection_file=tmp_path / "actor.json",
            verifier_connection_file=tmp_path / "verifier.json",
            diagnostic=True,
        )
    with pytest.raises(ValueError, match="does not accept actor_connection_file"):
        ExecutableCommand(
            action="export",
            task=reference,
            directory=tmp_path,
            actor_connection_file=tmp_path / "actor.json",
        )
    with pytest.raises(ValueError, match="requires"):
        ExecutableCommand(action="qualify")


@pytest.mark.parametrize("dry_run", [True, False])
def test_serve_preflight_does_not_execute_witnesses_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Readiness checks and denied consent cannot execute trusted native witness code."""
    from io import StringIO

    from rich.console import Console

    from exp.cli.evaluation.executable import run_executable_command
    from exp.optimize.evaluation.executable.package import export_executable_harbor

    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    package = export_executable_harbor(
        project, task, tmp_path / "package", name="test/preflight", qualification=qualification
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight executed native witnesses or opened a runtime")

    monkeypatch.setattr("exp.cli.evaluation.executable.verify_executable_package", forbidden)
    monkeypatch.setattr("exp.cli.evaluation.executable.open_executable_harbor_trial", forbidden)
    monkeypatch.setattr(
        "exp.cli.evaluation.executable.require_spend_consent", lambda *a, **kw: False
    )
    output = StringIO()
    run_executable_command(
        project,
        ExecutableCommand(
            action="serve",
            directory=package,
            actor_connection_file=tmp_path / "actor.json",
            verifier_connection_file=tmp_path / "verifier.json",
        ),
        console=Console(file=output),
        dry_run=dry_run,
        yes=False,
        non_interactive=True,
        code_revision="test",
    )
    if dry_run:
        assert "Full qualification is rechecked" in output.getvalue()
    assert not (tmp_path / "actor.json").exists()


@pytest.mark.parametrize("action", ["register", "export", "qualify"])
def test_dry_run_checks_references_and_runtime_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Invalid handoffs fail during readiness checks, not after a misleading dry-run success."""
    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.executable.task import load_executable_task

    project, reference, spec = qualified_fixture(tmp_path, monkeypatch)
    bad = ArtifactInput(artifact_id=reference.artifact_id, sha256="f" * 64)
    if action == "register":
        definition = load_executable_task(project, reference)
        definition.job.runtime_file.write_text("raise RuntimeError('MUST NOT IMPORT')\n")
        command = ExecutableCommand(action="register", definition=definition)
    elif action == "export":
        command = ExecutableCommand(
            action="export", task=bad, directory=tmp_path / "package", diagnostic=True
        )
    else:
        command = ExecutableCommand(
            action="qualify", qualification_spec=spec.model_copy(update={"task": bad})
        )
    path = tmp_path / "readiness.json"
    path.write_text(ContinuationCommand(action="harbor", harbor=command).model_dump_json())
    before = set(project.artifacts.list_ids())
    result = CliRunner().invoke(
        app,
        [
            "eval",
            project.paths.project_id,
            "--root",
            str(project.paths.root),
            "--continuations",
            str(path),
            "--dry-run",
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0, result.output
    assert "MUST NOT IMPORT" not in result.output
    assert set(project.artifacts.list_ids()) == before
    assert not (tmp_path / "package").exists()


def test_cli_prepares_and_executes_qualified_training_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI handoff produces a runnable frozen job with no calls during preparation."""

    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    _, _, transport = job_fixture(tmp_path / "transport", monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "prepare.json"
    operation_file = tmp_path / "generation.json"
    path.write_text(
        ContinuationCommand(
            action="harbor",
            harbor=ExecutableCommand(
                action="prepare-training",
                qualification=qualification,
                operation_file=operation_file,
            ),
        ).model_dump_json()
    )
    runner = CliRunner()
    args = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(path),
        "--yes",
        "--non-interactive",
    ]
    preview = runner.invoke(app, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert not operation_file.exists() and not transport.requests
    prepared = runner.invoke(app, args)
    assert prepared.exit_code == 0, prepared.output
    operation = ContinuationCommand.model_validate_json(operation_file.read_text())
    assert operation.action == "run" and operation.job is not None
    assert not transport.requests
    assert runner.invoke(app, args).exit_code != 0
    assert ContinuationCommand.model_validate_json(operation_file.read_text()) == operation
    args[args.index(str(path))] = str(operation_file)
    execution = runner.invoke(app, args)
    assert execution.exit_code == 0, execution.output
    assert len(transport.requests) == 3
    reference = continuation_result_reference(project, operation.job, code_revision="test")
    admitted = admit_continuations(
        project.artifacts, reference, created_at=datetime.now(UTC), code_revision="test"
    )
    report = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(admitted.artifact_id, "admission.json")
    )
    assert report.evidence is not None and report.evidence.qualification == qualification
    operation.job.runtime_file.unlink()
    operation_file.write_text(
        ContinuationCommand(
            action="export",
            export=ContinuationExport(
                studies=(reference,), dataset=SFTBuildSpec(held_out_fraction=0)
            ),
        ).model_dump_json()
    )
    exported = runner.invoke(app, args)
    assert exported.exit_code == 0, exported.output
    assert "1/2 branches" in exported.output and "Assistant targets: 1" in exported.output
    assert report.evidence.qualification_scope[:45] in exported.output
    dataset_id = project.artifacts.list_ids(artifact_type="sft-dataset")[0]
    dataset = load_verified_sft_dataset(project, dataset_id)
    assert len(dataset.rows) == 1 and len(dataset.sources) == 2
    assert all(source.acceptance_evidence == admitted for source in dataset.sources)
    assert len(transport.requests) == 3


def test_cli_authors_worlds_with_consent_then_replays_without_provider_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public command buys one typed draw and exercises actual native world challenges."""
    from exp.optimize.evaluation.factory.worlds import WorldProposals
    from exp.optimize.evaluation.factory.worlds_test import world_fixture

    project, spec, transport = world_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    operation = tmp_path / "propose-worlds.json"
    operation.write_text(
        ContinuationCommand(
            action="harbor",
            harbor=ExecutableCommand(
                action="propose-worlds",
                world_proposals=spec,
            ),
        ).model_dump_json()
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
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    refused = runner.invoke(app, args)
    assert refused.exit_code != 0 and not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")
    executed = runner.invoke(app, [*args, "--yes"])
    assert executed.exit_code == 0, executed.output
    report = WorldProposals.model_validate_json(executed.output[executed.output.index("{") :])
    assert report.search is not None and report.search.cells[1].variant is not None
    assert len(transport.requests) == 1
    project.model_catalog_path.unlink()
    spec.template.capture.runtime_file.unlink()
    replayed = runner.invoke(app, args)
    assert replayed.exit_code == 0, replayed.output
    assert WorldProposals.model_validate_json(replayed.output) == report
    assert len(transport.requests) == 1


def test_serve_spend_prompt_names_the_spec_file_the_user_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent identifies the exact `--continuations` file, not an invented spec name."""
    from exp.optimize.evaluation.executable.package import export_executable_harbor

    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    package = export_executable_harbor(
        project, task, tmp_path / "package", name="test/consent", qualification=qualification
    )
    commands: list[str] = []

    def decline(*args: object, command: str, **kwargs: object) -> bool:
        """Record the command shown to the operator and decline the spend.

        Args:
            *args: Positional consent arguments, ignored.
            command: Command identity shown in the spend prompt.
            **kwargs: Remaining consent arguments, ignored.

        Returns:
            False, so serving never starts.
        """
        commands.append(command)
        return False

    monkeypatch.setattr("exp.cli.evaluation.executable.require_spend_consent", decline)
    path = tmp_path / "my-serve-request.json"
    path.write_text(
        ContinuationCommand(
            action="harbor",
            harbor=ExecutableCommand(
                action="serve",
                directory=package,
                actor_connection_file=tmp_path / "actor.json",
                verifier_connection_file=tmp_path / "verifier.json",
            ),
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            project.paths.project_id,
            "--root",
            str(project.paths.root),
            "--continuations",
            str(path),
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert commands == [f"exp eval {project.paths.project_id} --continuations {path}"]
    assert not (tmp_path / "actor.json").exists()

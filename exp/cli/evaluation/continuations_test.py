"""The public continuation CLI executes and exports the same native workbook evidence."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuation_studies import ContinuationExport
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.project import artifact_input
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.observation_contrast_test import observation_fixture
from exp.optimize.evaluation.equivalence.policy import policy_identity
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy
from exp.optimize.model.sft.builder import (
    SFTBuildError,
    load_sft_dataset,
    load_verified_sft_dataset,
)
from exp.optimize.model.sft.contracts import SFTBuildSpec


def test_cli_compares_saved_observations_and_replays_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public operation validates dry, executes both arms and retains failed draws on replay."""
    project, spec, transport = observation_fixture(tmp_path, monkeypatch)
    request = tmp_path / "observations.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="compare-observations", observations=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests and not project.artifacts.list_ids(
        artifact_type="observation-contrast"
    )
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"unknown": 1' in result.output and "authored alternative is unverified" in result.output
    count = len(transport.requests)
    (tmp_path / "selected_runtime.py").unlink()
    project.model_catalog_path.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]
    assert len(transport.requests) == count


def test_cli_captures_builtin_workbook_without_custom_factory_or_credentials(
    tmp_path: Path,
) -> None:
    """A package-owned native adapter uses the same capture operation as other worlds."""
    from exp.common.core.artifacts import sha256_bytes
    from exp.optimize.evaluation.continuations.capture import CaptureDefinition
    from exp.optimize.evaluation.executable import workbook as workbook_runtime
    from exp.optimize.evaluation.executable.workbook_test import workbook_definition

    project, spec, task = workbook_definition(tmp_path)
    path = Path(workbook_runtime.__file__)
    request = tmp_path / "capture-native.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND,
            action="capture",
            capture=CaptureDefinition(
                task=task,
                sources=(spec.input_artifact,),
                runtime_file=path,
                runtime_sha256=sha256_bytes(path.read_bytes()),
                configuration=spec.model_dump(mode="json"),
                probe_sha256=spec.probe_sha256,
            ),
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.output
    assert len(project.artifacts.list_ids(artifact_type="continuation-checkpoint")) == 1
    assert not project.model_catalog_path.exists()


def test_cli_localizes_saved_failures_without_runtime_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis retains both native outcomes and replays after the runtime and catalog disappear."""
    from exp.optimize.evaluation.continuations.job import (
        continuation_result_reference,
        execute_continuation_job,
    )

    project, job, transport = job_fixture(tmp_path, monkeypatch)
    execute_continuation_job(project, job, code_revision="test")
    reference = continuation_result_reference(project, job, code_revision="test")
    job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    request = tmp_path / "analysis.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="analyse-failures", analysis=reference
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="failure-analysis")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"goal_failed"' in result.output and '"training_admitted": false' in result.output
    assert len(project.artifacts.list_ids(artifact_type="failure-analysis")) == 1
    assert runner.invoke(app, command).exit_code == 0
    assert len(transport.requests) == 3


def test_cli_captures_configured_native_and_learned_world_without_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One configuration reaches both runtime owners through the existing public capture action."""
    from exp.common.core.artifacts import sha256_bytes
    from exp.optimize.evaluation.continuations.capture import CaptureDefinition
    from exp.optimize.evaluation.executable import mixed as mixed_runtime
    from exp.optimize.evaluation.executable.mixed_test import mixed_fixture

    project, spec, image, transport = mixed_fixture(tmp_path, monkeypatch)
    factory = Path(mixed_runtime.__file__)
    capture = CaptureDefinition(
        task=spec.simulated.task,
        sources=(spec.simulated.world_model, image),
        runtime_file=factory,
        runtime_sha256=sha256_bytes(factory.read_bytes()),
        configuration=spec.model_dump(mode="json"),
        probe_sha256=spec.probe_sha256,
    )
    request = tmp_path / "mixed-capture.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="capture", capture=capture
        ).model_dump_json()
    )
    command = [
        "eval",
        project.paths.project_id,
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    assert runner.invoke(app, [*command, "--dry-run"]).exit_code == 0
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert not transport.requests
    spec.native.runtime_file.unlink()
    assert runner.invoke(app, command).exit_code == 0
    assert not transport.requests


def test_cli_registers_a_brief_without_hiding_unresolved_assumptions(tmp_path: Path) -> None:
    """Registration is a source-linked research record, never approval of its assumptions."""
    from exp.common.project import ProjectStore
    from exp.simulation.capability_briefs_test import brief

    project = ProjectStore(tmp_path, "case")
    request = tmp_path / "brief.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="register-brief", brief=brief()
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="capability-brief")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert "unresolved assumption" in result.output
    assert len(project.artifacts.list_ids(artifact_type="capability-brief")) == 1
    assert runner.invoke(app, command).exit_code == 0
    assert len(project.artifacts.list_ids(artifact_type="capability-brief")) == 1


def test_cli_evaluates_and_packages_every_saved_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public batch command links native feasibility, independent controls and actor panels."""
    from exp.optimize.evaluation.factory.candidates_test import batch_fixture

    project, spec, transport = batch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    request = tmp_path / "batch.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="evaluate-candidates", batch=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests and spec.destination is not None and not spec.destination.exists()
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"packaged": 2' in result.output and len(transport.requests) == 12
    assert project.artifacts.list_ids(artifact_type="candidate-batch")
    assert runner.invoke(app, command).exit_code == 0
    assert len(transport.requests) == 12


def test_cli_proposes_multiple_tasks_without_inheriting_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One paid logical proposal produces every candidate and can replay without the catalog."""
    from exp.optimize.evaluation.factory.proposals_test import proposal_fixture

    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    request = tmp_path / "proposals.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="propose-tasks", proposals=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert len(project.artifacts.list_ids(artifact_type="task-variant")) == 2
    assert not project.artifacts.list_ids(artifact_type="executable-qualification")
    project.model_catalog_path.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 1


def test_cli_compares_actions_then_replays_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public CLI preserves preregistration and all four native treatment measurements."""
    from exp.optimize.evaluation.continuations.action_contrast_test import contrast_fixture

    project, spec, transport = contrast_fixture(tmp_path, monkeypatch)
    request = tmp_path / "contrast.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="compare-actions", contrast=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    assert not project.artifacts.list_ids(artifact_type="action-contrast-registration")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"alternative_minus_baseline_bounds"' in result.output
    assert len(transport.requests) == 4
    spec.base.runtime_file.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 4


def test_cli_run_and_export_with_private_native_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI run, saved replay and witnessed export retain the planned branch denominator."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("EXP_RELEASE_REVISION", "c" * 40)
    request = tmp_path / "continuations.json"
    request.write_text(
        ContinuationCommand(spend=UNPRICED_SPEND, action="run", job=job).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    # The admission policy is registered through the CLI before any result it judges.
    policy = fixture_policy(margin=0.6)
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="register-admission-policy", admission_policy=policy
        ).model_dump_json()
    )
    preview = runner.invoke(app, [*command, "--dry-run"])
    assert preview.exit_code == 0 and "Dry run: not registered" in " ".join(preview.output.split())
    assert not project.artifacts.exists(policy_identity(policy))
    registered = runner.invoke(app, command)
    assert registered.exit_code == 0, registered.output
    policy_ref = artifact_input(project.artifacts.read(policy_identity(policy)).manifest)
    assert policy_ref.artifact_id in registered.output
    # The study pins the registered policy before it runs, so export can prove the order.
    job = job.model_copy(
        update={"study": job.study.model_copy(update={"admission_policy": policy_ref})}
    )
    request.write_text(
        ContinuationCommand(spend=UNPRICED_SPEND, action="run", job=job).model_dump_json()
    )
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    run = runner.invoke(app, command)
    assert run.exit_code == 0, run.output
    assert '"successes": 1' in run.output
    assert len(transport.requests) == 3
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0 and len(transport.requests) == 3
    results = [
        key
        for key in project.artifacts.list_ids(artifact_type="continuation-study")
        if key.endswith("-result")
    ]
    study = next(
        key for key in results if project.artifacts.read(key).manifest.code_revision != "test"
    )
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND,
            action="export",
            export=ContinuationExport(
                studies=(artifact_input(project.artifacts.read(study).manifest),),
                policy=policy_ref,
                dataset=SFTBuildSpec(held_out_fraction=0),
            ),
        ).model_dump_json()
    )
    exported = runner.invoke(app, command)
    assert exported.exit_code == 0, exported.output
    assert "1/2 branches" in exported.output and "Assistant targets: 1" in exported.output
    assert f"Admission policy {policy_ref.artifact_id}: 1 admitted, 1 refused" in exported.output
    assert len(transport.requests) == 3
    dataset = project.artifacts.list_ids(artifact_type="sft-dataset")[0]
    assert "(diagnostic)" in exported.output
    inspected = load_sft_dataset(project, dataset, require_accepted=False)
    assert len(inspected.rows) == 1
    with pytest.raises(SFTBuildError, match="diagnostic"):
        load_verified_sft_dataset(project, dataset)


def test_cli_captures_initial_world_without_actor_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public CLI creates the same native checkpoint consumed by subsequent run jobs."""
    import json

    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.continuations.capture_test import capture_fixture
    from exp.simulation.continuations import load_continuation_checkpoint

    project, definition, transport = capture_fixture(tmp_path, monkeypatch)
    request = tmp_path / "capture.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="capture", capture=definition
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="continuation-capture")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    # Consent can print before the JSON object; parse the returned artifact reference only.
    reference = ArtifactInput.model_validate(json.loads(result.output[result.output.index("{") :]))
    saved = load_continuation_checkpoint(project.artifacts, reference)
    assert saved.actor.calls == 0
    assert saved.task == definition.task
    assert not transport.requests


def test_cli_compares_actor_snapshots_without_hiding_unavailable_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor-panel CLI uses the same frozen studies and refuses an assumed tier ranking."""
    from exp.optimize.evaluation.continuations.actors_test import panel_fixture

    _, spec, transport = panel_fixture(tmp_path, monkeypatch)
    request = tmp_path / "actors.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="compare-actors", actors=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    run = runner.invoke(app, command)
    assert run.exit_code == 0, run.output
    assert '"separates_on_fixed_panel": false' in run.output
    assert '"unknown": 2' in run.output


def test_cli_records_a_prospective_critic_without_exposing_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI returns a durable estimate reference for a later continuation registration."""
    from exp.optimize.evaluation.continuations.critic_test import critic_fixture

    _, _, spec, transport = critic_fixture(tmp_path, monkeypatch)
    request = tmp_path / "critic.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="predict-success", critic=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"predicted_success": 0.75' in result.output
    assert '"reference"' in result.output
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 1


def test_cli_matches_checkpoints_then_runs_and_replays_registered_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full control comparison is available through the CLI, with credential-free replay."""
    import json

    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.continuations.experiment import CheckpointExperimentSpec
    from exp.optimize.evaluation.continuations.selection_test import selection_fixture

    _, selection = selection_fixture(tmp_path, monkeypatch)
    request = tmp_path / "selection.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="match-checkpoints", selection=selection
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    reference = ArtifactInput.model_validate(json.loads(result.output)["reference"])
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND,
            action="run-selection",
            experiment=CheckpointExperimentSpec(selection=reference),
        ).model_dump_json()
    )
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    run = runner.invoke(app, command)
    assert run.exit_code == 0, run.output
    assert '"planned_branches": 2' in run.output and '"unknown": 2' in run.output
    selection.base.runtime_file.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="match-checkpoints", selection=selection
        ).model_dump_json()
    )
    assert runner.invoke(app, command).exit_code == 0
    with pytest.raises(ValueError, match="requires only"):
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="run", job=selection.base, selection=selection
        )


def test_cli_action_intervention_changes_native_state_without_actor_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI publishes the imposed-effect capsule and replays it without runtime code."""
    from exp.optimize.evaluation.continuations.intervention import ActionInterventionSpec
    from exp.runtime.environments.tests.world_conformance_test import workbook_case

    _, job, transport = job_fixture(tmp_path, monkeypatch)
    spec = ActionInterventionSpec(
        base=job, action=workbook_case().first, rationale="One-row contrast."
    )
    request = tmp_path / "intervention.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="intervene", intervention=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"rows": 6' in result.output and "scripted action" in result.output
    assert not transport.requests
    job.runtime_file.unlink()
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output


def test_cli_publishes_a_variant_without_promoting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task families flow into executable qualification through the same public CLI."""
    import json

    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.factory.variants import load_task_variant
    from exp.optimize.evaluation.factory.variants_test import variant_fixture

    project, spec = variant_fixture(tmp_path, monkeypatch)
    request = tmp_path / "variant.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="make-variant", variant=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="task-variant")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    ref = ArtifactInput.model_validate(json.loads(result.output)["reference"])
    variant = load_task_variant(project, ref)
    assert variant.spec == spec and "candidate" in variant.scope
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0 and replay.output == result.output


def test_cli_measures_saved_decisions_without_provider_setup(tmp_path: Path) -> None:
    """The public CLI preserves failed slots and performs no credential or runtime setup."""
    from exp.optimize.evaluation.analysis.decision_calibration_test import study

    store, spec = study(tmp_path)
    before = store.list_ids()
    request = tmp_path / "decisions.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="measure-decisions", decisions=spec
        ).model_dump_json()
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "decisions",
            "--root",
            str(tmp_path),
            "--continuations",
            str(request),
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"confirmation_supported": false' in result.output and '"planned": 3' in result.output
    assert store.list_ids() == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_missing_continuation_artifact_is_a_readable_usage_error(
    tmp_path: Path, dry_run: bool
) -> None:
    """An invalid input reference is refused before inference, with no internal traceback."""
    request = tmp_path / "missing.json"
    request.write_text(
        '{"action":"analyse-failures","analysis":{"artifact_id":"missing-study","sha256":"'
        + "a" * 64
        + '"}}'
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "case",
            "--root",
            str(tmp_path),
            "--continuations",
            str(request),
            "--non-interactive",
            *(["--dry-run"] if dry_run else []),
        ],
    )
    assert result.exit_code == 2
    assert "missing-study" in result.output and "missing" in result.output
    assert "Traceback" not in result.output


def test_registry_is_the_one_list_of_actions_and_payloads() -> None:
    """Each action has exactly one payload field, and every payload field has one action.

    ``spend`` is the one shared non-payload field: the operation's spend-ceiling choices.
    """
    from typing import get_args

    from exp.cli.evaluation.continuations import CONTINUATION_ACTIONS

    names = [action.name for action in CONTINUATION_ACTIONS]
    payloads = [action.payload for action in CONTINUATION_ACTIONS]
    assert len(set(names)) == len(names) == 57
    assert len(set(payloads)) == len(payloads)
    assert set(payloads) == set(ContinuationCommand.model_fields) - {"action", "spend"}
    assert get_args(ContinuationCommand.model_fields["action"].annotation) == tuple(names)
    with pytest.raises(ValueError, match="harbor requires only its harbor payload"):
        ContinuationCommand.model_validate({"action": "harbor"})


def test_registry_runs_generic_actions_without_the_artifact_edits_extra(tmp_path: Path) -> None:
    """A fresh interpreter without openpyxl imports every action and refuses only workbooks."""
    import json
    import os
    import subprocess
    import sys

    from exp.simulation.capability_briefs_test import brief

    generic = tmp_path / "brief.json"
    generic.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="register-brief", brief=brief()
        ).model_dump_json()
    )
    workbook = tmp_path / "workbook.json"
    workbook.write_text(
        json.dumps(
            {
                "action": "native-workbook",
                "spend": UNPRICED_SPEND.model_dump(mode="json"),
                "native_workbook": {
                    "task_id": "fit-0",
                    "files": {"input.xlsx": str(tmp_path / "input.xlsx")},
                    "maximum_model_calls": 1,
                },
            }
        )
    )
    script = """
import importlib.abc,sys
class NoWorkbook(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        top=fullname.split('.')[0]
        if top in ('openpyxl','et_xmlfile'):
            raise ModuleNotFoundError('optional workbook dependency unavailable',name=top)
sys.meta_path.insert(0,NoWorkbook())
from pydantic import TypeAdapter
from typer.testing import CliRunner
from exp.cli.app import app
from exp.cli.evaluation.continuations import CONTINUATION_ACTIONS
from exp.runtime.environments.facts_derivers import FactsDeriverDefinition
root,generic,workbook=sys.argv[1:]
assert {'native-workbook','derive-facts'} <= {action.name for action in CONTINUATION_ACTIONS}
def run(spec):
    return CliRunner().invoke(
        app,['eval','case','--root',root,'--continuations',spec,'--non-interactive']
    )
result=run(generic)
assert result.exit_code==0,(result.output,result.exception)
result=run(workbook)
assert result.exit_code==2,(result.output,result.exception)
assert 'install experiential[artifact-edits]' in result.output,result.output
deriver=TypeAdapter(FactsDeriverDefinition).validate_python({'kind':'xlsx-workbook'})
assert 'openpyxl' not in sys.modules
try:
    deriver.derive('input.xlsx',b'')
except ValueError as error:
    assert 'install experiential[artifact-edits]' in str(error),error
else:
    raise AssertionError('a selected workbook deriver must still require its optional dependency')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    environment["COLUMNS"] = "400"
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path / "root"), str(generic), str(workbook)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr

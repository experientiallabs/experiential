"""Credential-free calibration of frozen task verifiers and saved expert judgments."""

import json
from pathlib import Path

from rich.console import Console

from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.tasks.calibration import CalibrationSpec, execute_calibration


def run_calibration(project: str, spec_path: Path, *, root: Path, dry_run: bool) -> None:
    """Validate or execute one saved-evidence calibration with no hosted calls.

    Args:
        project: Existing frozen simulator project.
        spec_path: Case pack, exact task and saved judge references.
        root: Existing artifact root.
        dry_run: Validate without publishing a result.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        spec = CalibrationSpec.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        reference = execute_calibration(
            store, spec, code_revision=installed_release_revision(), dry_run=dry_run
        )
        if reference is None:
            console.print("Calibration preflight passed. No writes or provider calls.")
            return
        result = json.loads(store.artifacts.read_bytes(reference.artifact_id, "report.json"))
        console.print(f"Calibration: {reference.artifact_id}")
        console.print(f"Cases: {result['planned']} · admitted labels: {result['independent']}")
        console.print(
            f"Mechanical false accepts: {result['false_acceptances']} · "
            f"false rejects: {result['false_rejections']}"
        )
        console.print(
            f"Task false accepts: {result['task_false_acceptances']} · "
            f"false rejects: {result['task_false_rejections']}"
        )
        console.print(f"Judge criterion disagreements: {result['judge_criterion_disagreements']}")
        console.print(
            f"Task decisions: {result['task_assessed']}/{result['task_planned']} assessed · "
            f"unassessable: {result['task_unassessable']} · "
            f"withheld: {result['task_withheld']}"
        )
        console.print(
            f"Development: {result['development_status']} · "
            f"selection: {result['selection_status']} · "
            f"confirmation: {result['confirmation_status']}"
        )
        console.print(
            f"Rubric: {result['rubric_status']} · qualification: {result['qualification']}"
        )
        console.print(
            f"Label sources: {result['label_authorities']} · groups: {result['lineage_groups']}"
        )
        console.print(f"Witnessed labels verified: {result['witnessed_verified']}")
        for refusal in result["qualification_refusals"]:
            console.print(
                f"Not qualified [{refusal['reason']}] {refusal['subject']}: {refusal['detail']}",
                markup=False,
            )
        if spec.harbor is not None:
            console.print(f"Harbor package: {spec.harbor.destination}")
            console.print("Package export is separate from a completed Harbor execution trial.")
        console.print("Saved evidence only. No provider calls or actor feedback.")

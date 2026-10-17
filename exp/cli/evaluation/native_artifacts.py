"""Provider-free CLI for explicit native artifact transformations."""

import json
from pathlib import Path

from rich.console import Console

from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.native_artifacts import (
    NativeMaterializationSpec,
    materialize_native_artifacts,
)
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter


def run_materialization(project: str, spec_path: Path, *, root: Path, dry_run: bool) -> None:
    """Execute a frozen workbook handoff, retaining divergence rather than repairing predictions."""
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        spec = NativeMaterializationSpec.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        reference = materialize_native_artifacts(
            store.artifacts,
            spec,
            WorkbookEditAdapter(),
            code_revision=installed_release_revision(),
            dry_run=dry_run,
        )
        if reference is None:
            console.print("Frozen bindings validated. Native edits were not executed.")
            return
        receipt = json.loads(store.artifacts.read_bytes(reference.artifact_id, "receipt.json"))
        console.print(f"Native artifacts: {reference.artifact_id}")
        console.print(f"Candidate/native consistency: {receipt['status']}")
        console.print("Diagnostic artifact execution. No provider calls or qualification claim.")
        if spec.destination is not None:
            console.print(f"Output files: {spec.destination}")

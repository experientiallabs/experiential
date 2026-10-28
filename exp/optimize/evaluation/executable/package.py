"""Harbor packages for executable tasks, reproduced from trusted artifacts before use."""

import tempfile
import tomllib
from pathlib import Path

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.qualification import load_executable_qualification
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.harbor.export import (
    preflight_harbor_destination,
    verify_harbor_export,
    write_harbor_package,
)
from exp.simulation.continuations import load_continuation_checkpoint


def export_executable_harbor(
    project: ProjectStore,
    task_ref: ArtifactInput,
    destination: Path,
    *,
    name: str,
    qualification: ArtifactInput | None = None,
    diagnostic: bool = False,
) -> Path:
    """Use the common Harbor package with a private executable checkpoint backend.

    Args:
        project: Project holding the frozen task, checkpoint and qualification.
        task_ref: Exact executable task to package.
        destination: New package directory; an existing path is refused.
        name: Safe organization/name pair written into ``task.toml``.
        qualification: Qualification report for this exact task, when qualified.
        diagnostic: Allow an unqualified, explicitly diagnostic package.

    Returns:
        The published package directory.
    """
    preflight_harbor_destination(destination, name=name)
    task = load_executable_task(project, task_ref)
    checkpoint = load_continuation_checkpoint(project.artifacts, task.job.study.checkpoint)
    qualified = False
    if qualification is not None:
        report = load_executable_qualification(project, qualification)
        if report.task != task_ref:
            raise ValueError("qualification belongs to another executable task")
        qualified = report.qualified
    if not qualified and not diagnostic:
        raise ValueError("observable contract is unqualified; use an explicit diagnostic export")
    return write_harbor_package(
        destination,
        name=name,
        instruction=checkpoint.task.instruction,
        public_context={
            "initial_context": checkpoint.task.initial_context,
            "actor_prefix": [
                message.model_dump(mode="json") for message in checkpoint.actor.messages
            ],
            "known_limits": list(task.known_limits),
        },
        binding={
            "backend": "executable-checkpoint",
            "task": task_ref.model_dump(mode="json"),
            "checkpoint": task.job.study.checkpoint.model_dump(mode="json"),
            "profile_sha256": task.profile(),
            "qualification": "qualified_observable_contract" if qualified else "diagnostic_only",
            "qualification_evidence": None
            if qualification is None
            else qualification.model_dump(mode="json"),
            "native_environment_accessed": False,
        },
        qualified=qualified,
        output_paths=task.output_paths,
        transport=task.harbor_transport,
        notice="Tools run in the declared executable or mixed world from a frozen checkpoint.",
    )


def verify_executable_package(project: ProjectStore, directory: Path) -> ArtifactInput:
    """Reproduce the package from trusted artifacts before importing its runtime.

    The local binding JSON is not a trust root. Both its references and all public files must
    reproduce the same export, so editing a binding cannot attach different grading or state.

    Args:
        project: Project holding the trusted task and qualification artifacts.
        directory: Package directory to verify.

    Returns:
        The executable task the package reproduces.
    """
    binding = verify_harbor_export(directory)
    if binding.get("backend") != "executable-checkpoint":
        raise ValueError("package does not name the executable checkpoint backend")
    task = ArtifactInput.model_validate(binding.get("task"))
    raw = binding.get("qualification_evidence")
    qualification = None if raw is None else ArtifactInput.model_validate(raw)
    config = tomllib.loads((directory / "task.toml").read_text())
    name = config["task"]["name"]
    with tempfile.TemporaryDirectory(prefix="exp-verify-package-") as temporary:
        expected = export_executable_harbor(
            project,
            task,
            Path(temporary) / "package",
            name=name,
            qualification=qualification,
            diagnostic=binding.get("qualification") == "diagnostic_only",
        )
        if verify_harbor_export(expected) != binding:
            raise ValueError("package does not reproduce its trusted executable task profile")
    return task

"""Harbor publication and bounded local serving of frozen executable task profiles."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Literal

from pydantic import Field, model_validator
from rich.console import Console

from exp.cli.shared.consent import require_spend_consent
from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuation_evidence import qualified_training_job
from exp.optimize.evaluation.executable_qualification import (
    ExecutableQualificationSpec,
    load_executable_qualification,
    preflight_executable_qualification,
    preflight_qualification_reference,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable_task import (
    ExecutableTask,
    export_executable_harbor,
    load_executable_task,
    persist_executable_task,
    preflight_executable_task,
    verify_executable_package,
)
from exp.optimize.evaluation.executable_trial import open_executable_harbor_trial
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.optimize.evaluation.harbor.export import preflight_harbor_destination, verify_harbor_export
from exp.optimize.evaluation.provider_grant import ProviderLimits
from exp.optimize.evaluation.world_proposals import (
    WorldProposalSpec,
    preflight_world_proposals,
    propose_worlds,
    world_proposal_request,
)
from exp.simulation.world_construction import (
    ConstructionSpec,
    persist_construction,
    preflight_construction,
)


class ExecutableCommand(ContractModel):
    """A single frozen executable task operation through the existing eval command.

    Attributes:
        action: Construct assets, register, qualify, prepare training, export, or serve.
        world_proposals: Cited author experiment feeding frozen worlds into native challenges.
        construction: Bounded declarative initial assets, with unqualified generated authority.
        definition: Task definition for registration only.
        task: Exact registered task for export.
        qualification_spec: Independent control evidence for qualification only.
        qualification: Saved scoped qualification for export, absent for diagnostics.
        comparison: Optional saved paired report for prepare-training, retained as measurements.
        comparison_case: Exact applicable candidate case in that report.
        operation_file: Optional new JSON file for the prepared continuation command.
            Existing files are never overwritten; dry-run does not create it.
        directory: New package path for export or immutable package to serve.
        name: Organization/name used for export.
        diagnostic: Explicit opt-in to a non-reward package.
        trial_id: Caller-selected identity; exact re-entry restores committed state.
        actor_connection_file: Private runtime connection file, never part of the package.
        verifier_connection_file: Separate private verifier connection file.
        duration_seconds: Finite bridge lifetime. Already executing calls drain before teardown.
        host: Loopback default; explicit interface override for an isolated container network.
        port: Fixed local port or zero for OS allocation.
        provider_limits: Explicit simulator allowance for this trial; immutable on resume.
    """

    action: Literal[
        "propose-worlds",
        "construct-assets",
        "register",
        "qualify",
        "prepare-training",
        "export",
        "serve",
    ]
    world_proposals: WorldProposalSpec | None = None
    construction: ConstructionSpec | None = None
    definition: ExecutableTask | None = None
    task: ArtifactInput | None = None
    qualification_spec: ExecutableQualificationSpec | None = None
    qualification: ArtifactInput | None = None
    comparison: ArtifactInput | None = None
    comparison_case: int | None = Field(default=None, ge=0, strict=True)
    operation_file: Path | None = None
    directory: Path | None = None
    name: str = "experiential/executable-task"
    diagnostic: bool = False
    trial_id: str = Field(default="trial-1", min_length=1, max_length=128)
    actor_connection_file: Path | None = None
    verifier_connection_file: Path | None = None
    duration_seconds: int = Field(default=600, ge=1, le=3600)
    host: str = "127.0.0.1"
    port: int = Field(default=0, ge=0, le=65535)
    provider_limits: ProviderLimits | None = None

    @model_validator(mode="after")
    def operation_fields(self) -> "ExecutableCommand":
        """Reject conflicting or ignored payloads before publication, consent or runtime imports."""
        allowed = {
            "propose-worlds": {"world_proposals"},
            "construct-assets": {"construction"},
            "register": {"definition"},
            "qualify": {"qualification_spec"},
            "prepare-training": {
                "qualification",
                "comparison",
                "comparison_case",
                "operation_file",
            },
            "export": {"task", "qualification", "directory", "name", "diagnostic"},
            "serve": {
                "directory",
                "trial_id",
                "actor_connection_file",
                "verifier_connection_file",
                "duration_seconds",
                "host",
                "port",
                "provider_limits",
            },
        }[self.action]
        required = {
            "propose-worlds": {"world_proposals"},
            "construct-assets": {"construction"},
            "register": {"definition"},
            "qualify": {"qualification_spec"},
            "prepare-training": {"qualification"},
            "export": {"task", "directory"},
            "serve": {"directory", "actor_connection_file", "verifier_connection_file"},
        }[self.action]
        for field, definition in type(self).model_fields.items():
            if (
                field != "action"
                and field not in allowed
                and getattr(self, field) != definition.default
            ):
                raise ValueError(f"{self.action} does not accept {field}")
        if any(getattr(self, field) is None for field in required):
            raise ValueError(f"{self.action} requires {', '.join(sorted(required))}")
        return self


def _connection_file(path: Path, endpoint: str, token: str, *, verifier: bool) -> None:
    """Create an exclusive owner-readable capability file without logging its contents."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {
                "EXP_HARBOR_ENDPOINT": endpoint,
                "EXP_HARBOR_VERIFIER_TOKEN" if verifier else "EXP_HARBOR_ACTOR_TOKEN": token,
            },
            stream,
        )


def run_executable_command(
    project: ProjectStore,
    command: ExecutableCommand,
    *,
    console: Console,
    dry_run: bool,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
) -> None:
    """Publish immutable artifacts or explicitly serve a bounded trusted environment runtime."""
    now = datetime.now(UTC)
    if command.action == "propose-worlds":
        spec = command.world_proposals
        if spec is None:
            raise ValueError("propose-worlds requires world_proposals")
        saved = project.artifacts.exists(spec.identity(code_revision) + "-response")
        if saved:
            world_proposal_request(project, spec)
        else:
            if project.artifacts.exists(spec.identity(code_revision) + "-intent"):
                raise ValueError("author dispatch has an unknown outcome; no implicit redraw")
            preflight_world_proposals(project, spec)
        if dry_run:
            console.print(
                "Author context and sources checked; no calls, binding or native execution."
            )
            return
        if not saved and not require_spend_consent(
            console,
            root=project.paths.root,
            yes=yes,
            estimated_cost_usd=None,
            cost_is_upper_bound=False,
            non_interactive=non_interactive,
            command=f"exp eval {project.paths.project_id} --continuations propose-worlds.json",
        ):
            return
        console.print_json(
            data=propose_worlds(project, spec, code_revision=code_revision).model_dump(mode="json")
        )
        return
    if command.action == "construct-assets":
        if command.construction is None:
            raise ValueError("construct-assets requires construction")
        if dry_run:
            files = preflight_construction(project.artifacts, command.construction)
            console.print_json(data={"assets": sorted(files), "status": "validated, not published"})
            return
        reference = persist_construction(
            project.artifacts, command.construction, created_at=now, code_revision=code_revision
        )
        console.print_json(
            data={
                "reference": reference.model_dump(mode="json"),
                "assets": [
                    "assets/tables.json",
                    *[f"assets/{name}" for name in sorted(command.construction.workbooks)],
                ],
                "scope": (
                    "authored initial assets; task qualification and native fidelity unestablished"
                ),
            }
        )
        return
    if command.action == "prepare-training":
        if command.qualification is None:
            raise ValueError("prepare-training requires qualification")
        output = command.operation_file
        if output is not None and (output.exists() or not output.parent.is_dir()):
            raise ValueError("operation_file needs a new file in an existing directory")
        job = qualified_training_job(
            project,
            qualification=command.qualification,
            comparison=command.comparison,
            comparison_case=command.comparison_case,
        )
        operation = {"action": "run", "job": job.model_dump(mode="json")}
        if output is not None and not dry_run:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(operation, stream, indent=2)
                stream.write("\n")
        console.print_json(data=operation)
        return
    if command.action == "register":
        if command.definition is None:
            raise ValueError("register requires a definition")
        if dry_run:
            preflight_executable_task(project, command.definition)
            console.print(
                "Task sources and runtime bytes checked. No artifact published or runtime imported."
            )
            return
        reference = persist_executable_task(
            project, command.definition, created_at=now, code_revision=code_revision
        )
        console.print_json(data=reference.model_dump(mode="json"))
        return
    if command.action == "qualify":
        if command.qualification_spec is None:
            raise ValueError("qualify requires qualification_spec")
        if dry_run:
            preflight_executable_qualification(project, command.qualification_spec)
            console.print(
                "Qualification sources checked. No native witnesses executed or result published."
            )
            return
        reference = qualify_executable_task(
            project, command.qualification_spec, created_at=now, code_revision=code_revision
        )
        report = load_executable_qualification(project, reference)
        console.print_json(
            data={"reference": reference.model_dump(mode="json"), **report.model_dump(mode="json")}
        )
        return
    if command.directory is None:
        raise ValueError("export and serve require a package directory")
    if command.action == "export":
        if command.task is None:
            raise ValueError("export requires an exact task reference")
        if dry_run:
            preflight_harbor_destination(command.directory, name=command.name)
            load_executable_task(project, command.task)
            if command.qualification is not None:
                preflight_qualification_reference(
                    project, command.qualification, command.task, diagnostic=command.diagnostic
                )
            elif not command.diagnostic:
                raise ValueError(
                    "observable contract is unqualified; use an explicit diagnostic export"
                )
            console.print(
                "Export sources and destination checked. "
                "No native witnesses executed or package written."
            )
            return
        path = export_executable_harbor(
            project,
            command.task,
            command.directory,
            name=command.name,
            qualification=command.qualification,
            diagnostic=command.diagnostic,
        )
        console.print(f"Harbor package: {path}")
        return
    binding = verify_harbor_export(command.directory)
    if binding.get("backend") != "executable-checkpoint":
        raise ValueError("serve requires an executable-checkpoint package")
    task = ArtifactInput.model_validate(binding.get("task"))
    definition = load_executable_task(project, task)
    preflight_executable_task(project, definition)
    if binding.get("profile_sha256") != definition.profile() or binding.get(
        "checkpoint"
    ) != definition.job.study.checkpoint.model_dump(mode="json"):
        raise ValueError("package differs from its frozen executable task; export it again")
    qualification = binding.get("qualification_evidence")
    qualification_ref = (
        None if qualification is None else ArtifactInput.model_validate(qualification)
    )
    if qualification_ref is not None:
        preflight_qualification_reference(
            project,
            qualification_ref,
            task,
            diagnostic=binding.get("qualification") == "diagnostic_only",
        )
    elif binding.get("qualification") != "diagnostic_only":
        raise ValueError("qualified package is missing qualification evidence; export it again")
    actor_file, verifier_file = command.actor_connection_file, command.verifier_connection_file
    if (
        actor_file is None
        or verifier_file is None
        or actor_file.resolve() == verifier_file.resolve()
    ):
        raise ValueError("serve needs distinct private actor and verifier connection files")
    for path in (actor_file, verifier_file):
        if (
            path.exists()
            or path.is_symlink()
            or path.resolve().is_relative_to(command.directory.resolve())
        ):
            raise ValueError(
                "connection files must be new private paths outside the exported package"
            )
        if not path.parent.is_dir():
            raise ValueError(
                "connection file parent is missing; create its private directory first"
            )
    if dry_run:
        console.print(
            "Package inventory and sources checked. No runtime imported, native witnesses "
            "executed or listener started. Full qualification is rechecked when serving."
        )
        return
    if command.provider_limits is not None:
        console.print(
            f"Simulator allowance: {command.provider_limits.maximum_physical_attempts} physical "
            f"attempts until {command.provider_limits.deadline.isoformat()}; "
            "resumes share this trial's counter."
        )
    if not require_spend_consent(
        console,
        root=project.paths.root,
        yes=yes,
        estimated_cost_usd=None,
        cost_is_upper_bound=False,
        non_interactive=non_interactive,
        command=f"exp eval {project.paths.project_id} --continuations executable-serve.json",
    ):
        return
    task = verify_executable_package(project, command.directory)
    created = []
    try:
        with open_executable_harbor_trial(
            project,
            task,
            trial_id=command.trial_id,
            code_revision=code_revision,
            qualification=qualification_ref,
            provider_limits=command.provider_limits,
        ) as trial:
            with serve_harbor_trial(trial, host=command.host, port=command.port) as endpoint:
                for path, token, verifier in (
                    (actor_file, endpoint.actor_token, False),
                    (verifier_file, endpoint.verifier_token, True),
                ):
                    _connection_file(path, endpoint.url, token, verifier=verifier)
                    created.append(path)
                console.print(
                    f"Executable trial ready for {command.duration_seconds}s. "
                    "Inject each private connection file only into its intended environment."
                )
                Event().wait(command.duration_seconds)
            console.print_json(data=trial.result())
    finally:
        for path in created:
            path.unlink(missing_ok=True)

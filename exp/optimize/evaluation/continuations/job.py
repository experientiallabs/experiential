"""Frozen continuation jobs: preflight, admitted execution and saved-result lookup.

A job binds one continuation study to an operator-selected trusted runtime file, an actor catalog
alias and finite provider allowances. Execution meters every actor HTTP attempt through the
admitted transport; saved results are read back without loading runtime code or credentials.
"""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.deciders import admitted_decider, preflight_decider
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudy,
    ContinuationStudySpec,
    run_continuation_study,
    study_identity,
)
from exp.optimize.evaluation.executable.loader import (
    ContinuationRuntime,
    load_runtime_definition,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task

# Shared ceiling on durable provider attempts registered by one continuation job and by every
# panel, experiment, contrast or campaign composed from jobs, counted before any dispatch.
MAXIMUM_PHYSICAL_ATTEMPTS = 100_000


class ContinuationJob(ContractModel):
    """One explicit local runtime implementation and its frozen actor study.

    Attributes:
        study: Frozen evidence and sampling contract.
        runtime_file: Explicit trusted Python file exporting ``continuation_runtime``.
            It receives ``(project, configuration)`` and returns ContinuationRuntime.
        runtime_sha256: Exact source bytes, checked before execution.
        configuration: Environment-specific configuration, without private credentials.
        actor_alias: Catalog alias whose snapshot must match the study.
        catalog_sha256: Complete catalog definition pinned before dispatch.
        maximum_model_calls: Original actor policy ceiling, including prefix calls.
        system_prompt: Original actor policy instruction.
        maximum_physical_attempts: Shared durable provider-attempt ceiling, including the
            attempts of any cascade decider.
        deadline: Frozen aware deadline for all actor provider calls.
    """

    study: ContinuationStudySpec
    runtime_file: Path
    runtime_sha256: Sha256
    configuration: JsonObject = Field(default_factory=dict)
    actor_alias: str = Field(min_length=1)
    catalog_sha256: Sha256
    maximum_model_calls: int = Field(gt=0, le=1000)
    system_prompt: str | None = None
    maximum_physical_attempts: int = Field(gt=0, le=MAXIMUM_PHYSICAL_ATTEMPTS)
    deadline: AwareDatetime

    def implementation(self, code_revision: str) -> Sha256:
        """Bind local runtime configuration as well as the installed application revision."""
        return sha256_json({"job": self.model_dump(mode="json"), "code": code_revision})


def preflight_continuation_job(project: ProjectStore, job: ContinuationJob) -> None:
    """Validate bytes, actor policy and catalog without loading local code or credentials."""
    if job.study.training_evidence is not None:
        # Evidence readers consume jobs, so validate this dependency after module initialization.
        from exp.optimize.evaluation.continuations.evidence import qualified_training_job

        evidence = job.study.training_evidence
        expected = qualified_training_job(
            project,
            qualification=evidence.qualification,
            comparison=evidence.comparison,
            comparison_case=evidence.comparison_case,
        )
        if job != expected:
            raise ValueError("generation differs from the qualified execution; prepare a fresh job")
    if sha256_bytes(job.runtime_file.read_bytes()) != job.runtime_sha256:
        raise ValueError("runtime source changed; register the exact reviewed file digest")
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != job.catalog_sha256:
        raise ValueError("model catalog changed; freeze a new continuation job")
    registry = RuntimeModelCatalog(catalog, environment={})
    snapshot, capabilities = registry.snapshot(job.actor_alias)
    if snapshot != job.study.actor_model:
        raise ValueError("actor alias differs from the registered model snapshot")
    require_admitted_transport(catalog, job.actor_alias, consumer="continuation actor")
    if job.study.cascade is not None:
        preflight_decider(catalog, job.study.cascade.decider)
    if (
        capabilities.maximum_output_tokens is not None
        and job.study.maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError("actor output allowance exceeds model capacity")
    efforts = tuple(catalog.models[job.actor_alias].supported_reasoning_efforts or ())
    if job.study.reasoning_effort is not None and job.study.reasoning_effort not in efforts:
        raise ValueError(
            f"actor reasoning effort {job.study.reasoning_effort!r} is not supported by "
            f"{job.actor_alias}; choose one of {', '.join(efforts) or 'none declared'}"
        )
    checkpoint = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    require_development_task(project.artifacts, checkpoint.task)
    actor = ChatAgentRuntime(
        maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
    )
    if not actor.remaining_calls(checkpoint.actor) or checkpoint.task.partition != "fit":
        raise ValueError("continuation needs a development checkpoint with remaining actor calls")


def saved_continuation_job(
    project: ProjectStore, job: ContinuationJob, *, code_revision: str
) -> ContinuationStudy | None:
    """Read the complete saved study without loading environment code or provider credentials."""
    identity = study_identity(job.study, job.implementation(code_revision)) + "-result"
    if not project.artifacts.exists(identity):
        return None
    return load_study(project.artifacts, artifact_input(project.artifacts.read(identity).manifest))


def execute_continuation_job(
    project: ProjectStore, job: ContinuationJob, *, code_revision: str
) -> ContinuationStudy:
    """Execute explicitly selected trusted local code and admit every actor HTTP attempt.

    Args:
        project: Trusted project and configured catalog.
        job: Exact runtime, actor, observation and physical-budget settings.
        code_revision: Installed implementation identity.

    Returns:
        Every planned branch, including unknown outcomes. Saved results make no new calls.
    """
    saved = saved_continuation_job(project, job, code_revision=code_revision)
    if saved is not None:
        return saved
    preflight_continuation_job(project, job)
    if job.study.training_evidence is not None:
        # Evidence qualification reads jobs and studies, so defer this cycle until execution.
        from exp.optimize.evaluation.continuations.evidence import verify_training_evidence

        verify_training_evidence(
            project.artifacts, job.study, execution_revision=job.implementation(code_revision)
        )
    if datetime.now(UTC) >= job.deadline:
        raise ValueError("continuation deadline expired before execution")
    revision = job.implementation(code_revision)
    ledger = DispatchLedger(
        project.artifacts.project_directory / (revision + "-physical.sqlite"),
        job.maximum_physical_attempts,
        revision,
    )
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision=revision,
        deadline=job.deadline,
    )
    catalog = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), transport_factory=lambda: transport
    )
    model = catalog.resolve(job.actor_alias)
    if model.snapshot != job.study.actor_model:
        raise ValueError("actor alias differs from the registered model snapshot")
    decider = (
        None
        if job.study.cascade is None
        else admitted_decider(catalog, job.study.cascade.decider, transport, job.deadline)
    )
    with load_continuation_runtime(project, job) as runtime:
        return run_continuation_study(
            project.artifacts,
            job.study,
            actor=ChatAgentRuntime(
                maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
            ),
            model=model.client,
            world=runtime.world,
            probe=runtime.probe,
            probe_sha256=runtime.probe_sha256,
            code_revision=revision,
            decider=decider,
        )


def load_continuation_runtime(project: ProjectStore, job: ContinuationJob) -> ContinuationRuntime:
    """Load the explicitly selected environment factory without composing an actor model.

    Only the checked source bytes execute. Private probes remain evaluator capabilities.
    Dependencies imported by the trusted factory are the operator's environment responsibility.
    The caller closes the returned runtime, normally with ``with``.
    """
    return load_runtime_definition(
        project,
        runtime_file=job.runtime_file,
        runtime_sha256=job.runtime_sha256,
        configuration=job.configuration,
        probe_sha256=job.study.probe_sha256,
    )


def continuation_result_reference(
    project: ProjectStore, job: ContinuationJob, *, code_revision: str
) -> ArtifactInput:
    """Return the verified immutable result reference for CLI handoffs and downstream evidence."""
    if saved_continuation_job(project, job, code_revision=code_revision) is None:
        raise ValueError("continuation job has no complete saved result")
    identity = study_identity(job.study, job.implementation(code_revision)) + "-result"
    return artifact_input(project.artifacts.read(identity).manifest)

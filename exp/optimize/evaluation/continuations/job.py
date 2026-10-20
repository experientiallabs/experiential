"""CLI composition of trusted executable runtimes and bounded continuation actors."""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudy,
    ContinuationStudySpec,
    WorldRuntime,
    run_continuation_study,
)
from exp.optimize.evaluation.executable.provider_grant import ProviderGrant
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import WorldCheckpoint, load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task


@dataclass(frozen=True)
class ContinuationRuntime:
    """Trusted environment implementation selected explicitly by the operator.

    Attributes:
        world: Existing executable or mixed runtime with exact restore semantics.
        probe: Evaluator-only state projection, never passed to the actor.
        probe_sha256: Digest of the projection definition checked against registration.
    """

    world: WorldRuntime
    probe: Callable[[WorldCheckpoint], JsonObject]
    probe_sha256: Sha256


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
        maximum_physical_attempts: Shared durable provider-attempt ceiling.
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
    maximum_physical_attempts: int = Field(gt=0, le=100_000)
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
    provider = catalog.connections[catalog.models[job.actor_alias].connection].provider
    if provider not in {
        "openai",
        "openai-compatible",
        "openrouter",
        "anthropic",
        "gemini",
        "azure",
    }:
        raise ValueError("continuation actor requires a provider using the admitted HTTP transport")
    if (
        capabilities.maximum_output_tokens is not None
        and job.study.maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError("actor output allowance exceeds model capacity")
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
    identity = (
        stable_id(
            "continuation-study",
            {"spec": sha256_json(job.study), "code": job.implementation(code_revision)},
        )
        + "-result"
    )
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
    runtime = load_continuation_runtime(project, job)
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
    )


def load_continuation_runtime(project: ProjectStore, job: ContinuationJob) -> ContinuationRuntime:
    """Load the explicitly selected environment factory without composing an actor model.

    Only the checked source bytes execute. Private probes remain evaluator capabilities.
    Dependencies imported by the trusted factory are the operator's environment responsibility.
    """
    return load_runtime_definition(
        project,
        runtime_file=job.runtime_file,
        runtime_sha256=job.runtime_sha256,
        configuration=job.configuration,
        probe_sha256=job.study.probe_sha256,
    )


def load_runtime_definition(
    project: ProjectStore,
    *,
    runtime_file: Path,
    runtime_sha256: Sha256,
    configuration: JsonObject,
    probe_sha256: Sha256,
    provider_grant: ProviderGrant | None = None,
) -> ContinuationRuntime:
    """Execute verified factory bytes for initial capture or exact checkpoint restoration."""
    if provider_grant is not None:
        provider_grant.require_configuration(project, configuration, runtime_sha256=runtime_sha256)
    module = load_runtime_module(runtime_file, runtime_sha256)
    entrypoint = "continuation_runtime" if provider_grant is None else "trial_runtime"
    factory = getattr(module, entrypoint, None)
    if not callable(factory):
        raise ValueError(f"runtime must export {entrypoint} for this execution authority")
    runtime = (
        factory(project, configuration)
        if provider_grant is None
        else factory(project, configuration, provider_grant)
    )
    if not isinstance(runtime, ContinuationRuntime):
        raise ValueError("runtime factory must return ContinuationRuntime")
    if runtime.probe_sha256 != probe_sha256:
        raise ValueError("runtime probe differs from the frozen definition")
    return runtime


def load_runtime_module(runtime_file: Path, runtime_sha256: Sha256) -> ModuleType:
    """Import only the explicitly selected source bytes, never model-generated executable code."""
    module_spec = importlib.util.spec_from_file_location(
        "exp_continuation_runtime_" + runtime_sha256, runtime_file
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("runtime file cannot be loaded as a Python module")
    module = importlib.util.module_from_spec(module_spec)
    source = runtime_file.read_bytes()
    if sha256_bytes(source) != runtime_sha256:
        raise ValueError("runtime source changed after preflight")
    exec(compile(source, str(runtime_file), "exec"), module.__dict__)
    return module


def continuation_result_reference(
    project: ProjectStore, job: ContinuationJob, *, code_revision: str
) -> ArtifactInput:
    """Return the verified immutable result reference for CLI handoffs and downstream evidence."""
    if saved_continuation_job(project, job, code_revision=code_revision) is None:
        raise ValueError("continuation job has no complete saved result")
    identity = (
        stable_id(
            "continuation-study",
            {
                "spec": sha256_json(job.study),
                "code": job.implementation(code_revision),
            },
        )
        + "-result"
    )
    return artifact_input(project.artifacts.read(identity).manifest)

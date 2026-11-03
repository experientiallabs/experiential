"""Induce and validate exact transition programs for a project, with durable model draws.

``induce_transitions`` builds fit episodes from the completed build, asks the inducer model
for one program per tool through the provider-free loop in
:mod:`exp.simulation.induction.proposals`, and retains every draw (intent before dispatch,
then the raw response), every parsed proposal and every validation round. The final result
binds the spec, the fit evidence digest, the final programs and their joint validation report;
its coverage is what the induced world backend executes. A saved result replays without
credentials or calls; an intent without a response is an unknown outcome and never redrawn.

``validate_transitions`` replays saved programs against the build's current fit evidence. It
is provider-free and deterministic, so the same programs and evidence always give the same
report; new or changed fit traces can only keep or shrink what induced code owns.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot, load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.induction.episodes import fit_episodes, fit_traces
from exp.runtime.environments.world_contracts import WorldManifest
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.induction.contracts import (
    InducedCoverage,
    InductionEpisode,
    ProgramProposal,
    SandboxLimits,
    TransitionProgram,
    ValidationReport,
)
from exp.simulation.induction.proposals import (
    InductionSettings,
    induce_programs,
    inducible_tools,
)
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.validation import validate_programs

RESULT_TYPE = "induced-transitions"
VALIDATION_TYPE = "induced-transitions-validation"


class InductionSpec(ContractModel):
    """One induction over a project's fit evidence.

    Attributes:
        manifest: World manifest of the simulated world the programs will serve; the same
            manifest as the grounded runtime definition they are composed with.
        initial_state: Declared starting value of the simulated domains in every fit episode,
            as in the grounded runtime definition; None when the starting world is unknown,
            in which case only steps after recorded state can be validated.
        tools: Actor-visible tool schemas by name; defaults to the fit traces' declared tools.
        inducer_alias: Catalog entry that proposes programs.
        inducer_model: Exact inducer snapshot.
        catalog_sha256: Credential-free catalog identity.
        settings: Attempt, support and prompt allowances.
        limits: Sandbox limits programs are validated and later executed under.
        maximum_episodes: Most fit traces used, in dataset order.
        maximum_physical_attempts: Durable ceiling on provider requests, retries included.
        deadline: Last time any provider request may start.
    """

    manifest: WorldManifest
    initial_state: JsonObject | None = None
    inducer_alias: str = Field(min_length=1)
    inducer_model: ModelSnapshot
    catalog_sha256: Sha256
    settings: InductionSettings = Field(default_factory=InductionSettings)
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    maximum_episodes: int = Field(default=64, ge=1, le=10_000, strict=True)
    maximum_physical_attempts: int = Field(ge=1, le=10_000, strict=True)
    deadline: AwareDatetime

    def identity(self, code_revision: str) -> str:
        """Return the result identity of this spec under one code revision."""
        return stable_id("induced-transitions", {"spec": sha256_json(self), "code": code_revision})


class InductionResult(ContractModel):
    """The saved outcome of one induction.

    Attributes:
        spec: The induction that ran.
        episodes_sha256: Digest of the exact fit episodes replayed.
        programs: The last program proposed for each tool, accepted or not.
        report: Joint validation of those programs; its coverage is what induced code owns.
    """

    spec: InductionSpec
    episodes_sha256: Sha256
    programs: tuple[TransitionProgram, ...]
    report: ValidationReport


class TransitionValidationSpec(ContractModel):
    """A provider-free re-validation of saved programs on the build's current fit evidence.

    Attributes:
        induction: Saved induction result whose programs are validated.
        minimum_support: Exact steps each program needs; defaults to the induction's.
    """

    induction: ArtifactInput
    minimum_support: int | None = Field(default=None, ge=1, strict=True)


class TransitionValidation(ContractModel):
    """The saved outcome of one re-validation.

    Attributes:
        induction: Induction whose programs were validated.
        episodes_sha256: Digest of the fit episodes replayed now.
        unchanged_evidence: Whether those episodes equal the ones the induction used.
        report: The new joint validation; its coverage replaces the induction's.
    """

    induction: ArtifactInput
    episodes_sha256: Sha256
    unchanged_evidence: bool
    report: ValidationReport


def preflight_induction(project: ProjectStore, spec: InductionSpec) -> tuple[InductionEpisode, ...]:
    """Check catalog, model and evidence without credentials or calls; return the episodes.

    Raises:
        ValueError: The catalog changed, the model differs or lacks typed tools, the admitted
            transport is unavailable, no fit trace calls a manifest tool, or the attempt
            ceiling cannot cover one proposal per inducible tool.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    if sha256_json(catalog.model_dump(mode="json")) != spec.catalog_sha256:
        raise ValueError("inducer catalog changed since the spec was written; refresh it")
    snapshot, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(
        spec.inducer_alias
    )
    if snapshot != spec.inducer_model or capabilities.supports_tools is not True:
        raise ValueError("the inducer needs its registered snapshot and typed tool output")
    if (
        capabilities.maximum_output_tokens is not None
        and spec.settings.maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError("inducer output allowance exceeds the model's output capacity")
    require_admitted_transport(catalog, spec.inducer_alias, consumer="transition inducer")
    episodes = _episodes(project, spec)
    tools = inducible_tools(spec.manifest, episodes)
    if not tools:
        raise ValueError(
            "no fit trace calls a tool whose domains the manifest gives the simulated side; "
            "check the manifest tool names against the build's traces"
        )
    if spec.maximum_physical_attempts < len(tools):
        raise ValueError(
            f"maximum_physical_attempts {spec.maximum_physical_attempts} cannot cover one "
            f"proposal for each of the {len(tools)} inducible tools"
        )
    return episodes


def induce_transitions(
    project: ProjectStore, spec: InductionSpec, *, code_revision: str
) -> ArtifactInput:
    """Run or replay one induction and return its saved result.

    Raises:
        ValueError: Preflight fails, or a draw has an unknown outcome.
        filelock.Timeout: Another process is running the same induction.
    """
    store = project.artifacts
    identity = spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            reference = artifact_input(store.read(identity).manifest)
            load_induction(store, reference)
            return reference
        episodes = preflight_induction(project, spec)
        build = project.load_project().build
        if build is None:
            raise ValueError("induction needs a completed build; run exp build first")
        draw = _durable_draw(project, spec, identity, code_revision)
        programs: dict[str, TransitionProgram] = {}
        # The fit episodes come from the build's trace dataset, split by its task set, so the
        # result cites both: its programs are traceable to the exact evidence they were fit to.
        inputs: list[ArtifactInput] = [build.task_set, build.trace_dataset]
        rounds = [0]

        def record(row: ProgramProposal | ValidationReport) -> None:
            """Retain every proposal and validation round before the loop continues."""
            if isinstance(row, ProgramProposal):
                if row.program is None:
                    programs.pop(row.tool, None)
                else:
                    programs[row.tool] = row.program
                response = artifact_input(
                    store.read(f"{identity}-{row.operation_id}-response").manifest
                )
                inputs.append(
                    _write_once(
                        store,
                        f"{identity}-{row.operation_id}-proposal",
                        "induced-transition-proposal",
                        (response,),
                        {"proposal.json": row},
                        code_revision,
                    )
                )
            else:
                rounds[0] += 1
                inputs.append(
                    _write_once(
                        store,
                        f"{identity}-round-{rounds[0]}",
                        "induced-transitions-round",
                        (),
                        {"report.json": row},
                        code_revision,
                    )
                )

        with ProgramSandbox(spec.limits) as sandbox:
            report = induce_programs(
                manifest=spec.manifest,
                tools=_tool_schemas(project, spec),
                episodes=episodes,
                draw=draw,
                record=record,
                sandbox=sandbox,
                settings=spec.settings,
            )
        result = InductionResult(
            spec=spec,
            episodes_sha256=sha256_json([row.model_dump(mode="json") for row in episodes]),
            programs=tuple(programs[tool] for tool in sorted(programs)),
            report=report,
        )
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type=RESULT_TYPE,
            envelope=_envelope(code_revision, inputs),
            files={"induction.json": result},
        )
        return artifact_input(manifest)


def load_induction(store: ArtifactStore, reference: ArtifactInput) -> InductionResult:
    """Load one saved induction, checking its identity and retained proposals.

    Raises:
        ValueError: The reference is not a saved induction or its content was changed.
    """
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != RESULT_TYPE:
        raise ValueError("reference is not a saved induced-transitions result")
    result = InductionResult.model_validate_json(
        store.read_bytes(reference.artifact_id, "induction.json")
    )
    if result.spec.identity(manifest.code_revision) != reference.artifact_id:
        raise ValueError("induction content differs from its frozen identity")
    proposed = set()
    for row in manifest.inputs:
        if row.artifact_id.endswith("-proposal"):
            proposal = ProgramProposal.model_validate_json(
                store.read_bytes(row.artifact_id, "proposal.json")
            )
            if proposal.program is not None:
                proposed.add(proposal.program)
    if not set(result.programs) <= proposed:
        raise ValueError("induction programs are not among its retained proposals")
    return result


def load_coverage(store: ArtifactStore, reference: ArtifactInput) -> InducedCoverage:
    """Return the coverage of a saved induction or of a saved re-validation."""
    manifest = store.read(reference.artifact_id).manifest
    if manifest.artifact_type == VALIDATION_TYPE and artifact_input(manifest) == reference:
        saved = TransitionValidation.model_validate_json(
            store.read_bytes(reference.artifact_id, "validation.json")
        )
        load_induction(store, saved.induction)
        return saved.report.coverage
    return load_induction(store, reference).report.coverage


def validate_transitions(
    project: ProjectStore, spec: TransitionValidationSpec, *, code_revision: str
) -> ArtifactInput:
    """Replay saved programs on the current fit evidence and save the new coverage."""
    store = project.artifacts
    induction = load_induction(store, spec.induction)
    episodes = _episodes(project, induction.spec)
    digest = sha256_json([row.model_dump(mode="json") for row in episodes])
    identity = stable_id(
        "induced-transitions-validation",
        {"spec": sha256_json(spec), "episodes": digest, "code": code_revision},
    )
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    with ProgramSandbox(induction.spec.limits) as sandbox:
        report = validate_programs(
            induction.programs,
            episodes,
            induction.spec.manifest,
            sandbox,
            minimum_support=spec.minimum_support or induction.spec.settings.minimum_support,
        )
    validation = TransitionValidation(
        induction=spec.induction,
        episodes_sha256=digest,
        unchanged_evidence=digest == induction.episodes_sha256,
        report=report,
    )
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=VALIDATION_TYPE,
        envelope=_envelope(code_revision, [spec.induction]),
        files={"validation.json": validation},
    )
    return artifact_input(manifest)


def _episodes(project: ProjectStore, spec: InductionSpec) -> tuple[InductionEpisode, ...]:
    """Return the fit episodes one induction spec replays."""
    return fit_episodes(
        project,
        tools=(tool.name for tool in spec.manifest.tools),
        initial_state=spec.initial_state,
        maximum_episodes=spec.maximum_episodes,
    )


def _tool_schemas(project: ProjectStore, spec: InductionSpec) -> tuple[ToolSchema, ...]:
    """Return actor-visible schemas of the manifest's tools as the fit traces declare them.

    Held-out traces are skipped by the fit-partition filter, so none of their schemas reaches
    the inducer prompt.
    """
    names = {tool.name for tool in spec.manifest.tools}
    schemas = {}
    for trace, _ in fit_traces(project):
        for tool in trace.tools:
            if tool.name in names:
                schemas.setdefault(tool.name, tool)
    return tuple(schemas[name] for name in sorted(schemas))


def _durable_draw(
    project: ProjectStore, spec: InductionSpec, identity: str, code_revision: str
) -> Callable[[ModelRequest, str], ModelResponse]:
    """Return a draw that replays saved responses and retains intent before every dispatch."""
    store = project.artifacts
    clients = []

    def draw(request: ModelRequest, operation: str) -> ModelResponse:
        """Send or replay one proposal request under its fixed operation identity."""
        base = f"{identity}-{operation}"
        if store.exists(base + "-response"):
            saved = ModelRequest.model_validate_json(
                store.read_bytes(base + "-intent", "request.json")
            )
            if saved != request:
                raise ValueError(
                    f"induction draw {operation} would send a different request than it saved; "
                    "the fit evidence or code changed, so start a new induction"
                )
            return ModelResponse.model_validate_json(
                store.read_bytes(base + "-response", "response.json")
            )
        if store.exists(base + "-intent"):
            raise ValueError(
                f"induction draw {operation} has an unknown outcome; no implicit redraw"
            )
        if not clients:
            transport = AdmittedJsonTransport(
                HttpxJsonTransport(),
                artifacts=store,
                code_revision=code_revision,
                deadline=spec.deadline,
                ledger=DispatchLedger(
                    store.project_directory / (identity + ".sqlite"),
                    spec.maximum_physical_attempts,
                    sha256_json(spec),
                ),
            )
            registry = RuntimeModelCatalog(
                load_model_catalog(project.model_catalog_path), transport_factory=lambda: transport
            )
            # Resolve credentials before the first intent so a refusal stays retryable.
            clients.append(registry.resolve(spec.inducer_alias).client)
        intent = store.write_json(
            artifact_id=base + "-intent",
            artifact_type="induced-transitions-intent",
            envelope=_envelope(code_revision, []),
            files={"request.json": request, "spec.json": spec},
        )
        response = clients[0].complete(request)
        if response.model != spec.inducer_model:
            raise ValueError("inducer response came from another model snapshot")
        store.write_json(
            artifact_id=base + "-response",
            artifact_type="induced-transitions-response",
            envelope=_envelope(code_revision, [artifact_input(intent)]),
            files={"response.json": response},
        )
        return response

    return draw


def _write_once(
    store: ArtifactStore,
    artifact_id: str,
    artifact_type: str,
    inputs: tuple[ArtifactInput, ...],
    files: dict[str, ContractModel],
    code_revision: str,
) -> ArtifactInput:
    """Write a deterministic artifact, or reuse an identical one left by an interrupted run."""
    if store.exists(artifact_id):
        saved = store.read(artifact_id).manifest
        for name, value in files.items():
            if type(value).model_validate_json(store.read_bytes(artifact_id, name)) != value:
                raise ValueError(f"saved {artifact_id} differs from its deterministic replay")
        return artifact_input(saved)
    return artifact_input(
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=_envelope(code_revision, list(inputs)),
            files=files,
        )
    )


def _envelope(code_revision: str, inputs: list[ArtifactInput]) -> ArtifactEnvelope:
    """Return provenance with sorted unique inputs."""
    unique = {row.artifact_id: row for row in inputs}
    return ArtifactEnvelope(
        schema_version=1,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
        inputs=tuple(unique[key] for key in sorted(unique)),
    )

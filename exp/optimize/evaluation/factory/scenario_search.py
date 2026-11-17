"""Bounded native scenario challenges before a new world becomes a task candidate.

Setup proposals and control scripts are untrusted candidates. Independently pinned acceptance
specifications label actual native effects. Discrimination earns development-candidate status,
never release qualification, a new independent lineage or training admission.
"""

from datetime import UTC, datetime
from typing import Literal, Self

from filelock import Timeout
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    redact_secret_text,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.research import ResearchFailure
from exp.optimize.evaluation.continuations.capture import preflight_capture
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.optimize.evaluation.executable.scenarios import (
    RecordScenarioSpec,
    RecordWitnessSpec,
    prepare_record_scenario,
    prepare_record_witness,
)
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.variants import (
    TaskVariantSpec,
    load_task_variant,
    publish_task_variant,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    LabelWitness,
    verify_label_witness,
)
from exp.simulation.evaluation.record_execution import RecordReplayRuntime
from exp.simulation.evaluation.tasks import TaskVerifier, VerifierChallenge, grade
from exp.simulation.evaluation.verifier_probes import VerifierProbeReport, probe_verifier


class ScenarioControl(ContractModel):
    """A proposed actor-visible script, labeled by native effects rather than its role.

    Attributes:
        role: Intended discrimination coverage, not an asserted correctness label.
        actions: Complete script from the reached scenario using actor tools only.
        finish_tick: Absolute clock cutoff including the script's duration.
    """

    role: Literal["solution", "alternative", "recovery", "targeted_error", "unwanted_effect"]
    actions: tuple[ToolCall, ...] = Field(max_length=128)
    finish_tick: int = Field(ge=0, le=1_000_000)


class ScenarioCandidate(ContractModel):
    """A prospective world construction and independently checkable control inventory.

    Attributes:
        scenario: Existing trusted native setup path, including source and capability pins.
        verifier: Candidate predicate, not permitted to label its own controls.
        acceptance: Independently sourced native-observation specification.
        controls: Proposed solutions and mistakes; every required role must execute.
    """

    scenario: RecordScenarioSpec
    verifier: TaskVerifier
    acceptance: ArtifactInput
    controls: tuple[ScenarioControl, ...] = Field(min_length=5, max_length=20)

    @model_validator(mode="after")
    def coverage(self) -> Self:
        """Require explicit alternative, recovery, targeted error and collateral-effect cases."""
        if {row.role for row in self.controls} != {
            "solution",
            "alternative",
            "recovery",
            "targeted_error",
            "unwanted_effect",
        }:
            raise ValueError("scenario controls must cover every required role")
        return self


class ScenarioResearchTarget(ContractModel):
    """Cited research motivating a discrimination experiment, without adjudicating blame.

    Attributes:
        failure: Original observations and competing explanations from the research contract.
        hypothesis: Explanation tested by the targeted_error controls across proposed worlds.
            Script relevance remains an authored claim; execution checks actual effects.
    """

    failure: ResearchFailure
    hypothesis: str = Field(min_length=1)

    @model_validator(mode="after")
    def named_hypothesis(self) -> Self:
        """Keep the selected explanation and its alternatives unambiguous."""
        names = [row.id for row in self.failure.hypotheses]
        if len(set(names)) != len(names) or self.hypothesis not in names:
            raise ValueError("scenario target requires one uniquely named research hypothesis")
        return self


class ScenarioSearchSpec(ContractModel):
    """Finite native prototype comparison, with one preserved intent and source family.

    Attributes:
        base: Original executable task supplying actor policy and source lineage.
        brief: Capability brief for the retained intent.
        prompt: Public task instruction, without checker details or expected answers.
        intent: User outcome to preserve while changing the world.
        candidates: Two to sixteen distinct prospective constructions; all are retained.
        target: Optional exact research observations and selected hypothesis. No causal
            attribution is inferred when its control is successfully distinguished.
    """

    base: ArtifactInput
    brief: ArtifactInput
    prompt: str = Field(min_length=1, max_length=20000)
    intent: str = Field(min_length=1, max_length=4000)
    candidates: tuple[ScenarioCandidate, ...] = Field(min_length=2, max_length=16)
    target: ScenarioResearchTarget | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def distinct(self) -> Self:
        """Changing scripts alone cannot masquerade as constructing a different world."""
        if len({sha256_json(row.scenario) for row in self.candidates}) != len(self.candidates):
            raise ValueError("scenario prototypes require distinct construction definitions")
        return self

    def identity(self, revision: str) -> str:
        """Freeze the entire candidate population before native execution."""
        return stable_id("scenario-search", {"spec": sha256_json(self), "code": revision})


class ScenarioCell(ContractModel):
    """One retained prototype disposition, without selecting away operational failures.

    Attributes:
        index: Original candidate position.
        scenario: Reached setup result when available.
        variant: New executable task variant only after all discrimination checks pass.
        witnesses: Native evidence collected, including failing control outcomes.
        refusals: Concrete unmet requirements or incomplete-execution class.
        verifier_probes: Synthetic specification checks keyed by their actual baseline witness.
            These are not native observations and confer no independent qualification.
    """

    index: int
    scenario: ArtifactInput | None = None
    variant: ArtifactInput | None = None
    witnesses: tuple[ArtifactInput, ...] = ()
    refusals: tuple[str, ...] = ()
    verifier_probes: dict[str, VerifierProbeReport] = Field(default_factory=dict)


class ScenarioSearchResult(ContractModel):
    """All construction outcomes and their limited interpretation.

    Attributes:
        spec: Exact pre-execution registration.
        cells: Every proposed construction in original order.
        scope: Deliberately narrower than independent verifier or release qualification.
    """

    spec: ScenarioSearchSpec
    cells: tuple[ScenarioCell, ...]
    scope: str = (
        "native development discrimination under supplied setup and goal specifications; "
        "same source family; requires fresh independent qualification before delivery; "
        "no production fidelity, general strategy diagnosis or training admission"
    )


def preflight_scenario_search(
    project: ProjectStore, spec: ScenarioSearchSpec, *, pending: frozenset[int] | None = None
) -> None:
    """Validate pins and lineage without importing a runtime or buying model calls."""
    _sources(project, spec)
    base = load_executable_task(project, spec.base)
    parent = load_continuation_checkpoint(project.artifacts, base.job.study.checkpoint)
    for index, candidate in enumerate(spec.candidates):
        capture = candidate.scenario.capture
        if pending is None or index in pending:
            preflight_capture(project, capture)
        if (
            base.job.study.checkpoint not in capture.sources
            or capture.task.lineage_group_id != parent.task.lineage_group_id
            or capture.task.source_trace_ids != parent.task.source_trace_ids
            or capture.task.tools != parent.task.tools
            or capture.maximum_model_calls != base.job.maximum_model_calls
            or capture.system_prompt != base.job.system_prompt
        ):
            raise ValueError(
                "scenario construction must retain base lineage, tools and actor policy"
            )
        manifest = project.artifacts.read(candidate.acceptance.artifact_id).manifest
        if artifact_input(manifest) != candidate.acceptance or manifest.artifact_type != (
            "evaluation-specification"
        ):
            raise ValueError("scenario acceptance requires an exact independent specification")
        if grade(candidate.verifier, {"steps": [], "stop_reason": "failure"}).status != (
            "unassessable"
        ):
            raise ValueError("scenario verifier must abstain when state evidence is absent")


def _sources(project: ProjectStore, spec: ScenarioSearchSpec) -> tuple[ArtifactInput, ...]:
    """Retain and verify research locators alongside world, intent and acceptance provenance."""
    sources = {spec.base, spec.brief}
    for candidate in spec.candidates:
        sources.update((*candidate.scenario.capture.sources, candidate.acceptance))
    if spec.target is not None:
        for observation in spec.target.failure.observations:
            observation.read(project.artifacts)
            sources.add(observation.source)
    for source in sources:
        if artifact_input(project.artifacts.read(source.artifact_id).manifest) != source:
            raise ValueError("scenario source changed; supply its exact immutable reference")
    return tuple(sorted(sources, key=lambda row: row.artifact_id))


def _cell(
    project: ProjectStore, spec: ScenarioSearchSpec, index: int, revision: str
) -> ScenarioCell:
    """Materialize one world and reject nondiscriminating or unsupported control claims."""
    candidate = spec.candidates[index]
    reached_ref = None
    witnesses = []
    refusals = []
    probes: dict[str, VerifierProbeReport] = {}
    positive = [
        row for row in candidate.controls if row.role in {"solution", "alternative", "recovery"}
    ]
    identities = {
        sha256_json(
            [{"name": action.name, "arguments": action.arguments} for action in row.actions]
        )
        for row in positive
    }
    if len(identities) != len(positive):
        refusals.append("valid alternatives require distinct scripts, not renamed call IDs")
    variant = None
    phase = "native_setup"
    try:
        reached = prepare_record_scenario(project, candidate.scenario, code_revision=revision)
        reached_ref = artifact_input(
            project.artifacts.read(candidate.scenario.identity(revision) + "-result").manifest
        )
        point = load_continuation_checkpoint(project.artifacts, reached.checkpoint)
        capture = candidate.scenario.capture.model_copy(update={"task": point.task})
        phase = "runtime_binding"
        with load_runtime_definition(
            project,
            runtime_file=capture.runtime_file,
            runtime_sha256=capture.runtime_sha256,
            configuration=capture.configuration,
            probe_sha256=capture.probe_sha256,
        ) as runtime:
            if not isinstance(runtime.world, ExecutableWorldRuntime):
                raise ValueError("scenario controls require the declared native world")
            replay = RecordReplayRuntime(runtime.world, runtime.probe, runtime.probe_sha256)
            for control in candidate.controls:
                phase = "control:" + control.role
                output = prepare_record_witness(
                    project,
                    RecordWitnessSpec(
                        capture=capture,
                        checkpoint=reached.checkpoint,
                        actions=control.actions,
                        finish_tick=control.finish_tick,
                        specification=candidate.acceptance,
                    ),
                    code_revision=revision,
                )
                reference = ArtifactInput.model_validate(output["witness"])
                witnesses.append(reference)
                witness = LabelWitness.model_validate_json(
                    project.artifacts.read_bytes(reference.artifact_id, "witness.json")
                )
                expected = (
                    "fail" if control.role in {"targeted_error", "unwanted_effect"} else "pass"
                )
                challenge = VerifierChallenge(
                    id=reference.artifact_id,
                    outcome=witness.observation,
                    expected=expected,
                    authority="native_execution",
                    witness=reference,
                    rationale=control.role,
                    source_refs=(candidate.acceptance,),
                )
                refusal = verify_label_witness(
                    project.artifacts,
                    challenge,
                    label=expected,
                    compared=(candidate.verifier,),
                    record_runtime=replay,
                )
                assessed = grade(candidate.verifier, witness.observation)
                if refusal is None and expected == "pass" and assessed.status == "pass":
                    specification = EvaluationSpecification.model_validate_json(
                        project.artifacts.read_bytes(
                            candidate.acceptance.artifact_id, "specification.json"
                        )
                    )
                    tested = probe_verifier(
                        candidate.verifier, specification.predicate, witness.observation
                    )
                    probes[reference.artifact_id] = tested
                    if tested.false_passes:
                        refusals.append(
                            control.role
                            + ": verifier accepted "
                            + str(len(tested.false_passes))
                            + " synthetic counterexamples rejected by the independent"
                            + " specification; "
                            + "inspect verifier_probes before publishing"
                        )
                if control.role == "recovery":
                    steps = witness.observation.get("steps")
                    if not isinstance(steps, list) or not any(
                        isinstance(step, dict)
                        and isinstance(step.get("reply"), dict)
                        and step["reply"].get("is_error") is True
                        for step in steps[:-1]
                    ):
                        refusals.append(
                            "recovery requires an observed tool error before a valid completion"
                        )
                if refusal is not None:
                    refusals.append(control.role + ": " + refusal.reason + ": " + refusal.detail)
                if assessed.status != expected:
                    refusals.append(
                        control.role + ": verifier does not distinguish the native outcome"
                    )
                if control.role == "targeted_error" and not (
                    assessed.goal.status == "fail" and assessed.policy.status == "pass"
                ):
                    refusals.append("targeted_error must fail the goal using permitted effects")
                if control.role == "unwanted_effect" and not (
                    assessed.goal.status == "pass" and assessed.policy.status == "fail"
                ):
                    refusals.append("unwanted_effect must achieve the goal with prohibited effects")
        if not refusals:
            phase = "task_publication"
            variant = publish_task_variant(
                project,
                TaskVariantSpec(
                    base=spec.base,
                    brief=spec.brief,
                    intent=spec.intent,
                    prompt=spec.prompt,
                    verifier=candidate.verifier,
                    kind="scenario_variation",
                    scenario=reached_ref,
                ),
                created_at=datetime.now(UTC),
                code_revision=revision,
            )
    except Timeout:
        raise  # Busy work remains pending; the owning process may still finish it.
    except (ValueError, OSError, RuntimeError) as error:
        refusals.append(
            phase + ": " + type(error).__name__ + ": " + redact_secret_text(str(error))[0][:2000]
        )
    return ScenarioCell(
        index=index,
        scenario=reached_ref,
        variant=variant,
        witnesses=tuple(witnesses),
        refusals=tuple(refusals),
        verifier_probes=probes,
    )


def run_scenario_search(
    project: ProjectStore, spec: ScenarioSearchSpec, *, code_revision: str
) -> ScenarioSearchResult:
    """Run all registered constructions with saved-cell replay and no automatic redraws."""
    spec = ScenarioSearchSpec.model_validate_json(spec.model_dump_json())
    store = project.artifacts
    identity = spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity + "-result"):
            saved = ScenarioSearchResult.model_validate_json(
                store.read_bytes(identity + "-result", "result.json")
            )
            registration = store.read(identity).manifest
            if (
                saved.spec != spec
                or len(saved.cells) != len(spec.candidates)
                or ScenarioSearchSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
                != spec
                or registration.inputs != _sources(project, spec)
            ):
                raise ValueError("saved scenario search differs from its registration")
            expected = {artifact_input(registration)}
            for index, cell in enumerate(saved.cells):
                child = store.read(identity + f"-cell-{index}").manifest
                expected.add(artifact_input(child))
                if (
                    cell.index != index
                    or ScenarioCell.model_validate_json(
                        store.read_bytes(child.artifact_id, "cell.json")
                    )
                    != cell
                ):
                    raise ValueError("scenario search changed its original cell inventory")
                refs = {
                    *cell.witnesses,
                    *((cell.scenario,) if cell.scenario else ()),
                    *((cell.variant,) if cell.variant else ()),
                }
                if set(child.inputs) != {
                    artifact_input(registration),
                    *refs,
                    *deferred_execution(store, child.artifact_id),
                }:
                    raise ValueError("scenario cell lost its source inventory")
                for ref in refs:
                    if artifact_input(store.read(ref.artifact_id).manifest) != ref:
                        raise ValueError("scenario search dependency changed")
                if cell.variant is not None:
                    variant = load_task_variant(project, cell.variant)
                    if (
                        cell.refusals
                        or len(cell.witnesses) != len(spec.candidates[index].controls)
                        or variant.spec.scenario != cell.scenario
                        or variant.spec.base != spec.base
                        or variant.spec.verifier != spec.candidates[index].verifier
                    ):
                        raise ValueError("scenario search published an unsupported task variant")
            if set(store.read(identity + "-result").manifest.inputs) != expected:
                raise ValueError("scenario search lost its registered result inventory")
            return saved
        pending = frozenset(
            i
            for i in range(len(spec.candidates))
            if not store.exists(identity + f"-cell-{i}")
            and not store.exists(identity + f"-cell-{i}-intent")
        )
        preflight_scenario_search(project, spec, pending=pending)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=_sources(project, spec),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="scenario-search-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells = []
        for index in range(len(spec.candidates)):
            cell_id = identity + f"-cell-{index}"
            if store.exists(cell_id):
                cell = ScenarioCell.model_validate_json(store.read_bytes(cell_id, "cell.json"))
            elif store.exists(cell_id + "-intent") and not deferred_execution(store, cell_id):
                cell = ScenarioCell(
                    index=index,
                    refusals=(
                        "unfinished construction intent; inspect saved evidence "
                        "before a new search",
                    ),
                )
            else:
                if not store.exists(cell_id + "-intent"):
                    store.write_json(
                        artifact_id=cell_id + "-intent",
                        artifact_type="scenario-search-intent",
                        envelope=envelope.model_copy(update={"inputs": (registration,)}),
                        files={"candidate.json": spec.candidates[index]},
                    )
                try:
                    cell = _cell(project, spec, index, code_revision)
                except Timeout:
                    record_execution_deferral(store, cell_id)
                    raise
            if not store.exists(cell_id):
                refs = {registration, *cell.witnesses, *deferred_execution(store, cell_id)}
                refs.update(ref for ref in (cell.scenario, cell.variant) if ref is not None)
                store.write_json(
                    artifact_id=cell_id,
                    artifact_type="scenario-search-cell",
                    envelope=envelope.model_copy(
                        update={"inputs": tuple(sorted(refs, key=lambda r: r.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
        result = ScenarioSearchResult(spec=spec, cells=tuple(cells))
        inputs = (
            registration,
            *(
                artifact_input(store.read(identity + f"-cell-{i}").manifest)
                for i in range(len(cells))
            ),
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="scenario-search-result",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(inputs, key=lambda r: r.artifact_id))}
            ),
            files={"result.json": result},
        )
        return result

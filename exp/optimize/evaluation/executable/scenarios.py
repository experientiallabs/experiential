"""Reach a native seed state, then hand its exact capsule to existing continuation jobs.

The operator explicitly selects trusted runtime code through the existing capture contract.
Setup capabilities are separate from actor tools; only declared state paths become public.
This publishes a development seed, not a qualified evaluation or a fidelity claim.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.continuations.capture import CaptureDefinition, preflight_capture
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.world_artifacts import load_world_image
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.development_scope import require_development_task
from exp.simulation.evaluation.admission import record_label_witness
from exp.simulation.evaluation.record_execution import (
    RecordReplayRuntime,
    execute_record_script,
    load_record_execution,
    require_record_runtime,
)
from exp.simulation.reconstruction.models import path_parts, resolve


class RecordScenarioSpec(ContractModel):
    """A reachable native initialization with explicit public and setup capabilities.

    Attributes:
        capture: Existing trusted runtime, source evidence and public actor policy.
            Runtime configuration freezes the manifest, baseline, seed and schedule.
        setup_tools: Additional tool schemas available only to the setup script.
        setup_actions: Ordered calls reaching the seed state from the native baseline.
        finish_tick: Absolute clock cutoff, after all setup actions have finished.
        public_fields: Public context keys mapped to private-probe JSON pointers. The
            operator explicitly selects this projection; private state is not copied wholesale.
    """

    capture: CaptureDefinition
    setup_tools: tuple[ToolSchema, ...] = ()
    setup_actions: tuple[ToolCall, ...] = Field(default=(), max_length=128)
    finish_tick: int = Field(ge=0, le=1_000_000, strict=True)
    public_fields: dict[str, str] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def separate_capabilities(self) -> "RecordScenarioSpec":
        """Refuse ambiguous setup tools, repeated actions and implicit whole-state disclosure."""
        tools = (*self.capture.task.tools, *self.setup_tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("setup tools must be unique and separate from actor tool names")
        names = {tool.name for tool in tools}
        if any(action.name not in names for action in self.setup_actions):
            raise ValueError("setup action lacks an explicitly declared tool capability")
        if len({action.call_id for action in self.setup_actions}) != len(self.setup_actions):
            raise ValueError("setup actions require unique logical call IDs")
        for name, pointer in self.public_fields.items():
            if not name or name in self.capture.task.initial_context or not pointer:
                raise ValueError("public fields need new context keys and explicit non-root paths")
            path_parts(pointer)
        return self

    def identity(self, code_revision: str) -> str:
        """Freeze setup, public projection and implementation before loading runtime code."""
        return stable_id("record-scenario", {"spec": sha256_json(self), "code": code_revision})


class RecordScenarioResult(ContractModel):
    """Exact source-backed seed capsule ready for the existing continuation/Harbor path.

    Attributes:
        specification: Durable setup intent including source and runtime pins.
        execution: Native setup receipt, replayed twice before publication.
        checkpoint: Paired initial actor state and reached native kernel checkpoint.
        scope: Explicit development status without inferred production accuracy.
    """

    specification: ArtifactInput
    execution: ArtifactInput
    checkpoint: ArtifactInput
    scope: Literal[
        "reachable native development seed; not qualified or production-fidelity evidence"
    ] = "reachable native development seed; not qualified or production-fidelity evidence"


class RecordWitnessSpec(ContractModel):
    """Explicit native execution and independent label specification for one seed task.

    Attributes:
        capture: Trusted runtime source, private probe and public task matching the checkpoint.
        checkpoint: Paired actor/world seed; the actor is not invoked by this operation.
        actions: Complete native script to label, retaining every response and state effect.
        finish_tick: Final absolute clock cutoff after the script.
        specification: Independently authored native-observation predicate, not a candidate judge.
    """

    capture: CaptureDefinition
    checkpoint: ArtifactInput
    actions: tuple[ToolCall, ...] = Field(max_length=128)
    finish_tick: int = Field(ge=0, le=1_000_000, strict=True)
    specification: ArtifactInput


def prepare_record_witness(
    project: ProjectStore, spec: RecordWitnessSpec, *, code_revision: str
) -> JsonObject:
    """Execute a script from the exact native seed and publish its re-derived label witness.

    A witness measures state under the selected backend. It does not approve the predicate
    under qualification, establish production fidelity or supervise a final explanation.
    """
    preflight_capture(project, spec.capture)
    checkpoint = load_continuation_checkpoint(project.artifacts, spec.checkpoint)
    if sha256_json(checkpoint.task) != sha256_json(spec.capture.task) or not isinstance(
        checkpoint.world, ExecutableCheckpoint
    ):
        raise ValueError("record witness needs the exact public task and native seed checkpoint")
    with load_runtime_definition(
        project,
        runtime_file=spec.capture.runtime_file,
        runtime_sha256=spec.capture.runtime_sha256,
        configuration=spec.capture.configuration,
        probe_sha256=spec.capture.probe_sha256,
    ) as runtime:
        if not isinstance(runtime.world, ExecutableWorldRuntime):
            raise ValueError("record witnesses require a wholly native executable world")
        replay = RecordReplayRuntime(runtime.world, runtime.probe, runtime.probe_sha256)
        now = datetime.now(UTC)
        reference = execute_record_script(
            project.artifacts,
            replay,
            task=checkpoint.task,
            baseline=checkpoint.world,
            actions=spec.actions,
            finish_tick=spec.finish_tick,
            sources=(*spec.capture.sources, spec.checkpoint),
            created_at=now,
            code_revision=code_revision,
        )
        execution = load_record_execution(project.artifacts, reference)
        witness = record_label_witness(
            project.artifacts,
            specification=spec.specification,
            outcome=execution.outcome,
            receipt=reference,
            record_runtime=replay,
            created_at=now,
            code_revision=code_revision,
        )
    return {
        "receipt": reference.model_dump(mode="json"),
        "witness": witness.model_dump(mode="json"),
        "outcome": execution.outcome,
        "scope": "native state/effects under the selected backend; no production fidelity claim",
    }


def saved_record_scenario(
    project: ProjectStore, spec: RecordScenarioSpec, *, code_revision: str
) -> RecordScenarioResult | None:
    """Replay a saved seed without importing runtime code or repeating setup."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    result = RecordScenarioResult.model_validate_json(
        store.read_bytes(identity + "-result", "result.json")
    )
    registration = store.read(identity).manifest
    if (
        artifact_input(registration) != result.specification
        or RecordScenarioSpec.model_validate_json(store.read_bytes(identity, "specification.json"))
        != spec
    ):
        raise ValueError("saved scenario does not reproduce its specification")
    expected_inputs = tuple(
        sorted(
            (result.specification, result.execution, result.checkpoint), key=lambda r: r.artifact_id
        )
    )
    if store.read(identity + "-result").manifest.inputs != expected_inputs:
        raise ValueError("saved scenario lost a declared dependency")
    execution = load_record_execution(store, result.execution)
    setup_task = spec.capture.task.model_copy(
        update={"tools": (*spec.capture.task.tools, *spec.setup_tools)}
    )
    if (
        sha256_json(execution.task) != sha256_json(setup_task)
        or sha256_json([step.action.model_dump(mode="json") for step in execution.steps])
        != sha256_json([action.model_dump(mode="json") for action in spec.setup_actions])
        or execution.finish_tick != spec.finish_tick
        or set(execution.sources) != {*spec.capture.sources, result.specification}
        or registration.inputs
        != tuple(sorted(spec.capture.sources, key=lambda row: row.artifact_id))
    ):
        raise ValueError("scenario receipt does not reproduce its registered setup and sources")
    checkpoint = load_continuation_checkpoint(store, result.checkpoint)
    require_development_task(store, checkpoint.task)
    if sha256_json(checkpoint.world) != sha256_json(load_world_image(store, execution.final)):
        raise ValueError("scenario seed differs from its reached native state")
    expected_task = _public_task(spec, execution.outcome)
    actor = ChatAgentRuntime(
        maximum_model_calls=spec.capture.maximum_model_calls,
        system_prompt=spec.capture.system_prompt,
    )
    if sha256_json(checkpoint.task) != sha256_json(expected_task) or sha256_json(
        checkpoint.actor
    ) != sha256_json(actor.initial_checkpoint(expected_task)):
        raise ValueError("scenario public initialization or actor policy changed")
    if set(checkpoint.sources) != {*spec.capture.sources, result.specification, result.execution}:
        raise ValueError("scenario checkpoint lost its setup provenance")
    return result


def _public_task(spec: RecordScenarioSpec, outcome: JsonObject) -> TaskCase:
    """Derive only selected initialization facts, retaining the original source lineage."""
    context = dict(spec.capture.task.initial_context)
    for key, pointer in spec.public_fields.items():
        try:
            context[key] = resolve(outcome.get("final_state"), pointer)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"public initialization field {key!r} is unavailable at {pointer}"
            ) from error
    return spec.capture.task.model_copy(update={"initial_context": context})


def prepare_record_scenario(
    project: ProjectStore, spec: RecordScenarioSpec, *, code_revision: str
) -> RecordScenarioResult:
    """Execute declared setup twice and publish a frozen seed with no actor/model calls.

    Unknown startup intents are retained and never implicitly repeated. A failed preparation
    requires a new explicit definition after inspecting its failure, as initial capture does.
    """
    store = project.artifacts
    identity = spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_record_scenario(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_capture(project, spec.capture)
        if store.exists(identity):
            raise ValueError(
                "scenario has an unfinished startup intent; inspect it and "
                "register a new definition"
            )
        now = datetime.now(UTC)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision=code_revision,
            inputs=tuple(sorted(spec.capture.sources, key=lambda r: r.artifact_id)),
        )
        registration = artifact_input(
            store.write_json(
                artifact_id=identity,
                artifact_type="reachable-record-scenario",
                envelope=envelope,
                files={"specification.json": spec},
            )
        )
        with load_runtime_definition(
            project,
            runtime_file=spec.capture.runtime_file,
            runtime_sha256=spec.capture.runtime_sha256,
            configuration=spec.capture.configuration,
            probe_sha256=spec.capture.probe_sha256,
        ) as runtime:
            if not isinstance(runtime.world, ExecutableWorldRuntime):
                raise ValueError("record scenarios require a wholly native executable world")
            require_record_runtime(runtime.world)
            setup_task = spec.capture.task.model_copy(
                update={"tools": (*spec.capture.task.tools, *spec.setup_tools)}
            )
            with runtime.world.open(setup_task) as world:
                baseline = world.checkpoint()
            execution_ref = execute_record_script(
                store,
                RecordReplayRuntime(runtime.world, runtime.probe, runtime.probe_sha256),
                task=setup_task,
                baseline=baseline,
                actions=spec.setup_actions,
                finish_tick=spec.finish_tick,
                sources=(*spec.capture.sources, registration),
                created_at=now,
                code_revision=code_revision,
            )
        execution = load_record_execution(store, execution_ref)
        if any(step.reply.is_error for step in execution.steps):
            raise ValueError("scenario setup returned a tool error; inspect the execution receipt")
        final = load_world_image(store, execution.final)
        if not isinstance(final, ExecutableCheckpoint):
            raise ValueError("scenario produced a non-native checkpoint")
        if any(event.status != "applied" for event in final.fired):
            raise ValueError("scenario setup contains unsuccessful scheduled events")
        task = _public_task(spec, execution.outcome)
        actor = ChatAgentRuntime(
            maximum_model_calls=spec.capture.maximum_model_calls,
            system_prompt=spec.capture.system_prompt,
        )
        checkpoint = persist_continuation_checkpoint(
            store,
            ContinuationCheckpoint(
                task=task,
                actor=actor.initial_checkpoint(task),
                world=final,
                sources=(*spec.capture.sources, registration, execution_ref),
            ),
            created_at=now,
            code_revision=code_revision,
        )
        result = RecordScenarioResult(
            specification=registration, execution=execution_ref, checkpoint=checkpoint
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="record-scenario-result",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted(
                            (registration, execution_ref, checkpoint), key=lambda r: r.artifact_id
                        )
                    )
                }
            ),
            files={"result.json": result},
        )
        return result

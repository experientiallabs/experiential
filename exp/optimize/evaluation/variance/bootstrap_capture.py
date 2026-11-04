"""Capture an accepted start, validate it by its provenance and register its mining job.

The start is captured through the existing machinery: the accepted draft becomes a
:class:`~exp.optimize.evaluation.executable.grounded.GroundedRuntimeDefinition` over the build's
world model with the family's rule book (and rule memory set) bound, and
:func:`~exp.optimize.evaluation.continuations.capture.capture_initial_world` captures its
initial checkpoint exactly once under the template's trusted runtime file.

Validation depends on what the start claims to be:

- ``observed``: the captured checkpoint must hold the recorded state snapshot and restore exactly
  through the runtime's own restore path
  (:func:`~exp.optimize.evaluation.executable.grounded.restores_exactly`).
- ``reconstructed``: the recorded prefix of the named source trace is executed in a fresh restore
  of the start; every step must agree with what was recorded (equal error status, and for a
  successful step the equal decoded observation).
- ``authored``: the reviewers' coherence verdict is already recorded; each reveal probe is
  executed once from its own fresh restore, and every first-reveal stance must find the reveals
  consistent with the fact ledger and instruction.

Executing replay and reveal calls in a learned world may buy simulator predictions within the
runtime's own provider allowance; the actor is never involved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, sha256_json
from exp.common.models import ToolCall, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.capture import CaptureDefinition, capture_initial_world
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    preflight_grounded_runtime,
    restores_exactly,
)
from exp.optimize.evaluation.executable.loader import load_runtime_definition
from exp.optimize.evaluation.variance.bootstrap_agents import (
    FamilyKnowledge,
    StartMaterial,
    check_reveals,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapSpec,
    BootstrapTask,
    FamilyInput,
    ReplayStep,
    Reveal,
    RuntimeTemplate,
    StartDraft,
    StartValidation,
)
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorld, ExecutableWorldRuntime
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint


def start_definition(
    template: RuntimeTemplate,
    family: FamilyInput,
    knowledge: FamilyKnowledge,
    task: BootstrapTask,
    draft: StartDraft,
) -> GroundedRuntimeDefinition:
    """Return the grounded runtime of an accepted draft over the family's frozen knowledge.

    A family with a frozen runtime keeps it exactly (world model, index binding, rule book,
    rule memory, limits) and replaces only the task, manifest, initial state and evaluator
    projection; instance facts of the template's own task never carry over to another task.
    Otherwise the definition is built from the template with the family's rule book and rule
    memory bound. The selected task keeps its identity, lineage, partition and
    sources; its instruction and tools are the draft's.

    Args:
        template: The spec's runtime template.
        family: The family input.
        knowledge: The family's knowledge.
        task: The selected task.
        draft: The accepted draft.

    Returns:
        The definition.
    """
    public = task.task.model_copy(update={"instruction": draft.instruction, "tools": draft.tools})
    start = {
        "task": public.model_dump(mode="json"),
        "manifest": draft.manifest.model_dump(mode="json"),
        "initial_state": draft.initial_state,
        "probe_paths": draft.probe_paths,
    }
    if family.runtime is not None:
        frozen = family.runtime.model_dump(mode="json", exclude={"facts"})
        return GroundedRuntimeDefinition.model_validate({**frozen, **start})
    bind = template.bind_rule_book
    return GroundedRuntimeDefinition.model_validate(
        {
            **start,
            "world_model": template.world_model.model_dump(mode="json"),
            "catalog_sha256": template.catalog_sha256,
            "seed": template.seed,
            "maximum_output_tokens": template.maximum_output_tokens,
            "maximum_reasks": template.maximum_reasks,
            "maximum_physical_attempts": template.maximum_physical_attempts,
            "deadline": template.deadline.isoformat(),
            "rule_book": knowledge.rulebook.model_dump(mode="json") if bind else None,
            "tag_retrieval": None
            if template.tag_retrieval is None
            else template.tag_retrieval.model_dump(mode="json"),
            "rule_memory": None
            if not bind or knowledge.rule_memory is None
            else knowledge.rule_memory.model_dump(mode="json"),
        }
    )


def capture_start(
    project: ProjectStore,
    spec: BootstrapSpec,
    template: RuntimeTemplate,
    definition: GroundedRuntimeDefinition,
    sources: Sequence[ArtifactInput],
    *,
    code_revision: str,
) -> ArtifactInput:
    """Capture, or replay, the start's initial checkpoint through the capture owner.

    Args:
        project: Project receiving the checkpoint.
        spec: The registration.
        template: The family's runtime template.
        definition: The start's runtime.
        sources: Saved evidence the start was authored, reviewed and researched from.
        code_revision: Producing source identity.

    Returns:
        The saved continuation checkpoint.
    """
    preflight_grounded_runtime(project, definition)
    capture = CaptureDefinition(
        task=definition.task,
        sources=tuple(sorted(set(sources), key=lambda ref: ref.artifact_id)),
        runtime_file=template.runtime_file,
        runtime_sha256=template.runtime_sha256,
        configuration=definition.model_dump(mode="json"),
        probe_sha256=definition.probe_sha256,
        maximum_model_calls=spec.job.maximum_model_calls,
        system_prompt=spec.job.system_prompt,
    )
    return capture_initial_world(project, capture, code_revision=code_revision)


def validate_start(
    project: ProjectStore,
    spec: BootstrapSpec,
    template: RuntimeTemplate,
    material: StartMaterial,
    draft: StartDraft,
    definition: GroundedRuntimeDefinition,
    checkpoint: ArtifactInput,
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> StartValidation:
    """Validate a captured start the way its provenance requires.

    Args:
        project: Project holding the start.
        spec: The registration.
        template: The family's runtime template.
        material: What the start was authored from.
        draft: The accepted draft.
        definition: The start's runtime.
        checkpoint: The captured start.
        identity: Task bootstrap identity.
        sources: Saved artifacts the material came from.
        code_revision: Producing source identity.

    Returns:
        The validation.
    """
    if draft.provenance == "observed":
        assert material.recorded_state is not None
        held = restores_exactly(project, definition, checkpoint, material.recorded_state)
        detail = (
            "the start holds the recorded state and restores exactly"
            if held
            else "the start does not hold, or does not restore, the recorded state exactly"
        )
        return StartValidation(
            provenance="observed", method="exact_restore", passed=held, detail=detail
        )
    if draft.provenance == "reconstructed":
        assert draft.replay_trace is not None
        steps = material.recorded_prefix(draft.replay_trace)[: draft.prefix_steps]
        replay = replay_prefix(project, template, definition, checkpoint, steps)
        disagree = [row.locator for row in replay if not row.agrees]
        detail = (
            f"{len(replay)} recorded steps of {draft.replay_trace} replay to their observations"
            if not disagree
            else f"replay disagrees with the recorded prefix at {disagree}"
        )
        return StartValidation(
            provenance="reconstructed",
            method="prefix_replay",
            passed=not disagree,
            detail=detail,
            replay=replay,
        )
    reveals = first_reveals(project, template, definition, checkpoint, draft)
    checks = check_reveals(
        project,
        spec,
        material,
        draft,
        reveals,
        identity=identity,
        sources=sources,
        code_revision=code_revision,
    )
    problems = [
        f"{draw.variant}: {draw.failure}"
        if answer is None
        else f"{draw.variant}: {'; '.join(row.explanation for row in answer.conflicts)}"
        for draw, answer in checks
        if answer is None or not answer.consistent
    ]
    detail = (
        "the reviewers accepted the start's coherence and every stance finds its first reveals "
        "consistent with its facts"
        if not problems
        else "first reveals contradict the start: " + " | ".join(problems)
    )
    return StartValidation(
        provenance="authored",
        method="coherence_and_first_reveal",
        passed=not problems,
        detail=detail,
        reveals=reveals,
        checks=tuple(draw for draw, _ in checks),
    )


def replay_prefix(
    project: ProjectStore,
    template: RuntimeTemplate,
    definition: GroundedRuntimeDefinition,
    checkpoint: ArtifactInput,
    steps: Sequence[JsonObject],
) -> tuple[ReplayStep, ...]:
    """Execute recorded steps in order in one fresh restore of the start and compare each.

    Args:
        project: Project holding the start.
        template: The family's runtime template.
        definition: The start's runtime.
        checkpoint: The captured start.
        steps: Recorded trajectory steps, as the trajectory view renders them.

    Returns:
        One comparison per step.
    """
    rows: list[ReplayStep] = []
    with _restored(project, template, definition, checkpoint) as session:
        for index, step in enumerate(steps):
            action = step["action"]
            assert isinstance(action, dict)
            arguments = action.get("arguments")
            call = ToolCall(
                name=str(action["name"]),
                call_id=f"replay-{index}",
                arguments=arguments if isinstance(arguments, dict) else {},
            )
            observation = session.execute(call)
            replayed = _decode(observation)
            recorded_error = step.get("failure") is not None
            agrees = observation.is_error == recorded_error and (
                recorded_error or replayed == step.get("observation")
            )
            rows.append(
                ReplayStep(
                    locator=str(step["locator"]),
                    action=action,
                    recorded=step.get("observation"),
                    recorded_error=recorded_error,
                    replayed=replayed,
                    replayed_error=observation.is_error,
                    agrees=agrees,
                )
            )
    return tuple(rows)


def first_reveals(
    project: ProjectStore,
    template: RuntimeTemplate,
    definition: GroundedRuntimeDefinition,
    checkpoint: ArtifactInput,
    draft: StartDraft,
) -> tuple[Reveal, ...]:
    """Execute each reveal probe once, each from its own fresh restore of the start.

    Returns:
        What each probe revealed, in probe order.
    """
    rows: list[Reveal] = []
    for index, probe in enumerate(draft.reveal_probes):
        with _restored(project, template, definition, checkpoint) as session:
            observation = session.execute(
                ToolCall(name=probe.name, call_id=f"reveal-{index}", arguments=probe.arguments)
            )
        rows.append(
            Reveal(
                probe=index,
                name=probe.name,
                arguments=probe.arguments,
                content=_decode(observation),
                is_error=observation.is_error,
            )
        )
    return tuple(rows)


def mining_job(
    project: ProjectStore,
    spec: BootstrapSpec,
    runtime: RuntimeTemplate,
    definition: GroundedRuntimeDefinition,
    checkpoint: ArtifactInput,
    draft: StartDraft,
) -> ContinuationJob:
    """Freeze the start's mining job and check it is runnable, without loading runtime code.

    Args:
        project: Project holding the start and catalog.
        spec: The registration.
        runtime: The family's runtime template.
        definition: The start's runtime.
        checkpoint: The captured start.
        draft: The accepted draft, whose verifier the study declares.

    Returns:
        The job, preflighted.
    """
    template = spec.job
    catalog = load_model_catalog(project.model_catalog_path)
    snapshot = RuntimeModelCatalog(catalog, environment={}).snapshot(template.actor_alias)[0]
    study = ContinuationStudySpec(
        checkpoint=checkpoint,
        actor_model=snapshot,
        verifier=draft.verifier,
        probe_sha256=definition.probe_sha256,
        repeats=template.repeats,
        selector=f"start bootstrap: {draft.provenance} initial start of {definition.task.task_id}",
        temperature=template.temperature,
        maximum_output_tokens=template.maximum_output_tokens,
    )
    job = ContinuationJob(
        study=study,
        runtime_file=runtime.runtime_file,
        runtime_sha256=runtime.runtime_sha256,
        configuration=definition.model_dump(mode="json"),
        actor_alias=template.actor_alias,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        maximum_model_calls=template.maximum_model_calls,
        system_prompt=template.system_prompt,
        maximum_physical_attempts=template.maximum_physical_attempts,
        deadline=template.deadline,
    )
    preflight_continuation_job(project, job)
    return job


@contextmanager
def _restored(
    project: ProjectStore,
    template: RuntimeTemplate,
    definition: GroundedRuntimeDefinition,
    checkpoint: ArtifactInput,
) -> Iterator[ExecutableWorld]:
    """Open the template runtime and restore the start into one fresh, isolated session."""
    start = load_continuation_checkpoint(project.artifacts, checkpoint)
    world = start.world
    if not isinstance(world, ExecutableCheckpoint):
        raise ValueError("a bootstrapped start must be an executable checkpoint")
    with load_runtime_definition(
        project,
        runtime_file=template.runtime_file,
        runtime_sha256=template.runtime_sha256,
        configuration=definition.model_dump(mode="json"),
        probe_sha256=definition.probe_sha256,
    ) as runtime:
        if not isinstance(runtime.world, ExecutableWorldRuntime):
            raise ValueError("the template runtime does not serve a restorable executable world")
        with runtime.world.restore(start.world_task, world) as session:
            yield session


def _decode(observation: Observation) -> JsonValue:
    """Return an observation's content, decoded when it is JSON."""
    try:
        value: JsonValue = json.loads(observation.content)
    except ValueError:
        return observation.content
    return value

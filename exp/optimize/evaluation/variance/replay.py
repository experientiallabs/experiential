"""Replay a variant that forks before its decision back to the decision, by forced recorded calls.

A variant whose edited fact was revealed before the decision forks at an earlier boundary
(:func:`~exp.optimize.evaluation.variance.variants.build_variant_checkpoint`). Forcing an option
there would not force the decision, so the turns between the fork and the decision are replayed
first. Replay never samples the actor: each tool call the original trajectory's actor recorded in
those turns is imposed, in order, through the scripted action-intervention owner
(:func:`~exp.optimize.evaluation.continuations.intervention.capture_action_intervention`), which
writes a durable intent before every call and never redispatches an unknown outcome. The world
therefore reaches the decision in the edited variant, and the decision checkpoint keeps the
actor's recorded messages verbatim while every tool reply is the real observation the edited
world returned. The actor's call count is the original decision's, so the remaining allowance at
the replayed decision is exactly the original one.

A forced prefix is only coherent when the recorded calls still make sense in the edited world.
The replay is marked ``replay_diverged``, saves no decision checkpoint and is never contrasted
when any of these holds:

- a recorded call is refused before execution (not a visible tool, schema-invalid arguments, a
  call identity that collides with the retained prefix);
- a recorded call raises, or its observation in the edited world is an error;
- the actor's own recorded text in a replayed turn (assistant content or tool-call arguments)
  states a value the scenario's edits replaced, so its recorded calls depend on the original fact;
- the replayed calls changed an edited path, so the edits no longer describe the decision's
  before-state;
- the replayed decision checkpoint does not hold every baseline part of the restorable state
  (:func:`~exp.optimize.evaluation.variance.variants.freeze_state`).

A diverged replay is saved and replayed as a finding, never coerced into alignment: no recorded
call is altered, no observation is overwritten with the original and no state is patched.

Dependence the actor never wrote down (a call chosen because of an old value it did not repeat)
cannot be detected from the recorded messages; the result's scope says so.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.intervention import (
    ActionInterventionSpec,
    capture_action_intervention,
    preflight_intervention,
)
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.contracts import (
    BASELINE_STATE,
    FrozenState,
    Scenario,
    json_scalars,
    message_texts,
    resolve_pointer,
    revealing_texts,
)
from exp.optimize.evaluation.variance.records import bound_job
from exp.optimize.evaluation.variance.variants import (
    VariantCheckpoint,
    freeze_state,
    grounded_state,
)
from exp.runtime.environments.interface import Observation
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)

REPLAY_TYPE = "decision-replay"
"""Artifact type of a saved replay to the decision."""

ReplayStatus = Literal["replayed", "replay_diverged"]
"""Whether the forced prefix reached the decision coherently."""


class ReplayedCall(ContractModel):
    """One recorded actor call imposed in the edited world.

    Attributes:
        turn: Boundary index the call's turn ends at; the fork is the boundary before the first.
        call: The recorded call, imposed unchanged.
        original: The tool reply the actor saw for it in the original trajectory.
        observation: The edited world's reply, when the call executed.
        intervention: The saved action intervention, when the call executed.
        failure: Why the call was refused or did not execute, when it did not.
    """

    turn: int = Field(ge=1)
    call: ToolCall
    original: str | None = None
    observation: Observation | None = None
    intervention: ArtifactInput | None = None
    failure: str | None = None


class DecisionReplay(ContractModel):
    """A variant replayed from its fork to its decision, or why the replay diverged.

    Attributes:
        scenario_id: The scenario the variant realizes.
        fork: The persisted variant checkpoint the replay started from.
        decision: The decision's checkpoint in the original trajectory.
        decision_step: Boundary index of the decision.
        status: ``replayed`` when the forced prefix reached the decision coherently.
        checkpoint: The edited world at the decision with the replayed conversation; None when
            the replay diverged.
        calls: Every recorded call in order, as far as the replay got.
        problems: Why the replay diverged; empty when it did not.
        frozen: What the replayed decision checkpoint holds of the restorable state, once built;
            instance facts are bound by the world, not this replay, so they are not recorded.
        scope: What the replay does and does not establish.
    """

    scenario_id: str
    fork: ArtifactInput
    decision: ArtifactInput
    decision_step: int = Field(ge=1)
    status: ReplayStatus
    checkpoint: ArtifactInput | None = None
    calls: tuple[ReplayedCall, ...] = ()
    problems: tuple[str, ...] = ()
    frozen: FrozenState | None = None
    scope: str = (
        "recorded actor calls imposed in the edited world, not a fresh actor draw; observations "
        "are the edited world's; dependence on a replaced fact the actor never wrote down is not "
        "detected"
    )


def replay_identity(
    variant: VariantCheckpoint,
    boundaries: tuple[ArtifactInput, ...],
    scenario: Scenario,
    job: ContinuationJob,
    *,
    code_revision: str,
) -> str:
    """Freeze the replay before any runtime code loads.

    Args:
        variant: The persisted variant.
        boundaries: The original trajectory's boundaries from the fork to the decision.
        scenario: The variant's scenario.
        job: Frozen job supplying the runtime and actor policy.
        code_revision: Installed implementation identity.

    Returns:
        The stable artifact identity of the saved replay.
    """
    return stable_id(
        "decision-replay",
        {
            "variant": None if variant.reference is None else variant.reference.model_dump(),
            "boundaries": [reference.model_dump(mode="json") for reference in boundaries],
            "scenario": sha256_json(scenario),
            "job": job.implementation(code_revision),
        },
    )


def replay_to_decision(
    project: ProjectStore,
    *,
    variant: VariantCheckpoint,
    branch_checkpoints: tuple[ArtifactInput, ...],
    scenario: Scenario,
    job: ContinuationJob,
    code_revision: str,
) -> DecisionReplay:
    """Impose the original trajectory's recorded calls from the fork until the decision.

    Args:
        project: Trusted project holding the trajectory and the variant.
        variant: A persisted variant that forks before its decision.
        branch_checkpoints: Every saved boundary of the trajectory in order, start first.
        scenario: The variant's scenario, whose edits must still hold at the decision.
        job: Frozen job supplying the runtime file and actor policy; its study is rebound to
            each intermediate checkpoint.
        code_revision: Installed implementation identity.

    Returns:
        The replay, saved as a ``decision-replay`` artifact and replayed exactly once saved.

    Raises:
        ValueError: The variant is not persisted, does not fork before its decision, or the
            boundaries are not the trajectory it was built from.
    """
    reference, fork, replay = variant.reference, variant.fork_turn, variant.replay_turns
    if reference is None or fork is None or not replay:
        raise ValueError(
            f"variant {variant.scenario_id} is unpersisted or forks at its decision; only a "
            "persisted variant with replay turns is replayed"
        )
    step = fork + replay
    if (
        step >= len(branch_checkpoints)
        or branch_checkpoints[step] != variant.base
        or branch_checkpoints[fork] != variant.fork
    ):
        raise ValueError(
            "boundaries are not the trajectory the variant was built from; pass the same "
            "checkpoints, start first"
        )
    boundaries = branch_checkpoints[fork : step + 1]
    store = project.artifacts
    identity = replay_identity(variant, boundaries, scenario, job, code_revision=code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            return DecisionReplay.model_validate_json(store.read_bytes(identity, "replay.json"))
        result = _replay(project, variant, reference, boundaries, scenario, job, code_revision)
        inputs = {reference, *boundaries}
        inputs |= {call.intervention for call in result.calls if call.intervention is not None}
        if result.checkpoint is not None:
            inputs.add(result.checkpoint)
        store.write_json(
            artifact_id=identity,
            artifact_type=REPLAY_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"replay.json": result},
        )
        return result


def _replay(
    project: ProjectStore,
    variant: VariantCheckpoint,
    reference: ArtifactInput,
    boundaries: tuple[ArtifactInput, ...],
    scenario: Scenario,
    job: ContinuationJob,
    code_revision: str,
) -> DecisionReplay:
    """Check the recorded turns, impose their calls and build the decision checkpoint.

    Args:
        project: Trusted project.
        variant: The variant.
        reference: Its persisted checkpoint.
        boundaries: The original boundaries from the fork to the decision.
        scenario: The variant's scenario.
        job: Frozen job.
        code_revision: Installed implementation identity.

    Returns:
        The unsaved replay.
    """
    store = project.artifacts
    fork = variant.fork_turn or 0
    loaded = [load_continuation_checkpoint(store, boundary) for boundary in boundaries]
    turns = [
        (fork + index, loaded[index].actor.messages[len(loaded[index - 1].actor.messages) :])
        for index in range(1, len(loaded))
    ]
    result = DecisionReplay(
        scenario_id=variant.scenario_id,
        fork=reference,
        decision=boundaries[-1],
        decision_step=fork + len(turns),
        status="replay_diverged",
    )
    replaced = _replaced_values(loaded[0], scenario)
    stated = [
        f"turn {turn}: the actor's recorded messages state replaced {', '.join(found)}, so its "
        "recorded calls depend on the original fact"
        for turn, messages in turns
        if (
            found := revealing_texts(
                message_texts(m for m in messages if m.role == "assistant"),
                replaced,
                frozenset(),
            )
        )
    ]
    if stated:
        return result.model_copy(update={"problems": tuple(stated)})
    current = reference
    calls: list[ReplayedCall] = []
    for turn, messages in turns:
        replies = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
        for message in messages:
            if message.assistant_action is None:
                continue
            for call in message.assistant_action.tool_calls:
                row, current = _impose(
                    project, job, current, turn, call, replies.get(call.call_id), code_revision
                )
                calls.append(row)
                if row.failure is not None:
                    return result.model_copy(
                        update={
                            "calls": tuple(calls),
                            "problems": (f"turn {turn} call {call.call_id}: {row.failure}",),
                        }
                    )
    result = result.model_copy(update={"calls": tuple(calls)})
    final = load_continuation_checkpoint(store, current)
    moved = _moved_edits(final, scenario)
    if moved:
        return result.model_copy(
            update={
                "problems": (
                    f"replayed calls changed edited paths {moved}, so the edits no longer "
                    "describe the decision's before-state",
                )
            }
        )
    observed = {row.call.call_id: row.observation for row in calls}
    rebuilt = list(loaded[0].actor.messages)
    for _, messages in turns:
        for message in messages:
            seen = observed.get(message.tool_call_id) if message.role == "tool" else None
            rebuilt.append(
                message if seen is None else message.model_copy(update={"content": seen.content})
            )
    decision = loaded[-1]
    replayed = ContinuationCheckpoint(
        task=decision.task,
        environment_task=decision.environment_task,
        actor=decision.actor.model_copy(update={"messages": tuple(rebuilt)}),
        world=final.world,
        sources=tuple(
            sorted({reference, current, boundaries[-1]}, key=lambda ref: ref.artifact_id)
        ),
    )
    frozen = freeze_state(replayed, None)
    result = result.model_copy(update={"frozen": frozen})
    missing = frozen.unrestorable(BASELINE_STATE)
    if missing:
        return result.model_copy(
            update={"problems": ("incomplete decision state: " + "; ".join(missing),)}
        )
    checkpoint = persist_continuation_checkpoint(
        store, replayed, created_at=datetime.now(UTC), code_revision=code_revision
    )
    return result.model_copy(update={"status": "replayed", "checkpoint": checkpoint})


def _impose(
    project: ProjectStore,
    job: ContinuationJob,
    current: ArtifactInput,
    turn: int,
    call: ToolCall,
    original: str | None,
    code_revision: str,
) -> tuple[ReplayedCall, ArtifactInput]:
    """Impose one recorded call at the current checkpoint through the intervention owner.

    Args:
        project: Trusted project.
        job: Frozen job, rebound to ``current``.
        current: Checkpoint the call is imposed at.
        turn: The call's turn.
        call: The recorded call.
        original: The reply the actor saw for it originally.
        code_revision: Installed implementation identity.

    Returns:
        The replayed call and the checkpoint after it, which stays ``current`` on failure.
    """
    row = ReplayedCall(turn=turn, call=call, original=original)
    spec = ActionInterventionSpec(
        base=bound_job(job, current),
        action=call,
        rationale=f"replay recorded call {call.call_id} of turn {turn} toward the decision",
    )
    try:
        preflight_intervention(project, spec)
    except ValueError as error:
        return row.model_copy(update={"failure": f"refused: {_reason(error)}"}), current
    try:
        effect = capture_action_intervention(project, spec, code_revision=code_revision)
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"failure": f"did not execute: {_reason(error)}"}), current
    saved = artifact_input(
        project.artifacts.read(spec.identity(code_revision) + "-result").manifest
    )
    row = row.model_copy(update={"observation": effect.observation, "intervention": saved})
    if effect.observation.is_error:
        return row.model_copy(
            update={"failure": "the edited world returned an error observation"}
        ), current
    return row, effect.checkpoint


def _replaced_values(fork: ContinuationCheckpoint, scenario: Scenario) -> tuple[JsonValue, ...]:
    """Return the original scalars at each edited path that the edit's value does not keep.

    Args:
        fork: The original trajectory's checkpoint at the fork, before any edit.
        scenario: The scenario.

    Returns:
        Distinct replaced scalars.
    """
    state = grounded_state(fork.world) or {}
    found: dict[str, JsonValue] = {}
    for edit in scenario.edits:
        kept = {json.dumps(value) for value in json_scalars(edit.value)}
        exists, value = resolve_pointer(state, edit.path)
        if exists:
            for item in json_scalars(value):
                if json.dumps(item) not in kept:
                    found.setdefault(json.dumps(item), item)
    return tuple(found.values())


def _moved_edits(final: ContinuationCheckpoint, scenario: Scenario) -> list[str]:
    """Return the edited paths whose value at the replayed decision differs from the edit."""
    state = grounded_state(final.world) or {}
    return sorted(
        edit.path
        for edit in scenario.edits
        if not _holds(resolve_pointer(state, edit.path), edit.value)
    )


def _holds(found: tuple[bool, JsonValue], value: JsonValue) -> bool:
    """Whether a resolved pointer exists and holds exactly a JSON value."""
    exists, current = found
    return exists and json.dumps(current, sort_keys=True) == json.dumps(value, sort_keys=True)


def _reason(error: BaseException) -> str:
    """Return a bounded failure reason naming the error type."""
    return (type(error).__name__ + ": " + str(error))[:2000]

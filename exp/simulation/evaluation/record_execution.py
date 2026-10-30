"""Replay-verified native record execution evidence, separate from production-fidelity claims.

Receipts are private evidence. Verification requires an explicitly supplied native runtime;
loading a receipt never imports or executes code named by its contents. The existing kernel
owns execution, rollback, clocks, events and randomness. No model or judge runs here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.record_world import RecordWorldFactory
from exp.runtime.environments.world_artifacts import load_world_image, persist_world_image
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.development_scope import require_development_task

RECORD_RECEIPT_TYPE = "native-record-execution"


@dataclass(frozen=True)
class RecordReplayRuntime:
    """Explicit trusted native runtime and its independently selected private state reader.

    Attributes:
        world: Existing isolated, restorable kernel, with wholly native state ownership.
        probe: Evaluator-only projection from a complete native checkpoint.
        probe_sha256: Frozen implementation/configuration identity of the state reader.
            The caller selects trusted code; receipts cannot supply executable source.
    """

    world: ExecutableWorldRuntime
    probe: Callable[[ExecutableCheckpoint], JsonObject]
    probe_sha256: Sha256


class RecordStep(ContractModel):
    """One exact attempted action and its committed kernel result.

    Attributes:
        action: Original action, including its logical identity.
        reply: Actual native reply, including business errors.
        after: Complete resulting image, clock, events and random position.
    """

    action: ToolCall
    reply: Observation
    after: ArtifactInput


class RecordExecution(ContractModel):
    """Bounded execution under a declared native implementation, not reconstructed truth.

    Attributes:
        authority: Local native record execution only.
        task: Tool capabilities used for this script, never a private expected answer.
        baseline: Exact starting executable checkpoint.
        steps: Ordered actions and results, including errors without implicit retries.
        finish_tick: Explicit final clock cutoff after all actions.
        final: Complete image after advancing to the cutoff.
        sources: Immutable source evidence and declared assumptions.
        backend_definition: Exact record factory identity used by the kernel.
        probe_sha256: Frozen evaluator-only projection definition.
        outcome: Private state projections and observed steps, derived from the native probe.
    """

    authority: Literal["native_record_execution"] = "native_record_execution"
    task: TaskCase
    baseline: ArtifactInput
    steps: tuple[RecordStep, ...] = Field(max_length=128)
    finish_tick: int = Field(ge=0, le=1_000_000, strict=True)
    final: ArtifactInput
    sources: tuple[ArtifactInput, ...] = Field(min_length=1)
    backend_definition: JsonObject
    probe_sha256: Sha256
    outcome: JsonObject


def require_record_runtime(runtime: ExecutableWorldRuntime) -> None:
    """Reject simulated owners, other backends and identities changed since construction."""
    if runtime.owner != "native":
        raise ValueError("record evidence requires an explicitly selected native runtime")
    if any(owner != "native" for owner in runtime.manifest.ownership().values()):
        raise ValueError("record evidence requires wholly native state; mixed evidence is separate")
    if (
        sha256_json(runtime.manifest) != runtime.manifest_sha256
        or sha256_json(runtime.factory.definition) != runtime.backend_sha256
    ):
        raise ValueError("record runtime definition changed after freezing")


def record_state(runtime: ExecutableWorldRuntime, checkpoint: ExecutableCheckpoint) -> JsonObject:
    """Read private native state without dispatching any tools or changing the checkpoint."""
    require_record_runtime(runtime)
    factory = runtime.factory
    if not isinstance(factory, RecordWorldFactory):
        raise ValueError(
            "SQLite projection requires a record factory; supply a native probe for other backends"
        )
    if (
        checkpoint.backend_sha256 != runtime.backend_sha256
        or checkpoint.manifest_sha256 != runtime.manifest_sha256
        or checkpoint.owner != "native"
        or checkpoint.ownership != runtime.manifest.ownership()
    ):
        raise ValueError("record projection belongs to a different native runtime")
    backend = factory.restore(checkpoint.image)
    try:
        if backend.capture() != checkpoint.image:
            raise ValueError("record image does not reproduce its declared native state")
        return backend.projection()
    finally:
        backend.close()


def _image(store: ArtifactStore, reference: ArtifactInput) -> ExecutableCheckpoint:
    """Require a complete native image rather than an unrelated or mixed checkpoint."""
    image = load_world_image(store, reference)
    if not isinstance(image, ExecutableCheckpoint):
        raise ValueError("record evidence requires a native executable checkpoint")
    return image


def _execute(
    replay: RecordReplayRuntime,
    task: TaskCase,
    baseline: ExecutableCheckpoint,
    actions: tuple[ToolCall, ...],
    finish_tick: int,
) -> tuple[tuple[tuple[Observation, ExecutableCheckpoint], ...], ExecutableCheckpoint, JsonObject]:
    """Execute one finite script in an isolated restored world with a private state probe."""
    runtime = replay.world
    require_record_runtime(runtime)
    if task.reference_answer is not None or task.partition != "fit":
        raise ValueError("record scripts need a public development task without expected answers")
    if len(actions) > 128 or len({action.call_id for action in actions}) != len(actions):
        raise ValueError("record scripts need at most 128 actions with unique logical call IDs")
    allowed = {tool.name for tool in task.tools}
    if any(action.name not in allowed for action in actions):
        raise ValueError("record script calls a tool outside its declared task capability")
    expected_tick = baseline.clock + sum(runtime.manifest.tool(a.name).duration for a in actions)
    if not expected_tick <= finish_tick <= 1_000_000:
        raise ValueError(
            "finish_tick must follow the complete script within the finite clock bound"
        )
    initial = replay.probe(baseline.model_copy(deep=True))
    rows, observed = [], []
    with runtime.restore(task, baseline) as world:
        for action in actions:
            reply = world.execute(action)
            checkpoint = world.checkpoint()
            rows.append((reply, checkpoint))
            observed.append(
                {
                    "action": action.model_dump(mode="json"),
                    "reply": reply.model_dump(mode="json"),
                    "status": "valid",
                    "after": replay.probe(checkpoint.model_copy(deep=True)),
                }
            )
        world.advance(finish_tick - world.clock)
        final = world.checkpoint()
    outcome: JsonObject = {
        "initial_state": initial,
        "final_state": replay.probe(final.model_copy(deep=True)),
        "steps": observed,
        "final_answer": "",
        "stop_reason": "completed",
    }
    return tuple(rows), final, outcome


def execute_record_script(
    store: ArtifactStore,
    replay: RecordReplayRuntime,
    *,
    task: TaskCase,
    baseline: ExecutableCheckpoint,
    actions: tuple[ToolCall, ...],
    finish_tick: int,
    sources: tuple[ArtifactInput, ...],
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Replay twice from the same baseline before publishing a native execution receipt.

    This requires the existing isolated native kernel contract. External-effect transports
    cannot use this restart protocol. Failed setup never publishes a successful receipt.
    """
    require_development_task(store, task)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("record execution needs unique immutable source evidence")
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("record execution source differs from its declared digest")
    first = _execute(replay, task, baseline, actions, finish_tick)
    second = _execute(replay, task, baseline, actions, finish_tick)
    if (
        sha256_json(first[2]) != sha256_json(second[2])
        or sha256_json(first[1]) != sha256_json(second[1])
        or any(
            sha256_json(left) != sha256_json(right)
            for left_pair, right_pair in zip(first[0], second[0], strict=True)
            for left, right in zip(left_pair, right_pair, strict=True)
        )
    ):
        raise ValueError("record script does not replay exactly; no native receipt published")

    def save(image: ExecutableCheckpoint) -> ArtifactInput:
        """Persist one complete kernel checkpoint with the existing image owner."""
        return persist_world_image(store, image, created_at=created_at, code_revision=code_revision)

    rows, final, outcome = first
    receipt = RecordExecution(
        task=task,
        baseline=save(baseline),
        steps=tuple(
            RecordStep(action=action, reply=reply, after=save(after))
            for action, (reply, after) in zip(actions, rows, strict=True)
        ),
        finish_tick=finish_tick,
        final=save(final),
        sources=sources,
        backend_definition=replay.world.factory.definition,
        probe_sha256=replay.probe_sha256,
        outcome=outcome,
    )
    identity = stable_id(
        RECORD_RECEIPT_TYPE, {"receipt": sha256_json(receipt), "code": code_revision}
    )
    inputs = _inputs(receipt)
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=RECORD_RECEIPT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, inputs=inputs, code_revision=code_revision
            ),
            files={"receipt.json": receipt},
        )
    return artifact_input(store.read(identity).manifest)


def _inputs(receipt: RecordExecution) -> tuple[ArtifactInput, ...]:
    """Close provenance over source evidence and every retained native checkpoint."""
    return tuple(
        sorted(
            {
                receipt.baseline,
                receipt.final,
                *receipt.sources,
                *(step.after for step in receipt.steps),
            },
            key=lambda row: row.artifact_id,
        )
    )


def load_record_execution(store: ArtifactStore, reference: ArtifactInput) -> RecordExecution:
    """Check receipt bytes and provenance only; this does not grant native label authority."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != RECORD_RECEIPT_TYPE:
        raise ValueError("record execution reference differs from its exact saved receipt")
    receipt = RecordExecution.model_validate_json(
        store.read_bytes(reference.artifact_id, "receipt.json")
    )
    if manifest.inputs != _inputs(receipt) or reference.artifact_id != stable_id(
        RECORD_RECEIPT_TYPE, {"receipt": sha256_json(receipt), "code": manifest.code_revision}
    ):
        raise ValueError("record receipt identity or provenance differs")
    for source in manifest.inputs:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("record receipt dependency differs from its saved digest")
    return receipt


def verify_record_execution(
    store: ArtifactStore, reference: ArtifactInput, replay: RecordReplayRuntime
) -> RecordExecution:
    """Re-execute under an explicitly trusted runtime and compare every reply and checkpoint."""
    receipt = load_record_execution(store, reference)
    require_development_task(store, receipt.task)
    require_record_runtime(replay.world)
    if (
        receipt.backend_definition != replay.world.factory.definition
        or receipt.probe_sha256 != replay.probe_sha256
    ):
        raise ValueError("record witness runtime differs from the producing backend definition")
    rows, final, outcome = _execute(
        replay,
        receipt.task,
        _image(store, receipt.baseline),
        tuple(step.action for step in receipt.steps),
        receipt.finish_tick,
    )
    if sha256_json(outcome) != sha256_json(receipt.outcome) or sha256_json(final) != sha256_json(
        _image(store, receipt.final)
    ):
        raise ValueError("record execution does not reproduce its outcome or final checkpoint")
    for step, (reply, after) in zip(receipt.steps, rows, strict=True):
        if sha256_json(reply) != sha256_json(step.reply) or sha256_json(after) != sha256_json(
            _image(store, step.after)
        ):
            raise ValueError(
                "record execution does not reproduce an intermediate reply or checkpoint"
            )
    return receipt

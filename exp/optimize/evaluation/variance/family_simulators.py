"""One simulator per environment family, shared by every task and isolated attempt in it.

An environment family (:mod:`exp.simulation.mining.environment_families`) is a set of traces
whose tool contracts and world semantics are compatible. Everything learned about that
environment is learned once per family and reused:

- **one rule memory**: :func:`learn_family_rules` runs an injected learner (normally the rules
  stage over the family's ``rules_traces``) once per family and learner identity, saves the
  result, and replays it with no call;
- **one simulator definition**: :class:`FamilySimulator` is the stable contract later stages
  consume. It binds the family identity and status, member traces, reconciled tool contracts,
  shared state domains, the scoped corpus and index, the world model, the frozen rule book and
  rule memory, the frozen runtime template and the family's tasks with their starting states.
  Its identity covers everything except the task list, so adding a task never re-keys it.

Tasks share the simulator: :func:`definition_for` derives one task's
:class:`~exp.optimize.evaluation.executable.grounded.GroundedRuntimeDefinition` from the frozen
template by replacing only the task and its initial state. Attempts never share state:
:func:`run_isolated_attempts` restores the same saved start into a fresh, isolated world for each
attempt through the existing continuation restore owner, and verifies the start was not mutated.
Five actors times sixteen attempts are eighty independent runs against one simulator definition,
not eighty learned simulators.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.continuations.study import (
    CheckpointSession,
    WorldRuntime,
    restore_continuation_world,
)
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    RuleMemoryBinding,
)
from exp.optimize.evaluation.variance.build_records import FamilyScope
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.mining.environment_families import (
    EnvironmentFamily,
    FamilyStatus,
    FamilyTask,
)
from exp.simulation.mining.environment_review import StateDomain
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.rules.model import ToolContract

FAMILY_SIMULATOR_PROTOCOL = "family-simulator-v1"
"""Version of the shared simulator contract and attempt isolation; bump when either changes."""

SIMULATOR_TYPE = "environment-family-simulator"
"""Artifact type of a saved :class:`FamilySimulator` (``simulator.json``)."""

LEARNING_TYPE = "environment-family-learning"
"""Artifact type of one family's saved learned rules (``learned.json``)."""


class FamilySimulator(ContractModel):
    """The one simulator definition of an environment family.

    Attributes:
        family_id: The family identity, the key of every per-family input.
        status: ``compatible``, or ``admitted_uncertain`` when explicitly admitted.
        cluster_id: The tag cluster it came from.
        path: Its position in the review tree; a ``/`` means it came from a split.
        members: Member world-learning trace identities.
        tool_contracts: Contracts parsed over every member, reconciled by tool name.
        state_domains: Shared state domains, as the reviewers named them.
        world_semantics: What the same actions do in this environment.
        corpus: The scoped corpus its index holds.
        index: Its scoped, tagged fit-only index.
        world_model: Its world model over that index.
        tag_retrieval: Its tag retrieval binding over placed probe tasks, when any.
        rules_traces: World-learning traces its rule memory is learned from.
        rule_book: Its frozen learned rule book, once learned.
        rule_memory: Its frozen rule memory set and recall policy, once learned.
        rules_report: The ``rules-stage-report`` its rules were learned in, when the learner
            ran the rules stage.
        runtime: Frozen runtime template every task definition derives from, once bound.
        tasks: Its tasks, each with its starting state.
        families: The saved ``environment-families`` record it came from.
    """

    protocol: str = FAMILY_SIMULATOR_PROTOCOL
    family_id: str
    status: FamilyStatus
    cluster_id: str
    path: str
    members: tuple[str, ...] = Field(min_length=1)
    tool_contracts: dict[str, ToolContract] = Field(default_factory=dict)
    state_domains: tuple[StateDomain, ...] = ()
    world_semantics: str = ""
    corpus: CorpusScope
    index: ArtifactInput
    world_model: ArtifactInput
    tag_retrieval: ArtifactInput | None = None
    rules_traces: tuple[str, ...]
    rule_book: ArtifactInput | None = None
    rule_memory: RuleMemoryBinding | None = None
    rules_report: ArtifactInput | None = None
    runtime: GroundedRuntimeDefinition | None = None
    tasks: tuple[FamilyTask, ...] = ()
    families: ArtifactInput

    def identity(self) -> str:
        """Return the shared definition's identity; tasks and the template's task are excluded."""
        runtime = None
        if self.runtime is not None:
            runtime = self.runtime.model_dump(mode="json", exclude={"task", "initial_state"})
        body = self.model_dump(mode="json", exclude={"tasks", "runtime"})
        return stable_id(SIMULATOR_TYPE, {"simulator": sha256_json(body), "runtime": runtime})

    def task(self, task_id: str) -> FamilyTask:
        """Return one task of the family.

        Raises:
            KeyError: The task is not in this family.
        """
        for row in self.tasks:
            if row.task_id == task_id:
                return row
        raise KeyError(f"task {task_id} is not in environment family {self.family_id}")


def family_simulator(
    family: EnvironmentFamily, scope: FamilyScope, families: ArtifactInput
) -> FamilySimulator:
    """Assemble a family's unlearned simulator from its review and its build scope.

    Raises:
        ValueError: The scope belongs to another family.
    """
    if scope.family_id != family.family_id:
        raise ValueError(f"scope {scope.family_id} is not family {family.family_id}'s scope")
    return FamilySimulator(
        family_id=family.family_id,
        status=family.status,
        cluster_id=family.cluster_id,
        path=family.path,
        members=family.members,
        tool_contracts=family.tool_contracts,
        state_domains=family.state_domains,
        world_semantics=family.world_semantics,
        corpus=scope.retrieval_scope,
        index=scope.index,
        world_model=scope.world_model,
        tag_retrieval=scope.tag_retrieval,
        rules_traces=scope.rules_traces,
        tasks=family.tasks,
        families=families,
    )


class FamilyRulesInput(ContractModel):
    """What one family's rule learning reads.

    Attributes:
        family_id: The family.
        traces: Its world-learning traces.
        tool_contracts: Its reconciled contracts.
    """

    family_id: str
    traces: tuple[str, ...]
    tool_contracts: dict[str, ToolContract]


class LearnedRules(ContractModel):
    """One family's learned rules.

    Attributes:
        rule_book: Saved rule book.
        rule_memory: Saved rule memory set with its recall policy.
        rules_report: Saved ``rules-stage-report``, when the learner ran the rules stage.
    """

    rule_book: ArtifactInput | None = None
    rule_memory: RuleMemoryBinding | None = None
    rules_report: ArtifactInput | None = None


class FamilyLearner(Protocol):
    """Learns one family's rules, normally by running the rules stage over its traces."""

    @property
    def identity(self) -> str:
        """Return the learner's registration identity (its spec, model and code)."""
        ...

    def __call__(self, inputs: FamilyRulesInput) -> LearnedRules:
        """Learn and save the family's rules."""
        ...


def learn_family_rules(
    store: ArtifactStore,
    simulator: FamilySimulator,
    learner: FamilyLearner,
    *,
    code_revision: str,
) -> FamilySimulator:
    """Learn a family's rules once and bind them, or replay the saved learning with no call.

    Returns:
        The simulator with its frozen rule book and rule memory bound.

    Raises:
        ValueError: The learner bound nothing.
    """
    inputs = FamilyRulesInput(
        family_id=simulator.family_id,
        traces=simulator.rules_traces,
        tool_contracts=simulator.tool_contracts,
    )
    identity = stable_id(
        LEARNING_TYPE,
        {
            "protocol": FAMILY_SIMULATOR_PROTOCOL,
            "inputs": sha256_json(inputs),
            "learner": learner.identity,
        },
    )
    if store.exists(identity):
        learned = LearnedRules.model_validate_json(store.read_bytes(identity, "learned.json"))
    else:
        learned = learner(inputs)
        if learned.rule_book is None and learned.rule_memory is None:
            raise ValueError(f"the learner bound no rules for family {simulator.family_id}")
        refs = [ref for ref in (learned.rule_book, learned.rules_report) if ref is not None]
        if learned.rule_memory is not None:
            refs.append(learned.rule_memory.memory_set)
        _write(store, identity, LEARNING_TYPE, "learned.json", learned, code_revision, refs)
    return simulator.model_copy(
        update={
            "rule_book": learned.rule_book,
            "rule_memory": learned.rule_memory,
            "rules_report": learned.rules_report,
        }
    )


def bind_runtime(
    simulator: FamilySimulator, template: GroundedRuntimeDefinition
) -> FamilySimulator:
    """Freeze the family's runtime template over its own world model and learned rules.

    The template's manifest, catalog, allowance and probe paths are kept; its world model, rule
    book, rule memory and tag binding are replaced by the family's.
    """
    frozen = GroundedRuntimeDefinition.model_validate(
        template.model_dump()
        | {
            "world_model": simulator.world_model,
            "rule_book": simulator.rule_book,
            "rule_memory": simulator.rule_memory,
            "tag_retrieval": simulator.tag_retrieval,
        }
    )
    return simulator.model_copy(update={"runtime": frozen})


def definition_for(
    simulator: FamilySimulator, task: TaskCase, initial_state: JsonObject
) -> GroundedRuntimeDefinition:
    """Derive one task's runtime definition from the family's frozen template.

    Only the task and its initial state differ between the family's tasks.

    Raises:
        KeyError: The task is not in the family.
        ValueError: No runtime template is bound.
    """
    simulator.task(task.task_id)
    if simulator.runtime is None:
        raise ValueError(f"family {simulator.family_id} has no bound runtime template")
    return GroundedRuntimeDefinition.model_validate(
        simulator.runtime.model_dump() | {"task": task, "initial_state": initial_state}
    )


class AttemptSlot(ContractModel):
    """One planned attempt against a family's shared simulator.

    Attributes:
        simulator: Identity of the shared simulator definition.
        task_id: The task.
        actor: The actor (policy tier) making the attempt.
        attempt: Its index within the actor's group.
        start_sha256: Digest of the saved start every attempt restores.
    """

    simulator: str
    task_id: str
    actor: str
    attempt: int = Field(ge=0)
    start_sha256: str

    def identity(self) -> str:
        """Return the attempt's identity."""
        return stable_id("family-attempt", sha256_json(self))


class IsolatedRun(ContractModel):
    """One attempt's run in its own restored world.

    Attributes:
        slot: The attempt.
        final_sha256: Digest of the world's checkpoint when the attempt ended.
    """

    slot: AttemptSlot
    final_sha256: str


def plan_attempts(
    simulator: FamilySimulator,
    task_id: str,
    actors: Sequence[str],
    attempts: int,
    start: WorldCheckpoint,
) -> tuple[AttemptSlot, ...]:
    """Plan ``len(actors) * attempts`` independent attempts at one task from one start.

    Raises:
        KeyError: The task is not in the family.
        ValueError: Actors repeat or no attempt is planned.
    """
    simulator.task(task_id)
    if len(set(actors)) != len(actors) or not actors or attempts < 1:
        raise ValueError("plan attempts for distinct actors and at least one attempt each")
    identity, start_sha256 = simulator.identity(), sha256_json(start)
    return tuple(
        AttemptSlot(
            simulator=identity,
            task_id=task_id,
            actor=actor,
            attempt=index,
            start_sha256=start_sha256,
        )
        for actor in actors
        for index in range(attempts)
    )


def run_isolated_attempts[Result](
    world: WorldRuntime,
    task: TaskCase,
    start: WorldCheckpoint,
    slots: Sequence[AttemptSlot],
    episode: Callable[[CheckpointSession, AttemptSlot], Result],
) -> tuple[tuple[IsolatedRun, Result], ...]:
    """Run every attempt in a fresh world restored from the same start; never share mutation.

    Args:
        world: The family's executable or mixed world runtime.
        task: The task every slot attempts.
        start: The saved start every attempt restores.
        slots: Planned attempts, all against one simulator definition and this task and start.
        episode: Drives one attempt in its own restored session.

    Returns:
        Each attempt's run record and episode result, in slot order.

    Raises:
        ValueError: Slots name several simulators, another task or start, or the start changed.
    """
    digest = sha256_json(start)
    if len({slot.simulator for slot in slots}) > 1:
        raise ValueError("isolated attempts must share one simulator definition")
    if any(slot.task_id != task.task_id or slot.start_sha256 != digest for slot in slots):
        raise ValueError("every slot must attempt this task from this start")
    runs: list[tuple[IsolatedRun, Result]] = []
    for slot in slots:
        with restore_continuation_world(world, task, start) as session:
            result = episode(session, slot)
            final = session.checkpoint()
        runs.append((IsolatedRun(slot=slot, final_sha256=sha256_json(final)), result))
    if sha256_json(start) != digest:
        raise ValueError("an attempt mutated the shared start; attempts must restore it")
    return tuple(runs)


def save_family_simulator(
    store: ArtifactStore, simulator: FamilySimulator, *, code_revision: str
) -> ArtifactInput:
    """Save a family simulator content-addressed, or adopt the identical saved one."""
    identity = stable_id(SIMULATOR_TYPE, {"record": sha256_json(simulator)})
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    refs = [simulator.families, simulator.index, simulator.world_model]
    refs += [ref for ref in (simulator.tag_retrieval, simulator.rule_book) if ref is not None]
    return _write(store, identity, SIMULATOR_TYPE, "simulator.json", simulator, code_revision, refs)


def load_family_simulator(store: ArtifactStore, reference: ArtifactInput) -> FamilySimulator:
    """Read a saved family simulator, refusing a changed or mistyped artifact."""
    stored = store.read(reference.artifact_id)
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != SIMULATOR_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not the pinned {SIMULATOR_TYPE}")
    return FamilySimulator.model_validate_json(stored.payloads["simulator.json"])


def _write(
    store: ArtifactStore,
    identity: str,
    kind: str,
    name: str,
    record: ContractModel,
    code_revision: str,
    inputs: Sequence[ArtifactInput],
) -> ArtifactInput:
    """Write one record as a new artifact."""
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=kind,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
        ),
        files={name: record},
    )
    return artifact_input(manifest)

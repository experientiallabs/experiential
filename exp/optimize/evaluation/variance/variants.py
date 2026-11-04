"""Coherent variant checkpoints: edit a decision's world state without contradicting history.

A scenario edits declared variation axes of the world state just before a decision. Writing those
edits into the decision's own checkpoint is wrong whenever the actor or the world model has
already seen the edited values: the retained conversation would then contradict the world. A
variant therefore forks from the last saved boundary *before* the first turn that revealed any
edited value, applies the edits there, and leaves the actor to replay the turns up to the decision.

:func:`build_variant_checkpoint` records every check it makes, and persists a checkpoint only when
none fails:

- **trajectory**: the base checkpoint is the decision's boundary in the supplied trajectory, and
  every boundary extends the previous one's actor conversation for the same task.
- **fork_placement**: the earliest revealing turn is found by scanning each turn's new actor
  messages (tool results, assistant text and tool-call arguments) for the old and new values at
  every edited path and every value under it, and for a JSON key naming the edited field. A
  reveal in the initial prompt cannot be forked around and is refused.
- **world_state**: the fork's world holds the edited domains as grounded simulator state (the
  ``predictor.json`` image part). Native images, such as workbook bytes, cannot be edited by a
  pointer and are refused; they need a new initialization built by their owner.
- **edits**: each edit is a declared axis with a permitted value, applies to the fork state and
  adds or removes no domain.
- **public_context**: no edited path resolves in the task's public ``initial_context`` and no
  edited value appears in it.
- **instance_facts**: no instance fact names an edited path (keys compared as path tokens, either
  containing the other), and no fact names the edited field with an old value.
- **clock_and_events**: forks with pending events or outages are marked for explicit review. An
  edited key's spelling is not read as evidence that it is time-dependent.
- **retained_messages** and **retained_world_history**: the conversation and the predictor
  history kept at the fork mention no edited value.
- **replay_stability**: the turns the actor replays did not themselves change an edited path, so
  the edits still describe the decision's before-state.
- **initialization**: the instruction value and start mode agree with the decision's instruction
  axis (a changed instruction needs a fresh briefing). A ``resume`` variant keeps the coherent
  prefix and its remaining calls; a ``fresh_briefing`` variant starts at the fork on a new public
  task (the instruction value plus the briefing in its initial context) with the same remaining
  calls and no replay, its world restored under the original environment task. The briefing's
  entitled paths are checked structurally against the actor's real view at the fork (the retained
  messages and task prompt): each exists, is not edited and shows observed values. What the
  briefing text gives away was judged by the model leak reviewer when the scenario was authored,
  against the view at the decision.
- **briefing_view**: when a fresh-briefing variant forks before the decision, the actor's view at
  the fork is narrower than the view the leak reviewer read, so a fact observed only between the
  fork and the decision could have passed review. Such a variant is marked for explicit review.
- **complete_state**: :func:`freeze_state` records what the variant checkpoint actually holds for
  every :data:`~exp.optimize.evaluation.variance.contracts.StatePart`; the baseline parts, files
  or facts the caller requires, and the facts when supplied must all be held.
- **reachability**: the replay fits the actor's call allowance when one is supplied; otherwise it
  is marked for review. The replay length is always reported; reaching the decision again is
  :func:`~exp.optimize.evaluation.variance.replay.replay_to_decision`, which forces the original
  trajectory's recorded calls in the edited world and may diverge.

Value matching is deliberately conservative: a value counts as revealed when it appears as a
whole token anywhere in the text, regardless of which field it came from, so short or common
values can refuse a fork that a field-aware check would allow.

:func:`verify_restoration` proves a saved variant restores to the stored edited state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.variance.contracts import (
    BASELINE_STATE,
    ActorView,
    DecisionState,
    FrozenPart,
    FrozenState,
    InitializationMode,
    Scenario,
    StatePart,
    apply_edits,
    briefing_problems,
    instruction_problems,
    json_scalars,
    message_texts,
    resolve_pointer,
    revealing_texts,
)
from exp.runtime.agents.chat import ChatCheckpoint, task_messages
from exp.runtime.environments.world_artifacts import WorldCheckpoint
from exp.runtime.environments.world_contracts import ExecutableCheckpoint, NativeImage
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)

CheckName = Literal[
    "trajectory",
    "fork_placement",
    "world_state",
    "edits",
    "public_context",
    "instance_facts",
    "clock_and_events",
    "retained_messages",
    "retained_world_history",
    "replay_stability",
    "initialization",
    "briefing_view",
    "complete_state",
    "reachability",
]
"""Every coherence check a variant records."""

_PREDICTOR_PART = "predictor.json"


class CoherenceCheck(ContractModel):
    """The recorded result of one coherence check.

    Attributes:
        name: Which check.
        status: ``passed``; ``failed``, which blocks persistence; or ``review``, which is
            persisted but needs an explicit human or witness decision before use.
        detail: What was checked and, when it did not pass, what was found.
    """

    name: CheckName
    status: Literal["passed", "failed", "review"]
    detail: str = Field(min_length=1)


class VariantCheckpoint(ContractModel):
    """One scenario's variant checkpoint, or the reasons it could not be built.

    Attributes:
        scenario_id: The scenario the variant realizes.
        base: The decision's checkpoint in the saved trajectory.
        reference: The persisted variant checkpoint, or None when any check failed.
        fork: The saved boundary the variant was built from, or None when none was chosen.
        fork_turn: Boundary index of the fork (0 is the trajectory's start), or None.
        revealing_turn: First turn whose actor messages revealed an edited value, or None when
            no retained turn did. Turn 0 is the initial prompt.
        replay_turns: Tool turns the actor must take again to reach the decision, or None.
        edited_state_sha256: Digest of the edited simulated state stored in the variant.
        checks: Every check made, in order.
        problems: The details of every failed check; empty when the variant was persisted.
        initialization: How the actor starts the variant.
        frozen: What the variant checkpoint holds of the restorable state, once built.
    """

    scenario_id: str
    base: ArtifactInput
    reference: ArtifactInput | None
    fork: ArtifactInput | None
    fork_turn: int | None
    revealing_turn: int | None
    replay_turns: int | None
    edited_state_sha256: Sha256 | None
    checks: tuple[CoherenceCheck, ...]
    problems: tuple[str, ...]
    initialization: InitializationMode = "resume"
    frozen: FrozenState | None = None

    @property
    def needs_review(self) -> tuple[CoherenceCheck, ...]:
        """Checks that passed only conditionally and need an explicit decision."""
        return tuple(check for check in self.checks if check.status == "review")


def build_variant_checkpoint(
    project: ProjectStore,
    *,
    base: ArtifactInput,
    branch_checkpoints: tuple[ArtifactInput, ...],
    decision: DecisionState,
    scenario: Scenario,
    facts: InstanceFacts | None,
    code_revision: str,
    maximum_model_calls: int | None = None,
    required_state: frozenset[StatePart] = frozenset(),
) -> VariantCheckpoint:
    """Fork a saved trajectory before any observation of the edited state and apply a scenario.

    Args:
        project: Project whose artifacts hold the trajectory.
        base: Checkpoint at the decision, ``branch_checkpoints[decision.step]``.
        branch_checkpoints: Every saved boundary of the trajectory in order, start first;
            boundary ``k`` follows tool turn ``k``.
        decision: The decision state the scenario varies.
        scenario: The variant, whose edits must be declared axes with permitted values.
        facts: Instance facts supplied to the world model for this task, if any.
        code_revision: Producing implementation identity.
        maximum_model_calls: The actor's call allowance, when known, to check the replay fits.
        required_state: State parts required beyond the baseline, such as ``files``.

    Returns:
        The variant with every check; ``reference`` is None and ``problems`` lists the failed
        checks when the variant cannot be built coherently. Nothing is persisted then.
    """
    run = _Run(scenario_id=scenario.scenario_id, base=base)
    run.initialization = scenario.initialization
    if decision.step >= len(branch_checkpoints) or branch_checkpoints[decision.step] != base:
        return run.fail(
            "trajectory",
            f"base is not boundary {decision.step} of the supplied trajectory; pass the "
            "trajectory's checkpoints in order, start first",
        )
    loaded = [
        load_continuation_checkpoint(project.artifacts, reference)
        for reference in branch_checkpoints[: decision.step + 1]
    ]
    broken = _trajectory_problem(loaded)
    if broken is not None:
        return run.fail("trajectory", broken)
    run.check("trajectory", "passed", f"{len(loaded)} boundaries extend one conversation")

    paths = tuple(edit.path for edit in scenario.edits)
    states = [grounded_state(checkpoint.world) for checkpoint in loaded]
    values = _watched_values(scenario, states)
    keys = frozenset(name for path in paths if (name := _field_name(path)) is not None)
    run.add(_public_context_check(loaded[-1], paths, values))
    run.add(_facts_check(facts, paths, values))
    revealing = _first_revealing_turn(loaded, values, keys)
    if revealing == 0:
        return run.fail(
            "fork_placement",
            "the initial prompt already shows an edited value; no saved boundary precedes it, so "
            "build and validate a new initialization instead",
            revealing_turn=0,
        )
    fork = decision.step if revealing is None else revealing - 1
    run.fork, run.fork_turn = branch_checkpoints[fork], fork
    fresh = scenario.initialization == "fresh_briefing"
    run.revealing_turn, run.replay_turns = revealing, 0 if fresh else decision.step - fork
    run.check(
        "fork_placement",
        "passed",
        "no retained turn reveals an edited value"
        if revealing is None
        else f"turn {revealing} reveals an edited value; forking at boundary {fork}",
    )

    forked = loaded[fork]
    grounded = _grounded_parts(forked.world)
    if isinstance(grounded, str):
        return run.fail("world_state", grounded)
    state, history = grounded
    foreign = sorted({path.split("/")[1] for path in paths} - set(state))
    if foreign:
        return run.fail(
            "world_state",
            f"edits target domains {foreign} the simulated world does not hold; native state "
            "needs a new initialization from its owner",
        )
    run.check("world_state", "passed", f"simulated domains {sorted(state)} are editable")

    undeclared = _undeclared_edits(decision, scenario)
    if undeclared:
        return run.fail("edits", "; ".join(undeclared))
    try:
        edited = apply_edits(state, scenario.edits)
    except ValueError as error:
        return run.fail("edits", str(error))
    if set(edited) != set(state):
        return run.fail("edits", "edits add a simulated domain; edit inside existing domains")
    run.check("edits", "passed", f"{len(paths)} declared axes applied at boundary {fork}")

    run.add(_clock_check(forked.world))
    retained = revealing_texts(message_texts(forked.actor.messages), values, keys)
    run.check(
        "retained_messages",
        "failed" if retained else "passed",
        f"retained actor messages show edited values {retained}"
        if retained
        else f"{len(forked.actor.messages)} retained messages show no edited value",
    )
    seen = revealing_texts(_history_texts(history, forked.world), values, keys)
    run.check(
        "retained_world_history",
        "failed" if seen else "passed",
        f"retained world history shows edited values {seen}"
        if seen
        else f"{len(history)} retained world turns show no edited value",
    )
    if not fresh:
        run.add(_replay_check(states[fork : decision.step + 1], paths, fork))
    started = _initialize(loaded[0], forked, decision, scenario, state)
    if isinstance(started, str):
        run.check("initialization", "failed", started)
    else:
        run.check("initialization", "passed", f"{scenario.initialization} start is coherent")
    if scenario.briefing is not None:
        unseen = len(loaded[-1].actor.messages) - len(forked.actor.messages)
        run.check(
            "briefing_view",
            "review" if unseen else "passed",
            f"the fork drops {unseen} messages the leak reviewer read at the decision; confirm "
            "the briefing states nothing only they show"
            if unseen
            else "the leak reviewer read exactly the actor's view at the fork",
        )
    run.add(_reachability_check(forked, run.replay_turns, maximum_model_calls))
    if run.problems or isinstance(started, str):
        return run.result()
    task, actor = started
    variant = ContinuationCheckpoint(
        task=task,
        environment_task=forked.environment_task if task == forked.task else forked.world_task,
        actor=actor,
        world=_with_state(forked.world, edited, history),
        sources=(*forked.sources, branch_checkpoints[fork])
        if branch_checkpoints[fork] not in forked.sources
        else forked.sources,
    )
    run.frozen = freeze_state(variant, facts)
    declared: frozenset[StatePart] = frozenset({"conditioning_facts"} if facts else ())
    required = BASELINE_STATE | required_state | declared
    missing = run.frozen.unrestorable(required)
    if missing:
        return run.fail("complete_state", "; ".join(missing))
    run.check("complete_state", "passed", f"{len(required)} required state parts are held")
    run.reference = persist_continuation_checkpoint(
        project.artifacts,
        variant,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
    )
    run.edited_state_sha256 = sha256_json(edited)
    return run.result()


def verify_restoration(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    runtime: ExecutableWorldRuntime | None = None,
    expected_state_sha256: Sha256 | None = None,
) -> bool:
    """Prove a saved variant restores to the edited state it stores.

    The checkpoint is loaded through the digest-verified continuation owner and its simulated
    state decoded. Without ``runtime`` the proof compares image contents: each domain's digest in
    the image must equal the digest of the decoded state, which is exactly the check the grounded
    backend and the world kernel make on restore. With ``runtime``, the world is restored through
    the kernel's own restore path and recaptured, and the recaptured image must equal the stored
    one.

    Args:
        project: Project holding the checkpoint.
        reference: The variant checkpoint.
        runtime: The executable world runtime the trajectory ran in, for a real restore.
        expected_state_sha256: The edited state digest recorded when the variant was built.

    Returns:
        Whether every comparison held.
    """
    try:
        checkpoint = load_continuation_checkpoint(project.artifacts, reference)
        grounded = _grounded_parts(checkpoint.world)
        if isinstance(grounded, str):
            return False
        state, _ = grounded
        for image in _grounded_images(checkpoint.world):
            if {key: sha256_json(value) for key, value in state.items()} != image.domains:
                return False
        if expected_state_sha256 is not None and sha256_json(state) != expected_state_sha256:
            return False
        if runtime is None:
            return True
        world = checkpoint.world
        if not isinstance(world, ExecutableCheckpoint):
            return False
        with runtime.restore(checkpoint.world_task, world) as restored:
            recaptured = restored.checkpoint()
        return recaptured.image == world.image and recaptured.clock == world.clock
    except ValueError:
        return False


def freeze_state(checkpoint: ContinuationCheckpoint, facts: InstanceFacts | None) -> FrozenState:
    """Record what a checkpoint actually holds of every part of the restorable state.

    Args:
        checkpoint: A variant or decision checkpoint.
        facts: Instance facts conditioning the world model, when supplied.

    Returns:
        One entry per state part: ``absent`` when it holds none (a world with no executable row
        holds no kernel state), ``external`` for native domains whose image has no bytes.
    """
    world = checkpoint.world
    mixed = None if isinstance(world, ExecutableCheckpoint) else world
    rows: list[ExecutableCheckpoint] = (
        [r for r in (world.native, world.candidate) if r]
        if not isinstance(world, ExecutableCheckpoint)
        else [world]
    )
    policy = None if mixed is None else mixed.policy_sha256
    actor, tools = checkpoint.actor, [t.model_dump(mode="json") for t in checkpoint.task.tools]
    files = {
        f"{r.owner}/{name}": sha256_bytes(data)
        for r in rows
        for name, data in r.image.parts.items()
        if name != _PREDICTOR_PART
    }
    outside = sorted({r.owner for r in rows if r.image.domains and not r.image.parts})
    held: dict[StatePart, tuple[JsonValue, str]] = {
        "domain_state": ([[r.owner, dict(r.image.domains)] for r in rows], "domain digests"),
        "files": (files, f"file parts {sorted(files)}"),
        "identity_permissions": (
            [[r.manifest_sha256, r.backend_sha256, dict(r.ownership), policy] for r in rows],
            "manifest, backend, domain ownership and routing policy",
        ),
        "clock": ([r.clock for r in rows], f"logical time {[r.clock for r in rows]}"),
        "pending_events": (
            [[[e.model_dump(mode="json") for e in r.pending], dict(r.degraded)] for r in rows],
            f"{sum(len(r.pending) for r in rows)} pending events, with outages",
        ),
        "scheduler_random": (
            [[r.next_event, r.seed, r.draws] for r in rows],
            "event counter, seed and draws; a simulator model's future draws are not state",
        ),
        "event_cursor": (
            [[r.steps, [e.model_dump(mode="json") for e in r.fired]] for r in rows]
            + ([] if mixed is None or not rows else [[mixed.sequence, mixed.shadow_halted]]),
            f"steps {[r.steps for r in rows]} with every fired event",
        ),
        "conditioning_facts": (
            None if facts is None else facts.model_dump(mode="json"),
            "instance facts by digest",
        ),
        "tool_session": ([actor.runtime_sha256, actor.calls, tools], f"{actor.calls} calls spent"),
        "conversation": (
            [m.model_dump(mode="json") for m in actor.messages],
            f"{len(actor.messages)} actor-visible messages",
        ),
    }
    parts: list[FrozenPart] = []
    for part, (value, detail) in held.items():
        if part == "domain_state" and outside:
            row = FrozenPart(part=part, status="external", detail=f"{outside} hold no image bytes")
        elif value is None or value == {} or value == []:
            row = FrozenPart(part=part, status="absent", detail="the checkpoint holds none")
        else:
            row = FrozenPart(part=part, status="held", sha256=sha256_json(value), detail=detail)
        parts.append(row)
    return FrozenState(parts=tuple(parts))


def _initialize(
    start: ContinuationCheckpoint,
    forked: ContinuationCheckpoint,
    decision: DecisionState,
    scenario: Scenario,
    state: JsonObject,
) -> tuple[TaskCase, ChatCheckpoint] | str:
    """Return a variant's public task and actor start, or why its initialization is incoherent.

    Args:
        start: The trajectory's first boundary, whose conversation is the task prompt.
        forked: The fork boundary.
        decision: The decision state with its instruction axis.
        scenario: The variant.
        state: Simulated state at the fork, before the edits.

    Returns:
        The task and actor checkpoint, or the reasons there are none.
    """
    problems = instruction_problems(decision, scenario)
    axis = decision.instruction_axis
    value = None if axis is None else axis.value(scenario.instruction)
    if value and value.kind == "original" and value.instruction != forked.task.instruction:
        problems.append("the instruction axis's original value is not the task's instruction")
    if scenario.briefing is None:
        return "; ".join(problems) if problems else (forked.task, forked.actor)
    view = ActorView(
        instruction=forked.task.instruction, observed=message_texts(forked.actor.messages)
    )
    problems.extend(briefing_problems(state, scenario, view))
    tail, opening = task_messages(forked.task), start.actor.messages
    leading = opening[: len(opening) - len(tail)]
    if len(opening) < len(tail) or opening[len(leading) :] != tail:
        problems.append("the start conversation is not the chat task prompt a briefing replaces")
    if "briefing" in forked.task.initial_context:
        problems.append("the task's initial context already has a briefing entry")
    if problems:
        return "; ".join(problems)
    task = forked.task.model_copy(
        update={
            "instruction": forked.task.instruction if value is None else value.instruction,
            "initial_context": {**forked.task.initial_context, "briefing": scenario.briefing.text},
        }
    )
    return task, ChatCheckpoint(
        task_sha256=sha256_json(task),
        runtime_sha256=forked.actor.runtime_sha256,
        messages=(*leading, *task_messages(task)),
        calls=forked.actor.calls,
    )


class _Run:
    """Accumulates checks and the fork placement while one variant is built."""

    def __init__(self, *, scenario_id: str, base: ArtifactInput) -> None:
        """Start with no checks and no fork.

        Args:
            scenario_id: The scenario being realized.
            base: The decision's checkpoint.
        """
        self.scenario_id, self.base = scenario_id, base
        self.checks: list[CoherenceCheck] = []
        self.fork: ArtifactInput | None = None
        self.fork_turn: int | None = None
        self.revealing_turn: int | None = None
        self.replay_turns: int | None = None
        self.reference: ArtifactInput | None = None
        self.edited_state_sha256: Sha256 | None = None
        self.initialization: InitializationMode = "resume"
        self.frozen: FrozenState | None = None

    @property
    def problems(self) -> tuple[str, ...]:
        """Details of every failed check."""
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if c.status == "failed")

    def add(self, check: CoherenceCheck) -> None:
        """Record one check."""
        self.checks.append(check)

    def check(
        self, name: CheckName, status: Literal["passed", "failed", "review"], detail: str
    ) -> None:
        """Record one check from its parts.

        Args:
            name: Which check.
            status: Its outcome.
            detail: What was found.
        """
        self.add(CoherenceCheck(name=name, status=status, detail=detail))

    def fail(
        self, name: CheckName, detail: str, *, revealing_turn: int | None = None
    ) -> VariantCheckpoint:
        """Record a blocking failure and return the unpersisted result.

        Args:
            name: Which check failed.
            detail: Why.
            revealing_turn: The revealing turn, when the failure is about it.

        Returns:
            The result without a reference.
        """
        if revealing_turn is not None:
            self.revealing_turn = revealing_turn
        self.check(name, "failed", detail)
        return self.result()

    def result(self) -> VariantCheckpoint:
        """Freeze the run into its result."""
        return VariantCheckpoint(
            scenario_id=self.scenario_id,
            base=self.base,
            reference=self.reference,
            fork=self.fork,
            fork_turn=self.fork_turn,
            revealing_turn=self.revealing_turn,
            replay_turns=self.replay_turns,
            edited_state_sha256=self.edited_state_sha256,
            checks=tuple(self.checks),
            problems=self.problems,
            initialization=self.initialization,
            frozen=self.frozen,
        )


def _trajectory_problem(loaded: list[ContinuationCheckpoint]) -> str | None:
    """Return why the boundaries are not one trajectory, or None when they are.

    Args:
        loaded: Boundaries in order.

    Returns:
        The problem, or None.
    """
    for index in range(1, len(loaded)):
        previous, current = loaded[index - 1], loaded[index]
        if current.task != previous.task or current.world_task != previous.world_task:
            return f"boundary {index} belongs to another task"
        prefix = previous.actor.messages
        if current.actor.messages[: len(prefix)] != prefix or len(current.actor.messages) <= len(
            prefix
        ):
            return f"boundary {index} does not extend boundary {index - 1}'s conversation"
    return None


def _grounded_images(world: WorldCheckpoint) -> tuple[NativeImage, ...]:
    """Return the grounded simulator images a world holds.

    An image served with transition programs also carries their partition part; the simulated
    state and history still live only in the predictor part.
    """
    worlds = (
        (world,) if isinstance(world, ExecutableCheckpoint) else (world.native, world.candidate)
    )
    return tuple(
        row.image for row in worlds if row is not None and _PREDICTOR_PART in row.image.parts
    )


def _grounded_parts(world: WorldCheckpoint) -> tuple[JsonObject, list[JsonValue]] | str:
    """Decode the simulated state and predictor history, or say why the world is not editable.

    Args:
        world: A saved world.

    Returns:
        ``(state, history)``, or the reason the world cannot be edited by pointer.
    """
    images = _grounded_images(world)
    if len(images) != 1:
        return (
            "the world holds no single grounded simulator image (predictor.json); native "
            "images cannot be edited by pointer, so build a new initialization from their owner"
        )
    wire = json.loads(images[0].parts[_PREDICTOR_PART])
    if (
        not isinstance(wire, dict)
        or not isinstance(wire.get("state"), dict)
        or not isinstance(wire.get("history"), list)
    ):
        return "the grounded image has malformed state or history"
    return wire["state"], wire["history"]


def grounded_state(world: WorldCheckpoint) -> JsonObject | None:
    """Return a boundary's simulated state, or None when it holds none."""
    parts = _grounded_parts(world)
    return None if isinstance(parts, str) else parts[0]


def _with_state(
    world: WorldCheckpoint, state: JsonObject, history: list[JsonValue]
) -> WorldCheckpoint:
    """Replace the grounded image's state, keeping its history, other parts and kernel fields.

    Args:
        world: The fork's world.
        state: The edited simulated state.
        history: The predictor history retained at the fork.

    Returns:
        The world with a new image whose domain digests match the edited state.
    """
    predictor = canonical_json_bytes({"state": state, "history": history})
    domains = {key: sha256_json(value) for key, value in state.items()}

    def replaced(row: ExecutableCheckpoint) -> ExecutableCheckpoint:
        """Swap the predictor part of one grounded executable checkpoint's image."""
        parts = {**row.image.parts, _PREDICTOR_PART: predictor}
        return row.model_copy(update={"image": NativeImage(parts=parts, domains=domains)})

    if isinstance(world, ExecutableCheckpoint):
        return replaced(world)
    if world.candidate is not None and _PREDICTOR_PART in world.candidate.image.parts:
        return world.model_copy(update={"candidate": replaced(world.candidate)})
    assert world.native is not None
    return world.model_copy(update={"native": replaced(world.native)})


def _watched_values(scenario: Scenario, states: list[JsonObject | None]) -> tuple[JsonValue, ...]:
    """Collect every old and new scalar at or under an edited path.

    Args:
        scenario: The variant.
        states: Simulated state at each boundary up to the decision.

    Returns:
        Distinct scalar values whose appearance counts as revealing an edited path.
    """
    found: list[JsonValue] = []
    for edit in scenario.edits:
        found.extend(json_scalars(edit.value))
        for state in states:
            if state is None:
                continue
            exists, value = resolve_pointer(state, edit.path)
            if exists:
                found.extend(json_scalars(value))
    unique: dict[str, JsonValue] = {}
    for value in found:
        unique.setdefault(json.dumps(value), value)
    return tuple(unique.values())


def _field_name(path: str) -> str | None:
    """Return the last non-index token of a pointer, the field name an observation would use."""
    for raw in reversed(path.split("/")[1:]):
        token = raw.replace("~1", "/").replace("~0", "~")
        if token and not token.isdigit():
            return token
    return None


def _history_texts(history: list[JsonValue], world: WorldCheckpoint) -> tuple[str, ...]:
    """Return the predictor history and fired-event details as texts to scan."""
    texts = [json.dumps(turn, sort_keys=True) for turn in history]
    worlds = (
        (world,) if isinstance(world, ExecutableCheckpoint) else (world.native, world.candidate)
    )
    for row in worlds:
        if row is not None:
            texts.extend(event.detail for event in row.fired)
    return tuple(texts)


def _first_revealing_turn(
    loaded: list[ContinuationCheckpoint], values: tuple[JsonValue, ...], keys: frozenset[str]
) -> int | None:
    """Return the first turn whose new actor messages reveal an edited value.

    Args:
        loaded: Boundaries in order; turn ``k`` is the messages boundary ``k`` adds.
        values: Old and new scalars at the edited paths.
        keys: Field names of the edited paths.

    Returns:
        The turn (0 is the initial prompt), or None when no retained turn reveals one.
    """
    seen = 0
    for turn, checkpoint in enumerate(loaded):
        messages = checkpoint.actor.messages[seen:]
        seen = len(checkpoint.actor.messages)
        if revealing_texts(message_texts(messages), values, keys):
            return turn
    return None


def _undeclared_edits(decision: DecisionState, scenario: Scenario) -> list[str]:
    """Return why any edit is not a declared axis with a permitted value."""
    axes = {axis.path: axis for axis in decision.axes}
    problems: list[str] = []
    for edit in scenario.edits:
        axis = axes.get(edit.path)
        if axis is None:
            problems.append(f"edit {edit.path} is not a declared variation axis")
        elif json.dumps(edit.value, sort_keys=True) not in {
            json.dumps(value, sort_keys=True) for value in axis.values
        }:
            problems.append(f"edit {edit.path} uses a value the axis does not permit")
    return problems


def _public_context_check(
    checkpoint: ContinuationCheckpoint, paths: tuple[str, ...], values: tuple[JsonValue, ...]
) -> CoherenceCheck:
    """Refuse edits to anything the public task context states.

    Args:
        checkpoint: A boundary of the trajectory; every boundary shares its task.
        paths: Edited paths.
        values: Old and new scalars at those paths.

    Returns:
        Failed when a path resolves in the context or a watched value appears in it.
    """
    tasks: dict[str, TaskCase] = {
        "task": checkpoint.task,
        "environment task": checkpoint.world_task,
    }
    found: list[str] = []
    for label, task in tasks.items():
        context = task.initial_context
        found.extend(
            f"{label} initial_context states {path}"
            for path in paths
            if any(resolve_pointer(context, prefix)[0] for prefix in _prefixes(path))
        )
        found.extend(
            f"{label} initial_context shows {item}"
            for item in revealing_texts((json.dumps(context),), values, frozenset())
        )
    if found:
        return CoherenceCheck(
            name="public_context",
            status="failed",
            detail="; ".join(sorted(set(found))) + "; the actor is told this, so it cannot vary",
        )
    return CoherenceCheck(
        name="public_context", status="passed", detail="no edited path or value is public context"
    )


def _prefixes(path: str) -> tuple[str, ...]:
    """Return a pointer and every ancestor except the root, deepest first."""
    tokens = path.split("/")[1:]
    return tuple("/" + "/".join(tokens[:end]) for end in range(len(tokens), 0, -1))


def _facts_check(
    facts: InstanceFacts | None, paths: tuple[str, ...], values: tuple[JsonValue, ...]
) -> CoherenceCheck:
    """Refuse edits that an instance fact describes.

    Args:
        facts: Instance facts supplied to the world model, if any.
        paths: Edited paths.
        values: Old and new scalars at those paths.

    Returns:
        Failed when a fact's key overlaps an edited path, or names the edited field with a
        watched value.
    """
    if facts is None:
        return CoherenceCheck(name="instance_facts", status="passed", detail="no instance facts")
    conflicts: list[str] = []
    watched = {json.dumps(value) for value in values}
    for fact in facts.facts:
        key = tuple(token for token in fact.key.split("/") if token)
        for path in paths:
            tokens = tuple(path.split("/")[1:])
            shorter = min(len(key), len(tokens))
            if key and key[:shorter] == tokens[:shorter]:
                conflicts.append(f"fact {fact.key!r} describes {path}")
            elif (
                key
                and key[-1] == _field_name(path)
                and any(json.dumps(item) in watched for item in json_scalars(fact.value))
            ):
                conflicts.append(f"fact {fact.key!r} names the field of {path} with its value")
    if conflicts:
        return CoherenceCheck(
            name="instance_facts",
            status="failed",
            detail="; ".join(sorted(set(conflicts)))
            + "; facts condition every prediction and cannot be edited",
        )
    return CoherenceCheck(
        name="instance_facts",
        status="passed",
        detail=f"none of {len(facts.facts)} facts describes an edited path",
    )


def _clock_check(world: WorldCheckpoint) -> CoherenceCheck:
    """Mark forks with pending events or outages for explicit review.

    Both are recorded world state scheduled or degraded under the old values, so an edit may
    contradict them. Whether an edited value is itself time-dependent is not guessed from its
    key's spelling: the decision-state author declares what the retained history fixes as
    invariants, and the witness and replay execute the edited world.

    Args:
        world: The fork's world.

    Returns:
        Review when events are pending or tools are under an outage.
    """
    worlds = (
        (world,) if isinstance(world, ExecutableCheckpoint) else (world.native, world.candidate)
    )
    pending = [event.event_id for row in worlds if row is not None for event in row.pending]
    outages = sorted({tool for row in worlds if row is not None for tool in row.degraded})
    notes = []
    if pending:
        notes.append(f"events {pending} are pending and were scheduled under the old state")
    if outages:
        notes.append(f"tools {outages} are under an outage")
    if notes:
        return CoherenceCheck(
            name="clock_and_events",
            status="review",
            detail="; ".join(notes) + "; confirm they agree with the edited state",
        )
    return CoherenceCheck(
        name="clock_and_events", status="passed", detail="no event pending and no outage"
    )


def _replay_check(
    states: list[JsonObject | None], paths: tuple[str, ...], fork: int
) -> CoherenceCheck:
    """Refuse a fork whose replayed turns themselves changed an edited path.

    Args:
        states: Simulated state from the fork to the decision.
        paths: Edited paths.
        fork: Boundary index of the fork.

    Returns:
        Failed when any replayed boundary holds a different value at an edited path.
    """
    changed = sorted(
        {
            path
            for path in paths
            for state in states[1:]
            if state is None
            or states[0] is None
            or resolve_pointer(state, path) != resolve_pointer(states[0], path)
        }
    )
    if changed:
        return CoherenceCheck(
            name="replay_stability",
            status="failed",
            detail=f"turns after boundary {fork} change {changed}, so edits made at the fork do "
            "not describe the decision's before-state",
        )
    return CoherenceCheck(
        name="replay_stability",
        status="passed",
        detail=f"{len(states) - 1} replayed turns leave every edited path unchanged",
    )


def _reachability_check(
    fork: ContinuationCheckpoint, replay_turns: int | None, maximum_model_calls: int | None
) -> CoherenceCheck:
    """Check the actor can replay to the decision and still make it.

    Args:
        fork: The fork checkpoint.
        replay_turns: Tool turns to replay.
        maximum_model_calls: The actor's call allowance, if known.

    Returns:
        Failed when the replay and the decision exceed the allowance; review when it is unknown.
    """
    replay = replay_turns or 0
    if maximum_model_calls is None:
        return CoherenceCheck(
            name="reachability",
            status="review",
            detail=f"the actor replays {replay} turns from {fork.actor.calls} spent calls; "
            "supply its allowance to check the decision stays reachable",
        )
    needed = fork.actor.calls + replay + 1
    if needed > maximum_model_calls:
        return CoherenceCheck(
            name="reachability",
            status="failed",
            detail=f"replaying {replay} turns and deciding needs {needed} calls but the "
            f"allowance is {maximum_model_calls}",
        )
    return CoherenceCheck(
        name="reachability",
        status="passed",
        detail=f"replaying {replay} turns and deciding needs {needed} of "
        f"{maximum_model_calls} calls",
    )

"""Freeze an authored world against its rule book, and review it against an independent rubric.

:func:`freeze_world` turns a ready world draft into a frozen world, or records why it cannot be
one. It checks the simulator, never an actor:

- **Properties.** The world's replay fixtures are executed through its programs and every
  property the rules stage derives from the rule book is checked on the resulting transitions.
  A supported rule the world's own fixtures violate rejects the world.
- **Boundary workloads.** The rules stage's boundary workloads are generated from every
  variant's initial state and executed through the gated tool's program; their triage keeps a
  contradiction of an unwitnessed rule diagnostic. A workload of a predicted tool stays
  unexecuted unless an observed transition is supplied.
- **Authorization and state gaps.** Every precondition of a state-writing tool is load-bearing.
  Each variant runs the tool once in its own state, and once more with each read the
  precondition makes deleted. A supported error or refusal, or an explicit ``OutOfDomain``
  unknown, resolves the gap; a write that silently succeeds while the precondition is false or
  undetermined rejects the world. A predicted tool whose supported precondition cannot be
  decided in a variant rejects that variant. A precondition without a supported predicate is an
  unsupported decision boundary: the family stays diagnostic, never trainable.
- **Restoration.** Each surviving variant is started through the existing grounded runtime,
  saved as a continuation checkpoint and proven to restore to its exact initial state with
  :func:`~exp.optimize.evaluation.variance.variants.verify_restoration`.

:func:`review_compatibility` compares a world and a rubric, each authored independently from the
same packet, against that packet: does the world initialize every variant the rubric scores, can
the judge observe every record each criterion needs, and does the world bind the rules the rubric
cites. Every disagreement is recorded; neither draft is edited to agree with the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Literal

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
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
)
from exp.optimize.evaluation.rules.stage import load_rulebook
from exp.optimize.evaluation.variance.authoring_packet import AuthoringPacket, load_packet
from exp.optimize.evaluation.variance.contracts import resolve_pointer
from exp.optimize.evaluation.variance.rubric_author import Rubric
from exp.optimize.evaluation.variance.variants import grounded_state, verify_restoration
from exp.optimize.evaluation.variance.world_author import (
    WorldSpec,
    load_world_draft,
    world_runtime_definition,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.world_contracts import ToolEffectPolicy
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.induction.contracts import ProgramOutcome, TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.rules.model import Predicate, Rule, RuleBook, Transition, evaluate
from exp.simulation.rules.properties import (
    DEFAULT_REFUSAL,
    PropertyReport,
    RefusalCriteria,
    check_transitions,
    is_refusal,
    properties_from,
)
from exp.simulation.rules.workloads import Triage, Workload, boundary_workloads, triage

WORLD_FREEZE_PROTOCOL = "world-freeze-v1"
"""Version of the freeze checks and compatibility review; change it whenever either changes."""

FREEZE_TYPE = "world-freeze"
"""Artifact type of a saved freeze report."""

Resolution = Literal[
    "consistent",
    "enforced",
    "refused",
    "unknown",
    "unexplained_refusal",
    "silently_allowed",
    "undecidable",
    "broken",
    "unexecuted",
]
"""How one authorization probe resolved."""

DisagreementKind = Literal[
    "uncovered_variant",
    "unscored_variant",
    "unobservable_evidence",
    "unbound_rule",
]
"""What a world and a rubric disagree about."""


class GateProbe(ContractModel):
    """One execution of a state-writing tool against its load-bearing precondition.

    Attributes:
        variant_id: Variant whose initial state the probe ran in.
        rule_id: The precondition.
        tool: The gated tool.
        case: ``variant`` runs in the variant's own state; ``missing:<pointer>`` deletes one
            read of the precondition first.
        predicate: The precondition's value on the probed state: True, False, or None when a
            read is absent.
        program_status: The program outcome, or None for a predicted tool.
        resolution: How the probe resolved.
        detail: One sentence on why.
    """

    variant_id: str
    rule_id: str
    tool: str
    case: str
    predicate: bool | None
    program_status: str | None
    resolution: Resolution
    detail: str


class WorkloadRun(ContractModel):
    """One boundary workload of one variant and what executing it showed.

    Attributes:
        variant_id: Variant whose initial state the workload edits.
        workload: The rules stage's workload.
        executed: Whether the gated program ran or an observed transition was supplied.
        triage: The rules stage's triage, when executed.
    """

    variant_id: str
    workload: Workload
    executed: bool
    triage: Triage | None = None


class VariantFreeze(ContractModel):
    """One variant's frozen start, or why it is diagnostic or rejected.

    Attributes:
        variant_id: Packet variant.
        status: ``frozen`` may qualify for training; ``diagnostic`` is usable for analysis
            only; ``rejected`` is not a valid world.
        definition_sha256: Digest of the grounded runtime definition, when one was bound.
        checkpoint: Saved starting checkpoint, when restoration was attempted.
        restored: Whether the checkpoint restored to the exact initial state; None when no
            runtime was supplied.
        problems: Why the variant is rejected.
        diagnostics: Why the variant cannot qualify for training.
    """

    variant_id: str
    status: Literal["frozen", "diagnostic", "rejected"]
    definition_sha256: Sha256 | None = None
    checkpoint: ArtifactInput | None = None
    restored: bool | None = None
    problems: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class WorldFreeze(ContractModel):
    """The complete record of freezing one authored world.

    Attributes:
        protocol: Freeze protocol.
        identity: Identity of this freeze, which names its saved report.
        packet_sha256: Digest of the authoring packet.
        world: The saved world draft.
        world_sha256: Digest of the authored world.
        properties: Property results over the fixture replays and executed workloads.
        workloads: Every boundary workload per variant.
        probes: Every authorization probe.
        diagnostic_rules: Rules whose boundaries keep the family diagnostic.
        problems: World-level defects; any rejects every variant.
        variants: One record per variant.
        trainable: Whether at least one variant is frozen and the world has no defect.
    """

    protocol: str = WORLD_FREEZE_PROTOCOL
    identity: str
    packet_sha256: Sha256
    world: ArtifactInput
    world_sha256: Sha256
    properties: PropertyReport
    workloads: tuple[WorkloadRun, ...]
    probes: tuple[GateProbe, ...]
    diagnostic_rules: tuple[str, ...]
    problems: tuple[str, ...]
    variants: tuple[VariantFreeze, ...]
    trainable: bool


class Disagreement(ContractModel):
    """One recorded disagreement between an authored world and an authored rubric.

    Attributes:
        kind: What they disagree about.
        subject: The variant, criterion or rule concerned.
        world: What the world says or lacks.
        rubric: What the rubric says or needs.
        sources: Packet locators that bear on it, for research or adjudication.
    """

    kind: DisagreementKind
    subject: str
    world: str
    rubric: str
    sources: tuple[str, ...] = ()


class CompatibilityReview(ContractModel):
    """Every disagreement between one world and one rubric authored from one packet.

    Attributes:
        protocol: Freeze protocol the review belongs to.
        packet_sha256: The packet both were authored from.
        world_sha256: The world reviewed.
        rubric_sha256: The rubric reviewed.
        disagreements: Every disagreement, unreconciled.
        compatible: True exactly when there is none.
    """

    protocol: str = WORLD_FREEZE_PROTOCOL
    packet_sha256: Sha256
    world_sha256: Sha256
    rubric_sha256: Sha256
    disagreements: tuple[Disagreement, ...]
    compatible: bool


def review_compatibility(
    packet: AuthoringPacket, world: WorldSpec, rubric: Rubric
) -> CompatibilityReview:
    """Compare an independently authored world and rubric against their shared packet.

    Args:
        packet: The packet both authors read.
        world: The ready world.
        rubric: The ready rubric.

    Returns:
        The review, listing every disagreement without changing either draft.

    Raises:
        ValueError: The world or rubric was authored from another packet.
    """
    if rubric.packet_sha256 != packet.identity:
        raise ValueError("the rubric was authored from another packet")
    found: list[Disagreement] = []
    initialized = {row.variant_id for row in world.initializations}
    criteria = rubric.body.criteria
    for variant in packet.variants:
        scored = [
            row.criterion_id
            for row in criteria
            if row.role == "required_correctness" and variant.variant_id in row.scenario_ids
        ]
        if variant.variant_id not in initialized:
            found.append(
                Disagreement(
                    kind="uncovered_variant",
                    subject=variant.variant_id,
                    world="the world does not initialize this variant",
                    rubric=f"criteria {scored} score it",
                    sources=(f"variant:{variant.variant_id}",),
                )
            )
        if not scored:
            found.append(
                Disagreement(
                    kind="unscored_variant",
                    subject=variant.variant_id,
                    world="the world initializes this variant",
                    rubric="no required correctness criterion applies to it",
                    sources=(f"variant:{variant.variant_id}",),
                )
            )
    hooks = {hook.kind for hook in world.evidence_hooks}
    exposed = (*world.probe_paths.values(), *(h.state_path for h in world.evidence_hooks))
    for row in criteria:
        for need in row.evidence_needed:
            if need.kind not in hooks:
                found.append(
                    Disagreement(
                        kind="unobservable_evidence",
                        subject=row.criterion_id,
                        world=f"no evidence hook records {need.kind}",
                        rubric=f"needs {need.kind}: {need.detail}",
                        sources=row.supported_by,
                    )
                )
            if need.state_path is not None and not any(
                path is not None and _within(need.state_path, path) for path in exposed
            ):
                found.append(
                    Disagreement(
                        kind="unobservable_evidence",
                        subject=row.criterion_id,
                        world=f"no probe or hook exposes {need.state_path}",
                        rubric=f"needs state {need.state_path}: {need.detail}",
                        sources=row.supported_by,
                    )
                )
    bindings = {binding.rule_id: binding.use for binding in world.rule_bindings}
    for row in criteria:
        if row.role != "required_correctness":
            continue
        for ref in row.supported_by:
            if not ref.startswith("rule:"):
                continue
            rule_id = ref[len("rule:") :]
            use = bindings.get(rule_id)
            if use is None or use == "diagnostic":
                found.append(
                    Disagreement(
                        kind="unbound_rule",
                        subject=rule_id,
                        world="the world does not bind it"
                        if use is None
                        else "the world binds it as diagnostic only",
                        rubric=f"required criterion {row.criterion_id} rests on it",
                        sources=(ref,),
                    )
                )
    return CompatibilityReview(
        packet_sha256=packet.identity,
        world_sha256=world.identity,
        rubric_sha256=rubric.identity,
        disagreements=tuple(found),
        compatible=not found,
    )


def freeze_world(
    project: ProjectStore,
    packet_reference: ArtifactInput,
    world_reference: ArtifactInput,
    *,
    sandbox: ProgramSandbox,
    code_revision: str,
    base: GroundedRuntimeDefinition | None = None,
    actor: ChatAgentRuntime | None = None,
    examples: Mapping[str, JsonObject] | None = None,
    observed: Mapping[str, Transition] | None = None,
    refusal: RefusalCriteria = DEFAULT_REFUSAL,
) -> WorldFreeze:
    """Freeze one ready authored world, or record exactly why its variants cannot qualify.

    Args:
        project: Project holding the packet, the world draft and the rule book.
        packet_reference: The saved authoring packet.
        world_reference: The saved ready world draft authored from it.
        sandbox: Bounded executor for the world's programs.
        code_revision: Producing source identity.
        base: Host-owned grounded runtime definition for the task; without it restoration
            is not proven and every variant stays diagnostic.
        actor: Chat actor whose initial context starts fresh-briefing variants; resume
            variants keep the packet checkpoint's actor.
        examples: Example arguments per tool; defaults to the first fixture call of the tool.
        observed: Observed transitions keyed ``<variant_id>:<workload_id>`` for workloads of
            predicted tools.
        refusal: How a refused action is recognized.

    Returns:
        The freeze, also saved as a ``world-freeze`` artifact and replayed from it.

    Raises:
        ValueError: The world draft is not ready or was authored from another packet.
    """
    packet = load_packet(project, packet_reference)
    draft = load_world_draft(project, world_reference)
    if draft.world is None or draft.packet_sha256 != packet.identity:
        raise ValueError(
            f"world draft {draft.identity} is {draft.status} for packet {draft.packet_sha256}; "
            "freeze only a ready world authored from this packet"
        )
    world = draft.world
    identity = stable_id(
        "world-freeze",
        {
            "protocol": WORLD_FREEZE_PROTOCOL,
            "packet": packet_reference.model_dump(mode="json"),
            "world": world_reference.model_dump(mode="json"),
            "base": None if base is None else sha256_json(base),
            "actor": None if actor is None else actor.identity,
            "examples": dict(examples or {}),
            "observed": {k: v.model_dump(mode="json") for k, v in sorted((observed or {}).items())},
            "refusal": refusal.model_dump(mode="json"),
            "code": code_revision,
        },
    )
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return WorldFreeze.model_validate_json(store.read_bytes(result_id, "freeze.json"))
        book = load_rulebook(project, packet.rule_book)
        checks = _Checks(world, book, sandbox, _examples(world, examples), refusal)
        checks.fixtures()
        for row in world.initializations:
            checks.workloads(row.variant_id, row.state, observed or {})
            checks.gates(row.variant_id, row.state)
        properties = check_transitions(properties_from(book), checks.located, refusal)
        supported = {rule.rule_id for rule in book.rules if rule.status == "supported"}
        for result in properties.results:
            fixture = [loc for loc in result.violations if loc.startswith("fixture:")]
            if result.rule_id in supported and fixture:
                checks.problems.append(
                    f"the world's own fixtures violate supported rule {result.rule_id} at "
                    f"{fixture[0]}"
                )
        contradicted = {
            run.workload.rule_id
            for run in checks.runs
            if run.triage is not None and run.triage.needs_witness
        }
        diagnostic = sorted(contradicted | checks.unsupported)
        variants = tuple(
            _freeze_variant(
                project,
                packet,
                packet_reference,
                world,
                row.variant_id,
                checks,
                diagnostic,
                base=base,
                actor=actor,
                code_revision=code_revision,
            )
            for row in world.initializations
        )
        freeze = WorldFreeze(
            identity=identity,
            packet_sha256=packet.identity,
            world=world_reference,
            world_sha256=world.identity,
            properties=properties,
            workloads=tuple(checks.runs),
            probes=tuple(checks.probes),
            diagnostic_rules=tuple(diagnostic),
            problems=tuple(checks.problems),
            variants=variants,
            trainable=not checks.problems and any(v.status == "frozen" for v in variants),
        )
        inputs = {packet_reference, world_reference, packet.rule_book}
        inputs |= {row.checkpoint for row in variants if row.checkpoint is not None}
        store.write_json(
            artifact_id=result_id,
            artifact_type=FREEZE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"freeze.json": freeze},
        )
        return freeze


class _Checks:
    """Executes the world's programs for fixtures, workloads and gate probes and keeps results."""

    def __init__(
        self,
        world: WorldSpec,
        book: RuleBook,
        sandbox: ProgramSandbox,
        examples: dict[str, JsonObject],
        refusal: RefusalCriteria,
    ) -> None:
        """Bind the world, its rule book and the executor.

        Args:
            world: The authored world.
            book: The packet's rule book.
            sandbox: Bounded program executor.
            examples: Example arguments per tool.
            refusal: How a refused action is recognized.
        """
        self.world, self.book, self.sandbox = world, book, sandbox
        self.examples, self.refusal = examples, refusal
        self.programs = {program.tool: program for program in world.programs}
        self.policies = {policy.name: policy for policy in world.manifest().tools}
        self.located: list[tuple[str, Transition]] = []
        self.runs: list[WorkloadRun] = []
        self.probes: list[GateProbe] = []
        self.problems: list[str] = []
        self.unsupported: set[str] = set()
        self.variant_problems: dict[str, list[str]] = {}
        self.variant_diagnostics: dict[str, list[str]] = {}

    def run(
        self, tool: str, state: JsonObject, arguments: JsonObject
    ) -> tuple[ProgramOutcome | None, Transition | None]:
        """Execute one call through the tool's program, if it has one.

        Args:
            tool: Tool name.
            state: Complete simulated state before the call.
            arguments: Call arguments.

        Returns:
            The outcome (None for a predicted tool) and the transition of a reply or error.
        """
        program: TransitionProgram | None = self.programs.get(tool)
        policy = self.policies.get(tool)
        if program is None or policy is None:
            return None, None
        visible = {
            name: deepcopy(state[name]) for name in sorted(policy.domains()) if name in state
        }
        outcome = self.sandbox.run(program, visible, arguments, {})
        if outcome.status not in ("reply", "error_reply"):
            return outcome, None
        after = deepcopy(state)
        for name in policy.writes:
            if outcome.state is not None and name in outcome.state:
                after[name] = deepcopy(outcome.state[name])
        observation = outcome.reply
        transition = Transition(
            before=deepcopy(state),
            after=after,
            action={"name": tool, "arguments": deepcopy(arguments)},
            observation=observation,
        )
        if outcome.status == "error_reply" and is_refusal(transition, self.refusal) is not True:
            transition = transition.model_copy(update={"observation": {"error": observation}})
        return outcome, transition

    def fixtures(self) -> None:
        """Replay every fixture through the programs and keep each executed transition."""
        for episode in self.world.fixtures:
            state = deepcopy(episode.initial_state)
            for index, step in enumerate(episode.steps):
                _, transition = self.run(step.tool, state, step.arguments)
                if transition is None:
                    break
                self.located.append((f"fixture:{episode.episode_id}:{index}", transition))
                state = transition.after

    def workloads(
        self, variant_id: str, state: JsonObject, observed: Mapping[str, Transition]
    ) -> None:
        """Generate and execute every boundary workload of one variant and triage it.

        Args:
            variant_id: The variant.
            state: Its initial state.
            observed: Supplied transitions for workloads of predicted tools.
        """
        for workload in boundary_workloads(self.book, state, examples=self.examples):
            tool = str(workload.action["name"])
            arguments = workload.action.get("arguments")
            _, transition = self.run(
                tool, workload.state, arguments if isinstance(arguments, dict) else {}
            )
            key = f"{variant_id}:{workload.workload_id}"
            transition = transition or observed.get(key)
            if transition is None:
                self.runs.append(
                    WorkloadRun(variant_id=variant_id, workload=workload, executed=False)
                )
                continue
            self.located.append((f"workload:{key}", transition))
            (found,) = triage((workload,), {workload.workload_id: transition}, self.refusal)
            self.runs.append(
                WorkloadRun(variant_id=variant_id, workload=workload, executed=True, triage=found)
            )

    def gates(self, variant_id: str, state: JsonObject) -> None:
        """Probe every load-bearing precondition of a state-writing tool in one variant.

        Args:
            variant_id: The variant.
            state: Its initial state.
        """
        for rule in self.book.rules:
            policy = self.policies.get(rule.tool or "")
            if rule.kind != "precondition" or policy is None or not policy.writes:
                continue
            if rule.status != "supported" or rule.predicate is None:
                self.unsupported.add(rule.rule_id)
                continue
            arguments = deepcopy(self.examples.get(policy.name, {}))
            self._probe(variant_id, rule, rule.predicate, policy, state, arguments, "variant")
            for path in _reads(rule.predicate):
                if not resolve_pointer(state, path)[0]:
                    continue
                trimmed = _without(state, path)
                if trimmed is not None:
                    self._probe(
                        variant_id,
                        rule,
                        rule.predicate,
                        policy,
                        trimmed,
                        arguments,
                        f"missing:{path}",
                    )

    def _probe(
        self,
        variant_id: str,
        rule: Rule,
        predicate: Predicate,
        policy: ToolEffectPolicy,
        state: JsonObject,
        arguments: JsonObject,
        case: str,
    ) -> None:
        """Run one gate probe and record how it resolved.

        Args:
            variant_id: The variant.
            rule: The supported precondition.
            predicate: Its predicate.
            policy: The gated tool's policy.
            state: The probed state.
            arguments: Call arguments.
            case: ``variant`` or ``missing:<pointer>``.
        """
        action: JsonObject = {"name": policy.name, "arguments": arguments}
        value = evaluate(predicate, Transition(before=state, after=state, action=action))
        outcome, transition = self.run(policy.name, state, arguments)
        resolution, detail = _resolve(value, outcome, transition, self.refusal)
        self.probes.append(
            GateProbe(
                variant_id=variant_id,
                rule_id=rule.rule_id,
                tool=policy.name,
                case=case,
                predicate=value,
                program_status=None if outcome is None else outcome.status,
                resolution=resolution,
                detail=detail,
            )
        )
        reason = f"{rule.rule_id} on {policy.name} ({case}): {detail}"
        if resolution in ("silently_allowed", "broken"):
            self.problems.append(reason)
        elif resolution == "undecidable" and case == "variant":
            self.variant_problems.setdefault(variant_id, []).append(reason)
        elif resolution in ("unknown", "unexplained_refusal") and case == "variant":
            self.variant_diagnostics.setdefault(variant_id, []).append(reason)


def _resolve(
    value: bool | None,
    outcome: ProgramOutcome | None,
    transition: Transition | None,
    refusal: RefusalCriteria,
) -> tuple[Resolution, str]:
    """Classify one gate probe from the precondition's value and what the world did.

    Args:
        value: The precondition on the probed state.
        outcome: The program outcome, or None for a predicted tool.
        transition: The executed transition, for a reply or error reply.
        refusal: How a refusal is recognized.

    Returns:
        The resolution and one sentence explaining it.
    """
    if outcome is None:
        if value is None:
            return "undecidable", (
                "the tool is predicted and the precondition reads absent state, so the rule "
                "check cannot refuse it; supply the state, model it with an erroring program, "
                "or reject the variant"
            )
        return "enforced", "the tool is predicted and the supported rule check decides it"
    if outcome.status == "out_of_domain":
        return "unknown", f"the program declares the input unsupported: {outcome.detail}"
    if transition is None:
        return "broken", f"the program {outcome.status}: {outcome.detail}"
    refused = is_refusal(transition, refusal)
    if refused:
        if value is True:
            return "unexplained_refusal", "the program refused although the precondition holds"
        return "refused", "the program refused the call with an error reply"
    changed = transition.after != transition.before
    if value is True:
        return "consistent", "the precondition holds and the program permitted the call"
    if not changed and refused is False:
        return "consistent", "the program permitted the call without any state effect"
    return "silently_allowed", (
        "the program performed a state-writing call while the precondition was "
        + ("false" if value is False else "undetermined because its state is missing")
    )


def _freeze_variant(
    project: ProjectStore,
    packet: AuthoringPacket,
    packet_reference: ArtifactInput,
    world: WorldSpec,
    variant_id: str,
    checks: _Checks,
    diagnostic: list[str],
    *,
    base: GroundedRuntimeDefinition | None,
    actor: ChatAgentRuntime | None,
    code_revision: str,
) -> VariantFreeze:
    """Decide one variant's status and prove its start restores, when it can be started.

    Args:
        project: Project receiving the checkpoint.
        packet: The packet.
        packet_reference: Its saved artifact.
        world: The authored world.
        variant_id: The variant.
        checks: Results of the executed checks.
        diagnostic: Rules keeping the family diagnostic.
        base: Host-owned grounded runtime definition, if any.
        actor: Chat actor for fresh-briefing starts, if any.
        code_revision: Producing source identity.

    Returns:
        The variant's freeze record.
    """
    problems = [*checks.problems, *checks.variant_problems.get(variant_id, [])]
    diagnostics = [*checks.variant_diagnostics.get(variant_id, [])]
    diagnostics += [
        f"rule {rule_id} has an unsupported decision boundary" for rule_id in diagnostic
    ]
    unexecuted = [
        run.workload.workload_id
        for run in checks.runs
        if run.variant_id == variant_id and not run.executed
    ]
    if unexecuted:
        diagnostics.append(f"boundary workloads {unexecuted} were not executed")
    if world.unsupported:
        diagnostics.append(f"the world declares unsupported behaviors {list(world.unsupported)}")
    if problems:
        return VariantFreeze(
            variant_id=variant_id,
            status="rejected",
            problems=tuple(problems),
            diagnostics=tuple(diagnostics),
        )
    if base is None:
        diagnostics.append("no runtime definition was supplied, so restoration is unproven")
        return VariantFreeze(
            variant_id=variant_id, status="diagnostic", diagnostics=tuple(diagnostics)
        )
    row = world.initialization(variant_id)
    definition = world_runtime_definition(packet, world, variant_id, base=base)
    try:
        reference = _start(
            project, packet, packet_reference, definition, row.mode, actor, code_revision
        )
    except ValueError as error:
        problems.append(f"the variant could not be started: {error}")
        return VariantFreeze(
            variant_id=variant_id,
            status="rejected",
            definition_sha256=sha256_json(definition),
            problems=tuple(problems),
            diagnostics=tuple(diagnostics),
        )
    with continuation_runtime(project, definition.model_dump(mode="json")) as runtime:
        executable = runtime.world if isinstance(runtime.world, ExecutableWorldRuntime) else None
        restored = executable is not None and verify_restoration(
            project, reference, runtime=executable, expected_state_sha256=sha256_json(row.state)
        )
    restored = (
        restored
        and grounded_state(load_continuation_checkpoint(project.artifacts, reference).world)
        == row.state
    )
    if not restored:
        problems.append("the frozen start does not restore to its exact initial state")
    status: Literal["frozen", "diagnostic", "rejected"] = (
        "rejected" if problems else "diagnostic" if diagnostics else "frozen"
    )
    return VariantFreeze(
        variant_id=variant_id,
        status=status,
        definition_sha256=sha256_json(definition),
        checkpoint=reference,
        restored=restored,
        problems=tuple(problems),
        diagnostics=tuple(diagnostics),
    )


def _start(
    project: ProjectStore,
    packet: AuthoringPacket,
    packet_reference: ArtifactInput,
    definition: GroundedRuntimeDefinition,
    mode: Literal["resume", "fresh_briefing"],
    actor: ChatAgentRuntime | None,
    code_revision: str,
) -> ArtifactInput:
    """Open the variant's world, pair it with its actor context and save the checkpoint.

    Raises:
        ValueError: A resume variant's packet checkpoint belongs to another task, or a fresh
            briefing has no actor to start it.
    """
    if mode == "resume":
        source = load_continuation_checkpoint(project.artifacts, packet.checkpoint)
        if source.task != definition.task:
            raise ValueError("the packet checkpoint's actor belongs to another task")
        context = source.actor
    elif actor is None:
        raise ValueError("a fresh briefing variant needs a chat actor to start it")
    else:
        context = actor.initial_checkpoint(definition.task)
    with (
        continuation_runtime(project, definition.model_dump(mode="json")) as runtime,
        runtime.world.open(definition.task) as session,
    ):
        start = ContinuationCheckpoint(
            task=definition.task,
            actor=context,
            world=session.checkpoint(),
            sources=(packet_reference,),
        )
    return persist_continuation_checkpoint(
        project.artifacts, start, created_at=datetime.now(UTC), code_revision=code_revision
    )


def _examples(world: WorldSpec, supplied: Mapping[str, JsonObject] | None) -> dict[str, JsonObject]:
    """Return example arguments per tool: supplied ones, else each tool's first fixture call."""
    found: dict[str, JsonObject] = {}
    for episode in world.fixtures:
        for step in episode.steps:
            found.setdefault(step.tool, step.arguments)
    return {**found, **dict(supplied or {})}


def _reads(predicate: Predicate) -> tuple[str, ...]:
    """Return every before-state pointer a predicate reads, in predicate order."""
    found: list[str] = []
    pending = [predicate]
    while pending:
        node = pending.pop(0)
        pending[0:0] = list(node.args)
        for operand in (node.left, node.right):
            if operand is not None and operand.scope == "before" and operand.path:
                if operand.path not in found:
                    found.append(operand.path)
    return tuple(found)


def _without(state: JsonObject, path: str) -> JsonObject | None:
    """Return a copy of a state with one object key deleted, or None when it cannot be."""
    tokens = [raw.replace("~1", "/").replace("~0", "~") for raw in path.split("/")[1:]]
    if len(tokens) < 2:
        return None
    trimmed = deepcopy(state)
    parent = "/" + "/".join(path.split("/")[1:-1])
    found, container = resolve_pointer(trimmed, parent)
    if not found or not isinstance(container, dict):
        return None
    del container[tokens[-1]]
    return trimmed


def _within(path: str, parent: str) -> bool:
    """Return whether a pointer equals or sits inside another."""
    return path == parent or path.startswith(parent + "/")

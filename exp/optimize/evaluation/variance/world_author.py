"""The world author: one forced draw that implements a decision family's executable world.

The world author reads one saved :class:`~exp.optimize.evaluation.variance.authoring_packet.
AuthoringPacket` and nothing else: never the rubric author's output, never a reward. It answers
through one forced ``submit_world_draft`` call, bought once and replayed exactly by the shared
author-response owner, and returns either a ready :class:`WorldSpec` or ``needs_research`` with
explicit :class:`~exp.optimize.evaluation.variance.authoring_packet.ResearchRequest` entries. A
blocked answer carries no world, so it never needs an invented checkpoint, program or artifact.

A ready world is expressed in the existing runtime contracts:

- state domains and per-tool read/write policies that form a
  :class:`~exp.runtime.environments.world_contracts.WorldManifest`;
- one initialization per packet variant (``resume`` keeps the checkpoint's coherent state and
  prefix, ``fresh_briefing`` adds a public briefing), each a complete simulated initial state;
- rule bindings to the packet's rule book, public tool contracts, private probe paths and the
  evidence hooks the episode judge relies on;
- induced transition programs in the induction ABI, ``transition(state, args, facts)``: mutate
  only the visible private state and return the JSON reply, raise ``ToolError(reply)`` for a
  modeled error or ``OutOfDomain(reason)`` for an unsupported input; the
  :class:`~exp.simulation.induction.sandbox.ProgramSandbox` builds the outcome envelope.

The host refuses an answer that cites unsupplied locators, misses or invents a variant or a
tool, enforces a rule the evidence does not support, leaves a probe or hook unresolvable, or
whose programs the existing induction validation does not accept on the packet's source episodes
plus the author's own replay fixtures. :func:`world_runtime_definition` then binds a ready world
into a :class:`~exp.optimize.evaluation.executable.grounded.GroundedRuntimeDefinition` for one
variant, programs included: covered tools are served by their validated programs and every other
call by the grounded model under the rule book. Whether an actor's episode in that world succeeds
is never decided here.

This module is the single forced draw and the host checks every world answer passes;
:mod:`~exp.optimize.evaluation.variance.world_agent` runs the agentic author (several prompt sets,
each planning, drafting and repairing from these checks' concrete failures) over the same checks.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    ResearchRequest,
    author_request,
    bounded_reason,
    buy_author_draw,
    check_research,
    forced_tool_arguments,
    load_packet,
    save_author_result,
)
from exp.optimize.evaluation.variance.contracts import resolve_pointer
from exp.optimize.evaluation.variance.episodes import EventKind
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.induction import runner
from exp.simulation.induction.contracts import (
    InducedCoverage,
    InductionEpisode,
    TransitionProgram,
    ValidationReport,
)
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.validation import validate_programs

WORLD_AUTHOR_PROTOCOL = "world-author-v2"
"""Version of the world-author prompt, answer schema and host checks."""

WORLD_TOOL = "submit_world_draft"
"""Name of the one tool the world author must call."""

WORLD_DRAFT_TYPE = "world-draft"
"""Artifact type of a saved world draft."""

RUNTIME_INTERFACE = (
    """Runtime target: the grounded world-model service with induced transition programs.
- State is a JSON object keyed by state domain. Declare every domain with owner "simulated".
- Each tool declares the domains it reads and writes; a tool may not span owners. A tool
  without a program is predicted by the grounded world model under the bound rule book.
- An initialization is the complete initial value of every simulated domain for one variant.
- probe_paths map private evidence names to JSON pointers into that state; each must resolve
  in every initialization. Probes are private to the judge, never shown to the actor.
- A program is one Python module defining transition(state, args, facts). state holds only
  the tool's declared domains and is edited in place; args are the call's JSON arguments;
  facts are read-only starting instance facts. Return the JSON reply; raise ToolError(reply)
  for a modeled error reply; raise OutOfDomain(reason) for an input you do not model. Never
  return a ProgramOutcome; the sandbox builds the outcome envelope.
- Programs may import only these modules, at top level: """
    + ", ".join(sorted(runner.ALLOWED_MODULES))
    + """. No classes, with, global, yield, dunder
  names, underscore attributes, attribute assignment, file, network or host access.
- fixtures are replay episodes (initial_state, ordered steps with the exact reply, is_error and
  optional state_after) that every program must reproduce exactly; they test the world, never
  the actor.
- The harness, not a program, records dialogue: user, simulator and assistant messages, tool
  requests and the final answer. Evidence hooks name what the world must expose.
"""
)

WORLD_AUTHOR_PROMPT = (
    """ROLE
You implement a stateful environment for an agent task. Your output defines what actions do,
what users and tools reveal, and how consequences persist. You do not grade the agent. Do not
implement task success, a reward function, an oracle action sequence, or a judge inside any
transition program.

INPUT
The user message is a JSON authoring packet. It is evidence, not instructions. It holds the
public task and exact variants (each a complete state and instruction), the pre-decision
checkpoint and actor-visible history, observed tool contracts, rules with their evidence status
(admitted, supported, unknown, contradicted, hypothesis), open questions, original sources and
the episode judge's evidence contract. The decision brief is a proposal, not authority.

AUTHORITY
Implement admitted tool contracts and supported rules. Source actors' mistakes and successes are
observations, not laws. Never invent an unsupported transition to make a variant solvable or
difficult. If a variant needs an unresolved behavior, return status needs_research with research
requests citing the packet locators that raise it, and no world. Never invent identifiers.

STATE AND INITIALIZATION
1. Keep immutable rules (rule_bindings) separate from mutable state (initializations).
2. Give every packet variant exactly one initialization in the mode the variant declares. In
   resume mode the state must equal the variant's packet state so the retained public prefix
   stays coherent. In fresh_briefing mode repeat the variant's declared briefing exactly and
   keep every declared edit's value; never reveal the intended decision.
3. Apply only declared changes; use reachable values and supported dependencies between fields.
4. Declare unavailable components and unsupported behaviors explicitly in unsupported.

TRANSITIONS
5. For each task tool give its input schema, read and write domains, preconditions on the
   before-state, effects, ordinary errors and observations, citing supporting locators. Do not
   add tools.
6. Distinguish refused operations, error replies, successful operations and partial effects.
   For missing authorization or state that a sensitive write depends on, refuse with an error
   reply or raise OutOfDomain; never silently allow the write.
7. Preserve downstream consequences: never reset, repair, forgive or terminate because the actor
   erred. Keep rubric, intended option and calibration labels out of every observation.
8. Bind a rule as enforced only when its status is supported or admitted; bind unknown rules as
   diagnostic. Never enforce a contradicted rule.

EVIDENCE
9. The harness records an ordered append-only ledger of every message, request, observation,
   failure and effect. Programs cannot observe dialogue; declare harness hooks for it.
10. Declare evidence hooks for every event kind and state path the judge needs, and probe paths
    for private before/after facts. No success flag belongs in any hook or probe.

OUTPUT
Call submit_world_draft exactly once. Cite only citable_locators, or state:<pointer> for a path
that exists in the packet states.

"""
    + RUNTIME_INTERFACE
)


class AuthoredTool(ContractModel):
    """The authored public contract and effect policy of one task tool.

    Attributes:
        name: Exact task tool name.
        input_schema: JSON schema of the tool's arguments.
        reads: Domains the tool reads.
        writes: Domains the tool may change.
        preconditions: Before-state conditions, including authorization scope.
        effects: Successful effects.
        errors: Ordinary error replies and refusals.
        observations: What a reply reveals.
        supported_by: Packet locators the contract rests on.
    """

    name: str = Field(min_length=1, max_length=256)
    input_schema: JsonObject
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    preconditions: str = Field(min_length=1, max_length=2000)
    effects: str = Field(min_length=1, max_length=2000)
    errors: str = Field(min_length=1, max_length=2000)
    observations: str = Field(min_length=1, max_length=2000)
    supported_by: tuple[str, ...] = Field(min_length=1)


class RuleBinding(ContractModel):
    """How the world uses one rule of the packet's rule book.

    Attributes:
        rule_id: Rule identity.
        use: ``enforced`` by the runtime's rule check, ``implemented`` by a program, or
            ``diagnostic``: recorded but not trusted to decide behavior.
        note: How the binding is realized.
    """

    rule_id: str = Field(min_length=1)
    use: Literal["enforced", "implemented", "diagnostic"]
    note: str = Field(min_length=1, max_length=1000)


class VariantInitialization(ContractModel):
    """The exact starting world of one packet variant.

    Attributes:
        variant_id: Packet variant identity.
        mode: ``resume`` keeps the checkpoint's prefix; ``fresh_briefing`` starts anew.
        state: Complete initial value of every simulated domain.
        briefing: Public briefing for fresh_briefing mode; None for resume.
    """

    variant_id: str = Field(min_length=1)
    mode: Literal["resume", "fresh_briefing"]
    state: JsonObject
    briefing: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _briefing(self) -> Self:
        """Require a briefing exactly in fresh_briefing mode."""
        if (self.mode == "fresh_briefing") != (self.briefing is not None):
            raise ValueError("a fresh_briefing initialization has a briefing; resume has none")
        return self


class EvidenceHook(ContractModel):
    """One record the judge relies on, and who must produce it.

    Attributes:
        kind: The episode event kind.
        owner: ``harness`` for dialogue and dispatch, ``world`` for replies, errors and state.
        detail: What the record must contain.
        state_path: State pointer the record exposes, when it exposes state.
    """

    kind: EventKind
    owner: Literal["harness", "world"]
    detail: str = Field(min_length=1, max_length=1000)
    state_path: str | None = Field(default=None, pattern=r"^(/[^/]*)+$")


class WorldSpec(ContractModel):
    """A ready authored world in existing runtime terms.

    Attributes:
        domains: Simulated state domains.
        tools: One authored contract per task tool.
        rule_bindings: How each used rule is bound.
        initializations: One per packet variant.
        probe_paths: Private evidence names mapped to state pointers.
        evidence_hooks: Records the judge relies on.
        programs: Induced transition programs, at most one per tool.
        fixtures: Replay fixtures every program must reproduce exactly.
        unsupported: Behaviors the world explicitly does not model.
        unresolved: Open questions that do not block a ready world.
    """

    domains: tuple[StateDomain, ...] = Field(min_length=1)
    tools: tuple[AuthoredTool, ...] = Field(min_length=1)
    rule_bindings: tuple[RuleBinding, ...] = ()
    initializations: tuple[VariantInitialization, ...] = Field(min_length=1)
    probe_paths: dict[str, str] = Field(min_length=1, max_length=128)
    evidence_hooks: tuple[EvidenceHook, ...] = Field(min_length=1)
    programs: tuple[TransitionProgram, ...] = ()
    fixtures: tuple[InductionEpisode, ...] = ()
    unsupported: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def manifest(self) -> WorldManifest:
        """Return the world manifest the domains and tool policies form."""
        return WorldManifest(
            domains=self.domains,
            tools=tuple(
                ToolEffectPolicy(name=tool.name, reads=tool.reads, writes=tool.writes)
                for tool in self.tools
            ),
        )

    def initialization(self, variant_id: str) -> VariantInitialization:
        """Return one variant's initialization.

        Raises:
            KeyError: The world does not initialize that variant.
        """
        for row in self.initializations:
            if row.variant_id == variant_id:
                return row
        raise KeyError(f"world has no initialization for variant {variant_id!r}")

    @property
    def identity(self) -> Sha256:
        """Digest of the complete world, bound into a judge definition as ``world_sha256``."""
        return sha256_json(self)


class WorldAnswer(ContractModel):
    """What the world author answers.

    Attributes:
        status: ``ready`` with a world, or ``needs_research`` with research requests.
        world: The world, exactly when ready.
        research: Evidence needed before a ready world; required when blocked.
        limitations: What the author could not ensure.
    """

    status: Literal["ready", "needs_research"]
    world: WorldSpec | None = None
    research: tuple[ResearchRequest, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_shape(self) -> Self:
        """A ready answer carries a world; a blocked one carries research and no world."""
        if self.status == "ready" and self.world is None:
            raise ValueError("a ready world answer carries its world")
        if self.status == "needs_research" and (self.world is not None or not self.research):
            raise ValueError("a needs_research answer carries research requests and no world")
        return self


class WorldDraft(ContractModel):
    """One world-author draw: a ready world, a research request, or why there is neither.

    Attributes:
        identity: Stable identity of the draw.
        packet_sha256: Digest of the packet the author read.
        status: ``ready``; ``needs_research``; ``refused`` when the saved answer failed a host
            check or program validation; ``unavailable`` when no response was obtained.
        response: Saved provider response, when one exists.
        world: The validated world, exactly when ready.
        research: The author's research requests.
        validation: Induction validation of the programs, when any were validated.
        failure: Why the draft is refused or unavailable.
    """

    identity: str = Field(min_length=1)
    packet_sha256: Sha256
    status: Literal["ready", "needs_research", "refused", "unavailable"]
    response: ArtifactInput | None = None
    world: WorldSpec | None = None
    research: tuple[ResearchRequest, ...] = ()
    validation: ValidationReport | None = None
    failure: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Keep each status's shape, so a blocked draft can never carry a fake world."""
        if (self.status == "ready") != (self.world is not None):
            raise ValueError("exactly a ready world draft carries a world")
        if (self.status in ("refused", "unavailable")) != (self.failure is not None):
            raise ValueError("exactly a refused or unavailable world draft carries a failure")
        if (self.status == "unavailable") != (self.response is None):
            raise ValueError("every world draft but an unavailable one keeps its saved response")
        if self.status == "needs_research" and not self.research:
            raise ValueError("a needs_research world draft carries research requests")
        return self


def world_protocol_sha256() -> Sha256:
    """Return the identity of the world-author prompt, schema and program target."""
    return sha256_json(
        {
            "protocol": WORLD_AUTHOR_PROTOCOL,
            "prompt": WORLD_AUTHOR_PROMPT,
            "schema": WorldAnswer.model_json_schema(),
            "runner": runner.RUNNER_PROTOCOL,
        }
    )


def world_request(packet: AuthoringPacket, *, output_tokens: int) -> ModelRequest:
    """Render one world-author draw from the packet alone.

    Args:
        packet: The authoring packet.
        output_tokens: Response allowance.

    Returns:
        The forced-tool request.
    """
    return author_request(
        WORLD_AUTHOR_PROMPT,
        packet,
        tool=WORLD_TOOL,
        schema=WorldAnswer.model_json_schema(),
        output_tokens=output_tokens,
    )


class WorldReviewError(ValueError):
    """A world answer failed a host check; carries the program validation when it ran.

    Attributes:
        report: Induction validation of the answer's programs, when validation ran.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        """Keep the failure and the validation report a repair round quotes."""
        super().__init__(message)
        self.report = report


def review_world_answer(
    arguments: object, packet: AuthoringPacket, sandbox: ProgramSandbox
) -> tuple[WorldAnswer, ValidationReport | None]:
    """Validate world-author arguments against the packet and validate every program.

    Args:
        arguments: The forced tool call's arguments.
        packet: The packet the author was shown.
        sandbox: Bounded executor for program validation.

    Returns:
        The answer and, for a ready world with programs, their validation report.

    Raises:
        ValueError: The answer is malformed, cites unsupplied locators, misses or invents a
            variant or tool, enforces an unsupported rule, leaves a probe or hook unresolvable,
            breaks a resume state, or has a program the induction validation does not accept
            or that the served runtime would never run.
    """
    answer = WorldAnswer.model_validate(arguments)
    check_research(packet, answer.research)
    world = answer.world
    if world is None:
        return answer, None
    invented = packet.unsupplied([ref for tool in world.tools for ref in tool.supported_by])
    if invented:
        raise ValueError(f"world cites locators that were not supplied: {invented}")
    _check_tools(world, packet)
    _check_rules(world, packet)
    _check_initializations(world, packet)
    manifest = world.manifest()
    if any(domain.owner != "simulated" for domain in manifest.domains):
        raise ValueError("authored worlds own simulated domains only")
    for name, pointer in world.probe_paths.items():
        for row in world.initializations:
            if not resolve_pointer(row.state, pointer)[0]:
                raise ValueError(f"probe {name} at {pointer} is absent from {row.variant_id}")
    for hook in world.evidence_hooks:
        if hook.kind not in packet.evidence.kinds():
            raise ValueError(f"evidence hook {hook.kind} is outside the judge's event contract")
        expected = "harness" if hook.kind in packet.evidence.harness_events else "world"
        if hook.owner != expected:
            raise ValueError(f"evidence hook {hook.kind} is recorded by the {expected}")
        if hook.state_path is not None and not all(
            resolve_pointer(row.state, hook.state_path)[0] for row in world.initializations
        ):
            raise ValueError(f"evidence hook path {hook.state_path} is absent from a variant")
    if not world.programs:
        return answer, None
    report = validate_programs(
        world.programs, (*packet.observed_episodes, *world.fixtures), manifest, sandbox
    )
    rejected = [row for row in report.programs if row.status != "accepted"]
    if rejected:
        reasons = "; ".join(
            f"{row.tool}: {row.status} ({row.reason or 'no reason'}"
            + (f"; {row.failures[0].status} {row.failures[0].detail or ''}" if row.failures else "")
            + ")"
            for row in rejected
        )
        raise WorldReviewError(
            f"induction validation did not accept every program: {reasons}", report
        )
    unserved = sorted(
        {program.tool for program in world.programs} - set(report.coverage.induced_tools)
    )
    if unserved:
        raise WorldReviewError(
            f"programs for {unserved} are valid but would never be served: a tool without a "
            "program touches their state domains, so the grounded model owns them; give every "
            "tool touching those domains a program, or drop these programs",
            report,
        )
    return answer, report


def author_world(
    project: ProjectStore,
    settings: AuthorSettings,
    packet_reference: ArtifactInput,
    *,
    sandbox: ProgramSandbox,
    identity: str,
    code_revision: str,
) -> WorldDraft:
    """Buy or replay one world-author draw over a saved packet and save its checked result.

    The draw takes its own lock named by ``identity``; the caller must not hold that lock.

    Args:
        project: Project whose artifacts hold the draw.
        settings: Frozen author model and allowance.
        packet_reference: The saved authoring packet; the author reads nothing else.
        sandbox: Bounded executor for program validation.
        identity: Stable identity of this draw.
        code_revision: Producing source identity.

    Returns:
        The draft. A draft with a saved response is saved as a ``world-draft`` artifact and
        replayed from it; a draw with no response is returned unsaved.
    """
    packet = load_packet(project, packet_reference)
    request = world_request(packet, output_tokens=settings.output_tokens)
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return WorldDraft.model_validate_json(store.read_bytes(result_id, "draft.json"))
        bought = buy_author_draw(
            project,
            settings,
            request,
            identity=identity,
            sources=(packet_reference,),
            code_revision=code_revision,
        )
        if isinstance(bought, str):
            return WorldDraft(
                identity=identity,
                packet_sha256=packet.identity,
                status="unavailable",
                failure=bought,
            )
        reference, response = bought
        try:
            answer, report = review_world_answer(
                forced_tool_arguments(response, WORLD_TOOL), packet, sandbox
            )
        except ValueError as error:
            draft = WorldDraft(
                identity=identity,
                packet_sha256=packet.identity,
                status="refused",
                response=reference,
                failure=bounded_reason(error),
            )
        else:
            draft = WorldDraft(
                identity=identity,
                packet_sha256=packet.identity,
                status=answer.status,
                response=reference,
                world=answer.world,
                research=answer.research,
                validation=report,
            )
        save_author_result(
            project,
            artifact_id=result_id,
            artifact_type=WORLD_DRAFT_TYPE,
            result=draft,
            inputs=(reference, packet_reference),
            code_revision=code_revision,
        )
        return draft


def load_world_draft(project: ProjectStore, reference: ArtifactInput) -> WorldDraft:
    """Read one exact saved world draft.

    Args:
        project: Project holding the draft.
        reference: The ``world-draft`` artifact, as :func:`world_draft_reference` returns it.

    Returns:
        The draft.

    Raises:
        ValueError: The reference is not the saved world draft it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        WORLD_DRAFT_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned world draft")
    return WorldDraft.model_validate_json(stored.payloads["draft.json"])


def world_draft_reference(project: ProjectStore, draft: WorldDraft) -> ArtifactInput:
    """Return the saved artifact of a draft that has a saved response.

    Raises:
        ValueError: The draft is unavailable, so nothing was saved.
    """
    if draft.response is None:
        raise ValueError(f"world draft {draft.identity} is unavailable; nothing was saved")
    return artifact_input(project.artifacts.read(draft.identity + "-result").manifest)


def world_coverage(packet: AuthoringPacket, world: WorldSpec) -> InducedCoverage | None:
    """Validate a ready world's programs and return the coverage the served runtime runs.

    Validation replays the packet's source episodes and the world's fixtures under the world's
    manifest in a default-limit sandbox, exactly as authoring did, so the coverage (and every
    digest of a definition that binds it) depends only on the packet and the world.

    Args:
        packet: The packet the world was authored from.
        world: The ready world.

    Returns:
        The coverage, or None when the world has no programs.

    Raises:
        ValueError: A program is not accepted or would not be served.
    """
    if not world.programs:
        return None
    with ProgramSandbox() as sandbox:
        report = validate_programs(
            world.programs, (*packet.observed_episodes, *world.fixtures), world.manifest(), sandbox
        )
    served = set(report.coverage.induced_tools)
    rejected = sorted({program.tool for program in world.programs} - served)
    if rejected:
        raise ValueError(f"the world's programs for {rejected} are not served; re-author them")
    return report.coverage


def world_runtime_definition(
    packet: AuthoringPacket,
    world: WorldSpec,
    variant_id: str,
    *,
    base: GroundedRuntimeDefinition,
    coverage: InducedCoverage | None = None,
) -> GroundedRuntimeDefinition:
    """Bind one variant of a ready world, programs included, into the grounded runtime.

    The host keeps its world model, catalog, allowance and deadline; the world supplies the
    manifest, initial state, probes and validated programs, the packet supplies the rule book and
    the variant's exact public instruction. Covered tools are served by their programs; every
    other call, and every call outside a program's validated domain, by the grounded model under
    the rule book. The programs are part of the definition, so its digest covers them.

    Args:
        packet: The packet the world was authored from.
        world: The ready world.
        variant_id: Which variant to start.
        base: Host-owned grounded runtime definition for the same task.
        coverage: The world's coverage from :func:`world_coverage`, to validate it once for
            several variants; computed when omitted.

    Returns:
        The validated definition.

    Raises:
        ValueError: The base runs another task, the coverage is not the world's, or the binding
            fails a definition check.
        KeyError: The world does not initialize that variant.
    """
    if base.task.task_id != packet.task.task_id:
        raise ValueError("the base runtime runs another task than the packet's")
    if coverage is None:
        coverage = world_coverage(packet, world)
    elif set(coverage.programs) != set(world.programs) or coverage.manifest != world.manifest():
        raise ValueError("the supplied coverage is not this world's validated programs")
    row = world.initialization(variant_id)
    variant = packet.variant(variant_id)
    # A resumed nudge keeps the task's own instruction; the nudge reaches the actor as a message.
    resumed = variant.nudge is not None and row.mode == "resume"
    instruction = packet.task.instruction if resumed else variant.instruction
    if row.briefing is not None:
        instruction = row.briefing + "\n\n" + instruction
    configuration = base.model_dump(mode="json")
    configuration.update(
        {
            "task": packet.task.model_copy(update={"instruction": instruction}).model_dump(
                mode="json"
            ),
            "manifest": world.manifest().model_dump(mode="json"),
            "initial_state": row.state,
            "probe_paths": dict(world.probe_paths),
            "rule_book": packet.rule_book.model_dump(mode="json"),
        }
    )
    configuration.pop("programs", None)
    if coverage is not None:
        configuration["programs"] = coverage.model_dump(mode="json")
    return GroundedRuntimeDefinition.model_validate(configuration)


def _check_tools(world: WorldSpec, packet: AuthoringPacket) -> None:
    """Require exactly the task's tools and programs only for authored tools.

    Raises:
        ValueError: A task tool is missing, a tool is invented, or a program targets an
            unknown tool.
    """
    authored = [tool.name for tool in world.tools]
    task = {tool.name for tool in packet.task.tools}
    if len(set(authored)) != len(authored) or set(authored) != task:
        raise ValueError(
            f"world must author exactly the task tools {sorted(task)}, got {sorted(authored)}"
        )
    unknown = sorted({program.tool for program in world.programs} - task)
    if unknown:
        raise ValueError(f"programs target tools the task does not have: {unknown}")


def _check_rules(world: WorldSpec, packet: AuthoringPacket) -> None:
    """Require bindings to packet rules, enforcing only supported ones.

    Raises:
        ValueError: A binding names an unknown rule or enforces an unsupported one.
    """
    rules = {rule.rule_id: rule for rule in packet.rules}
    for binding in world.rule_bindings:
        rule = rules.get(binding.rule_id)
        if rule is None:
            raise ValueError(f"world binds rule {binding.rule_id}, which the packet lacks")
        if binding.use != "diagnostic" and rule.status != "supported":
            raise ValueError(
                f"world {binding.use} rule {binding.rule_id} whose status is {rule.status}; "
                "bind it as diagnostic or return needs_research"
            )


def _check_initializations(world: WorldSpec, packet: AuthoringPacket) -> None:
    """Require one coherent initialization per packet variant.

    Raises:
        ValueError: A variant is missing or invented, a state names other domains, a resume
            state differs from the variant's packet state, a resumed instruction variant is not
            a nudge (whose message the resumed actor receives), or a fresh briefing drops an
            edit.
    """
    variants = {variant.variant_id: variant for variant in packet.variants}
    named = [row.variant_id for row in world.initializations]
    if len(set(named)) != len(named) or set(named) != set(variants):
        raise ValueError(
            f"world must initialize exactly the packet variants {sorted(variants)}, "
            f"got {sorted(named)}"
        )
    domains = {domain.name for domain in world.domains}
    for row in world.initializations:
        variant = variants[row.variant_id]
        if set(row.state) != domains:
            raise ValueError(f"initialization {row.variant_id} must name exactly the domains")
        if row.mode != variant.initialization:
            raise ValueError(
                f"initialization {row.variant_id} is {row.mode}, but the variant declares "
                f"{variant.initialization}"
            )
        if variant.briefing is not None and row.briefing != variant.briefing:
            raise ValueError(
                f"initialization {row.variant_id} rewrites the variant's declared briefing"
            )
        instructed = variant.axis in ("instruction", "state_and_instruction")
        if row.mode == "resume" and instructed and variant.nudge is None:
            raise ValueError(
                f"resume initialization {row.variant_id} keeps a prefix that shows the original "
                "instruction; an instruction variant needs fresh_briefing"
            )
        if row.mode == "resume" and sha256_json(row.state) != variant.state_sha256:
            raise ValueError(
                f"resume initialization {row.variant_id} differs from its variant state, so the "
                "retained prefix would contradict the world; use fresh_briefing or match it"
            )
        for edit in variant.edits:
            found, value = resolve_pointer(row.state, edit.path)
            if not found or sha256_json(value) != sha256_json(edit.value):
                raise ValueError(f"initialization {row.variant_id} drops edit {edit.path}")

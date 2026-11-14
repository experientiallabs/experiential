"""The agentic world author: several prompt sets, each planning, drafting and repairing a world.

:func:`author_world_agentic` runs every requested versioned prompt set of :data:`PROMPT_SETS`
(``contract-first``, ``state-machine-first``, ``failure-mode-first``) as its own trajectory:

1. **plan**: one forced ``submit_world_plan`` call naming the state machine, the tools worth a
   program, the failure modes and the fixtures that will exercise them;
2. **draft**: one forced ``submit_world_draft`` call over the packet and the author's own plan;
3. **repair**: while the draft fails a host check of
   :func:`~exp.optimize.evaluation.variance.world_author.review_world_answer` (packet citations,
   variants, rules, probes, hooks, and the programs run through
   :func:`~exp.simulation.induction.validation.validate_programs` in the
   :class:`~exp.simulation.induction.sandbox.ProgramSandbox`), up to ``maximum_repairs`` more
   draws that quote the previous answer and its exact failures, including the failing replay
   steps with expected and actual replies;
4. **final**: the first ready or ``needs_research`` draft, or the last refusal.

Every draw is bought once through the shared author-response owner and replayed exactly, and
every round is recorded. Each trajectory's final draft is saved as an ordinary ``world-draft``
artifact, so :func:`~exp.optimize.evaluation.variance.world_freeze.freeze_world` freezes it
unchanged. The author reads only the packet and its own earlier outputs; it never reads the
rubric author's output, and its plan never becomes authority over the packet.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelMessage, ModelRequest
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    ResearchRequest,
    author_request,
    bounded_reason,
    buy_author_draw,
    forced_tool_arguments,
    load_packet,
    save_author_result,
)
from exp.optimize.evaluation.variance.world_author import (
    WORLD_AUTHOR_PROMPT,
    WORLD_DRAFT_TYPE,
    WORLD_TOOL,
    WorldAnswer,
    WorldDraft,
    WorldReviewError,
    review_world_answer,
    world_protocol_sha256,
)
from exp.simulation.induction.contracts import ValidationReport
from exp.simulation.induction.sandbox import ProgramSandbox

WORLD_AGENT_PROTOCOL = "world-agent-v4"
"""Version of the prompt sets, phases, plan schema, repair feedback and selection."""

PLAN_TOOL = "submit_world_plan"
"""Name of the one tool the plan phase must call."""

PARALLEL_PROMPT_SETS = True
"""Run the prompt sets' trajectories at once (tests with an ordered fake transport turn it off)."""

WORLD_AUTHORING_TYPE = "world-authoring"
"""Artifact type of a saved agentic authoring record."""

Phase = Literal["plan", "draft", "repair"]
"""One phase of a trajectory."""


class WorldPromptSet(ContractModel):
    """One versioned way of approaching the same world.

    Attributes:
        name: Prompt set name.
        version: Version of its text; change it whenever the text changes.
        approach: The approach paragraph added to the shared world-author prompt.
    """

    name: str
    version: str
    approach: str

    def system(self, phase: Phase) -> str:
        """Return the system prompt of one phase."""
        return (
            f"{WORLD_AUTHOR_PROMPT}\nAPPROACH ({self.version})\n{self.approach}\n\n{_PHASES[phase]}"
        )

    @property
    def sha256(self) -> Sha256:
        """Digest of every phase prompt of this set, pinned by each trajectory."""
        return sha256_json({phase: self.system(phase) for phase in _PHASES})


_PHASES: dict[Phase, str] = {
    "plan": """PHASE plan
Do not write programs yet. Call submit_world_plan exactly once: describe the state machine
(domains, states and the transitions each tool causes), which tools deserve an exact program and
which stay with the grounded model, every failure mode you must model, and the replay fixtures
that will exercise each program path. Cite packet locators exactly as citable_locators lists
them. Record open questions as research notes; a question never stops the draft: a tool whose
reply the packet does not pin stays with the grounded model.""",
    "draft": """PHASE draft
The user message holds the authoring packet and your own plan. Your plan is a working note, not
evidence: where it disagrees with the packet, the packet wins. Call submit_world_draft once.""",
    "repair": """PHASE repair
Your previous answer failed the host checks quoted in host_failures, exactly as the validator and
the program sandbox reported them (failing replay steps carry the expected and actual replies).
Fix the world so every check passes. Never fix a check by enforcing an unsupported rule,
dropping a variant, deleting a fixture that exposes a real defect or hard-coding a fixture's
answer. Where host_failures says needs_research was declined, return status ready: leave each
unresolved tool or behavior to the grounded model (no program), list the open question under
unresolved (never unsupported), and keep every variant. Keep what passed. Call submit_world_draft once.""",
}

PROMPT_SETS: dict[str, WorldPromptSet] = {
    row.name: row
    for row in (
        WorldPromptSet(
            name="contract-first",
            version="contract-first-v1",
            approach="""Work contract-first. Start from each tool's admitted contract and its
observed replies. Write a program only where the contract and the source episodes pin the reply
and the state effect exactly; leave every other tool to the grounded model and the rule book.""",
        ),
        WorldPromptSet(
            name="state-machine-first",
            version="state-machine-first-v1",
            approach="""Work state-machine-first. Model the mutable state as explicit states and
transitions. For each tool state the guard on the before-state and the after-state it produces,
then implement programs for the transitions whose guard and effect are fully supported.""",
        ),
        WorldPromptSet(
            name="failure-mode-first",
            version="failure-mode-first-v1",
            approach="""Work failure-mode-first. Enumerate refusals, error replies, missing
authorization or state, partial effects and retries before the success paths. Every sensitive
write refuses or raises OutOfDomain when its precondition is false or undetermined, and the
fixtures cover each failure mode as well as the success path.""",
        ),
    )
}
"""Every versioned prompt set, by name."""

DEFAULT_PROMPT_SETS = tuple(PROMPT_SETS)
"""Prompt sets run when the caller names none, in selection order."""


class WorldPlan(ContractModel):
    """The plan phase's answer: the author's own working note for its draft.

    Attributes:
        state_machine: Domains, states and the transitions each tool causes.
        programs: Tools that deserve an exact program.
        predicted: Tools left to the grounded model and the rule book.
        failure_modes: Refusals, errors and partial effects to model.
        fixtures: Replay fixtures planned per program path.
        supported_by: Packet locators the plan rests on.
        research: Questions the packet does not answer.
    """

    state_machine: str = Field(min_length=1, max_length=16000)
    programs: tuple[str, ...] = ()
    predicted: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    supported_by: tuple[str, ...] = ()
    research: tuple[ResearchRequest, ...] = ()


class WorldRound(ContractModel):
    """One recorded round of a trajectory.

    Attributes:
        phase: Which phase.
        index: Round position in the trajectory, the plan first.
        response: Saved provider response, when one was obtained.
        status: ``planned`` for an accepted plan; a draft's status otherwise.
        failure: The exact host failure a repair round quotes, or why no response exists.
        validation: Program validation of the round's draft, when it ran.
    """

    phase: Phase
    index: int = Field(ge=0)
    response: ArtifactInput | None = None
    status: Literal["planned", "ready", "needs_research", "refused", "unavailable"]
    failure: str | None = None
    validation: ValidationReport | None = None


class WorldTrajectory(ContractModel):
    """One prompt set's plan, drafts and repairs, and its final draft.

    Attributes:
        prompt_set: Prompt set name.
        prompt_sha256: Digest of the set's phase prompts.
        rounds: Every round in order.
        draft: The final draft.
        saved: The saved ``world-draft`` artifact, when the final draft has a response.
    """

    prompt_set: str
    prompt_sha256: Sha256
    rounds: tuple[WorldRound, ...]
    draft: WorldDraft
    saved: ArtifactInput | None = None


class WorldAuthoring(ContractModel):
    """The complete record of one agentic authoring run.

    Attributes:
        protocol: Agent protocol.
        author_protocol_sha256: Digest of the shared world-author prompt, schema and target.
        identity: Identity of the run, which names every draw and saved draft.
        packet_sha256: Digest of the packet every trajectory read.
        maximum_repairs: Repair rounds allowed per trajectory.
        trajectories: One per prompt set, in the requested order.
        selected: The first prompt set whose final draft is ready, or None.
        draft: The selected saved world draft, or None.
    """

    protocol: str = WORLD_AGENT_PROTOCOL
    author_protocol_sha256: Sha256
    identity: str
    packet_sha256: Sha256
    maximum_repairs: int = Field(ge=0)
    trajectories: tuple[WorldTrajectory, ...]
    selected: str | None = None
    draft: ArtifactInput | None = None


def world_agent_protocol_sha256() -> Sha256:
    """Return the identity of every prompt set, the phases, the plan schema and the author."""
    return sha256_json(
        {
            "protocol": WORLD_AGENT_PROTOCOL,
            "author": world_protocol_sha256(),
            "sets": {name: row.sha256 for name, row in PROMPT_SETS.items()},
            "plan": WorldPlan.model_json_schema(),
        }
    )


def author_world_agentic(
    project: ProjectStore,
    settings: AuthorSettings,
    packet_reference: ArtifactInput,
    *,
    sandbox: ProgramSandbox,
    identity: str,
    code_revision: str,
    prompt_sets: Sequence[str] = DEFAULT_PROMPT_SETS,
    maximum_repairs: int = 2,
) -> WorldAuthoring:
    """Run, or replay, the agentic world author over a saved packet.

    The run takes its own lock named by ``identity``; the caller must not hold that lock.

    Args:
        project: Project whose artifacts hold every draw.
        settings: Frozen author model and allowance, shared by every draw.
        packet_reference: The saved authoring packet; the author reads nothing else.
        sandbox: Bounded executor for program validation.
        identity: Stable identity of the run.
        code_revision: Producing source identity.
        prompt_sets: Prompt set names to run, in selection order.
        maximum_repairs: Repair rounds allowed per trajectory after its first draft.

    Returns:
        The authoring record, also saved as a ``world-authoring`` artifact and replayed from it.

    Raises:
        ValueError: A prompt set is unknown or repeated, or ``maximum_repairs`` is negative.
    """
    unknown = sorted(set(prompt_sets) - PROMPT_SETS.keys())
    if unknown or not prompt_sets or len(set(prompt_sets)) != len(prompt_sets):
        raise ValueError(
            f"name distinct prompt sets from {sorted(PROMPT_SETS)}; got {list(prompt_sets)}"
        )
    if maximum_repairs < 0:
        raise ValueError("maximum_repairs cannot be negative")
    # A world with programs and fixtures outgrows a small output allowance; an operator floor
    # (EXP_WORLD_OUTPUT_TOKENS_FLOOR) raises it for new draws without touching the spec.
    floor = int(os.environ.get("EXP_WORLD_OUTPUT_TOKENS_FLOOR", "0") or 0)
    if floor > settings.output_tokens:
        settings = settings.model_copy(update={"output_tokens": floor})
    packet = load_packet(project, packet_reference)
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-authoring"
        if store.exists(result_id):
            return WorldAuthoring.model_validate_json(store.read_bytes(result_id, "draft.json"))
        run = _Run(project, settings, packet, packet_reference, sandbox, code_revision)
        # The prompt sets are independent draws: run them at once (order is kept); the shared
        # program sandbox is used under one lock.
        if not PARALLEL_PROMPT_SETS:
            trajectories = tuple(
                run.trajectory(PROMPT_SETS[name], f"{identity}-{name}", maximum_repairs)
                for name in prompt_sets
            )
        else:
            # The first ready world wins: stop waiting for the other prompt sets' repair rounds
            # (they finish in the background and their draws stay saved); each unfinished set is
            # recorded as superseded.
            pool = ThreadPoolExecutor(max_workers=max(1, len(prompt_sets)))
            futures = {
                pool.submit(
                    copy_context().run,
                    run.trajectory,
                    PROMPT_SETS[name],
                    f"{identity}-{name}",
                    maximum_repairs,
                ): name
                for name in prompt_sets
            }
            finished: dict[str, WorldTrajectory] = {}
            for future in as_completed(futures):
                row = future.result()
                finished[futures[future]] = row
                if row.draft.status == "ready":
                    break
            pool.shutdown(wait=False, cancel_futures=True)
            trajectories = tuple(
                finished.get(name)
                or WorldTrajectory(
                    prompt_set=name,
                    prompt_sha256=PROMPT_SETS[name].sha256,
                    rounds=(),
                    draft=WorldDraft(
                        identity=f"{identity}-{name}",
                        packet_sha256=packet.identity,
                        status="unavailable",
                        failure="superseded: another prompt set's world was ready first",
                    ),
                )
                for name in prompt_sets
            )
        chosen = next((row for row in trajectories if row.draft.status == "ready"), None)
        authoring = WorldAuthoring(
            author_protocol_sha256=world_protocol_sha256(),
            identity=identity,
            packet_sha256=packet.identity,
            maximum_repairs=maximum_repairs,
            trajectories=trajectories,
            selected=None if chosen is None else chosen.prompt_set,
            draft=None if chosen is None else chosen.saved,
        )
        save_author_result(
            project,
            artifact_id=result_id,
            artifact_type=WORLD_AUTHORING_TYPE,
            result=authoring,
            inputs=(packet_reference, *(row.saved for row in trajectories if row.saved)),
            code_revision=code_revision,
        )
        return authoring


def load_world_authoring(project: ProjectStore, reference: ArtifactInput) -> WorldAuthoring:
    """Read one exact saved agentic authoring record.

    Raises:
        ValueError: The reference is not the saved authoring record it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        WORLD_AUTHORING_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned world authoring run")
    return WorldAuthoring.model_validate_json(stored.payloads["draft.json"])


class _Run:
    """Shared inputs of every trajectory of one run."""

    def __init__(
        self,
        project: ProjectStore,
        settings: AuthorSettings,
        packet: AuthoringPacket,
        packet_reference: ArtifactInput,
        sandbox: ProgramSandbox,
        code_revision: str,
    ) -> None:
        """Bind the project, author settings, packet, executor and producing revision."""
        self.project, self.settings, self.sandbox = project, settings, sandbox
        self.sandbox_lock = threading.Lock()
        self.packet, self.packet_reference, self.code_revision = (
            packet,
            packet_reference,
            code_revision,
        )

    def trajectory(self, prompt: WorldPromptSet, identity: str, repairs: int) -> WorldTrajectory:
        """Plan, draft and repair one prompt set's world, recording every round.

        Args:
            prompt: The prompt set.
            identity: Identity of the trajectory; names its draws and saved draft.
            repairs: Repair rounds allowed after the first draft.

        Returns:
            The trajectory with its final draft, saved when it has a response.
        """
        rounds: list[WorldRound] = []
        plan = self._plan(prompt, identity, rounds)
        draft = WorldDraft(
            identity=identity,
            packet_sha256=self.packet.identity,
            status="unavailable" if rounds[-1].response is None else "refused",
            response=rounds[-1].response,
            failure=rounds[-1].failure or "the plan was refused",
        )
        context: JsonObject = {"your_plan": None if plan is None else plan.model_dump(mode="json")}
        for index in range(repairs + 1) if plan is not None else ():
            phase: Phase = "draft" if index == 0 else "repair"
            draft, feedback = self._draft(prompt, phase, identity, len(rounds), context, rounds)
            if feedback is None:
                break
            context = {**context, **feedback}
        saved = None
        if draft.response is not None:
            saved = save_author_result(
                self.project,
                artifact_id=identity + "-result",
                artifact_type=WORLD_DRAFT_TYPE,
                result=draft,
                inputs=(
                    self.packet_reference,
                    *(row.response for row in rounds if row.response is not None),
                ),
                code_revision=self.code_revision,
            )
        return WorldTrajectory(
            prompt_set=prompt.name,
            prompt_sha256=prompt.sha256,
            rounds=tuple(rounds),
            draft=draft,
            saved=saved,
        )

    def _plan(
        self, prompt: WorldPromptSet, identity: str, rounds: list[WorldRound]
    ) -> WorldPlan | None:
        """Buy or replay the plan and record its round; None when there is no usable plan."""
        bought = buy_author_draw(
            self.project,
            self.settings,
            self._request(prompt, "plan", PLAN_TOOL, WorldPlan.model_json_schema(), {}),
            identity=identity + "-plan",
            sources=(self.packet_reference,),
            code_revision=self.code_revision,
        )
        if isinstance(bought, str):
            rounds.append(WorldRound(phase="plan", index=0, status="unavailable", failure=bought))
            return None
        reference, response = bought
        try:
            plan = WorldPlan.model_validate(forced_tool_arguments(response, PLAN_TOOL))
            # The plan is a working note, not evidence: notes citing locators the packet never
            # supplied are dropped rather than refusing the whole plan.
            known = {variant.variant_id for variant in self.packet.variants}
            research = tuple(
                item
                for item in plan.research
                if not self.packet.unsupplied(list(item.cited)) and set(item.blocks) <= known
            )
            invented = set(self.packet.unsupplied(list(plan.supported_by)))
            plan = plan.model_copy(
                update={
                    "research": research,
                    "supported_by": tuple(r for r in plan.supported_by if r not in invented),
                }
            )
        except ValueError as error:
            rounds.append(
                WorldRound(
                    phase="plan",
                    index=0,
                    response=reference,
                    status="refused",
                    failure=bounded_reason(error),
                )
            )
            return None
        rounds.append(WorldRound(phase="plan", index=0, response=reference, status="planned"))
        return plan

    def _draft(
        self,
        prompt: WorldPromptSet,
        phase: Phase,
        identity: str,
        index: int,
        context: JsonObject,
        rounds: list[WorldRound],
    ) -> tuple[WorldDraft, JsonObject | None]:
        """Buy or replay one draft or repair, review it and record its round.

        Returns:
            The draft, and the repair feedback when another round may fix it (None to stop).
        """
        bought = buy_author_draw(
            self.project,
            self.settings,
            self._request(prompt, phase, WORLD_TOOL, WorldAnswer.model_json_schema(), context),
            identity=f"{identity}-{phase}-{index}",
            sources=(self.packet_reference,),
            code_revision=self.code_revision,
        )
        if isinstance(bought, str):
            rounds.append(
                WorldRound(phase=phase, index=index, status="unavailable", failure=bought)
            )
            unavailable = WorldDraft(
                identity=identity,
                packet_sha256=self.packet.identity,
                status="unavailable",
                failure=bought,
            )
            return unavailable, None
        reference, response = bought
        arguments: object = None
        try:
            arguments = forced_tool_arguments(response, WORLD_TOOL)
            with self.sandbox_lock:
                answer, report = review_world_answer(arguments, self.packet, self.sandbox)
        except ValueError as error:
            failure = bounded_reason(error)
            report = error.report if isinstance(error, WorldReviewError) else None
            rounds.append(
                WorldRound(
                    phase=phase,
                    index=index,
                    response=reference,
                    status="refused",
                    failure=failure,
                    validation=report,
                )
            )
            feedback: JsonObject = {
                "previous_answer": json.loads(json.dumps(arguments)),
                "host_failures": [failure],
                "failing_programs": []
                if report is None
                else [
                    row.model_dump(mode="json")
                    for row in report.programs
                    if row.status != "accepted"
                ],
            }
            draft = WorldDraft(
                identity=identity,
                packet_sha256=self.packet.identity,
                status="refused",
                response=reference,
                failure=failure,
            )
            return draft, feedback
        rounds.append(
            WorldRound(
                phase=phase,
                index=index,
                response=reference,
                status=answer.status,
                validation=report,
            )
        )
        if answer.status == "needs_research":
            declined: JsonObject = {
                "previous_answer": json.loads(json.dumps(arguments)),
                "host_failures": [
                    "needs_research declined: the grounded model simulates every tool without "
                    "an exact program from the state, the rule book and similar real outputs. "
                    "Return status ready; leave these questions' tools to the grounded model and "
                    "list each question under unresolved (not unsupported): "
                    + json.dumps([item.question for item in answer.research])[:6000]
                ],
                "failing_programs": [],
            }
        else:
            declined = None
        draft = WorldDraft(
            identity=identity,
            packet_sha256=self.packet.identity,
            status=answer.status,
            response=reference,
            world=answer.world,
            research=answer.research,
            validation=report,
        )
        return draft, declined

    def _request(
        self,
        prompt: WorldPromptSet,
        phase: Phase,
        tool: str,
        schema: JsonObject,
        context: JsonObject,
    ) -> ModelRequest:
        """Render one phase's forced request: the packet plus the author's own earlier work.

        Args:
            prompt: The prompt set.
            phase: The phase.
            tool: The forced tool.
            schema: Its input schema.
            context: The author's plan, previous answer and host failures, as applicable.

        Returns:
            The request.
        """
        request = author_request(
            prompt.system(phase),
            self.packet,
            tool=tool,
            schema=schema,
            output_tokens=self.settings.output_tokens,
        )
        if not context:
            return request
        system, user = request.messages
        payload = {**json.loads(str(user.content)), **context}
        return request.model_copy(
            update={
                "messages": (
                    system,
                    ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
                )
            }
        )

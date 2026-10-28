"""Ask a model for one transition program per tool, validate jointly and re-ask with feedback.

The loop is provider-free: callers inject ``draw``, which sends one provider-neutral request
under a caller-chosen operation identity and returns the model response, owning durability,
budgets and replay. Each round proposes a program for every tool still pending, records every
proposal, validates all current programs together by replaying the fit episodes, records the
report, and re-asks the rejected tools with their exact failing steps until each tool is
accepted or has used its attempts. Validation is deterministic, so replaying saved responses
reproduces every round and the final coverage exactly.

Prompts carry only fit evidence: tool schemas, declared effects, episode starting states,
starting instance facts and the recorded calls and replies. Held-out evidence never enters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import (
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolChoice,
)
from exp.common.tasks import ToolSchema
from exp.runtime.environments.world_contracts import WorldManifest
from exp.simulation.induction import runner
from exp.simulation.induction.contracts import (
    InductionEpisode,
    ProgramProposal,
    ProgramReport,
    TransitionProgram,
    ValidationReport,
)
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.validation import validate_programs

SUBMIT_TOOL = "submit_program"

type Draw = Callable[[ModelRequest, str], ModelResponse]
type Record = Callable[[ProgramProposal | ValidationReport], None]


class InductionSettings(ContractModel):
    """Finite allowances of one induction.

    Attributes:
        maximum_attempts: Proposals per tool, including re-asks after a failed validation.
        minimum_support: Exact fit steps a program needs before it can be accepted.
        maximum_prompt_characters: Evidence characters per request; whole episodes are added
            in order until the next would exceed it.
        maximum_feedback_failures: Failing steps quoted back on a re-ask.
        maximum_output_tokens: Output allowance of each proposal.
        temperature: Sampling temperature of each proposal.
    """

    maximum_attempts: int = Field(default=3, ge=1, le=8, strict=True)
    minimum_support: int = Field(default=2, ge=1, strict=True)
    maximum_prompt_characters: int = Field(default=60_000, ge=1_000, le=1_000_000, strict=True)
    maximum_feedback_failures: int = Field(default=6, ge=1, le=20, strict=True)
    maximum_output_tokens: int = Field(default=4096, ge=256, le=65_536, strict=True)
    temperature: float = Field(default=0.0, ge=0, le=2)

    def draws(self, tools: int) -> int:
        """Return the most provider draws an induction over ``tools`` tools can make."""
        return tools * self.maximum_attempts


def inducible_tools(
    manifest: WorldManifest, episodes: Iterable[InductionEpisode]
) -> tuple[str, ...]:
    """Return tools with fit steps whose declared domains are all simulated, sorted by name."""
    simulated = {row.name for row in manifest.domains if row.owner == "simulated"}
    called = {step.tool for episode in episodes for step in episode.steps}
    return tuple(
        sorted(
            tool.name
            for tool in manifest.tools
            if tool.name in called and tool.domains() <= simulated
        )
    )


def induce_programs(
    *,
    manifest: WorldManifest,
    tools: Iterable[ToolSchema],
    episodes: Iterable[InductionEpisode],
    draw: Draw,
    record: Record,
    sandbox: ProgramSandbox,
    settings: InductionSettings,
) -> ValidationReport:
    """Propose, validate and re-ask until every inducible tool is accepted or out of attempts.

    Args:
        manifest: World manifest declaring every tool's domains.
        tools: Actor-visible tool schemas, giving each tool's description and arguments.
        episodes: Fit episodes only.
        draw: Durable model call for one request under one operation identity.
        record: Durable sink for every proposal and every validation report.
        sandbox: Bounded executor for validation.
        settings: Attempt, support and prompt allowances.

    Returns:
        The final joint validation report, whose coverage is what induced code owns.
    """
    episodes = tuple(episodes)
    schemas = {tool.name: tool for tool in tools}
    targets = inducible_tools(manifest, episodes)
    current: dict[str, TransitionProgram] = {}
    attempts = dict.fromkeys(targets, 0)
    feedback: dict[str, str] = {}
    pending = set(targets)
    report = validate_programs((), episodes, manifest, sandbox)
    while pending:
        for tool in sorted(pending):
            attempt = attempts[tool]
            operation = f"tool-{targets.index(tool)}-attempt-{attempt}"
            request = program_request(
                tool,
                manifest=manifest,
                schemas=schemas,
                episodes=episodes,
                settings=settings,
                feedback=feedback.get(tool),
            )
            proposal = parse_proposal(draw(request, operation), tool, attempt, operation)
            record(proposal)
            attempts[tool] += 1
            if proposal.program is not None:
                current[tool] = proposal.program
            else:
                current.pop(tool, None)
                feedback[tool] = f"Your previous response was refused: {proposal.refusal}"
        report = validate_programs(
            current.values(),
            episodes,
            manifest,
            sandbox,
            minimum_support=settings.minimum_support,
        )
        record(report)
        verdicts = {row.tool: row for row in report.programs}
        for tool, row in verdicts.items():
            if row.status != "accepted":
                feedback[tool] = _feedback(current[tool], row, settings)
        pending = {
            tool
            for tool in targets
            if attempts[tool] < settings.maximum_attempts
            and (tool not in verdicts or verdicts[tool].status != "accepted")
        }
    return report


def program_request(
    tool: str,
    *,
    manifest: WorldManifest,
    schemas: dict[str, ToolSchema],
    episodes: tuple[InductionEpisode, ...],
    settings: InductionSettings,
    feedback: str | None,
) -> ModelRequest:
    """Build the provider-neutral request for one tool's program from fit evidence only."""
    policy = manifest.tool(tool)
    schema = schemas.get(tool)
    effects = [
        {"tool": row.name, "reads": sorted(row.domains()), "writes": sorted(row.writes)}
        for row in manifest.tools
    ]
    evidence: list[JsonObject] = []
    used = 0
    for episode in sorted(episodes, key=lambda row: (tool not in _tools(row), row.episode_id)):
        if tool not in _tools(episode):
            break
        rendered: JsonObject = {
            "episode": episode.episode_id,
            "initial_state": episode.initial_state,
            "facts": episode.facts,
            "calls": [
                {
                    "index": index,
                    "tool": step.tool,
                    "args": step.arguments,
                    "reply": step.reply,
                    "is_error": step.is_error,
                    **({"state_after": step.state_after} if step.state_after is not None else {}),
                }
                for index, step in enumerate(episode.steps)
            ],
        }
        size = len(json.dumps(rendered, sort_keys=True))
        if evidence and used + size > settings.maximum_prompt_characters:
            break
        evidence.append(rendered)
        used += size
    context: JsonObject = {
        "tool": tool,
        "description": None if schema is None else schema.description,
        "input_schema": None if schema is None else schema.input_schema,
        "reads": sorted(policy.domains()),
        "writes": sorted(policy.writes),
        "all_tool_effects": effects,
        "allowed_modules": {
            name: list(exported) for name, (_, exported) in sorted(runner.ALLOWED_MODULES.items())
        },
        "episodes": evidence,
    }
    messages = [
        ModelMessage(role="system", content=_INSTRUCTIONS),
        ModelMessage(role="user", content=json.dumps(context, sort_keys=True)),
    ]
    if feedback is not None:
        messages.append(ModelMessage(role="user", content=feedback))
    return ModelRequest(
        messages=tuple(messages),
        tools=(_SUBMIT_SCHEMA,),
        tool_choice=ToolChoice(name=SUBMIT_TOOL),
        temperature=settings.temperature,
        maximum_output_tokens=settings.maximum_output_tokens,
    )


def parse_proposal(
    response: ModelResponse, tool: str, attempt: int, operation: str
) -> ProgramProposal:
    """Return the program of one completed response, or a recorded refusal."""
    refusal = None
    source = None
    calls = response.output.tool_calls
    if response.finish_reason != ModelFinishReason.COMPLETED:
        refusal = f"generation stopped: {response.finish_reason.value}"
    elif len(calls) != 1 or calls[0].name != SUBMIT_TOOL:
        refusal = f"expected exactly one {SUBMIT_TOOL} call"
    else:
        value = calls[0].arguments.get("source")
        if not isinstance(value, str) or not value.strip():
            refusal = "the submitted source is empty"
        else:
            source = value
    return ProgramProposal(
        tool=tool,
        attempt=attempt,
        operation_id=operation,
        program=None if source is None else TransitionProgram(tool=tool, source=source),
        refusal=refusal,
    )


def _feedback(
    program: TransitionProgram, report: ProgramReport, settings: InductionSettings
) -> str:
    """Quote the previous program and its failing steps back for one re-ask."""
    failures = [
        row.model_dump(mode="json", exclude_none=True)
        for row in report.failures[: settings.maximum_feedback_failures]
    ]
    return (
        "Your previous program was not accepted: "
        + (report.reason or report.status)
        + ". Fix it so every recorded reply and state is reproduced exactly, then submit the "
        + "complete corrected program.\nPrevious program:\n"
        + program.source
        + "\nFailing steps (episode, call index, status, expected and actual replies):\n"
        + json.dumps(failures, sort_keys=True)
    )


def _tools(episode: InductionEpisode) -> set[str]:
    """Return the tools one episode calls."""
    return {step.tool for step in episode.steps}


_SUBMIT_SCHEMA = ToolSchema(
    name=SUBMIT_TOOL,
    description="Submit the complete Python source of the transition program.",
    input_schema={
        "type": "object",
        "required": ["source"],
        "additionalProperties": False,
        "properties": {"source": {"type": "string", "description": "Complete program text."}},
    },
)

_INSTRUCTIONS = """You write exact Python transition programs for one tool of a simulated world.

Write a module that defines transition(state, args, facts):
- state maps each domain the tool reads or writes to its JSON value. Edit the writable domains in
  place to the exact state after the call; never change read-only domains.
- args are the call's JSON arguments. facts maps starting instance-fact keys to values; they
  describe the starting world only and are read-only.
- Return the exact reply value. A reply that was recorded as JSON is a JSON value (dict, list,
  number, string, bool or null); a text reply is a string. For an error reply, raise
  ToolError(reply) with the exact recorded error reply.
- For any input you cannot model exactly from the evidence, raise OutOfDomain("reason"). Never
  guess: an input outside the evidence goes to another simulator.

The program must reproduce every recorded reply and every recorded state_after exactly when the
episodes are replayed in order from their initial_state. Other tools' calls change state too; use
their recorded replies to understand the state layout.

Restrictions: only top-level `import` of the allowed modules, each exposing only its listed
names; plain function definitions and constant assignments at top level; no classes, global,
with, yield, bare except, names starting with two underscores or attributes starting with an
underscore; no file, network, time or random access (they are unavailable). Keep the program
small and deterministic.

Submit the complete program with the submit_program tool."""

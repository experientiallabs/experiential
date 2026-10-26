"""Contracts for induced transition programs: evidence, programs, validation and coverage.

An induced program is exact Python for one tool's state transition. It is proposed by a model
from fit evidence, then accepted only where replaying that evidence reproduces every recorded
reply (and recorded state, when the evidence carries it) exactly. The accepted programs, the
input domain each one was validated on and the state domains the induced code owns form an
:class:`InducedCoverage`, the only input the induced world backend executes.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256, sha256_json
from exp.runtime.environments.world_contracts import WorldManifest, spanning_tools

# How one covered step replayed: ``exact``, a named mismatch, or ``unverifiable`` when the state
# the step reads is not known (its episode has no starting state for it, or an earlier step of a
# tool without an accepted program changed it).
StepStatus = Literal[
    "exact",
    "reply_mismatch",
    "error_mismatch",
    "state_mismatch",
    "undeclared_write",
    "invalid_state",
    "out_of_domain",
    "failed",
    "refused",
    "unverifiable",
]

FAILING_STATUSES: frozenset[StepStatus] = frozenset(
    {
        "reply_mismatch",
        "error_mismatch",
        "state_mismatch",
        "undeclared_write",
        "invalid_state",
        "out_of_domain",
        "failed",
        "refused",
    }
)


class SandboxLimits(ContractModel):
    """Finite resource limits for every program execution.

    Attributes:
        memory_bytes: Address-space limit of the worker process.
        cpu_seconds: CPU-time limit of one request, re-armed in the worker before each one.
        request_timeout_seconds: Wall-clock limit of one execution, including transfer.
        maximum_steps: Program lines one execution may run before it fails.
        maximum_output_bytes: Largest accepted worker response line.
    """

    memory_bytes: int = Field(default=512 * 1024 * 1024, ge=64 * 1024 * 1024, strict=True)
    cpu_seconds: int = Field(default=60, ge=1, le=3600, strict=True)
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    maximum_steps: int = Field(default=200_000, ge=100, le=50_000_000, strict=True)
    maximum_output_bytes: int = Field(default=8_000_000, ge=1024, le=64_000_000, strict=True)


class TransitionProgram(ContractModel):
    """One proposed program for one tool.

    Attributes:
        tool: Exact tool name the program implements.
        source: Program text defining ``transition(state, args, facts)``.
    """

    tool: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1)

    @property
    def sha256(self) -> Sha256:
        """Return the program identity bound by reports and coverage."""
        return sha256_json(self)


class InductionStep(ContractModel):
    """One recorded tool call with the reply the real environment returned.

    Attributes:
        tool: Called tool name.
        arguments: Recorded JSON arguments.
        reply: Recorded reply, decoded from JSON when it was JSON text.
        is_error: Whether the recorded reply is an error observation.
        state_after: Recorded state of named domains after the call, when the evidence has
            it; every named domain is compared exactly.
    """

    tool: str = Field(min_length=1, max_length=256)
    arguments: JsonObject
    reply: JsonValue
    is_error: bool = False
    state_after: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)


class InductionEpisode(ContractModel):
    """One fit episode replayed in order from its starting state.

    Attributes:
        episode_id: Unique episode identity, such as the trace ID.
        initial_state: Starting value of known simulated domains; absent domains are unknown
            and steps that read them are unverifiable until a recorded state names them.
        facts: Starting instance facts (key to value) every program receives read-only.
        steps: Recorded calls in call order.
    """

    episode_id: str = Field(min_length=1, max_length=512)
    initial_state: JsonObject = Field(default_factory=dict)
    facts: JsonObject = Field(default_factory=dict)
    steps: tuple[InductionStep, ...] = Field(min_length=1)


class ProgramOutcome(ContractModel):
    """What one sandboxed execution produced.

    Attributes:
        status: ``reply`` and ``error_reply`` carry a reply and state; ``out_of_domain`` is
            the program's own refusal; ``failed`` an exception, budget or limit; ``refused`` a
            program outside the admitted language subset.
        reply: Reply value of a reply or error reply.
        state: Visible state after the call, for a reply or error reply.
        lines: Program lines the execution ran.
        detail: Reason for a non-reply outcome.
    """

    status: Literal["reply", "error_reply", "out_of_domain", "failed", "refused"]
    reply: JsonValue = None
    state: JsonObject | None = None
    lines: tuple[int, ...] = ()
    detail: str | None = None


class ProgramDomain(ContractModel):
    """The inputs one accepted program was validated on.

    A later call is inside the domain only when its argument names match a validated call and
    every program line it runs also ran on a validated step. Any new code path, argument shape,
    program refusal or failure is outside the domain and goes to the predictor.

    Attributes:
        argument_keys: Sorted argument-name sets of validated calls.
        lines: Program lines run by validated calls.
    """

    argument_keys: tuple[tuple[str, ...], ...]
    lines: tuple[int, ...]

    def admits(self, arguments: JsonObject, outcome: ProgramOutcome) -> bool:
        """Return whether one execution stayed on validated arguments and code paths."""
        return (
            outcome.status in ("reply", "error_reply")
            and tuple(sorted(arguments)) in self.argument_keys
            and set(outcome.lines) <= set(self.lines)
        )


class StepCheck(ContractModel):
    """The replay result of one step of a program's tool.

    Attributes:
        episode_id: Episode of the step.
        index: Step position in the episode.
        status: Exact, a named mismatch, or unverifiable.
        expected: Recorded reply, kept for mismatches only.
        actual: Program reply, kept for mismatches only.
        detail: Short reason for a non-exact status.
    """

    episode_id: str
    index: int = Field(ge=0)
    status: StepStatus
    expected: JsonValue = None
    actual: JsonValue = None
    detail: str | None = None


class ProgramReport(ContractModel):
    """The validation verdict of one proposed program.

    Attributes:
        tool: Tool the program implements.
        program_sha256: Identity of the validated program.
        status: ``accepted`` when every evaluable step was exact and support was met;
            ``rejected`` otherwise; ``unsupported`` when the tool cannot run as induced code.
        evaluable: Steps whose read state was known.
        exact: Evaluable steps reproduced exactly.
        unverifiable: Steps whose read state was unknown.
        failures: Up to the first twenty failing checks, in replay order.
        reason: Why the program was rejected or unsupported.
        domain: Validated input domain of an accepted program.
    """

    tool: str
    program_sha256: Sha256
    status: Literal["accepted", "rejected", "unsupported"]
    evaluable: int = Field(ge=0)
    exact: int = Field(ge=0)
    unverifiable: int = Field(ge=0)
    failures: tuple[StepCheck, ...] = ()
    reason: str | None = None
    domain: ProgramDomain | None = None


class InducedCoverage(ContractModel):
    """Accepted programs and the state domains induced code owns for one world manifest.

    Ownership follows the world kernel's rule that no tool spans owners: every domain of an
    induced tool is induced, and no predicted tool touches an induced domain.

    Attributes:
        manifest: World manifest the programs were validated under.
        programs: Accepted programs of the induced tools, one per tool.
        domains: Validated input domain of each induced tool.
        induced_tools: Tools the induced code executes inside their validated domain.
        induced_domains: Simulated state domains only induced code writes at episode start.
        limits: Sandbox limits the programs were validated under.
    """

    manifest: WorldManifest
    programs: tuple[TransitionProgram, ...] = ()
    domains: dict[str, ProgramDomain] = Field(default_factory=dict)
    induced_tools: tuple[str, ...] = ()
    induced_domains: tuple[str, ...] = ()
    limits: SandboxLimits = Field(default_factory=SandboxLimits)

    @model_validator(mode="after")
    def single_owner(self) -> Self:
        """Refuse coverage whose partition lets a tool span induced and predicted domains."""
        names = [program.tool for program in self.programs]
        if len(set(names)) != len(names) or set(names) != set(self.induced_tools):
            raise ValueError("coverage needs exactly one accepted program per induced tool")
        if set(self.domains) != set(self.induced_tools):
            raise ValueError("coverage needs one validated domain per induced tool")
        simulated = {row.name for row in self.manifest.domains if row.owner == "simulated"}
        if not set(self.induced_domains) <= simulated:
            raise ValueError("induced code may own simulated domains only")
        policies = {tool.name: tool for tool in self.manifest.tools}
        if not set(self.induced_tools) <= policies.keys():
            raise ValueError("induced tools must be declared in the world manifest")
        routed = [tool for tool in self.manifest.tools if tool.domains() <= simulated]
        owners = {
            name: "induced" if name in self.induced_domains else "predicted" for name in simulated
        }
        if spanning := spanning_tools(routed, owners):
            raise ValueError(
                f"tool {spanning[0]!r} spans induced and predicted domains; induced coverage "
                "must give every tool's domains one owner"
            )
        for name in self.induced_tools:
            if not policies[name].domains() <= set(self.induced_domains):
                raise ValueError(f"induced tool {name!r} touches a domain induced code lacks")
        return self

    def program(self, tool: str) -> TransitionProgram:
        """Return the accepted program of one induced tool."""
        for program in self.programs:
            if program.tool == tool:
                return program
        raise KeyError(f"tool {tool!r} has no accepted induced program")


class ValidationReport(ContractModel):
    """Every program's verdict over the fit episodes and the coverage they support.

    Attributes:
        manifest_sha256: Digest of the world manifest replayed under.
        episodes: Fit episodes replayed.
        steps: Recorded steps across those episodes.
        rounds: Joint replays until no candidate program failed.
        programs: One report per proposed program.
        coverage: Accepted programs and owned domains.
    """

    manifest_sha256: Sha256
    episodes: int = Field(ge=0)
    steps: int = Field(ge=0)
    rounds: int = Field(ge=0)
    programs: tuple[ProgramReport, ...]
    coverage: InducedCoverage


class ProgramProposal(ContractModel):
    """One retained model proposal, with the program it parsed to or why it did not parse.

    Attributes:
        tool: Tool the proposal targets.
        attempt: Zero-based proposal attempt for the tool.
        operation_id: Identity of the model draw that produced it.
        program: Parsed program, or None when the response held no usable program.
        refusal: Why the response produced no program.
    """

    tool: str
    attempt: int = Field(ge=0)
    operation_id: str
    program: TransitionProgram | None = None
    refusal: str | None = None

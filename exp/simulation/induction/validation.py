"""Validate induced programs by replaying fit episodes, and derive what induced code owns.

Every episode replays in order from its starting state, threading one exact state through the
calls. A candidate program runs on each of its tool's steps whose read domains are known and
must reproduce the recorded reply, error flag and any recorded state exactly, write only its
tool's declared domains and keep every domain schema-valid. Steps of other tools, and steps
whose program failed, make the domains they write unknown, so a later read of those domains is
``unverifiable``: never counted as evidence for or against a program.

Validation is joint and conservative. A program with any failing step is rejected and the whole
replay runs again without it, until a round rejects nothing, because one wrong write could make
another program's correct read look exact. Programs below the support threshold are rejected
too. Ownership is then closed under the world kernel's rule that no tool spans owners: a state
domain is induced only when every simulated tool that touches it has an accepted program, and an
accepted tool whose domains are not all induced is left to the predictor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from exp.common.core.artifacts import JsonObject, sha256_json, strict_equal
from exp.runtime.environments.world_contracts import WorldManifest
from exp.runtime.environments.world_responses import validate_state_projection
from exp.simulation.induction.contracts import (
    FAILING_STATUSES,
    InducedCoverage,
    InductionEpisode,
    ProgramDomain,
    ProgramReport,
    StepCheck,
    TransitionProgram,
    ValidationReport,
)
from exp.simulation.induction.sandbox import ProgramSandbox

_REPORTED_FAILURES = 20


@dataclass
class _Evidence:
    """One candidate's checks in one replay round, with the inputs of its exact steps."""

    checks: list[StepCheck] = field(default_factory=list)
    argument_keys: set[tuple[str, ...]] = field(default_factory=set)
    lines: set[int] = field(default_factory=set)


def validate_programs(
    programs: Iterable[TransitionProgram],
    episodes: Iterable[InductionEpisode],
    manifest: WorldManifest,
    sandbox: ProgramSandbox,
    *,
    minimum_support: int = 1,
) -> ValidationReport:
    """Replay fit episodes with candidate programs and return every verdict and the coverage.

    Args:
        programs: At most one proposed program per tool.
        episodes: Fit episodes with their starting state and recorded steps.
        manifest: World manifest declaring each tool's read and write domains.
        sandbox: Bounded executor for the programs.
        minimum_support: Exact steps a program needs before it can be accepted.

    Returns:
        One report per program and the coverage of the accepted, owned programs.

    Raises:
        ValueError: Two programs target one tool or the support threshold is not positive.
    """
    if minimum_support < 1:
        raise ValueError("minimum_support must be at least one exact step")
    programs = tuple(programs)
    proposed = {program.tool: program for program in programs}
    if len(proposed) != len(programs):
        raise ValueError("validate at most one program per tool")
    episodes = tuple(episodes)
    simulated = {row.name for row in manifest.domains if row.owner == "simulated"}
    policies = {tool.name: tool for tool in manifest.tools}
    final: dict[str, ProgramReport] = {}
    for tool, program in proposed.items():
        policy = policies.get(tool)
        if policy is None or not policy.domains() <= simulated:
            final[tool] = ProgramReport(
                tool=tool,
                program_sha256=program.sha256,
                status="unsupported",
                evaluable=0,
                exact=0,
                unverifiable=0,
                reason="the tool is not declared, or touches state the simulated side does not own",
            )
    candidates = {tool: program for tool, program in proposed.items() if tool not in final}
    rounds = 0
    evidence: dict[str, _Evidence] = {}
    while candidates:
        rounds += 1
        evidence = _replay(candidates, episodes, manifest, sandbox)
        rejected = {}
        for tool, program in candidates.items():
            report = _report(tool, program, evidence[tool], minimum_support)
            if report.status != "accepted":
                rejected[tool] = report
        final.update(rejected)
        if not rejected:
            break
        candidates = {tool: program for tool, program in candidates.items() if tool not in rejected}
    accepted = {
        tool: _report(tool, program, evidence[tool], minimum_support)
        for tool, program in candidates.items()
    }
    owned_tools, owned_domains = induced_partition(manifest, accepted)
    for tool, report in accepted.items():
        final[tool] = report
    coverage = InducedCoverage(
        manifest=manifest,
        programs=tuple(candidates[tool] for tool in sorted(owned_tools)),
        domains={tool: _domain(accepted[tool]) for tool in sorted(owned_tools)},
        induced_tools=tuple(sorted(owned_tools)),
        induced_domains=tuple(sorted(owned_domains)),
        limits=sandbox.limits,
    )
    return ValidationReport(
        manifest_sha256=sha256_json(manifest),
        episodes=len(episodes),
        steps=sum(len(episode.steps) for episode in episodes),
        rounds=rounds,
        programs=tuple(final[tool] for tool in sorted(final)),
        coverage=coverage,
    )


def induced_partition(
    manifest: WorldManifest, accepted: Iterable[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the tools and domains induced code can own without any tool spanning owners.

    A simulated domain touched by a tool without an accepted program belongs to the predictor;
    an accepted tool that touches such a domain is left to the predictor too, which can in turn
    move its other domains. The fixpoint is the largest consistent induced partition.

    Args:
        manifest: World manifest with each tool's declared domains.
        accepted: Tools with accepted programs.

    Returns:
        The induced tools and the simulated domains they own.
    """
    simulated = {row.name for row in manifest.domains if row.owner == "simulated"}
    routed = {tool.name: tool.domains() for tool in manifest.tools if tool.domains() <= simulated}
    induced = {name for name in accepted if name in routed}
    while True:
        predicted = {
            domain for name, domains in routed.items() if name not in induced for domain in domains
        }
        kept = {name for name in induced if not routed[name] & predicted}
        if kept == induced:
            break
        induced = kept
    return frozenset(induced), frozenset(domain for name in induced for domain in routed[name])


def _replay(
    candidates: Mapping[str, TransitionProgram],
    episodes: tuple[InductionEpisode, ...],
    manifest: WorldManifest,
    sandbox: ProgramSandbox,
) -> dict[str, _Evidence]:
    """Replay every episode once with the current candidates and collect their checks."""
    evidence = {tool: _Evidence() for tool in candidates}
    simulated = {row.name for row in manifest.domains if row.owner == "simulated"}
    schemas = {
        row.name: row.state_schema for row in manifest.domains if row.state_schema is not None
    }
    policies = {tool.name: tool for tool in manifest.tools}
    for episode in episodes:
        state: JsonObject = {
            name: deepcopy(value)
            for name, value in episode.initial_state.items()
            if name in simulated
        }
        unknown = simulated - state.keys()
        for index, step in enumerate(episode.steps):
            policy = policies.get(step.tool)
            writes = frozenset(policy.writes) if policy is not None else frozenset(simulated)
            program = candidates.get(step.tool)
            if program is None or policy is None:
                unknown |= writes
                _ground(state, unknown, step.state_after, simulated)
                continue
            if policy.domains() & unknown:
                evidence[step.tool].checks.append(
                    StepCheck(
                        episode_id=episode.episode_id,
                        index=index,
                        status="unverifiable",
                        detail="read state is unknown at this step",
                    )
                )
                unknown |= writes
                _ground(state, unknown, step.state_after, simulated)
                continue
            visible = {name: deepcopy(state[name]) for name in sorted(policy.domains())}
            outcome = sandbox.run(program, visible, step.arguments, episode.facts)
            check = StepCheck(episode_id=episode.episode_id, index=index, status="exact")
            after = outcome.state or {}
            if outcome.status in ("out_of_domain", "failed", "refused"):
                check = check.model_copy(
                    update={"status": outcome.status, "detail": outcome.detail}
                )
            elif (outcome.status == "error_reply") != step.is_error:
                check = check.model_copy(
                    update={
                        "status": "error_mismatch",
                        "expected": step.reply,
                        "actual": outcome.reply,
                        "detail": f"recorded is_error={step.is_error}",
                    }
                )
            elif not strict_equal(outcome.reply, step.reply):
                check = check.model_copy(
                    update={
                        "status": "reply_mismatch",
                        "expected": step.reply,
                        "actual": outcome.reply,
                    }
                )
            elif set(after) != set(visible) or any(
                not strict_equal(after[name], visible[name])
                for name in visible
                if name not in writes
            ):
                check = check.model_copy(
                    update={
                        "status": "undeclared_write",
                        "detail": "the program changed or removed a domain its tool may not write",
                    }
                )
            else:
                try:
                    validate_state_projection(
                        after, {name: schemas[name] for name in writes if name in schemas}
                    )
                except ValueError as error:
                    check = check.model_copy(
                        update={"status": "invalid_state", "detail": str(error)[:500]}
                    )
            recorded = step.state_after or {}
            if check.status == "exact":
                mismatched = sorted(
                    name
                    for name, value in recorded.items()
                    if name in after and not strict_equal(after[name], value)
                )
                if mismatched:
                    check = check.model_copy(
                        update={
                            "status": "state_mismatch",
                            "detail": f"recorded state differs in {mismatched!r}",
                        }
                    )
            evidence[step.tool].checks.append(check)
            if check.status == "exact":
                for name in writes:
                    state[name] = after[name]
                evidence[step.tool].argument_keys.add(tuple(sorted(step.arguments)))
                evidence[step.tool].lines.update(outcome.lines)
            else:
                unknown |= writes
            _ground(state, unknown, step.state_after, simulated)
    return evidence


def _ground(
    state: JsonObject, unknown: set[str], recorded: JsonObject | None, simulated: set[str]
) -> None:
    """Adopt recorded post-step state: a recorded domain is known again from this point."""
    for name, value in (recorded or {}).items():
        if name in simulated:
            state[name] = deepcopy(value)
            unknown.discard(name)


def _report(
    tool: str, program: TransitionProgram, evidence: _Evidence, minimum_support: int
) -> ProgramReport:
    """Summarize one candidate's checks into a verdict."""
    failures = tuple(check for check in evidence.checks if check.status in FAILING_STATUSES)
    exact = sum(check.status == "exact" for check in evidence.checks)
    unverifiable = sum(check.status == "unverifiable" for check in evidence.checks)
    reason = None
    if failures:
        reason = f"{len(failures)} evaluable step(s) were not reproduced exactly"
    elif exact < minimum_support:
        reason = f"{exact} exact step(s); acceptance needs at least {minimum_support}"
    return ProgramReport(
        tool=tool,
        program_sha256=program.sha256,
        status="rejected" if reason else "accepted",
        evaluable=exact + len(failures),
        exact=exact,
        unverifiable=unverifiable,
        failures=failures[:_REPORTED_FAILURES],
        reason=reason,
        domain=None
        if reason
        else ProgramDomain(
            argument_keys=tuple(sorted(evidence.argument_keys)), lines=tuple(sorted(evidence.lines))
        ),
    )


def _domain(report: ProgramReport) -> ProgramDomain:
    """Return the validated domain of an accepted report."""
    if report.domain is None:
        raise ValueError(f"program for {report.tool!r} has no validated domain")
    return report.domain

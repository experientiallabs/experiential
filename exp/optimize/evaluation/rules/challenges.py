"""Executed rule challenges: run boundary workloads in a bound world and update rule status.

The rules stage generates boundary workloads (:func:`~exp.simulation.rules.workloads.
boundary_workloads`), each a state and an action on one side of a learned rule's boundary with
the outcome the rule predicts. :func:`run_challenges` executes every workload through one
:class:`ChallengeExecutor`, triages what the world did against the rule's prediction with
:func:`~exp.simulation.rules.workloads.triage`, and records per rule:

- ``supported`` when at least one executed workload agreed and none contradicted it;
- ``contradicted`` when an executed workload contradicted it and the rule has no supporting
  independent (native or specification) witness: the rule or the world is wrong and nothing here
  decides which, so the rule is listed in ``diagnostic`` and any family built on it stays
  diagnostic until an independent witness exists;
- a contradiction of an independently witnessed rule leaves the rule's status unchanged and is
  listed in ``world_defects``: the world, not the rule, is wrong;
- otherwise (nothing executed or determined) the rule keeps its prior status.

Executed witnesses are simulated: they are recorded as ``challenge:<identity>:<case>`` evidence on
the updated book, never as native witnesses, so :func:`~exp.simulation.rules.merge.
diagnostic_rules` still lists every rule without an independent witness.

Two executors are provided. :class:`ProgramChallenges` runs a tool's validated transition program
in the :class:`~exp.simulation.induction.sandbox.ProgramSandbox` and leaves other tools
unexecuted unless an observed transition is supplied. :class:`RuntimeChallenges` restores the
bound grounded runtime (programs first when it binds them, the grounded model under the rule book
otherwise) at each workload's state and executes the action there.

Every case result is saved as it completes and the report under an identity that binds the rule
book, the cases, the executor and the code, so a repeated run replays without executing anything
or calling a model. Infrastructure stops are raised and never recorded as world behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
from types import TracebackType
from typing import Literal, Protocol, Self

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
    simulated_image,
    with_simulated_image,
)
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.environments.world_contracts import ExecutableCheckpoint, ToolEffectPolicy
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.evaluation.workflow import INFRASTRUCTURE_STOPS
from exp.simulation.induction.contracts import ProgramOutcome, TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.rules.model import RuleBook, RuleStatus, Transition
from exp.simulation.rules.properties import DEFAULT_REFUSAL, RefusalCriteria, is_refusal
from exp.simulation.rules.workloads import Triage, Workload, triage

CHALLENGE_PROTOCOL = "rule-challenges-v1"
"""Version of challenge execution, triage use and status update; change it when they change."""

CHALLENGES_TYPE = "rule-challenges"
"""Artifact type of a saved challenge report."""

CHALLENGE_CASE_TYPE = "rule-challenge-case"
"""Artifact type of one saved executed case."""


class ChallengeCase(ContractModel):
    """One workload to execute, within a scope such as a variant.

    Attributes:
        scope: Where the workload's base state came from, such as a variant id; empty for one
            base state.
        workload: The rules stage's workload.
    """

    scope: str = ""
    workload: Workload

    @property
    def key(self) -> str:
        """Identity of the case, ``<scope>:<workload_id>`` or the workload id alone."""
        return (
            f"{self.scope}:{self.workload.workload_id}" if self.scope else self.workload.workload_id
        )


class ChallengeExecutor(Protocol):
    """Executes one workload in a bound world."""

    @property
    def identity(self) -> Sha256:
        """Digest of everything that decides what the executor does."""
        ...

    def execute(self, case: ChallengeCase) -> Transition | str | None:
        """Execute one case.

        Returns:
            The observed transition; a failure reason when the world failed to answer; None
            when this executor cannot execute the case's tool.
        """
        ...


class ChallengeRun(ContractModel):
    """What executing one case showed.

    Attributes:
        case: The case.
        status: ``executed``, ``unexecuted`` when the executor cannot run its tool, or ``failed``
            when the world failed to answer.
        transition: The observed transition, when executed.
        triage: The triage of that transition against the rule's prediction.
        detail: Why the case is unexecuted or failed.
    """

    case: ChallengeCase
    status: Literal["executed", "unexecuted", "failed"]
    transition: Transition | None = None
    triage: Triage | None = None
    detail: str | None = None


class RuleChallenge(ContractModel):
    """Every challenge of one rule and the status they support.

    Attributes:
        rule_id: The rule.
        prior: Its status in the challenged book.
        status: Its status after execution.
        consistent: Case keys whose outcome agreed with the rule.
        contradicting: Case keys whose outcome contradicted it.
        undetermined: Case keys executed without a determined verdict.
        unexecuted: Case keys not executed or failed.
        witnessed: Whether the rule has a supporting independent witness.
    """

    rule_id: str
    prior: RuleStatus
    status: RuleStatus
    consistent: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    undetermined: tuple[str, ...] = ()
    unexecuted: tuple[str, ...] = ()
    witnessed: bool


class ChallengeReport(ContractModel):
    """The complete record of executing one set of challenges.

    Attributes:
        protocol: Challenge protocol.
        identity: Identity of the run, which names its saved report.
        rulebook: The challenged rule book artifact.
        executor_sha256: Identity of the executor.
        runs: One per case, in case order.
        rules: One per challenged rule, in book order.
        diagnostic: Rules contradicted without an independent witness; families on them stay
            diagnostic.
        world_defects: Independently witnessed rules the world contradicted.
        book: The rule book with updated statuses and executed evidence.
    """

    protocol: str = CHALLENGE_PROTOCOL
    identity: str
    rulebook: ArtifactInput
    executor_sha256: Sha256
    runs: tuple[ChallengeRun, ...]
    rules: tuple[RuleChallenge, ...]
    diagnostic: tuple[str, ...]
    world_defects: tuple[str, ...]
    book: RuleBook

    def observed(self) -> dict[str, Transition]:
        """Return every executed transition by case key."""
        return {run.case.key: run.transition for run in self.runs if run.transition is not None}


def run_challenges(
    project: ProjectStore,
    *,
    book: RuleBook,
    rulebook: ArtifactInput,
    cases: Sequence[ChallengeCase],
    executor: ChallengeExecutor,
    code_revision: str,
    refusal: RefusalCriteria = DEFAULT_REFUSAL,
) -> ChallengeReport:
    """Execute, or replay, every challenge case and update the challenged rules' status.

    Args:
        project: Project receiving the case results and the report.
        book: The rule book the workloads were generated from.
        rulebook: Its saved artifact.
        cases: The cases, with distinct keys.
        executor: The bound world.
        code_revision: Producing source identity.
        refusal: How a refused action is recognized.

    Returns:
        The report, also saved as a ``rule-challenges`` artifact and replayed from it.

    Raises:
        ValueError: Case keys repeat, or a case challenges a rule the book lacks.
        FileLockTimeout: Another process is running the same challenges.
    """
    keys = [case.key for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("challenge cases need distinct keys; give each base state its own scope")
    known = {rule.rule_id for rule in book.rules}
    strange = sorted({case.workload.rule_id for case in cases} - known)
    if strange:
        raise ValueError(f"challenge cases name rules the book lacks: {strange}")
    identity = stable_id(
        "rule-challenges",
        {
            "protocol": CHALLENGE_PROTOCOL,
            "rulebook": rulebook.model_dump(mode="json"),
            "book": sha256_json(book),
            "cases": sha256_json([case.model_dump(mode="json") for case in cases]),
            "executor": executor.identity,
            "refusal": refusal.model_dump(mode="json"),
            "code": code_revision,
        },
    )
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return ChallengeReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(rulebook,),
        )
        runs: list[ChallengeRun] = []
        for index, case in enumerate(cases):
            case_id = f"{identity}-case-{index}"
            if store.exists(case_id):
                runs.append(ChallengeRun.model_validate_json(store.read_bytes(case_id, "run.json")))
                continue
            run = _execute(case, executor, refusal)
            store.write_json(
                artifact_id=case_id,
                artifact_type=CHALLENGE_CASE_TYPE,
                envelope=envelope,
                files={"run.json": run},
            )
            runs.append(run)
        report = _report(identity, book, rulebook, executor.identity, tuple(runs))
        store.write_json(
            artifact_id=result_id,
            artifact_type=CHALLENGES_TYPE,
            envelope=envelope,
            files={"report.json": report},
        )
        return report


def challenge_world(
    project: ProjectStore,
    *,
    book: RuleBook,
    rulebook: ArtifactInput,
    cases: Sequence[ChallengeCase],
    definition: GroundedRuntimeDefinition | None,
    programs: ProgramChallenges,
    code_revision: str,
) -> ChallengeReport:
    """Execute challenges in a served world when one is bound, else through its programs.

    Args:
        project: Project receiving the results.
        book: The challenged rule book.
        rulebook: Its saved artifact.
        cases: The cases.
        definition: The served runtime definition, programs included, or None.
        programs: The program executor, used only without a definition.
        code_revision: Producing source identity.

    Returns:
        The saved, replayable report.
    """
    with ExitStack() as owned:
        executor: ChallengeExecutor = programs
        if definition is not None:
            executor = owned.enter_context(
                RuntimeChallenges(project, definition, refusal=programs.refusal)
            )
        return run_challenges(
            project,
            book=book,
            rulebook=rulebook,
            cases=cases,
            executor=executor,
            code_revision=code_revision,
            refusal=programs.refusal,
        )


def program_transition(
    program: TransitionProgram,
    policy: ToolEffectPolicy,
    sandbox: ProgramSandbox,
    state: JsonObject,
    arguments: JsonObject,
    refusal: RefusalCriteria = DEFAULT_REFUSAL,
) -> tuple[ProgramOutcome, Transition | None]:
    """Execute one call through a tool's program and return its transition over the full state.

    Args:
        program: The tool's program.
        policy: The tool's effect policy; the program sees only its domains.
        sandbox: Bounded executor.
        state: Complete simulated state before the call.
        arguments: Call arguments.
        refusal: How a refusal is recognized; an error reply it does not recognize is recorded
            as ``{"error": reply}``.

    Returns:
        The outcome and, for a reply or error reply, the transition.
    """
    visible = {name: deepcopy(state[name]) for name in sorted(policy.domains()) if name in state}
    outcome = sandbox.run(program, visible, arguments, {})
    if outcome.status not in ("reply", "error_reply"):
        return outcome, None
    after = deepcopy(state)
    for name in policy.writes:
        if outcome.state is not None and name in outcome.state:
            after[name] = deepcopy(outcome.state[name])
    transition = _transition(state, after, policy.name, arguments, outcome.reply)
    if outcome.status == "error_reply" and is_refusal(transition, refusal) is not True:
        transition = transition.model_copy(update={"observation": {"error": outcome.reply}})
    return outcome, transition


class ProgramChallenges:
    """Executes workloads through validated transition programs only.

    A tool without a program is left unexecuted unless an observed transition is supplied for
    the case; a program that fails or declares the input outside its domain fails the case.
    """

    def __init__(
        self,
        programs: Sequence[TransitionProgram],
        policies: Sequence[ToolEffectPolicy],
        sandbox: ProgramSandbox,
        *,
        observed: Mapping[str, Transition] | None = None,
        refusal: RefusalCriteria = DEFAULT_REFUSAL,
    ) -> None:
        """Bind programs, effect policies, the executor and supplied observations.

        Args:
            programs: Validated programs, at most one per tool.
            policies: Effect policy of every tool.
            sandbox: Bounded executor.
            observed: Observed transitions by case key for tools without a program.
            refusal: How a refusal is recognized.
        """
        self.programs = {program.tool: program for program in programs}
        self.policies = {policy.name: policy for policy in policies}
        self.sandbox, self.refusal = sandbox, refusal
        self.observed = dict(observed or {})

    @property
    def identity(self) -> Sha256:
        """Bind the programs, policies, sandbox limits and supplied observations."""
        return sha256_json(
            {
                "kind": "programs",
                "programs": sorted(program.sha256 for program in self.programs.values()),
                "policies": [
                    self.policies[name].model_dump(mode="json") for name in sorted(self.policies)
                ],
                "limits": self.sandbox.limits.model_dump(mode="json"),
                "observed": {
                    k: v.model_dump(mode="json") for k, v in sorted(self.observed.items())
                },
            }
        )

    def execute(self, case: ChallengeCase) -> Transition | str | None:
        """Run the case's tool program, or fall back to a supplied observation."""
        tool = str(case.workload.action.get("name"))
        program, policy = self.programs.get(tool), self.policies.get(tool)
        if program is None or policy is None:
            return self.observed.get(case.key)
        outcome, transition = program_transition(
            program, policy, self.sandbox, case.workload.state, _arguments(case), self.refusal
        )
        return transition or f"the program {outcome.status}: {outcome.detail}"


class RuntimeChallenges:
    """Executes workloads in a bound grounded runtime, restored at each workload's state.

    The runtime opens on the first case, so a replayed run never opens it. Use as a context
    manager so the runtime and its program worker are released.
    """

    def __init__(
        self,
        project: ProjectStore,
        definition: GroundedRuntimeDefinition,
        *,
        refusal: RefusalCriteria = DEFAULT_REFUSAL,
    ) -> None:
        """Bind the project and the runtime definition without opening it.

        Args:
            project: Project the runtime belongs to.
            definition: The bound world, with its programs when it has any.
            refusal: How a refusal is recognized.
        """
        self.project, self.definition, self.refusal = project, definition, refusal
        self._owned = ExitStack()
        self._runtime: ContinuationRuntime | None = None
        self._template: ExecutableCheckpoint | None = None

    @property
    def identity(self) -> Sha256:
        """Bind the exact runtime definition."""
        return sha256_json(
            {"kind": "runtime", "definition": self.definition.model_dump(mode="json")}
        )

    def __enter__(self) -> Self:
        """Return this executor; the runtime opens lazily."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the runtime on every exit path."""
        del exception_type, exception, traceback
        self._owned.close()

    def execute(self, case: ChallengeCase) -> Transition | str | None:
        """Restore the world at the workload's state, run its action and observe the result."""
        world, template = self._open()
        workload = case.workload
        arguments = _arguments(case)
        action = ToolCall(
            name=str(workload.action.get("name")),
            call_id="challenge-" + workload.workload_id,
            arguments=arguments,
        )
        try:
            start = with_simulated_image(template, deepcopy(workload.state), [])
            with world.restore(self.definition.task, start) as session:
                observation = session.execute(action)
                after, _ = simulated_image(session.checkpoint())
        except INFRASTRUCTURE_STOPS:
            raise
        except Exception as error:  # noqa: BLE001 - a world failure is a recorded case result.
            return f"{type(error).__name__}: {error}"[:2000]
        reply: JsonValue = observation.content
        try:
            reply = json.loads(observation.content)
        except json.JSONDecodeError:
            pass
        transition = _transition(workload.state, after, action.name, arguments, reply)
        if observation.is_error and is_refusal(transition, self.refusal) is not True:
            transition = transition.model_copy(update={"observation": {"error": reply}})
        return transition

    def _open(self) -> tuple[ExecutableWorldRuntime, ExecutableCheckpoint]:
        """Open the runtime and capture a fresh checkpoint whose image each case rewrites."""
        if self._runtime is None or self._template is None:
            runtime = self._owned.enter_context(
                continuation_runtime(self.project, self.definition.model_dump(mode="json"))
            )
            if not isinstance(runtime.world, ExecutableWorldRuntime):
                raise ValueError("challenges execute only in the executable grounded world")
            with runtime.world.open(self.definition.task) as session:
                self._template = session.checkpoint()
            self._runtime = runtime
        world = self._runtime.world
        assert isinstance(world, ExecutableWorldRuntime)
        return world, self._template


def _arguments(case: ChallengeCase) -> JsonObject:
    """Return the workload action's arguments object."""
    arguments = case.workload.action.get("arguments")
    return deepcopy(arguments) if isinstance(arguments, dict) else {}


def _transition(
    before: JsonObject, after: JsonObject, tool: str, arguments: JsonObject, reply: JsonValue
) -> Transition:
    """Build one transition record."""
    return Transition(
        before=deepcopy(before),
        after=deepcopy(after),
        action={"name": tool, "arguments": deepcopy(arguments)},
        observation=reply,
    )


def _execute(
    case: ChallengeCase, executor: ChallengeExecutor, refusal: RefusalCriteria
) -> ChallengeRun:
    """Execute one case and triage what the world did."""
    result = executor.execute(case)
    if result is None:
        return ChallengeRun(
            case=case, status="unexecuted", detail="the executor cannot run this tool"
        )
    if isinstance(result, str):
        return ChallengeRun(case=case, status="failed", detail=result)
    workload = case.workload
    (found,) = triage((workload,), {workload.workload_id: result}, refusal)
    return ChallengeRun(case=case, status="executed", transition=result, triage=found)


def _report(
    identity: str,
    book: RuleBook,
    rulebook: ArtifactInput,
    executor: Sha256,
    runs: tuple[ChallengeRun, ...],
) -> ChallengeReport:
    """Summarize every case per rule and update the challenged rules' status."""
    rows: list[RuleChallenge] = []
    rules = []
    for rule in book.rules:
        mine = [run for run in runs if run.case.workload.rule_id == rule.rule_id]
        if not mine:
            rules.append(rule)
            continue
        verdicts: dict[str, list[str]] = {}
        for run in mine:
            verdict = "unexecuted" if run.triage is None else run.triage.verdict
            verdicts.setdefault(verdict, []).append(run.case.key)
        consistent = verdicts.get("consistent_with_rule", [])
        contradicting = verdicts.get("contradicts_rule", [])
        witnessed = any(witness.supports for witness in rule.witnesses)
        status: RuleStatus = rule.status
        if contradicting and not witnessed:
            status = "contradicted"
        elif consistent and not contradicting:
            status = "supported"
        rows.append(
            RuleChallenge(
                rule_id=rule.rule_id,
                prior=rule.status,
                status=status,
                consistent=tuple(consistent),
                contradicting=tuple(contradicting),
                undetermined=tuple(verdicts.get("undetermined", [])),
                unexecuted=tuple(verdicts.get("unexecuted", [])),
                witnessed=witnessed,
            )
        )
        rules.append(
            rule.model_copy(
                update={
                    "status": status,
                    "evidence": (
                        *rule.evidence,
                        *(f"challenge:{identity}:{key}" for key in consistent),
                    ),
                    "counter_evidence": (
                        *rule.counter_evidence,
                        *(f"challenge:{identity}:{key}" for key in contradicting),
                    ),
                }
            )
        )
    return ChallengeReport(
        identity=identity,
        rulebook=rulebook,
        executor_sha256=executor,
        runs=runs,
        rules=tuple(rows),
        diagnostic=tuple(row.rule_id for row in rows if row.contradicting and not row.witnessed),
        world_defects=tuple(row.rule_id for row in rows if row.contradicting and row.witnessed),
        book=book.model_copy(update={"rules": tuple(rules)}),
    )

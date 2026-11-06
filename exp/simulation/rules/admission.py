"""Rule admission by decision tags applied to every trajectory of a task family.

A learned rule is a claim about the whole family, so one author's reading of the trajectories it
was drafted from is not enough to admit it. The rule is compiled into yes/no questions of three
roles, each asked about a whole recorded trajectory:

- ``applies``: does this trajectory exercise the situation the rule is about?
- ``consistent``: is what the trajectory observed consistent with the rule?
- ``contradicts``: does the trajectory observe something that contradicts the rule?

The decision model answers every question about every trajectory of the family, giving one
probability per (trajectory, question). This module reads that matrix and decides, without any
model call:

- An answer is confident only outside the uncertain band (``no_threshold`` < p <
  ``yes_threshold``). A role reads ``yes`` when every one of its questions is a confident yes,
  ``no`` when every one is a confident no, and ``uncertain`` otherwise, so two prompt variants
  that disagree leave the role uncertain.
- A trajectory supports the rule when it applies, is consistent, and does not contradict. It
  contradicts the rule when it applies, contradicts, and is not consistent. Anything else that
  applies is uncertain, and uncertain trajectories are kept in the record but never counted.
- A rule is admitted when supporting trajectories come from at least ``minimum_independent``
  distinct lineages and no trajectory confidently contradicts it. Lineage is the leakage lineage
  of the exposure plan: near-duplicate trajectories share one, so they count once.

Admitted rules also become the family's action-guard and invariant table
(:class:`GuardTable`), one machine-readable source for the Python runtime guard (through
:func:`guard_rulebook` and the existing checker) and for a later model-checking specification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.simulation.rules.memory_set import RuleFile
from exp.simulation.rules.model import Predicate, Rule, RuleBook, RuleKind, RuleStatus

ADMISSION_PROTOCOL = "rule-tag-admission-v1"
"""Version of the reading and admission rule; change it whenever either changes."""

GUARD_TABLE_FORMAT = "rule-guard-table-v1"
"""Version of the action-guard and invariant table layout."""

QuestionRole = Literal["applies", "consistent", "contradicts"]
"""What one compiled question asks about a trajectory."""

ROLES: tuple[QuestionRole, ...] = ("applies", "consistent", "contradicts")

RoleAnswer = Literal["yes", "no", "uncertain"]
"""A role's reading over its questions for one trajectory."""

Reading = Literal["supports", "contradicts", "not_applicable", "uncertain"]
"""What one trajectory says about one rule."""


class RuleQuestion(ContractModel):
    """One compiled yes/no question about a whole trajectory.

    Attributes:
        question_id: Stable identity, derived from the rule, role and text.
        role: ``applies``, ``consistent`` or ``contradicts``.
        text: The exact instruction the decision model receives.
        variant: Compiler prompt variant that wrote it.
    """

    question_id: str = Field(min_length=1, max_length=256)
    role: QuestionRole
    text: str = Field(min_length=1, max_length=1200)
    variant: str = Field(default="", max_length=64)


def question_id_for(rule_id: str, role: QuestionRole, text: str) -> str:
    """Stable identity of one compiled question."""
    return f"{rule_id}.{role}.{sha256_json(' '.join(text.lower().split()))[:12]}"


class AdmissionPolicy(ContractModel):
    """Frozen thresholds of the admission rule.

    Attributes:
        yes_threshold: Probability at or above which an answer is a confident yes.
        no_threshold: Probability at or below which an answer is a confident no.
        minimum_independent: Distinct supporting lineages a rule needs to be admitted.
    """

    yes_threshold: float = Field(default=0.7, gt=0.5, le=1.0)
    no_threshold: float = Field(default=0.3, ge=0.0, lt=0.5)
    minimum_independent: int = Field(default=2, ge=2, le=64, strict=True)

    @property
    def identity(self) -> Sha256:
        """Bind the thresholds and the admission protocol."""
        return sha256_json({"protocol": ADMISSION_PROTOCOL, "policy": self.model_dump()})


class TrajectoryTags(ContractModel):
    """One trajectory's answers to every compiled question of one rule.

    Attributes:
        trace_id: The trajectory.
        lineage_id: Its leakage lineage; near-duplicates share one.
        values: One probability per question, in the matrix's column order.
    """

    trace_id: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    values: tuple[float, ...]


class RuleTagMatrix(ContractModel):
    """The corpus-wide tag matrix of one rule: every trajectory times every question.

    Attributes:
        rule_id: The rule.
        question_ids: Column order.
        rows: One row per family trajectory, in corpus order.
    """

    rule_id: str
    question_ids: tuple[str, ...]
    rows: tuple[TrajectoryTags, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> RuleTagMatrix:
        """Every row answers every column with a probability."""
        for row in self.rows:
            if len(row.values) != len(self.question_ids):
                raise ValueError(f"row {row.trace_id} does not answer every question")
            if any(not 0.0 <= value <= 1.0 for value in row.values):
                raise ValueError(f"row {row.trace_id} holds a value outside [0, 1]")
        return self


class TrajectoryVerdict(ContractModel):
    """What one trajectory says about one rule, with each role's reading.

    Attributes:
        trace_id: The trajectory.
        lineage_id: Its lineage.
        applies: Reading of the ``applies`` questions.
        consistent: Reading of the ``consistent`` questions.
        contradicts: Reading of the ``contradicts`` questions.
        reading: The combined reading.
    """

    trace_id: str
    lineage_id: str
    applies: RoleAnswer
    consistent: RoleAnswer
    contradicts: RoleAnswer
    reading: Reading


class AdmissionVerdict(ContractModel):
    """The admission decision of one rule over its corpus-wide tag matrix.

    Attributes:
        rule_id: The rule.
        admitted: Whether the rule is admitted.
        status: ``supported`` when admitted, ``contradicted`` when any trajectory confidently
            contradicts it, ``unknown`` otherwise.
        supporting: Supporting trajectories, in corpus order.
        supporting_lineages: Distinct lineages of the supporting trajectories, in first-seen order.
        duplicate_support: Supporting trajectories whose lineage was already counted.
        contradicting: Trajectories that confidently contradict the rule.
        uncertain: Trajectories whose reading is uncertain; kept, never counted.
        not_applicable: Trajectories that do not exercise the rule's situation.
        trajectories: Every trajectory's verdict, in corpus order.
        reason: One plain sentence explaining the decision.
    """

    rule_id: str
    admitted: bool
    status: RuleStatus
    supporting: tuple[str, ...] = ()
    supporting_lineages: tuple[str, ...] = ()
    duplicate_support: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    uncertain: tuple[str, ...] = ()
    not_applicable: tuple[str, ...] = ()
    trajectories: tuple[TrajectoryVerdict, ...] = ()
    reason: str = Field(min_length=1, max_length=2000)

    def summary(self) -> str:
        """The verdict as one paragraph for a rule file's status rationale."""
        independent = len(self.supporting_lineages)
        parts = [
            f"Tag admission: {'admitted' if self.admitted else 'not admitted'} "
            f"({self.status}). {self.reason}",
            f"Supporting trajectories: {_list(self.supporting)} from {independent} independent "
            f"lineage{'s' if independent != 1 else ''}.",
        ]
        if self.duplicate_support:
            parts.append(f"Same-lineage support not counted: {_list(self.duplicate_support)}.")
        parts.append(f"Contradicting trajectories: {_list(self.contradicting)}.")
        if self.uncertain:
            parts.append(f"Uncertain, kept and not counted: {_list(self.uncertain)}.")
        return " ".join(parts)


def read_role(values: Sequence[float], policy: AdmissionPolicy) -> RoleAnswer:
    """Read one role's answers: yes or no only when every answer is confident that way.

    A role with no question is uncertain, since nothing was asked.
    """
    if not values:
        return "uncertain"
    if all(value >= policy.yes_threshold for value in values):
        return "yes"
    if all(value <= policy.no_threshold for value in values):
        return "no"
    return "uncertain"


def read_trajectory(
    questions: Sequence[RuleQuestion], row: TrajectoryTags, policy: AdmissionPolicy
) -> TrajectoryVerdict:
    """Combine one trajectory's role readings into what it says about the rule.

    Args:
        questions: The rule's questions, in the matrix's column order.
        row: The trajectory's answers.
        policy: Thresholds.

    Returns:
        The trajectory's verdict.
    """
    by_role: dict[QuestionRole, list[float]] = {role: [] for role in ROLES}
    for question, value in zip(questions, row.values, strict=True):
        by_role[question.role].append(value)
    applies = read_role(by_role["applies"], policy)
    consistent = read_role(by_role["consistent"], policy)
    contradicts = read_role(by_role["contradicts"], policy)
    reading: Reading
    if applies == "no":
        reading = "not_applicable"
    elif applies == "uncertain":
        reading = "uncertain"
    elif contradicts == "yes" and consistent == "no":
        reading = "contradicts"
    elif consistent == "yes" and contradicts == "no":
        reading = "supports"
    else:
        reading = "uncertain"
    return TrajectoryVerdict(
        trace_id=row.trace_id,
        lineage_id=row.lineage_id,
        applies=applies,
        consistent=consistent,
        contradicts=contradicts,
        reading=reading,
    )


def admit(
    questions: Sequence[RuleQuestion], matrix: RuleTagMatrix, policy: AdmissionPolicy
) -> AdmissionVerdict:
    """Decide one rule's admission from its corpus-wide tag matrix.

    Args:
        questions: The rule's compiled questions, in the matrix's column order.
        matrix: Every trajectory's answers.
        policy: Thresholds and the independent-support minimum.

    Returns:
        The verdict. A rule without questions, or without any role, is never admitted.

    Raises:
        ValueError: The questions do not match the matrix columns.
    """
    if tuple(q.question_id for q in questions) != matrix.question_ids:
        raise ValueError(f"questions of rule {matrix.rule_id} do not match its matrix columns")
    missing = [role for role in ROLES if not any(q.role == role for q in questions)]
    if missing:
        return AdmissionVerdict(
            rule_id=matrix.rule_id,
            admitted=False,
            status="unknown",
            reason=f"The rule has no compiled {', '.join(missing)} question, so it was not "
            "tagged and cannot be admitted.",
        )
    verdicts = tuple(read_trajectory(questions, row, policy) for row in matrix.rows)
    supporting = [v for v in verdicts if v.reading == "supports"]
    lineages: list[str] = []
    duplicates: list[str] = []
    for verdict in supporting:
        if verdict.lineage_id in lineages:
            duplicates.append(verdict.trace_id)
        else:
            lineages.append(verdict.lineage_id)
    contradicting = tuple(v.trace_id for v in verdicts if v.reading == "contradicts")
    enough = len(lineages) >= policy.minimum_independent
    admitted = enough and not contradicting
    status: RuleStatus = "contradicted" if contradicting else "supported" if admitted else "unknown"
    if contradicting:
        reason = (
            f"{len(contradicting)} trajectory confidently contradicts the rule, which blocks "
            "admission whatever the support."
            if len(contradicting) == 1
            else f"{len(contradicting)} trajectories confidently contradict the rule, which "
            "blocks admission whatever the support."
        )
    elif admitted:
        reason = (
            f"{len(lineages)} independent lineages agree with the rule (at least "
            f"{policy.minimum_independent} required) and no trajectory contradicts it."
        )
    else:
        reason = (
            f"Only {len(lineages)} independent lineage{'s' if len(lineages) != 1 else ''} agree "
            f"with the rule; at least {policy.minimum_independent} are required."
        )
    return AdmissionVerdict(
        rule_id=matrix.rule_id,
        admitted=admitted,
        status=status,
        supporting=tuple(v.trace_id for v in supporting),
        supporting_lineages=tuple(lineages),
        duplicate_support=tuple(duplicates),
        contradicting=contradicting,
        uncertain=tuple(v.trace_id for v in verdicts if v.reading == "uncertain"),
        not_applicable=tuple(v.trace_id for v in verdicts if v.reading == "not_applicable"),
        trajectories=verdicts,
        reason=reason,
    )


def lineage_of(trace_id: str, lineages: Mapping[str, str]) -> str:
    """A trajectory's lineage, or its own identity when the plan names none."""
    return lineages.get(trace_id) or f"trace:{trace_id}"


class GuardEntry(ContractModel):
    """One admitted rule as a guard, postcondition or invariant.

    Attributes:
        rule_id: The rule.
        tool: The action it guards, or None for every action.
        kind: Its kind.
        statement: The truism as admitted.
        predicate: Its checkable form, when one exists; the runtime checker enforces it.
        questions: Its compiled questions, so a semantic guard can ask the decision model.
        supporting: Supporting trajectories.
        supporting_lineages: Distinct supporting lineages.
        statement_sha256: Digest of the statement the questions were compiled from.
    """

    rule_id: str
    tool: str | None
    kind: RuleKind
    statement: str
    predicate: Predicate | None = None
    questions: tuple[RuleQuestion, ...] = ()
    supporting: tuple[str, ...] = ()
    supporting_lineages: tuple[str, ...] = ()
    statement_sha256: Sha256

    @property
    def checkable(self) -> bool:
        """Whether the runtime checker can evaluate it without a model."""
        return self.predicate is not None


class WithdrawnGuard(ContractModel):
    """An admitted rule left out of the table, with why.

    Attributes:
        rule_id: The rule.
        reason: Removed, demoted, or edited after admission.
    """

    rule_id: str
    reason: str


class GuardTable(ContractModel):
    """One family's action guards, postconditions and invariants, from admitted rules only.

    Attributes:
        format: Layout version.
        family: Task family the table belongs to.
        tools: Every tool of the family, so an unguarded action is visible.
        policy: Identity of the admission policy the rules were admitted under.
        action_guards: Preconditions: an action whose guard is false must be refused.
        postconditions: Effects and observations that must hold after a successful action.
        invariants: Conditions every resulting state must satisfy.
        withdrawn: Admitted rules governance removed, demoted or rewrote.
    """

    format: Literal["rule-guard-table-v1"] = GUARD_TABLE_FORMAT
    family: str = Field(min_length=1)
    tools: tuple[str, ...] = ()
    policy: Sha256
    action_guards: tuple[GuardEntry, ...] = ()
    postconditions: tuple[GuardEntry, ...] = ()
    invariants: tuple[GuardEntry, ...] = ()
    withdrawn: tuple[WithdrawnGuard, ...] = ()

    def entries(self) -> tuple[GuardEntry, ...]:
        """Every entry, guards first."""
        return (*self.action_guards, *self.postconditions, *self.invariants)


def guard_table(
    family: str,
    tools: Sequence[str],
    rules: Sequence[RuleFile],
    verdicts: Sequence[AdmissionVerdict],
    compiled: Mapping[str, tuple[RuleQuestion, ...]],
    admitted_statements: Mapping[str, str],
    policy: AdmissionPolicy,
) -> GuardTable:
    """Build the family's guard table from its final rules and their admission verdicts.

    A rule enters only when it was admitted, is still in the set with status ``supported``, and
    its truism is the one its questions were compiled from. Every other admitted rule is listed
    as withdrawn, so nothing silently disappears.

    Args:
        family: Task family identity.
        tools: Every tool of the family.
        rules: The final rule files.
        verdicts: Admission verdicts.
        compiled: Compiled questions by rule.
        admitted_statements: Truism each admitted rule was compiled from, by rule.
        policy: The admission policy.

    Returns:
        The table, entries in rule order.
    """
    by_id = {rule.rule_id: rule for rule in rules}
    sections: dict[str, list[GuardEntry]] = {"guard": [], "post": [], "invariant": []}
    withdrawn: list[WithdrawnGuard] = []
    for verdict in verdicts:
        if not verdict.admitted:
            continue
        rule = by_id.get(verdict.rule_id)
        statement = admitted_statements.get(verdict.rule_id, "")
        if rule is None:
            withdrawn.append(WithdrawnGuard(rule_id=verdict.rule_id, reason="removed"))
            continue
        if rule.status != "supported":
            withdrawn.append(
                WithdrawnGuard(rule_id=rule.rule_id, reason=f"governance set it {rule.status}")
            )
            continue
        if rule.truism != statement:
            withdrawn.append(
                WithdrawnGuard(rule_id=rule.rule_id, reason="truism edited after admission")
            )
            continue
        entry = GuardEntry(
            rule_id=rule.rule_id,
            tool=rule.tool,
            kind=rule.kind,
            statement=rule.truism,
            predicate=rule.predicate,
            questions=compiled.get(rule.rule_id, ()),
            supporting=verdict.supporting,
            supporting_lineages=verdict.supporting_lineages,
            statement_sha256=sha256_json(rule.truism),
        )
        key = {"precondition": "guard", "invariant": "invariant"}.get(rule.kind, "post")
        sections[key].append(entry)
    return GuardTable(
        family=family,
        tools=tuple(sorted(tools)),
        policy=policy.identity,
        action_guards=tuple(sections["guard"]),
        postconditions=tuple(sections["post"]),
        invariants=tuple(sections["invariant"]),
        withdrawn=tuple(withdrawn),
    )


def guard_rulebook(table: GuardTable) -> RuleBook:
    """The table's checkable entries as a rule book the runtime checker enforces.

    Entries without a predicate need the decision model and are left to a semantic guard.
    """
    return RuleBook(
        contracts={},
        rules=tuple(
            Rule(
                rule_id=entry.rule_id,
                tool=entry.tool,
                kind=entry.kind,
                statement=entry.statement,
                predicate=entry.predicate,
                status="supported",
                support=len(entry.supporting),
                contradiction=0,
            )
            for entry in table.entries()
            if entry.checkable
        ),
    )


def _list(items: Sequence[str]) -> str:
    """Items in backticks, or none."""
    return ", ".join(f"`{item}`" for item in items) or "none"

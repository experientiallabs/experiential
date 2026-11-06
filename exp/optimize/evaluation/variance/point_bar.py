"""The point qualification bar: meta-rules decide what counts as a point, not a fixed count.

A :class:`QualificationBar` is a versioned set of yes/no meta-rules about one actor step, such as
"the step is a consequential choice among two or more executable alternatives". The bar is a
:class:`~exp.simulation.mining.questions.QuestionBank` (:meth:`QualificationBar.bank`), so the
corpus tag author can author it like any other bank (:meth:`QualificationBar.from_bank` adopts an
authored bank and names its required rules), and the decision model (Clef) answers every rule
about every step of the whole trajectory through the unchanged tagging seam, over the step view.
There is no step horizon and no per-selector point cap here.

:func:`qualify_points` reads those answers per point (a step with boundary ``k`` is decided at
point ``k``): a point qualifies when, on one of its steps, every required rule is confidently
true; a confidently false required rule rejects the step; an answer inside the ambiguity band or a
missing row leaves it uncertain, and uncertain stays unqualified. A measured blast radius of zero
rejects a point (outcomes do not differ from there).

Selectors receive the rules (:func:`bar_instruction`) and justify every pick against each required
rule with cited events (:class:`PickJustification`); :func:`check_picks` refuses a pick whose
point did not clear the bar or whose justification misses a rule or cites unknown events.

:func:`schedule_points` continues qualified points in blast-radius order while each one's
reserved cost fits the remaining shared ledger, and defers the rest; a deferred point carries
into the next phase. Every point moves through :data:`~.reward_states.POINT_MACHINE`
(candidate, tagged, qualified / rejected / uncertain, continued / deferred); illegal moves are
refused and recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.state_machine import Lifecycle, Refusal, Transition
from exp.optimize.evaluation.variance.blast_radius import BlastRadius
from exp.optimize.evaluation.variance.reward_states import POINT_MACHINE, PointState
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tag_selection import DEFAULT_AMBIGUITY_BAND, DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix, TraceTaggingError
from exp.simulation.mining.traceml import StepViewSpec, TraceTrajectory, step_units

BAR_FAMILY = "bar.meta"
"""Question family of every meta-rule."""


class MetaRule(ContractModel):
    """One yes/no meta-rule about one actor step.

    Attributes:
        rule_id: Stable identity, the bank column.
        text: The proposition the decision model answers about the focal step.
        required: Whether a point needs it confidently true to qualify.
    """

    rule_id: str = Field(pattern=r"^bar\.[a-z][a-z0-9_]{0,63}$")
    text: str = Field(min_length=1, max_length=1000)
    required: bool = True


class QualificationBar(ContractModel):
    """A versioned bar of meta-rules.

    Attributes:
        bar_id: Bank identity.
        version: Revision; a changed rule is a new version.
        rules: The meta-rules in column order.
    """

    bar_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    rules: tuple[MetaRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> QualificationBar:
        """Require distinct rules and at least one required rule."""
        ids = [rule.rule_id for rule in self.rules]
        if len(set(ids)) != len(ids) or not any(rule.required for rule in self.rules):
            raise ValueError("bar rules must be distinct and include a required rule")
        return self

    def bank(self) -> QuestionBank:
        """The bar as a question bank for the decision model."""
        return QuestionBank(
            bank_id=self.bar_id,
            version=self.version,
            questions=tuple(
                Question(question_id=rule.rule_id, family=BAR_FAMILY, text=rule.text)
                for rule in self.rules
            ),
        )

    @classmethod
    def from_bank(cls, bank: QuestionBank, required: Sequence[str]) -> QualificationBar:
        """Adopt an authored bank (for example from the corpus tag author) as a bar."""
        unknown = sorted(set(required) - set(bank.question_ids()))
        if unknown:
            raise ValueError(f"required rules {unknown} are not questions of {bank.bank_id}")
        return cls(
            bar_id=bank.bank_id,
            version=bank.version,
            rules=tuple(
                MetaRule(rule_id=q.question_id, text=q.text, required=q.question_id in required)
                for q in bank.questions
            ),
        )


DEFAULT_BAR = QualificationBar(
    bar_id="point-bar-v1",
    version=1,
    rules=(
        MetaRule(
            rule_id="bar.consequential_choice",
            text="Is the focal step a consequential choice among two or more alternatives the "
            "agent could actually have executed at that moment?",
        ),
        MetaRule(
            rule_id="bar.downstream_failure",
            text="Does a judged error or failure follow later in the trajectory that is causally "
            "downstream of the choice made at the focal step?",
        ),
        MetaRule(
            rule_id="bar.observable_facts",
            text="Were the facts that decide the right choice observable to the agent at the "
            "focal step (stated, returned by a tool, or readable in the workspace)?",
        ),
        MetaRule(
            rule_id="bar.not_forced",
            text="Is the choice at the focal step left open by the instructions, rather than "
            "dictated by them?",
        ),
        MetaRule(
            rule_id="bar.not_trivial_retry",
            text="Is the focal step something other than a trivial retry or repetition of an "
            "earlier action with the same arguments?",
        ),
        MetaRule(
            rule_id="bar.outcomes_diverge",
            text="Would different agents or attempts plausibly reach different outcomes from "
            "this step?",
            required=False,
        ),
    ),
)
"""The default bar; an authored bar replaces it by version."""


class StepVerdict(ContractModel):
    """The bar on one step.

    Attributes:
        locator: Step citation.
        index: Step position.
        status: ``qualified``, ``rejected`` or ``uncertain``.
        values: Probability per rule, empty without a row.
        failed: Required rules confidently false.
        unsure: Required rules inside the band.
    """

    locator: str
    index: int = Field(ge=0)
    status: str
    values: dict[str, float] = Field(default_factory=dict)
    failed: tuple[str, ...] = ()
    unsure: tuple[str, ...] = ()


class PointQualification(ContractModel):
    """The bar on one point.

    Attributes:
        point: Boundary index.
        state: Final qualification state (qualified, rejected or uncertain).
        steps: The verdict of every step decided at the point.
        reasons: Why.
    """

    point: int = Field(ge=0)
    state: PointState
    steps: tuple[StepVerdict, ...]
    reasons: tuple[str, ...] = ()


class BarRecord(ContractModel):
    """One trajectory's qualification, with its persisted lifecycle."""

    bar_id: str
    bar_version: int
    trace_id: str
    points: tuple[PointQualification, ...]
    transitions: tuple[Transition, ...] = ()
    refusals: tuple[Refusal, ...] = ()

    @property
    def qualified(self) -> tuple[int, ...]:
        """Qualified points in trajectory order."""
        return tuple(p.point for p in self.points if p.state == PointState.QUALIFIED)


def _step(
    values: dict[str, float] | None,
    bar: QualificationBar,
    threshold: float,
    band: tuple[float, float],
    locator: str,
    index: int,
) -> StepVerdict:
    """The bar on one step's answers."""
    if values is None:
        return StepVerdict(locator=locator, index=index, status="uncertain", unsure=("no row",))
    low, high = band
    required = [rule.rule_id for rule in bar.rules if rule.required]
    failed = tuple(r for r in required if values[r] < low)
    unsure = tuple(r for r in required if low <= values[r] <= high)
    true = all(values[r] > max(threshold, high) for r in required)
    status = "rejected" if failed else "qualified" if true and not unsure else "uncertain"
    return StepVerdict(
        locator=locator, index=index, status=status, values=values, failed=failed, unsure=unsure
    )


def qualify_points(
    trajectory: TraceTrajectory,
    matrix: TagMatrix | None,
    bar: QualificationBar = DEFAULT_BAR,
    *,
    view: StepViewSpec | None = None,
    threshold: float = DEFAULT_FIRE_THRESHOLD,
    ambiguity_band: tuple[float, float] = DEFAULT_AMBIGUITY_BAND,
    blast: Mapping[int, BlastRadius] | None = None,
) -> BarRecord:
    """Apply the bar to every point of the whole trajectory.

    Args:
        trajectory: The attempt, every step (no horizon).
        matrix: The decision model's answers to :meth:`QualificationBar.bank` over the step view,
            or None when tagging failed (every point uncertain).
        bar: The bar.
        view: Step rendering the answers were bought over.
        threshold: Probability strictly above which a rule holds.
        ambiguity_band: Closed band inside which an answer is uncertain.
        blast: Measured blast radius by point; a scored radius of zero rejects the point.
    """
    units = step_units(trajectory, view)
    columns = {} if matrix is None else {q: i for i, q in enumerate(matrix.question_ids)}
    missing = sorted({rule.rule_id for rule in bar.rules} - set(columns))
    if matrix is not None and missing:
        raise ValueError(f"the tag matrix does not answer bar rules {missing}")
    steps: dict[int, list[StepVerdict]] = {}
    for step, unit in zip(trajectory.steps, units, strict=True):
        try:
            row = None if matrix is None else matrix.values_for(unit)
        except (TraceTaggingError, ValueError):
            row = None
        values = None if row is None else {r.rule_id: row[columns[r.rule_id]] for r in bar.rules}
        verdict = _step(values, bar, threshold, ambiguity_band, step.locator, step.index)
        steps.setdefault(step.boundary, []).append(verdict)
    states = POINT_MACHINE.lifecycle()
    points: list[PointQualification] = []
    for point, verdicts in sorted(steps.items()):
        subject = f"point-{point}"
        states.start(subject)
        tagged = any(v.values for v in verdicts)
        if tagged:
            states.move(subject, PointState.TAGGED, f"{len(verdicts)} steps answered")
        statuses = {v.status for v in verdicts}
        reasons: list[str] = []
        radius = None if blast is None else blast.get(point)
        if radius is not None and radius.score == 0.0:
            target, reasons = PointState.REJECTED, ["measured blast radius is zero"]
        elif "qualified" in statuses:
            target = PointState.QUALIFIED
            reasons.append("a step clears every required rule")
        elif statuses == {"rejected"}:
            target = PointState.REJECTED
            failed = sorted({r for v in verdicts for r in v.failed})
            reasons.append(f"required rules confidently false: {failed}")
        else:
            target = PointState.UNCERTAIN
            reasons.append("no step confidently clears the bar")
        states.move(subject, target, "; ".join(reasons))
        points.append(
            PointQualification(
                point=point,
                state=states.state(subject),
                steps=tuple(verdicts),
                reasons=tuple(reasons),
            )
        )
    return BarRecord(
        bar_id=bar.bar_id,
        bar_version=bar.version,
        trace_id=trajectory.trace_id,
        points=tuple(points),
        transitions=tuple(states.transitions),
        refusals=tuple(states.refusals),
    )


def bar_instruction(bar: QualificationBar = DEFAULT_BAR) -> str:
    """The text selectors receive: every rule, and how each pick must be justified."""
    lines = [
        f"QUALIFICATION BAR {bar.bar_id} v{bar.version}",
        "A point counts only when it clears this bar. Points may be anywhere in the trajectory;",
        "there is no fixed number to choose. For every pick, justify each REQUIRED rule as",
        "holding, citing the event ids that show it. The host refuses a pick whose point did not",
        "clear the bar or whose justification misses a required rule or cites unknown events.",
    ]
    lines.extend(
        f"- {rule.rule_id} ({'required' if rule.required else 'signal'}): {rule.text}"
        for rule in bar.rules
    )
    return "\n".join(lines)


class RuleCitation(ContractModel):
    """One rule justified for one pick."""

    rule_id: str
    holds: bool
    event_refs: tuple[str, ...] = Field(default=(), max_length=32)


class PickJustification(ContractModel):
    """A selector's pick and its justification against the bar."""

    point: int = Field(ge=0)
    rules: tuple[RuleCitation, ...] = Field(max_length=32)
    rationale: str = Field(min_length=1, max_length=3000)


class PickRefusal(ContractModel):
    """A refused pick."""

    point: int
    problems: tuple[str, ...]


def check_picks(
    picks: Sequence[PickJustification],
    record: BarRecord,
    events: Sequence[str],
    bar: QualificationBar = DEFAULT_BAR,
) -> tuple[tuple[int, ...], tuple[PickRefusal, ...]]:
    """Accept picks that clear the bar with a full cited justification; refuse the rest.

    Returns:
        Accepted points (deduplicated, in pick order) and every refusal with its problems.
    """
    if (record.bar_id, record.bar_version) != (bar.bar_id, bar.version):
        raise ValueError("the qualification record was made under another bar")
    state = {p.point: p.state for p in record.points}
    known = set(events)
    required = [rule.rule_id for rule in bar.rules if rule.required]
    accepted: list[int] = []
    refused: list[PickRefusal] = []
    for pick in picks:
        problems: list[str] = []
        if state.get(pick.point) != PointState.QUALIFIED:
            problems.append(f"point {pick.point} did not clear the bar ({state.get(pick.point)})")
        cited = {row.rule_id: row for row in pick.rules}
        for rule in required:
            row = cited.get(rule)
            if row is None or not row.holds:
                problems.append(f"{rule} is not justified as holding")
            elif not row.event_refs or set(row.event_refs) - known:
                problems.append(f"{rule} cites no events or unknown events")
        if problems:
            refused.append(PickRefusal(point=pick.point, problems=tuple(problems)))
        elif pick.point not in accepted:
            accepted.append(pick.point)
    return tuple(accepted), tuple(refused)


class PointSchedule(ContractModel):
    """One phase's continuation order under the shared ledger.

    Attributes:
        continued: Points continued this phase, in blast order.
        deferred: Qualified points that did not fit; they carry to the next phase.
        reserved: Attempts reserved for the continued points.
        remaining: Ledger left after the reservation.
        transitions: Persisted moves of this phase.
        refusals: Refused moves (for example a point that never qualified).
    """

    continued: tuple[int, ...]
    deferred: tuple[int, ...]
    reserved: int = Field(ge=0)
    remaining: int = Field(ge=0)
    transitions: tuple[Transition, ...] = ()
    refusals: tuple[Refusal, ...] = ()


def schedule_points(
    record: BarRecord,
    ranked: Sequence[BlastRadius],
    *,
    cost: Mapping[int, int],
    remaining: int,
) -> PointSchedule:
    """Continue qualified points in blast order while each fits; defer the others.

    Args:
        record: The trajectory's qualification, including points deferred in earlier phases
            (their persisted transitions resume them at ``deferred``).
        ranked: The blast-radius ranking; qualified points it does not rank follow in
            trajectory order.
        cost: Attempts each point's continuations reserve (for example M times tiers times the
            call bound).
        remaining: Attempts left on the single cumulative ledger.
    """
    if remaining < 0:
        raise ValueError("the remaining ledger cannot be negative")
    states = _resume(record)
    order = {row.point: row.rank for row in ranked if row.score is not None}
    candidates = [
        p.point
        for p in record.points
        if states.state(f"point-{p.point}") in (PointState.QUALIFIED, PointState.DEFERRED)
    ]
    candidates.sort(key=lambda point: (point not in order, order.get(point, 0), point))
    first = len(states.transitions)
    continued: list[int] = []
    deferred: list[int] = []
    left = remaining
    for point in candidates:
        subject = f"point-{point}"
        need = cost[point]
        if need <= left:
            states.move(subject, PointState.CONTINUED, f"reserves {need} of {left}")
            continued.append(point)
            left -= need
        elif states.state(subject) == PointState.QUALIFIED:
            states.move(subject, PointState.DEFERRED, f"needs {need}, {left} left")
            deferred.append(point)
        else:
            deferred.append(point)
    return PointSchedule(
        continued=tuple(continued),
        deferred=tuple(deferred),
        reserved=remaining - left,
        remaining=left,
        transitions=tuple(states.transitions[first:]),
        refusals=tuple(states.refusals),
    )


def advance(record: BarRecord, schedule: PointSchedule) -> BarRecord:
    """Persist a phase's moves on the record, so the next phase resumes from them."""
    return record.model_copy(
        update={
            "transitions": (*record.transitions, *schedule.transitions),
            "refusals": (*record.refusals, *schedule.refusals),
        }
    )


def _resume(record: BarRecord) -> Lifecycle[PointState]:
    """Rebuild every point's state from the persisted transitions, rechecking each move."""
    states = POINT_MACHINE.lifecycle()
    for point in record.points:
        states.start(f"point-{point.point}")
    for row in record.transitions:
        if row.subject in states.current:
            states.move(row.subject, PointState(row.target), row.reason)
    return states

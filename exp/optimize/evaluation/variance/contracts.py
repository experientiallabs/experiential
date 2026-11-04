"""What judges, critics and scenario authors return, and the checks every answer must pass.

Each role answers with one of these models through a forced structured call. The models carry only
data: steps, probabilities, JSON-pointer state paths and propositions. Every reference an answer
makes is checked against what the role was actually shown, and every authored scenario is checked
mechanically before it can become an environment:

- edits replace only declared variation axes, with only their permitted values;
- invariants hold the same value before and after the edits, compared as values, so replacing a
  parent object cannot slip a changed invariant through;
- no two variants share their complete (world state, instruction obligations) identity. An
  instruction-only variant may share its world state with another variant when its instruction
  value changes obligations; an equivalent rewording does not, so wording alone never makes a
  variant distinct, and a state-only decision (no instruction axis) cannot vary wording at all;
- instruction variation is its own declared axis: a goal or consent change cites what
  authorizes it and binds the criteria that apply under it, and a changed instruction cannot
  resume a prefix whose first message stated the original, so it needs a fresh briefing;
- a fresh briefing's entitled paths exist, are not edited and show values the actor had
  observed at its start. Whether the briefing text leaks the intended option, an edited value or
  unobserved knowledge is a question about meaning, judged by the model leak reviewer
  (:mod:`~exp.optimize.evaluation.variance.briefing_review`), never by word matching here;
- correct choices are option identities from a declared set, so "refund" and "issue refund"
  cannot pass as different decisions;
- every deciding fact is revealed by a witness of permitted tool calls that fits in the actor's
  remaining budget, so a scenario cannot require guessing a hidden value.

A variant also freezes the complete restorable state it starts from (:class:`FrozenState`): domain
data and files, identity and permissions, clock, pending events, scheduler and random state, event
cursor, conditioning facts, tool/session state and conversation. A briefing may describe that
state; it never replaces it, and a variant whose required parts are not held is refused.

These checks are structural. Whether the witness calls really reveal the evidence, whether the
edited state is coherent with the retained conversation and instance facts, and whether the
correct option really leads to better outcomes than the others are established by executed
witnesses and competing-action continuations from the same checkpoint, never by these gates.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256
from exp.common.models import ModelMessage

AuditCategory = Literal[
    "instruction_verifier_mismatch",
    "answer_leakage",
    "toolset_or_harness_gap",
    "model_tier_anomaly",
    "guess_required",
    "sandbox_workaround_required",
    "decision_space_too_narrow",
    "inefficiency_misgraded",
]
"""The eval-audit checklist: the four classic flaws plus four that teach harmful behavior."""

StatePart = Literal[
    "domain_state",
    "files",
    "identity_permissions",
    "clock",
    "pending_events",
    "scheduler_random",
    "event_cursor",
    "conditioning_facts",
    "tool_session",
    "conversation",
]
"""Every part of the restorable state a variant freezes."""

BASELINE_STATE: frozenset[StatePart] = frozenset(
    {
        "domain_state",
        "identity_permissions",
        "clock",
        "pending_events",
        "scheduler_random",
        "event_cursor",
        "tool_session",
        "conversation",
    }
)
"""Parts every variant requires; files and conditioning facts are required when declared."""

InitializationMode = Literal["resume", "fresh_briefing"]
"""How the actor starts a variant: the retained prefix, or an authored new task."""

MAXIMUM_SCENARIOS = 8
"""Most variants one scenario set may author."""

InstructionKind = Literal["original", "reworded", "goal_change", "consent_change"]
"""What an instruction value does to the actor's obligations."""


class EstimationPoint(ContractModel):
    """One step of a trajectory a judge wants the value estimated before.

    Attributes:
        step: Index of the trajectory step the estimate is taken just before.
        decision: The choice the actor made at that step, stated as an action.
        alternatives: Plausible different choices the actor could have made there.
        deciding_state: JSON-pointer paths of the world state that should decide the choice.
        rationale: Why the outcome is likely to hinge on this decision.
        evidence_refs: Supplied locators backing the choice of this point.
    """

    step: int = Field(ge=0)
    decision: str = Field(min_length=1, max_length=600)
    alternatives: tuple[str, ...] = Field(min_length=1, max_length=5)
    deciding_state: tuple[str, ...] = Field(max_length=12)
    rationale: str = Field(min_length=1, max_length=1200)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class JudgeSelection(ContractModel):
    """One judge attempt's estimation points in the picked trajectory.

    Attributes:
        points: Distinct steps to estimate value at, in any order.
        audit: Eval-design problems the judge saw while reading the group.
        limitations: What the judge could not determine from the evidence.
    """

    points: tuple[EstimationPoint, ...] = Field(min_length=1, max_length=8)
    audit: tuple[AuditFlag, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> JudgeSelection:
        """Require each step at most once."""
        steps = [point.step for point in self.points]
        if len(set(steps)) != len(steps):
            raise ValueError("a judge may choose each step at most once")
        return self


class AuditFlag(ContractModel):
    """One eval-design problem found while reading trajectories.

    Attributes:
        category: Which checklist item the problem falls under.
        explanation: The specific problem, tied to what was observed.
        evidence_refs: Supplied locators that show it.
    """

    category: AuditCategory
    explanation: str = Field(min_length=1, max_length=1500)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CriticEstimate(ContractModel):
    """One critic's probability of success from one estimation point.

    Attributes:
        step: The estimation point's step.
        probability: Estimated probability that a continuation from here succeeds.
        rationale: The evidence the estimate rests on.
        evidence_refs: Supplied locators backing the estimate.
    """

    step: int = Field(ge=0)
    probability: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=1200)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CriticAssessment(ContractModel):
    """One critic attempt's estimates for every chosen point.

    Attributes:
        estimates: Exactly one estimate per chosen point.
        audit: Eval-design problems the critic saw while reading the group.
        limitations: What the critic could not determine from the evidence.
    """

    estimates: tuple[CriticEstimate, ...] = Field(min_length=1)
    audit: tuple[AuditFlag, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)


class VariationAxis(ContractModel):
    """One place in the world state where changing the value changes the right decision.

    Attributes:
        path: JSON pointer into the world state; scenarios replace exactly this path.
        values: Two or more values the path may take; scenarios may use no others.
        effect: How the right choice changes with the value.
    """

    path: str = Field(pattern=r"^(/[^/]*)+$")
    values: tuple[JsonValue, ...] = Field(min_length=2, max_length=8)
    effect: str = Field(min_length=1, max_length=800)


class OptionCall(ContractModel):
    """The exact first tool call that executes one option as a competing branch.

    Attributes:
        tool: A tool the actor may call.
        arguments: The call's JSON arguments.
    """

    tool: str = Field(min_length=1, max_length=256)
    arguments: JsonObject = Field(default_factory=dict)


class DecisionOption(ContractModel):
    """One action the actor could take at the decision.

    Attributes:
        option_id: Stable identity; scenarios name their correct choice by it.
        action: The action, stated concretely enough to execute as a competing branch.
        call: The exact first tool call that executes the option, when the author gave one;
            contrasts force it as the option's branch.
    """

    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    action: str = Field(min_length=1, max_length=600)
    call: OptionCall | None = Field(default=None, exclude_if=lambda value: value is None)


class Criterion(ContractModel):
    """One rubric proposition a decision grader answers with a probability.

    Attributes:
        criterion_id: Stable identity within one decision state.
        text: A single yes/no proposition about the finished episode.
    """

    criterion_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    text: str = Field(min_length=1, max_length=400)


class InstructionValue(ContractModel):
    """One exact task instruction the actor may be given, and what it obliges.

    Attributes:
        value_id: Stable identity; scenarios name their instruction by it.
        instruction: The exact instruction text.
        kind: ``original`` is the task's own instruction; ``reworded`` says the same thing in
            other words and changes no obligation; ``goal_change`` and ``consent_change`` change
            what the actor is asked or permitted to do.
        equivalent_to: For a rewording, the value whose obligations it keeps; None otherwise.
        obligations: What the actor must and must not do under this instruction.
        authorized_by: Supplied locators authorizing a goal or consent change; required for one.
        criteria: Criterion identities that apply under this instruction; required for a goal or
            consent change, empty for a rewording, which inherits its equivalent's.
    """

    value_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    instruction: str = Field(min_length=1, max_length=4000)
    kind: InstructionKind
    equivalent_to: str | None = None
    obligations: str = Field(min_length=1, max_length=800)
    authorized_by: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _bound(self) -> Self:
        """Tie rewordings to an equivalent and changes to an authorization and criteria."""
        if (self.kind == "reworded") != (self.equivalent_to is not None):
            raise ValueError("exactly a reworded instruction names the value it is equivalent to")
        if self.kind == "reworded" and self.criteria:
            raise ValueError(
                "a rewording changes no obligation, so it binds no criteria of its own"
            )
        if self.kind in ("goal_change", "consent_change") and not (
            self.authorized_by and self.criteria
        ):
            raise ValueError(
                "a goal or consent change must cite what authorizes it and bind its own criteria"
            )
        return self


class InstructionAxis(ContractModel):
    """The declared instruction variation of one decision, separate from its state axes.

    Attributes:
        values: Two to six instruction values; exactly one is the original.
        effect: How the right choice changes with the instruction.
    """

    values: tuple[InstructionValue, ...] = Field(min_length=2, max_length=6)
    effect: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        """Require distinct values, one original and rewordings of non-rewordings."""
        ids = {value.value_id: value for value in self.values}
        if len(ids) != len(self.values):
            raise ValueError("instruction value identities must be distinct")
        if sum(value.kind == "original" for value in self.values) != 1:
            raise ValueError("an instruction axis has exactly one original value")
        for value in self.values:
            target = ids.get(value.equivalent_to or "")
            if value.kind == "reworded" and (target is None or target.kind == "reworded"):
                raise ValueError(f"rewording {value.value_id} must reword a declared non-rewording")
        if len({value.instruction.strip() for value in self.values}) != len(self.values):
            raise ValueError("instruction values must differ in text")
        return self

    def value(self, instruction: str | None) -> InstructionValue | None:
        """Return the value with exactly this text, the original for None, else None."""
        for value in self.values:
            if (instruction is None and value.kind == "original") or (
                instruction is not None and value.instruction.strip() == instruction.strip()
            ):
                return value
        return None

    def obligation(self, instruction: str | None) -> str | None:
        """Return the identity of the obligations an instruction carries: a rewording's equivalent.

        Args:
            instruction: Exact instruction text, or None for the original.

        Returns:
            The value identity whose obligations apply, or None for an undeclared instruction.
        """
        value = self.value(instruction)
        if value is None:
            return None
        return value.equivalent_to or value.value_id


class DecisionState(ContractModel):
    """The authored representation of the world just before a decision worth varying.

    Attributes:
        step: Trajectory step whose before-state this describes.
        decision: The decision the state sets up.
        options: The competing actions; two or more, compared by identity, never by prose.
        relevant_paths: State paths the decision depends on.
        axes: Where the state may vary, with the permitted values.
        invariants: State paths whose values no scenario may change.
        instruction_axis: The declared instruction variation, separate from the state axes;
            None (the default) for a state-only decision, whose instruction is frozen.
        criteria: Rubric propositions for grading an episode started here.
        audit: Eval-design problems this state would carry into an environment.
        evidence_refs: Supplied locators backing the representation.
    """

    step: int = Field(ge=0)
    decision: str = Field(min_length=1, max_length=600)
    options: tuple[DecisionOption, ...] = Field(min_length=2, max_length=6)
    relevant_paths: tuple[str, ...] = Field(min_length=1, max_length=24)
    axes: tuple[VariationAxis, ...] = Field(min_length=1, max_length=8)
    invariants: tuple[str, ...] = ()
    instruction_axis: InstructionAxis | None = None
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=32)
    audit: tuple[AuditFlag, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> DecisionState:
        """Keep options, axes, invariants and criteria distinct and non-overlapping."""
        paths = [axis.path for axis in self.axes]
        if len(set(paths)) != len(paths):
            raise ValueError("each variation axis must name a distinct path")
        if any(_overlaps(path, invariant) for path in paths for invariant in self.invariants):
            raise ValueError("a variation axis cannot contain or sit inside an invariant")
        for name, identities in (
            ("option", [option.option_id for option in self.options]),
            ("criterion", [criterion.criterion_id for criterion in self.criteria]),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(f"{name} identities must be distinct")
        if self.instruction_axis is not None:
            known = {criterion.criterion_id for criterion in self.criteria}
            bound = {c for value in self.instruction_axis.values for c in value.criteria}
            if bound - known:
                raise ValueError(
                    f"instruction values bind unknown criteria {sorted(bound - known)}"
                )
        return self

    @property
    def vary_instruction(self) -> bool:
        """Whether the decision declares an instruction axis, so scenarios may change it."""
        return self.instruction_axis is not None


class StateEdit(ContractModel):
    """Set one declared variation axis to one of its permitted values.

    Attributes:
        path: JSON pointer of the axis.
        value: Value to place there.
    """

    path: str = Field(pattern=r"^(/[^/]*)+$")
    value: JsonValue


class WitnessStep(ContractModel):
    """One permitted tool call by which the actor can observe deciding evidence.

    Attributes:
        tool: Name of a tool the actor may call.
        arguments: Arguments of the call.
        reveals: Deciding state paths whose values the call's observation shows.
    """

    tool: str = Field(min_length=1)
    arguments: JsonObject
    reveals: tuple[str, ...] = Field(min_length=1)


class FreshBriefing(ContractModel):
    """An authored new task that starts the actor without the retained prefix.

    Attributes:
        text: What the actor is told about its situation: only facts it is entitled to know at
            the start, never the intended action and never knowledge it had not observed.
        entitled_paths: State paths whose values the briefing explains; each must have been
            observable to the actor at the start and none may be edited.
    """

    text: str = Field(min_length=1, max_length=4000)
    entitled_paths: tuple[str, ...] = Field(default=(), max_length=24)


class ActorView(ContractModel):
    """What the actor had legitimately seen at a variant's start, for briefing leak checks.

    Attributes:
        instruction: The original public task instruction.
        observed: Every actor-visible text at the start: task prompt, messages and replies.
    """

    instruction: str = Field(min_length=1)
    observed: tuple[str, ...] = ()


class Scenario(ContractModel):
    """One variant of the world before the decision, with the option it makes correct.

    Attributes:
        scenario_id: Stable identity within one scenario set.
        edits: Axis replacements applied to the decision state's before-state; empty for an
            instruction-only variant or the unedited world.
        instruction: Exact text of the instruction value the actor is given; None gives the
            original. Allowed only when the decision state declares an instruction axis.
        initialization: ``resume`` keeps the coherent actor-visible prefix and its remaining
            budget; ``fresh_briefing`` starts the actor on an authored new task.
        briefing: The fresh briefing, exactly when ``initialization`` is ``fresh_briefing``.
        correct_option: Identity of the option that is correct in this variant.
        deciding_paths: Edited-state paths whose values make that option correct.
        witness: Permitted tool calls, before the decision, that reveal every deciding path.
        rationale: Why those values make the option correct.
        criteria: Criterion identities that apply to this variant.
    """

    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    edits: tuple[StateEdit, ...] = Field(default=(), max_length=16)
    instruction: str | None = Field(default=None, max_length=4000)
    initialization: InitializationMode = "resume"
    briefing: FreshBriefing | None = None
    correct_option: str = Field(min_length=1, max_length=64)
    deciding_paths: tuple[str, ...] = Field(min_length=1, max_length=12)
    witness: tuple[WitnessStep, ...] = Field(min_length=1, max_length=16)
    rationale: str = Field(min_length=1, max_length=1200)
    criteria: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _initialized(self) -> Self:
        """Require a briefing exactly for a fresh-briefing start."""
        if (self.briefing is not None) != (self.initialization == "fresh_briefing"):
            raise ValueError("a scenario has a briefing exactly when it starts by fresh briefing")
        return self


class ScenarioSet(ContractModel):
    """Scenarios that vary one decision, as authored.

    Attributes:
        scenarios: Two to eight variants.
        limitations: What the author could not ensure.
    """

    scenarios: tuple[Scenario, ...] = Field(min_length=2, max_length=MAXIMUM_SCENARIOS)
    limitations: tuple[str, ...] = Field(min_length=1)


class ScenarioProblem(ContractModel):
    """One structural reason a scenario or scenario set cannot become an environment.

    Attributes:
        scenario_id: The scenario at fault, or None for a set-level problem.
        reason: What check failed.
    """

    scenario_id: str | None
    reason: str


class ScenarioLimits(ContractModel):
    """What the actor can do before the decision, against which witnesses are checked.

    Attributes:
        tools: Tools the actor may call.
        remaining_calls: Tool calls left to the actor at the decision point.
    """

    tools: frozenset[str] = Field(min_length=1)
    remaining_calls: int = Field(ge=0)


class FrozenPart(ContractModel):
    """What a checkpoint actually holds for one part of the restorable state.

    Attributes:
        part: Which part.
        status: ``held`` when the checkpoint restores it; ``absent`` when it holds none;
            ``external`` when it lives outside the checkpoint and cannot be restored from it.
        sha256: Digest of the held content; present exactly when held.
        detail: What was found.
    """

    part: StatePart
    status: Literal["held", "absent", "external"]
    sha256: Sha256 | None = None
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def _digest(self) -> Self:
        """Bind a digest to exactly the held parts."""
        if (self.sha256 is not None) != (self.status == "held"):
            raise ValueError("a frozen part has a digest exactly when it is held")
        return self


class FrozenState(ContractModel):
    """The complete restorable state a variant starts from, part by part.

    Attributes:
        parts: One entry per :data:`StatePart`, in declaration order.
    """

    parts: tuple[FrozenPart, ...]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Require every state part exactly once."""
        if tuple(row.part for row in self.parts) != _STATE_PARTS:
            raise ValueError("a frozen state records every state part once, in order")
        return self

    def unrestorable(self, required: frozenset[StatePart]) -> tuple[str, ...]:
        """Return why each required part is not restorable; empty when all are held."""
        return tuple(
            f"{row.part} is {row.status}: {row.detail}"
            for row in self.parts
            if row.part in required and row.status != "held"
        )


JudgeSelection.model_rebuild()
CriticAssessment.model_rebuild()
DecisionState.model_rebuild()

_STATE_PARTS: tuple[StatePart, ...] = get_args(StatePart)


def resolve_pointer(state: JsonObject, path: str) -> tuple[bool, JsonValue]:
    """Read one JSON pointer from a state.

    Args:
        state: World state object.
        path: RFC 6901 pointer such as ``/accounts/0/balance``.

    Returns:
        Whether the path exists, and its value when it does.
    """
    current: JsonValue = state
    for raw in path.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def apply_edits(state: JsonObject, edits: tuple[StateEdit, ...]) -> JsonObject:
    """Return a copy of a state with every edit applied.

    Args:
        state: World state object; it is not modified.
        edits: Edits applied in order; a missing final key is created inside an existing object.

    Returns:
        The edited copy.

    Raises:
        ValueError: An edit's parent container does not exist or a list index is out of range.
    """
    edited: JsonObject = {key: _copy(value) for key, value in state.items()}
    for edit in edits:
        tokens = [raw.replace("~1", "/").replace("~0", "~") for raw in edit.path.split("/")[1:]]
        parent_path = "/" + "/".join(edit.path.split("/")[1:-1]) if len(tokens) > 1 else ""
        exists, parent = (True, edited) if not parent_path else resolve_pointer(edited, parent_path)
        last = tokens[-1]
        if exists and isinstance(parent, dict):
            parent[last] = _copy(edit.value)
        elif exists and isinstance(parent, list) and last.isdigit() and int(last) < len(parent):
            parent[int(last)] = _copy(edit.value)
        else:
            raise ValueError(f"state edit {edit.path} has no existing parent to write into")
    return edited


def check_scenarios(
    before: JsonObject,
    decision: DecisionState,
    scenarios: ScenarioSet,
    limits: ScenarioLimits,
    view: ActorView | None = None,
) -> tuple[ScenarioProblem, ...]:
    """Structurally check a scenario set against the decision state it varies.

    These checks are necessary, not sufficient: passing them does not show that the world model
    behaves correctly in the edited state, that the witness calls really reveal the evidence, or
    that the chosen option leads to better outcomes. Those need executed witnesses, competing
    actions from the same checkpoint and independent specifications.

    Args:
        before: World state just before the decision's step.
        decision: The authored decision state.
        scenarios: The authored variants.
        limits: Tools and remaining calls available to the actor at the decision.
        view: What the actor had seen at the start, needed to leak-check a fresh briefing; a
            fresh briefing without it is refused.

    Returns:
        Every problem found; empty when the set may proceed to executed checks.
    """
    problems: list[ScenarioProblem] = []
    identities = [scenario.scenario_id for scenario in scenarios.scenarios]
    if len(set(identities)) != len(identities):
        problems.append(ScenarioProblem(scenario_id=None, reason="scenario ids repeat"))
    if len({scenario.correct_option for scenario in scenarios.scenarios}) < 2:
        problems.append(
            ScenarioProblem(
                scenario_id=None,
                reason="every scenario makes the same option correct; the set has no decision "
                "variance",
            )
        )
    seen: dict[tuple[str, str | None], Scenario] = {}
    for scenario in scenarios.scenarios:
        found = _scenario_problems(before, decision, scenario, limits, view)
        problems.extend(found)
        if found:
            continue
        axis = decision.instruction_axis
        key = (
            json.dumps(apply_edits(before, scenario.edits), sort_keys=True),
            None if axis is None else axis.obligation(scenario.instruction),
        )
        other = seen.setdefault(key, scenario)
        if other is scenario:
            continue
        reason = (
            f"its world state and instruction are identical to scenario {other.scenario_id}"
            if axis is None or axis.value(other.instruction) == axis.value(scenario.instruction)
            else f"it differs from scenario {other.scenario_id} only by an equivalent rewording "
            "of the instruction, which changes no obligation and cannot establish decision "
            "variance"
        )
        problems.append(_problem(scenario, reason))
    return tuple(problems)


def instruction_problems(decision: DecisionState, scenario: Scenario) -> list[str]:
    """Check a scenario's instruction value and initialization against the instruction axis.

    Args:
        decision: The decision state, whose instruction axis is separate from its state axes.
        scenario: The variant.

    Returns:
        Why the instruction or initialization is not coherent; empty when it is.
    """
    axis = decision.instruction_axis
    if axis is None:
        if scenario.instruction is None:
            return []
        return [
            "replaces the instruction, which is frozen: the decision is state-only and wording "
            "alone cannot vary it"
        ]
    value = axis.value(scenario.instruction)
    if value is None:
        return ["its instruction is not a declared instruction value"]
    found: list[str] = []
    if value.kind != "original" and scenario.initialization == "resume":
        found.append(
            f"instruction value {value.value_id} changes the instruction the resumed prefix "
            "already stated; start it by fresh_briefing"
        )
    bound = next(v for v in axis.values if v.value_id == (value.equivalent_to or value.value_id))
    applicable = set(bound.criteria if bound is not None else ())
    if applicable and not set(scenario.criteria) <= applicable:
        found.append(
            f"applies criteria {sorted(set(scenario.criteria) - applicable)} that instruction "
            f"value {value.value_id} does not bind"
        )
    return found


def briefing_problems(before: JsonObject, scenario: Scenario, view: ActorView) -> list[str]:
    """Find the structural problems of a fresh briefing's entitled paths.

    Each entitled path must exist in the edited state, must not overlap an edited path, and every
    scalar under it must appear in some text the actor had observed. These are the only briefing
    checks made in code; what the briefing text itself gives away is judged by the leak reviewer.

    Args:
        before: World state just before the decision's step.
        scenario: A fresh-briefing variant.
        view: What the actor had seen at the variant's start.

    Returns:
        Every problem found; empty when none was.
    """
    if scenario.briefing is None:
        return []
    edited = apply_edits(before, scenario.edits)
    edits = {edit.path for edit in scenario.edits}
    found: list[str] = []
    for path in scenario.briefing.entitled_paths:
        exists, held = resolve_pointer(edited, path)
        if not exists or any(_overlaps(path, edit) for edit in edits):
            found.append(f"entitled path {path} is absent or edited, so it cannot be briefed")
        elif not all(
            any(mentions(seen, item) for seen in view.observed) for item in json_scalars(held)
        ):
            found.append(f"entitled path {path} was not observable at the start")
    return found


def mentions(text: str, value: JsonValue) -> bool:
    """Return whether a text states a scalar as a whole token.

    Strings and numbers match as whole tokens, case-insensitively for strings; booleans and
    null match their JSON spelling.

    Args:
        text: Text to scan.
        value: Scalar to look for.

    Returns:
        Whether the value appears.
    """
    needle = (
        value
        if isinstance(value, str)
        else json.dumps(value)
        if not isinstance(value, float) or not value.is_integer()
        else str(int(value))
    ).strip()
    return bool(needle) and bool(
        re.search(rf"(?<![\w.]){re.escape(needle)}(?![\w])", text, flags=re.IGNORECASE)
    )


def json_scalars(value: JsonValue) -> Iterator[JsonValue]:
    """Yield every scalar inside a JSON value."""
    if isinstance(value, dict):
        for item in value.values():
            yield from json_scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_scalars(item)
    else:
        yield value


def message_texts(messages: Iterable[ModelMessage]) -> tuple[str, ...]:
    """Return every actor-visible text of some messages: content and tool-call arguments."""
    texts: list[str] = []
    for message in messages:
        if message.content is not None:
            texts.append(message.content)
        if message.assistant_action is not None:
            if message.assistant_action.content is not None:
                texts.append(message.assistant_action.content)
            texts.extend(
                json.dumps(call.arguments, sort_keys=True)
                for call in message.assistant_action.tool_calls
            )
    return tuple(texts)


def revealing_texts(
    texts: Iterable[str], values: tuple[JsonValue, ...], keys: frozenset[str]
) -> list[str]:
    """Return which watched values or field names appear in some texts.

    Strings and numbers match as whole tokens, case-insensitively for strings; booleans and null
    match their JSON spelling. A field name matches only as a key of a JSON object in the text.

    Args:
        texts: Texts to scan.
        values: Watched scalars.
        keys: Watched field names.

    Returns:
        Sorted descriptions of what was found.
    """
    found: set[str] = set()
    for text in texts:
        found.update(f"value {json.dumps(value)}" for value in values if mentions(text, value))
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        found.update(f"key {key!r}" for key in keys if key in json_keys(parsed))
    return sorted(found)


def json_keys(value: JsonValue) -> set[str]:
    """Return every object key anywhere inside a JSON value."""
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys |= json_keys(item)
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys |= json_keys(item)
        return keys
    return set()


def _scenario_problems(
    before: JsonObject,
    decision: DecisionState,
    scenario: Scenario,
    limits: ScenarioLimits,
    view: ActorView | None,
) -> list[ScenarioProblem]:
    """Check one scenario's option, edits, invariants, instruction, briefing and witness.

    Args:
        before: World state just before the decision's step.
        decision: The decision state the scenario varies.
        scenario: The variant to check.
        limits: Tools and remaining calls available to the actor.
        view: What the actor had seen at the start, for a fresh briefing.

    Returns:
        Problems with this scenario.
    """
    found: list[ScenarioProblem] = []
    axes = {axis.path: axis for axis in decision.axes}
    if scenario.correct_option not in {option.option_id for option in decision.options}:
        found.append(_problem(scenario, f"unknown option {scenario.correct_option!r}"))
    found.extend(_problem(scenario, reason) for reason in instruction_problems(decision, scenario))
    unknown = sorted(set(scenario.criteria) - {c.criterion_id for c in decision.criteria})
    if unknown:
        found.append(_problem(scenario, f"unknown criteria {unknown}"))
    edited_paths = [edit.path for edit in scenario.edits]
    if len(set(edited_paths)) != len(edited_paths):
        found.append(_problem(scenario, "edits the same axis twice"))
    for edit in scenario.edits:
        axis = axes.get(edit.path)
        if axis is None:
            found.append(_problem(scenario, f"edit {edit.path} is not a declared variation axis"))
        elif not any(_same(edit.value, value) for value in axis.values):
            found.append(
                _problem(scenario, f"edit {edit.path} uses a value the axis does not permit")
            )
    try:
        edited = apply_edits(before, scenario.edits)
    except ValueError as error:
        return [*found, _problem(scenario, str(error))]
    for invariant in decision.invariants:
        if resolve_pointer(before, invariant) != resolve_pointer(edited, invariant):
            found.append(_problem(scenario, f"invariant {invariant} changed"))
    if scenario.briefing is not None and view is None:
        found.append(
            _problem(scenario, "a fresh briefing cannot be leak-checked without the actor's view")
        )
    elif view is not None:
        leaks = briefing_problems(before, scenario, view)
        found.extend(_problem(scenario, reason) for reason in leaks)
    for path in scenario.deciding_paths:
        exists, _ = resolve_pointer(edited, path)
        if not exists:
            found.append(_problem(scenario, f"deciding fact {path} is absent from the state"))
    revealed = {path for step in scenario.witness for path in step.reveals}
    hidden = sorted(set(scenario.deciding_paths) - revealed)
    if hidden:
        found.append(
            _problem(
                scenario,
                f"no witness call reveals deciding facts {hidden}, so the choice could only be "
                "guessed",
            )
        )
    forbidden = sorted({step.tool for step in scenario.witness} - limits.tools)
    if forbidden:
        found.append(_problem(scenario, f"witness uses tools the actor cannot call: {forbidden}"))
    if len(scenario.witness) > limits.remaining_calls:
        found.append(
            _problem(
                scenario,
                f"witness needs {len(scenario.witness)} calls but only "
                f"{limits.remaining_calls} remain",
            )
        )
    return found


def _problem(scenario: Scenario, reason: str) -> ScenarioProblem:
    """Build a problem tied to one scenario."""
    return ScenarioProblem(scenario_id=scenario.scenario_id, reason=reason)


def _same(left: JsonValue, right: JsonValue) -> bool:
    """Compare two JSON values exactly, distinguishing booleans from numbers."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _overlaps(path: str, other: str) -> bool:
    """Return whether one pointer equals, contains or sits inside the other."""
    return path == other or path.startswith(other + "/") or other.startswith(path + "/")


def _copy(value: JsonValue) -> JsonValue:
    """Deep-copy a JSON value without sharing containers."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value

"""What judges, critics and scenario authors return, and the checks every answer must pass.

Each role answers with one of these models through a forced structured call. The models carry only
data: steps, probabilities, JSON-pointer state paths and propositions. Every reference an answer
makes is checked against what the role was actually shown, and every authored scenario is checked
mechanically before it can become an environment:

- edits replace only declared variation axes, with only their permitted values;
- invariants hold the same value before and after the edits, compared as values, so replacing a
  parent object cannot slip a changed invariant through;
- every variant's world state differs from every other variant's (the unedited world may be one);
- the instruction is frozen unless the decision state declares instruction variation;
- correct choices are option identities from a declared set, so "refund" and "issue refund"
  cannot pass as different decisions;
- every deciding fact is revealed by a witness of permitted tool calls that fits in the actor's
  remaining budget, so a scenario cannot require guessing a hidden value.

These checks are structural. Whether the witness calls really reveal the evidence, whether the
edited state is coherent with the retained conversation and instance facts, and whether the
correct option really leads to better outcomes than the others are established by executed
witnesses and competing-action continuations from the same checkpoint, never by these gates.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue

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


class DecisionOption(ContractModel):
    """One action the actor could take at the decision.

    Attributes:
        option_id: Stable identity; scenarios name their correct choice by it.
        action: The action, stated concretely enough to execute as a competing branch.
    """

    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    action: str = Field(min_length=1, max_length=600)


class Criterion(ContractModel):
    """One rubric proposition a decision grader answers with a probability.

    Attributes:
        criterion_id: Stable identity within one decision state.
        text: A single yes/no proposition about the finished episode.
    """

    criterion_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    text: str = Field(min_length=1, max_length=400)


class DecisionState(ContractModel):
    """The authored representation of the world just before a decision worth varying.

    Attributes:
        step: Trajectory step whose before-state this describes.
        decision: The decision the state sets up.
        options: The competing actions; two or more, compared by identity, never by prose.
        relevant_paths: State paths the decision depends on.
        axes: Where the state may vary, with the permitted values.
        invariants: State paths whose values no scenario may change.
        vary_instruction: Whether scenarios may replace the task instruction. Off by default:
            a causal variant changes the world, not what the actor is told.
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
    vary_instruction: bool = False
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
        return self


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


class Scenario(ContractModel):
    """One variant of the world before the decision, with the option it makes correct.

    Attributes:
        scenario_id: Stable identity within one scenario set.
        edits: Axis replacements applied to the decision state's before-state.
        instruction: Replacement task instruction; allowed only when the decision state
            declares instruction variation.
        correct_option: Identity of the option that is correct in this variant.
        deciding_paths: Edited-state paths whose values make that option correct.
        witness: Permitted tool calls, before the decision, that reveal every deciding path.
        rationale: Why those values make the option correct.
        criteria: Criterion identities that apply to this variant.
    """

    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    edits: tuple[StateEdit, ...] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, max_length=4000)
    correct_option: str = Field(min_length=1, max_length=64)
    deciding_paths: tuple[str, ...] = Field(min_length=1, max_length=12)
    witness: tuple[WitnessStep, ...] = Field(min_length=1, max_length=16)
    rationale: str = Field(min_length=1, max_length=1200)
    criteria: tuple[str, ...] = Field(min_length=1)


class ScenarioSet(ContractModel):
    """Scenarios that vary one decision, as authored.

    Attributes:
        scenarios: Two to eight variants.
        limitations: What the author could not ensure.
    """

    scenarios: tuple[Scenario, ...] = Field(min_length=2, max_length=8)
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


JudgeSelection.model_rebuild()
CriticAssessment.model_rebuild()


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
    states: dict[str, JsonObject] = {}
    for scenario in scenarios.scenarios:
        found = _scenario_problems(before, decision, scenario, limits)
        problems.extend(found)
        if not found:
            states[scenario.scenario_id] = apply_edits(before, scenario.edits)
    seen: dict[str, str] = {}
    for scenario_id, state in states.items():
        key = json.dumps(state, sort_keys=True)
        if key in seen:
            problems.append(
                ScenarioProblem(
                    scenario_id=scenario_id,
                    reason=f"its world state is identical to scenario {seen[key]}",
                )
            )
        seen.setdefault(key, scenario_id)
    return tuple(problems)


def _scenario_problems(
    before: JsonObject, decision: DecisionState, scenario: Scenario, limits: ScenarioLimits
) -> list[ScenarioProblem]:
    """Check one scenario's option, edits, invariants, criteria and observation witness.

    Args:
        before: World state just before the decision's step.
        decision: The decision state the scenario varies.
        scenario: The variant to check.
        limits: Tools and remaining calls available to the actor.

    Returns:
        Problems with this scenario.
    """
    found: list[ScenarioProblem] = []
    axes = {axis.path: axis for axis in decision.axes}
    if scenario.correct_option not in {option.option_id for option in decision.options}:
        found.append(_problem(scenario, f"unknown option {scenario.correct_option!r}"))
    if scenario.instruction is not None and not decision.vary_instruction:
        found.append(_problem(scenario, "replaces the instruction, which is frozen"))
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

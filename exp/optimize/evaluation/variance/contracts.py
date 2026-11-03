"""What judges, critics and scenario authors return, and the checks every answer must pass.

Each role answers with one of these models through a forced structured call. The models carry only
data: steps, probabilities, JSON-pointer state paths and propositions. Every reference an answer
makes is checked against what the role was actually shown, and every authored scenario is checked
mechanically before it can become an environment:

- edits touch only the declared variation axes and never an invariant;
- the facts that decide a scenario exist in its edited state, so solving it never requires a
  guess (a task that can only be solved by guessing teaches the model to hallucinate);
- across one scenario set at least two different choices are correct, so the environment rewards
  making the decision rather than repeating one memorized action.
"""

from __future__ import annotations

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
        path: JSON pointer into the world state.
        values: At least two values the path may take across scenarios.
        effect: How the right choice changes with the value.
    """

    path: str = Field(pattern=r"^(/[^/]*)+$")
    values: tuple[JsonValue, ...] = Field(min_length=2, max_length=8)
    effect: str = Field(min_length=1, max_length=800)


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
        relevant_paths: State paths the decision depends on.
        axes: Where the state may vary and why each change matters.
        invariants: State paths no scenario may change.
        criteria: Rubric propositions for grading an episode started here.
        audit: Eval-design problems this state would carry into an environment.
        evidence_refs: Supplied locators backing the representation.
    """

    step: int = Field(ge=0)
    decision: str = Field(min_length=1, max_length=600)
    relevant_paths: tuple[str, ...] = Field(min_length=1, max_length=24)
    axes: tuple[VariationAxis, ...] = Field(min_length=1, max_length=8)
    invariants: tuple[str, ...] = ()
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=32)
    audit: tuple[AuditFlag, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> DecisionState:
        """Keep axes, invariants and criteria distinct and non-overlapping."""
        paths = [axis.path for axis in self.axes]
        if len(set(paths)) != len(paths):
            raise ValueError("each variation axis must name a distinct path")
        if set(paths) & set(self.invariants):
            raise ValueError("a variation axis cannot also be an invariant")
        identities = [criterion.criterion_id for criterion in self.criteria]
        if len(set(identities)) != len(identities):
            raise ValueError("criterion identities must be distinct")
        return self


class StateEdit(ContractModel):
    """Set one world-state path to a value.

    Attributes:
        path: JSON pointer of an existing or new leaf.
        value: Value to place there.
    """

    path: str = Field(pattern=r"^(/[^/]*)+$")
    value: JsonValue


class Scenario(ContractModel):
    """One variant of the world before the decision, with the choice it makes correct.

    Attributes:
        scenario_id: Stable identity within one scenario set.
        edits: State edits applied to the decision state's before-state.
        instruction: Replacement task instruction, or None to keep the original.
        correct_choice: The decision that is correct in this variant.
        deciding_paths: Edited-state paths whose values make that choice correct.
        rationale: Why those values make the choice correct.
        criteria: Criterion identities that apply to this variant.
    """

    scenario_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    edits: tuple[StateEdit, ...] = Field(min_length=1, max_length=16)
    instruction: str | None = Field(default=None, max_length=4000)
    correct_choice: str = Field(min_length=1, max_length=600)
    deciding_paths: tuple[str, ...] = Field(min_length=1, max_length=12)
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
    """One mechanical reason a scenario or scenario set cannot become an environment.

    Attributes:
        scenario_id: The scenario at fault, or None for a set-level problem.
        reason: What check failed.
    """

    scenario_id: str | None
    reason: str


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
    before: JsonObject, decision: DecisionState, scenarios: ScenarioSet
) -> tuple[ScenarioProblem, ...]:
    """Mechanically check a scenario set against the decision state it varies.

    Args:
        before: World state just before the decision's step.
        decision: The authored decision state.
        scenarios: The authored variants.

    Returns:
        Every problem found; empty when the set may become environments.
    """
    problems: list[ScenarioProblem] = []
    axes = {axis.path for axis in decision.axes}
    criteria = {criterion.criterion_id for criterion in decision.criteria}
    identities = [scenario.scenario_id for scenario in scenarios.scenarios]
    if len(set(identities)) != len(identities):
        problems.append(ScenarioProblem(scenario_id=None, reason="scenario ids repeat"))
    choices = {" ".join(s.correct_choice.lower().split()) for s in scenarios.scenarios}
    if len(choices) < 2:
        problems.append(
            ScenarioProblem(
                scenario_id=None,
                reason="every scenario makes the same choice correct; the set has no decision "
                "variance",
            )
        )
    for scenario in scenarios.scenarios:
        problems.extend(_scenario_problems(before, scenario, axes, set(decision.invariants)))
        unknown = sorted(set(scenario.criteria) - criteria)
        if unknown:
            problems.append(
                ScenarioProblem(
                    scenario_id=scenario.scenario_id, reason=f"unknown criteria {unknown}"
                )
            )
    return tuple(problems)


def _scenario_problems(
    before: JsonObject, scenario: Scenario, axes: set[str], invariants: set[str]
) -> list[ScenarioProblem]:
    """Check one scenario's edits and that its deciding facts are present.

    Args:
        before: World state just before the decision's step.
        scenario: The variant to check.
        axes: Paths the decision state allows to vary.
        invariants: Paths no scenario may change.

    Returns:
        Problems with this scenario.
    """
    found: list[ScenarioProblem] = []
    for edit in scenario.edits:
        if not any(edit.path == axis or edit.path.startswith(axis + "/") for axis in axes):
            found.append(_problem(scenario, f"edit {edit.path} is outside every variation axis"))
        if any(edit.path == path or edit.path.startswith(path + "/") for path in invariants):
            found.append(_problem(scenario, f"edit {edit.path} changes an invariant"))
    try:
        edited = apply_edits(before, scenario.edits)
    except ValueError as error:
        return [*found, _problem(scenario, str(error))]
    for path in scenario.deciding_paths:
        exists, _ = resolve_pointer(edited, path)
        if not exists:
            found.append(
                _problem(
                    scenario,
                    f"deciding fact {path} is absent from the scenario state, so the right choice "
                    "could only be guessed",
                )
            )
    return found


def _problem(scenario: Scenario, reason: str) -> ScenarioProblem:
    """Build a problem tied to one scenario."""
    return ScenarioProblem(scenario_id=scenario.scenario_id, reason=reason)


def _copy(value: JsonValue) -> JsonValue:
    """Deep-copy a JSON value without sharing containers."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value

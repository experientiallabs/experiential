"""Acceptance controls for a start's verifier under probe-coordinate grading.

A verifier is only trustworthy if the grade tells known outcomes apart. For one start (its raw
initial state, probe paths and verifier) these controls synthesize workflows over the start's
own probe projection and grade each with the real grader:

* ``correct``: every rule satisfied, protected values kept; expected ``pass``.
* ``no_op``: the final state is the initial one; expected ``fail`` (a task that passes when the
  agent does nothing does not measure the agent).
* ``wrong_content``: correct, except each rule-checked value is replaced by a value its rule
  refuses; expected ``fail``.
* ``missing_observation``: the final state was not observed; expected ``unassessable``.
* ``observed_absence``: a required probe present at the start is absent from the observed final
  state; expected ``fail``, distinct from ``missing_observation``.

Values are synthesized only from the verifier's own finite rules, so a control a verifier cannot
express (a rule-free goal, a rule needing a value the controls cannot invent) is reported as
``not_constructible`` rather than guessed.
"""

from __future__ import annotations

import copy
from typing import Literal

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.optimize.evaluation.variance.verifier_paths import keyed_by_paths, project_verifier, split
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, grade
from exp.simulation.reconstruction.models import resolve

Control = Literal["correct", "no_op", "wrong_content", "missing_observation", "observed_absence"]
EXPECTED: dict[Control, str] = {
    "correct": "pass",
    "no_op": "fail",
    "wrong_content": "fail",
    "missing_observation": "unassessable",
    "observed_absence": "fail",
}
"""The grade each control must receive."""


class ControlResult(ContractModel):
    """One control's grade against what it must be.

    Attributes:
        control: Which control.
        expected: The status it must receive.
        status: The status it received, or ``not_constructible``.
        reason: The grader's reason, or why the control could not be built.
        ok: Whether the status is the expected one.
    """

    control: Control
    expected: str
    status: str
    reason: str
    ok: bool


def projection(state: JsonObject, probes: dict[str, str]) -> JsonObject:
    """The probe projection of a raw state; a probe that does not resolve is absent."""
    values: JsonObject = {}
    for name, pointer in probes.items():
        try:
            values[name] = copy.deepcopy(resolve(state, pointer))
        except ValueError:
            continue
    return values


def _set(value: JsonObject, pointer: str, item: JsonValue) -> bool:
    """Write ``item`` at a pointer in the state's own key style, filling missing parents.

    A missing or null parent becomes an object, as a recorded run fills one, except inside a
    files map (:func:`~.verifier_paths.keyed_by_paths`): the simulator holds each file there
    under one flat key and never nests directories, so a witness may not either.

    Returns False when the pointer runs through a text, number or boolean, which no state can
    hold a value inside, or would nest objects inside a files map.
    """
    keys = split(pointer)
    node: JsonValue = value
    for key in keys[:-1]:
        if isinstance(node, dict):
            child = node.get(key)
            if child is None and keyed_by_paths(node):
                # No run nests a directory inside a files map; the pointer names nothing real.
                return False
            if child is None:
                # A missing or null value may become an object, as a recorded run fills it.
                child = {}
                node[key] = child
            elif not isinstance(child, (dict, list)):
                # A rule reading inside text or a number names nothing any state can hold.
                return False
            node = child
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return False
    if isinstance(node, dict) and keys:
        node[keys[-1]] = item
        return True
    if isinstance(node, list) and keys and keys[-1].isdigit() and int(keys[-1]) < len(node):
        node[int(keys[-1])] = item
        return True
    return False


_TYPED: dict[str, JsonValue] = {
    "null": None,
    "boolean": True,
    "object": {},
    "array": [],
    "number": 1.5,
    "integer": 1,
    "string": "synthesized",
}


def _satisfying(rule: OutcomeRule, current: JsonValue) -> JsonValue:
    """A value the rule accepts, keeping the current one when it already does."""
    if rule.operator == "equals":
        return rule.value
    if rule.operator == "type":
        return _TYPED.get(str(rule.value), current)
    if rule.operator == "min_items":
        size = rule.value if isinstance(rule.value, int) else 0
        return (
            list(current)[:size] + [None] * max(0, size - len(current))
            if isinstance(current, list)
            else [None] * size
        )
    if rule.operator == "max_items":
        limit = rule.value if isinstance(rule.value, int) else 0
        return list(current)[:limit] if isinstance(current, list) else []
    return current if current is not None else "synthesized"


def _refused(rule: OutcomeRule) -> tuple[bool, JsonValue]:
    """A value the rule refuses, and whether one exists (``exists`` refuses only absence)."""
    if rule.operator == "equals":
        return True, {"not": rule.value}
    if rule.operator == "type":
        return True, None if rule.value != "null" else "not null"
    if rule.operator == "min_items" and isinstance(rule.value, int) and rule.value > 0:
        return True, []
    if rule.operator == "max_items" and isinstance(rule.value, int):
        return True, [None] * (rule.value + 1)
    return False, None


def _workflow(initial: JsonObject, final: JsonObject | None, tool: str = "control") -> JsonObject:
    """A completed one-step workflow from ``initial`` to ``final`` through one allowed call."""
    after = final if final is not None else initial
    return {
        "initial_state": initial,
        "final_state": final,
        "final_answer": "done",
        "steps": [{"status": "valid", "action": {"name": tool, "arguments": {}}, "after": after}],
        "stop_reason": "completed",
    }


def run_controls(
    verifier: JsonObject, state: JsonObject, probes: dict[str, str], *, add: bool = True
) -> tuple[ControlResult, ...]:
    """Grade the five controls for one start.

    Args:
        verifier: The start's verifier, as authored or saved.
        state: The start's raw initial state.
        probes: The start's probe paths.
        add: Whether verifier pointers no probe covers get a probe, as authoring now does.

    Returns:
        One result per control, in :data:`EXPECTED` order.
    """
    probes = dict(probes)
    moved, _ = project_verifier(verifier, state, probes, add=add)
    graded = TaskVerifier.model_validate(moved)
    initial = projection(state, probes)
    tool = (graded.allowed_tools or ("control",))[0]
    rows: list[ControlResult] = []

    def workflow(value: JsonObject | None) -> JsonObject:
        return _workflow(initial, value, tool)

    def record(control: Control, value: JsonObject | None, why: str = "") -> None:
        if value is None:
            rows.append(
                ControlResult(
                    control=control,
                    expected=EXPECTED[control],
                    status="not_constructible",
                    reason=why,
                    ok=False,
                )  # fmt: skip
            )
            return
        result = grade(graded, value)
        rows.append(
            ControlResult(
                control=control,
                expected=EXPECTED[control],
                status=result.status,
                reason=result.reason,
                ok=result.status == EXPECTED[control],
            )
        )

    final = copy.deepcopy(initial)
    built = True
    for pointer in graded.required_paths:
        if pointer.startswith("/final_state/"):
            rest = pointer[len("/final_state") :]
            try:
                resolve(final, rest)
            except ValueError:
                built &= _set(final, rest, "synthesized")
    for rule in graded.rules:
        if not rule.path.startswith("/final_state/"):
            built = False
            continue
        rest = rule.path[len("/final_state") :]
        try:
            current = resolve(final, rest)
        except ValueError:
            current = None
        built &= _set(final, rest, _satisfying(rule, current))
    record(
        "correct",
        workflow(final) if built else None,
        "a rule path is outside the final state or reads inside a text or number",
    )
    record("no_op", workflow(copy.deepcopy(initial)))
    wrong = copy.deepcopy(final)
    changed = False
    for rule in graded.rules:
        exists, bad = _refused(rule)
        if exists and rule.path.startswith("/final_state/"):
            changed |= _set(wrong, rule.path[len("/final_state") :], bad)
    record(
        "wrong_content",
        workflow(wrong) if built and changed else None,
        "no rule refuses a present value",
    )
    record(
        "missing_observation",
        {**workflow(None), "final_state": None, "steps": [{"status": "unavailable"}]},
    )
    absent_from = next(
        (
            split(p)[1]
            for p in graded.required_paths
            if p.startswith("/final_state/") and split(p)[1] in initial
        ),
        None,
    )
    if absent_from is None:
        record("observed_absence", None, "no required path names a probe present at the start")
    else:
        gone = {key: val for key, val in final.items() if key != absent_from}
        record("observed_absence", workflow(gone))
    return tuple(rows)

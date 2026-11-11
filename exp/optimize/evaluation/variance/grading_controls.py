"""Acceptance controls for a start's verifier under probe-coordinate grading.

A verifier is only trustworthy if the grade tells known outcomes apart. For one start (its raw
initial state, probe paths and verifier) these controls synthesize workflows over the start's
own probe projection and grade each with the real grader:

* ``correct``: every rule satisfied, protected values kept; expected ``pass``.
* ``no_op``: the final state is the initial one; expected ``fail`` (a task that passes when the
  agent does nothing does not measure the agent).
* ``wrong_content``: correct, except every value a rule reads outside the preserved paths stays
  present with placeholder content of the same type (placeholder text, a flipped boolean, a
  changed number, placeholder items); expected ``fail``. A verifier that only checks a target
  exists, has a type or has enough items passes it and does not tell a solution from a stub.
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
from exp.common.core.json_schema import expand_local_references
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


def _schema_at(schema: JsonObject, pointer: str) -> JsonObject:
    """Read a field's declared schema from the expanded workflow schema."""
    node = schema
    for part in split(pointer):
        properties = node.get("properties")
        child = properties.get(part) if isinstance(properties, dict) else None
        if child is None and part.isdecimal():
            child = node.get("items")
        if not isinstance(child, dict):
            return {}
        node = child
    return node


def _schema_seed(schema: JsonObject) -> JsonValue:
    """Construct a simple typed item; the real grader still checks the complete witness."""
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    values = schema.get("enum")
    if isinstance(values, list) and values:
        return copy.deepcopy(values[0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((value for value in kind if isinstance(value, str)), None)
    if kind == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            return {}
        result: JsonObject = {}
        for key in required:
            if isinstance(key, str):
                child = properties.get(key)
                result[key] = _schema_seed(child if isinstance(child, dict) else {})
        return result
    return copy.deepcopy(_TYPED.get(kind)) if isinstance(kind, str) else None


def _satisfying(rule: OutcomeRule, current: JsonValue, schema: JsonObject) -> JsonValue:
    """A value the rule accepts, keeping the current one when it already does."""
    if rule.operator == "equals":
        return rule.value
    if rule.operator == "type":
        return _TYPED.get(str(rule.value), current)
    if rule.operator == "min_items":
        size = rule.value if isinstance(rule.value, int) else 0
        values = list(current) if isinstance(current, list) else []
        item = schema.get("items")
        seed = _schema_seed(item if isinstance(item, dict) else {})
        return values + [copy.deepcopy(seed) for _ in range(max(0, size - len(values)))]
    if rule.operator == "max_items":
        limit = rule.value if isinstance(rule.value, int) else 0
        return list(current)[:limit] if isinstance(current, list) else []
    return current if current is not None else "synthesized"


PLACEHOLDER = "placeholder"
"""Text a wrong-content control writes where the verifier reads a text."""


def _placeholder(value: JsonValue) -> JsonValue:
    """The same value's shape with placeholder content: present, same type, wrong content.

    Text becomes placeholder text, a boolean flips, a number changes, a list keeps its length
    and an object its keys, each holding placeholders; null stays null.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return PLACEHOLDER if value != PLACEHOLDER else PLACEHOLDER + " text"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, list):
        return [_placeholder(item) for item in value]
    if isinstance(value, dict):
        return {key: _placeholder(item) for key, item in value.items()}
    return value


def _protected(pointer: str, preserved: tuple[str, ...]) -> bool:
    """Whether a ``/final_state`` pointer reads inside, or holds, a preserved path."""
    keys = split(pointer)[1:]
    for path in preserved:
        kept = split(path)
        if keys[: len(kept)] == kept or kept[: len(keys)] == keys:
            return True
    return False


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

    schema = expand_local_references(graded.outcome_schema)
    final = copy.deepcopy(initial)
    built = True
    read = [*graded.required_paths, *(rule.path for rule in graded.rules)]

    def parent(pointer: str) -> bool:
        """Whether another read lies inside this pointer, so its value must hold children."""
        keys = split(pointer)
        return any(
            len(other) > len(keys) and other[: len(keys)] == keys for other in map(split, read)
        )

    for pointer in graded.required_paths:
        if pointer.startswith("/final_state/"):
            rest = pointer[len("/final_state") :]
            try:
                resolve(final, rest)
            except ValueError:
                if not parent(pointer):
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
        if rule.operator == "exists" and current is None and parent(rule.path):
            # A deeper read fills this value with an object; text here would leave it nothing.
            continue
        built &= _set(final, rest, _satisfying(rule, current, _schema_at(schema, rule.path)))
    record(
        "correct",
        workflow(final) if built else None,
        "a rule path is outside the final state or reads inside a text or number",
    )
    record("no_op", workflow(copy.deepcopy(initial)))
    wrong = copy.deepcopy(final)
    changed = False
    preserved = () if graded.preserve_state else graded.preserved_paths
    for rule in graded.rules:
        if graded.preserve_state or _protected(rule.path, preserved):
            continue
        rest = rule.path[len("/final_state") :]
        try:
            # From the correct witness, so a parent and its child read agree on one placeholder.
            current = resolve(final, rest)
        except ValueError:
            continue
        placeholder = _placeholder(current)
        if placeholder != current:
            changed |= _set(wrong, rest, placeholder)
    record(
        "wrong_content",
        workflow(wrong) if built and changed else None,
        "no rule reads a value outside the preserved paths that placeholder content can change",
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

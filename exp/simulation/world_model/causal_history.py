"""The investigated trajectory's own earlier steps, labelled by who authored each part."""

import json
from collections.abc import Mapping, Sequence

from exp.common.core.artifacts import JsonObject, JsonValue, strict_equal

CAUSAL_HISTORY_BUDGET = 48_000
"""Characters of serialized steps; whole steps beyond it are dropped, oldest first."""

NATIVE = "native-observation"
PREDICTED = "simulator-prediction"
AUTHORITY: JsonObject = {
    NATIVE: "the recorded reply of the native environment, disclosed for that step",
    PREDICTED: "the simulator's own predicted reply and committed state edits; a belief, not "
    "evidence of what the native environment did",
}


def causal_history(
    transitions: Sequence[JsonValue],
    operation_id: str,
    disclosed: Mapping[str, JsonValue],
    budget: int = CAUSAL_HISTORY_BUDGET,
) -> JsonObject:
    """Every committed step up to and including ``operation_id``, never a later one.

    Each step carries its public actions, the simulator's prediction as committed (replies and
    the state edits from its before to its after state) and, when disclosed, the recorded
    reply. Only this trajectory's transitions and disclosures are read; retrieval evidence and
    other traces' memory are left out, so the history adds no cross-trace provenance. Steps are
    kept whole: when they exceed ``budget``, the oldest are dropped and counted, and the
    investigated step is kept even alone over budget.

    Args:
        transitions: The session's committed transitions, in step order.
        operation_id: The investigated step.
        disclosed: Recorded replies disclosed for this trajectory's steps, by operation.
        budget: Character budget for the serialized steps.

    Returns:
        ``authority`` labels, ``omitted_earlier_steps``, an ``omitted`` marker when any were
        dropped, and the ordered ``steps``.
    """
    operations = [row.get("operation_id") if isinstance(row, dict) else None for row in transitions]
    if operation_id not in operations:
        raise ValueError("causal history requires a committed step of this trajectory")
    end = operations.index(operation_id)
    rows = [_step(index, transitions[index], operation_id, disclosed) for index in range(end + 1)]
    kept, used = [rows[-1]], len(_text(rows[-1]))
    for row in reversed(rows[:-1]):
        used += len(_text(row))
        if used > budget:
            break
        kept.append(row)
    omitted = len(rows) - len(kept)
    history: JsonObject = {"authority": AUTHORITY, "omitted_earlier_steps": omitted}
    if omitted:
        history["omitted"] = f"{omitted} earlier steps omitted"
    history["steps"] = list(reversed(kept))
    return history


def _step(
    index: int, transition: JsonValue, investigated: str, disclosed: Mapping[str, JsonValue]
) -> JsonObject:
    """One step's public actions, committed prediction and disclosed recorded reply."""
    if not isinstance(transition, dict):
        raise ValueError("causal history requires committed transitions")
    operation = str(transition["operation_id"])
    actions = transition.get("actions")
    replies = transition.get("replies")
    recorded: JsonObject = {"authority": NATIVE, "disclosed": operation in disclosed}
    if operation in disclosed:
        recorded["reply"] = disclosed[operation]
    return {
        "step": index,
        "operation_id": operation,
        "investigated": operation == investigated,
        "actions": [
            {"name": row.get("name"), "arguments": row.get("arguments", {})}
            for row in actions or ()
            if isinstance(row, dict)
        ]
        if isinstance(actions, list)
        else [],
        "simulator_prediction": {
            "authority": PREDICTED,
            "replies": replies if isinstance(replies, list) else [],
            "state_edits": _edits(transition.get("before", {}), transition.get("after", {}), []),
        },
        "recorded_reply": recorded,
    }


def _edits(before: JsonValue, after: JsonValue, path: list[JsonValue]) -> list[JsonValue]:
    """Generic delta ops (``set``/``delete``) that turn ``before`` into ``after``.

    Objects are compared key by key; any other changed value is set whole.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        edits: list[JsonValue] = []
        for key in sorted(before.keys() | after.keys()):
            if key not in after:
                edits.append({"op": "delete", "path": [*path, key]})
            elif key not in before:
                edits.append({"op": "set", "path": [*path, key], "value": after[key]})
            else:
                edits.extend(_edits(before[key], after[key], [*path, key]))
        return edits
    if strict_equal(before, after):
        return []
    return [{"op": "set", "path": path, "value": after}]


def _text(step: JsonObject) -> str:
    """A step's serialized form, as the investigator receives it."""
    return json.dumps(step, ensure_ascii=False, allow_nan=False)

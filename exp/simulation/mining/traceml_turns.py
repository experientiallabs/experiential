"""Number trajectory points by actor turn, outside the variance build's source identity.

The actor saves one checkpoint per assistant turn, after every call of the turn is observed, so
a step's fork point is its turn's index, never its tool-call position. This module lives apart
from :mod:`exp.simulation.mining.traceml` because that module is in the variance build's import
closure (``build_source_identity``): editing it re-keys every saved variance build.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.core.artifacts import JsonValue
from exp.simulation.mining.traceml import TraceTrajectory, trajectory_from_workflow_steps

POINT_NUMBERING = "actor-turn-v1"
"""How step boundaries number points: by actor turn, every call of one response sharing it.

Records that carry point numbers (TraceML labels, point bars, behavior differences) are keyed
by it, so records saved under the earlier call-position numbering are never replayed as turns.
"""


def workflow_step_turns(
    steps: Sequence[JsonValue], model_calls: Sequence[JsonValue]
) -> tuple[int, ...]:
    """Return the actor turn of each saved step, matched to the turn's call by ``call_id``.

    A continuation outcome records one ``model_calls`` receipt per actor turn and one ``steps``
    row per dispatched tool call, in call order. Calls of a length-stopped response are never
    dispatched and hold no step.

    Args:
        steps: Saved step mappings in order; anything else is skipped.
        model_calls: Saved actor call receipts in order, each with its ``response``.

    Returns:
        One zero-based turn index per step mapping.

    Raises:
        ValueError: A step's action does not match the next call its turns made, or a step is
            left over after every recorded call.
    """
    kept = [raw for raw in steps if isinstance(raw, dict)]
    turns: list[int] = []
    for turn, receipt in enumerate(model_calls):
        response = receipt.get("response") if isinstance(receipt, dict) else None
        if not isinstance(response, dict) or response.get("finish_reason") == "length":
            continue
        output = response.get("output")
        calls = output.get("tool_calls") if isinstance(output, dict) else None
        for call in calls if isinstance(calls, list) else ():
            if len(turns) == len(kept):
                break
            action = kept[len(turns)].get("action")
            action = action if isinstance(action, dict) else {}
            expected = (call.get("call_id"), call.get("name")) if isinstance(call, dict) else None
            if (action.get("call_id"), action.get("name")) != expected:
                raise ValueError(
                    f"step {len(turns) + 1} action {action.get('call_id')!r} is not call "
                    f"{expected!r} of actor turn {turn}; the steps and actor calls disagree"
                )
            turns.append(turn)
    if len(turns) != len(kept):
        raise ValueError(
            f"{len(kept) - len(turns)} of {len(kept)} steps match no recorded actor call; "
            "their turns, and so their fork points, are unknown"
        )
    return tuple(turns)


def trajectory_by_turns(
    trace_id: str,
    tier: str,
    steps: Sequence[JsonValue],
    *,
    turns: Sequence[int],
    task: str = "",
    locator_prefix: str = "step",
    initial_state: JsonValue = None,
    final_answer: JsonValue = None,
) -> TraceTrajectory:
    """Build a workflow trajectory whose step boundaries are actor turns.

    Each tool step is decided at ``turns[i]``; a final answer is decided in the turn after the
    last step's.

    Raises:
        ValueError: ``turns`` does not give one non-negative, non-decreasing turn per step.
    """
    kept = [raw for raw in steps if isinstance(raw, dict)]
    turns = tuple(turns)
    if len(turns) != len(kept):
        raise ValueError(f"trajectory {trace_id} has {len(kept)} steps but {len(turns)} turns")
    if any(turn < 0 for turn in turns) or list(turns) != sorted(turns):
        raise ValueError(f"trajectory {trace_id} step turns must be non-negative and in order")
    base = trajectory_from_workflow_steps(
        trace_id,
        tier,
        kept,
        task=task,
        locator_prefix=locator_prefix,
        initial_state=initial_state,
        final_answer=final_answer,
    )
    final = turns[-1] + 1 if turns else 0
    remapped = tuple(
        step.model_copy(update={"boundary": turns[i] if i < len(turns) else final})
        for i, step in enumerate(base.steps)
    )
    return base.model_copy(update={"steps": remapped})

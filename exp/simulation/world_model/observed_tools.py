"""Tool signatures observed in recorded calls, for traces that publish no tool contract.

Traces may declare the tools an agent saw (``exp.request.tools``). Many exports do not, and a
mined task then has an empty tool inventory, which a grounded session refuses for every supplied
action. This module is the single owner of the fallback: each observed tool name becomes a
permissive schema listing the argument names seen in recorded calls, with no inferred types,
requiredness, or behavior.
"""

from collections.abc import Iterable

from pydantic import JsonValue

from exp.common.tasks import TaskCase, ToolSchema
from exp.simulation.retrieval.contracts import RAGTransition

OBSERVED_TOOL_DESCRIPTION = "Observed signature only; requirements and behavior are unspecified."


def observed_tool_schemas(calls: Iterable[tuple[str, JsonValue]]) -> tuple[ToolSchema, ...]:
    """Merge recorded calls into one permissive schema per tool name.

    Args:
        calls: Recorded ``(tool name, arguments)`` pairs, in any order.

    Returns:
        One schema per distinct name, sorted by name. Properties are the union of observed
        argument names; additional properties stay allowed because unobserved arguments may
        exist.

    Raises:
        ValueError: A recorded call's arguments are not a JSON object.
    """
    arguments: dict[str, set[str]] = {}
    for name, values in calls:
        if not isinstance(values, dict):
            raise ValueError("Observed tool arguments must be JSON objects for tool simulation")
        arguments.setdefault(name, set()).update(values)
    return tuple(
        ToolSchema(
            name=name,
            description=OBSERVED_TOOL_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {key: {} for key in sorted(keys)},
                "additionalProperties": True,
            },
        )
        for name, keys in sorted(arguments.items())
    )


def observed_calls(transitions: Iterable[RAGTransition]) -> list[tuple[str, JsonValue]]:
    """Recorded tool calls of real transitions, skipping message transitions.

    Args:
        transitions: Observed transitions from a verified retrieval index.

    Returns:
        ``(tool name, arguments)`` pairs in transition order.
    """
    return [
        (transition.action.tool_name, transition.action.tool_arguments)
        for transition in transitions
        if transition.action.kind == "tool_call" and transition.action.tool_name is not None
    ]


def public_task_tools(
    task: TaskCase, calls: Iterable[tuple[str, JsonValue]]
) -> tuple[ToolSchema, ...]:
    """Return the task's published tools, or observed signatures when it declares none.

    Args:
        task: Mined or authored task whose declared inventory is authoritative when present.
        calls: Recorded calls visible to the caller, such as fit evidence plus the task's own
            public actions.

    Returns:
        The declared inventory unchanged, or the observed fallback for an empty one.
    """
    return task.tools or observed_tool_schemas(calls)

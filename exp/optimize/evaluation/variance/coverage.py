"""Which probe tasks declare tools the learned rule book has no contract for.

A rule book's tool contracts are learned from world-learning traces only. A mining-probe task can
declare a tool those traces never exercised, and then the world model has no learned interface,
behavior, or rule for it: any scenario family built from that task is simulated against a tool
the world never learned.

Policy: a probe task that uses a tool without a contract keeps its scenario families diagnostic.
Such families are reported, not trained or rewarded on. The schemas probe tasks declare are never
imported into the rule book as contracts, because that would let probe data shape the world model
and move evidence across the frozen exposure roles. Coverage is closed only by learning the tool
from world-learning evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.tasks.task import TaskCase
from exp.simulation.rules.model import RuleBook
from exp.simulation.rules.tool_names import ToolNames


def uncovered_tools(book: RuleBook, tasks: Sequence[TaskCase]) -> dict[str, tuple[str, ...]]:
    """Map each task to the declared tools that have no contract in the rule book.

    Only tool names are compared, by canonical key (:class:`ToolNames`), so a declared
    ``Write`` is covered by a ``write_file`` contract. A task's declared schema is never read
    into the book, so a declared tool stays uncovered until world learning yields its contract.

    Args:
        book: Rule book learned from world-learning evidence.
        tasks: Probe tasks whose declared tools are checked.

    Returns:
        Task identity to its uncovered tool names, in declaration order without repeats, for
        every task with at least one uncovered tool, in task order.

    Raises:
        ValueError: Two tasks share an identity.
    """
    seen: set[str] = set()
    known = ToolNames(book.contracts)
    uncovered: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"task {task.task_id!r} appears more than once")
        seen.add(task.task_id)
        names = known.uncovered(tool.name for tool in task.tools)
        if names:
            uncovered[task.task_id] = names
    return uncovered


def families_without_tool_rules(book: RuleBook, tasks: Sequence[TaskCase]) -> tuple[str, ...]:
    """Task identities whose scenario families must stay diagnostic for missing tool contracts.

    A task appears here when it declares any tool the rule book has no contract for. Under the
    module policy its families are diagnostic only, and its declared schemas are not imported
    into the rule book to cover the gap.

    Args:
        book: Rule book learned from world-learning evidence.
        tasks: Probe tasks whose declared tools are checked.

    Returns:
        Identities of tasks with at least one uncovered tool, in task order.

    Raises:
        ValueError: Two tasks share an identity.
    """
    return tuple(uncovered_tools(book, tasks))

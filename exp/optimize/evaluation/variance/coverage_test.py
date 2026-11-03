"""Probe tasks with tools the rule book never learned keep their families diagnostic."""

import pytest

from exp.common.tasks.task import TaskCase, ToolSchema
from exp.optimize.evaluation.variance.coverage import families_without_tool_rules, uncovered_tools
from exp.simulation.rules.model import RuleBook, ToolContract


def _tool(name: str) -> ToolSchema:
    """A declared tool with a schema a probe task could carry.

    Args:
        name: Tool name.

    Returns:
        The declared tool.
    """
    return ToolSchema(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    )


def _task(task_id: str, *tools: str) -> TaskCase:
    """A mined probe task declaring the named tools.

    Args:
        task_id: Task identity.
        *tools: Declared tool names.

    Returns:
        The task.
    """
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="fit",
        instruction="Resolve the customer's request.",
        tools=tuple(_tool(name) for name in tools),
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


BOOK = RuleBook(contracts={"lookup": ToolContract(name="lookup", calls=2)})


def test_tools_without_contracts_are_reported_per_task() -> None:
    """Only declared tools missing from the book are listed, in declaration order."""
    tasks = (
        _task("covered", "lookup"),
        _task("partial", "refund", "lookup", "escalate"),
        _task("bare"),
        _task("unknown", "cancel"),
    )
    assert uncovered_tools(BOOK, tasks) == {
        "partial": ("refund", "escalate"),
        "unknown": ("cancel",),
    }
    assert families_without_tool_rules(BOOK, tasks) == ("partial", "unknown")


def test_declared_schemas_are_never_imported_into_the_book() -> None:
    """Checking coverage leaves the rule book exactly as it was learned."""
    before = BOOK.model_copy(deep=True)
    tasks = (_task("probe", "refund"),)
    assert uncovered_tools(BOOK, tasks) == {"probe": ("refund",)}
    assert BOOK == before
    assert "refund" not in BOOK.contracts
    assert uncovered_tools(BOOK, tasks) == {"probe": ("refund",)}


def test_an_empty_book_leaves_every_tool_task_diagnostic() -> None:
    """With no learned contracts every task that declares a tool is diagnostic."""
    tasks = (_task("one", "lookup"), _task("two"))
    assert families_without_tool_rules(RuleBook(contracts={}), tasks) == ("one",)
    assert families_without_tool_rules(BOOK, ()) == ()


def test_a_repeated_task_identity_is_refused() -> None:
    """Two tasks under one identity cannot be told apart, so the check refuses them."""
    with pytest.raises(ValueError, match="appears more than once"):
        uncovered_tools(BOOK, (_task("same", "refund"), _task("same", "lookup")))

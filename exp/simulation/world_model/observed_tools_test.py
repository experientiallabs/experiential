"""Observed tool signatures: one owner for tasks whose traces publish no tool contract."""

import pytest

from exp.common.tasks import TaskCase, ToolSchema
from exp.simulation.world_model.observed_tools import observed_tool_schemas, public_task_tools


def task(tools: tuple[ToolSchema, ...] = ()) -> TaskCase:
    """A mined task with optional declared tools."""
    return TaskCase(
        task_id="task",
        lineage_group_id="lineage",
        partition="held_out",
        instruction="Return my order",
        tools=tools,
        workload_weight=1,
        source_trace_ids=("trace",),
    )


def test_signatures_merge_argument_names_per_tool_in_name_order() -> None:
    """Every observed argument name is a permissive property; nothing is inferred."""
    tools = observed_tool_schemas(
        [
            ("get_order", {"order_id": "1"}),
            ("find_user", {}),
            ("get_order", {"order_id": "2", "verbose": True}),
        ]
    )
    assert [tool.name for tool in tools] == ["find_user", "get_order"]
    assert tools[1].input_schema == {
        "type": "object",
        "properties": {"order_id": {}, "verbose": {}},
        "additionalProperties": True,
    }
    assert tools[0].description == (
        "Observed signature only; requirements and behavior are unspecified."
    )


def test_non_object_arguments_are_refused() -> None:
    """A call whose arguments are not a JSON object cannot become a tool signature."""
    with pytest.raises(ValueError, match="JSON objects"):
        observed_tool_schemas([("get_order", "order 1")])


def test_published_contract_wins_and_observation_fills_its_absence() -> None:
    """Declared trace tools are kept as published; an empty inventory gets observed ones."""
    declared = (ToolSchema(name="get_order", description="Published", input_schema={}),)
    calls = [("get_order", {"order_id": "1"})]
    assert public_task_tools(task(declared), calls) == declared
    assert [tool.name for tool in public_task_tools(task(), calls)] == ["get_order"]

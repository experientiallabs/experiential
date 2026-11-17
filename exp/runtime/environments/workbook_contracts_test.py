"""Workbook contracts catch the observed header type corruption and readback mismatch."""

import io
import json

import pytest
from openpyxl import Workbook

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.workbook_contracts import simulated_workbook_manifest
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter, workbook_projection
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.environments.world_responses import validate_reply_state, validate_state_projection


def test_semantic_manifest_refuses_string_header_with_numeric_type() -> None:
    """The actual failed prediction cannot enter simulated state under the declared contract."""
    manifest = simulated_workbook_manifest(
        WorldManifest(
            domains=(StateDomain(name="input.xlsx", owner="native"),),
            tools=(ToolEffectPolicy(name="inspect_workbook", reads=("input.xlsx",)),),
        )
    )
    schema = manifest.domains[0].state_schema
    assert schema is not None
    header: JsonObject = {"value": "key", "data_type": "s", "number_format": "General"}
    state: JsonObject = {"input.xlsx": {"sheets": [{"name": "Data", "rows": [[header]]}]}}
    validate_state_projection(state, {"input.xlsx": schema})
    header["data_type"] = "n"
    with pytest.raises(ValueError, match="state.*schema"):
        validate_state_projection(state, {"input.xlsx": schema})
    assert len(manifest.tool("inspect_workbook").reply_bindings) == 6


@pytest.mark.parametrize("tool", ["describe_workbook", "inspect_workbook"])
def test_empty_sheet_reads_match_the_native_virtual_blank_cell(tool: str) -> None:
    """An observed empty sheet has native dimensions 1x1, not missing or guessed state."""
    workbook = Workbook()
    buffer = io.BytesIO()
    workbook.save(buffer)
    state: JsonObject = {"input.xlsx": workbook_projection(workbook)}
    arguments: JsonObject = {"source": "input.xlsx"}
    if tool == "inspect_workbook":
        arguments["sheet"] = "Sheet"
    call = ToolCall(name=tool, call_id="empty", arguments=arguments)
    actual = WorkbookEditAdapter().apply({"input.xlsx": buffer.getvalue()}, (call,), ())
    policy = simulated_workbook_manifest(
        WorldManifest(
            domains=(StateDomain(name="input.xlsx", owner="native"),),
            tools=(ToolEffectPolicy(name=tool, reads=("input.xlsx",)),),
        )
    ).tool(tool)
    validate_reply_state(
        Observation(content=json.dumps({"projection": actual.projection, "written": {}})),
        policy.reply_bindings,
        state,
        state,
        action=call,
        domains=policy.domains(),
    )
    assert state == {"input.xlsx": {"sheets": [{"name": "Sheet", "rows": []}]}}


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("describe_workbook", {}),
        ("describe_workbook", {"offset": 0, "limit": 1}),
        ("describe_workbook", {"offset": 1, "limit": 1}),
        ("describe_workbook", {"offset": 9, "limit": 1}),
        ("inspect_workbook", {"sheet": "Data", "row_count": 2}),
    ],
)
def test_bindings_match_real_adapter_reads_including_page_boundaries(
    tool: str,
    arguments: JsonObject,
) -> None:
    """Native execution independently supplies the pagination and range expected behavior."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    sheet.append(["key", "payload"])
    sheet.append([1, 2])
    workbook.create_sheet("Protected").append([42])
    buffer = io.BytesIO()
    workbook.save(buffer)
    state: JsonObject = {"input.xlsx": workbook_projection(workbook)}
    call = ToolCall(name=tool, call_id="c", arguments={"source": "input.xlsx", **arguments})
    actual = WorkbookEditAdapter().apply({"input.xlsx": buffer.getvalue()}, (call,), ())
    reply = {"projection": actual.projection, "written": {}}
    policy = simulated_workbook_manifest(
        WorldManifest(
            domains=(StateDomain(name="input.xlsx", owner="native"),),
            tools=(ToolEffectPolicy(name=tool, reads=("input.xlsx",)),),
        )
    ).tool(tool)
    validate_reply_state(
        Observation(content=json.dumps(reply)),
        policy.reply_bindings,
        state,
        state,
        action=call,
        domains=policy.domains(),
    )
    if tool == "describe_workbook":
        projection = actual.projection["input.xlsx"]
        assert isinstance(projection, dict)
        del projection["next_offset"]
        with pytest.raises(ValueError, match="unavailable"):
            validate_reply_state(
                Observation(content=json.dumps(reply)),
                policy.reply_bindings,
                state,
                state,
                action=call,
                domains=policy.domains(),
            )

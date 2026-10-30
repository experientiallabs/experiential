"""Declared semantic workbook contracts for a simulated owner, without native execution.

These contracts describe the bounded workbook adapter's representation and read behavior.
They establish internal consistency, not held-out fidelity or general spreadsheet support.
"""

from exp.common.core.artifacts import JsonObject
from exp.runtime.environments.world_contracts import WorldManifest
from exp.runtime.environments.world_responses import ReplyStateBinding
from exp.runtime.environments.world_state_paths import (
    ArgumentKey,
    ArgumentValue,
    ArrayLength,
    EmptyArrayDefault,
    PageCursor,
    ProjectItems,
    SelectItem,
    SliceItems,
)


def workbook_state_schema() -> JsonObject:
    """Require typed cells, including the dependency between cell type and actual value."""
    cell: JsonObject = {
        "type": "object",
        "required": ["value", "data_type", "number_format"],
        "additionalProperties": False,
        "properties": {
            "value": {},
            "data_type": {"type": "string"},
            "number_format": {"type": "string"},
        },
        "anyOf": [
            {
                "properties": {
                    "data_type": {"enum": ["s", "inlineStr", "str", "e", "d"]},
                    "value": {"type": ["string", "null"]},
                }
            },
            {"properties": {"data_type": {"const": "n"}, "value": {"type": ["number", "null"]}}},
            {"properties": {"data_type": {"const": "b"}, "value": {"type": "boolean"}}},
        ],
    }
    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["sheets"],
        "properties": {
            "sheets": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "rows"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rows": {"type": "array", "items": {"type": "array", "items": cell}},
                    },
                },
            }
        },
    }


def simulated_workbook_manifest(manifest: WorldManifest) -> WorldManifest:
    """Bind semantic workbook projections and current-state readback to simulated domains.

    Callers supply full ``workbook_projection`` values per file and null for absent outputs.
    This explicit conversion is for the bounded XLSX adapter only. Its private native file
    digest projection has a different schema and must not be used as semantic state.
    """
    source = ArgumentKey(argument="source")
    sheet = (source, "sheets", SelectItem(select="name", argument="sheet"))
    cells = EmptyArrayDefault(
        empty_array_default=([{"value": None, "data_type": "n", "number_format": "General"}],)
    )
    reply = ("projection", source)
    bindings = (
        ReplyStateBinding(
            reply_path=(*reply, "start_row"),
            state_path=(source, ArgumentValue(argument_value="start_row", default=1)),
        ),
        ReplyStateBinding(
            reply_path=(*reply, "start_column"),
            state_path=(source, ArgumentValue(argument_value="start_column", default=1)),
        ),
        ReplyStateBinding(reply_path=(*reply, "sheet"), state_path=(*sheet, "name")),
        ReplyStateBinding(
            reply_path=(*reply, "max_row"),
            state_path=(*sheet, "rows", cells, ArrayLength(length=True)),
        ),
        ReplyStateBinding(
            reply_path=(*reply, "max_column"),
            state_path=(*sheet, "rows", cells, 0, ArrayLength(length=True)),
        ),
        ReplyStateBinding(
            reply_path=(*reply, "rows"),
            state_path=(
                *sheet,
                "rows",
                cells,
                SliceItems(
                    slice_start="start_row",
                    slice_count="row_count",
                    default_start=1,
                    default_count=1,
                    offset=1,
                ),
                SliceItems(
                    slice_start="start_column",
                    slice_count="column_count",
                    default_start=1,
                    default_count=1,
                    offset=1,
                    each=True,
                ),
            ),
        ),
    )
    rows = []
    for tool in manifest.tools:
        if tool.name == "inspect_workbook":
            tool = tool.model_copy(
                update={
                    "reply_bindings": bindings,
                    "reply_schema": {
                        "type": "object",
                        "required": ["projection", "written"],
                        "properties": {"projection": {"type": "object"}, "written": {"const": {}}},
                        "additionalProperties": False,
                    },
                }
            )
        elif tool.name == "describe_workbook":
            page = SliceItems(
                slice_start="offset",
                slice_count="limit",
                default_start=0,
                default_count=32,
                clamp=True,
            )
            tool = tool.model_copy(
                update={
                    "reply_bindings": (
                        ReplyStateBinding(
                            reply_path=(*reply, "offset"),
                            state_path=(source, ArgumentValue(argument_value="offset", default=0)),
                        ),
                        ReplyStateBinding(
                            reply_path=(*reply, "next_offset"),
                            state_path=(source, "sheets", PageCursor(page_cursor=page)),
                        ),
                        ReplyStateBinding(
                            reply_path=(*reply, "total_sheets"),
                            state_path=(source, "sheets", ArrayLength(length=True)),
                        ),
                        ReplyStateBinding(
                            reply_path=(*reply, "sheets"),
                            state_path=(
                                source,
                                "sheets",
                                SliceItems(
                                    slice_start="offset",
                                    slice_count="limit",
                                    default_start=0,
                                    default_count=32,
                                    clamp=True,
                                ),
                                ProjectItems(
                                    fields={
                                        "name": ("name",),
                                        "max_row": ("rows", cells, ArrayLength(length=True)),
                                        "max_column": ("rows", cells, 0, ArrayLength(length=True)),
                                    }
                                ),
                            ),
                        ),
                    ),
                    "reply_schema": {
                        "type": "object",
                        "required": ["projection", "written"],
                        "properties": {"projection": {"type": "object"}, "written": {"const": {}}},
                        "additionalProperties": False,
                    },
                }
            )
        rows.append(tool)
    return WorldManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "domains": [
                {
                    **row.model_dump(mode="json"),
                    "owner": "simulated",
                    "state_schema": workbook_state_schema(),
                }
                for row in manifest.domains
            ],
            "tools": [row.model_dump(mode="json") for row in rows],
        }
    )

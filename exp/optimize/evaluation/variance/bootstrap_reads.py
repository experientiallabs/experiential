"""Explicit, complete step selections over already admitted bootstrap trace views."""

from collections.abc import Mapping

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.optimize.evaluation.rules.agent_loop import ToolStatus, bounded, tool_schema

BOOTSTRAP_READ_PROTOCOL = "bootstrap-step-reads-v1"

STEP_READ_TOOL = tool_schema(
    "read_trace_steps",
    "Read an explicit complete range of native trace steps. start_index is a zero-based "
    "position in the trace's steps list; count is exact, never silently clamped. "
    "Use returned total_steps and next_start_index to select another range. "
    "Only returned steps and their native citations are delivered; this is not a whole trace.",
    {
        "trace_id": {"type": "string", "minLength": 1},
        "start_index": {"type": "integer", "minimum": 0},
        "count": {"type": "integer", "minimum": 1, "maximum": 16},
    },
    ("trace_id", "start_index", "count"),
)


def read_whole_trace(
    views: Mapping[str, tuple[JsonObject, frozenset[str]]],
    arguments: JsonObject,
    maximum_bytes: int,
) -> tuple[ToolStatus, JsonObject | str, tuple[str, ...]]:
    """Preserve complete-trace reads; extra range arguments never select a subset."""
    trace_id = str(arguments.get("trace_id", ""))
    if trace_id not in views:
        return "refused", f"no readable trace {trace_id!r}", ()
    view, locators = views[trace_id]
    status, result = bounded(view, maximum_bytes)
    assert isinstance(result, (dict, str))
    return status, result, tuple(sorted(locators)) if status == "ok" else ()


class TraceStepSelection(ContractModel):
    """An exact caller-selected range, with no implicit truncation or aliases.

    Attributes:
        trace_id: Exact admitted trace identity.
        start_index: Zero-based native step-list position.
        count: Exact number of complete steps to return.
    """

    trace_id: str = Field(min_length=1, strict=True)
    start_index: int = Field(ge=0, strict=True)
    count: int = Field(ge=1, le=16, strict=True)


def read_trace_steps(
    views: Mapping[str, tuple[JsonObject, frozenset[str]]],
    arguments: JsonObject,
    maximum_bytes: int,
) -> tuple[ToolStatus, JsonObject | str, tuple[str, ...]]:
    """Return exactly the requested complete rows of an already admitted native view.

    Native view metadata supplies citations; quoted data never supplies citation authority.
    The frozen result allowance is applied here; the loop still checks complete-request bytes.
    """
    try:
        selection = TraceStepSelection.model_validate(arguments)
    except ValueError:
        return "refused", "use exact trace_id, integer start_index and positive integer count", ()
    if selection.trace_id not in views:
        return "refused", "trace is outside the admitted bootstrap material", ()
    view, inventory = views[selection.trace_id]
    rows = view["steps"]
    assert isinstance(rows, list)
    end = selection.start_index + selection.count
    if end > len(rows):
        return (
            "refused",
            f"exact range exceeds total_steps={len(rows)}; select an in-range count",
            (),
        )
    selected = rows[selection.start_index : end]
    refs: list[str] = []
    for row in selected:
        assert isinstance(row, dict)
        for key in ("locator", "observation_locator"):
            reference = row.get(key)
            if isinstance(reference, str):
                assert reference in inventory
                refs.append(reference)
    citations = tuple(dict.fromkeys(refs))
    value: JsonObject = {
        "protocol": BOOTSTRAP_READ_PROTOCOL,
        "trace_id": selection.trace_id,
        "coverage": {
            "unit": "zero-based-step-index",
            "start_index": selection.start_index,
            "end_index_exclusive": end,
            "total_steps": len(rows),
            "selected_range_complete": True,
            "all_steps_selected": selection.start_index == 0 and end == len(rows),
            "next_start_index": end if end < len(rows) else None,
        },
        "steps": selected,
        "citations": list(citations),
    }
    status, result = bounded(value, maximum_bytes)
    assert isinstance(result, (dict, str))
    return status, result, citations if status == "ok" else ()

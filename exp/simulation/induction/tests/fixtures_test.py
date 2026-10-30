"""Shared fixtures: a native SQLite record world, a workbook-like tool and scripted models.

The record world is the real :class:`RecordWorldFactory`; its native replies and table
projections are the recorded evidence. The workbook-like tool is an independent reference
implementation over the bounded workbook state schema. Programs below are what a scripted model
"proposes"; they are written separately from the references they must reproduce.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from exp.common.core.artifacts import JsonObject, JsonValue, sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.record_world import RecordTool, RecordToolError, RecordWorldFactory
from exp.runtime.environments.workbook_contracts import workbook_state_schema
from exp.runtime.environments.world_contracts import (
    StateDomain,
    ToolEffectPolicy,
    WorldAccess,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext
from exp.simulation.induction.contracts import InductionEpisode, InductionStep, TransitionProgram
from exp.simulation.induction.proposals import SUBMIT_TOOL
from exp.simulation.world_model.executable import GroundedWorldFactory, PredictorEvent
from exp.simulation.world_model.grounded_session import GroundedTransitionEngine
from exp.simulation.world_model.runtime_test import _runtime

RECORD_SCHEMA = """
CREATE TABLE tickets (id INTEGER PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL);
INSERT INTO tickets VALUES (1, 'seed', 'open');
"""
RECORD_TOOLS = ("open_ticket", "get_ticket", "close_ticket", "count_open", "add_note")


def _tickets(arguments: JsonObject) -> WorldAccess:
    """Every ticket tool reads and writes the tickets table."""
    del arguments
    return WorldAccess(reads=frozenset({"tickets"}), writes=frozenset({"tickets"}))


def _read_tickets(arguments: JsonObject) -> WorldAccess:
    """Read-only ticket tools."""
    del arguments
    return WorldAccess(reads=frozenset({"tickets"}))


def _notes(arguments: JsonObject) -> WorldAccess:
    """The note tool only writes notes."""
    del arguments
    return WorldAccess(reads=frozenset({"notes"}), writes=frozenset({"notes"}))


def _open(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Insert one open ticket."""
    del context
    cursor = db.execute(
        "INSERT INTO tickets (title, status) VALUES (?, 'open')", (arguments["title"],)
    )
    return {"id": cursor.lastrowid, "status": "open"}


def _get(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Read one ticket or fail as a business error."""
    del context
    row = db.execute(
        "SELECT id, title, status FROM tickets WHERE id = ?", (arguments["id"],)
    ).fetchone()
    if row is None:
        raise RecordToolError(f"ticket {arguments['id']} not found")
    return {"id": row[0], "title": row[1], "status": row[2]}


def _close(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Close one open ticket."""
    del context
    row = db.execute("SELECT status FROM tickets WHERE id = ?", (arguments["id"],)).fetchone()
    if row is None:
        raise RecordToolError(f"ticket {arguments['id']} not found")
    if row[0] == "closed":
        raise RecordToolError(f"ticket {arguments['id']} is already closed")
    db.execute("UPDATE tickets SET status = 'closed' WHERE id = ?", (arguments["id"],))
    return {"id": arguments["id"], "status": "closed"}


def _count(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Count open tickets."""
    del arguments, context
    return {"open": db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'").fetchone()[0]}


def _note(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Append one note."""
    del context
    db.execute("INSERT INTO notes (text) VALUES (?)", (arguments["text"],))
    return "noted"


def record_factory() -> RecordWorldFactory:
    """Return the native ticket service."""
    return RecordWorldFactory(
        service={"name": "tickets", "version": 1},
        schema=RECORD_SCHEMA + "CREATE TABLE notes (text TEXT NOT NULL);",
        tools={
            "open_ticket": RecordTool(access=_tickets, run=_open),
            "get_ticket": RecordTool(access=_read_tickets, run=_get),
            "close_ticket": RecordTool(access=_tickets, run=_close),
            "count_open": RecordTool(access=_read_tickets, run=_count),
            "add_note": RecordTool(access=_notes, run=_note),
        },
    )


def record_manifest(owner: str = "simulated") -> WorldManifest:
    """Return the ticket world's domains and effects under one owner."""
    owner_literal = "native" if owner == "native" else "simulated"
    return WorldManifest(
        domains=tuple(StateDomain(name=name, owner=owner_literal) for name in ("notes", "tickets")),
        tools=(
            ToolEffectPolicy(name="open_ticket", reads=("tickets",), writes=("tickets",)),
            ToolEffectPolicy(name="get_ticket", reads=("tickets",)),
            ToolEffectPolicy(name="close_ticket", reads=("tickets",), writes=("tickets",)),
            ToolEffectPolicy(name="count_open", reads=("tickets",)),
            ToolEffectPolicy(name="add_note", reads=("notes",), writes=("notes",)),
        ),
    )


def task(names: Iterable[str], task_id: str = "induction") -> TaskCase:
    """Return a public authored task exposing the named tools."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=task_id,
        partition="fit",
        instruction="Operate the environment.",
        workload_weight=1,
        origin="authored",
        tools=tuple(
            ToolSchema(
                name=name, description=f"Fixture tool {name}.", input_schema={"type": "object"}
            )
            for name in names
        ),
    )


def decoded(content: str) -> JsonValue:
    """Decode a recorded reply from JSON text when it is JSON, as fidelity replay does."""
    try:
        return json.loads(content)
    except ValueError:
        return content


def record_episode(
    episode_id: str,
    calls: Sequence[tuple[str, Mapping[str, JsonValue]]],
    *,
    with_state: bool = True,
) -> InductionEpisode:
    """Run calls against the native record world and record replies and table state."""
    factory = record_factory()
    runtime = ExecutableWorldRuntime(record_manifest("native"), factory)
    steps = []
    with runtime.open(task(RECORD_TOOLS)) as world:
        initial = factory.restore(world.checkpoint().image).state_projection()
        for index, (name, arguments) in enumerate(calls):
            observation = world.execute(
                ToolCall(name=name, call_id=f"{episode_id}-{index}", arguments=dict(arguments))
            )
            after = factory.restore(world.checkpoint().image).state_projection()
            steps.append(
                InductionStep(
                    tool=name,
                    arguments=dict(arguments),
                    reply=decoded(observation.content),
                    is_error=observation.is_error,
                    state_after=after if with_state else None,
                )
            )
    return InductionEpisode(
        episode_id=episode_id, initial_state=initial if with_state else {}, steps=tuple(steps)
    )


RECORD_FIT_CALLS: tuple[tuple[str, JsonObject], ...] = (
    ("count_open", {}),
    ("open_ticket", {"title": "printer"}),
    ("get_ticket", {"id": 2}),
    ("close_ticket", {"id": 1}),
    ("get_ticket", {"id": 1}),
    ("close_ticket", {"id": 9}),
    ("count_open", {}),
    ("get_ticket", {"id": 7}),
    ("add_note", {"text": "called back"}),
    ("open_ticket", {"title": "wifi"}),
)

RECORD_PROGRAMS = {
    "open_ticket": """
import json


def transition(state, args, facts):
    table = state["tickets"]
    ids = [row[0] for row in table["rows"]]
    ticket = (max(ids) if ids else 0) + 1
    table["rows"].append([ticket, args["title"], "open"])
    table["rows"].sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {"id": ticket, "status": "open"}
""",
    "get_ticket": """
def transition(state, args, facts):
    for row in state["tickets"]["rows"]:
        if row[0] == args["id"]:
            return {"id": row[0], "title": row[1], "status": row[2]}
    raise ToolError("ticket " + str(args["id"]) + " not found")
""",
    "close_ticket": """
def transition(state, args, facts):
    for row in state["tickets"]["rows"]:
        if row[0] == args["id"]:
            if row[2] == "closed":
                raise ToolError("ticket " + str(args["id"]) + " is already closed")
            row[2] = "closed"
            return {"id": args["id"], "status": "closed"}
    raise ToolError("ticket " + str(args["id"]) + " not found")
""",
    "count_open": """
def transition(state, args, facts):
    return {"open": len([row for row in state["tickets"]["rows"] if row[2] == "open"])}
""",
}

WORKBOOK = "book.xlsx"
WORKBOOK_TOOLS = ("read_cell", "write_cell", "sum_column")


def workbook_manifest() -> WorldManifest:
    """Return one simulated workbook domain under the bounded workbook state schema."""
    return WorldManifest(
        domains=(
            StateDomain(name=WORKBOOK, owner="simulated", state_schema=workbook_state_schema()),
        ),
        tools=(
            ToolEffectPolicy(name="read_cell", reads=(WORKBOOK,)),
            ToolEffectPolicy(name="write_cell", reads=(WORKBOOK,), writes=(WORKBOOK,)),
            ToolEffectPolicy(name="sum_column", reads=(WORKBOOK,)),
        ),
    )


def _cell(value: JsonValue) -> JsonObject:
    """Return one typed workbook cell."""
    kind = "n" if isinstance(value, (int, float)) and not isinstance(value, bool) else "s"
    return {"value": value, "data_type": kind, "number_format": "General"}


def workbook_state() -> JsonObject:
    """Return the starting workbook: a header and two numeric rows."""
    return {
        WORKBOOK: {
            "sheets": [
                {
                    "name": "Sheet1",
                    "rows": [
                        [_cell("item"), _cell("amount")],
                        [_cell("apples"), _cell(3)],
                        [_cell("pears"), _cell(4)],
                    ],
                }
            ]
        }
    }


def _address(cell: str) -> tuple[int, int]:
    """Return zero-based (row, column) of an A1 address with a single-letter column."""
    return int(cell[1:]) - 1, ord(cell[0]) - ord("A")


def workbook_reference(
    state: JsonObject, name: str, arguments: JsonObject
) -> tuple[JsonValue, bool]:
    """Apply one call to the reference workbook in place; return (reply, is_error)."""
    book = cast(dict[str, list[JsonObject]], state[WORKBOOK])
    sheet = next((row for row in book["sheets"] if row["name"] == arguments.get("sheet")), None)
    if sheet is None:
        return {"error": f"no sheet {arguments.get('sheet')}"}, True
    rows = cast(list[list[JsonObject]], sheet["rows"])
    if name == "sum_column":
        column = ord(str(arguments["column"])) - ord("A")
        total = sum(
            cast(int, row[column]["value"])
            for row in rows
            if column < len(row) and row[column]["data_type"] == "n"
        )
        return {"sum": total}, False
    row_index, column = _address(str(arguments["cell"]))
    if name == "read_cell":
        if row_index >= len(rows) or column >= len(rows[row_index]):
            return {"cell": arguments["cell"], "value": None}, False
        return {"cell": arguments["cell"], "value": rows[row_index][column]["value"]}, False
    while len(rows) <= row_index:
        rows.append([])
    while len(rows[row_index]) <= column:
        rows[row_index].append({"value": None, "data_type": "n", "number_format": "General"})
    rows[row_index][column] = _cell(arguments["value"])
    return {"written": arguments["cell"]}, False


def workbook_episode(
    episode_id: str,
    calls: Sequence[tuple[str, Mapping[str, JsonValue]]],
    *,
    with_state: bool = True,
) -> InductionEpisode:
    """Run calls against the reference workbook and record replies and state."""
    state = workbook_state()
    initial = json.loads(json.dumps(state))
    steps = []
    for name, arguments in calls:
        reply, is_error = workbook_reference(state, name, dict(arguments))
        steps.append(
            InductionStep(
                tool=name,
                arguments=dict(arguments),
                reply=reply,
                is_error=is_error,
                state_after=json.loads(json.dumps(state)) if with_state else None,
            )
        )
    return InductionEpisode(
        episode_id=episode_id, initial_state=initial if with_state else {}, steps=tuple(steps)
    )


WORKBOOK_FIT_CALLS: tuple[tuple[str, JsonObject], ...] = (
    ("read_cell", {"sheet": "Sheet1", "cell": "B2"}),
    ("sum_column", {"sheet": "Sheet1", "column": "B"}),
    ("write_cell", {"sheet": "Sheet1", "cell": "B4", "value": 5}),
    ("read_cell", {"sheet": "Sheet1", "cell": "A4"}),
    ("write_cell", {"sheet": "Sheet1", "cell": "A4", "value": "plums"}),
    ("sum_column", {"sheet": "Sheet1", "column": "B"}),
    ("read_cell", {"sheet": "Sheet1", "cell": "C9"}),
)

WORKBOOK_PROGRAMS = {
    "read_cell": """
def sheet_rows(state, name):
    for sheet in state["book.xlsx"]["sheets"]:
        if sheet["name"] == name:
            return sheet["rows"]
    raise OutOfDomain("unknown sheet")


def transition(state, args, facts):
    rows = sheet_rows(state, args["sheet"])
    row = int(args["cell"][1:]) - 1
    column = ord(args["cell"][0]) - ord("A")
    if row < len(rows) and column < len(rows[row]):
        return {"cell": args["cell"], "value": rows[row][column]["value"]}
    return {"cell": args["cell"], "value": None}
""",
    "write_cell": """
def transition(state, args, facts):
    rows = None
    for sheet in state["book.xlsx"]["sheets"]:
        if sheet["name"] == args["sheet"]:
            rows = sheet["rows"]
    if rows is None:
        raise OutOfDomain("unknown sheet")
    row = int(args["cell"][1:]) - 1
    column = ord(args["cell"][0]) - ord("A")
    while len(rows) <= row:
        rows.append([])
    while len(rows[row]) <= column:
        rows[row].append({"value": None, "data_type": "n", "number_format": "General"})
    value = args["value"]
    kind = "n" if isinstance(value, (int, float)) and not isinstance(value, bool) else "s"
    rows[row][column] = {"value": value, "data_type": kind, "number_format": "General"}
    return {"written": args["cell"]}
""",
    "sum_column": """
def transition(state, args, facts):
    column = ord(args["column"]) - ord("A")
    for sheet in state["book.xlsx"]["sheets"]:
        if sheet["name"] == args["sheet"]:
            return {"sum": sum(r[column]["value"] for r in sheet["rows"]
                               if column < len(r) and r[column]["data_type"] == "n")}
    raise OutOfDomain("unknown sheet")
""",
}


def programs(sources: Mapping[str, str]) -> tuple[TransitionProgram, ...]:
    """Return programs for the given tool sources."""
    return tuple(TransitionProgram(tool=tool, source=source) for tool, source in sources.items())


SNAPSHOT = ModelSnapshot(
    billing_source=BillingSource.CUSTOMER_MANAGED,
    provider="fixture",
    model_id="inducer",
    capabilities_sha256=ModelCapabilities().identity_sha256(),
    connection_sha256=sha256_json({"connection": "inducer"}),
)


class ScriptedInducer:
    """A scripted model that answers each tool's attempts with registered sources in order.

    Attributes:
        sources: Per tool, the source (or None for a non-program reply) of each attempt.
        requests: Every request received, with its operation identity.
    """

    def __init__(self, sources: Mapping[str, Sequence[str | None]]) -> None:
        """Register the replies."""
        self.sources = {tool: list(rows) for tool, rows in sources.items()}
        self.requests: list[tuple[str, ModelRequest]] = []

    def __call__(self, request: ModelRequest, operation: str) -> ModelResponse:
        """Answer one request for the tool it names."""
        self.requests.append((operation, request))
        tool = json.loads(str(request.messages[1].content))["tool"]
        source = self.sources[tool].pop(0)
        output = (
            AssistantAction(content="I cannot model this tool.")
            if source is None
            else AssistantAction(
                tool_calls=(ToolCall(name=SUBMIT_TOOL, call_id="p", arguments={"source": source}),)
            )
        )
        return ModelResponse(output=output, model=SNAPSHOT, economics=OperationEconomics())


def grounded_fallback(
    manifest: WorldManifest,
    public_task: TaskCase,
    initial_state: JsonObject,
    proposals: list[tuple[str, JsonValue, list[JsonObject]]],
) -> tuple[GroundedWorldFactory, list[PredictorEvent], list[JsonObject]]:
    """Return the real grounded predictor with scripted (call_id, reply, edits) proposals."""
    predictor, _, _ = _runtime("", capabilities=ModelCapabilities(supports_tools=True))
    engine = GroundedTransitionEngine(
        predictor, public_task, maximum_output_tokens=1024, output_protocol="json-tool"
    )
    events: list[PredictorEvent] = []
    requests: list[JsonObject] = []

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Return the next registered proposal for the single call in the request."""
        del operation
        requests.append(body)
        call_id, reply, edits = proposals.pop(0)
        return ModelResponse(
            model=predictor.artifact.model,
            economics=OperationEconomics(),
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        name="submit_transition",
                        call_id="proposal",
                        arguments={
                            "observations": [
                                {
                                    "call_id": call_id,
                                    "reply": reply,
                                    "encoding": "text" if isinstance(reply, str) else "json",
                                }
                            ],
                            "effect": {"kind": "edits", "edits": edits},
                        },
                    ),
                )
            ),
        ).model_dump(mode="json")

    factory = GroundedWorldFactory(
        engine, manifest, initial_state, complete=complete, record=events.append
    )
    return factory, events, requests

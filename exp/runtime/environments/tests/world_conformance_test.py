"""One restoration conformance suite, run identically on every executable-world backend.

Two backends pass it through the same interface: native XLSX artifacts edited by the workbook
adapter (the SpreadsheetBench shape), and a stateful SQLite ticket service with mutable
records, a sandboxed outbox, scheduled escalations and an outage window.
"""

import io
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from openpyxl import Workbook

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.record_world import RecordTool, RecordToolError, RecordWorldFactory
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter
from exp.runtime.environments.world_contracts import (
    AuthorityError,
    RestoreError,
    StateDomain,
    ToolEffectPolicy,
    UndeclaredEffectError,
    WorldAccess,
    WorldEvent,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext


@dataclass(frozen=True)
class WorldCase:
    """One backend bound to the calls the conformance suite needs.

    Attributes:
        runtime: Restorable runtime under test.
        task: Public task exposing the backend's tools.
        first: A state-changing call.
        second: A different state-changing call admitted after ``first``.
        outside_policy: A call whose resolved access exceeds its tool's declared policy.
        outside_authority: A call that touches a simulated-owned domain.
        branched: A native domain that ``first`` writes.
    """

    runtime: ExecutableWorldRuntime
    task: TaskCase
    first: ToolCall
    second: ToolCall
    outside_policy: ToolCall
    outside_authority: ToolCall
    branched: str


def _call(name: str, call_id: str, **arguments: JsonValue) -> ToolCall:
    """Build one tool call."""
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _task(*names: str) -> TaskCase:
    """Return a public task exposing exactly the named tools."""
    return TaskCase(
        task_id="world-conformance",
        lineage_group_id="world-conformance",
        partition="fit",
        instruction="Operate the environment.",
        workload_weight=1,
        origin="authored",
        tools=tuple(
            ToolSchema(name=name, description="Fixture tool.", input_schema={"type": "object"})
            for name in names
        ),
    )


def _outage(tool: str) -> WorldEvent:
    """Degrade one tool from t=5 for three ticks."""
    return WorldEvent(event_id="outage", due=5, kind="outage", tools=(tool,), duration=3)


def _workbook() -> bytes:
    """Return a typed workbook with enough rows for several deletions."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Orders"
    sheet.append(["id", "amount"])
    for index in range(1, 7):
        sheet.append([index, index * 10.5])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _edit(call_id: str, source: str, destination: str, start: int = 2) -> ToolCall:
    """Delete one row through the bounded workbook adapter."""
    return _call(
        "edit_workbook",
        call_id,
        source=source,
        destination=destination,
        deletions=[{"sheet": "Orders", "start": start, "count": 1}],
    )


def workbook_case() -> WorldCase:
    """Bind the SpreadsheetBench-shaped artifact world."""
    manifest = WorldManifest(
        domains=(
            StateDomain(name="input.xlsx", owner="native"),
            StateDomain(name="output.xlsx", owner="native"),
            StateDomain(name="answer.xlsx", owner="simulated"),
        ),
        tools=(
            ToolEffectPolicy(
                name="edit_workbook", reads=("input.xlsx", "output.xlsx"), writes=("output.xlsx",)
            ),
        ),
    )
    factory = ArtifactWorldFactory(
        WorkbookEditAdapter(), files={"input.xlsx": _workbook()}, outputs=("output.xlsx",)
    )
    nightly = WorldEvent(
        event_id="nightly-cleanup",
        due=3,
        kind="action",
        action=_edit("nightly", "output.xlsx", "output.xlsx", start=3),
    )
    runtime = ExecutableWorldRuntime(
        manifest, factory, seed=7, events=(nightly, _outage("edit_workbook"))
    )
    return WorldCase(
        runtime=runtime,
        task=_task("edit_workbook"),
        first=_edit("first", "input.xlsx", "output.xlsx"),
        second=_edit("second", "output.xlsx", "output.xlsx"),
        outside_policy=_edit("overwrite-input", "input.xlsx", "input.xlsx"),
        outside_authority=_edit("write-answer", "input.xlsx", "answer.xlsx"),
        branched="output.xlsx",
    )


TICKET_SCHEMA = """
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL, opened_at INTEGER NOT NULL, agent TEXT, escalation TEXT
);
CREATE TABLE outbox (
    id INTEGER PRIMARY KEY, ticket INTEGER NOT NULL, message TEXT NOT NULL, sent_at INTEGER NOT NULL
);
INSERT INTO tickets VALUES (1, 'Printer jam', 'low', 'open', 0, NULL, NULL);
"""


def _tickets(arguments: JsonObject) -> WorldAccess:
    """Read and write only ticket records."""
    del arguments
    return WorldAccess(reads=frozenset({"tickets"}), writes=frozenset({"tickets"}))


def _ticket(db: sqlite3.Connection, ticket: JsonValue) -> tuple[str, str | None]:
    """Return one ticket's status and pending escalation, refusing unknown tickets."""
    row = db.execute("SELECT status, escalation FROM tickets WHERE id = ?", (ticket,)).fetchone()
    if row is None:
        raise RecordToolError(f"ticket {ticket} does not exist")
    return row[0], row[1]


def open_ticket(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Open a ticket, assign a seeded agent and schedule its SLA escalation."""
    agent = "ana" if context.random() < 0.5 else "bo"
    cursor = db.execute(
        "INSERT INTO tickets (title, priority, status, opened_at, agent) "
        "VALUES (?, ?, 'open', ?, ?)",
        (arguments["title"], arguments["priority"], context.now, agent),
    )
    ticket = cursor.lastrowid
    delay = 1 if arguments["priority"] == "high" else 3
    event = context.schedule(
        ToolCall(
            call_id=f"escalate-{ticket}", name="escalate_ticket", arguments={"ticket": ticket}
        ),
        delay=delay,
    )
    db.execute("UPDATE tickets SET escalation = ? WHERE id = ?", (event, ticket))
    return {"ticket": ticket, "agent": agent, "escalates_at": context.now + delay}


def escalate_ticket(
    db: sqlite3.Connection, arguments: JsonObject, context: WorldContext
) -> JsonValue:
    """Escalate a ticket that is still open."""
    status, _ = _ticket(db, arguments["ticket"])
    if status == "open":
        db.execute(
            "UPDATE tickets SET status = 'escalated', escalation = NULL WHERE id = ?",
            (arguments["ticket"],),
        )
    return {"ticket": arguments["ticket"], "status": "escalated" if status == "open" else status}


def _close_access(arguments: JsonObject) -> WorldAccess:
    """Closing writes tickets; a notification also writes the outbox sink."""
    writes = {"tickets", "outbox"} if arguments.get("notify") is True else {"tickets"}
    return WorldAccess(reads=frozenset({"tickets"}), writes=frozenset(writes))


def close_ticket(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Close a ticket, cancel its escalation and optionally notify the customer."""
    _, escalation = _ticket(db, arguments["ticket"])
    if escalation is not None:
        context.cancel(escalation)
    db.execute(
        "UPDATE tickets SET status = 'closed', escalation = NULL WHERE id = ?",
        (arguments["ticket"],),
    )
    if arguments.get("notify") is True:
        db.execute(
            "INSERT INTO outbox (ticket, message, sent_at) VALUES (?, 'closed', ?)",
            (arguments["ticket"], context.now),
        )
    return {"ticket": arguments["ticket"], "status": "closed"}


def _refund_access(arguments: JsonObject) -> WorldAccess:
    """Refunds live in the simulated billing domain."""
    del arguments
    return WorldAccess(writes=frozenset({"refunds"}))


def refund_ticket(
    db: sqlite3.Connection, arguments: JsonObject, context: WorldContext
) -> JsonValue:
    """Never reached natively: refunds are owned by the simulator."""
    del db, context
    return {"refunded": arguments["ticket"]}


TICKET_TOOLS: dict[str, RecordTool] = {
    "open_ticket": RecordTool(access=_tickets, run=open_ticket),
    "escalate_ticket": RecordTool(access=_tickets, run=escalate_ticket),
    "close_ticket": RecordTool(access=_close_access, run=close_ticket),
    "refund_ticket": RecordTool(access=_refund_access, run=refund_ticket),
}

TICKET_MANIFEST = WorldManifest(
    domains=(
        StateDomain(name="tickets", owner="native"),
        StateDomain(name="outbox", owner="native", kind="effect", sandboxed=True),
        StateDomain(name="refunds", owner="simulated"),
    ),
    tools=(
        ToolEffectPolicy(name="open_ticket", reads=("tickets",), writes=("tickets",)),
        ToolEffectPolicy(name="escalate_ticket", writes=("tickets",)),
        ToolEffectPolicy(name="close_ticket", reads=("tickets",), writes=("tickets",)),
        ToolEffectPolicy(name="refund_ticket", writes=("refunds",)),
    ),
)


def ticket_case() -> WorldCase:
    """Bind the stateful ticket service."""
    factory = RecordWorldFactory(
        service={"name": "ticket-service", "version": 1}, schema=TICKET_SCHEMA, tools=TICKET_TOOLS
    )
    runtime = ExecutableWorldRuntime(
        TICKET_MANIFEST, factory, seed=7, events=(_outage("open_ticket"),)
    )
    return WorldCase(
        runtime=runtime,
        task=_task(*TICKET_TOOLS),
        first=_call("open_ticket", "first", title="VPN down", priority="low"),
        second=_call("close_ticket", "second", ticket=1),
        outside_policy=_call("close_ticket", "notify", ticket=1, notify=True),
        outside_authority=_call("refund_ticket", "refund", ticket=1),
        branched="tickets",
    )


CASES: dict[str, Callable[[], WorldCase]] = {"workbook": workbook_case, "tickets": ticket_case}


@pytest.fixture(params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> WorldCase:
    """Run every conformance test once per backend."""
    return CASES[request.param]()


def test_snapshot_mutate_restore_is_byte_and_state_equal(case: WorldCase) -> None:
    """A restored world equals its checkpoint byte for byte, after the source moved on."""
    with case.runtime.open(case.task) as world:
        reply = world.execute(case.first)
        assert not reply.is_error
        assert isinstance(json.loads(reply.content), dict)
        saved = world.checkpoint()
        world.execute(case.second)
        assert world.domain_digests() != saved.image.domains
        with case.runtime.restore(case.task, saved) as restored:
            assert restored.checkpoint() == saved
            assert restored.checkpoint().image.parts == saved.image.parts
            assert restored.domain_digests() == saved.image.domains


def test_fork_effects_never_reach_parent_or_siblings(case: WorldCase) -> None:
    """Forks share a starting point and nothing else, even when restored from one checkpoint."""
    with case.runtime.open(case.task) as parent:
        parent.execute(case.first)
        start = parent.checkpoint()
        persisted = start.model_dump_json()
        with (
            case.runtime.restore(case.task, start) as left,
            case.runtime.restore(case.task, start) as right,
            parent.fork() as child,
        ):
            left.execute(case.second)
            assert parent.checkpoint() == start
            assert right.checkpoint() == start
            assert child.checkpoint() == start
            right.advance(4)
            assert left.checkpoint().clock == start.clock + 1
            assert parent.checkpoint() == start
            assert child.checkpoint() == start
            assert left.domain_digests() != right.domain_digests()
        assert start.model_dump_json() == persisted


def test_clock_and_pending_events_resume_after_restore(case: WorldCase) -> None:
    """A restored world resumes its clock, queue and outage, then replays identically."""
    with case.runtime.open(case.task) as world:
        world.execute(case.first)
        assert [event.status for event in world.advance(3)] == ["applied"]
        saved = world.checkpoint()
        assert saved.clock == 4
        assert [event.event_id for event in saved.pending] == ["outage"]
        with case.runtime.restore(case.task, saved) as restored:
            assert restored.clock == saved.clock
            assert restored.pending == saved.pending
            original = world.advance(2)
            replayed = restored.advance(2)
            assert original == replayed
            assert [event.event_id for event in replayed] == ["outage"]
            during = world.checkpoint()
            assert world.execute(case.first).is_error
            assert restored.execute(case.first).is_error
            assert world.checkpoint() == restored.checkpoint()
        with case.runtime.restore(case.task, during) as resumed:
            reply = resumed.execute(case.first)
            assert reply.is_error and "unavailable until t=8" in reply.content
            assert resumed.checkpoint() == world.checkpoint()


def test_replay_from_a_checkpoint_is_deterministic(case: WorldCase) -> None:
    """Two restorations driven by the same calls end in identical checkpoints."""
    with case.runtime.open(case.task) as world:
        saved = world.checkpoint()
    endings = []
    for _ in range(2):
        with case.runtime.restore(case.task, saved) as replay:
            replay.execute(case.first)
            replay.execute(case.second)
            replay.advance(10)
            endings.append(replay.checkpoint())
    assert endings[0] == endings[1]
    assert endings[0].fired


def test_checkpoint_survives_serialization(case: WorldCase) -> None:
    """A checkpoint persisted as JSON restores the same world."""
    with case.runtime.open(case.task) as world:
        world.execute(case.first)
        saved = world.checkpoint()
    loaded = type(saved).model_validate_json(saved.model_dump_json())
    assert loaded == saved
    with case.runtime.restore(case.task, loaded) as restored:
        assert restored.checkpoint() == saved


def test_effect_outside_policy_is_refused_before_execution(case: WorldCase) -> None:
    """Resolved access beyond the declared policy changes no state, clock or queue."""
    with case.runtime.open(case.task) as world:
        world.execute(case.first)
        before = world.checkpoint()
        with pytest.raises(UndeclaredEffectError, match="outside its declared effect policy"):
            world.execute(case.outside_policy)
        assert world.checkpoint() == before


def test_write_outside_domain_authority_is_refused(case: WorldCase) -> None:
    """A native world never touches a simulated-owned domain."""
    with case.runtime.open(case.task) as world:
        before = world.checkpoint()
        with pytest.raises(AuthorityError, match="owned by the simulated side"):
            world.execute(case.outside_authority)
        assert world.checkpoint() == before


def test_branch_transfer_is_enforced_and_restored(case: WorldCase) -> None:
    """After a branch the native backend may not touch the domain, even after restore."""
    with case.runtime.open(case.task) as world:
        world.branch(case.branched)
        with pytest.raises(AuthorityError):
            world.execute(case.first)
        saved = world.checkpoint()
        assert saved.ownership[case.branched] == "simulated"
        with case.runtime.restore(case.task, saved) as restored, pytest.raises(AuthorityError):
            restored.execute(case.first)


def test_foreign_or_tampered_checkpoints_are_refused(case: WorldCase) -> None:
    """Restoration fails closed for another manifest or an image that does not verify."""
    with case.runtime.open(case.task) as world:
        world.execute(case.first)
        saved = world.checkpoint()
        world.execute(case.second)
        later = world.checkpoint()
    foreign = saved.model_copy(update={"manifest_sha256": "0" * 64})
    with pytest.raises(RestoreError, match="manifest"), case.runtime.restore(case.task, foreign):
        pass
    swapped = saved.model_copy(
        update={"image": later.image.model_copy(update={"domains": saved.image.domains})}
    )
    with pytest.raises(RestoreError, match="digests"), case.runtime.restore(case.task, swapped):
        pass

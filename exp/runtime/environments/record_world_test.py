"""SQLite record worlds: native images, transactions and code-bound identity."""

import sqlite3
from collections.abc import Callable
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.record_world import RecordTool, RecordToolError, RecordWorldFactory
from exp.runtime.environments.world_contracts import WorldAccess
from exp.runtime.environments.worlds import WorldContext

SCHEMA = """
CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL, payload BLOB);
INSERT INTO items VALUES (1, 'seed', x'00ff');
"""


def items(arguments: JsonObject) -> WorldAccess:
    """Every fixture tool reads and writes the items table."""
    del arguments
    return WorldAccess(reads=frozenset({"items"}), writes=frozenset({"items"}))


def add(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Insert one item, then fail as a business error when asked to."""
    del context
    db.execute("INSERT INTO items (name) VALUES (?)", (arguments["name"],))
    if arguments.get("reject") is True:
        raise RecordToolError("item rejected by the service")
    return {"added": arguments["name"]}


def add_twice(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """A different implementation under the same tool name."""
    add(db, arguments, context)
    return add(db, arguments, context)


def _builtin_run() -> Callable[[sqlite3.Connection, JsonObject, WorldContext], JsonValue]:
    """Return a callable with no retrievable Python source."""
    return cast(Callable[[sqlite3.Connection, JsonObject, WorldContext], JsonValue], len)


def _factory(run: RecordTool | None = None) -> RecordWorldFactory:
    """Bind the fixture service."""
    return RecordWorldFactory(
        service={"name": "items"},
        schema=SCHEMA,
        tools={"add": run or RecordTool(access=items, run=add)},
    )


def _task() -> TaskCase:
    """Return a task with no tools; the backend is driven directly."""
    return TaskCase(
        task_id="records",
        lineage_group_id="records",
        partition="fit",
        instruction="Fixture.",
        workload_weight=1,
        origin="authored",
    )


def _add(name: str, *, reject: bool = False) -> ToolCall:
    """Build one add call."""
    return ToolCall(call_id=name, name="add", arguments={"name": name, "reject": reject})


def _context() -> WorldContext:
    """Return a context for direct backend calls whose tools never schedule or draw."""
    from exp.runtime.environments.worlds_test import _runtime

    with _runtime().open(_task()) as world:
        return WorldContext(world)


def test_image_is_a_loadable_sqlite_database_and_restores_isolated() -> None:
    """The native image opens in plain SQLite, and restored worlds do not share a connection."""
    world = _factory().create(_task())
    image = world.capture()
    plain = sqlite3.connect(":memory:")
    plain.deserialize(image.parts["database"])
    assert plain.execute("SELECT name, payload FROM items").fetchall() == [("seed", b"\x00\xff")]
    left, right = _factory().restore(image), _factory().restore(image)
    left.execute(_add("left"), _context())
    assert right.domain_digests() == image.domains
    assert left.domain_digests() != image.domains
    assert left.capture().parts["database"][24:28] == bytes(4)


def test_business_error_rolls_back_its_transaction() -> None:
    """A rejected call is an error observation and leaves no partial insert."""
    world = _factory().create(_task())
    before = world.domain_digests()
    reply = world.execute(_add("bad", reject=True), _context())
    assert reply.is_error and reply.content == "item rejected by the service"
    assert world.domain_digests() == before


def test_definition_binds_tool_source_and_refuses_sourceless_tools() -> None:
    """Changing a tool's code changes the identity checkpoints bind."""
    changed = _factory(RecordTool(access=items, run=add_twice))
    assert changed.definition != _factory().definition
    with pytest.raises(ValueError, match="source-defined"):
        _factory(RecordTool(access=items, run=_builtin_run()))
    with pytest.raises(ValueError, match="database part"):
        _factory().restore(_factory().create(_task()).capture().model_copy(update={"parts": {}}))


def test_equal_histories_give_identical_images_after_restore() -> None:
    """An original and a restored database driven by the same calls serialize identically."""
    original = _factory().create(_task())
    original.execute(_add("first"), _context())
    restored = _factory().restore(original.capture())
    for world in (original, restored):
        world.execute(_add("second"), _context())
        world.execute(_add("third", reject=True), _context())
    assert original.capture() == restored.capture()


def schedule_then_reject(
    db: sqlite3.Connection, arguments: JsonObject, context: WorldContext
) -> JsonValue:
    """Stage state, a future action and randomness before rejecting the business operation."""
    db.execute("INSERT INTO items (name) VALUES ('partial')")
    context.random()
    context.schedule(_add("future"), delay=3)
    raise RecordToolError("rejected after staging")


def test_business_rejection_discards_scheduled_events_and_random_draws() -> None:
    """Rolling back a DB transaction must also discard that call's kernel-side effects."""
    from exp.runtime.environments.world_contracts import (
        StateDomain,
        ToolEffectPolicy,
        WorldManifest,
    )
    from exp.runtime.environments.worlds import ExecutableWorldRuntime

    runtime = ExecutableWorldRuntime(
        WorldManifest(
            domains=(StateDomain(name="items", owner="native"),),
            tools=(ToolEffectPolicy(name="add", writes=("items",)),),
        ),
        _factory(RecordTool(access=items, run=schedule_then_reject)),
    )
    with runtime.open(_task()) as world:
        before = world.checkpoint()
        assert world.execute(_add("bad")).is_error
        after = world.checkpoint()
        assert after.image == before.image
        assert after.pending == before.pending
        assert after.draws == before.draws and after.next_event == before.next_event

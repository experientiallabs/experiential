"""A stateful record service on native SQLite, as an executable-world backend.

Each user table is one state domain. Tools are source-defined Python callables: ``access``
resolves the tables a call reads and writes from its arguments before execution, and ``run``
executes inside one SQLite transaction with the world's clock, scheduler and seeded random
stream. A ``RecordToolError`` is a business failure: the transaction rolls back and the actor
receives an error observation. The native checkpoint is SQLite's own serialized database image.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, JsonValue, Sha256, sha256_bytes, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.world_contracts import NativeImage, WorldAccess
from exp.runtime.environments.worlds import WorldContext

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
RECORD_WORLD_PROTOCOL = "sqlite-record-world-v1"

_DATABASE_PART = "database"


class RecordToolError(Exception):
    """A business failure reported to the actor; the call's transaction is rolled back."""


@dataclass(frozen=True)
class RecordTool:
    """One record-service tool implemented by source-defined Python callables.

    Attributes:
        access: Resolves the tables one call reads and writes, without touching state.
        run: Executes the call on the open transaction and returns a JSON reply.
    """

    access: Callable[[JsonObject], WorldAccess]
    run: Callable[[sqlite3.Connection, JsonObject, WorldContext], JsonValue]


class RecordWorldFactory:
    """Create and restore SQLite record worlds for one schema and tool set."""

    def __init__(
        self, *, service: JsonObject, schema: str, tools: Mapping[str, RecordTool]
    ) -> None:
        """Bind a service identity, the schema and seed script, and the tool implementations.

        Args:
            service: Caller-declared service identity, such as a name and version.
            schema: SQL script creating the tables and their initial records.
            tools: Tool implementations keyed by exact tool name.

        Raises:
            ValueError: No tools are given or a tool callable has no retrievable source.
        """
        if not tools:
            raise ValueError("a record world needs at least one tool")
        self._schema = schema
        self._tools = dict(tools)
        self._definition: JsonObject = {
            "name": RECORD_WORLD_PROTOCOL,
            "sqlite": sqlite3.sqlite_version,
            "service": service,
            "schema_sha256": sha256_bytes(schema.encode()),
            "tools": {
                name: {"access": _source_digest(tool.access), "run": _source_digest(tool.run)}
                for name, tool in sorted(self._tools.items())
            },
        }

    @property
    def definition(self) -> JsonObject:
        """Return the schema, tool source and engine identity bound by checkpoints."""
        return self._definition

    def create(self, task: TaskCase) -> RecordWorld:
        """Create a fresh in-memory database from the schema script."""
        del task
        connection = _connect()
        try:
            connection.executescript(self._schema)
            connection.commit()
        except BaseException:
            connection.close()
            raise
        return RecordWorld(connection, self._tools)

    def restore(self, image: NativeImage) -> RecordWorld:
        """Deserialize a captured database image into a new, isolated connection."""
        if set(image.parts) != {_DATABASE_PART}:
            raise ValueError("record world images contain exactly one database part")
        connection = _connect()
        try:
            connection.deserialize(image.parts[_DATABASE_PART])
        except BaseException:
            connection.close()
            raise
        return RecordWorld(connection, self._tools)


class RecordWorld:
    """One isolated SQLite database driven by the executable-world kernel."""

    def __init__(self, connection: sqlite3.Connection, tools: Mapping[str, RecordTool]) -> None:
        """Own one connection; it is never shared with another world."""
        self._connection = connection
        self._tools = tools

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve one call's tables through its tool's access callable."""
        return self._tool(action.name).access(action.arguments)

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Run one call in a transaction, reporting business failures as error observations."""
        tool = self._tool(action.name)
        try:
            reply = tool.run(self._connection, action.arguments, context)
        except RecordToolError as exc:
            self._connection.rollback()
            context.discard()
            return Observation(content=str(exc), is_error=True)
        except BaseException:
            self._connection.rollback()
            raise
        self._connection.commit()
        return Observation(content=json.dumps(reply, sort_keys=True, separators=(",", ":")))

    def domain_digests(self) -> dict[str, Sha256]:
        """Digest every user table's columns and rows in a content-defined order."""
        digests: dict[str, Sha256] = {}
        tables = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (table,) in tables:
            quoted = '"' + str(table).replace('"', '""') + '"'
            cursor = self._connection.execute(f"SELECT * FROM {quoted}")  # noqa: S608 - catalog name
            columns = [column[0] for column in cursor.description]
            rows = sorted(
                (json.dumps([_cell(value) for value in row]) for row in cursor.fetchall()),
            )
            digests[str(table)] = sha256_json({"columns": columns, "rows": rows})
        return digests

    def capture(self) -> NativeImage:
        """Return SQLite's serialized database image and the table digests."""
        return NativeImage(
            parts={_DATABASE_PART: _canonical_image(self._connection.serialize())},
            domains=self.domain_digests(),
        )

    def projection(self) -> JsonObject:
        """Return evaluator-only table contents in a deterministic, lossless JSON shape.

        This reads actual SQLite values, independently of tool replies. Callers must select
        explicit public fields before exposing any of this private projection to an actor.
        """
        tables: JsonObject = {}
        names = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (name,) in names:
            quoted = '"' + str(name).replace('"', '""') + '"'
            cursor = self._connection.execute(f"SELECT * FROM {quoted}")  # noqa: S608
            rows = [[_cell(value) for value in row] for row in cursor.fetchall()]
            tables[str(name)] = {
                "columns": [column[0] for column in cursor.description],
                "rows": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
            }
        return {"tables": tables}

    def state_projection(self) -> JsonObject:
        """Expose table-keyed private state for declared reply consistency checks."""
        tables = self.projection()["tables"]
        assert isinstance(tables, dict)
        return tables

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    def _tool(self, name: str) -> RecordTool:
        """Return one tool implementation, refusing names the service does not implement."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"record service does not implement tool {name!r}")
        return tool


def _connect() -> sqlite3.Connection:
    """Open a private in-memory database with explicit transactions."""
    return sqlite3.connect(":memory:", check_same_thread=False, autocommit=False)


def _canonical_image(image: bytes) -> bytes:
    """Zero SQLite header bookkeeping that differs between connections holding equal pages.

    Bytes 24-27 (file change counter) and 92-95 (version-valid-for) are reset together, which
    SQLite reads as a valid header; bytes 96-99 record the library that last wrote the file.
    """
    if len(image) < 100 or not image.startswith(b"SQLite format 3\0"):
        raise ValueError("record world capture did not produce a SQLite database image")
    header = bytearray(image[:100])
    header[24:28] = bytes(4)
    header[92:100] = bytes(8)
    return bytes(header) + image[100:]


def _cell(value: object) -> JsonValue:
    """Project one SQLite value into JSON without losing blobs."""
    if isinstance(value, bytes):
        return {"blob": value.hex()}
    if value is None or isinstance(value, (int, float, str)):
        return value
    raise ValueError(f"unsupported SQLite value type {type(value).__name__}")


def _source_digest(function: Callable[..., object]) -> Sha256:
    """Digest a tool callable's source so checkpoints bind the code that produced them."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "record tools must be source-defined Python functions so checkpoints can bind them"
        ) from exc
    return sha256_bytes(source.encode())

"""Private disk-backed JSON documents for incremental trace normalization."""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO

import ijson
from pydantic import JsonValue

from exp.common.core.artifacts import SourceIdentity
from exp.common.traces.ingest.otlp import TraceNormalizationIssue
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError


@dataclass(frozen=True)
class JsonNode:
    """One JSON value stored on private temporary disk.

    Attributes:
        database: Owning scratch connection.
        identity: Node row identity.
        kind: map, array, or scalar.
        scalar: Encoded scalar JSON, absent for containers.
    """

    database: sqlite3.Connection
    identity: int
    kind: str
    scalar: str | None

    def get(self, key: str) -> JsonNode | None:
        """Read the last declared map key, matching json.loads duplicate-key semantics."""
        row = self.database.execute(
            "SELECT id,kind,value FROM json_nodes WHERE parent=? AND name=? "
            "ORDER BY id DESC LIMIT 1",
            (self.identity, key),
        ).fetchone()
        return None if row is None else JsonNode(self.database, *row)

    def children(self) -> Iterator[JsonNode]:
        """Read array or object values in source order without collecting their payloads."""
        for row in self.database.execute(
            "SELECT id,kind,value FROM json_nodes WHERE parent=? ORDER BY id", (self.identity,)
        ):
            yield JsonNode(self.database, *row)

    def value(self) -> JsonValue:
        """Materialize one requested source record, never its surrounding collection."""
        if self.kind == "scalar":
            return json.loads(self.scalar or "null")
        if self.kind == "array":
            return [child.value() for child in self.children()]
        return {
            name: JsonNode(self.database, identity, kind, value).value()
            for identity, name, kind, value in self.database.execute(
                "SELECT id,name,kind,value FROM json_nodes WHERE parent=? ORDER BY id",
                (self.identity,),
            )
        }


class JsonArchive:
    """A verified source snapshot whose JSON containers remain on temporary disk.

    The incremental parser owns JSON syntax. SQLite indexes provide random access to
    wrapper fields without materializing a whole export. Memory scales with nesting
    and the largest scalar during parsing, then the trace currently being normalized.
    """

    def __init__(
        self, path: Path, directory: Path, database: sqlite3.Connection, vendor: str
    ) -> None:
        """Snapshot exact source bytes, validate UTF-8, and decode JSON or recover JSONL lines."""
        self.database = database
        self.issues: list[TraceNormalizationIssue] = []
        self.jsonl = False
        self.duplicate_keys = False
        database.executescript(
            "CREATE TABLE json_nodes(id INTEGER PRIMARY KEY,parent INTEGER,name TEXT,"
            "kind TEXT,value TEXT);"
            "CREATE INDEX json_parent ON json_nodes(parent,name,id);"
        )
        snapshot = directory / "source"
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            with path.open("rb") as source, snapshot.open("xb") as target:
                snapshot.chmod(0o600)
                while chunk := source.read(64 * 1024):
                    decoder.decode(chunk)
                    digest.update(chunk)
                    target.write(chunk)
                decoder.decode(b"", final=True)
        except (OSError, UnicodeDecodeError) as exc:
            raise VendorTraceFormatError(
                f"Cannot read {vendor} export; select a readable UTF-8 file."
            ) from exc
        self.source = SourceIdentity(
            kind="otlp" if vendor == "otlp" else "file",
            source_id=str(path) if vendor in {"otlp", "posthog"} else f"{vendor}:{path}",
            sha256=digest.hexdigest(),
        )
        try:
            with snapshot.open("rb") as stream:
                self._parse(stream)
        except ijson.JSONError:
            self.jsonl = True
            self.duplicate_keys = False
            database.execute("DELETE FROM json_nodes")
            with snapshot.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    database.execute("SAVEPOINT json_line")
                    duplicate_keys = self.duplicate_keys
                    try:
                        self._parse(io.BytesIO(line))
                    except ijson.JSONError:
                        database.execute("ROLLBACK TO json_line")
                        self.duplicate_keys = duplicate_keys
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as exc:
                            message = exc.msg
                        else:
                            message = "unsupported JSON number or syntax"
                        self.issues.append(
                            TraceNormalizationIssue(
                                f"line-{line_number}", f"invalid JSONL record: {message}"
                            )
                        )
                    finally:
                        database.execute("RELEASE json_line")
            if (
                not self.issues
                and not database.execute("SELECT 1 FROM json_nodes LIMIT 1").fetchone()
            ):
                raise VendorTraceFormatError(f"{vendor} export contains no records") from None
        database.commit()

    def _parse(self, stream: BinaryIO) -> None:
        """Index incremental parser events with only the current ancestor stack in memory."""
        stack: list[tuple[int, str | None]] = []
        for event, value in ijson.basic_parse(stream):
            if event == "map_key":
                identity, _ = stack[-1]
                if self.database.execute(
                    "SELECT 1 FROM json_nodes WHERE parent=? AND name=? LIMIT 1", (identity, value)
                ).fetchone():
                    self.duplicate_keys = True
                stack[-1] = (identity, value)
                continue
            if event in {"end_map", "end_array"}:
                stack.pop()
                continue
            parent, key = stack[-1] if stack else (None, None)
            if event in {"start_map", "start_array"}:
                kind = event.removeprefix("start_")
                scalar = None
            else:
                kind = "scalar"
                scalar = json.dumps(
                    float(value) if isinstance(value, Decimal) else value, ensure_ascii=False
                )
            row = self.database.execute(
                "INSERT INTO json_nodes(parent,name,kind,value) VALUES (?,?,?,?)",
                (parent, key, kind, scalar),
            )
            if event in {"start_map", "start_array"}:
                assert row.lastrowid is not None
                stack.append((row.lastrowid, None))

    def documents(self) -> Iterator[JsonNode]:
        """Read complete decoded documents in file order, excluding malformed JSONL lines."""
        for row in self.database.execute(
            "SELECT id,kind,value FROM json_nodes WHERE parent IS NULL ORDER BY id"
        ):
            yield JsonNode(self.database, *row)


def records(
    node: JsonNode,
    *,
    vendor: str,
    wrappers: tuple[str, ...],
    keys: tuple[str, ...],
    chat: bool = False,
) -> Iterator[JsonNode]:
    """Flatten explicit collection shapes without materializing their record arrays."""
    if node.kind == "array":
        if (
            chat
            and next(node.children(), None) is not None
            and all(
                child.kind == "map" and child.get("role") is not None for child in node.children()
            )
        ):
            yield node
            return
        for child in node.children():
            yield from records(child, vendor=vendor, wrappers=wrappers, keys=keys)
        return
    if node.kind != "map":
        raise VendorTraceFormatError(f"{vendor} exports must contain record objects")
    if any(node.get(key) is not None for key in keys):
        yield node
        return
    for key in wrappers:
        child = node.get(key)
        if child is not None and child.kind == "array":
            yield from records(child, vendor=vendor, wrappers=wrappers, keys=keys)
            return
    expected = ", ".join((*keys, *wrappers))
    raise VendorTraceFormatError(f"{vendor} record has none of the expected keys: {expected}")

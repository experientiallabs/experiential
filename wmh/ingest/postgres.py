"""Postgres trace source: rows in a table -> normalized `Trace`s. Transport only, no new schema.

Teams that don't run an observability vendor usually have agent runs in Postgres already, in one
of three shapes this adapter reads directly:

  1. **Row per step/span**: a `trace_id` column plus a JSON `payload` column holding one span or
     event in any shape the file adapters accept (OTLP span, Langfuse observation, PostHog event,
     ...). Rows sharing a `trace_id` become one episode.
  2. **Row per message**: a session/thread table where each row's payload is one chat message
     (`{"role": ..., ...}`). Rows are assembled per `trace_id` into one conversation and read by
     the `chat-json` converter.
  3. **Row per trace**: each row's payload is a self-contained session (a `{"messages": [...]}`
     blob, a Langfuse trace object, an OTLP envelope, ...). No `trace_id` column needed.

The payload column's *format* is auto-detected with the same detector files use
(`wmh.ingest.detect`), so Postgres stays pure transport and schema knowledge lives in the existing
adapters. Default columns: `trace_id` (used when present), `payload` (required), `created_at`
(ordering, when present): all overridable via `VendorPull.trace_id_column` /
`payload_column` / `order_column`. When the table has a `trace_id` column its value overrides any
trace id inside the payload, so episode boundaries always follow the table.

The driver (`psycopg`) is an optional extra (`world-model-harness[postgres]`), imported lazily
inside the pull path like the vendor SDKs. Driver failures are re-raised as stdlib
`PermissionError` (bad credentials) / `ConnectionError` (unreachable), so callers: the streaming
ingest's error classifier in particular: never need psycopg imported.
"""

from __future__ import annotations

import json
import os
import re

from pydantic import JsonValue

from wmh.core.types import Trace
from wmh.ingest.adapter import VendorPull, get_adapter, register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.detect import detect_format
from wmh.ingest.normalize import SpanRecord

DSN_ENV = "WMH_POSTGRES_DSN"

# A table is `name` or `schema.name`; a column is one identifier. Validated (not escaped) because
# identifiers can't be bound as query parameters and an interpolated name must not smuggle SQL.
_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_TABLE_RE = re.compile(rf"^{_IDENTIFIER}(\.{_IDENTIFIER})?$")
_COLUMN_RE = re.compile(rf"^{_IDENTIFIER}$")

_DEFAULT_TRACE_ID_COLUMN = "trace_id"
_DEFAULT_PAYLOAD_COLUMN = "payload"
_DEFAULT_ORDER_COLUMN = "created_at"


def _validate_table(table: str) -> str:
    if not table or not _TABLE_RE.match(table):
        raise ValueError(
            f"invalid postgres table name {table!r}: pass --table as <table> or <schema>.<table> "
            "(letters, digits, underscores)"
        )
    return table


def _validate_column(column: str, *, flag: str) -> str:
    if not _COLUMN_RE.match(column):
        raise ValueError(f"invalid postgres column name {column!r} for {flag}")
    return column


def _is_message(payload: JsonValue) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("role"), str)


class PostgresAdapter(BaseTraceAdapter):
    """Read trace rows from a Postgres table; payload format shared with the file adapters."""

    name = "postgres"

    def from_file(self, path: str) -> list[Trace]:
        raise ValueError(
            "the postgres source reads a live database, not a file: pass --dsn/--table "
            "(export the rows to JSON/JSONL and auto-detect if you prefer a file)"
        )

    def spans_from_pull(self, pull: VendorPull) -> list[SpanRecord]:
        rows = self._fetch_rows(pull)
        payloads = [payload for _, payload in rows]
        if not payloads:
            return []
        if all(_is_message(p) for p in payloads):
            rows = self._assemble_conversations(rows)
            inner: BaseTraceAdapter = self._inner_adapter("chat-json")
        else:
            inner = self._inner_adapter(detect_format(payloads))
        # Same global uniqueness re-stamp as `collect_all`, plus the table's trace-id override:
        # a row's `trace_id` column value defines the episode boundary regardless of any id the
        # payload itself carries.
        spans: list[SpanRecord] = []
        for trace_id, payload in rows:
            for span in inner.spans_from_payload(payload):
                if trace_id is not None:
                    span.trace_id = trace_id
                span.span_id = f"{len(spans):012d}-{span.span_id}"
                spans.append(span)
        return spans

    @staticmethod
    def _inner_adapter(fmt: str) -> BaseTraceAdapter:
        adapter = get_adapter(fmt)
        if not isinstance(adapter, BaseTraceAdapter):
            raise ValueError(f"postgres rows detected as {fmt!r}, which cannot read row payloads")
        return adapter

    @staticmethod
    def _assemble_conversations(
        rows: list[tuple[str | None, JsonValue]],
    ) -> list[tuple[str | None, JsonValue]]:
        """Group message-per-row payloads into one `{"messages": [...]}` payload per trace id."""
        by_trace: dict[str, list[JsonValue]] = {}
        for trace_id, payload in rows:
            if trace_id is None:
                raise ValueError(
                    "rows look like individual chat messages, which need a trace-id column to "
                    "group them into sessions; pass --trace-id-column (or store one session per "
                    "row as a {'messages': [...]} blob)"
                )
            by_trace.setdefault(trace_id, []).append(payload)
        return [
            (trace_id, {"trace_id": trace_id, "messages": messages})
            for trace_id, messages in by_trace.items()
        ]

    def _fetch_rows(self, pull: VendorPull) -> list[tuple[str | None, JsonValue]]:
        """`(trace_id, decoded payload)` per row, in `order_column` order when the table has one.

        Overridable seam: tests swap it for canned rows; everything above it is driver-free.
        """
        if pull.table is None:
            raise ValueError("the postgres source needs --table <table-with-trace-rows>")
        table = _validate_table(pull.table)
        dsn = pull.dsn or os.environ.get(DSN_ENV)
        if not dsn:
            raise ValueError(
                f"the postgres source needs a connection string: pass --dsn or set ${DSN_ENV}"
            )
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise ValueError(
                "the postgres source needs the psycopg driver: "
                "pip install 'world-model-harness[postgres]'"
            ) from exc
        from psycopg import sql

        table_ident = sql.Identifier(*table.split("."))
        try:
            with psycopg.connect(dsn, connect_timeout=10) as conn, conn.cursor() as cur:
                try:
                    cur.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(table_ident))
                except psycopg.errors.UndefinedTable as exc:
                    raise ValueError(
                        f"postgres table {table!r} does not exist; check --table "
                        "(schema-qualify it if needed, e.g. public.agent_traces)"
                    ) from exc
                description = cur.description or []
                columns = [column.name for column in description]
                payload_column, trace_id_column, order_column = self._resolve_columns(
                    pull, table, columns
                )
                selected: sql.Composable = (
                    sql.Identifier(trace_id_column) if trace_id_column else sql.SQL("NULL")
                )
                query = sql.SQL("SELECT {}, {} FROM {}").format(
                    selected, sql.Identifier(payload_column), table_ident
                )
                params: tuple[str, ...] = ()
                if pull.since is not None and order_column is not None:
                    query += sql.SQL(" WHERE {} >= %s").format(sql.Identifier(order_column))
                    params = (pull.since,)
                if order_column is not None:
                    query += sql.SQL(" ORDER BY {}").format(sql.Identifier(order_column))
                cur.execute(query, params)
                fetched = cur.fetchall()
                has_trace_id = trace_id_column is not None
        except psycopg.OperationalError as exc:
            message = str(exc).strip()
            if "authentication" in message or "password" in message:
                raise PermissionError(f"postgres authentication failed: {message}") from exc
            raise ConnectionError(f"could not connect to postgres: {message}") from exc
        rows: list[tuple[str | None, JsonValue]] = []
        for fetched_row in fetched:
            trace_id = str(fetched_row[0]) if has_trace_id and fetched_row[0] is not None else None
            payload: JsonValue = fetched_row[1]
            if isinstance(payload, (str, bytes, bytearray)):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue  # tolerate a corrupt row; keep the rest (same as corrupt JSONL lines)
            rows.append((trace_id, payload))
        return rows

    @staticmethod
    def _resolve_columns(
        pull: VendorPull, table: str, columns: list[str]
    ) -> tuple[str, str | None, str | None]:
        """Resolve `(payload, trace_id, order)` column names against the table's real columns.

        The payload column is required (explicit or the `payload` default); trace-id and order
        columns participate only when explicitly given or present under their default names.
        """
        payload_column = _validate_column(
            pull.payload_column or _DEFAULT_PAYLOAD_COLUMN, flag="--payload-column"
        )
        if payload_column not in columns:
            raise ValueError(
                f"table {table} has no {payload_column!r} column (columns: {columns}); "
                "pass --payload-column <json-column>"
            )
        trace_id_column: str | None
        if pull.trace_id_column is not None:
            trace_id_column = _validate_column(pull.trace_id_column, flag="--trace-id-column")
            if trace_id_column not in columns:
                raise ValueError(
                    f"table {table} has no {trace_id_column!r} column (columns: {columns})"
                )
        else:
            present = _DEFAULT_TRACE_ID_COLUMN in columns
            trace_id_column = _DEFAULT_TRACE_ID_COLUMN if present else None
        order_column: str | None
        if pull.order_column is not None:
            order_column = _validate_column(pull.order_column, flag="--order-column")
            if order_column not in columns:
                raise ValueError(
                    f"table {table} has no {order_column!r} column (columns: {columns})"
                )
        else:
            order_column = _DEFAULT_ORDER_COLUMN if _DEFAULT_ORDER_COLUMN in columns else None
        return payload_column, trace_id_column, order_column


register_adapter(PostgresAdapter())

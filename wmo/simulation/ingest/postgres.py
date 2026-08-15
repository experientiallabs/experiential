"""Read agent trace rows from a Postgres table and normalize them into canonical evidence.

Teams without an observability vendor usually already store agent runs in Postgres. This source is
transport only: it selects declared columns from one declared table and hands each row payload to
the explicitly declared document normalizer. It never guesses the payload format, never invents a
trace identity, and never rewrites a payload.

The caller declares three things:

- ``table`` and ``payload_column``: where the JSON payloads live,
- ``payload_format``: which canonical document normalizer reads those payloads,
- ``row_shape``: whether each row payload is a whole document (``document``) or one chat message of
  a conversation assembled by ``trace_id_column`` (``message``).

Identifiers cannot be bound as query parameters, so the table and column names are validated
against a strict identifier pattern and quoted through ``psycopg.sql`` composition.

The driver is the optional ``postgres`` extra. The row reader is a protocol, so a caller (a test, or
an application with its own connection pool) can supply rows directly without the driver installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes
from wmo.simulation.ingest.braintrust import normalize_braintrust_payloads
from wmo.simulation.ingest.chat_json import normalize_chat_json_payloads
from wmo.simulation.ingest.langfuse import normalize_langfuse_payloads
from wmo.simulation.ingest.langsmith import normalize_langsmith_payloads
from wmo.simulation.ingest.mastra import normalize_mastra_payloads
from wmo.simulation.ingest.otel_genai import normalize_otel_genai_payloads
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.phoenix import normalize_phoenix_payloads

DSN_ENV = "WMO_POSTGRES_DSN"

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_TABLE_PATTERN = re.compile(rf"^{_IDENTIFIER}(\.{_IDENTIFIER})?$")
_COLUMN_PATTERN = re.compile(rf"^{_IDENTIFIER}$")
_MISSING_POSTGRES_EXTRA = (
    "the postgres trace source requires the optional postgres dependencies; run "
    "`uv sync --extra postgres` or install `world-model-optimizer[postgres]`"
)

PostgresPayloadFormat = Literal[
    "braintrust",
    "chat-json",
    "langfuse",
    "langsmith",
    "mastra",
    "otel-genai",
    "phoenix",
]
PostgresRowShape = Literal["document", "message"]


class PostgresSourceError(ValueError):
    """Raised when a Postgres trace source is misconfigured or cannot be read."""


@dataclass(frozen=True)
class PostgresRow:
    """One selected trace row.

    Args:
        source_trace_id: Value of the declared trace-id column, if the source declares one.
        payload: Decoded JSON payload of the declared payload column.
    """

    source_trace_id: str | None
    payload: JsonValue


@dataclass(frozen=True)
class PostgresSourceConfig:
    """One explicit Postgres trace source.

    Args:
        table: Table name as ``table`` or ``schema.table``.
        payload_format: Canonical document normalizer that reads the payload column.
        dsn: Connection string. Read from ``WMO_POSTGRES_DSN`` when omitted.
        payload_column: JSON column holding each row payload.
        trace_id_column: Column holding the source trace identity, required for message rows.
        order_column: Column used to order rows and to bound them by ``since``.
        row_shape: Whether a row payload is a whole document or one chat message.
        since: Inclusive lower bound compared against ``order_column``.
    """

    table: str
    payload_format: PostgresPayloadFormat
    dsn: str | None = None
    payload_column: str = "payload"
    trace_id_column: str | None = None
    order_column: str | None = None
    row_shape: PostgresRowShape = "document"
    since: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers and the declared row shape.

        Raises:
            PostgresSourceError: An identifier is unsafe or the row shape is unsupported.
        """
        if not _TABLE_PATTERN.fullmatch(self.table):
            raise PostgresSourceError(
                f"invalid postgres table {self.table!r}: use <table> or <schema>.<table>"
            )
        _validate_column(self.payload_column, label="payload column")
        if self.trace_id_column is not None:
            _validate_column(self.trace_id_column, label="trace id column")
        if self.order_column is not None:
            _validate_column(self.order_column, label="order column")
        if self.row_shape == "message":
            if self.payload_format != "chat-json":
                raise PostgresSourceError(
                    "message rows are assembled into chat conversations, so row_shape='message' "
                    "requires payload_format='chat-json'"
                )
            if self.trace_id_column is None:
                raise PostgresSourceError(
                    "message rows need a trace id column to group them into conversations"
                )
        if self.since is not None and self.order_column is None:
            raise PostgresSourceError("bounding rows by since requires an order column")


_DECLARATION_FIELDS = frozenset(
    {
        "table",
        "payload_format",
        "dsn",
        "payload_column",
        "trace_id_column",
        "order_column",
        "row_shape",
        "since",
    }
)
_CONFIG_ADAPTER = TypeAdapter(PostgresSourceConfig)


class PostgresRowReader(Protocol):
    """Reads the declared trace rows of one Postgres source."""

    def read_rows(self, config: PostgresSourceConfig) -> Sequence[PostgresRow]:
        """Return the declared rows of one Postgres source in source order.

        Args:
            config: Explicit Postgres trace source.

        Returns:
            Selected rows in ``order_column`` order when the source declares one.
        """
        ...


def read_postgres_source_file(path: Path) -> PostgresSourceConfig:
    """Read one declared Postgres trace source from a local JSON declaration.

    The declaration mirrors the source fields, for example::

        {
          "table": "public.agent_traces",
          "payload_format": "chat-json",
          "payload_column": "payload",
          "trace_id_column": "trace_id",
          "order_column": "created_at",
          "row_shape": "message",
          "since": "2024-05-01T00:00:00Z"
        }

    The connection string is read from ``dsn`` when the declaration carries one, otherwise from
    ``WMO_POSTGRES_DSN``, so a checked-in declaration never has to hold a credential.

    Args:
        path: Local JSON source declaration.

    Returns:
        The declared Postgres trace source.

    Raises:
        PostgresSourceError: The declaration is unreadable, malformed, or invalid.
    """
    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PostgresSourceError(f"unreadable postgres source declaration {path}: {exc}") from None
    try:
        declaration = json.loads(raw)
    except json.JSONDecodeError:
        raise PostgresSourceError(f"postgres source declaration {path} is not valid JSON") from None
    if not isinstance(declaration, dict):
        raise PostgresSourceError(f"postgres source declaration {path} must be a JSON object")
    unsupported = sorted(set(declaration) - _DECLARATION_FIELDS)
    if unsupported:
        raise PostgresSourceError(
            f"postgres source declaration {path} has unsupported fields: {', '.join(unsupported)}"
        )
    try:
        return _CONFIG_ADAPTER.validate_python(declaration)
    except ValidationError as exc:
        raise PostgresSourceError(
            f"invalid postgres source declaration {path}: {exc.errors()[0]['msg']}"
        ) from None


def load_postgres_source(
    config: PostgresSourceConfig,
    *,
    reader: PostgresRowReader | None = None,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read one Postgres trace source into canonical trace evidence.

    Args:
        config: Explicit Postgres trace source.
        reader: Row reader. The bundled psycopg reader is used when omitted.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        source_id: Optional durable source label. The table identity is used when omitted.

    Returns:
        Canonical traces and every retained validation exclusion.

    Raises:
        PostgresSourceError: The source cannot be read or its rows cannot be decoded.
    """
    rows = tuple((reader or PsycopgRowReader()).read_rows(config))
    digest_input: JsonValue = [
        {"trace_id": row.source_trace_id, "payload": row.payload} for row in rows
    ]
    source = SourceIdentity(
        kind="production",
        source_id=source_id or f"postgres:{config.table}",
        sha256=hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest(),
    )
    return normalize_postgres_rows(
        rows,
        config=config,
        source=source,
        semantic_convention_version=semantic_convention_version,
    )


def normalize_postgres_rows(
    rows: Sequence[PostgresRow],
    *,
    config: PostgresSourceConfig,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize selected Postgres rows through their declared document normalizer.

    Args:
        rows: Selected rows in source order.
        config: Explicit Postgres trace source.
        source: Immutable identity of the selected rows.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Row exclusions collected before normalization.

    Returns:
        Canonical traces and every retained validation exclusion.

    Raises:
        PostgresSourceError: A message row is missing its trace identity.
    """
    issues = list(initial_issues)
    payloads: list[JsonValue] = (
        _conversation_payloads(rows, issues)
        if config.row_shape == "message"
        else _row_payloads(rows)
    )
    normalizer = _NORMALIZERS[config.payload_format]
    return normalizer(
        payloads,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
    )


def _row_payloads(rows: Sequence[PostgresRow]) -> list[JsonValue]:
    """Return the payload of every document row in source order.

    Args:
        rows: Selected rows in source order.

    Returns:
        Row payloads in source order.
    """
    return [row.payload for row in rows]


def _conversation_payloads(
    rows: Sequence[PostgresRow],
    issues: list[TraceNormalizationIssue],
) -> list[JsonValue]:
    """Assemble message rows into one chat conversation document per source trace identity.

    Args:
        rows: Selected message rows in source order.
        issues: Accumulator for rows excluded because they declare no trace identity.

    Returns:
        One conversation document per source trace identity, in first-seen order.
    """
    conversations: dict[str, list[JsonValue]] = {}
    for index, row in enumerate(rows, start=1):
        if row.source_trace_id is None:
            issues.append(
                TraceNormalizationIssue(f"row-{index}", "message row declares no trace identity")
            )
            continue
        conversations.setdefault(row.source_trace_id, []).append(row.payload)
    return [
        {"trace_id": trace_id, "messages": messages} for trace_id, messages in conversations.items()
    ]


class PsycopgRowReader:
    """Reads Postgres trace rows with the optional psycopg driver."""

    def read_rows(self, config: PostgresSourceConfig) -> Sequence[PostgresRow]:
        """Select the declared columns of one Postgres source.

        Args:
            config: Explicit Postgres trace source.

        Returns:
            Selected rows in ``order_column`` order when the source declares one. Rows tied on the
            order column are broken by the trace identity and then by the payload text, so equal
            timestamps cannot reorder a conversation between builds.

        Raises:
            PostgresSourceError: The driver, connection string, table, or a column is unusable.
        """
        _require_postgres_dependencies()
        import psycopg
        from psycopg import sql

        dsn = _resolved_dsn(config)
        table = sql.Identifier(*config.table.split("."))
        trace_id: sql.Composable = (
            sql.Identifier(config.trace_id_column)
            if config.trace_id_column is not None
            else sql.SQL("NULL")
        )
        query = sql.SQL("SELECT {}, {} FROM {}").format(
            trace_id, sql.Identifier(config.payload_column), table
        )
        parameters: tuple[str, ...] = ()
        if config.order_column is not None:
            if config.since is not None:
                query += sql.SQL(" WHERE {} >= %s").format(sql.Identifier(config.order_column))
                parameters = (config.since,)
            order_terms: list[sql.Composable] = [sql.Identifier(config.order_column)]
            if config.trace_id_column is not None:
                order_terms.append(sql.Identifier(config.trace_id_column))
            order_terms.append(sql.SQL("{}::text").format(sql.Identifier(config.payload_column)))
            query += sql.SQL(" ORDER BY ") + sql.SQL(", ").join(order_terms)
        try:
            with psycopg.connect(dsn, connect_timeout=10) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, parameters)
                    fetched = cursor.fetchall()
        except psycopg.Error as exc:
            raise PostgresSourceError(f"postgres trace source query failed: {exc}") from None
        return tuple(
            decode_postgres_row(fetched_row, index=index)
            for index, fetched_row in enumerate(fetched, start=1)
        )


def decode_postgres_row(fetched: Sequence[object], *, index: int) -> PostgresRow:
    """Decode one selected row into a trace row.

    Args:
        fetched: Selected trace identity and payload values, in that order.
        index: One-based row position used in validation messages.

    Returns:
        Decoded trace row.

    Raises:
        PostgresSourceError: The row shape or its payload cannot be decoded.
    """
    if len(fetched) != 2:
        raise PostgresSourceError(f"postgres row {index} did not select two columns")
    raw_trace_id, raw_payload = fetched
    source_trace_id = None if raw_trace_id is None else str(raw_trace_id)
    return PostgresRow(
        source_trace_id=source_trace_id,
        payload=_decoded_payload(raw_payload, index=index),
    )


def _decoded_payload(raw: object, *, index: int) -> JsonValue:
    """Decode one row payload without repairing it.

    Args:
        raw: Payload column value, already decoded by the driver or still JSON text.
        index: One-based row position used in validation messages.

    Returns:
        Decoded JSON payload.

    Raises:
        PostgresSourceError: The payload is not JSON text or a decoded JSON value.
    """
    if isinstance(raw, str | bytes | bytearray):
        try:
            decoded: JsonValue = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise PostgresSourceError(f"postgres row {index} payload is not valid JSON") from None
        else:
            return decoded
    try:
        return _JSON_VALUE_ADAPTER.validate_python(raw)
    except ValidationError:
        raise PostgresSourceError(f"postgres row {index} payload is not a JSON value") from None


def _resolved_dsn(config: PostgresSourceConfig) -> str:
    """Return the declared connection string of one Postgres source.

    Args:
        config: Explicit Postgres trace source.

    Returns:
        Connection string from the configuration or the environment.

    Raises:
        PostgresSourceError: Neither the configuration nor the environment declares one.
    """
    dsn = config.dsn or os.environ.get(DSN_ENV)
    if not dsn:
        raise PostgresSourceError(
            f"the postgres trace source needs a connection string: set dsn or ${DSN_ENV}"
        )
    return dsn


def _require_postgres_dependencies() -> None:
    """Verify that the optional postgres driver is importable.

    Raises:
        PostgresSourceError: The optional postgres dependencies are not installed.
    """
    if not _psycopg_installed():
        raise PostgresSourceError(_MISSING_POSTGRES_EXTRA)


def _psycopg_installed() -> bool:
    """Return whether the optional psycopg driver is importable."""
    return find_spec("psycopg") is not None


def _validate_column(column: str, *, label: str) -> None:
    """Validate one Postgres column identifier.

    Args:
        column: Column name.
        label: Configuration label used in the validation message.

    Raises:
        PostgresSourceError: The column name is not a plain identifier.
    """
    if not _COLUMN_PATTERN.fullmatch(column):
        raise PostgresSourceError(f"invalid postgres {label} {column!r}")


class _PayloadNormalizer(Protocol):
    """Normalizes decoded documents of one canonical source format."""

    def __call__(
        self,
        payloads: Sequence[JsonValue],
        *,
        source: SourceIdentity,
        semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
        initial_issues: Sequence[TraceNormalizationIssue] = (),
    ) -> TraceNormalizationResult:
        """Normalize decoded documents into canonical traces.

        Args:
            payloads: Decoded documents in source order.
            source: Immutable identity of the source rows.
            semantic_convention_version: Pinned GenAI semantic-convention version.
            initial_issues: Exclusions collected before normalization.

        Returns:
            Canonical traces and every retained validation exclusion.
        """
        ...


_NORMALIZERS: dict[PostgresPayloadFormat, _PayloadNormalizer] = {
    "braintrust": normalize_braintrust_payloads,
    "chat-json": normalize_chat_json_payloads,
    "langfuse": normalize_langfuse_payloads,
    "langsmith": normalize_langsmith_payloads,
    "mastra": normalize_mastra_payloads,
    "otel-genai": normalize_otel_genai_payloads,
    "phoenix": normalize_phoenix_payloads,
}

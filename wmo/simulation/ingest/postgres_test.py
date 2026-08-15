"""Tests for the Postgres trace source."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

import pytest

from wmo.simulation.ingest import postgres as postgres_module
from wmo.simulation.ingest.postgres import (
    DSN_ENV,
    PostgresRow,
    PostgresSourceConfig,
    PostgresSourceError,
    PsycopgRowReader,
    decode_postgres_row,
    load_postgres_source,
    normalize_postgres_rows,
)


class _StubReader:
    """Returns canned rows instead of querying Postgres."""

    def __init__(self, rows: Sequence[PostgresRow]) -> None:
        """Record the rows this reader returns for every source."""
        self.rows = tuple(rows)
        self.configs: list[PostgresSourceConfig] = []

    def read_rows(self, config: PostgresSourceConfig) -> Sequence[PostgresRow]:
        """Record the requested source and return the canned rows."""
        self.configs.append(config)
        return self.rows


def _conversation(trace_id: str) -> dict[str, object]:
    """Return one chat conversation document with declared model identity."""
    return {
        "trace_id": trace_id,
        "provider": "openai",
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "It is 18C in Paris."},
        ],
    }


_CHAT_CONFIG = PostgresSourceConfig(
    table="public.agent_traces",
    payload_format="chat-json",
    dsn="postgresql://localhost/wmo",
)


def test_load_postgres_source_normalizes_document_rows() -> None:
    """Each document row is normalized by its declared canonical normalizer."""
    reader = _StubReader(
        [
            PostgresRow(source_trace_id="conversation-1", payload=_conversation("conversation-1")),
            PostgresRow(source_trace_id="conversation-2", payload=_conversation("conversation-2")),
        ]
    )
    config = replace(_CHAT_CONFIG, trace_id_column="trace_id", order_column="created_at")

    result = load_postgres_source(config, reader=reader)

    assert reader.configs == [config]
    assert result.issues == ()
    assert len(result.traces) == 2
    trace = result.traces[0]
    assert trace.task == "What is the weather in Paris?"
    assert trace.source.identity.kind == "production"
    assert trace.source.identity.source_id == "postgres:public.agent_traces"
    assert {trace.spans[0].attributes["wmo.source.trace.id"] for trace in result.traces} == {
        "conversation-1",
        "conversation-2",
    }


def test_load_postgres_source_hashes_selected_rows() -> None:
    """The source digest covers the selected trace identities and payloads."""
    rows = [PostgresRow(source_trace_id="conversation-1", payload=_conversation("conversation-1"))]
    config = _CHAT_CONFIG

    first = load_postgres_source(config, reader=_StubReader(rows))
    same = load_postgres_source(config, reader=_StubReader(rows))
    other = load_postgres_source(
        config,
        reader=_StubReader(
            [PostgresRow(source_trace_id="conversation-2", payload=_conversation("conversation-2"))]
        ),
    )

    assert first.traces[0].source.identity.sha256 == same.traces[0].source.identity.sha256
    assert first.traces[0].source.identity.sha256 != other.traces[0].source.identity.sha256


def test_load_postgres_source_accepts_an_explicit_source_id() -> None:
    """A caller can label the pulled corpus with a durable source id."""
    rows = [PostgresRow(source_trace_id="conversation-1", payload=_conversation("conversation-1"))]

    result = load_postgres_source(
        _CHAT_CONFIG, reader=_StubReader(rows), source_id="postgres:prod-us-east"
    )

    assert result.traces[0].source.identity.source_id == "postgres:prod-us-east"


def test_normalize_postgres_rows_assembles_message_rows() -> None:
    """Message rows sharing a trace identity become one conversation in source order."""
    rows = [
        PostgresRow(
            source_trace_id="conversation-1",
            payload={"role": "user", "content": "What is the weather in Paris?"},
        ),
        PostgresRow(
            source_trace_id="conversation-1",
            payload={"role": "assistant", "content": "It is 18C in Paris."},
        ),
    ]
    config = replace(_CHAT_CONFIG, row_shape="message", trace_id_column="trace_id")

    result = normalize_postgres_rows(
        rows,
        config=config,
        source=load_postgres_source(config, reader=_StubReader(rows)).traces[0].source.identity,
    )

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "What is the weather in Paris?"
    assert trace.spans[0].attributes["wmo.source.trace.id"] == "conversation-1"


def test_load_postgres_source_retains_untraceable_message_rows_as_issues() -> None:
    """A message row without a trace identity is excluded and reported."""
    rows = [
        PostgresRow(
            source_trace_id="conversation-1",
            payload={"role": "user", "content": "What is the weather in Paris?"},
        ),
        PostgresRow(
            source_trace_id="conversation-1",
            payload={"role": "assistant", "content": "It is 18C in Paris."},
        ),
        PostgresRow(payload={"role": "user", "content": "And in Berlin?"}, source_trace_id=None),
    ]
    config = replace(_CHAT_CONFIG, row_shape="message", trace_id_column="trace_id")

    result = load_postgres_source(config, reader=_StubReader(rows))

    assert [issue.source_record for issue in result.issues] == ["row-3"]
    assert "trace identity" in result.issues[0].message


def test_load_postgres_source_dispatches_to_the_declared_vendor_format() -> None:
    """A declared vendor format reads the payload column with that vendor's normalizer."""
    observation: dict[str, object] = {
        "id": "observation-1",
        "traceId": "langfuse-trace-1",
        "type": "GENERATION",
        "startTime": "2024-05-01T00:00:00Z",
        "endTime": "2024-05-01T00:00:01Z",
        "model": "gpt-4o",
        "metadata": {"provider": "openai"},
        "input": "What is the weather in Paris?",
        "output": "It is 18C in Paris.",
    }
    rows = [PostgresRow(source_trace_id="langfuse-trace-1", payload=observation)]

    result = load_postgres_source(
        replace(_CHAT_CONFIG, payload_format="langfuse"), reader=_StubReader(rows)
    )

    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].spans[0].attributes["wmo.source.trace.id"] == "langfuse-trace-1"


def test_postgres_source_config_rejects_unsafe_identifiers() -> None:
    """Table and column identifiers are validated because they cannot be bound as parameters."""
    with pytest.raises(PostgresSourceError, match="invalid postgres table"):
        replace(_CHAT_CONFIG, table="traces; drop table users")
    with pytest.raises(PostgresSourceError, match="payload column"):
        replace(_CHAT_CONFIG, payload_column="payload->>'x'")
    with pytest.raises(PostgresSourceError, match="trace id column"):
        replace(_CHAT_CONFIG, trace_id_column="trace id")
    with pytest.raises(PostgresSourceError, match="order column"):
        replace(_CHAT_CONFIG, order_column="created_at desc")


def test_postgres_source_config_rejects_unsupported_row_shapes() -> None:
    """Message rows are only assembled for chat JSON and need a trace identity column."""
    with pytest.raises(PostgresSourceError, match="payload_format='chat-json'"):
        replace(
            _CHAT_CONFIG,
            row_shape="message",
            payload_format="langfuse",
            trace_id_column="trace_id",
        )
    with pytest.raises(PostgresSourceError, match="trace id column"):
        replace(_CHAT_CONFIG, row_shape="message")


def test_postgres_source_config_requires_an_order_column_for_since() -> None:
    """Bounding a pull by time needs the column that orders the rows."""
    with pytest.raises(PostgresSourceError, match="order column"):
        replace(_CHAT_CONFIG, since="2024-05-01T00:00:00Z")


def test_decode_postgres_row_reads_driver_and_text_payloads() -> None:
    """A payload column arrives already decoded or as JSON text, and is never repaired."""
    decoded = decode_postgres_row(["conversation-1", {"messages": []}], index=1)
    text = decode_postgres_row([None, json.dumps({"messages": []})], index=2)

    assert decoded == PostgresRow(source_trace_id="conversation-1", payload={"messages": []})
    assert text == PostgresRow(source_trace_id=None, payload={"messages": []})

    with pytest.raises(PostgresSourceError, match="not valid JSON"):
        decode_postgres_row([None, "{oops"], index=3)
    with pytest.raises(PostgresSourceError, match="did not select two columns"):
        decode_postgres_row(["conversation-1"], index=4)


def test_psycopg_row_reader_requires_the_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver-backed reader states which extra to install when psycopg is absent."""
    monkeypatch.setattr(postgres_module, "_psycopg_installed", lambda: False)

    with pytest.raises(PostgresSourceError, match="optional postgres dependencies"):
        PsycopgRowReader().read_rows(_CHAT_CONFIG)


def test_psycopg_row_reader_requires_a_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pull without a configured or environment connection string fails before connecting."""
    monkeypatch.delenv(DSN_ENV, raising=False)
    config = PostgresSourceConfig(table="agent_traces", payload_format="chat-json")

    with pytest.raises(PostgresSourceError, match="connection string"):
        PsycopgRowReader().read_rows(config)

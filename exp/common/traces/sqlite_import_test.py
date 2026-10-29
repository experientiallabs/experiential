"""Streaming import hashes and batches preserve evidence without corpus-size buffers."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.common.traces.sqlite import SQLiteTraceStore, _import_id
from exp.common.traces.sqlite_import import BATCH_BYTES, BATCH_RECORDS, prepare_import
from exp.common.traces.trace import Trace
from exp.common.traces.trace_test import _trace


def test_one_pass_identity_and_oversize_trace_are_preserved() -> None:
    """Batch sizing never becomes a total count or per-trace admission limit."""
    traces = tuple(
        _trace().model_copy(update={"task": str(index)}) for index in range(BATCH_RECORDS + 5)
    )
    large = _trace().model_copy(update={"task": "x" * (BATCH_BYTES + 1)})
    traces = (*traces, large)
    source = traces[0].source.identity
    with prepare_import("otlp", source, iter(traces), {}) as prepared:
        assert prepared.import_id == _import_id("otlp", source, traces, {})
        assert prepared.count == len(traces)
        batches = tuple(prepared.batches())
        assert len(batches) >= 3
        assert all(len(batch) <= BATCH_RECORDS for batch in batches)
        assert len(batches[-1]) == 1 and "x" * (BATCH_BYTES + 1) in batches[-1][0].payload


def test_failed_source_iterator_never_publishes_or_creates_shared_storage(tmp_path: Path) -> None:
    """A late source failure is isolated to private preparation, with its spool closed."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    trace = _trace()

    def broken() -> Iterator[Trace]:
        """Supply some records, then reproduce a mid-stream reader failure."""
        yield trace
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        store.write_import(
            "powerset",
            source_format="otlp",
            source=trace.source.identity,
            traces=broken(),
            metadata={},
        )
    assert not store.path.exists()

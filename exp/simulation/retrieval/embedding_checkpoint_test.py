"""Durable batch resume, process coordination, and cache corruption regressions."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import pytest
from filelock import FileLock

from exp.common.core.artifacts import SourceIdentity, canonical_json_bytes, sha256_json
from exp.common.progress import ProgressEvent
from exp.common.project import ArtifactStore, ProjectPaths, artifact_input
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.common.traces.ingest import TraceNormalizationResult, persist_trace_dataset
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy
from exp.simulation.retrieval import embedding_checkpoint
from exp.simulation.retrieval.build import persist_trace_rag
from exp.simulation.retrieval.contracts import RAGAction, RAGLineageBinding
from exp.simulation.retrieval.embedding import default_rag_embedder, embed_rag_texts
from exp.simulation.retrieval.embedding_checkpoint import (
    RAGEmbeddingCheckpoint,
    RAGEmbeddingCheckpointError,
)
from exp.simulation.retrieval.transitions import render_rag_key

_CREATED_AT = datetime(2026, 9, 24, tzinfo=UTC)


def _checkpoint(project: Path, *, chunk_bytes: int = 2_048) -> RAGEmbeddingCheckpoint:
    """Create a checkpoint under the deterministic local model identity."""
    return RAGEmbeddingCheckpoint(
        project, default_rag_embedder().snapshot, maximum_chunk_bytes=chunk_bytes
    )


def _constant(texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
    """Return fixture unit vectors in exact input order."""
    return tuple((1.0, 0.0) for _ in texts)


def _never_embed(texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
    """Fail if a completed or corrupt cache dispatches to the provider."""
    pytest.fail(f"unexpected embedding of {len(texts)} cached inputs")


def test_saved_chunks_are_reused_without_storing_input_text(tmp_path: Path) -> None:
    """Only text hashes and validated vectors survive a fresh checkpoint instance."""
    text = "private prompt text that must never occur in the checkpoint database"
    first = _checkpoint(tmp_path)
    assert first.get_or_embed((text,), _constant) == ((1.0, 0.0),)
    fresh = _checkpoint(tmp_path)
    assert fresh.get_or_embed((text,), _never_embed) == ((1.0, 0.0),)
    assert text.encode() not in first.path.read_bytes()
    with closing(sqlite3.connect(first.path)) as connection:
        assert connection.execute("SELECT text_sha256 FROM vectors").fetchall() == [
            (hashlib.sha256(text.encode()).hexdigest(),)
        ]


@pytest.mark.parametrize("changed", ["connection_sha256", "model_id", "revision"])
def test_changed_model_connection_or_revision_cannot_reuse_vectors(
    tmp_path: Path, changed: str
) -> None:
    """Durable reuse includes exact endpoint and model revision, beyond input equality."""
    first = _checkpoint(tmp_path)
    first.get_or_embed(("same text",), _constant)
    snapshot = default_rag_embedder().snapshot.model_copy(
        update={changed: "f" * 64 if changed == "connection_sha256" else "other"}
    )
    second = RAGEmbeddingCheckpoint(tmp_path, snapshot, maximum_chunk_bytes=2_048)
    assert second.path != first.path
    assert second.load(("same text",)) == {}
    assert _checkpoint(tmp_path, chunk_bytes=512).load(("same text",)) == {}


@pytest.mark.parametrize("corruption", ["digest", "vector", "dimensions", "identity", "schema"])
def test_corrupt_saved_data_fails_before_provider_dispatch(tmp_path: Path, corruption: str) -> None:
    """A corrupt checkpoint cannot silently trigger paid replacement calls or enter an index."""
    checkpoint = _checkpoint(tmp_path)
    checkpoint.get_or_embed(("fixture",), _constant)
    with closing(sqlite3.connect(checkpoint.path)) as connection, connection:
        if corruption == "digest":
            connection.execute("UPDATE vectors SET checksum = 'incorrect'")
        elif corruption == "vector":
            payload = b'{"values":[2.0,0.0]}'
            key = hashlib.sha256(b"fixture").hexdigest()
            checksum = hashlib.sha256(
                checkpoint.identity.encode() + key.encode() + payload
            ).hexdigest()
            connection.execute("UPDATE vectors SET vector = ?, checksum = ?", (payload, checksum))
        elif corruption == "dimensions":
            connection.execute("UPDATE metadata SET dimensions = 3")
        elif corruption == "identity":
            connection.execute("UPDATE metadata SET identity = 'other'")
        else:
            connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RAGEmbeddingCheckpointError):
        _checkpoint(tmp_path).get_or_embed(("fixture",), _never_embed)


def test_failed_batch_validation_preserves_all_earlier_checkpoints(tmp_path: Path) -> None:
    """A later provider dimension change cannot save any part of the invalid batch."""
    checkpoint = _checkpoint(tmp_path)
    checkpoint.get_or_embed(("first",), _constant)
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        checkpoint.get_or_embed(("second",), lambda texts: ((1.0, 0.0, 0.0),))
    assert _checkpoint(tmp_path).load(("first", "second")) == {"first": (1.0, 0.0)}


def test_query_only_embedding_stays_memory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standalone query embedding creates no project runtime cache."""
    monkeypatch.chdir(tmp_path)
    key = render_rag_key(
        task="fixture", initial_context={}, action=RAGAction(kind="message", content="hello")
    )
    assert len(embed_rag_texts(default_rag_embedder(), (key,))) == 1
    assert list(tmp_path.iterdir()) == []


def test_progress_never_reports_a_batch_before_its_commit(tmp_path: Path) -> None:
    """An independent SQLite reader sees every reported completed chunk immediately."""
    checkpoint = _checkpoint(tmp_path)
    counts: list[int] = []

    def observe(event: ProgressEvent) -> None:
        """Check actual durable rows at the instant progress reaches the caller."""
        assert event.completed is not None
        with closing(sqlite3.connect(checkpoint.path)) as connection:
            saved = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            assert saved == event.completed
        counts.append(event.completed)

    keys = tuple(
        render_rag_key(
            task=f"request {index}",
            initial_context={},
            action=RAGAction(kind="message", content="shared action"),
        )
        for index in range(150)
    )
    embed_rag_texts(default_rag_embedder(), keys, checkpoint=checkpoint, progress=observe)
    assert counts == [0, 100, 151]


def test_contended_checkpoint_reports_retry_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live lock owner reports a retry instead of a traceback or paid duplicate."""
    checkpoint = _checkpoint(tmp_path)
    checkpoint.load(())
    monkeypatch.setattr(embedding_checkpoint, "_LOCK_TIMEOUT_SECONDS", 0)
    with FileLock(checkpoint.path.with_suffix(".lock")):
        with pytest.raises(RAGEmbeddingCheckpointError, match="retry after it finishes"):
            checkpoint.get_or_embed(("fixture",), _never_embed)


class _ProviderState:
    """Loopback provider evidence and a one-time interrupted request configuration."""

    def __init__(self, *, fail_request: int | None = None) -> None:
        """Initialize request evidence without raw customer data."""
        self.fail_request = fail_request
        self.requests: list[tuple[str, ...]] = []
        self.successful: list[tuple[str, ...]] = []
        self.lock = threading.Lock()


@contextmanager
def _provider(state: _ProviderState) -> Iterator[str]:
    """Serve OpenAI-compatible vectors on loopback and optionally sever one request."""

    class Handler(BaseHTTPRequestHandler):
        """Keep fixture HTTP logging quiet while recording exact request inputs."""

        def log_message(self, format: str, *args: str) -> None:
            """Suppress the standard handler's stderr request log."""

        def do_POST(self) -> None:
            """Return a valid vector batch or interrupt transport before a response."""
            assert self.path == "/v1/embeddings"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = tuple(body["input"])
            assert len(texts) <= 100
            with state.lock:
                state.requests.append(texts)
                fail = len(state.requests) == state.fail_request
                if not fail:
                    state.successful.append(texts)
            if fail:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            time.sleep(0.02)
            payload = canonical_json_bytes(
                {"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(len(texts))]}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            worker.join(timeout=5)


def _source(store: ArtifactStore, count: int) -> str:
    """Persist synthetic multi-transition evidence for the actual index-building boundary."""
    traces: list[Trace] = []
    for index in range(count):
        traces.append(
            Trace(
                trace_id=f"fixture-{index}",
                task=f"private fixture request {index}",
                spans=(
                    TraceSpan(
                        span_id=f"call-{index}",
                        name="chat",
                        started_at=_CREATED_AT,
                        ended_at=_CREATED_AT + timedelta(seconds=1),
                        attributes={
                            "gen_ai.operation.name": "chat",
                            "gen_ai.tool.name": "lookup",
                            "gen_ai.tool.call.id": f"tool-{index}",
                            "gen_ai.tool.call.arguments": {"item": index},
                        },
                    ),
                    TraceSpan(
                        span_id=f"result-{index}",
                        name="tool",
                        started_at=_CREATED_AT + timedelta(seconds=2),
                        ended_at=_CREATED_AT + timedelta(seconds=3),
                        attributes={
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": "lookup",
                            "gen_ai.tool.call.id": f"tool-{index}",
                            "gen_ai.tool.message": f"observed result {index}",
                        },
                    ),
                ),
                source=TraceSource(
                    identity=SourceIdentity(kind="otlp", source_id="fixture", sha256="a" * 64),
                    semantic_convention_version="1.37.0",
                ),
            )
        )
    result = persist_trace_dataset(
        TraceNormalizationResult(traces=tuple(traces), issues=()),
        store,
        created_at=_CREATED_AT,
        code_revision="fixture",
    )
    return result.manifest.artifact_id


def _build_in_process(root: str, endpoint: str, source_id: str, result_path: str) -> None:
    """Run the real client and persisted build with no inherited cache in a spawned process."""
    store = ArtifactStore(ProjectPaths(Path(root), "checkpoint"))
    snapshot = default_rag_embedder().snapshot.model_copy(
        update={"connection_sha256": sha256_json({"base_url": endpoint})}
    )
    client = OpenAICompatibleClient(
        model=snapshot,
        api_key="loopback-fixture",
        base_url=endpoint,
        retry_policy=RetryPolicy(maximum_attempts=1),
        timeout_seconds=5,
    )
    source_input = artifact_input(store.read(source_id).manifest)
    lineages = tuple(
        RAGLineageBinding(
            trace_id=f"fixture-{index}",
            lineage_id=f"lineage-{index}",
            partition="fit" if index % 2 == 0 else "held_out",
        )
        for index in range(120)
    )
    binding = replace(default_rag_embedder(), client=client, snapshot=snapshot)
    try:
        partition_sets: tuple[frozenset[Literal["fit", "held_out"]], ...] = (
            frozenset({"fit", "held_out"}),
            frozenset({"fit"}),
        )
        for partitions in partition_sets:
            persist_trace_rag(
                store,
                (source_input,),
                lineages,
                created_at=_CREATED_AT,
                code_revision="fixture",
                embedder=binding,
                included_partitions=partitions,
            )
    except ProviderTransportError:
        Path(result_path).write_text("transport-failure")
    else:
        Path(result_path).write_text("completed")


def test_loopback_build_resumes_successful_batches_after_process_restart(tmp_path: Path) -> None:
    """A failed second request preserves the first batch across process restart and fit reuse."""
    root = tmp_path / ".exp"
    store = ArtifactStore(ProjectPaths(root, "checkpoint"))
    source_id = _source(store, 120)
    state = _ProviderState(fail_request=2)
    context = multiprocessing.get_context("spawn")
    result = tmp_path / "result.txt"
    with _provider(state) as endpoint:
        for expected in ("transport-failure", "completed"):
            process = context.Process(
                target=_build_in_process, args=(str(root), endpoint, source_id, str(result))
            )
            process.start()
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join()
            assert process.exitcode == 0
            assert result.read_text() == expected
            if expected == "transport-failure":
                database = next(
                    (store.project_directory / "runtime" / "rag-embeddings").glob("*.sqlite3")
                )
                with closing(sqlite3.connect(database)) as connection:
                    assert connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 100
    assert len(state.requests) == 4
    assert state.requests[1] == state.requests[2]
    successful = [text for batch in state.successful for text in batch]
    assert len(successful) == 240
    assert len(set(successful)) == len(successful)
    indexes = (store.project_directory / "artifacts").glob("trace-rag-*")
    assert len([path for path in indexes if path.is_dir()]) == 2


def test_two_build_processes_share_completed_batches_without_duplicate_calls(
    tmp_path: Path,
) -> None:
    """Independent concurrent builders dispatch each exact input once after lock rechecking."""
    root = tmp_path / ".exp"
    store = ArtifactStore(ProjectPaths(root, "checkpoint"))
    source_id = _source(store, 120)
    state = _ProviderState()
    context = multiprocessing.get_context("spawn")
    outputs = [tmp_path / f"result-{index}.txt" for index in range(2)]
    with _provider(state) as endpoint:
        processes = [
            context.Process(
                target=_build_in_process, args=(str(root), endpoint, source_id, str(output))
            )
            for output in outputs
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join()
            assert process.exitcode == 0
    assert all(output.read_text() == "completed" for output in outputs)
    successful = [text for batch in state.successful for text in batch]
    assert len(successful) == len(set(successful)) == 240

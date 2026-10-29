"""Project-scoped embedding checkpoints in shared SQLite without stored input text.

A process lock covers cache lookup, provider dispatch, and durable commit. SQLite transactions
cover metadata only. Completed batches survive failure and restart. An interrupted request whose
response was never committed remains uncertain and may be sent again; the cache does not promise
exactly-once provider execution.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

from filelock import FileLock, Timeout

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import Embedding, ModelSnapshot
from exp.common.project.database import content_database_path
from exp.common.project.paths import ProjectPaths
from exp.common.project.records import ProjectRecordError, ProjectRecords
from exp.common.sqlite.schema import ContentSchemaError
from exp.simulation.retrieval.contracts import RAG_KEY_SCHEMA_VERSION

_LOCK_TIMEOUT_SECONDS = 10
Vector = tuple[float, ...]
EmbeddingBatch = Callable[[Sequence[str]], tuple[Vector, ...]]


class RAGEmbeddingCheckpointError(ValueError):
    """Saved embedding state is corrupt, incompatible, or unavailable for safe reuse."""


class RAGEmbeddingCheckpoint:
    """Durable vectors under one exact project, model, connection, and chunk identity.

    Attributes:
        path: Shared content database containing hashes and validated vectors only.
        lock_path: Per-namespace process coordination file; contains no checkpoint data.
        records: Project-scoped metadata and validated vector records.
        identity: Canonical metadata binding the embedding representation.
        snapshot: Exact model and secret-free connection whose vectors can be reused.
        maximum_chunk_bytes: Maximum UTF-8 input size admitted to this namespace.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        snapshot: ModelSnapshot,
        *,
        maximum_chunk_bytes: int,
    ) -> None:
        """Choose an exact project namespace without opening a provider connection.

        Args:
            paths: Explicit content root and project identity.
            snapshot: Immutable model and connection identity.
            maximum_chunk_bytes: Largest admitted UTF-8 input chunk.
        """
        self.snapshot = snapshot
        self.identity = canonical_json_bytes(
            {
                "key_schema_version": RAG_KEY_SCHEMA_VERSION,
                "embedder": snapshot.model_dump(mode="json"),
                "maximum_chunk_bytes": maximum_chunk_bytes,
            }
        ).decode("utf-8")
        self.maximum_chunk_bytes = maximum_chunk_bytes
        namespace = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        self._paths = paths
        self.path = content_database_path(paths.root)
        self.lock_path = paths.runtime_directory / "rag-embeddings" / f"{namespace}.lock"
        self.records = ProjectRecords(paths.root, paths.project_id, f"rag-embeddings/{namespace}")

    def load(self, texts: Sequence[str]) -> dict[str, Vector]:
        """Read and validate completed inputs before reporting resumed progress."""
        with self._locked():
            return self._read(texts)

    def seed(self, vectors: Mapping[str, Vector]) -> None:
        """Checkpoint an already populated invocation cache, rejecting conflicting values."""
        if not vectors:
            return
        with self._locked():
            existing = self._read(tuple(vectors))
            if any(existing[text] != vectors[text] for text in existing):
                raise RAGEmbeddingCheckpointError(
                    "RAG embedding memory cache differs from its saved checkpoint"
                )
            self._write({text: vector for text, vector in vectors.items() if text not in existing})

    def get_or_embed(self, texts: Sequence[str], embed: EmbeddingBatch) -> tuple[Vector, ...]:
        """Serialize missing work and durably commit a whole validated batch.

        Args:
            texts: A bounded batch of exact provider inputs.
            embed: Provider boundary called outside SQLite transactions for absent inputs.

        Returns:
            Validated cached and newly committed vectors in the requested order.

        Raises:
            RAGEmbeddingCheckpointError: Saved state cannot be safely reused or committed.
            ValueError: Provider vectors violate their count, shape, or unit-vector contract.
        """
        with self._locked():
            vectors = self._read(texts)
            missing = tuple(dict.fromkeys(text for text in texts if text not in vectors))
            if missing:
                generated = embed(missing)
                if len(generated) != len(missing):
                    raise ValueError(
                        "RAG embedder returned a vector count different from its inputs"
                    )
                completed = dict(zip(missing, generated, strict=True))
                self._write(completed)
                vectors.update(completed)
            return tuple(vectors[text] for text in texts)

    def _read(self, texts: Sequence[str]) -> dict[str, Vector]:
        """Verify record digests, input bindings and vectors without storing lookup text."""
        dimensions = self._dimensions()
        result: dict[str, Vector] = {}
        for text in texts:
            key = self._text_digest(text)
            payload = self.records.read(f"vector/{key}")
            if payload is None:
                continue
            try:
                saved = json.loads(payload)
                if saved["identity"] != self.identity or saved["key"] != key:
                    raise ValueError("vector identity differs")
                vector = Embedding.model_validate(saved["embedding"]).values
                if len(vector) != dimensions:
                    raise ValueError("vector dimensions differ")
            except (ValueError, KeyError, TypeError) as exc:
                raise RAGEmbeddingCheckpointError(
                    "RAG embedding checkpoint vector is invalid"
                ) from exc
            result[text] = vector
        return result

    def _write(self, vectors: Mapping[str, Vector]) -> None:
        """Validate and commit one whole batch with its namespace dimensionality."""
        if not vectors:
            return
        with self.records.transaction():
            dimensions = self._dimensions()
            for text, vector in vectors.items():
                validated = Embedding(values=vector)
                if dimensions is None:
                    dimensions = len(validated.values)
                if len(validated.values) != dimensions:
                    raise ValueError("RAG embedder returned vectors with inconsistent dimensions")
                key = self._text_digest(text)
                self.records.write(
                    f"vector/{key}",
                    canonical_json_bytes(
                        {
                            "identity": self.identity,
                            "key": key,
                            "embedding": validated.model_dump(mode="json"),
                        }
                    ),
                    exclusive=True,
                )
            self.records.write(
                "metadata",
                canonical_json_bytes({"identity": self.identity, "dimensions": dimensions}),
            )

    def _dimensions(self) -> int | None:
        """Return validated namespace dimensionality, including the empty-cache sentinel."""
        payload = self.records.read("metadata")
        if payload is None:
            if self.records.list_ids():
                raise RAGEmbeddingCheckpointError("RAG embedding checkpoint metadata is missing")
            return None
        try:
            saved = json.loads(payload)
            dimensions = saved["dimensions"]
            if saved["identity"] != self.identity:
                raise ValueError("identity differs")
            if dimensions is not None and (type(dimensions) is not int or dimensions < 1):
                raise ValueError("dimensions are invalid")
            return dimensions
        except (ValueError, KeyError, TypeError) as exc:
            raise RAGEmbeddingCheckpointError(
                "RAG embedding checkpoint metadata is invalid"
            ) from exc

    def _text_digest(self, text: str) -> str:
        """Hash an admitted input without ever writing the input itself."""
        payload = text.encode("utf-8")
        if not payload or len(payload) > self.maximum_chunk_bytes:
            raise ValueError("RAG embedding checkpoint needs nonempty bounded input chunks")
        return hashlib.sha256(payload).hexdigest()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Keep provider work serialized without holding the shared database write lock."""
        self._prepare_path()
        lock = FileLock(self.lock_path, timeout=_LOCK_TIMEOUT_SECONDS, mode=0o600)
        try:
            lock.acquire()
        except Timeout:
            raise RAGEmbeddingCheckpointError(
                "another build is embedding this project's inputs; retry after it finishes"
            ) from None
        try:
            self._prepare_path()
            yield
        except (ProjectRecordError, ContentSchemaError, sqlite3.Error) as exc:
            raise RAGEmbeddingCheckpointError(
                f"RAG embedding checkpoint could not be verified or committed at {self.path}; "
                "restore verified evidence before retrying"
            ) from exc
        finally:
            lock.release()

    def _prepare_path(self) -> None:
        """Create private coordination directories, rejecting symlinked descendants."""
        current = self._paths.root
        current.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("projects", self._paths.project_id, "runtime", "rag-embeddings"):
            current /= name
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise RAGEmbeddingCheckpointError("RAG embedding coordination directory is unsafe")
            current.mkdir(mode=0o700, exist_ok=True)
            if current.is_symlink() or not current.is_dir():
                raise RAGEmbeddingCheckpointError("RAG embedding coordination directory is unsafe")
        if self.lock_path.is_symlink() or (
            self.lock_path.exists() and not self.lock_path.is_file()
        ):
            raise RAGEmbeddingCheckpointError("RAG embedding coordination file is unsafe")

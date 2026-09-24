"""Project-local checkpoints for validated embedding batches without stored input text.

A process lock covers cache lookup, provider dispatch, and durable commit. Completed batches
survive failure and restart. An interrupted request whose response was never committed remains
uncertain and may be sent again; the cache does not promise exactly-once provider execution.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import Embedding, ModelSnapshot
from exp.simulation.retrieval.contracts import RAG_KEY_SCHEMA_VERSION

_SCHEMA_VERSION = 1
_LOCK_TIMEOUT_SECONDS = 10
Vector = tuple[float, ...]
EmbeddingBatch = Callable[[Sequence[str]], tuple[Vector, ...]]


class RAGEmbeddingCheckpointError(ValueError):
    """Saved embedding state is corrupt, incompatible, or unavailable for safe reuse."""


class RAGEmbeddingCheckpoint:
    """Durable pure-text vectors under one exact model, connection, and chunk identity.

    Attributes:
        path: Project runtime SQLite file containing hashes and validated vectors only.
        identity: Canonical metadata binding the file to the embedding representation.
        snapshot: Exact model and secret-free connection whose vectors can be reused.
        maximum_chunk_bytes: Maximum UTF-8 input size admitted to this namespace.
    """

    def __init__(
        self,
        project_directory: Path,
        snapshot: ModelSnapshot,
        *,
        maximum_chunk_bytes: int,
    ) -> None:
        """Choose a secret-free namespace without opening a provider connection."""
        self.snapshot = snapshot
        self.identity = canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                "key_schema_version": RAG_KEY_SCHEMA_VERSION,
                "embedder": snapshot.model_dump(mode="json"),
                "maximum_chunk_bytes": maximum_chunk_bytes,
            }
        ).decode("utf-8")
        self.maximum_chunk_bytes = maximum_chunk_bytes
        namespace = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        self._project_directory = project_directory.absolute()
        self.path = self._project_directory / "runtime" / "rag-embeddings" / f"{namespace}.sqlite3"
        self._integrity_checked = False

    def load(self, texts: Sequence[str]) -> dict[str, Vector]:
        """Read and validate completed inputs before reporting resumed progress."""
        with self._locked_connection() as connection:
            return self._read(connection, texts)

    def seed(self, vectors: Mapping[str, Vector]) -> None:
        """Checkpoint an already populated invocation cache, rejecting conflicting values."""
        if not vectors:
            return
        with self._locked_connection() as connection:
            existing = self._read(connection, tuple(vectors))
            if any(existing[text] != vectors[text] for text in existing):
                raise RAGEmbeddingCheckpointError(
                    "RAG embedding memory cache differs from its saved checkpoint"
                )
            self._write(
                connection,
                {text: vector for text, vector in vectors.items() if text not in existing},
            )

    def get_or_embed(self, texts: Sequence[str], embed: EmbeddingBatch) -> tuple[Vector, ...]:
        """Serialize missing work and commit a validated batch before returning its vectors.

        Args:
            texts: A bounded batch of exact provider inputs.
            embed: Provider boundary called only for inputs absent after acquiring the lock.

        Returns:
            Validated cached and newly committed vectors in the requested order.

        Raises:
            RAGEmbeddingCheckpointError: Saved state cannot be safely reused or committed.
            ValueError: Provider vectors violate their count, shape, or unit-vector contract.
        """
        with self._locked_connection() as connection:
            vectors = self._read(connection, texts)
            missing = tuple(dict.fromkeys(text for text in texts if text not in vectors))
            if missing:
                generated = embed(missing)
                if len(generated) != len(missing):
                    raise ValueError(
                        "RAG embedder returned a vector count different from its inputs"
                    )
                completed = dict(zip(missing, generated, strict=True))
                self._write(connection, completed)
                vectors.update(completed)
            return tuple(vectors[text] for text in texts)

    def _read(self, connection: sqlite3.Connection, texts: Sequence[str]) -> dict[str, Vector]:
        """Verify row digests and vector contracts without persisting the lookup text."""
        dimensions = self._dimensions(connection)
        result: dict[str, Vector] = {}
        for text in texts:
            key = self._text_digest(text)
            row = connection.execute(
                "SELECT vector, checksum FROM vectors WHERE text_sha256 = ?", (key,)
            ).fetchone()
            if row is None:
                continue
            payload, checksum = row
            if not isinstance(payload, bytes) or self._checksum(key, payload) != checksum:
                raise RAGEmbeddingCheckpointError("RAG embedding checkpoint vector digest differs")
            try:
                vector = Embedding.model_validate_json(payload).values
            except ValueError as exc:
                raise RAGEmbeddingCheckpointError(
                    "RAG embedding checkpoint vector is invalid"
                ) from exc
            if len(vector) != dimensions:
                raise RAGEmbeddingCheckpointError("RAG embedding checkpoint dimensions differ")
            result[text] = vector
        return result

    def _write(self, connection: sqlite3.Connection, vectors: Mapping[str, Vector]) -> None:
        """Validate a whole batch and commit every row atomically with full SQLite durability."""
        if not vectors:
            return
        dimensions = self._dimensions(connection)
        rows: list[tuple[str, bytes, str]] = []
        for text, vector in vectors.items():
            validated = Embedding(values=vector)
            if dimensions is None:
                dimensions = len(validated.values)
            if len(validated.values) != dimensions:
                raise ValueError("RAG embedder returned vectors with inconsistent dimensions")
            key = self._text_digest(text)
            payload = canonical_json_bytes(validated)
            rows.append((key, payload, self._checksum(key, payload)))
        with connection:
            connection.execute("UPDATE metadata SET dimensions = ? WHERE id = 1", (dimensions,))
            connection.executemany(
                "INSERT INTO vectors (text_sha256, vector, checksum) VALUES (?, ?, ?)", rows
            )

    def _dimensions(self, connection: sqlite3.Connection) -> int | None:
        """Return validated namespace dimensionality, including the empty-cache sentinel."""
        row = connection.execute("SELECT dimensions FROM metadata WHERE id = 1").fetchone()
        if row is None or (row[0] is not None and (not isinstance(row[0], int) or row[0] < 1)):
            raise RAGEmbeddingCheckpointError("RAG embedding checkpoint dimensions are invalid")
        return row[0]

    def _text_digest(self, text: str) -> str:
        """Hash an admitted input without ever writing the input itself."""
        payload = text.encode("utf-8")
        if not payload or len(payload) > self.maximum_chunk_bytes:
            raise ValueError("RAG embedding checkpoint needs nonempty bounded input chunks")
        return hashlib.sha256(payload).hexdigest()

    def _checksum(self, key: str, payload: bytes) -> str:
        """Bind a vector payload to both its text digest and complete namespace identity."""
        return hashlib.sha256(
            self.identity.encode("utf-8") + key.encode("ascii") + payload
        ).hexdigest()

    @contextmanager
    def _locked_connection(self) -> Iterator[sqlite3.Connection]:
        """Keep provider work serialized until its SQLite batch commit is durable."""
        self._prepare_path()
        lock = FileLock(self.path.with_suffix(".lock"), timeout=_LOCK_TIMEOUT_SECONDS, mode=0o600)
        try:
            lock.acquire()
        except Timeout:
            raise RAGEmbeddingCheckpointError(
                "another build is embedding this project's inputs; retry after it finishes"
            ) from None
        except OSError as exc:
            raise RAGEmbeddingCheckpointError(
                "RAG embedding checkpoint cannot be locked; "
                f"check permissions at {self.path.parent}"
            ) from exc
        try:
            self._prepare_path()
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            except OSError as exc:
                raise RAGEmbeddingCheckpointError(
                    f"RAG embedding checkpoint cannot be created at {self.path}; "
                    "check directory permissions and disk space"
                ) from exc
            else:
                os.close(descriptor)
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True)) as connection:
                connection.execute("PRAGMA synchronous = FULL")
                self._initialize(connection)
                yield connection
        except RAGEmbeddingCheckpointError as exc:
            raise RAGEmbeddingCheckpointError(
                f"{exc}; inspect or restore {self.path} before retrying. "
                "Removing the checkpoint will repeat its embedding provider calls."
            ) from exc
        except sqlite3.Error as exc:
            raise RAGEmbeddingCheckpointError(
                f"RAG embedding checkpoint could not be read or committed at {self.path}; "
                "check file permissions, disk space, and cache integrity before retrying"
            ) from exc
        finally:
            lock.release()

    def _initialize(self, connection: sqlite3.Connection) -> None:
        """Initialize an empty database or fail closed on metadata and integrity drift."""
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            if connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone():
                raise RAGEmbeddingCheckpointError("RAG embedding checkpoint schema is incompatible")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK (id = 1), "
                    "identity TEXT NOT NULL, dimensions INTEGER CHECK (dimensions > 0))"
                )
                connection.execute(
                    "CREATE TABLE vectors (text_sha256 TEXT PRIMARY KEY, vector BLOB NOT NULL, "
                    "checksum TEXT NOT NULL)"
                )
                connection.execute("INSERT INTO metadata VALUES (1, ?, NULL)", (self.identity,))
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif version != _SCHEMA_VERSION:
            raise RAGEmbeddingCheckpointError("RAG embedding checkpoint schema is incompatible")
        if not self._integrity_checked:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RAGEmbeddingCheckpointError("RAG embedding checkpoint database is corrupt")
            self._integrity_checked = True
        identity = connection.execute("SELECT identity FROM metadata WHERE id = 1").fetchone()
        if identity != (self.identity,):
            raise RAGEmbeddingCheckpointError("RAG embedding checkpoint identity differs")
        if (
            self._dimensions(connection) is None
            and connection.execute("SELECT 1 FROM vectors LIMIT 1").fetchone()
        ):
            raise RAGEmbeddingCheckpointError("RAG embedding checkpoint dimensions are missing")

    def _prepare_path(self) -> None:
        """Create private project-owned cache directories and reject linked state files."""
        current = self._project_directory
        if current.is_symlink() or not current.is_dir():
            raise RAGEmbeddingCheckpointError("RAG embedding project must be a real directory")
        for name in ("runtime", "rag-embeddings"):
            current /= name
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise RAGEmbeddingCheckpointError("RAG embedding cache directory is unsafe")
            try:
                current.mkdir(mode=0o700, exist_ok=True)
            except OSError as exc:
                raise RAGEmbeddingCheckpointError(
                    f"RAG embedding cache directory cannot be created at {current}; "
                    "check permissions and disk space before retrying"
                ) from exc
            if current.is_symlink() or not current.is_dir():
                raise RAGEmbeddingCheckpointError("RAG embedding cache directory is unsafe")
        for path in (self.path, self.path.with_suffix(".lock")):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise RAGEmbeddingCheckpointError("RAG embedding cache file is unsafe")

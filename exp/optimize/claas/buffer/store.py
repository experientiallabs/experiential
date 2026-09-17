"""Scoped SQLite queue with immutable leases and atomic checkpoint acknowledgements."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from exp.common.claas import Experience, ExperienceProvenance
from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.core.artifacts import sha256_json
from exp.optimize.claas.service.configuration import BufferStatus, RunConfiguration
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingCheckpoint,
    TrainingExample,
    TrainingResult,
    validate_training_batch,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 experience_id TEXT NOT NULL UNIQUE, response_id TEXT NOT NULL UNIQUE,
 experience TEXT NOT NULL, scalar_reward REAL, text_feedback TEXT,
 state TEXT NOT NULL, rejection_reason TEXT, batch_id TEXT,
 request_id TEXT UNIQUE, request TEXT, result TEXT, size_bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
 batch_id TEXT PRIMARY KEY, payload TEXT NOT NULL, result TEXT,
 receipt_limit_bytes INTEGER NOT NULL
);
"""


class ExperienceBuffer:
    """One application recipe's retained records and recoverable optimizer lease."""

    def __init__(
        self,
        path: Path,
        spec: ClaasTrainingSpec,
        limits: RunConfiguration,
        *,
        process_lock: FileLock | None = None,
    ) -> None:
        """Bind a new database or reject a different recipe without modifying it."""
        self.path = path.resolve()
        self.spec = spec
        self.limits = limits
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock = process_lock or FileLock(
            self.path.parent / "run.lock", timeout=0, mode=0o600
        )
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            identity = sha256_json(spec)
            previous = connection.execute(
                "SELECT value FROM metadata WHERE key='recipe'"
            ).fetchone()
            if previous is not None and previous[0] != identity:
                raise ValueError("buffer recipe differs; choose a fresh run directory")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('recipe', ?)", (identity,))
            self._check_capacity(connection)
            connection.commit()
            self.path.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Use bounded connections with complete single-file SQLite commits."""
        try:
            self._process_lock.acquire()
        except Timeout:
            raise ValueError(
                "another learner owns this run directory; use its API or stop it before importing"
            ) from None
        try:
            connection = sqlite3.connect(self.path, timeout=1.0, isolation_level=None)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                yield connection
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        finally:
            self._process_lock.release()

    def checkpoint(self) -> TrainingCheckpoint | None:
        """Return only the checkpoint committed atomically with queue consumption."""
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='checkpoint'").fetchone()
        return TrainingCheckpoint.model_validate_json(row[0]) if row else None

    def status(self) -> BufferStatus:
        """Count every retained state and expose rejection reasons without raw content."""
        with self._connect() as connection:
            counts = dict(connection.execute("SELECT state, COUNT(*) FROM records GROUP BY state"))
            reasons = dict(
                connection.execute(
                    "SELECT rejection_reason, COUNT(*) FROM records WHERE state='rejected' "
                    "GROUP BY rejection_reason"
                )
            )
            size = self._retained_bytes(connection)
        return BufferStatus(**counts, retained_bytes=size, rejection_reasons=reasons)

    def ready_ids(self) -> tuple[str, ...]:
        """Freeze ordered ready identities for a finite drain snapshot."""
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT experience_id FROM records WHERE state='ready' ORDER BY sequence"
                )
            )

    def replay(self, request: GenerationRequest) -> GenerationResult | None:
        """Replay a completed request exactly, rejecting identifier reuse with different input."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT request, result FROM records WHERE request_id=?", (request.request_id,)
            ).fetchone()
        if row is None:
            return None
        if row[0] != request.model_dump_json():
            raise ValueError("request_id already belongs to a different generation request")
        return GenerationResult.model_validate_json(row[1])

    def record_generation(self, request: GenerationRequest, result: GenerationResult) -> None:
        """Persist a completed exact sample before acknowledging the generation request."""
        experience = Experience(
            experience_id=result.response_id,
            response_id=result.response_id,
            scope=self.spec.scope,
            protocol="chat_completions",
            captured_at=datetime.now(UTC),
            request=request.model_dump(mode="json"),
            response=result.model_dump(mode="json"),
            provenance=ExperienceProvenance(
                source_kind="traffic",
                source_id=request.request_id,
                model_id=result.exact_tokens.model_id,
                model_revision=result.exact_tokens.model_revision,
                policy_revision=result.exact_tokens.policy_revision,
            ),
            exact_tokens=result.exact_tokens,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert(connection, experience, None, None, request=request, result=result)
            self._check_capacity(connection)
            connection.commit()

    def import_examples(self, examples: tuple[TrainingExample, ...]) -> None:
        """Import the complete set atomically, retaining explicit unsupported-data rejection."""
        if not examples:
            raise ValueError("import requires at least one training example")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for example in examples:
                if example.experience.scope != self.spec.scope:
                    raise ValueError("import belongs to another application scope")
                self._insert(
                    connection, example.experience, example.scalar_reward, example.text_feedback
                )
            self._check_capacity(connection)
            connection.commit()

    def feedback(
        self, response_id: str, *, scalar_reward: float | None, text_feedback: str | None
    ) -> None:
        """Join feedback atomically; exact retries are safe after lease or consumption."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM records WHERE response_id=?", (response_id,)
            ).fetchone()
            if row is None:
                raise ValueError(
                    "unknown response_id; generate or import its exact experience first"
                )
            if scalar_reward is None and text_feedback is None:
                raise ValueError("provide scalar_reward or text_feedback")
            scalar = row["scalar_reward"] if scalar_reward is None else scalar_reward
            text = row["text_feedback"] if text_feedback is None else text_feedback
            example = TrainingExample(
                experience=Experience.model_validate_json(row["experience"]),
                scalar_reward=scalar,
                text_feedback=text,
            )
            if row["scalar_reward"] == scalar and row["text_feedback"] == text:
                connection.commit()
                return
            if row["state"] not in {"pending_feedback", "ready"}:
                raise ValueError("feedback is frozen after lease, consumption, or rejection")
            if (row["scalar_reward"] is not None and row["scalar_reward"] != scalar) or (
                row["text_feedback"] is not None and row["text_feedback"] != text
            ):
                raise ValueError(
                    "feedback conflicts with an existing signal; submit a new experience"
                )
            state, reason = self._classify(example.experience, scalar, text)
            size = self._size(row["experience"], scalar, text, row["request"], row["result"])
            connection.execute(
                "UPDATE records SET scalar_reward=?, text_feedback=?, state=?, rejection_reason=?, "
                "size_bytes=? WHERE response_id=?",
                (scalar, text, state, reason, size, response_id),
            )
            self._check_capacity(connection)
            connection.commit()

    def _classify(
        self, experience: Experience, scalar: float | None, text: str | None
    ) -> tuple[str, str | None]:
        """Retain missing feedback separately from permanently unsupported or stale samples."""
        # Validate evidence with placeholder signals before waiting on actual feedback.
        example = TrainingExample(
            experience=experience,
            scalar_reward=scalar if scalar is not None else 0,
            text_feedback=text if text is not None else "pending",
        )
        checkpoint = self.checkpoint()
        revision = checkpoint.policy_revision if checkpoint else self.spec.initial_policy_revision
        try:
            validate_training_batch(
                self.spec,
                TrainingBatch(
                    batch_id="validation", expected_policy_revision=revision, examples=(example,)
                ),
                checkpoint,
            )
        except ValueError as error:
            return "rejected", str(error)
        ready = (self.spec.objective == "reinforce" or text is not None) and (
            self.spec.objective == "sdpo" or scalar is not None
        )
        return ("ready" if ready else "pending_feedback"), None

    def _insert(
        self,
        connection: sqlite3.Connection,
        experience: Experience,
        scalar: float | None,
        text: str | None,
        *,
        request: GenerationRequest | None = None,
        result: GenerationResult | None = None,
    ) -> None:
        """Insert one immutable record or accept an exact import replay."""
        payload = experience.model_dump_json()
        request_json = request.model_dump_json() if request else None
        result_json = result.model_dump_json() if result else None
        previous = connection.execute(
            "SELECT * FROM records WHERE experience_id=? OR response_id=?",
            (experience.experience_id, experience.response_id),
        ).fetchone()
        if previous is not None:
            if (
                previous["experience"] == payload
                and previous["scalar_reward"] == scalar
                and previous["text_feedback"] == text
                and request is None
            ):
                return
            raise ValueError(
                "experience or response identity already has different immutable content"
            )
        state, reason = self._classify(experience, scalar, text)
        size = self._size(payload, scalar, text, request_json, result_json)
        connection.execute(
            "INSERT INTO records (experience_id,response_id,experience,scalar_reward,"
            "text_feedback,state,"
            "rejection_reason,request_id,request,result,size_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                experience.experience_id,
                experience.response_id,
                payload,
                scalar,
                text,
                state,
                reason,
                request.request_id if request else None,
                request_json,
                result_json,
                size,
            ),
        )

    @staticmethod
    def _size(
        payload: str,
        scalar: float | None,
        text: str | None,
        request: str | None,
        result: str | None,
    ) -> int:
        """Count every retained content payload conservatively, including response replay."""
        return (
            sum(len(value.encode()) for value in (payload, text or "", request or "", result or ""))
            + 512
        )

    def _check_capacity(self, connection: sqlite3.Connection) -> None:
        """Fail the entire mutation before commit when any configured bound is exceeded."""
        count, _, largest = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes),0), COALESCE(MAX(size_bytes),0) FROM records"
        ).fetchone()
        if (
            count > self.limits.maximum_buffer_records
            or self._retained_bytes(connection) > self.limits.maximum_buffer_bytes
        ):
            raise ValueError(
                "experience buffer is full; archive this run and choose a fresh directory"
            )
        if largest > self.limits.maximum_record_bytes:
            raise ValueError(
                "experience exceeds maximum_record_bytes; reduce the request or feedback"
            )

    @staticmethod
    def _retained_bytes(connection: sqlite3.Connection) -> int:
        """Count retained payloads and reserve receipt/checkpoint capacity before training.

        Each unacknowledged lease reserves its frozen receipt limit twice, once
        for the TrainingResult and once for the checkpoint metadata. A successful
        acknowledgement replaces both reservations with the actual byte counts.
        SQLite pages and indexes are outside this content-retention ceiling.
        """
        records = connection.execute("SELECT COALESCE(SUM(size_bytes),0) FROM records").fetchone()[
            0
        ]
        batches = connection.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(payload AS BLOB)) + "
            "COALESCE(LENGTH(CAST(result AS BLOB)),2 * receipt_limit_bytes)),0) FROM batches"
        ).fetchone()[0]
        metadata = connection.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(value AS BLOB))),0) FROM metadata"
        ).fetchone()[0]
        return records + batches + metadata

    def inflight(self) -> TrainingBatch | None:
        """Recover the exact unacknowledged batch instead of creating another optimization ID."""
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM batches WHERE result IS NULL").fetchall()
        if len(rows) > 1:
            raise ValueError("buffer contains multiple in-flight batches; restore consistent state")
        return TrainingBatch.model_validate_json(rows[0][0]) if rows else None

    def lease(self, *, allowed_ids: tuple[str, ...] | None = None) -> TrainingBatch | None:
        """Freeze one bounded batch, explicitly rejecting examples that became stale."""
        allowed = set(allowed_ids) if allowed_ids is not None else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = self.inflight()
            if pending is not None:
                connection.commit()
                return pending
            checkpoint = self.checkpoint()
            revision = (
                checkpoint.policy_revision if checkpoint else self.spec.initial_policy_revision
            )
            rows = connection.execute(
                "SELECT * FROM records WHERE state='ready' ORDER BY sequence"
            ).fetchall()
            examples: list[TrainingExample] = []
            tokens = 0
            for row in rows:
                if allowed is not None and row["experience_id"] not in allowed:
                    continue
                experience = Experience.model_validate_json(row["experience"])
                state, reason = self._classify(
                    experience, row["scalar_reward"], row["text_feedback"]
                )
                if state != "ready":
                    connection.execute(
                        "UPDATE records SET state=?, rejection_reason=? WHERE experience_id=?",
                        (state, reason, experience.experience_id),
                    )
                    continue
                exact = experience.exact_tokens
                if exact is None:
                    raise ValueError("ready experience lacks exact token evidence")
                length = len(exact.prompt_token_ids) + len(exact.response_token_ids)
                if tokens + length > self.spec.max_batch_tokens:
                    break
                examples.append(
                    TrainingExample(
                        experience=experience,
                        scalar_reward=row["scalar_reward"],
                        text_feedback=row["text_feedback"],
                    )
                )
                tokens += length
                if len(examples) == self.spec.max_batch_examples:
                    break
            if not examples:
                connection.commit()
                return None
            batch = TrainingBatch(
                batch_id=uuid4().hex, expected_policy_revision=revision, examples=tuple(examples)
            )
            validate_training_batch(self.spec, batch, checkpoint)
            connection.execute(
                "INSERT INTO batches VALUES (?, ?, NULL, ?)",
                (batch.batch_id, batch.model_dump_json(), self.limits.maximum_update_receipt_bytes),
            )
            connection.executemany(
                "UPDATE records SET state='inflight', batch_id=? WHERE experience_id=?",
                [(batch.batch_id, item.experience.experience_id) for item in examples],
            )
            self._check_capacity(connection)
            connection.commit()
            return batch

    def acknowledge(self, batch: TrainingBatch, result: TrainingResult) -> None:
        """Commit checkpoint and consumed identities together, permitting only exact retry."""
        checkpoint = result.checkpoint
        expected_ids = tuple(item.experience.experience_id for item in batch.examples)
        previous = self.checkpoint()
        previous_step = previous.step if previous else 0
        if (
            result.consumed_experience_ids != expected_ids
            or checkpoint.scope != self.spec.scope
            or checkpoint.adapter_id != self.spec.adapter_id
            or not checkpoint.policy_history
            or checkpoint.policy_history[0] != checkpoint.policy_revision
            or len(checkpoint.policy_history) < 2
            or batch.expected_policy_revision != checkpoint.policy_history[1]
        ):
            raise ValueError("training result differs from its leased batch or checkpoint lineage")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload,result,receipt_limit_bytes FROM batches WHERE batch_id=?",
                (batch.batch_id,),
            ).fetchone()
            if row is None or row[0] != batch.model_dump_json():
                raise ValueError("training result has no identical durable batch lease")
            payload = result.model_dump_json()
            checkpoint_payload = checkpoint.model_dump_json()
            if max(len(payload.encode()), len(checkpoint_payload.encode())) > row[2]:
                raise ValueError(
                    "training receipt exceeds its reserved byte limit; inspect the runtime output"
                )
            if row[1] is not None:
                if row[1] != payload:
                    raise ValueError("training batch already has a different acknowledgement")
                connection.commit()
                return
            if checkpoint.step != previous_step + 1:
                raise ValueError("training checkpoint does not advance exactly one update")
            connection.execute(
                "UPDATE batches SET result=? WHERE batch_id=?", (payload, batch.batch_id)
            )
            connection.execute(
                "UPDATE records SET state='consumed' WHERE batch_id=?", (batch.batch_id,)
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('checkpoint', ?)",
                (checkpoint_payload,),
            )
            self._check_capacity(connection)
            connection.commit()

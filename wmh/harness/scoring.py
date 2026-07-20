"""Immutable, evaluator-neutral score evidence for harness optimization."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.harness.doc import HarnessDoc

_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_SAFE_ARTIFACT_PART = r"^[A-Za-z0-9._-]+$"
MAX_CELL_SUMMARY_CHARS = 16_000


class ScoreContext(BaseModel):
    """Content identities that make one evaluation request replayable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_set_digest: str = Field(strict=True, pattern=_SHA256_PATTERN)
    evaluator_digest: str = Field(strict=True, pattern=_SHA256_PATTERN)
    execution_config_digest: str = Field(strict=True, pattern=_SHA256_PATTERN)

    @property
    def context_hash(self) -> str:
        """Return a stable identity for the complete scoring context."""
        return _sha256(self.model_dump_json())


class ScoreRequest(BaseModel):
    """One exact task-by-attempt matrix evaluated under a frozen context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: ScoreContext
    task_ids: tuple[str, ...]
    attempts: int = Field(strict=True, ge=1)

    @field_validator("attempts", mode="before")
    @classmethod
    def _reject_boolean_attempts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempts must be an integer, not boolean")
        return value

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("task_ids must be nonempty")
        if len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        for task_id in value:
            if not task_id:
                raise ValueError("task_ids must not contain empty values")
            if len(task_id) > 512:
                raise ValueError("task_ids must not contain values longer than 512 characters")
            validate_durable_text(task_id, field="task id")
        return value


class EvaluationArtifact(BaseModel):
    """Content-addressed reference to evaluator-owned raw evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(strict=True)
    content_hash: str = Field(strict=True, pattern=_SHA256_PATTERN)
    size_bytes: int = Field(strict=True, ge=0)
    media_type: str = Field(default="text/plain", strict=True, min_length=1, max_length=256)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        _validate_artifact_path(value)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _reject_boolean_size(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("artifact size must be an integer, not boolean")
        return value

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str) -> str:
        validate_durable_text(value, field="artifact media type")
        return value

    @classmethod
    def from_text(
        cls,
        *,
        path: str,
        content: str,
        media_type: str = "text/plain",
    ) -> Self:
        """Build a manifest entry without retaining the potentially large content."""
        validate_durable_text(content, field=f"artifact {path!r} content")
        return cls.from_bytes(path=path, content=content.encode("utf-8"), media_type=media_type)

    @classmethod
    def from_bytes(
        cls,
        *,
        path: str,
        content: bytes,
        media_type: str = "application/octet-stream",
    ) -> Self:
        """Build a manifest entry for arbitrary evaluator evidence without retaining it."""
        return cls(
            path=path,
            content_hash=_sha256_bytes(content),
            size_bytes=len(content),
            media_type=media_type,
        )


class ScoreCell(BaseModel):
    """Normalized score and evidence index for one task attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    task_id: str = Field(strict=True, min_length=1, max_length=512)
    attempt: int = Field(strict=True, ge=1)
    score: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    # The evaluator owns pass semantics. Optimization ranks by ``score``; ``passed`` is retained
    # as diagnostic and reporting evidence rather than silently deriving a second objective.
    passed: bool = Field(strict=True)
    summary: str = Field(default="", strict=True, max_length=MAX_CELL_SUMMARY_CHARS)
    artifact_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("attempt", mode="before")
    @classmethod
    def _reject_boolean_attempt(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempt must be an integer, not boolean")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def _reject_boolean_score(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("score must be numeric, not boolean")
        return value

    @field_validator("task_id", "summary")
    @classmethod
    def _validate_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "score cell")
        validate_durable_text(value, field=str(field_name))
        return value

    @field_validator("artifact_paths")
    @classmethod
    def _validate_artifact_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("artifact_paths must be unique")
        for path in value:
            _validate_artifact_path(path)
        return tuple(sorted(value))


class HarnessScoreReport(BaseModel):
    """Canonical scorecard for one exact candidate and evaluation request."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source_run_id: str = Field(strict=True, min_length=1, max_length=1024)
    candidate_execution_hash: str = Field(strict=True, pattern=r"^[0-9a-f]{32}$")
    request: ScoreRequest
    cells: tuple[ScoreCell, ...]
    artifacts: tuple[EvaluationArtifact, ...]

    @field_validator("source_run_id", "candidate_execution_hash")
    @classmethod
    def _validate_identity_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "score identity")
        validate_durable_text(value, field=str(field_name))
        return value

    @field_validator("cells")
    @classmethod
    def _canonicalize_cells(cls, value: tuple[ScoreCell, ...]) -> tuple[ScoreCell, ...]:
        return tuple(sorted(value, key=lambda cell: (cell.task_id, cell.attempt)))

    @field_validator("artifacts")
    @classmethod
    def _canonicalize_artifacts(
        cls, value: tuple[EvaluationArtifact, ...]
    ) -> tuple[EvaluationArtifact, ...]:
        return tuple(sorted(value, key=lambda artifact: artifact.path))

    @model_validator(mode="after")
    def _validate_matrix_and_artifacts(self) -> Self:
        observed = [(cell.task_id, cell.attempt) for cell in self.cells]
        duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate score cell(s): {duplicates}")
        expected = {
            (task_id, attempt)
            for task_id in self.request.task_ids
            for attempt in range(1, self.request.attempts + 1)
        }
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        if missing or extra:
            raise ValueError(f"score cells do not match request: missing={missing}, extra={extra}")

        paths = [artifact.path for artifact in self.artifacts]
        duplicate_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
        if duplicate_paths:
            raise ValueError(f"duplicate artifact path(s): {duplicate_paths}")
        known_paths = set(paths)
        missing_paths = sorted(
            {path for cell in self.cells for path in cell.artifact_paths if path not in known_paths}
        )
        if missing_paths:
            raise ValueError(f"score cells reference missing artifact(s): {missing_paths}")
        return self

    @property
    def score(self) -> float:
        """Return the primary optimization objective over the complete matrix."""
        return fmean(cell.score for cell in self.cells)

    @property
    def pass_rate(self) -> float:
        """Return the evaluator-authoritative pass fraction for reporting."""
        return fmean(1.0 if cell.passed else 0.0 for cell in self.cells)

    @property
    def report_hash(self) -> str:
        """Commit to source run, candidate, context, cells, and artifact manifests."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return _sha256(encoded)


class ArtifactReader(Protocol):
    """Read evaluator-owned raw evidence by safe relative path."""

    def read_bytes(self, path: str) -> bytes: ...


@dataclass(frozen=True)
class HarnessScore:
    """A durable score report paired with its non-inline raw artifact reader."""

    report: HarnessScoreReport
    artifacts: ArtifactReader


class HarnessScorer(Protocol):
    """Synchronous evaluator injected into a harness optimization loop."""

    def score(
        self,
        candidate: HarnessDoc,
        *,
        request: ScoreRequest,
    ) -> HarnessScore: ...


def score_harness(
    scorer: HarnessScorer,
    candidate: HarnessDoc,
    *,
    request: ScoreRequest,
) -> HarnessScore:
    """Score once, reject identity drift, and detach immutable trusted metadata."""
    scored = scorer.score(candidate, request=request)
    if scored.report.candidate_execution_hash != candidate.doc_hash:
        raise ValueError(
            "scorer returned a candidate execution hash that does not match the scored harness"
        )
    if scored.report.request != request:
        raise ValueError("scorer returned a different score request than the caller supplied")
    snapshot = HarnessScoreReport.model_validate(scored.report.model_dump(mode="python"))
    artifacts = _VerifiedArtifactReader(snapshot.artifacts, scored.artifacts)
    artifacts.verify_all()
    return HarnessScore(report=snapshot, artifacts=artifacts)


class _VerifiedArtifactReader:
    """Check evaluator-owned bytes against the immutable manifest on every read."""

    def __init__(
        self,
        manifest: Sequence[EvaluationArtifact],
        source: ArtifactReader,
    ) -> None:
        self._manifest = {artifact.path: artifact for artifact in manifest}
        self._source = source

    def verify_all(self) -> None:
        """Fail before publication when any manifest entry is absent or changed."""
        for path in sorted(self._manifest):
            self.read_bytes(path)

    def read_bytes(self, path: str) -> bytes:
        """Read and verify one artifact."""
        artifact = self._manifest.get(path)
        if artifact is None:
            raise ValueError(f"artifact {path!r} is not present in the score manifest")
        content = self._source.read_bytes(path)
        if not isinstance(content, bytes):
            raise TypeError(f"artifact {path!r} reader returned non-bytes content")
        if len(content) != artifact.size_bytes:
            raise ValueError(
                f"artifact {path!r} size differs from its manifest: "
                f"expected {artifact.size_bytes}, got {len(content)}"
            )
        digest = _sha256_bytes(content)
        if digest != artifact.content_hash:
            raise ValueError(
                f"artifact {path!r} digest differs from its manifest: "
                f"expected {artifact.content_hash}, got {digest}"
            )
        return content


def _validate_artifact_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"artifact path must be a safe relative POSIX path, got {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"artifact path must be canonical and traversal-free, got {path!r}")
    if any(re.fullmatch(_SAFE_ARTIFACT_PART, part) is None for part in parts):
        raise ValueError(f"artifact path contains unsupported characters: {path!r}")


def _sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()

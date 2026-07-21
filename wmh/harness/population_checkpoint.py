"""Crash-safe local checkpoints for fixed-iteration harness population optimization."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from filelock import BaseFileLock, FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from wmh.core.types import JsonObject
from wmh.harness.archive_io import copy_score_artifacts, write_source_tree, write_text
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.live_session import EventKind, SessionEvent
from wmh.harness.population import (
    PopulationIteration,
    PopulationOptimizationResult,
)
from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
    validate_candidate_turn,
)
from wmh.harness.runtime import TokenUsage
from wmh.harness.scoring import HarnessScore, HarnessScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig

_SCHEMA_VERSION = 1
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_CHECKPOINT_DIR = "checkpoint"
_IDENTITY_FILE = "identity.json"
_CONTROL_FILE = "control.json"
_LOCK_FILE = "run.lock"
_MANIFEST_FILE = "manifest.json"


class PopulationCheckpointError(RuntimeError):
    """A local optimization checkpoint cannot be used safely."""


class PopulationCheckpointLockError(PopulationCheckpointError):
    """Another local process currently owns the optimization run."""


class PopulationCheckpointStateError(PopulationCheckpointError):
    """The checkpoint is partial, complete, or in an invalid state."""


class PopulationCheckpointIdentity(BaseModel):
    """Every immutable input that can change optimization or publication behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = _SCHEMA_VERSION
    output_name: str = Field(min_length=1, max_length=512)
    artifact_root: str = Field(min_length=1)
    seed_reference: str | None = None
    seed_source_tree_hash: str = Field(min_length=1)
    score_request: ScoreRequest
    iterations: int = Field(strict=True, ge=1)
    planned_score_cells: int = Field(strict=True, ge=1)
    max_score_cells: int = Field(strict=True, ge=1)
    harbor_job_template: JsonObject
    meta_provider: ProviderConfig
    agent_provider: ProviderConfig
    optimizer_document_hash: str = Field(min_length=1)
    harness_backend: Literal["local", "e2b"]
    e2b_template: str | None = None
    environment_command_timeout_sec: int = Field(strict=True, ge=1)
    project_timeout_sec: float = Field(gt=0)
    max_source_files: int = Field(strict=True, ge=1)
    max_source_bytes: int = Field(strict=True, ge=1)
    max_history_candidates: int = Field(strict=True, ge=1)
    max_history_bytes: int = Field(strict=True, ge=1)

    @field_validator("iterations", "planned_score_cells", "max_score_cells", mode="before")
    @classmethod
    def _reject_boolean_counts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("checkpoint counts must be integers, not booleans")
        return value

    @model_validator(mode="after")
    def _validate_score_cell_plan(self) -> PopulationCheckpointIdentity:
        expected = (
            (self.iterations + 1) * len(self.score_request.task_ids) * self.score_request.attempts
        )
        if self.planned_score_cells != expected:
            raise ValueError("planned_score_cells must equal (iterations + 1) * tasks * attempts")
        if self.planned_score_cells > self.max_score_cells:
            raise ValueError("planned score cells exceed max_score_cells")
        return self


class CheckpointPublicationIntent(BaseModel):
    """Immutable winner publication target persisted before any visible side effect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    publication_id: str = Field(pattern=_SHA256_PATTERN)
    boundary_hash: str = Field(pattern=_SHA256_PATTERN)
    harness_name: str
    harness_version: int = Field(strict=True, ge=1)
    document_hash: str
    prior_champion_version: int | None = Field(default=None, strict=True, ge=1)
    archive_manifest: str
    outcome_path: str

    @field_validator("archive_manifest", "outcome_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or candidate.as_posix() != value
            or ".." in candidate.parts
        ):
            raise ValueError("checkpoint publication path must be canonical and relative")
        return value


class CheckpointPublication(BaseModel):
    """Durable local publication evidence recorded only at terminal completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    publication_id: str = Field(pattern=_SHA256_PATTERN)
    harness_name: str
    harness_version: int = Field(strict=True, ge=1)
    document_hash: str
    archive_manifest: str
    archive_manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    outcome_path: str
    outcome_hash: str = Field(pattern=_SHA256_PATTERN)


class PopulationCheckpointControl(BaseModel):
    """Small atomic control record; boundary data itself is append-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = _SCHEMA_VERSION
    identity_hash: str = Field(pattern=_SHA256_PATTERN)
    state: Literal["ready", "in_progress", "complete"]
    committed_step: int = Field(strict=True, ge=-1)
    latest_boundary_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    active_kind: Literal["setup", "step", "cleanup", "finalize"] | None = None
    active_step: int | None = Field(default=None, ge=0)
    known_score_cells: int = Field(strict=True, ge=0)
    project_sandbox_usage: SandboxUsage | None = None
    publication_intent: CheckpointPublicationIntent | None = None
    publication: CheckpointPublication | None = None

    @model_validator(mode="after")
    def _validate_state_shape(self) -> PopulationCheckpointControl:
        if (self.committed_step == -1) != (self.latest_boundary_hash is None):
            raise ValueError("checkpoint boundary hash does not match committed_step")
        if self.state == "ready":
            if self.active_kind is not None or self.active_step is not None:
                raise ValueError("ready checkpoint cannot contain an active step")
            if self.publication is not None:
                raise ValueError("ready checkpoint cannot contain publication evidence")
            if self.publication_intent is not None:
                raise ValueError("ready checkpoint cannot contain publication intent")
        elif self.state == "in_progress":
            if self.active_kind is None:
                raise ValueError("in-progress checkpoint requires an active kind")
            if self.active_kind in {"cleanup", "finalize"}:
                if self.active_step is not None:
                    raise ValueError("cleanup or finalization cannot contain an active score step")
            elif self.active_step != self.committed_step + 1:
                raise ValueError("active checkpoint step must follow its committed prefix")
            if self.publication is not None:
                raise ValueError("in-progress checkpoint cannot contain publication evidence")
            if (self.active_kind == "finalize") != (self.publication_intent is not None):
                raise ValueError("finalization state must contain exactly one publication intent")
        else:
            if self.active_kind is not None or self.active_step is not None:
                raise ValueError("complete checkpoint cannot contain an active step")
            if self.publication is None:
                raise ValueError("complete checkpoint requires publication evidence")
            if self.publication_intent is not None:
                raise ValueError("complete checkpoint cannot contain publication intent")
        return self


class _FileRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(strict=True, ge=0)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or candidate.as_posix() != value
            or ".." in candidate.parts
        ):
            raise ValueError("checkpoint file path must be canonical and relative")
        return value


class _SeedManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = _SCHEMA_VERSION
    source_tree_hash: str
    files: tuple[_FileRecord, ...]


class _BoundaryManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = _SCHEMA_VERSION
    index: int = Field(strict=True, ge=0)
    previous_manifest_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    outcome: Literal["seed", "scored", "invalid"]
    candidate_id: str
    source_tree_hash: str | None = None
    document_hash: str | None = None
    score_report_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    score_cells: int = Field(strict=True, ge=0)
    worker_usage: TokenUsage | None = None
    error_reason: str | None = None
    files: tuple[_FileRecord, ...]

    @model_validator(mode="after")
    def _validate_outcome(self) -> _BoundaryManifest:
        if self.index == 0:
            if self.outcome != "seed" or self.candidate_id != "candidate-0000":
                raise ValueError("checkpoint step zero must be candidate-0000 seed evidence")
        elif self.outcome == "seed":
            raise ValueError("only checkpoint step zero can contain seed evidence")
        expected_id = f"candidate-{self.index:04d}"
        if self.candidate_id != expected_id:
            raise ValueError("checkpoint candidate identity does not match its step")
        if self.outcome in {"seed", "scored"}:
            if (
                self.source_tree_hash is None
                or self.document_hash is None
                or self.score_report_hash is None
                or self.score_cells < 1
                or self.error_reason is not None
            ):
                raise ValueError("scored checkpoint boundary is incomplete")
            if self.outcome == "scored" and self.worker_usage is None:
                raise ValueError("scored proposal checkpoint requires reported worker usage")
            if self.outcome == "seed" and self.worker_usage is not None:
                raise ValueError("seed checkpoint cannot contain proposer worker usage")
        elif (
            self.document_hash is not None
            or self.score_report_hash is not None
            or self.score_cells != 0
            or self.error_reason is None
        ):
            raise ValueError("invalid checkpoint boundary has scored fields")
        return self


class _EventRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EventKind
    payload: JsonObject


class _DirectoryArtifactReader:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read_bytes(self, path: str) -> bytes:
        return (self._root / path).read_bytes()


class PopulationCheckpointStore:
    """Own one locked local run and publish only complete optimization boundaries."""

    def __init__(
        self,
        run_dir: Path,
        lock_file: BaseFileLock,
        identity: PopulationCheckpointIdentity,
        control: PopulationCheckpointControl,
        seed: HarnessSourceTree,
        result: PopulationOptimizationResult | None,
    ) -> None:
        self.run_dir = run_dir
        self.root = run_dir / _CHECKPOINT_DIR
        self._lock_file = lock_file
        self.identity = identity
        self.control = control
        self.seed = seed
        self._result = result
        self._segment_base_usage: SandboxUsage | None = None
        self._segment_has_prior_usage = False

    @classmethod
    def create(
        cls,
        run_dir: Path,
        *,
        identity: PopulationCheckpointIdentity,
        seed: HarnessSourceTree,
    ) -> PopulationCheckpointStore:
        """Create and lock one new empty optimization prefix."""
        if seed.tree_hash != identity.seed_source_tree_hash:
            raise ValueError("checkpoint seed source does not match its identity")
        run_dir.mkdir(parents=True, exist_ok=False)
        root = run_dir / _CHECKPOINT_DIR
        root.mkdir()
        lock_file = _acquire_lock(root / _LOCK_FILE)
        try:
            identity_path = root / _IDENTITY_FILE
            _write_json_durable(identity_path, identity.model_dump(mode="json"))
            identity_hash = _hash_file(identity_path)
            seed_dir = root / "seed"
            write_source_tree(seed_dir / "source", seed)
            _sync_tree(seed_dir)
            seed_manifest = _SeedManifest(
                source_tree_hash=seed.tree_hash,
                files=_file_records(seed_dir),
            )
            _write_json_durable(
                seed_dir / _MANIFEST_FILE,
                seed_manifest.model_dump(mode="json"),
            )
            control = PopulationCheckpointControl(
                identity_hash=identity_hash,
                state="ready",
                committed_step=-1,
                known_score_cells=0,
            )
            _write_json_durable(root / _CONTROL_FILE, control.model_dump(mode="json"))
            return cls(run_dir, lock_file, identity, control, seed, None)
        except BaseException:
            _release_lock(lock_file)
            raise

    @classmethod
    def open(cls, run_dir: Path) -> PopulationCheckpointStore:
        """Lock and verify one ready or safely recoverable checkpoint."""
        root = run_dir / _CHECKPOINT_DIR
        if not root.is_dir():
            raise PopulationCheckpointError(f"checkpoint does not exist: {root}")
        lock_file = _acquire_lock(root / _LOCK_FILE)
        try:
            _discard_temporary_files(root)
            identity_path = root / _IDENTITY_FILE
            identity = PopulationCheckpointIdentity.model_validate_json(
                identity_path.read_text(encoding="utf-8")
            )
            control = PopulationCheckpointControl.model_validate_json(
                (root / _CONTROL_FILE).read_text(encoding="utf-8")
            )
            if control.identity_hash != _hash_file(identity_path):
                raise PopulationCheckpointError("checkpoint identity hash differs")
            resumable_cleanup = (
                control.state == "in_progress"
                and control.active_kind == "cleanup"
                and control.active_step is None
            )
            resumable_finalization = (
                control.state == "in_progress"
                and control.active_kind == "finalize"
                and control.active_step is None
            )
            if control.state != "ready" and not resumable_cleanup and not resumable_finalization:
                raise PopulationCheckpointStateError(
                    f"checkpoint state {control.state!r} is not resumable"
                )
            seed = _read_seed(root / "seed", identity)
            store = cls(run_dir, lock_file, identity, control, seed, None)
            store._validate_layout()
            store._result = store._load_committed_result()
            if resumable_cleanup:
                if store._result is None or control.committed_step < 0:
                    raise PopulationCheckpointStateError(
                        "cleanup recovery requires one fully committed boundary"
                    )
                store._replace_control(
                    state="ready",
                    active_kind=None,
                    active_step=None,
                    project_sandbox_usage=None,
                )
            elif resumable_finalization:
                store._validate_publication_intent()
            return store
        except BaseException:
            _release_lock(lock_file)
            raise

    def close(self) -> None:
        """Release this process's exclusive local run lock."""
        _release_lock(self._lock_file)

    def __enter__(self) -> PopulationCheckpointStore:
        self._assert_locked()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def result(self) -> PopulationOptimizationResult | None:
        """Return the fully verified committed prefix, if the seed was scored."""
        return self._result

    @property
    def publication_id(self) -> str:
        """Return the stable identity for publishing this exact committed winner."""
        self._assert_locked()
        if (
            self._result is None
            or self.control.committed_step != self.identity.iterations
            or self.control.latest_boundary_hash is None
        ):
            raise PopulationCheckpointStateError(
                "publication identity requires a fully committed optimization"
            )
        payload = {
            "boundary_hash": self.control.latest_boundary_hash,
            "document_hash": self._result.best.candidate.doc_hash,
            "identity_hash": self.control.identity_hash,
            "output_name": self.identity.output_name,
        }
        return _sha256_bytes(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )

    def assert_identity(self, expected: PopulationCheckpointIdentity) -> None:
        """Reject any current input drift without mutating checkpoint state."""
        self._assert_locked()
        if self.identity == expected:
            return
        stored = self.identity.model_dump(mode="json")
        current = expected.model_dump(mode="json")
        changed = sorted(key for key in stored if stored[key] != current[key])
        raise PopulationCheckpointError("checkpoint identity differs for: " + ", ".join(changed))

    def begin_setup(self) -> None:
        """Persist intent before creating or restoring a spend-capable project."""
        self._require_ready()
        if self.control.committed_step >= self.identity.iterations:
            raise PopulationCheckpointStateError("optimization prefix is already fully committed")
        self._segment_base_usage = self.control.project_sandbox_usage
        self._segment_has_prior_usage = self.control.committed_step >= 0
        self._replace_control(
            state="in_progress",
            active_kind="setup",
            active_step=self.control.committed_step + 1,
            project_sandbox_usage=None,
        )

    def before_step(self, index: int) -> None:
        """Persist intent immediately before one seed or proposal/score step."""
        self._assert_locked()
        expected = self.control.committed_step + 1
        if index != expected:
            raise PopulationCheckpointStateError(
                f"checkpoint expected step {expected}, received {index}"
            )
        if (
            self.control.state != "in_progress"
            or self.control.active_kind != "setup"
            or self.control.active_step != index
        ):
            raise PopulationCheckpointStateError("checkpoint has no matching active setup")
        self._replace_control(
            state="in_progress",
            active_kind="step",
            active_step=index,
            project_sandbox_usage=None,
        )

    def commit_boundary(
        self,
        result: PopulationOptimizationResult,
    ) -> None:
        """Append one verified delta while the live project remains non-resumable."""
        self._assert_locked()
        index = len(result.iterations)
        if index == 0 and len(result.population) != 1:
            raise ValueError("seed checkpoint boundary contains extra population entries")
        if self.control.state != "in_progress" or self.control.active_kind != "step":
            raise PopulationCheckpointStateError("checkpoint has no active optimization step")
        if self.control.active_step != index or index != self.control.committed_step + 1:
            raise PopulationCheckpointStateError("result does not match the active checkpoint step")
        self._validate_result_prefix(result)
        manifest, manifest_hash = self._write_boundary(result)
        known_cells = self.control.known_score_cells + manifest.score_cells
        if known_cells > self.identity.max_score_cells:
            raise PopulationCheckpointStateError("committed score cells exceed max_score_cells")
        self._replace_control(
            state="in_progress",
            committed_step=index,
            latest_boundary_hash=manifest_hash,
            active_kind="cleanup",
            active_step=None,
            known_score_cells=known_cells,
            project_sandbox_usage=None,
        )
        self._result = result

    def finish_project_segment(self, segment_usage: SandboxUsage | None) -> None:
        """Return to ready only after the active project has closed successfully."""
        self._assert_locked()
        if self.control.state != "in_progress" or self.control.active_kind != "cleanup":
            raise PopulationCheckpointStateError(
                "checkpoint is not between committed project steps"
            )
        if self.control.committed_step < 0:
            raise PopulationCheckpointStateError("project segment has no committed boundary")
        self._replace_control(
            state="ready",
            active_kind=None,
            active_step=None,
            project_sandbox_usage=self._total_segment_usage(segment_usage),
        )

    def begin_finalization(
        self,
        *,
        publication_id: str,
        harness_version: int,
        document_hash: str,
        prior_champion_version: int | None,
        archive_manifest: str,
        outcome_path: str,
    ) -> None:
        """Persist intent before archive, store, alias, or outcome publication."""
        self._require_ready()
        if self.control.committed_step != self.identity.iterations:
            raise PopulationCheckpointStateError("optimization prefix is not fully committed")
        if self._result is None:
            raise PopulationCheckpointStateError("optimization result is missing")
        if publication_id != self.publication_id:
            raise ValueError("publication_id differs from the committed optimization")
        if document_hash != self._result.best.candidate.doc_hash:
            raise ValueError("publication document differs from the committed winner")
        boundary_hash = self.control.latest_boundary_hash
        if boundary_hash is None:
            raise PopulationCheckpointStateError("optimization boundary hash is missing")
        intent = CheckpointPublicationIntent(
            publication_id=publication_id,
            boundary_hash=boundary_hash,
            harness_name=self.identity.output_name,
            harness_version=harness_version,
            document_hash=document_hash,
            prior_champion_version=prior_champion_version,
            archive_manifest=archive_manifest,
            outcome_path=outcome_path,
        )
        self._replace_control(
            state="in_progress",
            active_kind="finalize",
            active_step=None,
            publication_intent=intent,
        )

    def mark_complete(
        self,
        *,
        saved: HarnessDoc,
        archive_manifest: Path,
        outcome_path: Path,
    ) -> None:
        """Record terminal publication only after every referenced artifact is durable."""
        self._assert_locked()
        if self.control.state != "in_progress" or self.control.active_kind != "finalize":
            raise PopulationCheckpointStateError("checkpoint is not finalizing")
        intent = self.control.publication_intent
        if intent is None:
            raise PopulationCheckpointStateError("checkpoint publication intent is missing")
        if (
            saved.name != intent.harness_name
            or saved.version != intent.harness_version
            or saved.doc_hash != intent.document_hash
        ):
            raise ValueError("saved harness does not match checkpoint publication intent")
        if self._result is None:
            raise PopulationCheckpointError("checkpoint result is missing during publication")
        expected_saved = self._result.best.candidate.model_copy(
            update={"name": self.identity.output_name, "version": saved.version}
        )
        if saved != expected_saved:
            raise ValueError("saved harness differs from the checkpoint winner")
        if not archive_manifest.is_file() or not outcome_path.is_file():
            raise ValueError("checkpoint publication artifacts are incomplete")
        if (
            _relative_to_run(self.run_dir, archive_manifest) != intent.archive_manifest
            or _relative_to_run(self.run_dir, outcome_path) != intent.outcome_path
        ):
            raise ValueError("checkpoint publication paths differ from intent")
        publication = CheckpointPublication(
            publication_id=intent.publication_id,
            harness_name=saved.name,
            harness_version=saved.version,
            document_hash=saved.doc_hash,
            archive_manifest=_relative_to_run(self.run_dir, archive_manifest),
            archive_manifest_hash=_hash_file(archive_manifest),
            outcome_path=_relative_to_run(self.run_dir, outcome_path),
            outcome_hash=_hash_file(outcome_path),
        )
        self._replace_control(
            state="complete",
            active_kind=None,
            active_step=None,
            publication_intent=None,
            publication=publication,
        )

    def _validate_publication_intent(self) -> None:
        intent = self.control.publication_intent
        if intent is None or self._result is None:
            raise PopulationCheckpointStateError("finalizing checkpoint is incomplete")
        if self.control.committed_step != self.identity.iterations:
            raise PopulationCheckpointStateError(
                "finalizing checkpoint does not contain the complete optimization"
            )
        if (
            intent.publication_id != self.publication_id
            or intent.boundary_hash != self.control.latest_boundary_hash
            or intent.harness_name != self.identity.output_name
            or intent.document_hash != self._result.best.candidate.doc_hash
        ):
            raise PopulationCheckpointStateError(
                "finalizing checkpoint publication intent differs from committed result"
            )

    def _validate_result_prefix(self, result: PopulationOptimizationResult) -> None:
        if self._result is None:
            if self.control.committed_step != -1:
                raise PopulationCheckpointError("checkpoint result prefix is missing")
            if result.population[0].source != self.seed:
                raise ValueError("seed boundary differs from the checkpoint seed")
            if result.population[0].score.report.request != self.identity.score_request:
                raise ValueError("seed boundary uses a different score request")
            return
        prior = self._result
        if len(prior.iterations) != self.control.committed_step:
            raise PopulationCheckpointError("checkpoint prefix length differs from control")
        if len(result.iterations) != len(prior.iterations) + 1:
            raise ValueError("result did not append exactly one consumed slot")
        if _population_signature(result.population[: len(prior.population)]) != (
            _population_signature(prior.population)
        ):
            raise ValueError("result changed the committed evaluated population prefix")
        if _iteration_signature(result.iterations[:-1]) != _iteration_signature(prior.iterations):
            raise ValueError("result changed the committed proposal prefix")

    def _write_boundary(
        self,
        result: PopulationOptimizationResult,
    ) -> tuple[_BoundaryManifest, str]:
        index = len(result.iterations)
        destination = self.root / "steps" / f"{index:04d}"
        destination.mkdir(parents=True, exist_ok=False)
        previous_hash = self.control.latest_boundary_hash
        if index == 0:
            evaluated = result.population[0]
            _write_evaluation(destination, evaluated, source_dir="source")
            manifest_values: dict[str, object] = {
                "index": 0,
                "previous_manifest_hash": previous_hash,
                "outcome": "seed",
                "candidate_id": evaluated.candidate_id,
                "source_tree_hash": evaluated.source.tree_hash,
                "document_hash": evaluated.candidate.doc_hash,
                "score_report_hash": evaluated.score.report.report_hash,
                "score_cells": len(evaluated.score.report.cells),
            }
        else:
            iteration = result.iterations[-1]
            if iteration.index != index:
                raise ValueError("result boundary index is not contiguous")
            if iteration.error is not None:
                turn = iteration.error
                validate_candidate_turn(turn)
                _write_turn(destination / "proposal", turn)
                manifest_values = {
                    "index": index,
                    "previous_manifest_hash": previous_hash,
                    "outcome": "invalid",
                    "candidate_id": turn.candidate_id,
                    "source_tree_hash": (
                        turn.source.tree_hash if turn.source is not None else None
                    ),
                    "score_cells": 0,
                    "worker_usage": turn.worker_usage,
                    "error_reason": turn.reason,
                }
            else:
                assert iteration.proposal is not None
                assert iteration.evaluation is not None
                turn = iteration.proposal
                validate_candidate_turn(turn)
                _write_turn(destination / "proposal", turn)
                _write_evaluation(
                    destination / "evaluation",
                    iteration.evaluation,
                    source_dir=None,
                )
                manifest_values = {
                    "index": index,
                    "previous_manifest_hash": previous_hash,
                    "outcome": "scored",
                    "candidate_id": turn.candidate_id,
                    "source_tree_hash": turn.source.tree_hash,
                    "document_hash": turn.candidate.doc_hash,
                    "score_report_hash": iteration.evaluation.score.report.report_hash,
                    "score_cells": len(iteration.evaluation.score.report.cells),
                    "worker_usage": turn.worker_usage,
                }
        _sync_tree(destination)
        manifest = _BoundaryManifest.model_validate(
            {**manifest_values, "files": _file_records(destination)}
        )
        manifest_path = destination / _MANIFEST_FILE
        _write_json_durable(manifest_path, manifest.model_dump(mode="json"))
        _sync_directory(destination.parent)
        manifest_hash = _hash_file(manifest_path)
        loaded, loaded_hash = _read_boundary_manifest(destination)
        if loaded != manifest or loaded_hash != manifest_hash:
            raise PopulationCheckpointError("checkpoint boundary verification differs after write")
        return manifest, manifest_hash

    def _load_committed_result(self) -> PopulationOptimizationResult | None:
        if self.control.committed_step == -1:
            if self.control.known_score_cells != 0:
                raise PopulationCheckpointError("empty checkpoint has committed score cells")
            return None
        population: list[EvaluatedCandidate] = []
        iterations: list[PopulationIteration] = []
        previous_hash: str | None = None
        known_cells = 0
        for index in range(0, self.control.committed_step + 1):
            directory = self.root / "steps" / f"{index:04d}"
            manifest, manifest_hash = _read_boundary_manifest(directory)
            if manifest.index != index or manifest.previous_manifest_hash != previous_hash:
                raise PopulationCheckpointError("checkpoint boundary hash chain differs")
            previous_hash = manifest_hash
            known_cells += manifest.score_cells
            if index == 0:
                evaluated = _read_evaluation(directory, manifest, source_dir="source")
                if evaluated.source != self.seed:
                    raise PopulationCheckpointError("scored seed differs from checkpoint seed")
                population.append(evaluated)
                continue
            if manifest.outcome == "invalid":
                error = _read_invalid_turn(directory / "proposal", manifest)
                iterations.append(PopulationIteration(index=index, error=error))
                continue
            proposal = _read_valid_turn(directory / "proposal", manifest)
            evaluated = _read_evaluation(
                directory / "evaluation",
                manifest,
                source_dir=None,
                source=proposal.source,
            )
            population.append(evaluated)
            iterations.append(
                PopulationIteration(index=index, proposal=proposal, evaluation=evaluated)
            )
        if previous_hash != self.control.latest_boundary_hash:
            raise PopulationCheckpointError("checkpoint control boundary hash differs")
        if known_cells != self.control.known_score_cells:
            raise PopulationCheckpointError("checkpoint known score-cell count differs")
        if known_cells > self.identity.max_score_cells:
            raise PopulationCheckpointError("checkpoint exceeds max_score_cells")
        if any(item.score.report.request != self.identity.score_request for item in population):
            raise PopulationCheckpointError("checkpoint score request differs from identity")
        best = max(population, key=lambda item: item.score.report.score)
        return PopulationOptimizationResult(
            population=tuple(population),
            iterations=tuple(iterations),
            best=best,
        )

    def _validate_layout(self) -> None:
        allowed = {_IDENTITY_FILE, _CONTROL_FILE, _LOCK_FILE, "seed", "steps"}
        actual = {path.name for path in self.root.iterdir()}
        if not actual.issubset(allowed):
            raise PopulationCheckpointError("checkpoint contains unexpected top-level paths")
        steps_root = self.root / "steps"
        expected_steps = {f"{index:04d}" for index in range(0, self.control.committed_step + 1)}
        actual_steps = (
            {path.name for path in steps_root.iterdir()} if steps_root.is_dir() else set()
        )
        if actual_steps != expected_steps:
            raise PopulationCheckpointError("checkpoint step directories differ from control")

    def _total_segment_usage(self, segment: SandboxUsage | None) -> SandboxUsage | None:
        if segment is None:
            return None
        if self._segment_has_prior_usage and self._segment_base_usage is None:
            return None
        if self._segment_base_usage is None:
            return segment
        return SandboxUsage(
            count=self._segment_base_usage.count + segment.count,
            seconds=self._segment_base_usage.seconds + segment.seconds,
        )

    def _replace_control(self, **changes: object) -> None:
        self._assert_locked()
        updated = self.control.model_copy(update=changes)
        updated = PopulationCheckpointControl.model_validate(updated.model_dump(mode="python"))
        _write_json_durable(self.root / _CONTROL_FILE, updated.model_dump(mode="json"))
        self.control = updated

    def _require_ready(self) -> None:
        self._assert_locked()
        if self.control.state != "ready":
            raise PopulationCheckpointStateError(
                f"checkpoint state {self.control.state!r} is not ready"
            )

    def _assert_locked(self) -> None:
        if not self._lock_file.is_locked:
            raise PopulationCheckpointLockError("checkpoint lock is closed")


def _write_turn(
    destination: Path,
    turn: CandidateProposal | CandidateProposalError,
) -> None:
    write_text(destination / "REQUEST.md", turn.request)
    write_text(
        destination / "events.json",
        json.dumps(
            [{"kind": event.kind, "payload": event.payload} for event in turn.events],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(destination / "status.json", turn.status_json)
    if turn.source is not None:
        write_source_tree(destination / "source", turn.source)


def _write_evaluation(
    destination: Path,
    evaluated: EvaluatedCandidate,
    *,
    source_dir: str | None,
) -> None:
    if source_dir is not None:
        write_source_tree(destination / source_dir, evaluated.source)
    write_text(destination / "score.json", evaluated.score.report.model_dump_json(indent=2))
    copy_score_artifacts(destination / "artifacts", evaluated.score)


def _read_evaluation(
    directory: Path,
    manifest: _BoundaryManifest,
    *,
    source_dir: str | None,
    source: HarnessSourceTree | None = None,
) -> EvaluatedCandidate:
    if source_dir is not None:
        source = _read_source_tree(directory / source_dir)
    if source is None:
        raise PopulationCheckpointError("checkpoint evaluation source is missing")
    report = HarnessScoreReport.model_validate_json(
        (directory / "score.json").read_text(encoding="utf-8")
    )
    score = HarnessScore(report=report, artifacts=_DirectoryArtifactReader(directory / "artifacts"))
    evaluated = EvaluatedCandidate(
        candidate_id=manifest.candidate_id,
        source=source,
        score=score,
    )
    if (
        evaluated.source.tree_hash != manifest.source_tree_hash
        or evaluated.candidate.doc_hash != manifest.document_hash
        or report.report_hash != manifest.score_report_hash
        or len(report.cells) != manifest.score_cells
    ):
        raise PopulationCheckpointError("checkpoint evaluation differs from its manifest")
    _ = evaluated.fingerprint
    return evaluated


def _read_valid_turn(directory: Path, manifest: _BoundaryManifest) -> CandidateProposal:
    source = _read_source_tree(directory / "source")
    candidate = source.to_doc(manifest.candidate_id)
    if manifest.worker_usage is None:
        raise PopulationCheckpointError("scored proposal worker usage is missing")
    turn = CandidateProposal(
        candidate_id=manifest.candidate_id,
        source=source,
        candidate=candidate,
        events=_read_events(directory / "events.json"),
        worker_usage=manifest.worker_usage,
        request=(directory / "REQUEST.md").read_text(encoding="utf-8"),
        status_json=(directory / "status.json").read_text(encoding="utf-8"),
    )
    validate_candidate_turn(turn)
    return turn


def _read_invalid_turn(
    directory: Path,
    manifest: _BoundaryManifest,
) -> CandidateProposalError:
    source_dir = directory / "source"
    source = _read_source_tree(source_dir) if source_dir.is_dir() else None
    source_hash = source.tree_hash if source is not None else None
    if source_hash != manifest.source_tree_hash:
        raise PopulationCheckpointError("invalid proposal source differs from its manifest")
    turn = CandidateProposalError(
        manifest.candidate_id,
        manifest.error_reason or "",
        source=source,
        events=_read_events(directory / "events.json"),
        worker_usage=manifest.worker_usage,
        request=(directory / "REQUEST.md").read_text(encoding="utf-8"),
        status_json=(directory / "status.json").read_text(encoding="utf-8"),
    )
    validate_candidate_turn(turn)
    return turn


def _read_events(path: Path) -> tuple[SessionEvent, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PopulationCheckpointError("checkpoint events must contain a JSON list")
    records = tuple(_EventRecord.model_validate(item) for item in raw)
    return tuple(SessionEvent(kind=record.kind, payload=record.payload) for record in records)


def _read_seed(
    directory: Path,
    identity: PopulationCheckpointIdentity,
) -> HarnessSourceTree:
    manifest_path = directory / _MANIFEST_FILE
    manifest = _SeedManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    _verify_file_records(directory, manifest.files, ignored={_MANIFEST_FILE})
    source = _read_source_tree(directory / "source")
    if (
        source.tree_hash != manifest.source_tree_hash
        or source.tree_hash != identity.seed_source_tree_hash
    ):
        raise PopulationCheckpointError("checkpoint seed source hash differs")
    return source


def _read_boundary_manifest(directory: Path) -> tuple[_BoundaryManifest, str]:
    manifest_path = directory / _MANIFEST_FILE
    manifest = _BoundaryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    _verify_file_records(directory, manifest.files, ignored={_MANIFEST_FILE})
    return manifest, _hash_file(manifest_path)


def _read_source_tree(directory: Path) -> HarnessSourceTree:
    if not directory.is_dir():
        raise PopulationCheckpointError(f"checkpoint source directory is missing: {directory}")
    files: list[HarnessSourceFile] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PopulationCheckpointError("checkpoint source cannot contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PopulationCheckpointError("checkpoint source contains a non-file entry")
        files.append(
            HarnessSourceFile(
                path=path.relative_to(directory).as_posix(),
                content=path.read_text(encoding="utf-8"),
            )
        )
    return HarnessSourceTree(files=tuple(files))


def _population_signature(population: Sequence[EvaluatedCandidate]) -> tuple[str, ...]:
    return tuple(item.fingerprint for item in population)


def _iteration_signature(iterations: Sequence[PopulationIteration]) -> str:
    payload: list[JsonObject] = []
    for iteration in iterations:
        turn: CandidateProposal | CandidateProposalError
        if iteration.error is not None:
            turn = iteration.error
            outcome = "invalid"
            reason: JsonValue = turn.reason
        else:
            assert iteration.proposal is not None
            turn = iteration.proposal
            outcome = "scored"
            reason = None
        payload.append(
            {
                "index": iteration.index,
                "candidate_id": turn.candidate_id,
                "outcome": outcome,
                "reason": reason,
                "source_tree_hash": turn.source.tree_hash if turn.source is not None else None,
                "request": turn.request,
                "status_json": turn.status_json,
                "events": [{"kind": event.kind, "payload": event.payload} for event in turn.events],
                "worker_usage": (
                    turn.worker_usage.model_dump(mode="json")
                    if turn.worker_usage is not None
                    else None
                ),
            }
        )
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )


def _file_records(directory: Path) -> tuple[_FileRecord, ...]:
    records: list[_FileRecord] = []
    for path in sorted(directory.rglob("*")):
        if path.name == _MANIFEST_FILE and path.parent == directory:
            continue
        if path.is_symlink():
            raise PopulationCheckpointError("checkpoint cannot contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PopulationCheckpointError("checkpoint contains a non-file entry")
        content = path.read_bytes()
        records.append(
            _FileRecord(
                path=path.relative_to(directory).as_posix(),
                content_hash=_sha256_bytes(content),
                size_bytes=len(content),
            )
        )
    return tuple(records)


def _verify_file_records(
    directory: Path,
    records: Sequence[_FileRecord],
    *,
    ignored: set[str],
) -> None:
    expected = {record.path for record in records}
    if len(expected) != len(records):
        raise PopulationCheckpointError("checkpoint manifest contains duplicate file paths")
    actual: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise PopulationCheckpointError("checkpoint cannot contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PopulationCheckpointError("checkpoint contains a non-file entry")
        relative = path.relative_to(directory).as_posix()
        if relative in ignored:
            continue
        actual.add(relative)
    if actual != expected:
        raise PopulationCheckpointError("checkpoint files differ from their manifest")
    for record in records:
        path = directory / record.path
        content = path.read_bytes()
        if len(content) != record.size_bytes or _sha256_bytes(content) != record.content_hash:
            raise PopulationCheckpointError(
                f"checkpoint file differs from its manifest: {record.path}"
            )


def _write_json_durable(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _sync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in directory.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _sync_directory(path)
    _sync_directory(directory)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _acquire_lock(path: Path) -> BaseFileLock:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = FileLock(path)
    try:
        lock_file.acquire(timeout=0)
    except Timeout as error:
        raise PopulationCheckpointLockError(
            f"another process owns checkpoint lock: {path}"
        ) from error
    return lock_file


def _release_lock(lock_file: BaseFileLock) -> None:
    if lock_file.is_locked:
        lock_file.release()


def _discard_temporary_files(root: Path) -> None:
    changed: set[Path] = set()
    for path in list(root.rglob("*")):
        if not _is_atomic_metadata_tail(root, path):
            continue
        if path.is_symlink() or not path.is_file():
            raise PopulationCheckpointError("checkpoint temporary entry is not a regular file")
        path.unlink()
        changed.add(path.parent)
    for directory in sorted(changed, key=lambda item: len(item.parts), reverse=True):
        _sync_directory(directory)


def _is_atomic_metadata_tail(root: Path, path: Path) -> bool:
    match = re.fullmatch(r"(.+)\.tmp-[0-9a-f]{32}", path.name)
    if match is None:
        return False
    canonical_name = match.group(1)
    relative_parent = path.parent.relative_to(root)
    if relative_parent == Path("."):
        return canonical_name in {_IDENTITY_FILE, _CONTROL_FILE}
    if relative_parent == Path("seed"):
        return canonical_name == _MANIFEST_FILE
    parts = relative_parent.parts
    return (
        len(parts) == 2
        and parts[0] == "steps"
        and len(parts[1]) == 4
        and parts[1].isdigit()
        and canonical_name == _MANIFEST_FILE
    )


def _relative_to_run(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError as error:
        raise ValueError("checkpoint publication path is outside its run directory") from error

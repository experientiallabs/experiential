"""Agentic proposal of complete harness source trees from scored project history."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from wmh.agents.project import (
    DEFAULT_SOURCE_TREE_MAX_BYTES,
    DEFAULT_SOURCE_TREE_MAX_FILES,
    AgentProjectRun,
    ProjectSourceStage,
)
from wmh.core.text import validate_durable_text
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.runtime import HarnessSearchCancelled, TokenUsage
from wmh.harness.scoring import EvaluationArtifact, HarnessScore
from wmh.harness.source_tree import HarnessSourceTree
from wmh.providers.base import ToolCallingProvider

DEFAULT_MAX_HISTORY_CANDIDATES = 1_024
DEFAULT_MAX_HISTORY_BYTES = 512 * 1024 * 1024
MAX_HISTORY_ARTIFACT_PATH_BYTES = 1_024


class CandidateProject(Protocol):
    """Project operations required to propose one complete source candidate."""

    workspace: str

    def write_text(self, path: str, content: str) -> None: ...

    def stage_source_tree(
        self,
        tree: HarnessSourceTree,
        *,
        max_files: int,
        max_bytes: int,
    ) -> ProjectSourceStage: ...

    def snapshot_source_tree(
        self,
        stage: ProjectSourceStage,
        *,
        max_files: int,
        max_bytes: int,
    ) -> HarnessSourceTree: ...

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        on_event: Callable[[SessionEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
        retry_recoverable: bool = True,
    ) -> AgentProjectRun: ...


@dataclass(frozen=True)
class EvaluatedCandidate:
    """One complete source candidate paired with its immutable raw score evidence."""

    candidate_id: str
    source: HarnessSourceTree
    score: HarnessScore

    def __post_init__(self) -> None:
        if not self.candidate_id or len(self.candidate_id) > 512:
            raise ValueError("candidate_id must contain between 1 and 512 characters")
        validate_durable_text(self.candidate_id, field="candidate id")
        candidate = self.source.to_doc(self.candidate_id)
        if candidate.doc_hash != self.score.report.candidate_doc_hash:
            raise ValueError(
                "score document hash does not match the evaluated candidate source tree"
            )

    @property
    def candidate(self) -> HarnessDoc:
        """Reparse the complete source into its validated harness document."""
        return self.source.to_doc(self.candidate_id)

    @property
    def fingerprint(self) -> str:
        """Return the immutable identity used to enforce append-only project history."""
        for artifact in self.score.report.artifacts:
            _verified_artifact_content(self.score, artifact)
        payload = {
            "candidate_id": self.candidate_id,
            "doc_hash": self.candidate.doc_hash,
            "report_hash": self.score.report.report_hash,
            "source_tree_hash": self.source.tree_hash,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateProposal:
    """One complete, host-captured and reparsed candidate proposal."""

    candidate_id: str
    source: HarnessSourceTree
    candidate: HarnessDoc
    events: tuple[SessionEvent, ...]
    worker_usage: TokenUsage


class CandidateProposalError(RuntimeError):
    """One proposal turn that did not publish a new valid complete candidate."""

    def __init__(
        self,
        candidate_id: str,
        reason: str,
        *,
        source: HarnessSourceTree | None = None,
        events: Sequence[SessionEvent] = (),
        worker_usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(f"{candidate_id}: {reason}")
        self.candidate_id = candidate_id
        self.source = source
        self.events = tuple(events)
        self.worker_usage = worker_usage


class CandidateProposer(Protocol):
    """Produce exactly one complete candidate from append-only evaluated history."""

    def propose(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal: ...


class ProjectCandidateProposer:
    """Run one contained coding turn against complete scored project history."""

    def __init__(
        self,
        project: CandidateProject,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        *,
        max_source_files: int = DEFAULT_SOURCE_TREE_MAX_FILES,
        max_source_bytes: int = DEFAULT_SOURCE_TREE_MAX_BYTES,
        max_history_candidates: int = DEFAULT_MAX_HISTORY_CANDIDATES,
        max_history_bytes: int = DEFAULT_MAX_HISTORY_BYTES,
    ) -> None:
        _require_positive_int(max_source_files, field="max_source_files")
        _require_positive_int(max_source_bytes, field="max_source_bytes")
        _require_positive_int(max_history_candidates, field="max_history_candidates")
        _require_positive_int(max_history_bytes, field="max_history_bytes")
        self._project = project
        self._agent = agent
        self._provider = provider
        self._max_source_files = max_source_files
        self._max_source_bytes = max_source_bytes
        self._max_history_candidates = max_history_candidates
        self._max_history_bytes = max_history_bytes
        self._history_fingerprints: tuple[str, ...] = ()
        self._proposal_count = 0

    def propose(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal:
        """Produce one new source tree without selecting or materializing a host parent."""
        self._check_cancelled(should_cancel)
        frozen_history = tuple(history)
        if not frozen_history:
            raise ValueError("candidate proposal history must include an evaluated seed")
        candidate_ids = [item.candidate_id for item in frozen_history]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate proposal history contains duplicate candidate_id values")
        score_request = frozen_history[0].score.report.request
        if any(item.score.report.request != score_request for item in frozen_history[1:]):
            raise ValueError("all candidate history entries must use the same score request")
        self._validate_history_bounds(frozen_history)
        fingerprints = tuple(item.fingerprint for item in frozen_history)
        if fingerprints[: len(self._history_fingerprints)] != self._history_fingerprints:
            raise ValueError("candidate proposal history must preserve its append-only prefix")
        if len(fingerprints) < len(self._history_fingerprints):
            raise ValueError("candidate proposal history must preserve its append-only prefix")

        self._materialize_history(frozen_history, start=len(self._history_fingerprints))
        self._history_fingerprints = fingerprints
        self._write_history_manifest(frozen_history)
        self._check_cancelled(should_cancel)

        previous_proposal_id = (
            f"candidate-{self._proposal_count:04d}" if self._proposal_count > 0 else None
        )
        next_proposal_count = self._proposal_count + 1
        candidate_id = f"candidate-{next_proposal_count:04d}"
        proposal_dir = f"proposals/{candidate_id}"
        stage = self._project.stage_source_tree(
            HarnessSourceTree(files=()),
            max_files=self._max_source_files,
            max_bytes=self._max_source_bytes,
        )
        absolute_stage = f"{self._project.workspace}/{stage.path}"
        request = _proposal_request(
            candidate_id=candidate_id,
            absolute_stage=absolute_stage,
            proposal_dir=proposal_dir,
            history_count=len(frozen_history),
            previous_proposal_id=previous_proposal_id,
        )
        self._project.write_text(f"{proposal_dir}/REQUEST.md", request)
        self._proposal_count = next_proposal_count

        events: list[SessionEvent] = []
        run_error: str | None = None
        run_result: AgentProjectRun | None = None
        try:
            run_result = self._project.run(
                self._agent,
                self._provider,
                request,
                on_event=events.append,
                should_cancel=should_cancel,
                writable_files=(),
                retry_recoverable=False,
            )
        except HarnessSearchCancelled:
            self._write_events(proposal_dir, events)
            raise
        except Exception as error:  # noqa: BLE001 - persist the failed turn before rejecting it
            run_error = str(error)
        self._write_events(proposal_dir, events)
        self._check_cancelled(should_cancel)

        source: HarnessSourceTree | None = None
        snapshot_error: str | None = None
        try:
            source = self._project.snapshot_source_tree(
                stage,
                max_files=self._max_source_files,
                max_bytes=self._max_source_bytes,
            )
        except Exception as error:  # noqa: BLE001 - preserve a failed proposal as one iteration
            snapshot_error = str(error)

        if source is not None:
            self._write_source_tree(f"{proposal_dir}/source", source)

        validation_errors: list[str] = []
        if not any(event.kind == "submit" for event in events):
            validation_errors.append("agent turn did not submit a completed candidate")
        if snapshot_error is not None:
            validation_errors.append(f"source snapshot failed: {snapshot_error}")
        if run_error is not None:
            validation_errors.append(f"agent turn failed: {run_error}")

        candidate: HarnessDoc | None = None
        if source is not None:
            try:
                candidate = source.to_doc(candidate_id)
            except (TypeError, ValueError) as error:
                validation_errors.append(str(error))

        worker_usage = run_result.worker_usage if run_result is not None else None
        self._write_status(
            proposal_dir,
            candidate_id=candidate_id,
            source=source,
            candidate=candidate,
            agent_error=run_error,
            validation_errors=validation_errors,
        )
        if validation_errors or source is None or candidate is None:
            reason = "; ".join(validation_errors) or "candidate source was not captured"
            raise CandidateProposalError(
                candidate_id,
                reason,
                source=source,
                events=events,
                worker_usage=worker_usage,
            )
        assert run_result is not None
        return CandidateProposal(
            candidate_id=candidate_id,
            source=source,
            candidate=candidate,
            events=tuple(events),
            worker_usage=run_result.worker_usage,
        )

    def _validate_history_bounds(self, history: Sequence[EvaluatedCandidate]) -> None:
        """Reject unbounded evaluator-controlled history before reading or writing its bytes."""
        if len(history) > self._max_history_candidates:
            raise ValueError(
                f"candidate history exceeds max_history_candidates={self._max_history_candidates}"
            )
        total_bytes = 0
        for evaluated in history:
            evaluated.source.validate_bounds(
                max_files=self._max_source_files,
                max_bytes=self._max_source_bytes,
            )
            total_bytes += evaluated.source.total_bytes
            total_bytes += len(evaluated.score.report.model_dump_json().encode("utf-8"))
            for artifact in evaluated.score.report.artifacts:
                if len(artifact.path.encode("utf-8")) > MAX_HISTORY_ARTIFACT_PATH_BYTES:
                    raise ValueError(
                        f"artifact path exceeds {MAX_HISTORY_ARTIFACT_PATH_BYTES} bytes: "
                        f"{artifact.path!r}"
                    )
                total_bytes += artifact.size_bytes
        if total_bytes > self._max_history_bytes:
            raise ValueError(
                f"candidate history exceeds max_history_bytes={self._max_history_bytes}"
            )

    def _materialize_history(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        start: int,
    ) -> None:
        """Append every new complete candidate and exact artifact byte to the project."""
        for index, evaluated in enumerate(history[start:], start=start):
            directory = f"history/candidate-{index:04d}"
            self._write_source_tree(f"{directory}/source", evaluated.source)
            self._project.write_text(
                f"{directory}/evaluation/report.json",
                evaluated.score.report.model_dump_json(indent=2),
            )
            artifact_manifest: list[dict[str, str | int]] = []
            for artifact in evaluated.score.report.artifacts:
                content = _verified_artifact_content(evaluated.score, artifact)
                project_path, encoding, rendered = _render_artifact(directory, artifact, content)
                self._project.write_text(project_path, rendered)
                artifact_manifest.append(
                    {
                        "artifact_path": artifact.path,
                        "content_hash": artifact.content_hash,
                        "encoding": encoding,
                        "media_type": artifact.media_type,
                        "project_path": project_path,
                        "size_bytes": artifact.size_bytes,
                    }
                )
            self._project.write_text(
                f"{directory}/evaluation/artifacts.json",
                _json(artifact_manifest),
            )

    def _write_history_manifest(self, history: Sequence[EvaluatedCandidate]) -> None:
        candidates = []
        for index, evaluated in enumerate(history):
            directory = f"history/candidate-{index:04d}"
            candidates.append(
                {
                    "candidate_id": evaluated.candidate_id,
                    "doc_hash": evaluated.candidate.doc_hash,
                    "evaluation_artifacts": f"{directory}/evaluation/artifacts.json",
                    "score": evaluated.score.report.score,
                    "score_report": f"{directory}/evaluation/report.json",
                    "source_dir": f"{directory}/source",
                    "source_tree_hash": evaluated.source.tree_hash,
                }
            )
        self._project.write_text(
            "history/manifest.json",
            _json(
                {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "proposal_history_dir": "proposals",
                }
            ),
        )

    def _write_source_tree(self, directory: str, source: HarnessSourceTree) -> None:
        for item in source.files:
            self._project.write_text(f"{directory}/{item.path}", item.content)

    def _write_events(self, proposal_dir: str, events: Sequence[SessionEvent]) -> None:
        self._project.write_text(
            f"{proposal_dir}/events.json",
            _json([{"kind": event.kind, "payload": event.payload} for event in events]),
        )

    def _write_status(
        self,
        proposal_dir: str,
        *,
        candidate_id: str,
        source: HarnessSourceTree | None,
        candidate: HarnessDoc | None,
        agent_error: str | None,
        validation_errors: Sequence[str],
    ) -> None:
        self._project.write_text(
            f"{proposal_dir}/status.json",
            _json(
                {
                    "agent_error": agent_error,
                    "candidate_doc_hash": candidate.doc_hash if candidate is not None else None,
                    "candidate_id": candidate_id,
                    "source_tree_hash": source.tree_hash if source is not None else None,
                    "valid": not validation_errors and candidate is not None,
                    "validation_error": "; ".join(validation_errors) or None,
                }
            ),
        )

    @staticmethod
    def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harness search cancelled")


def _proposal_request(
    *,
    candidate_id: str,
    absolute_stage: str,
    proposal_dir: str,
    history_count: int,
    previous_proposal_id: str | None,
) -> str:
    previous_trace = (
        f"The immediately preceding coding-turn trace is "
        f"`proposals/{previous_proposal_id}/events.json`; its captured source and status are "
        f"in the same directory."
        if previous_proposal_id is not None
        else "There is no earlier coding turn in this project."
    )
    return f"""Produce exactly one complete harness candidate: {candidate_id}.

Read `history/manifest.json`. It indexes all {history_count} evaluated candidates, their complete
source directories, full score reports, and raw evaluator artifacts. Earlier coding-turn traces
remain under `proposals/`. {previous_trace} Use the full population as evidence. Do not select or
assume a host-designated source to extend.

Your only candidate output is this initially empty directory:
`{absolute_stage}`

Use Bash to inspect immutable project evidence and to copy, create, edit, delete, and test files
inside that directory. Leave one complete standalone harness source tree there. Do not modify
`history/`, `proposals/`, or any other project path. When the candidate is complete, call submit.
The host snapshots the directory once after this turn and will not ask for a repair.

This turn's immutable request and trace are stored under `{proposal_dir}/`.
"""


def _verified_artifact_content(score: HarnessScore, artifact: EvaluationArtifact) -> bytes:
    content = score.artifacts.read_bytes(artifact.path)
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if len(content) != artifact.size_bytes or digest != artifact.content_hash:
        raise ValueError(f"artifact content differs from its manifest for {artifact.path!r}")
    return content


def _render_artifact(
    directory: str,
    artifact: EvaluationArtifact,
    content: bytes,
) -> tuple[str, str, str]:
    try:
        rendered = content.decode("utf-8")
        validate_durable_text(rendered, field=f"artifact {artifact.path!r}")
    except (UnicodeDecodeError, ValueError):
        return (
            f"{directory}/evaluation/artifacts-base64/{artifact.path}.b64",
            "base64",
            base64.b64encode(content).decode("ascii"),
        )
    return f"{directory}/evaluation/artifacts/{artifact.path}", "utf-8", rendered


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _require_positive_int(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")

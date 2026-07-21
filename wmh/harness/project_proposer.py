"""Agentic proposal of complete harness source trees from scored project history."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
MAX_DIRECTIVE_CHARS = 20_000
# The seed's complete source tree is materialized here before staging, so ``stage_from_seed``
# copies it in-sandbox from this path rather than re-uploading it file by file from the host.
SEED_SOURCE_DIR = "history/candidate-0000/source"


class CandidateProject(Protocol):
    """Project operations required to propose one complete source candidate."""

    workspace: str

    def write_text(self, path: str, content: str) -> None: ...

    def read_text(self, path: str) -> str: ...

    def stage_source_tree(
        self,
        tree: HarnessSourceTree,
        *,
        max_files: int,
        max_bytes: int,
        copy_from: str | None = None,
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
    request: str = ""
    status_json: str = ""


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
        request: str = "",
        status_json: str = "",
    ) -> None:
        super().__init__(f"{candidate_id}: {reason}")
        self.candidate_id = candidate_id
        self.reason = reason
        self.source = source
        self.events = tuple(events)
        self.worker_usage = worker_usage
        self.request = request
        self.status_json = status_json


class CandidateProposer(Protocol):
    """Produce exactly one complete candidate from append-only evaluated history."""

    def propose(
        self,
        history: Sequence[EvaluatedCandidate],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> CandidateProposal: ...


@runtime_checkable
class ResumableCandidateProposer(CandidateProposer, Protocol):
    """Restore append-only proposal state before producing another candidate."""

    def restore(
        self,
        history: Sequence[EvaluatedCandidate],
        turns: Sequence[CandidateProposal | CandidateProposalError],
    ) -> None: ...


class ProjectCandidateProposer:
    """Run one contained coding turn against complete scored project history."""

    def __init__(
        self,
        project: CandidateProject,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        *,
        directive: str = "",
        on_event: Callable[[SessionEvent], None] | None = None,
        max_source_files: int = DEFAULT_SOURCE_TREE_MAX_FILES,
        max_source_bytes: int = DEFAULT_SOURCE_TREE_MAX_BYTES,
        max_history_candidates: int = DEFAULT_MAX_HISTORY_CANDIDATES,
        max_history_bytes: int = DEFAULT_MAX_HISTORY_BYTES,
        stage_from_seed: bool = False,
    ) -> None:
        _require_positive_int(max_source_files, field="max_source_files")
        _require_positive_int(max_source_bytes, field="max_source_bytes")
        _require_positive_int(max_history_candidates, field="max_history_candidates")
        _require_positive_int(max_history_bytes, field="max_history_bytes")
        if directive:
            validate_durable_text(directive, field="proposal directive")
            if len(directive) > MAX_DIRECTIVE_CHARS:
                raise ValueError(f"proposal directive exceeds {MAX_DIRECTIVE_CHARS} characters")
        self._directive = directive
        self._on_event = on_event
        self._project = project
        self._agent = agent
        self._provider = provider
        self._max_source_files = max_source_files
        self._max_source_bytes = max_source_bytes
        self._max_history_candidates = max_history_candidates
        self._max_history_bytes = max_history_bytes
        # When set, every proposal turn stages the seed (candidate-0000) source tree instead of an
        # empty directory, so the agent edits a valid, complete harness in place rather than
        # assembling one from scratch. Off by default so the population lane stays byte-identical.
        self._stage_from_seed = stage_from_seed
        self._history_fingerprints: tuple[str, ...] = ()
        self._proposal_count = 0

    def restore(
        self,
        history: Sequence[EvaluatedCandidate],
        turns: Sequence[CandidateProposal | CandidateProposalError],
    ) -> None:
        """Restore exact committed proposal traces into a fresh project."""
        if self._history_fingerprints or self._proposal_count:
            raise ValueError("candidate proposer restore requires fresh proposer state")
        frozen_history = tuple(history)
        frozen_turns = tuple(turns)
        fingerprints = self._validate_restored_state(frozen_history, frozen_turns)

        self._materialize_history(frozen_history, fingerprints=fingerprints, start=0)
        self._write_history_manifest(frozen_history)
        history_count = 1
        # When staging from the seed, every turn's pre-populated base is the seed source
        # (candidate-0000), never the captured turn source, so restored staging matches propose().
        seed_base = frozen_history[0].source
        for index, turn in enumerate(frozen_turns, start=1):
            candidate_id = f"candidate-{index:04d}"
            proposal_dir = f"proposals/{candidate_id}"
            staged_base = (
                seed_base if self._stage_from_seed else (turn.source or HarnessSourceTree(files=()))
            )
            stage = self._project.stage_source_tree(
                staged_base,
                max_files=self._max_source_files,
                max_bytes=self._max_source_bytes,
                copy_from=SEED_SOURCE_DIR if self._stage_from_seed else None,
            )
            expected_request = _proposal_request(
                candidate_id=candidate_id,
                absolute_stage=f"{self._project.workspace}/{stage.path}",
                proposal_dir=proposal_dir,
                history_count=history_count,
                previous_proposal_id=(f"candidate-{index - 1:04d}" if index > 1 else None),
                directive=self._directive,
                stage_from_seed=self._stage_from_seed,
            )
            if turn.request != expected_request:
                raise ValueError(f"restored request differs for {candidate_id}")
            self._project.write_text(f"{proposal_dir}/REQUEST.md", turn.request)
            self._write_events(proposal_dir, turn.events)
            if turn.source is not None:
                self._write_source_tree(f"{proposal_dir}/source", turn.source)
            self._project.write_text(f"{proposal_dir}/status.json", turn.status_json)
            if isinstance(turn, CandidateProposal):
                history_count += 1

        self._history_fingerprints = fingerprints
        self._proposal_count = len(frozen_turns)

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

        self._materialize_history(
            frozen_history,
            fingerprints=fingerprints,
            start=len(self._history_fingerprints),
        )
        self._history_fingerprints = fingerprints
        self._write_history_manifest(frozen_history)
        self._check_cancelled(should_cancel)

        previous_proposal_id = (
            f"candidate-{self._proposal_count:04d}" if self._proposal_count > 0 else None
        )
        next_proposal_count = self._proposal_count + 1
        candidate_id = f"candidate-{next_proposal_count:04d}"
        proposal_dir = f"proposals/{candidate_id}"
        staged_base = (
            frozen_history[0].source if self._stage_from_seed else HarnessSourceTree(files=())
        )
        stage = self._project.stage_source_tree(
            staged_base,
            max_files=self._max_source_files,
            max_bytes=self._max_source_bytes,
            copy_from=SEED_SOURCE_DIR if self._stage_from_seed else None,
        )
        absolute_stage = f"{self._project.workspace}/{stage.path}"
        request = _proposal_request(
            candidate_id=candidate_id,
            absolute_stage=absolute_stage,
            proposal_dir=proposal_dir,
            history_count=len(frozen_history),
            previous_proposal_id=previous_proposal_id,
            directive=self._directive,
            stage_from_seed=self._stage_from_seed,
        )
        self._project.write_text(f"{proposal_dir}/REQUEST.md", request)
        self._proposal_count = next_proposal_count

        events: list[SessionEvent] = []

        def sink(event: SessionEvent) -> None:
            events.append(event)
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:  # noqa: BLE001 - a sink error must never abort the proposal
                    pass

        run_error: str | None = None
        run_result: AgentProjectRun | None = None
        try:
            run_result = self._project.run(
                self._agent,
                self._provider,
                request,
                on_event=sink,
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
        status_json = self._write_status(
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
                request=request,
                status_json=status_json,
            )
        assert run_result is not None
        return CandidateProposal(
            candidate_id=candidate_id,
            source=source,
            candidate=candidate,
            events=tuple(events),
            worker_usage=run_result.worker_usage,
            request=request,
            status_json=status_json,
        )

    def _validate_restored_state(
        self,
        history: tuple[EvaluatedCandidate, ...],
        turns: tuple[CandidateProposal | CandidateProposalError, ...],
    ) -> tuple[str, ...]:
        if not history:
            raise ValueError("restored history must include an evaluated seed")
        candidate_ids = [item.candidate_id for item in history]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("restored history contains duplicate candidate_id values")
        if history[0].candidate_id != "candidate-0000":
            raise ValueError("restored history seed must use candidate-0000")
        score_request = history[0].score.report.request
        if any(item.score.report.request != score_request for item in history[1:]):
            raise ValueError("all restored history entries must use the same score request")
        self._validate_history_bounds(history)
        fingerprints = tuple(item.fingerprint for item in history)

        population_index = 1
        for index, turn in enumerate(turns, start=1):
            expected_id = f"candidate-{index:04d}"
            if turn.candidate_id != expected_id:
                raise ValueError("restored proposal candidate_id values must match consumed slots")
            validate_candidate_turn(turn)
            if isinstance(turn, CandidateProposal):
                if population_index >= len(history):
                    raise ValueError("restored proposal is missing its evaluated history entry")
                evaluated = history[population_index]
                if evaluated.candidate_id != turn.candidate_id:
                    raise ValueError("restored proposal and evaluated history identities differ")
                if evaluated.source != turn.source:
                    raise ValueError("restored proposal and evaluated history sources differ")
                if evaluated.candidate.doc_hash != turn.candidate.doc_hash:
                    raise ValueError("restored proposal and evaluated history documents differ")
                population_index += 1
        if population_index != len(history):
            raise ValueError("restored history contains an evaluation without a proposal turn")
        return fingerprints

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
        fingerprints: Sequence[str],
        start: int,
    ) -> None:
        """Append every new complete candidate and exact artifact byte to the project."""
        for index, evaluated in enumerate(history[start:], start=start):
            directory = f"history/candidate-{index:04d}"
            self._bind_history_slot(directory, fingerprints[index])
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

    def _bind_history_slot(self, directory: str, fingerprint: str) -> None:
        """Bind one history index before non-atomic writes so retries cannot mix candidates."""
        identity_path = f"{directory}/fingerprint.txt"
        try:
            existing = self._project.read_text(identity_path)
        except FileNotFoundError:
            self._project.write_text(identity_path, fingerprint)
            return
        if existing != fingerprint:
            raise ValueError(
                f"{directory} is already bound to another candidate after a partial write; "
                "retry with the original history or create a fresh project"
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
    ) -> str:
        status_json = _json(
            {
                "agent_error": agent_error,
                "candidate_doc_hash": candidate.doc_hash if candidate is not None else None,
                "candidate_id": candidate_id,
                "source_tree_hash": source.tree_hash if source is not None else None,
                "valid": not validation_errors and candidate is not None,
                "validation_error": "; ".join(validation_errors) or None,
            }
        )
        self._project.write_text(f"{proposal_dir}/status.json", status_json)
        return status_json

    @staticmethod
    def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harness search cancelled")


def validate_candidate_turn(turn: CandidateProposal | CandidateProposalError) -> None:
    """Validate one exact proposal trace before durable restore or checkpointing."""
    if not turn.request:
        raise ValueError("restored proposal request is missing")
    validate_durable_text(turn.request, field="restored proposal request")
    if not turn.status_json:
        raise ValueError("restored proposal status_json is missing")
    validate_durable_text(turn.status_json, field="restored proposal status_json")
    try:
        status = json.loads(turn.status_json)
    except json.JSONDecodeError as error:
        raise ValueError("restored proposal status_json is invalid") from error
    if not isinstance(status, dict):
        raise ValueError("restored proposal status_json must contain an object")
    if status.get("candidate_id") != turn.candidate_id:
        raise ValueError("restored proposal status candidate_id differs")
    expected_valid = isinstance(turn, CandidateProposal)
    if status.get("valid") is not expected_valid:
        raise ValueError("restored proposal status validity differs")
    expected_source_hash = turn.source.tree_hash if turn.source is not None else None
    if status.get("source_tree_hash") != expected_source_hash:
        raise ValueError("restored proposal status source hash differs")
    candidate_hash: str | None = None
    if turn.source is not None:
        try:
            candidate_hash = turn.source.to_doc(turn.candidate_id).doc_hash
        except (TypeError, ValueError):
            pass
    if status.get("candidate_doc_hash") != candidate_hash:
        raise ValueError("restored proposal status document hash differs")
    if isinstance(turn, CandidateProposal):
        if not any(event.kind == "submit" for event in turn.events):
            raise ValueError("restored valid proposal trace has no submit event")
        if turn.candidate.name != turn.candidate_id:
            raise ValueError("restored proposal document name differs")
        if candidate_hash != turn.candidate.doc_hash:
            raise ValueError("restored proposal source and document differ")


def _proposal_request(
    *,
    candidate_id: str,
    absolute_stage: str,
    proposal_dir: str,
    history_count: int,
    previous_proposal_id: str | None,
    directive: str = "",
    stage_from_seed: bool = False,
) -> str:
    previous_trace = (
        f"The immediately preceding coding-turn trace is "
        f"`proposals/{previous_proposal_id}/events.json`; its captured source and status are "
        f"in the same directory."
        if previous_proposal_id is not None
        else "There is no earlier coding turn in this project."
    )
    directive_block = (
        "\nThe user directing this optimization gave the following feedback. Treat it as the"
        f" primary objective of this candidate:\n\n{directive}\n"
        if directive
        else ""
    )
    output_block = (
        "Your candidate output directory is ALREADY POPULATED with the current harness's\n"
        f"complete source tree:\n`{absolute_stage}`\n\n"
        "Edit it in place to satisfy the objective. Do not delete SYSTEM.md or config.toml,\n"
        "and the final directory must remain a complete standalone harness source tree."
        if stage_from_seed
        else (
            "Your only candidate output is this initially empty directory:\n"
            f"`{absolute_stage}`\n\n"
            "Leave one complete standalone harness source tree there."
        )
    )
    return f"""Produce exactly one complete harness candidate: {candidate_id}.

Read `history/manifest.json`. It indexes all {history_count} evaluated candidates, their complete
source directories, full score reports, and raw evaluator artifacts. Earlier coding-turn traces
remain under `proposals/`. {previous_trace} Use the full population as evidence. Do not select or
assume a host-designated source to extend.
{directive_block}
{output_block}

Use Bash to inspect immutable project evidence and to copy, create, edit, delete, and test files
inside that directory. Do not modify `history/`, `proposals/`, or any other project path. When the
candidate is complete, call submit. The host snapshots the directory once after this turn and will
not ask for a repair.

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

"""Behavioral tests for complete-source project candidate proposal."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import cast

import pytest

from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import AgentProjectRun, ProjectSourceStage
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.project_proposer import (
    CandidateProposalError,
    EvaluatedCandidate,
    ProjectCandidateProposer,
)
from wmh.harness.runtime import HarnessSearchCancelled, TokenUsage
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ToolCallingProvider

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


class _BytesReader:
    def __init__(self, content: dict[str, bytes]) -> None:
        self.content = content

    def read_bytes(self, path: str) -> bytes:
        return self.content[path]


@dataclass
class _RunCall:
    instruction: str
    writable_files: tuple[str, ...]
    retry_recoverable: bool


class _FakeProject:
    workspace = "/home/user/project"

    def __init__(self, snapshots: list[HarnessSourceTree]) -> None:
        self.files: dict[str, str] = {}
        self.snapshots = snapshots
        self.staged: list[HarnessSourceTree] = []
        self.snapshot_calls: list[ProjectSourceStage] = []
        self.run_calls: list[_RunCall] = []
        self.emit_submit = True
        self.run_behavior: (
            Callable[[_FakeProject, tuple[str, ...], Callable[[SessionEvent], None] | None], None]
            | None
        ) = None

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def stage_source_tree(
        self,
        tree: HarnessSourceTree,
        *,
        max_files: int,
        max_bytes: int,
    ) -> ProjectSourceStage:
        del max_files, max_bytes
        self.staged.append(tree)
        return ProjectSourceStage(
            path=f".scratch/source-stages/stage-{len(self.staged):06d}",
            sandbox_generation=1,
        )

    def snapshot_source_tree(
        self,
        stage: ProjectSourceStage,
        *,
        max_files: int,
        max_bytes: int,
    ) -> HarnessSourceTree:
        del max_files, max_bytes
        self.snapshot_calls.append(stage)
        return self.snapshots.pop(0)

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
    ) -> AgentProjectRun:
        del agent, provider, should_cancel
        granted = tuple(writable_files or ())
        self.run_calls.append(
            _RunCall(
                instruction=instruction,
                writable_files=granted,
                retry_recoverable=retry_recoverable,
            )
        )
        if self.run_behavior is not None:
            self.run_behavior(self, granted, on_event)
        event = SessionEvent(kind="submit", payload={"answer": "candidate complete"})
        if self.emit_submit and on_event is not None:
            on_event(event)
        return AgentProjectRun(
            answer="candidate complete",
            events=((event,) if self.emit_submit else ()),
            worker_usage=TokenUsage(input_tokens=11, output_tokens=7, calls=1),
        )


def _source(prompt: str = "base") -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content=('[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n'),
            ),
        )
    )


def _evaluated(
    candidate_id: str = "seed",
    *,
    source: HarnessSourceTree | None = None,
    artifacts: dict[str, bytes] | None = None,
    score: float = 0.5,
) -> EvaluatedCandidate:
    tree = source or _source()
    candidate = tree.to_doc(candidate_id)
    raw = artifacts or {
        "traces/task.jsonl": b'{"tool":"bash","ok":true}\n',
        "screens/final.bin": b"\x00\xff",
    }
    manifest = tuple(
        EvaluationArtifact.from_bytes(path=path, content=content) for path, content in raw.items()
    )
    request = ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST_A,
            evaluator_digest=_DIGEST_B,
            execution_config_digest=_DIGEST_C,
        ),
        task_ids=("task-1",),
        attempts=1,
    )
    report = HarnessScoreReport(
        source_run_id=f"run-{candidate_id}",
        candidate_doc_hash=candidate.doc_hash,
        request=request,
        cells=(
            ScoreCell(
                task_id="task-1",
                attempt=1,
                score=score,
                passed=False,
                summary="raw result",
                artifact_paths=tuple(raw),
            ),
        ),
        artifacts=manifest,
    )
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        source=tree,
        score=HarnessScore(report=report, artifacts=_BytesReader(raw)),
    )


def _provider() -> ToolCallingProvider:
    return cast(ToolCallingProvider, object())


def _emit_bash_activity(
    project: _FakeProject,
    granted: tuple[str, ...],
    on_event: Callable[[SessionEvent], None] | None,
) -> None:
    del project
    assert granted == ()
    if on_event is not None:
        on_event(SessionEvent(kind="tool_call", payload={"name": "bash"}))


def test_evaluated_candidate_rejects_score_identity_drift() -> None:
    evaluated = _evaluated()
    changed = _source("changed")

    with pytest.raises(ValueError, match="score document hash"):
        EvaluatedCandidate(
            candidate_id="seed",
            source=changed,
            score=evaluated.score,
        )


def test_project_proposer_materializes_complete_history_and_returns_one_candidate() -> None:
    candidate_source = _source("improved")
    project = _FakeProject([candidate_source])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated()

    result = proposer.propose((seed,))

    assert result.candidate_id == "candidate-0001"
    assert result.source == candidate_source
    assert result.candidate.system_prompt() == "improved"
    assert result.worker_usage == TokenUsage(input_tokens=11, output_tokens=7, calls=1)
    assert len(project.run_calls) == 1
    assert project.staged == [seed.source]
    assert len(project.snapshot_calls) == 1
    assert project.run_calls[0].writable_files == ()
    assert project.run_calls[0].retry_recoverable is False
    assert "/home/user/project/.scratch/source-stages/stage-000001" in (
        project.run_calls[0].instruction
    )

    manifest = json.loads(project.files["history/manifest.json"])
    assert manifest["candidate_count"] == 1
    assert manifest["candidates"][0]["candidate_id"] == "seed"
    assert manifest["candidates"][0]["source_dir"] == "history/candidate-0000/source"
    assert project.files["history/candidate-0000/source/SYSTEM.md"] == "base"
    report = json.loads(project.files["history/candidate-0000/evaluation/report.json"])
    assert report["candidate_doc_hash"] == _source().to_doc("seed").doc_hash
    assert (
        project.files["history/candidate-0000/evaluation/artifacts/traces/task.jsonl"]
        == '{"tool":"bash","ok":true}\n'
    )
    encoded = project.files[
        "history/candidate-0000/evaluation/artifacts-base64/screens/final.bin.b64"
    ]
    assert base64.b64decode(encoded) == b"\x00\xff"
    assert project.files["proposals/candidate-0001/source/SYSTEM.md"] == "improved"
    events = json.loads(project.files["proposals/candidate-0001/events.json"])
    assert [event["kind"] for event in events] == ["tool_call", "submit"]


def test_every_live_proposal_stages_the_fixed_seed_not_a_later_high_scorer() -> None:
    seed_source = _source("fixed seed")
    first_source = _source("higher-scoring candidate")
    second_source = _source("independent second candidate")
    project = _FakeProject([first_source, second_source])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000", source=seed_source, score=0.1)

    first = proposer.propose((seed,))
    higher_scorer = _evaluated("candidate-0001", source=first.source, score=0.9)
    second = proposer.propose((seed, higher_scorer))

    assert second.candidate_id == "candidate-0002"
    assert project.staged == [seed_source, seed_source]
    assert project.staged[1] != higher_scorer.source


def test_fixed_seed_scaffold_is_freely_replaceable_and_final_source_is_standalone() -> None:
    base = _source("fixed seed")
    seed = HarnessSourceTree(
        files=(*base.files, HarnessSourceFile(path="seed-only.txt", content="delete me"))
    )
    replacement = _source("standalone replacement")
    project = _FakeProject([replacement])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    result = proposer.propose((_evaluated("candidate-0000", source=seed),))

    assert project.staged == [seed]
    assert result.source == replacement
    assert "seed-only.txt" not in result.source.file_map()
    assert result.source.to_doc(result.candidate_id).system_prompt() == "standalone replacement"
    request = project.run_calls[0].instruction
    assert "fixed first evaluated seed" in request
    assert "Every proposal slot receives that same fixed scaffold" in request
    assert "create,\nedit, delete, replace, and test files" in request
    assert "must not import from or otherwise depend on `history/`, `proposals/`" in request
    assert "must contain a UTF-8 `SYSTEM.md`" in request
    assert "optional `config.toml`, `runtime.py`, `skills/*.md`, and code files" in request
    assert "portable source format and parse together as one complete harness" in request
    assert "initially empty" not in request


def test_project_proposer_history_is_append_only_and_previous_trace_remains_visible() -> None:
    first_source = _source("first")
    second_source = _source("second")
    project = _FakeProject([first_source, second_source])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated()

    first = proposer.propose((seed,))
    scored_first = _evaluated("first", source=first.source)
    second = proposer.propose((seed, scored_first))

    assert second.candidate_id == "candidate-0002"
    assert len(project.run_calls) == 2
    assert "proposals/candidate-0001/events.json" in project.run_calls[1].instruction
    manifest = json.loads(project.files["history/manifest.json"])
    assert [item["candidate_id"] for item in manifest["candidates"]] == ["seed", "first"]
    assert project.files["proposals/candidate-0001/source/SYSTEM.md"] == "first"

    with pytest.raises(ValueError, match="append-only prefix"):
        proposer.propose((scored_first, seed))
    assert len(project.run_calls) == 2


def test_partial_history_write_cannot_be_reused_for_a_different_candidate() -> None:
    class _HistoryWriteFailsOnceProject(_FakeProject):
        def __init__(self) -> None:
            super().__init__([_source("proposal")])
            self.failed = False

        def write_text(self, path: str, content: str) -> None:
            if path.endswith("evaluation/report.json") and not self.failed:
                self.failed = True
                raise RuntimeError("history write interrupted")
            super().write_text(path, content)

    project = _HistoryWriteFailsOnceProject()
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    original = _evaluated("original", source=_source("original"))

    with pytest.raises(RuntimeError, match="history write interrupted"):
        proposer.propose((original,))

    assert project.files["history/candidate-0000/source/SYSTEM.md"] == "original"
    changed = _evaluated("changed", source=_source("changed"))
    with pytest.raises(ValueError, match="already bound to another candidate"):
        proposer.propose((changed,))
    assert project.files["history/candidate-0000/source/SYSTEM.md"] == "original"
    assert project.run_calls == []

    recovered = proposer.propose((original,))

    assert recovered.candidate_id == "candidate-0001"
    assert len(project.run_calls) == 1


def test_project_proposer_restores_exact_valid_and_invalid_turns_before_continuing() -> None:
    first_source = _source("first")
    third_source = _source("third")
    invalid_source = HarnessSourceTree(
        files=(HarnessSourceFile(path="notes.txt", content="unfinished"),)
    )
    original_project = _FakeProject([first_source, invalid_source, third_source])
    original_project.run_behavior = _emit_bash_activity
    original = ProjectCandidateProposer(original_project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000")

    first = original.propose((seed,))
    scored_first = _evaluated("candidate-0001", source=first.source)
    original_project.emit_submit = False
    with pytest.raises(CandidateProposalError) as raised:
        original.propose((seed, scored_first))
    invalid = raised.value
    original_project.emit_submit = True

    restored_project = _FakeProject([third_source])
    restored_project.run_behavior = _emit_bash_activity
    restored = ProjectCandidateProposer(restored_project, optimizer_agent(), _provider())
    restored.restore((seed, scored_first), (first, invalid))

    original_trace = {
        path: content
        for path, content in original_project.files.items()
        if path.startswith("proposals/")
    }
    restored_trace = {
        path: content
        for path, content in restored_project.files.items()
        if path.startswith("proposals/")
    }
    assert restored_trace == original_trace
    assert restored_project.staged == [first.source, invalid.source]

    original_third = original.propose((seed, scored_first))
    third = restored.propose((seed, scored_first))
    assert third.candidate_id == "candidate-0003"
    assert original_project.staged == [seed.source, seed.source, seed.source]
    assert restored_project.staged == [first.source, invalid.source, seed.source]
    assert third.request == original_third.request
    assert "proposals/candidate-0002/events.json" in restored_project.run_calls[0].instruction


@pytest.mark.parametrize("field", ["request", "status_json"])
def test_project_proposer_restore_rejects_incomplete_trace_before_writes(field: str) -> None:
    project = _FakeProject([_source("first")])
    project.run_behavior = _emit_bash_activity
    original = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000")
    proposal = original.propose((seed,))
    incomplete = replace(proposal, **{field: ""})
    restored_project = _FakeProject([_source("unused")])
    restored = ProjectCandidateProposer(restored_project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match=field):
        restored.restore(
            (seed, _evaluated("candidate-0001", source=proposal.source)),
            (incomplete,),
        )

    assert restored_project.files == {}
    assert restored_project.staged == []


def test_project_proposer_restore_rejects_valid_turn_without_submit_before_writes() -> None:
    project = _FakeProject([_source("first")])
    original = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000")
    proposal = original.propose((seed,))
    incomplete = replace(proposal, events=())
    restored_project = _FakeProject([_source("unused")])
    restored = ProjectCandidateProposer(restored_project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match="submit event"):
        restored.restore(
            (seed, _evaluated("candidate-0001", source=proposal.source)),
            (incomplete,),
        )

    assert restored_project.files == {}
    assert restored_project.staged == []


def test_project_proposer_restore_rejects_tampered_request_without_an_agent_run() -> None:
    project = _FakeProject([_source("first")])
    original = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000")
    proposal = original.propose((seed,))
    tampered = replace(proposal, request=proposal.request + "changed\n")
    restored_project = _FakeProject([_source("unused")])
    restored = ProjectCandidateProposer(restored_project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match="request differs"):
        restored.restore(
            (seed, _evaluated("candidate-0001", source=proposal.source)),
            (tampered,),
        )

    assert restored_project.run_calls == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("candidate_id", "candidate-9999", "candidate_id"),
        ("valid", False, "validity"),
        ("source_tree_hash", "sha256:" + "f" * 64, "source hash"),
        ("candidate_doc_hash", "sha256:" + "e" * 64, "document hash"),
    ],
)
def test_project_proposer_restore_rejects_tampered_status_before_writes(
    field: str,
    value: str | bool,
    match: str,
) -> None:
    project = _FakeProject([_source("first")])
    original = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated("candidate-0000")
    proposal = original.propose((seed,))
    status = json.loads(proposal.status_json)
    status[field] = value
    tampered = replace(proposal, status_json=json.dumps(status, indent=2, sort_keys=True))
    restored_project = _FakeProject([_source("unused")])
    restored = ProjectCandidateProposer(restored_project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match=match):
        restored.restore(
            (seed, _evaluated("candidate-0001", source=proposal.source)),
            (tampered,),
        )

    assert restored_project.files == {}
    assert restored_project.staged == []
    assert restored_project.run_calls == []


def test_pre_turn_stage_failure_does_not_advance_proposal_identity() -> None:
    class _StageFailsOnceProject(_FakeProject):
        def __init__(self) -> None:
            super().__init__([_source("first")])
            self.stage_attempts = 0

        def stage_source_tree(
            self,
            tree: HarnessSourceTree,
            *,
            max_files: int,
            max_bytes: int,
        ) -> ProjectSourceStage:
            self.stage_attempts += 1
            if self.stage_attempts == 1:
                raise RuntimeError("stage unavailable")
            return super().stage_source_tree(
                tree,
                max_files=max_files,
                max_bytes=max_bytes,
            )

    project = _StageFailsOnceProject()
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated()

    with pytest.raises(RuntimeError, match="stage unavailable"):
        proposer.propose((seed,))

    assert project.run_calls == []
    assert not any(path.startswith("proposals/") for path in project.files)

    result = proposer.propose((seed,))

    assert result.candidate_id == "candidate-0001"
    assert "There is no earlier coding turn" in project.run_calls[0].instruction
    assert "proposals/candidate-0001/events.json" not in project.run_calls[0].instruction


def test_append_only_history_binds_source_score_and_artifact_content() -> None:
    project = _FakeProject([_source("first")])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated()
    proposer.propose((seed,))

    changed_source = _evaluated("seed", source=_source("changed seed"))
    with pytest.raises(ValueError, match="append-only prefix"):
        proposer.propose((changed_source,))
    assert len(project.run_calls) == 1

    corrupt_reader = EvaluatedCandidate(
        candidate_id=seed.candidate_id,
        source=seed.source,
        score=HarnessScore(
            report=seed.score.report,
            artifacts=_BytesReader(
                {
                    "traces/task.jsonl": b"changed",
                    "screens/final.bin": b"\x00\xff",
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="artifact content differs"):
        proposer.propose((corrupt_reader,))
    assert len(project.run_calls) == 1


def test_history_rejects_a_second_evaluation_matrix_before_project_writes() -> None:
    seed = _evaluated()
    second = _evaluated("second", source=_source("second"))
    changed_request = second.score.report.request.model_copy(
        update={
            "context": second.score.report.request.context.model_copy(
                update={"execution_config_digest": "sha256:" + "d" * 64}
            )
        }
    )
    changed_report = second.score.report.model_copy(update={"request": changed_request})
    mixed = EvaluatedCandidate(
        candidate_id=second.candidate_id,
        source=second.source,
        score=HarnessScore(report=changed_report, artifacts=second.score.artifacts),
    )
    project = _FakeProject([_source("unused")])
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match="same score request"):
        proposer.propose((seed, mixed))

    assert project.files == {}
    assert project.run_calls == []


def test_history_is_bounded_before_materializing_evaluator_controlled_files() -> None:
    seed = _evaluated()
    second = _evaluated("second", source=_source("second"))
    project = _FakeProject([_source("unused")])
    proposer = ProjectCandidateProposer(
        project,
        optimizer_agent(),
        _provider(),
        max_history_candidates=1,
    )

    with pytest.raises(ValueError, match="max_history_candidates"):
        proposer.propose((seed, second))
    assert project.files == {}

    byte_bounded = ProjectCandidateProposer(
        project,
        optimizer_agent(),
        _provider(),
        max_history_bytes=1,
    )
    with pytest.raises(ValueError, match="max_history_bytes"):
        byte_bounded.propose((seed,))
    assert project.files == {}
    assert project.run_calls == []


def test_invalid_candidate_consumes_turn_without_repair_or_fallback() -> None:
    invalid = HarnessSourceTree(files=(HarnessSourceFile(path="notes.txt", content="partial"),))
    valid = _source("later")
    project = _FakeProject([invalid, valid])
    project.emit_submit = False
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())
    seed = _evaluated()

    with pytest.raises(CandidateProposalError, match="submit") as raised:
        proposer.propose((seed,))

    assert raised.value.candidate_id == "candidate-0001"
    assert raised.value.source == invalid
    assert len(project.run_calls) == 1
    assert project.staged == [seed.source]
    assert project.files["proposals/candidate-0001/source/notes.txt"] == "partial"

    project.emit_submit = True
    project.run_behavior = _emit_bash_activity
    result = proposer.propose((seed,))
    assert result.candidate_id == "candidate-0002"
    assert result.candidate.system_prompt() == "later"
    assert len(project.run_calls) == 2
    assert project.staged == [seed.source, seed.source]
    assert "proposals/candidate-0001/events.json" in project.run_calls[1].instruction


def test_complete_duplicate_is_returned_without_a_host_novelty_gate() -> None:
    seed = _evaluated()
    duplicate_project = _FakeProject([seed.source])
    duplicate_project.run_behavior = _emit_bash_activity
    duplicate_proposer = ProjectCandidateProposer(duplicate_project, optimizer_agent(), _provider())

    duplicate = duplicate_proposer.propose((seed,))

    assert duplicate.source == seed.source
    assert duplicate.candidate.doc_hash == seed.candidate.doc_hash
    assert len(duplicate_project.run_calls) == 1


def test_repeated_invalid_source_remains_an_invalid_consumed_turn() -> None:
    seed = _evaluated()
    failed = HarnessSourceTree(
        files=(HarnessSourceFile(path="notes.txt", content="same failed source"),)
    )
    failed_project = _FakeProject([failed, failed])
    failed_project.run_behavior = _emit_bash_activity
    failed_proposer = ProjectCandidateProposer(failed_project, optimizer_agent(), _provider())
    with pytest.raises(CandidateProposalError, match="missing required SYSTEM.md"):
        failed_proposer.propose((seed,))

    with pytest.raises(CandidateProposalError, match="missing required SYSTEM.md"):
        failed_proposer.propose((seed,))

    assert len(failed_project.run_calls) == 2


def test_submitted_snapshot_fails_closed_when_agent_run_errors() -> None:
    project = _FakeProject([_source("durable")])

    def fail_after_ready(
        project: _FakeProject,
        granted: tuple[str, ...],
        on_event: Callable[[SessionEvent], None] | None,
    ) -> None:
        _emit_bash_activity(project, granted, on_event)
        if on_event is not None:
            on_event(SessionEvent(kind="submit", payload={"answer": "candidate complete"}))
        raise RuntimeError("terminal frame lost")

    project.run_behavior = fail_after_ready
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    with pytest.raises(CandidateProposalError, match="agent turn failed") as raised:
        proposer.propose((_evaluated(),))

    assert raised.value.source == _source("durable")
    assert raised.value.worker_usage is None
    status = json.loads(project.files["proposals/candidate-0001/status.json"])
    assert status["agent_error"] == "terminal frame lost"
    assert status["valid"] is False


def test_invalid_tree_is_not_repaired_and_preserves_raw_source_and_trace() -> None:
    invalid = HarnessSourceTree(
        files=(HarnessSourceFile(path="config.toml", content="[harness]\n"),)
    )
    project = _FakeProject([invalid])
    project.run_behavior = _emit_bash_activity
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    with pytest.raises(CandidateProposalError, match="missing required SYSTEM.md") as raised:
        proposer.propose((_evaluated(),))

    assert raised.value.source == invalid
    assert len(project.run_calls) == 1
    assert len(project.snapshot_calls) == 1
    assert project.files["proposals/candidate-0001/source/config.toml"] == "[harness]\n"
    status = json.loads(project.files["proposals/candidate-0001/status.json"])
    assert status["valid"] is False
    assert "missing required SYSTEM.md" in status["validation_error"]


def test_cancellation_propagates_without_snapshotting_a_candidate() -> None:
    project = _FakeProject([_source("unused")])

    def cancel(
        project: _FakeProject,
        granted: tuple[str, ...],
        on_event: Callable[[SessionEvent], None] | None,
    ) -> None:
        del project, granted, on_event
        raise HarnessSearchCancelled("stop")

    project.run_behavior = cancel
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    with pytest.raises(HarnessSearchCancelled, match="stop"):
        proposer.propose((_evaluated(),))

    assert len(project.run_calls) == 1
    assert project.snapshot_calls == []


def test_history_artifact_content_is_verified_before_agent_exposure() -> None:
    evaluated = _evaluated()
    corrupt = EvaluatedCandidate(
        candidate_id=evaluated.candidate_id,
        source=evaluated.source,
        score=HarnessScore(
            report=evaluated.score.report,
            artifacts=_BytesReader(
                {
                    "traces/task.jsonl": b"changed",
                    "screens/final.bin": b"\x00\xff",
                }
            ),
        ),
    )
    project = _FakeProject([_source("unused")])
    proposer = ProjectCandidateProposer(project, optimizer_agent(), _provider())

    with pytest.raises(ValueError, match="artifact content differs"):
        proposer.propose((corrupt,))

    assert project.run_calls == []

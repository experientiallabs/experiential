"""Tests for the fresh-project-per-slot proposer (in-memory fake project, no E2B)."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import pytest
from llm_waterfall import ChatResponse

from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import AgentProjectRun, ProjectBashResult
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import CandidateProposalError, EvaluatedCandidate
from wmh.harness.project_proposer import ProjectCandidateProposer
from wmh.harness.runtime import TokenUsage
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig, ProviderKind, ToolCallingProvider


@dataclass(frozen=True)
class _RunCall:
    instruction: str
    retry_recoverable: bool


class _FakeProject:
    workspace = "/home/user/project"

    def __init__(self, snapshots: list[HarnessSourceTree | Exception]) -> None:
        self.files: dict[str, str] = {}
        self.snapshots = snapshots
        self.staged: list[tuple[HarnessSourceTree, str]] = []
        self.bash_commands: list[str] = []
        self.bash_results: dict[str, ProjectBashResult] = {}
        self.run_calls: list[_RunCall] = []
        self.emit_submit = True
        self.run_raises: Exception | None = None
        self.closed = 0

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def run_bash(self, command: str) -> ProjectBashResult:
        self.bash_commands.append(command)
        for marker, result in self.bash_results.items():
            if marker in command:
                return result
        return ProjectBashResult(stdout="", stderr="", exit_code=0)

    def stage_source_tree(self, tree: HarnessSourceTree, dest: str) -> None:
        self.staged.append((tree, dest))

    def snapshot_source_tree(
        self, directory: str, *, max_files: int, max_bytes: int
    ) -> HarnessSourceTree:
        del directory, max_files, max_bytes
        item = self.snapshots.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

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
        del agent, provider, should_cancel, writable_files
        self.run_calls.append(_RunCall(instruction, retry_recoverable))
        if self.run_raises is not None:
            raise self.run_raises
        if self.emit_submit and on_event is not None:
            on_event(SessionEvent(kind="submit", payload={"answer": "done"}))
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())

    def close(self) -> None:
        self.closed += 1


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="meta")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("the fake project never calls the provider")


def _tree(prompt: str, *extra: tuple[str, str]) -> HarnessSourceTree:
    files = [HarnessSourceFile(path="SYSTEM.md", content=prompt)]
    files.extend(HarnessSourceFile(path=path, content=content) for path, content in extra)
    return HarnessSourceTree(files=tuple(files))


def _seed_with_trial(tmp_path: Path) -> EvaluatedCandidate:
    trial = tmp_path / "harbor" / "wmh-abc" / "t1__trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "wmh-run.json").write_text('{"steps": ["s1"]}', encoding="utf-8")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "output.txt").write_text("PASS", encoding="utf-8")
    (trial / "config.json").write_text("{}", encoding="utf-8")  # ceremony: never copied
    tree = _tree("seed")
    doc = tree.to_doc("candidate-0000")
    report = ScoreReport(
        doc_hash=doc.doc_hash,
        request=ScoreRequest(task_ids=("t1",), attempts=1),
        reward_mode="positive-binary",
        cells=(
            ScoreCell(task_id="t1", attempt=1, reward=1.0, passed=True, artifact_dir=str(trial)),
        ),
    )
    return EvaluatedCandidate("candidate-0000", tree, report)


def _proposer(
    projects: list[_FakeProject], run_dir: Path, **kwargs: int
) -> ProjectCandidateProposer:
    return ProjectCandidateProposer(
        optimizer_agent(),
        _Provider(),
        project_factory=lambda: projects.pop(0),
        run_dir=run_dir,
        **kwargs,
    )


def test_proposer_materializes_history_stages_seed_and_returns_one_candidate(
    tmp_path: Path,
) -> None:
    candidate = _tree("improved", ("src/agent-loop.ts", "export {};"))
    project = _FakeProject([candidate])
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"

    proposal = _proposer([project], run_dir).propose((seed,), slot=1)

    assert proposal.candidate_id == "candidate-0001"
    assert proposal.source == candidate
    assert project.staged == [(seed.source, "candidate")]
    assert project.run_calls[0].retry_recoverable is False
    request = project.run_calls[0].instruction
    assert "/home/user/project/candidate" in request
    assert "kebab-case" in request  # the filename grammar is load-bearing prompt content
    assert "history/candidate-0000/source" in request
    # Complete history: source, report, transcript, and verifier output; NOT harbor ceremony.
    assert project.files["history/candidate-0000/source/SYSTEM.md"] == "seed"
    assert project.files["history/candidate-0000/trials/t1/attempt-1/wmh-run.json"] == (
        '{"steps": ["s1"]}'
    )
    assert project.files["history/candidate-0000/trials/t1/attempt-1/verifier/output.txt"] == (
        "PASS"
    )
    assert not any("config.json" in path for path in project.files)
    report = json.loads(project.files["history/candidate-0000/report.json"])
    assert report["score"] == 1.0
    manifest = json.loads(project.files["history/manifest.json"])
    assert manifest["candidates"][0]["by_task"] == {"t1": 1.0}
    # The interface gate checked the candidate's one code file inside the project.
    assert project.bash_commands == [
        "node --experimental-strip-types --check candidate/src/agent-loop.ts"
    ]
    assert project.closed == 1
    slot_dir = run_dir / "proposals" / "slot-0001"
    assert (slot_dir / "REQUEST.md").is_file()
    assert (slot_dir / "source" / "SYSTEM.md").read_text(encoding="utf-8") == "improved"
    assert json.loads((slot_dir / "status.json").read_text(encoding="utf-8"))["valid"] is True


def test_missing_submit_consumes_the_slot_with_persisted_evidence(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved")])
    project.emit_submit = False
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"

    with pytest.raises(CandidateProposalError, match="did not submit") as excinfo:
        _proposer([project], run_dir).propose((seed,), slot=1)

    slot_dir = run_dir / "proposals" / "slot-0001"
    assert excinfo.value.evidence_dir == str(slot_dir)
    status = json.loads((slot_dir / "status.json").read_text(encoding="utf-8"))
    assert status["valid"] is False
    assert (slot_dir / "events.json").is_file()


def test_interface_validation_failure_preserves_node_stderr(tmp_path: Path) -> None:
    project = _FakeProject([_tree("improved", ("src/agent-loop.ts", "export {"))])
    project.bash_results["node "] = ProjectBashResult(
        stdout="", stderr="SyntaxError: unexpected end of input", exit_code=1
    )
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="SyntaxError") as excinfo:
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)

    assert "interface validation failed for src/agent-loop.ts" in excinfo.value.reason


def test_agent_transport_failure_consumes_the_slot_instead_of_propagating(
    tmp_path: Path,
) -> None:
    project = _FakeProject([_tree("leftover")])
    project.run_raises = RuntimeError("server disconnected mid-turn")
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="server disconnected"):
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)
    assert project.closed == 1


def test_invalid_snapshot_tree_is_slot_evidence(tmp_path: Path) -> None:
    project = _FakeProject([ValueError("paths differ only by letter case")])
    seed = _seed_with_trial(tmp_path)

    with pytest.raises(CandidateProposalError, match="letter case"):
        _proposer([project], tmp_path / "run").propose((seed,), slot=1)


def test_oversized_evidence_is_head_tail_truncated_never_fatal(tmp_path: Path) -> None:
    seed = _seed_with_trial(tmp_path)
    transcript = Path(seed.report.cells[0].artifact_dir) / "agent" / "wmh-run.json"
    transcript.write_text("H" * 200 + "MIDDLE" + "T" * 200, encoding="utf-8")
    project = _FakeProject([_tree("improved")])

    _proposer([project], tmp_path / "run", max_history_file_bytes=64).propose((seed,), slot=1)

    copied = project.files["history/candidate-0000/trials/t1/attempt-1/wmh-run.json"]
    assert "bytes truncated" in copied
    assert copied.startswith("HHH")
    assert copied.endswith("TTT")
    assert "MIDDLE" not in copied


def test_each_slot_gets_a_fresh_project_carrying_prior_proposal_evidence(
    tmp_path: Path,
) -> None:
    seed = _seed_with_trial(tmp_path)
    run_dir = tmp_path / "run"
    first = _FakeProject([_tree("improved")])
    first.emit_submit = False  # slot 1 fails and leaves evidence in the run dir
    second = _FakeProject([_tree("improved")])
    proposer = _proposer([first, second], run_dir)

    with pytest.raises(CandidateProposalError):
        proposer.propose((seed,), slot=1)
    proposal = proposer.propose((seed,), slot=2)

    assert proposal.candidate_id == "candidate-0002"
    assert first.closed == 1 and second.closed == 1
    # The failed turn's trace teaches the next fresh project.
    assert "proposals/slot-0001/status.json" in second.files
    assert "proposals/slot-0001/REQUEST.md" in second.files
    assert not any(path.startswith("proposals/slot-0002") for path in first.files)

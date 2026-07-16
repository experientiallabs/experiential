"""Tests for provider and persistent-project delta proposers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import pytest
from llm_waterfall import ChatResponse

from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProjectRun
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.proposer import ProjectDeltaProposer, ProposalFailure, ProviderDeltaProposer
from wmh.harness.runtime import HarnessSearchCancelled, TokenUsage
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
    VerifyResult,
)


def _trigger() -> FailureSignature:
    return FailureSignature(mechanism="verification", task_ids=["t1"])


def _payload(parent: HarnessDoc, content: str) -> str:
    core = parent.surface("prompt:core")
    assert core is not None
    return json.dumps(
        {
            "expected_effect": "t1 passes",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": content,
                    "rationale": "verify before submit",
                }
            ],
        }
    )


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, messages: list[Message], **kwargs: object) -> Completion:
        del system, messages, kwargs
        self.calls += 1
        return Completion(text=self.reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, kind=ProviderKind.BEDROCK, model="m")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("fake project never calls the provider")


class _FlakyProvider(_Provider):
    def __init__(self, replies: list[str | Exception]) -> None:
        super().__init__("")
        self.replies = replies

    def complete(self, system: str, messages: list[Message], **kwargs: object) -> Completion:
        del system, messages, kwargs
        reply = self.replies[self.calls]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply)


class _Project:
    workspace = "/home/user/project"

    def __init__(self, outputs: list[str]) -> None:
        self.files: dict[str, str] = {}
        self.outputs = outputs
        self.runs = 0

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def read_text(self, path: str) -> str:
        return self.files[path]

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, should_cancel
        self.runs += 1
        assert f"exactly {len(self.outputs)}" in instruction
        round_dir = f"round-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{round_dir}/proposal-{index:02d}.json"] = output
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())


def _manifest_content(project: _Project, path: str) -> tuple[dict[str, object], list[str]]:
    manifest = json.loads(project.files[path])
    chunks = [
        project.files[str(absolute).removeprefix(f"{project.workspace}/")]
        for absolute in manifest["content_files"]
    ]
    return manifest, chunks


def _parent_surface_manifests(project: _Project, root_path: str) -> list[dict[str, object]]:
    root = json.loads(project.files[root_path])
    index_path = str(root["surface_index_manifest"]).removeprefix(f"{project.workspace}/")
    _index_manifest, index_chunks = _manifest_content(project, index_path)
    index = json.loads("".join(index_chunks))
    return [
        json.loads(project.files[str(item["manifest_file"]).removeprefix(f"{project.workspace}/")])
        for item in index
    ]


class _InterruptedProject(_Project):
    """Write a prefix of the batch, then lose the runner's terminal control frame."""

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel
        self.runs += 1
        round_dir = f"round-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{round_dir}/proposal-{index:02d}.json"] = output
        raise RuntimeError("Server disconnected after durable writes")


class _FailedProject(_Project):
    """Fail the project turn before any proposal output reaches durable storage."""

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel
        self.runs += 1
        raise RuntimeError("provider down")


def test_provider_proposer_produces_requested_sibling_count() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider(_payload(parent, "careful"))

    proposals = ProviderDeltaProposer(provider).propose_batch(
        parent, _trigger(), "evidence", history=[], count=3
    )

    assert provider.calls == 3
    assert len(proposals) == 3
    assert all(proposal is not None for proposal in proposals)


def test_provider_proposer_isolates_one_failed_sibling_call() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _FlakyProvider(
        [_payload(parent, "first"), RuntimeError("rate limited"), _payload(parent, "third")]
    )

    proposals = ProviderDeltaProposer(provider).propose_batch(
        parent, _trigger(), "evidence", history=[], count=3
    )

    assert provider.calls == 3
    assert proposals[0] is not None and not isinstance(proposals[0], ProposalFailure)
    assert proposals[1] == ProposalFailure(reason="rate limited")
    assert proposals[2] is not None and not isinstance(proposals[2], ProposalFailure)


def test_provider_proposer_checks_cancellation_between_sibling_calls() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider(_payload(parent, "careful"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        ProviderDeltaProposer(provider).propose_batch(
            parent,
            _trigger(),
            "evidence",
            history=[],
            count=3,
            should_cancel=lambda: provider.calls >= 1,
        )

    assert provider.calls == 1


def test_project_proposer_propagates_project_cancellation() -> None:
    parent = HarnessDoc.baseline("parent")
    callback = lambda: False  # noqa: E731 - identity is the behavior under test

    class _CancellingProject(_Project):
        def run(
            self,
            agent: HarnessDoc,
            provider: ToolCallingProvider,
            instruction: str,
            *,
            should_cancel: Callable[[], bool] | None = None,
        ) -> AgentProjectRun:
            del agent, provider, instruction
            assert should_cancel is callback
            raise HarnessSearchCancelled("harness search cancelled")

    proposer = ProjectDeltaProposer(_CancellingProject([]), meta_agent(), _Provider("unused"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.propose_batch(
            parent,
            _trigger(),
            "inspect failures",
            history=[],
            count=2,
            should_cancel=callback,
        )


def test_project_proposer_checks_cancellation_between_context_writes() -> None:
    parent = HarnessDoc.baseline("parent")

    class _CountingProject(_Project):
        def __init__(self) -> None:
            super().__init__([_payload(parent, "careful")])
            self.writes = 0

        def write_text(self, path: str, content: str) -> None:
            self.writes += 1
            super().write_text(path, content)

    project = _CountingProject()
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.propose_batch(
            parent,
            _trigger(),
            "inspect failures",
            history=[],
            count=1,
            should_cancel=lambda: project.writes >= 2,
        )

    assert project.writes == 2
    assert project.runs == 0


def test_project_proposer_uses_one_agent_turn_and_keeps_round_files() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful"), _payload(parent, "verify")])
    provider = _Provider("unused")

    proposer = ProjectDeltaProposer(project, meta_agent(), provider)
    proposals = proposer.propose_batch(parent, _trigger(), "inspect failures", history=[], count=2)
    first_files = dict(project.files)
    second = proposer.propose_batch(
        parent,
        _trigger(),
        "inspect the next failures",
        history=[proposal for proposal in proposals if isinstance(proposal, HarnessDelta)],
        count=2,
    )

    assert project.runs == 2
    assert len(proposals) == 2
    assert len(second) == 2
    assert all(proposal is not None for proposal in proposals)
    assert "context/round-0001/parent.json" in project.files
    assert "context/round-0001/evidence.json" in project.files
    assert "context/round-0001/history.json" in project.files
    assert "context/round-0002/history.json" in project.files
    assert "proposal-01.json" in project.files["context/round-0002/REQUEST.md"]
    assert "failure evidence manifest" in project.files["context/round-0002/REQUEST.md"]
    assert "content_files in listed order" in project.files["context/round-0002/REQUEST.md"]
    assert all(project.files[path] == content for path, content in first_files.items())
    assert {path for path in project.files if path.startswith("parents/")} == {
        path for path in first_files if path.startswith("parents/")
    }
    parent_context = json.loads(project.files["context/round-0001/parent.json"])
    surface_manifests = _parent_surface_manifests(project, "context/round-0001/parent.json")
    assert parent_context["doc_hash"] == parent.doc_hash
    assert {
        surface["id"]: surface["content_hash"] for surface in surface_manifests
    } == parent.surface_hashes()
    assert all("content" not in surface for surface in surface_manifests)
    for surface, manifest_surface in zip(parent.surfaces, surface_manifests, strict=True):
        files = [
            path.removeprefix(f"{project.workspace}/")
            for path in cast("list[str]", manifest_surface["content_files"])
        ]
        assert "".join(project.files[path] for path in files) == surface.content
        assert "source_file" not in manifest_surface


def test_project_parent_manifest_splits_large_surfaces_below_read_cap() -> None:
    content = "0123456789" * 4_001
    parent = HarnessDoc(
        name="large",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(
                id="tool_policy:main",
                kind=SurfaceKind.TOOL_POLICY,
                content="submit",
            ),
            Surface(
                id="code:large",
                kind=SurfaceKind.CODE,
                content=content,
                path="src/large.ts",
            ),
        ],
    )
    project = _Project([_payload(parent, "careful")])

    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    manifest_text = project.files["context/round-0001/parent.json"]
    surfaces = _parent_surface_manifests(project, "context/round-0001/parent.json")
    code_surface = next(surface for surface in surfaces if surface["id"] == "code:large")
    relative_files = [
        path.removeprefix(f"{project.workspace}/")
        for path in cast("list[str]", code_surface["content_files"])
    ]
    chunks = [project.files[path] for path in relative_files]
    source_file = cast("str", code_surface["source_file"]).removeprefix(f"{project.workspace}/")
    assert len(manifest_text) < 16_000
    assert len(chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert "".join(chunks) == content
    assert source_file == f"parents/{parent.doc_hash}/parent-source/src/large.ts"
    assert project.files[source_file] == content


def test_real_pi_parent_manifest_itself_fits_one_project_read() -> None:
    parent = default_agent("parent")
    project = _Project([_payload(parent, "careful")])

    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    manifest_text = project.files["context/round-0001/parent.json"]
    manifest = json.loads(manifest_text)
    surfaces = _parent_surface_manifests(project, "context/round-0001/parent.json")
    assert len(manifest_text) < 16_000
    assert manifest["surface_count"] == len(parent.surfaces)
    assert len(surfaces) == len(parent.surfaces)
    for surface in surfaces:
        chunks = [
            project.files[path.removeprefix(f"{project.workspace}/")]
            for path in cast("list[str]", surface["content_files"])
        ]
        assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert all(surface.get("source_file") for surface in surfaces if surface["kind"] == "code")


def test_project_context_preserves_evidence_and_compacts_judged_history() -> None:
    parent = HarnessDoc.baseline("parent")
    large_change = "changed source\n" * 2_001
    project = _Project([_payload(parent, large_change)])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    first = proposer.propose_batch(parent, _trigger(), "first evidence", history=[], count=1)[0]
    assert isinstance(first, HarnessDelta)
    first.verdict = GateRecord(accepted=False, reason="screened out after exact trace review")
    second = first.model_copy(
        deep=True,
        update={"delta_id": "second-history-entry", "expected_effect": "different prediction"},
    )
    evidence = "failure trace line\n" * 1_501
    history = [second, first]

    proposer.propose_batch(parent, _trigger(), evidence, history=history, count=1)

    evidence_manifest, evidence_chunks = _manifest_content(
        project, "context/round-0002/evidence.json"
    )
    history_manifest, history_chunks = _manifest_content(project, "context/round-0002/history.json")
    reconstructed_history = "".join(history_chunks)
    judged_history = json.loads(reconstructed_history)

    assert evidence_manifest["format"] == "markdown"
    assert evidence_manifest["content_length"] == len(evidence)
    assert len(evidence_chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in evidence_chunks)
    assert "".join(evidence_chunks) == evidence
    assert history_manifest["format"] == "json-array"
    assert history_manifest["entry_count"] == 2
    assert history_manifest["content_length"] == len(reconstructed_history)
    assert all(len(chunk) <= 12_000 for chunk in history_chunks)
    assert len(reconstructed_history) < len(large_change)
    assert [entry["delta_id"] for entry in judged_history] == [second.delta_id, first.delta_id]
    assert all("content" not in entry["ops"][0] for entry in judged_history)
    assert all(entry["ops"][0]["content_length"] == len(large_change) for entry in judged_history)
    assert judged_history[0]["proposal_file"] is None
    assert judged_history[1]["proposal_file"].endswith("/proposals/round-0001/proposal-01.json")


def test_project_proposer_persists_candidate_evaluation_beside_its_proposal() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful")])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    proposal = proposer.propose_batch(parent, _trigger(), "inspect failures", history=[], count=1)[
        0
    ]
    assert isinstance(proposal, HarnessDelta)
    evidence = "candidate trace\n" * 2_001

    proposer.record_evaluation(proposal, stage="screen", content=evidence)

    manifest_path = "evaluations/round-0001/proposal-01/screen.json"
    manifest, chunks = _manifest_content(project, manifest_path)
    assert manifest["delta_id"] == proposal.delta_id
    assert manifest["stage"] == "screen"
    assert len(chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert "".join(chunks) == evidence


def test_project_proposer_checks_cancellation_before_evaluation_writes() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful")])
    cancelled = False
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    proposal = proposer.propose_batch(
        parent,
        _trigger(),
        "inspect failures",
        history=[],
        count=1,
        should_cancel=lambda: cancelled,
    )[0]
    assert isinstance(proposal, HarnessDelta)
    cancelled = True

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.record_evaluation(proposal, stage="screen", content="candidate trace")

    assert not any(path.startswith("evaluations/") for path in project.files)


def test_project_proposer_stamps_missing_parent_preconditions() -> None:
    parent = HarnessDoc.baseline("parent")
    raw = json.loads(_payload(parent, "careful"))
    raw["preconditions"] = {}
    project = _Project([json.dumps(raw)])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    proposal = proposals[0]
    assert isinstance(proposal, HarnessDelta)
    assert proposal.preconditions == {"prompt:core": parent.surface_hashes()["prompt:core"]}


def test_project_proposer_salvages_outputs_written_before_runner_disconnect() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _InterruptedProject([_payload(parent, "careful")])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert isinstance(proposals[0], HarnessDelta)
    assert proposals[1] == ProposalFailure(reason="Server disconnected after durable writes")


def test_project_proposer_only_salvages_fully_parsed_outputs_after_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _InterruptedProject([_payload(parent, "careful"), "{"])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert isinstance(proposals[0], HarnessDelta)
    assert proposals[1:] == [
        ProposalFailure(reason="Server disconnected after durable writes"),
        ProposalFailure(reason="Server disconnected after durable writes"),
    ]


def test_project_proposer_keeps_a_clean_malformed_output_unusable() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project(["{"])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert proposals == [None]


def test_project_proposer_marks_every_missing_output_as_a_proposal_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _FailedProject([])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert proposals == [ProposalFailure(reason="provider down")] * 3

"""Tests for provider and persistent-project delta proposers."""

from __future__ import annotations

import json

from llm_waterfall import ChatResponse

from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProjectRun
from wmh.harness.delta import FailureSignature, HarnessDelta
from wmh.harness.doc import HarnessDoc
from wmh.harness.proposer import ProjectDeltaProposer, ProposalFailure, ProviderDeltaProposer
from wmh.harness.runner_link import TokenUsage
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
    ) -> AgentProjectRun:
        del agent, provider
        self.runs += 1
        assert f"exactly {len(self.outputs)}" in instruction
        round_dir = f"round-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{round_dir}/proposal-{index:02d}.json"] = output
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())


class _InterruptedProject(_Project):
    """Write a prefix of the batch, then lose the runner's terminal control frame."""

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
    ) -> AgentProjectRun:
        del agent, provider, instruction
        self.runs += 1
        round_dir = f"round-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{round_dir}/proposal-{index:02d}.json"] = output
        raise RuntimeError("Server disconnected after durable writes")


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
    assert "context/round-0001/history.json" in project.files
    assert "context/round-0002/history.json" in project.files
    assert "proposal-01.json" in project.files["context/round-0002/REQUEST.md"]
    assert all(project.files[path] == content for path, content in first_files.items())
    parent_context = json.loads(project.files["context/round-0001/parent.json"])
    assert parent_context["doc_hash"] == parent.doc_hash
    assert {
        surface["id"]: surface["content_hash"] for surface in parent_context["surfaces"]
    } == parent.surface_hashes()


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

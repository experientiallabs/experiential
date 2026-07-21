"""Tests for provider and persistent-project delta proposers."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from typing import cast

import pytest
from llm_waterfall import ChatResponse

from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProjectRun
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.mutate import parse_delta
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


def _skill_payload(content: str, *, slug: str = "parse-json") -> str:
    return json.dumps(
        {
            "expected_effect": "the agent parses JSON responses reliably",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": f"skill:{slug}",
                    "kind": "skill",
                    "content": content,
                    "rationale": "teach response parsing without changing unrelated behavior",
                }
            ],
        }
    )


def _proposal_failure_reason(proposal: object) -> str:
    assert isinstance(proposal, ProposalFailure)
    return proposal.reason


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
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, should_cancel, writable_files
        self.runs += 1
        assert f"exactly {len(self.outputs)}" in instruction
        iteration_dir = f"iteration-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{iteration_dir}/proposal-{index:02d}.json"] = output
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
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel, writable_files
        self.runs += 1
        iteration_dir = f"iteration-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{iteration_dir}/proposal-{index:02d}.json"] = output
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
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel, writable_files
        self.runs += 1
        raise RuntimeError("provider down")


class _RepairingProject(_Project):
    """Drive one proposal iteration with explicit per-turn rewrites of the same slot files."""

    def __init__(
        self,
        outputs: list[str],
        repairs: list[dict[int, str]],
        *,
        extra_rewrites: dict[int, str] | None = None,
        initial_error: str | None = None,
    ) -> None:
        super().__init__(outputs)
        self.repairs = repairs
        self.extra_rewrites = extra_rewrites or {}
        self.initial_error = initial_error
        self.instructions: list[str] = []
        self.write_grants: list[tuple[str, ...] | None] = []

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, should_cancel
        self.runs += 1
        self.instructions.append(instruction)
        self.write_grants.append(None if writable_files is None else tuple(sorted(writable_files)))
        if self.runs == 1:
            writes = {index: output for index, output in enumerate(self.outputs, start=1)}
        else:
            repair_index = self.runs - 2
            writes = self.repairs[repair_index] if repair_index < len(self.repairs) else {}
            writes = {**self.extra_rewrites, **writes}
        for index, output in writes.items():
            self.files[f"proposals/iteration-0001/proposal-{index:02d}.json"] = output
        if self.runs == 1 and self.initial_error is not None:
            raise RuntimeError(self.initial_error)
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())


class _RestoreFailingProject(_RepairingProject):
    """Lose the durable-write channel while the host restores a valid sibling."""

    def write_text(self, path: str, content: str) -> None:
        if self.runs >= 2 and path == "proposals/iteration-0001/proposal-01.json":
            raise RuntimeError("valid sibling restoration failed")
        super().write_text(path, content)


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
            writable_files: Collection[str] | None = None,
        ) -> AgentProjectRun:
            del agent, provider, instruction, writable_files
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


def test_project_proposer_uses_one_agent_turn_and_keeps_iteration_files() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful"), _payload(parent, "verify")])
    provider = _Provider("unused")

    proposer = ProjectDeltaProposer(project, meta_agent(), provider)
    proposals = proposer.propose_batch(parent, _trigger(), "inspect failures", history=[], count=2)
    first_files = dict(project.files)
    project.outputs = [_payload(parent, "careful next"), _payload(parent, "verify next")]
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
    assert "context/iteration-0001/parent.json" in project.files
    assert "context/iteration-0001/evidence.json" in project.files
    assert "context/iteration-0001/history.json" in project.files
    assert "context/iteration-0002/history.json" in project.files
    assert "proposal-01.json" in project.files["context/iteration-0002/REQUEST.md"]
    assert "failure evidence manifest" in project.files["context/iteration-0002/REQUEST.md"]
    assert "content_files in listed order" in project.files["context/iteration-0002/REQUEST.md"]
    assert all(project.files[path] == content for path, content in first_files.items())
    assert {path for path in project.files if path.startswith("parents/")} == {
        path for path in first_files if path.startswith("parents/")
    }
    parent_context = json.loads(project.files["context/iteration-0001/parent.json"])
    surface_manifests = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
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
                id="code:src-large-ts",
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

    manifest_text = project.files["context/iteration-0001/parent.json"]
    surfaces = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
    code_surface = next(surface for surface in surfaces if surface["id"] == "code:src-large-ts")
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

    manifest_text = project.files["context/iteration-0001/parent.json"]
    manifest = json.loads(manifest_text)
    surfaces = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
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
        project, "context/iteration-0002/evidence.json"
    )
    history_manifest, history_chunks = _manifest_content(
        project, "context/iteration-0002/history.json"
    )
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
    assert judged_history[1]["proposal_file"].endswith("/proposals/iteration-0001/proposal-01.json")


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

    manifest_path = "evaluations/iteration-0001/proposal-01/screen.json"
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


def test_project_proposer_repairs_skill_frontmatter_and_protects_valid_sibling() -> None:
    parent = HarnessDoc.baseline("parent")
    valid_raw = _payload(parent, "careful")
    invalid_skill = _skill_payload("Parse the JSON response before formatting the answer.")
    repaired_skill = _skill_payload(
        "---\n"
        "name: parse-json\n"
        "description: Parse a JSON API response before formatting the requested fields\n"
        "---\n"
        "Parse the JSON response, select the requested fields, and format only after parsing."
    )
    # The fake repair agent also tries to overwrite slot 1. The host must restore that valid file
    # byte-for-byte and re-read only slot 2.
    project = _RepairingProject(
        [valid_raw, invalid_skill],
        [{2: repaired_skill}],
        extra_rewrites={1: "{"},
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    assert project.write_grants == [
        (
            "proposals/iteration-0001/proposal-01.json",
            "proposals/iteration-0001/proposal-02.json",
        ),
        ("proposals/iteration-0001/proposal-02.json",),
    ]
    assert project.files["proposals/iteration-0001/proposal-01.json"] == valid_raw
    assert "name: <slug>" in project.instructions[0]
    assert "rewrite ONLY these" in project.instructions[1]
    assert "invalid files:" in project.instructions[1]
    assert "proposals/iteration-0001/proposal-02.json" in project.instructions[1]
    assert "Do not rewrite them" in project.instructions[1]
    first_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-01.json"]
    )
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-02.json"]
    )
    assert first_report["valid_slots"] == [1]
    assert "skill file has no frontmatter" in first_report["errors"][0]["reason"]
    assert final_report["valid_slots"] == [1, 2]
    assert final_report["errors"] == []
    for proposal in proposals:
        assert isinstance(proposal, HarnessDelta)
        assert proposal.child_doc_hash is None
        apply_delta(parent, proposal.model_copy(deep=True), "preflight-proven")


def test_project_proposer_fails_if_valid_sibling_provenance_cannot_be_restored() -> None:
    """Never return an in-memory delta whose durable proposal file may have changed."""
    parent = HarnessDoc.baseline("parent")
    project = _RestoreFailingProject(
        [_payload(parent, "careful"), _skill_payload("missing frontmatter")],
        [
            {
                2: _skill_payload(
                    "---\nname: parse-json\ndescription: Parse JSON responses\n---\nParse first."
                )
            }
        ],
        extra_rewrites={1: "{"},
    )

    with pytest.raises(RuntimeError, match="valid sibling restoration failed"):
        ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
            parent, _trigger(), "inspect failures", history=[], count=2
        )


def test_project_proposer_repairs_history_and_sibling_duplicates() -> None:
    parent = HarnessDoc.baseline("parent")
    sibling = _payload(parent, "sibling")
    historic_raw = _payload(parent, "historic")
    historic = parse_delta(parent, _trigger(), historic_raw)
    assert isinstance(historic, HarnessDelta)
    project = _RepairingProject(
        [sibling, sibling, historic_raw],
        [{2: _payload(parent, "repaired sibling"), 3: _payload(parent, "repaired history")}],
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[historic], count=3
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    delta_ids = [proposal.delta_id for proposal in proposals if isinstance(proposal, HarnessDelta)]
    assert len(delta_ids) == len(set(delta_ids)) == 3
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-01.json"])
    reasons = [error["reason"] for error in report["errors"]]
    assert any("duplicates valid sibling proposal-01" in reason for reason in reasons)
    assert any("already present in judged history" in reason for reason in reasons)


def test_project_proposer_repairs_semantically_identical_children_and_no_ops() -> None:
    base = HarnessDoc.baseline("parent")
    parent = HarnessDoc(
        name="parent",
        surfaces=[
            *base.surfaces,
            Surface(id="prompt:extra", kind=SurfaceKind.PROMPT, content="extra"),
        ],
    )

    def payload(*, reverse: bool, core: str = "core changed", extra: str = "extra changed") -> str:
        ops = [
            {
                "op": "replace",
                "surface_id": "prompt:core",
                "content": core,
                "rationale": "change the main instruction",
            },
            {
                "op": "replace",
                "surface_id": "prompt:extra",
                "content": extra,
                "rationale": "change the supporting instruction",
            },
        ]
        if reverse:
            ops.reverse()
        return json.dumps(
            {
                "expected_effect": "the two prompt sections work together",
                "preconditions": {},
                "ops": ops,
            }
        )

    core_surface = parent.surface("prompt:core")
    assert core_surface is not None
    no_op = _payload(parent, core_surface.content)
    project = _RepairingProject(
        [payload(reverse=False), payload(reverse=True), no_op],
        [
            {
                2: payload(reverse=True, extra="independent extra"),
                3: _payload(parent, "nonempty semantic change"),
            }
        ],
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-01.json"])
    reasons = [error["reason"] for error in report["errors"]]
    assert any("duplicates valid sibling proposal-01" in reason for reason in reasons)
    assert any("semantic no-op" in reason for reason in reasons)


def test_project_proposer_never_returns_a_delta_that_remains_invalid_after_two_repairs() -> None:
    parent = HarnessDoc.baseline("parent")
    invalid = _skill_payload("Still missing required frontmatter.")
    project = _RepairingProject([invalid], [{1: invalid}, {1: invalid}])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert project.runs == 3
    assert "skill file has no frontmatter" in _proposal_failure_reason(proposals[0])
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-03.json"]
    )
    assert final_report["valid_slots"] == []
    assert "skill file has no frontmatter" in final_report["errors"][0]["reason"]


def test_project_proposer_repairs_partial_durable_outputs_after_runner_disconnect() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _RepairingProject(
        [_payload(parent, "careful"), _skill_payload("missing frontmatter")],
        [
            {
                2: _skill_payload(
                    "---\nname: parse-json\ndescription: Parse JSON responses\n---\nParse first."
                )
            }
        ],
        initial_error="Server disconnected after durable writes",
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-02.json"]
    )
    assert final_report["valid_slots"] == [1, 2]
    assert final_report["errors"] == []


def test_project_proposer_rejects_a_runtime_kind_switch_when_fixed_before_search() -> None:
    parent = HarnessDoc.baseline("parent")
    switch = json.dumps(
        {
            "expected_effect": "run the vendored node harness",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-node",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _RepairingProject([switch], [{1: switch}, {1: switch}])

    proposals = ProjectDeltaProposer(
        project,
        meta_agent(),
        _Provider("unused"),
        preserve_runtime_kind=True,
    ).propose_batch(parent, _trigger(), "inspect failures", history=[], count=1)

    assert "must preserve the parent's runtime kind 'kit-python'" in _proposal_failure_reason(
        proposals[0]
    )
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-03.json"])
    assert "must preserve the parent's runtime kind 'kit-python'" in report["errors"][0]["reason"]


def test_project_proposer_allows_runtime_kind_switch_for_generic_local_search() -> None:
    parent = HarnessDoc.baseline("parent")
    switch = json.dumps(
        {
            "expected_effect": "run the vendored node harness",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-node",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _Project([switch])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    proposal = proposals[0]
    assert isinstance(proposal, HarnessDelta)
    assert apply_delta(parent, proposal, "child").runtime_kind() == "pi-node"


def test_project_proposer_repairs_an_unknown_runtime_kind_before_search() -> None:
    parent = HarnessDoc.baseline("parent")
    invalid = json.dumps(
        {
            "expected_effect": "run a misspelled execution engine",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-nod",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _RepairingProject([invalid], [{1: invalid}, {1: invalid}])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert "unsupported runtime kind 'pi-nod'" in _proposal_failure_reason(proposals[0])
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-03.json"])
    assert "unsupported runtime kind 'pi-nod'" in report["errors"][0]["reason"]


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


def test_project_proposer_reports_the_exact_clean_malformed_output_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project(["{"])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert _proposal_failure_reason(proposals[0]) == (
        "proposal is not a parseable typed delta JSON object"
    )


def test_project_proposer_marks_every_missing_output_as_a_proposal_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _FailedProject([])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert proposals == [ProposalFailure(reason="provider down")] * 3

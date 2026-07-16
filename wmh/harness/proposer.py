"""Typed delta proposers for direct providers and project-backed meta agents."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from wmh.agents.project import AgentProjectRun
from wmh.core.types import JsonObject
from wmh.harness.delta import FailureSignature, HarnessDelta
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import parse_delta, propose_delta
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.providers.base import Provider, ToolCallingProvider

_CONTEXT_CONTENT_CHUNK_CHARS = 12_000


class AgentProject(Protocol):
    """Project operations required by the optimizer wiring."""

    workspace: str

    def write_text(self, path: str, content: str) -> None: ...

    def read_text(self, path: str) -> str: ...

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentProjectRun: ...


class DeltaProposer(Protocol):
    """Produce sibling deltas against one selected parent."""

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]: ...


@dataclass(frozen=True)
class ProposalFailure:
    """One proposal slot whose provider or agent call failed."""

    reason: str


class ProviderDeltaProposer:
    """Adapt the original single-completion proposer to the batched search contract."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    @property
    def provider(self) -> Provider:
        """Return the provider used for direct proposal calls."""
        return self._provider

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Make ``count`` independent proposal calls against the same parent."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for _ in range(count):
            _check_cancelled(should_cancel)
            try:
                proposal = propose_delta(
                    parent,
                    trigger,
                    evidence,
                    self._provider,
                    history=history,
                )
            except HarnessSearchCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - isolate one flaky sibling call
                proposal = ProposalFailure(reason=str(error))
            proposals.append(proposal)
        return proposals


class ProjectDeltaProposer:
    """Wire a normal agent in a persistent project into harness search."""

    def __init__(
        self,
        project: AgentProject,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
    ) -> None:
        self._project = project
        self._agent = agent
        self._provider = provider
        self._round = 0
        self._evaluation_dirs: dict[str, str] = {}
        self._proposal_files: dict[str, str] = {}
        self._parent_manifests: dict[str, JsonObject] = {}
        self._should_cancel: Callable[[], bool] | None = None

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Run one meta-agent turn that writes ``count`` proposal files."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        self._should_cancel = should_cancel
        _check_cancelled(should_cancel)
        self._round += 1
        round_dir = f"round-{self._round:04d}"
        context_dir = f"context/{round_dir}"
        proposal_dir = f"proposals/{round_dir}"
        parent_context = self._parent_manifests.get(parent.doc_hash)
        if parent_context is None:
            parent_context = _materialize_parent(
                self._project,
                parent,
                context_dir=f"parents/{parent.doc_hash}",
                should_cancel=should_cancel,
            )
            self._parent_manifests[parent.doc_hash] = parent_context
        _write_project_text(
            self._project,
            f"{context_dir}/parent.json",
            json.dumps(parent_context, indent=2),
            should_cancel=should_cancel,
        )
        evidence_context = _materialize_context_content(
            self._project,
            evidence,
            directory=f"{context_dir}/evidence",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/evidence.json",
            json.dumps(
                {
                    "kind": "failure-evidence",
                    "format": "markdown",
                    **evidence_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        history_content = json.dumps(
            [
                _project_history_entry(
                    delta,
                    proposal_file=self._proposal_files.get(delta.delta_id),
                    evaluation_dir=self._evaluation_dirs.get(delta.delta_id),
                    workspace=self._project.workspace,
                )
                for delta in history
            ],
            indent=2,
        )
        history_context = _materialize_context_content(
            self._project,
            history_content,
            directory=f"{context_dir}/history",
            extension=".json.part",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/history.json",
            json.dumps(
                {
                    "kind": "judged-history",
                    "format": "json-array",
                    "entry_count": len(history),
                    **history_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        request = _project_request(
            workspace=self._project.workspace,
            context_dir=context_dir,
            proposal_dir=proposal_dir,
            count=count,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/REQUEST.md",
            request,
            should_cancel=should_cancel,
        )
        run_error: Exception | None = None
        try:
            self._project.run(
                self._agent,
                self._provider,
                request,
                should_cancel=should_cancel,
            )
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - durable project files may still be complete
            run_error = error
        _check_cancelled(should_cancel)

        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for index in range(1, count + 1):
            try:
                raw = _read_project_text(
                    self._project,
                    f"{proposal_dir}/proposal-{index:02d}.json",
                    should_cancel=should_cancel,
                )
            except Exception:  # noqa: BLE001 - a missing output is one unusable proposal
                if run_error is None:
                    proposals.append(None)
                else:
                    proposals.append(ProposalFailure(reason=str(run_error)))
                continue
            try:
                proposal = parse_delta(parent, trigger, raw)
            except Exception as error:  # noqa: BLE001 - isolate one malformed sibling output
                proposals.append(
                    ProposalFailure(
                        reason=(
                            str(run_error)
                            if run_error is not None
                            else f"invalid proposal output: {error}"
                        )
                    )
                )
            else:
                if proposal is None and run_error is not None:
                    proposals.append(ProposalFailure(reason=str(run_error)))
                else:
                    stamped = _stamp_project_preconditions(parent, proposal)
                    proposals.append(stamped)
                    if stamped is not None:
                        self._proposal_files.setdefault(
                            stamped.delta_id,
                            f"{self._project.workspace}/{proposal_dir}/proposal-{index:02d}.json",
                        )
                        self._evaluation_dirs.setdefault(
                            stamped.delta_id,
                            f"evaluations/{round_dir}/proposal-{index:02d}",
                        )
        return proposals

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        """Persist one candidate's judged evidence for later project-agent rounds."""
        should_cancel = self._should_cancel
        _check_cancelled(should_cancel)
        root = self._evaluation_dirs.get(
            delta.delta_id,
            f"evaluations/by-delta/{delta.delta_id}",
        )
        context = _materialize_context_content(
            self._project,
            content,
            directory=f"{root}/{stage}",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{root}/{stage}.json",
            json.dumps(
                {
                    "kind": "candidate-evaluation",
                    "stage": stage,
                    "delta_id": delta.delta_id,
                    "format": "markdown",
                    **context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise the shared search signal without converting it into a failed proposal slot."""
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")


def _write_project_text(
    project: AgentProject,
    path: str,
    content: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Make each E2B filesystem RPC a cancellation boundary."""
    _check_cancelled(should_cancel)
    project.write_text(path, content)
    _check_cancelled(should_cancel)


def _read_project_text(
    project: AgentProject,
    path: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> str:
    """Read one project output without hiding cancellation behind the next round."""
    _check_cancelled(should_cancel)
    content = project.read_text(path)
    _check_cancelled(should_cancel)
    return content


def _materialize_parent(
    project: AgentProject,
    parent: HarnessDoc,
    *,
    context_dir: str,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write a bounded manifest plus individually readable parent-surface chunks.

    A real pi document is hundreds of kilobytes, while one project ``read_file`` observation is
    intentionally capped. Putting every surface inline in one parent.json therefore hid most of
    the harness from the proposer. The manifest remains small and points to ordered chunks below
    the read cap; concatenating a surface's chunks reconstructs its exact content.
    """
    surface_index: list[JsonObject] = []
    for index, surface in enumerate(parent.surfaces, 1):
        surface_dir = f"{context_dir}/parent-surfaces/surface-{index:03d}"
        content_context = _materialize_context_content(
            project,
            surface.content,
            directory=surface_dir,
            extension=".txt",
            include_contract=False,
            should_cancel=should_cancel,
        )
        source_file: str | None = None
        if surface.path is not None:
            source_relative = f"{context_dir}/parent-source/{surface.path}"
            _write_project_text(
                project,
                source_relative,
                surface.content,
                should_cancel=should_cancel,
            )
            source_file = f"{project.workspace}/{source_relative}"
        entry: JsonObject = {
            "id": surface.id,
            "kind": surface.kind.value,
            "content_hash": surface.content_hash,
            **content_context,
        }
        if surface.budget is not None:
            entry["budget"] = surface.budget
        if surface.path is not None:
            entry["path"] = surface.path
            entry["source_file"] = source_file
        surface_manifest_relative = f"{surface_dir}/manifest.json"
        _write_project_text(
            project,
            surface_manifest_relative,
            json.dumps(entry, indent=2),
            should_cancel=should_cancel,
        )
        surface_index.append(
            {
                "id": surface.id,
                "kind": surface.kind.value,
                "content_hash": surface.content_hash,
                "manifest_file": f"{project.workspace}/{surface_manifest_relative}",
            }
        )
    index_content = json.dumps(surface_index, indent=2)
    index_context = _materialize_context_content(
        project,
        index_content,
        directory=f"{context_dir}/parent-surface-index",
        extension=".json.part",
        should_cancel=should_cancel,
    )
    index_manifest_relative = f"{context_dir}/parent-surfaces.json"
    _write_project_text(
        project,
        index_manifest_relative,
        json.dumps(
            {
                "kind": "parent-surface-index",
                "format": "json-array",
                "entry_count": len(surface_index),
                **index_context,
            },
            indent=2,
        ),
        should_cancel=should_cancel,
    )
    return {
        "name": parent.name,
        "version": parent.version,
        "doc_hash": parent.doc_hash,
        "source_root": f"{project.workspace}/{context_dir}/parent-source",
        "surface_count": len(surface_index),
        "surface_index_manifest": f"{project.workspace}/{index_manifest_relative}",
        "content_contract": (
            "Read surface_index_manifest, then concatenate its content_files and parse that JSON "
            "array. Each index entry points to one independently readable surface manifest. "
            "Within a surface manifest, concatenate content_files exactly to reconstruct the "
            "surface. Pathful code is also mirrored beneath source_root at its exact path."
        ),
    }


def _materialize_context_content(
    project: AgentProject,
    content: str,
    *,
    directory: str,
    extension: str,
    include_contract: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write exact ordered chunks that each fit in one project ``read_file`` result."""
    chunk_count = max(1, math.ceil(len(content) / _CONTEXT_CONTENT_CHUNK_CHARS))
    width = max(3, len(str(chunk_count)))
    content_files: list[str] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * _CONTEXT_CONTENT_CHUNK_CHARS
        chunk = content[start : start + _CONTEXT_CONTENT_CHUNK_CHARS]
        relative = (
            f"{directory}/part-{chunk_index + 1:0{width}d}-of-{chunk_count:0{width}d}{extension}"
        )
        _write_project_text(
            project,
            relative,
            chunk,
            should_cancel=should_cancel,
        )
        content_files.append(f"{project.workspace}/{relative}")
    result: JsonObject = {
        "content_length": len(content),
        "content_files": content_files,
    }
    if include_contract:
        result["content_contract"] = (
            "Read content_files in listed order and concatenate them exactly. Each file is "
            "independently readable without truncation."
        )
    return result


def _project_history_entry(
    delta: HarnessDelta,
    *,
    proposal_file: str | None,
    evaluation_dir: str | None,
    workspace: str,
) -> JsonObject:
    """Compact judged metadata while raw proposals retain exact replacement payloads.

    Re-serializing every prior full code surface into every later round makes persistent history
    quadratic in run length. The project already owns each raw proposal file, so history carries
    queryable identities, rationales, sizes, and verdicts plus a direct pointer to the exact bytes.
    """
    ops: list[JsonObject] = []
    for op in delta.ops:
        item: JsonObject = {
            "op": op.op,
            "surface_id": op.surface_id,
            "rationale": op.rationale[:2_000],
            "content_length": len(op.content) if op.content is not None else 0,
        }
        if op.kind is not None:
            item["kind"] = op.kind.value
        if op.path is not None:
            item["path"] = op.path
        if op.budget is not None:
            item["budget"] = op.budget
        ops.append(item)
    return {
        "delta_id": delta.delta_id,
        "parent_doc_hash": delta.parent_doc_hash,
        "child_doc_hash": delta.child_doc_hash,
        "trigger": delta.trigger.model_dump(mode="json"),
        "preconditions": dict(delta.preconditions),
        "expected_effect": delta.expected_effect[:2_000],
        "ops": ops,
        "verdict": delta.verdict.model_dump(mode="json") if delta.verdict is not None else None,
        "proposal_file": proposal_file,
        "evaluation_dir": (f"{workspace}/{evaluation_dir}" if evaluation_dir is not None else None),
        "content_contract": (
            "Exact op content remains in proposal_file; this entry intentionally omits it to "
            "keep cumulative judged history linear and fast."
        ),
    }


def _stamp_project_preconditions(
    parent: HarnessDoc, proposal: HarnessDelta | None
) -> HarnessDelta | None:
    """Stamp missing concurrency metadata from the exact project-round parent.

    The ordinary agent still chooses every semantic operation. The host owns this mechanical
    identity field because it wrote the immutable parent snapshot for the same synchronous round.
    An explicitly supplied but incorrect hash is preserved so normal validation rejects it.
    """
    if proposal is None:
        return None
    for op in proposal.ops:
        if op.op not in ("replace", "remove") or op.surface_id in proposal.preconditions:
            continue
        surface = parent.surface(op.surface_id)
        if surface is not None:
            proposal.preconditions[op.surface_id] = surface.content_hash
    return proposal


def _project_request(*, workspace: str, context_dir: str, proposal_dir: str, count: int) -> str:
    """Render one filesystem-first proposal task for the ordinary meta agent."""
    absolute_context = f"{workspace}/{context_dir}"
    absolute_proposals = f"{workspace}/{proposal_dir}"
    outputs = "\n".join(
        f"- {absolute_proposals}/proposal-{index:02d}.json" for index in range(1, count + 1)
    )
    return f"""Produce exactly {count} independent harness proposals for this optimization round.

Read:
- parent manifest: {absolute_context}/parent.json
  - follow surface_index_manifest to find every independently readable surface manifest
  - each surface manifest lists ordered content_files; concatenate them to inspect exact content
  - pathful code is also mirrored under source_root with exact source_file paths for direct reads
- failure evidence manifest: {absolute_context}/evidence.json
  - read its content_files in listed order and concatenate them exactly
- judged history manifest: {absolute_context}/history.json
  - read its content_files in listed order, concatenate them exactly, then parse the JSON array
- earlier raw proposals, when useful: {workspace}/proposals/
- earlier candidate evaluation manifests and traces: {workspace}/evaluations/

Write exactly these files, without changing earlier rounds:
{outputs}

Each file must be one JSON object:
{{"expected_effect":"<falsifiable prediction>",
 "preconditions":{{"<surface id>":"<hash copied from parent>"}},
 "ops":[{{"op":"add|replace|remove","surface_id":"<kind:slug>",
           "kind":"<required for add>","content":"<full content>",
           "rationale":"<why this helps>"}}]}}

For a replacement, you may omit content and use compact exact edits instead:
"edits":[{{"old":"<nonempty text occurring exactly once>","new":"<replacement>"}}].
The optimizer expands those edits against the parent before validation. Every proposal must be
focused, valid against the same supplied parent, and meaningfully different from its siblings.
After all files exist, call submit with a short summary."""

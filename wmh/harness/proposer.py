"""Typed delta proposers for direct providers and project-backed meta agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from wmh.agents.project import AgentProjectRun
from wmh.core.types import JsonObject
from wmh.harness.delta import FailureSignature, HarnessDelta
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import parse_delta, propose_delta
from wmh.providers.base import Provider, ToolCallingProvider


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
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Make ``count`` independent proposal calls against the same parent."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for _ in range(count):
            try:
                proposal = propose_delta(
                    parent,
                    trigger,
                    evidence,
                    self._provider,
                    history=history,
                )
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

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        """Run one meta-agent turn that writes ``count`` proposal files."""
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        self._round += 1
        round_dir = f"round-{self._round:04d}"
        context_dir = f"context/{round_dir}"
        proposal_dir = f"proposals/{round_dir}"
        self._project.write_text(
            f"{context_dir}/parent.json", json.dumps(_parent_context(parent), indent=2)
        )
        self._project.write_text(f"{context_dir}/evidence.md", evidence)
        self._project.write_text(
            f"{context_dir}/history.json",
            json.dumps([delta.model_dump(mode="json") for delta in history], indent=2),
        )
        request = _project_request(
            workspace=self._project.workspace,
            context_dir=context_dir,
            proposal_dir=proposal_dir,
            count=count,
        )
        self._project.write_text(f"{context_dir}/REQUEST.md", request)
        run_error: Exception | None = None
        try:
            self._project.run(self._agent, self._provider, request)
        except Exception as error:  # noqa: BLE001 - durable project files may still be complete
            run_error = error

        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for index in range(1, count + 1):
            try:
                raw = self._project.read_text(f"{proposal_dir}/proposal-{index:02d}.json")
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
                    proposals.append(_stamp_project_preconditions(parent, proposal))
        return proposals


def _parent_context(parent: HarnessDoc) -> JsonObject:
    """Serialize a parent with the computed identities a project agent must copy."""
    return {
        "name": parent.name,
        "version": parent.version,
        "doc_hash": parent.doc_hash,
        "surfaces": [
            {
                "id": surface.id,
                "kind": surface.kind.value,
                "content": surface.content,
                "content_hash": surface.content_hash,
                "budget": surface.budget,
                "path": surface.path,
            }
            for surface in parent.surfaces
        ],
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
- parent document: {absolute_context}/parent.json
- failure evidence: {absolute_context}/evidence.md
- judged history: {absolute_context}/history.json
- earlier raw proposals, when useful: {workspace}/proposals/

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
